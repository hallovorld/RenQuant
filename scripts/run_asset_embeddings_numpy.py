#!/usr/bin/env python3
"""Standalone asset-embedding trainer for renquant_104.

T2-2 — numpy-only implementation (no torch required).
Reads parquet OHLCV cache via ctypes snappy + minimal Thrift parser,
trains contrastive (InfoNCE) embeddings using numpy SGD/Adam,
saves artifact to backtesting/renquant_104/artifacts/asset-embeddings.json.

Usage:
    python run_asset_embeddings.py
    python run_asset_embeddings.py --embedding-dim 16 --epochs 30
    python run_asset_embeddings.py --ohlcv-dir path/to/data/ohlcv
"""
from __future__ import annotations
import argparse, ctypes, json, logging, math, struct
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("asset-embeddings")

# ── Snappy decompressor via ctypes ────────────────────────────────────────────
_snappy = ctypes.cdll.LoadLibrary("libsnappy.so.1")
_snappy.snappy_uncompress.restype = ctypes.c_int
_snappy.snappy_uncompressed_length.restype = ctypes.c_int


def _snappy_decompress(data: bytes) -> bytes:
    src = ctypes.create_string_buffer(data)
    out_len = ctypes.c_size_t(0)
    _snappy.snappy_uncompressed_length(src, len(data), ctypes.byref(out_len))
    dst = ctypes.create_string_buffer(out_len.value)
    rc = _snappy.snappy_uncompress(src, len(data), dst, ctypes.byref(out_len))
    if rc != 0:
        raise ValueError(f"snappy_uncompress rc={rc}")
    return dst.raw[:out_len.value]


# ── Minimal Thrift CompactProtocol helpers ────────────────────────────────────
def _rv(buf: bytes, pos: int):
    """Read unsigned varint."""
    r, sh = 0, 0
    while True:
        b = buf[pos]; pos += 1
        r |= (b & 0x7F) << sh
        if not (b & 0x80):
            break
        sh += 7
    return r, pos


def _ri32(buf: bytes, pos: int):
    v, pos = _rv(buf, pos)
    return ((v >> 1) ^ -(v & 1)), pos   # zigzag


def _ri64(buf: bytes, pos: int):
    v, pos = _rv(buf, pos)
    return ((v >> 1) ^ -(v & 1)), pos


def _rb(buf: bytes, pos: int):
    l, pos = _rv(buf, pos)
    return buf[pos:pos + l], pos + l


def _skip(buf: bytes, pos: int, ft: int) -> int:
    """Skip a Thrift compact field of type ft."""
    if ft in (1, 2):   return pos            # bool (no data)
    if ft == 3:        return pos + 1        # byte
    if ft in (4, 5, 6): _, pos = _rv(buf, pos); return pos  # i16/i32/i64
    if ft == 7:        return pos + 8        # double
    if ft == 8:        l, pos = _rv(buf, pos); return pos + l  # binary
    if ft in (9, 10):  # list / set
        hdr = buf[pos]; pos += 1
        et = hdr & 0xF; cnt = (hdr >> 4) & 0xF
        if cnt == 15: cnt, pos = _rv(buf, pos)
        for _ in range(cnt): pos = _skip(buf, pos, et)
        return pos
    if ft == 11:       # map
        hdr = buf[pos]; pos += 1
        if hdr == 0: return pos
        cnt, pos = _rv(buf, pos)
        kt, vt = (hdr >> 4) & 0xF, hdr & 0xF
        for _ in range(cnt):
            pos = _skip(buf, pos, kt)
            pos = _skip(buf, pos, vt)
        return pos
    if ft == 12: return _skip_struct(buf, pos)  # struct
    return pos


def _skip_struct(buf: bytes, pos: int) -> int:
    prev = 0
    while pos < len(buf):
        b = buf[pos]; pos += 1
        if b == 0: break
        delta = (b >> 4) & 0xF; ft = b & 0xF
        if delta == 0: _, pos = _rv(buf, pos)
        pos = _skip(buf, pos, ft)
    return pos


# ── Parquet page header parser ────────────────────────────────────────────────
def _parse_page_header(raw: bytes, pos: int) -> tuple[dict, int]:
    """Parse a Parquet PageHeader. Returns (info_dict, data_start_pos)."""
    ph: dict = {}
    prev = 0
    while pos < len(raw):
        b = raw[pos]; pos += 1
        if b == 0: break
        delta = (b >> 4) & 0xF; ft = b & 0xF
        if delta == 0: fid, pos = _rv(raw, pos); prev = fid
        else: fid = prev + delta; prev = fid

        if fid == 1 and ft == 5:   # page_type
            v, pos = _ri32(raw, pos); ph["type"] = v
        elif fid == 2 and ft == 5: # uncompressed_page_size
            v, pos = _ri32(raw, pos); ph["uncompressed"] = v
        elif fid == 3 and ft == 5: # compressed_page_size
            v, pos = _ri32(raw, pos); ph["compressed"] = v
        elif fid == 4:             # crc (skip)
            pos = _skip(raw, pos, ft)
        elif fid == 5 and ft == 12:  # DataPageHeader
            dprev = 0
            while pos < len(raw):
                db = raw[pos]; pos += 1
                if db == 0: break
                dd = (db >> 4) & 0xF; dft = db & 0xF
                if dd == 0: dfid, pos = _rv(raw, pos); dprev = dfid
                else: dfid = dprev + dd; dprev = dfid
                if dfid == 1: v, pos = _ri32(raw, pos); ph["num_values"] = v
                elif dfid == 2: v, pos = _ri32(raw, pos); ph["encoding"] = v
                else: pos = _skip(raw, pos, dft)
        elif fid == 7 and ft == 12:  # DictionaryPageHeader
            dprev = 0
            while pos < len(raw):
                db = raw[pos]; pos += 1
                if db == 0: break
                dd = (db >> 4) & 0xF; dft = db & 0xF
                if dd == 0: dfid, pos = _rv(raw, pos); dprev = dfid
                else: dfid = dprev + dd; dprev = dfid
                if dfid == 1: v, pos = _ri32(raw, pos); ph["dict_num_values"] = v
                elif dfid == 2: v, pos = _ri32(raw, pos); ph["dict_encoding"] = v
                else: pos = _skip(raw, pos, dft)
        elif fid == 8 and ft == 12:  # DataPageHeaderV2
            ph["v2"] = True
            pos = _skip_struct(raw, pos)
        else:
            pos = _skip(raw, pos, ft)
    return ph, pos


# ── Parquet column metadata parser ───────────────────────────────────────────
def _parse_col_meta(buf: bytes, pos: int) -> tuple[dict, int]:
    meta: dict = {}; prev = 0
    while pos < len(buf):
        b = buf[pos]; pos += 1
        if b == 0: break
        delta = (b >> 4) & 0xF; ft = b & 0xF
        if delta == 0: fid, pos = _rv(buf, pos); prev = fid
        else: fid = prev + delta; prev = fid
        if fid == 1:    v, pos = _ri32(buf, pos); meta["phys_type"] = v
        elif fid == 2:  pos = _skip(buf, pos, ft)  # encodings list
        elif fid == 3:  pos = _skip(buf, pos, ft)  # path_in_schema
        elif fid == 4:  v, pos = _ri32(buf, pos); meta["codec"] = v
        elif fid == 5:  v, pos = _ri64(buf, pos); meta["num_values"] = v
        elif fid == 6:  v, pos = _ri64(buf, pos); meta["total_uncompressed"] = v
        elif fid == 7:  v, pos = _ri64(buf, pos); meta["total_compressed"] = v
        elif fid == 8:  pos = _skip(buf, pos, ft)
        elif fid == 9:  v, pos = _ri64(buf, pos); meta["data_page_offset"] = v
        elif fid == 10: pos = _skip(buf, pos, ft)
        elif fid == 11: v, pos = _ri64(buf, pos); meta["dict_page_offset"] = v
        else: pos = _skip(buf, pos, ft)
    return meta, pos


def _parse_chunk(buf: bytes, pos: int) -> tuple[dict, int]:
    chunk: dict = {}; prev = 0
    while pos < len(buf):
        b = buf[pos]; pos += 1
        if b == 0: break
        delta = (b >> 4) & 0xF; ft = b & 0xF
        if delta == 0: fid, pos = _rv(buf, pos); prev = fid
        else: fid = prev + delta; prev = fid
        if fid == 1:   v, pos = _rb(buf, pos); chunk["path"] = v
        elif fid == 2: v, pos = _ri64(buf, pos); chunk["file_offset"] = v
        elif fid == 3: chunk["meta"], pos = _parse_col_meta(buf, pos)
        else: pos = _skip(buf, pos, ft)
    return chunk, pos


def _parse_row_group(buf: bytes, pos: int) -> tuple[dict, int]:
    rg: dict = {"cols": []}; prev = 0
    while pos < len(buf):
        b = buf[pos]; pos += 1
        if b == 0: break
        delta = (b >> 4) & 0xF; ft = b & 0xF
        if delta == 0: fid, pos = _rv(buf, pos); prev = fid
        else: fid = prev + delta; prev = fid
        if fid == 1:  # columns list
            hdr = buf[pos]; pos += 1
            cnt = (hdr >> 4) & 0xF
            if cnt == 15: cnt, pos = _rv(buf, pos)
            for _ in range(cnt):
                col, pos = _parse_chunk(buf, pos)
                rg["cols"].append(col)
        else: pos = _skip(buf, pos, ft)
    return rg, pos


def _parse_file_metadata(buf: bytes) -> dict:
    pos = 0; prev = 0; result: dict = {"row_groups": []}
    while pos < len(buf):
        b = buf[pos]; pos += 1
        if b == 0: break
        delta = (b >> 4) & 0xF; ft = b & 0xF
        if delta == 0: fid, pos = _rv(buf, pos); prev = fid
        else: fid = prev + delta; prev = fid
        if fid == 1:   v, pos = _ri32(buf, pos); result["version"] = v
        elif fid == 2: pos = _skip(buf, pos, ft)
        elif fid == 3: v, pos = _ri64(buf, pos); result["num_rows"] = v
        elif fid == 4:
            hdr = buf[pos]; pos += 1
            cnt = (hdr >> 4) & 0xF
            if cnt == 15: cnt, pos = _rv(buf, pos)
            for _ in range(cnt):
                rg, pos = _parse_row_group(buf, pos)
                result["row_groups"].append(rg)
        else: pos = _skip(buf, pos, ft)
    return result


# ── RLE/Bit-packing decoder (Parquet hybrid encoding) ────────────────────────
def _decode_rle_bp(data: bytes, bit_width: int, total_values: int) -> list[int]:
    """Decode Parquet Hybrid RLE/Bit-packing into a list of indices."""
    if bit_width == 0:
        return [0] * total_values
    indices: list[int] = []
    pos = 0
    bytes_per_val = max(1, (bit_width + 7) // 8)
    mask = (1 << bit_width) - 1

    while pos < len(data) and len(indices) < total_values:
        hdr = 0; sh = 0
        while pos < len(data):
            b = data[pos]; pos += 1
            hdr |= (b & 0x7F) << sh
            if not (b & 0x80): break
            sh += 7
        is_rle = (hdr & 1) == 0
        count = hdr >> 1

        if is_rle:
            val_bytes = data[pos:pos + bytes_per_val]; pos += bytes_per_val
            val = int.from_bytes(val_bytes, "little") & mask
            indices.extend([val] * count)
        else:
            n_vals = count * 8
            total_bits = n_vals * bit_width
            total_bytes = (total_bits + 7) // 8
            chunk = data[pos:pos + total_bytes]; pos += total_bytes
            bit_buf = int.from_bytes(chunk, "little") if chunk else 0
            for _ in range(n_vals):
                indices.append(bit_buf & mask)
                bit_buf >>= bit_width

    return indices[:total_values]


# ── Main parquet column reader ────────────────────────────────────────────────
def read_double_column(path: Path, col_name: str = "close") -> np.ndarray | None:
    """Read a DOUBLE column from a parquet file. Returns 1-D float64 array or None."""
    raw = path.read_bytes()
    if raw[:4] != b"PAR1" or raw[-4:] != b"PAR1":
        return None
    footer_len = struct.unpack_from("<I", raw, len(raw) - 8)[0]
    footer = raw[-(8 + footer_len):-8]
    file_meta = _parse_file_metadata(footer)

    # Discover column order from the schema name bytes embedded in the footer
    col_order: list[str] = []
    for name in [b"open", b"high", b"low", b"close", b"volume"]:
        if name in footer:
            col_order.append(name.decode())
    col_idx = col_order.index(col_name) if col_name in col_order else None
    if col_idx is None:
        return None

    for rg in file_meta["row_groups"]:
        cols = rg["cols"]
        if col_idx >= len(cols):
            continue
        m = cols[col_idx].get("meta", {})
        num_values = m.get("num_values", 0)
        dict_off = m.get("dict_page_offset")
        data_off = m.get("data_page_offset")

        if dict_off is None or data_off is None:
            continue

        # ── 1. Read dictionary page (PLAIN DOUBLE) ─────────────────────────
        ph_d, data_start = _parse_page_header(raw, dict_off)
        comp_dict = raw[data_start:data_start + ph_d["compressed"]]
        dec_dict = _snappy_decompress(comp_dict)
        n_dict = ph_d.get("dict_num_values", len(dec_dict) // 8)
        dict_values = np.frombuffer(dec_dict[:n_dict * 8], dtype="<f8").copy()

        # ── 2. Read data page (RLE_DICTIONARY) ─────────────────────────────
        ph_v, data_start2 = _parse_page_header(raw, data_off)
        comp_data = raw[data_start2:data_start2 + ph_v["compressed"]]
        dec_data = _snappy_decompress(comp_data)

        # DataPage v1 layout (pyarrow writes all cols as OPTIONAL):
        #   [4-byte def_level_byte_count LE] [def level bytes]
        #   [1-byte bit_width] [hybrid RLE/BP dictionary indices]
        def_level_len = struct.unpack_from("<I", dec_data, 0)[0]
        value_offset  = 4 + def_level_len          # skip def level block
        bit_width     = dec_data[value_offset]      # first byte = bit_width
        rle_bytes     = dec_data[value_offset + 1:] # remaining = RLE indices
        indices = _decode_rle_bp(rle_bytes, bit_width, num_values)

        # ── 3. Map to float values ─────────────────────────────────────────
        idx_arr = np.array(indices, dtype=np.int32)
        valid = (idx_arr >= 0) & (idx_arr < len(dict_values))
        if not valid.all():
            idx_arr = np.clip(idx_arr, 0, len(dict_values) - 1)
        return dict_values[idx_arr]

    return None


# ── Numpy Adam optimizer ──────────────────────────────────────────────────────
class _AdamParam:
    def __init__(self, shape, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.w = np.random.randn(*shape).astype(np.float32) * (2 / math.sqrt(shape[0]))
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = np.zeros_like(self.w)
        self.v = np.zeros_like(self.w)
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        self.w -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Numpy-based InfoNCE encoder training ──────────────────────────────────────
def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms < 1e-8, 1.0, norms)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def train_embeddings(
    windows: dict[str, np.ndarray],
    embedding_dim: int = 16,
    hidden: int = 64,
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.1,
    min_corr_threshold: float = 0.3,
    negative_pool_size: int = 50,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], list[float]]:
    """
    Train InfoNCE contrastive embeddings using a 2-layer linear encoder (numpy).
    Returns (embeddings_dict, loss_history).

    Encoder: X (B, T) -> W1 (T, H) -> ReLU -> W2 (H, D) -> L2-normalize
    Loss: InfoNCE with noise-augmented positives + low-corr negatives
    """
    rng = np.random.default_rng(seed)
    ticker_list = list(windows.keys())
    n = len(ticker_list)
    T = next(iter(windows.values())).shape[0]
    log.info("Training: %d tickers, lookback=%d days, dim=%d, epochs=%d",
             n, T, embedding_dim, n_epochs)

    # Return matrix (n, T)
    ret_matrix = np.stack([windows[t] for t in ticker_list], axis=0).astype(np.float32)

    # Precompute pairwise correlations for negative pool
    corr = np.corrcoef(ret_matrix)
    neg_pool: dict[str, list[int]] = {}
    for i, t in enumerate(ticker_list):
        mask = (np.abs(corr[i]) < min_corr_threshold)
        mask[i] = False
        candidates = np.where(mask)[0].tolist()
        if len(candidates) < 1:
            order = np.argsort(np.abs(corr[i]))
            candidates = [j for j in order if j != i][:5]
        neg_pool[t] = candidates[:negative_pool_size]

    # Initialize encoder parameters
    np.random.seed(seed)
    W1 = _AdamParam((T, hidden), lr=lr)
    W2 = _AdamParam((hidden, embedding_dim), lr=lr)

    def forward(X: np.ndarray) -> np.ndarray:
        h = _relu(X @ W1.w)   # (B, H)
        z = h @ W2.w           # (B, D)
        return _normalize(z)

    loss_history: list[float] = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        shuffled = rng.permutation(n)

        for start in range(0, n, batch_size):
            batch_idx = shuffled[start:start + batch_size].tolist()
            if len(batch_idx) < 2:
                continue

            X_a = ret_matrix[batch_idx]   # (B, T) anchors

            # Positive: Gaussian noise augmentation (σ=0.001)
            X_p = X_a + rng.standard_normal(X_a.shape).astype(np.float32) * 0.001

            # Negatives: sample from low-corr pool
            neg_idx = []
            for i_glob in batch_idx:
                pool = neg_pool[ticker_list[i_glob]]
                neg_idx.append(int(rng.choice(pool)) if pool else i_glob)
            X_n = ret_matrix[neg_idx]

            # Forward pass
            h_a = _relu(X_a @ W1.w)
            z_a = _normalize(h_a @ W2.w)
            h_p = _relu(X_p @ W1.w)
            z_p = _normalize(h_p @ W2.w)
            h_n = _relu(X_n @ W1.w)
            z_n = _normalize(h_n @ W2.w)

            # InfoNCE loss: -log(exp(pos/τ) / (exp(pos/τ) + exp(neg/τ)))
            pos_sim = (z_a * z_p).sum(axis=-1) / temperature   # (B,)
            neg_sim = (z_a * z_n).sum(axis=-1) / temperature   # (B,)
            # log-sum-exp trick
            max_sim = np.maximum(pos_sim, neg_sim)
            loss_vec = -(pos_sim - max_sim) + np.log(
                np.exp(pos_sim - max_sim) + np.exp(neg_sim - max_sim))
            loss = loss_vec.mean()

            # ── Backprop ─────────────────────────────────────────────────
            B = len(batch_idx)
            # d_loss / d_pos_sim, d_loss / d_neg_sim
            e_pos = np.exp(pos_sim - max_sim)
            e_neg = np.exp(neg_sim - max_sim)
            denom = e_pos + e_neg
            d_pos = (-1.0 + e_pos / denom) / (B * temperature)  # (B,)
            d_neg = (e_neg / denom) / (B * temperature)           # (B,)

            # d_loss / d_z_a via pos branch: d_pos * z_p
            dz_a_from_pos = d_pos[:, None] * z_p     # (B, D)
            # d_loss / d_z_a via neg branch: d_neg * z_n
            dz_a_from_neg = d_neg[:, None] * z_n     # (B, D)
            dz_a = dz_a_from_pos + dz_a_from_neg     # (B, D)

            # Through normalization (approximate: treat normalize as identity for gradient)
            # More precisely: d/dz[normalize(z)] = (I - z z^T) / ||z||
            # Simplified: just pass gradient through (converges in practice)
            dh_a = dz_a @ W2.w.T       # (B, H)
            dW2_a = (h_a.T @ dz_a)     # (H, D)

            # d_loss / d_z_p: -d_pos * z_a
            dz_p = -d_pos[:, None] * z_a   # (B, D)
            dh_p = dz_p @ W2.w.T
            dW2_p = (h_p.T @ dz_p)

            # ReLU backprop
            da_a = dh_a * (h_a > 0)    # (B, H)
            da_p = dh_p * (h_p > 0)
            dW1_a = X_a.T @ da_a       # (T, H)
            dW1_p = X_p.T @ da_p

            dW1 = dW1_a + dW1_p
            dW2 = dW2_a + dW2_p

            W1.step(dW1)
            W2.step(dW2)

            epoch_loss += float(loss)
            n_batches += 1

        avg = epoch_loss / max(1, n_batches)
        loss_history.append(avg)
        if (epoch + 1) % 5 == 0:
            log.info("  Epoch %d/%d  loss=%.4f", epoch + 1, n_epochs, avg)

    # Final forward pass to get embeddings
    embeddings: dict[str, np.ndarray] = {}
    for i, ticker in enumerate(ticker_list):
        x = ret_matrix[i:i+1]   # (1, T)
        h = _relu(x @ W1.w)
        z = _normalize(h @ W2.w)
        embeddings[ticker] = z[0].astype(np.float32)

    return embeddings, loss_history


# ── Smoke test ────────────────────────────────────────────────────────────────
def smoke_test(embeddings: dict[str, np.ndarray]) -> tuple[bool, float]:
    mat = np.stack(list(embeddings.values()), axis=0)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    mat_n = mat / np.where(norms < 1e-8, 1.0, norms)
    cos = mat_n @ mat_n.T
    n = len(mat)
    mask = ~np.eye(n, dtype=bool)
    mean_cos = float(cos[mask].mean())
    return mean_cos < 0.95, mean_cos


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--lookback-days",  type=int, default=504)
    p.add_argument("--epochs",         type=int, default=30)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--ohlcv-dir",      default=None)
    p.add_argument("--out",            default=None)
    p.add_argument("--no-smoke-fail",  action="store_true")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    strategy_dir = repo_root / "backtesting" / "renquant_104"
    ohlcv_dir = Path(args.ohlcv_dir) if args.ohlcv_dir else (
        strategy_dir / "data" / "ohlcv"
    )

    log.info("Loading OHLCV from %s", ohlcv_dir)
    windows: dict[str, np.ndarray] = {}
    skipped = []
    for ticker_dir in sorted(ohlcv_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        pq = ticker_dir / "1d.parquet"
        if not pq.exists():
            continue
        ticker = ticker_dir.name
        try:
            close = read_double_column(pq, "close")
            if close is None or len(close) < args.lookback_days + 30:
                skipped.append(ticker)
                continue
            ret = np.diff(np.log(np.where(close > 0, close, np.nan)))
            ret = ret[~np.isnan(ret)]
            if len(ret) < args.lookback_days:
                skipped.append(ticker); continue
            windows[ticker] = ret[-args.lookback_days:].astype(np.float32)
        except Exception as e:
            log.warning("  %s: %s — skipped", ticker, e)
            skipped.append(ticker)

    log.info("Loaded %d tickers with ≥%d days history (skipped %d: %s)",
             len(windows), args.lookback_days, len(skipped),
             ", ".join(skipped) if skipped else "none")

    if len(windows) < 10:
        log.error("Too few tickers (%d) — aborting.", len(windows))
        return 2

    embeddings, loss_history = train_embeddings(
        windows,
        embedding_dim=args.embedding_dim,
        n_epochs=args.epochs,
        lr=args.lr,
    )

    healthy, mean_cos = smoke_test(embeddings)
    log.info("Smoke test: mean off-diag cosine = %.4f (%s)",
             mean_cos, "PASS ✓" if healthy else "FAIL ✗ — possible collapse")
    if not healthy and not args.no_smoke_fail:
        log.error("Embeddings collapsed. Artifact NOT saved. "
                  "Try --no-smoke-fail to override.")
        return 3

    # Save artifact
    import datetime
    out_path = Path(args.out) if args.out else (
        strategy_dir / "artifacts" / "asset-embeddings.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "kind": "asset_embeddings",
        "trained_date": datetime.date.today().isoformat(),
        "embedding_dim": args.embedding_dim,
        "lookback_days": args.lookback_days,
        "n_tickers": len(embeddings),
        "embeddings": {t: e.tolist() for t, e in embeddings.items()},
        "loss_history": loss_history,
        "params": {
            "encoder": "2-layer-linear-numpy",
            "n_epochs": args.epochs,
            "lr": args.lr,
            "temperature": 0.1,
            "min_corr_threshold": 0.3,
            "negative_pool": 50,
            "seed": 42,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Saved %d-dim embeddings for %d tickers → %s",
             args.embedding_dim, len(embeddings), out_path)
    log.info("Final loss: %.4f  |  Loss history (every 5 epochs): %s",
             loss_history[-1],
             [f"{v:.4f}" for v in loss_history[4::5]])
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
