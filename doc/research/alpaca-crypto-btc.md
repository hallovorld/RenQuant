# Alpaca Crypto Trading — BTC Feasibility & Plan (2026-04-26)


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

**User question**: 用 Alpaca 交易 BTC 等虚拟货币的可行性？需要哪些改动？
BTC 太贵了，怎么买？

**TL;DR**: Alpaca 支持加密货币交易（包括 BTC/ETH/LTC/etc.），**支持 fractional**
（minimum buy ≈ $1）。与 equities 共享 portfolio API 但有几个重要差异。
预计需要约 **3 周开发** 才能从 renquant_104 (equities-only) 扩展到
multi-asset (equities + crypto)。

---

## 1. Alpaca 加密货币 API 现状（2026-04 baseline）

### 1.1 支持的资产
- 主流：BTC, ETH, LTC, BCH, AVAX, DOT, LINK, SHIB, SOL, USDT, USDC
- 全 list ≈ 25-30 个 trading pair（vs USD）
- 不支持币币交易（e.g. ETH/BTC）；都是 vs USD

### 1.2 关键差异 vs equities

| 维度 | Equities | Crypto |
|---|---|---|
| 交易时间 | 9:30-16:00 ET, 工作日 | **24/7/365** |
| Tick size | $0.01 | 0.000001 BTC = ~$0.09 |
| Min order | 1 share OR $1 fractional | **$1 fractional**（BTC ~ 0.00001） |
| Settlement | T+2 | T+0 (instant) |
| Wash sale | IRS rule applies | **No wash sale** for crypto |
| Long-term capital gains | yes (>365d) | yes (>365d) |
| Short selling | yes (with margin) | **NOT supported** at Alpaca |
| Margin | 2x intraday, 1x overnight | **NOT supported** |
| API endpoint | `/v2/orders` | `/v1beta1/crypto/orders` (different prefix) |
| Account separation | 同 account | **同 account, 同 cash pool** |
| Position type | `asset_class=us_equity` | `asset_class=crypto` |
| Data feed | IEX/CTP | Alpaca-aggregated from CB/Binance/Kraken |
| OHLCV history | yfinance fallback | **No yfinance**; need direct Alpaca data API |

### 1.3 BTC fractional buying

**已经原生支持**。例：
```python
broker.submit_order(
    symbol="BTC/USD",
    qty=0.001,         # ~$90 at BTC=$90k
    side="buy",
    type="market",
)
# OR notional-based:
broker.submit_order(
    symbol="BTC/USD",
    notional=100,       # buy $100 of BTC, fractional auto-computed
    side="buy",
    type="market",
)
```
**Min trade size: $1**. BTC 太贵不是问题——直接用 notional API。

---

## 2. 改动范围（按 layer）

### 2.1 `common/data/` — data layer

新加 `AlpacaCryptoSource(DataSource)`:

```python
class AlpacaCryptoSource(DataSource):
    def fetch_ohlcv(self, symbol: str, ...) -> pd.DataFrame:
        # symbol="BTC/USD" not "BTC"
        # uses alpaca-py's CryptoHistoricalDataClient
        ...
```

`fetch_ohlcv()` 顶层路由：如果 symbol 含 "/USD" 或在 crypto 名单 →
crypto source；否则 equity source.

**Crypto bar resolutions**: 1Min, 5Min, 15Min, 1Hour, 4Hour, 1Day, 1Week.
24/7 → 一天 24 个 hour bar，不是 7 个。

### 2.2 `kernel/intraday.py` — bar store

- 现有 `HourlyBarStore` / `MinuteBarStore` 都按 NY market 时间戳建索引；crypto 是 UTC 24/7。
- 加 `CryptoHourlyBarStore(_TimeframedBarStore)`，子目录 `data/crypto/intraday/{SYMBOL}/1h.parquet`，索引 UTC。

### 2.3 `live/alpaca_broker.py` — order routing

主要改动：
```python
def submit_order(self, ticker, ..., asset_class="us_equity"):
    if asset_class == "crypto":
        # use crypto order endpoint
        # symbol must be "BTC/USD" not "BTC"
        # validate symbol pair format
        ...
```

Position fetch (`get_all_positions`) 已经返回 mixed equity + crypto，
按 `position.asset_class` 区分即可。

### 2.4 `kernel/regime.py` — regime detection

**SPY 不适用于 crypto**。需要 crypto-side regime：
- 选项 A: 用 BTC 自身做 regime detector（自相关）
- 选项 B: 用 crypto-aggregate index (BITX 或 GBTC) 做 regime
- 选项 C: 不做 crypto regime，crypto 永远算 BULL_VOL（高 vol 默认）

**Recommended**: 选项 B，用 BITX 复制 SPY 的 GMM 流程。

### 2.5 `kernel/exits.py` — risk control

Crypto vol >> equity vol。需要：
- Stop-loss 阈值放宽：equity 8% → crypto 15-20%
- Trailing stop 阈值放宽：18% → 25-30%
- Max drawdown：portfolio-level 15% → crypto-side 25%
- **No wash-sale rule** for crypto → simpler exit logic

### 2.6 `kernel/sizing.py` — Kelly + position cap

- Kelly cap (max_position_pct) for crypto SHOULD be much lower than
  equities, since crypto sigma 是 equity 的 ~3-5x：
  - Equity max_position_pct = 0.20
  - Crypto max_position_pct = 0.05 (推荐)
- Total crypto allocation cap: max(20% NAV)

### 2.7 `kernel/pipeline/...` — InferencePipeline

- 现有 InferencePipeline 假设所有资产是 equities。需要：
  - 按资产类别拆分 candidate scan / sell loop
  - 或者：把 asset_class 当作一个 feature，在 task 内 dispatch
- **Recommended**: 加 `ctx.asset_class_per_ticker: dict[str, str]`，每个 task 在内部 branch

### 2.8 Strategy config

```json
{
  "watchlist": [
    "AAPL", "MSFT", ...,            // equities
    "BTC/USD", "ETH/USD"             // crypto pairs (symbol format change)
  ],
  "asset_classes": {
    "BTC/USD": "crypto",
    "ETH/USD": "crypto",
    "_default": "us_equity"
  },
  "crypto": {
    "regime_detector": "BITX",         // SPY-equivalent for crypto
    "max_position_pct": 0.05,
    "stop_loss_pct": 0.20,
    "trailing_stop_pct": 0.30,
    "max_total_allocation_pct": 0.20   // total crypto exposure cap
  }
}
```

### 2.9 Backtesting: LEAN 不支持 Alpaca-style crypto

LEAN supports crypto via Coinbase but pair formats / fees differ from
Alpaca. **Recommended**: backtest crypto separately via the
`SimAdapter + run_backtest` path (which doesn't depend on LEAN);
crypto OHLCV from Alpaca data API.

---

## 3. 实施 plan — 3 周分 6 个 PR

| PR | Scope | 工时 | 风险 |
|---|---|---|---|
| 1 | `common/data/AlpacaCryptoSource` + `CryptoBarStore` | 2 days | 🟢 |
| 2 | `live/alpaca_broker.py` crypto order routing + tests | 2 days | 🟡 |
| 3 | Crypto regime detector (BITX-based GMM) + artifact | 3 days | 🟡 |
| 4 | InferencePipeline asset_class dispatch (sizing/exits) | 4 days | 🟠 |
| 5 | Strategy config schema + new `renquant_105_crypto` strategy | 2 days | 🟢 |
| 6 | E2E sim + paper-trade verification + ops runbook | 2 days | 🟠 |

**总计**: ~15 工作日 ≈ 3 周 calendar time。

### 3.1 第一周（PR 1-2）— infra
- AlpacaCryptoSource fetch_ohlcv tested against live API
- CryptoBarStore parquet cache works end-to-end
- AlpacaBroker.submit_order 支持 crypto, validated paper-trade

### 3.2 第二周（PR 3-4）— logic
- Crypto regime detector trained on 2020-2026 BTC history
- Pipeline dispatch: candidates filtered by asset_class
- Sizing: crypto Kelly + cap correctly applied

### 3.3 第三周（PR 5-6）— integration
- New strategy `renquant_105_crypto_eq_blend` (50/50 split)
- Backtest 2024-01-01 → 2026-04-25
- Paper-trade for 1 week, verify live behavior

---

## 4. 风险 / decision points

### 4.1 单独策略 vs 混合策略
- **单独**：renquant_105_crypto，纯 crypto-only。简单，不影响 104。
- **混合**：renquant_105_blend，equity + crypto 共用 InferencePipeline。
  逻辑复杂但可分散风险（crypto + equity 相关性低）。

**Recommended**: 先单独做（简单 + 可独立验证），熟悉 crypto 行为再考虑混合。

### 4.2 Position cap 是否太保守
- BTC 30 day annualized vol ~ 60%，equity ~ 18%
- 同 Kelly target 下，crypto 仓位应是 equity 的 (0.18/0.60)² = 9%
- 5% max_position_pct 看起来合理（同 Kelly 的 9% 还小一点，留 buffer）

### 4.3 Tax 处理
- Crypto 不受 wash sale 限制 → 可频繁交易
- 但 crypto 短期收益（<1 yr）按 ordinary income 征税（最高 37%）
  vs equity short-term 也是 37%，所以影响相同
- 长期持有（>1 yr）crypto 是 LTCG（15-20%）—— 同 equity

### 4.4 Data 质量
- Alpaca crypto data 来自 aggregator (CB/Binance/Kraken)
- 不同 venue 的 BTC 价差可能 >$50；Alpaca 取的是 reference price
- 高频策略可能受影响；我们的策略是 daily/hourly，**impact 可忽略**

### 4.5 24/7 交易的 operational burden
- 现在 daily_104 在 1:55 PM PT 跑一次/天
- Crypto 24/7 需要 hourly 跑（或起码每 4 小时）
- 如果不增加 cadence，crypto signals 会滞后 24h
- **Recommended Phase 1**: 仍每天 1:55 PT 跑一次 crypto；Phase 2 加 hourly

---

## 5. BTC 太贵问题 — 详细回答

```
BTC 当前 ~$90k/coin
我们 NAV = $10k
single-share buy = $90k  ← impossible
```

**解决方案**：
- ✅ Alpaca **原生支持 fractional**
- ✅ Min order = $1
- 用 `notional=$X` 而不是 `qty=N`：
  ```python
  broker.submit_order("BTC/USD", notional=500, side="buy")
  # buys $500 of BTC, qty auto = 500 / 90000 = 0.00556 BTC
  ```
- 在我们的 pipeline 中，`SizeAndEmitTask` 现在算 `shares = invest / price`
  得 0；需要改成 `qty = invest / price`（保留小数），按 6 位精度发单。

**改动**：`adapters/runner.py` 在 crypto path 不做 `int(invest / px)`，
直接 round to 6 decimals 提交 notional 单。

---

## 6. 立即下一步

1. **用户决定**：要不要做？做的话 standalone (renquant_105_crypto)
   还是 blend (renquant_105_eq_crypto)?
2. 如果 yes：
   - 在 `doc/roadmap.md` 顶部加这个 task
   - 起 PR 1（AlpacaCryptoSource）开发分支
3. 如果 no / defer：留这个文档作为参考

预计 **standalone version**: 3 周开发 + 1 周 paper trade verify = **4 周到生产**。

---

## 7. 参考

- Alpaca crypto API docs: https://alpaca.markets/docs/trading/crypto/
- alpaca-py SDK CryptoHistoricalDataClient
- Bitcoin price series: ~70% annual vol baseline; events drive ~150-200% intraday spikes
- Carhart 1997 / Asness 2013 — momentum factors apply to crypto with similar half-life as equity (~10-30d)
- Burniske-Tatar 2017 — *Cryptoassets* — practitioner guide to risk metrics for crypto in portfolio context
