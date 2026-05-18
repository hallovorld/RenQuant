# renquant_106 — momentum/CTA strategy (PROTOTYPE)

Per 2026-05-18 user request: explore trend-following strategy as
diversification from renquant_104's mean-reversion-biased stock-picker.

Status: skeleton only. Not for live trading.

Signal: Jegadeesh-Titman 12-1 momentum + 200-MA crossover + 52w-high distance.
Universe: top-N momentum tickers from wl200.
Position sizing: vol-targeted at 10% portfolio vol.
Risk: trailing 20% from peak.

References:
- Jegadeesh-Titman 1993 JF
- Moskowitz-Ooi-Pedersen 2012 JFE "Time series momentum"
- Hurst-Ooi-Pedersen 2017 JPM "A century of evidence on trend-following"
