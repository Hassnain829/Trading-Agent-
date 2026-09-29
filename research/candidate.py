"""
The ONE file the research loop edits (karpathy/autoresearch pattern; see research/program.md).

PARAMS overrides strategy knobs for a backtest only; live trading is never changed from here.
Only the names in evaluate.ALLOWED are accepted. Leave PARAMS empty to test the current rules.
"""

PARAMS = {
    # "SCALP_RSI_PULLBACK": 40.0,
}
EXIT_MODE = "FIXED"  # FIXED | BREAKEVEN | PARTIAL | TRAIL
NOTE = "baseline: the adopted rules"  # one line: the idea behind this candidate
