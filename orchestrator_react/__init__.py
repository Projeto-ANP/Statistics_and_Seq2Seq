"""ReAct combiner agent over a closed catalog of deterministic tools.

Isolated package: imports nothing from `orchestrator/` or `orchestrator_langchain/`
at module level, so the catalog can be tested on synthetic data without dragging in
the heavy dependency chain (aeon, pywt, pyts, darts) or the LLM layer.

Layout (mirrors Section 3.4 of the specification):
    config.py     - single configuration object per experiment run (ablations)
    metrics.py    - self-contained MAPE/SMAPE/RMSE/POCID/MSMAPE/MAE
    state.py      - application state: raw data, handles, attempt history
    combiners.py  - combination functions (same code in backtest and final apply)
    weighting.py  - weight recipes, computed in code and referenced by handle
    features.py   - series profile (STL, stationarity, outliers, catch22)
    tools.py      - the closed catalog, sections 3.4.1-3.4.5
    registry.py   - dispatch, argument validation and call trace
"""

from orchestrator_react.config import ReactConfig
from orchestrator_react.state import Attempt, ReactState

__all__ = ["ReactConfig", "ReactState", "Attempt"]
