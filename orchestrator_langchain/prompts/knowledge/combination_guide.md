# Forecast Combination Methods — Decision Guide

Academic references and empirical rules for selecting forecast combination strategies.

---

## When to Use Each Method

### Equal-Weight Mean (`baseline_mean`)
**Use when**: All models have similar RMSE on validation (spread ratio < 0.1) AND no significant per-horizon heterogeneity.
**Avoid when**: One or more models are clear outliers, or different horizons are dominated by different models.
**Reference**: Timmermann (2006) — equal weighting is surprisingly competitive but is beaten by adaptive methods when model quality is heterogeneous.

---

### Median (`robust_median`)
**Use when**: Model disagreement is high (spread > 0.25) and you suspect outlier predictions from one or more models.
**Reference**: Genre et al. (2013) — median outperforms mean when expert forecasts are skewed.

---

### Trimmed Mean
**Use when**: A few models are consistently poor but you don't know which ones in advance.
**Key parameter**: `trim_ratio` = 0.1–0.2 is usually safe; 0.3+ risks discarding good models.

---

### DTW Barycenter Averaging — DBA (`dba_combination`)
**Use when**: Models disagree in shape/phase (not just amplitude). DBA finds a centroid that minimises the sum of DTW distances — useful when predictions have temporal alignment differences.
**Use when**: `relative_spread_mean >= 0.25` AND `seasonality_corr_variance >= 0.3`.
**Avoid when**: All models produce near-identical shapes (DBA gives same result as mean but slower).
**Reference**: Petitjean et al. (2011) — *A global averaging method for dynamic time warping*. Pattern Recognition.
**Computational note**: O(n_models × horizon²) per forecast — add `top_k` to limit models.

---

### Top-k Mean Per Horizon (`topk_mean_per_horizon`)
**Use when**: RMSE spread is moderate (0.2–0.5), n_unique_winners >= 2, n_windows >= 2.
**Key parameter**: `top_k = sqrt(n_models)` is a robust default (Makridakis M4 Competition, 2020).
**Reference**: Makridakis et al. (2020) — M4 Competition findings. The combination that won used selective averaging.

---

### Inverse-RMSE Weighted (`inverse_rmse_weights_per_horizon`)
**Use when**: Large RMSE spread between models (spread ratio > 0.3), sufficient validation windows (n_windows >= 3).
**Key parameters**: Use `shrinkage = 0.2–0.35` when n_windows <= 5 to prevent weight collapse.
**Reference**: Timmermann (2006) — *Forecast Combinations*. Handbook of Economic Forecasting.
- Outperforms equal weighting when the best model's RMSE is at least 20% better than the worst.

---

### Ridge Stacking (`ridge_stacking_per_horizon`)
**Use when**: n_windows >= 2 × n_models_used, models are collinear (similar but not identical predictions).
**Key parameters**: `l2 = 10–50` for small windows; reduce to `5–10` when n_windows >= 10.
**Reference**: Gaillard & Goude (2015) — *Forecasting Electricity Consumption by Aggregating Specialized Experts*.

---

### Exponentially/Polynomially Weighted Average (EWA/PWA)
**Use when**: You expect concept drift — recent performance is more informative than historical.
**Reference**: Cesa-Bianchi & Lugosi (2006) — *Prediction, Learning, and Games*.

---

### ADE Dynamic Error (`ade_dynamic_error_per_horizon`)
**Use when**: The best model changes over time (non-stationary). Tracks recent errors via EMA.
**Reference**: Cerqueira et al. (2019) — *Arbitrage of Forecasting Experts*. Machine Learning.

---

### Best Single / Best Per Horizon (`best_single_by_validation`, `best_per_horizon_by_validation`)
**Use when**: One model dominates (RMSE gap > 30%), OR different models dominate at different horizons (n_unique_winners >= 3).
**Risk**: Instability with n_windows <= 3 — combine with shrinkage or fallback to top-k.

---

## M4 Competition Key Takeaway (Makridakis et al., 2020)
> "Simple combination methods (equal weights, median) perform surprisingly well, but selective averaging with sqrt(n) models consistently beats them across series."

**Practical rule**: Start with `topk_mean_per_horizon` with `k = floor(sqrt(n_models))`, then upgrade to weighted methods if validation shows RMSE spread > 0.3.

---

## Signal → Method Lookup Table

| Validation Signal | Primary Method | Backup |
|---|---|---|
| All models similar RMSE (spread < 0.1) | `baseline_mean` | `robust_median` |
| High spread (0.3–0.5) | `topk_mean_per_horizon` k=√n | `inverse_rmse_weights` |
| Very high spread (>0.5) + phase shifts | `dba_combination` | `trimmed_mean` |
| n_unique_winners >= 3 | `best_per_horizon_by_validation` | `topk_mean_per_horizon` |
| One clear dominant model | `best_single_by_validation` | `inverse_rmse_weights` k=3 |
| n_windows >= 6, collinear preds | `ridge_stacking_per_horizon` | `inverse_rmse_weights` |
| Concept drift suspected | `ade_dynamic_error` | `exp_weighted_average` |
| Outlier models present | `robust_median` or `trimmed_mean` | `dba_combination` |
