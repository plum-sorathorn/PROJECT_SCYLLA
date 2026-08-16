# Sweep and Gate Ablation Handoff

## 1. Context

`whale_quality` regressed to approximately -100% PnL. The RCA verdict is:

1. Fix the gate architecture first.
2. Re-tune on the current synthetic-only dataset.
3. Replace the strategy only if it is still dead after the ablation and re-tune.

The fail-open synthetic whale/IQR gates have been fixed. Already-fixed files:

- `backend/config/strategy_defaults.json`: added per-strategy
  `real_vol_oi_floor`, `synth_vol_oi_floor`, and
  `synth_max_quantile_spread` floors/caps.
- `backend/config/_strategy_loader.py`: centralized loading of the JSON defaults.
- `backend/routers/ml_model.py`: strategy defaults now resolve through the
  loader; whale gates fail closed; the new gate values are applied; and the
  backtest response cache key includes all three new gate values.
- `frontend/js/api.js`: Unusual Activity Registry request timeout increased from
  25,000 ms to 50,000 ms.
- `backend/routers/_yf_safe.py`: yfinance safe-call timeout increased from 12 s
  to 24 s.

Still open: the gate ablation has not been run, and the optimal parameters must
be re-tuned on the current synthetic-only five-ensemble dataset. The sweep was
previously using test increment 100 while production defaults use 500; this
handoff changes the sweep to read the common configured value (currently 500).
Do not use the old +115.24% result as a regression target: it was measured on
the pre-purge dataset at increment 100.

## 2. STEP 1: Run the gate ablation FIRST

Start the backend in a separate terminal:

```powershell
cd C:\Users\plum\Documents\FIN Works\PROJECT_SCYLLA\backend
.venv\Scripts\Activate.ps1
uvicorn main:app --host 127.0.0.1 --port 6900
```

Health check:

```powershell
curl http://127.0.0.1:6900/health
```

Run each POST and save the response exactly as shown:

```powershell
curl.exe -X POST http://127.0.0.1:6900/api/ml/backtest -H "Content-Type: application/json" -d '{"mode":"walkforward","strategy_type":"whale_quality"}' -o backend/cache/ablation_G0.json
curl.exe -X POST http://127.0.0.1:6900/api/ml/backtest -H "Content-Type: application/json" -d '{"mode":"walkforward","strategy_type":"whale_quality","use_synthetic":true,"synth_vol_oi_floor":5.0,"synth_max_quantile_spread":0.132}' -o backend/cache/ablation_G1.json
curl.exe -X POST http://127.0.0.1:6900/api/ml/backtest -H "Content-Type: application/json" -d '{"mode":"walkforward","strategy_type":"whale_quality","use_synthetic":true,"synth_vol_oi_floor":0.0,"synth_max_quantile_spread":99.0}' -o backend/cache/ablation_G2.json
```

**Warning:** each is a cold run and can take up to approximately 30 minutes.
The prediction cache was invalidated by the cache-key change. Do not kill a run
early.

Interpret the results as follows:

| Observation | Interpretation |
|---|---|
| G0 escapes -100% | The fix worked. Proceed to the sweep. |
| G0 is still -100% | The whale gate was not the root cause; escalate to dataset/dead-strategy hypotheses and consider replacement. |
| G0 trade count is near the sweep's 123, not thousands | The whale population is restored. |
| G0 equals G1 | Resolution from `strategy_defaults.json` is working. |
| G2 reproduces -100% | Causation is confirmed. |
| G2 is identical to G0 | The cache key is still broken; stop and fix it before continuing. |

## 3. STEP 2: Re-run the sweep

Only run this after the ablation passes:

```powershell
cd C:\Users\plum\Documents\FIN Works\PROJECT_SCYLLA
backend\.venv\Scripts\Activate.ps1
python scripts\sweep_strategies_v2.py
```

The script accepts no CLI arguments. `SCYLLA_MAX_WORKERS` is an environment
variable used by its process-pool cap, but the sweep itself currently sets its
outer stage worker count to 1. It imports `api_backtest` directly and runs
standalone against the local database/cache; uvicorn on port 6900 is **not** a
prerequisite. The activated environment and backend dependencies are required.

The script now reads both `walkforward_train_window` and
`walkforward_test_increment` from `backend/config/_strategy_loader.py`; both
currently resolve to 500, matching `strategy_defaults.json`.

It writes `backend/cache/sweep_optimal_v2.json`. The schema is an object keyed
by strategy (`whale_quality`, `contrarian_trend`, `vol_regime`). A successful
entry contains `params`, `metrics` (Sharpe, Sortino, win rate, trades, return,
drawdown, profit factor, and score), and `evaluation:
"out-of-sample_walkforward"`. A failed entry contains `params: null`,
`metrics: null`, and an `error`.

Static runnability/gate review:

- It sets `use_synthetic=True` and `use_costs=True`, and uses the local DB
  through `api_backtest`; it does not call the HTTP backend.
- It does not sweep the new floor fields. Because omitted request fields are
  resolved by `ml_model.py`, the current strategy defaults are applied during
  each run, including `real_vol_oi_floor`, `synth_vol_oi_floor`, and
  `synth_max_quantile_spread`. Thus it is gate-aware for the shipped defaults,
  but not a floor-ablation or floor-parameter sweep.
- Minimal future change if floors themselves must be tuned: add the three floor
  fields to the parameter ranges/strategy grids and pass them explicitly into
  `BacktestRequestSchema`; do not make that change as part of this handoff.
- The script logs elapsed seconds per coarse/refined stage, but documents no
  total runtime estimate. It runs 15 coarse plus up to 24 refined combinations
  per strategy (up to 117 backtests total), so expect a long run; cold backtests
  can individually take up to approximately 30 minutes.
- No dead `docs/` path, deleted-real-row reference, or hardcoded ensemble count
  was found in this script. It does intentionally use a 15,000-row subset and
  the three fixed active strategy names.

## 4. STEP 3: Write the results back

After reviewing the sweep and ablation, write new optimal values per strategy
to `backend/config/strategy_defaults.json`:

- `prob_threshold`
- `kelly_multiplier`
- `kelly_cap`
- `hard_stop_loss`
- `stop_lambda`
- `max_quantile_spread`
- `min_median_return`
- `max_iv`
- `max_concurrent_trades`
- `profit_threshold`
- `real_vol_oi_floor`
- `synth_vol_oi_floor`
- `synth_max_quantile_spread`

Keep the sweep artifact in `backend/cache/sweep_optimal_v2.json`. This is a
gitignore exception and **is tracked**.

`strategy_defaults.json` is the single source of truth. The frontend reads it
through `/api/ml/strategy-defaults`; HTML form defaults mirror the `vol_regime`
block as a last resort.

C++ is not wired to this config. `cpp_core/include/data_fetcher.h` has its own
`minVolOI=2.0`; update it by hand only if C++ thresholds are intended to match
the Python strategy thresholds.

## 5. Known open issues / gotchas

- `contrarian_trend.synth_max_quantile_spread` is currently `0.0`. This is
  harmless today because that block has no IQR gate, but it would block every
  trade if one is added. Correct it to mirror its `max_quantile_spread` before
  adding such a gate.
- Unreviewed working-tree scope creep: `frontend/index.html` and
  `frontend/js/backtest.js` are modified, and `.gemini/settings.json` is
  deleted. Review these before committing.
- RCA-latent bugs not yet fixed: possible concurrent over-allocation (up to 10
  positions may size from the same pre-debit `current_equity`); possible
  mismatch between `calibration_target_pct` and `profit_threshold` (0.404), so
  Kelly may optimize a different success event; the `dte<=2` time-stop can
  realize a deep `entry_mar` loss bypassing the `-0.197` hard stop; and four
  disagreeing whale thresholds exist across layers (2.0 / 5.0 / 8.0 / training
  filter 0.05-10).
- Baseline caveat: +115% was measured on the pre-purge dataset at increment
  100. Do not treat it as a regression target. Success means positive return,
  profit factor > 1.2, maximum drawdown under approximately 35%, and a trade
  count in the same order as approximately 123.
