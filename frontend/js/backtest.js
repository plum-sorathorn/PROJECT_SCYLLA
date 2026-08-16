'use strict';

// Strategy numeric defaults live in backend/config/strategy_defaults.json only.
// Frontend populates forms via /api/ml/strategy-defaults → state.strategyDefaults /
// state.optimalParams (see loadOptimalParams + _buildOptimalParamsFromStrategyDefaults).

function mapApiOptimalToFormInputs(params) {
  if (!params) return null;
  const pct = (v) => (v == null ? undefined : Math.round(Number(v) * 1000) / 10);
  const pct2 = (v) => (v == null ? undefined : Math.round(Number(v) * 100));
  return {
    prob:        pct2(params.prob_threshold),
    kelly:       params.kelly_multiplier,
    kelly_cap:   pct2(params.kelly_cap),
    stop:        params.stop_lambda,
    hard_stop:   pct(params.hard_stop_loss),
    concurrent:  params.max_concurrent_trades,
    profit:      pct2(params.profit_threshold),
    median_ret:  pct(params.min_median_return),
    max_spread:  params.max_quantile_spread,
    max_iv:      params.max_iv || 0,
    walkforward_train_window: params.walkforward_train_window,
    walkforward_test_increment: params.walkforward_test_increment,
  };
}

/**
 * Build a synthetic state.optimalParams object from the consolidated
 * strategy_defaults.json (served by /api/ml/strategy-defaults and stored in
 * state.strategyDefaults). Produces the same { [strategy_type]: { params } } shape
 * that the sweep_optimal.json served via /api/ml/optimal-params would produce,
 * so the change handler at L1767 and the submit path can consume both with no
 * branching. This replaces the deleted FALLBACK_OPT_PARAMS hardcoded map.
 *
 * pct2(0.05) === 5 (numbers, not strings) — sanity-checked at call sites.
 */
function _buildOptimalParamsFromStrategyDefaults(strategyDefaults) {
  const out = {};
  if (!strategyDefaults || typeof strategyDefaults !== 'object') return out;
  const strategies = strategyDefaults.strategies || {};
  for (const [name, params] of Object.entries(strategies)) {
    out[name] = { params };
  }
  return out;
}

let _lastDatasetInfo = null;

function applyDatasetInfoToSlider(info) {
  if (!info) return;
  const slider = $('bt-lookback-days');
  if (!slider) return;
  const span = Number(info.data_span_days) || 0;
  if (span <= 0) {
    slider.max = 0;
    slider.value = 0;
    return;
  }
  const step = Number(slider.step) || 30;
  const newMax = Math.max(step, Math.ceil(span / step) * step);
  const curVal = Number(slider.value) || 0;
  slider.max = String(newMax);
  if (curVal > newMax) {
    slider.value = String(newMax);
    const display = $('bt-lookback-display');
    if (display) {
      const yrs = (newMax / 365).toFixed(1);
      display.textContent = `LAST ${yrs} YR (${newMax} D)`;
    }
  }
}


async function runBacktestSimulation(e) {
  console.log('[BACKTEST] Button clicked, event:', e?.type);
  if (e) e.preventDefault();
  console.log('[BACKTEST] state.backtestLoading:', state.backtestLoading);
  if (state.backtestLoading) {
    console.warn('[BACKTEST] Already loading, ignoring click');
    return;
  }
  state.backtestLoading = true;
  setLoading('backtest-loading', true);
  console.log('[BACKTEST] Starting simulation...');
  
  const modeEl = $('bt-mode');
  if (!modeEl) {
    console.error('bt-mode element not found');
    state.backtestLoading = false;
    setLoading('backtest-loading', false);
    return;
  }
  const mode = modeEl.value;
  const initialCapital = parseFloat($('bt-initial-capital').value) || 100000;
  const probThreshold = (parseFloat($('bt-prob-threshold').value) || 40) / 100.0;
  const maxRisk = (parseFloat($('bt-max-risk').value) || 3.0) / 100.0;
  const trainWindow = parseInt($('bt-train-window').value) || 500;
  const testIncrement = parseInt($('bt-test-increment').value) || 100;
  const isSweep = $('bt-sweep-enable').checked;
  const strategyType = $('bt-strategy-type')?.value || 'standard';
  const maxConcurrentTrades = parseInt($('bt-max-concurrent')?.value) || 10;
  const scanTime = $('bt-scan-time')?.value.trim() || '10:00:00';
  const minKellyFraction = (parseFloat($('bt-min-kelly-fraction')?.value) || 1.0) / 100.0;
  const hardStopLossRaw = parseFloat($('bt-hard-stop-loss')?.value) || 3.5;
  const hardStopLoss = hardStopLossRaw > 1.0 ? hardStopLossRaw / 100.0 : hardStopLossRaw;
  const lookbackDaysRaw = parseInt($('bt-lookback-days')?.value) || 0;
  const lookbackDays = lookbackDaysRaw > 0 ? lookbackDaysRaw : null;
  const profitThresholdInput = parseFloat($('bt-profit-threshold')?.value);
  const profitThreshold = !isNaN(profitThresholdInput) ? (profitThresholdInput > 1.0 ? profitThresholdInput / 100.0 : profitThresholdInput) : 0.08;
  // Strategy-specific params — read from dedicated inputs if they exist, else use optimal defaults
  const kellyCapRaw = parseFloat($('bt-kelly-cap')?.value) || 20.0;
  const kellyCap = kellyCapRaw > 1.0 ? kellyCapRaw / 100.0 : kellyCapRaw;
  const maxQuantileSpread = parseFloat($('bt-max-quantile-spread')?.value) || 0.20;
  const minMedianReturnRaw = parseFloat($('bt-min-median-return')?.value) || 2.0;
  const minMedianReturn = minMedianReturnRaw > 1.0 ? minMedianReturnRaw / 100.0 : minMedianReturnRaw;
  const slippagePctRaw = parseFloat($('bt-slippage-pct')?.value);
  const slippagePct = !isNaN(slippagePctRaw) ? (slippagePctRaw > 1.0 ? slippagePctRaw / 100.0 : slippagePctRaw) : 0.01;

  let confirmDirectDev = false;
  if (mode === 'direct_dev') {
    if (!state.directDevConfirmed) {
      showConfirmModal(() => {
        state.directDevConfirmed = true;
        runBacktestSimulation();
      });
      return;
    }
    confirmDirectDev = true;
  }
  
  try {
    if (isSweep) {
      const kellyMin = parseFloat($('bt-kelly-min').value) || 0.1;
      const kellyMax = parseFloat($('bt-kelly-max').value) || 1.0;
      const kellyStep = parseFloat($('bt-kelly-step').value) || 0.2;
      const stopMin = parseFloat($('bt-stop-min').value) || 0.5;
      const stopMax = parseFloat($('bt-stop-max').value) || 2.5;
      const stopStep = parseFloat($('bt-stop-step').value) || 0.5;

      const kellyList = [];
      for (let k = kellyMin; k <= kellyMax; k = parseFloat((k + kellyStep).toFixed(4))) {
        kellyList.push(k);
      }
      const stopList = [];
      for (let s = stopMin; s <= stopMax; s = parseFloat((s + stopStep).toFixed(4))) {
        stopList.push(s);
      }

      const totalCombos = kellyList.length * stopList.length;
      if (totalCombos > 20) {
        if (!state.booting) {
          showToast(`The parameter grid has ${totalCombos} combinations. Please adjust min/max/step values so that total combinations are 20 or fewer.`, 'error');
        } else {
          console.warn(`Boot backtest sweep combo count exceeds 20: ${totalCombos}`);
        }
        state.backtestLoading = false;
        setLoading('backtest-loading', false);
        return;
      }

      const sweepResults = [];
      const promises = [];

      for (const kMult of kellyList) {
        for (const sLam of stopList) {
          const reqBody = {
            mode: mode,
            initial_capital: initialCapital,
            prob_threshold: probThreshold,
            kelly_multiplier: kMult,
            kelly_cap: kellyCap,
            stop_lambda: sLam,
            max_risk_pct_per_trade: maxRisk,
            walkforward_train_window: trainWindow,
            walkforward_test_increment: testIncrement,
            confirm_direct_dev: confirmDirectDev,
            strategy_type: strategyType,
            max_concurrent_trades: maxConcurrentTrades,
            scan_time: scanTime,
            min_kelly_fraction: minKellyFraction,
            hard_stop_loss: hardStopLoss,
            lookback_days: lookbackDays,
            profit_threshold: profitThreshold,
            max_quantile_spread: maxQuantileSpread,
            min_median_return: minMedianReturn,
            slippage_pct: slippagePct,
            use_synthetic: $('bt-use-synthetic')?.checked ?? true
          };

          promises.push(
            fetch(`${API_BASE}/api/ml/backtest`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(reqBody)
            }).then(async r => {
              const data = await r.json();
              if (!r.ok) throw new Error(data.detail || 'Backtest error');
              sweepResults.push({
                kelly: kMult,
                lambda: sLam,
                sharpe: data.summary.sharpe,
                pnl: data.summary.cumulative_pnl_pct,
                data: data
              });
            })
          );
        }
      }

      await Promise.all(promises);
      state.backtestSweepResults = sweepResults;
      
      renderSweepHeatmap();
      
      // Load first cell results as default active
      if (sweepResults.length > 0) {
        // Sort sweep results so they display or activate consistently
        sweepResults.sort((a, b) => b.pnl - a.pnl); // sort descending by PnL
        loadBacktestData(sweepResults[0].data);
      }

      const hmCard = $('card-backtest-heatmap');
      if (hmCard) hmCard.style.display = 'block';
    } else {
      const kellyMultiplier = parseFloat($('bt-kelly-multiplier')?.value) || 0.20;
      const stopLambda = parseFloat($('bt-stop-lambda')?.value) || 1.0;

      const reqBody = {
        mode: mode,
        initial_capital: initialCapital,
        prob_threshold: probThreshold,
        kelly_multiplier: kellyMultiplier,
        kelly_cap: kellyCap,
        stop_lambda: stopLambda,
        max_risk_pct_per_trade: maxRisk,
        walkforward_train_window: trainWindow,
        walkforward_test_increment: testIncrement,
        confirm_direct_dev: confirmDirectDev,
        strategy_type: strategyType,
        max_concurrent_trades: maxConcurrentTrades,
        scan_time: scanTime,
        min_kelly_fraction: minKellyFraction,
        hard_stop_loss: hardStopLoss,
        lookback_days: lookbackDays,
        profit_threshold: profitThreshold,
        max_quantile_spread: maxQuantileSpread,
        min_median_return: minMedianReturn,
        slippage_pct: slippagePct,
        use_synthetic: $('bt-use-synthetic')?.checked ?? true
      };

      const r = await fetch(`${API_BASE}/api/ml/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqBody)
      });
      const data = await r.json();
      if (!r.ok) {
        if (!state.booting) {
          showToast(`Backtest Failed: ${data.detail || 'Unknown error'}`, 'error');
        } else {
          console.error(`Boot backtest failed: ${data.detail || 'Unknown error'}`);
        }
        state.backtestLoading = false;
        setLoading('backtest-loading', false);
        return;
      }

      state.backtestSweepResults = null;
      const hmCard = $('card-backtest-heatmap');
      if (hmCard) hmCard.style.display = 'none';
      loadBacktestData(data);
    }
  } catch (err) {
    showToast(`Simulation failed: ${err.message}`, 'error');
    console.error('Backtest error:', err);
  } finally {
    setLoading('backtest-loading', false);
    state.backtestLoading = false;
    state.directDevConfirmed = false;
  }
}


function loadBacktestData(data) {
  if (!data) return;
  state.backtestResults = data;
  
  const warningBanner = $('bt-warning-banner');
  if (warningBanner) {
    warningBanner.style.display = data.in_sample_warning ? 'flex' : 'none';
  }

  const badgeEl = $('bt-summary-badge');
  if (badgeEl) badgeEl.textContent = (data.mode || 'walkforward').toUpperCase();

  const pnlEl = $('bt-sum-pnl');
  if (pnlEl && data.summary?.cumulative_pnl_pct !== undefined) {
    pnlEl.textContent = data.summary.cumulative_pnl_pct.toFixed(2) + '%';
  }

  const pnlUsdEl = $('bt-sum-pnl-usd');
  if (pnlUsdEl && data.summary?.cumulative_pnl_usd !== undefined) {
    pnlUsdEl.textContent = (data.summary.cumulative_pnl_usd >= 0 ? '+$' : '-$') + Math.abs(data.summary.cumulative_pnl_usd).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
    pnlUsdEl.className = 'tile-value ' + (data.summary.cumulative_pnl_usd >= 0 ? 'accent-call' : 'accent-coral');
  }

  const cagrEl = $('bt-sum-cagr');
  if (cagrEl && data.summary?.cagr_pct !== undefined) {
    cagrEl.textContent = data.summary.cagr_pct.toFixed(2) + '%';
  }

  const sharpeEl = $('bt-sum-sharpe');
  if (sharpeEl && data.summary?.sharpe !== undefined) {
    sharpeEl.textContent = data.summary.sharpe.toFixed(2);
  }

  const sortinoEl = $('bt-sum-sortino');
  if (sortinoEl && data.summary?.sortino !== undefined) {
    sortinoEl.textContent = data.summary.sortino.toFixed(2);
  }

  const winlossEl = $('bt-sum-winloss');
  if (winlossEl && data.summary?.win_loss_ratio !== undefined) {
    winlossEl.textContent = data.summary.win_loss_ratio.toFixed(2);
  }

  const profitfactorEl = $('bt-sum-profitfactor');
  if (profitfactorEl && data.summary?.profit_factor !== undefined) {
    profitfactorEl.textContent = data.summary.profit_factor.toFixed(2);
  }

  const maxddEl = $('bt-sum-maxdd');
  if (maxddEl && data.summary?.max_drawdown_pct !== undefined) {
    maxddEl.textContent = data.summary.max_drawdown_pct.toFixed(2) + '%';
  }

  const triggeredEl = $('bt-sum-triggered');
  if (triggeredEl && data.summary) {
    triggeredEl.textContent = `${data.summary.trades_triggered || 0} / ${data.summary.trades_total_available || 0}`;
  }

  // Win rate (new field)
  const winRateEl = $('bt-sum-winrate');
  if (winRateEl && data.summary?.win_rate_pct !== undefined) {
    winRateEl.textContent = data.summary.win_rate_pct.toFixed(1) + '%';
  }

  // Trade days info for Sharpe transparency
  const tradeDaysEl = $('bt-sum-tradedays');
  if (tradeDaysEl && data.summary?.trade_days_used_for_sharpe !== undefined) {
    tradeDaysEl.textContent = data.summary.trade_days_used_for_sharpe;
  }

  // Data span tile — populated from the backtest response (covers the case
  // where dataset-info wasn't fetched yet, e.g. first backtest before page-load fetch).
  if (data.data_span_days !== undefined) {
    applyDatasetInfoToSlider({
      data_span_days: data.data_span_days,
      data_count: data.data_count,
      is_synthetic_filter: data.is_synthetic_filter,
    });
  }
  const spanEl = $('bt-sum-dataspan');
  const spanSubEl = $('bt-sum-dataspan-sub');
  if (spanEl) {
    const spanDays = Number(data.data_span_days) || 0;
    if (spanDays > 0) {
      const years = spanDays / 365.0;
      spanEl.textContent = years >= 1.0
        ? `${years.toFixed(1)} yr`
        : `${spanDays} d`;
    } else {
      spanEl.textContent = '--';
    }
  }
  if (spanSubEl) {
    const ds = data.data_start || '----';
    const de = data.data_end || '----';
    const cnt = Number(data.data_count) || 0;
    const cntStr = cnt >= 1000 ? `${(cnt / 1000).toFixed(cnt >= 10000 ? 0 : 1)}K` : String(cnt);
    const tag = data.is_synthetic_filter ? 'synthetic' : 'real';
    spanSubEl.textContent = `${ds} → ${de} · ${cntStr} trades (${tag})`;
  }
  
  const sumCard = $('card-backtest-summary');
  if (sumCard) sumCard.style.display = 'block';

  // Display methodology warnings
  const warningsEl = $('bt-warnings-container');
  if (warningsEl && data.warnings && data.warnings.length > 0) {
    warningsEl.innerHTML = data.warnings.map(w => {
      const [tag, ...rest] = w.split(': ');
      const msg = rest.join(': ');
      return `<div class="bt-warning-line"><span class="badge-mid" style="font-size: 9px; margin-right: 6px;">${tag.toUpperCase()}</span><span style="color: var(--text-muted); font-size: 11px;">${msg}</span></div>`;
    }).join('');
    warningsEl.style.display = 'block';
  } else if (warningsEl) {
    warningsEl.style.display = 'none';
  }

  renderBacktestCharts(data);
  const eqCard = $('card-backtest-chart-equity');
  if (eqCard) eqCard.style.display = 'block';
  const ddCard = $('card-backtest-chart-drawdown');
  if (ddCard) ddCard.style.display = 'block';

  renderBacktestLedgerTable();
  const tradesCard = $('card-backtest-trades');
  if (tradesCard) tradesCard.style.display = 'block';
}


function renderBacktestCharts(data) {
  if (state.charts.btEquity) state.charts.btEquity.destroy();
  if (state.charts.btDrawdown) state.charts.btDrawdown.destroy();

  if (!data || !data.equity_curve) return;

  const labels = data.equity_curve.map(pt => pt.date);
  const equityVals = data.equity_curve.map(pt => pt.equity);

  let peak = equityVals[0] || 100000;
  const ddVals = equityVals.map(eq => {
    if (eq > peak) peak = eq;
    return peak > 0 ? -((peak - eq) / peak * 100) : 0;
  });

  const canvasEquity = $('chart-bt-equity');
  if (canvasEquity) {
    const ctxEquity = canvasEquity.getContext('2d');
    state.charts.btEquity = new Chart(ctxEquity, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Portfolio Equity ($)',
          data: equityVals,
          borderColor: '#005566',
          backgroundColor: 'rgba(0, 85, 102, 0.05)',
          borderWidth: 1.5,
          tension: 0.1,
          fill: true,
          pointRadius: labels.length > 100 ? 0 : 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            grid: { color: '#f0f0f0' },
            ticks: { color: '#666666', font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 9 } }
          },
          y: {
            grid: { color: '#f0f0f0' },
            ticks: { color: '#666666', font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 10 } }
          }
        }
      }
    });
  }

  const canvasDD = $('chart-bt-drawdown');
  if (canvasDD) {
    const ctxDD = canvasDD.getContext('2d');
    state.charts.btDrawdown = new Chart(ctxDD, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Drawdown (%)',
          data: ddVals,
          borderColor: '#8b3a3a',
          backgroundColor: 'rgba(139, 58, 58, 0.05)',
          borderWidth: 1.5,
          tension: 0.1,
          fill: true,
          pointRadius: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            grid: { color: '#f0f0f0' },
            ticks: { color: '#666666', font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 9 } }
          },
          y: {
            grid: { color: '#f0f0f0' },
            ticks: {
              color: '#666666',
              font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 10 },
              callback: value => value.toFixed(1) + '%'
            }
          }
        }
      }
    });
  }
}


function renderSweepHeatmap() {
  const container = $('bt-heatmap-container');
  if (!container || !state.backtestSweepResults) return;

  const metric = $('bt-heatmap-metric').value;

  const kellys = [...new Set(state.backtestSweepResults.map(r => r.kelly))].sort((a, b) => b - a);
  const lambdas = [...new Set(state.backtestSweepResults.map(r => r.lambda))].sort((a, b) => a - b);

  container.innerHTML = '';

  container.style.gridTemplateColumns = `80px repeat(${lambdas.length}, 1fr)`;
  container.style.display = 'grid';
  container.style.gap = '3px';

  const topLeft = document.createElement('div');
  topLeft.className = 'heatmap-header-cell';
  topLeft.innerHTML = 'K \\ λ';
  container.appendChild(topLeft);

  for (const lam of lambdas) {
    const lamHeader = document.createElement('div');
    lamHeader.className = 'heatmap-header-cell';
    lamHeader.textContent = lam.toFixed(1);
    container.appendChild(lamHeader);
  }

  const vals = state.backtestSweepResults.map(r => r[metric]);
  const minVal = Math.min(...vals);
  const maxVal = Math.max(...vals);
  const range = maxVal - minVal || 1.0;

  for (const k of kellys) {
    const kLabel = document.createElement('div');
    kLabel.className = 'heatmap-header-cell';
    kLabel.style.justifyContent = 'flex-start';
    kLabel.textContent = k.toFixed(2);
    container.appendChild(kLabel);

    for (const lam of lambdas) {
      const match = state.backtestSweepResults.find(r => r.kelly === k && r.lambda === lam);
      const cell = document.createElement('div');
      cell.className = 'heatmap-cell';
      
      if (match) {
        const val = match[metric];
        cell.textContent = metric === 'pnl' ? val.toFixed(1) + '%' : val.toFixed(2);
        
        let r, g, b, alpha;
        if (val >= 0) {
          const ratio = maxVal > 0 ? val / maxVal : 0;
          r = Math.round(18 + ratio * (143 - 18));
          g = Math.round(19 + ratio * (163 - 19));
          b = Math.round(22 + ratio * (130 - 22));
          alpha = 0.3 + ratio * 0.5;
        } else {
          const ratio = minVal < 0 ? val / minVal : 0;
          r = Math.round(18 + ratio * (191 - 18));
          g = Math.round(19 + ratio * (90 - 19));
          b = Math.round(22 + ratio * (90 - 22));
          alpha = 0.3 + ratio * 0.5;
        }
        
        cell.style.background = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        
        cell.addEventListener('click', () => {
          document.querySelectorAll('.heatmap-cell').forEach(c => c.style.border = '1px solid rgba(255,255,255,0.02)');
          cell.style.border = '1px solid var(--accent)';
          loadBacktestData(match.data);
        });
      } else {
        cell.textContent = '—';
        cell.style.background = 'var(--surface)';
      }
      
      container.appendChild(cell);
    }
  }
}


function renderBacktestLedgerTable() {
  const tbody = $('bt-trades-tbody');
  if (!tbody || !state.backtestResults) return;

  let txs = [...state.backtestResults.transactions];

  const tickerFilter = ($('bt-filter-ticker')?.value || '').trim().toUpperCase();
  if (tickerFilter) {
    txs = txs.filter(t => t.ticker.toUpperCase().includes(tickerFilter));
  }

  const exitFilter = $('bt-filter-exit')?.value || 'ALL';
  if (exitFilter !== 'ALL') {
    txs = txs.filter(t => t.exit_reason === exitFilter);
  }

  txs.sort((a, b) => {
    let av = a[state.backtestSortKey], bv = b[state.backtestSortKey];
    if (typeof av === 'string') av = av.toLowerCase(), bv = bv.toLowerCase();
    if (av === undefined || av === null) return 1;
    if (bv === undefined || bv === null) return -1;
    return (av < bv ? -1 : av > bv ? 1 : 0) * state.backtestSortDir;
  });

  if (txs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="table-empty">[ NO TRANSACTIONS FOUND MATCHING CRITERIA ]</td></tr>`;
    return;
  }

  tbody.innerHTML = txs.map(t => {
    const isProfit = t.exit_reason === 'profit_hit';
    const isStop = t.exit_reason === 'stop_hit';
    
    let statusBadge;
    if (isProfit) {
      statusBadge = '<span class="badge-buy">PROFIT HIT</span>';
    } else if (isStop) {
      statusBadge = '<span class="badge-sell">STOP HIT</span>';
    } else {
      statusBadge = '<span class="badge-mid">EXPIRED</span>';
    }
      
    const pnlClass = t.pnl_usd >= 0 ? 'accent-call' : 'accent-coral';
    const returnClass = t.observed_return >= 0 ? 'accent-call' : 'accent-coral';

    return `
      <tr>
        <td style="font-weight: 600;">${t.ticker}</td>
        <td>${t.trade_date}</td>
        <td style="font-family: var(--font-mono); font-size: 11px;">${t.contract}</td>
        <td>${(t.p_success * 100).toFixed(1)}%</td>
        <td>${(t.kelly_fraction * 100).toFixed(1)}%</td>
        <td style="font-family: var(--font-mono);">$${t.position_size_usd.toLocaleString()}</td>
        <td style="font-family: var(--font-mono);">${(t.max_adverse_return * 100).toFixed(2)}%</td>
        <td class="${returnClass}" style="font-family: var(--font-mono);">${t.observed_return >= 0 ? '+' : ''}${(t.observed_return * 100).toFixed(2)}%</td>
        <td class="${pnlClass}" style="font-family: var(--font-mono); font-weight: 600;">${t.pnl_usd >= 0 ? '+$' : '-$'}${Math.abs(t.pnl_usd).toLocaleString()}</td>
        <td>${statusBadge}</td>
      </tr>
    `;
  }).join('');
}

// ── Toast Notification System ─────────────────────────────────
