'use strict';

async function checkHealth() {
  const coreEl  = $('status-core');
  const odpEl   = $('status-odp');

  try {
    const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    coreEl.classList.toggle('online', r.ok);
    coreEl.classList.toggle('offline', !r.ok);
  } catch {
    coreEl.className = 'status-dot offline';
  }

  try {
    const r = await fetch(`${API_BASE}/api/scanner?_probe=1&limit=1`, { signal: AbortSignal.timeout(4000) });
    odpEl.classList.toggle('online', r.ok);
    odpEl.classList.toggle('offline', !r.ok);
  } catch {
    odpEl.className = 'status-dot offline';
  }
}

async function fetchScanner(minVolOI = 8.0) {
  setLoading('scanner-loading', true);
  try {
    const r = await fetch(`${API_BASE}/api/scanner?min_vol_oi=${minVolOI}`, { signal: AbortSignal.timeout(50000) });
    const json = await r.json();
    state.scannerData = json.data || [];
    updateSummary(json.summary);
    renderScannerTable();
    updateExpectedMovePanel();
  } catch (e) {
    const tbody = $('scanner-tbody');
    if (tbody) {
      const isTimeout = e.name === 'AbortError';
      const msg = isTimeout 
        ? '[ DATA LOAD TIMED OUT — RETRYING SCANNER QUERY ]' 
        : `[ SCANNER OFFLINE — ${e.message || 'Check Connection'} ]`;
      tbody.innerHTML = `<tr><td colspan="15" class="table-empty">${msg}</td></tr>`;
    }
    console.error('Scanner fetch error:', e);
  } finally {
    setLoading('scanner-loading', false);
  }
}

async function fetchPCR() {
  setLoading('pcr-loading', true);
  try {
    const r = await fetch(`${API_BASE}/api/put-call-ratio`, { signal: AbortSignal.timeout(20000) });
    const json = await r.json();
    state.pcrData = json.data || {};
    renderPCRChart();
  } catch (e) {
    console.error('PCR fetch error:', e);
  } finally {
    setLoading('pcr-loading', false);
  }
}

async function fetchVolCon(ticker = 'SPY') {
  setLoading('volcon-loading', true);
  try {
    const r = await fetch(`${API_BASE}/api/volume-concentration?ticker=${ticker}`, { signal: AbortSignal.timeout(20000) });
    const json = await r.json();
    state.volConData = json.data || [];
    renderVolConChart();
  } catch (e) {
    console.error('VolCon fetch error:', e);
  } finally {
    setLoading('volcon-loading', false);
  }
}

async function fetchIVSkew(ticker = 'SPY') {
  setLoading('iv-loading', true);
  try {
    const r = await fetch(`${API_BASE}/api/iv-skew?ticker=${ticker}`, { signal: AbortSignal.timeout(20000) });
    const json = await r.json();
    state.ivData = json;
    renderIVGauges();
    renderIVSmileChart();
  } catch (e) {
    console.error('IV skew fetch error:', e);
  } finally {
    setLoading('iv-loading', false);
  }
}

async function fetchSwingAlignment() {
  const tickers = [...new Set(state.scannerData.map(r => r.ticker))].slice(0, 10);
  if (!tickers.length) {
    $('swing-grid').innerHTML = `<div class="exp-placeholder">[ AWAITING REGISTRY STREAM... ]</div>`;
    return;
  }

  try {
    const seen = new Set();
    const swingRows = state.scannerData.filter(r => {
      if (seen.has(r.ticker)) return false;
      seen.add(r.ticker);
      return true;
    }).slice(0, 12);

    const grid = $('swing-grid');
    if (!swingRows.length) {
      grid.innerHTML = `<div class="exp-placeholder">[ NO DATA PRESENT ]</div>`;
      return;
    }

    grid.innerHTML = swingRows.map(r => {
      const bullish = r.above50dSMA === 1;
      const tickerClass = bullish ? 'accent-gold' : 'accent-coral';
      const flag50 = r.above50dSMA === 1
        ? `<span class="swing-flag-label flag-bull">▲ ABOVE 50d SMA</span>`
        : r.above50dSMA === 0
        ? `<span class="swing-flag-label flag-bear">▼ BELOW 50d SMA</span>`
        : `<span class="swing-flag-label flag-unknown">? 50d SMA</span>`;
      const flag200 = r.above200dSMA === 1
        ? `<span class="swing-flag-label flag-bull">▲ ABOVE 200d SMA</span>`
        : r.above200dSMA === 0
        ? `<span class="swing-flag-label flag-bear">▼ BELOW 200d SMA</span>`
        : `<span class="swing-flag-label flag-unknown">? 200d SMA</span>`;

      return `
        <div class="swing-row">
          <div class="swing-ticker ${tickerClass}">${r.ticker}</div>
          <div class="swing-price">$${fmt(r.underlierPrice)}</div>
          <div class="swing-flags">
            ${flag50}
            ${flag200}
          </div>
        </div>`;
    }).join('');
  } catch (e) {
    console.error('Swing alignment error:', e);
  }
}

async function fetchTacticalBundle(minVolOI, volconTicker, ivTicker) {
  setLoading('scanner-loading', true);
  setLoading('pcr-loading', true);
  setLoading('volcon-loading', true);
  setLoading('iv-loading', true);
  try {
    const params = new URLSearchParams({
      min_vol_oi: String(minVolOI),
      volcon_ticker: volconTicker,
      iv_ticker: ivTicker,
    });
    const r = await fetch(`${API_BASE}/api/tactical-bundle?${params}`, { signal: AbortSignal.timeout(30000) });
    const bundle = await r.json();
    if (bundle.scanner) {
      state.scannerData = bundle.scanner.data || [];
      if (bundle.scanner.summary) updateSummary(bundle.scanner.summary);
      renderScannerTable();
      if (typeof updateExpectedMovePanel === 'function') updateExpectedMovePanel();
    }
    if (bundle.put_call_ratio) {
      state.pcrData = bundle.put_call_ratio.data || {};
      renderPCRChart();
    }
    if (bundle.volume_concentration) {
      state.volConData = bundle.volume_concentration.data || [];
      renderVolConChart();
    }
    if (bundle.iv_skew) {
      state.ivData = bundle.iv_skew;
      renderIVGauges();
      renderIVSmileChart();
    }
  } catch (e) {
    console.error('[SCYLLA] tactical-bundle fetch failed:', e);
  } finally {
    setLoading('scanner-loading', false);
    setLoading('pcr-loading', false);
    setLoading('volcon-loading', false);
    setLoading('iv-loading', false);
  }
}

async function refreshTactical() {
  await Promise.all([
    fetchTacticalBundle(
      parseFloat($('filter-minvoloi').value) || 2.0,
      $('volcon-ticker').value || 'SPY',
      $('iv-ticker').value || 'SPY',
    ),
    fetchSwingAlignment(),
  ]);
}

async function fetchStats() {
  try {
    const r = await fetch(`${API_BASE}/api/ml/stats`, { signal: AbortSignal.timeout(10000) });
    const data = await r.json();
    state.mlStats = data;
    
    const totalTradesEl = $('dash-total-trades');
    if (totalTradesEl) totalTradesEl.textContent = data.total_trades ?? 0;
    
    const labeledTradesEl = $('dash-labeled-trades');
    if (labeledTradesEl) labeledTradesEl.textContent = data.labeled_trades ?? 0;
    
    const winRateEl = $('dash-win-rate');
    if (winRateEl) winRateEl.textContent = fmtPct(data.success_ratio);
    
    const statusEl = $('dash-model-status');
    if (statusEl) {
      if (data.model_ready) {
        statusEl.textContent = 'MODEL READY';
        statusEl.className = 'tile-value accent-call';
      } else {
        statusEl.textContent = 'NO WEIGHTS';
        statusEl.className = 'tile-value accent-coral';
      }
    }
  } catch (e) {
    console.error('Error fetching stats:', e);
  }
}

async function fetchTrades() {
  setLoading('ledger-loading', true);
  try {
    const r = await fetch(`${API_BASE}/api/ml/trades?limit=50`, { signal: AbortSignal.timeout(10000) });
    const json = await r.json();
    state.mlTrades = json.data || [];
    renderLedgerTable();
  } catch (e) {
    const tbody = $('ledger-tbody');
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="11" class="table-empty">[ ERROR: PIPELINE SERVER OFFLINE — CONNECTION REFUSED ]</td></tr>`;
    }
    console.error('Ledger fetch error:', e);
  } finally {
    setLoading('ledger-loading', false);
  }
}

async function runLabeling(forceReLabel = false) {
  const btnId = forceReLabel ? 'btn-force-relabel' : 'btn-run-labeling';
  const btn = $(btnId);
  const prevText = btn.textContent;
  btn.textContent = forceReLabel ? 'RE-LABELING ALL TRADES...' : 'LABELING RUNNING...';
  btn.disabled = true;
  
  if (forceReLabel && !confirm('This will reset and re-compute labels for ALL trades. This may take several minutes. Continue?')) {
    btn.textContent = prevText;
    btn.disabled = false;
    return;
  }
  
  try {
    const horizon = $('cfg-horizon').value || 10;
    const threshold = $('cfg-threshold').value || 0.03;
    const forceParam = forceReLabel ? '&force=true' : '';
    const r = await fetch(`${API_BASE}/api/ml/label?horizon_days=${horizon}&profit_threshold=${threshold}${forceParam}`, { method: 'POST' });
    const res = await r.json();
    showToast(`Labeling run finished. Labeled ${res.labeled_count} trades.`, 'success');
    await refreshML();
  } catch (e) {
    showToast(`Labeling worker failed: ${e.message}`, 'error');
  } finally {
    btn.textContent = prevText;
    btn.disabled = false;
  }
}

async function runRetraining() {
  const btn = $('btn-trigger-retrain');
  const prevText = btn.textContent;
  btn.textContent = 'PIPELINE RETRAINING...';
  btn.disabled = true;
  
  try {
    const r = await fetch(`${API_BASE}/api/ml/train`, { method: 'POST' });
    const res = await r.json();
    
    if (res.status === 'success') {
      const m = res.metrics;
      if ($('sum-train-acc')) $('sum-train-acc').textContent = fmtPct(m.train_accuracy);
      if ($('sum-test-acc')) $('sum-test-acc').textContent = fmtPct(m.test_accuracy);
      
      if ($('sum-cv-roc-auc')) $('sum-cv-roc-auc').textContent = fmtPct(m.cv_roc_auc_mean);
      if ($('sum-test-f1')) $('sum-test-f1').textContent = fmtPct(m.test_f1);
      
      // Render Feature Importances
      renderFeatureImportances(res.feature_importances);
      showToast(`Model retraining successful! Trained on ${m.samples_count} labeled trade instances.`, 'success');
      await refreshML();
    } else {
      showToast(`Retraining returned abnormal status: ${res.message}`, 'error');
    }
  } catch (e) {
    showToast(`Retraining pipeline failed. Make sure scikit-learn is installed in your python backend. Error: ${e.message}`, 'error');
  } finally {
    btn.textContent = prevText;
    btn.disabled = false;
  }
}

async function fetchSettings() {
  try {
    const r = await fetch(`${API_BASE}/api/ml/settings`, { signal: AbortSignal.timeout(10000) });
    const json = await r.json();
    state.mlSettings = json;
    
    if ($('cfg-horizon')) $('cfg-horizon').value = json.horizon_days || '10';
    if ($('cfg-threshold')) $('cfg-threshold').value = Number(json.profit_threshold).toFixed(2);
  } catch (e) {
    console.error('Settings fetch error:', e);
  }
}

async function saveSettings() {
  const horizon = $('cfg-horizon')?.value || '10';
  const threshold = $('cfg-threshold')?.value || '1.0';
  const statusEl = $('settings-status');
  
  try {
    if (statusEl) statusEl.textContent = 'SAVING...';
    const r = await fetch(`${API_BASE}/api/ml/settings?horizon_days=${horizon}&profit_threshold=${threshold}`, {
      method: 'POST',
      signal: AbortSignal.timeout(10000)
    });
    const res = await r.json();
    if (res.status === 'success') {
      if (statusEl) {
        statusEl.textContent = 'CONFIGURATION PERSISTED';
        statusEl.style.color = 'var(--buy)';
        setTimeout(() => {
          statusEl.textContent = '';
          statusEl.style.color = 'var(--text-muted)';
        }, 3000);
      }
      await fetchSettings();
    }
  } catch (e) {
    if (statusEl) {
      statusEl.textContent = 'SAVE FAILED';
      statusEl.style.color = 'var(--put)';
    }
  }
}

async function fetchModelRuns() {
  try {
    const r = await fetch(`${API_BASE}/api/ml/model-runs`, { signal: AbortSignal.timeout(10000) });
    const json = await r.json();
    state.mlModelRuns = json.data || [];
    renderModelRunsTable();
    renderModelRunsChart();
  } catch (e) {
    console.error('Model runs fetch error:', e);
  }
}

async function fetchOpenTrades() {
  setLoading('dashboard-loading', true);
  try {
    const dashProbEl = $('dash-prob-threshold');
    const dashKellyEl = $('dash-min-kelly');
    
    const probThreshold = (dashProbEl && dashProbEl.value !== '' ? parseFloat(dashProbEl.value) : (parseFloat($('bt-prob-threshold')?.value) || 0)) / 100.0;
    const minKellyFraction = (dashKellyEl && dashKellyEl.value !== '' ? parseFloat(dashKellyEl.value) : (parseFloat($('bt-min-kelly-fraction')?.value) || 0)) / 100.0;
    
    const r = await fetch(`${API_BASE}/api/ml/open-trades?prob_threshold=${probThreshold}&min_kelly_fraction=${minKellyFraction}`, { signal: AbortSignal.timeout(45000) });
    const json = await r.json();
    state.dashboardOpenTrades = json.data || [];

    // Update summary tiles
    const openCount = $('dash-open-count');
    if (openCount) openCount.textContent = json.count || 0;

    // Compute avg p_success from trades that have it
    const tradesWithP = state.dashboardOpenTrades.filter(t => t.p_success != null);
    const avgProb = $('dash-avg-prob');
    if (avgProb && tradesWithP.length > 0) {
      const avg = tradesWithP.reduce((s, t) => s + t.p_success, 0) / tradesWithP.length;
      avgProb.textContent = (avg * 100).toFixed(1) + '%';
    } else if (avgProb) {
      avgProb.textContent = '--';
    }

    // Render highest probability hero card
    const topTrade = json.highest_probability;
    const topCard = $('card-top-trade');
    if (topTrade && topCard) {
      topCard.style.display = 'block';
      $('top-trade-ticker').textContent = topTrade.ticker || '--';
      $('top-trade-prob').textContent = topTrade.p_success != null ? (topTrade.p_success * 100).toFixed(1) + '%' : '--';
      const contract = `${topTrade.ticker} ${topTrade.expiration || '--'} ${(topTrade.option_type || '')[0]?.toUpperCase() || ''}${topTrade.strike}`;
      $('top-trade-contract').textContent = contract;
      $('top-trade-kelly').textContent = topTrade.kelly_fraction != null ? (topTrade.kelly_fraction * 100).toFixed(1) + '%' : '--';
      $('top-trade-strategy').textContent = (topTrade.predicted_strategy || '--').replace(/_/g, ' ');
      $('top-trade-date').textContent = topTrade.timestamp ? topTrade.timestamp.split(' ')[0] : '--';

      // Color the probability
      const probEl = $('top-trade-prob');
      if (topTrade.p_success >= 0.70) {
        probEl.style.color = 'var(--call)';
      } else if (topTrade.p_success <= 0.40) {
        probEl.style.color = 'var(--put)';
      } else {
        probEl.style.color = 'var(--accent)';
      }
    } else if (topCard) {
      topCard.style.display = 'none';
    }

    renderOpenTradesTable();
  } catch (e) {
    const tbody = $('open-trades-tbody');
    const msg = e.name === 'AbortError' ? 'Live Yahoo Finance scanner request timed out. Retrying recommended.' : e.message;
    if (tbody) tbody.innerHTML = `<tr><td colspan="12" class="table-empty">[ WARNING: ${msg} ]</td></tr>`;
    console.error('Open trades fetch error:', e);
  } finally {
    setLoading('dashboard-loading', false);
  }
}

async function loadOptimalParams() {
  // Load strategy defaults from consolidated config
  try {
    const r = await fetch(`${API_BASE}/api/ml/strategy-defaults`, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    if (data && data.available && data.config) {
      state.strategyDefaults = data.config;
      console.info(`[strategy-defaults] loaded from ${data.path}`);
    } else {
      state.strategyDefaults = null;
      console.warn('[strategy-defaults] not found — using hardcoded fallback values');
    }
  } catch (e) {
    state.strategyDefaults = null;
    console.warn(`[strategy-defaults] fetch failed: ${e.message} — using fallback values`);
  }

  // Load sweep-optimal params (overrides defaults if available).
  // If /api/ml/optimal-params is unavailable (sweep_optimal.json missing),
  // fall back to the consolidated strategy_defaults served by /api/ml/strategy-defaults
  // (already loaded into state.strategyDefaults above) so the form is still
  // populated from the single source of truth instead of a hardcoded JS map.
  try {
    const r = await fetch(`${API_BASE}/api/ml/optimal-params`, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    if (data && data.available && data.optimal) {
      state.optimalParams = data.optimal;
      state.optimalParamsSource = data.evaluation || 'unknown';
      console.info(`[optimal-params] loaded ${Object.keys(data.optimal).length} strategies from ${data.path}`);
    } else {
      state.optimalParams = _buildOptimalParamsFromStrategyDefaults(state.strategyDefaults);
      state.optimalParamsSource = 'strategy_defaults';
      console.info('[optimal-params] sweep_optimal.json unavailable — using strategy_defaults.json (single source of truth)');
    }
  } catch (e) {
    state.optimalParams = _buildOptimalParamsFromStrategyDefaults(state.strategyDefaults);
    state.optimalParamsSource = 'strategy_defaults';
    console.info(`[optimal-params] fetch failed: ${e.message} — using strategy_defaults.json`);
  }

  // Auto-populate form inputs for initial strategy selection
  const stratEl = $('bt-strategy-type');
  if (stratEl) {
    stratEl.dispatchEvent(new Event('change'));
  }
}

async function loadDefaultBacktestCache() {
  if (state.backtestLoading) return;
  state.booting = true;
  setLoading('backtest-loading', true);
  try {
    const useSyntheticFlag = $('bt-use-synthetic')?.checked ?? true;
    const r = await fetch(`${API_BASE}/api/ml/backtest/default_cache?use_synthetic=${useSyntheticFlag}`);
    if (r.ok) {
      const data = await r.json();
      loadBacktestData(data);
      return;
    }
    console.warn('Default cache endpoint returned non-OK. Skipping automatic simulation on startup.');
    state.backtestLoading = false;
    setLoading('backtest-loading', false);
    // await runBacktestSimulation();
  } catch (err) {
    console.warn('Failed to fetch default backtest cache or run simulation:', err.message);
  } finally {
    setLoading('backtest-loading', false);
    state.booting = false;
  }
  fetchDatasetInfo();
  const synthToggle = $('bt-use-synthetic');
  if (synthToggle && !synthToggle.dataset.scyllaDatasetWired) {
    synthToggle.dataset.scyllaDatasetWired = '1';
    synthToggle.addEventListener('change', () => { fetchDatasetInfo(); });
  }
}

async function fetchDatasetInfo() {
  try {
    const useSyntheticFlag = $('bt-use-synthetic')?.checked ?? true;
    const r = await fetch(`${API_BASE}/api/ml/dataset-info?use_synthetic=${useSyntheticFlag}`);
    if (!r.ok) return;
    const info = await r.json();
    _lastDatasetInfo = info;
    applyDatasetInfoToSlider(info);
  } catch (err) {
    console.warn('fetchDatasetInfo failed:', err.message);
  }
}
