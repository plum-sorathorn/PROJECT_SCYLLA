/* ============================================================
   PROJECT: SCYLLA // TERMINAL & ML COCKPIT — Unified SPA Script
   Connects to C++ Core on port 8080 and python ML ODP on port 6900.
   ============================================================ */

'use strict';

const API_BASE = (window.location.protocol && window.location.protocol.startsWith('http')) 
  ? `${window.location.protocol}//${window.location.hostname || '127.0.0.1'}:6900`
  : 'http://127.0.0.1:8080';


async function refreshAll() {
  checkHealth();
  if ($('view-tactical') && $('view-tactical').style.display !== 'none') {
    await refreshTactical();
  } else if ($('view-dashboard') && $('view-dashboard').style.display !== 'none') {
    await refreshDashboard();
  } else {
    await refreshML();
  }
}

function setupEventListeners() {
  // Sort controllers
  document.querySelectorAll('#scanner-table th[data-sort]').forEach((th) => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      if (state.sortKey === key) {
        state.sortDir *= -1;
      } else {
        state.sortKey = key;
        state.sortDir = -1;
      }
      
      document.querySelectorAll('#scanner-table th').forEach((h) => {
        h.classList.remove('active-sort');
        const textNode = Array.from(h.childNodes).find(n => n.nodeType === Node.TEXT_NODE);
        if (textNode) {
          textNode.textContent = textNode.textContent.replace(/[▲▼]/g, '').trim() + ' ';
        }
      });
      
      th.classList.add('active-sort');
      const textNode = Array.from(th.childNodes).find(n => n.nodeType === Node.TEXT_NODE);
      if (textNode) {
        textNode.textContent = textNode.textContent.trim() + (state.sortDir === -1 ? ' ▼' : ' ▲');
      }
      
      renderScannerTable();
    });
  });

  // Filter controllers
  $('btn-apply-filter').addEventListener('click', () => {
    const minVol = parseFloat($('filter-minvoloi').value) || 2.0;
    fetchScanner(minVol);
  });

  $('filter-exclude-indices').addEventListener('change', () => {
    renderScannerTable();
  });

  $('filter-min-dte')?.addEventListener('input', (e) => {
    $('val-min-dte').textContent = e.target.value;
    renderScannerTable();
  });

  $('filter-min-premium')?.addEventListener('input', () => {
    renderScannerTable();
  });

  $('filter-min-oi')?.addEventListener('input', () => {
    renderScannerTable();
  });

  $('filter-exclude-weeklies')?.addEventListener('change', () => {
    renderScannerTable();
  });

  $('volcon-ticker').addEventListener('change', (e) => {
    fetchVolCon(e.target.value);
  });

  $('btn-load-iv').addEventListener('click', () => {
    const ticker = $('iv-ticker').value;
    fetchIVSkew(ticker);
  });

  // Global actions
  $('btn-refresh-all').addEventListener('click', () => refreshAll());

  // ML Form & Buttons
  $('btn-run-labeling')?.addEventListener('click', () => runLabeling(false));
  $('btn-force-relabel')?.addEventListener('click', () => runLabeling(true));
  $('btn-trigger-retrain')?.addEventListener('click', runRetraining);
  $('btn-save-settings')?.addEventListener('click', saveSettings);

  // Strategy filter for ledger table
  $('filter-ledger-strategy')?.addEventListener('change', renderLedgerTable);

  // Hash Navigation Routing
  window.addEventListener('hashchange', handleRouting);

  // Backtester Config & Controllers
  $('bt-mode')?.addEventListener('change', (e) => {
    const isWalkforward = e.target.value === 'walkforward';
    if (isWalkforward) {
      $('walkforward-settings-group').style.display = 'grid';
    } else {
      $('walkforward-settings-group').style.display = 'none';
    }
  });

  $('bt-strategy-type')?.addEventListener('change', (e) => {
    const val = e.target.value;
    const group1 = $('bt-max-concurrent-group');
    const group2 = $('bt-scan-time-group');
    if (group1) group1.style.display = 'none';
    if (group2) group2.style.display = 'none';

    // Show/hide strategy-specific fields
    document.querySelectorAll('.strategy-field').forEach(field => {
      const strategies = field.dataset.strategies;
      if (strategies) {
        const strategyList = strategies.split(',');
        field.style.display = strategyList.includes(val) ? 'flex' : 'none';
      }
    });

    // Populate the form from the latest sweep_optimal.json served by the backend,
    // or from the consolidated strategy_defaults.json if sweep_optimal.json is
    // unavailable. Both paths flow through state.optimalParams (single source of
    // truth in backend/config/strategy_defaults.json) — see _buildOptimalParamsFromStrategyDefaults.
    const p = (state.optimalParams && state.optimalParams[val] && state.optimalParams[val].params)
      ? mapApiOptimalToFormInputs(state.optimalParams[val].params)
      : null;
    if (p) {
      if (p.prob !== undefined && $('bt-prob-threshold')) $('bt-prob-threshold').value = p.prob;
      if (p.kelly !== undefined && $('bt-kelly-multiplier')) {
        $('bt-kelly-multiplier').value = p.kelly;
        if ($('kelly-val-display')) $('kelly-val-display').textContent = p.kelly.toFixed(2);
      }
      if (p.stop !== undefined && $('bt-stop-lambda')) {
        $('bt-stop-lambda').value = p.stop;
        if ($('stop-val-display')) $('stop-val-display').textContent = p.stop.toFixed(2);
      }
      if (p.concurrent !== undefined && $('bt-max-concurrent')) $('bt-max-concurrent').value = p.concurrent;
      if (p.hard_stop !== undefined && $('bt-hard-stop-loss')) $('bt-hard-stop-loss').value = p.hard_stop || 0;
      if (p.kelly_cap !== undefined && $('bt-kelly-cap')) $('bt-kelly-cap').value = p.kelly_cap;
      if (p.max_spread !== undefined && $('bt-max-quantile-spread')) $('bt-max-quantile-spread').value = p.max_spread;
      if (p.profit !== undefined && $('bt-profit-threshold')) $('bt-profit-threshold').value = p.profit;
      if (p.median_ret !== undefined && $('bt-min-median-return')) $('bt-min-median-return').value = p.median_ret;
    }
  });

  $('bt-kelly-multiplier')?.addEventListener('input', (e) => {
    $('kelly-val-display').textContent = parseFloat(e.target.value).toFixed(2);
  });

  $('bt-stop-lambda')?.addEventListener('input', (e) => {
    $('stop-val-display').textContent = parseFloat(e.target.value).toFixed(2);
  });

  $('bt-lookback-days')?.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    const display = $('bt-lookback-display');
    if (display) {
      if (val === 0) {
        display.textContent = 'ALL DATA (ENTIRE DATASET)';
      } else if (val >= 365) {
        const yrs = (val / 365).toFixed(1);
        display.textContent = `${yrs} ${yrs === '1.0' ? 'YEAR' : 'YEARS'} (${val} DAYS)`;
      } else {
        const months = Math.round(val / 30);
        display.textContent = `${val} DAYS (${months} ${months === 1 ? 'MONTH' : 'MONTHS'})`;
      }
    }
  });

  $('bt-sweep-enable')?.addEventListener('change', (e) => {
    const isChecked = e.target.checked;
    if (isChecked) {
      $('kelly-single-input').style.display = 'none';
      $('stop-single-input').style.display = 'none';
      $('kelly-sweep-inputs').style.display = 'grid';
      $('stop-sweep-inputs').style.display = 'grid';
      $('kelly-label-text').textContent = 'KELLY MULTIPLIER RANGE';
      $('stop-label-text').textContent = 'STOP MULTIPLIER (λ) RANGE';
      $('kelly-val-display').style.display = 'none';
      $('stop-val-display').style.display = 'none';
    } else {
      $('kelly-single-input').style.display = 'block';
      $('stop-single-input').style.display = 'block';
      $('kelly-sweep-inputs').style.display = 'none';
      $('stop-sweep-inputs').style.display = 'none';
      $('kelly-label-text').textContent = 'KELLY MULTIPLIER';
      $('stop-label-text').textContent = 'STOP MULTIPLIER (λ)';
      $('kelly-val-display').style.display = 'inline';
      $('stop-val-display').style.display = 'inline';
    }
  });

  $('backtest-form')?.addEventListener('submit', runBacktestSimulation);
  $('bt-heatmap-metric')?.addEventListener('change', renderSweepHeatmap);
  $('bt-filter-ticker')?.addEventListener('input', renderBacktestLedgerTable);
  $('bt-filter-exit')?.addEventListener('change', renderBacktestLedgerTable);

  // Sorting for backtest transaction ledger
  document.querySelectorAll('#bt-trades-table th[data-sort-bt]').forEach((th) => {
    th.addEventListener('click', () => {
      const key = th.dataset.sortBt;
      if (state.backtestSortKey === key) {
        state.backtestSortDir *= -1;
      } else {
        state.backtestSortKey = key;
        state.backtestSortDir = -1;
      }
      
      document.querySelectorAll('#bt-trades-table th').forEach((h) => {
        h.classList.remove('active-sort');
        const textNode = Array.from(h.childNodes).find(n => n.nodeType === Node.TEXT_NODE);
        if (textNode) {
          textNode.textContent = textNode.textContent.replace(/[▲▼]/g, '').trim() + ' ';
        }
      });
      
      th.classList.add('active-sort');
      const textNode = Array.from(th.childNodes).find(n => n.nodeType === Node.TEXT_NODE);
      if (textNode) {
        textNode.textContent = textNode.textContent.trim() + (state.backtestSortDir === -1 ? ' ▼' : ' ▲');
      }
      
      renderBacktestLedgerTable();
    });
  });

  // Sorting for open positions / live signals table
  document.querySelectorAll('#open-trades-table th[data-sort-dash]').forEach((th) => {
    th.style.cursor = 'pointer';
    th.addEventListener('click', () => {
      const key = th.dataset.sortDash;
      if (state.dashSortKey === key) {
        state.dashSortDir *= -1;
      } else {
        state.dashSortKey = key;
        state.dashSortDir = -1;
      }
      
      document.querySelectorAll('#open-trades-table th').forEach((h) => {
        h.classList.remove('active-sort');
        const arrowSpan = h.querySelector('.sort-arrow');
        if (arrowSpan) {
          arrowSpan.textContent = '';
        } else {
          const textNode = Array.from(h.childNodes).find(n => n.nodeType === Node.TEXT_NODE);
          if (textNode) textNode.textContent = textNode.textContent.replace(/[▲▼]/g, '').trim() + ' ';
        }
      });
      
      th.classList.add('active-sort');
      const arrowSpan = th.querySelector('.sort-arrow');
      if (arrowSpan) {
        arrowSpan.textContent = state.dashSortDir === -1 ? '▼' : '▲';
      } else {
        const textNode = Array.from(th.childNodes).find(n => n.nodeType === Node.TEXT_NODE);
        if (textNode) textNode.textContent = textNode.textContent.trim() + (state.dashSortDir === -1 ? ' ▼' : ' ▲');
      }
      
      renderOpenTradesTable();
    });
  });

  $('dash-filter-ticker')?.addEventListener('input', renderOpenTradesTable);
  $('dash-filter-strategy')?.addEventListener('change', renderOpenTradesTable);
  $('dash-filter-bt-strategy')?.addEventListener('change', renderOpenTradesTable);
  $('dash-prob-threshold')?.addEventListener('change', fetchOpenTrades);
  $('dash-min-kelly')?.addEventListener('change', fetchOpenTrades);
}

// ── Boot Sequence ────────────────────────────────────────────
(async function boot() {
  const statusEl = $('startup-status-text');
  const startTime = Date.now();
  state.booting = true;

  // Helper to safely dismiss startup loader overlay and re-enable interactive UI controls
  let loaderDismissed = false;
  const dismissLoader = () => {
    if (loaderDismissed) return;
    loaderDismissed = true;
    const loader = $('startup-loader');
    state.booting = false;
    if (loader) {
      loader.classList.add('fade-out');
      setTimeout(() => {
        loader.style.display = 'none';
        document.querySelectorAll('button, input, select, .filter-slider').forEach(el => {
          el.style.pointerEvents = 'auto';
        });
      }, 800);
    } else {
      document.querySelectorAll('button, input, select, .filter-slider').forEach(el => {
        el.style.pointerEvents = 'auto';
      });
    }
  };

  // Hard safety timeout: Dismiss loader after 3.5 seconds max regardless of network latency
  const hardTimeout = setTimeout(dismissLoader, 3500);

  // Disable interactive elements during startup sequence
  document.querySelectorAll('button, input, select, .filter-slider').forEach(el => {
    el.style.pointerEvents = 'none';
  });

  try {
    startClock();
    setupEventListeners();
    initSidebar();
  } catch (e) {
    console.warn('Diagnostics setup exception during boot:', e);
  }

  try {
    await loadOptimalParams();
  } catch (e) {
    console.warn('Error loading optimal params during boot:', e);
  }

  // Trigger strategy change to initialize field visibility
  try {
    const stratEl = $('bt-strategy-type');
    if (stratEl) {
      stratEl.dispatchEvent(new Event('change'));
    }
  } catch (e) {
    console.warn('Error initializing strategy fields:', e);
  }

  // Resolve initial view
  try {
    handleRouting();
  } catch (e) {
    console.warn('Routing exception during boot:', e);
  }

  // Step 1: Health Diagnostics
  if (statusEl) statusEl.textContent = 'RUNNING CORE SYSTEM DIAGNOSTICS...';
  
  let isCoreOnline = false;
  try {
    await checkHealth();
    isCoreOnline = $('status-core')?.classList.contains('online') || $('status-odp')?.classList.contains('online');
  } catch (e) {
    console.warn('Diagnostics exception during boot:', e);
  }

  // Step 2: Update status message based on connectivity
  if (statusEl) {
    if (isCoreOnline) {
      statusEl.textContent = 'CONNECTIVITY RESOLVED. RUNNING SIMULATIONS & RETRIEVING OPTION FLOWS...';
    } else {
      statusEl.textContent = 'ODP CORE OFFLINE. RETRIEVING CACHED FLOWS & SIMULATING BACKTESTS...';
    }
  }

  // Step 3: Trigger main visual telemetry data requests, dashboard, ML config, and backtest simulation in parallel
  const scannerPromise = fetchScanner();
  const pcrPromise = fetchPCR();
  const volConPromise = fetchVolCon('SPY');
  const ivSkewPromise = fetchIVSkew('SPY');
  const swingPromise = fetchSwingAlignment();
  const dashboardPromise = refreshDashboard();
  const mlPromise = refreshML();
  const backtestPromise = loadDefaultBacktestCache();

  // Set loaded flags so view transitions are immediate
  state.dashboardLoaded = true;
  state.mlLoaded = true;

  // Await ALL boot requests to finish loading before dismissing the loading screen
  try {
    await Promise.allSettled([
      scannerPromise,
      pcrPromise,
      volConPromise,
      ivSkewPromise,
      swingPromise,
      dashboardPromise,
      mlPromise,
      backtestPromise
    ]);
  } catch (e) {
    console.warn('Error during boot data fetch:', e);
  }

  // Ensure the loading animation stays visible for smooth presentation
  const elapsed = Date.now() - startTime;
  const minDuration = 1200;
  if (elapsed < minDuration) {
    await new Promise(resolve => setTimeout(resolve, minDuration - elapsed));
  }

  // Step 4: Dismiss loading screen animation AFTER all data is loaded in full
  clearTimeout(hardTimeout);
  dismissLoader();

  // Trigger background sync cycle (5 minutes)
  setInterval(refreshAll, 5 * 60 * 1000);
})();

