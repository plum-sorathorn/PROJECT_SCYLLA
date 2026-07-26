/* ============================================================
   PROJECT: SCYLLA // TERMINAL & ML COCKPIT — Unified SPA Script
   Connects to C++ Core on port 8080 and python ML ODP on port 6900.
   ============================================================ */

'use strict';

const API_BASE = (window.location.protocol && window.location.protocol.startsWith('http')) 
  ? `${window.location.protocol}//${window.location.hostname || '127.0.0.1'}:6900`
  : 'http://127.0.0.1:6900';

// ── State ──────────────────────────────────────────────────
const state = {
  // Application startup state
  booting: false,

  // Tactical Console State
  scannerData: [],
  pcrData: {},
  volConData: [],
  ivData: null,
  sortKey: 'volOiRatio',
  sortDir: -1,   // -1 = descending
  selectedRows: new Set(),
  charts: {},

  // ML Cockpit State (now lives in Backtest page)
  mlStats: {},
  mlTrades: [],
  mlImportance: [],
  mlModelMetrics: {},
  mlModelRuns: [],
  mlSettings: {},
  mlLoaded: false,

  // Dashboard State
  dashboardOpenTrades: [],
  dashboardLoaded: false,
  dashSortKey: 'timestamp',
  dashSortDir: -1,

  // Backtester State
  backtestResults: null,
  backtestSweepResults: null,
  backtestSortKey: 'trade_date',
  backtestSortDir: -1,
  backtestLoading: false,
  directDevConfirmed: false,

  // Optimal params loaded from /api/ml/optimal-params (scripts/sweep_optimal.json)
  optimalParams: null,
  optimalParamsSource: 'pending',
};

// ── Utility ─────────────────────────────────────────────────
const fmt = (n, d = 2) => (n == null ? '—' : Number(n).toFixed(d));
const fmtPct = (n) => (n == null ? '—' : (Number(n) * 100).toFixed(1) + '%');
const fmtK = (n) => {
  if (n == null) return '—';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + 'K';
  return String(n);
};
const $ = (id) => document.getElementById(id);

// ── SPA View Router ─────────────────────────────────────────
function showView(viewName) {
  const titleEl = $('header-title-text');
  const subEl = $('header-sub-text');
  const tacticalStrip = $('tactical-summary-strip');
  const tacticalView = $('view-tactical');
  const dashboardView = $('view-dashboard');
  const backtestView = $('view-backtest');

  // Hide all views first
  if (tacticalView) tacticalView.style.display = 'none';
  if (tacticalStrip) tacticalStrip.style.display = 'none';
  if (dashboardView) dashboardView.style.display = 'none';
  if (backtestView) backtestView.style.display = 'none';

  if (viewName === 'dashboard') {
    if (dashboardView) dashboardView.style.display = 'block';

    titleEl.innerHTML = 'PROJECT: SCYLLA <span class="header-sep">//</span> <span style="font-style: italic;">Live Signals</span>';
    subEl.textContent = 'LIVE SCANNER FLOW & ML SIGNAL ANALYSIS — REAL-TIME OPPORTUNITY RANKING';

    if (!state.dashboardLoaded) {
      refreshDashboard();
      state.dashboardLoaded = true;
    }
  } else if (viewName === 'backtest') {
    if (backtestView) backtestView.style.display = 'block';

    titleEl.innerHTML = 'PROJECT: SCYLLA <span class="header-sep">//</span> <span style="font-style: italic;">Backtester</span>';
    subEl.textContent = 'STRATEGY EVALUATION & WALKFOWARD PERFORMANCE SIMULATOR';

    // Load ML ops data into backtest page if not already loaded
    if (!state.mlLoaded) {
      refreshML();
      state.mlLoaded = true;
    }

    // Auto-load #1 strategy cached simulation on initial display
    if (!state.backtestResults && !state.backtestLoading) {
      state.booting = true;
      loadDefaultBacktestCache();
    }
  } else {
    if (tacticalView) tacticalView.style.display = 'block';
    if (tacticalStrip) tacticalStrip.style.display = 'block';

    titleEl.innerHTML = 'PROJECT: SCYLLA <span class="header-sep">//</span> <span style="font-style: italic;">Main Dashboard</span>';
    subEl.textContent = 'VOLATILITY & FLOW TELEMETRY — POWERED BY C++ CORE & OpenBB ODP';

    // Re-render tactical charts on display
    renderPCRChart();
    renderVolConChart();
    renderIVSmileChart();
  }

  // Update active state in sidebar nav
  document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
  const activeNavItem = $(`nav-${viewName}`);
  if (activeNavItem) {
    activeNavItem.classList.add('active');
  }
}

function handleRouting() {
  const hash = window.location.hash;
  if (hash === '#dashboard') {
    showView('dashboard');
  } else if (hash === '#backtest') {
    showView('backtest');
  } else {
    showView('tactical');
  }
}

// ── Black-Scholes Pricing Model ──────────────────────────────
function cnd(x) {
  const a1 = 0.319381530;
  const a2 = -0.356563782;
  const a3 = 1.781477937;
  const a4 = -1.821255978;
  const a5 = 1.330274429;
  const L = Math.abs(x);
  const K = 1.0 / (1.0 + 0.2316419 * L);
  let w = 1.0 - 1.0 / Math.sqrt(2.0 * Math.PI) * Math.exp(-L * L / 2.0) * (a1 * K + a2 * K * K + a3 * Math.pow(K, 3) + a4 * Math.pow(K, 4) + a5 * Math.pow(K, 5));
  if (x < 0.0) {
    w = 1.0 - w;
  }
  return w;
}

function blackScholes(type, S, K, T_days, sigma_pct, r = 0.045) {
  const T = T_days / 365.0; // time in years
  const sigma = sigma_pct / 100.0; // volatility as decimal
  
  if (T <= 0 || sigma <= 0) {
    return type === 'Call' ? Math.max(0, S - K) : Math.max(0, K - S);
  }
  
  const d1 = (Math.log(S / K) + (r + sigma * sigma / 2.0) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  
  if (type === 'Call') {
    return S * cnd(d1) - K * Math.exp(-r * T) * cnd(d2);
  } else {
    return K * Math.exp(-r * T) * cnd(-d2) - S * cnd(-d1);
  }
}

function loadTradeIntoPredictor(trade, source) {
  const ticker = (trade.ticker || '').toUpperCase();
  state.activeTicker = ticker;
}

// ── Clock ───────────────────────────────────────────────────
function startClock() {
  const el = $('header-clock');
  const tick = () => {
    const now = new Date();
    el.textContent = now.toLocaleTimeString('en-US', { hour12: false });
  };
  tick();
  setInterval(tick, 1000);
}

// ── Health checks ────────────────────────────────────────────
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

// ── Loading helpers ──────────────────────────────────────────
function setLoading(id, active) {
  const el = $(id);
  if (el) el.classList.toggle('active', active);

  if (id === 'backtest-loading') {
    state.backtestLoading = !!active;
    const btn = $('btn-run-backtest');
    if (btn) {
      btn.disabled = !!active;
      btn.style.opacity = active ? '0.6' : '1';
      btn.style.cursor = active ? 'not-allowed' : 'pointer';
      btn.style.pointerEvents = active ? 'none' : 'auto';
      btn.innerHTML = active 
        ? '<span class="spinner" style="display:inline-block; width:12px; height:12px; border:2px solid rgba(255,255,255,0.3); border-top-color:#fff; border-radius:50%; animation:spin 0.8s linear infinite; margin-right:8px;"></span> SIMULATING...' 
        : 'RUN BACKTEST SIMULATION';
    }
  }
}


// ══════════════════════════════════════════════════════════════
// TACTICAL CONSOLE LOGIC
// ══════════════════════════════════════════════════════════════

// ── WIDGET A: Unusual Options Scanner ───────────────────────
async function fetchScanner(minVolOI = 8.0) {
  setLoading('scanner-loading', true);
  try {
    const r = await fetch(`${API_BASE}/api/scanner?min_vol_oi=${minVolOI}`, { signal: AbortSignal.timeout(25000) });
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

// ── Summary statistics strip ──────────────────────────────────
function updateSummary(summary) {
  if (!summary) return;
  $('sum-whales').textContent  = summary.whaleSignalCount ?? '—';
  $('sum-maxvoloi').textContent = fmt(summary.maxVolOI) + 'x';
  $('sum-avgvoloi').textContent = fmt(summary.avgVolOI) + 'x';
  $('sum-agg-pcr').textContent  = fmt(summary.aggregatePCR, 3);
  $('sum-callvol').textContent  = fmtK(summary.totalCallVolume);
  $('sum-putvol').textContent   = fmtK(summary.totalPutVolume);
}

// ── Scanner Table Renderer ────────────────────────────────────
function renderScannerTable() {
  const tbody = $('scanner-tbody');
  let data = [...state.scannerData];

  // 1. Exclude Indices / ETFs if checkbox is checked
  const excludeIndices = $('filter-exclude-indices')?.checked;
  if (excludeIndices) {
    const INDEX_TICKERS = ['SPY', 'QQQ', 'IWM', 'DIA', 'ARKK', 'VIX'];
    data = data.filter(row => !INDEX_TICKERS.includes(row.ticker.toUpperCase()));
  }

  // 2. Interactive filter: Days to Expiration (DTE)
  const minDte = parseInt($('filter-min-dte')?.value) || 0;
  data = data.filter(row => row.dte === undefined || row.dte >= minDte);

  // 3. Interactive filter: Option Premium (Notional value)
  const minPremium = parseFloat($('filter-min-premium')?.value) || 0;
  data = data.filter(row => row.premium === undefined || row.premium >= minPremium);

  // 4. Interactive filter: Minimum Open Interest (OI)
  const minOi = parseInt($('filter-min-oi')?.value) || 0;
  data = data.filter(row => row.openInterest === undefined || row.openInterest >= minOi);

  // 5. Interactive filter: Exclude Weeklies
  const excludeWeeklies = $('filter-exclude-weeklies')?.checked;
  if (excludeWeeklies) {
    data = data.filter(row => row.isWeekly === false);
  }

  // Sort logic
  data.sort((a, b) => {
    let av = a[state.sortKey], bv = b[state.sortKey];
    if (typeof av === 'string') av = av.toLowerCase(), bv = bv.toLowerCase();
    if (av === undefined || av === null) return 1;
    if (bv === undefined || bv === null) return -1;
    return (av < bv ? -1 : av > bv ? 1 : 0) * state.sortDir;
  });

  if (!data.length) {
    tbody.innerHTML = `<tr><td colspan="15" class="table-empty">[ NO CONTRACT FLOWS DETECTED MATCHING SEARCH VECTORS ]</td></tr>`;
    return;
  }

  const rows = data.map((row, idx) => {
    const isWhale = row.isWhaleSignal;
    const typeEl = row.optionType === 'Call'
      ? `<span class="badge-call">CALL</span>`
      : `<span class="badge-put">PUT</span>`;

    const smaFlag = (val, label) => {
      if (val === 1)  return `<span class="flag-bull">▲ ${label}</span>`;
      if (val === 0)  return `<span class="flag-bear">▼ ${label}</span>`;
      return `<span class="flag-unknown">? ${label}</span>`;
    };

    const trendClass = {
      'BULL_ALIGNED':   'trend-bull-aligned',
      'BEAR_ALIGNED':   'trend-bear-aligned',
      'BULL_CONTRARIAN': 'trend-bull-contra',
      'NEUTRAL':         'trend-neutral',
      'UNKNOWN':         'trend-unknown',
    }[row.trendAlignment] || 'trend-neutral';

    const em = row.expectedMove > 0
      ? `<span class="em-plus">+${fmt(row.expectedMove)}</span> / <span class="em-minus">-${fmt(row.expectedMove)}</span>`
      : '—';

    const lastTrade = row.lastTradeDate ? row.lastTradeDate.replace(' ', '<br>') : '—';
    const actionEl = row.side === 'BUY'
      ? `<span class="badge-buy">BUY</span>`
      : row.side === 'SELL'
        ? `<span class="badge-sell">SELL</span>`
        : `<span class="badge-mid">MID</span>`;

    // Muted grey style for DTE indicator
    const expText = row.dte != null
      ? `${row.expiration}<br><span style="font-size: 10px; color: var(--text-muted); font-weight: 500;">${row.dte}d${row.isWeekly ? ' (W)' : ' (M)'}</span>`
      : row.expiration;

    // Gold accent color for option premium to draw visual focus
    const volText = row.premium != null
      ? `${fmtK(row.volume)}<br><span style="font-size: 10px; color: var(--accent); font-weight: 500;">$${fmtK(row.premium)}</span>`
      : fmtK(row.volume);

    return `
      <tr class="${isWhale ? 'whale-row' : ''}" data-ticker="${row.ticker}" data-em="${row.expectedMove}" data-price="${row.underlierPrice}" data-idx="${idx}">
        <td><strong>${row.ticker}</strong></td>
        <td>${expText}</td>
        <td>$${fmt(row.strike)}</td>
        <td>${typeEl}</td>
        <td>${volText}</td>
        <td>${fmtK(row.openInterest)}</td>
        <td class="${isWhale ? 'accent-gold' : ''}">${row.volOiRatio === 9999 ? '∞' : fmt(row.volOiRatio)}x</td>
        <td style="font-size: 10px; line-height: 1.1; color: var(--text-muted);">${lastTrade}</td>
        <td>${actionEl}</td>
        <td>${fmt(row.impliedVolatility)}%</td>
        <td>$${fmt(row.underlierPrice)}</td>
        <td>${smaFlag(row.above50dSMA, '50d')}</td>
        <td>${smaFlag(row.above200dSMA, '200d')}</td>
        <td class="${trendClass}">${(row.trendAlignment || '—').replace('_', ' ')}</td>
        <td>${em}</td>
      </tr>`;
  });

  tbody.innerHTML = rows.join('');

  // Row click → update expected move panel, other widgets, and populate ML console
  tbody.querySelectorAll('tr[data-ticker]').forEach((tr) => {
    tr.addEventListener('click', () => {
      document.querySelectorAll('#scanner-table tr, #ledger-table tr').forEach(r => r.classList.remove('active-row'));
      tr.classList.add('active-row');

      const ticker = tr.dataset.ticker;
      const em = parseFloat(tr.dataset.em);
      const price = parseFloat(tr.dataset.price);
      if (ticker) {
        addToExpectedMovePanel(ticker, em, price);
        updateAssetWidgets(ticker);

        const idx = parseInt(tr.dataset.idx);
        const trade = data[idx];
        if (trade) {
          loadTradeIntoPredictor(trade, 'REGISTRY');
        }
      }
    });
  });
}

function updateAssetWidgets(ticker) {
  const formattedTicker = ticker.toUpperCase();

  // Helper to ensure ticker option exists and select it
  const selectElement = (selectId) => {
    const select = $(selectId);
    if (!select) return false;

    let exists = false;
    for (let i = 0; i < select.options.length; i++) {
      if (select.options[i].value === formattedTicker) {
        exists = true;
        select.selectedIndex = i;
        break;
      }
    }

    if (!exists) {
      const option = document.createElement('option');
      option.value = formattedTicker;
      option.textContent = formattedTicker;
      select.appendChild(option);
      select.value = formattedTicker;
    }
    return true;
  };

  // 1. Update Widget C (Volume Concentration)
  if (selectElement('volcon-ticker')) {
    fetchVolCon(formattedTicker);
  }

  // 2. Update Extension 1 (IV Sandbox & Skew Finder)
  if (selectElement('iv-ticker')) {
    fetchIVSkew(formattedTicker);
  }
}

// ── WIDGET B: Put/Call Ratio ─────────────────────────────────
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

function renderPCRChart() {
  const ctx = $('chart-pcr');
  if (!ctx) return;

  const colors = { SPY: '#005566', QQQ: '#2d7a4a', IWM: '#8b3a3a' };
  const tickers = ['SPY', 'QQQ', 'IWM'];

  const allLabels = [];
  tickers.forEach((t) => {
    (state.pcrData[t] || []).forEach((pt) => {
      if (!allLabels.includes(pt.date)) allLabels.push(pt.date);
    });
  });
  allLabels.sort();

  const datasets = tickers.map((t) => {
    const pts = state.pcrData[t] || [];
    const dataMap = Object.fromEntries(pts.map(p => [p.date, p.putCallRatio]));
    return {
      label: t,
      data: allLabels.map(d => dataMap[d] ?? null),
      borderColor: colors[t],
      backgroundColor: colors[t] + '10',
      borderWidth: 1.5,
      pointRadius: 2.5,
      pointHoverRadius: 4.5,
      tension: 0.3,
      fill: false,
      spanGaps: true,
    };
  });

  if (state.charts.pcr) state.charts.pcr.destroy();

  state.charts.pcr = new Chart(ctx, {
    type: 'line',
    data: { labels: allLabels, datasets },
    options: chartDefaults({
      title: 'Put/Call Ratio Historical Trend',
      yLabel: 'P/C Ratio',
    }),
  });
}

// ── WIDGET C: Volume Concentration ──────────────────────────
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

function renderVolConChart() {
  const ctx = $('chart-volcon');
  if (!ctx) return;

  const labels = state.volConData.map(d => d.expiration);
  const callVols = state.volConData.map(d => d.callVolume);
  const putVols  = state.volConData.map(d => d.putVolume);

  if (state.charts.volcon) state.charts.volcon.destroy();

  state.charts.volcon = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Call Volume',
          data: callVols,
          backgroundColor: '#e8f5e9',
          borderColor: '#2d7a4a',
          borderWidth: 0.5,
          stack: 'vol',
        },
        {
          label: 'Put Volume',
          data: putVols,
          backgroundColor: '#ffebee',
          borderColor: '#8b3a3a',
          borderWidth: 0.5,
          stack: 'vol',
        },
      ],
    },
    options: chartDefaults({ title: 'Expiration Volume Distribution', yLabel: 'Volume', stacked: true }),
  });
}

// ── EXTENSION 1: IV Skew ─────────────────────────────────────
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

function renderIVGauges() {
  const d = state.ivData;
  if (!d) return;
  $('iv-current').textContent = `${fmt(d.currentIV)}%`;
  $('iv-rank').textContent    = `${fmt(d.ivRank)}%`;
  $('iv-pct').textContent     = `${fmt(d.ivPercentile)}%`;
  $('iv-rank-bar').style.width    = `${Math.min(d.ivRank, 100)}%`;
  $('iv-pct-bar').style.width     = `${Math.min(d.ivPercentile, 100)}%`;

  const ivRankEl = $('iv-rank');
  if (d.ivRank >= 75) ivRankEl.className = 'gauge-value accent-coral';
  else if (d.ivRank >= 50) ivRankEl.className = 'gauge-value accent-gold';
  else ivRankEl.className = 'gauge-value accent-silver';
}

function renderIVSmileChart() {
  const ctx = $('chart-ivsmile');
  if (!ctx || !state.ivData) return;

  const smile = state.ivData.smileData || [];
  const expirations = [...new Set(smile.map(p => p.expiration))].slice(0, 2);
  const colors = ['#005566', '#2d7a4a'];

  const datasets = expirations.map((exp, i) => {
    const pts = smile
      .filter(p => p.expiration === exp && p.optionType === 'Call')
      .sort((a, b) => a.strike - b.strike);
    return {
      label: exp,
      data: pts.map(p => ({ x: p.strike, y: p.iv })),
      borderColor: colors[i % 2],
      backgroundColor: colors[i % 2] + '10',
      borderWidth: 1.5,
      pointRadius: 2,
      showLine: true,
      tension: 0.4,
    };
  });

  if (state.charts.ivsmile) state.charts.ivsmile.destroy();

  state.charts.ivsmile = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: chartDefaults({ title: 'Implied Volatility Smile Profile', xLabel: 'Strike', yLabel: 'IV%' }),
  });
}

// ── EXTENSION 3: Expected Move Panel ─────────────────────────
function updateExpectedMovePanel() {
  const whales = state.scannerData
    .filter(r => r.isWhaleSignal && r.expectedMove > 0)
    .reduce((acc, r) => {
      if (!acc.find(x => x.ticker === r.ticker)) acc.push(r);
      return acc;
    }, [])
    .slice(0, 8);

  const grid = $('exp-move-grid');
  if (!grid) return;

  if (!whales.length) {
    grid.innerHTML = `<div class="exp-placeholder">[ NO INSTANT WHALE BLOCKS REGISTERED WITH VALUATION BOUNDS ]</div>`;
    return;
  }

  grid.innerHTML = whales.map(r => `
    <div class="exp-move-row">
      <div class="exp-ticker">${r.ticker}</div>
      <div class="exp-move-details">
        <div class="exp-move-range">
          <span class="em-plus">+$${fmt(r.expectedMove)}</span>
          &nbsp;/&nbsp;
          <span class="em-minus">-$${fmt(r.expectedMove)}</span>
        </div>
        <div class="exp-move-meta">
          PRICE: $${fmt(r.underlierPrice)} &nbsp;|&nbsp;
          RANGE: [$${fmt(r.underlierPrice - r.expectedMove)}&nbsp;–&nbsp;$${fmt(r.underlierPrice + r.expectedMove)}]
        </div>
      </div>
    </div>`).join('');
}

function addToExpectedMovePanel(ticker, em, price) {
  if (em <= 0) return;
  const grid = $('exp-move-grid');
  const existing = grid.querySelector(`[data-ticker="${ticker}"]`);
  if (existing) return;

  const placeholder = grid.querySelector('.exp-placeholder');
  if (placeholder) placeholder.remove();

  const div = document.createElement('div');
  div.className = 'exp-move-row';
  div.dataset.ticker = ticker;
  div.innerHTML = `
    <div class="exp-ticker">${ticker}</div>
    <div class="exp-move-details">
      <div class="exp-move-range">
        <span class="em-plus">+$${fmt(em)}</span>
        &nbsp;/&nbsp;
        <span class="em-minus">-$${fmt(em)}</span>
      </div>
      <div class="exp-move-meta">PRICE: $${fmt(price)} &nbsp;|&nbsp; RANGE: [$${fmt(price - em)}&nbsp;–&nbsp;$${fmt(price + em)}]</div>
    </div>`;
  grid.prepend(div);
}

// ── EXTENSION 2: Swing Alignment ─────────────────────────────
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

// ── Chart defaults factory ───────────────────────────────────
function chartDefaults({ title = '', xLabel = '', yLabel = '', stacked = false } = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 600, easing: 'easeInOutQuart' },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#ffffff',
        borderColor: '#e0e0e0',
        borderWidth: 0.5,
        titleColor: '#005566',
        bodyColor: '#1a1a1a',
        titleFont: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 11, weight: 'bold' },
        bodyFont: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 11 },
        padding: 10,
      },
    },
    scales: {
      x: {
        stacked,
        grid: { color: '#f0f0f0', drawBorder: false },
        ticks: {
          color: '#666666',
          font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 9 },
          maxTicksLimit: 8,
          maxRotation: 45,
        },
        border: { color: '#e0e0e0' },
        ...(xLabel ? { title: { display: true, text: xLabel, color: '#666666', font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 9 } } } : {}),
      },
      y: {
        stacked,
        grid: { color: '#f0f0f0', drawBorder: false },
        ticks: {
          color: '#666666',
          font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 9 },
        },
        border: { color: '#e0e0e0' },
        ...(yLabel ? { title: { display: true, text: yLabel, color: '#666666', font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 9 } } } : {}),
      },
    },
  };
}

async function refreshTactical() {
  await Promise.all([
    fetchScanner(parseFloat($('filter-minvoloi').value) || 2.0),
    fetchPCR(),
    fetchVolCon($('volcon-ticker').value || 'SPY'),
    fetchIVSkew($('iv-ticker').value || 'SPY'),
  ]);
  await fetchSwingAlignment();
}


// ══════════════════════════════════════════════════════════════
// MACHINE LEARNING COCKPIT LOGIC
// ══════════════════════════════════════════════════════════════

// ── Fetch ML Stats ───────────────────────────────────────────
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

// ── Fetch Logged Trades (Ledger) ─────────────────────────────
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

// ── Render Ledger Table ──────────────────────────────────────
function renderLedgerTable() {
  const tbody = $('ledger-tbody');
  
  if (!state.mlTrades.length) {
    tbody.innerHTML = `<tr><td colspan="12" class="table-empty">[ NO HISTORICAL OPTIONS LOGS RECORDED IN DATABASE ]</td></tr>`;
    return;
  }
  
  const strategyFilter = $('filter-ledger-strategy')?.value || 'ALL';
  let filteredTrades = [...state.mlTrades];
  if (strategyFilter !== 'ALL') {
    filteredTrades = filteredTrades.filter(row => row.predicted_strategy === strategyFilter);
  }
  
  if (!filteredTrades.length) {
    tbody.innerHTML = `<tr><td colspan="12" class="table-empty">[ NO HISTORICAL OPTIONS LOGS MATCHING SELECTED STRATEGY ]</td></tr>`;
    return;
  }
  
  const rows = filteredTrades.map((row, idx) => {
    const typeEl = row.option_type === 'Call'
      ? `<span class="badge-call">CALL</span>`
      : `<span class="badge-put">PUT</span>`;
      
    let outcomeEl = '<span class="badge-mid" style="background: rgba(123, 125, 130, 0.1); border-color: var(--text-muted); color: var(--text-muted);">PENDING</span>';
    if (row.labeled === 1) {
      if (row.label_success === 1) {
        outcomeEl = '<span class="badge-buy" style="box-shadow: 0 0 2px var(--buy);">SUCCESS</span>';
      } else {
        outcomeEl = '<span class="badge-sell" style="box-shadow: 0 0 2px var(--sell);">FAILURE</span>';
      }
    }
    
    const retVal = row.observed_return;
    const retEl = retVal != null
      ? `<span class="${retVal >= 0 ? 'em-plus' : 'em-minus'}">${retVal >= 0 ? '+' : ''}${fmtPct(retVal)}</span>`
      : '—';
      
    const predP50 = row.predicted_p50;
    const predP50El = predP50 != null
      ? `<span style="font-family: var(--font-mono); color: var(--accent);">${predP50 >= 0 ? '+' : ''}${(predP50 * 100).toFixed(1)}%</span>`
      : '—';
      
    const logDate = row.timestamp ? row.timestamp.split(' ')[0] : '—';
    const evalDate = row.evaluation_date || '—';
    
    return `
      <tr data-idx="${idx}">
        <td style="font-family: var(--font-mono);">${logDate}</td>
        <td><strong>${row.ticker}</strong></td>
        <td>${row.expiration}</td>
        <td style="font-family: var(--font-mono);">$${fmt(row.strike)}</td>
        <td>${typeEl}</td>
        <td style="font-family: var(--font-mono);">${fmt(row.vol_oi_ratio)}x</td>
        <td style="font-family: var(--font-mono); font-weight: 500; color: var(--accent);">$${fmtK(row.premium)}</td>
        <td style="font-family: var(--font-mono);">$${fmt(row.underlier_price)}</td>
        <td style="font-family: var(--font-mono);">${evalDate}</td>
        <td style="font-family: var(--font-mono);">${predP50El}</td>
        <td style="font-family: var(--font-mono);">${retEl}</td>
        <td>${outcomeEl}</td>
      </tr>
    `;
  });
  
  tbody.innerHTML = rows.join('');

  // Row click listener to load into Inference Console
  tbody.querySelectorAll('tr[data-idx]').forEach((tr) => {
    tr.addEventListener('click', () => {
      document.querySelectorAll('#scanner-table tr, #ledger-table tr').forEach(r => r.classList.remove('active-row'));
      tr.classList.add('active-row');
      
      const idx = parseInt(tr.dataset.idx);
      const trade = filteredTrades[idx];
      if (trade) {
        loadTradeIntoPredictor(trade, 'LEDGER');
      }
    });
  });
}

// ── Run Labeling Worker ──────────────────────────────────────
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
    alert(`Labeling run finished. Labeled ${res.labeled_count} trades.`);
    await refreshML();
  } catch (e) {
    alert(`Labeling worker failed: ${e.message}`);
  } finally {
    btn.textContent = prevText;
    btn.disabled = false;
  }
}

// ── Run Retraining Pipeline ──────────────────────────────────
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
      $('sum-train-acc').textContent = fmtPct(m.train_accuracy);
      $('sum-test-acc').textContent = fmtPct(m.test_accuracy);
      
      if ($('sum-cv-roc-auc')) $('sum-cv-roc-auc').textContent = fmtPct(m.cv_roc_auc_mean);
      if ($('sum-test-f1')) $('sum-test-f1').textContent = fmtPct(m.test_f1);
      
      // Render Feature Importances
      renderFeatureImportances(res.feature_importances);
      alert(`Model retraining successful! Trained on ${m.samples_count} labeled trade instances.`);
      await refreshML();
    } else {
      alert(`Retraining returned abnormal status: ${res.message}`);
    }
  } catch (e) {
    alert(`Retraining pipeline failed. Make sure scikit-learn is installed in your python backend. Error: ${e.message}`);
  } finally {
    btn.textContent = prevText;
    btn.disabled = false;
  }
}

// ── Render Feature Importance Bars ───────────────────────────
function renderFeatureImportances(list) {
  const wrapper = $('feature-list-wrapper');
  if (!list || !list.length) {
    wrapper.innerHTML = `<div class="exp-placeholder">[ NO FEATURE IMPORTANCE DATA AVAILABLE ]</div>`;
    return;
  }
  
  const rows = list.map((item) => {
    const cleanLabel = item.feature.toUpperCase().replace('_', ' ');
    const pctVal = (item.importance * 100).toFixed(1) + '%';
    const barWidth = (item.importance * 100) + '%';
    
    return `
      <div class="feature-bar-row" style="display: flex; flex-direction: column; gap: 4px;">
        <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 10px;">
          <span>${cleanLabel}</span>
          <span class="accent-gold">${pctVal}</span>
        </div>
        <div style="height: 6px; background: var(--border); border-radius: var(--radius); overflow: hidden;">
          <div style="height: 100%; background: var(--accent); width: ${barWidth}; transition: width 0.5s ease-out;"></div>
        </div>
      </div>
    `;
  });
  
  wrapper.innerHTML = rows.join('');
}

// Obsolete manual inference logic removed.

// ── Fetch settings ───────────────────────────────────────────
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

// ── Save settings ────────────────────────────────────────────
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

// ── Fetch model runs ─────────────────────────────────────────
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

// ── Render model runs table ──────────────────────────────────
function renderModelRunsTable() {
  const tbody = $('training-runs-tbody');
  if (!tbody) return;
  
  if (!state.mlModelRuns.length) {
    tbody.innerHTML = `<tr><td colspan="9" class="table-empty">[ NO TRAINING LOGS FOUND ]</td></tr>`;
    return;
  }
  
  const rows = state.mlModelRuns.map((run) => {
    return `
      <tr>
        <td style="font-family: var(--font-mono);">${run.timestamp}</td>
        <td style="font-family: var(--font-mono);">${run.samples_count}</td>
        <td style="font-family: var(--font-mono);">${fmtPct(run.train_accuracy)}</td>
        <td style="font-family: var(--font-mono);">${fmtPct(run.test_accuracy)}</td>
        <td style="font-family: var(--font-mono);">${fmtPct(run.test_precision)}</td>
        <td style="font-family: var(--font-mono);">${fmtPct(run.test_recall)}</td>
        <td style="font-family: var(--font-mono);">${fmtPct(run.test_f1)}</td>
        <td style="font-family: var(--font-mono);">${fmtPct(run.test_roc_auc)}</td>
        <td style="font-family: var(--font-mono);">${fmtPct(run.cv_roc_auc_mean)}</td>
      </tr>
    `;
  });
  
  tbody.innerHTML = rows.join('');
}

// ── Render model runs chart ──────────────────────────────────
let runsChart = null;
function renderModelRunsChart() {
  const canvas = $('chart-training-runs');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  // Sort runs chronologically ascending by timestamp
  const sortedRuns = [...state.mlModelRuns].sort((a, b) => 
    new Date(a.timestamp || 0) - new Date(b.timestamp || 0)
  );

  const labels = sortedRuns.map(run => {
    if (!run.timestamp) return '--';
    const d = new Date(run.timestamp);
    return isNaN(d.getTime()) ? run.timestamp : d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  });
  
  const p10Loss = sortedRuns.map(run => run.pinball_loss_p10);
  const p25Loss = sortedRuns.map(run => run.pinball_loss_p25);
  const p50Loss = sortedRuns.map(run => run.pinball_loss_p50);
  const p75Loss = sortedRuns.map(run => run.pinball_loss_p75);
  const p90Loss = sortedRuns.map(run => run.pinball_loss_p90);
  const rocData = sortedRuns.map(run => run.test_roc_auc);
  
  if (runsChart) {
    runsChart.destroy();
  }
  
  runsChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'P10 Pinball Loss',
          data: p10Loss,
          borderColor: 'rgba(244, 63, 94, 0.8)', // Coral/red
          backgroundColor: 'transparent',
          tension: 0,
          yAxisID: 'y'
        },
        {
          label: 'P25 Pinball Loss',
          data: p25Loss,
          borderColor: 'rgba(249, 115, 22, 0.8)', // Orange
          backgroundColor: 'transparent',
          tension: 0,
          yAxisID: 'y'
        },
        {
          label: 'P50 (Median) Loss',
          data: p50Loss,
          borderColor: 'rgba(6, 182, 212, 0.8)', // Cyan
          backgroundColor: 'transparent',
          tension: 0,
          yAxisID: 'y'
        },
        {
          label: 'P75 Pinball Loss',
          data: p75Loss,
          borderColor: 'rgba(16, 185, 129, 0.8)', // Green
          backgroundColor: 'transparent',
          tension: 0,
          yAxisID: 'y'
        },
        {
          label: 'P90 Pinball Loss',
          data: p90Loss,
          borderColor: 'rgba(217, 70, 239, 0.8)', // Purple
          backgroundColor: 'transparent',
          tension: 0,
          yAxisID: 'y'
        },
        {
          label: 'Test ROC AUC (RHS)',
          data: rocData,
          borderColor: 'rgba(0, 85, 102, 0.9)', // Teal
          backgroundColor: 'transparent',
          borderDash: [5, 5],
          tension: 0,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#666666', font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 9 } }
        }
      },
      scales: {
        x: {
          grid: { color: '#f0f0f0' },
          ticks: { color: '#666666', font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' } }
        },
        y: {
          position: 'left',
          grid: { color: '#f0f0f0' },
          ticks: { color: '#666666', font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' } },
          title: {
            display: true,
            text: 'Pinball Loss',
            color: '#666666',
            font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 10 }
          }
        },
        y1: {
          position: 'right',
          grid: { drawOnChartArea: false },
          min: 0,
          max: 1.0,
          ticks: { color: 'rgba(0, 85, 102, 0.7)', font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' } },
          title: {
            display: true,
            text: 'ROC AUC',
            color: 'rgba(0, 85, 102, 0.7)',
            font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 10 }
          }
        }
      }
    }
  });
}

async function refreshML() {
  await Promise.all([
    fetchSettings(),
    fetchModelRuns()
  ]);
}

// ── Dashboard: Fetch and render open trades ──────────────────
async function refreshDashboard() {
  await Promise.all([
    fetchOpenTrades(),
    fetchStats()
  ]);
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
        probEl.style.color = '#8FA382';
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

const STRATEGY_DEFAULT_PARAMS = {
  quantile_confidence: { prob: 40, kelly: 0.75, kelly_cap: 12, stop: 1.5, hard_stop: 25, max_spread: 0.20, median_ret: 0, max_iv: 0, profit: 27.5 },
  trend_breakout: { prob: 38, kelly: 0.90, kelly_cap: 12, stop: 1.5, hard_stop: 25, max_spread: 0, median_ret: 2, max_iv: 0, profit: 30 },
  iv_regime_adaptive: { prob: 38, kelly: 0.75, kelly_cap: 12, stop: 1.5, hard_stop: 25, max_spread: 0, median_ret: 0, max_iv: 0, profit: 30 },
};

function getStrategyParams(strategyType) {
  if (state.optimalParams && state.optimalParams[strategyType] && state.optimalParams[strategyType].params) {
    return mapApiOptimalToFormInputs(state.optimalParams[strategyType].params);
  }
  if (state.strategyDefaults && state.strategyDefaults.strategies && state.strategyDefaults.strategies[strategyType]) {
    const s = state.strategyDefaults.strategies[strategyType];
    return mapApiOptimalToFormInputs(s);
  }
  return FALLBACK_OPT_PARAMS[strategyType] || null;
}

function isEligibleForStrategy(trade, strategyType) {
  if (!strategyType || strategyType === 'ALL') return true;
  const p = getStrategyParams(strategyType);
  if (!p) return true;

  const pSuccess = Number(trade.p_success) || 0;
  const p10 = Number(trade.predicted_p10) || 0;
  const p50 = Number(trade.predicted_p50) || 0;
  const p90 = Number(trade.predicted_p90) || 0;
  const iqr = p90 - p10;
  const iv = Number(trade.implied_vol) || 0;
  const optType = trade.option_type || '';
  const side = trade.side || '';
  const trend = trade.trend_alignment || '';

  const probThreshold = (p.prob != null ? p.prob : 40) / 100.0;
  const minMedianReturn = (p.median_ret != null ? p.median_ret : 3) / 100.0;
  const maxQuantileSpread = p.max_spread || 5.0;
  const maxIv = p.max_iv || 0;

  if (strategyType === 'quantile_confidence') {
    return pSuccess >= probThreshold && iqr <= maxQuantileSpread;
  }
  if (strategyType === 'trend_breakout') {
    const isBull = (optType === 'Call' && trend === 'BULL_ALIGNED');
    const isBear = (optType === 'Put' && trend === 'BEAR_ALIGNED');
    return pSuccess >= probThreshold && p50 >= minMedianReturn && (isBull || isBear);
  }
  if (strategyType === 'iv_regime_adaptive') {
    const curProb = iv >= 30.0 ? Math.max(probThreshold, 0.50) : probThreshold;
    return pSuccess >= curProb;
  }
  return pSuccess >= probThreshold;
}

function renderOpenTradesTable() {
  const tbody = $('open-trades-tbody');
  if (!tbody) return;

  let trades = [...state.dashboardOpenTrades];

  // Filter by ticker
  const tickerFilter = $('dash-filter-ticker')?.value?.trim().toUpperCase();
  if (tickerFilter) {
    trades = trades.filter(t => (t.ticker || '').toUpperCase().includes(tickerFilter));
  }

  // Filter by ML-predicted strategy regime (SIDEWAYS / BULLISH_BREAKOUT / etc.)
  const strategyFilter = $('dash-filter-strategy')?.value || 'ALL';
  if (strategyFilter !== 'ALL') {
    trades = trades.filter(t => t.predicted_strategy === strategyFilter);
  }

  // Filter by backtester strategy type — mirrors api_backtest entry rules.
  // Uses each strategy's optimal OOS parameters from sweep_optimal.json.
  const btStrategyFilter = $('dash-filter-bt-strategy')?.value || 'ALL';
  const bannerEl = $('dash-bt-strategy-banner');
  const bannerTextEl = $('dash-strategy-banner-text');
  const bannerMetricsEl = $('dash-strategy-banner-metrics');

  if (btStrategyFilter !== 'ALL') {
    const totalBefore = trades.length;
    trades = trades.filter(t => isEligibleForStrategy(t, btStrategyFilter));
    const countEl = $('dash-bt-strategy-count');
    if (countEl) countEl.textContent = `${trades.length}/${totalBefore} MATCH`;

    // Render Strategy Threshold Banner
    if (bannerEl) {
      bannerEl.style.display = 'flex';
      const p = getStrategyParams(btStrategyFilter);
      const metrics = (state.optimalParams && state.optimalParams[btStrategyFilter] && state.optimalParams[btStrategyFilter].metrics);
      const evalType = (state.optimalParams && state.optimalParams[btStrategyFilter] && state.optimalParams[btStrategyFilter].evaluation) || 'OOS';

      if (bannerTextEl && p) {
        const rules = [
          `MIN P%: ${p.prob}%`,
          `KELLY MULT: ${p.kelly}`,
          `KELLY CAP: ${p.kelly_cap}%`,
          p.max_spread ? `MAX SPREAD: ${p.max_spread}` : null,
          p.median_ret ? `MIN MEDIAN RET: ${p.median_ret}%` : null,
        ].filter(Boolean).join(' | ');
        bannerTextEl.innerHTML = `<strong>ACTIVE STRATEGY [${btStrategyFilter.toUpperCase().replace(/_/g, ' ')}]:</strong> ${rules}`;
      }

      if (bannerMetricsEl && metrics) {
        bannerMetricsEl.innerHTML = `[${evalType.toUpperCase()}] SHARPE: <strong>${fmt(metrics.sharpe)}</strong> | WIN RATE: <strong>${fmtPct(metrics.win_rate / 100)}</strong> | RETURN: <strong>+${fmt(metrics.total_return)}%</strong>`;
      } else if (bannerMetricsEl) {
        bannerMetricsEl.innerHTML = ``;
      }
    }
  } else {
    const countEl = $('dash-bt-strategy-count');
    if (countEl) countEl.textContent = '';
    if (bannerEl) bannerEl.style.display = 'none';
  }

  // Update Hero Card for top trade matching selected strategy filter
  const topCard = $('card-top-trade');
  if (topCard) {
    const topTrade = trades.length > 0
      ? trades.reduce((prev, curr) => ((curr.p_success || 0) > (prev.p_success || 0) ? curr : prev), trades[0])
      : null;

    if (topTrade) {
      topCard.style.display = 'block';
      if ($('top-trade-ticker')) $('top-trade-ticker').textContent = topTrade.ticker || '--';
      if ($('top-trade-prob')) $('top-trade-prob').textContent = topTrade.p_success != null ? (topTrade.p_success * 100).toFixed(1) + '%' : '--';
      const contract = `${topTrade.ticker} ${topTrade.expiration || '--'} ${(topTrade.option_type || '')[0]?.toUpperCase() || ''}${topTrade.strike}`;
      if ($('top-trade-contract')) $('top-trade-contract').textContent = contract;
      if ($('top-trade-kelly')) $('top-trade-kelly').textContent = topTrade.kelly_fraction != null ? (topTrade.kelly_fraction * 100).toFixed(1) + '%' : '--';
      if ($('top-trade-strategy')) $('top-trade-strategy').textContent = (topTrade.predicted_strategy || '--').replace(/_/g, ' ');
      if ($('top-trade-date')) $('top-trade-date').textContent = topTrade.timestamp ? topTrade.timestamp.split(' ')[0] : '--';

      const probEl = $('top-trade-prob');
      if (probEl) {
        if (topTrade.p_success >= 0.70) probEl.style.color = '#8FA382';
        else if (topTrade.p_success <= 0.40) probEl.style.color = 'var(--put)';
        else probEl.style.color = 'var(--accent)';
      }
    } else {
      topCard.style.display = 'none';
    }
  }

  // Sort
  trades.sort((a, b) => {
    let av = a[state.dashSortKey], bv = b[state.dashSortKey];
    if (typeof av === 'string') av = av.toLowerCase(), bv = bv.toLowerCase();
    if (av === undefined || av === null) return 1;
    if (bv === undefined || bv === null) return -1;
    return (av < bv ? -1 : av > bv ? 1 : 0) * state.dashSortDir;
  });

  if (trades.length === 0) {
    const stratName = btStrategyFilter !== 'ALL' ? btStrategyFilter.toUpperCase().replace(/_/g, ' ') : 'THIS FILTER';
    tbody.innerHTML = `<tr><td colspan="12" class="table-empty">[ NO SIGNALS CURRENTLY MEET ENTRY RULES FOR ${stratName} ]</td></tr>`;
    return;
  }

  tbody.innerHTML = trades.map(row => {
    const typeEl = row.option_type === 'Call'
      ? `<span class="badge-call">CALL</span>`
      : `<span class="badge-put">PUT</span>`;

    const sideEl = row.side === 'BUY'
      ? `<span class="badge-buy">BUY</span>`
      : (row.side === 'SELL' ? `<span class="badge-sell">SELL</span>` : `<span class="badge-mid">MID</span>`);

    const pSuccess = row.p_success != null ? (row.p_success * 100).toFixed(1) + '%' : '--';
    const pClass = row.p_success >= 0.70 ? 'accent-call' : (row.p_success <= 0.40 ? 'accent-coral' : '');
    const kellyStr = row.kelly_fraction != null ? (row.kelly_fraction * 100).toFixed(1) + '%' : '--';
    const stratStr = (row.predicted_strategy || '--').replace(/_/g, ' ');
    let logDate = '--';
    if (row.timestamp) {
      const parts = row.timestamp.split(' ');
      if (parts.length >= 2) {
        logDate = `${parts[0]} <span style="color: var(--text-muted); font-size: 10px;">${parts[1]}</span>`;
      } else {
        logDate = row.timestamp;
      }
    }

    return `
      <tr>
        <td style="font-family: var(--font-mono); font-size: 11px; white-space: nowrap;">${logDate}</td>
        <td><strong>${row.ticker}</strong></td>
        <td>${row.expiration || '--'}</td>
        <td style="font-family: var(--font-mono);">$${fmt(row.strike)}</td>
        <td>${typeEl}</td>
        <td>${sideEl}</td>
        <td style="font-family: var(--font-mono);">${fmt(row.vol_oi_ratio)}x</td>
        <td style="font-family: var(--font-mono); font-weight: 500; color: var(--accent);">$${fmtK(row.premium)}</td>
        <td style="font-family: var(--font-mono);">${row.dte || '--'}</td>
        <td style="font-family: var(--font-mono); font-weight: 600;" class="${pClass}">${pSuccess}</td>
        <td style="font-family: var(--font-mono); color: var(--buy);">${kellyStr}</td>
        <td style="font-family: var(--font-mono); font-size: 10px;">${stratStr}</td>
      </tr>
    `;
  }).join('');
}


// ══════════════════════════════════════════════════════════════
// CONTROLS & BOOT STRAP
// ══════════════════════════════════════════════════════════════

// ── Refresh Active View ──────────────────────────────────────
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

// ── Wire Interactive Events ──────────────────────────────────
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

// ── Optimal params (from scripts/sweep_optimal.json via /api/ml/optimal-params) ──
// Units: sweep_optimal.json stores all values as fractions (0-1) for thresholds
// and absolute numbers for things like max_concurrent_trades. The HTML inputs
// expect prob/kelly_cap/hard_stop/median_ret in percent (×100), kelly/stop_lambda
// as fractions, max_spread/max_iv as absolute, profit_threshold in percent.
const FALLBACK_OPT_PARAMS = {
  'quantile_confidence': { prob: 40, kelly: 0.75, kelly_cap: 12, stop: 1.5, hard_stop: 25, concurrent: 8, profit: 27.5, median_ret: 0,  max_spread: 0.20, max_iv: 0 },
  'trend_breakout':      { prob: 38, kelly: 0.90, kelly_cap: 12, stop: 1.5, hard_stop: 25, concurrent: 8, profit: 30, median_ret: 2,  max_spread: 0, max_iv: 0 },
  'iv_regime_adaptive':  { prob: 38, kelly: 0.75, kelly_cap: 12, stop: 1.5, hard_stop: 25, concurrent: 8, profit: 30, median_ret: 0,  max_spread: 0, max_iv: 0 },
};

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
  };
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

  // Load sweep-optimal params (overrides defaults if available)
  try {
    const r = await fetch(`${API_BASE}/api/ml/optimal-params`, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    if (data && data.available && data.optimal) {
      state.optimalParams = data.optimal;
      state.optimalParamsSource = data.evaluation || 'unknown';
      console.info(`[optimal-params] loaded ${Object.keys(data.optimal).length} strategies from ${data.path}`);
    } else {
      state.optimalParams = null;
      state.optimalParamsSource = 'unavailable';
      console.warn('[optimal-params] sweep_optimal.json not found — using fallback (in-sample) values');
    }
  } catch (e) {
    state.optimalParams = null;
    state.optimalParamsSource = 'unavailable';
    console.warn(`[optimal-params] fetch failed: ${e.message} — using fallback values`);
  }

  // Auto-populate form inputs for initial strategy selection
  const stratEl = $('bt-strategy-type');
  if (stratEl) {
    stratEl.dispatchEvent(new Event('change'));
  }
}


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

    // Populate the form from the latest sweep_optimal.json served by the backend.
    // Falls back to a hardcoded map (in-sample-only, not for trading) if the file is missing.
    const p = (state.optimalParams && state.optimalParams[val] && state.optimalParams[val].params)
      ? mapApiOptimalToFormInputs(state.optimalParams[val].params)
      : (FALLBACK_OPT_PARAMS[val] || null);
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


// ══════════════════════════════════════════════════════════════
// SIDEBAR COLLAPSE & BACKTEST ENGINE CLIENT INTERACTION
// ══════════════════════════════════════════════════════════════

function initSidebar() {
  const sidebar = $('sidebar');
  const toggleBtn = $('sidebar-toggle');
  
  if (!sidebar || !toggleBtn) return;
  
  // Load state from localStorage
  const isCollapsed = localStorage.getItem('scylla:sidebar_collapsed') === 'true';
  if (isCollapsed) {
    sidebar.classList.add('collapsed');
  } else {
    sidebar.classList.remove('collapsed');
  }
  
  toggleBtn.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    const nowCollapsed = sidebar.classList.contains('collapsed');
    localStorage.setItem('scylla:sidebar_collapsed', nowCollapsed);
  });
}

function showConfirmModal(onConfirm) {
  const modal = $('confirm-modal');
  const btnConfirm = $('btn-modal-confirm');
  const btnCancel = $('btn-modal-cancel');
  
  if (!modal) return;
  modal.style.display = 'flex';
  
  // Clone button to strip existing event listeners cleanly
  const newConfirm = btnConfirm.cloneNode(true);
  btnConfirm.replaceWith(newConfirm);
  
  newConfirm.addEventListener('click', () => {
    modal.style.display = 'none';
    onConfirm();
  });
  
  btnCancel.onclick = () => {
    modal.style.display = 'none';
    state.directDevConfirmed = false;
  };
}

async function loadDefaultBacktestCache() {
  if (state.backtestLoading) return;
  state.booting = true;
  setLoading('backtest-loading', true);
  try {
    const r = await fetch(`${API_BASE}/api/ml/backtest/default_cache`);
    if (r.ok) {
      const data = await r.json();
      loadBacktestData(data);
      return;
    }
    console.warn('Default cache endpoint returned non-OK, running simulation...');
    state.backtestLoading = false;
    setLoading('backtest-loading', false);
    await runBacktestSimulation();
  } catch (err) {
    console.warn('Failed to fetch default backtest cache or run simulation:', err.message);
  } finally {
    setLoading('backtest-loading', false);
    state.booting = false;
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
          alert(`The parameter grid has ${totalCombos} combinations. Please adjust min/max/step values so that total combinations are 20 or fewer.`);
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
            slippage_pct: slippagePct
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
        slippage_pct: slippagePct
      };

      const r = await fetch(`${API_BASE}/api/ml/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqBody)
      });
      const data = await r.json();
      if (!r.ok) {
        if (!state.booting) {
          alert(`Backtest Failed: ${data.detail || 'Unknown error'}`);
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
    alert(`Simulation failed: ${err.message}`);
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
