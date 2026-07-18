/* ============================================================
   PROJECT: SCYLLA // TERMINAL & ML COCKPIT — Unified SPA Script
   Connects to C++ Core on port 8080 and python ML ODP on port 6900.
   ============================================================ */

'use strict';

const API_BASE = 'http://127.0.0.1:6900';

// ── State ──────────────────────────────────────────────────
const state = {
  // Tactical Console State
  scannerData: [],
  pcrData: {},
  volConData: [],
  ivData: null,
  sortKey: 'volOiRatio',
  sortDir: -1,   // -1 = descending
  selectedRows: new Set(),
  charts: {},

  // ML Cockpit State
  mlStats: {},
  mlTrades: [],
  mlImportance: [],
  mlModelMetrics: {},
  mlModelRuns: [],
  mlSettings: {},
  mlLoaded: false
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
  const mlCockpitBtn = $('btn-ml-cockpit');
  const consoleBtn = $('btn-console');
  const tacticalStrip = $('tactical-summary-strip');
  const mlStrip = $('ml-summary-strip');
  const tacticalView = $('view-tactical');
  const mlView = $('view-ml');

  if (viewName === 'ml') {
    tacticalView.style.display = 'none';
    tacticalStrip.style.display = 'none';
    mlCockpitBtn.style.display = 'none';

    mlView.style.display = 'block';
    mlStrip.style.display = 'block';
    consoleBtn.style.display = 'inline-flex';

    titleEl.innerHTML = 'PROJECT: SCYLLA <span class="header-sep">//</span> <span style="font-style: italic;">ML Cockpit</span>';
    subEl.textContent = 'SELF-TRAINING PREDICTOR — CLASSIFICATION & TELEMETRY SYSTEMS';

    if (!state.mlLoaded) {
      refreshML();
      state.mlLoaded = true;
    } else {
      // Re-trigger chart rendering on visible canvas
      renderModelRunsChart();
    }
  } else {
    mlView.style.display = 'none';
    mlStrip.style.display = 'none';
    consoleBtn.style.display = 'none';

    tacticalView.style.display = 'block';
    tacticalStrip.style.display = 'block';
    mlCockpitBtn.style.display = 'inline-flex';

    titleEl.innerHTML = 'PROJECT: SCYLLA <span class="header-sep">//</span> <span style="font-style: italic;">Tactical Console</span>';
    subEl.textContent = 'VOLATILITY & FLOW TELEMETRY — POWERED BY C++ CORE & OpenBB ODP';

    // Re-render tactical charts on display
    renderPCRChart();
    renderVolConChart();
    renderIVSmileChart();
  }
}

function handleRouting() {
  const hash = window.location.hash;
  if (hash === '#ml') {
    showView('ml');
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

function recalculateTheoreticalPremium() {
  if (!$('inf-autocalc').checked) return;

  const S = parseFloat($('inf-underlier').value) || 0;
  const K = parseFloat($('inf-strike').value) || 0;
  const type = $('inf-type').value;
  const dte = parseFloat($('inf-dte').value) || 0;
  const iv = parseFloat($('inf-iv').value) || 0;
  const vol = parseFloat($('inf-volume').value) || 0;

  if (S <= 0 || K <= 0 || dte <= 0 || iv <= 0 || vol <= 0) {
    return;
  }

  const pricePerShare = blackScholes(type, S, K, dte, iv);
  const totalPremium = pricePerShare * vol * 100;
  $('inf-premium').value = totalPremium.toFixed(2);
}

// ── ML Inference Form Target Profile Sync ────────────────────
function loadTradeIntoPredictor(trade, source) {
  const ticker = (trade.ticker || '').toUpperCase();
  const strike = trade.strike ?? 0;
  const optionType = trade.optionType || trade.option_type || 'Call';
  const side = trade.side || 'BUY';
  const dte = trade.dte ?? 30;
  const volOiRatio = trade.volOiRatio ?? trade.vol_oi_ratio ?? 1.0;
  const impliedVol = trade.impliedVolatility ?? trade.implied_vol ?? 30.0;
  const premium = trade.premium ?? 0;
  const trendAlignment = trade.trendAlignment ?? trade.trend_alignment ?? 'NEUTRAL';
  const underlierPrice = trade.underlierPrice ?? trade.underlier_price ?? 0;
  const volume = trade.volume ?? 1000;

  // Uncheck autocalc when loading recorded trade values
  $('inf-autocalc').checked = false;

  $('inf-underlier').value = underlierPrice;
  $('inf-strike').value = strike;
  $('inf-type').value = optionType;
  $('inf-side').value = side;
  $('inf-dte').value = dte;
  $('inf-ratio').value = volOiRatio;
  $('inf-iv').value = impliedVol;
  $('inf-volume').value = volume;
  $('inf-premium').value = premium;
  $('inf-trend').value = trendAlignment;

  const profileEl = $('prediction-target-profile');
  if (profileEl) {
    profileEl.textContent = `TARGET PROFILE: ${ticker} $${fmt(strike)} ${optionType.toUpperCase()} (${dte} DTE) [SOURCE: ${source}]`;
  }

  // Trigger inference automatically
  runManualInference({ preventDefault: () => {} });
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
  if (!el) return;
  el.classList.toggle('active', active);
}


// ══════════════════════════════════════════════════════════════
// TACTICAL CONSOLE LOGIC
// ══════════════════════════════════════════════════════════════

// ── WIDGET A: Unusual Options Scanner ───────────────────────
async function fetchScanner(minVolOI = 2.0) {
  setLoading('scanner-loading', true);
  try {
    const r = await fetch(`${API_BASE}/api/scanner?min_vol_oi=${minVolOI}`);
    const json = await r.json();
    state.scannerData = json.data || [];
    updateSummary(json.summary);
    renderScannerTable();
    updateExpectedMovePanel();
  } catch (e) {
    const tbody = $('scanner-tbody');
    tbody.innerHTML = `<tr><td colspan="15" class="table-empty">[ ERROR: ENGINE OFFLINE — CONNECTION REFUSED AT ${API_BASE} ]</td></tr>`;
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
    const r = await fetch(`${API_BASE}/api/put-call-ratio`);
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

  const colors = { SPY: '#E2E2E2', QQQ: '#D4AF37', IWM: '#BF5A5A' };
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
    const r = await fetch(`${API_BASE}/api/volume-concentration?ticker=${ticker}`);
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
          backgroundColor: '#E2E2E225',
          borderColor: '#E2E2E2',
          borderWidth: 1,
          stack: 'vol',
        },
        {
          label: 'Put Volume',
          data: putVols,
          backgroundColor: '#BF5A5A25',
          borderColor: '#BF5A5A',
          borderWidth: 1,
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
    const r = await fetch(`${API_BASE}/api/iv-skew?ticker=${ticker}`);
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
  const colors = ['#E2E2E2', '#D4AF37'];

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
        backgroundColor: '#121316',
        borderColor: '#222428',
        borderWidth: 1,
        titleColor: '#D4AF37',
        bodyColor: '#E2E2E2',
        titleFont: { family: 'Share Tech Mono', size: 11 },
        bodyFont: { family: 'Share Tech Mono', size: 11 },
        padding: 10,
      },
    },
    scales: {
      x: {
        stacked,
        grid: { color: '#22242833', drawBorder: false },
        ticks: {
          color: '#7B7D82',
          font: { family: 'Share Tech Mono', size: 9 },
          maxTicksLimit: 8,
          maxRotation: 45,
        },
        border: { color: '#222428' },
        ...(xLabel ? { title: { display: true, text: xLabel, color: '#7B7D82', font: { family: 'Share Tech Mono', size: 9 } } } : {}),
      },
      y: {
        stacked,
        grid: { color: '#22242844', drawBorder: false },
        ticks: {
          color: '#7B7D82',
          font: { family: 'Share Tech Mono', size: 9 },
        },
        border: { color: '#222428' },
        ...(yLabel ? { title: { display: true, text: yLabel, color: '#7B7D82', font: { family: 'Share Tech Mono', size: 9 } } } : {}),
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
    const r = await fetch(`${API_BASE}/api/ml/stats`);
    const data = await r.json();
    state.mlStats = data;
    
    $('sum-total-trades').textContent = data.total_trades ?? 0;
    $('sum-labeled-trades').textContent = data.labeled_trades ?? 0;
    $('sum-win-rate').textContent = fmtPct(data.success_ratio);
    
    const statusEl = $('sum-model-status');
    if (data.model_ready) {
      statusEl.textContent = 'MODEL READY';
      statusEl.className = 'tile-value accent-call';
    } else {
      statusEl.textContent = 'NO WEIGHTS';
      statusEl.className = 'tile-value accent-coral';
    }
  } catch (e) {
    console.error('Error fetching stats:', e);
  }
}

// ── Fetch Logged Trades (Ledger) ─────────────────────────────
async function fetchTrades() {
  setLoading('ledger-loading', true);
  try {
    const r = await fetch(`${API_BASE}/api/ml/trades?limit=50`);
    const json = await r.json();
    state.mlTrades = json.data || [];
    renderLedgerTable();
  } catch (e) {
    const tbody = $('ledger-tbody');
    tbody.innerHTML = `<tr><td colspan="11" class="table-empty">[ ERROR: PIPELINE SERVER OFFLINE — CONNECTION REFUSED ]</td></tr>`;
    console.error('Ledger fetch error:', e);
  } finally {
    setLoading('ledger-loading', false);
  }
}

// ── Render Ledger Table ──────────────────────────────────────
function renderLedgerTable() {
  const tbody = $('ledger-tbody');
  
  if (!state.mlTrades.length) {
    tbody.innerHTML = `<tr><td colspan="11" class="table-empty">[ NO HISTORICAL OPTIONS LOGS RECORDED IN DATABASE ]</td></tr>`;
    return;
  }
  
  const rows = state.mlTrades.map((row, idx) => {
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
      const trade = state.mlTrades[idx];
      if (trade) {
        loadTradeIntoPredictor(trade, 'LEDGER');
      }
    });
  });
}

// ── Run Labeling Worker ──────────────────────────────────────
async function runLabeling() {
  const btn = $('btn-run-labeling');
  const prevText = btn.textContent;
  btn.textContent = 'LABELING RUNNING...';
  btn.disabled = true;
  
  try {
    const horizon = $('cfg-horizon').value || 10;
    const threshold = $('cfg-threshold').value || 0.03;
    const r = await fetch(`${API_BASE}/api/ml/label?horizon_days=${horizon}&profit_threshold=${threshold}`, { method: 'POST' });
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

// ── Run Manual Prediction / Inference ────────────────────────
async function runManualInference(e) {
  e.preventDefault();
  
  const data = {
    underlierPrice: parseFloat($('inf-underlier').value),
    strike: parseFloat($('inf-strike').value),
    optionType: $('inf-type').value,
    side: $('inf-side').value,
    dte: parseInt($('inf-dte').value),
    volOiRatio: parseFloat($('inf-ratio').value),
    impliedVolatility: parseFloat($('inf-iv').value),
    premium: parseFloat($('inf-premium').value),
    trendAlignment: $('inf-trend').value,
    volume: 1000,          // standard placeholders
    openInterest: 200      // standard placeholders
  };
  
  try {
    const r = await fetch(`${API_BASE}/api/ml/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    
    const res = await r.json();
    const prob = res.probability;
    const modelType = res.model_type;
    
    const pctStr = (prob * 100).toFixed(1) + '%';
    const pctEl = $('prediction-percentage');
    pctEl.textContent = pctStr;
    
    // Dynamically color score based on classification strength
    if (prob >= 0.70) {
      pctEl.style.color = '#8FA382'; // Bull green/sage glow
      pctEl.style.textShadow = '0 0 6px rgba(143, 163, 130, 0.4)';
    } else if (prob <= 0.40) {
      pctEl.style.color = 'var(--put)'; // Bear coral red glow
      pctEl.style.textShadow = '0 0 6px rgba(191, 90, 90, 0.4)';
    } else {
      pctEl.style.color = 'var(--accent)'; // Standard gold glow
      pctEl.style.textShadow = '0 0 6px rgba(212, 175, 55, 0.4)';
    }
    
    $('model-type-badge').textContent = modelType.toUpperCase().replace('_', ' ');
    
  } catch (e) {
    alert(`Prediction inference call failed: ${e.message}`);
  }
}

// ── Fetch settings ───────────────────────────────────────────
async function fetchSettings() {
  try {
    const r = await fetch(`${API_BASE}/api/ml/settings`);
    const json = await r.json();
    state.mlSettings = json;
    
    $('cfg-horizon').value = json.horizon_days || '10';
    $('cfg-threshold').value = Number(json.profit_threshold).toFixed(2);
  } catch (e) {
    console.error('Settings fetch error:', e);
  }
}

// ── Save settings ────────────────────────────────────────────
async function saveSettings() {
  const horizon = $('cfg-horizon').value;
  const threshold = $('cfg-threshold').value;
  const statusEl = $('settings-status');
  
  try {
    statusEl.textContent = 'SAVING...';
    const r = await fetch(`${API_BASE}/api/ml/settings?horizon_days=${horizon}&profit_threshold=${threshold}`, {
      method: 'POST'
    });
    const res = await r.json();
    if (res.status === 'success') {
      statusEl.textContent = 'CONFIGURATION PERSISTED';
      statusEl.style.color = 'var(--buy)';
      setTimeout(() => {
        statusEl.textContent = '';
        statusEl.style.color = 'var(--text-muted)';
      }, 3000);
      await fetchSettings();
    }
  } catch (e) {
    statusEl.textContent = 'SAVE FAILED';
    statusEl.style.color = 'var(--put)';
  }
}

// ── Fetch model runs ─────────────────────────────────────────
async function fetchModelRuns() {
  try {
    const r = await fetch(`${API_BASE}/api/ml/model-runs`);
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
  
  const labels = state.mlModelRuns.map(run => {
    const d = new Date(run.timestamp);
    return d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  }).reverse();
  
  const rocData = state.mlModelRuns.map(run => run.test_roc_auc).reverse();
  const accData = state.mlModelRuns.map(run => run.test_accuracy).reverse();
  
  if (runsChart) {
    runsChart.destroy();
  }
  
  runsChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Test ROC AUC',
          data: rocData,
          borderColor: 'rgba(212, 175, 55, 0.9)',
          backgroundColor: 'transparent',
          tension: 0.1
        },
        {
          label: 'Test Accuracy',
          data: accData,
          borderColor: 'rgba(143, 163, 130, 0.9)',
          backgroundColor: 'transparent',
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: 'rgba(255, 255, 255, 0.7)', font: { family: 'Share Tech Mono' } }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: 'rgba(255,255,255,0.4)', font: { family: 'Share Tech Mono' } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: 'rgba(255,255,255,0.4)', font: { family: 'Share Tech Mono' } }
        }
      }
    }
  });
}

async function refreshML() {
  await Promise.all([
    fetchStats(),
    fetchTrades(),
    fetchSettings(),
    fetchModelRuns()
  ]);
}


// ══════════════════════════════════════════════════════════════
// CONTROLS & BOOT STRAP
// ══════════════════════════════════════════════════════════════

// ── Refresh Active View ──────────────────────────────────────
async function refreshAll() {
  checkHealth();
  if ($('view-tactical').style.display !== 'none') {
    await refreshTactical();
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
  $('inference-form').addEventListener('submit', runManualInference);
  $('btn-run-labeling').addEventListener('click', runLabeling);
  $('btn-trigger-retrain').addEventListener('click', runRetraining);
  $('btn-save-settings').addEventListener('click', saveSettings);

  // Monitor form input modifications to sync state and auto-calculate
  document.querySelectorAll('#inference-form input, #inference-form select').forEach((input) => {
    input.addEventListener('input', () => {
      // Re-calculate first if autocalc is enabled and this input isn't premium itself
      if (input.id !== 'inf-premium') {
        recalculateTheoreticalPremium();
      } else {
        // If they manually edit premium, turn off autocalc!
        $('inf-autocalc').checked = false;
      }
      
      const profileEl = $('prediction-target-profile');
      if (profileEl) {
        profileEl.textContent = 'TARGET PROFILE: CUSTOM MANUAL PARAMETERS';
      }
      document.querySelectorAll('#scanner-table tr, #ledger-table tr').forEach(r => r.classList.remove('active-row'));
    });
  });

  // Re-calculate when auto-calc checkbox state toggles
  $('inf-autocalc').addEventListener('change', () => {
    recalculateTheoreticalPremium();
  });

  // Hash Navigation Routing
  window.addEventListener('hashchange', handleRouting);
}

// ── Boot Sequence ────────────────────────────────────────────
(async function boot() {
  startClock();
  setupEventListeners();
  
  // Resolve initial view
  handleRouting();

  // Stagger loading to prevent API hammering
  checkHealth();

  // Load tactical scanner first
  setTimeout(() => fetchScanner(), 200);
  setTimeout(() => fetchPCR(), 600);
  setTimeout(() => fetchVolCon('SPY'), 1000);
  setTimeout(() => fetchIVSkew('SPY'), 1400);

  // Load swing convergence alignment slightly later
  setTimeout(() => fetchSwingAlignment(), 4000);

  // Trigger background sync cycle (5 minutes)
  setInterval(refreshAll, 5 * 60 * 1000);
})();
