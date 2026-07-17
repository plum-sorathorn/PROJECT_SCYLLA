/* ============================================================
   PROJECT: SCYLLA // TERMINAL — Application JavaScript
   Connects to C++ Core on port 8080 and renders all widgets.
   Architecture: C++ Core (8080) → Python ODP (6900) → yfinance
   ============================================================ */

'use strict';

const API_BASE = 'http://127.0.0.1:6900';

// ── State ──────────────────────────────────────────────────
const state = {
  scannerData: [],
  pcrData: {},
  volConData: [],
  ivData: null,
  sortKey: 'volOiRatio',
  sortDir: -1,   // -1 = descending
  selectedRows: new Set(),
  charts: {},
};

// ── Utility ─────────────────────────────────────────────────
const fmt = (n, d = 2) => (n == null ? '—' : Number(n).toFixed(d));
const fmtK = (n) => {
  if (n == null) return '—';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + 'K';
  return String(n);
};
const $ = (id) => document.getElementById(id);

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

  // If C++ core is up, ODP is reachable through it (no direct call needed)
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

// ── WIDGET A: Unusual Options Scanner ───────────────────────
async function fetchScanner(minVolOI = 2.0) {
  setLoading('scanner-loading', true);
  try {
    const r = await fetch(`${API_BASE}/api/scanner`);
    const json = await r.json();
    state.scannerData = json.data || [];
    updateSummary(json.summary);
    renderScannerTable();
    updateExpectedMovePanel();
  } catch (e) {
    const tbody = $('scanner-tbody');
    tbody.innerHTML = `<tr><td colspan="13" class="table-empty">[ ERROR: CANNOT REACH C++ CORE AT ${API_BASE} ]</td></tr>`;
    console.error('Scanner fetch error:', e);
  } finally {
    setLoading('scanner-loading', false);
  }
}

function updateSummary(summary) {
  if (!summary) return;
  $('sum-whales').textContent  = summary.whaleSignalCount ?? '—';
  $('sum-maxvoloi').textContent = fmt(summary.maxVolOI) + 'x';
  $('sum-avgvoloi').textContent = fmt(summary.avgVolOI) + 'x';
  $('sum-agg-pcr').textContent  = fmt(summary.aggregatePCR, 3);
  $('sum-callvol').textContent  = fmtK(summary.totalCallVolume);
  $('sum-putvol').textContent   = fmtK(summary.totalPutVolume);
}

function renderScannerTable() {
  const tbody = $('scanner-tbody');
  const data = [...state.scannerData];

  // Sort
  data.sort((a, b) => {
    let av = a[state.sortKey], bv = b[state.sortKey];
    if (typeof av === 'string') av = av.toLowerCase(), bv = bv.toLowerCase();
    if (av === undefined || av === null) return 1;
    if (bv === undefined || bv === null) return -1;
    return (av < bv ? -1 : av > bv ? 1 : 0) * state.sortDir;
  });

  if (!data.length) {
    tbody.innerHTML = `<tr><td colspan="13" class="table-empty">[ NO DATA — VERIFY BACKEND STATUS ]</td></tr>`;
    return;
  }

  const rows = data.map((row) => {
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

    return `
      <tr class="${isWhale ? 'whale-row' : ''}" data-ticker="${row.ticker}" data-em="${row.expectedMove}" data-price="${row.underlierPrice}">
        <td><strong>${row.ticker}</strong></td>
        <td>${row.expiration}</td>
        <td>$${fmt(row.strike)}</td>
        <td>${typeEl}</td>
        <td>${fmtK(row.volume)}</td>
        <td>${fmtK(row.openInterest)}</td>
        <td class="${isWhale ? 'accent-blue' : ''}">${row.volOiRatio === 9999 ? '∞' : fmt(row.volOiRatio)}x</td>
        <td>${fmt(row.impliedVolatility)}%</td>
        <td>$${fmt(row.underlierPrice)}</td>
        <td>${smaFlag(row.above50dSMA, '50d')}</td>
        <td>${smaFlag(row.above200dSMA, '200d')}</td>
        <td class="${trendClass}">${(row.trendAlignment || '—').replace('_', ' ')}</td>
        <td>${em}</td>
      </tr>`;
  });

  tbody.innerHTML = rows.join('');

  // Row click → update expected move panel
  tbody.querySelectorAll('tr[data-ticker]').forEach((tr) => {
    tr.addEventListener('click', () => {
      const ticker = tr.dataset.ticker;
      const em = parseFloat(tr.dataset.em);
      const price = parseFloat(tr.dataset.price);
      if (ticker) addToExpectedMovePanel(ticker, em, price);
    });
  });
}

// ── Sort controls ────────────────────────────────────────────
document.querySelectorAll('#scanner-table th[data-sort]').forEach((th) => {
  th.addEventListener('click', () => {
    const key = th.dataset.sort;
    if (state.sortKey === key) {
      state.sortDir *= -1;
    } else {
      state.sortKey = key;
      state.sortDir = -1;
    }
    document.querySelectorAll('#scanner-table th').forEach(h => h.classList.remove('active-sort'));
    th.classList.add('active-sort');
    renderScannerTable();
  });
});

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

  const colors = { SPY: '#00E5FF', QQQ: '#E040FB', IWM: '#69FF47' };
  const tickers = ['SPY', 'QQQ', 'IWM'];

  // Gather all labels (dates/expirations)
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
      backgroundColor: colors[t] + '18',
      borderWidth: 1.5,
      pointRadius: 3,
      pointHoverRadius: 5,
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
      title: 'Put/Call Ratio by Expiry Cycle',
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
          backgroundColor: '#00E5FF44',
          borderColor: '#00E5FF',
          borderWidth: 1,
          stack: 'vol',
        },
        {
          label: 'Put Volume',
          data: putVols,
          backgroundColor: '#E040FB44',
          borderColor: '#E040FB',
          borderWidth: 1,
          stack: 'vol',
        },
      ],
    },
    options: chartDefaults({ title: 'Volume by Expiry', yLabel: 'Volume', stacked: true }),
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

  // Color IV rank by level
  const ivRankEl = $('iv-rank');
  if (d.ivRank >= 75) ivRankEl.className = 'gauge-value accent-mag';
  else if (d.ivRank >= 50) ivRankEl.className = 'gauge-value accent-blue';
  else ivRankEl.className = 'gauge-value accent-green';
}

function renderIVSmileChart() {
  const ctx = $('chart-ivsmile');
  if (!ctx || !state.ivData) return;

  const smile = state.ivData.smileData || [];
  const expirations = [...new Set(smile.map(p => p.expiration))].slice(0, 2);
  const colors = ['#00E5FF', '#E040FB'];

  const datasets = expirations.map((exp, i) => {
    const pts = smile
      .filter(p => p.expiration === exp && p.optionType === 'Call')
      .sort((a, b) => a.strike - b.strike);
    return {
      label: exp,
      data: pts.map(p => ({ x: p.strike, y: p.iv })),
      borderColor: colors[i % 2],
      backgroundColor: colors[i % 2] + '18',
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
    options: chartDefaults({ title: 'Volatility Smile', xLabel: 'Strike', yLabel: 'IV%' }),
  });
}

// ── EXTENSION 3: Expected Move Panel ─────────────────────────
function updateExpectedMovePanel() {
  // Auto-populate from top whale signals in scanner
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
    grid.innerHTML = `<div class="exp-placeholder">[ NO WHALE SIGNALS WITH EXPECTED MOVE DATA ]</div>`;
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
  // Pull tickers from current scanner data (top 10 unique)
  const tickers = [...new Set(state.scannerData.map(r => r.ticker))].slice(0, 10);
  if (!tickers.length) {
    $('swing-grid').innerHTML = `<div class="exp-placeholder">[ AWAITING SCANNER DATA ]</div>`;
    return;
  }

  try {
    // Derive swing data from already-fetched scanner rows (SMA flags are embedded)
    const seen = new Set();
    const swingRows = state.scannerData.filter(r => {
      if (seen.has(r.ticker)) return false;
      seen.add(r.ticker);
      return true;
    }).slice(0, 12);

    const grid = $('swing-grid');
    if (!swingRows.length) {
      grid.innerHTML = `<div class="exp-placeholder">[ NO DATA ]</div>`;
      return;
    }

    grid.innerHTML = swingRows.map(r => {
      const bullish = r.above50dSMA === 1;
      const tickerClass = bullish ? 'accent-blue' : 'accent-mag';
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

      const trendClass = {
        'BULL_ALIGNED': 'trend-bull-aligned',
        'BEAR_ALIGNED': 'trend-bear-aligned',
        'BULL_CONTRARIAN': 'trend-bull-contra',
      }[r.trendAlignment] || 'trend-neutral';

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
        backgroundColor: '#13161F',
        borderColor: '#1E2338',
        borderWidth: 1,
        titleColor: '#00E5FF',
        bodyColor: '#C8D3E8',
        titleFont: { family: 'Share Tech Mono', size: 11 },
        bodyFont: { family: 'Share Tech Mono', size: 11 },
        padding: 10,
      },
    },
    scales: {
      x: {
        stacked,
        grid: { color: '#1E233833', drawBorder: false },
        ticks: {
          color: '#4A556E',
          font: { family: 'Share Tech Mono', size: 9 },
          maxTicksLimit: 8,
          maxRotation: 45,
        },
        border: { color: '#1E2338' },
        ...(xLabel ? { title: { display: true, text: xLabel, color: '#4A556E', font: { family: 'Share Tech Mono', size: 9 } } } : {}),
      },
      y: {
        stacked,
        grid: { color: '#1E233844', drawBorder: false },
        ticks: {
          color: '#4A556E',
          font: { family: 'Share Tech Mono', size: 9 },
        },
        border: { color: '#1E2338' },
        ...(yLabel ? { title: { display: true, text: yLabel, color: '#4A556E', font: { family: 'Share Tech Mono', size: 9 } } } : {}),
      },
    },
  };
}

// ── Filter / controls wiring ─────────────────────────────────
$('btn-apply-filter').addEventListener('click', () => {
  const minVol = parseFloat($('filter-minvoloi').value) || 2.0;
  fetchScanner(minVol);
});

$('btn-refresh-all').addEventListener('click', () => refreshAll());

$('volcon-ticker').addEventListener('change', (e) => {
  fetchVolCon(e.target.value);
});

$('btn-load-iv').addEventListener('click', () => {
  const ticker = $('iv-ticker').value;
  fetchIVSkew(ticker);
});

// ── Main refresh cycle ───────────────────────────────────────
async function refreshAll() {
  await Promise.all([
    fetchScanner(parseFloat($('filter-minvoloi').value) || 2.0),
    fetchPCR(),
    fetchVolCon($('volcon-ticker').value || 'SPY'),
    fetchIVSkew($('iv-ticker').value || 'SPY'),
  ]);
  await fetchSwingAlignment();
  await checkHealth();
}

// ── Boot sequence ────────────────────────────────────────────
(async function boot() {
  startClock();

  // Stagger initial loads to avoid hammering the backend
  checkHealth();

  setTimeout(() => fetchScanner(), 200);
  setTimeout(() => fetchPCR(), 600);
  setTimeout(() => fetchVolCon('SPY'), 1000);
  setTimeout(() => fetchIVSkew('SPY'), 1400);

  // After scanner loads, populate swing alignment
  setTimeout(() => fetchSwingAlignment(), 4000);

  // Auto-refresh every 5 minutes
  setInterval(refreshAll, 5 * 60 * 1000);
})();

