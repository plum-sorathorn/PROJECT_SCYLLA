'use strict';

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

function startClock() {
  const el = $('header-clock');
  const tick = () => {
    const now = new Date();
    el.textContent = now.toLocaleTimeString('en-US', { hour12: false });
  };
  tick();
  setInterval(tick, 1000);
}

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

function showToast(msg, type = 'error') {
  if (msg.includes('TimeoutError') || msg.includes('AbortError') || msg.includes('The operation was aborted')) {
    msg = 'Request timed out. The server might be busy or offline.';
  }
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.style.position = 'fixed';
    container.style.bottom = '20px';
    container.style.right = '20px';
    container.style.zIndex = '9999';
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '10px';
    document.body.appendChild(container);
  }
  const toast = document.createElement('div');
  toast.textContent = msg;
  toast.style.background = type === 'error' ? 'var(--bg-loss, #ffebee)' : 'var(--bg-profit, #e8f5e9)';
  toast.style.color = type === 'error' ? 'var(--color-loss, #8b3a3a)' : 'var(--color-profit, #2d7a4a)';
  toast.style.padding = '12px 16px';
  toast.style.borderRadius = '4px';
  toast.style.border = '1px solid ' + (type === 'error' ? 'var(--color-loss, #8b3a3a)' : 'var(--color-profit, #2d7a4a)');
  toast.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
  toast.style.fontFamily = 'var(--font-mono, monospace)';
  toast.style.fontSize = '12px';
  toast.style.opacity = '0';
  toast.style.transition = 'opacity 0.3s ease';
  container.appendChild(toast);
  setTimeout(() => toast.style.opacity = '1', 10);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 5000);
}
