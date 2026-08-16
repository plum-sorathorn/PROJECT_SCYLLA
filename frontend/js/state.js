'use strict';

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
