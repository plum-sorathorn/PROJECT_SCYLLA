'use strict';

async function refreshDashboard() {
  await Promise.all([
    fetchOpenTrades(),
    fetchStats()
  ]);
}

function getStrategyParams(strategyType) {
  if (state.optimalParams && state.optimalParams[strategyType] && state.optimalParams[strategyType].params) {
    return mapApiOptimalToFormInputs(state.optimalParams[strategyType].params);
  }
  if (state.strategyDefaults && state.strategyDefaults.strategies && state.strategyDefaults.strategies[strategyType]) {
    const s = state.strategyDefaults.strategies[strategyType];
    return mapApiOptimalToFormInputs(s);
  }
  return null;
}

function isEligibleForStrategy(trade, strategyType) {
  if (!strategyType || strategyType === 'ALL') return true;
  const p = getStrategyParams(strategyType);
  if (!p) return true;

  const pSuccess = Number(trade.p_success) || 0;
  const p10 = Number(trade.predicted_p10) || 0;
  const p90 = Number(trade.predicted_p90) || 0;
  const p50 = Number(trade.predicted_p50) || 0;
  const iqr = p90 - p10;
  const iv = Number(trade.implied_vol) || 0;
  const optType = trade.option_type || '';
  const trend = trade.trend_alignment || '';

  const probThreshold = (p.prob != null ? p.prob : 35) / 100.0;
  const minMedianReturn = (p.median_ret != null ? p.median_ret : 0) / 100.0;
  const maxQuantileSpread = p.max_spread || 5.0;

  if (strategyType === 'whale_quality') {
    // Fades high-confidence, tight-IQR, mid-IV setups on TIER_A tickers
    return pSuccess >= probThreshold && (maxQuantileSpread <= 0 || iqr <= maxQuantileSpread);
  }
  if (strategyType === 'contrarian_trend') {
    // Fades trend: BULL_ALIGNED+Put or BEAR_ALIGNED+Call
    const isContrarian = (optType === 'Put' && trend === 'BULL_ALIGNED') ||
                         (optType === 'Call' && trend === 'BEAR_ALIGNED');
    return pSuccess >= probThreshold && isContrarian;
  }
  if (strategyType === 'vol_regime') {
    // Low-IV: tighter threshold; High-IV: looser IQR allowed
    const curProb = iv < 30.0 ? Math.max(probThreshold, 0.40) : probThreshold;
    const iqrOk = iv >= 30.0 ? (iqr <= 0.35) : true;
    return pSuccess >= curProb && iqrOk;
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
        if (topTrade.p_success >= 0.70) probEl.style.color = 'var(--call)';
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
