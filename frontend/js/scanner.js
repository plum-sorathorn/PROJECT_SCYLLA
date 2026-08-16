'use strict';

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
          backgroundColor: 'rgba(0, 85, 102, 0.15)',
          hoverBackgroundColor: 'rgba(0, 85, 102, 0.25)',
          borderColor: 'rgba(0, 85, 102, 1)',
          borderWidth: 1.5,
          borderRadius: { topLeft: 4, topRight: 4, bottomLeft: 0, bottomRight: 0 },
          borderSkipped: false,
          stack: 'vol',
          barPercentage: 0.78,
          categoryPercentage: 0.82,
        },
        {
          label: 'Put Volume',
          data: putVols,
          backgroundColor: 'rgba(139, 58, 58, 0.15)',
          hoverBackgroundColor: 'rgba(139, 58, 58, 0.25)',
          borderColor: 'rgba(139, 58, 58, 1)',
          borderWidth: 1.5,
          borderRadius: { topLeft: 0, topRight: 0, bottomLeft: 4, bottomRight: 4 },
          borderSkipped: false,
          stack: 'vol',
          barPercentage: 0.78,
          categoryPercentage: 0.82,
        },
      ],
    },
    options: chartDefaults({ title: 'Expiration Volume Distribution', yLabel: 'Volume', stacked: true }),
  });
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
