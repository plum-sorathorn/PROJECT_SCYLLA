'use strict';

let runsChart = null;

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
