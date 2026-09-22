/* ============================================================
   MindfulTech – Chart.js Helpers
   ============================================================ */

const MT_COLORS = {
  primary: 'rgba(13, 148, 136, 1)',
  primaryBg: 'rgba(13, 148, 136, 0.15)',
  accent: 'rgba(99, 102, 241, 1)',
  accentBg: 'rgba(99, 102, 241, 0.15)',
  green: 'rgba(34, 197, 94, 1)',
  greenBg: 'rgba(34, 197, 94, 0.15)',
  amber: 'rgba(245, 158, 11, 1)',
  amberBg: 'rgba(245, 158, 11, 0.15)',
  red: 'rgba(239, 68, 68, 1)',
  redBg: 'rgba(239, 68, 68, 0.15)',
  purple: 'rgba(168, 85, 247, 1)',
  purpleBg: 'rgba(168, 85, 247, 0.15)',
};

const SHARED_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: true, position: 'top', labels: { font: { family: 'Inter', size: 12 } } },
  },
  scales: {
    x: { grid: { display: false }, ticks: { font: { family: 'Inter' } } },
    y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,.06)' }, ticks: { font: { family: 'Inter' } } },
  },
};

/* ---- Analytics Page ---- */

async function renderAnalyticsCharts() {
  try {
    const resp = await fetch('/api/chart-data');
    const data = await resp.json();

    if (!data.labels || data.labels.length === 0) {
      document.getElementById('no-data-msg')?.classList.remove('d-none');
      return;
    }

    createLineChart('screenTimeChart', data.labels, [{
      label: 'Screen Time (hrs)', data: data.screen_time,
      borderColor: MT_COLORS.primary, backgroundColor: MT_COLORS.primaryBg,
    }]);

    createLineChart('socialMediaChart', data.labels, [{
      label: 'Social Media (hrs)', data: data.social_media,
      borderColor: MT_COLORS.accent, backgroundColor: MT_COLORS.accentBg,
    }]);

    createBarChart('focusTimeChart', data.labels, [{
      label: 'Focus Time (hrs)', data: data.focus_time,
      backgroundColor: MT_COLORS.green, borderColor: MT_COLORS.green,
    }]);

    createLineChart('sleepChart', data.labels, [{
      label: 'Sleep (hrs)', data: data.sleep,
      borderColor: MT_COLORS.purple, backgroundColor: MT_COLORS.purpleBg,
    }]);

    createBarChart('productivityChart', data.labels, [{
      label: 'Productivity (1-10)', data: data.productivity,
      backgroundColor: MT_COLORS.amber, borderColor: MT_COLORS.amber,
    }]);

    createLineChart('riskChart', data.labels, [{
      label: 'Risk Level (1=Low, 2=Mod, 3=High)', data: data.risk_levels,
      borderColor: MT_COLORS.red, backgroundColor: MT_COLORS.redBg,
      stepped: true,
    }]);

    /* Scatter charts */
    if (data.screen_time && data.focus_time) {
      createScatterChart('scatterScreenFocus',
        data.screen_time, data.focus_time,
        'Screen Time (hrs)', 'Focus Time (hrs)',
        MT_COLORS.primary
      );
    }
    if (data.sleep && data.productivity) {
      createScatterChart('scatterSleepProd',
        data.sleep, data.productivity,
        'Sleep (hrs)', 'Productivity (1-10)',
        MT_COLORS.purple
      );
    }

  } catch (err) {
    console.error('Failed to load chart data:', err);
  }
}

/* ---- Dashboard Trends ---- */

async function renderDashboardTrends() {
  try {
    const resp = await fetch('/api/chart-data');
    const data = await resp.json();
    if (!data.labels || data.labels.length === 0) return;

    createLineChart('dashScreenTrend', data.labels, [{
      label: 'Screen Time (hrs)', data: data.screen_time,
      borderColor: MT_COLORS.primary, backgroundColor: MT_COLORS.primaryBg,
    }]);

    createLineChart('dashFocusTrend', data.labels, [{
      label: 'Focus Time (hrs)', data: data.focus_time,
      borderColor: MT_COLORS.green, backgroundColor: MT_COLORS.greenBg,
    }]);
  } catch (err) {
    console.error('Failed to load dashboard trends:', err);
  }
}

/* ---- Chart constructors ---- */

function createLineChart(canvasId, labels, datasets) {
  const el = document.getElementById(canvasId);
  if (!el) return;
  new Chart(el.getContext('2d'), {
    type: 'line',
    data: {
      labels,
      datasets: datasets.map(ds => ({
        ...ds, fill: true, tension: 0.35,
        pointRadius: 4, pointHoverRadius: 6, borderWidth: 2,
      })),
    },
    options: SHARED_OPTIONS,
  });
}

function createBarChart(canvasId, labels, datasets) {
  const el = document.getElementById(canvasId);
  if (!el) return;
  new Chart(el.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: datasets.map(ds => ({
        ...ds, borderWidth: 1, borderRadius: 6, maxBarThickness: 40,
      })),
    },
    options: SHARED_OPTIONS,
  });
}

function createScatterChart(canvasId, xData, yData, xLabel, yLabel, color) {
  const el = document.getElementById(canvasId);
  if (!el) return;
  const points = xData.map((x, i) => ({ x, y: yData[i] }));
  new Chart(el.getContext('2d'), {
    type: 'scatter',
    data: {
      datasets: [{
        label: `${xLabel} vs ${yLabel}`,
        data: points,
        backgroundColor: color,
        pointRadius: 5,
        pointHoverRadius: 7,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: true, position: 'top', labels: { font: { family: 'Inter', size: 12 } } } },
      scales: {
        x: { title: { display: true, text: xLabel, font: { family: 'Inter' } }, grid: { color: 'rgba(0,0,0,.06)' } },
        y: { title: { display: true, text: yLabel, font: { family: 'Inter' } }, grid: { color: 'rgba(0,0,0,.06)' }, beginAtZero: true },
      },
    },
  });
}

function renderRiskDonut(canvasId, riskLevel) {
  const el = document.getElementById(canvasId);
  if (!el) return;
  const colorMap = { Low: MT_COLORS.green, Moderate: MT_COLORS.amber, High: MT_COLORS.red };
  const valMap = { Low: 1, Moderate: 2, High: 3 };
  const color = colorMap[riskLevel] || MT_COLORS.primary;
  const val = valMap[riskLevel] || 1;
  new Chart(el.getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: ['Risk', 'Remaining'],
      datasets: [{ data: [val, 3 - val], backgroundColor: [color, 'rgba(226,232,240,.4)'], borderWidth: 0 }],
    },
    options: {
      responsive: true,
      cutout: '72%',
      plugins: { legend: { display: false } },
    },
  });
}
