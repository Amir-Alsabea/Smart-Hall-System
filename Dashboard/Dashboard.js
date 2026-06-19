"use strict";

document.documentElement.setAttribute(
    'data-theme',
    sessionStorage.getItem('theme') || 'dark'
);

const API_CV = 'http://localhost:5050/api/state';
const API_ENV = 'http://localhost:5051/api/state';

let focusHist = Array(60).fill(null);
let chart;
let envConnected = false;
let cvConnected = false;

// ── Clock ────────────────────────────────────────────────────────────────
function tick() {
    document.getElementById('clk').textContent =
        new Date().toLocaleTimeString('en-US',
            { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true });
}
setInterval(tick, 1000); tick();

function updateChartTheme() {
    const light = document.documentElement.getAttribute('data-theme') === 'light';
    chart.options.scales.y.grid.color = light ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.04)';
    chart.options.scales.y.ticks.color = light ? '#94a3b8' : '#475569';
    chart.options.plugins.tooltip.backgroundColor = light ? '#ffffff' : '#13161e';
    chart.options.plugins.tooltip.borderColor = light ? '#e2e5ef' : '#1e2333';
    chart.options.plugins.tooltip.titleColor = light ? '#475569' : '#94a3b8';
    chart.options.plugins.tooltip.bodyColor = light ? '#1e293b' : '#e2e8f0';
    chart.update('none');
}

// ── Chart ────────────────────────────────────────────────────────────────
function initChart() {
    const ctx = document.getElementById('focus-chart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(60).fill(''), datasets: [{
                data: focusHist,
                borderColor: '#00e5a0',
                backgroundColor: 'rgba(0,229,160,0.05)',
                borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0, spanGaps: true
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 300 },
            plugins: {
                legend: { display: false }, tooltip: {
                    backgroundColor: '#13161e', borderColor: '#1e2333', borderWidth: 1,
                    titleColor: '#94a3b8', bodyColor: '#e2e8f0',
                    titleFont: { size: 11 }, bodyFont: { family: 'JetBrains Mono', size: 13 },
                    callbacks: { label: c => ` ${Math.round(c.raw)}%` }
                }
            },
            scales: {
                x: { display: false },
                y: {
                    min: 0, max: 100,
                    grid: { color: 'rgba(255,255,255,0.04)' }, border: { display: false },
                    ticks: { color: '#475569', font: { size: 11, family: 'JetBrains Mono' }, callback: v => v + '%' }
                }
            }
        }
    });
}
initChart();

// ── Nav ──────────────────────────────────────────────────────────────────
function switchView(name, btn) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById('view-' + name).classList.add('active');
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
}

// ── Student card ─────────────────────────────────────────────────────────
function stuCard(id, s) {
    const fc = s.status === 'FOCUSED';
    const sc = Math.round(s.score || 0);
    const clr = fc ? '#00e5a0' : '#f43f5e';
    const ini = (s.name || '??').replace(/[^a-zA-Z ]/g, '').split(' ')
        .filter(Boolean).map(w => w[0]).join('').toUpperCase().slice(0, 2) || '??';
    const reasons = (s.reasons || []).map(r => `<span class="rtag">${r}</span>`).join('');
    return `<div class="stu-card">
    <div class="stu-top">
      <div class="stu-avatar">${ini}</div>
      <span class="stu-status-pill ${fc ? 'sp-focused' : 'sp-unfocused'}">${s.status}</span>
    </div>
    <div class="stu-name" title="${s.name || id}">${s.name || id}</div>
    <div class="stu-meta">YAW ${s.yaw || '—'} · PITCH ${s.pitch || '—'}</div>
    <div class="score-row"><span>Focus Score</span>
      <span class="score-num" style="color:${clr}">${sc}%</span></div>
    <div class="score-track">
      <div class="score-fill" style="width:${sc}%;background:${clr}"></div></div>
    ${reasons ? `<div class="reason-tags">${reasons}</div>` : ''}
  </div>`;
}

function setEnvCell(valId, statusId, raw, unit, lo, hi) {
    if (raw == null) return;
    document.getElementById(valId).innerHTML =
        raw + `<span class="env-unit-light"> ${unit}</span>`;
    const st = document.getElementById(statusId);
    if (raw < lo || raw > hi) {
        st.textContent = '⚠ Out of range';
        st.className = 'env-status es-err';
    } else {
        st.textContent = '✓ Normal';
        st.className = 'env-status es-ok';
    }
}

function renderEnv(d) {
    const s = d.sensors || {};

    // Show/hide no-data notice
    const hasData = Object.keys(s).length > 0;
    document.getElementById('env-no-data').style.display = hasData ? 'none' : '';
    document.getElementById('env-focus-banner').style.display = hasData ? '' : 'none';

    if (!hasData) return;

    // Sensor cells — key names from ML snapshot(): temperature, humidity, light, noise, motion
    setEnvCell('e-temp', 'e-temp-s', s.temperature != null ? +s.temperature.toFixed(1) : null, '°C', 20, 24);
    setEnvCell('e-hum', 'e-hum-s', s.humidity != null ? +s.humidity.toFixed(1) : null, '%', 57, 63);
    setEnvCell('e-light', 'e-light-s', s.light != null ? Math.round(s.light) : null, 'lx', 360, 1600);
    setEnvCell('e-noise', 'e-noise-s', s.noise != null ? Math.round(s.noise) : null, 'ADC', 2000, 2500);

    // PIR — ML sends "motion" (0 or 1)
    if (s.motion != null) {
        const pirEl = document.getElementById('e-pir');
        const pirSt = document.getElementById('e-pir-s');
        pirEl.textContent = s.motion ? 'DETECTED' : 'ABSENT';
        pirSt.textContent = s.motion ? 'Motion present' : 'No motion';
        pirSt.className = 'env-status ' + (s.motion ? 'es-ok' : 'es-warn');
    }

    // ML classifier label banner
    const ef = d.env_focus || {};
    const lbl = ef.label || '—';
    const valEl = document.getElementById('efb-val');
    valEl.textContent = lbl;
    valEl.className = 'efb-val ' +
        (lbl === 'Focused' ? 'focused' :
            lbl === 'Half Focus' ? 'half' :
                lbl === 'Not Focused' ? 'not-focused' : '');

    // Confidence breakdown
    const conf = ef.conf || {};
    const confStr = Object.entries(conf)
        .map(([k, v]) => `${k}: ${(v * 100).toFixed(0)}%`)
        .join('  ·  ');
    document.getElementById('efb-conf').textContent = confStr;
}

// ── CV data renderer (students / attendance / alerts) ────────────────────
function render(d) {
    const sum = d.summary || {};
    const total = sum.total || 0;
    document.getElementById('s-total').innerHTML = total;
    document.getElementById('s-focused').innerHTML = sum.focused || 0;
    document.getElementById('s-unfocused').innerHTML = sum.not_focused || 0;
    const rate = total ? (sum.focus_pct || 0) : null;
    document.getElementById('s-rate').innerHTML = rate !== null
        ? `${rate}<span class="metric-unit">%</span>`
        : `—<span class="metric-unit">%</span>`;
    document.getElementById('s-foot').textContent = total
        ? `${total} student${total !== 1 ? 's' : ''} in frame`
        : 'No students in frame';

    focusHist.push(total ? (sum.focus_pct || 0) : null);
    focusHist = focusHist.slice(-60);
    chart.data.datasets[0].data = focusHist;
    chart.update('none');

    // Students
    const students = d.students || {};
    const keys = Object.keys(students);
    const lbl = `${keys.length} detected`;
    document.getElementById('stu-count').textContent = lbl;
    document.getElementById('stu-count2').textContent = lbl;
    const html = keys.length
        ? keys.map(id => stuCard(id, students[id])).join('')
        : '<div class="stu-empty-light">Waiting for students…</div>';
    document.getElementById('stu-grid').innerHTML = html;
    document.getElementById('stu-grid2').innerHTML = html;

    // Alerts
    const alerts = d.alerts || [];
    const cnt = alerts.length;

    document.getElementById('sidebar-alert-list').innerHTML = cnt
        ? alerts.map(a => `<div class="sidebar-al-item">
        <div class="sidebar-al-who">${a.student}</div>
        <div class="sidebar-al-msg">${a.message}</div>
        <div class="sidebar-al-t">${a.time}</div>
      </div>`).join('')
        : '<div class="sidebar-al-empty">No alerts yet</div>';

    const ab = document.getElementById('alert-badge');
    const nb = document.getElementById('alert-nav-badge');
    if (cnt > 0) {
        ab.textContent = cnt; ab.style.display = '';
        nb.textContent = cnt; nb.style.display = '';
    } else {
        ab.style.display = 'none';
        nb.style.display = 'none';
    }

    document.getElementById('alert-history-list').innerHTML = cnt
        ? alerts.map(a => `<div class="al-item">
        <div class="al-who">${a.student}</div>
        <div class="al-msg">${a.message}</div>
        <div class="al-t">${a.time}</div>
      </div>`).join('')
        : '<div class="al-empty">No alerts yet</div>';
    document.getElementById('alert-history-lbl').textContent =
        `${cnt} alert${cnt !== 1 ? 's' : ''}`;

    // Attendance
    const att = d.attendance || [];
    document.getElementById('att-badge').textContent = att.length;
    document.getElementById('att-lbl').textContent =
        `${att.length} student${att.length !== 1 ? 's' : ''} present`;
    document.getElementById('att-body').innerHTML = att.length
        ? att.map((a, i) => `<tr>
        <td class="row-num">${String(i + 1).padStart(2, '0')}</td>
        <td><span class="att-av-sm">${(a.name || '?').slice(0, 2).toUpperCase()}</span>${a.name}</td>
        <td class="att-id-cell">${a.id}</td>
        <td class="att-time-cell">${a.time}</td>
        <td><span class="present-badge">✓ Present</span></td>
      </tr>`).join('')
        : '<tr><td colspan="5" class="att-empty-light">No students recognised yet</td></tr>';
}

// ── Poll CV (port 5050) ──────────────────────────────────────────────────
async function poll() {
    const dot = document.getElementById('conn-dot');
    const txt = document.getElementById('conn-txt');
    try {
        const r = await fetch(API_CV, { cache: 'no-store' });
        const d = await r.json();
        if (!cvConnected) {
            cvConnected = true;
            dot.className = 'conn-dot blink';
            dot.removeAttribute('style');
            txt.textContent = 'Online';
        }
        render(d);
    } catch {
        cvConnected = false;
        dot.className = 'conn-dot err';
        txt.textContent = 'Offline';
    }
}

// ── Poll ENV (port 5051) ─────────────────────────────────────────────────
async function pollEnv() {
    const dot = document.getElementById('env-conn-dot');
    const txt = document.getElementById('env-conn-txt');
    try {
        const r = await fetch(API_ENV, { cache: 'no-store' });
        const d = await r.json();
        if (!envConnected) {
            envConnected = true;
            dot.className = 'env-conn-dot online';
            txt.textContent = 'Online';
        }
        renderEnv(d);
    } catch {
        envConnected = false;
        dot.className = 'env-conn-dot err';
        txt.textContent = 'Offline';
        // clear env display to show "no data" state
        document.getElementById('env-no-data').style.display = '';
        document.getElementById('env-focus-banner').style.display = 'none';
    }
}

// Update toggleTheme to save preference
function toggleTheme() {
    const html = document.documentElement;
    const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    sessionStorage.setItem('theme', next);
    if (chart) updateChartTheme();
}

// ── Start both pollers ───────────────────────────────────────────────────
poll(); setInterval(poll, 1000);
pollEnv(); setInterval(pollEnv, 1000);
