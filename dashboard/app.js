/**
 * app.js — SupportEnv Dashboard Logic
 *
 * Connects to the FastAPI server at API_BASE, drives a simulated
 * rule-based agent step-by-step, and renders everything live.
 */

// ── Config ──────────────────────────────────────────────────────────────────
const API_BASE = window.location.protocol === 'file:' 
  ? 'http://127.0.0.1:7860' 
  : window.location.origin;

const STEP_DELAY_MS = 700;  // ms between agent steps (animation pacing)

// ── State ────────────────────────────────────────────────────────────────────
let currentTask   = 'easy';
let rewardHistory = [];
let stepLog       = [];
let isRunning     = false;
let episodeDone   = false;

// ── WebSocket State ────────────────────────────────────────────────────────
let ws            = null;
let reconnectWait = 1000; // ms
const MAX_RECONNECT = 30000; // 30s max wait
let heartbeatTimer = null;

// ── Task metadata ────────────────────────────────────────────────────────────
const TASK_DESCS = {
  easy:   'Classify 5 tickets into the correct category (billing, technical, general, complaint, positive). Actions: classify.',
  medium: 'Classify and prioritize 6 tickets. You must assign both category and priority correctly. Actions: classify, prioritize.',
  hard:   'Full resolution pipeline across 7 tickets. Classify → prioritize → respond or escalate → close. All 5 actions available.',
};

const TASK_ALLOWED = {
  easy:   ['classify'],
  medium: ['classify', 'prioritize'],
  hard:   ['classify', 'prioritize', 'respond', 'escalate', 'close'],
};

// ── Category/priority color mapping ──────────────────────────────────────────
const CAT_COLORS = {
  billing:   'badge-amber',
  technical: 'badge-cyan',
  general:   'badge-green',
  complaint: 'badge-red',
  positive:  'badge-purple',
};

const PRI_COLORS = {
  high:   'badge-red',
  medium: 'badge-amber',
  low:    'badge-green',
};

const ACTION_COLORS = {
  classify:   '#00d4ff',
  prioritize: '#ffab00',
  respond:    '#00e676',
  escalate:   '#ff5252',
  close:      '#b388ff',
};

// ─────────────────────────────────────────────────────────────────────────────
// API helpers
// ─────────────────────────────────────────────────────────────────────────────

async function apiPost(path, body) {
  const res = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API ${path} returned ${res.status}`);
  return res.json();
}

async function apiGet(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(`API ${path} returned ${res.status}`);
  return res.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// Task selector
// ─────────────────────────────────────────────────────────────────────────────

function selectTask(task) {
  currentTask = task;
  ['easy', 'medium', 'hard'].forEach(t => {
    document.getElementById(`btn-${t}`).classList.toggle('active', t === task);
  });
  document.getElementById('taskDesc').textContent = TASK_DESCS[task];
}

// ─────────────────────────────────────────────────────────────────────────────
// Reset
// ─────────────────────────────────────────────────────────────────────────────

async function handleReset() {
  if (isRunning) return;

  document.getElementById('resetBtn').disabled = true;
  document.getElementById('runBtn').disabled   = false;

  rewardHistory = [];
  stepLog       = [];
  episodeDone   = false;

  // Reset UI
  renderRewardChart();
  renderLog();
  hideScore();
  setAgentBanner(false);
  updateStats({ task: currentTask, step_count: 0, max_steps: '—', cumulative_reward: 0, tickets: [] });

  try {
    await apiPost('/env/reset', { task: currentTask, seed: 42 });
    const state = await apiGet('/env/state');
    renderTickets(state.tickets, []);
    updateStats(state, true);
    document.getElementById('taskDesc').textContent = TASK_DESCS[currentTask];
  } catch (err) {
    showApiError(err);
  }

  document.getElementById('resetBtn').disabled = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Inject Custom Ticket
// ─────────────────────────────────────────────────────────────────────────────

async function handleInjectTicket() {
  const text     = document.getElementById('customTicketText').value;
  const category = document.getElementById('customTicketCat').value;
  const priority = document.getElementById('customTicketPri').value;
  const escalate = document.getElementById('customTicketEscalate').checked;

  if (!text.trim()) {
    alert("Please enter ticket text.");
    return;
  }

  const statusEl = document.getElementById('injectStatus');
  statusEl.textContent = 'injecting…';
  statusEl.className   = 'badge badge-amber';

  try {
    await apiPost('/env/ticket', {
      text,
      category,
      priority,
      persona: "polite",
      requires_escalation: escalate
    });
    
    // Clear form
    document.getElementById('customTicketText').value = '';
    statusEl.textContent = 'success';
    statusEl.className   = 'badge badge-green';
    setTimeout(() => {
      statusEl.textContent = 'ready';
      statusEl.className   = 'badge badge-purple';
    }, 2000);

  } catch (err) {
    statusEl.textContent = 'error';
    statusEl.className   = 'badge badge-error';
    showApiError(err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Run agent
// ─────────────────────────────────────────────────────────────────────────────

async function handleRun() {
  if (isRunning || episodeDone) return;
  isRunning = true;

  document.getElementById('runBtn').disabled   = true;
  document.getElementById('resetBtn').disabled = true;

  try {
    // Fetch current state to get tickets
    const state = await apiGet('/env/state');
    const tickets = state.tickets;
    const allowed = TASK_ALLOWED[currentTask];

    let done = state.done;
    let safetyCounter = 0;
    let lastActionLog = null; // To detect repetitive failed actions
    let repeatActionCount = 0;

    while (!done && safetyCounter < 150) {
      safetyCounter++;

      // Find next action to take using client-side rule-based agent
      const action = decideNextAction(tickets, allowed);
      if (!action) break;

      // Loop prevention: If we've tried the exact same action/ticket too many times without success
      const actionKey = `${action.action_type}:${action.ticket_id}`;
      if (actionKey === lastActionLog) {
        repeatActionCount++;
        if (repeatActionCount > 3) {
          console.error(`🛑 Agent stuck in a loop on ticket #${action.ticket_id}. Stopping.`);
          appendToTerminal(`System: [ERROR] Agent stuck in loop on Ticket #${action.ticket_id}. Manual intervention required.`, 'error');
          break;
        }
      } else {
        lastActionLog = actionKey;
        repeatActionCount = 0;
      }

      setAgentBanner(true, `Acting on ticket #${action.ticket_id}: ${action.action_type}…`);
      await sleep(STEP_DELAY_MS);

      const result = await apiPost('/env/step', action);
      
      // Compliance: Handles tuple-as-dict {observation, reward, done, info}
      const obs = result.observation;
      done = result.done;

      // Update local ticket state
      syncTicketFromObs(tickets, obs.tickets);

      const reward = result.reward;
      rewardHistory.push(reward);
      stepLog.push({
        step:     result.info.step,
        action:   action.action_type,
        ticketId: action.ticket_id,
        reward,
        reason:   result.info.reason,
        cumulative: obs.cumulative_reward,
      });

      renderTickets(tickets, [action.ticket_id]);
      renderRewardChart();
      renderLog();
      updateStats({
        task:              currentTask,
        step_count:        result.info.step,
        max_steps:         state.max_steps,
        cumulative_reward: obs.cumulative_reward,
        tickets,
      });
    }

    setAgentBanner(false);
    episodeDone = true;

    // Fetch final grade
    const grade = await apiGet('/env/grade');
    showScore(grade);
    updateStats({ score: grade.score });

  } catch (err) {
    showApiError(err);
  }

  isRunning = false;
  document.getElementById('resetBtn').disabled = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Client-side rule agent (mirrors server RuleBasedAgent logic)
// ─────────────────────────────────────────────────────────────────────────────

function decideNextAction(tickets, allowed) {
  for (const ticket of tickets) {
    if (ticket.status !== 'open') continue;
    const text = ticket.text.toLowerCase();

    if (allowed.includes('classify') && !ticket.category) {
      return { action_type: 'classify', ticket_id: ticket.id, content: guessCategory(text) };
    }
    if (allowed.includes('prioritize') && !ticket.priority) {
      return { action_type: 'prioritize', ticket_id: ticket.id, content: guessPriority(text) };
    }
    if (allowed.includes('respond') || allowed.includes('escalate')) {
      if (shouldEscalate(text) && allowed.includes('escalate')) {
        return { action_type: 'escalate', ticket_id: ticket.id, content: null };
      }
      if (allowed.includes('respond')) {
        return { action_type: 'respond', ticket_id: ticket.id, content: generateResponse(text) };
      }
    }
  }
  return null; // all done
}

function guessCategory(text) {
  if (/refund|charge|invoice|billing|payment|subscription|cancel/.test(text)) return 'billing';
  if (/crash|error|bug|slow|login|api|password|blank|500/.test(text))          return 'technical';
  if (/angry|unacceptable|furious|disgrace|terrible|disappointed/.test(text))  return 'complaint';
  if (/thank|great|fantastic|love|amazing|excellent|5 stars/.test(text))       return 'positive';
  return 'general';
}

function guessPriority(text) {
  if (/immediately|urgent|critical|blocking|legal|crash|furious|deleted|twice/.test(text)) return 'high';
  if (/slow|intermittent|sometimes|occasional/.test(text)) return 'medium';
  return 'low';
}

function shouldEscalate(text) {
  return /furious|legal action|disgrace|ignoring|unacceptable|shameful/.test(text);
}

function generateResponse(text) {
  if (/refund|charge|billing|invoice|payment/.test(text))
    return 'We sincerely apologize for the billing issue. A refund has been processed and will appear within 3–5 business days.';
  if (/crash|error|bug|slow|login|api|password/.test(text))
    return 'Our engineering team has been notified of this technical issue and is actively investigating. We will provide an update within 24 hours.';
  if (/angry|disappointed|terrible|disgrace/.test(text))
    return 'We deeply apologize for your experience. A senior manager will contact you within 2 hours to resolve this complaint.';
  if (/thank|great|amazing|love/.test(text))
    return 'Thank you so much for your kind words! Your feedback has been shared with the entire team.';
  return 'Thank you for contacting us. A support representative will respond to your enquiry within 24 hours.';
}

function syncTicketFromObs(local, obsTickets) {
  obsTickets.forEach(ot => {
    const lt = local.find(t => t.id === ot.id);
    if (lt) {
      lt.category = ot.category;
      lt.priority = ot.priority;
      lt.status   = ot.status;
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────────────────────

function renderTickets(tickets, flashIds = []) {
  const grid = document.getElementById('ticketGrid');
  if (!tickets || tickets.length === 0) {
    grid.innerHTML = `<div class="empty-state"><div class="empty-icon">📋</div><div class="empty-text">No tickets loaded — click <strong>Reset</strong> to start</div></div>`;
    return;
  }

  document.getElementById('ticketCountBadge').textContent = `${tickets.length} ticket${tickets.length !== 1 ? 's' : ''}`;

  grid.innerHTML = tickets.map((t, i) => {
    const statusClass = t.status !== 'open' ? `status-${t.status}` : '';
    const flashClass  = flashIds.includes(t.id) ? 'just-updated' : '';
    const delay       = i * 0.05;

    const catBadge = t.category
      ? `<span class="tag badge ${CAT_COLORS[t.category] || 'badge-cyan'}">${t.category}</span>`
      : `<span class="tag" style="background:rgba(255,255,255,0.06);color:var(--text-muted)">unclassified</span>`;

    const priBadge = t.priority
      ? `<span class="tag badge ${PRI_COLORS[t.priority] || 'badge-cyan'}">${t.priority}</span>`
      : '';

    const statusBadge = t.status !== 'open'
      ? `<span class="tag badge ${t.status === 'closed' ? 'badge-green' : 'badge-amber'}">${t.status}</span>`
      : '';

    return `
      <div class="ticket-card ${statusClass} ${flashClass}" style="animation-delay:${delay}s" id="ticket-${t.id}">
        <div class="ticket-id">#${String(t.id).padStart(3,'0')}</div>
        <div class="ticket-text">${escapeHtml(t.text)}</div>
        <div class="ticket-tags">
          ${catBadge}
          ${priBadge}
          ${statusBadge}
        </div>
      </div>`;
  }).join('');
}

function renderRewardChart() {
  const chart = document.getElementById('rewardChart');
  const last  = rewardHistory[rewardHistory.length - 1] ?? 0;

  document.getElementById('rewardBadge').textContent =
    (last >= 0 ? 'Δ +' : 'Δ ') + last.toFixed(3);

  if (rewardHistory.length === 0) {
    chart.innerHTML = '';
    return;
  }

  const maxAbs = Math.max(...rewardHistory.map(Math.abs), 0.1);
  const MAX_H  = 100;

  chart.innerHTML = rewardHistory.slice(-40).map((r, i, arr) => {
    const pct    = Math.abs(r) / maxAbs;
    const height = Math.max(4, pct * MAX_H);
    const color  = r >= 0
      ? `hsl(${140 + pct * 20}, 80%, 50%)`
      : `hsl(${0 + pct * 20}, 75%, 55%)`;
    const delay  = (i / arr.length) * 0.3;
    return `<div class="chart-bar" style="height:${height}px;background:${color};animation-delay:${delay}s" title="Step ${i+1}: ${r >= 0 ? '+' : ''}${r.toFixed(3)}"></div>`;
  }).join('');
}

function renderLog() {
  const scroll = document.getElementById('logScroll');
  document.getElementById('logBadge').textContent = `${stepLog.length} step${stepLog.length !== 1 ? 's' : ''}`;

  if (stepLog.length === 0) {
    scroll.innerHTML = `<div style="text-align:center;color:var(--text-muted);font-size:12px;padding:20px;">Waiting for first action…</div>`;
    return;
  }

  const color = e => ACTION_COLORS[e.action] || '#00d4ff';
  scroll.innerHTML = [...stepLog].reverse().map(e => `
    <div class="log-entry">
      <span class="log-step">${String(e.step).padStart(2,'0')}</span>
      <span class="log-action" style="color:${color(e)}">${e.action}</span>
      <span class="log-reward ${e.reward >= 0 ? 'pos' : 'neg'}">${e.reward >= 0 ? '+' : ''}${e.reward.toFixed(2)}</span>
      <span class="log-reason">${escapeHtml(e.reason)}</span>
    </div>
  `).join('');
}

function updateStats(data, fromReset = false) {
  if (data.task)             document.getElementById('statTask').textContent = data.task.toUpperCase();
  if (data.step_count !== undefined) {
    document.getElementById('statStep').textContent = data.step_count;
    if (data.max_steps) document.getElementById('statStepMax').textContent = `/ ${data.max_steps} max`;
  }
  if (data.cumulative_reward !== undefined) {
    const r   = parseFloat(data.cumulative_reward);
    const el  = document.getElementById('statReward');
    el.textContent = (r >= 0 ? '+' : '') + r.toFixed(3);
    el.className   = 'stat-value ' + (r > 0 ? 'positive' : r < 0 ? 'negative' : '');
  }
  if (data.tickets) {
    const total    = data.tickets.length;
    const resolved = data.tickets.filter(t => t.status !== 'open').length;
    document.getElementById('statTickets').textContent  = total || '—';
    document.getElementById('statResolved').textContent = `${resolved} resolved`;
  }
  if (data.score !== undefined) {
    document.getElementById('statScore').textContent = (data.score * 100).toFixed(1) + '%';
  }
}

function showScore(grade) {
  const panel = document.getElementById('scorePanel');
  panel.classList.add('visible');

  document.getElementById('scorePct').textContent     = (grade.score * 100).toFixed(1) + '%';
  document.getElementById('scoreSummary').textContent = grade.summary;

  const rows = document.getElementById('breakdownRows');
  rows.innerHTML = Object.entries(grade.breakdown || {}).map(([key, val]) => `
    <div class="breakdown-row">
      <span class="breakdown-key">${key}</span>
      <div class="breakdown-bar-wrap">
        <div class="breakdown-bar-bg">
          <div class="breakdown-bar-fill" style="width:${(val*100).toFixed(1)}%"></div>
        </div>
      </div>
      <span class="breakdown-val">${(val*100).toFixed(0)}%</span>
    </div>
  `).join('');
}

function hideScore() {
  document.getElementById('scorePanel').classList.remove('visible');
  document.getElementById('statScore').textContent = '—';
}

function setAgentBanner(visible, text = 'Agent is thinking…') {
  const banner = document.getElementById('agentBanner');
  banner.classList.toggle('visible', visible);
  document.getElementById('agentBannerText').textContent = text;
}

function showApiError(err) {
  setAgentBanner(true, `⚠ API Error: ${err.message}. Make sure the server is running on port 7860.`);
  console.error(err);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  selectTask('easy');
  initWebSocket();
});

// ── WebSocket Client ────────────────────────────────────────────────────────

function initWebSocket() {
  const wsStatus = document.getElementById('wsStatus');
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl    = `${protocol}//${window.location.host}${window.location.pathname.replace(/\/dashboard\/?$/, '')}/ws`;
  
  // If running from file: or weird origin, use hardcoded base
  const finalUrl = (window.location.protocol === 'file:' || !window.location.host) 
    ? 'ws://127.0.0.1:7860/ws' 
    : wsUrl;

  console.log(`📡 Connecting to Live Stream: ${finalUrl}`);
  ws = new WebSocket(finalUrl);

  ws.onopen = () => {
    console.log('✅ Live Stream Connected');
    wsStatus.textContent = 'online';
    wsStatus.className   = 'badge badge-green';
    reconnectWait = 1000; // reset wait
    appendToTerminal('System: [CONNECTED] Live-Stream Terminal initialized.', 'system');

    // Start Heartbeat to keep connection alive through proxies
    if (heartbeatTimer) clearInterval(heartbeatTimer);
    heartbeatTimer = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 15000); // 15s
  };

  ws.onmessage = async (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'system') {
        appendToTerminal(`System: ${data.message}`, 'system');
      } else if (data.type === 'event') {
        const colorClass = data.reward > 0 ? 'success' : data.reward < 0 ? 'error' : '';
        appendToTerminal(data.message, colorClass);
        
        // Auto-refresh UI on key events
        if (data.event === 'ticket_added' || data.event === 'reset') {
          const state = await apiGet('/env/state');
          renderTickets(state.tickets);
          updateStats(state);
        }
      }
    } catch (err) {
      console.error('WS Message Error:', err);
    }
  };

  ws.onclose = (event) => {
    console.warn(`❌ Live Stream Disconnected (Code: ${event.code}). Reconnecting in ${reconnectWait}ms…`);
    wsStatus.textContent = 'offline';
    wsStatus.className   = 'badge badge-error';
    
    if (heartbeatTimer) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }

    let reason = event.reason || "Abnormal Closure";
    appendToTerminal(`System: [DISCONNECTED] Code: ${event.code} | Reason: ${reason} | Reconnecting...`, 'error');
    
    setTimeout(() => {
      reconnectWait = Math.min(reconnectWait * 1.5, MAX_RECONNECT);
      initWebSocket();
    }, reconnectWait);
  };

  ws.onerror = (err) => {
    console.error('WS Error:', err);
  };
}

function appendToTerminal(message, type = '') {
  const grid = document.getElementById('terminalGrid');
  const line = document.createElement('div');
  line.className = `terminal-line ${type}`;
  
  const now = new Date();
  const timeStr = now.toTimeString().split(' ')[0];
  
  line.textContent = `[${timeStr}] ${message}`;
  grid.appendChild(line);

  // Auto-scroll to bottom
  grid.scrollTop = grid.scrollHeight;
  
  // Keep only last 100 lines
  while (grid.children.length > 100) {
    grid.removeChild(grid.firstChild);
  }
}
