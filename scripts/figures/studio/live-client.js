/** Local authoritative simulator connection. All displayed states come from the service. */
export class LocalMissionClient {
  constructor({ onState, onConnection, base }) {
    this.bases = base ? [base] : ['http://127.0.0.1:8882', 'http://127.0.0.1:8894'];
    this.base = this.bases[0];
    this.onState = onState;
    this.onConnection = onConnection;
    this.revision = -1;
    this.sessionId = null;
    this.connected = false;
    this.closed = false;
  }

  async request(route, options = {}, timeout = 27000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(this.base + route, { ...options, signal: controller.signal, cache: 'no-store' });
      const body = await response.json();
      if (!response.ok) throw new Error(body.message || body.error || `Local simulator returned ${response.status}`);
      return body;
    } finally { clearTimeout(timer); }
  }

  receive(state) {
    const statuses = ['idle', 'starting', 'running', 'pausing', 'paused', 'completing', 'completed', 'stopped', 'failed', 'cancelled'];
    if (!Number.isInteger(state?.revision) || state.revision < 0 || !statuses.includes(state.status)) throw new Error('Unsupported local simulator state.');
    if (state.execution_model !== 'sensed_nonlinear') throw new Error('This viewer requires the current sensed nonlinear simulator.');
    if (this.sessionId && state.session_id !== this.sessionId) throw new Error('Local simulator restarted; reconnecting');
    if (state.revision <= this.revision) return;
    this.revision = state.revision;
    this.onState(state);
  }

  async command(name, body = {}) {
    if (!this.connected || !this.sessionId) throw new Error('Live connection unavailable');
    const envelope = { protocol_version: 2, session_id: this.sessionId, payload: body };
    const state = await this.request('/' + name, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(envelope) }, 15000);
    this.receive(state);
    return state;
  }

  async start() {
    while (!this.closed) {
      try {
        if (!this.connected) {
          let health;
          for (const base of this.bases) {
            this.base = base;
            try {
              const candidate = await this.request('/health', {}, 4000);
              if (candidate.service === 'orbexa-live-v2' && candidate.protocol_version === 2 &&
                  candidate.execution_model === 'sensed_nonlinear' && candidate.status === 'ready' &&
                  typeof candidate.session_id === 'string' && candidate.session_id.length) {
                health = candidate; break;
              }
            } catch { /* Try only the explicitly configured loopback services. */ }
          }
          if (!health) throw new Error('Start the current sensed nonlinear ORBEX-A simulator.');
          this.sessionId = health.session_id || null;
          this.revision = -1;
          this.receive(await this.request('/state?after=0&timeout=0'));
          this.connected = true;
          this.onConnection(true);
          continue;
        }
        this.receive(await this.request(`/state?after=${Math.max(0, this.revision)}&timeout=${this.revision < 0 ? 0 : 20}`));
      } catch (error) {
        this.connected = false;
        this.onConnection(false, error.message);
        await new Promise(resolve => setTimeout(resolve, 3000));
      }
    }
  }

  close() { this.closed = true; }
}

/** Saved replay controls never create a client, poll, or send simulator commands. */
export function recordedControls({ liveUiHref } = {}) {
  const get = id => document.getElementById(id);
  const group = document.querySelector('#simulation-controls .simulation-knobs');
  group.setAttribute('role', 'group');
  group.setAttribute('aria-label', 'Next run settings');
  group.title = 'Next run settings; these controls do not report saved mission values.';
  ['release-distance', 'cluster-spacing', 'target-spin', 'live-run', 'live-pause', 'live-reset'].forEach(id => { get(id).disabled = true; });
  get('live-connection').textContent = 'Recorded mission';
  get('live-connection').dataset.connected = 'false';
  const link = get('open-live-simulator');
  let url;
  try {
    const href = liveUiHref === undefined ? link.getAttribute('href') : liveUiHref;
    if (typeof href !== 'string' || !href.trim()) throw new Error('Invalid local UI address');
    url = new URL(href, location.href);
    if (!['http:', 'https:'].includes(url.protocol) || !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) || url.username || url.password) throw new Error('Invalid local UI address');
  } catch {
    link.hidden = true;
    throw new Error('Live controls require a loopback HTTP(S) address.');
  }
  link.href = url.href;
  link.textContent = 'Live controls';
  link.title = 'Open the configured local simulator in a separate tab; your browser may ask before opening a local address.';
  link.hidden = false;
  return url.href;
}

export function simulationControls({ onState, onReset, onFollow, onError }) {
  const get = id => document.getElementById(id);
  const knobs = [['release-distance', 'release_distance_m', ' m'], ['cluster-spacing', 'cluster_spacing_m', ' m'], ['target-spin', 'target_spin_deg_s', '°/s']];
  let state, connected = false, connectionFailed = false, busy = false, dirty = false, timer;
  const hosted = location.protocol === 'https:';
  let resetQueue = Promise.resolve();
  const values = () => Object.fromEntries(knobs.map(([id, name]) => [name, Number(get(id).value)]));
  const labels = () => knobs.forEach(([id, , unit]) => { get(id + '-value').textContent = get(id).value + unit; });
  const paint = () => {
    get('live-connection').textContent = !connected ? hosted ? connectionFailed ? 'Live connection unavailable' : 'Connecting…' : 'Local simulator offline' : busy ? 'Updating…' : dirty ? 'Ready' : ({idle:'Ready',starting:'Starting…',running:'Computing approach',pausing:'Pausing…',paused:'Paused',completing:'Completing mission…',completed:'Complete',stopped:'Stopped',failed:'Stopped · see details',cancelled:'Ready'})[state?.status] || 'Connected';
    get('live-connection').dataset.connected = String(connected);
    get('open-live-simulator').hidden = !(hosted && connectionFailed && !connected);
    knobs.forEach(([id]) => { get(id).disabled = !connected || busy; });
    get('live-run').disabled = !connected || busy;
    get('live-pause').disabled = !connected || busy || !['running','pausing','paused'].includes(state?.status);
    get('live-pause').textContent = state?.status === 'paused' ? 'Resume' : 'Pause';
    get('live-reset').disabled = !connected || busy;
  };
  const client = new LocalMissionClient({
    base: document.body.dataset.liveApiBase || undefined,
    onConnection(value, error) { connected = value; connectionFailed = !value; get('live-connection').title = error || ''; paint(); },
    onState(value) {
      state = value;
      if (!dirty && value.parameters) {
        knobs.forEach(([id, name]) => { if (Number.isFinite(value.parameters[name])) get(id).value = value.parameters[name]; });
        labels();
      }
      if (!dirty) onState(value);
      paint();
    },
  });
  const action = async callback => {
    if (busy) return;
    busy = true; paint();
    try { await callback(); }
    catch (error) { onError(error.message); }
    finally { busy = false; paint(); }
  };
  const reset = () => {
    resetQueue = resetQueue.catch(() => {}).then(() => action(async () => {
      await client.command('reset');
      onReset();
    }));
    return resetQueue;
  };
  knobs.forEach(([id]) => get(id).addEventListener('input', () => {
    labels(); dirty = true; onReset(); paint();
    clearTimeout(timer); timer = setTimeout(reset, 350);
  }));
  get('live-run').addEventListener('click', () => action(async () => {
    clearTimeout(timer);
    if (dirty) {
      await client.command('reset');
      onReset();
    }
    onFollow();
    if (!dirty && ['running','pausing','paused','completing'].includes(state?.status)) {
      if (['pausing','paused'].includes(state.status)) await client.command('pause', { paused: false });
    } else {
      dirty = false;
      await client.command('run', values());
    }
  }));
  get('live-pause').addEventListener('click', () => action(() => client.command('pause', { paused: state?.status !== 'paused' })));
  get('live-reset').addEventListener('click', () => { clearTimeout(timer); dirty = true; reset(); });
  labels(); paint(); client.start();
  return client;
}
