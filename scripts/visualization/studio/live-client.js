/** Local authoritative simulator connection. All displayed states come from the service. */
export class LocalMissionClient {
  constructor({ onState, onConnection, base = 'http://127.0.0.1:8882' }) {
    this.base = base;
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
    if (this.sessionId && state.session_id !== this.sessionId) throw new Error('Local simulator restarted; reconnecting');
    if (state.revision <= this.revision) return;
    this.revision = state.revision;
    this.onState(state);
  }

  async command(name, body = {}) {
    if (!this.connected) throw new Error('Live connection unavailable');
    const state = await this.request('/' + name, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, 15000);
    this.receive(state);
    return state;
  }

  async start() {
    while (!this.closed) {
      try {
        if (!this.connected) {
          const health = await this.request('/health', {}, 4000);
          if (health.service !== 'orbexa-live-v1' || health.status !== 'ready') throw new Error('Local simulator unavailable');
          this.sessionId = health.session_id || null;
          this.revision = -1;
          this.connected = true;
          this.onConnection(true);
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
