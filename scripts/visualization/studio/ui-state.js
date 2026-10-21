// URL state is presentation only; it never changes a simulation declaration.
export const STUDIO_TABS = Object.freeze(['mission', 'plots', 'theory']);
export const CAMERA_MODES = Object.freeze(['mission', 'overview', 'docking', 'top', 'follow']);
export const PLAYBACK_SPEEDS = Object.freeze([1, 10, 30, 100, 300]);
export const VIEW_FLAGS = Object.freeze({ paths: true, tubes: true, grid: false, envelope: false, loop: false });

export function theoryKey(document) {
  return String(document?.id || document?.href?.split(/[?#]/)[0].split('/').filter(Boolean).at(-1)?.replace(/\.html?$/i, '') || '');
}

function number(value, fallback) {
  if (!['number', 'string'].includes(typeof value) || String(value).trim() === '') return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function normalizeUiState(input = {}, limits = {}) {
  const theories = limits.theories || [], chasers = limits.chasers || [];
  const start = number(limits.start, 0), end = Math.max(start, number(limits.end, start));
  let topic = theories.includes(input.theory) ? input.theory : theories[0] || '';
  if (typeof input.theory === 'string' && theories.includes(theoryKey({ href: input.theory }))) topic = theoryKey({ href: input.theory });
  if (!theories.includes(input.theory) && /^\d+$/.test(String(input.theory)) && theories[Number(input.theory)]) topic = theories[Number(input.theory)];
  let chaser = chasers.includes(input.chaser) ? input.chaser : chasers[0] || '';
  if (!chasers.includes(input.chaser) && /^\d+$/.test(String(input.chaser)) && chasers[Number(input.chaser)]) chaser = chasers[Number(input.chaser)];
  const speed = number(input.speed, 30);
  const state = { tab: STUDIO_TABS.includes(input.tab) ? input.tab : 'mission', theory: topic, chaser,
    camera: CAMERA_MODES.includes(input.camera) ? input.camera : 'mission',
    speed: PLAYBACK_SPEEDS.includes(speed) ? speed : 30,
    time: Math.max(start, Math.min(end, number(input.time, start))) };
  for (const [key, fallback] of Object.entries(VIEW_FLAGS)) {
    state[key] = [true, '1', 'true'].includes(input[key]) ? true : [false, '0', 'false'].includes(input[key]) ? false : fallback;
  }
  return state;
}

export function readUiState(address, limits) {
  const url = new URL(address), hash = url.hash.slice(1), values = Object.fromEntries(url.searchParams);
  if (STUDIO_TABS.includes(hash) && !url.searchParams.has('tab')) values.tab = hash;
  else if (hash.startsWith('tab=') || hash.startsWith('theory=')) {
    for (const [key, value] of new URLSearchParams(hash)) if (!url.searchParams.has(key)) values[key] = value;
    if (hash.startsWith('theory=') && !values.tab) values.tab = 'theory';
  }
  return normalizeUiState(values, limits);
}

export function writeUiState(address, input, limits = {}) {
  const url = new URL(address), state = normalizeUiState(input, limits);
  url.searchParams.set('tab', state.tab);
  for (const key of ['theory', 'chaser', 'camera', 'speed', 'time']) {
    const value = state[key], defaults = { theory: '', chaser: '', camera: 'mission', speed: 30, time: number(limits.start, 0) };
    if (value === defaults[key]) url.searchParams.delete(key);
    else url.searchParams.set(key, String(value));
  }
  for (const [key, fallback] of Object.entries(VIEW_FLAGS)) {
    if (state[key] === fallback) url.searchParams.delete(key);
    else url.searchParams.set(key, state[key] ? '1' : '0');
  }
  // A stale link cannot select an older record hidden in the catalog.
  if (limits.mission) url.searchParams.set('mission', limits.mission);
  else url.searchParams.delete('mission');
  if (STUDIO_TABS.includes(url.hash.slice(1)) || /^#(?:tab|theory)=/.test(url.hash)) url.hash = '';
  return url;
}

export function isInteractiveTarget(target) {
  return Boolean(target?.isContentEditable || target?.closest?.('input, select, textarea, button, a, dialog, [contenteditable], [role="slider"], [role="textbox"], [role="combobox"], [role="button"], [role="checkbox"]'));
}

export class StudioHistory {
  constructor({ window, limits, capture, apply }) {
    this.window = window; this.limits = limits; this.capture = capture; this.apply = apply;
    this.restoring = false; this.timer = null;
    this.onNavigation = () => this.restore();
    this.onLeave = () => this.save();
    window.addEventListener('popstate', this.onNavigation);
    window.addEventListener('hashchange', this.onNavigation);
    window.addEventListener('pagehide', this.onLeave);
  }
  restore() {
    this.cancel(); this.restoring = true;
    try { this.apply(readUiState(this.window.location.href, this.limits())); }
    finally { this.restoring = false; }
  }
  cancel() { if (this.timer !== null) this.window.clearTimeout(this.timer); this.timer = null; }
  save(push = false) {
    this.cancel();
    if (this.restoring) return;
    const state = this.capture();
    if (!state) return;
    const url = writeUiState(this.window.location.href, state, this.limits());
    if (url.href === this.window.location.href) return;
    const previous = this.window.history.state;
    const value = { ...(previous && typeof previous === 'object' ? previous : {}), orbexaStudio: true };
    try { this.window.history[push ? 'pushState' : 'replaceState'](value, '', url.href); }
    catch { /* A browser that denies History writes still has a usable local view. */ }
  }
  schedule() {
    if (this.restoring || this.timer !== null) return;
    this.timer = this.window.setTimeout(() => { this.timer = null; this.save(); }, 750);
  }
  destroy() {
    this.cancel();
    this.window.removeEventListener('popstate', this.onNavigation);
    this.window.removeEventListener('hashchange', this.onNavigation);
    this.window.removeEventListener('pagehide', this.onLeave);
  }
}
