import { MissionScene } from './scene.js';
import { MetricChart } from './charts.js';
import { SceneVideoExporter, downloadVideo } from './recorder.js';

const $ = id => document.getElementById(id);
const state = { index: 0, selected: 0, playing: false, tab: 'replay', generation: 0 };
const fmtDistance = value => value === null || !Number.isFinite(value) ? 'Not recorded' : value < 0.01 ? `${(value * 1000).toFixed(2)} mm` : `${value.toFixed(3)} m`;
const fmtRatio = value => value === null || !Number.isFinite(value) ? 'Not recorded' : value.toFixed(3);
let catalog, dataset, currentEntry, scene;
let exportAbort;
let lifecycleClock;

function outcomeLabel() {
  if (dataset?.synthetic) return dataset.outcome.label;
  const physical = currentEntry?.physical_outcome;
  if (!physical) return dataset?.outcome.label ?? '';
  return physical.success ? `${physical.attached_chaser_count} surface attachments · modeled mission completed` : 'Physical mission incomplete';
}

window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.source !== $('lifecycle-frame').contentWindow) return;
  if (event.data?.type === 'orbexa.video.export') {
    try { downloadVideo(event.data.blob, event.data.filename); }
    catch (error) { $('error').hidden = false; $('error').textContent = error.message; }
  }
  if (event.data?.type === 'orbexa.lifecycle.clock') {
    lifecycleClock = event.data;
    paintFooter();
  }
});

function paintFooter() {
  if (!dataset) return;
  if (state.tab === 'lifecycle') {
    $('footer-outcome').textContent = dataset.synthetic ? 'Synthetic post-capture physics fixture' : outcomeLabel();
    $('footer-clock').textContent = lifecycleClock ? `t = ${lifecycleClock.elapsed.toFixed(1)} s after attachment · node ${lifecycleClock.index + 1} / ${lifecycleClock.count}` : 'Post-capture timeline';
  } else {
    $('footer-outcome').textContent = `${outcomeLabel()} · recorded model evidence`;
    $('footer-clock').textContent = `q = ${dataset.q[state.index].toFixed(4)} rad · node ${state.index + 1} / ${dataset.q.length} · ${dataset.tracks.length} ${dataset.tracks.length === 1 ? 'chaser' : 'chasers'}`;
  }
}

try {
  scene = new MissionScene($('viewport'), selected => { state.selected = selected; $('chaser').value = String(selected); paint(); });
} catch (error) {
  $('scene-error').hidden = false;
  $('scene-error').textContent = '3D rendering is unavailable in this browser. Measurements, videos and evidence remain available. ' + error.message;
}

const charts = [
  new MetricChart($('goal-chart'), 'Goal distance', 'm', 'goal_distance_m', seek, { logarithmic: true }),
  new MetricChart($('occupancy-chart'), 'Position tube occupancy', 'ratio', 'position_occupancy', seek, { threshold: 1 }),
  new MetricChart($('reserve-chart'), 'Ancillary input reserve', 'ratio · preceding interval', 'input_reserve_ratio', seek, { threshold: 1 }),
];

async function readJSON(path, signal) {
  const response = await fetch(path, { signal, cache: 'no-cache' });
  if (!response.ok) throw new Error(`Cannot load saved evidence (${response.status}): ${path}`);
  return response.json();
}

function option(value, title) {
  const result = document.createElement('option');
  result.value = value;
  result.textContent = title;
  return result;
}

function setLink(id, href) {
  $(id).hidden = !href;
  if (href) $(id).href = href;
  else $(id).removeAttribute('href');
}

function stop() {
  state.playing = false;
  $('play').setAttribute('aria-label', 'Play recorded mission');
  $('play').innerHTML = '<svg viewBox="0 0 24 24"><path d="m8 4 12 8-12 8Z"/></svg>';
}

function clockAt(index) { return dataset.elapsed_s ? dataset.elapsed_s[index] : 30 * (dataset.q[index] - dataset.q[0]) / (dataset.q.at(-1) - dataset.q[0] || 1); }

function seek(index) {
  if (!dataset || state.exporting) return;
  stop();
  state.index = Math.max(0, Math.min(dataset.q.length - 1, Number(index)));
  paint();
}

function play() {
  if (!dataset || dataset.q.length < 2 || state.exporting) return;
  if (state.playing) { stop(); return; }
  if (state.index === dataset.q.length - 1) state.index = 0;
  state.playing = true;
  state.started = performance.now();
  state.startClock = clockAt(state.index);
  $('play').setAttribute('aria-label', 'Pause recorded mission');
  $('play').innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></svg>';
}

function paint() {
  if (!dataset) return;
  const i = state.index;
  const track = dataset.tracks[state.selected];
  $('scrub').value = String(i);
  $('scrub').setAttribute('aria-valuetext', `Node ${i + 1} of ${dataset.q.length}, true anomaly ${dataset.q[i].toFixed(4)} rad`);
  $('chaser-title').textContent = track.label;
  $('metric-goal').textContent = fmtDistance(track.goal_distance_m[i]);
  $('metric-tube').textContent = fmtDistance(track.tube_radius_m[i]);
  $('metric-occupancy').textContent = fmtRatio(track.position_occupancy[i]);
  $('metric-sliding').textContent = fmtRatio(track.sliding_occupancy[i]);
  $('metric-reserve').textContent = fmtRatio(track.input_reserve_ratio[i]);
  $('metric-phase').textContent = dataset.phases[i].replaceAll('_', ' ');
  $('node-label').textContent = `${i + 1} / ${dataset.q.length}`;
  $('elapsed-label').textContent = dataset.elapsed_s ? `${dataset.elapsed_s[i].toFixed(1)} s` : 'Not recorded';
  $('anomaly').textContent = dataset.q[i].toFixed(4);
  paintFooter();
  $('first').disabled = $('previous').disabled = i === 0;
  $('last').disabled = $('next').disabled = i === dataset.q.length - 1;
  scene?.update(i, state.selected, { paths: $('paths').checked, tubes: $('tubes').checked, grid: $('grid').checked, envelope: $('envelope').checked });
  charts.forEach(chart => chart.update(i, state.selected));
}

function showTab(name) {
  if (state.exporting) return;
  state.tab = name;
  if (name !== 'replay') stop();
  if (name !== 'videos') $('mission-video').pause();
  for (const section of ['replay', 'lifecycle', 'videos', 'evidence', 'theory']) {
    $(section).hidden = section !== name;
    $(`tab-${section}`).setAttribute('aria-selected', String(section === name));
  }
  if (name === 'replay') scene?.resize();
  if (name === 'lifecycle' && !$('lifecycle-frame').getAttribute('src') && $('lifecycle-frame').dataset.source) $('lifecycle-frame').src = $('lifecycle-frame').dataset.source;
  $('lifecycle-frame').contentWindow?.postMessage({type:'orbexa.lifecycle.visibility',active:name==='lifecycle'},location.origin);
  paintFooter();
}

async function loadScenario(key) {
  const generation = ++state.generation;
  clearMission();
  $('error').hidden = true;
  const entry = catalog.entries.find(item => item.key === key);
  $('mission-title').textContent = entry.title;
  $('mission-subtitle').textContent = 'Loading recorded numerical evidence…';
  $('outcome').textContent = 'Loading';
  try {
    const data = await readJSON(entry.dataset);
    if (generation !== state.generation) return;
    if (data.schema !== 'orbexa.mission.studio.data.v1' || !data.q.length || !data.tracks.length) throw new Error('Unsupported or empty saved replay dataset');
    dataset = data;
    lifecycleClock = null;
    currentEntry = entry;
    state.index = 0;
    state.selected = 0;
    $('mission-title').textContent = entry.title;
    $('mission-subtitle').textContent = `${data.synthetic ? 'Synthetic physics fixture' : 'Recorded adaptive tube MPC mission'} · ${data.q.length} shared state ${data.q.length === 1 ? 'sample' : 'samples'}`;
    $('outcome').textContent = outcomeLabel();
    $('outcome').classList.toggle('failed', !(entry.physical_outcome?.success ?? data.outcome.success));
    $('footer-outcome').textContent = `${data.outcome.label} · sampled model evidence`;
    $('chaser').replaceChildren(...data.tracks.map((track, number) => option(String(number), track.label)));
    $('count-label').textContent = String(data.tracks.length);
    $('scrub').max = String(data.q.length - 1);
    $('play').disabled = data.q.length < 2;
    $('scrub').disabled = $('chaser').disabled = false;
    $('export-scene').disabled = !scene || !SceneVideoExporter.supported();
    $('timeline-start').textContent = data.q[0].toFixed(2);
    $('timeline-end').textContent = data.q.at(-1).toFixed(2);
    $('speed').title = data.elapsed_s ? 'Multiplier of recorded physical elapsed time' : 'Multiplier of a normalized 30 second presentation; elapsed physical time was not recorded';
    setLink('download', entry.data_href);
    setLink('raw-evidence', entry.data_href);
    setLink('family-evidence', entry.evidence_href);
    setLink('video-download', entry.video_href);
    $('lifecycle-frame').hidden = !entry.lifecycle_href;
    $('lifecycle-unavailable').hidden = Boolean(entry.lifecycle_href);
    $('lifecycle-frame').dataset.source = entry.lifecycle_href ? `lifecycle.html?data=${encodeURIComponent(entry.lifecycle_href)}` : '';
    if (entry.lifecycle_href && state.tab === 'lifecycle') $('lifecycle-frame').src = $('lifecycle-frame').dataset.source;
    else $('lifecycle-frame').removeAttribute('src');
    $('mission-video').pause();
    if (entry.video_href) $('mission-video').src = entry.video_href;
    else { $('mission-video').removeAttribute('src'); $('mission-video').load(); }
    $('mission-video').hidden = !entry.video_href;
    $('video-unavailable').hidden = Boolean(entry.video_href);
    $('scope-text').textContent = data.scope;
    $('provenance').replaceChildren(...[['Model', data.model], ['Dynamics scope', data.dynamics_scope], ['Controller', data.controller], ['Raw data SHA-256', entry.raw_sha256], ['Outcome', data.outcome.message]].map(([title, value]) => {
      const row = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
      dt.textContent = title;
      dd.textContent = value || 'Not recorded';
      row.append(dt, dd);
      return row;
    }));
    const notes = [...data.notes, 'Actual markers are display glyphs; no chaser body volume or attitude is inferred.', 'Tubes use each chaser’s own component bounds at its recorded nominal center.', 'The three charts and geometry share one selected recorded node. Video playback is separate.'];
    $('dataset-notes').replaceChildren(...notes.map(text => { const li = document.createElement('li'); li.textContent = text; return li; }));
    scene?.load(data);
    charts.forEach(chart => chart.load(data));
    paint();
  } catch (error) {
    if (generation !== state.generation) return;
    clearMission();
    $('error').hidden = false;
    $('error').textContent = error.message;
    $('outcome').textContent = 'Saved replay unavailable';
    $('footer-outcome').textContent = 'Saved replay unavailable';
    $('footer-clock').textContent = 'No recorded node selected';
    for (const id of ['metric-goal', 'metric-tube', 'metric-occupancy', 'metric-sliding', 'metric-reserve', 'metric-phase', 'node-label', 'elapsed-label']) $(id).textContent = '—';
    for (const id of ['first', 'previous', 'next', 'last']) $(id).disabled = true;
    $('play').disabled = true;
  }
}

function clearMission() {
  dataset = currentEntry = null;
  stop();
  scene?.clear();
  $('mission-video').pause();
  $('mission-video').removeAttribute('src');
  $('mission-video').load();
  $('mission-video').hidden = true;
  $('lifecycle-frame').removeAttribute('src');
  $('lifecycle-frame').dataset.source = '';
  $('lifecycle-frame').hidden = true;
  $('lifecycle-unavailable').hidden = false;
  $('video-unavailable').hidden = false;
  for (const id of ['download', 'raw-evidence', 'family-evidence', 'video-download']) setLink(id, null);
  for (const chart of charts) { chart.container.replaceChildren(); chart.svg = null; chart.data = null; }
  for (const id of ['metric-goal', 'metric-tube', 'metric-occupancy', 'metric-sliding', 'metric-reserve', 'metric-phase', 'node-label', 'elapsed-label', 'count-label', 'anomaly']) $(id).textContent = '—';
  $('scope-text').textContent = '';
  $('provenance').replaceChildren();
  $('dataset-notes').replaceChildren();
  $('mission-subtitle').textContent = 'No recorded dataset selected';
  $('footer-outcome').textContent = 'No recorded dataset selected';
  $('footer-clock').textContent = 'No recorded node selected';
  for (const id of ['first', 'previous', 'next', 'last', 'play', 'export-scene', 'scrub', 'chaser']) $(id).disabled = true;
}

function showTheory() {
  const document = catalog.theory[Number($('theory-select').value)];
  $('theory-frame').hidden = !document;
  setLink('theory-open', document?.href);
  if (document) $('theory-frame').src = document.href;
}

$('scenario').addEventListener('change', () => loadScenario($('scenario').value));
$('chaser').addEventListener('change', () => { state.selected = Number($('chaser').value); paint(); });
$('camera').addEventListener('change', () => scene?.setCamera($('camera').value));
for (const id of ['paths', 'tubes', 'grid', 'envelope']) $(id).addEventListener('change', paint);
$('scrub').addEventListener('input', () => seek($('scrub').value));
$('play').addEventListener('click', play);
$('first').addEventListener('click', () => seek(0));
$('last').addEventListener('click', () => seek(dataset.q.length - 1));
$('previous').addEventListener('click', () => seek(state.index - 1));
$('next').addEventListener('click', () => seek(state.index + 1));
$('speed').addEventListener('change', () => { if (state.playing) { state.started = performance.now(); state.startClock = clockAt(state.index); } });
$('fullscreen').addEventListener('click', async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('viewport').requestFullscreen(); }
  catch (error) { $('scene-help').textContent = 'Full screen is unavailable in this browser.'; }
});
for (const name of ['replay', 'lifecycle', 'videos', 'evidence', 'theory']) $(`tab-${name}`).addEventListener('click', () => showTab(name));
$('theory-select').addEventListener('change', showTheory);
$('export-scene').disabled = !scene || !SceneVideoExporter.supported();
$('cancel-export').addEventListener('click', () => exportAbort?.abort());
$('export-scene').addEventListener('click', async () => {
  if (!dataset || !scene || state.exporting) return;
  stop();
  state.exporting = true;
  exportAbort = new AbortController();
  $('export-scene').disabled = true;
  $('cancel-export').hidden = false;
  $('scenario').disabled = $('scrub').disabled = $('chaser').disabled = true;
  for (const name of ['lifecycle', 'videos', 'evidence', 'theory']) $(`tab-${name}`).disabled = true;
  try {
    const output = await new SceneVideoExporter().record(scene, dataset, currentEntry.title, {
      signal: exportAbort.signal,
      onFrame: index => { state.index = index; paint(); },
      onProgress: progress => { $('export-status').textContent = `Exporting recorded scene · ${progress}%`; },
    });
    downloadVideo(output.blob, `ORBEX-A-${currentEntry.key}-scene.${output.extension}`);
    $('export-status').textContent = 'Scene video created. Download requested.';
  } catch (error) { $('export-status').textContent = error.message; }
  finally {
    state.exporting = false;
    $('export-scene').disabled = false;
    $('cancel-export').hidden = true;
    $('scenario').disabled = $('scrub').disabled = $('chaser').disabled = false;
    for (const name of ['lifecycle', 'videos', 'evidence', 'theory']) $(`tab-${name}`).disabled = false;
    paint();
  }
});
document.addEventListener('keydown', event => {
  if (state.tab !== 'replay' || ['INPUT', 'SELECT', 'BUTTON', 'TEXTAREA'].includes(event.target.tagName)) return;
  if (event.code === 'Space') { event.preventDefault(); play(); }
  if (event.code === 'ArrowLeft') { event.preventDefault(); seek(state.index - 1); }
  if (event.code === 'ArrowRight') { event.preventDefault(); seek(state.index + 1); }
});

function tick(now) {
  if (state.playing && dataset) {
    const multiplier = dataset.elapsed_s ? Number($('speed').value) : Number($('speed').value) / 40;
    const targetClock = state.startClock + (now - state.started) / 1000 * multiplier;
    let index = state.index;
    while (index < dataset.q.length - 1 && clockAt(index + 1) <= targetClock) index++;
    if (index !== state.index) { state.index = index; paint(); }
    if (targetClock >= clockAt(dataset.q.length - 1)) stop();
  }
  if (state.tab === 'replay') scene?.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

try {
  catalog = await readJSON('catalog.json');
  if (catalog.schema !== 'orbexa.mission.studio.catalog.v1' || !catalog.entries.length) throw new Error('No recorded mission catalog is available');
  $('scenario').replaceChildren(...catalog.entries.map(entry => option(entry.key, entry.title)));
  $('theory-select').replaceChildren(...catalog.theory.map((document, index) => option(String(index), document.title)));
  showTheory();
  const requested = new URLSearchParams(location.search).get('mission');
  const key = catalog.entries.some(entry => entry.key === requested) ? requested : catalog.entries[0].key;
  $('scenario').value = key;
  await loadScenario(key);
} catch (error) {
  $('error').hidden = false;
  $('error').textContent = error.message;
}
