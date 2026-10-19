import { MissionScene } from './scene.js';
import { MetricChart } from './charts.js';
import { SceneVideoExporter, downloadVideo, sha256Bytes } from './recorder.js';
import { renderTheoryFigure } from './theory-figures.js';
import { missionTimeline, sampleAtTime, phaseTitle, releaseContextVisible } from './mission-timeline.js';
import { TubeDetail } from './tube-detail.js';
import { simulationControls, recordedControls } from './live-client.js';
import { LifecycleScene } from './lifecycle-scene.js';

const $ = id => document.getElementById(id);
const state = { index: 0, selected: 0, playing: false, tab: 'mission', generation: 0 };
const fmtDistance = value => value === null || !Number.isFinite(value) ? 'Not recorded' : value < 0.01 ? `${(value * 1000).toFixed(2)} mm` : `${value.toFixed(3)} m`;
const fmtRatio = value => value === null || !Number.isFinite(value) ? 'Not recorded' : value.toFixed(3);
let catalog, dataset, currentEntry, scene, timeline, lifecycleData, lifecycleScene;
let exportAbort, recordingDataUrl;
let lifecycleReady = false, waitingForLifecycle = false;
let liveJobId = null, followLive = false, liveSnapshot, renderedLiveSignature;
const timeLabel = seconds => seconds < 60 ? `${seconds.toFixed(1)} s` : `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(1).padStart(4, '0')}`;
const currentSample = () => timeline?.samples[state.index];
const approachIndex = () => currentSample()?.kind === 'approach' ? currentSample().index : dataset.q.length - 1;
const hasRelease = () => { const value = dataset?.deployment?.carrier_center_lvlh_m; return Array.isArray(value) && value.length === 3 && value.every(Number.isFinite); };

function outcomeLabel() {
  if (dataset?.synthetic) return dataset.outcome.label;
  const physical = currentEntry?.physical_outcome;
  const terminalLive = liveJobId && liveSnapshot?.job_id === liveJobId && ['completed','stopped','failed'].includes(liveSnapshot.status);
  if (terminalLive && (liveSnapshot.success !== true || physical?.success !== true)) return 'Mission unfinished';
  if (!physical) return dataset?.outcome.message === 'max_mpc_updates_reached_before_terminal_tube_dwell' ? 'Approach unfinished' : dataset?.outcome.label ?? '';
  return physical.success ? `${physical.attached_chaser_count} attached · mission complete` : 'Mission unfinished';
}

window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.source !== $('lifecycle-frame').contentWindow) return;
  if (!dataset || event.data?.session !== String(state.generation)) return;
  if (event.data?.type === 'orbexa.lifecycle.request' && lifecycleData) {
    $('lifecycle-frame').contentWindow.postMessage({ type: 'orbexa.lifecycle.load', session: String(state.generation), data: lifecycleData }, location.origin);
  }
  if (event.data?.type === 'orbexa.video.export') {
    try { downloadVideo(event.data.blob, event.data.filename); }
    catch (error) { $('error').hidden = false; $('error').textContent = error.message; }
  }
  if (event.data?.type === 'orbexa.lifecycle.ready') {
    if (!lifecycleData || event.data.count !== lifecycleData.time_s.length || event.data.firstTime !== lifecycleData.time_s[0] || event.data.lastTime !== lifecycleData.time_s.at(-1)) return;
    lifecycleReady = true;
    syncLifecycle();
    paint();
    if (waitingForLifecycle) { waitingForLifecycle = false; play(); }
  }
  if (event.data?.type === 'orbexa.lifecycle.error') {
    stop();
    $('error').hidden = false;
    $('error').textContent = `Attached-flight view unavailable: ${event.data.message}`;
  }
});

function syncLifecycle() {
  if (lifecycleScene) return;
  const sample = currentSample(), active = state.tab === 'mission' && sample?.kind === 'lifecycle';
  const frame = $('lifecycle-frame');
  if (!frame.getAttribute('src')) return;
  frame.contentWindow?.postMessage({ type: 'orbexa.lifecycle.visibility', session: String(state.generation), active }, location.origin);
  if (lifecycleReady && sample?.kind === 'lifecycle') frame.contentWindow?.postMessage({ type: 'orbexa.lifecycle.seek', session: String(state.generation), index: sample.index }, location.origin);
}

function paintFooter() {
  if (!dataset || !timeline) return;
  const sample = currentSample();
  $('footer-outcome').textContent = dataset.synthetic ? 'Synthetic physics fixture' : 'Recorded simulation';
  $('footer-clock').textContent = `${phaseTitle(sample.phase, hasRelease())} · t = ${timeLabel(sample.time)} · sample ${state.index + 1} / ${timeline.samples.length}`;
}

try {
  scene = new MissionScene($('viewport'), selected => { state.selected = selected; $('chaser').value = String(selected); paint(); });
} catch (error) {
  $('scene-error').hidden = false;
  $('scene-error').textContent = '3D rendering is unavailable in this browser. Measurements, videos and evidence remain available. ' + error.message;
}

const charts = [
  new MetricChart($('goal-chart'), 'Goal distance', 'm', 'goal_distance_m', seekApproach, { logarithmic: true }),
  new MetricChart($('occupancy-chart'), 'Position tube occupancy', 'ratio', 'position_occupancy', seekApproach, { threshold: 1 }),
  new MetricChart($('reserve-chart'), 'Ancillary input reserve', 'ratio · preceding interval', 'input_reserve_ratio', seekApproach, { threshold: 1 }),
];
const tubeDetail = new TubeDetail($('tube-detail'));

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
  waitingForLifecycle = false;
  $('play').setAttribute('aria-label', 'Play recorded mission');
  $('play').innerHTML = '<svg viewBox="0 0 24 24"><path d="m8 4 12 8-12 8Z"/></svg>';
}

function clockAt(index) { return timeline.samples[index].time; }

function seek(index) {
  if (!timeline || state.exporting || !Number.isFinite(Number(index))) return;
  stop();
  followLive = false;
  state.index = Math.max(0, Math.min(timeline.samples.length - 1, Math.trunc(Number(index))));
  paint();
}

function seekApproach(index) { seek(index); }

function play() {
  if (!timeline || timeline.samples.length < 2 || state.exporting) return;
  if (state.playing || waitingForLifecycle) { stop(); return; }
  followLive = false;
  if (state.index === timeline.samples.length - 1) { state.index = 0; paint(); }
  if (currentSample().kind === 'lifecycle' && !lifecycleReady) {
    waitingForLifecycle = true;
    $('play').setAttribute('aria-label', 'Cancel waiting for attached-flight view');
    return;
  }
  state.playing = true;
  state.started = performance.now();
  state.startClock = clockAt(state.index);
  $('play').setAttribute('aria-label', 'Pause recorded mission');
  $('play').innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></svg>';
}

function paint() {
  if (!dataset || !timeline) return;
  const sample = currentSample(), i = approachIndex();
  const physical = sample.kind !== 'approach' && Boolean(lifecycleScene);
  const track = dataset.tracks[state.selected];
  const completedRecord = currentEntry.physical_outcome?.success ?? (dataset.outcome.complete && dataset.outcome.success);
  const activeLive = liveJobId && liveSnapshot?.job_id === liveJobId && ['starting','running','pausing','paused','completing'].includes(liveSnapshot.status);
  $('outcome').textContent = activeLive || completedRecord && state.index < timeline.samples.length - 1 ? phaseTitle(sample.phase, hasRelease()) : outcomeLabel();
  $('scrub').value = String(sample.time);
  $('scrub').setAttribute('aria-valuetext', `${phaseTitle(sample.phase, hasRelease())}, ${timeLabel(sample.time)}, recorded sample ${state.index + 1} of ${timeline.samples.length}`);
  $('chaser-title').textContent = track.label;
  $('metric-goal').textContent = fmtDistance(track.goal_distance_m[i]);
  $('metric-tube').textContent = fmtDistance(track.tube_radius_m[i]);
  $('metric-occupancy').textContent = fmtRatio(track.position_occupancy[i]);
  $('metric-sliding').textContent = fmtRatio(track.sliding_occupancy[i]);
  $('metric-reserve').textContent = fmtRatio(track.input_reserve_ratio[i]);
  $('metric-phase').textContent = phaseTitle(dataset.phases[i], hasRelease());
  $('node-label').textContent = `${i + 1} / ${dataset.q.length}`;
  $('elapsed-label').textContent = dataset.elapsed_s ? `${dataset.elapsed_s[i].toFixed(1)} s` : 'Not recorded';
  const labels = ['Goal distance', 'Tube radius', 'Position occupancy', 'Sliding occupancy', 'Input reserve ratio'];
  const ids = ['metric-goal', 'metric-tube', 'metric-occupancy', 'metric-sliding', 'metric-reserve'];
  lifecycleScene?.setVisible(physical);
  const options = { paths: $('paths').checked, tubes: $('tubes').checked, grid: $('grid').checked, envelope: $('envelope').checked };
  if (physical) {
    const pose = lifecycleScene.update(sample, state.selected, options);
    const body = lifecycleData;
    let values;
    if (sample.kind === 'contact') {
      labels.splice(0, labels.length, 'Pad gap', 'Tip extension', 'Bond activation', 'Pad force', 'Pad moment');
      const pad = pose.pads[track.id];
      values = [fmtDistance(pad.gap), fmtDistance(pad.extension), pad.activation.toFixed(3), `${pad.force.toFixed(3)} N`, `${pad.moment.toFixed(3)} N m`];
    } else {
      labels.splice(0, labels.length, 'Angular speed', 'Assembly propellant', 'Altitude', 'Osculating perigee', 'Applied force');
      values = [`${body.angular_speed_rad_s[sample.index].toExponential(3)} rad/s`, `${body.fuel_kg[sample.index].toFixed(3)} kg`,
        `${(body.altitude_m[sample.index] / 1000).toFixed(2)} km`, `${(body.perigee_altitude_m[sample.index] / 1000).toFixed(2)} km`,
        `${Math.hypot(...body.applied_force_body_n[sample.index]).toFixed(3)} N`];
    }
    ids.forEach((id, number) => { $(id).textContent = values[number]; });
    $('metric-phase').textContent = phaseTitle(sample.phase);
    $('node-label').textContent = `${sample.kind === 'contact' ? 'Contact' : 'Assembly'} ${sample.index + 1} / ${sample.kind === 'contact' ? body.finite_contact.time_s.length : body.time_s.length}`;
    $('elapsed-label').textContent = `${sample.time.toFixed(3)} s`;
  } else scene?.update(i, state.selected, options);
  ids.forEach((id, number) => { $(id).previousElementSibling.textContent = labels[number]; });
  const frameNote = document.querySelector('.frame-note');
  if (frameNote) { frameNote.style.whiteSpace = 'pre-line'; frameNote.textContent = physical ? 'Capture frame · target-relative meters\nFixed axes from the recorded capture epoch' :
    (dataset.state_frame === 'reference-orbit-frame-v1' ? 'Reference orbit frame · meters' : 'Physical LVLH · meters') + '\nX −transverse · Y radial · Z orbit normal'; }
  document.querySelectorAll('.viewport-key > span').forEach((item, number) => {
    item.hidden = physical && number === 1;
    item.lastChild.textContent = physical ? number === 2 ? 'Pad tip' : 'Actual body poses' : ['Actual', 'Nominal', 'COM goal'][number];
  });
  for (const id of ['tubes', 'grid']) $(id).disabled = physical;
  $('tubes').title = physical ? 'Free-flight tube samples end at the approach handoff' : '';
  $('grid').title = physical ? 'Capture-fixed axes are identified in the frame label' : '';
  $('camera').querySelector('option[value="overview"]').textContent = physical ? 'Capture overview' : 'Full route';
  $('anomaly').textContent = timeLabel(sample.time);
  $('clock-label').textContent = timeline.physical ? 'Mission time' : 'Presentation time';
  $('deployment-note').hidden = !hasRelease() || sample.kind !== 'approach' || !releaseContextVisible(dataset, i);
  paintFooter();
  $('first').disabled = $('previous').disabled = state.index === 0;
  $('last').disabled = $('next').disabled = state.index === timeline.samples.length - 1;
  $('mission-phases').querySelectorAll('button').forEach(button => button.setAttribute('aria-current', String(Number(button.dataset.index) <= state.index && Number(button.dataset.end) > state.index)));
  charts.forEach(chart => chart.update(i, state.selected));
  tubeDetail.update(track, i, $('tubes').checked && sample.kind === 'approach');
  showMissionPanel();
  syncLifecycle();
}

function showMissionPanel() {
  const inMission = state.tab === 'mission', sample = currentSample();
  const empty = !sample, attached = Boolean(sample) && sample.kind !== 'approach', shared = Boolean(lifecycleScene);
  const enteringApproach = $('replay').hidden && inMission && !attached;
  $('replay').hidden = !inMission || attached && !shared;
  $('lifecycle').hidden = !inMission || !attached || shared;
  $('replay').classList.toggle('empty', empty);
  $('mission-empty').hidden = !empty || !scene;
  $('mission-transport').hidden = $('mission-view-control').hidden = !inMission || empty;
  $('lifecycle-frame').hidden = !lifecycleReady;
  $('lifecycle-unavailable').hidden = lifecycleReady || !attached;
  if (attached && !lifecycleReady) $('lifecycle-unavailable').textContent = 'Loading the recorded attached-flight view…';
  if (enteringApproach) scene?.resize();
}

function showTab(name) {
  if (state.exporting) return;
  state.tab = name;
  if (name !== 'mission') stop();
  if (name !== 'plots') $('mission-video').pause();
  for (const section of ['plots', 'theory']) $(section).hidden = section !== name;
  for (const section of ['mission','plots','theory']) $(`tab-${section}`).setAttribute('aria-selected',String(section===name));
  showMissionPanel();
  syncLifecycle();
  paintFooter();
}

function paintPhases() {
  const existing = [...$('mission-phases').querySelectorAll('button')];
  if (existing.length === timeline.phases.length && existing.every((button, index) => button.dataset.phase === timeline.phases[index].phase && Number(button.dataset.index) === timeline.phases[index].index)) {
    existing.forEach((button, index) => { button.dataset.end = String(timeline.phases[index + 1]?.index ?? timeline.samples.length); });
    return;
  }
  $('mission-phases').replaceChildren(...timeline.phases.map((phase, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = phaseTitle(phase.phase, hasRelease());
    button.title = `Jump to ${timeLabel(phase.time)}`;
    button.dataset.index = String(phase.index);
    button.dataset.phase = phase.phase;
    button.dataset.end = String(timeline.phases[index + 1]?.index ?? timeline.samples.length);
    button.addEventListener('click', () => { seek(phase.index); showTab('mission'); });
    return button;
  }));
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
    const attachedData = entry.lifecycle_href ? await readJSON(entry.lifecycle_href) : null;
    if (generation !== state.generation) return;
    if (attachedData && attachedData.schema !== 'orbexa.lifecycle.studio.v1') throw new Error('Unsupported saved attached-flight record');
    const mission = missionTimeline(data, attachedData);
    dataset = data;
    const frameNote=document.querySelector('.frame-note');
    if(frameNote)frameNote.textContent=(data.state_frame==='reference-orbit-frame-v1'?
      'Reference orbit frame · meters':'Physical LVLH · meters')+'\nX −transverse · Y radial · Z orbit normal';
    if(frameNote)frameNote.style.whiteSpace='pre-line';
    timeline = mission;
    lifecycleData = attachedData;
    currentEntry = entry;
    state.index = 0;
    state.selected = 0;
    $('mission-title').textContent = entry.title;
    $('mission-subtitle').textContent = data.synthetic ? 'Synthetic physics fixture' : `${data.tracks.length} chasers · adaptive tube MPC`;
    $('mission-model').textContent = data.target.shape === 'composite_spacecraft' ? 'Declared component model' : 'Cylinder baseline';
    $('outcome').textContent = outcomeLabel();
    $('outcome').classList.toggle('failed', entry.physical_outcome?.success === false || data.outcome.complete && !data.outcome.success);
    $('footer-outcome').textContent = `${data.outcome.label} · sampled model evidence`;
    $('chaser').replaceChildren(...data.tracks.map((track, number) => option(String(number), track.label)));
    $('count-label').textContent = String(data.tracks.length);
    $('scrub').min = String(timeline.start);
    $('scrub').max = String(timeline.end);
    $('play').disabled = timeline.samples.length < 2;
    $('scrub').disabled = $('chaser').disabled = false;
    $('export-scene').disabled = !scene || (attachedData?.finite_contact ? mission.samples.length : data.q.length)<2 || !SceneVideoExporter.supported();
    $('timeline-start').textContent = timeLabel(timeline.start);
    $('timeline-end').textContent = timeLabel(timeline.end);
    paintPhases();
    $('speed').title = data.elapsed_s ? 'Multiplier of recorded physical elapsed time' : 'Multiplier of a normalized 30 second presentation; elapsed physical time was not recorded';
    setLink('download', entry.data_href);
    $('download').textContent = entry.data_kind === 'projection' ? 'Projection JSON' : 'Data';
    setLink('raw-evidence', entry.raw_href ?? (entry.data_kind === 'projection' ? null : entry.data_href));
    setLink('family-evidence', entry.evidence_href);
    setLink('video-download', entry.video_href);
    $('plot-gallery').replaceChildren(...(entry.figures || []).map(figure=>{
      const link=document.createElement('a'),image=document.createElement('img');
      link.href=figure.href; image.src=figure.href;image.alt=figure.title;image.loading='lazy';
      link.append(image);return link;
    }));
    $('lifecycle-frame').dataset.source = entry.lifecycle_href ? `lifecycle.html?v=${encodeURIComponent(catalog.asset_version || '')}&data=${encodeURIComponent(entry.lifecycle_href)}&controlled=1&session=${generation}` : '';
    if (entry.lifecycle_href && !attachedData?.finite_contact) $('lifecycle-frame').src = $('lifecycle-frame').dataset.source;
    else $('lifecycle-frame').removeAttribute('src');
    $('mission-video').pause();
    if (entry.video_href) $('mission-video').src = entry.video_href;
    else { $('mission-video').removeAttribute('src'); $('mission-video').load(); }
    $('mission-video').hidden = !entry.video_href;
    $('videos').hidden = !entry.video_href;
    $('video-unavailable').hidden = Boolean(entry.video_href);
    $('scope-text').textContent = data.scope;
    $('provenance').replaceChildren(...[['Model', data.model], ['Dynamics scope', data.dynamics_scope], ['Controller', data.controller], ['Raw data SHA-256', entry.raw_sha256], ['Outcome', data.outcome.message]].map(([title, value]) => {
      const row = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
      dt.textContent = title;
      dd.textContent = value || 'Not recorded';
      row.append(dt, dd);
      return row;
    }));
    const notes = [...data.notes, 'One mission clock selects saved samples; no intermediate physical states are invented. Paths include later samples. Stars are decorative.', 'Hull dots mark surface points; orange markers mark COM goals.', 'Tubes belong to each recorded nominal center. Occupancy 1 is the boundary; reserve covers the preceding interval.'];
    if (hasRelease()) notes.push('The shared dispenser marks the declared release location. Its shape and placement behind the payload are illustrative; ascent, carrier motion and ejection are outside this replay.');
    if (attachedData?.finite_contact) notes.push('Finite contact shows actual saved SI target, bus and pad-tip poses. Capture-fixed display axes remain unchanged through rigid attachment; physical states are not interpolated.');
    else if (attachedData && attachedData.time_s[0] > data.elapsed_s.at(-1)) notes.push('The interval between the final approach sample and the first attachment sample is retained on the mission clock. The browser holds the last recorded sample until the next recorded state.');
    $('dataset-notes').replaceChildren(...notes.map(text => { const li = document.createElement('li'); li.textContent = text; return li; }));
    scene?.load(data);
    if (scene && attachedData?.finite_contact) { lifecycleScene = new LifecycleScene(scene, attachedData, data); lifecycleReady = true; }
    $('export-scene').textContent = lifecycleScene ? 'Export mission video' : 'Export approach video';
    $('camera').value = 'mission';
    scene?.setCamera('mission');
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
  if (recordingDataUrl) { URL.revokeObjectURL(recordingDataUrl); recordingDataUrl = null; }
  $('export-status').textContent = '';
  lifecycleScene?.dispose(); lifecycleScene = null;
  dataset = currentEntry = timeline = lifecycleData = null;
  lifecycleReady = false;
  stop();
  scene?.clear();
  $('mission-video').pause();
  $('mission-video').removeAttribute('src');
  $('mission-video').load();
  $('mission-video').hidden = true;
  $('videos').hidden = true;
  $('lifecycle-frame').removeAttribute('src');
  $('lifecycle-frame').dataset.source = '';
  $('lifecycle-frame').hidden = true;
  $('mission-phases').replaceChildren();
  $('tube-detail').hidden = true;
  $('lifecycle-unavailable').hidden = false;
  $('video-unavailable').hidden = false;
  for (const id of ['download', 'raw-evidence', 'family-evidence', 'video-download']) setLink(id, null);
  for (const chart of charts) { chart.container.replaceChildren(); chart.svg = null; chart.data = null; }
  for (const id of ['metric-goal', 'metric-tube', 'metric-occupancy', 'metric-sliding', 'metric-reserve', 'metric-phase', 'node-label', 'elapsed-label', 'count-label', 'anomaly']) $(id).textContent = '—';
  $('scope-text').textContent = '';
  $('provenance').replaceChildren();
  $('dataset-notes').replaceChildren();
  $('plot-gallery').replaceChildren();
  $('mission-subtitle').textContent = 'No recorded dataset selected';
  $('footer-outcome').textContent = 'No recorded dataset selected';
  $('footer-clock').textContent = 'No recorded node selected';
  for (const id of ['first', 'previous', 'next', 'last', 'play', 'export-scene', 'scrub', 'chaser']) $(id).disabled = true;
}

function resetLiveView() {
  exportAbort?.abort();
  ++state.generation;
  clearMission();
  liveJobId = renderedLiveSignature = null;
  liveSnapshot = null;
  followLive = false;
  state.index = state.selected = 0;
  $('mission-title').textContent = 'Envisat';
  $('mission-subtitle').textContent = '';
  $('outcome').textContent = 'Ready';
  $('outcome').classList.remove('failed');
  $('error').hidden = true;
  showTab('mission');
}

function applyLiveSnapshot(snapshot) {
  liveSnapshot = snapshot;
  const data = snapshot.projection, attached = snapshot.lifecycle;
  if (state.exporting) return;
  if (!data) {
    if (snapshot.job_id && snapshot.job_id !== liveJobId) {
      ++state.generation; clearMission();
      liveJobId = snapshot.job_id; renderedLiveSignature = null;
      state.index = state.selected = 0;
      $('mission-title').textContent = 'Envisat'; $('outcome').textContent = 'Starting…';
    }
    return;
  }
  $('outcome').classList.toggle('failed', ['failed','stopped'].includes(snapshot.status));
  if (data.schema !== 'orbexa.mission.studio.data.v1' || !data.q?.length || !data.tracks?.length ||
      attached && attached.schema !== 'orbexa.lifecycle.studio.v1') throw new Error('Unsupported local simulator projection');
  const signature = JSON.stringify([snapshot.job_id, data.q.length, data.q.at(-1), data.outcome, attached?.time_s.length, attached?.time_s.at(-1), attached?.finite_contact?.source_history_sha256]);
  if (signature === renderedLiveSignature) {
    if (['completed','failed','stopped'].includes(snapshot.status)) $('outcome').textContent = outcomeLabel();
    const outcomeRow = $('provenance').children[2];
    if (outcomeRow?.lastElementChild) outcomeRow.lastElementChild.textContent = snapshot.message || data.outcome.message || 'Not recorded';
    return;
  }
  const mission = missionTimeline(data, attached);
  const newJob = liveJobId !== snapshot.job_id;
  const previousTime = currentSample()?.time;
  const selectedId = dataset?.tracks[state.selected]?.id;
  const addingLifecycle = !newJob && !lifecycleData && Boolean(attached);
  const continueJourney = addingLifecycle && followLive;
  const camera = !newJob && scene?.camera ? { position: scene.camera.position.clone(), target: scene.controls.target.clone(), mode: scene.mode, key: scene.frameKey } : null;
  if (newJob) { ++state.generation; clearMission(); }
  liveJobId = snapshot.job_id;
  renderedLiveSignature = signature;
  dataset = data;
  const frameNote=document.querySelector('.frame-note');
  if(frameNote){frameNote.style.whiteSpace='pre-line';frameNote.textContent=(data.state_frame==='reference-orbit-frame-v1'?
    'Reference orbit frame · meters':'Physical LVLH · meters')+'\nX −transverse · Y radial · Z orbit normal';}
  timeline = mission;
  const changedLifecycle = Boolean(attached) && (!lifecycleData || lifecycleData.time_s.length !== attached.time_s.length || lifecycleData.time_s.at(-1) !== attached.time_s.at(-1) || lifecycleData.finite_contact?.source_history_sha256 !== attached.finite_contact?.source_history_sha256);
  lifecycleData = attached;
  currentEntry = { key: 'envisat-live', title: 'Envisat', physical_outcome: attached ? { ...attached.outcome,
    success: snapshot.success === true && attached.outcome?.success === true, attached_chaser_count: Object.keys(attached.surface_contacts || {}).length } : null };
  state.selected = Math.max(0, data.tracks.findIndex(track => track.id === selectedId));
  state.index = newJob ? 0 : followLive && !addingLifecycle ? timeline.samples.length - 1 : sampleAtTime(timeline.samples, previousTime ?? timeline.start);
  if (newJob && followLive) state.index = timeline.samples.length - 1;
  if (continueJourney) followLive = false;
  $('mission-title').textContent = 'Envisat';
  $('mission-subtitle').textContent = `${data.tracks.length} chasers · ${data.deployment?.carrier_center_lvlh_m ? Math.hypot(...data.deployment.carrier_center_lvlh_m).toFixed(0) + ' m release' : 'recorded flight'}`;
  $('outcome').classList.toggle('failed', ['failed','stopped'].includes(snapshot.status));
  $('error').hidden = true;
  $('chaser').replaceChildren(...data.tracks.map((track, number) => option(String(number), track.label)));
  $('chaser').value = String(state.selected);
  $('count-label').textContent = String(data.tracks.length);
  $('scrub').min = String(timeline.start); $('scrub').max = String(timeline.end);
  $('timeline-start').textContent = timeLabel(timeline.start); $('timeline-end').textContent = timeLabel(timeline.end);
  $('play').disabled = timeline.samples.length < 2;
  $('scrub').disabled = $('chaser').disabled = false;
  $('export-scene').disabled = !scene || (attached?.finite_contact ? mission.samples.length : data.q.length) < 2 || !SceneVideoExporter.supported();
  $('scope-text').textContent = data.scope || snapshot.scope || '';
  $('mission-model').textContent = '';
  $('provenance').replaceChildren(...[['Model', data.model], ['Controller', data.controller], ['Outcome', snapshot.message || data.outcome.message], ['Source', 'Local authoritative simulator']].map(([label, value]) => {
    const row = document.createElement('div'), title = document.createElement('dt'), description = document.createElement('dd');
    title.textContent = label; description.textContent = value || 'Not recorded'; row.append(title, description); return row;
  }));
  paintPhases();
  lifecycleScene?.dispose(); lifecycleScene = null;
  scene?.load(data);
  if (camera) {
    scene.camera.position.copy(camera.position); scene.controls.target.copy(camera.target); scene.mode = camera.mode; scene.frameKey = camera.key;
  } else { $('camera').value = 'mission'; scene?.setCamera('mission'); }
  if (scene && attached?.finite_contact) { lifecycleScene = new LifecycleScene(scene, attached, data); lifecycleReady = true; }
  $('export-scene').textContent = lifecycleScene ? 'Export mission video' : 'Export approach video';
  charts.forEach(chart => chart.load(data));
  if (changedLifecycle) {
    if (!newJob) ++state.generation;
    lifecycleReady = Boolean(lifecycleScene);
    if (lifecycleScene) $('lifecycle-frame').removeAttribute('src');
    else $('lifecycle-frame').src = `lifecycle.html?v=${encodeURIComponent(catalog.asset_version || '')}&source=parent&controlled=1&session=${state.generation}`;
  }
  paint();
  if (continueJourney) play();
}

function showTheory() {
  const document = catalog.theory[Number($('theory-select').value)];
  $('theory-frame').hidden = !document;
  setLink('theory-open', document?.href);
  if (document) $('theory-frame').src = document.href;
  renderTheoryFigure($('theory-diagram'),document?.title || '');
}

$('scenario').addEventListener('change', () => loadScenario($('scenario').value));
$('chaser').addEventListener('change', () => { state.selected = Number($('chaser').value); paint(); });
$('camera').addEventListener('change', () => scene?.setCamera($('camera').value));
for (const id of ['paths', 'tubes', 'grid', 'envelope']) $(id).addEventListener('change', paint);
$('scrub').addEventListener('input', () => { if (timeline) seek(sampleAtTime(timeline.samples, Number($('scrub').value))); });
$('play').addEventListener('click', play);
$('first').addEventListener('click', () => seek(0));
$('last').addEventListener('click', () => { if (timeline) seek(timeline.samples.length - 1); });
$('previous').addEventListener('click', () => seek(state.index - 1));
$('next').addEventListener('click', () => seek(state.index + 1));
$('speed').addEventListener('change', () => { if (state.playing) { state.started = performance.now(); state.startClock = clockAt(state.index); } });
$('fullscreen').addEventListener('click', async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('viewport').requestFullscreen(); }
  catch (error) { $('scene-help').textContent = 'Full screen is unavailable in this browser.'; }
});
for (const name of ['mission', 'plots', 'theory']) $(`tab-${name}`).addEventListener('click', () => showTab(name));
$('theory-select').addEventListener('change', showTheory);
$('export-scene').disabled = !scene || !SceneVideoExporter.supported();
$('cancel-export').addEventListener('click', () => exportAbort?.abort());
$('export-scene').addEventListener('click', async () => {
  if (!dataset || (lifecycleScene ? timeline.samples.length : dataset.q.length)<2 || !scene || state.exporting) return;
  stop();
  state.exporting = true;
  exportAbort = new AbortController();
  $('export-scene').disabled = true;
  $('cancel-export').hidden = false;
  $('scenario').disabled = $('scrub').disabled = $('chaser').disabled = true;
  for (const name of ['mission', 'plots', 'theory']) $(`tab-${name}`).disabled = true;
  $('mission-phases').querySelectorAll('button').forEach(button => { button.disabled = true; });
  try {
    if (recordingDataUrl) { URL.revokeObjectURL(recordingDataUrl); recordingDataUrl = null; }
    $('export-status').textContent = 'Preparing recording…';
    const catalogResponse = await fetch('catalog.json', { cache: 'no-cache', signal: exportAbort.signal });
    if (!catalogResponse.ok) throw new Error('Cannot bind the recording to its saved catalog.');
    const catalogBytes = await catalogResponse.arrayBuffer();
    if (JSON.stringify(JSON.parse(new TextDecoder().decode(catalogBytes))) !== JSON.stringify(catalog)) {
      throw new Error('The mission catalog changed. Reload it before exporting.');
    }
    const sourceBinding = { entry_key: currentEntry.key, catalog_sha256: await sha256Bytes(catalogBytes),
      asset_version: catalog.asset_version, approach_raw_sha256: currentEntry.raw_sha256 ?? null,
      dataset_sha256: await sha256Bytes(JSON.stringify(dataset)), timeline_sha256: await sha256Bytes(JSON.stringify(timeline.samples)),
      lifecycle_projection_sha256: lifecycleData ? await sha256Bytes(JSON.stringify(lifecycleData)) : null,
      approach_canonical_sha256: lifecycleData?.approach_canonical_sha256 ?? null,
      lifecycle_canonical_sha256: lifecycleData?.lifecycle_canonical_sha256 ?? null,
      finite_contact_history_sha256: lifecycleData?.finite_contact?.source_history_sha256 ?? null };
    const exportData = lifecycleScene ? { elapsed_s: timeline.samples.map(sample => sample.time), phases: timeline.samples.map(sample => sample.phase) } : dataset;
    const output = await new SceneVideoExporter().record(scene, exportData, currentEntry.title, {
      signal: exportAbort.signal, sourceBinding,
      onFrame: index => {
        state.index = index; paint();
        const sample = lifecycleScene ? timeline.samples[index] : { kind: 'approach', index, phase: dataset.phases[index] };
        return { ...sample, selected_chaser: dataset.tracks[state.selected].id, camera_view: $('camera').value };
      },
      onProgress: progress => { $('export-status').textContent = `Exporting recorded scene · ${progress}%`; },
    });
    downloadVideo(output.blob, `ORBEX-A-${currentEntry.key}-scene.${output.extension}`);
    $('export-status').textContent = 'Scene video created. Download requested.';
    recordingDataUrl = URL.createObjectURL(new Blob([JSON.stringify(output.provenance, null, 2)+'\n'], { type: 'application/json' }));
    const recordingData = document.createElement('a');
    recordingData.href = recordingDataUrl; recordingData.download = `ORBEX-A-${currentEntry.key}-recording.json`;
    recordingData.textContent = 'Recording data'; $('export-status').append(' ', recordingData);
  } catch (error) { $('export-status').textContent = error.message; }
  finally {
    state.exporting = false;
    $('export-scene').disabled = false;
    $('cancel-export').hidden = true;
    $('scenario').disabled = $('scrub').disabled = $('chaser').disabled = false;
    for (const name of ['mission', 'plots', 'theory']) $(`tab-${name}`).disabled = false;
    $('mission-phases').querySelectorAll('button').forEach(button => { button.disabled = false; });
    paint();
    if (liveSnapshot) applyLiveSnapshot(liveSnapshot);
  }
});
document.addEventListener('keydown', event => {
  if (state.tab !== 'mission' || ['INPUT', 'SELECT', 'BUTTON', 'TEXTAREA'].includes(event.target.tagName)) return;
  if (event.code === 'Space') { event.preventDefault(); play(); }
  if (event.code === 'ArrowLeft') { event.preventDefault(); seek(state.index - 1); }
  if (event.code === 'ArrowRight') { event.preventDefault(); seek(state.index + 1); }
});

function tick(now) {
  if (state.playing && dataset) {
    const multiplier = timeline.physical ? Number($('speed').value) : Number($('speed').value) / 30;
    const targetClock = state.startClock + (now - state.started) / 1000 * multiplier;
    let index = state.index;
    while (index < timeline.samples.length - 1 && clockAt(index + 1) <= targetClock) index++;
    if (timeline.samples[index].kind === 'lifecycle' && !lifecycleReady) {
      stop();
      waitingForLifecycle = true;
      index = timeline.approachCount - 1;
    }
    if (index !== state.index) { state.index = index; paint(); }
    if (index === timeline.samples.length - 1) {
      stop();
      if ($('loop').checked) { state.index = 0; paint(); play(); }
    }
  }
  if (state.tab === 'mission' && (currentSample()?.kind === 'approach' || lifecycleScene)) scene?.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

try {
  catalog = await readJSON('catalog.json');
  if (catalog.schema !== 'orbexa.mission.studio.catalog.v1' || !Array.isArray(catalog.entries) || !catalog.entries.length && catalog.live !== true) throw new Error('No recorded mission catalog is available');
  $('scenario').replaceChildren(...catalog.entries.map(entry => option(entry.key, entry.title)));
  $('scenario').hidden = catalog.entries.length < 2;
  $('theory-select').replaceChildren(...catalog.theory.map((document, index) => option(String(index), document.title)));
  showTheory();
  if (!catalog.entries.length) resetLiveView();
  else {
    const requested = new URLSearchParams(location.search).get('mission');
    const key = catalog.entries.some(entry => entry.key === requested) ? requested : catalog.entries[0].key;
    $('scenario').value = key;
    await loadScenario(key);
  }
} catch (error) {
  $('error').hidden = false;
  $('error').textContent = error.message;
}

if (catalog?.live === true) simulationControls({
  onState: applyLiveSnapshot,
  onReset: resetLiveView,
  onFollow() { followLive = true; stop(); if (timeline) { state.index = timeline.samples.length - 1; paint(); } },
  onError(message) { $('error').hidden = false; $('error').textContent = message; },
});
else if (catalog) {
  try { recordedControls({ liveUiHref: catalog.live_ui_href }); }
  catch (error) { $('error').hidden = false; $('error').textContent = error.message; }
}
