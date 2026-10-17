/** Capture the numerical WebGL view into a local, captioned video. No upload. */
export function downloadVideo(blob, filename) {
  if (!(blob instanceof Blob) || !/^ORBEX-A-[A-Za-z0-9_-]+\.(webm|mp4)$/.test(filename)) throw new Error('Invalid local video export.');
  if (window.parent !== window) {
    window.parent.postMessage({ type: 'orbexa.video.export', blob, filename }, location.origin);
    return;
  }
  const url = URL.createObjectURL(blob), anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

export async function sha256Bytes(value) {
  const bytes = typeof value === 'string' ? new TextEncoder().encode(value) :
    value instanceof Blob ? await value.arrayBuffer() : value;
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
}

/** Contiguous saved phases receive equal screen time, including short contact. */
export function recordingPhaseGroups(data) {
  const times = data.elapsed_s || data.q;
  if (!Array.isArray(times) || times.length < 2 || !Array.isArray(data.phases) || data.phases.length !== times.length ||
      times.some((time, index) => !Number.isFinite(time) || index > 0 && time <= times[index - 1]) ||
      data.phases.some(phase => typeof phase !== 'string' || !phase)) {
    throw new Error('Video export requires aligned, increasing saved samples and phases.');
  }
  const groups = [];
  data.phases.forEach((phase, index) => {
    if (!index || phase !== data.phases[index - 1]) groups.push({ phase, start: index, end: index });
    else groups.at(-1).end = index;
  });
  return groups;
}

export function phaseBalancedIndex(groups, fraction) {
  if (!groups.length || !Number.isFinite(fraction) || fraction < 0 || fraction > 1) {
    throw new Error('Invalid recorded-phase presentation coordinate.');
  }
  const position = fraction * groups.length, group = groups[Math.min(groups.length - 1, Math.floor(position))];
  const local = fraction === 1 ? 1 : position - Math.floor(position);
  // Keep both saved phase endpoints visible. Intermediate states are selected,
  // never interpolated; repeated screen frames do not create physical samples.
  const interior = Math.max(0, Math.min(1, (local - .1) / .8));
  return group.start + Math.round(interior * (group.end - group.start));
}

/** Composite the existing selected-chaser inset; never rescale scene geometry. */
export function drawTubeDetailInset(context, detail, { width = 480, top = 120, bottom = 16 } = {}) {
  const source = detail?.querySelector('canvas');
  if (!context || detail?.hidden !== false || !source ||
      ![source.width, source.height, context.canvas.width, context.canvas.height, width, top, bottom].every(Number.isFinite) ||
      source.width <= 0 || source.height <= 0 || width <= 0 || top < 0 || bottom < 0) return { visible: false };
  const padding = 12, header = 60, footer = 34;
  const availableHeight = context.canvas.height - top - bottom;
  const panelWidth = Math.min(width, context.canvas.width * .4,
    (availableHeight - header - footer) * source.width / source.height + 2 * padding);
  if (panelWidth <= 2 * padding || availableHeight <= header + footer) return { visible: false };
  const imageWidth = panelWidth - 2 * padding, imageHeight = imageWidth * source.height / source.width;
  const panelHeight = header + imageHeight + footer;
  const x = context.canvas.width - panelWidth - 16, y = context.canvas.height - panelHeight - bottom;
  if (x < 0 || y < top) return { visible: false };
  const label = detail.ownerDocument.getElementById('chaser')?.selectedOptions?.[0]?.textContent || 'Selected chaser';
  const title = detail.querySelector('.tube-detail-title')?.textContent || 'Recorded position tube';
  context.save();
  try {
    context.fillStyle = '#07121ff2'; context.fillRect(x, y, panelWidth, panelHeight);
    context.strokeStyle = '#687580'; context.lineWidth = 1; context.strokeRect(x, y, panelWidth, panelHeight);
    context.fillStyle = '#e5edf5'; context.font = '22px Segoe UI, sans-serif';
    context.textAlign = 'left'; context.textBaseline = 'alphabetic';
    context.fillText(`${label} · magnified local view`, x + padding, y + 24, imageWidth);
    context.font = '20px Segoe UI, sans-serif';
    context.fillText(title, x + padding, y + 48, imageWidth);
    context.drawImage(source, x + padding, y + header, imageWidth, imageHeight);
    context.fillStyle = '#a6bfd4'; context.font = '19px Segoe UI, sans-serif';
    context.fillText('● Actual  ○ Nominal · scene at physical scale', x + padding, y + panelHeight - 12, imageWidth);
  } finally { context.restore(); }
  return { visible: true, selected_chaser: label, title, x, y, width: panelWidth, height: panelHeight,
    source_canvas: { width: source.width, height: source.height },
    semantics: 'Existing recorded-sample error inset magnified independently; underlying scene retains physical scale' };
}

export class SceneVideoExporter {
  static supported() { return typeof MediaRecorder !== 'undefined' && typeof HTMLCanvasElement.prototype.captureStream === 'function'; }

  async record(scene, data, title, { duration = 20, onFrame, onProgress, signal, sourceBinding = null } = {}) {
    if (!SceneVideoExporter.supported()) throw new Error('This browser cannot export a canvas video. Saved MP4 files remain available in Videos.');
    if (!Number.isFinite(duration) || duration <= 0) throw new Error('Video duration must be positive.');
    const groups = recordingPhaseGroups(data), draws = [];
    const canvas = document.createElement('canvas');
    canvas.width = 1280;
    canvas.height = 720;
    const context = canvas.getContext('2d');
    let stream, recorder;
    const chunks = [];
    let animation;
    let frameFailure = null;
    let started;
    const values = data.elapsed_s || data.q;
    const draw = () => {
      try {
      // A browser's first RAF timestamp can precede recorder startup. Measure
      // actual callback execution with the same monotonic clock as startup.
      const mediaElapsed = Math.max(0, performance.now() - started);
      const fraction = Math.min(1, mediaElapsed / (duration * 1000));
      const index = phaseBalancedIndex(groups, fraction);
      const sample = onFrame(index);
      scene.render();
      context.fillStyle = '#010306';
      context.fillRect(0, 0, 1280, 720);
      const source = scene.renderer.domElement;
      const scale = Math.min(1280 / source.width, 630 / source.height);
      const width = source.width * scale, height = source.height * scale;
      context.drawImage(source, (1280 - width) / 2, 52 + (630 - height) / 2, width, height);
      const inset = drawTubeDetailInset(context, document.getElementById('tube-detail'), { top: 70, bottom: 42 });
      context.fillStyle = '#e5edf5';
      context.font = '600 21px Segoe UI, sans-serif';
      context.fillText('ORBEX-A  ·  ' + title, 26, 32);
      context.font = '14px Segoe UI, sans-serif';
      context.fillStyle = '#a6bfd4';
      const elapsed = data.elapsed_s ? `  ·  t = ${data.elapsed_s[index].toFixed(1)} s` : '';
      const coordinate = data.q ? `q = ${data.q[index].toFixed(4)} rad` : 'Physical lifecycle';
      context.fillText(`${coordinate}${elapsed}  ·  node ${index + 1}/${values.length}  ·  ${data.phases[index]}`, 26, 696);
      context.textAlign = 'right';
      context.fillText('Saved states · variable playback compression', 1254, 696);
      context.textAlign = 'left';
      draws.push({ draw_index: draws.length, media_elapsed_ms: mediaElapsed, timeline_index: index,
        kind: sample?.kind ?? null, index: sample?.index ?? index,
        physical_time_s: data.elapsed_s ? values[index] : null, phase: data.phases[index],
        tube_inset_visible: inset.visible, selected_chaser: sample?.selected_chaser ?? null,
        camera_view: sample?.camera_view ?? null });
      onProgress?.(Math.round(fraction * 100));
      if (fraction < 1 && !signal?.aborted) animation = requestAnimationFrame(draw);
      else if (recorder.state !== 'inactive') recorder.stop();
      } catch (error) {
        frameFailure = error;
        if (recorder.state !== 'inactive') recorder.stop();
      }
    };
    const abort = () => { cancelAnimationFrame(animation); if (recorder.state !== 'inactive') recorder.stop(); };
    signal?.addEventListener('abort', abort, { once: true });
    try {
      if (signal?.aborted) throw new Error('Video export cancelled.');
      stream = canvas.captureStream(30);
      const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/mp4'].find(type => MediaRecorder.isTypeSupported(type));
      recorder = new MediaRecorder(stream, { ...(mimeType ? { mimeType } : {}), videoBitsPerSecond: 5_000_000 });
      const blobPromise = new Promise((resolve, reject) => {
        recorder.addEventListener('dataavailable', event => { if (event.data.size) chunks.push(event.data); });
        recorder.addEventListener('stop', () => resolve(new Blob(chunks, { type: recorder.mimeType })));
        recorder.addEventListener('error', () => reject(new Error('The browser video encoder failed.')));
      });
      recorder.start(250);
      started = performance.now();
      draw();
      const blob = await blobPromise;
      if (frameFailure) throw frameFailure;
      if (signal?.aborted) throw new Error('Video export cancelled.');
      if (!blob.size) throw new Error('The browser produced an empty video.');
      if (groups.some(group => !draws.some(row => row.timeline_index === group.start) ||
          !draws.some(row => row.timeline_index === group.end))) {
        throw new Error('Video export missed a saved phase endpoint. Try exporting again.');
      }
      const videoSha256 = await sha256Bytes(blob);
      if (signal?.aborted) throw new Error('Video export cancelled.');
      return { blob, extension: blob.type.includes('mp4') ? 'mp4' : 'webm', provenance: {
        schema: 'orbexa.recorded-scene-video.v1', completed: true, cancelled: false, errors: [], source_binding: sourceBinding,
        video: { sha256: videoSha256, bytes: blob.size, mime_type: blob.type, width: 1280, height: 720 },
        presentation: { policy: 'equal-contiguous-phase-budgets-endpoint-dwell-v1', requested_duration_s: duration,
          requested_capture_fps: 30, phase_endpoint_fraction: .1, physical_clock: Boolean(data.elapsed_s), groups,
          timing: 'Equal screen time per contiguous phase; saved states only, with repeated frames and variable physical-time compression.' },
        draws, scope: 'Actual render callbacks and video Blob identity. Capture/encoding may repeat, drop or coalesce frames; no one-to-one encoded-frame correspondence is asserted.' } };
    } finally {
      cancelAnimationFrame(animation);
      signal?.removeEventListener('abort', abort);
      if (recorder && recorder.state !== 'inactive') recorder.stop();
      stream?.getTracks().forEach(track => track.stop());
    }
  }
}
