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

export class SceneVideoExporter {
  static supported() { return typeof MediaRecorder !== 'undefined' && typeof HTMLCanvasElement.prototype.captureStream === 'function'; }

  async record(scene, data, title, { duration = 20, onFrame, onProgress, signal } = {}) {
    if (!SceneVideoExporter.supported()) throw new Error('This browser cannot export a canvas video. Saved MP4 files remain available in Videos.');
    const canvas = document.createElement('canvas');
    canvas.width = 1280;
    canvas.height = 720;
    const context = canvas.getContext('2d');
    let stream, recorder;
    const chunks = [];
    let animation;
    let frameFailure = null;
    const started = performance.now();
    const values = data.elapsed_s || data.q;
    const start = values[0], end = values.at(-1);
    let index = 0;
    const draw = now => {
      try {
      const fraction = Math.min(1, (now - started) / (duration * 1000));
      const clock = start + fraction * (end - start);
      while (index < values.length - 1 && values[index + 1] <= clock) index++;
      if (fraction === 1) index = values.length - 1;
      onFrame(index);
      scene.render();
      context.fillStyle = '#010306';
      context.fillRect(0, 0, 1280, 720);
      const source = scene.renderer.domElement;
      const scale = Math.min(1280 / source.width, 630 / source.height);
      const width = source.width * scale, height = source.height * scale;
      context.drawImage(source, (1280 - width) / 2, 52 + (630 - height) / 2, width, height);
      context.fillStyle = '#e5edf5';
      context.font = '600 21px Segoe UI, sans-serif';
      context.fillText('ORBEX-A  ·  ' + title, 26, 32);
      context.font = '14px Segoe UI, sans-serif';
      context.fillStyle = '#a6bfd4';
      const elapsed = data.elapsed_s ? `  ·  t = ${data.elapsed_s[index].toFixed(1)} s` : '';
      const coordinate = data.q ? `q = ${data.q[index].toFixed(4)} rad` : 'Physical lifecycle';
      context.fillText(`${coordinate}${elapsed}  ·  node ${index + 1}/${values.length}  ·  ${data.phases[index]}`, 26, 696);
      context.textAlign = 'right';
      context.fillText('Recorded states · representative visual model', 1254, 696);
      context.textAlign = 'left';
      onProgress(Math.round(fraction * 100));
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
      animation = requestAnimationFrame(draw);
      const blob = await blobPromise;
      if (frameFailure) throw frameFailure;
      if (signal?.aborted) throw new Error('Video export cancelled.');
      return { blob, extension: blob.type.includes('mp4') ? 'mp4' : 'webm' };
    } finally {
      cancelAnimationFrame(animation);
      signal?.removeEventListener('abort', abort);
      if (recorder && recorder.state !== 'inactive') recorder.stop();
      stream?.getTracks().forEach(track => track.stop());
    }
  }
}
