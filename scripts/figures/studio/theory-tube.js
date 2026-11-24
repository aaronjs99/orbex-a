/** Adjustable normalized geometry, explicitly separate from recorded flight. */
const NS = 'http://www.w3.org/2000/svg';

export function renderTubeExplainer(container, values) {
  if (container.dataset.figureKind !== 'sliding-tube') {
    container.dataset.figureKind = 'sliding-tube';
    container.innerHTML = `<div class="tube-explainer">
      <div class="tube-explainer-heading"><span>Sliding tube</span><small>Normalized geometry · illustrative</small></div>
      <svg role="img" aria-label="Adjustable normalized position and velocity error tube"></svg>
      <div class="tube-explainer-key"><span class="tube-key">Tube</span><span class="navigation-key">Navigation box</span><span class="state-key">State</span><output class="tube-membership" aria-live="polite"></output></div>
      <div class="tube-explainer-controls">
        <label>Tube width Φ / ΛΩ <output data-value="phi"></output><input aria-label="Illustrative sliding tube width" data-parameter="phi" type="range" min="0.15" max="1" step="0.05"></label>
        <label>Position error eₚ / Ω <output data-value="p"></output><input aria-label="Illustrative position error" data-parameter="p" type="range" min="-1.25" max="1.25" step="0.05"></label>
        <label>Sliding error s / ΛΩ <output data-value="s"></output><input aria-label="Illustrative sliding error" data-parameter="s" type="range" min="-1.25" max="1.25" step="0.05"></label>
      </div></div>`;
    for (const input of container.querySelectorAll('[data-parameter]')) {
      input.addEventListener('input', () => {
        const value = Number(input.value);
        if (!Number.isFinite(value)) return;
        values[input.dataset.parameter] = value;
        draw(container, values);
      });
    }
  }
  draw(container, values);
}

function draw(container, values) {
  const svg = container.querySelector('svg');
  const width = Math.max(280, container.clientWidth || 820), height = width < 520 ? 285 : 255;
  const left = 56, right = 22, top = 16, bottom = 40;
  const x = value => left + (value + 1.5) / 3 * (width - left - right);
  const y = value => top + (2.6 - value) / 5.2 * (height - top - bottom);
  const inside = Math.abs(values.p) <= 1 && Math.abs(values.s) <= values.phi;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.replaceChildren();
  function node(tag, attrs = {}, text = '') {
    const item = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) item.setAttribute(key, String(value));
    item.textContent = text; svg.append(item); return item;
  }
  node('title', {}, 'Normalized sliding tube and a separate navigation rectangle');
  node('desc', {}, 'The tube requires position error within plus or minus one and sliding error within the adjustable width. The state can leave either constraint. This is explanatory geometry, not mission data.');
  for (const tick of [-1, 0, 1]) {
    node('path', { d: `M${x(tick)} ${top}V${height - bottom}`, class: 'tube-explainer-grid' });
    node('text', { x: x(tick), y: height - bottom + 18, 'text-anchor': 'middle' }, tick.toString());
  }
  for (const tick of [-2, 0, 2]) {
    node('path', { d: `M${left} ${y(tick)}H${width - right}`, class: 'tube-explainer-grid' });
    node('text', { x: left - 12, y: y(tick) + 4, 'text-anchor': 'end' }, tick.toString());
  }
  const vertices = [[-1, 1 + values.phi], [1, -1 + values.phi], [1, -1 - values.phi], [-1, 1 - values.phi]];
  node('polygon', { points: vertices.map(point => `${x(point[0])},${y(point[1])}`).join(' '), class: 'tube-explainer-band' });
  node('path', { d: `M${x(-1)} ${y(1)}L${x(1)} ${y(-1)}`, class: 'tube-explainer-center' });
  node('rect', { x: x(-.1), y: y(.2), width: x(.5) - x(-.1), height: y(-.6) - y(.2), class: 'tube-explainer-navigation' });
  node('circle', { cx: x(values.p), cy: y(values.s - values.p), r: 6, class: inside ? 'tube-explainer-state' : 'tube-explainer-state outside' });
  node('text', { x: (left + width - right) / 2, y: height - 3, 'text-anchor': 'middle' }, 'Position error eₚ / Ω');
  node('text', { x: 15, y: (top + height - bottom) / 2, 'text-anchor': 'middle', transform: `rotate(-90 15 ${(top + height - bottom) / 2})` }, 'Velocity error eᵥ / ΛΩ');
  for (const input of container.querySelectorAll('[data-parameter]')) {
    const key = input.dataset.parameter;
    input.value = String(values[key]);
    container.querySelector(`[data-value="${key}"]`).textContent = values[key].toFixed(2);
  }
  const membership = container.querySelector('.tube-membership');
  membership.textContent = inside ? 'Inside tube' : 'Outside tube';
  membership.dataset.inside = String(inside);
}
