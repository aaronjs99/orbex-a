const NS = 'http://www.w3.org/2000/svg';
const node = (tag, attributes = {}, text = '') => {
  const element = document.createElementNS(NS, tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  element.textContent = text;
  return element;
};

/** All curves and cursors use the same recorded node indices as the scene. */
export class MetricChart {
  constructor(container, title, units, key, onSeek, { logarithmic = false, threshold = null } = {}) {
    Object.assign(this, { container, title, units, key, onSeek, logarithmic, threshold });
    this.resizeObserver = new ResizeObserver(() => {
      const width = this.container.clientWidth;
      if (this.data && width > 0 && width !== this.lastWidth) { this.load(this.data); this.update(this.index || 0, this.selected || 0); }
    });
    this.resizeObserver.observe(container);
  }

  load(data) {
    this.data = data;
    this.lastWidth = this.container.clientWidth;
    this.container.replaceChildren();
    const heading = document.createElement('h3');
    heading.textContent = this.title;
    const unit = document.createElement('span');
    unit.textContent = this.units + (this.logarithmic ? ' · log scale' : '');
    heading.append(unit);
    this.container.append(heading);
    const transform = value => this.logarithmic ? (value > 0 ? Math.log10(value) : null) : value;
    const values = data.tracks.flatMap(track => track[this.key].filter(value => value !== null && Number.isFinite(value)).map(transform)).filter(value => value !== null);
    if (!values.length) {
      const empty = document.createElement('div');
      empty.className = 'chart-empty';
      empty.textContent = 'Not recorded for this mission';
      this.container.append(empty);
      this.svg = null;
      return;
    }
    const W = Math.max(240, this.container.clientWidth - 30), H = 174, L = 48, R = 16, T = 13, B = 34;
    const physicalClock = Array.isArray(data.elapsed_s) && data.elapsed_s.length === data.q.length &&
      data.elapsed_s.every((value, index) => Number.isFinite(value) && (!index || value > data.elapsed_s[index - 1]));
    const clock = physicalClock ? data.elapsed_s : data.q;
    const lowQ = clock[0], highQ = clock.at(-1);
    const minimum = values.reduce((a, b) => Math.min(a, b), Infinity);
    const maximum = values.reduce((a, b) => Math.max(a, b), -Infinity);
    const narrowLog = this.logarithmic && maximum - minimum < 1;
    const logPad = Math.max(.001, (maximum - minimum) * .12);
    let low = this.logarithmic ? narrowLog ? minimum - logPad : Math.floor(minimum) : Math.min(0, minimum);
    let high = this.logarithmic ? narrowLog ? maximum + logPad : Math.ceil(maximum) : Math.max(0.1, this.threshold || 0, maximum) * 1.07;
    if (high === low) high = low + 1;
    this.x = q => L + (q - lowQ) / (highQ - lowQ || 1) * (W - L - R);
    const y = value => H - B - (value - low) / (high - low) * (H - B - T);
    this.svg = node('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${this.title}, ${this.units}, recorded samples` });
    this.container.append(this.svg);
    const tickStep = Math.max(1, Math.ceil((high - low) / 5));
    const ticks = this.logarithmic && !narrowLog ? Array.from({ length: Math.floor((high - low) / tickStep) + 1 }, (_, i) => low + i * tickStep) : Array.from({ length: 4 }, (_, i) => low + (high - low) * i / 3);
    if (this.logarithmic && !narrowLog && ticks.at(-1) !== high) ticks.push(high);
    for (const value of ticks) {
      const yy = y(value);
      this.svg.append(node('path', { d: `M${L} ${yy}H${W - R}`, stroke: '#243545', 'stroke-width': 1 }));
      const label = this.logarithmic ? narrowLog ? Number((10 ** value).toPrecision(4)).toString() : `1e${value}` : value.toFixed(high < 1 ? 2 : 1);
      this.svg.append(node('text', { x: L - 7, y: yy + 3, 'text-anchor': 'end' }, label));
    }
    for (let index = 0; index <= 3; index++) {
      const q = lowQ + (highQ - lowQ) * index / 3;
      const xx = this.x(q);
      this.svg.append(node('path', { d: `M${xx} ${T}V${H - B}`, stroke: '#243545', 'stroke-width': 1 }));
      this.svg.append(node('text', { x: xx, y: H - B + 17, 'text-anchor': 'middle' }, q.toFixed(physicalClock ? 1 : 3)));
    }
    if (this.threshold !== null) {
      this.svg.append(node('path', { d: `M${L} ${y(transform(this.threshold))}H${W - R}`, stroke: '#859ba9', 'stroke-dasharray': '4 5', 'stroke-width': 1 }));
    }
    this.lines = data.tracks.map(track => {
      let d = '', connected = false;
      track[this.key].forEach((raw, index) => {
        const value = raw === null ? null : transform(raw);
        if (value === null || !Number.isFinite(value)) { connected = false; return; }
        d += `${connected ? 'L' : 'M'}${this.x(clock[index]).toFixed(2)},${y(value).toFixed(2)}`;
        connected = true;
      });
      const path = node('path', { d, fill: 'none', stroke: track.color, 'stroke-width': 1.5, opacity: 0.6 });
      path.append(node('title', {}, track.label));
      this.svg.append(path);
      return path;
    });
    this.cursor = node('path', { stroke: '#e3eef6', 'stroke-width': 1, d: '' });
    this.clock = clock;
    const clockLabel = physicalClock ? 'Mission time (s)' : data.state_frame === 'reference-orbit-frame-v1' ? 'Reference q (rad)' : 'True anomaly (rad)';
    this.svg.append(this.cursor, node('text', { x: (W + L - R) / 2, y: H - 1, 'text-anchor': 'middle' }, clockLabel));
    this.svg.addEventListener('pointerdown', event => {
      const box = this.svg.getBoundingClientRect();
      const x = (event.clientX - box.left) / box.width * W;
      const q = lowQ + Math.max(0, Math.min(1, (x - L) / (W - L - R))) * (highQ - lowQ);
      let index = 0;
      clock.forEach((value, i) => { if (Math.abs(value - q) < Math.abs(clock[index] - q)) index = i; });
      this.onSeek(index);
    });
  }

  update(index, selected) {
    this.index = index;
    this.selected = selected;
    if (!this.svg) return;
    const x = this.x(this.clock[index]);
    this.cursor.setAttribute('d', `M${x} 13V140`);
    this.lines.forEach((line, number) => { line.setAttribute('opacity', number === selected ? 1 : 0.35); line.setAttribute('stroke-width', number === selected ? 2.2 : 1.2); });
  }
}
