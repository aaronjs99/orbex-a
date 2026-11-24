/** Magnified local position-error view; the main scene retains physical dimensions. */
const lengthLabel = value => !Number.isFinite(value) ? 'Not recorded' : Math.abs(value) < .001 ? `${(value * 1e6).toPrecision(3)} µm` : Math.abs(value) < 1 ? `${(value * 1000).toPrecision(3)} mm` : `${value.toPrecision(3)} m`;

export class TubeDetail {
  constructor(container) {
    this.container = container;
    this.title = document.createElement('div');
    this.title.className = 'tube-detail-title';
    this.canvas = document.createElement('canvas');
    this.canvas.width = 440;
    this.canvas.height = 260;
    this.canvas.setAttribute('role', 'img');
    this.legend = document.createElement('div');
    this.legend.className = 'tube-detail-legend';
    this.legend.textContent = '● Actual  ○ Nominal · magnified local view';
    container.append(this.title, this.canvas, this.legend);
  }

  update(track, index, visible) {
    const center = track?.center?.[index], omega = track?.omega?.[index], actual = track?.states?.[index];
    const valid = center?.length === 3 && omega?.length === 3 && actual?.length >= 3 &&
      [...center, ...omega, ...actual.slice(0, 3)].every(Number.isFinite) && omega.every(value => value >= 0);
    this.container.hidden = !visible || !valid;
    if (!visible || !valid) return;
    const error = actual.slice(0, 3).map((value, axis) => value - center[axis]);
    const radius = Math.hypot(...omega);
    this.title.textContent = radius === 0 ? 'Initial tube · point' : `Tube detail · radius ${lengthLabel(radius)}`;
    this.legend.textContent = radius === 0 ? 'Grows after release' : '● Actual  ○ Nominal · magnified local view';
    this.canvas.setAttribute('aria-label', `${track.label} magnified position tube, half widths ${omega.map(lengthLabel).join(', ')}, position error ${error.map(lengthLabel).join(', ')}. The physical scene remains at true scale.`);
    const context = this.canvas.getContext('2d');
    const width = this.canvas.width, height = this.canvas.height;
    context.clearRect(0, 0, width, height);
    const corners = Array.from({ length: 8 }, (_, bit) => omega.map((value, axis) => (bit & (1 << axis) ? 1 : -1) * value));
    const project = row => [(row[0] - row[1]) * .75, (row[0] + row[1]) * .35 - row[2]];
    const projected = [...corners, error].map(project);
    const extent = Math.max(...projected.flat().map(Math.abs), 1e-15);
    const scale = Math.min(width * .37, height * .34) / extent;
    const point = row => { const p = project(row); return [width / 2 + p[0] * scale, height / 2 + p[1] * scale]; };
    const line = (from, to, color, lineWidth = 2) => {
      context.beginPath(); context.moveTo(...point(from)); context.lineTo(...point(to));
      context.strokeStyle = color; context.lineWidth = lineWidth; context.stroke();
    };
    corners.forEach((corner, bit) => { for (const axis of [1, 2, 4]) if (!(bit & axis)) line(corner, corners[bit | axis], track.color); });
    const axisLength = extent * .7;
    ['X', 'Y', 'Z'].forEach((label, axis) => {
      const tip = [0, 0, 0]; tip[axis] = axisLength;
      line([0, 0, 0], tip, '#71849a', 1);
      const p = point(tip); context.fillStyle = '#a8bbcd'; context.font = '19px Segoe UI, sans-serif'; context.fillText(label, p[0] + 5, p[1] - 5);
    });
    const dot = (row, filled, color) => {
      context.beginPath(); context.arc(...point(row), 6, 0, Math.PI * 2);
      context.strokeStyle = color; context.fillStyle = color; context.lineWidth = 2;
      if (filled) context.fill(); else context.stroke();
    };
    dot([0, 0, 0], false, '#e5edf5');
    dot(error, true, track.color);
    context.fillStyle = '#91a2b6'; context.font = '18px Segoe UI, sans-serif';
    context.fillText(`Box half widths: ${omega.map(lengthLabel).join(' / ')}`, 8, height - 12);
  }
}
