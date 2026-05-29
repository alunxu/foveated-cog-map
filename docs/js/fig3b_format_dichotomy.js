/* ============================================================
   Figure 3b — Format dichotomy
   Scene-invariance (LOSO R²) vs transplant cost.
   Blind sits in the "scene-invariant + robust" quadrant;
   all five visual sensors cluster in "scene-conditional".
   ============================================================ */

(function () {
  const POINTS = [
    { cond: 'Blind',     x: 0.92, y: 0.025, color: '#3a3a3a', marker: 'circle',       size: 11 },
    { cond: 'Coarse',    x: 0.37, y: 0.175, color: '#3b7dd8', marker: 'square',       size: 8 },
    { cond: 'Log-polar', x: 0.24, y: 0.155, color: '#9333ea', marker: 'triangleDown', size: 8 },
    { cond: 'Foveated',  x: 0.28, y: 0.145, color: '#d92e2e', marker: 'diamond',      size: 8 },
    { cond: 'Uniform',   x: 0.19, y: 0.128, color: '#22a155', marker: 'triangle',     size: 8 }
  ];

  window.renderFig3b = function (container) {
    if (typeof container === 'string') container = document.getElementById(container);
    container.innerHTML = '';

    const W = 640, H = 380;
    const M = { top: 50, right: 36, bottom: 50, left: 64 };
    const iw = W - M.left - M.right;
    const ih = H - M.top - M.bottom;

    const xMin = 0, xMax = 1.05;
    const yMin = 0, yMax = 0.26;
    const xs = v => M.left + (v - xMin) / (xMax - xMin) * iw;
    const ys = v => M.top + (yMax - v) / (yMax - yMin) * ih;

    const SVGNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    // grid
    [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(v => {
      const x = xs(v);
      const line = document.createElementNS(SVGNS, 'line');
      line.setAttribute('class', 'axis-grid');
      line.setAttribute('x1', x); line.setAttribute('x2', x);
      line.setAttribute('y1', M.top); line.setAttribute('y2', M.top + ih);
      svg.appendChild(line);
    });
    [0.05, 0.10, 0.15, 0.20, 0.25].forEach(v => {
      const y = ys(v);
      const line = document.createElementNS(SVGNS, 'line');
      line.setAttribute('class', 'axis-grid');
      line.setAttribute('x1', M.left); line.setAttribute('x2', M.left + iw);
      line.setAttribute('y1', y); line.setAttribute('y2', y);
      svg.appendChild(line);
    });

    // divider at x = 0.5 (scene-invariant threshold)
    const divX = xs(0.5);
    const div = document.createElementNS(SVGNS, 'line');
    div.setAttribute('class', 'axis-zero');
    div.setAttribute('x1', divX); div.setAttribute('x2', divX);
    div.setAttribute('y1', M.top); div.setAttribute('y2', M.top + ih);
    svg.appendChild(div);

    // axes
    const xax = document.createElementNS(SVGNS, 'line');
    xax.setAttribute('class', 'axis-line');
    xax.setAttribute('x1', M.left); xax.setAttribute('x2', M.left + iw);
    xax.setAttribute('y1', M.top + ih); xax.setAttribute('y2', M.top + ih);
    svg.appendChild(xax);
    const yax = document.createElementNS(SVGNS, 'line');
    yax.setAttribute('class', 'axis-line');
    yax.setAttribute('x1', M.left); yax.setAttribute('x2', M.left);
    yax.setAttribute('y1', M.top); yax.setAttribute('y2', M.top + ih);
    svg.appendChild(yax);

    // x ticks
    [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(v => {
      const x = xs(v);
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('class', 'axis-num');
      t.setAttribute('x', x); t.setAttribute('y', M.top + ih + 14);
      t.setAttribute('text-anchor', 'middle');
      t.textContent = v.toFixed(1);
      svg.appendChild(t);
    });
    // y ticks
    [0, 0.05, 0.10, 0.15, 0.20, 0.25].forEach(v => {
      const y = ys(v);
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('class', 'axis-num');
      t.setAttribute('x', M.left - 6); t.setAttribute('y', y + 3.5);
      t.setAttribute('text-anchor', 'end');
      t.textContent = v.toFixed(2);
      svg.appendChild(t);
    });

    // axis titles
    const xT = document.createElementNS(SVGNS, 'text');
    xT.setAttribute('class', 'axis-title');
    xT.setAttribute('x', M.left + iw/2); xT.setAttribute('y', H - 10);
    xT.setAttribute('text-anchor', 'middle');
    xT.innerHTML = 'LOSO <tspan font-style="italic">R²</tspan> (scene-invariance)';
    svg.appendChild(xT);

    const yT = document.createElementNS(SVGNS, 'text');
    yT.setAttribute('class', 'axis-title');
    yT.setAttribute('transform', `translate(16, ${M.top + ih/2}) rotate(-90)`);
    yT.setAttribute('text-anchor', 'middle');
    yT.textContent = 'transplant cost';
    svg.appendChild(yT);

    // quadrant annotations
    const annTopLeft = document.createElementNS(SVGNS, 'text');
    annTopLeft.setAttribute('x', M.left + 8); annTopLeft.setAttribute('y', M.top + 14);
    annTopLeft.setAttribute('class', 'annot-quad');
    annTopLeft.setAttribute('fill', '#a03838');
    annTopLeft.setAttribute('font-style', 'italic');
    annTopLeft.innerHTML = '<tspan x="' + (M.left + 8) + '">scene-conditional</tspan><tspan x="' + (M.left + 8) + '" dy="13">+ brittle to transplant</tspan>';
    svg.appendChild(annTopLeft);

    const annBotRight = document.createElementNS(SVGNS, 'text');
    annBotRight.setAttribute('x', M.left + iw - 8); annBotRight.setAttribute('y', M.top + ih - 64);
    annBotRight.setAttribute('class', 'annot-quad');
    annBotRight.setAttribute('text-anchor', 'end');
    annBotRight.setAttribute('fill', '#2a6f4f');
    annBotRight.setAttribute('font-style', 'italic');
    annBotRight.innerHTML = '<tspan x="' + (M.left + iw - 8) + '" text-anchor="end">scene-invariant</tspan><tspan x="' + (M.left + iw - 8) + '" text-anchor="end" dy="13">+ robust to transplant</tspan>';
    svg.appendChild(annBotRight);

    // points
    POINTS.forEach(p => {
      const cx = xs(p.x), cy = ys(p.y);
      drawMarker(svg, p.marker, cx, cy, p.color, p.size);

      // label slightly above/right of the point
      const lab = document.createElementNS(SVGNS, 'text');
      lab.setAttribute('class', 'annot-text');
      lab.setAttribute('fill', p.color);
      lab.setAttribute('font-weight', '600');
      lab.setAttribute('font-size', '12');
      // Blind sits at far right, label to the left
      if (p.cond === 'Blind') {
        lab.setAttribute('x', cx + 13); lab.setAttribute('y', cy + 4);
        lab.setAttribute('text-anchor', 'start');
      } else if (p.cond === 'Uniform') {
        lab.setAttribute('x', cx + 9); lab.setAttribute('y', cy + 4);
        lab.setAttribute('text-anchor', 'start');
      } else if (p.cond === 'Log-polar') {
        lab.setAttribute('x', cx); lab.setAttribute('y', cy - 9);
        lab.setAttribute('text-anchor', 'middle');
      } else {
        lab.setAttribute('x', cx + 10); lab.setAttribute('y', cy + 4);
        lab.setAttribute('text-anchor', 'start');
      }
      lab.textContent = p.cond;
      svg.appendChild(lab);

      attachTooltip(svg.lastChild, () => '');
    });

    container.appendChild(svg);
  };

  function drawMarker(svg, kind, cx, cy, color, size) {
    const SVGNS = 'http://www.w3.org/2000/svg';
    let el;
    const r = size;
    if (kind === 'circle') {
      el = document.createElementNS(SVGNS, 'circle');
      el.setAttribute('cx', cx); el.setAttribute('cy', cy); el.setAttribute('r', r);
    } else if (kind === 'square') {
      el = document.createElementNS(SVGNS, 'rect');
      el.setAttribute('x', cx - r); el.setAttribute('y', cy - r);
      el.setAttribute('width', r*2); el.setAttribute('height', r*2);
    } else if (kind === 'diamond') {
      el = document.createElementNS(SVGNS, 'polygon');
      el.setAttribute('points', `${cx},${cy-r*1.2} ${cx+r*1.2},${cy} ${cx},${cy+r*1.2} ${cx-r*1.2},${cy}`);
    } else if (kind === 'triangle') {
      el = document.createElementNS(SVGNS, 'polygon');
      el.setAttribute('points', `${cx},${cy-r*1.2} ${cx+r*1.2},${cy+r*0.9} ${cx-r*1.2},${cy+r*0.9}`);
    } else {
      el = document.createElementNS(SVGNS, 'polygon');
      el.setAttribute('points', `${cx-r*1.2},${cy-r*0.9} ${cx+r*1.2},${cy-r*0.9} ${cx},${cy+r*1.2}`);
    }
    el.setAttribute('fill', color);
    el.setAttribute('fill-opacity', '0.92');
    el.setAttribute('stroke', '#fff');
    el.setAttribute('stroke-width', 1.4);
    el.setAttribute('class', 'dot');
    svg.appendChild(el);
    return el;
  }

  function getTip() {
    let t = document.getElementById('shared-tooltip');
    if (!t) { t = document.createElement('div'); t.id = 'shared-tooltip'; t.className = 'tt-tooltip'; document.body.appendChild(t); }
    return t;
  }
  function attachTooltip(el, html) {
    if (!html()) return;
    el.addEventListener('mouseenter', () => { const t = getTip(); t.innerHTML = html(); t.classList.add('is-visible'); });
    el.addEventListener('mousemove',  e => { const t = getTip(); t.style.left = (e.clientX + 12) + 'px'; t.style.top = (e.clientY + 12) + 'px'; });
    el.addEventListener('mouseleave', () => { getTip().classList.remove('is-visible'); });
  }
})();
