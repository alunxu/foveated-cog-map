/* ============================================================
   Figure 4c — Future-position R² vs predictive horizon k
   How far into the future can h₂ predict position?
   Blind stays high → carries a forward-rolling trajectory;
   visual conditions decay to chance.
   ============================================================ */

(function () {
  const K = [0, 1, 2, 5, 10, 20, 50];

  const SERIES = {
    'Blind':     { color: '#3a3a3a', vals: [0.97, 0.97, 0.97, 0.97, 0.96, 0.92, 0.75], err: [0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.10] },
    'Coarse':    { color: '#3b7dd8', vals: [0.40, 0.40, 0.41, 0.42, 0.45, 0.23, -0.10], err: [0.08, 0.08, 0.10, 0.13, 0.13, 0.15, 0.20] },
    'Foveated':  { color: '#d92e2e', vals: [0.22, 0.22, 0.23, 0.23, 0.25, 0.22, -1.45], err: [0.05, 0.05, 0.06, 0.08, 0.13, 0.18, 0.18] },
    'Log-polar': { color: '#9333ea', vals: [-0.05, -0.05, -0.04, -0.06, -0.10, -0.65, -1.45], err: [0.04, 0.05, 0.05, 0.05, 0.08, 0.18, 0.20] },
    'Uniform':   { color: '#22a155', vals: [-0.05, -0.05, -0.05, -0.06, -0.18, -0.30, -1.45], err: [0.04, 0.04, 0.04, 0.05, 0.08, 0.12, 0.15] }
  };

  const MARKER = { 'Blind': 'circle', 'Coarse': 'square', 'Foveated': 'diamond', 'Log-polar': 'triangleDown', 'Uniform': 'triangle' };

  window.renderFig4c = function (container) {
    if (typeof container === 'string') container = document.getElementById(container);
    container.innerHTML = '';

    const W = 560, H = 360;
    const M = { top: 22, right: 22, bottom: 50, left: 60 };
    const iw = W - M.left - M.right;
    const ih = H - M.top - M.bottom;

    const yMin = -1.55, yMax = 1.10;
    const yScale = v => M.top + (yMax - v) / (yMax - yMin) * ih;
    // x axis: piecewise positions for non-linear k-spacing
    const xPos = [0, 0.18, 0.3, 0.45, 0.6, 0.78, 0.98]; // visual placement of [0,1,2,5,10,20,50]
    const xScale = i => M.left + xPos[i] * iw;

    const SVGNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    // Shade negative region
    const negRect = document.createElementNS(SVGNS, 'rect');
    negRect.setAttribute('x', M.left); negRect.setAttribute('y', yScale(0));
    negRect.setAttribute('width', iw); negRect.setAttribute('height', yScale(yMin) - yScale(0));
    negRect.setAttribute('fill', '#d92e2e'); negRect.setAttribute('opacity', '0.06');
    svg.appendChild(negRect);

    // y-grid
    [-1.5, -1.0, -0.5, 0, 0.5, 1.0].forEach(v => {
      const y = yScale(v);
      const line = document.createElementNS(SVGNS, 'line');
      line.setAttribute('class', v === 0 ? 'axis-zero' : 'axis-grid');
      line.setAttribute('x1', M.left); line.setAttribute('x2', M.left + iw);
      line.setAttribute('y1', y); line.setAttribute('y2', y);
      svg.appendChild(line);
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('class', 'axis-num');
      t.setAttribute('x', M.left - 6); t.setAttribute('y', y + 3.5);
      t.setAttribute('text-anchor', 'end');
      t.textContent = v.toFixed(1).replace('-', '−');
      svg.appendChild(t);
    });

    // x-axis baseline
    const xax = document.createElementNS(SVGNS, 'line');
    xax.setAttribute('class', 'axis-line');
    xax.setAttribute('x1', M.left); xax.setAttribute('x2', M.left + iw);
    xax.setAttribute('y1', M.top + ih); xax.setAttribute('y2', M.top + ih);
    svg.appendChild(xax);

    // x ticks
    K.forEach((k, i) => {
      const x = xScale(i);
      const tk = document.createElementNS(SVGNS, 'line');
      tk.setAttribute('class', 'axis-tick');
      tk.setAttribute('x1', x); tk.setAttribute('x2', x);
      tk.setAttribute('y1', M.top + ih); tk.setAttribute('y2', M.top + ih + 4);
      svg.appendChild(tk);
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('class', 'axis-num');
      t.setAttribute('x', x); t.setAttribute('y', M.top + ih + 16);
      t.setAttribute('text-anchor', 'middle');
      t.textContent = k;
      svg.appendChild(t);
    });

    const xT = document.createElementNS(SVGNS, 'text');
    xT.setAttribute('class', 'axis-title');
    xT.setAttribute('x', M.left + iw/2); xT.setAttribute('y', H - 10);
    xT.setAttribute('text-anchor', 'middle');
    xT.textContent = 'predictive horizon k (steps ahead)';
    svg.appendChild(xT);

    const yT = document.createElementNS(SVGNS, 'text');
    yT.setAttribute('class', 'axis-title');
    yT.setAttribute('transform', `translate(14, ${M.top + ih/2}) rotate(-90)`);
    yT.setAttribute('text-anchor', 'middle');
    yT.innerHTML = 'future-position <tspan font-style="italic">R²</tspan>';
    svg.appendChild(yT);

    // series
    Object.entries(SERIES).forEach(([cond, s]) => {
      // error bars
      s.vals.forEach((v, i) => {
        const x = xScale(i);
        const e = s.err[i] || 0;
        if (e > 0) {
          const line = document.createElementNS(SVGNS, 'line');
          line.setAttribute('class', 'errbar');
          line.setAttribute('x1', x); line.setAttribute('x2', x);
          const top = yScale(Math.min(yMax - 0.01, v + e));
          const bot = yScale(Math.max(yMin + 0.01, v - e));
          line.setAttribute('y1', top); line.setAttribute('y2', bot);
          line.setAttribute('stroke', s.color); line.setAttribute('stroke-opacity', '0.55');
          svg.appendChild(line);
        }
      });
      // line
      const path = document.createElementNS(SVGNS, 'path');
      const dStr = s.vals.map((v, i) => `${i ? 'L' : 'M'}${xScale(i)},${yScale(Math.max(yMin + 0.01, Math.min(yMax - 0.01, v)))}`).join(' ');
      path.setAttribute('d', dStr);
      path.setAttribute('class', 'lineSeries');
      path.setAttribute('stroke', s.color);
      svg.appendChild(path);

      // markers
      s.vals.forEach((v, i) => {
        const cy = yScale(Math.max(yMin + 0.01, Math.min(yMax - 0.01, v)));
        drawMarker(svg, MARKER[cond], xScale(i), cy, s.color, cond, K[i], v);
      });
    });

    container.appendChild(svg);

    // legend
    const legend = document.createElement('div');
    legend.className = 'chiprow';
    Object.entries(SERIES).forEach(([cond, s]) => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.innerHTML = `<span class="chip-swatch" style="background:${s.color}"></span>${cond}`;
      legend.appendChild(chip);
    });
    container.appendChild(legend);
  };

  function drawMarker(svg, kind, cx, cy, color, cond, k, v) {
    const SVGNS = 'http://www.w3.org/2000/svg';
    let el;
    const r = 4;
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
    el.setAttribute('stroke', '#fff'); el.setAttribute('stroke-width', 1.1);
    el.setAttribute('class', 'dot');
    attachTooltip(el, () => `<div class="tt-row"><span>${cond} · k=${k}</span><span class="tt-v">R²=${v.toFixed(2)}</span></div>`);
    svg.appendChild(el);
  }

  function getTip() {
    let t = document.getElementById('shared-tooltip');
    if (!t) { t = document.createElement('div'); t.id = 'shared-tooltip'; t.className = 'tt-tooltip'; document.body.appendChild(t); }
    return t;
  }
  function attachTooltip(el, html) {
    el.addEventListener('mouseenter', () => { const t = getTip(); t.innerHTML = html(); t.classList.add('is-visible'); });
    el.addEventListener('mousemove',  e => { const t = getTip(); t.style.left = (e.clientX + 12) + 'px'; t.style.top = (e.clientY + 12) + 'px'; });
    el.addEventListener('mouseleave', () => { getTip().classList.remove('is-visible'); });
  }
})();
