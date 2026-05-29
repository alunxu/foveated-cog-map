/* ============================================================
   Figure 2b — Substitution mechanism
   Top-layer GPS R² as a function of training frames (M).
   Rich sensors trade away GPS readability during training.
   ============================================================ */

(function () {
  const FRAMES = [50, 100, 150, 200, 250]; // M frames

  // [mean, ±err] per training-frame for each condition
  const SERIES = {
    'Blind':     { color: '#3a3a3a', vals: [0.95, 0.97, 0.97, 0.96, 0.96], err: [0.02, 0.02, 0.02, 0.02, 0.02] },
    'Coarse':    { color: '#3b7dd8', vals: [0.92, 0.54, 0.43, 0.58, 0.58], err: [0.04, 0.16, 0.13, 0.10, 0.09] },
    'Foveated':  { color: '#d92e2e', vals: [0.92, 0.86, 0.74, 0.28, 0.32], err: [0.03, 0.05, 0.10, 0.17, 0.15] },
    'Log-polar': { color: '#9333ea', vals: [0.85, 0.72, 0.74, 0.66, -0.02], err: [0.06, 0.12, 0.09, 0.10, 0.30] },
    'Uniform':   { color: '#22a155', vals: [0.83, 0.72, 0.37, -0.40, -1.00], err: [0.05, 0.08, 0.14, 0.20, 0.16] }
  };

  const MARKER = {
    'Blind': 'circle', 'Coarse': 'square', 'Foveated': 'diamond',
    'Log-polar': 'triangleDown', 'Uniform': 'triangle'
  };

  window.renderFig2b = function (container) {
    if (typeof container === 'string') container = document.getElementById(container);
    container.innerHTML = '';

    const W = 560, H = 360;
    const M = { top: 18, right: 18, bottom: 50, left: 56 };
    const iw = W - M.left - M.right;
    const ih = H - M.top - M.bottom;

    const xMin = 30, xMax = 270;
    const yMin = -1.05, yMax = 1.05;
    const xScale = v => M.left + (v - xMin) / (xMax - xMin) * iw;
    const yScale = v => M.top + (yMax - v) / (yMax - yMin) * ih;

    const SVGNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    // softly shaded region for negative R²
    const negRect = document.createElementNS(SVGNS, 'rect');
    negRect.setAttribute('x', M.left); negRect.setAttribute('y', yScale(0));
    negRect.setAttribute('width', iw); negRect.setAttribute('height', yScale(yMin) - yScale(0));
    negRect.setAttribute('fill', '#d92e2e'); negRect.setAttribute('opacity', '0.05');
    svg.appendChild(negRect);

    // grid (horizontal)
    for (let v = -1; v <= 1.001; v += 0.25) {
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
      t.textContent = v.toFixed(2).replace('-', '−');
      svg.appendChild(t);
    }

    // x-axis at zero line is implicit; draw axis baseline at yMin
    const xAxis = document.createElementNS(SVGNS, 'line');
    xAxis.setAttribute('class', 'axis-line');
    xAxis.setAttribute('x1', M.left); xAxis.setAttribute('x2', M.left + iw);
    xAxis.setAttribute('y1', M.top + ih); xAxis.setAttribute('y2', M.top + ih);
    svg.appendChild(xAxis);

    FRAMES.forEach(f => {
      const x = xScale(f);
      const tick = document.createElementNS(SVGNS, 'line');
      tick.setAttribute('class', 'axis-tick');
      tick.setAttribute('x1', x); tick.setAttribute('x2', x);
      tick.setAttribute('y1', M.top + ih); tick.setAttribute('y2', M.top + ih + 4);
      svg.appendChild(tick);
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('class', 'axis-num');
      t.setAttribute('x', x); t.setAttribute('y', M.top + ih + 16);
      t.setAttribute('text-anchor', 'middle');
      t.textContent = f;
      svg.appendChild(t);
    });

    // axis titles
    const xT = document.createElementNS(SVGNS, 'text');
    xT.setAttribute('class', 'axis-title');
    xT.setAttribute('x', M.left + iw/2); xT.setAttribute('y', H - 10);
    xT.setAttribute('text-anchor', 'middle');
    xT.textContent = 'training frames (M)';
    svg.appendChild(xT);

    const yT = document.createElementNS(SVGNS, 'text');
    yT.setAttribute('class', 'axis-title');
    yT.setAttribute('transform', `translate(14, ${M.top + ih/2}) rotate(-90)`);
    yT.setAttribute('text-anchor', 'middle');
    yT.innerHTML = 'top-layer GPS <tspan font-style="italic">R²</tspan>';
    svg.appendChild(yT);

    // draw series ----------------------------------------------
    Object.entries(SERIES).forEach(([cond, s]) => {
      // error band-ish: vertical bars at each point
      s.vals.forEach((v, i) => {
        const x = xScale(FRAMES[i]);
        const e = s.err[i] || 0;
        if (e > 0) {
          const line = document.createElementNS(SVGNS, 'line');
          line.setAttribute('class', 'errbar');
          line.setAttribute('x1', x); line.setAttribute('x2', x);
          line.setAttribute('y1', yScale(v + e)); line.setAttribute('y2', yScale(v - e));
          line.setAttribute('stroke', s.color); line.setAttribute('stroke-opacity', '0.55');
          svg.appendChild(line);
        }
      });
      // line
      const path = document.createElementNS(SVGNS, 'path');
      const dStr = s.vals.map((v, i) => `${i ? 'L' : 'M'}${xScale(FRAMES[i])},${yScale(v)}`).join(' ');
      path.setAttribute('d', dStr);
      path.setAttribute('class', 'lineSeries');
      path.setAttribute('stroke', s.color);
      svg.appendChild(path);

      // markers
      s.vals.forEach((v, i) => {
        drawMarker(svg, MARKER[cond], xScale(FRAMES[i]), yScale(v), s.color, cond, FRAMES[i], v);
      });
    });

    container.appendChild(svg);

    // legend ----------------------------------------------------
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

  // ---------- helpers ----------
  function drawMarker(svg, kind, cx, cy, color, cond, f, v) {
    const SVGNS = 'http://www.w3.org/2000/svg';
    let el;
    const r = 4.2;
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
    } else { // triangleDown
      el = document.createElementNS(SVGNS, 'polygon');
      el.setAttribute('points', `${cx-r*1.2},${cy-r*0.9} ${cx+r*1.2},${cy-r*0.9} ${cx},${cy+r*1.2}`);
    }
    el.setAttribute('fill', color);
    el.setAttribute('stroke', '#fff');
    el.setAttribute('stroke-width', 1.1);
    el.setAttribute('class', 'dot');
    attachTooltip(el, () => `<div class="tt-row"><span>${cond} · ${f}M</span><span class="tt-v">R²=${v.toFixed(2)}</span></div>`);
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
