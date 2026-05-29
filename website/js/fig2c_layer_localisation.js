/* ============================================================
   Figure 2c — Layer localisation
   GPS R² per LSTM layer (L0, L1, L2) for each condition.
   Cells coloured on a diverging scale.
   ============================================================ */

(function () {
  // rows × cols
  const ROWS = ['Blind', 'Coarse', 'Foveated', 'Log-polar', 'Uniform'];
  const COLS = ['L0', 'L1', 'L2'];
  const DATA = [
    [+0.99, +0.54, +0.95],   // Blind
    [+0.96, -0.09, +0.69],   // Coarse
    [+0.95, +0.02, +0.56],   // Foveated
    [+0.96, +0.10, +0.32],   // Log-polar
    [+0.96, +0.15, +0.66]    // Uniform
  ];

  // diverging scale: red (neg) → white (0) → blue (pos)
  // domain is roughly [-2, 1] in the paper; map symmetrically to [-1, 1]
  function colorFor(v) {
    // v in [-1, 1] approx
    const t = Math.max(-1, Math.min(1, v));
    if (t >= 0) {
      // 0 (white) → +1 (blue)
      const k = t;            // 0→1
      const r = lerp(255, 29, k);
      const g = lerp(255, 78, k);
      const b = lerp(255, 216, k);
      return `rgb(${r|0},${g|0},${b|0})`;
    } else {
      const k = -t;           // 0→1
      const r = lerp(255, 185, k);
      const g = lerp(255, 28, k);
      const b = lerp(255, 28, k);
      return `rgb(${r|0},${g|0},${b|0})`;
    }
  }
  function lerp(a, b, t) { return a + (b - a) * t; }

  window.renderFig2c = function (container) {
    if (typeof container === 'string') container = document.getElementById(container);
    container.innerHTML = '';

    const W = 480, H = 360;
    const M = { top: 32, right: 70, bottom: 50, left: 86 };
    const iw = W - M.left - M.right;
    const ih = H - M.top - M.bottom;

    const cellW = iw / COLS.length;
    const cellH = ih / ROWS.length;

    const SVGNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    // cells
    ROWS.forEach((cond, r) => {
      COLS.forEach((layer, c) => {
        const v = DATA[r][c];
        const x = M.left + c * cellW;
        const y = M.top + r * cellH;
        const rect = document.createElementNS(SVGNS, 'rect');
        rect.setAttribute('x', x + 1); rect.setAttribute('y', y + 1);
        rect.setAttribute('width', cellW - 2); rect.setAttribute('height', cellH - 2);
        rect.setAttribute('fill', colorFor(v));
        rect.setAttribute('rx', 4);
        rect.setAttribute('class', 'bar');
        attachTooltip(rect, () => `<div class="tt-row"><span>${cond} · ${layer}</span><span class="tt-v">R²=${v >= 0 ? '+' : ''}${v.toFixed(2)}</span></div>`);
        svg.appendChild(rect);

        // label inside the cell
        const t = document.createElementNS(SVGNS, 'text');
        t.setAttribute('x', x + cellW/2); t.setAttribute('y', y + cellH/2 + 4);
        t.setAttribute('text-anchor', 'middle');
        t.setAttribute('font-family', 'DM Mono, monospace');
        t.setAttribute('font-size', '13');
        t.setAttribute('font-weight', '600');
        // contrast: if very saturated, use white; else dark
        const sat = Math.abs(v);
        t.setAttribute('fill', sat > 0.45 ? '#fff' : '#1a2942');
        t.textContent = `${v >= 0 ? '+' : ''}${v.toFixed(2)}`;
        svg.appendChild(t);
      });
    });

    // row labels
    ROWS.forEach((cond, r) => {
      const y = M.top + r * cellH + cellH/2 + 4;
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('x', M.left - 10); t.setAttribute('y', y);
      t.setAttribute('text-anchor', 'end');
      t.setAttribute('class', 'axis-label');
      t.setAttribute('fill', condColor(cond));
      t.setAttribute('font-weight', '600');
      t.textContent = cond;
      svg.appendChild(t);
    });

    // column labels
    COLS.forEach((layer, c) => {
      const x = M.left + c * cellW + cellW/2;
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('x', x); t.setAttribute('y', M.top - 10);
      t.setAttribute('text-anchor', 'middle');
      t.setAttribute('class', 'axis-label');
      t.setAttribute('font-weight', '600');
      t.textContent = layer;
      svg.appendChild(t);
    });

    // x-axis title
    const xT = document.createElementNS(SVGNS, 'text');
    xT.setAttribute('class', 'axis-title');
    xT.setAttribute('x', M.left + iw/2); xT.setAttribute('y', H - 14);
    xT.setAttribute('text-anchor', 'middle');
    xT.textContent = 'LSTM layer';
    svg.appendChild(xT);

    // color legend (vertical bar on the right)
    const legX = M.left + iw + 18;
    const legW = 10;
    const legTop = M.top, legBot = M.top + ih;
    const grad = document.createElementNS(SVGNS, 'linearGradient');
    grad.setAttribute('id', 'fig2c-grad');
    grad.setAttribute('x1', 0); grad.setAttribute('y1', 0);
    grad.setAttribute('x2', 0); grad.setAttribute('y2', 1);
    // stops top → bottom map +1 → -1
    const stops = [
      [0,    'rgb(29,78,216)'],
      [0.45, 'rgb(180,205,240)'],
      [0.5,  'rgb(255,255,255)'],
      [0.55, 'rgb(240,200,200)'],
      [1,    'rgb(185,28,28)']
    ];
    stops.forEach(([off, col]) => {
      const s = document.createElementNS(SVGNS, 'stop');
      s.setAttribute('offset', off); s.setAttribute('stop-color', col);
      grad.appendChild(s);
    });
    const defs = document.createElementNS(SVGNS, 'defs');
    defs.appendChild(grad);
    svg.appendChild(defs);

    const legRect = document.createElementNS(SVGNS, 'rect');
    legRect.setAttribute('x', legX); legRect.setAttribute('y', legTop);
    legRect.setAttribute('width', legW); legRect.setAttribute('height', legBot - legTop);
    legRect.setAttribute('fill', 'url(#fig2c-grad)');
    legRect.setAttribute('stroke', '#1a2942'); legRect.setAttribute('stroke-width', 0.5);
    legRect.setAttribute('rx', 2);
    svg.appendChild(legRect);

    // ticks on color bar
    const legTicks = [+1, +0.5, 0, -0.5, -1];
    legTicks.forEach(v => {
      // map  +1 → top, -1 → bottom
      const y = legTop + (1 - (v + 1) / 2) * (legBot - legTop);
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('x', legX + legW + 4); t.setAttribute('y', y + 3.5);
      t.setAttribute('class', 'axis-num');
      t.textContent = (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(1);
      svg.appendChild(t);
    });

    const legT = document.createElementNS(SVGNS, 'text');
    legT.setAttribute('class', 'axis-title');
    legT.setAttribute('transform', `translate(${W - 8}, ${(legTop + legBot)/2}) rotate(90)`);
    legT.setAttribute('text-anchor', 'middle');
    legT.innerHTML = 'GPS <tspan font-style="italic">R²</tspan>';
    svg.appendChild(legT);

    container.appendChild(svg);
  };

  function condColor(c) {
    return ({
      'Blind': '#3a3a3a', 'Coarse': '#3b7dd8', 'Foveated': '#d92e2e',
      'Log-polar': '#9333ea', 'Uniform': '#22a155'
    })[c];
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
