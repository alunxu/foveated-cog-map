/* ============================================================
   Figure 5a — Probe-readability ≠ policy use
   x: GPS R² on h2 (memory readability of position)
   y: shortcut SPL drop after replacing memory with a baseline
      (= how much the policy ACTUALLY relies on memory)
   Labels next to each point quote the disorientation distance.
   ============================================================ */

(function () {
  const POINTS = [
    { cond: 'Blind',     x:  0.9,  y: 0.72,  dist: '−1.7 m',  marker: 'circle',       size: 9,
      labelPos: 'below', distPos: 'left' },
    { cond: 'Coarse',    x:  0.5,  y: 0.13,  dist: '−5.0 m',  marker: 'square',       size: 7,
      labelPos: 'above', distPos: 'right' },
    { cond: 'Foveated',  x:  0.4,  y: 0.06,  dist: '−13.7 m', marker: 'diamond',      size: 8,
      labelPos: 'below', distPos: 'right' },
    { cond: 'Log-polar', x: -0.1,  y: 0.10,  dist: '−6.3 m',  marker: 'triangleDown', size: 7,
      labelPos: 'above', distPos: 'right' },
    { cond: 'Uniform',   x: -1.5,  y: 0.12,  dist: '−3.2 m',  marker: 'triangle',     size: 7,
      labelPos: 'above', distPos: 'right' }
  ];
  const COLORS = {
    'Blind':     '#3a3a3a',
    'Coarse':    '#3b7dd8',
    'Foveated':  '#d92e2e',
    'Log-polar': '#9333ea',
    'Uniform':   '#22a155'
  };

  // Parse "−1.7 m" → 1.7
  function parseDistM(s) {
    return Math.abs(parseFloat(s.replace('−', '-')));
  }

  const MAX_DIST   = 13.7; // foveated — sets scale
  const MAX_ARROW  = 68;   // px at MAX_DIST

  window.renderFig5a = function (container) {
    if (typeof container === 'string') container = document.getElementById(container);
    container.innerHTML = '';

    const W = 580, H = 380;
    const M = { top: 38, right: 30, bottom: 56, left: 62 };
    const iw = W - M.left - M.right;
    const ih = H - M.top - M.bottom;

    const xMin = -1.7, xMax = 1.25;
    const yMin = -0.02, yMax = 1.0;
    const xs = v => M.left + (v - xMin) / (xMax - xMin) * iw;
    const ys = v => M.top  + (yMax - v) / (yMax - yMin) * ih;

    const SVGNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    const divX = xs(0.3);
    const divY = ys(0.2);

    // ── quadrant tints (drawn first, under everything) ────────
    const QUADS = [
      // top-left: UNREADABLE, USED — warm red
      { x: M.left, y: M.top,  w: divX - M.left,       h: divY - M.top,       fill: 'rgba(180,60,60,0.07)'  },
      // top-right: READABLE, USED — green
      { x: divX,   y: M.top,  w: M.left + iw - divX,  h: divY - M.top,       fill: 'rgba(30,140,80,0.07)'  },
      // bottom-left: UNREADABLE, UNUSED — cool gray
      { x: M.left, y: divY,   w: divX - M.left,        h: M.top + ih - divY,  fill: 'rgba(100,110,160,0.07)' },
      // bottom-right: READABLE, UNUSED — amber
      { x: divX,   y: divY,   w: M.left + iw - divX,   h: M.top + ih - divY,  fill: 'rgba(190,140,30,0.07)' }
    ];
    QUADS.forEach(q => {
      const r = document.createElementNS(SVGNS, 'rect');
      r.setAttribute('x', q.x); r.setAttribute('y', q.y);
      r.setAttribute('width', q.w); r.setAttribute('height', q.h);
      r.setAttribute('fill', q.fill);
      svg.appendChild(r);
    });

    // ── grid ─────────────────────────────────────────────────
    [-1.5, -1.0, -0.5, 0, 0.5, 1.0].forEach(v => {
      const x = xs(v);
      const line = document.createElementNS(SVGNS, 'line');
      line.setAttribute('class', 'axis-grid');
      line.setAttribute('x1', x); line.setAttribute('x2', x);
      line.setAttribute('y1', M.top); line.setAttribute('y2', M.top + ih);
      svg.appendChild(line);
    });
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(v => {
      const y = ys(v);
      const line = document.createElementNS(SVGNS, 'line');
      line.setAttribute('class', 'axis-grid');
      line.setAttribute('x1', M.left); line.setAttribute('x2', M.left + iw);
      line.setAttribute('y1', y); line.setAttribute('y2', y);
      svg.appendChild(line);
    });

    // ── quadrant dividers ─────────────────────────────────────
    const dxL = document.createElementNS(SVGNS, 'line');
    dxL.setAttribute('class', 'axis-zero');
    dxL.setAttribute('x1', divX); dxL.setAttribute('x2', divX);
    dxL.setAttribute('y1', M.top); dxL.setAttribute('y2', M.top + ih);
    svg.appendChild(dxL);

    const dyL = document.createElementNS(SVGNS, 'line');
    dyL.setAttribute('class', 'axis-zero');
    dyL.setAttribute('x1', M.left); dyL.setAttribute('x2', M.left + iw);
    dyL.setAttribute('y1', divY); dyL.setAttribute('y2', divY);
    svg.appendChild(dyL);

    // ── axes ─────────────────────────────────────────────────
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

    // ── x ticks ───────────────────────────────────────────────
    [-1.5, -1.0, -0.5, 0, 0.5, 1.0].forEach(v => {
      const x = xs(v);
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('class', 'axis-num');
      t.setAttribute('x', x); t.setAttribute('y', M.top + ih + 14);
      t.setAttribute('text-anchor', 'middle');
      t.textContent = v.toFixed(1).replace('-', '−');
      svg.appendChild(t);
    });
    // ── y ticks ───────────────────────────────────────────────
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(v => {
      const y = ys(v);
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('class', 'axis-num');
      t.setAttribute('x', M.left - 6); t.setAttribute('y', y + 3.5);
      t.setAttribute('text-anchor', 'end');
      t.textContent = v.toFixed(1);
      svg.appendChild(t);
    });

    // ── axis titles ───────────────────────────────────────────
    const xT = document.createElementNS(SVGNS, 'text');
    xT.setAttribute('class', 'axis-title');
    xT.setAttribute('x', M.left + iw / 2); xT.setAttribute('y', H - 10);
    xT.setAttribute('text-anchor', 'middle');
    xT.innerHTML = 'GPS <tspan font-style="italic">R²</tspan> on h₂';
    svg.appendChild(xT);

    const yT = document.createElementNS(SVGNS, 'text');
    yT.setAttribute('class', 'axis-title');
    yT.setAttribute('transform', `translate(14, ${M.top + ih / 2}) rotate(-90)`);
    yT.setAttribute('text-anchor', 'middle');
    yT.textContent = 'shortcut SPL drop';
    svg.appendChild(yT);

    // ── quadrant labels ───────────────────────────────────────
    function quadLabel(x, y, color, text, anchor) {
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('x', x); t.setAttribute('y', y);
      t.setAttribute('text-anchor', anchor);
      t.setAttribute('fill', color);
      t.setAttribute('font-weight', '600');
      t.setAttribute('font-size', '10');
      t.setAttribute('font-style', 'italic');
      t.textContent = text;
      svg.appendChild(t);
    }
    // Top labels: centered within each quadrant
    const tlCx = (M.left + divX) / 2;
    const trCx = (divX + M.left + iw) / 2;
    quadLabel(tlCx,           M.top + 14, '#a03838', 'UNREADABLE, USED',   'middle');
    quadLabel(trCx,           M.top + 14, '#2a6f4f', 'READABLE, USED',     'middle');
    // Bottom labels near inner corners
    quadLabel(M.left + 6,     M.top + ih - 4, '#9a9080', 'UNREADABLE, UNUSED', 'start');
    quadLabel(M.left + iw - 6, M.top + ih - 4, '#b07c1f', 'READABLE, UNUSED',  'end');

    // ── arrows + points + labels ──────────────────────────────
    POINTS.forEach(p => {
      const cx = xs(p.x), cy = ys(p.y);
      const c  = COLORS[p.cond];
      const distVal  = parseDistM(p.dist);
      const arrowLen = (distVal / MAX_DIST) * MAX_ARROW;

      // Arrow (drawn before marker so marker sits on top)
      if (arrowLen > 0) {
        const ax2 = cx - arrowLen;
        const hs  = 4; // arrowhead half-size
        const arr = document.createElementNS(SVGNS, 'path');
        arr.setAttribute('d',
          `M${cx},${cy} L${ax2},${cy} M${ax2 + hs},${cy - hs * 0.65} L${ax2},${cy} L${ax2 + hs},${cy + hs * 0.65}`
        );
        arr.setAttribute('stroke', c);
        arr.setAttribute('stroke-width', 1.5);
        arr.setAttribute('fill', 'none');
        arr.setAttribute('stroke-opacity', '0.75');
        svg.appendChild(arr);
      }

      // Marker (on top of arrow)
      drawMarker(svg, p.marker, cx, cy, c, p.size);
      attachTooltip(svg.lastChild,
        `<div class="tt-row"><span>${p.cond}</span><span class="tt-v">R²=${p.x.toFixed(2)}, ΔSPL=${p.y.toFixed(2)}</span></div>
         <div class="tt-row"><span>disorient.</span><span class="tt-v">${p.dist}</span></div>`);

      // Condition label
      const lab = document.createElementNS(SVGNS, 'text');
      lab.setAttribute('fill', c);
      lab.setAttribute('font-weight', '600');
      lab.setAttribute('font-size', '12');
      lab.setAttribute('font-family', 'DM Sans, sans-serif');
      let labX, labY, anchor;
      if (p.labelPos === 'above') {
        labX = cx; labY = cy - p.size - 4; anchor = 'middle';
      } else {
        labX = cx; labY = cy + p.size + 12; anchor = 'middle';
      }
      lab.setAttribute('x', labX); lab.setAttribute('y', labY);
      lab.setAttribute('text-anchor', anchor);
      lab.textContent = p.cond;
      svg.appendChild(lab);

      // Distance label
      const distLab = document.createElementNS(SVGNS, 'text');
      distLab.setAttribute('font-size', '11');
      distLab.setAttribute('font-weight', '500');
      distLab.setAttribute('fill', c);
      distLab.setAttribute('font-family', 'DM Mono, monospace');
      if (p.distPos === 'left') {
        distLab.setAttribute('x', cx - p.size - 6);
        distLab.setAttribute('y', cy + 3.5);
        distLab.setAttribute('text-anchor', 'end');
      } else {
        distLab.setAttribute('x', cx + p.size + 5);
        distLab.setAttribute('y', cy + 3.5);
        distLab.setAttribute('text-anchor', 'start');
      }
      distLab.textContent = p.dist;
      svg.appendChild(distLab);
    });

    container.appendChild(svg);
  };

  function drawMarker(svg, kind, cx, cy, color, size) {
    const SVGNS = 'http://www.w3.org/2000/svg';
    const r = size;
    let el;
    if (kind === 'circle') {
      el = document.createElementNS(SVGNS, 'circle');
      el.setAttribute('cx', cx); el.setAttribute('cy', cy); el.setAttribute('r', r);
    } else if (kind === 'square') {
      el = document.createElementNS(SVGNS, 'rect');
      el.setAttribute('x', cx - r); el.setAttribute('y', cy - r);
      el.setAttribute('width', r * 2); el.setAttribute('height', r * 2);
    } else if (kind === 'diamond') {
      el = document.createElementNS(SVGNS, 'polygon');
      el.setAttribute('points', `${cx},${cy-r*1.2} ${cx+r*1.2},${cy} ${cx},${cy+r*1.2} ${cx-r*1.2},${cy}`);
    } else if (kind === 'triangleDown') {
      el = document.createElementNS(SVGNS, 'polygon');
      el.setAttribute('points', `${cx-r*1.2},${cy-r*0.9} ${cx+r*1.2},${cy-r*0.9} ${cx},${cy+r*1.2}`);
    } else {
      el = document.createElementNS(SVGNS, 'polygon');
      el.setAttribute('points', `${cx},${cy-r*1.2} ${cx+r*1.2},${cy+r*0.9} ${cx-r*1.2},${cy+r*0.9}`);
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
    el.addEventListener('mouseenter', () => { const t = getTip(); t.innerHTML = html; t.classList.add('is-visible'); });
    el.addEventListener('mousemove',  e  => { const t = getTip(); t.style.left = (e.clientX + 12) + 'px'; t.style.top = (e.clientY + 12) + 'px'; });
    el.addEventListener('mouseleave', () => { getTip().classList.remove('is-visible'); });
  }
})();
