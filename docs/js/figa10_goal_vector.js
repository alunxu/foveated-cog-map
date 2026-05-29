/* ============================================================
   Figure A10 — Goal vector decoding
   Ridge probe R² for {goal distance, goal direction, goal vector (2D)}
   across five conditions. Off-scale outliers shown as red labels.
   ============================================================ */

(function () {
  const DATA = [
    { cond: 'Blind',     dist: 0.90, dir: -0.30, vec: -0.32 },
    { cond: 'Coarse',    dist: 0.97, dir: -0.30, vec: -0.15 },
    { cond: 'Log-polar', dist: 0.97, dir: -0.35, vec: -0.55, vecOutlier: '−0.9' },
    { cond: 'Foveated',  dist: 0.97, dir: -0.15, vec: -0.14 },
    { cond: 'Uniform',   dist: 0.27, dir: -0.45, vec: -0.50, dirOutlier: '−3.4', vecOutlier: '−22.1' }
  ];
  const COLORS = {
    'Blind':     '#3a3a3a',
    'Coarse':    '#3b7dd8',
    'Foveated':  '#d92e2e',
    'Log-polar': '#9333ea',
    'Uniform':   '#22a155'
  };

  window.renderFigA10 = function (container) {
    if (typeof container === 'string') container = document.getElementById(container);
    container.innerHTML = '';

    const W = 640, H = 380;
    const M = { top: 32, right: 24, bottom: 56, left: 58 };
    const iw = W - M.left - M.right;
    const ih = H - M.top - M.bottom;

    const yMin = -0.55, yMax = 1.05;
    const ys = v => M.top + (yMax - v) / (yMax - yMin) * ih;

    const bandW = iw / DATA.length;
    const groupPad = 0.18;
    const subBarW = (bandW * (1 - groupPad)) / 3; // 3 metrics

    const SVGNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    // patterns for hatched direction + cross-hatched vector
    const defs = document.createElementNS(SVGNS, 'defs');
    Object.entries(COLORS).forEach(([cond, c]) => {
      const idDir = `hatch-a10-dir-${cond.replace('-', '').toLowerCase()}`;
      const pD = document.createElementNS(SVGNS, 'pattern');
      pD.setAttribute('id', idDir);
      pD.setAttribute('patternUnits', 'userSpaceOnUse');
      pD.setAttribute('width', 6); pD.setAttribute('height', 6);
      pD.setAttribute('patternTransform', 'rotate(45)');
      const rD = document.createElementNS(SVGNS, 'rect');
      rD.setAttribute('width', 6); rD.setAttribute('height', 6);
      rD.setAttribute('fill', c); rD.setAttribute('fill-opacity', '0.22');
      const lD = document.createElementNS(SVGNS, 'line');
      lD.setAttribute('x1', 0); lD.setAttribute('y1', 0);
      lD.setAttribute('x2', 0); lD.setAttribute('y2', 6);
      lD.setAttribute('stroke', c); lD.setAttribute('stroke-width', 1.8);
      pD.appendChild(rD); pD.appendChild(lD);
      defs.appendChild(pD);

      // cross hatch (denser) for "vector (2D)"
      const idVec = `hatch-a10-vec-${cond.replace('-', '').toLowerCase()}`;
      const pV = document.createElementNS(SVGNS, 'pattern');
      pV.setAttribute('id', idVec);
      pV.setAttribute('patternUnits', 'userSpaceOnUse');
      pV.setAttribute('width', 6); pV.setAttribute('height', 6);
      pV.setAttribute('patternTransform', 'rotate(45)');
      const rV = document.createElementNS(SVGNS, 'rect');
      rV.setAttribute('width', 6); rV.setAttribute('height', 6);
      rV.setAttribute('fill', c); rV.setAttribute('fill-opacity', '0.32');
      const l1 = document.createElementNS(SVGNS, 'line');
      l1.setAttribute('x1', 0); l1.setAttribute('y1', 0);
      l1.setAttribute('x2', 0); l1.setAttribute('y2', 6);
      l1.setAttribute('stroke', c); l1.setAttribute('stroke-width', 1.4);
      const l2 = document.createElementNS(SVGNS, 'line');
      l2.setAttribute('x1', 3); l2.setAttribute('y1', 0);
      l2.setAttribute('x2', 3); l2.setAttribute('y2', 6);
      l2.setAttribute('stroke', c); l2.setAttribute('stroke-width', 1.4);
      pV.appendChild(rV); pV.appendChild(l1); pV.appendChild(l2);
      defs.appendChild(pV);
    });
    svg.appendChild(defs);

    // gridlines
    [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(v => {
      const y = ys(v);
      const line = document.createElementNS(SVGNS, 'line');
      line.setAttribute('class', v === 0 ? 'axis-zero' : 'axis-grid');
      line.setAttribute('x1', M.left); line.setAttribute('x2', M.left + iw);
      line.setAttribute('y1', y); line.setAttribute('y2', y);
      svg.appendChild(line);
    });

    // y ticks
    [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(v => {
      const y = ys(v);
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('class', 'axis-num');
      t.setAttribute('x', M.left - 6); t.setAttribute('y', y + 3.5);
      t.setAttribute('text-anchor', 'end');
      t.textContent = v.toFixed(1).replace('-', '−');
      svg.appendChild(t);
    });

    // y title
    const yT = document.createElementNS(SVGNS, 'text');
    yT.setAttribute('class', 'axis-title');
    yT.setAttribute('transform', `translate(14, ${M.top + ih / 2}) rotate(-90)`);
    yT.setAttribute('text-anchor', 'middle');
    yT.innerHTML = 'Ridge probe <tspan font-style="italic">R²</tspan>';
    svg.appendChild(yT);

    const baseY = ys(0);

    // bars per condition × 3 metrics
    DATA.forEach((d, i) => {
      const cx = M.left + bandW * (i + 0.5);
      const c = COLORS[d.cond];
      const hatchDir = `url(#hatch-a10-dir-${d.cond.replace('-', '').toLowerCase()})`;
      const hatchVec = `url(#hatch-a10-vec-${d.cond.replace('-', '').toLowerCase()})`;

      // x for the 3 sub-bars (centered around cx)
      const x0 = cx - 1.5 * subBarW;       // dist
      const x1 = cx - 0.5 * subBarW;       // dir
      const x2 = cx + 0.5 * subBarW;       // vec

      // helper to draw a single (possibly negative) bar
      const drawBar = (xPos, val, fill, strokeColor, outlierLabel, metricName) => {
        const yTop = ys(Math.max(val, 0));
        const yBot = ys(Math.min(val, 0));
        const r = document.createElementNS(SVGNS, 'rect');
        r.setAttribute('class', 'bar');
        r.setAttribute('x', xPos); r.setAttribute('y', yTop);
        r.setAttribute('width', subBarW - 1.5); r.setAttribute('height', Math.max(0, yBot - yTop));
        r.setAttribute('fill', fill);
        if (strokeColor) {
          r.setAttribute('stroke', strokeColor);
          r.setAttribute('stroke-width', 1);
          r.setAttribute('stroke-opacity', '0.55');
        }
        r.setAttribute('rx', 1.5);
        attachTooltip(r, `<div class="tt-row"><span>${d.cond} · ${metricName}</span><span class="tt-v">${outlierLabel || formatVal(val)}</span></div>`);
        svg.appendChild(r);
      };

      drawBar(x0, d.dist, c, null, null, 'goal distance');
      drawBar(x1, d.dir, hatchDir, c, d.dirOutlier, 'goal direction');
      drawBar(x2, d.vec, hatchVec, c, d.vecOutlier, 'goal vector (2D)');

      // condition x-label
      const xt = document.createElementNS(SVGNS, 'text');
      xt.setAttribute('class', 'axis-label');
      xt.setAttribute('x', cx); xt.setAttribute('y', M.top + ih + 18);
      xt.setAttribute('text-anchor', 'middle');
      xt.setAttribute('fill', c);
      xt.setAttribute('font-weight', '600');
      xt.textContent = d.cond;
      svg.appendChild(xt);

      // outlier labels (red, below the truncated bar) — direction
      if (d.dirOutlier) {
        const out = document.createElementNS(SVGNS, 'text');
        out.setAttribute('font-size', '10.5');
        out.setAttribute('font-weight', '600');
        out.setAttribute('fill', '#c0292e');
        out.setAttribute('font-family', 'DM Mono, monospace');
        out.setAttribute('text-anchor', 'middle');
        out.setAttribute('x', x1 + (subBarW - 1.5) / 2);
        out.setAttribute('y', M.top + ih + 32);
        out.textContent = d.dirOutlier;
        svg.appendChild(out);
      }
      if (d.vecOutlier) {
        const out = document.createElementNS(SVGNS, 'text');
        out.setAttribute('font-size', '10.5');
        out.setAttribute('font-weight', '600');
        out.setAttribute('fill', '#c0292e');
        out.setAttribute('font-family', 'DM Mono, monospace');
        out.setAttribute('text-anchor', 'middle');
        out.setAttribute('x', x2 + (subBarW - 1.5) / 2);
        out.setAttribute('y', M.top + ih + 32);
        out.textContent = d.vecOutlier;
        svg.appendChild(out);
      }
    });

    // zero baseline emphasized
    const zL = document.createElementNS(SVGNS, 'line');
    zL.setAttribute('class', 'axis-line');
    zL.setAttribute('x1', M.left); zL.setAttribute('x2', M.left + iw);
    zL.setAttribute('y1', baseY); zL.setAttribute('y2', baseY);
    zL.setAttribute('stroke-width', '1.2');
    svg.appendChild(zL);

    // y-axis
    const yax = document.createElementNS(SVGNS, 'line');
    yax.setAttribute('class', 'axis-line');
    yax.setAttribute('x1', M.left); yax.setAttribute('x2', M.left);
    yax.setAttribute('y1', M.top); yax.setAttribute('y2', M.top + ih);
    svg.appendChild(yax);

    container.appendChild(svg);

    // legend
    const legend = document.createElement('div');
    legend.className = 'chiprow';
    legend.innerHTML = `
      <span class="chip"><span class="chip-swatch" style="background:#444;border-radius:3px"></span>Goal distance</span>
      <span class="chip"><span class="chip-swatch" style="background:repeating-linear-gradient(45deg,#444 0 2px,transparent 2px 5px)"></span>Goal direction</span>
      <span class="chip"><span class="chip-swatch" style="background:repeating-linear-gradient(45deg,#444 0 2px,transparent 2px 4px)"></span>Goal vector (2D)</span>
      <span class="chip"><span style="color:#c0292e;font-family:'DM Mono',monospace;font-weight:600;font-size:11px">−x.x</span>&nbsp;off-scale value</span>
    `;
    container.appendChild(legend);
  };

  function formatVal(v) {
    if (v === null || v === undefined) return 'n/a';
    const s = v >= 0 ? '+' : '−';
    return s + Math.abs(v).toFixed(2);
  }

  function getTip() {
    let t = document.getElementById('shared-tooltip');
    if (!t) { t = document.createElement('div'); t.id = 'shared-tooltip'; t.className = 'tt-tooltip'; document.body.appendChild(t); }
    return t;
  }
  function attachTooltip(el, html) {
    el.addEventListener('mouseenter', () => { const t = getTip(); t.innerHTML = html; t.classList.add('is-visible'); });
    el.addEventListener('mousemove',  e => { const t = getTip(); t.style.left = (e.clientX + 12) + 'px'; t.style.top = (e.clientY + 12) + 'px'; });
    el.addEventListener('mouseleave', () => { getTip().classList.remove('is-visible'); });
  }
})();
