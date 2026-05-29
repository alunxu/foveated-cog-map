/* ============================================================
   Figure 2a — Magnitude
   What does h2 encode?
   Linear vs MLP probe R² for GPS (position) and DtG (control),
   with MINE mutual-information bits on a secondary axis.
   ============================================================ */

(function () {
  const DATA = [
    // condition, GPS-linear R², GPS-MLP R² (≥ linear), DtG-linear R², MINE bits, error bars
    { cond: 'Blind',     gpsLin: 0.96, gpsMlp: 0.96, dtg: 0.93, mine: 6.05, errLin: 0.012 },
    { cond: 'Coarse',    gpsLin: 0.58, gpsMlp: 0.78, dtg: 0.97, mine: 4.62, errLin: 0.130 },
    { cond: 'Foveated',  gpsLin: 0.30, gpsMlp: 0.72, dtg: 0.95, mine: 4.45, errLin: 0.150 },
    { cond: 'Log-polar', gpsLin: -0.10, gpsMlp: 0.52, dtg: 0.95, mine: 4.70, errLin: 0.610 },
    { cond: 'Uniform',   gpsLin: -1.20, gpsMlp: 0.47, dtg: 0.92, mine: 4.58, errLin: 1.150 }
  ];

  const COLORS = {
    'Blind':     '#3a3a3a',
    'Coarse':    '#3b7dd8',
    'Foveated':  '#d92e2e',
    'Log-polar': '#9333ea',
    'Uniform':   '#22a155'
  };

  // softer shade for DtG control bars (~50 % alpha)
  function softColor(hex) {
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    return `rgba(${r},${g},${b},0.42)`;
  }

  window.renderFig2a = function (container) {
    if (typeof container === 'string') container = document.getElementById(container);
    container.innerHTML = '';

    const W = 640, H = 380;
    const M = { top: 22, right: 70, bottom: 56, left: 56 };
    const iw = W - M.left - M.right;
    const ih = H - M.top - M.bottom;

    // y-axes  ---------------------------------------------------
    const yMin = -2.5, yMax = 1.0;
    const yScale = v => M.top + (yMax - v) / (yMax - yMin) * ih;
    const mineMin = 0, mineMax = 7;
    const mineScale = v => M.top + (mineMax - v) / (mineMax - mineMin) * ih;

    // x bands ---------------------------------------------------
    const conds = DATA.map(d => d.cond);
    const bandW = iw / conds.length;
    const groupPad = 0.18;
    const subBarW = (bandW * (1 - groupPad)) / 2; // two sub-bars (h2 + DtG control)

    // SVG -------------------------------------------------------
    const SVGNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Magnitude: probe R² and MINE bits across encoder bandwidths');

    // hatched-pattern defs (one per condition)
    const defs = document.createElementNS(SVGNS, 'defs');
    Object.entries(COLORS).forEach(([cond, c]) => {
      const id = `hatch-2a-${cond.replace('-', '').toLowerCase()}`;
      const p = document.createElementNS(SVGNS, 'pattern');
      p.setAttribute('id', id);
      p.setAttribute('patternUnits', 'userSpaceOnUse');
      p.setAttribute('width', 6); p.setAttribute('height', 6);
      p.setAttribute('patternTransform', 'rotate(45)');
      const r = document.createElementNS(SVGNS, 'rect');
      r.setAttribute('width', 6); r.setAttribute('height', 6);
      r.setAttribute('fill', c); r.setAttribute('fill-opacity', '0.18');
      const l = document.createElementNS(SVGNS, 'line');
      l.setAttribute('x1', 0); l.setAttribute('y1', 0);
      l.setAttribute('x2', 0); l.setAttribute('y2', 6);
      l.setAttribute('stroke', c); l.setAttribute('stroke-width', 2.2);
      p.appendChild(r); p.appendChild(l);
      defs.appendChild(p);
    });
    svg.appendChild(defs);

    // grid
    const g = document.createElementNS(SVGNS, 'g');
    for (let v = -2; v <= 1; v += 1) {
      const y = yScale(v);
      const line = document.createElementNS(SVGNS, 'line');
      line.setAttribute('class', v === 0 ? 'axis-zero' : 'axis-grid');
      line.setAttribute('x1', M.left); line.setAttribute('x2', M.left + iw);
      line.setAttribute('y1', y); line.setAttribute('y2', y);
      g.appendChild(line);
    }
    svg.appendChild(g);

    // y-axis ticks
    for (let v = -2.5; v <= 1.0001; v += 0.5) {
      const y = yScale(v);
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('class', 'axis-num');
      t.setAttribute('x', M.left - 6); t.setAttribute('y', y + 3.5);
      t.setAttribute('text-anchor', 'end');
      t.textContent = v.toFixed(1).replace('-', '−');
      svg.appendChild(t);
    }

    // right axis (MINE)
    for (let v = 0; v <= 7.001; v += 1) {
      const y = mineScale(v);
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('class', 'axis-num');
      t.setAttribute('x', M.left + iw + 7); t.setAttribute('y', y + 3.5);
      t.setAttribute('fill', '#b07c1f');
      t.textContent = v;
      svg.appendChild(t);
    }

    // y-axis labels
    const yLabel = document.createElementNS(SVGNS, 'text');
    yLabel.setAttribute('class', 'axis-title');
    yLabel.setAttribute('transform', `translate(14, ${M.top + ih/2}) rotate(-90)`);
    yLabel.setAttribute('text-anchor', 'middle');
    yLabel.innerHTML = 'probe <tspan font-style="italic">R²</tspan>';
    svg.appendChild(yLabel);

    const yLabel2 = document.createElementNS(SVGNS, 'text');
    yLabel2.setAttribute('class', 'axis-title');
    yLabel2.setAttribute('transform', `translate(${W - 14}, ${M.top + ih/2}) rotate(90)`);
    yLabel2.setAttribute('text-anchor', 'middle');
    yLabel2.setAttribute('fill', '#b07c1f');
    yLabel2.textContent = 'MINE I(h₂; pos)  (bits)';
    svg.appendChild(yLabel2);

    // x-axis baseline (drawn at y=yMin) and tick labels at bottom
    const xAxis = document.createElementNS(SVGNS, 'line');
    xAxis.setAttribute('class', 'axis-line');
    xAxis.setAttribute('x1', M.left); xAxis.setAttribute('x2', M.left + iw);
    xAxis.setAttribute('y1', M.top + ih); xAxis.setAttribute('y2', M.top + ih);
    svg.appendChild(xAxis);

    const xTitle = document.createElementNS(SVGNS, 'text');
    xTitle.setAttribute('class', 'axis-title');
    xTitle.setAttribute('x', M.left + iw / 2);
    xTitle.setAttribute('y', H - 8);
    xTitle.setAttribute('text-anchor', 'middle');
    xTitle.textContent = 'encoder bandwidth';
    svg.appendChild(xTitle);

    // bars + ticks + MINE diamonds
    DATA.forEach((d, i) => {
      const cx = M.left + bandW * (i + 0.5);
      const xH2  = cx - subBarW - 1;          // h2 (GPS) bar
      const xDtG = cx + 1;                    // DtG control bar

      const c = COLORS[d.cond];
      const hatchId = `url(#hatch-2a-${d.cond.replace('-', '').toLowerCase()})`;
      const zeroY = yScale(0);

      // ------- GPS linear bar (filled solid) -------
      const linTopV = Math.min(d.gpsLin, 0) <= 0 ? 0 : d.gpsLin;
      // We split into "above zero" and "below zero" cleanly:
      const yLinTop = yScale(Math.max(d.gpsLin, 0));
      const yLinBot = yScale(Math.min(d.gpsLin, 0));
      const linRect = document.createElementNS(SVGNS, 'rect');
      linRect.setAttribute('class', 'bar');
      linRect.setAttribute('x', xH2);
      linRect.setAttribute('y', yLinTop);
      linRect.setAttribute('width', subBarW);
      linRect.setAttribute('height', Math.max(0, yLinBot - yLinTop));
      linRect.setAttribute('fill', c);
      linRect.setAttribute('fill-opacity', '0.92');
      linRect.setAttribute('rx', 1.5);
      attachTooltip(linRect, () => `<div class="tt-row"><span>${d.cond} · GPS linear</span><span class="tt-v">${d.gpsLin.toFixed(2)}</span></div>`);
      svg.appendChild(linRect);

      // ------- MLP-2 increment (hatched, drawn from linear-top up to mlp-top) -------
      if (d.gpsMlp > d.gpsLin) {
        const yMlpTop = yScale(d.gpsMlp);
        const yMlpBot = yScale(Math.max(d.gpsLin, 0));
        // hatched body
        const mlpRect = document.createElementNS(SVGNS, 'rect');
        mlpRect.setAttribute('class', 'bar');
        mlpRect.setAttribute('x', xH2);
        mlpRect.setAttribute('y', yMlpTop);
        mlpRect.setAttribute('width', subBarW);
        mlpRect.setAttribute('height', Math.max(0, yMlpBot - yMlpTop));
        mlpRect.setAttribute('fill', hatchId);
        mlpRect.setAttribute('stroke', c);
        mlpRect.setAttribute('stroke-width', 1);
        mlpRect.setAttribute('stroke-opacity', '0.7');
        mlpRect.setAttribute('rx', 1.5);
        attachTooltip(mlpRect, () => `<div class="tt-row"><span>${d.cond} · GPS MLP-2</span><span class="tt-v">${d.gpsMlp.toFixed(2)}</span></div>`);
        svg.appendChild(mlpRect);
      }

      // error bar on the GPS linear value
      if (d.errLin > 0) {
        const ex = xH2 + subBarW / 2;
        const eTop = yScale(d.gpsLin + d.errLin);
        const eBot = yScale(d.gpsLin - d.errLin);
        const eLine = document.createElementNS(SVGNS, 'line');
        eLine.setAttribute('class', 'errbar');
        eLine.setAttribute('x1', ex); eLine.setAttribute('x2', ex);
        eLine.setAttribute('y1', eTop); eLine.setAttribute('y2', eBot);
        eLine.setAttribute('stroke', '#222');
        svg.appendChild(eLine);
        [eTop, eBot].forEach(y => {
          const cap = document.createElementNS(SVGNS, 'line');
          cap.setAttribute('class', 'errbar');
          cap.setAttribute('x1', ex - 4); cap.setAttribute('x2', ex + 4);
          cap.setAttribute('y1', y); cap.setAttribute('y2', y);
          cap.setAttribute('stroke', '#222');
          svg.appendChild(cap);
        });
      }

      // ------- DtG control bar (lighter shade) -------
      const yDtgTop = yScale(Math.max(d.dtg, 0));
      const yDtgBot = yScale(Math.min(d.dtg, 0));
      const dtgRect = document.createElementNS(SVGNS, 'rect');
      dtgRect.setAttribute('class', 'bar');
      dtgRect.setAttribute('x', xDtG);
      dtgRect.setAttribute('y', yDtgTop);
      dtgRect.setAttribute('width', subBarW);
      dtgRect.setAttribute('height', Math.max(0, yDtgBot - yDtgTop));
      dtgRect.setAttribute('fill', softColor(c));
      dtgRect.setAttribute('stroke', c);
      dtgRect.setAttribute('stroke-opacity', '0.35');
      dtgRect.setAttribute('rx', 1.5);
      attachTooltip(dtgRect, () => `<div class="tt-row"><span>${d.cond} · DtG (control)</span><span class="tt-v">${d.dtg.toFixed(2)}</span></div>`);
      svg.appendChild(dtgRect);

      // MINE diamond
      const my = mineScale(d.mine);
      const mineSize = 5;
      const dmd = document.createElementNS(SVGNS, 'polygon');
      const px = cx;
      dmd.setAttribute('points', `${px},${my-mineSize} ${px+mineSize},${my} ${px},${my+mineSize} ${px-mineSize},${my}`);
      dmd.setAttribute('fill', '#e8b04a');
      dmd.setAttribute('stroke', '#7a5410');
      dmd.setAttribute('stroke-width', 1);
      dmd.setAttribute('class', 'dot');
      attachTooltip(dmd, () => `<div class="tt-row"><span>${d.cond} · MINE</span><span class="tt-v">${d.mine.toFixed(2)} bits</span></div>`);
      svg.appendChild(dmd);

      // x-tick label
      const xt = document.createElementNS(SVGNS, 'text');
      xt.setAttribute('class', 'axis-label');
      xt.setAttribute('x', cx);
      xt.setAttribute('y', M.top + ih + 16);
      xt.setAttribute('text-anchor', 'middle');
      xt.setAttribute('fill', c);
      xt.setAttribute('font-weight', '600');
      xt.textContent = d.cond.toLowerCase();
      svg.appendChild(xt);
    });

    container.appendChild(svg);

    // legend
    const legend = document.createElement('div');
    legend.className = 'chiprow';
    legend.innerHTML = `
      <span class="chip"><span class="chip-swatch" style="background:#333;border-radius:3px"></span>GPS linear R²</span>
      <span class="chip"><span class="chip-swatch" style="background:repeating-linear-gradient(45deg,#333 0 2px,transparent 2px 5px)"></span>GPS MLP-2 (above linear)</span>
      <span class="chip"><span class="chip-swatch" style="background:rgba(60,60,60,.42);border:1px solid rgba(60,60,60,.5)"></span>DtG control</span>
      <span class="chip"><span class="chip-swatch" style="background:#e8b04a;transform:rotate(45deg);border-radius:0"></span>MINE bits (right axis)</span>
    `;
    container.appendChild(legend);
  };

  // ---------------- shared tooltip (single instance) ----------------
  function getTip() {
    let t = document.getElementById('shared-tooltip');
    if (!t) {
      t = document.createElement('div');
      t.id = 'shared-tooltip';
      t.className = 'tt-tooltip';
      document.body.appendChild(t);
    }
    return t;
  }
  function attachTooltip(el, html) {
    el.addEventListener('mouseenter', e => { const t = getTip(); t.innerHTML = html(); t.classList.add('is-visible'); });
    el.addEventListener('mousemove',  e => { const t = getTip(); t.style.left = (e.clientX + 12) + 'px'; t.style.top = (e.clientY + 12) + 'px'; });
    el.addEventListener('mouseleave', () => { getTip().classList.remove('is-visible'); });
  }
})();
