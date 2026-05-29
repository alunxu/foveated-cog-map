/* ============================================================
   Figure 5c — Readable axis is one window, not the substrate
   probe R² drop after β-scrubbing (zeroing the readable axis)
   per condition × {linear probe, MLP probe}
   ============================================================ */

(function () {
  const DATA = [
    { cond: 'Blind',     lin: +0.07, mlp: +0.01 },
    { cond: 'Coarse',    lin: +0.26, mlp: +0.08 },
    { cond: 'Foveated',  lin: +0.32, mlp: +0.03 },
    { cond: 'Log-polar', lin: +0.37, mlp: -0.09 },
    { cond: 'Uniform',   lin: -0.16, mlp: null   /* n/a */ }
  ];
  const COLORS = {
    'Blind':     '#3a3a3a',
    'Coarse':    '#3b7dd8',
    'Foveated':  '#d92e2e',
    'Log-polar': '#9333ea',
    'Uniform':   '#22a155'
  };

  window.renderFig5c = function (container) {
    if (typeof container === 'string') container = document.getElementById(container);
    container.innerHTML = '';

    const W = 560, H = 380;
    const M = { top: 32, right: 22, bottom: 56, left: 60 };
    const iw = W - M.left - M.right;
    const ih = H - M.top - M.bottom;

    const yMin = -0.20, yMax = 0.55;
    const ys = v => M.top + (yMax - v) / (yMax - yMin) * ih;

    const bandW = iw / DATA.length;
    const groupPad = 0.20;
    const subBarW = (bandW * (1 - groupPad)) / 2;

    const SVGNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    // hatch patterns
    const defs = document.createElementNS(SVGNS, 'defs');
    Object.entries(COLORS).forEach(([cond, c]) => {
      const id = `hatch-5c-${cond.replace('-', '').toLowerCase()}`;
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

    // gridlines
    [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5].forEach(v => {
      const y = ys(v);
      const line = document.createElementNS(SVGNS, 'line');
      line.setAttribute('class', v === 0 ? 'axis-zero' : 'axis-grid');
      line.setAttribute('x1', M.left); line.setAttribute('x2', M.left + iw);
      line.setAttribute('y1', y); line.setAttribute('y2', y);
      svg.appendChild(line);
    });

    // y ticks
    [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5].forEach(v => {
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
    yT.innerHTML = 'probe <tspan font-style="italic">R²</tspan> drop after β-scrubbing';
    svg.appendChild(yT);

    // baseline
    const baseY = ys(0);

    // bars
    DATA.forEach((d, i) => {
      const cx = M.left + bandW * (i + 0.5);
      const xLin = cx - subBarW - 1;
      const xMlp = cx + 1;
      const c = COLORS[d.cond];
      const hatchId = `url(#hatch-5c-${d.cond.replace('-', '').toLowerCase()})`;

      // linear bar (solid)
      const yLinTop = ys(Math.max(d.lin, 0));
      const yLinBot = ys(Math.min(d.lin, 0));
      const linRect = document.createElementNS(SVGNS, 'rect');
      linRect.setAttribute('class', 'bar');
      linRect.setAttribute('x', xLin);
      linRect.setAttribute('y', yLinTop);
      linRect.setAttribute('width', subBarW);
      linRect.setAttribute('height', Math.max(0, yLinBot - yLinTop));
      linRect.setAttribute('fill', c);
      linRect.setAttribute('fill-opacity', '0.92');
      linRect.setAttribute('rx', 1.5);
      attachTooltip(linRect, `<div class="tt-row"><span>${d.cond} · linear</span><span class="tt-v">${formatVal(d.lin)}</span></div>`);
      svg.appendChild(linRect);

      // top label for linear bar
      const linLabel = document.createElementNS(SVGNS, 'text');
      linLabel.setAttribute('font-size', '11');
      linLabel.setAttribute('font-weight', '600');
      linLabel.setAttribute('font-family', 'DM Mono, monospace');
      linLabel.setAttribute('text-anchor', 'middle');
      linLabel.setAttribute('fill', c);
      linLabel.setAttribute('x', xLin + subBarW / 2);
      linLabel.setAttribute('y', d.lin >= 0 ? yLinTop - 5 : yLinBot + 13);
      linLabel.textContent = formatVal(d.lin);
      svg.appendChild(linLabel);

      // MLP bar (hatched) - if not n/a
      if (d.mlp !== null) {
        const yMlpTop = ys(Math.max(d.mlp, 0));
        const yMlpBot = ys(Math.min(d.mlp, 0));
        const mlpRect = document.createElementNS(SVGNS, 'rect');
        mlpRect.setAttribute('class', 'bar');
        mlpRect.setAttribute('x', xMlp);
        mlpRect.setAttribute('y', yMlpTop);
        mlpRect.setAttribute('width', subBarW);
        mlpRect.setAttribute('height', Math.max(0, yMlpBot - yMlpTop));
        mlpRect.setAttribute('fill', hatchId);
        mlpRect.setAttribute('stroke', c);
        mlpRect.setAttribute('stroke-width', 1);
        mlpRect.setAttribute('stroke-opacity', '0.6');
        mlpRect.setAttribute('rx', 1.5);
        attachTooltip(mlpRect, `<div class="tt-row"><span>${d.cond} · MLP</span><span class="tt-v">${formatVal(d.mlp)}</span></div>`);
        svg.appendChild(mlpRect);

        const mlpLabel = document.createElementNS(SVGNS, 'text');
        mlpLabel.setAttribute('font-size', '11');
        mlpLabel.setAttribute('font-weight', '500');
        mlpLabel.setAttribute('font-family', 'DM Mono, monospace');
        mlpLabel.setAttribute('text-anchor', 'middle');
        mlpLabel.setAttribute('fill', c);
        mlpLabel.setAttribute('fill-opacity', '0.85');
        mlpLabel.setAttribute('x', xMlp + subBarW / 2);
        mlpLabel.setAttribute('y', d.mlp >= 0 ? yMlpTop - 5 : yMlpBot + 13);
        mlpLabel.textContent = formatVal(d.mlp);
        svg.appendChild(mlpLabel);
      } else {
        // n/a label where MLP bar would go
        const na = document.createElementNS(SVGNS, 'text');
        na.setAttribute('font-size', '11');
        na.setAttribute('font-style', 'italic');
        na.setAttribute('fill', c);
        na.setAttribute('fill-opacity', '0.7');
        na.setAttribute('text-anchor', 'middle');
        na.setAttribute('x', xMlp + subBarW / 2);
        na.setAttribute('y', baseY - 5);
        na.textContent = 'n/a';
        svg.appendChild(na);
      }

      // x-axis condition label
      const xt = document.createElementNS(SVGNS, 'text');
      xt.setAttribute('class', 'axis-label');
      xt.setAttribute('x', cx); xt.setAttribute('y', M.top + ih + 18);
      xt.setAttribute('text-anchor', 'middle');
      xt.setAttribute('fill', c);
      xt.setAttribute('font-weight', '600');
      xt.textContent = d.cond;
      svg.appendChild(xt);
    });

    // baseline on top of bars (zero line emphasized)
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
      <span class="chip"><span class="chip-swatch" style="background:#333;border-radius:3px"></span>linear probe drop</span>
      <span class="chip"><span class="chip-swatch" style="background:repeating-linear-gradient(45deg,#333 0 2px,transparent 2px 5px)"></span>MLP probe drop</span>
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
