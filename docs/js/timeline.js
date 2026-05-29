(function () {
  'use strict';

  var SECTION_IDS = [
    'abstract', 'motivation', 'hypothesis', 'related', 'methods',
    'result1', 'result2', 'result3', 'result4', 'result5',
    'discussion', 'conclusion', 'future', 'references', 'code-data'
  ];

  var SECTION_LABELS = [
    'Abstract', 'Motivation', 'Hypothesis', 'Related Work', 'Methods',
    'Result 1', 'Result 2', 'Result 3', 'Result 4', 'Result 5',
    'Discussion', 'Conclusion', 'Future Work', 'References', 'Code & Data'
  ];

  var dots = [];
  var sections = [];
  var rail = null;
  var progressDot = null;
  var rafPending = false;
  var isVisible = false;
  var idleTimer = null;
  var idleHidden = false;

  var IDLE_DELAY = 5000; // ms

  function showRailIfVisible() {
    if (!isVisible || !rail) return;
    idleHidden = false;
    rail.classList.remove('timeline-idle-hidden');
  }

  function startIdleTimer() {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(function () {
      if (isVisible && rail) {
        idleHidden = true;
        rail.classList.add('timeline-idle-hidden');
      }
    }, IDLE_DELAY);
  }

  function positionRail() {
    var col = document.querySelector('.content-column');
    if (!col || !rail) return;

    var rect = col.getBoundingClientRect();
    var contentLeft = rect.left + parseFloat(getComputedStyle(col).paddingLeft);
    document.documentElement.style.setProperty('--content-left-offset', contentLeft + 'px');

    rail.style.left = '20px';
  }

  function setDotPositions() {
    var vh = window.innerHeight;
    var n = dots.length;
    var TOP_Y = vh * 0.14;
    var BOT_Y = vh * 0.86;
    for (var i = 0; i < n; i++) {
      var y = n === 1 ? vh * 0.5 : TOP_Y + (BOT_Y - TOP_Y) * i / (n - 1);
      dots[i].style.top = Math.round(y) + 'px';
    }
    var line = rail.querySelector('.timeline-line');
    if (line) {
      line.style.top = Math.round(TOP_Y) + 'px';
      line.style.bottom = Math.round(vh - BOT_Y) + 'px';
      line.style.height = 'auto';
    }
  }

  function getActiveSectionIndex() {
    var vh = window.innerHeight;
    var target = vh * 0.40;
    var best = -1;
    var bestDist = Infinity;

    for (var i = 0; i < sections.length; i++) {
      var s = sections[i];
      if (!s) continue;
      var r = s.getBoundingClientRect();
      if (r.bottom < 0 || r.top > vh) continue;
      var h2 = s.querySelector('h2');
      var anchor = h2 ? h2.getBoundingClientRect().top : r.top;
      var dist = Math.abs(anchor - target);
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }

    if (best === -1) {
      for (var j = sections.length - 1; j >= 0; j--) {
        var sec = sections[j];
        if (!sec) continue;
        if (sec.getBoundingClientRect().top < vh * 0.5) {
          best = j;
          break;
        }
      }
    }

    return best;
  }

  function updateProgressDot(activeIdx) {
    if (!progressDot || !isVisible || activeIdx < 0 || activeIdx >= sections.length - 1) {
      if (progressDot) progressDot.style.opacity = '0';
      return;
    }

    var s0 = sections[activeIdx];
    var s1 = sections[activeIdx + 1];
    if (!s0 || !s1) { progressDot.style.opacity = '0'; return; }

    var vh = window.innerHeight;
    var h0 = s0.querySelector('h2');
    var h1 = s1.querySelector('h2');
    var y0 = h0 ? h0.getBoundingClientRect().top : s0.getBoundingClientRect().top;
    var y1 = h1 ? h1.getBoundingClientRect().top : s1.getBoundingClientRect().top;

    if (y1 === y0) { progressDot.style.opacity = '0'; return; }

    var target = vh * 0.40;
    var progress = (target - y0) / (y1 - y0);
    progress = Math.max(0, Math.min(1, progress));

    if (progress < 0.05 || progress > 0.95) {
      progressDot.style.opacity = '0';
      return;
    }

    var d0top = parseFloat(dots[activeIdx].style.top);
    var d1top = parseFloat(dots[activeIdx + 1].style.top);
    progressDot.style.top = Math.round(d0top + progress * (d1top - d0top)) + 'px';
    progressDot.style.opacity = '1';
  }

  function updateDots() {
    if (!isVisible) return;
    var active = getActiveSectionIndex();
    for (var i = 0; i < dots.length; i++) {
      dots[i].classList.remove('dot-active', 'dot-past', 'dot-future');
      if (i < active) {
        dots[i].classList.add('dot-past');
      } else if (i === active) {
        dots[i].classList.add('dot-active');
      } else {
        dots[i].classList.add('dot-future');
      }
    }
    updateProgressDot(active);
  }

  function scheduleUpdate() {
    if (!rafPending) {
      rafPending = true;
      requestAnimationFrame(function () {
        rafPending = false;
        updateDots();
      });
    }
  }

  function init() {
    rail = document.getElementById('timeline-rail');
    if (!rail) return;

    dots = Array.prototype.slice.call(document.querySelectorAll('.timeline-dot'));
    sections = SECTION_IDS.map(function (id) { return document.getElementById(id); });

    // Add section labels inside each dot
    dots.forEach(function (dot, i) {
      var label = document.createElement('span');
      label.className = 'timeline-label';
      label.textContent = SECTION_LABELS[i] || '';
      dot.appendChild(label);
    });

    // Create the scroll progress dot
    progressDot = document.createElement('div');
    progressDot.className = 'timeline-progress-dot';
    rail.appendChild(progressDot);

    // Wire up click navigation
    dots.forEach(function (dot, i) {
      var sectionId = SECTION_IDS[i];
      dot.style.pointerEvents = 'all';
      dot.style.cursor = 'pointer';
      dot.addEventListener('click', function () {
        var target = document.getElementById(sectionId);
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });

    positionRail();
    setDotPositions();

    window.addEventListener('scroll', function () {
      showRailIfVisible();
      startIdleTimer();
      scheduleUpdate();
    }, { passive: true });
    window.addEventListener('resize', function () {
      positionRail();
      setDotPositions();
      updateDots();
    });

    var motivationEl = document.getElementById('motivation');
    if (motivationEl && 'IntersectionObserver' in window) {
      var motObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function () {
          var rect = motivationEl.getBoundingClientRect();
          if (rect.top < window.innerHeight * 0.95) {
            isVisible = true;
            rail.classList.remove('timeline-hidden');
            updateDots();
            startIdleTimer();
          } else {
            isVisible = false;
            rail.classList.add('timeline-hidden');
          }
        });
      }, { threshold: [0, 0.01, 0.1] });
      motObserver.observe(motivationEl);
    }

    var endEl = document.getElementById('end');
    var endImageBox = document.querySelector('.end-image-box');
    if (endEl && 'IntersectionObserver' in window) {
      var endObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            rail.classList.add('timeline-hidden');
            if (endImageBox) endImageBox.classList.add('end-visible');
          } else if (isVisible) {
            rail.classList.remove('timeline-hidden');
          }
        });
      }, { threshold: 0.05 });
      endObserver.observe(endEl);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
