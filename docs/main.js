(function () {
  'use strict';
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));

  const SEQ =
    'CGTAACGGTATCAGCTTGCAAGGTCAAC' +
    'TTGACA' +
    'GCATTCGATACGGTC' +
    'TATAAT' +
    'GCTAC' +
    'A' +
    'CGTTAGCCTGATTCACGGTA';
  const BOXES = [
    { start: 28, end: 33, kind: 'box', label: '−35', base: 'T' },
    { start: 49, end: 54, kind: 'box', label: '−10', base: 'A' },
  ];
  const TSS = 60;

  function renderTrack() {
    const seqEl = $('#trackSeq');
    const annoEl = $('#trackAnno');
    if (!seqEl) return;

    const frag = document.createDocumentFragment();
    for (let i = 0; i < SEQ.length; i++) {
      const b = SEQ[i];
      const span = document.createElement('span');
      span.className = 'base';
      span.dataset.b = b;
      span.textContent = b;
      const inBox = BOXES.some((x) => i >= x.start && i <= x.end);
      if (inBox) span.classList.add('box');
      if (i === TSS) span.classList.add('tss');
      frag.appendChild(span);
    }
    seqEl.appendChild(frag);
    const bases = $$('.base', seqEl);
    const cellW = () => (bases[1] ? bases[1].offsetLeft - bases[0].offsetLeft : 18);

    const mkAnno = (leftPx, widthPx, kind, label, colorVar, delay) => {
      const a = document.createElement('div');
      a.className = 'anno' + (kind === 'tss' ? ' anno--tss' : '');
      a.dataset.kind = kind === 'tss' ? 'tss' : 'box';
      a.style.left = leftPx + 'px';
      if (colorVar) a.style.setProperty('--c', colorVar);
      a.style.animationDelay = delay + 'ms';
      if (kind === 'tss') {
        a.innerHTML = '<span class="anno__tick"></span>+1&nbsp;TSS';
      } else {
        a.innerHTML = '<span class="anno__bracket" style="width:' + widthPx + 'px"></span>' + label + '&nbsp;box';
      }
      annoEl.appendChild(a);
    };

    const place = () => {
      annoEl.innerHTML = '';

      const seqLeft = seqEl.getBoundingClientRect().left;
      const rectOf = (i) => bases[i].getBoundingClientRect();
      const baseDelay = reduce ? 0 : 1400;
      BOXES.forEach((x, k) => {
        const r0 = rectOf(x.start), r1 = rectOf(x.end);
        const left = r0.left - seqLeft;
        const width = (r1.right - r0.left) - 2;
        const cvar = x.label === '−35' ? 'var(--base-g)' : 'var(--base-a)';
        mkAnno(left, width, 'box', x.label, cvar, baseDelay + k * 120);
      });
      const rt = rectOf(TSS);

      const tssLeft = (rt.left - seqLeft) + rt.width / 2;
      mkAnno(tssLeft, 0, 'tss', '', null, baseDelay + 260);
    };

    if (reduce) {
      bases.forEach((el) => el.classList.add('on'));
      place();
    } else {
      let i = 0;
      const step = () => {
        if (i < bases.length) {
          bases[i].classList.add('on');
          i++;

          if (i % 1 === 0) requestAnimationFrame(step);
        }
      };

      bases.forEach((el, k) => setTimeout(() => el.classList.add('on'), 1200 + k * 15));
      setTimeout(place, 1200 + bases.length * 15 + 40);
    }
    window.addEventListener('resize', debounce(place, 150));
  }

  function splitHeroTitle() {
    const h = $('#heroTitle');
    if (!h) return;
    const walk = (node) => {
      Array.from(node.childNodes).forEach((n) => {
        if (n.nodeType === 3) {
          const words = n.textContent.split(/(\s+)/);
          const f = document.createDocumentFragment();
          words.forEach((w) => {
            if (w.trim() === '') { f.appendChild(document.createTextNode(w)); return; }
            const s = document.createElement('span');
            s.className = 'word'; s.textContent = w;
            f.appendChild(s);
          });
          node.replaceChild(f, n);
        } else if (n.nodeType === 1) {
          n.classList.add('word');
        }
      });
    };
    walk(h);
    const words = $$('.word', h);
    words.forEach((w, k) => {
      if (reduce) { w.style.opacity = 1; w.style.transform = 'none'; return; }
      w.style.animation = 'rise 0.8s var(--ease-out) forwards';
      w.style.animationDelay = 0.35 + k * 0.06 + 's';
    });
  }

  function nav() {
    const mast = $('#masthead');
    const links = $$('.nav a');
    const map = links.map((a) => {
      const id = a.getAttribute('href').slice(1);
      return { a, el: document.getElementById(id) };
    }).filter((x) => x.el);

    const onScroll = () => {
      mast.classList.toggle('is-stuck', window.scrollY > 40);
      let current = null;
      const y = window.scrollY + window.innerHeight * 0.32;
      map.forEach((m) => { if (m.el.offsetTop <= y) current = m; });
      links.forEach((a) => a.classList.remove('is-active'));
      if (current) current.a.classList.add('is-active');
    };
    onScroll();
    window.addEventListener('scroll', throttle(onScroll, 100), { passive: true });
  }

  function mobileNav() {
    const btn = $('#navToggle');
    const drawer = $('#navDrawer');
    const mast = $('#masthead');
    if (!btn) return;
    const close = () => { drawer.classList.remove('open'); mast.classList.remove('nav-open'); btn.setAttribute('aria-expanded', 'false'); };
    btn.addEventListener('click', () => {
      const open = drawer.classList.toggle('open');
      mast.classList.toggle('nav-open', open);
      btn.setAttribute('aria-expanded', String(open));
    });
    $$('a', drawer).forEach((a) => a.addEventListener('click', close));
  }

  function reveals() {
    const items = $$('[data-reveal]');
    if (reduce || !('IntersectionObserver' in window)) {
      items.forEach((el) => el.classList.add('in'));
      return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
    items.forEach((el) => io.observe(el));
  }

  function figFilter() {
    const btns = $$('.filter');
    const cards = $$('#gallery .fig-card');
    btns.forEach((b) => b.addEventListener('click', () => {
      btns.forEach((x) => x.classList.remove('is-on'));
      b.classList.add('is-on');
      const f = b.dataset.filter;
      cards.forEach((c) => {
        const show = f === 'all' || c.dataset.act === f;
        c.style.display = show ? '' : 'none';
      });
    }));
  }

  function themeToggle() {
    const btn = $('#themeToggle');
    if (!btn) return;
    btn.addEventListener('click', () => {
      const cur = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
      const next = cur === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('dp2-theme', next); } catch (e) {}
    });
  }

  function lightbox() {
    const lb = $('#lightbox');
    if (!lb) return;
    const els = {
      img: $('#lbImg'), slug: $('#lbSlug'), tag: $('#lbTag'), act: $('#lbAct'),
      title: $('#lbTitle'), find: $('#lbFind'), scope: $('#lbScope'), scopeTxt: $('#lbScopeTxt'),
      count: $('#lbCount'),
    };
    const cards = $$('#gallery .fig-card');
    let order = [];
    let pos = 0;
    let lastFocus = null;

    const visible = () => cards.filter((c) => c.style.display !== 'none');
    const pick = (card, sel) => { const e = $(sel, card); return e ? e : null; };

    function fill(card) {
      els.img.src = card.dataset.full || '';
      const im = pick(card, 'img');
      els.img.alt = im ? im.getAttribute('alt') || '' : '';
      els.slug.textContent = (pick(card, '.fig-card__slug') || {}).textContent || '';
      els.tag.textContent = card.dataset.fig || '';
      els.tag.dataset.kind = card.dataset.kind || 'ed';
      els.act.textContent = (pick(card, '.act-tag') || {}).textContent || '';
      els.title.textContent = (pick(card, '.fig-card__title') || {}).textContent || '';

      const findEl = pick(card, '.fig-card__find');
      els.find.innerHTML = findEl ? findEl.innerHTML : '';
      if (card.dataset.scope) { els.scopeTxt.textContent = card.dataset.scope; els.scope.hidden = false; }
      else els.scope.hidden = true;
      els.count.textContent = (pos + 1) + ' / ' + order.length;
    }

    function open(card) {
      order = visible();
      pos = order.indexOf(card);
      if (pos < 0) { order = cards.slice(); pos = Math.max(0, order.indexOf(card)); }
      lastFocus = document.activeElement;
      fill(card);
      lb.hidden = false;
      lb.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
      const c = $('[data-close]', lb);
      if (c) c.focus();
    }
    function close() {
      lb.hidden = true;
      lb.setAttribute('aria-hidden', 'true');
      document.body.style.overflow = '';
      els.img.removeAttribute('src');
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    }
    function step(d) {
      if (order.length < 2) return;
      pos = (pos + d + order.length) % order.length;
      fill(order[pos]);
    }

    cards.forEach((card) => {
      const hit = $('.fig-card__hit', card);
      if (hit) hit.addEventListener('click', () => open(card));
    });
    $$('[data-close]', lb).forEach((b) => b.addEventListener('click', close));
    const prev = $('[data-prev]', lb), next = $('[data-next]', lb);
    if (prev) prev.addEventListener('click', () => step(-1));
    if (next) next.addEventListener('click', () => step(1));

    document.addEventListener('keydown', (e) => {
      if (lb.hidden) return;
      if (e.key === 'Escape') { e.preventDefault(); close(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
      else if (e.key === 'Tab') {

        const f = $$('button', lb).filter((el) => el.offsetParent !== null);
        if (!f.length) return;
        const first = f[0], last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    });
  }

  function debounce(fn, ms) { let t; return function () { clearTimeout(t); t = setTimeout(() => fn.apply(this, arguments), ms); }; }
  function throttle(fn, ms) { let last = 0, t; return function () { const now = Date.now(); const run = () => { last = now; fn.apply(this, arguments); }; if (now - last >= ms) run(); else { clearTimeout(t); t = setTimeout(run, ms - (now - last)); } }; }

  function boot() {
    renderTrack();
    splitHeroTitle();
    nav();
    mobileNav();
    reveals();
    figFilter();
    themeToggle();
    lightbox();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
