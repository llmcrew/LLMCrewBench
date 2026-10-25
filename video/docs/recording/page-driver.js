/**
 * page-driver.js - драйвер живой сцены: замер окна движения и покадровый seek.
 *
 * Вставляется в конец страницы (или в готовую копию страницы). Ему нужны три
 * вещи от сцены: `window.__step(dt)`, `window.__probe()` и `window.__ready`.
 * Драйвер сам:
 *
 *  1. ждёт `__ready`;
 *  2. замеряет окно движения - сколько шагов подряд canvas реально меняется,
 *     пока подряд не пройдёт `QUIET` шагов без изменений (потолок `MEASURE_CAP`);
 *  3. раскладывает окно на показ (~`SPAN_PER_STEP` кадров на шаг, 4-10 с) и
 *     зацикливает показ перезагрузкой.
 *
 * Для записи наружу выставлены `window.__dbscene.seek(n)` (ручной прогон до
 * шага n; назад не ходит) и сообщения `step|pause|play|reset`.
 *
 * СЕТКА И ЗАНАВЕС. С `?dbstage=1` страница становится не сценой, а кадром из
 * равных полос (`lanes`, по умолчанию 2 × 960 px). Каждая полоса - iframe с
 * той же страницей (`dbstage=0`), со своим замером и своим seek. Внешний
 * драйвер раскладывает полосы по раскадровке `curtain.js`: на своём ходу
 * полоса выезжает на весь кадр за `curtain` секунд, стоит клип и уходит
 * обратно. Поэтому кадр клипа = кадр раскадровки, и `seek(n)` ведёт сразу и
 * геометрию, и обе полосы. Без `dbstage` поведение прежнее: одна сцена.
 *
 * ВАЖНО: страница, которая перерисовывает кадр только в `__probe()`, без
 * драйвера стоит «как скриншот» - драйвер оборачивает `WebGLRenderer` и
 * перерисовывает сцену после каждого шага.
 *
 * Настройка - константы ниже. `MIN_WINDOW` - порог статичности: сцена, которая
 * меняется меньше 4 шагов, показывается одним кадром.
 */
(() => {
  const DT = 1 / 60;
  const QUIET = 90;          // столько шагов без изменений - окно кончилось
  const MEASURE_CAP = 1200;  // предел замера в шагах
  const MIN_WINDOW = 4;      // короче - сцена статична, повторять нечего
  const SPAN_PER_STEP = 6;   // кадров на шаг показа (~0.1 с при 60 к/с)
  const SPAN_MIN = 240;      // показ короче 4 с сливается
  const SPAN_MAX = 600;      // ...а длиннее 10 с - затягивается
  const HOLD_MS = 900;       // пауза перед повтором
  const MEASURE_MS = 8;      // бюджет замера на кадр, чтобы не морозить вкладку

  // three.js держит render собственным свойством экземпляра (this.render =
  // render), поэтому обёртка прототипа его не видит: подменяем конструктор, а
  // render перепривязываем на самом экземпляре.
  let lastRender = null;
  const wrap = () => {
    const W = window.THREE && window.THREE.WebGLRenderer;
    if (!W || W.__dbWrapped) return !!W;
    const Wrapped = function (...args) {
      const inst = new W(...args);
      const orig = inst.render.bind(inst);
      inst.render = function (scene, camera) {
        lastRender = [inst, scene, camera];
        return orig(scene, camera);
      };
      return inst;
    };
    Wrapped.prototype = W.prototype;
    Object.setPrototypeOf(Wrapped, W);
    Wrapped.__dbWrapped = true;
    window.THREE.WebGLRenderer = Wrapped;
    return true;
  };
  if (!wrap()) {
    let n = 0;
    const timer = setInterval(() => {
      if (wrap() || ++n > 1500) clearInterval(timer);
    }, 4);
  }

  const step = () => {
    if (typeof window.__step === 'function') {
      try { window.__step(DT); } catch (e) { /* сцена сама решит */ }
    }
  };
  const probe = () => {
    if (typeof window.__probe === 'function') {
      try { window.__probe(); } catch (e) { /* probe может кидать */ }
    }
  };
  const draw = () => {
    if (lastRender) {
      try { lastRender[0].render(lastRender[1], lastRender[2]); } catch (e) { /* пусто */ }
    }
  };
  // Один шаг - и сразу кадр: probe зовём всегда, часть страниц рисует в нём.
  const beat = () => { step(); probe(); draw(); };

  let offscreen = null, offctx = null;
  const sig = () => {  // дешёвая подпись кадра: 64x48 достаточно для «меняется/нет»
    try {
      const canvases = document.querySelectorAll('canvas');
      if (!canvases.length) return -1;
      if (!offscreen) {
        offscreen = document.createElement('canvas');
        offscreen.width = 64;
        offscreen.height = 48;
        offctx = offscreen.getContext('2d', { willReadFrequently: true });
      }
      // Сумма по ВСЕМ canvas: страница может держать второй canvas, а первый -
      // WebGL без preserveDrawingBuffer, и подпись по нему читалась бы пустой.
      let total = 0;
      for (const canvas of canvases) {
        if (!canvas.width || !canvas.height) continue;
        offctx.clearRect(0, 0, 64, 48);
        offctx.drawImage(canvas, 0, 0, 64, 48);
        const data = offctx.getImageData(0, 0, 64, 48).data;
        let sum = 0;
        for (let i = 0; i < data.length; i += 4) {
          sum = (sum + data[i] + data[i + 1] * 3 + data[i + 2] * 7) % 1000003;
        }
        total = (total + sum + canvases.length * 7) % 1000003;
      }
      return total;
    } catch (e) {
      return -2;
    }
  };

  // Замер должен пережить перезагрузку (её использует повтор): sessionStorage,
  // а если недоступен - window.name (оба живут в пределах вкладки).
  const KEY = 'dailybench.scene:' + location.pathname;
  const put = (value) => {
    try { sessionStorage.setItem(KEY, JSON.stringify(value)); } catch (e) { /* нет доступа */ }
    try { window.name = 'dbscene:' + JSON.stringify(value); } catch (e) { /* нет доступа */ }
  };
  const got = () => {
    try {
      const value = sessionStorage.getItem(KEY);
      if (value) return JSON.parse(value);
    } catch (e) { /* нет доступа */ }
    try {
      if (window.name.indexOf('dbscene:') === 0) return JSON.parse(window.name.slice(8));
    } catch (e) { /* нет доступа */ }
    return null;
  };

  let freeze = 0, span = SPAN_MAX, frames = 0, shown = 0;
  let phase = 'measuring', holdUntil = 0, auto = true;
  const scene = { mode: 'measuring', freeze: 0, step: 0 };
  // Ответы сами выставляют `window.__scene` - драйвер живёт под своим именем,
  // чтобы не отбирать его у страницы.
  window.__dbscene = scene;
  window.__beat = beat;

  // --- занавес и две полосы ------------------------------------------------
  // Сцена-полоса меряет собственное окно (ниже), а страница-сетка только
  // раскладывает полосы и выезжает нужной на весь кадр по раскадровке.
  const stage = {config: null, el: null, lanes: [], turns: []};
  const stageFacing = () => {
    const query = new URLSearchParams(location.search);
    if (query.get('dbstage') !== '1') return null;
    if (!window.__dbcurtain) return null;
    const number = (key, fallback) => Number(query.get(key) ?? fallback);
    const order = (query.get('order') || '')
      .split(',').filter((part) => part !== '').map(Number);
    // Раскадровка считает в кадрах: секунды из URL переводим и округляем.
    const fps = number('fps', 120);
    const config = {
      fps, secs: Math.round(number('secs', 23) * fps),
      open: Math.round(number('open', 20) * fps), lanes: number('lanes', 2),
      width: number('width', window.innerWidth), height: number('height', window.innerHeight),
      lead: Math.round(number('lead', 1.5) * fps),
      curtain: Math.round(number('curtain', 0.75) * fps),
      order: order.length ? order : undefined,
      clips: (query.get('clips') || '').split(',').filter(Boolean),
      frames: number('frames', 1),
    };
    try { return window.__dbcurtain.validate(config); } catch (e) {
      console.warn('dbstage: ' + e.message);
      return null;
    }
  };
  const mountStage = (config) => {
    stage.config = config;
    stage.turns = window.__dbcurtain.build(config);
    const rest = config.width / config.lanes;
    stage.el = document.createElement('div');
    stage.el.id = 'dbstage';
    stage.el.style.cssText = 'position:fixed;inset:0;overflow:hidden;background:#05080f';
    for (let index = 0; index < config.lanes; index++) {
      const lane = document.createElement('div');
      lane.className = 'db-lane';
      lane.style.cssText = 'position:absolute;top:0;left:0;height:100%;'
        + `width:${rest}px;transform:translateX(${index * rest}px);will-change:transform,width`;
      const frame = document.createElement('iframe');
      frame.style.cssText = 'display:block;width:100%;height:100%;border:0';
      const own = new URLSearchParams(location.search);
      own.set('dbstage', '0');
      own.set('lane', String(index));
      const query = own.toString();
      frame.src = location.pathname + (query ? '?' + query : '') + location.hash;
      lane.appendChild(frame);
      stage.el.appendChild(lane);
      stage.lanes.push(lane);
    }
    document.body.appendChild(stage.el);
    scene.mode = 'staged';
    // Внешнее окно движения - длина клипа: рекордер ведёт кадр клипа напрямую.
    scene.freeze = config.frames - 1;
    scene.storyboard = () => stage.turns.slice();
    scene.seek = (n) => {
      auto = false;
      const frame = Math.max(0, Math.min(config.frames - 1, n | 0));
      const state = window.__dbcurtain.frameState(stage.turns, frame, config);
      stage.lanes.forEach((lane, index) => {
        const open = index === state.lane;
        lane.style.zIndex = String(open ? 30 : 1 + index);
        lane.style.width = (open ? state.width : rest) + 'px';
        lane.style.transform = 'translateX(' + (open ? state.x : index * rest) + 'px)';
      });
      // Полосы идут своим ходом всё время. Автоплей стоит с момента готовности,
      // поэтому seek не упирается в уже пройденные шаги: считаем от базы полосы.
      const progress = config.frames > 1 ? frame / (config.frames - 1) : 1;
      stage.lanes.forEach((lane) => {
        const inner = lane.firstChild && lane.firstChild.contentWindow;
        const drive = inner && inner.__dbscene;
        if (!drive || !drive.seek) return;
        if (lane.dbBase === undefined) {
          inner.postMessage('pause', '*');
          lane.dbBase = drive.step | 0;
        }
        const own = drive.freeze | 0;
        const span = Math.max(0, own - lane.dbBase);
        drive.seek(lane.dbBase + Math.round(progress * span));
      });
      shown = frame;
      scene.step = frame;
      return frame;
    };
    const laneReady = () => stage.lanes.every((lane) => {
      const inner = lane.firstChild && lane.firstChild.contentWindow;
      const drive = inner && inner.__dbscene;
      if (!drive || drive.mode === 'measuring') return false;
      // Автоплей гасим сразу: иначе до первого seek полоса уходит вперёд.
      if (lane.dbBase === undefined) {
        inner.postMessage('pause', '*');
        lane.dbBase = drive.step | 0;
      }
      return true;
    });
    const waitLanes = () => {
      if (laneReady()) { window.__ready = true; return; }
      requestAnimationFrame(waitLanes);
    };
    window.__ready = false;
    waitLanes();
  };

  const publish = () => {
    scene.mode = phase;
    scene.freeze = freeze;
    scene.step = shown;
    // seek(n) - ручной покадровый прогон до шага n (гарантирует первый кадр)
    scene.seek = (n) => {
      auto = false;
      if (shown === 0) { beat(); shown++; }
      while (shown < (n | 0)) { beat(); shown++; }
      scene.step = shown;
      return shown;
    };
  };
  publish();
  const staged = stageFacing();
  if (staged) mountStage(staged);

  const play = () => {
    const cache = got();
    freeze = (cache && cache.freeze | 0) || 0;
    span = Math.min(SPAN_MAX, Math.max(SPAN_MIN, freeze * SPAN_PER_STEP));
    phase = freeze < MIN_WINDOW ? 'static' : 'play';
    frames = 0;
    shown = 0;
    publish();
    if (phase === 'play') requestAnimationFrame(tick);
  };

  const tick = () => {
    if (phase === 'play' && auto) {
      frames++;
      const target = Math.min(freeze, Math.round(frames * freeze / span));
      while (shown < target) { beat(); shown++; }
      scene.step = shown;
      if (frames >= span) {
        phase = 'hold';
        holdUntil = performance.now() + HOLD_MS;
        scene.mode = phase;
      }
    } else if (phase === 'hold' && auto && performance.now() >= holdUntil) {
      location.reload();  // сброса у страницы нет: повтор - только перезагрузкой
    }
    requestAnimationFrame(tick);
  };

  if (staged) {
    // Сетка не играет сама: кадры двигает рекордер через seek, полосы - свои
    // маленькие драйверы. Замер и автоплей остаются полосам.
  } else if (got() === null) {
    // Первый заход: дождаться кадра страницы, измерить окно и перезагрузиться -
    // показ пойдёт с чистого состояния, замер уже в кэше.
    const t0 = performance.now();
    let prev = null, lastChange = 0, k = 0, started = false;
    const measureFrame = () => {
      const until = performance.now();
      while (performance.now() - until < MEASURE_MS && k < MEASURE_CAP) {
        beat();
        k++;
        const s = sig();
        if (s !== prev) { lastChange = k; prev = s; }
        else if (k - lastChange >= QUIET) break;
      }
      if ((k >= MEASURE_CAP || k - lastChange >= QUIET) && k > 0) {
        put({ freeze: lastChange });
        location.reload();
        return;
      }
      requestAnimationFrame(measureFrame);
    };
    const wait = () => {
      if (window.__ready !== true && performance.now() - t0 < 8000) {
        requestAnimationFrame(wait);
        return;
      }
      probe();
      draw();
      prev = sig();
      if (!started) {
        started = true;
        requestAnimationFrame(measureFrame);
      }
    };
    requestAnimationFrame(wait);
  } else {
    play();
  }

  const resume = () => {
    auto = true;
    if (phase !== 'play') {
      phase = 'play';
      publish();
      requestAnimationFrame(tick);
    }
  };
  window.addEventListener('message', (event) => {
    const command = event.data;
    if (command === 'step') { auto = false; beat(); shown++; scene.step = shown; }
    else if (command === 'pause') { auto = false; }
    else if (command === 'play') { resume(); }
    else if (command === 'reset') { window.location.reload(); }
    else return;
    window.dispatchEvent(new Event('dailybench-step'));
  });

  window.addEventListener('keydown', (event) => {
    if (event.target && /INPUT|TEXTAREA/.test(event.target.tagName || '')) return;
    if (event.key === ' ') {
      event.preventDefault();
      auto = !auto;
      if (auto) resume();
    }
    if (event.key === 'ArrowRight') {
      event.preventDefault();
      auto = false;
      beat();
      shown++;
      scene.step = shown;
    }
    if (event.key === 'r' || event.key === 'R' || event.key === 'к' || event.key === 'К') {
      window.location.reload();
    }
  });
})();
