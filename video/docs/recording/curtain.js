/**
 * curtain.js - чистая раскадровка занавеса: две полосы, выезд на весь кадр.
 *
 * Подключается перед `page-driver.js` (или вставляется в страницу вместе с
 * ним) и не трогает DOM: только числа. Так её можно прогнать в node и сверить
 * с `layout.py` - рекордер и страница обязаны считать одинаково.
 *
 * Единицы - кадры мастера. Полоса стоит в своей половине, пока не её очередь;
 * на ходу выезжает за `curtain` кадров до окна показа и уходит за столько же
 * после. Между ходами - пауза `lead - curtain`.
 *
 * Настройка (все ключи обязательны):
 *   fps       - частота мастера (120);
 *   open      - кадров окна показа (20 с);
 *   secs      - кадров всей записи (целое число ходов);
 *   lanes     - число полос (2);
 *   width     - ширина кадра (1920);
 *   lead      - кадров лида вокруг показа (1.5 с);
 *   curtain   - кадров занавеса (0.75 с, ровно половина лида);
 *   order     - порядок полос в секции;
 *   clips     - клип на каждый ход (для подписи и трассировки).
 */
(function (root) {
  const fail = (message) => { throw new TypeError(message); };
  const whole = (value, name, {min = 0, positive = false} = {}) => {
    if (!Number.isInteger(value) || value < min || (positive && value === 0)) {
      fail(`${name} must be ${positive ? 'a positive' : `an integer >= ${min}`} integer`);
    }
    return value;
  };

  /**
   * Проверить раскладку и вернуть кадровые величины.
   * `curtain` обязан быть половиной `lead`: иначе полоса приезжает не к окну.
   */
  const validate = (config) => {
    if (!config || typeof config !== 'object') fail('config must be an object');
    const fps = whole(config.fps, 'fps', {min: 1, positive: true});
    const lanes = whole(config.lanes, 'lanes', {min: 1, positive: true});
    const width = whole(config.width, 'width', {min: 1, positive: true});
    const lead = whole(config.lead, 'lead', {min: 1, positive: true});
    const curtain = whole(config.curtain, 'curtain', {min: 2});
    const secs = whole(config.secs, 'secs', {min: 1, positive: true});
    const open = whole(config.open ?? secs, 'open', {min: 1, positive: true});
    if (width % lanes !== 0) fail(`width ${width} is not divisible by ${lanes} lanes`);
    if (Math.max(2, Math.round(lead * 0.5)) !== curtain) fail('curtain must be half of lead');
    if (curtain >= open) fail('curtain must be shorter than the open window');
    const turn = open + 2 * lead;
    if (secs % turn) fail(`recording is not a whole number of ${turn}-frame turns`);
    const order = config.order ?? Array.from({length: lanes}, (_, index) => index);
    if (!Array.isArray(order) || !order.length) fail('order must be a non-empty array');
    if (new Set(order).size !== order.length) fail('order repeats a lane');
    for (const lane of order) {
      whole(lane, 'order lane', {min: 0});
      if (lane >= lanes) fail(`order lane ${lane} outside 0..${lanes - 1}`);
    }
    const frames = whole(config.frames ?? secs, 'frames', {min: 1, positive: true});
    const height = whole(config.height ?? 1080, 'height', {min: 1, positive: true});
    const clips = Array.isArray(config.clips) ? config.clips.slice() : [];
    return {
      fps, lanes, width, height, lead, curtain, secs, open, frames,
      order: order.slice(), clips,
    };
  };

  /** Раскадровка записи: один объект на ход, без перекрытий. */
  const build = (config, start = 0) => {
    const settings = validate(config);
    const stride = settings.open + 2 * settings.lead;
    const first = whole(start, 'start', {min: 0});
    return Array.from({length: settings.secs / stride}, (_, position) => {
      const lane = settings.order[position % settings.order.length];
      const turnStart = first + position * stride;
      const openFrom = turnStart + settings.lead;
      const openTo = openFrom + settings.open;
      return {
        lane,
        clip: settings.clips[position % settings.clips.length] || '',
        turnStart,
        expandFrom: openFrom - settings.curtain,
        openFrom,
        openTo,
        returnTo: openTo + settings.curtain,
        turnEnd: turnStart + stride,
      };
    });
  };

  /** Доля занавеса кадра `frame`: 0 - полоса на месте, 1 - во весь кадр. */
  const facing = (turn, frame, curtain) => {
    if (frame < turn.expandFrom || frame >= turn.returnTo) return 0;
    if (frame < turn.openFrom) return (frame - turn.expandFrom) / curtain;
    if (frame < turn.openTo) return 1;
    return (turn.returnTo - frame) / curtain;
  };

  /**
   * Кадр -> полоса и геометрия между своей половиной и полным кадром.
   * Вне ходов все полосы стоят на месте.
   */
  const frameState = (turns, frame, config) => {
    const settings = validate(config);
    for (const turn of turns) {
      if (frame < turn.expandFrom || frame >= turn.returnTo) continue;
      const curtain = turn.openFrom - turn.expandFrom;
      const t = facing(turn, frame, Math.max(1, curtain));
      const rest = settings.width / settings.lanes;
      const x = turn.lane * rest * (1 - t);
      const width = rest + (settings.width - rest) * t;
      return {lane: turn.lane, curtain: t, x: Math.round(x), width: Math.round(width)};
    }
    return {lane: -1, curtain: 0, x: 0, width: settings.width / settings.lanes};
  };

  /** Геометрия полосы в сетке (без занавеса). */
  const laneRect = (index, config) => {
    const settings = validate(config);
    const width = settings.width / settings.lanes;
    return {x: index * width, width, height: Number(config.height ?? 1080)};
  };

  root.__dbcurtain = {validate, build, facing, frameState, laneRect};
})(typeof window !== 'undefined' ? window : globalThis);
