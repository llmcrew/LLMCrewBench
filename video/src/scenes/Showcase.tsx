// Сцена витрины: четыре вертикальные полосы с живыми решениями; по очереди каждая
// раскрывается на весь экран на OPEN_SECONDS и возвращается, затем следующая.
//
// Клип полосы играет ускоренно так, чтобы показать свой отрезок за время сцены.

import {AbsoluteFill, Freeze, OffthreadVideo, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_MONO, FONT_SANS} from '../video/theme';
import {fadeIn, fadeOut} from '../components/ui';
import {CLIP_SECONDS, OPEN_SECONDS, SHOWCASE_LEAD, SHOWCASE_MODELS, SHOWCASE_TURN_SECONDS} from '../video/data';

export type ShowcaseRun = {
  model: string;
  label: string;
  clip: string;
  missing: boolean;
  accent: string;
};

// Ширина полосы считается от фактического числа моделей: полосы вместе с зазорами
// и полями ровно заполняют кадр, сетка всегда центрирована.
const LANE_MARGIN = 113;
const LANE_GAP = 18;
const LANE_W = (1920 - LANE_MARGIN * 2 - LANE_GAP * (SHOWCASE_MODELS - 1)) / SHOWCASE_MODELS;
const X0 = (1920 - (LANE_W * SHOWCASE_MODELS + LANE_GAP * (SHOWCASE_MODELS - 1))) / 2;
const LANE_TOP = 196;
const LANE_BOTTOM = 70;
const LANE_H = 1080 - LANE_TOP - LANE_BOTTOM;

/** Полоса: стоит на месте, пока её очередь; на своём ходу выезжает на весь кадр. */
const Lane: React.FC<{
  run: ShowcaseRun;
  index: number;
  openIndex: number;
  openStart: number;
  openEnd: number;
  laneFrom: number;
  laneTo: number;
}> = ({run, index, openIndex, openStart, openEnd, laneFrom, laneTo}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  // Открыта ровно одна полоса — та, чья сейчас очередь.
  const isOpen = index === openIndex;
  const open = isOpen && frame >= openStart && frame < openEnd;
  // Полосы проявляются каскадом и успевают набрать полную плотность к началу
  // первого раскрытия (expandFrom).
  const appear = fadeIn(frame, 3 + index * 2, 6);

  // Раскрытие и возврат — доли бюджета SHOWCASE_LEAD вокруг окна показа: полоса
  // приходит на весь кадр ровно к началу своего отрезка, уходит на место сразу
  // после него, а сетке остаётся пауза до следующего хода (turn = OPEN + 2*LEAD).
  const leadFrames = SHOWCASE_LEAD * fps;
  const expandFrames = Math.max(2, Math.round(leadFrames * 0.4));
  const returnFrames = Math.max(2, Math.round(leadFrames * 0.36));
  const expandFrom = openStart - expandFrames;
  const expandTo = openStart;
  const returnFrom = openEnd;
  const returnTo = openEnd + returnFrames;

  const t = isOpen
    ? interpolate(frame, [expandFrom, expandTo, returnFrom, returnTo], [0, 1, 1, 0], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      })
    : 0;

  // Пока полоса стоит во весь кадр, остальные лежат ровно под ней и картинку не
  // меняют: их можно не рендерить. Иначе на каждой сцене декодировались бы все
  // четыре клипа сразу и рендер шёл бы ~кадр в секунду.
  const covered = !isOpen && frame >= expandTo && frame <= returnFrom;
  if (covered) return null;

  const x = interpolate(t, [0, 1], [X0 + index * (LANE_W + LANE_GAP), 0]);
  const y = interpolate(t, [0, 1], [LANE_TOP, 0]);
  const w = interpolate(t, [0, 1], [LANE_W, 1920]);
  const h = interpolate(t, [0, 1], [LANE_H, 1080]);
  const z = isOpen && frame >= expandFrom && frame <= returnTo ? 30 : 1 + index;

  // Темп клипа: свой отрезок за окно показа (сцена длинная, клип — CLIP_SECONDS).
  const windowFrames = Math.max(1, laneTo - laneFrom);
  const rate = CLIP_SECONDS / (windowFrames / fps);

  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        // В открытом состоянии карточка обязана накрыть кадр целиком: на кадре
        // с frameless-бордером её края заметны как обводка поверх краёв кадра.
        width: open ? 1920 : w,
        height: open ? 1080 : h,
        // border-box: рамка входит в размер, поэтому открытая карточка равна
        // кадру ровно, а полосы в сетке делят ширину без сдвига на рамку.
        boxSizing: 'border-box',
        zIndex: z,
        overflow: 'hidden',
        borderRadius: 12,
        // Акцентная рамка — пока полоса летит или стоит раскрытой (t > 0), и
        // сменяется на обычную, когда та садится в сетку: без скачка на месте.
        border: `1px solid ${t > 0 ? run.accent : C.line}`,
        background: '#05080f',
        boxShadow: open ? '0 24px 80px rgba(0,0,0,.55)' : 'none',
        opacity: appear,
      }}
    >
      {run.missing ? (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'grid',
            placeItems: 'center',
            textAlign: 'center',
            padding: 20,
            font: `600 ${interpolate(t, [0, 1], [18, 34])}px/1.3 ${FONT_SANS}`,
            color: C.faint,
          }}
        >
          модель
          <br />
          не справилась
        </div>
      ) : (
        <Sequence from={laneFrom} durationInFrames={windowFrames} name={`${run.label} клип`}>
          <Freeze frame={Math.max(1, windowFrames - 2)} active={(f) => f >= windowFrames - 1}>
            {/* Клип всегда 1920x1080 и уменьшается только transform'ом: элемент
                <video> не меняет размер при раскрытии и не пересобирается. */}
            <div
              style={{
                position: 'absolute',
                // Клип повторяет поведение objectFit:cover: при сжатой полосе
                // мимо кадра уходит не левый край, а пропорционально срезанные
                // поля — в центре всегда виден один и тот же участок клипа.
                left: (w - 1920 * Math.max(w / 1920, h / 1080)) / 2,
                top: (h - 1080 * Math.max(w / 1920, h / 1080)) / 2,
                width: 1920,
                height: 1080,
                transform: `scale(${Math.max(w / 1920, h / 1080)})`,
                transformOrigin: 'top left',
              }}
            >
              <OffthreadVideo
                src={staticFile(run.clip)}
                playbackRate={rate}
                muted
                style={{position: 'absolute', inset: 0, width: 1920, height: 1080, objectFit: 'cover', objectPosition: 'center'}}
              />
            </div>
          </Freeze>
        </Sequence>
      )}

      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: 0,
          padding: '14px 16px 30px',
          background: 'linear-gradient(180deg, rgba(5,8,15,.92), rgba(5,8,15,0))',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}
      >
        <span style={{width: 10, height: 10, borderRadius: 999, background: run.accent}} />
        <span style={{font: `600 ${interpolate(t, [0, 1], [17, 24])}px/1 ${FONT_SANS}`, color: C.ink}}>
          {run.label}
        </span>
      </div>
    </div>
  );
};

/**
 * Одна сцена витрины: все четыре решения по очереди раскрываются на весь экран.
 * Порядок обхода на каждой сцене свой (детерминированный), чтобы за ролик
 * каждая модель побывала первой и последней примерно поровну.
 */
export const ShowcaseScene: React.FC<{
  sceneId: string;
  title: string;
  runs: ShowcaseRun[];
  order: number[];
  seconds: number;
}> = ({sceneId, title, runs, order, seconds}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const turn = Math.round(SHOWCASE_TURN_SECONDS * fps);

  // Какая полоса сейчас раскрыта: по кадрам идём по order.
  const openedIndex = Math.min(SHOWCASE_MODELS - 1, Math.floor(frame / turn));
  // order — порядок моделей всего реестра (длиной 5). Для сцены с другим числом
  // полос приводим его к индексам runs без повторов: каждая полоса получает свой
  // ход ровно один раз, а не первая дважды.
  const slots = Array.from(new Set(order.map((v) => v % runs.length)));
  const openIndex = slots[openedIndex % slots.length] ?? 0;
  const openStart = openedIndex * SHOWCASE_TURN_SECONDS * fps + SHOWCASE_LEAD * fps;
  const openEnd = openStart + OPEN_SECONDS * fps;
  // Подпись живёт ровно время показа своего решения: к выезду она готова.
  const captionOpacity = interpolate(frame, [openStart, openStart + 16, openEnd - 8, openEnd + 8], [0, 1, 1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill style={{backgroundColor: C.bg}}>
      <div
        style={{
          position: 'absolute',
          top: 44,
          left: 0,
          right: 0,
          textAlign: 'center',
          // К адаптации первой полосы шапка уже не нужна: в открытом состоянии
          // по центру сверху подпись решения, а не заголовок сцены.
          opacity: Math.min(fadeIn(frame, 4) * 0.75, fadeOut(frame, openStart + 10, 30)),
        }}
      >
        <div style={{font: `700 40px/1.15 ${FONT_SANS}`, color: C.ink}}>
          {title.replace(new RegExp(`^${sceneId}\\s*[—-]\\s*`), '')}
        </div>
      </div>

      {runs.map((run, i) => (
        <Lane
          key={run.model}
          run={run}
          index={i}
          openIndex={openIndex}
          openStart={openStart}
          openEnd={openEnd}
          laneFrom={Math.max(1, Math.round(0.3 * fps))}
          laneTo={Math.round((seconds - 0.3) * fps)}
        />
      ))}

      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          bottom: 0,
          height: 128,
          // Подложка под подпись: на широком кадре клипа верхний градиент шапки
          // до подписи не достаёт, поэтому у низа свой. Гаснет вместе с подписью,
          // в закрытой сетке низ полос не затемняет. 39 — над полосами (до 30),
          // под подписью (40).
          zIndex: 39,
          background: 'linear-gradient(180deg, rgba(5,8,15,0), rgba(5,8,15,.92))',
          opacity: captionOpacity,
          pointerEvents: 'none',
        }}
      />

      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          bottom: 22,
          zIndex: 40,
          textAlign: 'center',
          opacity: captionOpacity,
        }}
      >
        <span style={{font: `500 21px/1.4 ${FONT_SANS}`, color: C.ink}}>
          решение {runs[openIndex]?.label ?? '—'} · сцена {sceneId} · {openedIndex + 1}/{runs.length} ·{' '}
          {runs[openIndex]?.missing ? 'страница не ожила' : 'живая страница, снята покадрово'}
        </span>
      </div>
    </AbsoluteFill>
  );
};

export {LANE_W, LANE_GAP, LANE_H};
