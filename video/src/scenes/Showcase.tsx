// Сцена витрины: пять вертикальных полос с живыми решениями; по очереди каждая
// раскрывается на весь экран на OPEN_SECONDS и возвращается, затем следующая.
//
// Клип полосы играет ускоренно так, чтобы показать свой отрезок за время сцены.

import {AbsoluteFill, Freeze, OffthreadVideo, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_MONO, FONT_SANS} from '../video/theme';
import {fadeIn} from '../components/ui';
import {CLIP_SECONDS, OPEN_SECONDS, SHOWCASE_LEAD, SHOWCASE_MODELS, SHOWCASE_TURN_SECONDS} from '../video/data';

export type ShowcaseRun = {
  model: string;
  label: string;
  clip: string;
  missing: boolean;
  accent: string;
};

const LANE_W = 350;
const LANE_GAP = 16;
const LANE_H = 780;
const X0 = (1920 - (LANE_W * 5 + LANE_GAP * 4)) / 2;

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
  const appear = fadeIn(frame, 6 + index * 4, 12);

  // Чужая раскрытая карточка накрывает полосу целиком: пока чужая карточка
  // стоит в полном размере, свою полосу рендерить не нужно. Без этого каждую
  // сцену декодировались все пять видео сразу и рендер шёл ~1 кадр/с.
  const covered = !isOpen && frame >= openStart + 10 && frame <= openEnd - 8;
  if (covered) return null;

  const t = isOpen
    ? interpolate(
        frame,
        [openStart - 10, openStart + 8, openEnd - 8, openEnd + 10],
        [0, 1, 1, 0],
        {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
      )
    : 0;
  const x = interpolate(t, [0, 1], [X0 + index * (LANE_W + LANE_GAP), 40]);
  const y = interpolate(t, [0, 1], [196, 54]);
  const w = interpolate(t, [0, 1], [LANE_W, 1840]);
  const h = interpolate(t, [0, 1], [LANE_H, 930]);
  const z = isOpen && frame >= openStart - 10 && frame < openEnd + 10 ? 30 : 1 + index;

  // Темп клипа: свой отрезок за окно показа (сцена длинная, клип — CLIP_SECONDS).
  const windowFrames = Math.max(1, laneTo - laneFrom);
  const rate = CLIP_SECONDS / (windowFrames / fps);

  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        width: w,
        height: h,
        zIndex: z,
        overflow: 'hidden',
        borderRadius: 12,
        border: `1px solid ${open ? run.accent : C.line}`,
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
            <OffthreadVideo
              src={staticFile(run.clip)}
              playbackRate={rate}
              muted
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                width: w,
                height: h,
                objectFit: 'cover',
                objectPosition: 'center',
              }}
            />
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
 * Одна сцена витрины: все пять решений по очереди раскрываются на весь экран.
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
  const openIndex = order[openedIndex] ?? openedIndex;
  const openStart = openedIndex * SHOWCASE_TURN_SECONDS * fps + SHOWCASE_LEAD * fps;
  const openEnd = openStart + OPEN_SECONDS * fps;

  return (
    <AbsoluteFill style={{backgroundColor: C.bg}}>
      <div
        style={{
          position: 'absolute',
          top: 44,
          left: 0,
          right: 0,
          textAlign: 'center',
          opacity: fadeIn(frame, 4) * 0.75,
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
          bottom: 22,
          zIndex: 40,
          textAlign: 'center',
          opacity: interpolate(frame, [openStart + 6, openEnd - 6], [0, 1], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          }),
        }}
      >
        <span style={{font: `500 21px/1.4 ${FONT_SANS}`, color: C.muted}}>
          решение {runs[openIndex]?.label ?? '—'} · сцена {sceneId} · {openedIndex + 1}/5 ·{' '}
          {runs[openIndex]?.missing ? 'страница не ожила' : 'живая страница, снята покадрово'}
        </span>
      </div>
    </AbsoluteFill>
  );
};

export {LANE_W, LANE_GAP, LANE_H};
