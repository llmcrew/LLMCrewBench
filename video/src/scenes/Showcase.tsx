// Две половины сравнения. Только доступные результаты получают полноэкранный ход.
// Неуспешная сторона остаётся в своей половине с причиной.

import {AbsoluteFill, OffthreadVideo, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_MONO, FONT_SANS} from '../video/theme';
import {fadeIn} from '../components/ui';
import {HEIGHT, WIDTH} from '../video/data';
import {videoFramingStyle} from '../video/framing';
import {CLIP_SECONDS, OPEN_SECONDS, SHOWCASE_LEAD, SHOWCASE_MODELS, SHOWCASE_TURN_SECONDS, presentationClip, type MediaStatus, type ShowcaseRun} from '../video/showcase-timing';


/** Причина плашки словами: отсутствие задачи не выдаётся за провал модели. */
const REASON: Record<MediaStatus, [string, string]> = {
  ok: ['', ''],
  not_run: ['не запускалась', 'этой задачи нет в прогоне'],
  model_failed: ['модель не справилась', 'страница не доставлена или падает до кадра'],
  capture_failed: ['сбой записи', 'страница жива, клип не снят'],
  media_missing: ['клип не доставлен', 'файла записи нет в поставке'],
  evaluator_unavailable: ['оценщик недоступен', 'страница есть, вердикта нет'],
  infra_failure: ['сорвала инфраструктура', 'транспорт, 503 или воркер'],
  skipped: ['задача пропущена', 'в прогоне отмечена как skipped'],
};

// Две равные полноэкранные половины без внешних полей.
const LANE_W = WIDTH / SHOWCASE_MODELS;

/** Полоса: стоит на месте, пока её очередь; на своём ходу выезжает на весь кадр. */
const Lane: React.FC<{
  run: ShowcaseRun;
  index: number;
  openIndex: number;
  openStart: number;
  openEnd: number;
  laneFrom: number;
  laneTo: number;
  overlayOnly?: boolean;
}> = ({run, index, openIndex, openStart, openEnd, laneFrom, laneTo, overlayOnly = false}) => {
  const note = run.resultNote;
  const repair = note?.repair;
  const sourceUnchanged = repair && repair.sourceSha256 === repair.repairedSha256;
  const clip = presentationClip(run);
  const hasVideo = clip !== null;
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  // Открыта ровно одна полоса - та, чья сейчас очередь.
  const isOpen = index === openIndex;

  // Раскрытие и возврат - доли бюджета SHOWCASE_LEAD вокруг окна показа: полоса
  // приходит на весь кадр ровно к началу своего отрезка, уходит на место сразу
  // после него, а сетке остаётся пауза до следующего хода (turn = OPEN + 2*LEAD).
  const leadFrames = SHOWCASE_LEAD * fps;
  const expandFrames = Math.max(2, Math.round(leadFrames * 0.5));
  const returnFrames = Math.max(2, Math.round(leadFrames * 0.5));
  const expandFrom = openStart - expandFrames;
  const expandTo = openStart;
  const returnFrom = openEnd;
  const returnTo = openEnd + returnFrames;
  const heroActive = isOpen && frame >= openStart && frame < returnTo;

  const t = isOpen
    ? interpolate(frame, [expandFrom, expandTo, returnFrom, returnTo], [0, 1, 1, 0], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      })
    : 0;

  // Пока полоса стоит во весь кадр, остальные лежат ровно под ней и картинку не
  // меняют: их можно не рендерить. Иначе на каждой сцене декодировались бы все
  // все клипы сразу и рендер шёл бы ~кадр в секунду.
  const covered = openIndex >= 0 && !isOpen && frame >= expandTo && frame <= returnFrom;
  if (covered) return null;

  const x = interpolate(t, [0, 1], [index * LANE_W, 0]);
  const w = interpolate(t, [0, 1], [LANE_W, WIDTH]);
  const h = HEIGHT;
  const z = isOpen && frame >= expandFrom && frame <= returnTo ? 30 : 1 + index;

  // Полоса в сетке играет клип размеренно, на всю длину сцены.
  const windowFrames = Math.max(1, laneTo - laneFrom);
  const rate = CLIP_SECONDS / (windowFrames / fps);
  // А на своём ходу - свой клип заново, с первого кадра, на всё окно показа с
  // возвратом: раскрытая сцена начинается с начала, а не с середины общего
  // хода, и уезжает вместе с полосой, без склейки на середине.
  const openFrames = Math.round(OPEN_SECONDS * fps);
  const heroFrames = openFrames + (returnTo - returnFrom);
  const openRate = CLIP_SECONDS / (heroFrames / fps);

  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: 0,
        width: w,
        height: HEIGHT,
        boxSizing: 'border-box',
        zIndex: z,
        overflow: 'hidden',
        borderRadius: 0,
        border: 0,
        borderRight: index === 0 && t === 0 ? `1px solid ${C.line}` : 0,
        background: overlayOnly ? 'transparent' : '#05080f',
      }}
    >
      {overlayOnly ? null : !hasVideo ? (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            textAlign: 'center',
            padding: 20,
            gap: 12,
            color: C.faint,
          }}
        >
          <div style={{font: `700 30px/1.2 ${FONT_SANS}`, color: C.ink}}>
            {note?.originalStatus === 'failed'
              ? 'Модель не справилась'
              : note?.originalStatus === 'partial'
                ? 'Частичный результат'
                : REASON[run.status][0]}
          </div>
          <div style={{font: `500 22px/1.4 ${FONT_SANS}`, maxWidth: 800}}>
            {note?.reason ?? REASON[run.status][1]}
          </div>
          {note ? (
            <div style={{font: `500 20px/1.4 ${FONT_SANS}`, maxWidth: 800}}>
              Ожидалось: {note.expected}
            </div>
          ) : null}
        </div>
      ) : (
        <>
          {/* Закрытая штора продолжает общий клип по времени сцены.
              Активный полноэкранный слой заменяет её, а не дублирует декодирование. */}
          {!heroActive ? (
            <Sequence
              from={laneFrom}
              durationInFrames={windowFrames}
              name={`${run.label} клип`}
              layout="none"
            >
              <OffthreadVideo
                src={staticFile(clip)}
                playbackRate={rate}
                muted
                style={videoFramingStyle(clip, w, h)}
              />
            </Sequence>
          ) : null}

          {/* Свой ход: клип переигрывается с нуля поверх размеренного и живёт
              до посадки полосы обратно в сетку. */}
          {isOpen ? (
            <Sequence
              from={Math.round(openStart)}
              durationInFrames={heroFrames}
              name={`${run.label}: показ с начала`}
            >
              <OffthreadVideo
                src={staticFile(clip)}
                startFrom={0}
                playbackRate={openRate}
                muted
                style={videoFramingStyle(clip, w, h)}
              />
            </Sequence>
          ) : null}
        </>
      )}
      {repair ? (
        <div
          style={{
            position: 'absolute',
            top: 104,
            left: 16,
            right: 16,
            zIndex: 2,
            width: 'fit-content',
            maxWidth: 'calc(100% - 32px)',
            padding: '7px 11px',
            border: `1px solid ${C.amber}`,
            borderRadius: 6,
            background: 'rgba(5,8,15,.94)',
            color: C.amber,
            font: `700 ${interpolate(t, [0, 1], [12, 19])}px/1.3 ${FONT_SANS}`,
            textAlign: 'center',
          }}
        >
          {sourceUnchanged
            ? 'Исходный код модели · расширенный запуск'
            : 'Технический ремонт · не исходный результат модели'}
        </div>
      ) : null}

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
          opacity: 1 - t,
          pointerEvents: 'none',
        }}
      >
        <span style={{width: 8, height: 8, flex: '0 0 auto', borderRadius: 999, background: run.accent}} />
        <span style={{font: `600 ${interpolate(t, [0, 1], [18, 23])}px/1 ${FONT_SANS}`, color: C.ink}}>
          {run.label}
        </span>
      </div>
    </div>
  );
};

/** Неуспешные стороны не получают ни раскрытия, ни отдельного окна показа. */
export const ShowcaseScene: React.FC<{
  sceneId: string;
  title: string;
  runs: ShowcaseRun[];
  order: number[];
  seconds: number;
  overlayOnly?: boolean;
}> = ({sceneId, title, runs, order, seconds, overlayOnly = false}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const turn = Math.round(SHOWCASE_TURN_SECONDS * fps);

  const openedIndex = order.length ? Math.min(order.length - 1, Math.floor(frame / turn)) : -1;
  const openIndex = order[openedIndex] ?? -1;
  const openStart = openedIndex < 0 ? 0 : openedIndex * SHOWCASE_TURN_SECONDS * fps + SHOWCASE_LEAD * fps;
  const openEnd = openedIndex < 0 ? 0 : openStart + OPEN_SECONDS * fps;
  const captionOpacity = openIndex < 0 ? 0 : interpolate(frame, [openStart, openStart + 16, openEnd - 8, openEnd + 8], [0, 1, 1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const openRun = runs[openIndex];
  const resultNote = openRun?.resultNote;
  const sourceUnchanged = resultNote?.repair
    && resultNote.repair.sourceSha256 === resultNote.repair.repairedSha256;
  const activeAccent = openRun?.accent ?? C.line;

  return (
    <AbsoluteFill style={{backgroundColor: overlayOnly ? 'transparent' : C.bg}}>
      <div
        style={{
          position: 'absolute',
          top: 34,
          left: 48,
          right: 48,
          zIndex: 40,
          textAlign: 'center',
          opacity: fadeIn(frame, 4),
          pointerEvents: 'none',
        }}
      >
        <div style={{font: `600 18px/1 ${FONT_MONO}`, letterSpacing: '0.08em', color: activeAccent, textShadow: '0 2px 12px rgba(0,0,0,.9)'}}>
          {sceneId}
        </div>
        <div style={{marginTop: 9, font: `700 36px/1.15 ${FONT_SANS}`, color: C.ink, textShadow: '0 2px 14px rgba(0,0,0,.9)'}}>
          {title.replace(new RegExp(`^${sceneId}\\s*[---]\\s*`), '')}
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
          laneFrom={0}
          laneTo={Math.round(seconds * fps)}
          overlayOnly={overlayOnly}
        />
      ))}

      {openRun ? (
        <div style={{position: 'absolute', bottom: 26, left: 48, right: 48, zIndex: 40, textAlign: 'center', font: `600 24px/1.25 ${FONT_SANS}`, color: C.ink, textShadow: '0 2px 12px rgba(0,0,0,.9)', opacity: captionOpacity, pointerEvents: 'none'}}>
          {openRun.label}
        </div>
      ) : null}

      {resultNote ? (
        <div
          style={{
            position: 'absolute',
            left: 48,
            right: 48,
            bottom: 76,
            zIndex: 40,
            maxWidth: 1600,
            margin: '0 auto',
            padding: '10px 18px',
            borderRadius: 8,
            background: 'rgba(5,8,15,.88)',
            font: `500 20px/1.35 ${FONT_SANS}`,
            color: C.ink,
            textAlign: 'center',
            opacity: captionOpacity,
            pointerEvents: 'none',
          }}
        >
          {resultNote.repair ? (
            <>
              <div>{sourceUnchanged ? 'Ограничение оценщика' : 'Исходная проверка не пройдена'}: {resultNote.reason}</div>
              <div>{sourceUnchanged ? 'Повторный запуск' : 'Техническое исправление'}: {resultNote.repair.change}</div>
            </>
          ) : (
            <div>Частичный результат: {resultNote.reason}</div>
          )}
          <div>Ожидалось: {resultNote.expected}</div>
        </div>
      ) : null}
    </AbsoluteFill>
  );
};

