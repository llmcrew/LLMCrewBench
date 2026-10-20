// Hero-сцена, вывод и аутро.

import {AbsoluteFill, OffthreadVideo, staticFile, useCurrentFrame} from 'remotion';
import {C, FONT_MONO, FONT_SANS} from '../video/theme';
import {Backdrop, Title, riseIn} from '../components/ui';
import {CLIP_SECONDS, SCENES} from '../video/data';
import models from '../data/runs.json';

type ModelRow = {
  dir: string;
  label: string;
  accent?: string;
  score: number;
  verdict: string;
  spend: {minutes: number; tools: number};
  K: {e: number; p: number};
  L: {e: number; p: number};
  totals?: Record<string, {earned: number; possible: number}>;
};

const MODELS = (models as unknown as ModelRow[]).slice().sort((a, b) => b.score - a.score);

/** Сцена hero на весь кадр: окно движения клипа укладывается в её длительность. */
const HERO_SECONDS = SCENES.find((s) => s.id === 'hero')!.seconds;

/** Hero: одна сцена раскрыта на весь кадр, без полос. */
export const Hero: React.FC<{clip: string; caption: string; model: string}> = ({clip, caption, model}) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{backgroundColor: '#05080f'}}>
      <OffthreadVideo
        src={staticFile(clip)}
        muted
        playbackRate={CLIP_SECONDS / HERO_SECONDS}
        style={{width: '100%', height: '100%', objectFit: 'cover'}}
      />
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: 0,
          padding: '34px 48px 70px',
          background: 'linear-gradient(180deg, rgba(5,8,15,.9), rgba(5,8,15,0))',
        }}
      >
        <div style={{font: `700 44px/1.2 ${FONT_SANS}`, color: C.ink}}>{caption}</div>
        <div style={{font: `500 20px/1 ${FONT_MONO}`, color: C.muted, marginTop: 8}}>{model}</div>
      </div>
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          bottom: 0,
          padding: '70px 48px 34px',
          textAlign: 'right',
          background: 'linear-gradient(0deg, rgba(5,8,15,.9), rgba(5,8,15,0))',
          opacity: riseIn(frame, 10).opacity,
        }}
      >
        <span style={{font: `500 20px/1 ${FONT_MONO}`, color: C.faint, textShadow: '0 1px 8px rgba(5,8,15,.9), 0 0 2px rgba(5,8,15,.8)'}}>
          живая сцена, не кадр · 120 fps, кадр клипа = кадр показа
        </span>
      </div>
    </AbsoluteFill>
  );
};

/** Вывод: где именно разошлись стенды (по числам, без оценок сверх данных). */
export const Verdict: React.FC = () => {
  const frame = useCurrentFrame();
  const best = MODELS[0];
  const worst = MODELS[MODELS.length - 1];
  const gap = Math.round((best.score - worst.score) * 10) / 10;
  // Категории, где стенды различаются: только они и составляют разницу итога.
  const cats = new Set(MODELS.flatMap((m) => Object.keys(m.totals ?? {})));
  const escored = [...cats]
    .filter((c) => 'ABCDEFGHIJ'.includes(c))
    .map((c) => {
      const e = MODELS.map((m) => {
        const t = m.totals?.[c] ?? {earned: 0, possible: 1};
        return t.possible ? (t.earned / t.possible) * 100 : 0;
      });
      return {c, delta: Math.round((e[0] - e[e.length - 1]) * 10) / 10};
    })
    .filter((x) => Math.abs(x.delta) > 0.5);
  const facts = [
    ['Итог', MODELS.map((m) => `${m.label} — ${m.verdict || `${m.score.toFixed(0)}/100`}`).join(' · '), best.accent],
    [
      'Где именно разошлись (1017 минус A)',
      escored.length
        ? escored.map((x) => `${x.c} ${x.delta > 0 ? '+' : ''}${x.delta} п.п.`).join(', ')
        : 'нет — общая часть A–J совпала',
      C.accent,
    ],
    ['Разница итога', `${gap} ${gap % 10 === 1 && gap !== 11 ? 'балл' : 'баллов'} — ровно на них`, worst.accent],
    [
      'Симуляции (K)',
      MODELS.map((m) => `${m.label}: ${Math.round((m.K.e / Math.max(1, m.K.p)) * 100)}%`).join(' · '),
      C.purple,
    ],
    ['Арт-сцены (L)', MODELS.map((m) => `${m.label}: ${m.L.p} из 145 в блоке · ${Math.round((m.L.e / Math.max(1, m.L.p)) * 100)}%`).join(' · '), C.pink],
  ] as const;
  return (
    <AbsoluteFill style={{backgroundColor: C.bg, color: C.ink}}>
      <Backdrop />
      <div style={{position: 'absolute', top: 96, left: 96, ...riseIn(frame, 6)}}>
        <Title size={58}>Что показали числа</Title>
        <Title sub delay={18}>Ниже витрина: те же стенды, те же сцены — можно смотреть глазами</Title>
      </div>
      <div style={{position: 'absolute', top: 320, bottom: 120, left: 96, right: 96, display: 'flex', flexDirection: 'column', gap: 14}}>
        {facts.map(([name, value, accent], i) => (
          <div
            key={name}
            style={{
              display: 'flex',
              alignItems: 'center',
              flex: 1,
              gap: 20,
              padding: '20px 26px',
              borderRadius: 12,
              border: `1px solid ${C.line}`,
              background: C.panel,
              ...riseIn(frame, 40 + i * 14),
            }}
          >
            <span style={{width: 10, height: 30, borderRadius: 4, background: accent}} />
            <span style={{font: `500 22px/1 ${FONT_SANS}`, color: C.muted, width: 430}}>{name}</span>
            <span style={{font: `600 26px/1 ${FONT_SANS}`}}>{value}</span>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

export const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{backgroundColor: C.bg, color: C.ink}}>
      <Backdrop />
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', gap: 22}}>
        <Title size={72} delay={6}>
          Артефакт — в репозитории
        </Title>
        <Title sub delay={24}>
          Живые сцены всех сорока девяти заданий, отчёт и сравнение стендов
        </Title>
        <div
          style={{
            marginTop: 26,
            padding: '18px 26px',
            borderRadius: 10,
            border: `1px solid ${C.line}`,
            background: C.bgSoft,
            font: `500 20px/1.6 ${FONT_MONO}`,
            color: C.muted,
            ...riseIn(frame, 46),
          }}
        >
          python -m dailybench publish … --out reports/comparison
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
