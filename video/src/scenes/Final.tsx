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
        <span style={{font: `500 20px/1 ${FONT_MONO}`, color: C.faint}}>живая сцена, не кадр</span>
      </div>
    </AbsoluteFill>
  );
};

/** Вывод: кто в чём силён (по числам, без оценочных суждений сверх данных). */
export const Verdict: React.FC = () => {
  const frame = useCurrentFrame();
  const best = MODELS[0];
  const fastest = MODELS.slice().sort((a, b) => a.spend.minutes - b.spend.minutes)[0];
  const topK = MODELS.slice().sort((a, b) => b.K.e / b.K.p - a.K.e / a.K.p)[0];
  const topL = MODELS.slice().sort((a, b) => b.L.e / b.L.p - a.L.e / a.L.p)[0];
  const facts = [
    ['Лучший итог', `${best.label} — ${best.verdict || best.score.toFixed(0)}/100`, best.accent],
    ['Больше всех инструментов', `${MODELS.slice().sort((a, b) => b.spend.tools - a.spend.tools)[0].label} — ${MODELS.slice().sort((a, b) => b.spend.tools - a.spend.tools)[0].spend.tools} вызовов`, C.accent],
    ['Быстрее всех по генерации', `${fastest.label} — ${Math.round(fastest.spend.minutes)} мин`, fastest.accent],
    ['Лучшие симуляции (K)', `${topK.label} — ${Math.round((topK.K.e / topK.K.p) * 100)}%`, topK.accent],
    ['Лучшие арт-сцены (L)', `${topL.label} — ${Math.round((topL.L.e / topL.L.p) * 100)}%`, topL.accent],
  ] as const;
  return (
    <AbsoluteFill style={{backgroundColor: C.bg, color: C.ink}}>
      <Backdrop />
      <div style={{position: 'absolute', top: 96, left: 96, ...riseIn(frame, 6)}}>
        <Title size={58}>Что показали числа</Title>
        <Title sub delay={18}>Ниже витрина: те же модели, те же сцены — можно смотреть глазами</Title>
      </div>
      <div style={{position: 'absolute', top: 320, left: 96, right: 96, display: 'flex', flexDirection: 'column', gap: 14}}>
        {facts.map(([name, value, accent], i) => (
          <div
            key={name}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 20,
              padding: '20px 26px',
              borderRadius: 12,
              border: `1px solid ${C.line}`,
              background: C.panel,
              ...riseIn(frame, 40 + i * 14),
            }}
          >
            <span style={{width: 10, height: 30, borderRadius: 4, background: accent}} />
            <span style={{font: `500 22px/1 ${FONT_SANS}`, color: C.muted, width: 460}}>{name}</span>
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
          Живые сцены всех сорока девяти заданий, отчёт и сравнение конфигураций
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
