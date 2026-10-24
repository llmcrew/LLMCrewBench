// Hero-сцена, вывод и аутро.

import {AbsoluteFill, Img, OffthreadVideo, staticFile, useCurrentFrame} from 'remotion';
import {C, FONT_MONO, FONT_SANS} from '../video/theme';
import {Backdrop, Title, riseIn} from '../components/ui';
import {HEIGHT, WIDTH} from '../video/data';
import {videoFramingStyle} from '../video/framing';
import models from '../data/runs.json';
import edit from '../data/edit.json';
import {showcaseTimeline} from '../video/showcase-data';

type ModelRow = {
  label: string;
  accent?: string;
  score: number;
  K: {e: number; p: number};
  L: {e: number; p: number};
  totals?: Record<string, {earned: number; possible: number}>;
};

const MODELS = (models as unknown as ModelRow[]).slice().sort((a, b) => b.score - a.score);

/** Число из runs.json с десятичной запятой для подписей на русском. */
const formatNumber = (value: number) => value.toLocaleString('ru-RU', {maximumFractionDigits: 1});
const formatScore = (score: number) => `${formatNumber(score)}/100`;

const SHOWCASE_CARDS = [
  {category: 'K', label: 'Симуляции', accent: C.purple},
  {category: 'L', label: 'Арт-сцены', accent: C.pink},
].map((card) => ({...card, count: showcaseTimeline.filter((scene) => scene.scene.startsWith(`${card.category}-`)).length}));

const VERDICT_CUES = [
  {at: edit.cues.verdictSummary, text: 'Общий балл не заменяет оценку качества и стабильности'},
  {at: edit.cues.verdictFailures, text: 'Результат нужно сохранить и проверить'},
  {at: edit.cues.verdictStability, text: 'Стабильность: запуск и доставка исходной работы'},
  {at: edit.cues.verdictQuality, text: 'Визуальное качество — оценка автора, не автоматический балл'},
  {at: 0, text: 'Баллы, расходы и исходные работы'},
];

/** Речевая подводка вынесена перед музыкальными показами, не поверх тестов. */
export const ShowcaseIntro: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{backgroundColor: C.bg, color: C.ink, fontFamily: FONT_SANS}}>
      <Backdrop />
      <div style={{position: 'absolute', left: 96, right: 96, top: 124}}>
        <Title size={72}>Теперь — сами работы</Title>
        <Title sub delay={18}>DeepSeek-V4.1-Flash и DeepSeek-V4.1-Flash 1017</Title>
      </div>
      <div style={{position: 'absolute', left: 96, right: 96, top: 400, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48}}>
        {SHOWCASE_CARDS.map(({category, label, accent, count}, index) => (
          <div key={category} style={{padding: 48, borderTop: `4px solid ${accent}`, background: C.panel, ...riseIn(frame, 24 + index * 14)}}>
            <div style={{font: `700 112px/1 ${FONT_MONO}`, color: accent}}>{count}</div>
            <div style={{font: `600 44px/1.3 ${FONT_SANS}`, marginTop: 28}}>{label}</div>
            <div style={{font: `400 25px/1.5 ${FONT_SANS}`, color: C.muted, marginTop: 20}}>Один набор заданий · сравнение по очереди</div>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

/** Hero: одна сцена раскрыта на весь кадр, без полос. */
export const Hero: React.FC<{clip: string; caption: string; model: string}> = ({clip, caption, model}) => {
  return (
    <AbsoluteFill style={{backgroundColor: '#05080f'}}>
      <OffthreadVideo
        src={staticFile(clip)}
        muted
        playbackRate={1}
        style={videoFramingStyle(clip, WIDTH, HEIGHT)}
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
    </AbsoluteFill>
  );
};

/** Вывод: сопоставление двух конфигураций по числам из runs.json. */
export const Verdict: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = VERDICT_CUES.find((item) => frame >= item.at)!;
  const first = MODELS[0];
  const last = MODELS[MODELS.length - 1];
  const comparison = `${first.label} − ${last.label}`;
  const gap = Math.round((first.score - last.score) * 10) / 10;
  const kCompatible = first.K.p > 0 && MODELS.every((model) => model.K.p === first.K.p);
  const lCompatible = first.L.p > 0 && MODELS.every((model) => model.L.p === first.L.p);
  const pct = (earned: number, possible: number) =>
    `${((earned / Math.max(1, possible)) * 100).toLocaleString('ru-RU', {maximumFractionDigits: 2})}%`;
  const lFact = `L: доставка ${MODELS.map((model) => `${model.label} ${model.L.e}/${model.L.p}`).join(' · ')}`
    + (lCompatible ? ' - техническая доставка, не оценка вкуса' : ' - покрытие разное; проценты несопоставимы');
  const cats = new Set(MODELS.flatMap((model) => Object.keys(model.totals ?? {})));
  const escored = [...cats]
    .filter((category) => 'ABCDEFGHIJ'.includes(category))
    .map((category) => {
      const percentages = MODELS.map((model) => {
        const total = model.totals?.[category] ?? {earned: 0, possible: 1};
        return total.possible ? (total.earned / total.possible) * 100 : 0;
      });
      return {category, delta: Math.round((percentages[0] - percentages[percentages.length - 1]) * 10) / 10};
    })
    .filter((item) => Math.abs(item.delta) > 0.5);
  const facts = [
    [
      'Итог A-J',
      MODELS.map((model) => `${model.label}: ${formatScore(model.score)}`).join(' · '),
      first.accent,
    ],
    [
      'Разница по категориям A-J',
      escored.length
        ? `${comparison}: ${escored.map((item) => `${item.category} ${item.delta > 0 ? '+' : ''}${formatNumber(item.delta)} п.п.`).join(', ')}`
        : `${comparison}: нет расхождений больше 0,5 п.п.`,
      C.accent,
    ],
    [
      'Разница итогового балла',
      `${comparison}: ${formatNumber(gap)} пункта из 100`,
      last.accent,
    ],
    [
      'K · симуляции',
      kCompatible
        ? MODELS.map((model) => `${model.label}: ${model.K.e}/${model.K.p} (${pct(model.K.e, model.K.p)})`).join(' · ')
        : `${MODELS.map((model) => `${model.label}: ${model.K.e}/${model.K.p}`).join(' · ')} - покрытие разное, без процентов`,
      C.purple,
    ],
    ['L · арт-сцены', lFact, C.pink],
  ] as const;
  return (
    <AbsoluteFill style={{backgroundColor: C.bg, color: C.ink}}>
      <Backdrop />
      <div style={{position: 'absolute', top: 78, left: 96, right: 96}}>
        <Title size={58}>Что показали числа</Title>
        <Title sub key={cue.at} delay={cue.at}>{cue.text}</Title>
      </div>
      <div style={{position: 'absolute', top: 290, bottom: 88, left: 72, right: 72, display: 'flex', flexDirection: 'column', gap: 14}}>
        {facts.map(([name, value, accent], i) => (
          <div
            key={name}
            style={{
              display: 'flex',
              alignItems: 'center',
              flex: 1,
              minHeight: 0,
              gap: 20,
              padding: '18px 26px',
              borderTop: `1px solid ${C.line}`,
              borderBottom: `1px solid ${C.line}`,
              background: C.panel,
              ...riseIn(frame, 40 + i * 14),
            }}
          >
            <span style={{width: 8, height: 42, flex: '0 0 8px', background: accent ?? C.accent}} />
            <span style={{font: `500 22px/1.2 ${FONT_SANS}`, color: C.muted, width: 390, flex: '0 0 390px'}}>{name}</span>
            <span style={{font: `600 24px/1.25 ${FONT_SANS}`, minWidth: 0}}>{value}</span>
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
      <div style={{position: 'absolute', top: 30, left: 96, right: 96}}>
      </div>
      <div style={{position: 'absolute', top: 145, bottom: 188, left: 96, right: 96, display: 'grid', gridTemplateColumns: '440px 1fr', gap: 88, alignItems: 'center'}}>
        <Img
          src={staticFile('llmcrew-watermark.png')}
          alt="LLMCrew - матрёшка с ламой"
          style={{width: 440, height: 620, objectFit: 'contain', ...riseIn(frame, 6)}}
        />
        <div style={{display: 'flex', justifyContent: 'space-between', gap: 48}}>
          {[
            {title: 'Канал', url: 'https://t.me/LLMCrew', label: 't.me/LLMCrew', qr: 'llmcrew-channel-qr.svg'},
            {title: 'Чат', url: 'https://t.me/LLMCrew_chat', label: 't.me/LLMCrew_chat', qr: 'llmcrew-chat-qr.svg'},
          ].map(({title, url, label, qr}, index) => (
            <a
              key={url}
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={`${title} LLMCrew в Telegram`}
              style={{flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 28, color: C.ink, textDecoration: 'none', ...riseIn(frame, 14 + index * 8)}}
            >
              <span style={{font: `500 36px/1.2 ${FONT_SANS}`}}>{title}</span>
              <Img
                src={staticFile(qr)}
                alt={`QR-код: ${url}`}
                style={{width: 336, height: 336, flexShrink: 0, borderRadius: 12}}
              />
              <span style={{font: `500 27px/1.4 ${FONT_MONO}`, whiteSpace: 'nowrap'}}>{label}</span>
            </a>
          ))}
        </div>
      </div>
      <div style={{position: 'absolute', bottom: 111, left: 96, right: 96, display: 'flex', justifyContent: 'center'}}>

      </div>
      <div style={{position: 'absolute', bottom: 34, left: 96, right: 96, font: `500 34px/1.3 ${FONT_SANS}`, textAlign: 'center', ...riseIn(frame, 58)}}>
        Создавай, измеряй и делись находками вместе с LLMCrew
      </div>
    </AbsoluteFill>
  );
};

