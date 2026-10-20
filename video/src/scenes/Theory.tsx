// Титульные и поясняющие сцены: интро, harness-цикл, сьют, карточки моделей,
// сводная таблица, hero-сцена, вывод, аутро. Данные — из прогонов (runs.json).

import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {C, FONT_MONO, FONT_SANS} from '../video/theme';
import {Backdrop, Chip, Pill, Title, fadeIn, riseIn} from '../components/ui';
import models from '../data/runs.json';

type ModelRow = {
  dir: string;
  label: string;
  model: string;
  score: number;
  verdict: string;
  quant: string;
  context: number;
  tasks: number;
  errors: number;
  K: {e: number; p: number};
  L: {e: number; p: number};
  totals: Record<string, {earned: number; possible: number}>;
  spend: {answer: number; reasoning: number; tools: number; minutes: number};
  accent?: string;
};

const MODELS = (models as unknown as ModelRow[]).slice().sort((a, b) => b.score - a.score);

const Shell: React.FC<{children: React.ReactNode}> = ({children}) => (
  <AbsoluteFill style={{backgroundColor: C.bg, color: C.ink}}>
    <Backdrop />
    {children}
  </AbsoluteFill>
);

export const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <Shell>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', gap: 26}}>
        <Title size={96} delay={6}>
          DailyBench-16
        </Title>
        <Title sub delay={26}>
          Две конфигурации · один сьют · сто баллов · визуальные блоки отдельно
        </Title>
        <div style={{display: 'flex', gap: 12, marginTop: 14}}>
          {MODELS.map((m, i) => (
            <Pill key={m.dir} tone={m.accent ?? C.accent} delay={40 + i * 8}>
              {m.label}
            </Pill>
          ))}
        </div>
        <div style={{font: `500 18px/1 ${FONT_MONO}`, color: C.faint, marginTop: 30, opacity: fadeIn(frame, 70)}}>
          локальные конфигурации · один провайдер · одна политика прогонов
        </div>
      </AbsoluteFill>
    </Shell>
  );
};

const CYCLE = ['Запрос и задача', 'Контекст: файлы, правила', 'Модель: ответ или вызов', 'Инструмент: read / edit / bash', 'Среда: проект и тесты', 'История сессии — и снова'];

export const HarnessDef: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <Shell>
      <div style={{position: 'absolute', top: 84, left: 96, ...riseIn(frame, 6)}}>
        <Title size={56}>Что измеряет бенчмарк</Title>
        <Title sub delay={20}>
          Не «ум» модели, а работу агента: решения, вызовы инструментов и проверяемый результат
        </Title>
      </div>
      <div style={{position: 'absolute', top: 320, bottom: 250, left: 96, right: 96, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gridAutoRows: '1fr', gap: 18}}>
        {CYCLE.map((step, i) => (
          <div
            key={step}
            style={{
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              padding: '22px 24px',
              borderRadius: 12,
              border: `1px solid ${i === 2 ? C.accent + '66' : C.line}`,
              background: i === 2 ? `${C.accent}10` : C.panel,
              font: `500 22px/1.35 ${FONT_SANS}`,
              color: i === 2 ? C.accent2 : C.ink,
              ...riseIn(frame, 30 + i * 14),
            }}
          >
            <div style={{font: `600 13px/1 ${FONT_MONO}`, color: C.faint, marginBottom: 10}}>
              {String(i + 1).padStart(2, '0')}
            </div>
            {step}
          </div>
        ))}
      </div>
      <div style={{position: 'absolute', bottom: 84, left: 96, right: 96, font: `400 20px/1.5 ${FONT_SANS}`, color: C.muted, ...riseIn(frame, 130)}}>
        Harness один и тот же для обоих прогонов: одинаковая политика, инструменты и сброс контекста. Модель и сид тоже
        одни — значит, разница в баллах это разница стендов.
      </div>
    </Shell>
  );
};

const CATEGORIES: [string, string, number, number][] = [
  ['A', 'Следование инструкциям', 5, 10],
  ['B', 'Рассуждение', 15, 15],
  ['C', 'Код', 3, 20],
  ['D', 'Вызовы инструментов', 7, 14],
  ['E', 'Длинный контекст', 5, 10],
  ['F', 'Сопротивление галлюцинациям', 1, 8],
  ['G', 'Структурированный вывод', 4, 8],
  ['H', 'Диалог', 3, 6],
  ['I', 'Поиск по файлам', 2, 4],
  ['J', 'Ревью кода', 2, 5],
];

export const Suite: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <Shell>
      <div style={{position: 'absolute', top: 78, left: 96, ...riseIn(frame, 6)}}>
        <Title size={54}>Сьют: 47 заданий и сто баллов</Title>
        <Title sub delay={18}>
          Каждый балл — наблюдаемый результат: разобранный ответ, файл на диске, прогон скрытых тестов, лог вызовов
        </Title>
      </div>
      <div style={{position: 'absolute', top: 280, bottom: 260, left: 96, width: 1120, display: 'grid', gridTemplateColumns: '1fr 1fr', gridAutoRows: '1fr', gap: 14}}>
        {CATEGORIES.map(([key, name, tasks, pts], i) => (
          <div
            key={key}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              padding: '18px 16px',
              borderRadius: 10,
              border: `1px solid ${C.line}`,
              background: C.panel,
              ...riseIn(frame, 30 + i * 9),
            }}
          >
            <span style={{font: `700 26px/1 ${FONT_MONO}`, color: C.accent, width: 32}}>{key}</span>
            <span style={{font: `500 20px/1.2 ${FONT_SANS}`, flex: 1}}>{name}</span>
            <span style={{font: `500 16px/1 ${FONT_MONO}`, color: C.muted}}>{tasks} задач</span>
            <span style={{font: `600 16px/1 ${FONT_MONO}`, color: C.ink}}>{pts} б.</span>
          </div>
        ))}
      </div>
      <div style={{position: 'absolute', top: 280, bottom: 260, left: 1260, width: 564, display: 'grid', gridTemplateRows: 'repeat(5, 1fr)', gap: 14}}>
        <div style={{gridRow: 'span 2', display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: 20, borderRadius: 12, border: `1px solid ${C.purple}55`, background: `${C.purple}10`, ...riseIn(frame, 60)}}>
          <div style={{font: `700 27px/1.2 ${FONT_SANS}`, color: C.purple}}>K — 20 симуляций</div>
          <div style={{font: `400 17px/1.5 ${FONT_SANS}`, color: C.muted, marginTop: 6}}>
            Three.js-сцены с проверяемой физикой: пробы состояния сверяются с независимым пересчётом на Python.
          </div>
        </div>
        <div style={{gridRow: 'span 2', display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: 20, borderRadius: 12, border: `1px solid ${C.pink}55`, background: `${C.pink}10`, ...riseIn(frame, 84)}}>
          <div style={{font: `700 27px/1.2 ${FONT_SANS}`, color: C.pink}}>L — 29 арт-сцен</div>
          <div style={{font: `400 17px/1.5 ${FONT_SANS}`, color: C.muted, marginTop: 6}}>
            Свободная форма: палитра, композиция, детали, движение. Смотрится глазами, в балл не входит.
          </div>
        </div>
        <div style={{display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: 20, borderRadius: 12, border: `1px solid ${C.line}`, background: C.bgSoft, ...riseIn(frame, 108)}}>
          <div style={{font: `500 17px/1.5 ${FONT_SANS}`, color: C.muted}}>
            Всё генерируется из одного сида: два прогона видят побайтово одинаковые задания.
          </div>
        </div>
      </div>
    </Shell>
  );
};

export const Models: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <Shell>
      <div style={{position: 'absolute', top: 74, left: 96, ...riseIn(frame, 6)}}>
        <Title size={54}>Две конфигурации, один сьют</Title>
        <Title sub delay={16}>Одна модель на двух ingress-стендах: что меняется, когда чинят бэкенд, а не модель</Title>
      </div>
      <div style={{position: 'absolute', top: 250, bottom: 210, left: 210, right: 210, display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 28, alignContent: 'center'}}>
        {MODELS.map((m, i) => (
          <div
            key={m.dir}
            style={{
              padding: 30,
              borderRadius: 14,
              border: `1px solid ${C.line}`,
              background: `linear-gradient(180deg, ${C.panelHi}, ${C.panel})`,
              ...riseIn(frame, 30 + i * 16),
            }}
          >
            <div style={{height: 4, width: 46, borderRadius: 999, background: m.accent ?? C.accent, marginBottom: 18}} />
            <div style={{font: `600 32px/1.25 ${FONT_SANS}`}}>{m.label}</div>
            <div style={{font: `500 15px/1.5 ${FONT_MONO}`, color: C.faint, marginTop: 10}}>
              {m.quant} · окно {Math.round(m.context / 1024)}K · {m.tasks} заданий
            </div>
            <div style={{marginTop: 22, font: `700 72px/1 ${FONT_SANS}`, color: m.accent ?? C.accent}}>
              {m.verdict || `${m.score.toFixed(0)}/100`}
            </div>
            <div style={{display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 24}}>
              <Chip>D {m.totals.D?.earned ?? 0}/{m.totals.D?.possible ?? 0}</Chip>
              <Chip>K {m.K.e}/{m.K.p}</Chip>
              <Chip>L {m.L.e}/{m.L.p}</Chip>
              <Chip>{Math.round(m.spend.minutes)} мин</Chip>
            </div>
          </div>
        ))}
      </div>
      <div style={{position: 'absolute', bottom: 76, left: 96, right: 96, font: `400 20px/1.5 ${FONT_SANS}`, color: C.muted, ...riseIn(frame, 140)}}>
        Обе колонки — одна и та же модель и один сид: разница только в стенде. Дробь на плашках — заработано/возможно по блокам D, K и L. Витрина покажет, что собрал каждый стенд.
      </div>
    </Shell>
  );
};
