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
          Пять моделей · один сьют · сто баллов · визуальные блоки отдельно
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
      <div style={{position: 'absolute', top: 320, left: 96, right: 96, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 18}}>
        {CYCLE.map((step, i) => (
          <div
            key={step}
            style={{
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
        Один и тот же harness для всех моделей: одинаковая политика, одинаковые инструменты, одинаковый сброс контекста
        между задачами. Разница в баллах — это разница моделей, а не обвязки.
      </div>
    </Shell>
  );
};

const CATEGORIES: [string, string, number, number][] = [
  ['A', 'Следование инструкциям', 8, 10],
  ['B', 'Рассуждение', 15, 15],
  ['C', 'Код', 3, 20],
  ['D', 'Вызовы инструментов', 12, 14],
  ['E', 'Длинный контекст', 5, 10],
  ['F', 'Сопротивление галлюцинациям', 8, 8],
  ['G', 'Структурированный вывод', 4, 8],
  ['H', 'Диалог', 6, 6],
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
      <div style={{position: 'absolute', top: 280, left: 96, width: 1120, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14}}>
        {CATEGORIES.map(([key, name, tasks, pts], i) => (
          <div
            key={key}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 16,
              padding: '18px 20px',
              borderRadius: 10,
              border: `1px solid ${C.line}`,
              background: C.panel,
              ...riseIn(frame, 30 + i * 9),
            }}
          >
            <span style={{font: `700 26px/1 ${FONT_MONO}`, color: C.accent, width: 36}}>{key}</span>
            <span style={{font: `500 22px/1.2 ${FONT_SANS}`, flex: 1}}>{name}</span>
            <span style={{font: `500 17px/1 ${FONT_MONO}`, color: C.muted}}>{tasks} задач</span>
            <span style={{font: `600 17px/1 ${FONT_MONO}`, color: C.ink}}>{pts} б.</span>
          </div>
        ))}
      </div>
      <div style={{position: 'absolute', top: 280, left: 1260, width: 564, display: 'flex', flexDirection: 'column', gap: 16}}>
        <div style={{padding: 24, borderRadius: 12, border: `1px solid ${C.purple}55`, background: `${C.purple}10`, ...riseIn(frame, 60)}}>
          <div style={{font: `700 30px/1.2 ${FONT_SANS}`, color: C.purple}}>K — 20 симуляций</div>
          <div style={{font: `400 19px/1.5 ${FONT_SANS}`, color: C.muted, marginTop: 8}}>
            Three.js-сцены с проверяемой физикой: пробы состояния сверяются с независимым пересчётом на Python.
          </div>
        </div>
        <div style={{padding: 24, borderRadius: 12, border: `1px solid ${C.pink}55`, background: `${C.pink}10`, ...riseIn(frame, 84)}}>
          <div style={{font: `700 30px/1.2 ${FONT_SANS}`, color: C.pink}}>L — 29 арт-сцен</div>
          <div style={{font: `400 19px/1.5 ${FONT_SANS}`, color: C.muted, marginTop: 8}}>
            Свободная форма: палитра, композиция, детали, движение. Смотрится глазами, в балл не входит.
          </div>
        </div>
        <div style={{padding: 22, borderRadius: 12, border: `1px solid ${C.line}`, background: C.bgSoft, ...riseIn(frame, 108)}}>
          <div style={{font: `500 19px/1.5 ${FONT_SANS}`, color: C.muted}}>
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
        <Title size={54}>Четыре конфигурации</Title>
        <Title sub delay={16}>Разные семейства и масштабы: интересен не итог сам по себе, а профиль по категориям</Title>
      </div>
      <div style={{position: 'absolute', top: 250, left: 84, right: 84, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16}}>
        {MODELS.map((m, i) => (
          <div
            key={m.dir}
            style={{
              padding: 22,
              borderRadius: 14,
              border: `1px solid ${C.line}`,
              background: `linear-gradient(180deg, ${C.panelHi}, ${C.panel})`,
              ...riseIn(frame, 30 + i * 16),
            }}
          >
            <div style={{height: 4, width: 46, borderRadius: 999, background: m.accent ?? C.accent, marginBottom: 18}} />
            <div style={{font: `600 24px/1.25 ${FONT_SANS}`}}>{m.label}</div>
            <div style={{font: `500 15px/1.5 ${FONT_MONO}`, color: C.faint, marginTop: 10}}>
              {m.quant} · окно {Math.round(m.context / 1024)}K
            </div>
            <div style={{marginTop: 20, font: `700 44px/1 ${FONT_SANS}`, color: m.accent ?? C.accent}}>
              {m.verdict || `${m.score.toFixed(0)}/100`}
            </div>
            <div style={{display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 20}}>
              <Chip>K {Math.round((m.K.e / Math.max(1, m.K.p)) * 100)}%</Chip>
              <Chip>L {Math.round((m.L.e / Math.max(1, m.L.p)) * 100)}%</Chip>
              <Chip>{Math.round(m.spend.minutes)} мин</Chip>
            </div>
          </div>
        ))}
      </div>
      <div style={{position: 'absolute', bottom: 76, left: 96, right: 96, font: `400 20px/1.5 ${FONT_SANS}`, color: C.muted, ...riseIn(frame, 140)}}>
        Слева направо по итогу: чем правее — тем ниже балл. Витрина в конце покажет, что именно каждая собрала в K и L.
      </div>
    </Shell>
  );
};
