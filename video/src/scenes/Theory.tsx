// Титры и пояснения в стилистике первого ролика LLMCrew.

import {useEffect, useState} from 'react';
import {AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, useDelayRender} from 'remotion';
import {C, FONT_MONO, FONT_SANS} from '../video/theme';
import {Backdrop, Title, riseIn} from '../components/ui';
import models from '../data/runs.json';
import edit from '../data/edit.json';

type ModelRow = {
  dir: string;
  label: string;
  score: number;
  verdict: string;
  quant: string;
  context: number;
  tasks: number;
  errors: number;
  K: {e: number; p: number};
  L: {e: number; p: number};
  totals: Record<string, {earned: number; possible: number}>;
  spend: {answer: number | null; reasoning: number | null; tools: number; minutes: number | null};
  accent?: string;
};

const MODELS = (models as unknown as ModelRow[]).slice().sort((a, b) => b.score - a.score);

const Shell: React.FC<{children: React.ReactNode}> = ({children}) => (
  <AbsoluteFill style={{backgroundColor: C.bg, color: C.ink, fontFamily: FONT_SANS}}>
    <Backdrop />
    {children}
  </AbsoluteFill>
);

export const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const cardsStart = edit.cues.introCards;
  const greetingOpacity = interpolate(frame, [cardsStart - 18, cardsStart], [1, 0], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp',
  });
  return (
    <Shell>
      <div style={{position: 'absolute', inset: 0, display: 'flex', justifyContent: 'center', alignItems: 'center', opacity: greetingOpacity}}>
        <div style={{font: `700 126px/1.1 ${FONT_SANS}`, letterSpacing: -5}}>привет интернет</div>
      </div>
      <div style={{position: 'absolute', inset: '88px 96px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 432px', gap: 72, pointerEvents: frame < cardsStart ? 'none' : 'auto', ...riseIn(frame, cardsStart, 24, 18)}}>
        <div style={{display: 'flex', flexDirection: 'column', justifyContent: 'center', minWidth: 0}}>
          <div style={{display: 'flex', alignItems: 'center', gap: 24}}>
            <Img src={staticFile('llmcrew-mascot.png')} alt="Матрёшка LLMCrew с ламой" style={{width: 72, height: 96, objectFit: 'contain', flexShrink: 0}} />
            <div style={{font: `500 30px/1.3 ${FONT_MONO}`, color: C.muted}}>LLMCrewBench</div>
          </div>
          <div style={{display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 44px minmax(0, 1fr)', gap: 16, marginTop: 48, alignItems: 'stretch'}}>
            {MODELS.map((model, index) => (
              <div key={model.dir} style={{gridColumn: index === 0 ? 1 : 3, gridRow: 1, minWidth: 0, minHeight: 160, padding: 24, boxSizing: 'border-box', display: 'flex', alignItems: 'center', border: `1px solid ${C.line}`, borderTop: `4px solid ${model.accent ?? C.accent}`, borderRadius: 16, background: C.panel, ...riseIn(frame, cardsStart + 12 + index * 12, 20, 16)}}>
                <div style={{font: `700 34px/1.3 ${FONT_SANS}`, color: model.accent, whiteSpace: 'nowrap'}}>{model.label}</div>
              </div>
            ))}
            <div style={{gridColumn: 2, gridRow: 1, alignSelf: 'center', textAlign: 'center', font: `500 24px/1 ${FONT_MONO}`, color: C.faint}}>vs</div>
          </div>
          <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 32, marginTop: 42, ...riseIn(frame, cardsStart + 36, 24, 16)}}>
            {[['2', 'конфигурации'], ['100', 'баллов A-J'], ['49', 'сцен K-L']].map(([value, label]) => (
              <div key={label} style={{borderTop: `1px solid ${C.line}`, paddingTop: 24}}>
                <div style={{font: `700 80px/1 ${FONT_MONO}`}}>{value}</div>
                <div style={{font: `400 24px/1.4 ${FONT_SANS}`, color: C.muted, marginTop: 14}}>{label}</div>
              </div>
            ))}
          </div>
        </div>
        <a href="https://github.com/llmcrew/LLMCrewBench" aria-label="Репозиторий LLMCrewBench" style={{alignSelf: 'center', padding: 36, boxSizing: 'border-box', display: 'flex', flexDirection: 'column', alignItems: 'center', border: `1px solid ${C.line}`, borderRadius: 24, background: C.panel, color: C.ink, textDecoration: 'none', ...riseIn(frame, cardsStart + 24, 24, 20)}}>
          <div style={{font: `700 28px/1.2 ${FONT_SANS}`}}>GitHub</div>
          <Img src={staticFile('llmcrewbench-qr.svg')} alt="QR-код репозитория LLMCrewBench" style={{width: 320, height: 320, flexShrink: 0, borderRadius: 8, marginTop: 28, marginBottom: 24}} />
          <div style={{font: `400 23px/1.4 ${FONT_MONO}`, color: C.muted}}>github.com/llmcrew/</div>
          <div style={{font: `500 28px/1.4 ${FONT_MONO}`}}>LLMCrewBench</div>
        </a>
      </div>
    </Shell>
  );
};

const CYCLE = [
  ['Контекст', 'Инструкции + файлы + история вызовов'],
  ['Модель', 'Ответ или аргументы вызова инструмента'],
  ['Инструмент', 'read / edit / bash'],
  ['Среда', 'Файлы, stdout, stderr, код возврата'],
  ['История', 'Результат инструмента → следующий запрос'],
  ['Проверка', 'Ответ / артефакт / состояние сцены'],
];

export const HarnessDef: React.FC = () => {
  const frame = useCurrentFrame();
  const screenshotStart = edit.cues.harnessScreenshot;
  const cycleOpacity = interpolate(frame, [screenshotStart, screenshotStart + 24], [1, 0], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp',
  });
  return (
    <Shell>
      <div style={{position: 'absolute', top: 54, left: 96, right: 96, display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
        <Title size={66}>Harness · Pi</Title>
        <a href="https://pi.dev/" target="_blank" rel="noopener noreferrer" style={{font: `500 28px/1.4 ${FONT_MONO}`, color: C.accent, textDecoration: 'none', ...riseIn(frame, 12)}}>pi.dev</a>
      </div>
      <div style={{opacity: cycleOpacity}} aria-hidden={frame >= screenshotStart + 24}>
        <div style={{position: 'absolute', top: 280, left: 96, right: 96, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24}}>
          {CYCLE.map(([label, value], i) => (
            <div key={label} style={{padding: 32, minHeight: 210, boxSizing: 'border-box', border: `1px solid ${C.line}`, background: C.panel, ...riseIn(frame, 20 + i * 14)}}>
              <div style={{font: `500 20px/1 ${FONT_MONO}`, color: C.accent}}>{i + 1} {i === 4 ? '↺' : '→'}</div>
              <div style={{fontSize: 34, fontWeight: 700, marginTop: 24}}>{label}</div>
              <div style={{fontSize: 26, lineHeight: 1.45, color: C.muted, marginTop: 18}}>{value}</div>
            </div>
          ))}
        </div>
        <div style={{position: 'absolute', top: 830, left: 96, right: 96, font: `400 28px/1.5 ${FONT_SANS}`, color: C.muted, ...riseIn(frame, 112)}}>
          Между заданиями контекст сбрасывается; политика и инструменты одинаковы для обеих конфигураций.
          Сравниваем полученный результат и фактические расходы.
        </div>
      </div>
      <figure aria-hidden={frame < screenshotStart} style={{position: 'absolute', top: 184, left: 96, right: 96, margin: 0, pointerEvents: frame < screenshotStart ? 'none' : 'auto', ...riseIn(frame, screenshotStart, 24)}}>
        <a href="https://pi.dev/" target="_blank" rel="noopener noreferrer" aria-label="Pi — агентный harness">
          <Img src={staticFile('pi-harness.png')} alt="Скриншот pi.dev: Pi — минимальный агентный harness" style={{display: 'block', width: '100%', height: 'auto', border: `1px solid ${C.line}`, borderRadius: 12, boxSizing: 'border-box'}} />
        </a>
        <figcaption style={{marginTop: 12, font: `400 22px/1.3 ${FONT_SANS}`, color: C.muted}}>
          Pi · агентный harness · pi.dev
        </figcaption>
      </figure>
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
      <div style={{position: 'absolute', top: 80, left: 96}}><Title size={66}>A-J · 47 заданий · 100 баллов</Title></div>
      <div style={{position: 'absolute', top: 250, left: 96, width: 1070, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16}}>
        {CATEGORIES.map(([key, label, tasks, points], i) => (
          <div key={key} style={{height: 120, padding: '20px 22px', boxSizing: 'border-box', border: `1px solid ${C.line}`, background: C.panel, display: 'flex', alignItems: 'center', gap: 18, ...riseIn(frame, 24 + i * 8)}}>
            <span style={{font: `700 29px/1 ${FONT_MONO}`, color: C.accent}}>{key}</span>
            <div style={{flex: 1}}>
              <div style={{font: `500 23px/1.25 ${FONT_SANS}`}}>{label}</div>
              <div style={{font: `400 18px/1.2 ${FONT_MONO}`, color: C.muted, marginTop: 10}}>заданий: {tasks}</div>
            </div>
            <span style={{font: `500 22px/1.3 ${FONT_MONO}`, textAlign: 'right'}}>{points}<br /><span style={{fontSize: 16, color: C.muted}}>баллов</span></span>
          </div>
        ))}
      </div>
      <div style={{position: 'absolute', top: 250, right: 96, width: 612, display: 'flex', flexDirection: 'column', gap: 24}}>
        {[
          ['K · 20 симуляций', 'Three.js / физика. Пробы состояния сверяются с независимым пересчётом.', C.purple],
          ['L · 29 арт-сцен', 'Свободная работа автора. Автоматом проверяется доставка HTML; художественное качество оценивает человек.', C.pink],
        ].map(([label, value, tone], i) => (
          <div key={label} style={{padding: 32, border: `1px solid ${tone}55`, background: `${tone}0c`, ...riseIn(frame, 50 + i * 24)}}>
            <div style={{fontSize: 36, fontWeight: 700, color: tone}}>{label}</div>
            <div style={{fontSize: 27, lineHeight: 1.45, color: C.muted, marginTop: 24}}>{value}</div>
          </div>
        ))}
        <div style={{font: `500 25px/1.4 ${FONT_SANS}`, color: C.amber, ...riseIn(frame, 110)}}>K-L не входят в 100 баллов A-J.</div>
      </div>
    </Shell>
  );
};

const MODEL_DETAILS = MODELS.map((model) => ({
  model,
  rows: [
    ['Формат в отчёте', model.quant],
    ['Контекст в прогоне', `${model.context.toLocaleString('ru-RU')} токенов`],
    ['A-J · результат', `${model.score.toLocaleString('ru-RU')}/100`],
    ['K · симуляции', `${model.K.e}/${model.K.p}`],
    ['L · доставка HTML', `${model.L.e}/${model.L.p}`],
    ['Токены ответа', model.spend.answer?.toLocaleString('ru-RU') ?? 'н/д'],
    ['Вызовы инструментов', model.spend.tools.toLocaleString('ru-RU')],
    ['Время работы агента', model.spend.minutes == null ? 'н/д' : `${model.spend.minutes.toLocaleString('ru-RU')} мин`],
  ],
}));

const VLLM_LAUNCH_FILE = 'deepseek-v41-flash-uncensored-1017.sh';

const VllmLaunchConfig: React.FC = () => {
  const {delayRender, continueRender, cancelRender} = useDelayRender();
  const [handle] = useState(() => delayRender('Loading vLLM launch command'));
  const [columns, setColumns] = useState<readonly [string, string]>(() => ['', '']);
  useEffect(() => {
    let mounted = true;
    let pending = true;
    const release = () => {
      if (pending) {
        pending = false;
        continueRender(handle);
      }
    };
    fetch(staticFile(VLLM_LAUNCH_FILE))
      .then((response) => {
        if (!response.ok) throw new Error(`Cannot load ${VLLM_LAUNCH_FILE}: ${response.status}`);
        return response.text();
      })
      .then((text) => {
        if (mounted) {
          const lines = text.trimEnd().split('\n');
          const middle = Math.ceil(lines.length / 2);
          setColumns([lines.slice(0, middle).join('\n'), lines.slice(middle).join('\n')]);
        }
        release();
      })
      .catch((error) => {
        if (mounted) cancelRender(error);
        release();
      });
    return () => {
      mounted = false;
      release();
    };
  }, [handle, continueRender, cancelRender]);
  return (
    <section aria-label="Конфигурация запуска vLLM" style={{position: 'absolute', top: 156, left: 96, right: 96, height: 680, border: `1px solid ${C.line}`, borderRadius: 12, background: C.panel, boxSizing: 'border-box', overflow: 'hidden'}}>
      <div style={{padding: '14px 26px', borderBottom: `1px solid ${C.line}`, font: `500 20px/1.4 ${FONT_MONO}`}}>
        <a href={staticFile(VLLM_LAUNCH_FILE)} download style={{color: C.accent, textDecoration: 'none'}}>{VLLM_LAUNCH_FILE}</a>
      </div>
      <div style={{display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: 40, padding: '24px 26px'}}>
        {columns.map((text, index) => (
          <pre key={index} style={{margin: 0, minWidth: 0, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', font: `400 22px/1.45 ${FONT_MONO}`, color: C.ink}}><code style={{font: 'inherit'}}>{text}</code></pre>
        ))}
      </div>
    </section>
  );
};

export const Models: React.FC = () => {
  const frame = useCurrentFrame();
  const screenshotStart = edit.cues.modelsScreenshot;
  const showLaunchConfig = frame >= edit.cues.modelsLaunchConfig;
  const tableOpacity = interpolate(frame, [screenshotStart, screenshotStart + 24], [1, 0], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp',
  });
  return (
    <Shell>
      <div style={{position: 'absolute', top: 44, left: 96, right: 96, display: 'flex', alignItems: 'center', gap: 28}}>
        <Img src={staticFile('deepseek-icon.png')} alt="Логотип DeepSeek" style={{width: 88, height: 88, objectFit: 'contain', ...riseIn(frame, 0)}} />
        <Title size={62}>DeepSeek-V4.1-Flash vs DeepSeek-V4.1-Flash 1017</Title>
      </div>
      <div style={{opacity: tableOpacity}} aria-hidden={frame >= screenshotStart + 24}>
        <div style={{position: 'absolute', top: 174, left: 96, right: 96, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 64}}>
          {MODEL_DETAILS.map(({model, rows}, i) => (
            <div key={model.dir} style={riseIn(frame, 12 + i * 12)}>
              <div style={{padding: '18px 24px', borderLeft: `3px solid ${model.accent}`, background: C.panel, font: `700 34px/1.2 ${FONT_SANS}`}}>{model.label}</div>
              {rows.map(([label, value]) => (
                <div key={label} style={{minHeight: 62, padding: '12px 0', boxSizing: 'border-box', borderBottom: `1px solid ${C.line}`, display: 'grid', gridTemplateColumns: '310px 1fr', gap: 20, alignItems: 'center'}}>
                  <div style={{font: `400 24px/1.35 ${FONT_SANS}`, color: C.muted}}>{label}</div>
                  <div style={{font: `500 25px/1.35 ${FONT_MONO}`, color: model.accent}}>{value}</div>
                </div>
              ))}
            </div>
          ))}
        </div>
        <div style={{position: 'absolute', top: 776, left: 96, right: 96, font: `400 20px/1.35 ${FONT_SANS}`, color: C.muted, ...riseIn(frame, 100)}}>
          96 заданий в каждом прогоне. K и L показываются отдельно от A-J; время рендера не является расходом генерации модели.
        </div>
      </div>
      {!showLaunchConfig ? <figure aria-hidden={frame < screenshotStart} style={{position: 'absolute', top: 156, left: 160, right: 160, margin: 0, pointerEvents: frame < screenshotStart ? 'none' : 'auto', ...riseIn(frame, screenshotStart, 24)}}>
        <a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main" target="_blank" rel="noopener noreferrer" aria-label="Файлы DeepSeek-V4.1-Flash-1017 на Hugging Face">
          <Img
            src={staticFile('deepseek-v4.1-flash-1017-files.png')}
            alt="Скриншот Hugging Face: файлы репозитория deepseek-ai/DeepSeek-V4.1-Flash-1017"
            style={{display: 'block', width: '100%', height: 'auto', border: `1px solid ${C.line}`, borderRadius: 12, boxSizing: 'border-box'}}
          />
        </a>
        <figcaption style={{marginTop: 10, font: `400 20px/1.3 ${FONT_SANS}`, color: C.muted}}>
          Hugging Face · deepseek-ai/DeepSeek-V4.1-Flash-1017 · файлы релиза
        </figcaption>
      </figure> : <VllmLaunchConfig />}
      <div style={{position: 'absolute', top: 872, bottom: 24, left: 96, right: 96, paddingTop: 12, boxSizing: 'border-box', borderTop: `1px solid ${C.line}`, display: 'grid', gridTemplateColumns: '140px 1fr 388px 420px', gap: 36, alignItems: 'center', ...riseIn(frame, 54)}}>
        <Img src={staticFile('llmcrew-watermark.png')} alt="LLMCrew - матрёшка с ламой" style={{width: 124, height: 172, objectFit: 'contain'}} />
        <div>
          <div style={{font: `700 34px/1.2 ${FONT_SANS}`, marginBottom: 10}}>LLMCrew</div>
          <div style={{font: `400 23px/1.4 ${FONT_SANS}`, color: C.muted}}>Создавай, измеряй и делись находками вместе с LLMCrew</div>
        </div>
        {[
          {title: 'Канал', url: 'https://t.me/LLMCrew', label: 't.me/LLMCrew', qr: 'llmcrew-channel-qr.svg'},
          {title: 'Чат', url: 'https://t.me/LLMCrew_chat', label: 't.me/LLMCrew_chat', qr: 'llmcrew-chat-qr.svg'},
        ].map(({title, url, label, qr}) => (
          <a key={url} href={url} target="_blank" rel="noopener noreferrer" aria-label={`${title} LLMCrew в Telegram`} style={{display: 'flex', alignItems: 'center', gap: 18, color: C.ink, textDecoration: 'none', minWidth: 0}}>
            <Img src={staticFile(qr)} alt={`QR-код: ${url}`} style={{width: 168, height: 168, flexShrink: 0, borderRadius: 8}} />
            <div>
              <div style={{font: `500 26px/1.2 ${FONT_SANS}`}}>{title}</div>
              <div style={{font: `500 20px/1.4 ${FONT_MONO}`, marginTop: 12, whiteSpace: 'nowrap'}}>{label}</div>
            </div>
          </a>
        ))}
      </div>
    </Shell>
  );
};
