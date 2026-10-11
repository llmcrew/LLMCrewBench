// Сводная таблица и графики результатов — из данных прогонов (без выдумки).

import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, FONT_MONO, FONT_SANS} from '../video/theme';
import {Backdrop, Title, riseIn} from '../components/ui';
import models from '../data/runs.json';

type ModelRow = {
  dir: string;
  label: string;
  accent?: string;
  score: number;
  verdict: string;
  errors: number;
  K: {e: number; p: number};
  L: {e: number; p: number};
  totals: Record<string, {earned: number; possible: number}>;
  spend: {answer: number; reasoning: number; tools: number; minutes: number};
};

const MODELS = (models as unknown as ModelRow[]).slice().sort((a, b) => b.score - a.score);

const CATS: [string, string][] = [
  ['A', 'Инструкции'],
  ['B', 'Рассужд'],
  ['C', 'Код'],
  ['D', 'Инструм'],
  ['E', 'Длинный ctx'],
  ['F', 'Анти-галлюц'],
  ['G', 'Структура'],
  ['H', 'Диалог'],
  ['I', 'Поиск'],
  ['J', 'Ревью'],
];

const Cell: React.FC<{value: number; tone?: string}> = ({value, tone = C.ink}) => (
  <span style={{font: `600 19px/1 ${FONT_MONO}`, color: value >= 90 ? C.green : value >= 60 ? tone : C.amber}}>
    {value.toFixed(0)}
  </span>
);

export const Board: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{backgroundColor: C.bg, color: C.ink}}>
      <Backdrop />
      <div style={{position: 'absolute', top: 66, left: 96, ...riseIn(frame, 6)}}>
        <Title size={52}>Результаты по категориям, % от максимума</Title>
        <Title sub delay={16}>Балл — из наблюдаемых результатов; визуальные K/L показаны отдельно и в итог не входят</Title>
      </div>

      <div style={{position: 'absolute', top: 246, left: 84, right: 84}}>
        <div style={{display: 'grid', gridTemplateColumns: `220px repeat(${CATS.length}, 1fr) 90px`, gap: 6, alignItems: 'center'}}>
          <span />
          {CATS.map(([key, name]) => (
            <div key={key} style={{textAlign: 'center'}}>
              <div style={{font: `700 20px/1 ${FONT_MONO}`, color: C.accent}}>{key}</div>
              <div style={{font: `400 13px/1.3 ${FONT_SANS}`, color: C.faint}}>{name}</div>
            </div>
          ))}
          <div style={{textAlign: 'right', font: `600 14px/1 ${FONT_MONO}`, color: C.muted}}>Итог</div>
        </div>

        {MODELS.map((m, row) => (
          <div
            key={m.dir}
            style={{
              display: 'grid',
              gridTemplateColumns: `220px repeat(${CATS.length}, 1fr) 90px`,
              gap: 6,
              alignItems: 'center',
              padding: '13px 0',
              borderTop: `1px solid ${C.line}`,
              ...riseIn(frame, 30 + row * 18),
            }}
          >
            <div style={{display: 'flex', alignItems: 'center', gap: 10}}>
              <span style={{width: 8, height: 26, borderRadius: 3, background: m.accent ?? C.accent}} />
              <span style={{font: `500 20px/1.2 ${FONT_SANS}`}}>{m.label}</span>
            </div>
            {CATS.map(([key]) => {
              const t = m.totals[key] ?? {earned: 0, possible: 1};
              const pct = t.possible ? (t.earned / t.possible) * 100 : 0;
              return (
                <div
                  key={key}
                  style={{
                    textAlign: 'center',
                    padding: '8px 0',
                    borderRadius: 6,
                    background: pct >= 90 ? `${C.green}18` : pct >= 60 ? `${C.panelHi}` : `${C.amber}18`,
                  }}
                >
                  <Cell value={pct} />
                </div>
              );
            })}
            <div style={{textAlign: 'right', font: `700 26px/1 ${FONT_SANS}`, color: m.accent ?? C.accent}}>
              {m.verdict || m.score.toFixed(0)}
            </div>
          </div>
        ))}
      </div>

      <div style={{position: 'absolute', bottom: 60, left: 96, right: 96, display: 'flex', gap: 20}}>
        {MODELS.map((m, i) => (
          <div
            key={m.dir}
            style={{
              flex: 1,
              padding: '16px 18px',
              borderRadius: 10,
              border: `1px solid ${C.line}`,
              background: C.panel,
              ...riseIn(frame, 130 + i * 10),
            }}
          >
            <div style={{font: `500 14px/1 ${FONT_MONO}`, color: C.faint}}>токены ответа</div>
            <div style={{font: `600 22px/1.2 ${FONT_MONO}`, marginTop: 6}}>
              {Math.round(m.spend.answer / 1000)}K
            </div>
            <div style={{font: `500 14px/1 ${FONT_MONO}`, color: C.faint, marginTop: 10}}>вызовы инструментов</div>
            <div style={{font: `600 22px/1.2 ${FONT_MONO}`, marginTop: 6}}>{m.spend.tools}</div>
            <div style={{font: `500 14px/1 ${FONT_MONO}`, color: C.faint, marginTop: 10}}>минут генерации</div>
            <div style={{font: `600 22px/1.2 ${FONT_MONO}`, marginTop: 6}}>{Math.round(m.spend.minutes)}</div>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
