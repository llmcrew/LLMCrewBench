// Таблица сравнения и панели расходов в оформлении первого ролика.

import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, FONT_MONO, FONT_SANS} from '../video/theme';
import {Backdrop, riseIn} from '../components/ui';
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
  spend: {answer: number | null; reasoning: number | null; tools: number; minutes: number | null};
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

const BOARD_COLUMNS = `300px repeat(${CATS.length}, minmax(0, 1fr)) 120px 120px 140px`;

const Cell: React.FC<{value?: number}> = ({value}) => (
  <span style={{
    font: `700 24px/1 ${FONT_MONO}`,
    color: value == null ? C.muted : value >= 90 ? C.green : value >= 60 ? C.amber : C.red,
  }}>
    {value == null ? '-' : `${value.toLocaleString('ru-RU', {maximumFractionDigits: 1})}%`}
  </span>
);

export const Board: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{backgroundColor: C.bg, color: C.ink, fontFamily: FONT_SANS}}>
      <Backdrop />
      <div style={{position: 'absolute', top: 76, left: 72, font: `700 66px/1.15 ${FONT_SANS}`, ...riseIn(frame, 6)}}>Результаты и расходы</div>
      <div style={{position: 'absolute', top: 166, left: 72, font: `400 25px/1.4 ${FONT_SANS}`, color: C.muted, ...riseIn(frame, 12)}}>
        A-J: % от максимума · K: баллы симуляций · L: доставка HTML · итог: /100
      </div>
      <div style={{position: 'absolute', top: 244, left: 72, right: 72}}>
        <div style={{display: 'grid', gridTemplateColumns: BOARD_COLUMNS, gap: 6, alignItems: 'end', paddingBottom: 18}}>
          <div style={{font: `500 18px/1 ${FONT_MONO}`, color: C.muted}}>КОНФИГУРАЦИЯ</div>
          {CATS.map(([key, name]) => (
            <div key={key} style={{textAlign: 'center'}}>
              <div style={{font: `700 20px/1 ${FONT_MONO}`, color: C.accent}}>{key}</div>
              <div style={{font: `500 17px/1.2 ${FONT_SANS}`, color: C.muted, marginTop: 8}}>{name}</div>
            </div>
          ))}
          <div style={{textAlign: 'center', borderLeft: `1px solid ${C.line}`}}>
            <div style={{font: `700 20px/1 ${FONT_MONO}`, color: C.accent2}}>K</div>
            <div style={{font: `500 17px/1.2 ${FONT_SANS}`, color: C.muted, marginTop: 8}}>Симуляции</div>
          </div>
          <div style={{textAlign: 'center'}}>
            <div style={{font: `700 20px/1 ${FONT_MONO}`, color: C.accent2}}>L</div>
            <div style={{font: `500 17px/1.2 ${FONT_SANS}`, color: C.muted, marginTop: 8}}>HTML</div>
          </div>
          <div style={{textAlign: 'right', font: `700 20px/1 ${FONT_MONO}`, color: C.accent}}>ИТОГ</div>
        </div>
        {MODELS.map((model, row) => (
          <div key={model.dir} style={{display: 'grid', gridTemplateColumns: BOARD_COLUMNS, gap: 6, alignItems: 'stretch', padding: '22px 0', borderTop: `1px solid ${C.line}`, ...riseIn(frame, 30 + row * 18)}}>
            <div style={{display: 'flex', alignItems: 'center', gap: 12, minWidth: 0}}>
              <span style={{width: 8, height: 48, flexShrink: 0, background: model.accent}} />
              <span style={{font: `600 22px/1.3 ${FONT_SANS}`, overflowWrap: 'anywhere'}}>{model.label}</span>
            </div>
            {CATS.map(([key]) => {
              const total = model.totals[key];
              const pct = total?.possible ? total.earned / total.possible * 100 : undefined;
              return (
                <div key={key} style={{display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 66, borderRadius: 6, background: pct == null ? C.panel : pct >= 90 ? `${C.green}18` : pct >= 60 ? `${C.amber}18` : `${C.red}18`}}>
                  <Cell value={pct} />
                </div>
              );
            })}
            {(['K', 'L'] as const).map((key) => (
              <div key={key} style={{display: 'flex', alignItems: 'center', justifyContent: 'center', background: C.panel, borderLeft: key === 'K' ? `1px solid ${C.line}` : undefined, font: `700 24px/1 ${FONT_MONO}`}}>
                {model[key].e}/{model[key].p}
              </div>
            ))}
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 8, borderLeft: `2px solid ${model.accent}`, background: C.bgSoft, font: `700 24px/1 ${FONT_MONO}`, color: model.accent, whiteSpace: 'nowrap'}}>
              {model.score.toLocaleString('ru-RU')}/100
            </div>
          </div>
        ))}
      </div>
      <div style={{position: 'absolute', top: 600, bottom: 64, left: 72, right: 72, display: 'flex', gap: 24}}>
        {MODELS.map((model, i) => (
          <div key={model.dir} style={{flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: '28px 32px', border: `1px solid ${C.line}`, background: C.panel, ...riseIn(frame, 100 + i * 10)}}>
            <div style={{display: 'flex', alignItems: 'center', gap: 14, paddingBottom: 20, borderBottom: `1px solid ${C.line}`}}>
              <span style={{width: 8, height: 30, background: model.accent}} />
              <span style={{font: `600 27px/1.2 ${FONT_SANS}`}}>{model.label}</span>
            </div>
            {[
              ['Токены ответа', model.spend.answer?.toLocaleString('ru-RU') ?? 'н/д'],
              ['Вызовы инструментов', model.spend.tools.toLocaleString('ru-RU')],
              ['Время работы агента', model.spend.minutes == null ? 'н/д' : `${model.spend.minutes.toLocaleString('ru-RU')} мин`],
            ].map(([label, value]) => (
              <div key={label} style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 24}}>
                <span style={{font: `400 25px/1.3 ${FONT_SANS}`, color: C.muted}}>{label}</span>
                <span style={{font: `700 32px/1 ${FONT_MONO}`, whiteSpace: 'nowrap'}}>{value}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
