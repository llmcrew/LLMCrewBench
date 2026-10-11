// Мелкие UI-примитивы ролика: появления, панели, чипы, счётчики.

import type {CSSProperties, ReactNode} from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {C, FONT_MONO, FONT_SANS} from '../video/theme';

export const fadeIn = (frame: number, start: number, dur = 12): number =>
  interpolate(frame, [start, start + dur], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

export const fadeOut = (frame: number, end: number, dur = 12): number =>
  interpolate(frame, [end - dur, end], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

export const riseIn = (frame: number, start: number, dur = 14, dy = 24): CSSProperties => ({
  opacity: fadeIn(frame, start, dur),
  transform: `translateY(${interpolate(frame, [start, start + dur], [dy, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  })}px)`,
});

export const Backdrop: React.FC = () => (
  <AbsoluteFill
    style={{
      backgroundImage: `linear-gradient(${C.line}22 1px, transparent 1px), linear-gradient(90deg, ${C.line}22 1px, transparent 1px)`,
      backgroundSize: '72px 72px',
      maskImage: 'radial-gradient(120% 90% at 50% 40%, #000 35%, transparent 100%)',
      WebkitMaskImage: 'radial-gradient(120% 90% at 50% 40%, #000 35%, transparent 100%)',
    }}
  />
);

export const Title: React.FC<{children: ReactNode; size?: number; delay?: number; sub?: boolean}> = ({
  children,
  size = 54,
  delay = 0,
  sub = false,
}) => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        font: sub ? `400 24px/1.4 ${FONT_SANS}` : `700 ${size}px/1.15 ${FONT_SANS}`,
        color: sub ? C.muted : C.ink,
        ...riseIn(frame, delay),
      }}
    >
      {children}
    </div>
  );
};

export const Pill: React.FC<{children: ReactNode; tone?: string; delay?: number; size?: number}> = ({
  children,
  tone = C.accent,
  delay = 0,
  size = 18,
}) => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '6px 14px',
        borderRadius: 999,
        border: `1px solid ${tone}55`,
        background: `${tone}14`,
        color: tone,
        font: `600 ${size}px/1 ${FONT_MONO}`,
        ...riseIn(frame, delay, 10, 12),
      }}
    >
      {children}
    </div>
  );
};

export const Chip: React.FC<{children: ReactNode; style?: CSSProperties}> = ({children, style}) => (
  <span
    style={{
      font: `500 15px/1 ${FONT_MONO}`,
      color: C.muted,
      border: `1px solid ${C.line}`,
      borderRadius: 999,
      padding: '6px 12px',
      ...style,
    }}
  >
    {children}
  </span>
);
