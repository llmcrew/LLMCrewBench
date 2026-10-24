// Тема видео: палитра и шрифты. Тёмная база под стать отчёту бенчмарка.

import {loadFont as loadSans} from '@remotion/google-fonts/Inter';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';

export const FONT_SANS = loadSans('normal', {
  weights: ['400', '500', '700'],
  subsets: ['cyrillic', 'latin'],
}).fontFamily;

export const FONT_MONO = loadMono('normal', {
  weights: ['400', '500', '700'],
  subsets: ['cyrillic', 'latin'],
}).fontFamily;

export const C = {
  bg: '#0a0f1e',
  bgSoft: '#0e1526',
  panel: '#111a2e',
  panelHi: '#16213a',
  line: '#23324e',
  ink: '#e8eefb',
  muted: '#a8b8d0',
  faint: '#8499b8',
  accent: '#4cc2ff',
  accent2: '#7ee0ff',
  green: '#3ddc97',
  amber: '#ffb454',
  red: '#ff6b6b',
  purple: '#b48cff',
  pink: '#ff7ac8',
} as const;
