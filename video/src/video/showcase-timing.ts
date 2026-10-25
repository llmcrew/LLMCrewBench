/** Shared presentation policy and timing; independent of React and media loading. */
export const CLIP_SECONDS = 20;
export const OPEN_SECONDS = CLIP_SECONDS;
export const SHOWCASE_LEAD = 1.5;
export const SHOWCASE_MODELS = 2;
export const SHOWCASE_TURN_SECONDS = OPEN_SECONDS + SHOWCASE_LEAD * 2;
export const SHOWCASE_EMPTY_SECONDS = 4;

export type MediaStatus =
  | 'ok'
  | 'skipped'
  | 'not_run'
  | 'model_failed'
  | 'evaluator_unavailable'
  | 'infra_failure'
  | 'capture_failed'
  | 'media_missing';

export type ResultNote = {
  model: string;
  scene: string;
  originalStatus: 'partial' | 'failed';
  reason: string;
  expected: string;
  repair?: {
    clip: string;
    change: string;
    sourceSha256: string;
    repairedSha256: string;
  } | null;
};

export type ShowcaseRun = {
  model: string;
  label: string;
  clip: string;
  status: MediaStatus;
  accent: string;
  resultNote?: Omit<ResultNote, 'model' | 'scene'>;
};

export type ShowcaseEntry = {
  scene: string;
  title: string;
  runs: ShowcaseRun[];
};

export type ShowcaseSection = ShowcaseEntry & {
  from: number;
  seconds: number;
  order: number[];
};

/** Repaired variants may remain visible in a half without earning a full-screen turn. */
export const presentationClip = (run: ShowcaseRun): string | null =>
  run.resultNote?.repair?.clip
  ?? (run.status === 'ok' && run.resultNote?.originalStatus !== 'failed' ? run.clip : null);

export const canOpenFullscreen = (run: ShowcaseRun): boolean => {
  if (presentationClip(run) === null) return false;
  const note = run.resultNote;
  if (note?.originalStatus !== 'failed') return true;
  // A longer startup wait does not change the model's source; a code repair does.
  return Boolean(note.repair && note.repair.sourceSha256 === note.repair.repairedSha256);
};

/** Keep original lane indices: filtering must not move the right-hand model to the left. */
export const fullscreenOrder = (runs: ShowcaseRun[], offset: number): number[] =>
  Array.from({length: runs.length}, (_, index) => (index + offset) % runs.length)
    .filter((index) => canOpenFullscreen(runs[index]));

export const buildShowcaseTimeline = (entries: ShowcaseEntry[]): ShowcaseSection[] => {
  let from = 0;
  return entries.map((entry, index) => {
    const order = fullscreenOrder(entry.runs, index);
    const seconds = order.length ? order.length * SHOWCASE_TURN_SECONDS : SHOWCASE_EMPTY_SECONDS;
    const section = {...entry, from, seconds, order};
    from += seconds;
    return section;
  });
};
