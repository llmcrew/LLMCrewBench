import assert from 'node:assert/strict';
import {test} from 'node:test';
import {
  buildShowcaseTimeline,
  canOpenFullscreen,
  fullscreenOrder,
  type MediaStatus,
  type ShowcaseRun,
} from '../src/video/showcase-timing.ts';

const work = (model: string, status: MediaStatus = 'ok'): ShowcaseRun => ({
  model,
  label: model,
  clip: `${model}.mp4`,
  status,
  accent: '#fff',
});

const failedNote = {
  originalStatus: 'failed' as const,
  reason: 'The original page did not render.',
  expected: 'A running scene.',
};

test('failed results cannot receive a full-screen turn even if a clip exists', () => {
  const failed = {...work('failed'), resultNote: failedNote};
  const repaired = {
    ...failed,
    resultNote: {
      ...failedNote,
      repair: {clip: 'repaired.mp4', change: 'A code change.', sourceSha256: 'original', repairedSha256: 'changed'},
    },
  };
  assert.equal(canOpenFullscreen(failed), false);
  assert.equal(canOpenFullscreen(repaired), false);
});

test('a longer wait for unchanged source is not treated as model failure', () => {
  const extended = {
    ...work('extended'),
    resultNote: {
      ...failedNote,
      repair: {clip: 'extended.mp4', change: 'A longer startup wait.', sourceSha256: 'same', repairedSha256: 'same'},
    },
  };
  assert.equal(canOpenFullscreen(extended), true);
});

test('a delivered partial result remains eligible', () => {
  const partial = {...work('partial'), resultNote: {...failedNote, originalStatus: 'partial' as const}};
  assert.equal(canOpenFullscreen(partial), true);
});

test('unavailable media never creates an empty full-screen presentation', () => {
  const statuses: MediaStatus[] = ['skipped', 'not_run', 'model_failed', 'evaluator_unavailable', 'infra_failure', 'capture_failed', 'media_missing'];
  for (const status of statuses) {
    assert.equal(canOpenFullscreen(work(status, status)), false, status);
  }
});

test('filtering retains the original lane index and gives every valid work one turn', () => {
  const failed = work('left', 'model_failed');
  const right = work('right');
  assert.deepEqual(fullscreenOrder([failed, right], 0), [1]);
  assert.deepEqual(fullscreenOrder([failed, right], 1), [1]);
  assert.deepEqual(fullscreenOrder([work('left'), right], 1).sort(), [0, 1]);
});

test('mixed and all-failed scenes leave no empty slots or gaps in the timeline', () => {
  const left = work('left');
  const right = work('right');
  const badLeft = work('left', 'model_failed');
  const badRight = work('right', 'model_failed');
  const entries = [[left, right], [badLeft, right], [badLeft, badRight], [left, badRight]]
    .map((runs, index) => ({scene: `scene-${index}`, title: 'Comparison', runs}));
  const timeline = buildShowcaseTimeline(entries);
  assert.deepEqual(timeline.map(({from, seconds}) => [from, seconds]), [[0, 46], [46, 23], [69, 4], [73, 23]]);
  assert.deepEqual(timeline[1].order, [1]);
  assert.deepEqual(timeline[2].order, []);
  assert.equal(timeline.at(-1)!.from + timeline.at(-1)!.seconds, 96);
});
