// Витрина: все 49 сцен подряд, каждая — пять последовательных раскрытий.

import {AbsoluteFill, Sequence} from 'remotion';
import {ShowcaseScene, type ShowcaseRun} from './Showcase';
import {FPS, SHOWCASE_SCENE_SECONDS} from '../video/data';
import {C} from '../video/theme';
import showcase from '../data/showcase.json';

export const SCENE_SECONDS = SHOWCASE_SCENE_SECONDS;

const sceneIndex = showcase as unknown as {
  scene: string;
  title: string;
  runs: ShowcaseRun[];
}[];

/** Порядок обхода полос: сдвигается со сценой, чтобы первыми побывали разные. */
const orderFor = (i: number): number[] => [0, 1, 2, 3, 4].map((k) => (k + i) % 5);

export const ShowcaseReel: React.FC = () => (
  <AbsoluteFill style={{backgroundColor: C.bg}}>
    {sceneIndex.map((s, i) => (
      <Sequence
        key={s.scene}
        from={Math.round(i * SCENE_SECONDS * FPS)}
        durationInFrames={Math.round(SCENE_SECONDS * FPS)}
      >
        <ShowcaseScene
          sceneId={s.scene}
          title={s.title}
          runs={s.runs}
          order={orderFor(i)}
          seconds={SCENE_SECONDS}
        />
      </Sequence>
    ))}
  </AbsoluteFill>
);

export {sceneIndex};
