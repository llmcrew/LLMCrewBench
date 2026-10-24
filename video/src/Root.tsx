import {Composition} from 'remotion';
import {DailyBench} from './video/DailyBench';
import {Section} from './video/Sections';
import {FPS, HEIGHT, SCENES, WIDTH, totalFrames, sceneFrames} from './video/data';
import type {SceneId} from './video/data';
import {ShowcaseReel} from './scenes/ShowcaseReel';

const SingleScene: React.FC<{id: SceneId; overlayOnly?: boolean}> = ({id, overlayOnly = false}) =>
  id === 'showcase' && overlayOnly ? <ShowcaseReel overlayOnly /> : <Section id={id} />;

export const RemotionRoot: React.FC = () => (
  <>
    <Composition
      id="DailyBench"
      component={DailyBench}
      durationInFrames={totalFrames()}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
    />
    {SCENES.map((s) => (
      <Composition
        key={s.id}
        id={`scene-${s.id}`}
        component={SingleScene}
        defaultProps={{id: s.id, overlayOnly: false}}
        durationInFrames={sceneFrames(s)}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
    ))}
  </>
);
