import {Composition} from 'remotion';
import {DailyBench} from './video/DailyBench';
import {Section} from './video/Sections';
import {FPS, HEIGHT, SCENES, WIDTH, totalFrames, sceneFrames} from './video/data';
import type {SceneId} from './video/data';

const SingleScene: React.FC<{id: SceneId}> = ({id}) => <Section id={id} />;

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
        defaultProps={{id: s.id}}
        durationInFrames={sceneFrames(s)}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
    ))}
  </>
);
