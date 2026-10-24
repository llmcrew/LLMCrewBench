// Все сцены сравнения; полноэкранные ходы выделены только доступным результатам.

import {AbsoluteFill, Sequence} from 'remotion';
import {ShowcaseScene} from './Showcase';
import {FPS} from '../video/data';
import {C} from '../video/theme';
import {showcaseTimeline} from '../video/showcase-data';

export const ShowcaseReel: React.FC<{overlayOnly?: boolean}> = ({overlayOnly = false}) => (
  <AbsoluteFill style={{backgroundColor: overlayOnly ? 'transparent' : C.bg}}>
    {showcaseTimeline.map((s) => (
      <Sequence
        key={s.scene}
        from={Math.round(s.from * FPS)}
        durationInFrames={Math.round(s.seconds * FPS)}
      >
        <ShowcaseScene
          sceneId={s.scene}
          title={s.title}
          runs={s.runs}
          order={s.order}
          seconds={s.seconds}
          overlayOnly={overlayOnly}
        />
      </Sequence>
    ))}
  </AbsoluteFill>
);
