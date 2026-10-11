// Линейный таймлайн: сцены идут встык, длительности — из SCENES.

import {AbsoluteFill, Sequence} from 'remotion';
import {SCENES, sceneFrames, sceneStarts} from './data';
import {Section} from './Sections';

export const DailyBench: React.FC = () => {
  const starts = sceneStarts();
  return (
    <AbsoluteFill style={{backgroundColor: '#05080f'}}>
      {SCENES.map((s, i) => (
        <Sequence
          key={s.id}
          from={starts[i].from}
          durationInFrames={sceneFrames(s)}
          name={`${s.tc} · ${s.id}`}
        >
          <Section id={s.id} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
