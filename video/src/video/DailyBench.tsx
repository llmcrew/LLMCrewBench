// Записанная речь, музыкальная витрина и визуальные сцены на общем таймлайне.

import {AbsoluteFill, Audio, Sequence, staticFile} from 'remotion';
import {SCENES, sceneFrames, sceneStarts} from './data';
import {Section} from './Sections';
import edit from '../data/edit.json';

export const DailyBench: React.FC = () => {
  const starts = sceneStarts();
  return (
    <AbsoluteFill style={{backgroundColor: '#05080f'}}>
      <Audio src={staticFile(edit.audio.mix)} />
      {SCENES.map((s, i) => (
        <Sequence
          key={s.id}
          from={starts[i].from}
          durationInFrames={sceneFrames(s)}
          name={starts[i].label}
        >
          <Section id={s.id} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
