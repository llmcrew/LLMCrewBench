// Диспетчер сцен по id.

import type {SceneId} from './data';
import {Intro, HarnessDef, Suite, Models} from '../scenes/Theory';
import {Board} from '../scenes/Board';
import {ShowcaseReel} from '../scenes/ShowcaseReel';
import {Hero, Verdict, Outro} from '../scenes/Final';

export const Section: React.FC<{id: SceneId}> = ({id}) => {
  switch (id) {
    case 'intro':
      return <Intro />;
    case 'harness':
      return <HarnessDef />;
    case 'suite':
      return <Suite />;
    case 'models':
      return <Models />;
    case 'board':
      return <Board />;
    case 'hero':
      return (
        <Hero
          clip="clips/deepseek-v4.1-flash-unc__K-1.mp4"
          caption="K-1 — баллистика: три снаряда под гравитацией"
          model="DeepSeek 4.1 Flash · 4/4"
        />
      );
    case 'showcase':
      return <ShowcaseReel />;
    case 'verdict':
      return <Verdict />;
    case 'outro':
      return <Outro />;
  }
};
