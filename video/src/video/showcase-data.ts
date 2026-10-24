import showcase from '../data/showcase.json';
import resultNotes from '../data/result-notes.json';
import {buildShowcaseTimeline, type ResultNote, type ShowcaseEntry} from './showcase-timing';

const notesByRun = new Map(
  (resultNotes as ResultNote[]).map(({model, scene, ...note}) => [`${model}\0${scene}`, note] as const),
);

export const showcaseTimeline = buildShowcaseTimeline(
  (showcase as unknown as ShowcaseEntry[]).map((entry) => ({
    ...entry,
    runs: entry.runs.map((run) => {
      const resultNote = notesByRun.get(`${run.model}\0${entry.scene}`);
      return resultNote ? {...run, resultNote} : run;
    }),
  })),
);

export const showcaseSeconds = showcaseTimeline.reduce((total, entry) => total + entry.seconds, 0);
