// Монтаж под записанную речь. edit.json — единый источник абсолютных кадров.
import edit from '../data/edit.json';

/** Мастер-клипы 120 fps показываются без ускорения в композиции 60 fps. */
export const FPS = edit.fps;
export const WIDTH = edit.width;
export const HEIGHT = edit.height;

export type SceneId =
  | 'intro'
  | 'harness'
  | 'suite'
  | 'models'
  | 'board'
  | 'showcase-intro'
  | 'hero'
  | 'showcase'
  | 'verdict'
  | 'outro';

export type SceneSpec = {
  id: SceneId;
  seconds: number;
  tc: string;
  frame: string;
  lines: string[];
};

export const timecode = (frame: number): string => {
  const seconds = Math.floor(frame / FPS);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
};

export const SCENES: SceneSpec[] = edit.scenes.map((scene) => ({
  id: scene.id as SceneId,
  seconds: scene.duration / FPS,
  tc: `${timecode(scene.from)}–${timecode(scene.from + scene.duration)}`,
  frame: scene.title,
  lines: edit.narration.filter((item) => item.scene === scene.id).map((item) => item.text),
}));

export const sceneFrames = (scene: SceneSpec): number => Math.round(scene.seconds * FPS);

export const sceneStarts = (): {id: SceneId; from: number; duration: number; label: string}[] =>
  edit.scenes.map((scene) => ({
    id: scene.id as SceneId,
    from: scene.from,
    duration: scene.duration,
    label: `${timecode(scene.from)} · ${scene.title}`,
  }));

export const totalFrames = (): number => edit.totalFrames;
