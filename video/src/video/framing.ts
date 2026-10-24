import type {CSSProperties} from 'react';
import regions from '../data/framing.json';
import {WIDTH, HEIGHT} from './data';

const framingByClip: Record<string, (typeof regions)[number]> = Object.fromEntries(
  regions.map((region) => [region.clip, region]),
);
const FULL_FRAME = {x: 0, y: 0, width: WIDTH, height: HEIGHT};

/** Presentation only: centre sparse K results without rewriting source MP4 files. */
export const videoFramingStyle = (clip: string, width: number, height: number): CSSProperties => {
  const region = framingByClip[clip];
  const source = region ?? FULL_FRAME;
  // Keep the sampled K content visible in a narrow curtain. Full-frame artwork
  // retains the reference video's centred cover crop rather than being reframed.
  const scale = region
    ? Math.min(width / source.width, height / source.height)
    : Math.max(width / source.width, height / source.height);
  return {
    position: 'absolute',
    left: (width - source.width * scale) / 2 - source.x * scale,
    top: (height - source.height * scale) / 2 - source.y * scale,
    width: WIDTH * scale,
    height: HEIGHT * scale,
    maxWidth: 'none',
    objectFit: 'fill',
  };
};
