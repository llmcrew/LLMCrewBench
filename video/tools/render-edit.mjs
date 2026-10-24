#!/usr/bin/env node
// Render real motion in Remotion; encode only genuinely static holds from stills.
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {bundle, BundlerInternals} from '@remotion/bundler';
import {renderMedia, renderStill, selectComposition} from '@remotion/renderer';

const VIDEO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(VIDEO, 'out', 'edit');
const EDIT_PATH = join(VIDEO, 'src', 'data', 'edit.json');
const edit = JSON.parse(readFileSync(EDIT_PATH, 'utf8'));
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const at = args.indexOf(name);
  return at < 0 ? fallback : args[at + 1];
};
const browserExecutable = option('--browser-executable', process.env.REMOTION_BROWSER_EXECUTABLE);
const concurrency = Number(option('--concurrency', '4'));
const narrativeOnly = args.includes('--narrative-only');
const assembleOnly = args.includes('--assemble-only');
const python = process.env.PYTHON ?? (process.platform === 'win32' ? 'python' : 'python3');
const finalFile = resolve(option('--output', join(VIDEO, 'out', 'DailyBench-voiceover-final.mp4')));
const showcaseFile = join(VIDEO, 'out', 'showcase-master.mp4');
mkdirSync(OUT, {recursive: true});

function run(command, argv, capture = false) {
  const result = spawnSync(command, argv, {
    cwd: VIDEO,
    stdio: capture ? 'pipe' : 'inherit',
    encoding: capture ? 'utf8' : undefined,
    maxBuffer: 8 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} exited ${result.status}: ${result.stderr ?? ''}`);
  return result.stdout;
}

function probe(file) {
  return JSON.parse(run('ffprobe', ['-v', 'error', '-show_entries',
    'stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,time_base:format=duration', '-of', 'json', file], true));
}

function checkVideo(file, frames) {
  const video = probe(file).streams.find((stream) => stream.codec_type === 'video');
  if (!video || video.codec_name !== 'h264' || video.width !== edit.width || video.height !== edit.height
      || video.avg_frame_rate !== `${edit.fps}/1` || video.time_base !== '1/90000' || Number(video.nb_frames) !== frames) {
    throw new Error(`Video does not match the edit (${frames} frames): ${file}\n${JSON.stringify(video)}`);
  }
}

function complete(file, count) {
  if (!existsSync(file)) return false;
  try { checkVideo(file, count); return true; } catch { return false; }
}

const fingerprint = createHash('sha256');
function hashSource(folder) {
  for (const item of readdirSync(folder, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
    const path = join(folder, item.name);
    if (item.isDirectory()) hashSource(path);
    else fingerprint.update(path).update(readFileSync(path));
  }
}
hashSource(join(VIDEO, 'src'));
fingerprint.update(readFileSync(fileURLToPath(import.meta.url)));
const cache = join(OUT, 'cache', fingerprint.digest('hex').slice(0, 16));
mkdirSync(cache, {recursive: true});
const concatPath = (path) => `file '${path.replaceAll('\\', '/').replaceAll("'", "'\\''")}'`;
function concat(files, target) {
  const list = `${target}.concat.txt`;
  writeFileSync(list, files.map(concatPath).join('\n') + '\n');
  run('ffmpeg', ['-v', 'warning', '-nostdin', '-y', '-f', 'concat', '-safe', '0', '-i', list,
    '-map', '0:v:0', '-c:v', 'copy', '-an', '-video_track_timescale', '90000', '-movflags', '+faststart', target]);
}

let serveUrl;
async function renderNarrative(scene) {
  const target = join(cache, `${scene.id}.mp4`);
  if (complete(target, scene.duration)) {
    console.log(`Reusing ${scene.id}: ${scene.duration} frames`);
    return target;
  }
  const composition = await selectComposition({serveUrl, id: `scene-${scene.id}`, browserExecutable});
  if (composition.durationInFrames !== scene.duration) throw new Error(`Duration drift: ${scene.id}`);
  const spans = [];
  let at = 0;
  for (const [start, end] of scene.motionRanges) {
    if (start < at || end <= start || end > scene.duration) throw new Error(`Invalid motion range: ${scene.id}`);
    if (start > at) spans.push({start: at, end: start, motion: false});
    spans.push({start, end, motion: true});
    at = end;
  }
  if (at < scene.duration) spans.push({start: at, end: scene.duration, motion: false});
  const parts = [];
  for (const span of spans) {
    const part = join(cache, `${scene.id}-${span.start}-${span.end}.mp4`);
    const count = span.end - span.start;
    if (!complete(part, count)) {
      console.log(`${scene.id}: ${span.motion ? 'motion' : 'static hold'} ${span.start}–${span.end}`);
      if (span.motion) {
        let reported = -1;
        await renderMedia({
          serveUrl, composition, browserExecutable, outputLocation: part, codec: 'h264',
          crf: 18, pixelFormat: 'yuv420p', colorSpace: 'bt709', x264Preset: 'veryfast', muted: true,
          concurrency, frameRange: [span.start, span.end - 1], timeoutInMilliseconds: 180000,
          onProgress: ({progress}) => {
            const percent = Math.floor(progress * 10) * 10;
            if (percent !== reported) { reported = percent; console.log(`${scene.id} ${span.start}: ${percent}%`); }
          },
        });
      } else {
        const still = join(cache, `${scene.id}-${span.start}.png`);
        await renderStill({serveUrl, composition, browserExecutable, output: still, frame: span.start, imageFormat: 'png'});
        run('ffmpeg', ['-v', 'warning', '-nostdin', '-y', '-loop', '1', '-framerate', String(edit.fps), '-i', still,
          '-frames:v', String(count), '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
          '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-pix_fmt', 'yuv420p',
          '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
          '-video_track_timescale', '90000', part]);
      }
    }
    checkVideo(part, count);
    parts.push(part);
  }
  concat(parts, target);
  checkVideo(target, scene.duration);
  return target;
}

// Bundle the real primary timing rules; never keep another K/L eligibility table.
async function showcaseTimeline() {
  const built = await BundlerInternals.esbuild.build({
    entryPoints: [join(VIDEO, 'src', 'video', 'showcase-data.ts')], bundle: true,
    platform: 'node', format: 'esm', write: false,
  });
  return (await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`)).showcaseTimeline;
}

function clock(frame, frames = false) {
  const whole = Math.floor(frame / edit.fps);
  const h = Math.floor(whole / 3600), m = Math.floor(whole / 60) % 60, s = whole % 60;
  const base = `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return frames ? `${base}:${String(frame % edit.fps).padStart(2, '0')}` : base;
}

async function chaptersAndParts(files) {
  const timeline = await showcaseTimeline();
  const showcase = edit.scenes.find((scene) => scene.id === 'showcase');
  if (timeline.reduce((sum, item) => sum + Math.round(item.seconds * edit.fps), 0) !== showcase.duration) {
    throw new Error('Showcase timing changed: rebuild edit.json before rendering');
  }
  const chapters = [], cuts = [];
  for (let index = 0; index < edit.scenes.length; index++) {
    const scene = edit.scenes[index];
    if (scene.id === 'showcase') {
      for (const item of timeline) {
        const from = scene.from + Math.round(item.from * edit.fps);
        const duration = Math.round(item.seconds * edit.fps);
        const row = {id: item.scene, title: item.title, from, duration, file: files[index], sourceStart: Math.round(item.from * edit.fps)};
        chapters.push(row); cuts.push(row);
      }
    } else {
      const row = {id: scene.id, title: scene.title, from: scene.from, duration: scene.duration, file: files[index], sourceStart: 0};
      chapters.push(row); cuts.push(row);
    }
  }
  const metadata = [';FFMETADATA1', 'title=DeepSeek Flash vs 1017 · LLMCrewBench',
    ...chapters.flatMap((item) => ['[CHAPTER]', 'TIMEBASE=1/60', `START=${item.from}`, `END=${item.from + item.duration}`, `title=${item.title.replace(/[=;#\\\n]/g, ' ')}`])].join('\n') + '\n';
  const metadataPath = join(OUT, 'chapters.ffmeta');
  writeFileSync(metadataPath, metadata);
  writeFileSync(join(OUT, 'chapters.txt'), chapters.map((item) => `${clock(item.from)} ${item.title}`).join('\n') + '\n');
  const csv = (value) => `"${String(value).replaceAll('"', '""')}"`;
  writeFileSync(join(OUT, 'timeline.csv'), ['id,title,in,out,duration_frames,source',
    ...cuts.map((item) => [item.id, item.title, clock(item.from, true), clock(item.from + item.duration, true), item.duration, item.file].map(csv).join(','))].join('\n') + '\n');
  const plan = {...edit, finalFile, cuts, audioFiles: {
    narration: join(VIDEO, 'public', edit.audio.narration),
    music: join(VIDEO, 'public', edit.audio.music),
    mix: join(VIDEO, 'public', edit.audio.mix),
  }};
  writeFileSync(join(OUT, 'timeline.json'), JSON.stringify(plan, null, 2) + '\n');
  return {metadataPath, plan};
}

if (!assembleOnly) {
  run(python, ['tools/preflight.py']);
  if (!narrativeOnly) run(python, ['tools/build_audio.py']);
  serveUrl = option('--serve-url', undefined) ?? await bundle({entryPoint: join(VIDEO, 'src', 'index.ts'), outDir: join(cache, 'bundle')});
}
const files = [];
for (const scene of edit.scenes) {
  if (scene.id === 'showcase') {
    if (!narrativeOnly && !assembleOnly) {
      const command = [join(VIDEO, 'tools', 'render-showcase.mjs'), '--serve-url', serveUrl, '--output', showcaseFile,
        '--concurrency', String(concurrency)];
      if (browserExecutable) command.push('--browser-executable', browserExecutable);
      run(process.execPath, command);
    }
    files.push(showcaseFile);
  } else {
    files.push(assembleOnly ? join(cache, `${scene.id}.mp4`) : await renderNarrative(scene));
  }
}
if (narrativeOnly) {
  console.log(`Narrative sections ready in ${cache}`);
} else {
  for (let i = 0; i < files.length; i++) checkVideo(files[i], edit.scenes[i].duration);
  const {metadataPath, plan} = await chaptersAndParts(files);
  const picture = join(OUT, 'picture-master.mp4');
  concat(files, picture);
  checkVideo(picture, edit.totalFrames);
  run('ffmpeg', ['-v', 'warning', '-nostdin', '-y', '-i', picture, '-i', plan.audioFiles.mix,
    '-f', 'ffmetadata', '-i', metadataPath, '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '2', '-map_chapters', '2',
    '-c:v', 'copy', '-c:a', 'copy', '-t', String(edit.totalFrames / edit.fps),
    '-movflags', '+faststart', finalFile]);
  checkVideo(finalFile, edit.totalFrames);
  const report = probe(finalFile);
  if (!report.streams.some((stream) => stream.codec_type === 'audio' && stream.codec_name === 'aac')) throw new Error('Missing final audio');
  writeFileSync(join(OUT, 'render-verification.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(`Final: ${finalFile}\nTimeline: ${join(OUT, 'timeline.json')}`);
}
