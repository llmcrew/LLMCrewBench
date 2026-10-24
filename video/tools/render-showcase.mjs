#!/usr/bin/env node
// The curtains stay in Remotion. Fixed full-screen plateaus use the same crop,
// rate and a transparent PNG of the real React captions over the original video.
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {bundle, BundlerInternals} from '@remotion/bundler';
import {openBrowser, renderMedia, renderStill, selectComposition} from '@remotion/renderer';

const VIDEO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const option = (name, fallback) => argv.includes(name) ? argv[argv.indexOf(name) + 1] : fallback;
const chrome = option('--browser-executable', process.env.REMOTION_BROWSER_EXECUTABLE);
const concurrency = Number(option('--concurrency', '8'));
const output = resolve(option('--output', join(VIDEO, 'out/showcase-master.mp4')));
const only = option('--only', '').split(',').filter(Boolean);
const force = argv.includes('--force');
const FPS = 60, W = 1920, H = 1080;

export async function loadShowcasePlan() {
  const result = await BundlerInternals.esbuild.build({
    stdin: {contents: `export {showcaseTimeline, showcaseSeconds} from './src/video/showcase-data.ts';
      export {presentationClip, CLIP_SECONDS, OPEN_SECONDS, SHOWCASE_LEAD, SHOWCASE_TURN_SECONDS} from './src/video/showcase-timing.ts';
      export {videoFramingStyle} from './src/video/framing.ts';`, resolveDir: VIDEO, loader: 'ts'},
    bundle: true, platform: 'node', format: 'esm', write: false,
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

function run(command, args, capture = false) {
  const result = spawnSync(command, args, {cwd: VIDEO, stdio: capture ? 'pipe' : 'inherit', encoding: capture ? 'utf8' : undefined, maxBuffer: 8 * 1024 * 1024});
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} exited ${result.status}: ${result.stderr ?? ''}`);
  return result.stdout;
}

function validate(file, count) {
  const meta = JSON.parse(run('ffprobe', ['-v', 'error', '-show_entries',
    'stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,time_base:format=duration', '-of', 'json', file], true));
  const stream = meta.streams.find((item) => item.codec_type === 'video');
  if (!stream || meta.streams.some((item) => item.codec_type === 'audio') || stream.codec_name !== 'h264'
      || stream.width !== W || stream.height !== H || stream.pix_fmt !== 'yuv420p'
      || stream.avg_frame_rate !== '60/1' || stream.time_base !== '1/90000' || Number(stream.nb_frames) !== count
      || Math.abs(Number(meta.format.duration) - count / FPS) > 1 / FPS) {
    throw new Error(`Invalid rendered segment: ${file}\n${JSON.stringify(meta)}`);
  }
}

function complete(file, count) {
  if (!existsSync(file)) return false;
  try { validate(file, count); return true; } catch { return false; }
}

function concat(files, target, count) {
  const list = `${target}.concat.txt`;
  writeFileSync(list, files.map((file) => `file '${file.replaceAll('\\', '/').replaceAll("'", "'\\''")}'`).join('\n') + '\n');
  run('ffmpeg', ['-v', 'warning', '-nostdin', '-y', '-f', 'concat', '-safe', '0', '-i', list,
    '-map', '0:v:0', '-c:v', 'copy', '-an', '-video_track_timescale', '90000', '-movflags', '+faststart', target]);
  validate(target, count);
}

function fingerprint(plan) {
  const hash = createHash('sha256');
  function walk(folder) {
    for (const item of readdirSync(folder, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = join(folder, item.name);
      if (item.isDirectory()) walk(file); else hash.update(file).update(readFileSync(file));
    }
  }
  walk(join(VIDEO, 'src'));
  hash.update(readFileSync(fileURLToPath(import.meta.url)));
  for (const clip of new Set(plan.showcaseTimeline.flatMap((scene) => scene.runs.map(plan.presentationClip).filter(Boolean)))) {
    const info = statSync(join(VIDEO, 'public', clip));
    hash.update(clip).update(String(info.size)).update(String(info.mtimeMs));
  }
  return hash.digest('hex').slice(0, 16);
}

function framingFilter(style) {
  const width = Math.round(Number(style.width)), height = Math.round(Number(style.height));
  const left = Math.round(Number(style.left)), top = Math.round(Number(style.top));
  const x = Math.max(0, -left), y = Math.max(0, -top);
  const padX = Math.max(0, left), padY = Math.max(0, top);
  const cropW = Math.min(W - padX, width - x), cropH = Math.min(H - padY, height - y);
  return `format=rgb24,scale=${width}:${height}:flags=lanczos,crop=${cropW}:${cropH}:${x}:${y},pad=${W}:${H}:${padX}:${padY}:color=0x05080f,format=rgba`;
}

async function main() {
  if (argv.includes('--help')) {
    console.log('node tools/render-showcase.mjs [--browser-executable <chrome>] [--serve-url <bundle>] [--output <mp4>] [--concurrency 8] [--only K-1,L-2] [--force]');
    return;
  }
  const plan = await loadShowcasePlan();
  const scenes = plan.showcaseTimeline.filter((scene) => !only.length || only.includes(scene.scene));
  if (!scenes.length || only.some((id) => !scenes.some((scene) => scene.scene === id))) throw new Error('Unknown or empty showcase selection');
  const cache = join(VIDEO, 'out/showcase-master.parts', fingerprint(plan));
  mkdirSync(cache, {recursive: true});
  mkdirSync(dirname(output), {recursive: true});
  const serveUrl = option('--serve-url', null) ?? await bundle({entryPoint: join(VIDEO, 'src/index.ts'), outDir: join(cache, 'bundle')});
  const browser = await openBrowser('chrome', {browserExecutable: chrome, logLevel: 'warn'});
  try {
    const composition = await selectComposition({serveUrl, id: 'scene-showcase', browserExecutable: chrome, puppeteerInstance: browser});
    if (composition.fps !== FPS || composition.width !== W || composition.height !== H
        || composition.durationInFrames !== Math.round(plan.showcaseSeconds * FPS)) throw new Error('Showcase composition drift');
    const files = [];
    const metadata = [];
    for (const scene of scenes) {
      const sceneStart = Math.round(scene.from * FPS), sceneCount = Math.round(scene.seconds * FPS);
      const sceneFile = join(cache, `${scene.scene}.mp4`);
      if (force || !complete(sceneFile, sceneCount)) {
        const pieces = [];
        let cursor = sceneStart;
        async function motion(start, end) {
          if (end <= start) return;
          const file = join(cache, `${scene.scene}-${start}-${end}-motion.mp4`);
          if (force || !complete(file, end - start)) {
            console.log(`${scene.scene}: real curtains ${start - sceneStart}–${end - sceneStart}`);
            let reported = -1;
            await renderMedia({composition, serveUrl, browserExecutable: chrome, puppeteerInstance: browser,
              inputProps: {id: 'showcase', overlayOnly: false}, codec: 'h264', outputLocation: file,
              frameRange: [start, end - 1], concurrency, muted: true, crf: 18, pixelFormat: 'yuv420p',
              colorSpace: 'bt709', x264Preset: 'veryfast', timeoutInMilliseconds: 180000,
              onProgress: ({progress}) => {const p = Math.floor(progress * 4) * 25; if (p !== reported) {reported = p; console.log(`${scene.scene}: ${p}%`);}},
            });
          }
          validate(file, end - start); pieces.push(file);
        }
        for (let turn = 0; turn < scene.order.length; turn++) {
          const openStart = Math.round((turn * plan.SHOWCASE_TURN_SECONDS + plan.SHOWCASE_LEAD) * FPS);
          const openEnd = openStart + Math.round(plan.OPEN_SECONDS * FPS);
          // Outside these limits the real caption is fading, so keep those frames in Remotion.
          const start = sceneStart + openStart + 16, end = sceneStart + openEnd - 8;
          await motion(cursor, start);
          const runInfo = scene.runs[scene.order[turn]];
          const clip = plan.presentationClip(runInfo);
          if (!clip) throw new Error(`No original presentation clip for ${scene.scene}`);
          const file = join(cache, `${scene.scene}-${turn}-plateau.mp4`);
          if (force || !complete(file, end - start)) {
            const overlay = join(cache, `${scene.scene}-${turn}-overlay.png`);
            await renderStill({composition: {...composition, props: {...composition.props, id: 'showcase', overlayOnly: true}},
              serveUrl, browserExecutable: chrome, puppeteerInstance: browser,
              inputProps: {id: 'showcase', overlayOnly: true}, frame: start, output: overlay, imageFormat: 'png'});
            const returnFrames = Math.max(2, Math.round(plan.SHOWCASE_LEAD * FPS * 0.5));
            const heroFrames = Math.round(plan.OPEN_SECONDS * FPS) + returnFrames;
            const openRate = plan.CLIP_SECONDS / (heroFrames / FPS);
            const trimStart = start - sceneStart - openStart;
            const count = end - start;
            const geometry = framingFilter(plan.videoFramingStyle(clip, W, H));
            const graph = `[0:v]setpts=(PTS-STARTPTS)/${openRate},fps=${FPS},trim=start_frame=${trimStart}:end_frame=${trimStart + count},setpts=PTS-STARTPTS,${geometry}[base];[base][1:v]overlay=0:0:format=auto:shortest=1,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[out]`;
            console.log(`${scene.scene}: original live video, full-screen turn ${turn + 1}`);
            run('ffmpeg', ['-v', 'warning', '-nostdin', '-y', '-i', join(VIDEO, 'public', clip),
              '-loop', '1', '-framerate', String(FPS), '-i', overlay, '-filter_complex_threads', '2', '-filter_complex', graph,
              '-map', '[out]', '-frames:v', String(count), '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
              '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
              '-color_range', 'tv', '-video_track_timescale', '90000', file]);
          }
          validate(file, end - start); pieces.push(file); cursor = end;
        }
        await motion(cursor, sceneStart + sceneCount);
        concat(pieces, sceneFile, sceneCount);
      }
      validate(sceneFile, sceneCount);
      files.push(sceneFile);
      metadata.push({scene: scene.scene, title: scene.title, from: sceneStart, duration: sceneCount, file: sceneFile, turns: scene.order.length});
      console.log(`Completed ${scene.scene}: ${sceneCount} frames`);
    }
    const count = metadata.reduce((sum, scene) => sum + scene.duration, 0);
    concat(files, output, count);
    writeFileSync(`${output}.json`, JSON.stringify({fps: FPS, width: W, height: H, totalFrames: count,
      durationSeconds: count / FPS, scenes: metadata}, null, 2) + '\n');
    console.log(`Finished ${output}: ${count} frames, ${count / FPS} seconds`);
  } finally {
    await browser.close({silent: true});
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {console.error(error.stack ?? error); process.exitCode = 1;});
}
