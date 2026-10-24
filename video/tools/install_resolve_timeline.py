#!/usr/bin/env python3
"""Install the finished 60-fps edit as a new Resolve timeline; preserve the recording."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

VIDEO = Path(__file__).resolve().parents[1]


def connect():
    api = os.environ.get("RESOLVE_SCRIPT_API")
    if api:
        modules = Path(api) / "Modules"
    elif sys.platform == "win32":
        modules = Path(os.environ["PROGRAMDATA"]) / "Blackmagic Design/DaVinci Resolve/Support/Developer/Scripting/Modules"
    else:
        raise RuntimeError("Set RESOLVE_SCRIPT_API to the installed Resolve Developer/Scripting directory")
    sys.path.insert(0, str(modules))
    resolve = importlib.import_module("DaVinciResolveScript").scriptapp("Resolve")
    if resolve is None:
        raise RuntimeError("Open DaVinci Resolve with external scripting enabled")
    return resolve


def audio_snapshot(timeline):
    return [
        (track, item.GetName(), item.GetStart(), item.GetEnd(), item.GetLeftOffset(), item.GetRightOffset())
        for track in range(1, timeline.GetTrackCount("audio") + 1)
        for item in timeline.GetItemListInTrack("audio", track) or []
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=VIDEO / "out/edit/timeline.json")
    parser.add_argument("--name", default="Flash vs 1017 — final — 60fps")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    for path in {cut["file"] for cut in plan["cuts"]} | set(plan["audioFiles"].values()) | {plan["finalFile"]}:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Render the complete edit first: {path}")

    resolve = connect()
    manager = resolve.GetProjectManager()
    project = manager.GetCurrentProject()
    if project is None or project.GetName() != plan["source"]["project"]:
        raise RuntimeError(f'Open the original project {plan["source"]["project"]!r} first')
    timelines = [project.GetTimelineByIndex(index) for index in range(1, project.GetTimelineCount() + 1)]
    if any(timeline.GetName() == args.name for timeline in timelines):
        raise RuntimeError("This final timeline already exists; select a new --name to preserve it")
    original = next(timeline for timeline in timelines if timeline.GetName() == plan["source"]["timeline"])
    before = audio_snapshot(original)
    exports = VIDEO / "out/resolve"
    exports.mkdir(parents=True, exist_ok=True)
    backup = VIDEO / plan["source"]["backup"]
    if not backup.exists() and not manager.ExportProject(project.GetName(), str(backup), True):
        raise RuntimeError("Could not back up the original project")

    pool = project.GetMediaPool()
    folder = pool.AddSubFolder(pool.GetRootFolder(), args.name)
    if folder is None or not pool.SetCurrentFolder(folder):
        raise RuntimeError("Could not create a media folder for the final edit")
    timeline = pool.CreateEmptyTimeline(args.name)
    if timeline is None:
        raise RuntimeError("Could not create the new timeline")
    if not timeline.SetSettings({"useCustomSettings": "1"}):
        raise RuntimeError("Could not enable timeline-specific settings")
    settings = {
        "timelineFrameRate": str(plan["fps"]),
        "timelineResolutionWidth": str(plan["width"]),
        "timelineResolutionHeight": str(plan["height"]),
    }
    if not timeline.SetSettings(settings) or float(timeline.GetSettings()["timelineFrameRate"]) != plan["fps"]:
        raise RuntimeError("Resolve did not accept the required 60-fps timeline")
    if not timeline.SetStartTimecode("01:00:00:00") or not project.SetCurrentTimeline(timeline):
        raise RuntimeError("Could not activate the final timeline")
    start = int(timeline.GetStartFrame())
    for kind, count in (("video", 1), ("audio", 2)):
        while timeline.GetTrackCount(kind) < count:
            if not timeline.AddTrack(kind, "stereo" if kind == "audio" else None):
                raise RuntimeError(f"Could not add {kind} track")
    timeline.SetTrackName("video", 1, "Picture — scenes and 49 comparisons")
    timeline.SetTrackName("audio", 1, "Narration — recorded voice")
    timeline.SetTrackName("audio", 2, "Music — instrumental score")

    imported = {}

    def media(path):
        if path not in imported:
            clips = pool.ImportMedia([str(Path(path).resolve())])
            if not clips:
                raise RuntimeError(f"Resolve did not import {path}")
            imported[path] = clips[0]
        return imported[path]

    # endFrame is exclusive here; subtracting one shortens every cut by a frame.
    for cut in plan["cuts"]:
        items = pool.AppendToTimeline([{
            "mediaPoolItem": media(cut["file"]),
            "startFrame": cut["sourceStart"],
            "endFrame": cut["sourceStart"] + cut["duration"],
            "mediaType": 1,
            "trackIndex": 1,
            "recordFrame": start + cut["from"],
        }])
        if not items or int(items[0].GetStart()) != start + cut["from"] or int(items[0].GetDuration()) != cut["duration"]:
            raise RuntimeError(f'Resolve changed a video cut: {cut["id"]}')
        if not timeline.AddMarker(cut["from"], "Cyan" if cut["id"].startswith("K-") else "Purple" if cut["id"].startswith("L-") else "Blue",
                                  cut["title"], "Music only" if cut["id"].startswith(("K-", "L-")) else cut["id"], 1, cut["id"]):
            raise RuntimeError(f'Could not add marker: {cut["id"]}')

    for track, key in ((1, "narration"), (2, "music")):
        items = pool.AppendToTimeline([{"mediaPoolItem": media(plan["audioFiles"][key]), "mediaType": 2,
                                        "trackIndex": track, "recordFrame": start}])
        if not items or int(items[0].GetStart()) != start or int(items[0].GetDuration()) != plan["totalFrames"]:
            raise RuntimeError(f"Resolve changed the {key} stem duration")
    # Keep the ready-to-watch master in the media pool, not on another audio track.
    media(plan["finalFile"])
    if int(timeline.GetEndFrame()) - start != plan["totalFrames"]:
        raise RuntimeError("Final timeline duration does not match the rendered edit")
    if audio_snapshot(original) != before:
        raise RuntimeError("The original recording timeline unexpectedly changed")

    if not resolve.OpenPage("edit") or not timeline.SetCurrentTimecode("01:00:08:00"):
        raise RuntimeError("Could not open the completed edit for preview")
    drt = exports / "Flash-vs-1017-final-60fps.drt"
    if not timeline.Export(str(drt), resolve.EXPORT_DRT, resolve.EXPORT_NONE):
        raise RuntimeError("Resolve did not export the final DRT")
    if not manager.SaveProject():
        raise RuntimeError("Resolve did not save the final project")
    drp = exports / "flashVS1017-final.drp"
    if not manager.ExportProject(project.GetName(), str(drp), True):
        raise RuntimeError("Resolve did not export the final project")
    image = exports / "final-timeline.png"
    still_exported = project.ExportCurrentFrameAsStill(str(image))
    report = {
        "project": project.GetName(), "timeline": timeline.GetName(), "fps": plan["fps"],
        "page": resolve.GetCurrentPage(), "timecode": timeline.GetCurrentTimecode(),
        "startFrame": start, "endFrame": int(timeline.GetEndFrame()), "durationFrames": plan["totalFrames"],
        "videoCuts": len(timeline.GetItemListInTrack("video", 1) or []),
        "markers": len(timeline.GetMarkers() or {}),
        "narrationTrack": timeline.GetTrackName("audio", 1), "musicTrack": timeline.GetTrackName("audio", 2),
        "originalTimelinePreserved": audio_snapshot(original) == before,
        "drt": str(drt), "drp": str(drp), "still": str(image) if still_exported else None,
    }
    (exports / "timeline-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
