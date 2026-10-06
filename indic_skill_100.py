#!/usr/bin/env python3
"""
Indic-Skill-100 (mock) dataset builder.

Stage 1  download : fetch short clips of informal trade tasks
                    (Pexels API, Pixabay API, yt-dlp for CC-licensed YouTube)
Stage 2  process  : ffmpeg -> video downsampled to 15 fps + mono 16 kHz WAV audio

Requirements:
    pip install requests yt-dlp
    ffmpeg + ffprobe on PATH
    export PEXELS_API_KEY=...   PIXABAY_API_KEY=...   (both free)

Usage:
    python indic_skill_100.py download --target 40
    python indic_skill_100.py process
    python indic_skill_100.py register --source mixkit   (after manual downloads)
    python indic_skill_100.py all --target 40
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path("indic_skill_100")
RAW = ROOT / "raw"
VID = ROOT / "video_15fps"
AUD = ROOT / "audio"
MANIFEST = ROOT / "manifest.csv"

QUERIES = {
    "plumbing": ["plumber fixing pipe", "plumbing repair tap"],
    "electrical": ["electrician wiring", "fixing electric socket"],
    "tailoring": ["tailor sewing machine", "hands stitching fabric"],
    "carpentry": ["carpenter sawing wood", "woodworking hands chisel"],
    "painting": ["painting wall roller", "house painter brush"],
    "welding": ["welding metal sparks", "welder working"],
    "masonry": ["bricklayer laying bricks", "mason plastering wall"],
    "mechanic": ["mechanic repairing engine", "hands using wrench"],
}
YT_QUERIES = {
    "plumbing": ["plumber pipe repair India"],
    "electrical": ["electrician wiring work India"],
    "tailoring": ["tailor stitching India"],
    "carpentry": ["carpenter woodwork India"],
    "painting": ["house painter wall India"],
    "welding": ["welder workshop India"],
    "masonry": ["mason bricklaying India"],
    "mechanic": ["bike mechanic repair India"],
}
MAX_SECONDS = 30
FPS = 15


# ---------- helpers ----------
def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def download_file(url, dest):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)


def log_manifest(rows):
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "task", "source", "source_id", "license", "url"])
        w.writerows(rows)


def count_raw():
    return len([p for p in RAW.glob("*") if p.suffix.lower() in VIDEO_EXTS])


# ---------- stage 1: download ----------
def from_pexels(task, query, need):
    key = os.environ.get("PEXELS_API_KEY")
    if not key or need <= 0:
        return 0
    r = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": key},
        params={"query": query, "per_page": 15, "orientation": "landscape"},
        timeout=30,
    )
    r.raise_for_status()
    got, rows = 0, []
    for v in r.json().get("videos", []):
        if got >= need:
            break
        if v["duration"] > MAX_SECONDS:
            continue
        dest = RAW / f"pexels_{v['id']}.mp4"
        if dest.exists():
            continue
        # pick the HD-ish file closest to 720p to keep sizes small
        files = [f for f in v["video_files"] if f.get("file_type") == "video/mp4" and f.get("height")]
        if not files:
            continue
        best = min(files, key=lambda f: abs(f["height"] - 720))
        download_file(best["link"], dest)
        rows.append([dest.name, task, "pexels", v["id"], "Pexels License", v["url"]])
        got += 1
    log_manifest(rows)
    return got


def from_pixabay(task, query, need):
    key = os.environ.get("PIXABAY_API_KEY")
    if not key or need <= 0:
        return 0
    r = requests.get(
        "https://pixabay.com/api/videos/",
        params={"key": key, "q": query, "per_page": 15},
        timeout=30,
    )
    r.raise_for_status()
    got, rows = 0, []
    for h in r.json().get("hits", []):
        if got >= need:
            break
        if h["duration"] > MAX_SECONDS:
            continue
        dest = RAW / f"pixabay_{h['id']}.mp4"
        if dest.exists():
            continue
        url = (h["videos"].get("medium") or h["videos"].get("small") or {}).get("url")
        if not url:
            continue
        download_file(url, dest)
        rows.append([dest.name, task, "pixabay", h["id"], "Pixabay License", h["pageURL"]])
        got += 1
    log_manifest(rows)
    return got



UA = {"User-Agent": "IndicSkill100/0.1 (research dataset; contact: ayanshmanan2007@gmail.com)"}
OPEN_MAX_SECONDS = 60
OPEN_MAX_BYTES = 60 * 1024 * 1024
VIDEO_EXTS = {".mp4", ".webm", ".ogv", ".mkv", ".mov"}


def from_commons(task, query, need):
    """Wikimedia Commons: no API key needed."""
    if need <= 0:
        return 0
    r = requests.get(
        "https://commons.wikimedia.org/w/api.php", headers=UA, timeout=30,
        params={"action": "query", "generator": "search",
                "gsrsearch": f"filetype:video {query}", "gsrnamespace": 6,
                "gsrlimit": 20, "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata", "format": "json"})
    r.raise_for_status()
    got, rows = 0, []
    for pg in r.json().get("query", {}).get("pages", {}).values():
        if got >= need:
            break
        ii = (pg.get("imageinfo") or [{}])[0]
        if not ii.get("url") or ii.get("size", 1e12) > OPEN_MAX_BYTES:
            continue
        if ii.get("duration", 0) > OPEN_MAX_SECONDS:
            continue
        ext = Path(ii["url"]).suffix.lower()
        if ext not in VIDEO_EXTS:
            continue
        dest = RAW / f"commons_{pg['pageid']}{ext}"
        if dest.exists():
            continue
        with requests.get(ii["url"], headers=UA, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(1 << 16):
                    f.write(chunk)
        lic = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "see page")
        rows.append([dest.name, task, "commons", pg["pageid"], lic, ii.get("descriptionurl", "")])
        got += 1
    log_manifest(rows)
    return got


def from_archive(task, query, need):
    """Internet Archive: Creative Commons-licensed movies, no API key needed."""
    if need <= 0:
        return 0
    r = requests.get(
        "https://archive.org/advancedsearch.php", headers=UA, timeout=30,
        params={"q": f"({query}) AND mediatype:movies AND licenseurl:*creativecommons*",
                "fl[]": "identifier", "rows": 15, "output": "json"})
    r.raise_for_status()
    got, rows = 0, []
    for d in r.json().get("response", {}).get("docs", []):
        if got >= need:
            break
        ident = d["identifier"]
        meta = requests.get(f"https://archive.org/metadata/{ident}", headers=UA, timeout=30).json()
        pick = None
        for fobj in meta.get("files", []):
            try:
                ok = (fobj["name"].lower().endswith(".mp4")
                      and float(fobj.get("length", 1e9)) <= OPEN_MAX_SECONDS
                      and int(fobj.get("size", 1e12)) <= OPEN_MAX_BYTES)
            except (ValueError, KeyError):
                ok = False
            if ok:
                pick = fobj["name"]
                break
        if not pick:
            continue
        dest = RAW / f"archive_{ident}.mp4"
        if dest.exists():
            continue
        from urllib.parse import quote
        download_file(f"https://archive.org/download/{ident}/{quote(pick)}", dest)
        lic = meta.get("metadata", {}).get("licenseurl", "CC (verify)")
        rows.append([dest.name, task, "archive.org", ident, lic, f"https://archive.org/details/{ident}"])
        got += 1
    log_manifest(rows)
    return got


def from_ytdlp(task, query, need):
    """Only Creative Commons-licensed, short YouTube videos."""
    if need <= 0:
        return 0
    before = set(RAW.glob("yt_*.mp4"))
    cmd = [
        sys.executable, "-m", "yt_dlp",
        f"ytsearch25:{query}",
        "--match-filters", f"license~='Creative Commons' & duration<={MAX_SECONDS}",
        "--max-downloads", str(need),
        "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "-o", str(RAW / "yt_%(id)s.%(ext)s"),
        "--no-playlist", "--quiet", "--no-warnings",
    ]
    run(cmd)  # exit code 101 is normal when --max-downloads is hit
    rows = []
    for p in set(RAW.glob("yt_*.mp4")) - before:
        info_path = p.with_suffix(".info.json")
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        rows.append([p.name, task, "youtube", info.get("id", p.stem[3:]),
                     info.get("license", "CC (verify)"), info.get("webpage_url", "")])
    log_manifest(rows)
    return len(rows)


def stage_download(target):
    RAW.mkdir(parents=True, exist_ok=True)
    per_task = -(-target // len(QUERIES))  # ceil
    for task, queries in QUERIES.items():
        task_count = 0
        for q in queries:
            for fn in (from_pexels, from_pixabay, from_commons, from_archive):
                try:
                    task_count += fn(task, q, per_task - task_count)
                except Exception as e:
                    print(f"[warn] {fn.__name__} '{q}': {e}")
        for q in YT_QUERIES[task]:
            try:
                task_count += from_ytdlp(task, q, per_task - task_count)
            except Exception as e:
                print(f"[warn] yt-dlp '{q}': {e}")
        print(f"{task}: {task_count} clips")
    print(f"Total raw clips: {count_raw()}")


# ---------- stage 2: process ----------
def has_audio(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)])
    return bool(r.stdout.strip())


def stage_process():
    VID.mkdir(parents=True, exist_ok=True)
    AUD.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in RAW.glob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not clips:
        sys.exit("No raw clips found. Run the download stage first.")
    for i, src in enumerate(clips, 1):
        out_v = VID / f"{src.stem}.mp4"  # always mp4 output
        out_a = AUD / f"{src.stem}.wav"
        if not out_v.exists():
            r = run(["ffmpeg", "-y", "-i", str(src),
                     "-vf", f"fps={FPS}",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                     "-an", str(out_v)])
            if r.returncode:
                print(f"[fail video] {src.name}: {r.stderr[-200:]}")
        if not out_a.exists():
            if has_audio(src):
                r = run(["ffmpeg", "-y", "-i", str(src), "-vn",
                         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_a)])
                if r.returncode:
                    print(f"[fail audio] {src.name}: {r.stderr[-200:]}")
            else:
                print(f"[no audio] {src.name}")
        print(f"[{i}/{len(clips)}] {src.name}")



# ---------- manual downloads ----------
LICENSES = {"pixabay": "Pixabay License", "mixkit": "Mixkit License",
            "coverr": "Coverr License", "pexels": "Pexels License"}


def stage_register(source):
    """Move clips from manual/<task>/*.mp4 into raw/ and log them."""
    import shutil
    RAW.mkdir(parents=True, exist_ok=True)
    manual = ROOT / "manual"
    rows = []
    for task_dir in sorted(p for p in manual.glob("*") if p.is_dir()):
        for f in task_dir.glob("*.mp4"):
            dest = RAW / f"{source}_{task_dir.name}_{f.stem}.mp4"
            shutil.move(str(f), dest)
            rows.append([dest.name, task_dir.name, source, f.stem,
                         LICENSES.get(source, "check source"), ""])
    log_manifest(rows)
    print(f"Registered {len(rows)} clips from '{source}'")


# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["download", "process", "all", "register"])
    ap.add_argument("--target", type=int, default=40, help="clips to download (30-50)")
    ap.add_argument("--source", default="manual", help="for register: pixabay/mixkit/coverr/pexels")
    a = ap.parse_args()
    if a.stage == "register":
        stage_register(a.source)
    if a.stage in ("download", "all"):
        stage_download(a.target)
    if a.stage in ("process", "all"):
        stage_process()