#!/usr/bin/env python3
"""
yt_collect.py - collect short task clips from YouTube (and any site yt-dlp supports)
into the Indic-Skill-100 folders, logging every clip in manifest.csv.

Requirements:  pip install -U yt-dlp     +  ffmpeg on PATH

Examples (any folder; output always goes next to this script):
  py yt_collect.py                                   # all trades, built-in Indic queries
  py yt_collect.py --tasks tailoring welding         # only some trades
  py yt_collect.py --per-query 6                     # more clips per search phrase
  py yt_collect.py --clip-length 20 --clip-start 10  # 20-second clips starting at 0:10
  py yt_collect.py --task carpentry --query "wood polish karigar"
  py yt_collect.py --task masonry --urls my_links.txt   # specific links, any site

By default only Creative Commons-licensed YouTube videos are downloaded (YouTube's CC
search filter + a license check). --any-license removes this; PRIVATE TESTING ONLY.
Each clip is a short SECTION of a longer video, not the whole video.

Next step afterwards:  py indic_skill_100.py process
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote_plus

BASE = Path(__file__).resolve().parent
ROOT = BASE / "indic_skill_100"
RAW = ROOT / "raw"
MANIFEST = ROOT / "manifest.csv"
ARCHIVE = ROOT / "yt_archive.txt"      # remembers downloaded IDs -> no duplicates
TMP = ROOT / "_last_run.txt"
CC_SEARCH_FLAG = "&sp=EgIwAQ%253D%253D"  # YouTube search filter: Creative Commons

DEFAULT_QUERIES = {
    "plumbing": ["plumber pipe repair India", "नल पाइप मरम्मत प्लंबर", "பிளம்பர் குழாய் பழுது"],
    "electrical": ["electrician wiring work India", "इलेक्ट्रीशियन वायरिंग", "எலக்ட்ரீசியன் வயரிங்"],
    "tailoring": ["tailor stitching India", "दर्जी सिलाई मशीन", "தையல் தொழில் டெய்லர்"],
    "carpentry": ["carpenter woodwork India", "बढ़ई लकड़ी का काम", "தச்சு வேலை"],
    "painting": ["house painter wall India", "पेंटर दीवार पेंटिंग", "பெயிண்டர் சுவர் வேலை"],
    "welding": ["welder workshop India", "वेल्डिंग का काम", "வெல்டிங் வேலை"],
    "masonry": ["mason bricklaying India", "राजमिस्त्री ईंट चिनाई", "கொத்தனார் வேலை"],
    "mechanic": ["bike mechanic repair India", "मैकेनिक बाइक रिपेयर", "மெக்கானிக் பைக் ரிப்பேர்"],
}

FORMAT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]"


def log_manifest(rows):
    if not rows:
        return
    ROOT.mkdir(exist_ok=True)
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "task", "source", "source_id", "license", "url"])
        w.writerows(rows)


def fetch(target, task, limit, a, is_search):
    """Run yt-dlp once; return number of new clips."""
    filters = []
    if a.clip_length > 0:
        filters.append(f"duration>={a.clip_start + a.clip_length}")
    if is_search:
        filters.append(f"duration<={a.max_seconds}")
        if not a.any_license:
            filters.append("license~='Creative Commons'")

    cmd = [
        sys.executable, "-m", "yt_dlp", target,
        "--max-downloads", str(limit),
        "--ignore-errors", "--no-warnings",
        "-f", FORMAT, "--merge-output-format", "mp4",
        "--download-archive", str(ARCHIVE),
        "-o", str(RAW / "yt_%(id)s.%(ext)s"),
        "--sleep-interval", "2",
        "--print-to-file", "after_move:%(filepath)s|%(id)s|%(license)s|%(webpage_url)s", str(TMP),
        "--no-simulate",
    ]
    if filters:
        cmd += ["--match-filters", " & ".join(filters)]
    if is_search:
        cmd += ["--playlist-end", str(a.search_depth)]
    else:
        cmd += ["--no-playlist"]
    if a.clip_length > 0:
        cmd += ["--download-sections", f"*{a.clip_start}-{a.clip_start + a.clip_length}",
                "--force-keyframes-at-cuts"]
    if a.subs:
        cmd += ["--write-subs", "--write-auto-subs",
                "--sub-langs", "hi.*,ta.*,te.*,bn.*,en.*", "--convert-subs", "srt"]

    TMP.unlink(missing_ok=True)
    subprocess.run(cmd, text=True)  # exit code 101 = --max-downloads reached (normal)

    rows = []
    if TMP.exists():
        for line in TMP.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) < 4 or not parts[0].lower().endswith(".mp4"):
                continue
            path, vid, lic, url = parts[0], parts[1], parts[2], "|".join(parts[3:])
            if lic in ("NA", ""):
                lic = "CC (verify)" if is_search else "user-verified"
            rows.append([Path(path).name, task, "youtube" if "youtu" in url else "other", vid, lic, url])
    log_manifest(rows)
    return len(rows)


def search_target(query, a):
    if a.any_license:
        return f"ytsearch{a.search_depth}:{query}"
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}{CC_SEARCH_FLAG}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=list(DEFAULT_QUERIES), help="trades to search (default: all)")
    ap.add_argument("--task", help="trade label for --query / --urls")
    ap.add_argument("--query", help="your own search phrase (needs --task)")
    ap.add_argument("--urls", help="text file, one video URL per line (needs --task)")
    ap.add_argument("--per-query", type=int, default=3, help="max new clips per search phrase")
    ap.add_argument("--search-depth", type=int, default=40, help="how many search results to look through")
    ap.add_argument("--max-seconds", type=int, default=900, help="skip source videos longer than this")
    ap.add_argument("--clip-start", type=int, default=5, help="clip starts at this second")
    ap.add_argument("--clip-length", type=int, default=30, help="clip length in seconds (0 = whole video)")
    ap.add_argument("--subs", action="store_true", help="also download subtitles (hi/ta/te/bn/en)")
    ap.add_argument("--any-license", action="store_true", help="PRIVATE TESTING ONLY: skip the CC filter")
    a = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {RAW}\n")
    if a.any_license:
        print("WARNING: license filter off. Do not redistribute or publish these clips.\n")
    total = 0

    if a.urls:
        if not a.task:
            sys.exit("--urls needs --task (e.g. --task plumbing)")
        urls = [u.strip() for u in Path(a.urls).read_text(encoding="utf-8").splitlines()
                if u.strip() and not u.startswith("#")]
        for u in urls:
            print(f"[url] {u}")
            total += fetch(u, a.task, 1, a, is_search=False)
    elif a.query:
        if not a.task:
            sys.exit("--query needs --task (e.g. --task carpentry)")
        print(f"[{a.task}] {a.query}")
        total += fetch(search_target(a.query, a), a.task, a.per_query, a, is_search=True)
    else:
        for task in a.tasks:
            if task not in DEFAULT_QUERIES:
                print(f"[skip] unknown trade '{task}'. Choices: {', '.join(DEFAULT_QUERIES)}")
                continue
            for q in DEFAULT_QUERIES[task]:
                print(f"[{task}] {q}")
                n = fetch(search_target(q, a), task, a.per_query, a, is_search=True)
                print(f"   -> {n} new")
                total += n

    TMP.unlink(missing_ok=True)
    print(f"\nDone. {total} new clips in {RAW}. Next: py indic_skill_100.py process")


if __name__ == "__main__":
    main()