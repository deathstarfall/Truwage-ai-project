#!/usr/bin/env python3
"""
Step 1 - Feature extraction.
  video : VideoMAE (MCG-NJU/videomae-base)  -> 768-d clip vector + (8 x 768) temporal vectors
  audio : wav2vec2 (facebook/wav2vec2-base) -> 768-d clip vector
Reads  indic_skill_100/video_15fps/*.mp4  and  indic_skill_100/audio/*.wav
Writes indic_skill_100/features/features.npz

Install:  py -m pip install torch transformers opencv-python soundfile numpy
Run:      py aqa_extract.py
Options:  --video-model MCG-NJU/videomae-base-finetuned-kinetics   (more semantic features)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, VideoMAEImageProcessor, VideoMAEModel, Wav2Vec2Model

BASE = Path(__file__).resolve().parent / "indic_skill_100"
VID, AUD, OUT = BASE / "video_15fps", BASE / "audio", BASE / "features"
NUM_FRAMES = 16  # VideoMAE-base expects 16 frames


def read_frames(path, n=NUM_FRAMES):
    """Return n RGB frames sampled evenly across the clip (or None)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None
    wanted = np.linspace(0, total - 1, n).astype(int).tolist()
    need, keep, i = set(wanted), {}, 0
    while i <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if i in need:
            keep[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    if not keep:
        return None
    frames = [keep[j] if j in keep else keep[max(keep)] for j in wanted]
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-model", default="MCG-NJU/videomae-base")
    ap.add_argument("--audio-model", default="facebook/wav2vec2-base")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}  (first run downloads ~700 MB of models)")
    proc = VideoMAEImageProcessor.from_pretrained(a.video_model)
    vmodel = VideoMAEModel.from_pretrained(a.video_model).to(dev).eval()
    afe = AutoFeatureExtractor.from_pretrained(a.audio_model)
    amodel = Wav2Vec2Model.from_pretrained(a.audio_model).to(dev).eval()
    t_steps = vmodel.config.num_frames // vmodel.config.tubelet_size  # 16 // 2 = 8

    clips = sorted(VID.glob("*.mp4"))
    if not clips:
        raise SystemExit("No clips in video_15fps. Run: py indic_skill_100.py process")

    names, vfeat, vtemp, afeat, has_audio = [], [], [], [], []
    for k, clip in enumerate(clips, 1):
        frames = read_frames(clip)
        if frames is None:
            print(f"[skip] unreadable video {clip.name}")
            continue
        with torch.no_grad():
            inp = proc(frames, return_tensors="pt").to(dev)
            h = vmodel(**inp).last_hidden_state              # (1, tokens, 768)
            vfeat.append(h.mean(1)[0].cpu().numpy())
            vtemp.append(h.reshape(1, t_steps, -1, h.shape[-1]).mean(2)[0].cpu().numpy())

        a_vec = np.zeros(amodel.config.hidden_size, dtype=np.float32)
        ok = False
        wav_path = AUD / f"{clip.stem}.wav"
        if wav_path.exists():
            wav, sr = sf.read(str(wav_path), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(1)
            if sr == 16000 and np.abs(wav).max() > 1e-3:  # skip silent tracks
                with torch.no_grad():
                    ai = afe(wav[: 16000 * 30], sampling_rate=16000, return_tensors="pt").to(dev)
                    a_vec = amodel(**ai).last_hidden_state.mean(1)[0].cpu().numpy()
                ok = True
        names.append(clip.stem)
        afeat.append(a_vec)
        has_audio.append(ok)
        print(f"[{k}/{len(clips)}] {clip.name}  audio={'yes' if ok else 'no'}")

    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "features.npz", names=np.array(names), video=np.stack(vfeat),
             temporal=np.stack(vtemp), audio=np.stack(afeat), has_audio=np.array(has_audio))
    print(f"\nSaved {len(names)} clips ({sum(has_audio)} with usable audio) -> {OUT / 'features.npz'}")


if __name__ == "__main__":
    main()
