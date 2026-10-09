"""Narration for Short 01 with Kokoro-82M (Apache-2.0 weights & voices, hexgrad/Kokoro-82M).
Run in a Python 3.12 env with kokoro installed: python narrate.py
Writes audio/narration.wav (24 kHz mono) and audio/timing.json (segment start/end seconds)."""
import json, os, numpy as np, soundfile as sf
from kokoro import KPipeline
HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "script_short01.json")))
os.makedirs(os.path.join(HERE, "audio"), exist_ok=True)
SR = 24000
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
parts, timing, t = [np.zeros(int(cfg["lead_in_s"] * SR), np.float32)], [], cfg["lead_in_s"]
for seg in cfg["segments"]:
    chunks = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in pipe(seg["spoken"], voice=cfg["voice"], speed=cfg["speed"])]
    audio = np.concatenate(chunks).astype(np.float32)
    timing.append({"id": seg["id"], "start": t, "end": t + len(audio) / SR, "caption": seg["caption"]})
    parts += [audio, np.zeros(int(cfg["gap_s"] * SR), np.float32)]
    t += len(audio) / SR + cfg["gap_s"]
parts.append(np.zeros(int(cfg["tail_s"] * SR), np.float32))
full = np.concatenate(parts)
sf.write(os.path.join(HERE, "audio", "narration.wav"), full, SR)
json.dump({"total_s": len(full) / SR, "segments": timing}, open(os.path.join(HERE, "audio", "timing.json"), "w"), indent=2)
print(f"total {len(full)/SR:.2f}s"); [print(f"{s['id']:8s} {s['start']:6.2f}-{s['end']:6.2f}") for s in timing]
