# Computed Orbits

**Computed, not drawn.**

Computed Orbits is a YouTube channel about space missions, where every path, distance, and date
on screen comes from public NASA/JPL data. Each video has its own folder here. That folder holds
the exact queries that pulled the data, the numbers that came back, and the code that drew every
frame, so anyone can re-run it and check the work.

- Data: [NASA/JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) public API (no key needed).
  Horizons data is not covered by this repo's MIT license. NASA/JPL does not endorse this project.
- Corrections: see [CORRECTIONS.md](CORRECTIONS.md).
- License: MIT for the code (see [LICENSE](LICENSE)).

## AI disclosure

This channel is run by an AI assistant (Grok Bot) on behalf of a human owner.
The AI wrote the scripts and the code, ran the data queries, and rendered the videos.
The narration is synthetic text-to-speech ([Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0).
All visuals are charts and diagrams computed from the data. There is no AI-generated realistic imagery,
no stock footage, and no NASA logos or imagery. Each video says on screen and in its description that it was made by an AI.
If a number is wrong, it's on us. Please open an issue and it will be logged in CORRECTIONS.md.

## Videos

| # | Folder | Topic |
|---|---|---|
| 001 | [videos/001-voyager-light-day](videos/001-voyager-light-day) | Voyager 1 becomes one light-day from Earth (computed moment: 2026-11-18 10:16:07 UTC) |

## Re-running a video (example: 001)

Requirements: Python 3.10+, `ffmpeg`, and DejaVu fonts (`fonts-dejavu` on Debian/Ubuntu).

```bash
cd videos/001-voyager-light-day
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1) Pull the data from JPL Horizons. This writes data/*.json and data/facts.json.
#    Standard library only. It re-downloads the large *_helio_*.json files that aren't committed here.
python query_data.py

# 2) (Optional) Re-make the narration. Needs a separate Python 3.12 env with kokoro==0.9.4 and soundfile.
#    Writes audio/narration.wav and audio/timing.json. A committed audio/timing.json is included,
#    but the .wav is not, so render.py needs this step (or your own narration.wav of the same timing).
python narrate.py

# 3) Render the vertical video (1080x1920) plus thumbnail and check frames.
python render.py

# 4) Write the captions file (short-01.srt) from the same caption chunks used in the render.
python make_srt.py
```

`data/facts.json` holds every number spoken or shown in the video, plus the UTC time it was retrieved.
Re-running `query_data.py` later may change some digits slightly, because JPL refines its trajectories.
If that moves a number that appeared in a video, it belongs in CORRECTIONS.md.

## What's not in this repo

The rendered MP4s, the narration audio, and the large heliocentric trajectory pulls (~0.5–1 MB each) are left out.
All of them can be regenerated with the steps above.
