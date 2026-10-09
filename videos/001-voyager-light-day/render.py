"""Render Short 01 v2 (Voyager 1 one light-day) as 1080x1920 30 fps frames with matplotlib, then mux with ffmpeg.
Inputs: data/*.json (from query_data.py), audio/narration.wav + audio/timing.json (from narrate.py).
Run: python render.py            -> short-01.mp4, thumbnail.png, frames.png, safezone.png
     ../.venv/bin/python render.py test 3 15  -> test_*.png stills + layout warnings (text outside the Shorts safe zone)
All plotted values come from the JSON data files; nothing is hand-drawn.
v2: Shorts safe-zone layout, 3D (orthographic) trajectory scene from Horizons XYZ, larger secondary text,
    0.3 s crossfades between scenes, end card wording that is true today."""
import json, os, math, subprocess, textwrap, datetime, shutil, sys
from multiprocessing import Pool
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Arc
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.dates as mdates
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(HERE, *p)
W, H, FPS, DPI = 1080, 1920, 30, 100
BG, ACC, VOY, TXT, MUT, GRID = "#070b18", "#4fd1ff", "#ffb347", "#ffffff", "#9fb3d1", "#1b2440"
AU = 149597870.7
# YouTube Shorts UI zones (px): top 8 %, bottom 20 %, right 12 %
SAFE_TOP, SAFE_BOTTOM, SAFE_RIGHT, SAFE_LEFT = int(0.08 * H), int(0.80 * H), int(0.88 * W), 40
CX = (SAFE_LEFT + SAFE_RIGHT) / 2          # horizontal centre of the safe area (495 px)
SEC = 25                                   # secondary text size in pt (25 pt = 34.7 px at 100 dpi)
CAP_TOP = 1075                             # caption box top (56 % of height); 2-line box ends ~1277 (66.5 %)
SRC_TOP = 1405                             # per-scene source line (ends before the 80 % line)
XFADE = 0.30
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TXT, "axes.edgecolor": MUT,
                     "axes.labelcolor": MUT, "xtick.color": MUT, "ytick.color": MUT})

facts = json.load(open(D("data", "facts.json")))
timing = json.load(open(D("audio", "timing.json")))
TOTAL = timing["total_s"]; NF = int(math.ceil(TOTAL * FPS))
SEG = {s["id"]: s for s in timing["segments"]}
geo = json.load(open(D("data", "v1_geocentric_daily_UT.json")))
traj = json.load(open(D("data", "v1_helio_5d.json")))
PLN = ["Jupiter", "Saturn", "Uranus", "Neptune"]
pos10 = {n: json.load(open(D("data", f"{n.lower()}_helio_10d.json"))) for n in PLN}
orbit = {n: json.load(open(D("data", f"{n.lower()}_orbit.json"))) for n in PLN}

def cal2dt(c):  # "A.D. 2026-Sep-01 00:00:00.0000"
    return datetime.datetime.strptime(c.replace("A.D. ", "")[:17], "%Y-%b-%d %H:%M")
geo_t = np.array([mdates.date2num(cal2dt(r["cal"])) for r in geo]); geo_h = np.array([r["lt"] / 3600 for r in geo])
tr_t = np.array([mdates.date2num(cal2dt(r["cal"])) for r in traj])
tr_xyz = np.array([[r["x"], r["y"], r["z"]] for r in traj]) / AU
tr_r = np.linalg.norm(tr_xyz, axis=1)
p_t = {n: np.array([mdates.date2num(cal2dt(r["cal"])) for r in v]) for n, v in pos10.items()}
p_xyz = {n: np.array([[r["x"], r["y"], r["z"]] for r in v]) / AU for n, v in pos10.items()}
o_xyz = {n: np.array([[r["x"], r["y"], r["z"]] for r in v] + [[v[0]["x"], v[0]["y"], v[0]["z"]]]) / AU for n, v in orbit.items()}
o_a = {n: float(np.mean(np.linalg.norm(o_xyz[n], axis=1))) for n in PLN}
cross = facts["crossing"]; cross_dt = datetime.datetime.fromisoformat(cross["utc"]).replace(tzinfo=None)
cross_num = mdates.date2num(cross_dt)
sw = facts["shrink_window"]; hv = facts["v1_helio_2026_11_18"]
LAT = hv["ecl_lat_position_deg"]; LAT_OK = 34.5 <= LAT <= 35.5   # only state "35 deg" if the data supports it

# scene boundaries from narration timing
b = lambda sid: SEG[sid]["start"] - 0.14
S = {"A": (0, b("define")), "B": (b("define"), b("method")), "C": (b("method"), b("result")),
     "D": (b("result"), b("twist")), "E": (b("twist"), b("outro")), "F": (b("outro"), TOTAL + 1)}
ORDER = ["A", "B", "C", "D", "E", "F"]

def ease(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3
def hms(s): s = int(round(s)); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

# ---------- captions: split each segment caption into <=2-line chunks, timed by character share ----------
def wrap(text, width=25):
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 > width and cur: lines.append(cur); cur = w
        else: cur = (cur + " " + w).strip()
    return lines + [cur]
def chunks_for(seg):
    import re
    raw = [c.strip() for c in re.split(r"(?<=[.?!:,])\s+", seg["caption"]) if c.strip()]
    clauses = []
    for c in raw:                                   # glue very short lead-in clauses ("On Nov 18,") to the next one
        if clauses and len(clauses[-1]) < 12 and clauses[-1][-1] not in ".?!": clauses[-1] += " " + c
        else: clauses.append(c)
    groups, cur = [], []
    for c in clauses:
        if cur and cur[-1][-1] in ".?!":            # start a new caption at each sentence end
            groups.append(wrap(" ".join(cur))); cur = []
        if len(wrap(" ".join(cur + [c]))) <= 2: cur.append(c)
        else:
            if cur: groups.append(wrap(" ".join(cur)))
            ls = wrap(c)
            while len(ls) > 2: groups.append(ls[:2]); ls = ls[2:]
            cur = [" ".join(ls)]
    if cur: groups.append(wrap(" ".join(cur)))
    total = sum(len(" ".join(g)) for g in groups); t = seg["start"]; out = []
    for g in groups:
        dur = (seg["end"] - seg["start"]) * len(" ".join(g)) / total
        out.append((t, t + dur, "\n".join(g))); t += dur
    return out
CAPS = [c for s in timing["segments"] for c in chunks_for(s)]
def caption_at(t):
    for a, b_, txt in CAPS:
        if a <= t < b_ + 0.25: return txt
    return None

# ---------- figure helpers ----------
def base_fig(transparent=False):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    if transparent: fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    else: ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), extent=[0, W, H, 0], cmap=matplotlib.colors.LinearSegmentedColormap.from_list("g", ["#0b1430", BG]), aspect="auto", zorder=-10)
    return fig, ax
def fig_array(fig):
    fig.canvas.draw(); return np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)
def pix_axes(fig, x0, y0, x1, y1, **kw):  # axes from pixel box (top-left origin)
    return fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H], **kw)
def T(ax, x, y, s, size=SEC, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    return ax.text(x, y, s, fontsize=size, **kw)

def source_line(ax, text):
    T(ax, SAFE_LEFT + 10, SRC_TOP, "\n".join(textwrap.wrap(text, 46)), size=SEC, color=MUT, ha="left", va="top", linespacing=1.3)

def chrome(ax, t):
    T(ax, SAFE_LEFT + 10, 190, "AI-narrated · computed from NASA/JPL data", size=26, color=MUT, ha="left")
    T(ax, SAFE_LEFT + 10, 236, "Grok bot kai", size=24, color=ACC, ha="left", fontweight="bold")
    cap = caption_at(t)
    if cap:
        n = cap.count("\n") + 1; hgt = 70 + 66 * n
        ax.add_patch(FancyBboxPatch((SAFE_LEFT + 10, CAP_TOP), SAFE_RIGHT - SAFE_LEFT - 20, hgt, boxstyle="round,pad=0,rounding_size=22", fc="#000000", ec="none", alpha=0.62, zorder=5))
        T(ax, CX, CAP_TOP + hgt / 2, cap, size=36, fontweight="bold", color=TXT, linespacing=1.25, zorder=6)

# ---------- scenes ----------
def prog(key, t): a, b_ = S[key]; return (t - a) / (b_ - a)

def scene_A(fig, ax, t):
    p = prog("A", t)
    T(ax, CX, 380, "1 LIGHT-DAY", size=84, fontweight="bold", color=TXT, alpha=ease(0.35 + p * 8))
    T(ax, CX, 492, "Voyager 1  ·  Nov 18, 2026", size=38, color=VOY, alpha=ease(0.2 + p * 6))
    T(ax, CX, 640, "One-way signal time, Earth \u2192 Voyager 1", size=SEC + 1, color=MUT)
    t0 = mdates.date2num(datetime.datetime(2026, 9, 27)); tn = t0 + (cross_num - t0) * ease(min(1, p * 1.12))
    val = 86400.0 if tn >= cross_num - 1e-6 else float(np.interp(tn, geo_t, geo_h)) * 3600
    T(ax, CX, 775, hms(min(val, 86400.0)), size=104, fontweight="bold", color=ACC, family="DejaVu Sans Mono")
    T(ax, CX, 900, f"on {mdates.num2date(tn):%b %d, %Y}  (computed)", size=SEC + 1, color=MUT)
    source_line(ax, "Source: NASA/JPL Horizons, Voyager 1 (ID -31), distance from Earth")

def scene_B(fig, ax, t):
    p0 = prog("B", t); p = ease(p0 * 1.6)
    T(ax, CX, 320, "Light-travel time from Earth", size=36, fontweight="bold")
    x0, x1, y = 80, 900, 620; L = x0 + (x1 - x0) * p
    ax.plot([x0, L], [y, y], color=ACC, lw=9, solid_capstyle="round")
    items = [(8.33 / 60, "Sun: 8.3 light-min", "below", "left", TXT), (4 + 10 / 60, "Neptune: ~4 h 10 min", "above", "left", TXT),
             (24, "Voyager 1\n(Nov 18): 24 h", "below", "right", VOY)]
    for hrs, lab, pos, ha, col in items:
        x = x0 + (x1 - x0) * hrs / 24
        if x <= L + 1:
            ax.plot([x, x], [y - 40, y + 40], color=col, lw=4)
            T(ax, x, y - 60 if pos == "above" else y + 60, lab, size=SEC + 2, ha=ha, va="bottom" if pos == "above" else "top", color=col, fontweight="bold" if col == VOY else "normal")
    a2 = ease(p0 * 2)
    T(ax, CX, 880, "1 light-day =", size=30, color=MUT, alpha=a2)
    T(ax, CX, 940, f"{facts['light_day_km']:,.0f} km", size=38, fontweight="bold", alpha=a2)
    T(ax, CX, 1005, f"= {facts['light_day_au']:.2f} \u00d7 the Earth\u2013Sun distance", size=SEC + 1, color=MUT, alpha=a2)
    source_line(ax, "Sources: speed of light (exact SI value); NASA \u201cWhat Is a Light-Day?\u201d (Sun, Neptune)")

# --- 3D trajectory (orthographic projection of Horizons heliocentric ecliptic XYZ) ---
LON_V = math.radians(hv["ecl_lon_position_deg"])
AZ, EL = LON_V + math.pi / 2, math.radians(26)   # view from the side of Voyager's direction so its climb is seen in profile
def proj(xyz):
    xyz = np.atleast_2d(xyz); x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    sx = -x * math.sin(AZ) + y * math.cos(AZ)
    sy = -(x * math.cos(AZ) + y * math.sin(AZ)) * math.sin(EL) + z * math.cos(EL)
    return np.column_stack([sx, sy])
TR2 = proj(tr_xyz); O2 = {n: proj(o_xyz[n]) for n in PLN}
C_BOX = (40, 440, 950, 965)                      # plot area in px (x0, y0, x1, y1)
C_ASPECT = (C_BOX[2] - C_BOX[0]) / (C_BOX[3] - C_BOX[1])
def _raw_box(k):
    rmax = tr_r[:k].max()
    pts = [TR2[:k], np.zeros((1, 2))] + [O2[n] for n in PLN if o_a[n] <= max(6.0, rmax * 1.05)]
    P = np.vstack(pts); return np.array([P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()])
_bx = np.array([_raw_box(k) for k in range(2, len(TR2) + 1)])
_bx[:, 0] = np.minimum.accumulate(_bx[:, 0]); _bx[:, 2] = np.minimum.accumulate(_bx[:, 2])
_bx[:, 1] = np.maximum.accumulate(_bx[:, 1]); _bx[:, 3] = np.maximum.accumulate(_bx[:, 3])
for _ in range(2):                                 # forward-backward smoothing of the camera box (no jumps)
    for i in range(1, len(_bx)): _bx[i] = 0.93 * _bx[i - 1] + 0.07 * _bx[i]
    for i in range(len(_bx) - 2, -1, -1): _bx[i] = 0.93 * _bx[i + 1] + 0.07 * _bx[i]
FINAL_W = 92.0                                     # final view width (au): Neptune's orbit stays readable; the path exits top-left
FINAL = (-58.0, FINAL_W - 58.0, -20.0, -20.0 + FINAL_W / C_ASPECT)
def cam_box(k):
    x0, x1, y0, y1 = _bx[k - 2]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) / 2 * 1.14, (y1 - y0) / 2 * 1.22
    hw, hh = max(hw, hh * C_ASPECT), max(hh, hw / C_ASPECT)
    box = np.array([cx - hw, cx + hw, cy - hh, cy + hh])
    w = min(1.0, max(0.0, (2 * hw - 0.6 * FINAL_W) / (0.4 * FINAL_W)))  # blend into the fixed final view as the box grows
    w = w * w * (3 - 2 * w)
    return tuple((1 - w) * box + w * np.array(FINAL)) if 2 * hw < FINAL_W else FINAL
FB = {n: np.argmin(np.abs(tr_t - mdates.date2num(cal2dt(facts["flybys"][n]["utc"])))) for n in ("Jupiter", "Saturn")}

def scene_C(fig, ax, t):
    p = prog("C", t)
    T(ax, CX, 300, "Voyager 1's real path, 1977 \u2192 2026", size=32, fontweight="bold")
    T(ax, CX, 352, "3D view, computed from JPL Horizons", size=SEC, color=MUT)
    f = min(1.0, max(0.0, p / 0.82)) ** 1.7
    k = max(2, int(round(2 + (len(TR2) - 2) * f)))
    now = tr_t[k - 1]
    x0, x1, y0, y1 = cam_box(k); span = x1 - x0
    pax = pix_axes(fig, *C_BOX, facecolor="none"); pax.axis("off")
    pax.set_xlim(x0, x1); pax.set_ylim(y0, y1)
    # scale rings in the ecliptic plane (drawn guides, labelled)
    rmax = tr_r[:k].max(); th = np.linspace(0, 2 * np.pi, 200)
    for R in (50, 100, 150):
        if 0.3 * rmax < R < 0.98 * rmax:
            ring = proj(np.column_stack([R * np.cos(th), R * np.sin(th), 0 * th]))
            pax.plot(ring[:, 0], ring[:, 1], color="#24345c", lw=1.5, ls=(0, (4, 4)))
            lab = proj([R * math.cos(AZ), R * math.sin(AZ), 0])[0]
            pax.text(lab[0], lab[1] - span * 0.012, f"{R} au", fontsize=SEC, color="#6f84a8", ha="center", va="top", clip_on=True)
    # planet orbits (one full orbit each, from Horizons) + positions on the current date
    placed = []
    for n in PLN[::-1]:                                   # outer planets first; skip labels that would overlap
        pax.plot(O2[n][:, 0], O2[n][:, 1], color="#3d5388", lw=2)
        i = min(np.searchsorted(p_t[n], now), len(p_t[n]) - 1); q = proj(p_xyz[n][i])[0]
        pax.plot(*q, "o", color=TXT, ms=7)
        if 2 * o_a[n] / span > 0.22 and x0 + 0.03 * span < q[0] < x1 - 0.22 * span and y0 + 0.04 * span < q[1] < y1 - 0.06 * span \
                and all(abs(q[0] - u) > 0.2 * span or abs(q[1] - v) > 0.06 * span for u, v in placed + [(0, 0)]):
            placed.append((q[0], q[1]))
            pax.text(q[0] + span * 0.015, q[1] + span * 0.012, n, fontsize=SEC, color=MUT, ha="left", va="bottom", clip_on=True)
    pax.plot(0, 0, "o", color="#ffd84a", ms=11); pax.text(span * 0.015, -span * 0.02, "Sun", fontsize=SEC, color="#ffd84a", va="top")
    # 35 deg illustration once the path is (nearly) complete
    if LAT_OK and p > 0.80:
        a3 = ease((p - 0.80) * 6); v = tr_xyz[k - 1]; foot = proj([v[0], v[1], 0])[0]; vp = TR2[k - 1]
        pax.plot([0, foot[0]], [0, foot[1]], color=TXT, lw=2.5, ls="--", alpha=0.8 * a3)
        pax.plot([0, vp[0]], [0, vp[1]], color=VOY, lw=2, ls="--", alpha=0.8 * a3)
        ang = math.degrees(math.atan2(vp[1], vp[0])); rr = span * 0.22
        pax.add_patch(Arc((0, 0), 2 * rr, 2 * rr, theta1=ang, theta2=180, color=VOY, lw=3, alpha=a3))
        mid = math.radians((ang + 180) / 2)
        pax.text(rr * 1.25 * math.cos(mid), rr * 1.25 * math.sin(mid), f"{LAT:.0f}\u00b0", fontsize=32, color=VOY, fontweight="bold", ha="center", va="center", alpha=a3)
        pax.text(foot[0] * 0.3 if abs(foot[0]) < span else -span * 0.35, -span * 0.03, "planets' plane", fontsize=SEC, color=TXT, ha="center", va="top", alpha=a3, clip_on=True)
        T(ax, CX, 412, f"Now {LAT:.0f}\u00b0 above the planets' orbital plane", size=SEC + 1, color=TXT, alpha=a3)
    pax.plot(TR2[:k, 0], TR2[:k, 1], color=VOY, lw=3.5)
    pax.plot(*TR2[k - 1], "o", color=VOY, ms=12)
    for n, idx in FB.items():
        if k - 1 >= idx and span < 45:
            q = TR2[idx]; pax.plot(*q, "o", color=VOY, ms=8, mfc=BG)
            pax.text(q[0] - span * 0.02, q[1] - span * 0.02, f"{n} flyby", fontsize=SEC, color=VOY, ha="right", va="top", clip_on=True)
    d = mdates.num2date(now)
    T(ax, SAFE_LEFT + 20, 1000, f"{d:%Y}", size=44, fontweight="bold", color=VOY, ha="left")
    T(ax, SAFE_LEFT + 200, 1004, f"{tr_r[k-1]:.1f} au from the Sun", size=SEC + 2, color=VOY, ha="left")
    vp = TR2[k - 1]
    if not (x0 <= vp[0] <= x1 and y0 <= vp[1] <= y1):   # Voyager has left the view: edge marker
        T(ax, SAFE_LEFT + 20, 470, "↖ Voyager 1 is off this view", size=SEC, color=VOY, ha="left")
    source_line(ax, "Source: NASA/JPL Horizons, Sun-centred XYZ of Voyager 1 (-31) and planets")

YT = [23 + 50 / 60, 23 + 55 / 60, 24, 24 + 5 / 60]
def light_chart(fig, upto, top, bottom, show_shrink, show_min):
    cax = pix_axes(fig, 190, top, 925, bottom, facecolor="none")
    m = geo_t <= upto
    cax.plot(geo_t[m], geo_h[m], color=ACC, lw=5)
    cax.axhline(24, color=VOY, lw=2.5, ls="--")
    cax.text(geo_t[0] + 3, 24.008, "24 h = one light-day", color=VOY, fontsize=SEC, va="bottom")
    cax.set_xlim(geo_t[0], geo_t[-1]); cax.set_ylim(23.78, 24.115)
    cax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[11, 1, 3, 5])); cax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    cax.yaxis.set_major_locator(FixedLocator(YT)); cax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v + 1e-9)}:{round((v % 1) * 60):02d}"))
    cax.tick_params(labelsize=SEC); cax.set_ylabel("signal time (h:mm)", fontsize=SEC)
    for s in ("top", "right"): cax.spines[s].set_visible(False)
    cax.grid(color=GRID, lw=1)
    if upto >= cross_num: cax.plot(cross_num, 24, "o", color=TXT, ms=13, zorder=5)
    if show_shrink:
        a0 = mdates.date2num(cal2dt(sw["first_day_rr_negative"])); a1 = mdates.date2num(cal2dt(sw["last_day_rr_negative"]))
        cax.axvspan(a0, a1, color=ACC, alpha=0.16 * show_shrink)
    if show_min:
        mn = sw["local_min"]; cax.plot(mdates.date2num(cal2dt(mn["date"])), mn["lt_s"] / 3600, "o", color=VOY, ms=12, zorder=6)
    return cax

def scene_D(fig, ax, t):
    p = prog("D", t)
    T(ax, CX, 300, "When is the signal exactly 24 h?", size=34, fontweight="bold")
    upto = geo_t[0] + (cross_num - geo_t[0]) * ease(p * 2.2)
    light_chart(fig, upto, 590, 960, 0, False)
    if p > 0.40: T(ax, CX, 372, f"Computed: {cross_dt:%b %d, %Y  %H:%M:%S} UTC", size=29, fontweight="bold")
    if p > 0.58:
        T(ax, CX, 428, "NASA's announced time:", size=SEC, color="#7dff9a")
        T(ax, CX, 470, "2:16:07 a.m. PST = 10:16:07 UTC  \u2713", size=SEC, color="#7dff9a")
    source_line(ax, "Sources: NASA/JPL Horizons (-31), 1-minute steps, UT; NASA \u201cWhat Is a Light-Day?\u201d")

def scene_E(fig, ax, t):
    p = prog("E", t)
    T(ax, CX, 300, "Twist: the gap shrinks for 3 months", size=32, fontweight="bold")
    upto = cross_num + (geo_t[-1] - cross_num) * ease(p * 2.0)
    shr = ease((p - 0.22) * 3) if p > 0.22 else 0
    tnever = SEG["never"]["start"]
    light_chart(fig, upto, 590, 960, shr, t >= tnever - 0.1)
    if shr > 0:
        T(ax, CX, 368, f"{cal2dt(sw['first_day_rr_negative']):%b %d} \u2013 {cal2dt(sw['last_day_rr_negative']):%b %d, %Y}:", size=SEC + 2, color=ACC, alpha=shr)
        T(ax, CX, 410, f"{sw['drop_km']/1e6:.1f} million km closer", size=SEC + 2, color=ACC, fontweight="bold", alpha=shr)
        w = facts["why_shrinks"]
        T(ax, CX, 462, f"Mar 8, 2027: Earth moves toward it at {w['earth_speed_toward_voyager_kms']:.1f} km/s,", size=SEC, color=MUT, alpha=shr)
        T(ax, CX, 502, f"Voyager recedes from the Sun at {w['voyager_heliocentric_recession_kms']:.1f} km/s", size=SEC, color=MUT, alpha=shr)
    if t >= tnever - 0.1:
        lt = sw["local_min"]["lt_s"]
        T(ax, CX, 548, f"Lowest (Apr 24, 2027): {int(lt//3600)} h {int(lt%3600//60):02d} min {int(round(lt%60)):02d} s", size=SEC + 1, fontweight="bold", color=VOY)
    source_line(ax, "Source: NASA/JPL Horizons (Earth and Voyager 1 position and velocity vectors)")

def scene_F(fig, ax, t):
    p = ease((t - S["F"][0]) / 0.6)
    T(ax, CX, 420, "COMPUTED,\nNOT DRAWN", size=80, fontweight="bold", linespacing=1.1, alpha=p)
    lines = [("Every number: NASA/JPL Horizons API", TXT), ("Voyager 1 (ID -31) \u00b7 geometric \u00b7 UTC", MUT),
             ("Query details: in the description", MUT), ("Script, visuals & voice: AI (Grok bot kai)", MUT),
             ("Narration: Kokoro-82M TTS (Apache-2.0)", MUT)]
    for i, (l, c) in enumerate(lines):
        T(ax, CX, 640 + i * 70, l, size=SEC + 1, color=c, alpha=p)
    source_line(ax, "Kai is an AI. Data courtesy NASA/JPL; not endorsed by NASA.")

SCENES = {"A": scene_A, "B": scene_B, "C": scene_C, "D": scene_D, "E": scene_E, "F": scene_F}

def scene_img(key, t):
    fig, ax = base_fig(); SCENES[key](fig, ax, t); a = fig_array(fig); plt.close(fig); return a
def compose(t):
    key = next(k for k in ORDER if S[k][0] <= t < S[k][1])
    i = ORDER.index(key); img = None
    if i > 0 and t - S[key][0] < XFADE / 2:                       # fade in from previous scene
        w = 0.5 + (t - S[key][0]) / XFADE; img = (1 - w) * scene_img(ORDER[i - 1], t) + w * scene_img(key, t)
    elif i < len(ORDER) - 1 and S[key][1] - t < XFADE / 2:        # fade out into next scene
        w = 0.5 - (S[key][1] - t) / XFADE; img = (1 - w) * scene_img(key, t) + w * scene_img(ORDER[i + 1], t)
    else: img = scene_img(key, t)
    fig, ax = base_fig(transparent=True); chrome(ax, t); c = fig_array(fig); plt.close(fig)
    al = c[..., 3:4] / 255.0
    out = img[..., :3] * (1 - al) + c[..., :3] * al
    return np.clip(out, 0, 255).astype(np.uint8)
def render_frame(i, path=None):
    Image.fromarray(compose(i / FPS)).save(path or D("frames", f"{i:05d}.png")); return i

def layout_check(t):
    """Report any text drawn outside the Shorts safe zone at time t (scene + chrome)."""
    key = next(k for k in ORDER if S[k][0] <= t < S[k][1]); bad = []
    for build in ("scene", "chrome"):
        fig, ax = base_fig(build == "chrome")
        if build == "scene": SCENES[key](fig, ax, t)
        else: chrome(ax, t)
        fig.canvas.draw(); r = fig.canvas.get_renderer()
        for a in fig.axes:
            for txt in a.texts:
                if not txt.get_text().strip() or txt.get_alpha() == 0: continue
                bb = txt.get_window_extent(r); x0, x1 = bb.x0, bb.x1; y0, y1 = H - bb.y1, H - bb.y0
                if a is not ax and txt.get_clip_on():
                    ab = a.get_window_extent(r); x0 = max(x0, ab.x0); x1 = min(x1, ab.x1)
                if x1 > SAFE_RIGHT + 1 or y0 < SAFE_TOP or y1 > SAFE_BOTTOM or x0 < 0:
                    bad.append(f"{key} t={t:.1f} '{txt.get_text()[:40]!r}' x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}]")
        plt.close(fig)
    return bad

def thumbnail():
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1280); ax.set_ylim(720, 0); ax.axis("off")
    ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), extent=[0, 1280, 720, 0], cmap=matplotlib.colors.LinearSegmentedColormap.from_list("g", ["#0b1430", BG]), aspect="auto", zorder=-10)
    ax.text(60, 150, "VOYAGER 1", fontsize=46, color=VOY, fontweight="bold", va="center")
    ax.text(60, 270, "1 LIGHT-DAY", fontsize=80, color=TXT, fontweight="bold", va="center")
    ax.text(60, 380, "from Earth", fontsize=44, color=TXT, va="center")
    ax.text(60, 500, f"Nov 18, 2026\n{cross_dt:%H:%M:%S} UTC", fontsize=40, color=ACC, fontweight="bold", va="center", linespacing=1.2)
    ax.text(60, 640, "computed from NASA/JPL Horizons data", fontsize=24, color=MUT, va="center")
    cax = fig.add_axes([0.70, 0.16, 0.27, 0.66], facecolor="none")
    cax.plot(geo_t, geo_h, color=ACC, lw=5); cax.axhline(24, color=VOY, lw=3, ls="--"); cax.plot(cross_num, 24, "o", color=TXT, ms=16)
    cax.set_xlim(geo_t[0], geo_t[-1]); cax.set_ylim(23.83, 24.08); cax.axis("off")
    cax.text(geo_t[-1], 24.006, "24 h", ha="right", color=VOY, fontsize=26, va="bottom", fontweight="bold")
    fig.savefig(D("thumbnail.png"), dpi=100, facecolor=BG); plt.close(fig)

def sheet_times():
    at = lambda k, f: S[k][0] + (min(S[k][1], TOTAL) - S[k][0]) * f
    return [at("A", 0.5), at("B", 0.8), at("C", 0.3), at("C", 0.62), at("C", 0.97), at("D", 0.9), at("E", 0.5), at("E", 0.96), at("F", 0.6)]

def grab(t, path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", D("short-01.mp4"), "-frames:v", "1", path], check=True)

def safezone_png():
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    cap_t = [a for a, b_, txt in CAPS if S["C"][0] < a < S["C"][1]][-1] + 0.3
    panels = []
    for t, lab in [(cap_t, f"Caption frame (t={cap_t:.1f} s)"), (S["F"][0] + 3.0, f"Outro frame (t={S['F'][0]+3.0:.1f} s)")]:
        grab(t, "/tmp/_sz.png"); im = Image.open("/tmp/_sz.png").convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        for box in [(0, 0, W, SAFE_TOP), (0, SAFE_BOTTOM, W, H), (SAFE_RIGHT, SAFE_TOP, W, SAFE_BOTTOM)]:
            d.rectangle(box, fill=(255, 40, 40, 90), outline=(255, 60, 60, 255), width=4)
        d.rectangle((0, int(0.55 * H), SAFE_RIGHT, int(0.72 * H)), outline=(80, 255, 120, 255), width=4)
        d.text((12, 12), "top 8% UI", font=small, fill=(255, 220, 220, 255))
        d.text((12, SAFE_BOTTOM + 12), "bottom 20%: title / channel / audio", font=small, fill=(255, 220, 220, 255))
        d.text((12, int(0.55 * H) - 34), "caption band 55-72%", font=small, fill=(150, 255, 170, 255))
        im = Image.alpha_composite(im, ov).convert("RGB").resize((540, 960))
        canvas = Image.new("RGB", (540, 1010), (20, 20, 28)); canvas.paste(im, (0, 50))
        ImageDraw.Draw(canvas).text((10, 8), lab, font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26), fill=(255, 255, 255))
        panels.append(canvas)
    out = Image.new("RGB", (1090, 1010), (20, 20, 28)); out.paste(panels[0], (0, 0)); out.paste(panels[1], (550, 0))
    out.save(D("safezone.png"))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        for s in map(float, sys.argv[2:]):
            render_frame(int(s * FPS), D(f"test_{s:05.1f}.png")); [print("LAYOUT:", m) for m in layout_check(s)]
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        bad = [m for i in range(0, NF, 5) for m in layout_check(i / FPS)]
        print("\n".join(bad) if bad else "layout OK: no text outside the safe zone"); sys.exit()
    shutil.rmtree(D("frames"), ignore_errors=True); os.makedirs(D("frames"))
    with Pool(os.cpu_count()) as pool:
        for n, _ in enumerate(pool.imap_unordered(render_frame, range(NF), chunksize=8)):
            if n % 300 == 0: print("frame", n, "/", NF, flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", D("frames", "%05d.png"), "-i", D("audio", "narration.wav"),
                    "-filter:a", "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest",
                    "-movflags", "+faststart", D("short-01.mp4")], check=True)
    shutil.rmtree(D("frames"), ignore_errors=True)
    thumbnail()
    times = sheet_times()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + sum([["-ss", f"{s:.3f}", "-i", D("short-01.mp4")] for s in times], []) +
                   ["-filter_complex", "".join(f"[{i}:v]scale=360:640,setsar=1,trim=end_frame=1[v{i}];" for i in range(9)) +
                    "".join(f"[v{i}]" for i in range(9)) + "xstack=inputs=9:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0|0_h0+h3|w0_h0+h3|w0+w1_h0+h3[out]",
                    "-map", "[out]", "-frames:v", "1", D("frames.png")], check=True)
    safezone_png()
    print("contact sheet times:", [round(x, 2) for x in times]); print("done")
