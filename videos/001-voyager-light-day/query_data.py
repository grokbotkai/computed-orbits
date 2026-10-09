"""Short 01 data query: Voyager 1 reaches one light-day from Earth.
All numbers used in the video are computed here from NASA/JPL Horizons (public API, no key)
and saved to data/ so the render is reproducible. Run: python query_data.py
"""
import json, math, os, sys, datetime
sys.path.insert(0, os.path.dirname(__file__))
from horizons import vectors, AU_KM
OUT = os.path.join(os.path.dirname(__file__), "data"); os.makedirs(OUT, exist_ok=True)
C_KMS = 299792.458                     # exact, SI definition
LIGHT_DAY_KM = C_KMS * 86400.0         # 25,902,068,371.2 km
facts = {"retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
         "source": "NASA/JPL Horizons API https://ssd.jpl.nasa.gov/api/horizons.api (geometric states, ecliptic J2000)",
         "light_day_km": LIGHT_DAY_KM, "light_day_au": LIGHT_DAY_KM / AU_KM}

def save(name, rows): json.dump(rows, open(os.path.join(OUT, name), "w"))

# 1) Geocentric range / one-way light time, daily, for the chart (UT)
geo, hdr = vectors("-31", "500@399", "2026-09-01", "2027-07-01", "1d", time_type="UT")
save("v1_geocentric_daily_UT.json", geo)

# 2) Exact crossing of one-way light time = 86400 s, 1-minute steps (UT), linear interpolation
fine, _ = vectors("-31", "500@399", "2026-11-18 09:00", "2026-11-18 12:00", "1m", time_type="UT")
cross = None
for a, b in zip(fine, fine[1:]):
    if a["lt"] < 86400.0 <= b["lt"]:
        frac = (86400.0 - a["lt"]) / (b["lt"] - a["lt"])
        jd = a["jd"] + frac * (b["jd"] - a["jd"])
        t0 = datetime.datetime.strptime(a["cal"].replace("A.D. ", "")[:17], "%Y-%b-%d %H:%M")
        secs = frac * 60.0
        t = (t0 + datetime.timedelta(seconds=round(secs))).replace(tzinfo=datetime.timezone.utc)
        cross = {"jd_ut": jd, "utc": t.isoformat(timespec="seconds"), "seconds_after_bracket_start": secs, "bracket": [a["cal"], b["cal"]],
                 "range_km": a["rg"] + frac * (b["rg"] - a["rg"]), "range_rate_kms": a["rr"]}
facts["crossing"] = cross

# 2b) Does it ever drop back below one light-day? Check daily through 2030
lt_later, _ = vectors("-31", "500@399", "2026-11-19", "2031-01-01", "1d", time_type="UT")
facts["min_lt_2026_11_19_to_2031_s"] = min(r["lt"] for r in lt_later)

# 3) Today's values and the early-2027 "gap shrinks" window
today = [r for r in geo if "2026-Sep-27" in r["cal"]][0]
facts["today_2026_09_27"] = {"range_km": today["rg"], "range_au": today["rg"] / AU_KM, "lt_s": today["lt"], "rr_kms": today["rr"]}
neg = [r for r in geo if r["rr"] < 0]
mx = max(geo, key=lambda r: r["lt"] if r["jd"] < neg[-1]["jd"] else -1)
mn = min([r for r in geo if r["jd"] > mx["jd"]], key=lambda r: r["lt"])
facts["shrink_window"] = {"first_day_rr_negative": neg[0]["cal"], "last_day_rr_negative": neg[-1]["cal"],
                          "local_max": {"date": mx["cal"], "lt_s": mx["lt"], "range_km": mx["rg"]},
                          "local_min": {"date": mn["cal"], "lt_s": mn["lt"], "range_km": mn["rg"]},
                          "drop_km": mx["rg"] - mn["rg"], "drop_light_s": mx["lt"] - mn["lt"],
                          "min_lt_after_crossing_s": min(r["lt"] for r in geo if r["jd"] > cross["jd_ut"]),
                          "most_negative_rr": min((r["rr"], r["cal"]) for r in geo)}

# 4) Why it shrinks: Earth's heliocentric velocity projected on the Earth->Voyager direction (at most negative rr date)
d = facts["shrink_window"]["most_negative_rr"][1].split()[1]
e, _ = vectors("399", "500@10", d, d + " 00:01", "1m"); v, _ = vectors("-31", "500@10", d, d + " 00:01", "1m")
e, v = e[0], v[0]
u = [v[k] - e[k] for k in "xyz"]; n = math.sqrt(sum(c * c for c in u)); u = [c / n for c in u]
facts["why_shrinks"] = {"date": d,
    "earth_speed_toward_voyager_kms": sum(e["v" + k] * uk for k, uk in zip("xyz", u)),
    "voyager_heliocentric_recession_kms": v["rr"],
    "voyager_ecliptic_latitude_deg": math.degrees(math.asin(v["z"] / v["rg"]))}

# 5) Heliocentric trajectory 1977 -> crossing (for the map) + planets over the same span
traj, _ = vectors("-31", "500@10", "1977-09-06", "2026-11-18", "5d"); save("v1_helio_5d.json", traj)
facts["helio_speed_2026_11_18_kms"] = traj[-1]["rr"]
planets = {"Earth": "399", "Jupiter": "599", "Saturn": "699", "Uranus": "799", "Neptune": "899"}
for name, cid in planets.items():
    p, _ = vectors(cid, "500@10", "1977-09-06", "2026-11-18", "10d"); save(f"{name.lower()}_helio_10d.json", p)

# 6) Planetary flybys (closest approach, hourly around known windows)
fb = {}
for name, cid, a, b in [("Jupiter", "599", "1979-03-01", "1979-03-10"), ("Saturn", "699", "1980-11-08", "1980-11-16")]:
    r, _ = vectors("-31", f"500@{cid}", a, b, "10m", time_type="UT")
    m = min(r, key=lambda x: x["rg"]); fb[name] = {"utc": m["cal"], "range_km": m["rg"]}
facts["flybys"] = fb
# 7) One full orbit of each outer planet (for the orbit rings in the 3D path scene), heliocentric, ecliptic J2000
for name, cid, stop, step in [("Jupiter", "599", "1989-08-01", "20d"), ("Saturn", "699", "2007-04-01", "50d"),
                              ("Uranus", "799", "2062-01-01", "150d"), ("Neptune", "899", "2143-01-01", "300d")]:
    p, _ = vectors(cid, "500@10", "1977-09-06", stop, step); save(f"{name.lower()}_orbit.json", p)

# 8) Voyager 1 heliocentric ecliptic latitude of position and of velocity at the crossing date (for the "35 deg" label)
last = traj[-1]; sp = math.sqrt(last["vx"]**2 + last["vy"]**2 + last["vz"]**2)
facts["v1_helio_2026_11_18"] = {"ecl_lat_position_deg": math.degrees(math.asin(last["z"] / last["rg"])),
                                "ecl_lon_position_deg": math.degrees(math.atan2(last["y"], last["x"])) % 360,
                                "ecl_lat_velocity_deg": math.degrees(math.asin(last["vz"] / sp)),
                                "height_above_ecliptic_au": last["z"] / 149597870.7}
json.dump(facts, open(os.path.join(OUT, "facts.json"), "w"), indent=2)
print(json.dumps(facts, indent=2))
