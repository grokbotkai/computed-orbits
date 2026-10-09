"""Minimal JPL Horizons API client (vectors). Public API, no key: https://ssd-api.jpl.nasa.gov/doc/horizons.html"""
import urllib.request, urllib.parse, json, re, time
API = "https://ssd.jpl.nasa.gov/api/horizons.api"
AU_KM = 149597870.7

def vectors(command, center, start, stop, step, ref_plane="ECLIPTIC", retries=3, time_type="TDB"):
    """Return list of dicts: jd, cal, x,y,z (km), vx,vy,vz (km/s), lt (s), rg (km), rr (km/s). Time scale per time_type (TDB default, or UT)."""
    params = {
        "format": "json", "COMMAND": f"'{command}'", "EPHEM_TYPE": "'VECTORS'",
        "CENTER": f"'{center}'", "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'", "VEC_TABLE": "'3'", "REF_PLANE": f"'{ref_plane}'",
        "OUT_UNITS": "'KM-S'", "CSV_FORMAT": "'YES'", "VEC_LABELS": "'NO'", "OBJ_DATA": "'NO'", "TIME_TYPE": f"'{time_type}'",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                res = json.load(r)
            break
        except Exception:
            if i == retries - 1: raise
            time.sleep(2)
    text = res["result"]
    if "$$SOE" not in text:
        raise RuntimeError(text[:2000])
    body = text.split("$$SOE")[1].split("$$EOE")[0].strip()
    rows = []
    for line in body.splitlines():
        p = [s.strip() for s in line.split(",")]
        if len(p) < 11: continue
        rows.append(dict(jd=float(p[0]), cal=p[1], x=float(p[2]), y=float(p[3]), z=float(p[4]),
                         vx=float(p[5]), vy=float(p[6]), vz=float(p[7]), lt=float(p[8]), rg=float(p[9]), rr=float(p[10])))
    return rows, text.split("$$SOE")[0]

def closest(command, center, start, stop, step):
    rows, _ = vectors(command, center, start, stop, step)
    m = min(rows, key=lambda r: r["rg"])
    return m, rows
