# -*- coding: utf-8 -*-
import os, math
import pandas as pd
import pygplates

MAX_MODEL_AGE = 1800.0
R_EARTH = 6371.0

BASE_DIR  = r"C:\Users\Guozhi\OneDrive - Australian National University\Code\PyGplate\Carbonatite1107"
MODEL_DIR = rf"{BASE_DIR}\PyGPlate\1.8Ga_model_GSF"

ROT_FILES = {
    ">=1000": rf"{MODEL_DIR}\1800_1000_rotfile.rot",
    "<1000" : rf"{MODEL_DIR}\1000_0_rotfile.rot",
}


CONV_410_1000_1 = rf"{BASE_DIR}\1000-410_Convergence.gpml"
CONV_410_1000_2 = rf"{BASE_DIR}\1000-410-Convergence_Merdith_et_al.gpml"

SUBD_BY_INTERVAL = [
    (0.0,   250.0,  [rf"{MODEL_DIR}\250-0_plate_boundaries.gpml"]),
    (250.0, 410.0,  [rf"{MODEL_DIR}\410-250_plate_boundaries.gpml"]),
    (410.0, 1000.0, [CONV_410_1000_1, CONV_410_1000_2]),
    (1000.0, 1800.0,[rf"{MODEL_DIR}\1800-1000_plate_boundaries.gpml"]),
]

INPUT_FILE  = rf"{BASE_DIR}\Kimber_Age_Nd_PaleoCoords.csv"
OUTPUT_FILE = rf"{BASE_DIR}\Kimber_Age_Nd_SubductionDistanceWoodhead.csv"

def pick_col(cols, prefer_list):
    cols_l = {c.lower(): c for c in cols}
    for name in prefer_list:
        if name.lower() in cols_l:
            return cols_l[name.lower()]
    for name in prefer_list:
        nl = name.lower()
        for c in cols:
            if nl in c.lower():
                return c
    return None

def normalize_lon(lon):
    lon = float(lon)
    return (lon + 180.0) % 360.0 - 180.0

def fix_lat_lon(lat, lon):
    lat = float(lat); lon = float(lon)
    lon = normalize_lon(lon)
    if (abs(lat) > 90.0) and (abs(lon) <= 90.0):
        lat, lon = lon, lat
        lon = normalize_lon(lon)
    return lat, lon

def is_subduction_feature(feat: pygplates.Feature) -> bool:
    
    try:
        ft = str(feat.get_feature_type()).lower()
    except Exception:
        return False

    if "transform" in ft:
        return False
    if "midoceanridge" in ft or ("ridge" in ft and "subduction" not in ft):
        return False

    return "subductionzone" in ft

def files_for_age(age):
    rot_file = ROT_FILES["<1000"] if age < 1000 else ROT_FILES[">=1000"]
    for amin, amax, candidates in SUBD_BY_INTERVAL:
        if amin <= age < amax:
            sub_files = [p for p in candidates if os.path.exists(p)]
            if sub_files and os.path.exists(rot_file):
                return sub_files, rot_file
    return None, None

def min_distance_km(lat, lon, rec):
    pt = pygplates.PointOnSphere(lat, lon)
    best = float("inf")
    for r in rec:
        try:
            g = r.get_reconstructed_geometry()
            if g is None:
                continue
            ang = pygplates.GeometryOnSphere.distance(pt, g)  # radians
            if ang is None:
                continue
            best = min(best, ang * R_EARTH)
        except Exception:
            continue
    return best if math.isfinite(best) else None


print("calculation")
print("MODEL_DIR exists:", os.path.exists(MODEL_DIR))
print("ROT >=1000 exists:", os.path.exists(ROT_FILES[">=1000"]))
print("ROT <1000  exists:", os.path.exists(ROT_FILES["<1000"]))
print("410–1000 gpml 1:", CONV_410_1000_1, "exists=", os.path.exists(CONV_410_1000_1))
print("410–1000 gpml 2:", CONV_410_1000_2, "exists=", os.path.exists(CONV_410_1000_2))

if (not os.path.exists(CONV_410_1000_1)) and (not os.path.exists(CONV_410_1000_2)):
    raise FileNotFoundError(
        "Nothing\n"
        f"{CONV_410_1000_1}\n{CONV_410_1000_2}\n"
        "Please check"
    )


df = pd.read_csv(INPUT_FILE)

age_col = pick_col(df.columns, ["Age_Ma", "Age(Ma)", "Age (Ma)", "Age"])
lat_col = pick_col(df.columns, ["Paleo_Lat", "Paleo Lat", "paleo_lat"])
lon_col = pick_col(df.columns, ["Paleo_Lon", "Paleo Lon", "paleo_lon"])

print("Columns:", list(df.columns))
print(f"Resolved: age_col={age_col} | lat_col={lat_col} | lon_col={lon_col}")

if not (age_col and lat_col and lon_col):
    raise RuntimeError("No Age / Paleo_Lat / Paleo_Lon 列，请检查表头")

df["_age"] = pd.to_numeric(df[age_col], errors="coerce")
df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")

print("Age max:", float(df["_age"].max()))
print("Age>1800:", int((df["_age"] > MAX_MODEL_AGE).sum()))
print("NaN Lat:", int(df["_lat"].isna().sum()))
print("NaN Lon:", int(df["_lon"].isna().sum()))

distances = []
stats = {"ok":0, "age_out":0, "skip_nan":0, "no_file":0, "no_trench":0, "no_result":0}

cache = {}  # key=(interval_tag, rot_file, age)-> rec list

def interval_tag(age):
    for amin, amax, _ in SUBD_BY_INTERVAL:
        if amin <= age < amax:
            return f"{int(amax)}-{int(amin)}"
    return "NA"

for _, row in df.iterrows():
    age = row["_age"]; lat = row["_lat"]; lon = row["_lon"]

    if not (math.isfinite(age) and math.isfinite(lat) and math.isfinite(lon)):
        distances.append(None)
        stats["skip_nan"] += 1
        continue

    if age < 0 or age > MAX_MODEL_AGE:
        distances.append(None)
        stats["age_out"] += 1
        continue

    lat, lon = fix_lat_lon(lat, lon)

    sub_files, rot_file = files_for_age(age)
    if not sub_files:
        distances.append(None)
        stats["no_file"] += 1
        continue

    key = (interval_tag(age), rot_file, float(age))
    if key not in cache:
        feats = []
        for f in sub_files:
            feats.extend(list(pygplates.FeatureCollection(f)))
        feats = [ft for ft in feats if is_subduction_feature(ft)]
        if not feats:
            cache[key] = []
        else:
            rec = []
            pygplates.reconstruct(feats, pygplates.RotationModel(rot_file), rec, age)
            cache[key] = rec

    rec = cache[key]
    if not rec:
        distances.append(None)
        stats["no_trench"] += 1
        continue

    d = min_distance_km(lat, lon, rec)
    if d is None:
        distances.append(None)
        stats["no_result"] += 1
    else:
        distances.append(d)
        stats["ok"] += 1

df["SubductionDistance_km"] = distances
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

s = pd.to_numeric(df["SubductionDistance_km"], errors="coerce")
print("\n统计:", stats)
print("Valid count:", int(s.notna().sum()))
if int(s.notna().sum()) > 0:
    print("min/max:", float(s.min()), float(s.max()))
    print("p95:", float(s.quantile(0.95)))
print("输出:", OUTPUT_FILE)
