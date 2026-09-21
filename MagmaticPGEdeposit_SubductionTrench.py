# -*- coding: utf-8 -*-
import os, math, collections, datetime
import pandas as pd
import pygplates


BASE_DIR  = r"C:\Users\Guozhi\OneDrive - Australian National University\Code\PyGplate\Carbonatite1107"
MODEL_DIR = rf"{BASE_DIR}\PyGPlate\1.8Ga_model_GSF"

INPUT_FILE  = rf"{BASE_DIR}\Sulfide deposit_PaleoCoords0210.csv"
OUTPUT_FILE = rf"{BASE_DIR}\Sulfide deposit_SubductionDistance0210.csv"

ROT_FILES = {
    ">=1000": rf"{MODEL_DIR}\1800_1000_rotfile.rot",
    "<1000" : rf"{MODEL_DIR}\1000_0_rotfile.rot",
}

COB_FILE = rf"{MODEL_DIR}\COBfile_1800_0.gpml"

CONV_410_1000_1 = rf"{BASE_DIR}\1000-410_Convergence.gpml"
CONV_410_1000_2 = rf"{BASE_DIR}\1000-410-Convergence_Merdith_et_al.gpml"

SUBD_BY_INTERVAL = [
    (0.0,    250.0,  [rf"{MODEL_DIR}\250-0_plate_boundaries.gpml",  COB_FILE]),
    (250.0,  410.0,  [rf"{MODEL_DIR}\410-250_plate_boundaries.gpml", COB_FILE]),
    (410.0,  1000.0, [CONV_410_1000_1, CONV_410_1000_2, COB_FILE]),
    (1000.0, 1800.0, [rf"{MODEL_DIR}\1800-1000_plate_boundaries.gpml", COB_FILE]),
]

R_EARTH = 6371.0
MODEL_MAX_AGE = 1800.0

PLAT_COL = "Paleo_Lat"
PLON_COL = "Paleo_Lon"


def normalize_lon(lon):
    lon = float(lon)
    return (lon + 180.0) % 360.0 - 180.0

def fix_lat_lon(lat, lon):
    lat = float(lat)
    lon = normalize_lon(lon)
    if (abs(lat) > 90.0) and (abs(lon) <= 90.0):
        lat, lon = lon, lat
        lon = normalize_lon(lon)
    return lat, lon

def get_age_ma(row):
    # 兼容各种列名
    for k in ["Age_Ma", "Age(Ma)", "Age (Ma)", "Age", "Age(Ma) "]:
        if k in row.index:
            v = pd.to_numeric(row[k], errors="coerce")
            if pd.notna(v):
                return float(v)
    if "Age (Ga)" in row.index:
        v = pd.to_numeric(row["Age (Ga)"], errors="coerce")
        if pd.notna(v):
            return float(v) * 1000.0
    return None

def is_subduction_zone(feat: pygplates.Feature) -> bool:
    """只认 gpml:SubductionZone"""
    try:
        ft = feat.get_feature_type()
        ft_str = ft.to_string() if hasattr(ft, "to_string") else str(ft)
        return ft_str.lower().endswith("subductionzone")
    except Exception:
        return False


EXCLUDE_PLATE_PAIRS_BY_INTERVAL = {
    "410-250": {(401, 0), (0, 401)},  
}
EXCLUDE_NAME_KEYWORDS_BY_INTERVAL = {
    "410-250": ["mongol-okhotsk"], 
}

def keep_subduction_feature(feat: pygplates.Feature, itag: str) -> bool:
    """
    True = feature 
    False = delete
    """
    if not is_subduction_zone(feat):
        return False

    name = (feat.get_name() or "").lower()
    for kw in EXCLUDE_NAME_KEYWORDS_BY_INTERVAL.get(itag, []):
        if kw in name:
            return False

    pid = feat.get_reconstruction_plate_id()
    cid = feat.get_conjugate_plate_id()
    if (pid, cid) in EXCLUDE_PLATE_PAIRS_BY_INTERVAL.get(itag, set()):
        return False

    return True

def load_features(paths):
    feats = []
    for f in [p for p in paths if os.path.exists(p)]:
        feats.extend(list(pygplates.FeatureCollection(f)))
    return feats

def reconstruct_geoms(subd_feats, rot_model, age):
    reconstructed = []
    pygplates.reconstruct(subd_feats, rot_model, reconstructed, age)
    geoms = []
    for rfg in reconstructed:
        g = rfg.get_reconstructed_geometry()
        if g:
            geoms.append(g)
    return geoms

def distance_point_geoms(lat, lon, geoms):
    if not geoms:
        return None
    pt = pygplates.PointOnSphere(lat, lon)
    best = float("inf")
    for g in geoms:
        try:
            d = pygplates.GeometryOnSphere.distance(pt, g) * R_EARTH
            if d < best:
                best = d
        except Exception:
            continue
    return best if math.isfinite(best) else None

def files_for_age(age):
    rot = ROT_FILES["<1000"] if age < 1000 else ROT_FILES[">=1000"]
    for amin, amax, flist in SUBD_BY_INTERVAL:
        if amin <= age < amax:
            exist_files = [p for p in flist if os.path.exists(p)]
            if exist_files and os.path.exists(rot):
                return exist_files, rot, (amin, amax)
            return None, None, None
    return None, None, None

def interval_tag(age):
    for amin, amax, _ in SUBD_BY_INTERVAL:
        if amin <= age < amax:
            return f"{int(amax)}-{int(amin)}"
    return "NA"

def nearest_subduction_feature(lat, lon, subd_feats, rot_model, age):
     
    pt = pygplates.PointOnSphere(lat, lon)
    reconstructed = []
    pygplates.reconstruct(subd_feats, rot_model, reconstructed, age)

    best = None
    for rfg in reconstructed:
        g = rfg.get_reconstructed_geometry()
        if not g:
            continue
        dkm = pygplates.GeometryOnSphere.distance(pt, g) * R_EARTH
        f = rfg.get_feature()
        name = f.get_name() if f.get_name() else ""
        pid = f.get_reconstruction_plate_id()
        cid = f.get_conjugate_plate_id()
        if (best is None) or (dkm < best[0]):
            best = (float(dkm), f, name, pid, cid)
    return best

def safe_to_csv(df, out_path):
     
    try:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path
    except PermissionError:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root, ext = os.path.splitext(out_path)
        out2 = f"{root}_{ts}{ext}"
        df.to_csv(out2, index=False, encoding="utf-8-sig")
        print(f"[WARN] Permission denied for {out_path}")
        print(f"[WARN] Wrote to {out2} instead (close Excel if you want fixed name).")
        return out2


print("on going")
print("MODEL_DIR exists:", os.path.exists(MODEL_DIR))
print("INPUT exists:", os.path.exists(INPUT_FILE))
print("ROT >=1000 exists:", os.path.exists(ROT_FILES[">=1000"]))
print("ROT <1000 exists:", os.path.exists(ROT_FILES["<1000"]))
print("COB exists:", os.path.exists(COB_FILE))
print("410–1000 gpml 1 exists:", os.path.exists(CONV_410_1000_1))
print("410–1000 gpml 2 exists:", os.path.exists(CONV_410_1000_2))

# ================= 主程序 =================
df = pd.read_csv(INPUT_FILE)

distances = []
stats = {
    "ok":0,
    "age_gt_model_max":0,
    "bad_age":0,
    "skip_nan":0,
    "no_file":0,
    "no_subduction":0,
    "no_result":0
}

rot_model_cache = {}
geom_cache = {}          # key=(itag, rot_file, age)-> geoms
typecount_cache = {}     # key=(itag, rot_file)-> Counter
subd_cache = {}          # key=(itag, rot_file)-> (subd_feats, rot_model)

for _, row in df.iterrows():

    age = get_age_ma(row)
    if age is None or (not math.isfinite(age)):
        distances.append(None)
        stats["bad_age"] += 1
        continue

    if age > MODEL_MAX_AGE or age < 0:
        distances.append(None)
        stats["age_gt_model_max"] += 1
        continue

    lat = pd.to_numeric(row.get(PLAT_COL, None), errors="coerce")
    lon = pd.to_numeric(row.get(PLON_COL, None), errors="coerce")
    if not (math.isfinite(lat) and math.isfinite(lon)):
        distances.append(None)
        stats["skip_nan"] += 1
        continue

    lat, lon = fix_lat_lon(lat, lon)

    sub_files, rot_file, _ = files_for_age(age)
    if sub_files is None:
        distances.append(None)
        stats["no_file"] += 1
        continue

    itag = interval_tag(age)
    type_key = (itag, rot_file)

    
    if type_key not in typecount_cache:
        feats_all = load_features(sub_files)
        c = collections.Counter()
        for f in feats_all:
            try:
                ft = f.get_feature_type()
                ft_str = ft.to_string() if hasattr(ft, "to_string") else str(ft)
            except Exception:
                ft_str = "UNKNOWN"
            c[ft_str] += 1
        typecount_cache[type_key] = c
        print(f"[TYPECOUNT] interval={itag} files={len(sub_files)} -> top types:",
              c.most_common(5))

    
    if type_key not in subd_cache:
        if rot_file not in rot_model_cache:
            rot_model_cache[rot_file] = pygplates.RotationModel(rot_file)
        rot_model = rot_model_cache[rot_file]

        feats = load_features(sub_files)

        
        all_subd = [f for f in feats if is_subduction_zone(f)]
        subd_feats = [f for f in all_subd if keep_subduction_feature(f, itag)]
        print(f"[SUBD] interval={itag} age={age:.1f} Ma -> SubductionZone feats:",
              len(all_subd), " kept:", len(subd_feats))

        subd_cache[type_key] = (subd_feats, rot_model)
    else:
        subd_feats, rot_model = subd_cache[type_key]

    
    key = (itag, rot_file, float(age))
    if key not in geom_cache:
        if not subd_feats:
            geom_cache[key] = []
        else:
            geom_cache[key] = reconstruct_geoms(subd_feats, rot_model, age)

    geoms = geom_cache[key]
    if not geoms:
        distances.append(None)
        stats["no_subduction"] += 1
        continue

    dkm = distance_point_geoms(lat, lon, geoms)
    distances.append(dkm)
    if dkm is not None:
        stats["ok"] += 1
    else:
        stats["no_result"] += 1

df["SubductionDistance_km"] = distances


out_written = safe_to_csv(df, OUTPUT_FILE)
print("输出:", out_written)
print("统计:", stats)


try:
    m = df["Deposit"].astype(str).str.contains("Noril", case=False, na=False)
    nor = df.loc[m].copy()
    print("\n[CHECK] Noril rows in output df:")
    if len(nor) == 0:
        print("  (none found)")
    else:
        print(nor[["Deposit", PLAT_COL, PLON_COL, "SubductionDistance_km"]])

        nor_row = nor.iloc[0]
        age = get_age_ma(nor_row)
        lat = float(pd.to_numeric(nor_row.get(PLAT_COL), errors="coerce"))
        lon = float(pd.to_numeric(nor_row.get(PLON_COL), errors="coerce"))
        lat, lon = fix_lat_lon(lat, lon)

        sub_files, rot_file, _ = files_for_age(age)
        itag = interval_tag(age)
        type_key = (itag, rot_file)

        if type_key in subd_cache:
            subd_feats, rot_model = subd_cache[type_key]
            best = nearest_subduction_feature(lat, lon, subd_feats, rot_model, age)
            print("\n[DEBUG Noril nearest trench]")
            print("age=", age, "Ma")
            print("best(km, name, pid, cid) =", (best[0], best[2], best[3], best[4]))

            out_dbg = rf"{BASE_DIR}\DEBUG_Noril_nearestTrench_{int(round(age))}Ma.gpml"
            pt_feat = pygplates.Feature()
            pt_feat.set_name(f"Noril_point_{int(round(age))}Ma")
            pt_feat.set_geometry(pygplates.PointOnSphere(lat, lon))
            pygplates.FeatureCollection([pt_feat, best[1]]).write(out_dbg)
            print("wrote:", out_dbg)
        else:
            print("[DEBUG] Noril: subd_cache miss for", type_key)

except Exception as e:
    print("[DEBUG ERROR]", repr(e))
