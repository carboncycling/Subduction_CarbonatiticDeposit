# -*- coding: utf-8 -*-
"""
Sampled Trench-to-CratonEdge Distance (time dependent, cross-version pygplates)
------------------------------------------------------------------------------
Fix for 410–1000 Ma:
  - Prefer using 1000-410_plate_boundaries.gpml (contains polylines) + Transforms
  - Convergence files are kept as optional extras (may not contain polylines in some models)

Outputs:
  - per_point CSV: one row per (age, trench feature, sample point)
  - summary CSV  : one row per age with distribution stats over ALL sample points
  - per_trench CSV: one row per (age, trench feature) with stats over its sampled points
"""

import os
import glob
import datetime
import math
import numpy as np
import pandas as pd
import pygplates


# ==============================
# 0) PATHS (EDIT THESE)
# ==============================
BASE_DIR  = r"C:\Users\Guozhi\OneDrive - Australian National University\Code\PyGplate\Carbonatite1107" 
MODEL_DIR = rf"{BASE_DIR}\PyGPlate\1.8Ga_model_GSF"

ROT_FILES = {
    ">=1000": rf"{MODEL_DIR}\1800_1000_rotfile.rot",
    "<1000" : rf"{MODEL_DIR}\1000_0_rotfile.rot",
}

COB_FILE = rf"{MODEL_DIR}\COBfile_1800_0.gpml"

# 410–1000 Ma files (plate boundaries + transforms are usually the reliable polyline sources)
PB_410_1000   = rf"{MODEL_DIR}\1000-410_plate_boundaries.gpml"
TR_410_1000   = rf"{MODEL_DIR}\1000-410_Transforms.gpml"

# Convergence files are OPTIONAL (sometimes not polylines)
CONV_410_1000_1 = rf"{MODEL_DIR}\1000-410_Convergence.gpml"
CONV_410_1000_2 = rf"{MODEL_DIR}\1000-410-Convergence_Merdith_et_al.gpml"

# Subduction boundary feature sets by interval
SUBD_BY_INTERVAL = [
    (0.0,    250.0,  [rf"{MODEL_DIR}\250-0_plate_boundaries.gpml",  COB_FILE]),
    (250.0,  410.0,  [rf"{MODEL_DIR}\410-250_plate_boundaries.gpml", COB_FILE]),
    (410.0,  1000.0, [
        PB_410_1000,
        TR_410_1000,
        CONV_410_1000_1,
        CONV_410_1000_2,
        COB_FILE
    ]),
    (1000.0, 1800.0, [rf"{MODEL_DIR}\1800-1000_plate_boundaries.gpml", COB_FILE]),
]

# IMPORTANT: set this to your real file location
CRATON_GPML = rf"{BASE_DIR}\shapes_cratons_Merdith_et_al.gpml"

# Outputs
OUT_PER_POINT  = rf"{BASE_DIR}\TrenchSample_to_CratonEdge_perPoint.csv"
OUT_SUMMARY    = rf"{BASE_DIR}\TrenchSample_to_CratonEdge_summary.csv"
OUT_PER_TRENCH = rf"{BASE_DIR}\TrenchSample_to_CratonEdge_perTrench.csv"

# Time slices (Ma)
AGES = np.arange(0, 1800 + 1, 10, dtype=float)

# Sampling step along trench (km)
STEP_KM = 100.0   # recommended: 50 or 100

# Earth radius (km)
R_EARTH = 6371.0


# ==============================
# 1) UTILITIES
# ==============================
def safe_to_csv(df, out_path):
    try:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path
    except PermissionError:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root, ext = os.path.splitext(out_path)
        out2 = f"{root}_{ts}{ext}"
        df.to_csv(out2, index=False, encoding="utf-8-sig")
        print(f"[WARN] Permission denied for {out_path}. Saved as: {out2}")
        return out2

def _existing(files):
    return [f for f in files if f and os.path.exists(f)]

def files_for_age(age):
    rot = ROT_FILES["<1000"] if age < 1000.0 else ROT_FILES[">=1000"]
    if not os.path.exists(rot):
        return None, None, None

    for amin, amax, flist in SUBD_BY_INTERVAL:
        if amin <= age < amax:
            exist_files = _existing(flist)

            # Also include any 1000-410*.gpml in MODEL_DIR (sometimes scattered)
            if amin == 410.0 and amax == 1000.0:
                for f in glob.glob(os.path.join(MODEL_DIR, "1000-410*.gpml")):
                    if os.path.exists(f) and f not in exist_files:
                        exist_files.append(f)

            if exist_files:
                return exist_files, rot, (amin, amax)
            return None, None, None

    return None, None, None

def load_features(files):
    feats = []
    for f in files:
        fc = pygplates.FeatureCollection(f)
        feats.extend(list(fc))
    return feats

def reconstruct_feature_geometries(features, rotation_model, age):
    reconstructed = []
    pygplates.reconstruct(features, rotation_model, reconstructed, age)
    return reconstructed


# ==============================
# 2) GEOMETRY HELPERS (cross-version)
# ==============================
def iter_polylines_from_geom(geom):
    """Yield PolylineOnSphere parts from geom (polyline or iterable container)."""
    if geom is None:
        return
    if isinstance(geom, pygplates.PolylineOnSphere):
        yield geom
        return
    try:
        for part in geom:
            if isinstance(part, pygplates.PolylineOnSphere):
                yield part
    except TypeError:
        pass


def polygon_to_boundary_polylines(polygon):
    """Convert PolygonOnSphere -> list[PolylineOnSphere] using version-safe fallbacks."""
    edges = []

    if hasattr(polygon, "get_exterior_ring"):
        try:
            ring = polygon.get_exterior_ring()
            if ring is not None and hasattr(ring, "to_lat_lon_list"):
                edges.append(pygplates.PolylineOnSphere(ring.to_lat_lon_list()))
                return edges
        except Exception:
            pass

    if hasattr(polygon, "to_lat_lon_list"):
        try:
            latlon = polygon.to_lat_lon_list()
            if not latlon:
                return edges
            if isinstance(latlon[0], (list, tuple)) and len(latlon[0]) > 0 and isinstance(latlon[0][0], (list, tuple)):
                for ring_latlon in latlon:
                    if ring_latlon and len(ring_latlon) >= 2:
                        edges.append(pygplates.PolylineOnSphere(ring_latlon))
            else:
                if len(latlon) >= 2:
                    edges.append(pygplates.PolylineOnSphere(latlon))
            return edges
        except Exception:
            pass

    if hasattr(polygon, "get_points"):
        try:
            pts = polygon.get_points()
            latlon = [p.to_lat_lon() for p in pts]
            if len(latlon) >= 2:
                edges.append(pygplates.PolylineOnSphere(latlon))
            return edges
        except Exception:
            pass

    return edges


def extract_craton_edges_from_reconstructed(craton_reconstructed):
    """Extract craton edges (PolylineOnSphere) from reconstructed craton geometries."""
    edges = []
    for rfg in craton_reconstructed:
        g = rfg.get_reconstructed_geometry()
        if g is None:
            continue

        if isinstance(g, pygplates.PolygonOnSphere):
            edges.extend(polygon_to_boundary_polylines(g))
            continue

        # Multi polygon: treat as iterable
        try:
            any_poly = False
            for part in g:
                if isinstance(part, pygplates.PolygonOnSphere):
                    any_poly = True
                    edges.extend(polygon_to_boundary_polylines(part))
            if any_poly:
                continue
        except TypeError:
            pass

        # If already lines
        for pl in iter_polylines_from_geom(g):
            edges.append(pl)

    return edges


# ==============================
# 3) GREAT-CIRCLE SAMPLING
# ==============================
def _deg2rad(x): return x * math.pi / 180.0
def _rad2deg(x): return x * 180.0 / math.pi

def latlon_to_unitvec(lat_deg, lon_deg):
    lat = _deg2rad(lat_deg)
    lon = _deg2rad(lon_deg)
    clat = math.cos(lat)
    return np.array([clat * math.cos(lon), clat * math.sin(lon), math.sin(lat)], dtype=float)

def unitvec_to_latlon(v):
    v = v / np.linalg.norm(v)
    lat = math.asin(v[2])
    lon = math.atan2(v[1], v[0])
    return _rad2deg(lat), _rad2deg(lon)

def central_angle_rad(v1, v2):
    dot = float(np.dot(v1, v2))
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)

def slerp(v1, v2, f):
    omega = central_angle_rad(v1, v2)
    if omega < 1e-12:
        return v1.copy()
    so = math.sin(omega)
    a = math.sin((1.0 - f) * omega) / so
    b = math.sin(f * omega) / so
    v = a * v1 + b * v2
    return v / np.linalg.norm(v)

def sample_polyline_every_km(polyline, step_km):
    """Sample points along a PolylineOnSphere at ~fixed spacing (km)."""
    latlon = polyline.to_lat_lon_list()
    if not latlon or len(latlon) < 2:
        return []

    vecs = [latlon_to_unitvec(lat, lon) for (lat, lon) in latlon]
    samples = [latlon[0]]
    carry = 0.0

    for i in range(len(vecs) - 1):
        v1, v2 = vecs[i], vecs[i + 1]
        omega = central_angle_rad(v1, v2)
        seg_km = omega * R_EARTH
        if seg_km <= 1e-9:
            continue

        dist_along = step_km - carry
        while dist_along <= seg_km + 1e-9:
            f = dist_along / seg_km
            v = slerp(v1, v2, f)
            lat, lon = unitvec_to_latlon(v)
            samples.append((lat, lon))
            dist_along += step_km

        carry = max(0.0, dist_along - seg_km)

    if samples[-1] != latlon[-1]:
        samples.append(latlon[-1])

    return samples


# ==============================
# 4) DISTANCE
# ==============================
def point_min_distance_to_edges_km(point_on_sphere, edges):
    best = None
    for e in edges:
        try:
            d_rad = pygplates.GeometryOnSphere.distance(point_on_sphere, e)
        except Exception:
            continue
        if d_rad is None:
            continue
        d_km = float(d_rad) * R_EARTH
        if best is None or d_km < best:
            best = d_km
    return best


# ==============================
# 5) PRELOAD CRATON FEATURES
# ==============================
if not os.path.exists(CRATON_GPML):
    raise FileNotFoundError(f"[ERROR] CRATON_GPML not found: {CRATON_GPML}")

craton_features = load_features([CRATON_GPML])


# ==============================
# 6) MAIN LOOP
# ==============================
per_point_rows  = []
per_trench_rows = []
summary_rows    = []

for age in AGES:
    sub_files, rot_file, interval = files_for_age(age)

    # Debug: show which files are used (VERY helpful for 410–1000)
    if sub_files:
        if 410.0 <= age < 1000.0:
            print(f"[DEBUG 410-1000] age={age:.1f} sub_files:")
            for f in sub_files:
                print("   -", f)

    if not sub_files:
        continue

    rotation_model = pygplates.RotationModel(rot_file)

    # Reconstruct subduction
    sub_feats = load_features(sub_files)
    sub_recon = reconstruct_feature_geometries(sub_feats, rotation_model, age)

    # Count polylines in reconstructed subduction (debug)
    n_sub_polylines = 0
    for rfg in sub_recon:
        g = rfg.get_reconstructed_geometry()
        for _ in iter_polylines_from_geom(g):
            n_sub_polylines += 1

    # Reconstruct cratons -> edges
    cr_recon  = reconstruct_feature_geometries(craton_features, rotation_model, age)
    cr_edges  = extract_craton_edges_from_reconstructed(cr_recon)

    if n_sub_polylines == 0 or not cr_edges:
        print(f"[WARN] age={age:.1f}  n_sub_polylines={n_sub_polylines}  craton_edges={len(cr_edges)}  -> skipped")
        continue

    all_point_dists = []
    trench_feature_count = 0
    point_count = 0

    for rfg in sub_recon:
        feat = rfg.get_feature()
        if feat is None:
            continue
        geom = rfg.get_reconstructed_geometry()
        if geom is None:
            continue

        fid  = str(feat.get_feature_id()) if feat.get_feature_id() else f"noid_{id(feat)}"
        name = feat.get_name() if feat.get_name() else ""
        pid  = feat.get_reconstruction_plate_id()
        cid  = feat.get_conjugate_plate_id()

        trench_dists = []
        trench_points = 0

        for pl in iter_polylines_from_geom(geom):
            sampled_latlon = sample_polyline_every_km(pl, STEP_KM)
            if not sampled_latlon:
                continue

            for (lat, lon) in sampled_latlon:
                p = pygplates.PointOnSphere(lat, lon)
                d_km = point_min_distance_to_edges_km(p, cr_edges)
                if d_km is None:
                    continue

                per_point_rows.append({
                    "Age_Ma": float(age),
                    "Interval": f"{int(interval[1])}-{int(interval[0])}",
                    "TrenchFeatureID": fid,
                    "TrenchName": name,
                    "TrenchPlateID": pid,
                    "TrenchConjugateID": cid,
                    "SampleLat": float(lat),
                    "SampleLon": float(lon),
                    "Step_km": float(STEP_KM),
                    "Dist_to_CratonEdge_km": float(d_km),
                    "RotFile": os.path.basename(rot_file),
                })

                all_point_dists.append(d_km)
                trench_dists.append(d_km)
                trench_points += 1
                point_count += 1

        if trench_points > 0:
            trench_feature_count += 1
            a = np.array(trench_dists, dtype=float)
            per_trench_rows.append({
                "Age_Ma": float(age),
                "Interval": f"{int(interval[1])}-{int(interval[0])}",
                "TrenchFeatureID": fid,
                "TrenchName": name,
                "TrenchPlateID": pid,
                "TrenchConjugateID": cid,
                "n_sample_points": int(trench_points),
                "min_km": float(np.min(a)),
                "median_km": float(np.median(a)),
                "mean_km": float(np.mean(a)),
                "p10_km": float(np.percentile(a, 10)),
                "p90_km": float(np.percentile(a, 90)),
                "RotFile": os.path.basename(rot_file),
            })

    if all_point_dists:
        a = np.array(all_point_dists, dtype=float)
        summary_rows.append({
            "Age_Ma": float(age),
            "n_trench_features_used": int(trench_feature_count),
            "n_sample_points": int(len(all_point_dists)),
            "min_km": float(np.min(a)),
            "median_km": float(np.median(a)),
            "mean_km": float(np.mean(a)),
            "p10_km": float(np.percentile(a, 10)),
            "p90_km": float(np.percentile(a, 90)),
            "Interval": f"{int(interval[1])}-{int(interval[0])}",
            "Step_km": float(STEP_KM),
            "RotFile": os.path.basename(rot_file),
        })

    print(f"[{age:6.1f} Ma] sub_polylines={n_sub_polylines:5d}  trenches_used={trench_feature_count:4d}  craton_edges={len(cr_edges):4d}  sample_pts={point_count:7d}")

# ==============================
# 7) SAVE
# ==============================
df_point  = pd.DataFrame(per_point_rows)
df_trench = pd.DataFrame(per_trench_rows)
df_sum    = pd.DataFrame(summary_rows)

out1 = safe_to_csv(df_point,  OUT_PER_POINT)
out2 = safe_to_csv(df_trench, OUT_PER_TRENCH)
out3 = safe_to_csv(df_sum,    OUT_SUMMARY)

print("\n[DONE]")
print("  per point :", out1)
print("  per trench:", out2)
print("  summary   :", out3)
print(f"  CRATON_GPML: {CRATON_GPML}")
print(f"  STEP_KM   : {STEP_KM}")
