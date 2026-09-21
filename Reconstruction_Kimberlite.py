# -*- coding: utf-8 -*-

import os, csv
import pygplates

# ===== 路径 =====
BASE       = r"C:\Users\Guozhi\OneDrive - Australian National University\Code\PyGplate\Carbonatite1107\PyGPlate"
MODEL_DIR  = rf"{BASE}\1.8Ga_model_GSF"

INPUT_CSV  = r"C:\Users\Guozhi\OneDrive - Australian National University\Code\PyGplate\Carbonatite1107\Kimber_Age_Nd.csv"
OUT_GPML   = r"C:\Users\Guozhi\OneDrive - Australian National University\Code\PyGplate\Carbonatite1107\AssignedPoints.gpml"
OUT_CSV    = r"C:\Users\Guozhi\OneDrive - Australian National University\Code\PyGplate\Carbonatite1107\Kimber_Age_Nd_PaleoCoords.csv"

ROT_FILES = [
    rf"{MODEL_DIR}\1800_1000_rotfile.rot",
    rf"{MODEL_DIR}\1000_0_rotfile.rot",
]
STATIC_POLY = rf"{MODEL_DIR}\static_polygons.gpmlz"


def pick_col(cols, keywords):

    if isinstance(keywords, str):
        keywords = [keywords]
    cols_l = [(c, c.lower().strip()) for c in cols]
    for kw in keywords:
        kw = kw.lower().strip()
        for c, cl in cols_l:
            if kw in cl:
                return c
    return None

def to_float(x):
    if x is None:
        raise ValueError("None")
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "none", "null"}:
        raise ValueError("empty/na")
    return float(s)


for p in ROT_FILES + [STATIC_POLY, INPUT_CSV]:
    if not os.path.isfile(p):
        raise FileNotFoundError("缺少文件: " + p)


rotation_model = pygplates.RotationModel(ROT_FILES)


point_features = []
rows_cache     = []

with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise RuntimeError("CSV nothing")


    col_age = pick_col(reader.fieldnames, ["age (ma)", "age", "ma"])
    col_lat = pick_col(reader.fieldnames, ["latitude", "lat"])
    col_lon = pick_col(reader.fieldnames, ["longitude", "long", "lon"])

    if not (col_age and col_lat and col_lon):
        raise RuntimeError(
            "CSV lost Age / Latitude / Longitude \n"
            f"Age={col_age}, Lat={col_lat}, Lon={col_lon}"
        )

    for row in reader:
        rows_cache.append(row)

        try:
            lat = to_float(row.get(col_lat))
            lon = to_float(row.get(col_lon))
        except Exception:
            point_features.append(None)
            continue

        pt = pygplates.PointOnSphere(lat, lon)
        pf = pygplates.Feature()
        pf.set_geometry(pt)
        point_features.append(pf)


features_for_partition = [pf for pf in point_features if pf is not None]

assigned_features = []
if features_for_partition:
    assigned_features = pygplates.partition_into_plates(
        STATIC_POLY,
        rotation_model,
        features_for_partition,
        properties_to_copy=[
            pygplates.PartitionProperty.reconstruction_plate_id,
            pygplates.PartitionProperty.valid_time_period
        ],
        reconstruction_time=0.0
    )


if assigned_features:
    pygplates.FeatureCollection(assigned_features).write(OUT_GPML)


assigned_iter = iter(assigned_features)
aligned_assigned = []
for pf in point_features:
    aligned_assigned.append(None if pf is None else next(assigned_iter, None))


stats = {"ok":0,"no_plate":0,"no_rotation":0,"recon_fail":0,"recon_empty":0,"skip":0,"bad_age":0}

rows_out = []
for row, assigned in zip(rows_cache, aligned_assigned):
    out_row = dict(row)
    out_row["PlateID"]   = ""
    out_row["Paleo_Lat"] = ""
    out_row["Paleo_Lon"] = ""
    out_row["Status"]    = ""

    
    if assigned is None:
        out_row["Status"] = "skip"; stats["skip"] += 1
        rows_out.append(out_row)
        continue

    # PlateID
    pid = assigned.get_reconstruction_plate_id()
    if pid is None:
        out_row["Status"] = "no_plate"; stats["no_plate"] += 1
        rows_out.append(out_row)
        continue
    pid = int(pid)
    out_row["PlateID"] = pid

    # Age (Ma)
    try:
        age = to_float(row.get(col_age))
    except Exception:
        out_row["Status"] = "bad_age"; stats["bad_age"] += 1
        rows_out.append(out_row); continue

    
    try:
        rotation_model.get_rotation(age, pid)
    except Exception:
        out_row["Status"] = "no_rotation"; stats["no_rotation"] += 1
        rows_out.append(out_row); continue

    
    rec = []
    try:
        pygplates.reconstruct([assigned], rotation_model, rec, age)
    except Exception:
        out_row["Status"] = "recon_fail"; stats["recon_fail"] += 1
        rows_out.append(out_row); continue

    if not rec or not rec[0].get_reconstructed_geometry():
        out_row["Status"] = "recon_empty"; stats["recon_empty"] += 1
        rows_out.append(out_row); continue

    rlat, rlon = rec[0].get_reconstructed_geometry().to_lat_lon()
    out_row["Paleo_Lat"] = rlat
    out_row["Paleo_Lon"] = rlon
    out_row["Status"]    = "ok"; stats["ok"] += 1

    rows_out.append(out_row)


if rows_out:
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

print("[DONE] PlateID+Reconstruction")
print("  GPML :", OUT_GPML)
print("  CSV  :", OUT_CSV)
print("  Number :", stats)
print("  Used:", {"Age": col_age, "Lat": col_lat, "Lon": col_lon})
