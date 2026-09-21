# -*- coding: utf-8 -*-
"""
步骤：
1) 用 static_polygons 在 0 Ma 分区，给每个样点分配 reconstruction_plate_id（PlateID）。
2) 导出一个 GPML（可在 GPlates 打开，看到带 PlateID 的点）。
3) 按每行 Age(Ma) 重建到古位置，导出 CSV。
"""

import os, csv
import pygplates

# ===== 路径 =====
BASE       = r"C:\Users\Guozhi\OneDrive - Australian National University\Code\PyGplate\Carbonatite1107\PyGplate"
MODEL_DIR  = rf"{BASE}\1.8Ga_model_GSF"

INPUT_CSV  = r"C:\Users\Guozhi\OneDrive - Australian National University\Code\PyGplate\Carbonatite1107\REEdeposit0907.csv"
OUT_GPML   = r"C:\Users\Guozhi\OneDrive - Australian National University\Code\PyGplate\Carbonatite1107\AssignedPoints.gpml"
OUT_CSV    = r"C:\Users\Guozhi\OneDrive - Australian National University\Code\PyGplate\Carbonatite1107\REEdeposit_PaleoCoords0907_4.csv"

ROT_FILES = [
    rf"{MODEL_DIR}\1800_1000_rotfile.rot",
    rf"{MODEL_DIR}\1000_0_rotfile.rot",
]
STATIC_POLY = rf"{MODEL_DIR}\static_polygons.gpmlz"   # 可为 .gpml 或 .gpmlz

# ===== 检查文件 =====
for p in ROT_FILES + [STATIC_POLY, INPUT_CSV]:
    if not os.path.isfile(p):
        raise FileNotFoundError("缺少文件: " + p)

# ===== 载入模型 =====
rotation_model   = pygplates.RotationModel(ROT_FILES)

# ===== 读取 CSV，构建点要素（用 lat/lon）=====
def pick(cols, key):
    key = key.lower()
    for c in cols:
        if key in c.lower():
            return c
    return None

point_features = []
rows_cache     = []  # 保存原始行，后面回填结果

with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    col_age = pick(reader.fieldnames, "age")
    col_lat = pick(reader.fieldnames, "lat")
    col_lon = pick(reader.fieldnames, "lon")
    if not (col_age and col_lat and col_lon):
        raise RuntimeError("CSV 缺少 Age / Latitude / Longitude 列（不区分大小写）")

    for row in reader:
        rows_cache.append(row)
        try:
            lat = float(row[col_lat]); lon = float(row[col_lon])
        except Exception:
            # 先留空，后面 status=skip
            pf = None
        else:
            pt = pygplates.PointOnSphere(lat, lon)
            pf = pygplates.Feature()
            pf.set_geometry(pt)
        point_features.append(pf)

# 过滤掉坐标不合法的（保持索引对齐，用 None 占位）
features_for_partition = [pf for pf in point_features if pf is not None]

# ===== 用 static_polygons 在 0 Ma 分区，复制 PlateID 与有效时段 =====
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
        reconstruction_time=0.0  # 关键：在 0 Ma 分区
    )

# 把 assigned_features 写成 GPML，方便 GPlates 打开看 PlateID
if assigned_features:
    pygplates.FeatureCollection(assigned_features).write(OUT_GPML)

# 为了和原始行对齐，给每个输入点配一个“已分配的要素”或 None
assigned_iter = iter(assigned_features)
aligned_assigned = []
for pf in point_features:
    if pf is None:
        aligned_assigned.append(None)
    else:
        aligned_assigned.append(next(assigned_iter, None))

# ===== 按每行 Age(Ma) 重建到古位置，并写 CSV =====
stats = {"ok":0,"no_plate":0,"no_rotation":0,"recon_fail":0,"recon_empty":0,"skip":0,"bad_age":0}

rows_out = []
for row, assigned in zip(rows_cache, aligned_assigned):
    out_row = dict(row)
    out_row["PlateID"]   = ""
    out_row["Paleo_Lat"] = ""
    out_row["Paleo_Lon"] = ""
    out_row["Status"]    = ""

    # 坐标不合法的行
    if assigned is None:
        out_row["Status"] = "skip"; stats["skip"] += 1
        rows_out.append(out_row)
        continue

    # 取 PlateID
    pid = assigned.get_reconstruction_plate_id()
    if pid is None:
        out_row["Status"] = "no_plate"; stats["no_plate"] += 1
        rows_out.append(out_row)
        continue
    pid = int(pid)
    out_row["PlateID"] = pid

    # 读取年龄
    try:
        age = float(row[pick(row.keys(), "age")])
    except Exception:
        out_row["Status"] = "bad_age"; stats["bad_age"] += 1
        rows_out.append(out_row); continue

    # 检查该 pid 在 age 是否有旋转
    try:
        rotation_model.get_rotation(age, pid)
    except Exception:
        out_row["Status"] = "no_rotation"; stats["no_rotation"] += 1
        rows_out.append(out_row); continue

    # 用“已分配”的要素去重建
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

# 写 CSV（含 BOM，Excel 直接开）
if rows_out:
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)

print("[DONE] PlateID 分配 + 古位置重建完成")
print("  GPML :", OUT_GPML)
print("  CSV  :", OUT_CSV)
print("  统计 :", stats)
