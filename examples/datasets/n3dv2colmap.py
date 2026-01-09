#!/usr/bin/env python3
import argparse
import math
import re
import shutil
import subprocess
import sqlite3
from pathlib import Path

import numpy as np

_CAM_DIR_RE = re.compile(r"^cam\d+$", re.IGNORECASE)


def run(cmd):
    print("==>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def rotmat_to_qvec(R):
    # Returns (qw,qx,qy,qz) for a proper rotation matrix
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return qw, qx, qy, qz


def list_frames(cam_dir: Path):
    frames = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if not frames:
        raise FileNotFoundError(f"No frames under {cam_dir}")

    def key(p: Path):
        stem = p.stem
        m = re.search(r"(?:frame[_-]?)(\d+)$", stem, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m2 = re.search(r"(\d+)$", stem)
        if m2:
            return int(m2.group(1))
        return 10**18

    frames.sort(key=key)
    return frames


def read_model_image_names(images_txt: Path):
    names = []
    for line in images_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # images.txt: IMAGE_ID ... CAMERA_ID NAME
        # followed by an optional second line for points2D (we leave it empty)
        if len(parts) >= 10:
            names.append(parts[-1])
    return names


def read_db_image_names(db_path: Path):
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT name FROM images").fetchall()
        return {r[0] for r in rows}
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path, help="N3DV scene dir (camXX/frames extracted, plus poses_bounds.npy)")
    ap.add_argument("--keyframe_index", type=int, default=0, help="Which extracted frame (sorted index) to use as template")
    ap.add_argument("--colmap_bin", default="colmap", help="Path to colmap executable")
    ap.add_argument("--use_gpu", type=int, default=1, help="Use GPU for feature extraction/matching (0/1)")
    ap.add_argument("--max_image_size", type=int, default=3200, help="FeatureExtraction.max_image_size")
    ap.add_argument("--workspace", type=Path, default=None, help="Workspace dir (default: scene_dir/colmap_workspace)")
    ap.add_argument("--dense", action="store_true", help="Also run MVS and output dense fused ply")
    args = ap.parse_args()

    scene = args.scene_dir
    poses_path = scene / "poses_bounds.npy"
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing {poses_path}")

    cam_dirs = sorted(
        [p for p in scene.iterdir() if p.is_dir() and _CAM_DIR_RE.match(p.name)],
        key=lambda p: int(p.name[3:]),
    )
    if not cam_dirs:
        raise FileNotFoundError(f"No camXX/ directories under {scene}. Did you run n3dv_extract_frames.py?")

    poses_bounds = np.load(str(poses_path))
    N = poses_bounds.shape[0]
    if N != len(cam_dirs):
        raise ValueError(f"poses_bounds has N={N} but found {len(cam_dirs)} camera folders (sorted camXX).")

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N,3,5)
    H, W, fl = poses[0, :, -1]
    H, W, fl = float(H), float(W), float(fl)
    cx, cy = W * 0.5, H * 0.5

    ws = args.workspace if args.workspace is not None else (scene / "colmap_workspace")
    img_ws = ws / "images"
    created = ws / "created" / "sparse"
    tri = ws / "triangulated" / "sparse"
    db = ws / "database.db"
    image_list = ws / "image_list.txt"

    # Reset workspace
    if ws.exists():
        shutil.rmtree(ws)
    img_ws.mkdir(parents=True, exist_ok=True)
    created.mkdir(parents=True, exist_ok=True)
    tri.mkdir(parents=True, exist_ok=True)

    # Stage ONE template PNG per camera into img_ws as camXX.png
    staged_names = []
    for cam_dir in cam_dirs:
        frames = list_frames(cam_dir)
        k = min(args.keyframe_index, len(frames) - 1)
        src = frames[k]

        dst = img_ws / f"{cam_dir.name}.png"
        if src.suffix.lower() == ".png":
            shutil.copy2(src, dst)
        else:
            # Convert to PNG if the source frame isn't PNG (keeps template extension consistent)
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), str(dst)])

        staged_names.append(dst.name)

    # Write image_list.txt (relative names under --image_path)
    image_list.write_text("\n".join(staged_names) + "\n")

    # Write cameras.txt / images.txt / points3D.txt (COLMAP text model)
    cam_txt = created / "cameras.txt"
    img_txt = created / "images.txt"
    pts_txt = created / "points3D.txt"

    with open(cam_txt, "w") as f:
        f.write(f"1 PINHOLE {int(W)} {int(H)} {fl} {fl} {cx} {cy}\n")

    with open(img_txt, "w") as f:
        for i, cam_dir in enumerate(cam_dirs, start=1):
            # LLFF pose: 3x5, columns are [down, right, back] + position
            P = poses[i - 1]
            R_llff = P[:, 0:3]
            t = P[:, 3]

            # Convert LLFF -> camera-to-world (OpenCV-ish)
            R_c2w = np.stack([R_llff[:, 1], R_llff[:, 0], -R_llff[:, 2]], axis=1)

            # World-to-camera for COLMAP
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t.reshape(3,)

            qw, qx, qy, qz = rotmat_to_qvec(R_w2c)

            name = f"{cam_dir.name}.png"  # must match DB 'images.name' exactly
            f.write(f"{i} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} 1 {name}\n\n")

    pts_txt.write_text("")

    # Fresh DB
    if db.exists():
        db.unlink()

    # feature_extractor: use image_list_path so DB names match images.txt names exactly :contentReference[oaicite:3]{index=3}
    run(
        [
            args.colmap_bin,
            "feature_extractor",
            "--database_path",
            str(db),
            "--image_path",
            str(img_ws),
            "--image_list_path",
            str(image_list),
            "--camera_mode",
            "1",  # SINGLE :contentReference[oaicite:4]{index=4}
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.camera_params",
            f"{fl},{fl},{cx},{cy}",
        ]
    )

    # Sanity-check: every NAME in images.txt must exist in DB (DB stores relative paths under image_path) :contentReference[oaicite:5]{index=5}
    model_names = read_model_image_names(img_txt)
    db_names = read_db_image_names(db)
    missing = [n for n in model_names if n not in db_names]
    if missing:
        raise RuntimeError(
            "Some template images were NOT inserted into COLMAP's database.\n"
            "Missing names (must match exactly):\n  - "
            + "\n  - ".join(missing)
            + "\n\nCommon causes: unreadable/corrupt PNG, missing PNG support in your COLMAP build, or files not present under workspace/images/.\n"
              f"Check: ls -l {img_ws} and try running: {args.colmap_bin} feature_extractor -h"
        )

    # exhaustive_matcher (GPU toggle uses FeatureMatching.use_gpu per CLI docs) :contentReference[oaicite:6]{index=6}
    run(
        [
            args.colmap_bin,
            "exhaustive_matcher",
            "--database_path",
            str(db),
        ]
    )

    # point_triangulator uses your known-pose model from created/sparse :contentReference[oaicite:7]{index=7}
    run(
        [
            args.colmap_bin,
            "point_triangulator",
            "--database_path",
            str(db),
            "--image_path",
            str(img_ws),
            "--input_path",
            str(created),
            "--output_path",
            str(tri),
        ]
    )

    # Copy to scene_dir/sparse/0/
    out_sparse0 = scene / "sparse" / "0"
    out_sparse0.mkdir(parents=True, exist_ok=True)
    for fn in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = tri / fn
        if not src.exists():
            raise FileNotFoundError(f"Expected {src} (COLMAP output).")
        shutil.copy2(src, out_sparse0 / fn)

    print(f"[OK] Wrote COLMAP model to: {out_sparse0}")

    if args.dense:
        dense = ws / "dense"
        dense.mkdir(parents=True, exist_ok=True)
        run(
            [
                args.colmap_bin,
                "image_undistorter",
                "--image_path",
                str(img_ws),
                "--input_path",
                str(tri),
                "--output_path",
                str(dense),
                "--output_type",
                "COLMAP",
                "--max_image_size",
                str(int(args.max_image_size)),
            ]
        )
        run(
            [
                args.colmap_bin,
                "patch_match_stereo",
                "--workspace_path",
                str(dense),
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "true",
            ]
        )
        fused = scene / "points3D.ply"
        run(
            [
                args.colmap_bin,
                "stereo_fusion",
                "--workspace_path",
                str(dense),
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric",
                "--StereoFusion.min_num_pixels",
                "3",
                "--output_path",
                str(fused),
            ]
        )
        print(f"[OK] Dense fused point cloud: {fused}")


if __name__ == "__main__":
    main()
