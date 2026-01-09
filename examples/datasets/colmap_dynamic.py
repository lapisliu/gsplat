import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager
from typing_extensions import assert_never

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


@dataclass
class TimestampCfg:
    """Configuration for computing per-frame timestamps.

    - fps: frames-per-second used to convert frame index -> seconds. If None/<=0, uses raw index.
    - d_fps: temporal downsample factor (keep every d_fps-th frame).
    - vid_time_start / vid_time_end: optional slicing window in seconds; interval is [start, end).
      If end<=0 or end<=start, the window is treated as open-ended.
    - normalize: if True, normalize timestamps to [0, 1] per camera after slicing/downsampling.
    """

    fps: Optional[float] = 30.0
    d_fps: Optional[float] = 1.0
    vid_time_start: float = 0.0
    vid_time_end: float = 0.0
    normalize: bool = False


class ParserDynamic:
    """Dynamic COLMAP parser that expands per-camera COLMAP templates into all frames on disk.

    Expected layout:
        dataset_root/
            cam01/frame_00001.jpg ...   (or)   cam02/000000.png ...
            sparse/0/  (or sparse/)  cameras.bin, images.bin, points3D.bin

    Splits:
        - `test_cams` is REQUIRED and specifies camera-folder names used for val/test.
        - train is all frames whose camera folder is NOT in `test_cams`.

    Notes:
        - Camera poses/intrinsics are repeated for all frames in that camera folder.
        - Per-frame timestamps are computed from frame indices parsed from filenames,
          grouped by camera folder.
    """

    # Matches "frame_000123" or trailing digits like "000123.png"
    _FRAME_RE = re.compile(r"(?:frame[_-]?)(\d+)|(\d+)$", re.IGNORECASE)

    def __init__(
        self,
        data_dir: str,
        test_cams: List[str],
        factor: int = 1,
        normalize: bool = False,
        timestamp_cfg: TimestampCfg = TimestampCfg(),
    ) -> None:
        if not test_cams:
            raise ValueError("test_cams is required and must be a non-empty list of camera folder names.")

        self.data_dir = data_dir
        self.factor = int(factor)
        self.normalize = bool(normalize)
        self.timestamp_cfg = timestamp_cfg
        self.test_cams: Set[str] = set(test_cams)

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist."

        # Optional pre-downsampled dynamic folder (d_{factor}/camXX/..)
        if self.factor != 1:
            self.data_dir = os.path.join(data_dir, f"d_{self.factor}")
            assert os.path.exists(self.data_dir), (
                f"Downsampled dynamic data directory {self.data_dir} does not exist."
            )
            print(f"[ParserDynamic] Using downsampled data directory: {self.data_dir}")

        # Effective FPS (useful for logging/metadata)
        self.effective_fps = (self.timestamp_cfg.fps or 0.0) / max(
            1, int(round(self.timestamp_cfg.d_fps or 1))
        )

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # 3D points
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        print(f"number of 3D points: {points.shape[0]}")

        imdata = manager.images
        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")

        # Build point visibility using COLMAP template image names (used by depth loss)
        point_indices_template: Dict[str, np.ndarray] = {}
        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        tmp: Dict[str, List[int]] = {}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                tmp.setdefault(image_name, []).append(point_idx)
        for k, v in tmp.items():
            point_indices_template[k] = np.array(v, dtype=np.int32)

        bottom = np.array([0, 0, 0, 1], dtype=np.float64).reshape(1, 4)

        # Gather per-camera COLMAP template records (sorted for determinism)
        colmap_images_sorted = [
            imdata[k] for k in sorted(imdata.keys(), key=lambda x: imdata[x].name)
        ]

        cam_template: List[Dict[str, Any]] = []
        for im in colmap_images_sorted:
            cam_key = os.path.splitext(os.path.basename(im.name))[0]
            cam_id = im.camera_id

            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], axis=1), bottom], axis=0)

            cam = manager.cameras[cam_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            K[:2, :] /= float(self.factor)

            type_ = cam.camera_type
            if type_ in (0, "SIMPLE_PINHOLE", 1, "PINHOLE"):
                dist = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ in (2, "SIMPLE_RADIAL"):
                dist = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ in (3, "RADIAL"):
                dist = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ in (4, "OPENCV"):
                dist = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ in (5, "OPENCV_FISHEYE"):
                dist = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            else:
                raise ValueError(f"Unsupported camera type: {type_}")

            cam_template.append(
                {
                    "cam_key": cam_key,
                    "template_image_name": im.name,
                    "camera_id": cam_id,
                    "w2c": w2c,
                    "K": K,
                    "dist": dist,
                    "camtype": camtype,
                    "nominal_size": (cam.width // self.factor, cam.height // self.factor),
                }
            )

            # import pdb; pdb.set_trace()

        # ---------- Frame listing + filtering (per camera folder) ----------

        def _list_frames(cam_key: str) -> List[str]:
            root = os.path.join(self.data_dir, cam_key)
            if not os.path.isdir(root):
                raise FileNotFoundError(
                    f"Camera folder not found for '{cam_key}'. Expected directory: {root}"
                )
            frames = [
                os.path.join(cam_key, f)
                for f in os.listdir(root)
                if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")
            ]

            def _ns_key(p: str):
                b = os.path.splitext(os.path.basename(p))[0]
                if b.isdigit():
                    return (0, int(b))
                fi = self._frame_index_from_name(p)
                if fi is not None:
                    return (0, fi)
                return (1, b)

            frames.sort(key=_ns_key)
            if not frames:
                raise FileNotFoundError(f"No frames found under {root}")
            return frames

        def _time_slice_frames(frames_rel: List[str]) -> List[str]:
            """Slice a camera's frame list by [vid_time_start, vid_time_end) in seconds."""
            fps = self.timestamp_cfg.fps
            if fps is None or fps <= 0:
                return frames_rel

            start_s = max(0.0, float(self.timestamp_cfg.vid_time_start or 0.0))
            end_s = float(self.timestamp_cfg.vid_time_end or 0.0)

            start_idx = int(np.floor(start_s * fps))
            if end_s <= 0 or end_s <= start_s:
                end_idx = len(frames_rel)  # open-ended
            else:
                end_idx = int(np.ceil(end_s * fps))  # exclusive

            start_idx = max(0, min(start_idx, len(frames_rel)))
            end_idx = max(start_idx, min(end_idx, len(frames_rel)))
            return frames_rel[start_idx:end_idx]

        d_step = max(1, int(round(self.timestamp_cfg.d_fps or 1)))

        def _apply_d_fps(frames_rel: List[str]) -> List[str]:
            """Keep only frames where frame_index % d_step == 0; fallback to stride."""
            if d_step == 1:
                return frames_rel
            kept: List[str] = []
            for rel in frames_rel:
                fi = self._frame_index_from_name(rel)
                if fi is None:
                    return frames_rel[::d_step]
                if (fi % d_step) == 0:
                    kept.append(rel)
            return kept

        # ---------- First pass: per-camera scaling + undistortion maps ----------

        self.mapx_dict: Dict[int, np.ndarray] = {}
        self.mapy_dict: Dict[int, np.ndarray] = {}
        self.roi_undist_dict: Dict[int, List[int]] = {}

        Ks_dict: Dict[int, np.ndarray] = {}
        params_dict: Dict[int, np.ndarray] = {}
        imsize_dict: Dict[int, Tuple[int, int]] = {}
        mask_dict: Dict[int, Optional[np.ndarray]] = {}

        for rec in cam_template:
            cam_id = rec["camera_id"]
            cam_key = rec["cam_key"]
            frames_rel = _apply_d_fps(_time_slice_frames(_list_frames(cam_key)))
            if not frames_rel:
                raise FileNotFoundError(
                    f"No frames remain for camera '{cam_key}' after time slicing / d_fps filtering."
                )

            first_frame_path = os.path.join(self.data_dir, frames_rel[0])
            im0 = imageio.imread(first_frame_path)[..., :3]
            h0, w0 = im0.shape[:2]

            # Scale K from COLMAP nominal size -> actual frame size
            colmap_w, colmap_h = rec["nominal_size"]
            s_h, s_w = h0 / float(colmap_h), w0 / float(colmap_w)
            K_scaled = rec["K"].copy()
            K_scaled[0, :] *= s_w
            K_scaled[1, :] *= s_h

            dist = rec["dist"]
            camtype = rec["camtype"]
            params_dict[cam_id] = dist

            if len(dist) == 0:
                Ks_dict[cam_id] = K_scaled
                imsize_dict[cam_id] = (w0, h0)
                mask_dict[cam_id] = None
                self.roi_undist_dict[cam_id] = [0, 0, w0, h0]
                continue

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K_scaled, dist, (w0, h0), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K_scaled, dist, None, K_undist, (w0, h0), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx, fy = float(K_scaled[0, 0]), float(K_scaled[1, 1])
                cx, cy = float(K_scaled[0, 2]), float(K_scaled[1, 2])
                grid_x, grid_y = np.meshgrid(
                    np.arange(w0, dtype=np.float32),
                    np.arange(h0, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                k1, k2, k3, k4 = dist
                r = 1.0 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8
                mapx = (fx * x1 * r + w0 // 2).astype(np.float32)
                mapy = (fy * y1 * r + h0 // 2).astype(np.float32)

                mask_full = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < w0 - 1, mapy < h0 - 1),
                )
                y_indices, x_indices = np.nonzero(mask_full)
                y_min, y_max = int(y_indices.min()), int(y_indices.max() + 1)
                x_min, x_max = int(x_indices.min()), int(x_indices.max() + 1)
                mask = mask_full[y_min:y_max, x_min:x_max]

                K_undist = K_scaled.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[cam_id] = mapx
            self.mapy_dict[cam_id] = mapy
            Ks_dict[cam_id] = K_undist
            self.roi_undist_dict[cam_id] = roi_undist
            imsize_dict[cam_id] = (int(roi_undist[2]), int(roi_undist[3]))
            mask_dict[cam_id] = mask

        # ---------- Second pass: expand per-camera templates into per-frame lists ----------

        image_names: List[str] = []
        image_paths: List[str] = []
        camtoworlds_list: List[np.ndarray] = []
        camera_ids_list: List[int] = []
        point_indices: Dict[str, np.ndarray] = {}

        for rec in cam_template:
            cam_id = rec["camera_id"]
            cam_key = rec["cam_key"]
            frames_rel = _apply_d_fps(_time_slice_frames(_list_frames(cam_key)))
            if not frames_rel:
                raise FileNotFoundError(
                    f"No frames remain for camera '{cam_key}' after time slicing / d_fps filtering."
                )

            c2w = np.linalg.inv(rec["w2c"])
            tmpl_name = rec["template_image_name"]
            tmpl_pts = point_indices_template.get(tmpl_name, np.empty((0,), dtype=np.int32))

            for rel in frames_rel:
                image_names.append(rel)
                image_paths.append(os.path.join(self.data_dir, rel))
                camtoworlds_list.append(c2w)
                camera_ids_list.append(cam_id)
                point_indices[rel] = tmpl_pts

        camtoworlds = np.stack(camtoworlds_list, axis=0)

        # Normalize world (same as Parser)
        if self.normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Optional fix for up-side-down scenes (kept consistent with Parser)
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4, dtype=np.float64)

        # Public fields (match Parser naming)
        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids_list
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.mask_dict = mask_dict
        self.points = points
        self.points_err = points_err
        self.points_rgb = points_rgb
        self.point_indices = point_indices
        self.transform = transform

        # Camera folder per image (first path segment)
        self.image_cameras: List[str] = [
            (os.path.normpath(n).split(os.sep)[0] if os.sep in n else "")
            for n in self.image_names
        ]

        # Per-frame timestamps
        self.timestamps: np.ndarray = self._compute_timestamps_index_only()
        assert len(self.timestamps) == len(self.image_names)

        # Split indices
        self.train_indices: np.ndarray = np.array(
            [i for i, cam in enumerate(self.image_cameras) if cam not in self.test_cams],
            dtype=np.int64,
        )
        self.test_indices: np.ndarray = np.array(
            [i for i, cam in enumerate(self.image_cameras) if cam in self.test_cams],
            dtype=np.int64,
        )
        self.val_indices: np.ndarray = self.test_indices

        # Scene scale
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = float(np.max(dists))

        # import pdb; pdb.set_trace()

    # ---- helpers ----

    def _frame_index_from_name(self, name: str) -> Optional[int]:
        base = os.path.basename(name)
        m = self._FRAME_RE.search(base)
        if not m:
            return None
        g = m.group(1) or m.group(2)
        try:
            return int(g)
        except Exception:
            return None

    def _compute_timestamps_index_only(self) -> np.ndarray:
        cfg = self.timestamp_cfg
        # group by camera folder
        per_cam_indices: Dict[str, List[int]] = {}
        for i, rel in enumerate(self.image_names):
            cam = rel.split(os.sep)[0] if os.sep in rel else ""
            per_cam_indices.setdefault(cam, []).append(i)

        def _normalize_vec(v: np.ndarray) -> np.ndarray:
            if v.size == 0:
                return v
            vmin, vmax = float(v.min()), float(v.max())
            if vmax == vmin:
                return np.zeros_like(v)
            return (v - vmin) / (vmax - vmin)

        ts = np.zeros(len(self.image_names), dtype=np.float32)
        for cam, idxs in per_cam_indices.items():
            vals: List[float] = []
            for i in idxs:
                name = self.image_names[i]
                fi = self._frame_index_from_name(name)
                vals.append(float(fi if fi is not None else len(vals)))
            v = np.asarray(vals, dtype=np.float32)
            if cfg.fps is not None and cfg.fps > 0:
                v = v / float(cfg.fps)
            if cfg.normalize:
                v = _normalize_vec(v)
            ts[idxs] = v
        return ts


class DatasetDynamic:
    """Dataset for ParserDynamic.

    Mirrors the behavior of the original 3DGS Dataset (undistort + optional random crop + optional depth loss),
    but also returns per-item `timestamps`.
    """

    def __init__(
        self,
        parser: ParserDynamic,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths

        split_l = split.lower()
        if split_l == "train":
            self.indices = np.asarray(self.parser.train_indices, dtype=np.int64)
        elif split_l in ("val", "test"):
            self.indices = np.asarray(self.parser.val_indices, dtype=np.int64)
        else:
            raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'test'.")

        if self.indices.size == 0:
            raise ValueError(f"No indices found for split '{split}'. Check test_cams and camera folders.")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = int(self.indices[item])
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        image_name = self.parser.image_names[index]  # e.g. "cam00/000000.png"
        cam_key = image_name.split(os.sep)[0] if os.sep in image_name else ""
        fi = self.parser._frame_index_from_name(image_name)
        frame_idx = int(fi if fi is not None else item)

        if len(params) > 0:
            mapx, mapy = self.parser.mapx_dict[camera_id], self.parser.mapy_dict[camera_id]
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x0, y0, w0, h0 = self.parser.roi_undist_dict[camera_id]
            image = image[y0 : y0 + h0, x0 : x0 + w0]

        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = int(np.random.randint(0, max(w - self.patch_size, 1)))
            y = int(np.random.randint(0, max(h - self.patch_size, 1)))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data: Dict[str, Any] = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # index within this dataset split
            "timestamps": torch.tensor([float(self.parser.timestamps[index])], dtype=torch.float32),
            "image_name": image_name,
            "cam_key": cam_key,
            "frame_idx": torch.tensor(frame_idx, dtype=torch.int64),
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices.get(image_name, None)
            if point_indices is None or point_indices.size == 0:
                data["points"] = torch.empty((0, 2), dtype=torch.float32)
                data["depths"] = torch.empty((0,), dtype=torch.float32)
                return data

            points_world = self.parser.points[point_indices]
            points_cam = (
                worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]
            ).T
            points_proj = (K @ points_cam.T).T
            pts = points_proj[:, :2] / points_proj[:, 2:3]
            depths = points_cam[:, 2]
            selector = (
                (pts[:, 0] >= 0)
                & (pts[:, 0] < image.shape[1])
                & (pts[:, 1] >= 0)
                & (pts[:, 1] < image.shape[0])
                & (depths > 0)
            )
            pts = pts[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(pts).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data
