#!/usr/bin/env python3
"""
Extract frames from N3DV camXX.mp4 videos into:

  scene_dir/
    cam00/frame_000000.png
    cam00/frame_000001.png
    ...
    cam01/frame_000000.png
    ...

Parallelizes across cameras (one ffmpeg process per camera).

Skip logic:
- After a successful extraction, writes camXX/.frames_done.json
- If that file exists and validates (video unchanged + frames complete), skip extraction.

Notes:
- Use --overwrite to force re-extraction (passes -y to ffmpeg). Overwrite behavior is commonly exposed via -y. :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

_CAM_RE = re.compile(r"(cam\d+)\.mp4$", re.IGNORECASE)


@dataclass(frozen=True)
class DoneRecord:
    video_path: str
    video_size: int
    video_mtime_ns: int
    ext: str
    start_number: int
    frame_count: int

    @staticmethod
    def load(path: Path) -> "DoneRecord":
        d = json.loads(path.read_text())
        return DoneRecord(
            video_path=d["video_path"],
            video_size=int(d["video_size"]),
            video_mtime_ns=int(d["video_mtime_ns"]),
            ext=str(d["ext"]),
            start_number=int(d["start_number"]),
            frame_count=int(d["frame_count"]),
        )

    def dump(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "video_path": self.video_path,
                    "video_size": self.video_size,
                    "video_mtime_ns": self.video_mtime_ns,
                    "ext": self.ext,
                    "start_number": self.start_number,
                    "frame_count": self.frame_count,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def _frame_glob(out_dir: Path, ext: str):
    return out_dir.glob(f"frame_*.{ext}")


def _count_frames_on_disk(out_dir: Path, ext: str) -> int:
    return sum(1 for _ in _frame_glob(out_dir, ext))


def _last_frame_path(out_dir: Path, ext: str, start_number: int, frame_count: int) -> Path:
    # If frame_count == N, expected indices are start_number ... start_number + N - 1
    last_idx = start_number + frame_count - 1
    return out_dir / f"frame_{last_idx:06d}.{ext}"


def _stat_sig(path: Path) -> Tuple[int, int]:
    st = path.stat()
    return int(st.st_size), int(st.st_mtime_ns)


def _is_done_and_complete(out_dir: Path, video: Path, ext: str, start_number: int) -> bool:
    done_path = out_dir / ".frames_done.json"
    if not done_path.exists():
        return False

    try:
        rec = DoneRecord.load(done_path)
    except Exception:
        return False

    if rec.ext != ext or rec.start_number != start_number:
        return False

    size, mtime_ns = _stat_sig(video)
    if rec.video_size != size or rec.video_mtime_ns != mtime_ns:
        return False

    if rec.frame_count <= 0:
        return False

    on_disk = _count_frames_on_disk(out_dir, ext)
    if on_disk != rec.frame_count:
        return False

    if not _last_frame_path(out_dir, ext, start_number, rec.frame_count).exists():
        return False

    return True


def _maybe_promote_existing_to_done(out_dir: Path, video: Path, ext: str, start_number: int) -> bool:
    """
    If frames exist but no .frames_done.json, try to infer completeness by:
    - finding the max frame index on disk and ensuring contiguous count equals (max_idx - start_number + 1)
    - requiring that frame_XXXXXX.{ext} naming is consistent
    If it looks complete, write .frames_done.json and return True; else False.
    """
    frames = sorted(_frame_glob(out_dir, ext))
    if not frames:
        return False

    idxs = []
    for p in frames:
        m = re.search(r"frame_(\d+)$", p.stem, re.IGNORECASE)
        if not m:
            return False
        idxs.append(int(m.group(1)))

    idxs.sort()
    # Require the first existing frame to match start_number, and require contiguity.
    if idxs[0] != start_number:
        return False
    expected_count = idxs[-1] - start_number + 1
    if expected_count != len(idxs):
        return False
    # Ensure no gaps.
    for a, b in zip(idxs, idxs[1:]):
        if b != a + 1:
            return False

    size, mtime_ns = _stat_sig(video)
    rec = DoneRecord(
        video_path=str(video),
        video_size=size,
        video_mtime_ns=mtime_ns,
        ext=ext,
        start_number=start_number,
        frame_count=expected_count,
    )
    rec.dump(out_dir / ".frames_done.json")
    return True


def _run_ffmpeg_one(
        video_path: str,
        out_dir: str,
        ext: str,
        start_number: int,
        overwrite: bool,
        ffmpeg_bin: str,
        ffmpeg_threads: Optional[int],
        loglevel: str,
) -> Tuple[str, str]:
    """
    Returns: (cam_name, status_string)
    """
    video = Path(video_path)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    m = _CAM_RE.search(video.name)
    if not m:
        raise ValueError(f"Video name does not match camXX.mp4: {video}")

    cam = m.group(1)

    # Skip if we know it's complete.
    if not overwrite and _is_done_and_complete(out_dir_p, video, ext, start_number):
        return cam, "SKIP(done)"

    # If frames exist but no sentinel, try to validate/promote to done.
    if not overwrite:
        promoted = _maybe_promote_existing_to_done(out_dir_p, video, ext, start_number)
        if promoted and _is_done_and_complete(out_dir_p, video, ext, start_number):
            return cam, "SKIP(promoted)"

        # If frames exist but not complete, don't call ffmpeg with -n (it will fail);
        # ask user to fix explicitly.
        if any(_frame_glob(out_dir_p, ext)):
            raise RuntimeError(
                f"[{cam}] Found existing frames in {out_dir_p} but they don't look complete.\n"
                f"Fix: rerun with --overwrite, or delete {out_dir_p} and rerun."
            )

    out_pat = str(out_dir_p / f"frame_%06d.{ext}")

    cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", loglevel]
    cmd += ["-y"] if overwrite else ["-n"]
    cmd += ["-i", str(video), "-vsync", "0"]
    if ffmpeg_threads is not None:
        cmd += ["-threads", str(int(ffmpeg_threads))]
    cmd += ["-start_number", str(int(start_number)), out_pat]

    print("==>", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"[{cam}] ffmpeg failed (exit {proc.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr}"
        )

    # Record completion
    count = _count_frames_on_disk(out_dir_p, ext)
    if count <= 0:
        raise RuntimeError(f"[{cam}] ffmpeg succeeded but no frames found in {out_dir_p}")

    size, mtime_ns = _stat_sig(video)
    rec = DoneRecord(
        video_path=str(video),
        video_size=size,
        video_mtime_ns=mtime_ns,
        ext=ext,
        start_number=start_number,
        frame_count=count,
    )
    rec.dump(out_dir_p / ".frames_done.json")

    return cam, f"OK({count})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path, help="N3DV scene dir containing camXX.mp4 and poses_bounds.npy")
    ap.add_argument("--ext", default="png", choices=["jpg", "png"], help="Output image extension")
    ap.add_argument("--start_number", type=int, default=0, help="First frame index in filenames")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames")
    ap.add_argument("--workers", type=int, default=None, help="Parallel jobs (default: min(num_videos, cpu_count))")
    ap.add_argument(
        "--backend",
        choices=["processes", "threads"],
        default="processes",
        help="Parallel scheduler for launching ffmpeg tasks",
    )
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary (default: ffmpeg)")
    ap.add_argument(
        "--ffmpeg_threads",
        type=int,
        default=1,
        help="Pass -threads N to EACH ffmpeg (default: 1 to avoid CPU oversubscription)",
    )
    ap.add_argument(
        "--loglevel",
        default="error",
        help="ffmpeg -loglevel (default: error; use info for debugging)",
    )
    args = ap.parse_args()

    scene = args.scene_dir
    videos = sorted(
        [p for p in scene.iterdir() if p.is_file() and _CAM_RE.search(p.name)],
        key=lambda p: int(_CAM_RE.search(p.name).group(1)[3:]),
    )
    if not videos:
        raise FileNotFoundError(f"No camXX.mp4 found under {scene}")

    jobs = []
    for vid in videos:
        cam = _CAM_RE.search(vid.name).group(1)
        out_dir = scene / cam
        jobs.append(
            (
                str(vid),
                str(out_dir),
                args.ext,
                int(args.start_number),
                bool(args.overwrite),
                args.ffmpeg,
                int(args.ffmpeg_threads) if args.ffmpeg_threads is not None else None,
                str(args.loglevel),
            )
        )

    max_workers = args.workers
    if max_workers is None:
        max_workers = min(len(jobs), os.cpu_count() or 1)

    Executor = ProcessPoolExecutor if args.backend == "processes" else ThreadPoolExecutor
    print(f"[INFO] {len(jobs)} videos, backend={args.backend}, workers={max_workers}, overwrite={args.overwrite}")

    ok = 0
    skipped = 0
    with Executor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_ffmpeg_one, *job) for job in jobs]
        for fut in as_completed(futs):
            cam, status = fut.result()
            print(f"[{cam}] {status}")
            if status.startswith("OK"):
                ok += 1
            elif status.startswith("SKIP"):
                skipped += 1

    print(f"[DONE] extracted={ok}, skipped={skipped}")
    print(f"[DONE] frames are in {scene}/camXX/frame_*.{args.ext}")


if __name__ == "__main__":
    main()
