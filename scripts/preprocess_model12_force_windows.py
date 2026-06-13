#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def pad_force(seq_len: int, force: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pad_len = seq_len - force.shape[0]
    if pad_len > 0:
        force = np.concatenate([np.zeros(pad_len, dtype=np.float32), force.astype(np.float32)], axis=0)
        mask = np.concatenate([np.zeros(pad_len, dtype=np.int64), mask.astype(np.int64)], axis=0)
    elif pad_len < 0:
        force = force[-seq_len:].astype(np.float32)
        mask = mask[-seq_len:].astype(np.int64)
    return force.astype(np.float32), mask.astype(np.int64)


def build_windows_for_sample(sample_dir: Path, window_size: int, frame_stride: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    seq = np.load(sample_dir / "sequences.npy", mmap_mode="r")
    seq_mask = np.load(sample_dir / "seq_masks.npy", mmap_mode="r")
    dssp = np.load(sample_dir / "dssp_labels.npy", mmap_mode="r")
    seq_len = int(window_size) * 5
    frames = np.arange(0, dssp.shape[0], max(int(frame_stride), 1), dtype=int)
    if len(frames) == 0 or frames[-1] != dssp.shape[0] - 1:
        frames = np.unique(np.r_[frames, dssp.shape[0] - 1]).astype(int)
    windows = np.zeros((len(frames), seq_len), dtype=np.float32)
    masks = np.zeros((len(frames), seq_len), dtype=np.int64)
    rows = []
    for out_i, frame_i in enumerate(frames):
        force_end = int(frame_i) * 5 + 1
        force_start = max(0, force_end - seq_len)
        force = np.asarray(seq[force_start:force_end], dtype=np.float32)
        mask = np.asarray(seq_mask[force_start:force_end], dtype=np.int64)
        force, mask = pad_force(seq_len, force, mask)
        windows[out_i] = force
        masks[out_i] = mask
        rows.append({
            "local_window_index": out_i,
            "frame_index": int(frame_i),
            "force_start": int(force_start),
            "force_end": int(force_end),
            "force_len": int(seq_len),
            "valid_force_points": int(mask.sum()),
        })
    return windows, masks, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute model12-style raw force windows for clean300")
    parser.add_argument("--manifest", default=str(ROOT / "outputs/preprocess_db_current_clean_300/model18_training_manifest.csv"))
    parser.add_argument("--outdir", default=str(ROOT / "outputs/preprocess_force_windows_clean300_model12_style"))
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=1, help="Dense frame stride for saved force-window DB")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    df = manifest[(manifest["ready"] == True)].copy().sort_values("sample_id_int")
    if args.max_samples is not None:
        df = df.head(args.max_samples)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    for _, row in df.iterrows():
        sample_id = int(row["sample_id_int"])
        sample_dir = Path(row["sample_dir"])
        out_sample = outdir / str(sample_id)
        out_sample.mkdir(parents=True, exist_ok=True)
        windows, masks, meta = build_windows_for_sample(sample_dir, args.window_size, args.frame_stride)
        np.save(out_sample / "force_window.npy", windows)
        np.save(out_sample / "force_window_mask.npy", masks)
        meta.insert(0, "sample_id_int", sample_id)
        meta.insert(1, "source_sample_dir", str(sample_dir))
        meta.to_csv(out_sample / "force_window_metadata.csv", index=False)
        sample_rows.append({
            "sample_id_int": sample_id,
            "split": row.get("split", ""),
            "length": int(row["length"]),
            "sample_dir": str(sample_dir),
            "force_window_dir": str(out_sample),
            "force_window_path": str(out_sample / "force_window.npy"),
            "force_window_mask_path": str(out_sample / "force_window_mask.npy"),
            "metadata_path": str(out_sample / "force_window_metadata.csv"),
            "windows": int(windows.shape[0]),
            "force_len": int(windows.shape[1]),
            "window_size": int(args.window_size),
            "frame_stride": int(args.frame_stride),
            "ready": True,
        })
        print(f"saved sample {sample_id}: {windows.shape}", flush=True)
    out_manifest = pd.DataFrame(sample_rows)
    out_manifest.to_csv(outdir / "force_window_manifest.csv", index=False)
    summary = {
        "samples": int(len(out_manifest)),
        "total_windows": int(out_manifest["windows"].sum()) if len(out_manifest) else 0,
        "window_size": int(args.window_size),
        "force_len": int(args.window_size * 5),
        "frame_stride": int(args.frame_stride),
        "source_manifest": str(args.manifest),
        "outdir": str(outdir),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
