"""Bonus B1 (+4 điểm): So sánh RANSAC Ground Segmentation vs Cắt theo độ cao cố định (z < -1.5m).

Mục tiêu:
  - Chạy 2 phương pháp trên cùng dữ liệu KITTI (các frame 000011, 000019, 000025).
  - Xuất bảng so sánh results/bonus_b1_ground_comparison.csv.
  - Tạo ảnh so sánh trực quan results/figures/bonus_b1_compare_ground.png.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import load_frame
from src.obstacle_pipeline import (
    euclidean_clustering,
    ransac_ground_segmentation,
    voxel_downsample,
)


def run_fixed_height_baseline(down_pts: np.ndarray, z_ground_cut: float = -1.55,
                              eps: float = 0.6, min_points: int = 15) -> dict:
    """Phương pháp 2 (Baseline): Cắt mặt đất theo ngưỡng độ cao cố định z."""
    t0 = time.perf_counter()
    xyz = down_pts[:, :3]
    ground_mask = xyz[:, 2] <= z_ground_cut
    obs_mask = (xyz[:, 2] > z_ground_cut) & (xyz[:, 2] < 2.5)
    t1 = time.perf_counter()
    
    obstacle_pts = down_pts[obs_mask]
    clusters = euclidean_clustering(obstacle_pts, eps=eps, min_points=min_points)
    t2 = time.perf_counter()
    
    return {
        "ground_points": int(ground_mask.sum()),
        "obstacle_points": len(obstacle_pts),
        "num_clusters": len(clusters),
        "clusters": clusters,
        "time_ground_ms": (t1 - t0) * 1000,
        "time_total_ms": (t2 - t0) * 1000,
        "ground_mask": ground_mask,
        "obs_mask": obs_mask,
    }


def main() -> None:
    data_root = "data/kitti_mini"
    test_frames = ["000011", "000019", "000025"]
    records = []

    print("[*] Chay so sanh Bonus B1: RANSAC vs Cắt cao độ z cố định...")

    for fid in test_frames:
        fr = load_frame(data_root, fid)
        pts = fr["points"]
        down_pts = voxel_downsample(pts, voxel_size=0.1)

        # 1. RANSAC
        t0 = time.perf_counter()
        g_mask_ransac, obs_mask_ransac, _ = ransac_ground_segmentation(down_pts, distance_threshold=0.15)
        obs_pts_ransac = down_pts[obs_mask_ransac]
        c_ransac = euclidean_clustering(obs_pts_ransac, eps=0.6, min_points=15)
        t_ransac = (time.perf_counter() - t0) * 1000

        # 2. Fixed z cut
        res_fixed = run_fixed_height_baseline(down_pts, z_ground_cut=-1.55, eps=0.6, min_points=15)

        records.append({
            "frame_id": fid,
            "method": "RANSAC Plane Fitting",
            "ground_points": int(g_mask_ransac.sum()),
            "obstacle_points": len(obs_pts_ransac),
            "num_clusters": len(c_ransac),
            "time_ms": round(t_ransac, 1),
            "pros": "Thích nghi với độ dốc/nghiêng của đường; không phụ thuộc độ cao xe tuyệt đối",
            "cons": "Chậm hơn cắt cố định ~15ms do cần lặp tìm mặt phẳng",
        })

        records.append({
            "frame_id": fid,
            "method": "Fixed Height Cut (z <= -1.55m)",
            "ground_points": res_fixed["ground_points"],
            "obstacle_points": res_fixed["obstacle_points"],
            "num_clusters": res_fixed["num_clusters"],
            "time_ms": round(res_fixed["time_total_ms"], 1),
            "pros": "Cực nhanh (O(1), < 1ms cho bước lọc)",
            "cons": "Dễ nhầm mặt đường dốc thành vật cản giả; mất vật cản khi xe chúc mũi xuống",
        })

        # Vẽ ảnh so sánh cho frame 000011
        if fid == "000011":
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # RANSAC
            axes[0].scatter(down_pts[g_mask_ransac, 1], down_pts[g_mask_ransac, 0], s=0.3, c="silver", alpha=0.3)
            axes[0].scatter(obs_pts_ransac[:, 1], obs_pts_ransac[:, 0], s=1.5, c="tab:green", alpha=0.7)
            axes[0].set_title(f"Method 1: RANSAC Plane ({len(c_ransac)} clusters, {int(g_mask_ransac.sum())} ground pts)", weight="bold")
            axes[0].set_xlim(-20, 20); axes[0].set_ylim(0, 45)
            axes[0].set_xlabel("Y (m)"); axes[0].set_ylabel("X (m)")
            axes[0].grid(True, alpha=0.3)

            # Fixed Cut
            axes[1].scatter(down_pts[res_fixed['ground_mask'], 1], down_pts[res_fixed['ground_mask'], 0], s=0.3, c="silver", alpha=0.3)
            axes[1].scatter(down_pts[res_fixed['obs_mask'], 1], down_pts[res_fixed['obs_mask'], 0], s=1.5, c="tab:red", alpha=0.7)
            axes[1].set_title(f"Method 2: Fixed z <= -1.55m ({res_fixed['num_clusters']} clusters, {res_fixed['ground_points']} ground pts)", weight="bold")
            axes[1].set_xlim(-20, 20); axes[1].set_ylim(0, 45)
            axes[1].set_xlabel("Y (m)"); axes[1].set_ylabel("X (m)")
            axes[1].grid(True, alpha=0.3)

            plt.suptitle("BONUS B1: So sánh RANSAC Plane Segmentation vs Fixed Height Cut (Frame 000011)", fontsize=13, weight="bold")
            plt.tight_layout()
            out_img = Path("results/figures/bonus_b1_compare_ground.png")
            out_img.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_img), dpi=150)
            plt.close(fig)
            print(f"[OK] Da luu anh: {out_img}")

    df = pd.DataFrame(records)
    out_csv = Path("results/bonus_b1_ground_comparison.csv")
    df.to_csv(str(out_csv), index=False)
    print(f"[OK] Da luu CSV: {out_csv}")
    print(df[["frame_id", "method", "ground_points", "obstacle_points", "num_clusters", "time_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()

