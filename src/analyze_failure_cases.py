"""Phân tích các trường hợp thất bại (Failure Cases) của pipeline Topic D.

Tạo 2 ảnh phân tích failure:
  1. fail_01_ransac_low_obstacle.png: Ngưỡng RANSAC quá lớn (0.40m) nuốt mất vật cản thấp (Lớp Geometry/Preprocess).
  2. fail_02_dbscan_clustering_drift.png: eps quá nhỏ làm vỡ vật thể ở xa, eps quá lớn làm dính các vật thể gần nhau (Lớp Preprocess).
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from starter.datasets import load_frame
from src.obstacle_pipeline import run_pipeline


def generate_ransac_failure_case(frame_data: dict, out_path: Path) -> None:
    """Tạo ảnh minh họa Failure Case 1: RANSAC nuốt chửng vật cản thấp khi ngưỡng quá cao."""
    pts = frame_data["points"]

    # Chạy ở 2 ngưỡng: Chuẩn (0.10m) vs Thất bại (0.40m)
    res_normal = run_pipeline(pts, voxel_size=0.08, distance_threshold=0.10, eps=0.6, min_points=15)
    res_failed = run_pipeline(pts, voxel_size=0.08, distance_threshold=0.40, eps=0.6, min_points=15)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Cột 1: Bình thường (0.10m)
    ax1 = axes[0]
    p_norm = res_normal["down_pts"]
    g_norm = res_normal["ground_mask"]
    obs_norm = res_normal["obs_mask"]
    ax1.scatter(p_norm[g_norm, 1], p_norm[g_norm, 0], s=0.3, c="gray", alpha=0.3, label="Mặt đất (inliers)")
    ax1.scatter(p_norm[obs_norm, 1], p_norm[obs_norm, 0], s=1.8, c="green", alpha=0.8, label="Vật cản (obstacles)")
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(0, 35)
    ax1.set_xlabel("Y - Left/Right (m)")
    ax1.set_ylabel("X - Forward (m)")
    ax1.set_title(f"HỢP LÝ: distance_threshold = 0.10 m\n(Giữ lại {res_normal['obstacle_points']:,} điểm vật cản, {res_normal['num_clusters']} cụm)", color="green", weight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Cột 2: Thất bại (0.40m)
    ax2 = axes[1]
    p_fail = res_failed["down_pts"]
    g_fail = res_failed["ground_mask"]
    obs_fail = res_failed["obs_mask"]
    ax2.scatter(p_fail[g_fail, 1], p_fail[g_fail, 0], s=0.3, c="gray", alpha=0.3, label="Mặt đất (inliers)")
    ax2.scatter(p_fail[obs_fail, 1], p_fail[obs_fail, 0], s=1.8, c="red", alpha=0.8, label="Vật cản còn sót")
    ax2.set_xlim(-15, 15)
    ax2.set_ylim(0, 35)
    ax2.set_xlabel("Y - Left/Right (m)")
    ax2.set_ylabel("X - Forward (m)")
    lost_pts = res_normal["obstacle_points"] - res_failed["obstacle_points"]
    ax2.set_title(f"FAILURE CASE: distance_threshold = 0.40 m\n(Mất {lost_pts:,} điểm vật cản, nuốt mất chân người/vật thấp)", color="red", weight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("FAILURE CASE 1: Lỗi Lớp GEOMETRY / PREPROCESS\n"
                 "RANSAC Plane Nuốt Vật Cản Thấp Sát Đất (Pallet, Chân người, Gờ đường) Khi Ngưỡng Khoảng Cách Quá Lớn",
                 fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Da tao: {out_path}")


def generate_clustering_failure_case(frame_data: dict, out_path: Path) -> None:
    """Tạo ảnh minh họa Failure Case 2: DBSCAN Over-segmentation và Under-segmentation."""
    pts = frame_data["points"]

    # eps quá nhỏ (0.25m) vs eps quá lớn (1.50m)
    res_under = run_pipeline(pts, voxel_size=0.08, distance_threshold=0.15, eps=0.25, min_points=10)
    res_over = run_pipeline(pts, voxel_size=0.08, distance_threshold=0.15, eps=1.50, min_points=10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    cmap = plt.get_cmap("tab20")

    # Cột 1: eps = 0.25m (Over-segmentation / Fragmentation)
    ax1 = axes[0]
    for i, c in enumerate(res_under["clusters"][:40]):
        color = cmap(i % 20)
        ax1.scatter(c.points[:, 1], c.points[:, 0], s=2.5, color=color)
        bmin, bmax = c.bbox_min, c.bbox_max
        rect = plt.Rectangle((bmin[1], bmin[0]), bmax[1] - bmin[1], bmax[0] - bmin[0],
                             fill=False, edgecolor=color, linewidth=1.2)
        ax1.add_patch(rect)
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(5, 35)
    ax1.set_xlabel("Y (m)")
    ax1.set_ylabel("X (m)")
    ax1.set_title(f"OVER-SEGMENTATION: eps = 0.25 m\n({res_under['num_clusters']} cụm vụn, xe/người bị xé nhỏ do khoảng cách beam lớn)", color="red", weight="bold")
    ax1.grid(True, alpha=0.3)

    # Cột 2: eps = 1.50m (Under-segmentation / Merging)
    ax2 = axes[1]
    for i, c in enumerate(res_over["clusters"][:20]):
        color = cmap(i % 20)
        ax2.scatter(c.points[:, 1], c.points[:, 0], s=2.5, color=color)
        bmin, bmax = c.bbox_min, c.bbox_max
        rect = plt.Rectangle((bmin[1], bmin[0]), bmax[1] - bmin[1], bmax[0] - bmin[0],
                             fill=False, edgecolor=color, linewidth=1.5)
        ax2.add_patch(rect)
    ax2.set_xlim(-15, 15)
    ax2.set_ylim(5, 35)
    ax2.set_xlabel("Y (m)")
    ax2.set_ylabel("X (m)")
    ax2.set_title(f"UNDER-SEGMENTATION: eps = 1.50 m\n({res_over['num_clusters']} cụm quá lớn, người đi bộ bị dính chùm vào xe)", color="red", weight="bold")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("FAILURE CASE 2: Lỗi Lớp PREPROCESS / METRIC\n"
                 "Đánh Đổi Của Bán Kính Láng Giềng EPS Trong Phân Cụm DBSCAN Khi Mật Độ Điểm LiDAR Thay Đổi Theo Cự Ly",
                 fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Da tao: {out_path}")


def main() -> None:
    data_root = "data/kitti_mini"
    frame_id = "000011"
    fr = load_frame(data_root, frame_id)

    fail_img1 = Path("results/figures/fail_01_ransac_low_obstacle.png")
    fail_img2 = Path("results/figures/fail_02_dbscan_clustering_drift.png")

    generate_ransac_failure_case(fr, fail_img1)
    generate_clustering_failure_case(fr, fail_img2)


if __name__ == "__main__":
    main()

