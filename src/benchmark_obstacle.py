"""Thí nghiệm benchmark quét tham số và đo latency p50/p95 cho Topic D.

Mục tiêu:
  1. Sweep distance_threshold của RANSAC (0.05m -> 0.35m): chứng minh ngưỡng cao làm mất vật cản thấp.
  2. Sweep voxel_size (0.03m -> 0.25m): đo latency p50/p95 và số cluster.
  3. Xuất file CSV vào results/ và vẽ biểu đồ vào results/figures/.
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
from src.obstacle_pipeline import run_pipeline


def sweep_distance_threshold(frame_data: dict,
                             thresholds: list[float],
                             fixed_voxel: float = 0.1,
                             fixed_eps: float = 0.6,
                             fixed_min_points: int = 15) -> pd.DataFrame:
    """Quét các mức distance_threshold của RANSAC."""
    records = []
    points = frame_data["points"]

    for d_th in thresholds:
        res = run_pipeline(
            points,
            voxel_size=fixed_voxel,
            distance_threshold=d_th,
            eps=fixed_eps,
            min_points=fixed_min_points,
        )
        
        # Đếm số vật cản thấp (< 0.45 m chiều cao dz)
        low_obstacles = sum(1 for c in res["clusters"] if c.size[2] < 0.45)
        # Đếm số vật cản ở cự ly gần (< 15 m)
        near_obstacles = sum(1 for c in res["clusters"] if c.min_dist < 15.0)

        records.append({
            "distance_threshold_m": d_th,
            "voxel_size_m": fixed_voxel,
            "ground_points": res["ground_points"],
            "ground_ratio_pct": 100.0 * res["ground_points"] / res["down_points"],
            "obstacle_points": res["obstacle_points"],
            "total_clusters": res["num_clusters"],
            "low_obstacles_h_under_0_45m": low_obstacles,
            "near_obstacles_dist_under_15m": near_obstacles,
            "nearest_obstacle_dist_m": round(res["nearest_obstacle_dist"], 2),
            "total_latency_ms": round(res["total_time_ms"], 2),
        })

    return pd.DataFrame(records)


def sweep_voxel_size_with_latency(frame_data: dict,
                                  voxel_sizes: list[float],
                                  fixed_d_th: float = 0.15,
                                  fixed_eps: float = 0.6,
                                  fixed_min_points: int = 15,
                                  repeats: int = 20) -> pd.DataFrame:
    """Quét các mức voxel_size và đo latency p50/p95 (theo Bonus B3)."""
    records = []
    points = frame_data["points"]

    for vs in voxel_sizes:
        latencies = []
        # Chạy warm-up bỏ qua lần đầu
        res = run_pipeline(points, voxel_size=vs, distance_threshold=fixed_d_th,
                           eps=fixed_eps, min_points=fixed_min_points)
        
        # Chạy đo latency lặp lại
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = run_pipeline(points, voxel_size=vs, distance_threshold=fixed_d_th,
                             eps=fixed_eps, min_points=fixed_min_points)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        lat_p50 = float(np.percentile(latencies, 50))
        lat_p95 = float(np.percentile(latencies, 95))
        fps = 1000.0 / lat_p50 if lat_p50 > 0 else 0.0

        records.append({
            "voxel_size_m": vs,
            "raw_points": res["raw_points"],
            "downsampled_points": res["down_points"],
            "point_reduction_pct": round(100.0 * (1 - res["down_points"] / res["raw_points"]), 1),
            "total_clusters": res["num_clusters"],
            "nearest_obstacle_dist_m": round(res["nearest_obstacle_dist"], 2),
            "latency_p50_ms": round(lat_p50, 2),
            "latency_p95_ms": round(lat_p95, 2),
            "fps_p50": round(fps, 1),
        })

    return pd.DataFrame(records)


def plot_benchmark_figures(df_dist: pd.DataFrame, df_voxel: pd.DataFrame, out_path: Path) -> None:
    """Vẽ 4 biểu đồ phân tích benchmark."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Ảnh hưởng của distance_threshold lên số cluster và vật cản thấp
    ax1 = axes[0, 0]
    ax1.plot(df_dist["distance_threshold_m"], df_dist["total_clusters"], marker="o",
             color="tab:blue", linewidth=2, label="Tổng số cụm vật cản")
    ax1.plot(df_dist["distance_threshold_m"], df_dist["low_obstacles_h_under_0_45m"],
             marker="s", color="tab:red", linewidth=2, linestyle="--", label="Vật cản thấp (< 0.45m)")
    ax1.set_xlabel("RANSAC Distance Threshold (m)")
    ax1.set_ylabel("Số lượng vật cản")
    ax1.set_title("Ảnh hưởng của ngưỡng RANSAC lên số vật cản")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Tỉ lệ inlier mặt đất theo distance_threshold
    ax2 = axes[0, 1]
    ax2.plot(df_dist["distance_threshold_m"], df_dist["ground_ratio_pct"], marker="^",
             color="tab:green", linewidth=2)
    ax2.set_xlabel("RANSAC Distance Threshold (m)")
    ax2.set_ylabel("Tỉ lệ điểm bị gán là mặt đất (%)")
    ax2.set_title("Tỉ lệ điểm mặt đất khi tăng ngưỡng RANSAC")
    ax2.grid(True, alpha=0.3)

    # 3. Latency p50 và p95 theo Voxel Size
    ax3 = axes[1, 0]
    ax3.plot(df_voxel["voxel_size_m"], df_voxel["latency_p50_ms"], marker="o",
             color="tab:purple", linewidth=2, label="Latency p50 (median)")
    ax3.plot(df_voxel["voxel_size_m"], df_voxel["latency_p95_ms"], marker="^",
             color="tab:orange", linewidth=2, linestyle=":", label="Latency p95")
    ax3.set_xlabel("Voxel Size (m)")
    ax3.set_ylabel("Độ trễ xử lý (ms)")
    ax3.set_title("Độ trễ tính toán (p50 / p95) theo Voxel Size")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Đánh đổi FPS và số điểm còn lại
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    l1 = ax4.plot(df_voxel["voxel_size_m"], df_voxel["fps_p50"], marker="s",
                  color="tab:blue", linewidth=2, label="FPS")
    l2 = ax4_twin.plot(df_voxel["voxel_size_m"], df_voxel["downsampled_points"], marker="d",
                       color="tab:red", linewidth=2, linestyle="--", label="Số điểm sau voxel")
    ax4.set_xlabel("Voxel Size (m)")
    ax4.set_ylabel("Tốc độ xử lý (FPS)", color="tab:blue")
    ax4_twin.set_ylabel("Số điểm (points)", color="tab:red")
    ax4.set_title("Đánh đổi giữa Tốc độ (FPS) và Mật độ điểm")
    
    # Gộp legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main() -> None:
    data_root = "data/kitti_mini"
    frame_id = "000011"
    print(f"[*] Dang doc du lieu frame {frame_id} tu {data_root}...")
    fr = load_frame(data_root, frame_id)

    # 1. Sweep Distance Threshold (7 mức)
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    print(f"[*] Chay thi nghiem 1: Sweep RANSAC distance_threshold ({len(thresholds)} muc)...")
    df_dist = sweep_distance_threshold(fr, thresholds)
    csv_dist = Path("results/obstacle_distance_sweep.csv")
    csv_dist.parent.mkdir(parents=True, exist_ok=True)
    df_dist.to_csv(str(csv_dist), index=False)
    print(f"    [OK] Da luu: {csv_dist}")

    # 2. Sweep Voxel Size (7 mức) + Latency p50/p95
    voxel_sizes = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
    print(f"[*] Chay thi nghiem 2: Sweep Voxel Size va do latency p50/p95 ({len(voxel_sizes)} muc, 20 repeats)...")
    df_voxel = sweep_voxel_size_with_latency(fr, voxel_sizes, repeats=20)
    csv_voxel = Path("results/obstacle_voxel_sweep.csv")
    df_voxel.to_csv(str(csv_voxel), index=False)
    print(f"    [OK] Da luu: {csv_voxel}")

    # 3. Vẽ biểu đồ benchmark
    plot_file = Path("results/figures/obstacle_benchmark_plots.png")
    plot_benchmark_figures(df_dist, df_voxel, plot_file)
    print(f"    [OK] Da luu bieu do: {plot_file}")

    print("\n--- KET QUA THI NGHIEM 1 (DISTANCE THRESHOLD) ---")
    print(df_dist[["distance_threshold_m", "ground_ratio_pct", "total_clusters", "low_obstacles_h_under_0_45m", "nearest_obstacle_dist_m"]].to_string(index=False))

    print("\n--- KET QUA THI NGHIEM 2 (VOXEL SIZE & LATENCY) ---")
    print(df_voxel[["voxel_size_m", "downsampled_points", "total_clusters", "latency_p50_ms", "latency_p95_ms", "fps_p50"]].to_string(index=False))


if __name__ == "__main__":
    main()

