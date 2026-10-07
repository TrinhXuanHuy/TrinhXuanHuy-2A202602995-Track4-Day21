"""Pipeline phát hiện vật cản cho robot/xe tự hành từ Point Cloud (Topic D).

Các bước xử lý:
  1. Voxel Downsampling: giảm mật độ điểm, chuẩn hóa độ phân giải không gian.
  2. Ground Removal (RANSAC Plane Segmentation): lọc mặt phẳng đường.
  3. Obstacle Clustering (Euclidean/DBSCAN): gom các điểm còn lại thành từng vật cản.
  4. Bounding Box & Feature Extraction: tính tâm, kích thước, khoảng cách tới ego.
  5. Projection & Visualization: vẽ BEV và chiếu bounding box lên ảnh camera.
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Them goc repo vao sys.path de import starter
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from starter.datasets import dataset_type, load_frame
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import project_velo_to_image, velo_to_cam



@dataclass
class ObstacleCluster:
    cluster_id: int
    points: np.ndarray       # (M, 3)
    bbox_min: np.ndarray     # (3,) [x_min, y_min, z_min]
    bbox_max: np.ndarray     # (3,) [x_max, y_max, z_max]
    center: np.ndarray       # (3,)
    size: np.ndarray         # (3,) [dx, dy, dz]
    min_dist: float          # khoảng cách gần nhất tới ego (m)
    num_points: int


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
    """Giảm số điểm bằng voxel grid hashing trong NumPy.
    
    Mỗi voxel giữ lại điểm đại diện (centroid hoặc điểm đầu tiên).
    """
    if len(points) == 0:
        return points
    xyz = points[:, :3]
    voxel_coords = np.floor(xyz / voxel_size).astype(np.int32)
    # Lấy index duy nhất
    _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
    return points[unique_indices]


def ransac_ground_segmentation(points: np.ndarray,
                               distance_threshold: float = 0.15,
                               max_iterations: int = 100,
                               normal_z_threshold: float = 0.8,
                               z_ground_max: float = -0.5,
                               random_seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phân tách mặt đất bằng RANSAC plane fitting.
    
    Quy ước Velodyne KITTI: x forward, y left, z up (đường thường có z ~ -1.73m).
    
    Trả về:
      ground_mask: (N,) bool
      obstacle_mask: (N,) bool
      plane_model: (4,) [a, b, c, d] sao cho a*x + b*y + c*z + d = 0
    """
    xyz = points[:, :3]
    N = len(xyz)
    if N < 3:
        return np.zeros(N, dtype=bool), np.ones(N, dtype=bool), np.zeros(4)

    rng = np.random.default_rng(random_seed)
    
    # Chỉ xét ứng viên mặt đất ở độ cao hợp lý (dưới gầm xe / dưới nắp capo)
    candidate_indices = np.where(xyz[:, 2] < z_ground_max)[0]
    if len(candidate_indices) < 3:
        candidate_indices = np.arange(N)

    best_inliers_count = -1
    best_plane = np.array([0.0, 0.0, 1.0, 1.73], dtype=np.float32)

    for _ in range(max_iterations):
        sample_idx = rng.choice(candidate_indices, size=3, replace=False)
        p1, p2, p3 = xyz[sample_idx]
        
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal = normal / norm_len
        
        # Mặt đất KITTI có vector pháp tuyến hướng lên z (normal_z ~ 1 hoặc -1)
        if abs(normal[2]) < normal_z_threshold:
            continue
        # Đảm bảo normal hướng lên trên
        if normal[2] < 0:
            normal = -normal

        d = -np.dot(normal, p1)
        
        # Khoảng cách từ tất cả các điểm tới mặt phẳng
        signed_dist = np.dot(xyz, normal) + d
        abs_dist = np.abs(signed_dist)
        
        inliers_count = np.count_nonzero(abs_dist < distance_threshold)
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

    # Tính lại inliers theo plane tốt nhất
    normal = best_plane[:3]
    d = best_plane[3]
    signed_dist = np.dot(xyz, normal) + d
    
    ground_mask = np.abs(signed_dist) < distance_threshold
    # Obstacle là các điểm nằm PHÍA TRÊN mặt đất (loại trừ phản xạ ngầm dưới đất)
    obstacle_mask = (signed_dist >= distance_threshold) & (xyz[:, 2] < 2.5) & (xyz[:, 2] > -2.5)

    return ground_mask, obstacle_mask, best_plane


def euclidean_clustering(points: np.ndarray,
                         eps: float = 0.5,
                         min_points: int = 15,
                         max_points: int = 5000) -> list[ObstacleCluster]:
    """Gom cụm các điểm vật cản bằng thuật toán Euclidean / DBSCAN dựa trên KD-Tree."""
    if len(points) == 0:
        return []

    xyz = points[:, :3]
    tree = cKDTree(xyz)
    N = len(xyz)
    visited = np.zeros(N, dtype=bool)
    clusters: list[ObstacleCluster] = []
    cluster_id = 0

    for i in range(N):
        if visited[i]:
            continue
        
        # Tìm láng giềng trong bán kính eps
        neighbors = tree.query_ball_point(xyz[i], r=eps)
        if len(neighbors) < min_points:
            visited[i] = True
            continue
        
        # Lan truyền cụm bằng BFS
        queue = collections.deque(neighbors)
        curr_cluster_indices = [i]
        visited[i] = True
        for nb in neighbors:
            visited[nb] = True

        while queue:
            curr_idx = queue.popleft()
            curr_cluster_indices.append(curr_idx)
            
            sub_neighbors = tree.query_ball_point(xyz[curr_idx], r=eps)
            if len(sub_neighbors) >= min_points:
                for sub_idx in sub_neighbors:
                    if not visited[sub_idx]:
                        visited[sub_idx] = True
                        queue.append(sub_idx)

        if min_points <= len(curr_cluster_indices) <= max_points:
            c_pts = xyz[curr_cluster_indices]
            b_min = c_pts.min(axis=0)
            b_max = c_pts.max(axis=0)
            center = (b_min + b_max) / 2.0
            size = b_max - b_min
            min_dist = float(np.min(np.linalg.norm(c_pts[:, :2], axis=1)))
            
            clusters.append(ObstacleCluster(
                cluster_id=cluster_id,
                points=c_pts,
                bbox_min=b_min,
                bbox_max=b_max,
                center=center,
                size=size,
                min_dist=min_dist,
                num_points=len(c_pts),
            ))
            cluster_id += 1

    # Sắp xếp các cụm theo khoảng cách gần xe nhất trước
    clusters.sort(key=lambda c: c.min_dist)
    return clusters


def cluster_to_camera_box2d(cluster: ObstacleCluster, calib: KittiCalib,
                            image_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    """Chiếu 8 góc của bounding box 3D từ Velodyne frame lên ảnh 2D camera."""
    bmin, bmax = cluster.bbox_min, cluster.bbox_max
    # 8 góc của 3D AABB trong velodyne frame
    corners_velo = np.array([
        [bmin[0], bmin[1], bmin[2]],
        [bmin[0], bmin[1], bmax[2]],
        [bmin[0], bmax[1], bmin[2]],
        [bmin[0], bmax[1], bmax[2]],
        [bmax[0], bmin[1], bmin[2]],
        [bmax[0], bmin[1], bmax[2]],
        [bmax[0], bmax[1], bmin[2]],
        [bmax[0], bmax[1], bmax[2]],
    ], dtype=np.float32)

    corners_cam = velo_to_cam(corners_velo, calib)
    # Nếu tất cả các góc nằm sau camera -> bỏ qua
    if (corners_cam[:, 2] <= 0.1).all():
        return None

    # Chiếu lên ảnh
    pts_homo = np.hstack([corners_cam, np.ones((8, 1), dtype=np.float32)])
    pts_2d = pts_homo @ calib.P2.T
    depth = pts_2d[:, 2]
    valid = depth > 0.1
    if not np.any(valid):
        return None

    u = pts_2d[valid, 0] / depth[valid]
    v = pts_2d[valid, 1] / depth[valid]
    
    H, W = image_shape[:2]
    u1, u2 = np.clip(np.min(u), 0, W - 1), np.clip(np.max(u), 0, W - 1)
    v1, v2 = np.clip(np.min(v), 0, H - 1), np.clip(np.max(v), 0, H - 1)
    
    if u2 - u1 < 4 or v2 - v1 < 4:
        return None
    return int(u1), int(v1), int(u2), int(v2)


def run_pipeline(points: np.ndarray,
                 voxel_size: float = 0.1,
                 distance_threshold: float = 0.15,
                 eps: float = 0.6,
                 min_points: int = 20) -> dict:
    """Chạy toàn bộ pipeline và đo thời gian chi tiết từng bước."""
    t0 = time.perf_counter()
    down_pts = voxel_downsample(points, voxel_size=voxel_size)
    t1 = time.perf_counter()
    
    ground_mask, obs_mask, plane = ransac_ground_segmentation(
        down_pts, distance_threshold=distance_threshold
    )
    t2 = time.perf_counter()
    
    obstacle_pts = down_pts[obs_mask]
    clusters = euclidean_clustering(obstacle_pts, eps=eps, min_points=min_points)
    t3 = time.perf_counter()

    return {
        "raw_points": len(points),
        "down_points": len(down_pts),
        "ground_points": int(ground_mask.sum()),
        "obstacle_points": len(obstacle_pts),
        "num_clusters": len(clusters),
        "clusters": clusters,
        "plane": plane,
        "down_pts": down_pts,
        "ground_mask": ground_mask,
        "obs_mask": obs_mask,
        "time_downsample_ms": (t1 - t0) * 1000,
        "time_ransac_ms": (t2 - t1) * 1000,
        "time_cluster_ms": (t3 - t2) * 1000,
        "total_time_ms": (t3 - t0) * 1000,
        "nearest_obstacle_dist": clusters[0].min_dist if clusters else float("inf"),
    }


def visualize_results(fr: dict, res: dict, out_path: Path, frame_id: str) -> None:
    """Vẽ dashboard trực quan gồm:
      1. Bird's Eye View (BEV) với các cụm và bounding box.
      2. Ảnh camera overlay các cụm vật cản phát hiện được.
      3. Phân bố khoảng cách & kích thước vật cản.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 1. BEV (Top-down view)
    ax_bev = fig.add_subplot(2, 2, 1)
    down_pts = res["down_pts"]
    g_mask = res["ground_mask"]
    
    # Ve mat dat mau xam nhat
    ax_bev.scatter(down_pts[g_mask, 1], down_pts[g_mask, 0], s=0.3, c="silver", alpha=0.3, label="Ground")
    
    # Ve cac cum vat can
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(res["clusters"][:25]):
        color = cmap(i % 20)
        ax_bev.scatter(c.points[:, 1], c.points[:, 0], s=2.0, color=color)
        # Bounding box BEV
        bmin, bmax = c.bbox_min, c.bbox_max
        rect = plt.Rectangle((bmin[1], bmin[0]), bmax[1] - bmin[1], bmax[0] - bmin[0],
                             fill=False, edgecolor=color, linewidth=1.5)
        ax_bev.add_patch(rect)
        if c.min_dist < 20:
            ax_bev.text(c.center[1], c.center[0], f"#{c.cluster_id} ({c.min_dist:.1f}m)",
                        fontsize=7, color="black", weight="bold")

    ax_bev.set_xlim(-25, 25)
    ax_bev.set_ylim(0, 50)
    ax_bev.set_xlabel("Y - Left/Right (m)")
    ax_bev.set_ylabel("X - Forward (m)")
    ax_bev.set_title(f"BEV Obstacle Detection: {len(res['clusters'])} clusters (Frame {frame_id})")
    ax_bev.grid(True, linestyle="--", alpha=0.5)

    # 2. Camera View Overlay
    ax_cam = fig.add_subplot(2, 2, 2)
    img_bgr = fr["image"].copy()
    calib = fr["calib"]
    
    for i, c in enumerate(res["clusters"]):
        color_bgr = tuple(int(val * 255) for val in cmap(i % 20)[:3][::-1])
        b2d = cluster_to_camera_box2d(c, calib, img_bgr.shape)
        if b2d:
            u1, v1, u2, v2 = b2d
            cv2.rectangle(img_bgr, (u1, v1), (u2, v2), color_bgr, 2)
            cv2.putText(img_bgr, f"{c.min_dist:.1f}m", (u1, max(15, v1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax_cam.imshow(img_rgb)
    ax_cam.set_title("Camera Projection: Obstacle Bounding Boxes")
    ax_cam.axis("off")

    # 3. Phan bo khoang cach vat can
    ax_dist = fig.add_subplot(2, 2, 3)
    if res["clusters"]:
        dists = [c.min_dist for c in res["clusters"]]
        ax_dist.hist(dists, bins=np.arange(0, 55, 5), color="skyblue", edgecolor="black")
        ax_dist.axvline(res["nearest_obstacle_dist"], color="red", linestyle="--",
                        label=f"Nearest: {res['nearest_obstacle_dist']:.2f} m")
        ax_dist.set_xlabel("Distance (m)")
        ax_dist.set_ylabel("Cluster Count")
        ax_dist.set_title("Obstacle Distance Distribution")
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)

    # 4. Thong ke Pipeline
    ax_info = fig.add_subplot(2, 2, 4)
    ax_info.axis("off")
    info_text = (
        f"--- PIPELINE STATS (Frame {frame_id}) ---\n"
        f"• Raw points: {res['raw_points']:,}\n"
        f"• Voxel downsampled: {res['down_points']:,} (-{100*(1 - res['down_points']/res['raw_points']):.1f}%)\n"
        f"• Ground points (RANSAC): {res['ground_points']:,} ({100*res['ground_points']/res['down_points']:.1f}%)\n"
        f"• Obstacle points: {res['obstacle_points']:,}\n"
        f"• Detected obstacles: {res['num_clusters']}\n"
        f"• Nearest obstacle: {res['nearest_obstacle_dist']:.2f} m\n\n"
        f"--- PROCESSING LATENCY ---\n"
        f"• Voxel downsampling: {res['time_downsample_ms']:.2f} ms\n"
        f"• RANSAC plane: {res['time_ransac_ms']:.2f} ms\n"
        f"• DBSCAN clustering: {res['time_cluster_ms']:.2f} ms\n"
        f"• TOTAL LATENCY: {res['total_time_ms']:.2f} ms ({1000/res['total_time_ms']:.1f} FPS)\n"
    )
    ax_info.text(0.05, 0.5, info_text, fontsize=11, family="monospace", va="center")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot/Drone Obstacle Detection Pipeline (Topic D)")
    parser.add_argument("--data-root", default="data/kitti_mini", help="Dataset root (KITTI)")
    parser.add_argument("--frame", default="000011", help="Frame ID (e.g. 000011)")
    parser.add_argument("--voxel-size", type=float, default=0.1, help="Voxel size (m)")
    parser.add_argument("--distance-threshold", type=float, default=0.15, help="RANSAC distance threshold (m)")
    parser.add_argument("--eps", type=float, default=0.6, help="DBSCAN clustering eps (m)")
    parser.add_argument("--min-points", type=int, default=15, help="DBSCAN min points")
    parser.add_argument("--out-dir", default="results/figures", help="Output directory")
    args = parser.parse_args()

    fr = load_frame(args.data_root, args.frame)
    res = run_pipeline(
        fr["points"],
        voxel_size=args.voxel_size,
        distance_threshold=args.distance_threshold,
        eps=args.eps,
        min_points=args.min_points,
    )

    out_file = Path(args.out_dir) / f"demo_obstacle_{args.frame}.png"
    visualize_results(fr, res, out_file, args.frame)
    print(f"[OK] Done frame {args.frame}: found {res['num_clusters']} obstacles "
          f"(nearest: {res['nearest_obstacle_dist']:.2f} m, latency: {res['total_time_ms']:.2f} ms)")
    print(f"     Saved figure to: {out_file}")



if __name__ == "__main__":
    main()
