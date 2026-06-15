import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm

# ──────────────────────────────────────────────
# Joint mapping utilities
# ──────────────────────────────────────────────

def nba16_to_coco17_3d(nba_3d):
    """NBA-16 (T,16,3) → COCO-17 (T,17,3)"""
    T = nba_3d.shape[0]
    coco = np.zeros((T, 17, 3), dtype=np.float32)
    coco[:, 0] = nba_3d[:, 8]
    coco[:, 1] = nba_3d[:, 8]
    coco[:, 2] = nba_3d[:, 8]
    coco[:, 3] = nba_3d[:, 8]
    coco[:, 4] = nba_3d[:, 8]
    coco[:, 5] = nba_3d[:, 10]
    coco[:, 6] = nba_3d[:, 13]
    coco[:, 7] = nba_3d[:, 11]
    coco[:, 8] = nba_3d[:, 14]
    coco[:, 9] = nba_3d[:, 12]
    coco[:, 10] = nba_3d[:, 15]
    coco[:, 11] = nba_3d[:, 4]
    coco[:, 12] = nba_3d[:, 1]
    coco[:, 13] = nba_3d[:, 5]
    coco[:, 14] = nba_3d[:, 2]
    coco[:, 15] = nba_3d[:, 6]
    coco[:, 16] = nba_3d[:, 3]
    return coco


def kinect25_to_coco17_3d(kp3d):
    """
    NTU RGB+D Kinect-25 (T,25,3) → COCO-17 (T,17,3)

    Kinect-25 joints:
      0  pelvis / spine_base   1  spine_mid        2  neck/chest
      3  head                  4  left_shoulder     5  left_elbow
      6  left_wrist            7  left_hand         8  right_shoulder
      9  right_elbow          10  right_wrist       11  right_hand
     12  left_hip             13  left_knee         14  left_ankle
     15  left_foot            16  right_hip         17  right_knee
     18  right_ankle          19  right_foot        20  spine_shoulder
     21  left_hand_tip        22  left_thumb        23  right_hand_tip
     24  right_thumb

    COCO-17:
      0 nose  1 L-eye  2 R-eye  3 L-ear  4 R-ear
      5 L-sho 6 R-sho  7 L-elb  8 R-elb  9 L-wri 10 R-wri
     11 L-hip 12 R-hip 13 L-kne 14 R-kne 15 L-ank 16 R-ank
    """
    T = kp3d.shape[0]
    coco = np.zeros((T, 17, 3), dtype=np.float32)
    # Head/face joints: approximate from neck(2) and head(3)
    coco[:, 0] = (kp3d[:, 2] + kp3d[:, 3]) * 0.5  # nose ≈ mid neck-head
    coco[:, 1] = kp3d[:, 3]   # left eye  → head
    coco[:, 2] = kp3d[:, 3]   # right eye → head
    coco[:, 3] = kp3d[:, 3]   # left ear  → head
    coco[:, 4] = kp3d[:, 3]   # right ear → head
    # Arms
    coco[:, 5]  = kp3d[:, 4]  # left shoulder
    coco[:, 6]  = kp3d[:, 8]  # right shoulder
    coco[:, 7]  = kp3d[:, 5]  # left elbow
    coco[:, 8]  = kp3d[:, 9]  # right elbow
    coco[:, 9]  = kp3d[:, 6]  # left wrist
    coco[:, 10] = kp3d[:, 10] # right wrist
    # Legs
    coco[:, 11] = kp3d[:, 12] # left hip
    coco[:, 12] = kp3d[:, 16] # right hip
    coco[:, 13] = kp3d[:, 13] # left knee
    coco[:, 14] = kp3d[:, 17] # right knee
    coco[:, 15] = kp3d[:, 14] # left ankle
    coco[:, 16] = kp3d[:, 18] # right ankle
    return coco


# ──────────────────────────────────────────────
# Rotation & projection
# ──────────────────────────────────────────────

def rotate_azimuth(pose_3d, theta):
    """Rotate around Y-axis by theta radians.  pose_3d: (..., 3)"""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([
        [ cos_t, 0, sin_t],
        [     0, 1,     0],
        [-sin_t, 0, cos_t],
    ])
    return pose_3d @ R.T


def _align_vectors(v_from, v_to):
    """Rodrigues rotation matrix: rotates unit vector v_from onto unit vector v_to."""
    v_from = v_from / (np.linalg.norm(v_from) + 1e-8)
    v_to   = v_to   / (np.linalg.norm(v_to)   + 1e-8)
    c = float(np.dot(v_from, v_to))
    if c > 1 - 1e-6:
        return np.eye(3, dtype=np.float64)
    if c < -1 + 1e-6:
        perp = np.array([1., 0., 0.]) if abs(v_from[0]) < 0.9 else np.array([0., 1., 0.])
        perp = perp - np.dot(perp, v_from) * v_from
        perp /= np.linalg.norm(perp)
        return (2 * np.outer(perp, perp) - np.eye(3)).astype(np.float64)
    v  = np.cross(v_from, v_to)
    s  = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s)


def rotate_azimuth_centered(coco3d, theta):
    """
    Body-aligned azimuth rotation for NTU Kinect camera-space coordinates.

    Problem: The Kinect camera is mounted ~1.5 m above the floor looking slightly
    downward, so in camera space the feet are ~0.4 m deeper (higher Z) than the
    head.  Rotating around the camera Y-axis converts this depth gradient into a
    horizontal lean at large angles (≈10° at ±90°).

    Fix: estimate the spine direction (hip→shoulder midpoint), build a small
    alignment rotation R_align that maps this direction onto the world [0,1,0]
    axis, apply R_align before the azimuth rotation, then undo it afterwards.
    This way the rotation axis is the body's own vertical, not the camera Y.

    coco3d : (..., 17, 3)  —  COCO-17 joints in Kinect camera space
    theta  : float  —  azimuth angle in radians
    """
    # ── 1. hip midpoint as translation centre ──────────────────────────────
    root = (coco3d[..., 11, :] + coco3d[..., 12, :]) / 2.0   # (..., 3)
    centered = coco3d - root[..., np.newaxis, :]               # (..., 17, 3)

    # ── 2. estimate spine direction (mean over sequence if multi-frame) ────
    sho_mid = (centered[..., 5, :] + centered[..., 6, :]) / 2.0   # (..., 3)
    hip_mid = (centered[..., 11, :] + centered[..., 12, :]) / 2.0  # ≈ 0

    if centered.ndim == 3:                 # (T, 17, 3)
        spine = (sho_mid - hip_mid).mean(axis=0)   # (3,)
    else:                                  # (17, 3)
        spine = sho_mid - hip_mid          # (3,)

    # ── 3. align spine → world Y, then rotate, then un-align ──────────────
    R_align = _align_vectors(spine, np.array([0., 1., 0.]))

    aligned  = centered.astype(np.float64) @ R_align.T   # (..., 17, 3)
    rotated  = rotate_azimuth(aligned, theta)             # (..., 17, 3)
    unaligned = rotated @ R_align                         # (..., 17, 3)  (R_align^-1 = R_align^T)

    return (unaligned + root[..., np.newaxis, :]).astype(np.float32)


def project_to_ntu_style(pose_3d, img_w=1920, img_h=1080):
    """Orthographic projection used for NBA / c-MAS output data.
    pose_3d: (..., J, 3) — last dim is (X, Y, Z)
    """
    scale = img_w * 0.06
    cx, cy = img_w / 2, img_h / 2
    x_pixel = -pose_3d[..., 0] * scale + cx
    y_pixel = -pose_3d[..., 1] * scale + cy
    return np.stack([x_pixel, y_pixel], axis=-1)


def project_kinect_perspective(pose_3d, img_w=1920, img_h=1080):
    """
    Perspective projection for NTU RGB+D Kinect-25 3D coords (metres, camera space).
    Kinect v2 RGB approximate intrinsics at 1920×1080.
    Coordinate system: X right, Y up, Z toward viewer (depth).
    Image space: Y increases downward.
    pose_3d: (..., J, 3) — last dim is (X, Y, Z)
    """
    fx = 1081.37
    fy = 1081.37
    cx = img_w / 2.0
    cy = img_h / 2.0

    Z = pose_3d[..., 2]
    Z = np.where(np.abs(Z) < 1e-6, 1e-6, Z)   # avoid div-by-zero

    x_pixel = pose_3d[..., 0] / Z * fx + cx
    y_pixel = -pose_3d[..., 1] / Z * fy + cy   # negate: Kinect Y-up → image Y-down
    return np.stack([x_pixel, y_pixel], axis=-1)


# ──────────────────────────────────────────────
# NBA multi-view processing  (original flow)
# ──────────────────────────────────────────────

def process_nba_multiview(input_dir, output_base, angle_sets):
    """Original per-sample .npy multi-view processing for c-MAS NBA output."""
    for set_name in angle_sets:
        os.makedirs(os.path.join(output_base, f"ntu_test_60class_multiview_{set_name}"), exist_ok=True)

    files = sorted(os.listdir(input_dir))
    print(f"총 {len(files)}개 처리 시작")

    for i, fname in enumerate(files):
        fpath = os.path.join(input_dir, fname)
        result = np.load(fpath, allow_pickle=True).item()
        nba_3d = result['motions'][0]           # (T, 16, 3)
        coco_3d = nba16_to_coco17_3d(nba_3d)   # (T, 17, 3)
        label = int(fname.split('_label')[1].replace('.npy', ''))

        for set_name, angles in angle_sets.items():
            views = []
            for theta in angles:
                rotated  = rotate_azimuth(coco_3d, theta)
                projected = project_to_ntu_style(rotated[None])[0]  # (T,17,2)
                conf = np.ones((rotated.shape[0], 17, 1), dtype=np.float32)
                views.append(np.concatenate([projected, conf], axis=-1))

            multiview = np.stack(views, axis=0)
            save_path = os.path.join(
                output_base,
                f"ntu_test_60class_multiview_{set_name}",
                fname,
            )
            np.save(save_path, {'multiview': multiview, 'label': label})

        if (i + 1) % 500 == 0:
            print(f"진행: {i+1}/{len(files)}")

    print("완료!")
    for set_name in angle_sets:
        cnt = len(os.listdir(os.path.join(output_base, f"ntu_test_60class_multiview_{set_name}")))
        print(f"  {set_name}: {cnt}개")


# ──────────────────────────────────────────────
# GT baseline:  Kinect-25 → COCO-17 → 2D  →  MotionBERT pkl
# ──────────────────────────────────────────────

def build_gt_motionbert_pkl(gt_pkl_path, output_pkl_path, theta=0.0,
                             img_w=1920, img_h=1080):
    """
    Read NTU60 GT 3D annotations (Kinect-25), convert to COCO-17 2D projected
    coordinates, and save a MotionBERT-compatible pkl.

    The output pkl has the same schema as ntu60_hrnet.pkl:
      {
        'split': { 'xsub_train': [...], 'xsub_val': [...], ... },
        'annotations': [
          {
            'frame_dir':      str,
            'label':          int,
            'total_frames':   int,
            'img_shape':      (H, W),
            'keypoint':       np.ndarray  (M, T, 17, 2)  float32 pixel coords
            'keypoint_score': np.ndarray  (M, T, 17)     float32 confidence
          }, ...
        ]
      }
    """
    print(f"Loading GT pkl: {gt_pkl_path}")
    with open(gt_pkl_path, 'rb') as f:
        gt_data = pickle.load(f)

    new_annotations = []
    for sample in tqdm(gt_data['annotations'], desc="Converting"):
        kp3d = sample['keypoint'].astype(np.float32)  # (M, T, 25, 3)
        M, T, _, _ = kp3d.shape

        kp2d_list = []
        for m in range(M):
            coco3d = kinect25_to_coco17_3d(kp3d[m])                    # (T, 17, 3)
            rotated = rotate_azimuth_centered(coco3d, theta)            # (T, 17, 3)
            proj2d  = project_kinect_perspective(                       # (T, 17, 2)
                rotated, img_w=img_w, img_h=img_h
            )
            kp2d_list.append(proj2d)

        keypoint       = np.stack(kp2d_list, axis=0).astype(np.float32)  # (M,T,17,2)
        keypoint_score = np.ones((M, T, 17), dtype=np.float32)

        new_annotations.append({
            'frame_dir':      sample['frame_dir'],
            'label':          sample['label'],
            'total_frames':   sample['total_frames'],
            'img_shape':      (img_h, img_w),
            'keypoint':       keypoint,
            'keypoint_score': keypoint_score,
        })

    out = {
        'split':       gt_data['split'],
        'annotations': new_annotations,
    }
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    print(f"Saving to: {output_pkl_path}")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(out, f)
    print("Done.")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projection utilities: NBA multi-view or Kinect GT baseline")
    subparsers = parser.add_subparsers(dest="mode")

    # --- nba mode (original) ---
    p_nba = subparsers.add_parser("nba", help="Process c-MAS NBA output into multi-view .npy files")
    p_nba.add_argument("--input_dir",   required=True, help="Directory of per-sample .npy files")
    p_nba.add_argument("--output_base", required=True, help="Root directory for output folders")

    # --- gt mode (new) ---
    p_gt = subparsers.add_parser("gt", help="Convert NTU GT Kinect-25 3D to MotionBERT-compatible pkl")
    p_gt.add_argument("--gt_pkl",    default="/workspace/clone_model/data/ntu60_3danno.pkl")
    p_gt.add_argument("--out_pkl",   default="/workspace/clone_model/MotionBERT/data/action/ntu60_gt_coco17.pkl")
    p_gt.add_argument("--theta",     type=float, default=0.0,
                      help="Azimuth rotation angle in radians (default 0 = front view)")
    p_gt.add_argument("--img_w",     type=int, default=1920)
    p_gt.add_argument("--img_h",     type=int, default=1080)

    args = parser.parse_args()

    if args.mode == "nba":
        angle_sets = {
            "setA": [0, np.pi/4, -np.pi/4],
            "setB": [0, np.pi/6, -np.pi/6],
            "setC": [0, np.pi/2, -np.pi/2],
            "setD": [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2],
        }
        process_nba_multiview(args.input_dir, args.output_base, angle_sets)

    elif args.mode == "gt":
        build_gt_motionbert_pkl(
            gt_pkl_path=args.gt_pkl,
            output_pkl_path=args.out_pkl,
            theta=args.theta,
            img_w=args.img_w,
            img_h=args.img_h,
        )

    else:
        parser.print_help()
