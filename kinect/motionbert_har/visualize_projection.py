"""
Visualize NTU60 GT skeletons:
  - Original Kinect-25 (3D)
  - COCO-17 projected at each rotation angle (0, ±30, ±45, ±90 deg)

Usage (gt_pkl/out_dir default to relative project paths):
    python visualize_projection.py \
        --sample_idx 240 \
        --frame 30
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Kinect-25 skeleton connections ────────────────────────────────────────
KINECT25_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),          # spine + head
    (20, 4), (4, 5), (5, 6), (6, 7),            # left arm
    (20, 8), (8, 9), (9, 10), (10, 11),         # right arm
    (0, 12), (12, 13), (13, 14), (14, 15),      # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),      # right leg
]

KINECT25_COLORS = {
    "spine":     "#FF6B6B",
    "left_arm":  "#4ECDC4",
    "right_arm": "#45B7D1",
    "left_leg":  "#96CEB4",
    "right_leg": "#FFEAA7",
}

def kinect25_bone_color(bone):
    s, e = bone
    if s in (0,1,2,20) or e in (0,1,2,20,3):
        return KINECT25_COLORS["spine"]
    if s in (4,5,6,7) or e in (4,5,6,7):
        return KINECT25_COLORS["left_arm"]
    if s in (8,9,10,11) or e in (8,9,10,11):
        return KINECT25_COLORS["right_arm"]
    if s in (12,13,14,15) or e in (12,13,14,15):
        return KINECT25_COLORS["left_leg"]
    return KINECT25_COLORS["right_leg"]

# ─── COCO-17 skeleton connections ──────────────────────────────────────────
COCO17_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6),                                      # shoulders
    (5, 7), (7, 9),                              # left arm
    (6, 8), (8, 10),                             # right arm
    (5, 11), (6, 12),                            # torso sides
    (11, 12),                                    # hips
    (11, 13), (13, 15),                          # left leg
    (12, 14), (14, 16),                          # right leg
]

COCO17_PART_COLOR = {
    "head":      "#FF6B6B",
    "left_arm":  "#4ECDC4",
    "right_arm": "#45B7D1",
    "torso":     "#DDA0DD",
    "left_leg":  "#96CEB4",
    "right_leg": "#FFEAA7",
}

def coco17_bone_color(bone):
    s, e = bone
    if s <= 4 or e <= 4:
        return COCO17_PART_COLOR["head"]
    if s == 5 and e == 6:
        return COCO17_PART_COLOR["torso"]
    if s in (5, 7) and e in (7, 9):
        return COCO17_PART_COLOR["left_arm"]
    if s in (6, 8) and e in (8, 10):
        return COCO17_PART_COLOR["right_arm"]
    if (s in (5, 6, 11, 12) and e in (11, 12)) or (s in (5,6) and e in (11,12)):
        return COCO17_PART_COLOR["torso"]
    if s in (11, 13) and e in (13, 15):
        return COCO17_PART_COLOR["left_leg"]
    return COCO17_PART_COLOR["right_leg"]

# ─── Mapping / projection functions (same as proj_nba2coco.py) ─────────────

def kinect25_to_coco17_3d(kp3d):
    T = kp3d.shape[0]
    coco = np.zeros((T, 17, 3), dtype=np.float32)
    coco[:, 0]  = (kp3d[:, 2] + kp3d[:, 3]) * 0.5
    coco[:, 1]  = kp3d[:, 3]
    coco[:, 2]  = kp3d[:, 3]
    coco[:, 3]  = kp3d[:, 3]
    coco[:, 4]  = kp3d[:, 3]
    coco[:, 5]  = kp3d[:, 4]
    coco[:, 6]  = kp3d[:, 8]
    coco[:, 7]  = kp3d[:, 5]
    coco[:, 8]  = kp3d[:, 9]
    coco[:, 9]  = kp3d[:, 6]
    coco[:, 10] = kp3d[:, 10]
    coco[:, 11] = kp3d[:, 12]
    coco[:, 12] = kp3d[:, 16]
    coco[:, 13] = kp3d[:, 13]
    coco[:, 14] = kp3d[:, 17]
    coco[:, 15] = kp3d[:, 14]
    coco[:, 16] = kp3d[:, 18]
    return coco

def rotate_azimuth(pose_3d, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return pose_3d @ R.T

def _align_vectors(v_from, v_to):
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
    """Body-aligned azimuth rotation.
    Aligns spine direction to world Y before rotating so the rotation axis is
    the body's own vertical, not the camera Y — eliminates the 10° forward lean
    caused by the Kinect being mounted above floor level."""
    root = (coco3d[..., 11, :] + coco3d[..., 12, :]) / 2.0
    centered = coco3d - root[..., np.newaxis, :]

    sho_mid = (centered[..., 5, :] + centered[..., 6, :]) / 2.0
    hip_mid = (centered[..., 11, :] + centered[..., 12, :]) / 2.0
    if centered.ndim == 3:
        spine = (sho_mid - hip_mid).mean(axis=0)
    else:
        spine = sho_mid - hip_mid

    R_align  = _align_vectors(spine, np.array([0., 1., 0.]))
    aligned  = centered.astype(np.float64) @ R_align.T
    rotated  = rotate_azimuth(aligned, theta)
    unaligned = rotated @ R_align
    return (unaligned + root[..., np.newaxis, :]).astype(np.float32)

def project_kinect_perspective(pose_3d, img_w=1920, img_h=1080):
    fx, fy = 1081.37, 1081.37
    cx, cy = img_w / 2.0, img_h / 2.0
    Z = pose_3d[..., 2]
    Z = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
    x_px = pose_3d[..., 0] / Z * fx + cx
    y_px = -pose_3d[..., 1] / Z * fy + cy
    return np.stack([x_px, y_px], axis=-1)

# ─── Drawing helpers ────────────────────────────────────────────────────────

def draw_kinect25_3d(ax, joints, title="Kinect-25\n(3D GT)"):
    """joints: (25, 3)  X-right, Y-up, Z-depth"""
    ax.set_facecolor("#1A1A2E")
    for bone in KINECT25_BONES:
        s, e = bone
        xs = [joints[s, 0], joints[e, 0]]
        ys = [joints[s, 2], joints[e, 2]]   # Z as depth axis
        zs = [joints[s, 1], joints[e, 1]]   # Y-up
        ax.plot(xs, ys, zs, color=kinect25_bone_color(bone), lw=2.5)
    ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1],
               c="#FFFFFF", s=25, zorder=5, linewidths=0)

    # auto-range: center + equal aspect
    all_xyz = np.stack([joints[:, 0], joints[:, 2], joints[:, 1]], axis=1)
    center = all_xyz.mean(axis=0)
    half = max((all_xyz.max(axis=0) - all_xyz.min(axis=0)).max() / 2 * 1.3, 0.3)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    ax.set_title(title, fontsize=9, fontweight="bold", color="#EEEEEE", pad=4)
    ax.set_xlabel("X", fontsize=7, color="#AAAACC")
    ax.set_ylabel("Z", fontsize=7, color="#AAAACC")
    ax.set_zlabel("Y", fontsize=7, color="#AAAACC")
    ax.tick_params(labelsize=5, colors="#AAAACC")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333355")
    ax.yaxis.pane.set_edgecolor("#333355")
    ax.zaxis.pane.set_edgecolor("#333355")
    ax.grid(True, color="#333355", linewidth=0.5)
    ax.view_init(elev=15, azim=-65)


def draw_coco17_2d(ax, joints_2d, theta_deg, pad_ratio=0.25):
    """joints_2d: (17, 2) pixel coords — auto-zoom to skeleton bounding box."""
    ax.set_facecolor("#1A1A2E")

    x_min, x_max = joints_2d[:, 0].min(), joints_2d[:, 0].max()
    y_min, y_max = joints_2d[:, 1].min(), joints_2d[:, 1].max()
    w = max(x_max - x_min, 1)
    h = max(y_max - y_min, 1)
    side = max(w, h)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    pad = side * pad_ratio
    ax.set_xlim(cx - side / 2 - pad, cx + side / 2 + pad)
    ax.set_ylim(cy + side / 2 + pad, cy - side / 2 - pad)   # y-down
    ax.set_aspect("equal")

    for bone in COCO17_BONES:
        s, e = bone
        xs = [joints_2d[s, 0], joints_2d[e, 0]]
        ys = [joints_2d[s, 1], joints_2d[e, 1]]
        ax.plot(xs, ys, color=coco17_bone_color(bone), lw=2.5, solid_capstyle="round")

    ax.scatter(joints_2d[:, 0], joints_2d[:, 1], c="#FFFFFF", s=30, zorder=5, linewidths=0)

    sign = "+" if theta_deg > 0 else ""
    label = "0°" if theta_deg == 0 else f"{sign}{theta_deg}°"
    ax.set_title(f"COCO-17\n({label})", fontsize=9, fontweight="bold", color="#EEEEEE",
                 pad=4)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")


# ─── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_pkl",     default=os.path.join(_THIS_DIR, '..', 'data', 'ntu60_3danno.pkl'))
    parser.add_argument("--sample_idx", type=int, default=240,
                        help="Index into annotations list")
    parser.add_argument("--frame",      type=int, default=None,
                        help="Frame index (default: middle frame)")
    parser.add_argument("--out_dir",    default=os.path.join(_THIS_DIR, 'vis_output'))
    parser.add_argument("--n_samples",  type=int, default=3,
                        help="How many different samples to visualize")
    return parser.parse_args()


ANGLES_DEG = [0, 30, -30, 45, -45, 90, -90]

NTU_ACTION_NAMES = {
    0: "drink water", 1: "eat meal", 2: "brush teeth", 3: "brush hair",
    4: "drop", 5: "pick up", 6: "throw", 7: "sit down",
    8: "stand up", 9: "clapping",
}


def visualize_sample(ann, out_path, person_idx=0):
    kp3d_all = ann["keypoint"].astype(np.float32)   # (M, T, 25, 3)
    T = kp3d_all.shape[1]
    frame_idx = T // 2
    kp3d = kp3d_all[person_idx, frame_idx]           # (25, 3)
    label = ann["label"]
    action_name = NTU_ACTION_NAMES.get(label, f"label_{label}")

    # Convert
    coco3d_seq = kinect25_to_coco17_3d(kp3d_all[person_idx])  # (T, 17, 3)
    coco3d = coco3d_seq[frame_idx]                              # (17, 3)

    n_angles = len(ANGLES_DEG)
    # layout: 1 row, (1 Kinect3D + n_angles COCO2D) columns
    n_cols = 1 + n_angles
    fig = plt.figure(figsize=(3.2 * n_cols, 4.5))
    fig.patch.set_facecolor("#12122A")

    title_str = (f"NTU60  ·  {ann['frame_dir']}  ·  action: {action_name}  "
                 f"·  frame {frame_idx}/{T}")
    fig.suptitle(title_str, fontsize=10, fontweight="bold", y=1.01, color="#EEEEEE")

    # ── Kinect-25 3D ──────────────────────────────────────────────────
    ax3d = fig.add_subplot(1, n_cols, 1, projection="3d")
    ax3d.set_facecolor("#1A1A2E")
    draw_kinect25_3d(ax3d, kp3d)

    # ── COCO-17 per angle ────────────────────────────────────────────
    for col, deg in enumerate(ANGLES_DEG):
        ax = fig.add_subplot(1, n_cols, 2 + col)
        theta = np.deg2rad(deg)
        rotated = rotate_azimuth_centered(coco3d[None], theta)[0]   # (17, 3)
        proj2d  = project_kinect_perspective(rotated[None])[0]        # (17, 2)
        draw_coco17_2d(ax, proj2d, deg)

    # ── Legend ───────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=KINECT25_COLORS["spine"],     label="Spine/Head"),
        mpatches.Patch(color=KINECT25_COLORS["left_arm"],  label="L-Arm"),
        mpatches.Patch(color=KINECT25_COLORS["right_arm"], label="R-Arm"),
        mpatches.Patch(color=KINECT25_COLORS["left_leg"],  label="L-Leg"),
        mpatches.Patch(color=KINECT25_COLORS["right_leg"], label="R-Leg"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=5, fontsize=7, framealpha=0.3,
               facecolor="#1A1A2E", labelcolor="#EEEEEE",
               edgecolor="#444466",
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    opts = parse_args()
    os.makedirs(opts.out_dir, exist_ok=True)

    with open(opts.gt_pkl, "rb") as f:
        data = pickle.load(f)

    annotations = data["annotations"]
    split_val   = set(data["split"]["xsub_val"])

    # Collect candidate samples covering diverse actions
    seen_labels = {}
    for i, ann in enumerate(annotations):
        if ann["frame_dir"] not in split_val:
            continue
        lb = ann["label"]
        if lb not in seen_labels:
            seen_labels[lb] = i
        if len(seen_labels) >= 60:
            break

    # Pick n_samples evenly spaced across action classes
    step = max(1, 60 // opts.n_samples)
    chosen = sorted(seen_labels.keys())[::step][: opts.n_samples]

    for lb in chosen:
        ann = annotations[seen_labels[lb]]
        name = NTU_ACTION_NAMES.get(lb, f"label{lb:02d}")
        out_path = os.path.join(opts.out_dir,
                                f"vis_{ann['frame_dir']}_{name.replace(' ','_')}.png")
        visualize_sample(ann, out_path)

    # Also always render the user-specified index
    ann = annotations[opts.sample_idx]
    out_path = os.path.join(opts.out_dir,
                            f"vis_{ann['frame_dir']}_idx{opts.sample_idx}.png")
    visualize_sample(ann, out_path)

    print(f"\nAll done → {opts.out_dir}")


if __name__ == "__main__":
    main()
