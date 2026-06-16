"""
Convert cMAS 3D output to NTU-style multi-view 2D projections.
NBA16 → COCO17 remapping → azimuth rotation → pixel projection.
Input:  ntu_test_60class_3d/   (cMAS 3D output, NBA16 joints)
Output: ntu_test_60class_multiview_{set}/  (2D multi-view .npy per sample)
"""
import numpy as np
import os

# NBA16 → COCO17 joint remapping
# NBA16: Hip(0) RHip(1) RKnee(2) RAnk(3) LHip(4) LKnee(5) LAnk(6)
#        Neck(7) Nose(8) Head(9) LSho(10) LElb(11) LWri(12) RSho(13) RElb(14) RWri(15)
# COCO17: Nose(0) LEye(1-4≈Nose) LSho(5) RSho(6) LElb(7) RElb(8)
#         LWri(9) RWri(10) LHip(11) RHip(12) LKnee(13) RKnee(14) LAnk(15) RAnk(16)
def nba16_to_coco17_3d(nba_3d):
    T = nba_3d.shape[0]
    coco = np.zeros((T, 17, 3), dtype=np.float32)
    coco[:, 0]  = nba_3d[:, 8]
    coco[:, 1]  = nba_3d[:, 8]   # LEye → approximated by Nose
    coco[:, 2]  = nba_3d[:, 8]   # REye → approximated by Nose
    coco[:, 3]  = nba_3d[:, 8]   # LEar → approximated by Nose
    coco[:, 4]  = nba_3d[:, 8]   # REar → approximated by Nose
    coco[:, 5]  = nba_3d[:, 10]
    coco[:, 6]  = nba_3d[:, 13]
    coco[:, 7]  = nba_3d[:, 11]
    coco[:, 8]  = nba_3d[:, 14]
    coco[:, 9]  = nba_3d[:, 12]
    coco[:, 10] = nba_3d[:, 15]
    coco[:, 11] = nba_3d[:, 4]
    coco[:, 12] = nba_3d[:, 1]
    coco[:, 13] = nba_3d[:, 5]
    coco[:, 14] = nba_3d[:, 2]
    coco[:, 15] = nba_3d[:, 6]
    coco[:, 16] = nba_3d[:, 3]
    return coco


def rotate_azimuth(pose_3d, theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]])
    return pose_3d @ R.T


def project_to_pixel(pose_3d, img_w=1920, img_h=1080):
    scale = img_w * 0.06
    x_pixel = -pose_3d[:, :, 0] * scale + img_w / 2
    y_pixel = -pose_3d[:, :, 1] * scale + img_h / 2
    return np.stack([x_pixel, y_pixel], axis=-1)


ANGLE_SETS = {
    "setA": [0, np.pi / 4, -np.pi / 4],
    "setB": [0, np.pi / 6, -np.pi / 6],
    "setC": [0, np.pi / 2, -np.pi / 2],
    "setD": [0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2],
}


if __name__ == "__main__":
    input_dir = "ntu_test_60class_3d"

    for set_name in ANGLE_SETS:
        os.makedirs(f"ntu_test_60class_multiview_{set_name}", exist_ok=True)

    files = sorted(os.listdir(input_dir))
    print(f"Processing {len(files)} samples")

    for i, fname in enumerate(files):
        result = np.load(os.path.join(input_dir, fname), allow_pickle=True).item()
        nba_3d = result['motions'][0]          # (T, 16, 3)
        coco_3d = nba16_to_coco17_3d(nba_3d)  # (T, 17, 3)
        label = int(fname.split('_label')[1].replace('.npy', ''))

        for set_name, angles in ANGLE_SETS.items():
            save_path = f"ntu_test_60class_multiview_{set_name}/{fname}"
            if os.path.exists(save_path):
                continue

            views = []
            for theta in angles:
                rotated   = rotate_azimuth(coco_3d, theta)
                projected = project_to_pixel(rotated)
                conf      = np.ones((coco_3d.shape[0], 17, 1), dtype=np.float32)
                views.append(np.concatenate([projected, conf], axis=-1))

            multiview = np.stack(views, axis=0)  # (V, T, 17, 3)
            np.save(save_path, {'multiview': multiview, 'label': label})

        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(files)}")

    print("Done")
