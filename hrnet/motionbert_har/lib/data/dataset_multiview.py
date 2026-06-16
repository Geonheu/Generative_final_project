import numpy as np
import pickle
import os
import torch
from torch.utils.data import Dataset
from lib.data.dataset_action import make_cam, coco2h36m, resample
from lib.utils.utils_data import crop_scale

# NTU25 → COCO17 joint index mapping
NTU25_TO_COCO17 = [3, 3, 3, 3, 3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]

ANGLE_SETS = {
    "setA": [0, np.pi / 4, -np.pi / 4],
    "setB": [0, np.pi / 6, -np.pi / 6],
    "setC": [0, np.pi / 2, -np.pi / 2],
    "setD": [0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2],
}


def ntu25_to_coco17(kp):
    return kp[:, NTU25_TO_COCO17, :]


def rotate_azimuth(pose_3d, theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]], dtype=np.float32)
    return pose_3d @ R.T


def project_to_pixel(pose_3d, img_w=1920, img_h=1080, scale=115.2):
    x_pixel = -pose_3d[:, :, 0] * scale + img_w / 2
    y_pixel = -pose_3d[:, :, 1] * scale + img_h / 2
    return np.stack([x_pixel, y_pixel], axis=-1)


def _to_motionbert_input(xy, conf, n_frames, resample_id, scale_range):
    motion     = xy[np.newaxis, resample_id]
    motion_cam = make_cam(motion, (1080, 1920))
    motion_c   = conf[np.newaxis, resample_id] if conf.ndim == 2 else conf[np.newaxis, resample_id, :, np.newaxis]
    motion     = np.concatenate([motion_cam, motion_c], axis=-1)
    motion     = coco2h36m(motion)
    fake       = np.zeros_like(motion)
    motion     = np.concatenate([motion, fake], axis=0)
    return crop_scale(motion, scale_range=scale_range)


class MixedViewTrainDataset(Dataset):
    """Train: view0=HRNet 2D, view1+=GT 3D rotated projection (xsub_train)."""
    def __init__(self, hrnet_pkl, gt3d_pkl, set_name, n_frames=243, scale_range=(2, 2)):
        self.angles   = ANGLE_SETS[set_name]
        self.n_frames = n_frames
        self.scale    = scale_range

        with open(hrnet_pkl, 'rb') as f:
            hrnet = pickle.load(f)
        train_set  = set(hrnet['split']['xsub_train'])
        hrnet_map  = {a['frame_dir']: a for a in hrnet['annotations'] if a['frame_dir'] in train_set}

        with open(gt3d_pkl, 'rb') as f:
            gt3d = pickle.load(f)
        gt3d_set  = set(gt3d['split']['xsub_train'])
        gt3d_map  = {a['frame_dir']: a for a in gt3d['annotations'] if a['frame_dir'] in gt3d_set}

        self.samples = []
        for fdir, ha in hrnet_map.items():
            if fdir not in gt3d_map:
                continue
            ga = gt3d_map[fdir]
            kp  = ga['keypoint'][0].astype(np.float32)
            kp -= kp[:, 0:1, :]
            self.samples.append((
                ha['keypoint'][0].astype(np.float32),
                ha['keypoint_score'][0].astype(np.float32),
                kp, ha['label'], ga['total_frames']
            ))

        print(f"MixedViewTrainDataset ({set_name}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gt_xy, gt_conf, kp_ntu25, label, T_3d = self.samples[idx]
        T_gt  = gt_xy.shape[0]
        kp    = ntu25_to_coco17(kp_ntu25)

        views = []
        for i, theta in enumerate(self.angles):
            if i == 0:
                motion = _to_motionbert_input(
                    gt_xy, gt_conf[:, :, np.newaxis] if gt_conf.ndim == 2 else gt_conf,
                    self.n_frames, resample(T_gt, self.n_frames, randomness=False), self.scale
                )
                # rebuild: gt_conf needs shape adjustment
                rid_gt = resample(T_gt, self.n_frames, randomness=False)
                m      = gt_xy[np.newaxis, rid_gt]
                mc     = make_cam(m, (1080, 1920))
                mconf  = gt_conf[np.newaxis, rid_gt, :, np.newaxis]
                m      = np.concatenate([mc, mconf], axis=-1)
                m      = coco2h36m(m)
                m      = np.concatenate([m, np.zeros_like(m)], axis=0)
                motion = crop_scale(m, scale_range=self.scale)
            else:
                rid_3d    = resample(T_3d, self.n_frames, randomness=False)
                rotated   = rotate_azimuth(kp, theta)
                projected = project_to_pixel(rotated)
                conf      = np.ones((T_3d, 17, 1), dtype=np.float32)
                xy_conf   = np.concatenate([projected, conf], axis=-1)
                m         = xy_conf[np.newaxis, rid_3d]
                mc        = make_cam(m[..., :2], (1080, 1920))
                mconf     = m[..., 2:3]
                m         = np.concatenate([mc, mconf], axis=-1)
                m         = coco2h36m(m)
                m         = np.concatenate([m, np.zeros_like(m)], axis=0)
                motion    = crop_scale(m, scale_range=self.scale)
            views.append(torch.tensor(motion, dtype=torch.float32))

        return views, label


class MixedViewTestDataset(Dataset):
    """Test: view0=HRNet 2D, view1+=cMAS virtual projections (xsub_val)."""
    def __init__(self, hrnet_pkl, multiview_dir, set_name, n_frames=243, scale_range=(2, 2)):
        self.n_views  = len(ANGLE_SETS[set_name])
        self.n_frames = n_frames
        self.scale    = scale_range

        with open(hrnet_pkl, 'rb') as f:
            hrnet = pickle.load(f)
        val_set    = set(hrnet['split']['xsub_val'])
        self.gt_map = {a['frame_dir']: a for a in hrnet['annotations'] if a['frame_dir'] in val_set}

        self.data = []
        for fname in sorted(os.listdir(multiview_dir)):
            fdir = fname.split('_label')[0]
            d    = np.load(os.path.join(multiview_dir, fname), allow_pickle=True).item()
            self.data.append((fdir, d['multiview'], d['label']))

        print(f"MixedViewTestDataset ({set_name}): {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fdir, multiview, label = self.data[idx]
        gt_ann = self.gt_map[fdir]
        T_orig = multiview.shape[1]
        rid    = resample(T_orig, self.n_frames, randomness=False)

        gt_xy   = gt_ann['keypoint'][0]
        gt_conf = gt_ann['keypoint_score'][0]
        T_gt    = gt_xy.shape[0]
        rid_gt  = resample(T_gt, self.n_frames, randomness=False)

        views = []
        for v in range(self.n_views):
            if v == 0:
                m     = gt_xy[np.newaxis, rid_gt]
                mc    = make_cam(m, (1080, 1920))
                mconf = gt_conf[np.newaxis, rid_gt, :, np.newaxis]
                m     = np.concatenate([mc, mconf], axis=-1)
            else:
                view  = multiview[v][np.newaxis, rid]
                mc    = make_cam(view[..., :2], (1080, 1920))
                mconf = view[..., 2:3]
                m     = np.concatenate([mc, mconf], axis=-1)
            m = coco2h36m(m)
            m = np.concatenate([m, np.zeros_like(m)], axis=0)
            m = crop_scale(m, scale_range=self.scale)
            views.append(torch.tensor(m, dtype=torch.float32))

        return views, label


class GTValDataset(Dataset):
    """Val set with GT 3D projection for both train and test views (used for Part B)."""
    def __init__(self, hrnet_pkl, gt3d_pkl, set_name, n_frames=243, scale_range=(2, 2)):
        self.angles   = ANGLE_SETS[set_name]
        self.n_frames = n_frames
        self.scale    = scale_range

        with open(hrnet_pkl, 'rb') as f:
            hrnet = pickle.load(f)
        val_set    = set(hrnet['split']['xsub_val'])
        hrnet_map  = {a['frame_dir']: a for a in hrnet['annotations'] if a['frame_dir'] in val_set}

        with open(gt3d_pkl, 'rb') as f:
            gt3d = pickle.load(f)
        gt3d_val  = set(gt3d['split']['xsub_val'])
        gt3d_map  = {a['frame_dir']: a for a in gt3d['annotations'] if a['frame_dir'] in gt3d_val}

        self.samples    = []
        self.frame_dirs = []
        for fdir, ha in hrnet_map.items():
            if fdir not in gt3d_map:
                continue
            ga  = gt3d_map[fdir]
            kp  = ga['keypoint'][0].astype(np.float32)
            kp -= kp[:, 0:1, :]
            self.samples.append((
                ha['keypoint'][0].astype(np.float32),
                ha['keypoint_score'][0].astype(np.float32),
                kp, ha['label'], ga['total_frames']
            ))
            self.frame_dirs.append(fdir)

        print(f"GTValDataset ({set_name}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gt_xy, gt_conf, kp_ntu25, label, T_3d = self.samples[idx]
        T_gt  = gt_xy.shape[0]
        kp    = ntu25_to_coco17(kp_ntu25)

        views = []
        for i, theta in enumerate(self.angles):
            if i == 0:
                rid_gt = resample(T_gt, self.n_frames, randomness=False)
                m      = gt_xy[np.newaxis, rid_gt]
                mc     = make_cam(m, (1080, 1920))
                mconf  = gt_conf[np.newaxis, rid_gt, :, np.newaxis]
                m      = np.concatenate([mc, mconf], axis=-1)
            else:
                rid_3d    = resample(T_3d, self.n_frames, randomness=False)
                rotated   = rotate_azimuth(kp, theta)
                projected = project_to_pixel(rotated)
                conf      = np.ones((T_3d, 17, 1), dtype=np.float32)
                xy_conf   = np.concatenate([projected, conf], axis=-1)
                m         = xy_conf[np.newaxis, rid_3d]
                mc        = make_cam(m[..., :2], (1080, 1920))
                mconf     = m[..., 2:3]
                m         = np.concatenate([mc, mconf], axis=-1)
            m = coco2h36m(m)
            m = np.concatenate([m, np.zeros_like(m)], axis=0)
            m = crop_scale(m, scale_range=self.scale)
            views.append(torch.tensor(m, dtype=torch.float32))

        return views, label
