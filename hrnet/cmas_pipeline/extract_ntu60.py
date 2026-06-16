"""
Extract NTU60 xsub_val samples from ntu60_hrnet.pkl and convert to NBA16 format.
Output: one .npy per sample in save_dir, filename: {frame_dir}_label{label:02d}.npy
"""
import pickle
import numpy as np
import os

# COCO17 → NBA16 joint mapping
# COCO17: Nose(0) LEye(1) REye(2) LEar(3) REar(4) LSho(5) RSho(6) LElb(7) RElb(8)
#         LWri(9) RWri(10) LHip(11) RHip(12) LKnee(13) RKnee(14) LAnk(15) RAnk(16)
# NBA16:  Hip(0) RHip(1) RKnee(2) RAnk(3) LHip(4) LKnee(5) LAnk(6)
#         Neck(7) Nose(8) Head(9) LSho(10) LElb(11) LWri(12) RSho(13) RElb(14) RWri(15)
def coco17_to_nba16(coco_seq, coco_conf):
    T = coco_seq.shape[0]
    nba_xy   = np.zeros((T, 16, 2), dtype=np.float32)
    nba_conf = np.zeros((T, 16),    dtype=np.float32)

    mapping = {1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15,
               8: 0, 10: 5, 11: 7, 12: 9, 13: 6, 14: 8, 15: 10}
    for dst, src in mapping.items():
        nba_xy[:, dst]   = coco_seq[:, src]
        nba_conf[:, dst] = coco_conf[:, src]

    nba_xy[:, 0]   = (coco_seq[:, 11]  + coco_seq[:, 12])  / 2
    nba_xy[:, 7]   = (coco_seq[:, 5]   + coco_seq[:, 6])   / 2
    nba_xy[:, 9]   = (coco_seq[:, 1]   + coco_seq[:, 2])   / 2
    nba_conf[:, 0] = (coco_conf[:, 11] + coco_conf[:, 12]) / 2
    nba_conf[:, 7] = (coco_conf[:, 5]  + coco_conf[:, 6])  / 2
    nba_conf[:, 9] = (coco_conf[:, 1]  + coco_conf[:, 2])  / 2

    return np.concatenate([nba_xy, nba_conf[:, :, np.newaxis]], axis=-1)


if __name__ == "__main__":
    pkl_path = "data/action/ntu60_hrnet.pkl"
    save_dir = "ntu_test_60class_nba"
    os.makedirs(save_dir, exist_ok=True)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    ann_dict = {ann['frame_dir']: ann for ann in data['annotations']}
    test_dirs = data['split']['xsub_val']
    filtered = [(d, ann_dict[d]) for d in test_dirs
                if d in ann_dict and ann_dict[d]['label'] < 60]

    print(f"Total samples: {len(filtered)}")

    for i, (frame_dir, ann) in enumerate(filtered):
        coco_seq  = ann['keypoint'][0].astype(np.float32)
        coco_conf = ann['keypoint_score'][0].astype(np.float32)
        label     = ann['label']

        nba_seq = coco17_to_nba16(coco_seq, coco_conf)
        root = nba_seq[:, 0:1, :2].copy()
        nba_seq[:, :, :2] -= root
        nba_seq[:, :, 1]   = -nba_seq[:, :, 1]
        nba_seq[:, :, :2] /= 100.0

        np.save(os.path.join(save_dir, f"{frame_dir}_label{label:02d}.npy"), nba_seq)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(filtered)}")

    print(f"Done. Files: {len(os.listdir(save_dir))}")
