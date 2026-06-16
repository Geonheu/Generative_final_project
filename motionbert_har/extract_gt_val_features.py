"""
Extract DSTformer features for xsub_val using GT 3D projection (no cMAS).
Used for Part B experiment: train/test both with GT projection, Exp3 subject split.
Output: data/gt_val_features/{set_name}.npz  shape=(N, V, 8704)
  - features: (N, V, 8704)
  - labels:   (N,)
  - frame_dirs: (N,)  for subject-based split
"""
import os, argparse, pickle, re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.data.dataset_multiview import GTValDataset, ANGLE_SETS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sets',          nargs='+', default=['setA', 'setB', 'setC', 'setD'])
    parser.add_argument('--config',        default='configs/action/MB_ft_NTU60_xsub.yaml')
    parser.add_argument('--backbone_ckpt', default='checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch.bin')
    parser.add_argument('--gt3d_pkl',      default='/home/navygrace/minji/ntu60_3danno.pkl')
    parser.add_argument('--hrnet_pkl',     default='data/action/ntu60_hrnet.pkl')
    parser.add_argument('--out_dir',       default='data/gt_val_features')
    parser.add_argument('--batch_size',    type=int, default=64)
    parser.add_argument('--gpu',           default='0')
    return parser.parse_args()


def collate_views(batch):
    n_views = len(batch[0][0])
    views  = [torch.stack([b[0][v] for b in batch]) for v in range(n_views)]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return views, labels


def extract(backbone, loader, device, n_views, desc):
    backbone.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for views, labels in tqdm(loader, desc=desc):
            vf = []
            for v in range(n_views):
                x = views[v].to(device)
                NB, M, T, J, C = x.shape
                feat = backbone.get_representation(x.reshape(NB * M, T, J, C))
                feat = feat.reshape(NB, M, T, J, -1).mean(2).mean(1).reshape(NB, -1)
                vf.append(feat.cpu())
            all_feats.append(torch.stack(vf, dim=1))
            all_labels.append(labels)
    return torch.cat(all_feats).numpy(), torch.cat(all_labels).numpy()


def main():
    opts = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opts.out_dir, exist_ok=True)

    cfg      = get_config(opts.config)
    backbone = load_backbone(cfg).to(device)
    ckpt     = torch.load(opts.backbone_ckpt, map_location='cpu')
    state    = {k.replace('module.backbone.', ''): v
                for k, v in ckpt['model'].items() if 'backbone' in k}
    backbone.load_state_dict(state, strict=True)
    print("Backbone loaded", flush=True)

    ldr_cfg = dict(batch_size=opts.batch_size, collate_fn=collate_views,
                   num_workers=4, pin_memory=True)

    for set_name in opts.sets:
        out_path = os.path.join(opts.out_dir, f'{set_name}.npz')
        if os.path.exists(out_path):
            print(f"[{set_name}] already exists, skipping", flush=True)
            continue

        ds      = GTValDataset(opts.hrnet_pkl, opts.gt3d_pkl, set_name)
        n_views = len(ANGLE_SETS[set_name])
        ldr     = DataLoader(ds, shuffle=False, **ldr_cfg)
        feats, labels = extract(backbone, ldr, device, n_views, f'GT val {set_name}')

        frame_dirs = np.array(ds.frame_dirs)
        np.savez(out_path, features=feats, labels=labels, frame_dirs=frame_dirs)
        print(f"[{set_name}] saved: {out_path}  shape={feats.shape}", flush=True)

    print("Done", flush=True)


if __name__ == '__main__':
    main()
