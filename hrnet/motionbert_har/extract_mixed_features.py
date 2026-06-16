"""
Extract DSTformer backbone features for all views and save to .npz.
train_{set}.npz: view0=HRNet 2D, view1+=GT 3D projection  (xsub_train, 40091 samples)
test_{set}.npz:  view0=HRNet 2D, view1+=cMAS projection   (xsub_val,   16487 samples)
Output shape: (N, V, 8704)
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MB_ROOT  = os.path.normpath(os.path.join(_THIS_DIR, '..', '..', 'models', 'MotionBERT'))
sys.path.insert(0, _MB_ROOT)

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.data.dataset_multiview import MixedViewTrainDataset, MixedViewTestDataset, ANGLE_SETS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sets', nargs='+', default=['setA', 'setB', 'setC', 'setD'])
    parser.add_argument('--config',        default=os.path.join(_MB_ROOT, 'configs/action/MB_ft_NTU60_xsub.yaml'))
    parser.add_argument('--backbone_ckpt', default=os.path.join(_MB_ROOT, 'save/MB_ft_NTU60_xsub/best_epoch.bin'))
    parser.add_argument('--gt3d_pkl',      default=os.path.join(_THIS_DIR, '..', '..', 'kinect', 'data', 'ntu60_3danno.pkl'))
    parser.add_argument('--hrnet_pkl',     default='data/action/ntu60_hrnet.pkl')
    parser.add_argument('--cmas_dir',      default=os.path.join(_THIS_DIR, '..', 'cmas_pipeline', 'ntu_test_60class_multiview_{set_name}'))
    parser.add_argument('--out_dir',       default='data/mixed_features')
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
        n_views  = len(ANGLE_SETS[set_name])
        cmas_dir = opts.cmas_dir.format(set_name=set_name)

        train_path = os.path.join(opts.out_dir, f'train_{set_name}.npz')
        if not os.path.exists(train_path):
            ds  = MixedViewTrainDataset(opts.hrnet_pkl, opts.gt3d_pkl, set_name)
            ldr = DataLoader(ds, shuffle=False, **ldr_cfg)
            f, l = extract(backbone, ldr, device, n_views, f'train {set_name}')
            np.savez(train_path, features=f, labels=l)
            print(f"[{set_name}] train saved: {f.shape}", flush=True)

        test_path = os.path.join(opts.out_dir, f'test_{set_name}.npz')
        if not os.path.exists(test_path):
            ds  = MixedViewTestDataset(opts.hrnet_pkl, cmas_dir, set_name)
            ldr = DataLoader(ds, shuffle=False, **ldr_cfg)
            f, l = extract(backbone, ldr, device, n_views, f'test {set_name}')
            np.savez(test_path, features=f, labels=l)
            print(f"[{set_name}] test saved: {f.shape}", flush=True)

    print("Done", flush=True)


if __name__ == '__main__':
    main()
