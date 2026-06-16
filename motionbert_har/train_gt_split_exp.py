"""
Subject-split experiments using GT 3D projection features (no domain gap).

Part A: Split xsub_train samples (train_{set}.npz) by subject.
        Holdout subjects P001/009/025/035 as test. Train and test both use GT projection.

Part B: Split xsub_val samples (data/gt_val_features/{set}.npz) by subject.
        Same split as Exp3 (P003/010/021/037 as test). Both train and test use GT projection.
        Run extract_gt_val_features.py first to generate gt_val_features.
"""
import os, re, argparse, pickle
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

SETS = ['setA', 'setB', 'setC', 'setD']
PART_A_TEST_SUBJECTS = [1, 9, 25, 35]
PART_B_TEST_SUBJECTS = [3, 10, 21, 37]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sets',           nargs='+', default=SETS)
    parser.add_argument('--hrnet_pkl',      default='data/action/ntu60_hrnet.pkl')
    parser.add_argument('--gt3d_pkl',       default='/home/navygrace/minji/ntu60_3danno.pkl')
    parser.add_argument('--train_feat_dir', default='data/mixed_features')
    parser.add_argument('--gt_val_dir',     default='data/gt_val_features')
    parser.add_argument('--save_dir',       default='checkpoint/gt_split_mlp')
    parser.add_argument('--epochs',         type=int,   default=100)
    parser.add_argument('--batch_size',     type=int,   default=256)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--hidden_dim',     type=int,   default=2048)
    parser.add_argument('--dropout',        type=float, default=0.5)
    parser.add_argument('--num_classes',    type=int,   default=60)
    parser.add_argument('--gpu',            default='0')
    return parser.parse_args()


class SplitDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]


def build_mlp(in_dim, hidden=2048, n_cls=60, drop=0.5):
    return nn.Sequential(
        nn.Dropout(drop), nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Dropout(drop), nn.Linear(hidden, n_cls),
    )


def train_eval(tr_ds, te_ds, opts, device, tag):
    model = build_mlp(tr_ds[0][0].shape[0], opts.hidden_dim, opts.num_classes, opts.dropout).to(device)
    opt   = optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opts.epochs)
    crit  = nn.CrossEntropyLoss()
    trl = DataLoader(tr_ds, opts.batch_size, shuffle=True,  num_workers=2)
    tel = DataLoader(te_ds, opts.batch_size, shuffle=False, num_workers=2)
    best = 0.0
    for ep in range(1, opts.epochs + 1):
        model.train()
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in tel:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        acc = correct / total * 100
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(opts.save_dir, f'best_{tag}.pth'))
    return best


def run_split(label, feats, labels, subj_ids, test_subjs, opts, device, tag_prefix):
    N, V, D    = feats.shape
    test_mask  = np.isin(subj_ids, test_subjs)
    train_mask = ~test_mask
    n_tr, n_te = train_mask.sum(), test_mask.sum()
    print(f"\n[{label}] train={n_tr}, test={n_te}", flush=True)

    results = {}
    for mode, x_tr, x_te in [
        ('sv',          feats[train_mask, 0, :],                      feats[test_mask, 0, :]),
        ('feat_avg',    feats[train_mask].mean(axis=1),                feats[test_mask].mean(axis=1)),
        ('feat_concat', feats[train_mask].reshape(n_tr, V * D),       feats[test_mask].reshape(n_te, V * D)),
    ]:
        acc = train_eval(
            SplitDataset(x_tr, labels[train_mask]),
            SplitDataset(x_te, labels[test_mask]),
            opts, device, f'{tag_prefix}_{mode}'
        )
        results[mode] = acc
        print(f"  {mode:<14} {acc:.2f}%", flush=True)
    return results


def get_train_subject_ids(hrnet_pkl, gt3d_pkl):
    with open(hrnet_pkl, 'rb') as f:
        hrnet = pickle.load(f)
    train_set = set(hrnet['split']['xsub_train'])
    hrnet_map = {a['frame_dir']: a for a in hrnet['annotations'] if a['frame_dir'] in train_set}

    with open(gt3d_pkl, 'rb') as f:
        gt3d = pickle.load(f)
    gt3d_set = set(gt3d['split']['xsub_train'])
    gt3d_map = {a['frame_dir']: a for a in gt3d['annotations'] if a['frame_dir'] in gt3d_set}

    frame_dirs = [fdir for fdir in hrnet_map if fdir in gt3d_map]
    return np.array([int(re.search(r'P(\d+)', d).group(1)) for d in frame_dirs])


def main():
    opts = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opts.save_dir, exist_ok=True)

    # Part A
    print("=" * 60)
    print(f"Part A: xsub_train GT proj split  (test={PART_A_TEST_SUBJECTS})")
    train_subj_ids = get_train_subject_ids(opts.hrnet_pkl, opts.gt3d_pkl)
    part_a = {}
    for s in opts.sets:
        d = np.load(os.path.join(opts.train_feat_dir, f'train_{s}.npz'))
        part_a[s] = run_split(f'A {s}', d['features'], d['labels'],
                               train_subj_ids, PART_A_TEST_SUBJECTS, opts, device, f'partA_{s}')

    # Part B
    print("\n" + "=" * 60)
    print(f"Part B: xsub_val GT proj split  (test={PART_B_TEST_SUBJECTS}, same as Exp3)")
    part_b = {}
    for s in opts.sets:
        path = os.path.join(opts.gt_val_dir, f'{s}.npz')
        if not os.path.exists(path):
            print(f"  [{s}] gt_val_features not found, run extract_gt_val_features.py first")
            continue
        d        = np.load(path)
        subj_ids = np.array([int(re.search(r'P(\d+)', fd).group(1)) for fd in d['frame_dirs']])
        part_b[s] = run_split(f'B {s}', d['features'], d['labels'],
                               subj_ids, PART_B_TEST_SUBJECTS, opts, device, f'partB_{s}')

    # Summary table
    print("\n" + "=" * 68)
    print(f"{'':>22} {'setA':>8} {'setB':>8} {'setC':>8} {'setD':>8}")
    print("-" * 56)
    for part_label, part_data in [("Part A (xsub_train GT proj)", part_a),
                                   ("Part B (xsub_val GT proj)",   part_b)]:
        print(f"\n{part_label}")
        for mode in ['sv', 'feat_avg', 'feat_concat']:
            vals = [part_data.get(s, {}).get(mode, float('nan')) for s in SETS]
            print(f"  {mode:<20} {vals[0]:>8.2f} {vals[1]:>8.2f} {vals[2]:>8.2f} {vals[3]:>8.2f}")

    print("\nRef: Exp3 (cMAS val GT proj, same subject split)")
    for mode, ref in [('sv',          dict(setA=87.42, setB=87.52, setC=87.42, setD=87.50)),
                      ('feat_avg',    dict(setA=86.45, setB=86.63, setC=83.07, setD=83.80)),
                      ('feat_concat', dict(setA=88.27, setB=87.95, setC=87.46, setD=87.80))]:
        vals = [ref[s] for s in SETS]
        print(f"  Exp3 {mode:<16} {vals[0]:>8.2f} {vals[1]:>8.2f} {vals[2]:>8.2f} {vals[3]:>8.2f}")
    print("=" * 68)


if __name__ == '__main__':
    main()
