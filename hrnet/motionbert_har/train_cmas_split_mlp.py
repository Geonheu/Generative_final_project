"""
Exp3: Subject-based split on cMAS val features (test_{set}.npz).
Subjects P003/010/021/037 held out as test; remaining val subjects used for training.
Trains single-view (view0) vs multi-view (feature concat) MLP — fair comparison with matched distribution.
"""
import os, re, argparse
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

CMAS_DIRS = {
    'setA': '/home/navygrace/minji/c-MAS/ntu_test_60class_multiview_setA',
    'setB': '/home/navygrace/minji/c-MAS/ntu_test_60class_multiview_setB',
    'setC': '/home/navygrace/minji/c-MAS/ntu_test_60class_multiview_setC',
    'setD': '/home/navygrace/minji/c-MAS/ntu_test_60class_multiview_setD',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sets',           nargs='+', default=['setA', 'setB', 'setC', 'setD'])
    parser.add_argument('--feat_dir',       default='data/mixed_features')
    parser.add_argument('--save_dir',       default='checkpoint/cmas_split_mlp')
    parser.add_argument('--test_subjects',  nargs='+', type=int, default=[3, 10, 21, 37])
    parser.add_argument('--epochs',         type=int,   default=100)
    parser.add_argument('--batch_size',     type=int,   default=256)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--hidden_dim',     type=int,   default=2048)
    parser.add_argument('--dropout',        type=float, default=0.5)
    parser.add_argument('--num_classes',    type=int,   default=60)
    parser.add_argument('--gpu',            default='0')
    return parser.parse_args()


def get_subject_ids(set_name):
    files = sorted(os.listdir(CMAS_DIRS[set_name]))
    return np.array([int(re.search(r'P(\d+)', f).group(1)) for f in files])


class SplitDataset(Dataset):
    def __init__(self, feats, labels):
        self.x = torch.tensor(feats, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]


def build_mlp(in_dim, hidden, n_cls, drop):
    return nn.Sequential(
        nn.Dropout(drop), nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Dropout(drop), nn.Linear(hidden, n_cls),
    )


def train_eval(train_ds, test_ds, opts, device, tag):
    model = build_mlp(train_ds[0][0].shape[0], opts.hidden_dim, opts.num_classes, opts.dropout).to(device)
    opt   = optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opts.epochs)
    crit  = nn.CrossEntropyLoss()
    trl = DataLoader(train_ds, opts.batch_size, shuffle=True,  num_workers=2)
    tel = DataLoader(test_ds,  opts.batch_size, shuffle=False, num_workers=2)

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


def main():
    opts = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opts.save_dir, exist_ok=True)

    print(f"\n{'Set':6} {'SV':>10} {'MV Concat':>12} {'Gain':>8}")
    print('-' * 42)

    for set_name in opts.sets:
        d      = np.load(os.path.join(opts.feat_dir, f'test_{set_name}.npz'))
        feats  = d['features']
        labels = d['labels']
        N, V, D = feats.shape

        subj_ids   = get_subject_ids(set_name)
        test_mask  = np.isin(subj_ids, opts.test_subjects)
        train_mask = ~test_mask

        sv_acc = train_eval(
            SplitDataset(feats[train_mask, 0, :], labels[train_mask]),
            SplitDataset(feats[test_mask,  0, :], labels[test_mask]),
            opts, device, f'{set_name}_sv'
        )
        mv_acc = train_eval(
            SplitDataset(feats[train_mask].reshape(train_mask.sum(), V * D), labels[train_mask]),
            SplitDataset(feats[test_mask].reshape(test_mask.sum(),  V * D), labels[test_mask]),
            opts, device, f'{set_name}_mv'
        )
        print(f"{set_name:6} {sv_acc:>9.2f}% {mv_acc:>11.2f}% {mv_acc-sv_acc:>+7.2f}%p", flush=True)


if __name__ == '__main__':
    main()
