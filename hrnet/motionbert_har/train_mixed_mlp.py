"""
Exp2: Train MLP on mixed features (view0=HRNet + view1+=GT 3D proj), test on cMAS.
Trains both feature-avg and feature-concat fusion variants.
"""
import os, argparse
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sets',        nargs='+', default=['setA', 'setB', 'setC', 'setD'])
    parser.add_argument('--feat_dir',    default='data/mixed_features')
    parser.add_argument('--save_dir',    default='checkpoint/mixed_mlp')
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--hidden_dim',  type=int,   default=2048)
    parser.add_argument('--dropout',     type=float, default=0.5)
    parser.add_argument('--num_classes', type=int,   default=60)
    parser.add_argument('--gpu',         default='0')
    return parser.parse_args()


class FeatDataset(Dataset):
    def __init__(self, npz_path, mode):
        d = np.load(npz_path)
        feats = torch.tensor(d['features'], dtype=torch.float32)
        self.labels = torch.tensor(d['labels'], dtype=torch.long)
        self.x = feats.mean(dim=1) if mode == 'avg' else feats.reshape(len(feats), -1)
    def __len__(self):  return len(self.labels)
    def __getitem__(self, i): return self.x[i], self.labels[i]


def build_mlp(in_dim, hidden, n_cls, drop):
    return nn.Sequential(
        nn.Dropout(drop), nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Dropout(drop), nn.Linear(hidden, n_cls),
    )


def run(train_ds, test_ds, opts, device, tag):
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
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
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

    print(f"\n{'Set':6} {'Mode':10} {'Acc':>10}")
    print('-' * 30)
    for set_name in opts.sets:
        tp = os.path.join(opts.feat_dir, f'train_{set_name}.npz')
        ep = os.path.join(opts.feat_dir, f'test_{set_name}.npz')
        if not os.path.exists(tp):
            continue
        for mode in ['avg', 'concat']:
            acc = run(FeatDataset(tp, mode), FeatDataset(ep, mode),
                      opts, device, f'{set_name}_{mode}')
            print(f"{set_name:6} feat-{mode:6} {acc:>9.2f}%", flush=True)


if __name__ == '__main__':
    main()
