"""
Exp2 variant: View-attention fusion MLP.
view0 acts as query; attention weights over all views before classification.
Train: mixed features (HRNet + GT proj). Test: mixed features (HRNet + cMAS).
"""
import os, argparse
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sets',        nargs='+', default=['setA', 'setB', 'setC', 'setD'])
    parser.add_argument('--feat_dir',    default='data/mixed_features')
    parser.add_argument('--save_dir',    default='checkpoint/view_attn_mlp')
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--attn_dim',    type=int,   default=512)
    parser.add_argument('--hidden_dim',  type=int,   default=2048)
    parser.add_argument('--dropout',     type=float, default=0.5)
    parser.add_argument('--num_classes', type=int,   default=60)
    parser.add_argument('--gpu',         default='0')
    return parser.parse_args()


class FeatDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.feats  = torch.tensor(d['features'], dtype=torch.float32)
        self.labels = torch.tensor(d['labels'],   dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.feats[i], self.labels[i]


class ViewAttentionMLP(nn.Module):
    """view0 as query; dot-product attention over all views; classify fused feature."""
    def __init__(self, feat_dim, attn_dim, hidden, n_cls, drop):
        super().__init__()
        self.scale  = attn_dim ** -0.5
        self.q_proj = nn.Linear(feat_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, attn_dim, bias=False)
        self.head   = nn.Sequential(
            nn.Dropout(drop), nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(hidden, n_cls),
        )

    def forward(self, feats):
        q       = self.q_proj(feats[:, 0:1, :])
        k       = self.k_proj(feats)
        weights = (torch.bmm(q, k.transpose(1, 2)) * self.scale).softmax(dim=-1)
        fused   = torch.bmm(weights, feats).squeeze(1)
        return self.head(fused)


def run(train_ds, test_ds, opts, device, tag):
    model = ViewAttentionMLP(8704, opts.attn_dim, opts.hidden_dim,
                             opts.num_classes, opts.dropout).to(device)
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

    print(f"\n{'Set':6} {'View-Attn':>12}")
    print('-' * 22)
    for set_name in opts.sets:
        tp = os.path.join(opts.feat_dir, f'train_{set_name}.npz')
        ep = os.path.join(opts.feat_dir, f'test_{set_name}.npz')
        if not os.path.exists(tp):
            continue
        acc = run(FeatDataset(tp), FeatDataset(ep), opts, device, set_name)
        print(f"{set_name:6} {acc:>11.2f}%", flush=True)


if __name__ == '__main__':
    main()
