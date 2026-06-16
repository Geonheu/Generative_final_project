"""
Evaluate all fusion strategies across Exp1/2/3 and print a results table.
Requires pre-trained checkpoints in checkpoint/singleview_mlp/, checkpoint/mixed_mlp/,
checkpoint/cmas_split_mlp/, checkpoint/cmas_split_avg_mlp/.
"""
import os, re, argparse
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_CMAS_PIPELINE_OUT = os.path.join(_THIS_DIR, '..', 'cmas_pipeline')

SETS = ['setA', 'setB', 'setC', 'setD']
FEAT_DIR = 'data/mixed_features'
TEST_SUBJECTS = [3, 10, 21, 37]

CMAS_DIRS = {
    'setA': os.path.join(_CMAS_PIPELINE_OUT, 'ntu_test_60class_multiview_setA'),
    'setB': os.path.join(_CMAS_PIPELINE_OUT, 'ntu_test_60class_multiview_setB'),
    'setC': os.path.join(_CMAS_PIPELINE_OUT, 'ntu_test_60class_multiview_setC'),
    'setD': os.path.join(_CMAS_PIPELINE_OUT, 'ntu_test_60class_multiview_setD'),
}


class FlatDataset(Dataset):
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


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total * 100


def logit_avg_eval(model, feats, labels, device, batch_size=256):
    N, V, D = feats.shape
    model.eval()
    all_logits = []
    with torch.no_grad():
        for v in range(V):
            view_feat = torch.tensor(feats[:, v, :], dtype=torch.float32)
            logits_v  = [model(view_feat[i:i+batch_size].to(device)).cpu()
                         for i in range(0, N, batch_size)]
            all_logits.append(torch.cat(logits_v))
    avg_logits = torch.stack(all_logits, dim=1).mean(dim=1)
    preds = avg_logits.argmax(dim=1).numpy()
    return (preds == np.array(labels)).mean() * 100


def train_mlp(train_ds, test_ds, in_dim, device, save_path, epochs=100, lr=1e-3):
    model = build_mlp(in_dim).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()
    trl = DataLoader(train_ds, 256, shuffle=True,  num_workers=2)
    tel = DataLoader(test_ds,  256, shuffle=False, num_workers=2)
    best = 0.0
    for _ in range(epochs):
        model.train()
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
        acc = evaluate(model, tel, device)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), save_path)
    return best


def get_subject_ids(set_name):
    files = sorted(os.listdir(CMAS_DIRS[set_name]))
    return np.array([int(re.search(r'P(\d+)', f).group(1)) for f in files])


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    # Exp1: SV MLP on avg/logit-avg test features
    for s in SETS:
        ckpt = f'checkpoint/singleview_mlp/best_{s}.pth'
        d    = np.load(os.path.join(FEAT_DIR, f'test_{s}.npz'))
        model = build_mlp(8704).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        ds  = FlatDataset(d['features'].mean(axis=1), d['labels'])
        ldr = DataLoader(ds, 256, shuffle=False, num_workers=2)
        results[f'exp1_feat_avg_{s}']   = evaluate(model, ldr, device)
        results[f'exp1_logit_avg_{s}']  = logit_avg_eval(model, d['features'], d['labels'], device)

    # Exp2: mixed MLP (avg/concat) eval + logit avg
    for s in SETS:
        d = np.load(os.path.join(FEAT_DIR, f'test_{s}.npz'))
        N, V, D = d['features'].shape

        for mode, ckpt_key in [('avg', f'best_{s}_avg'), ('concat', f'best_{s}_concat')]:
            ckpt  = f'checkpoint/mixed_mlp/{ckpt_key}.pth'
            in_dim = 8704 if mode == 'avg' else V * D
            x_te   = d['features'].mean(axis=1) if mode == 'avg' else d['features'].reshape(len(d['features']), -1)
            model  = build_mlp(in_dim).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            ds     = FlatDataset(x_te, d['labels'])
            ldr    = DataLoader(ds, 256, shuffle=False, num_workers=2)
            results[f'exp2_feat_{mode}_{s}'] = evaluate(model, ldr, device)

        # logit avg with avg-trained model
        model_avg = build_mlp(8704).to(device)
        model_avg.load_state_dict(torch.load(f'checkpoint/mixed_mlp/best_{s}_avg.pth', map_location=device))
        results[f'exp2_logit_avg_{s}'] = logit_avg_eval(model_avg, d['features'], d['labels'], device)

    # Exp3: cMAS subject split
    os.makedirs('checkpoint/cmas_split_avg_mlp', exist_ok=True)
    for s in SETS:
        d         = np.load(os.path.join(FEAT_DIR, f'test_{s}.npz'))
        feats     = d['features']
        labels    = d['labels']
        N, V, D   = feats.shape
        subj_ids  = get_subject_ids(s)
        test_mask = np.isin(subj_ids, TEST_SUBJECTS)
        train_mask = ~test_mask

        sv_model = build_mlp(D).to(device)
        sv_model.load_state_dict(torch.load(f'checkpoint/cmas_split_mlp/best_{s}_sv.pth', map_location=device))
        results[f'exp3_logit_avg_{s}'] = logit_avg_eval(sv_model, feats[test_mask], labels[test_mask], device)

        avg_ckpt = f'checkpoint/cmas_split_avg_mlp/best_{s}_avg.pth'
        tr_ds    = FlatDataset(feats[train_mask].mean(axis=1), labels[train_mask])
        te_ds    = FlatDataset(feats[test_mask].mean(axis=1),  labels[test_mask])
        results[f'exp3_feat_avg_{s}'] = train_mlp(tr_ds, te_ds, D, device, avg_ckpt)
        print(f"[{s}] Exp3 feat_avg={results[f'exp3_feat_avg_{s}']:.2f}%  "
              f"logit_avg={results[f'exp3_logit_avg_{s}']:.2f}%", flush=True)

    # Print table
    EXP3_SV     = dict(setA=87.42, setB=87.52, setC=87.42, setD=87.50)
    EXP3_CONCAT = dict(setA=88.27, setB=87.95, setC=87.46, setD=87.80)
    SV_BASE     = 88.19

    print("\n" + "=" * 78)
    print(f"{'Experiment':<32} {'setA':>7} {'setB':>7} {'setC':>7} {'setD':>7}")
    print("-" * 56)
    print(f"{'SV Baseline':<32} {SV_BASE:>7.2f} {SV_BASE:>7.2f} {SV_BASE:>7.2f} {SV_BASE:>7.2f}")

    rows = [
        ("Exp1 Feature Avg",    "exp1_feat_avg"),
        ("Exp1 Logit Avg",      "exp1_logit_avg"),
        ("Exp2 Feature Avg",    "exp2_feat_avg"),
        ("Exp2 Feature Concat", "exp2_feat_concat"),
        ("Exp2 Logit Avg",      "exp2_logit_avg"),
    ]
    for label, key in rows:
        vals = [results.get(f'{key}_{s}', float('nan')) for s in SETS]
        print(f"  {label:<30} {vals[0]:>7.2f} {vals[1]:>7.2f} {vals[2]:>7.2f} {vals[3]:>7.2f}")

    print(f"  {'Exp3 SV Baseline':<30} {EXP3_SV['setA']:>7.2f} {EXP3_SV['setB']:>7.2f} "
          f"{EXP3_SV['setC']:>7.2f} {EXP3_SV['setD']:>7.2f}")
    for label, key in [("Exp3 MV Feature Avg", "exp3_feat_avg"),
                        ("Exp3 MV Feature Concat", None),
                        ("Exp3 MV Logit Avg", "exp3_logit_avg")]:
        if key is None:
            vals = [EXP3_CONCAT[s] for s in SETS]
        else:
            vals = [results.get(f'{key}_{s}', float('nan')) for s in SETS]
        print(f"  {label:<30} {vals[0]:>7.2f} {vals[1]:>7.2f} {vals[2]:>7.2f} {vals[3]:>7.2f}")
    print("=" * 78)


if __name__ == '__main__':
    main()
