"""
Multi-view fusion evaluation for NTU60 action recognition.

Modes (--mode):
  gt_single  : GT single-view baseline (eval_gt_baseline)
  gt_combo   : GT multi-view logit/feature fusion combos (eval_multiview_combo)
  cmas       : c-MAS zero-shot evaluation — all GT-trained models applied (eval_cmas_full)
  gt0_cmas   : GT 0° + c-MAS other views — hypothesis test (eval_gt0_plus_cmas)

Usage:
    python eval.py --mode gt_single
    python eval.py --mode gt_combo
    python eval.py --mode cmas
    python eval.py --mode gt0_cmas
    python eval.py --mode gt_combo --combos 1 2 3 4 8   # specific combos only
"""

import os, sys, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MB_ROOT  = os.path.normpath(os.path.join(_THIS_DIR, '..', '..', 'models', 'MotionBERT'))
sys.path.insert(0, _MB_ROOT)

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import ActionNet

# ── Paths ─────────────────────────────────────────────────────────────────
MB_CONFIG  = os.path.join(_MB_ROOT, "configs/action/MB_ft_NTU60_xsub.yaml")
MB_WEIGHTS = os.path.join(_MB_ROOT, "save/MB_ft_NTU60_xsub/best_epoch.bin")
DATA_DIR   = "data/action"
LOGIT_DIR  = "view_attn_out/logits"
FEAT_DIR   = "view_attn_out/features"
CKPT_LOGIT = "view_attn_out/checkpoints_logit_combo"
CKPT_FEAT  = "view_attn_out/checkpoints_feat"

# ── Constants ─────────────────────────────────────────────────────────────
VIEW_ANGLES = [0, 30, -30, 45, -45, 90, -90]
ANGLE_TAGS  = {0:'theta_000', 30:'theta_p030', -30:'theta_n030',
               45:'theta_p045', -45:'theta_n045', 90:'theta_p090', -90:'theta_n090'}
N_CLASSES = 60; LOGIT_DIM = 60; FEAT_DIM = 2048; BATCH = 128

COMBOS = {
    1: [0],
    2: [0, 30, -30],
    3: [0, 45, -45],
    4: [0, 90, -90],
    8: [0, 30, -30, 45, -45, 90, -90],
}
COMBO_LABELS = {1:"0°", 2:"0°,±30°", 3:"0°,±45°", 4:"0°,±90°", 8:"0°,±30°,±45°,±90°"}


# ── Models ────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, mid=128, n=N_CLASSES, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid), nn.LayerNorm(mid), nn.GELU(),
            nn.Dropout(drop), nn.Linear(mid, n))
    def forward(self, x): return self.net(x)


class FusionMLP(nn.Module):
    def __init__(self, in_dim, n=N_CLASSES, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(drop), nn.Linear(512, n))
    def forward(self, x): return self.net(x)


class FeatureExtractor(nn.Module):
    def __init__(self, action_net):
        super().__init__()
        base = action_net.module if hasattr(action_net, 'module') else action_net
        self.backbone = base.backbone; self.feat_J = base.feat_J
        head = base.head
        self.dropout = head.dropout; self.fc1 = head.fc1
        self.bn = head.bn; self.relu = head.relu

    def forward(self, x):
        N, M, T, J, C = x.shape
        x    = x.reshape(N*M, T, J, C)
        feat = self.backbone.get_representation(x)
        feat = feat.reshape([N, M, T, self.feat_J, -1])
        feat = self.dropout(feat).permute(0, 1, 3, 4, 2).mean(dim=-1)
        feat = feat.reshape(N, M, -1).mean(dim=1)
        return self.relu(self.bn(self.fc1(feat)))


# ── Utilities ─────────────────────────────────────────────────────────────
def zscore(X):
    m = X.mean(dim=-1, keepdim=True)
    s = X.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (X - m) / s


def load_ckpt(model, path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    key  = 'state_dict' if 'state_dict' in ckpt else 'model'
    model.load_state_dict(ckpt[key])
    return model.to(device).eval()


def evaluate(model, X, y, device):
    model.eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=False)
    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds_all.append(model(xb.to(device)).argmax(1).cpu())
            labels_all.append(yb)
    preds  = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()
    return (preds == labels).mean() * 100., f1_score(labels, preds, average='macro') * 100.


def build_action_model(split='xsub'):
    args = get_config(MB_CONFIG)
    bb   = load_backbone(args)
    model = ActionNet(backbone=bb, dim_rep=args.dim_rep,
                      num_classes=args.action_classes, dropout_ratio=args.dropout_ratio,
                      version=args.model_version, hidden_dim=args.hidden_dim,
                      num_joints=args.num_joints)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    ckpt = torch.load(MB_WEIGHTS, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model, args


def infer_pkl(model, mb_args, pkl_path, split, batch=64):
    dataset = NTURGBD(data_path=pkl_path, data_split=split,
                      n_frames=mb_args.clip_len, random_move=False,
                      scale_range=mb_args.scale_range_test)
    loader  = DataLoader(dataset, batch_size=batch, shuffle=False,
                         num_workers=4, pin_memory=True)
    logits_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            if torch.cuda.is_available(): x = x.cuda()
            logits_all.append(model(x).cpu().numpy())
            labels_all.append(y.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


# ── Mode: gt_single ────────────────────────────────────────────────────────
def run_gt_single(split='xsub_val', batch=32):
    print(f"\n[gt_single]  split={split}")
    model, mb_args = build_action_model()
    pkl = f"{DATA_DIR}/ntu60_gt_coco17.pkl"
    dataset = NTURGBD(data_path=pkl, data_split=split,
                      n_frames=mb_args.clip_len, random_move=False,
                      scale_range=mb_args.scale_range_test)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False,
                        num_workers=4, pin_memory=True)

    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            if torch.cuda.is_available(): x = x.cuda()
            preds_all.append(model(x).argmax(1).cpu().numpy())
            labels_all.append(y.numpy())

    preds  = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    top1 = (preds == labels).mean() * 100.
    f1   = f1_score(labels, preds, average='macro') * 100.

    print(f"\n{'='*50}")
    print(f"  Split    : {split}  ({len(labels)} samples)")
    print(f"  Top-1 Acc: {top1:.2f}%")
    print(f"  Macro F1 : {f1:.2f}%")
    print('='*50)


# ── Mode: gt_combo ─────────────────────────────────────────────────────────
def run_gt_combo(combo_ids, split='xsub_val', pkl_prefix='ntu60_gt', batch=64):
    print(f"\n[gt_combo]  combos={combo_ids}  prefix={pkl_prefix}")
    model, mb_args = build_action_model()
    mb_args.batch  = batch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for cid in combo_ids:
        degs = COMBOS[cid]
        print(f"\n── C{cid}  [{COMBO_LABELS[cid]}] ──")
        logits_sum = None; labels = None

        for deg in degs:
            pkl = f"{DATA_DIR}/{pkl_prefix}_{ANGLE_TAGS[deg]}.pkl"
            print(f"   {deg:+4d}°  {pkl} ...", end=' ', flush=True)
            logits, lbl = infer_pkl(model, mb_args, pkl, split, batch)
            print(f"shape={logits.shape}")
            logits_sum = logits if logits_sum is None else logits_sum + logits
            if labels is None: labels = lbl
            else: assert (labels == lbl).all()

        preds = (logits_sum / len(degs)).argmax(1)
        top1  = (preds == labels).mean() * 100.
        f1    = f1_score(labels, preds, average='macro') * 100.
        results[cid] = (top1, f1)
        print(f"   Top-1: {top1:.2f}%  F1: {f1:.2f}%")

    print(f"\n{'='*58}")
    print(f"  {'C':<6} {'Views':<26} {'Top-1':>7} {'F1':>9}")
    print('='*58)
    for cid, (t, f) in results.items():
        print(f"  {cid:<6} {COMBO_LABELS[cid]:<26} {t:>6.2f}%  {f:>8.2f}%")
    print('='*58)


# ── Mode: cmas ─────────────────────────────────────────────────────────────
def _extract_cmas_val_feats(device):
    out = f'{FEAT_DIR}/features_cmas_val.npz'
    if os.path.exists(out):
        d = np.load(out)
        return torch.FloatTensor(d['features']), torch.LongTensor(d['labels'])

    print("Extracting c-MAS val features ...")
    model, mb_args = build_action_model()
    base = model.module if hasattr(model, 'module') else model
    extractor = FeatureExtractor(base)
    if torch.cuda.is_available():
        extractor = nn.DataParallel(extractor).cuda()
    extractor.eval()

    all_feats, labels = [], None
    for deg in VIEW_ANGLES:
        pkl = f"{DATA_DIR}/ntu60_cmas_{ANGLE_TAGS[deg]}.pkl"
        print(f"  {deg:+4d}° ...", end=' ', flush=True)
        dataset = NTURGBD(data_path=pkl, data_split='xsub_val',
                          n_frames=mb_args.clip_len, random_move=False,
                          scale_range=mb_args.scale_range_test)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        feats_v, labels_v = [], []
        with torch.no_grad():
            for x, y in loader:
                if torch.cuda.is_available(): x = x.cuda()
                feats_v.append(extractor(x).cpu().numpy())
                labels_v.append(y.numpy())
        feats_v  = np.concatenate(feats_v)
        labels_v = np.concatenate(labels_v)
        print(f"{feats_v.shape}")
        all_feats.append(feats_v)
        if labels is None: labels = labels_v

    stacked = np.stack(all_feats, axis=1)
    np.savez(out, features=stacked, labels=labels)
    return torch.FloatTensor(stacked), torch.LongTensor(labels)


def run_cmas(device):
    print("\n[cmas]  GT-trained models → zero-shot on c-MAS val (480 samples)")
    d = np.load(f'{LOGIT_DIR}/logits_cmas_val.npz')
    L = torch.FloatTensor(d['logits'])   # (480, 7, 60)
    y = torch.LongTensor(d['labels'])
    L_z = zscore(L)

    F, y_check = _extract_cmas_val_feats(device)
    assert (y == y_check).all()

    results = {}
    for cid, degs in COMBOS.items():
        idx = [VIEW_ANGLES.index(d) for d in degs]
        nv  = len(idx); lbl = COMBO_LABELS[cid]

        raw_acc = (L[:, idx, :].mean(1).numpy().argmax(1) == y.numpy()).mean() * 100.
        results[f'C{cid} Raw-Avg [{lbl}]'] = (raw_acc, None)

        acc, f1 = evaluate(load_ckpt(MLP(LOGIT_DIM),      f'{CKPT_LOGIT}/logit_c{cid}_avg.pth',    device),
                           L_z[:, idx, :].mean(1), y, device)
        results[f'C{cid} Logit-Avg [{lbl}]'] = (acc, f1)

        acc, f1 = evaluate(load_ckpt(MLP(nv*LOGIT_DIM),   f'{CKPT_LOGIT}/logit_c{cid}_concat.pth', device),
                           L_z[:, idx, :].reshape(len(L),-1), y, device)
        results[f'C{cid} Logit-Concat [{lbl}]'] = (acc, f1)

        acc, f1 = evaluate(load_ckpt(FusionMLP(FEAT_DIM),    f'{CKPT_FEAT}/c{cid}_avg.pth',    device),
                           F[:, idx, :].mean(1), y, device)
        results[f'C{cid} Feat-Avg [{lbl}]'] = (acc, f1)

        acc, f1 = evaluate(load_ckpt(FusionMLP(nv*FEAT_DIM), f'{CKPT_FEAT}/c{cid}_concat.pth', device),
                           F[:, idx, :].reshape(len(F),-1), y, device)
        results[f'C{cid} Feat-Concat [{lbl}]'] = (acc, f1)

    _print_table(results, "c-MAS Zero-shot Evaluation (GT-trained models)")


# ── Mode: gt0_cmas ─────────────────────────────────────────────────────────
def run_gt0_cmas(device):
    print("\n[gt0_cmas]  GT 0° + c-MAS other views — hypothesis test")

    # Mixed combos: index 0=GT0°, rest=c-MAS views
    mixed_combos = {
        1: [0],
        2: [0, 1, 2],
        3: [0, 3, 4],
        4: [0, 5, 6],
        8: [0, 1, 2, 3, 4, 5, 6],
    }
    mixed_labels = {
        1:"GT 0°", 2:"GT 0°+cMAS±30°", 3:"GT 0°+cMAS±45°",
        4:"GT 0°+cMAS±90°", 8:"GT 0°+cMAS all",
    }

    # Match frame_dirs
    with open(f'{DATA_DIR}/ntu60_cmas_theta_000.pkl', 'rb') as f:
        cmas_pkl = pickle.load(f)
    with open(f'{DATA_DIR}/ntu60_gt_theta_000.pkl', 'rb') as f:
        gt_pkl = pickle.load(f)
    cmas_fds = [a['frame_dir'] for a in cmas_pkl['annotations']]
    gt_val_fds = gt_pkl['split']['xsub_val']
    gt_map = {fd: i for i, fd in enumerate(gt_val_fds)}
    gi = np.array([gt_map[fd] for fd in cmas_fds if fd in gt_map])
    ci = np.array([i for i, fd in enumerate(cmas_fds) if fd in gt_map])
    N  = len(gi); print(f"Matched: {N} samples")

    # Load logits + features
    gt_l  = torch.FloatTensor(np.load(f'{LOGIT_DIR}/logits_val.npz')['logits'])
    gt_la = torch.LongTensor(np.load(f'{LOGIT_DIR}/logits_val.npz')['labels'])
    gt_f  = torch.FloatTensor(np.load(f'{FEAT_DIR}/features_val.npz')['features'])
    cm_l  = torch.FloatTensor(np.load(f'{LOGIT_DIR}/logits_cmas_val.npz')['logits'])
    cm_la = torch.LongTensor(np.load(f'{LOGIT_DIR}/logits_cmas_val.npz')['labels'])
    cm_f_data, _ = _extract_cmas_val_feats(device)

    assert (gt_la[gi] == cm_la[ci]).all(), "Label mismatch!"
    labels = torch.LongTensor(gt_la[gi].numpy())

    # Build mixed (view 0 = GT, rest = c-MAS)
    ML = cm_l[ci].clone(); ML[:, 0, :] = gt_l[gi, 0, :]
    MF = cm_f_data[ci].clone(); MF[:, 0, :] = gt_f[gi, 0, :]
    ML_z = zscore(ML)

    results = {}
    for cid, view_idx in mixed_combos.items():
        nv  = len(view_idx); lbl = mixed_labels[cid]

        if cid == 1:
            raw_preds = gt_l[gi, 0, :].numpy().argmax(1)
            l_avg = zscore(gt_l[gi, 0:1, :]).squeeze(1)
            l_cat = l_avg
            f_avg = gt_f[gi, 0, :]
            f_cat = f_avg
        else:
            raw_preds = ML[:, view_idx, :].mean(1).numpy().argmax(1)
            l_avg = ML_z[:, view_idx, :].mean(1)
            l_cat = ML_z[:, view_idx, :].reshape(N, -1)
            f_avg = MF[:, view_idx, :].mean(1)
            f_cat = MF[:, view_idx, :].reshape(N, -1)

        raw_acc = (raw_preds == gt_la[gi].numpy()).mean() * 100.
        results[f'C{cid} Raw-Avg [{lbl}]'] = (raw_acc, None)

        acc, f1 = evaluate(load_ckpt(MLP(LOGIT_DIM),      f'{CKPT_LOGIT}/logit_c{cid}_avg.pth',    device), l_avg, labels, device)
        results[f'C{cid} Logit-Avg [{lbl}]'] = (acc, f1)

        acc, f1 = evaluate(load_ckpt(MLP(nv*LOGIT_DIM),   f'{CKPT_LOGIT}/logit_c{cid}_concat.pth', device), l_cat, labels, device)
        results[f'C{cid} Logit-Concat [{lbl}]'] = (acc, f1)

        acc, f1 = evaluate(load_ckpt(FusionMLP(FEAT_DIM),    f'{CKPT_FEAT}/c{cid}_avg.pth',    device), f_avg, labels, device)
        results[f'C{cid} Feat-Avg [{lbl}]'] = (acc, f1)

        acc, f1 = evaluate(load_ckpt(FusionMLP(nv*FEAT_DIM), f'{CKPT_FEAT}/c{cid}_concat.pth', device), f_cat, labels, device)
        results[f'C{cid} Feat-Concat [{lbl}]'] = (acc, f1)

    _print_table(results, f"GT 0° + c-MAS Views  (N={N})")


# ── Summary table ─────────────────────────────────────────────────────────
def _print_table(results, title):
    print(f'\n{"="*65}\n  {title}\n{"="*65}')
    print(f"  {'Method':<42} {'Top-1':>8}  {'Macro F1':>9}")
    print('='*65)
    for name, (acc, f1) in results.items():
        f1s = f'{f1:.2f}%' if f1 else '    —  '
        print(f'  {name:<42} {acc:>7.2f}%  {f1s:>9}')
    print('='*65)


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,
                        choices=['gt_single', 'gt_combo', 'cmas', 'gt0_cmas'])
    parser.add_argument('--combos', type=int, nargs='+', default=[1, 2, 3, 4, 8],
                        help='Combo IDs to evaluate (gt_combo mode only)')
    parser.add_argument('--split',      default='xsub_val')
    parser.add_argument('--pkl_prefix', default='ntu60_gt',
                        help='pkl prefix: ntu60_gt or ntu60_cmas (gt_combo mode)')
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Mode: {args.mode}  Device: {device}")

    if args.mode == 'gt_single':
        run_gt_single(split=args.split, batch=args.batch)
    elif args.mode == 'gt_combo':
        run_gt_combo(args.combos, split=args.split,
                     pkl_prefix=args.pkl_prefix, batch=args.batch)
    elif args.mode == 'cmas':
        run_cmas(device)
    elif args.mode == 'gt0_cmas':
        run_gt0_cmas(device)


if __name__ == '__main__':
    main()
