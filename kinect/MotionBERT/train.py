"""
Multi-view fusion MLP training for NTU60 action recognition.

Modes (--mode):
  logit   : Extract GT logits (if needed) → train logit-level Avg/Concat MLPs
  feat    : Extract GT features (if needed) → train feature-level Avg/Concat MLPs
  mixed   : Extract c-MAS train logits/features → train mixed GT-0°+c-MAS MLPs

Usage:
    python train.py --mode logit
    python train.py --mode feat
    python train.py --mode mixed
"""

import os, pickle, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import ActionNet

# ── Paths ─────────────────────────────────────────────────────────────────
MB_CONFIG  = "configs/action/MB_ft_NTU60_xsub.yaml"
MB_WEIGHTS = "save/MB_ft_NTU60_xsub/best_epoch.bin"
DATA_DIR   = "data/action"
LOGIT_DIR  = "view_attn_out/logits"
FEAT_DIR   = "view_attn_out/features"
CKPT_LOGIT = "view_attn_out/checkpoints_logit_combo"
CKPT_FEAT  = "view_attn_out/checkpoints_feat"
CKPT_MIXED = "view_attn_out/checkpoints_mixed"

# ── Constants ─────────────────────────────────────────────────────────────
VIEW_ANGLES = [0, 30, -30, 45, -45, 90, -90]
ANGLE_TAGS  = {0:'theta_000', 30:'theta_p030', -30:'theta_n030',
               45:'theta_p045', -45:'theta_n045', 90:'theta_p090', -90:'theta_n090'}
N_CLASSES = 60; LOGIT_DIM = 60; FEAT_DIM = 2048

COMBOS = {
    1: [0],
    2: [0, 30, -30],
    3: [0, 45, -45],
    4: [0, 90, -90],
    8: [0, 30, -30, 45, -45, 90, -90],
}
COMBO_LABELS = {1:"0°", 2:"0°,±30°", 3:"0°,±45°", 4:"0°,±90°", 8:"0°,±30°,±45°,±90°"}

# mixed mode combos use view indices instead of angles
MIXED_COMBOS = {
    1: [0],
    2: [0, 1, 2],
    3: [0, 3, 4],
    4: [0, 5, 6],
    8: [0, 1, 2, 3, 4, 5, 6],
}
MIXED_LABELS = {
    1:"GT 0°", 2:"GT 0° + cMAS ±30°", 3:"GT 0° + cMAS ±45°",
    4:"GT 0° + cMAS ±90°", 8:"GT 0° + cMAS all",
}

# ── Training hyperparameters (change here) ────────────────────────────────
LR           = 1e-4
WD           = 1e-4
EPOCHS       = 100    # 150 for mixed mode
BATCH        = 256    # 128 for mixed mode
LABEL_SMOOTH = 0.05
BATCH_MB     = 64     # MotionBERT inference batch


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
        self.backbone = base.backbone
        self.feat_J   = base.feat_J
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


def load_motionbert():
    args  = get_config(MB_CONFIG)
    bb    = load_backbone(args)
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


def extract_one_view(model, mb_args, pkl_path, split, batch=64):
    dataset = NTURGBD(data_path=pkl_path, data_split=split,
                      n_frames=mb_args.clip_len, random_move=False,
                      scale_range=mb_args.scale_range_test)
    loader  = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    outs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            if torch.cuda.is_available(): x = x.cuda()
            outs.append(model(x).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(outs), np.concatenate(labels)


def train_model(model, X_tr, y_tr, X_va, y_va, name, ckpt_dir, device,
                epochs=EPOCHS, batch=BATCH):
    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch, shuffle=False)
    model = model.to(device)
    crit  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc, best_f1, best_ep = 0., 0., 0

    for ep in range(1, epochs + 1):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(X), y)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        sched.step()

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for X, y in va_loader:
                preds_all.append(model(X.to(device)).argmax(1).cpu())
                labels_all.append(y)
        preds  = torch.cat(preds_all).numpy()
        labels = torch.cat(labels_all).numpy()
        acc = (preds == labels).mean() * 100.
        f1  = f1_score(labels, preds, average='macro') * 100.

        if acc > best_acc:
            best_acc, best_f1, best_ep = acc, f1, ep
            torch.save({'state_dict': model.state_dict(), 'val_acc': acc, 'val_f1': f1},
                       os.path.join(ckpt_dir, f'{name}.pth'))

        if ep % 30 == 0 or ep == 1:
            print(f'  [{name}] Ep {ep:3d}  val_acc={acc:.2f}%  f1={f1:.2f}%')

    print(f'  [{name}] Best: {best_acc:.2f}%  F1={best_f1:.2f}%  @ ep {best_ep}')
    return best_acc, best_f1


# ── Extract GT logits + features ──────────────────────────────────────────
def extract_gt(device):
    need_logit = not (os.path.exists(f'{LOGIT_DIR}/logits_train.npz') and
                      os.path.exists(f'{LOGIT_DIR}/logits_val.npz'))
    need_feat  = not (os.path.exists(f'{FEAT_DIR}/features_train.npz') and
                      os.path.exists(f'{FEAT_DIR}/features_val.npz'))
    if not need_logit and not need_feat:
        print("GT logits/features already exist, skipping extraction.")
        return

    print("Loading MotionBERT ...")
    mb_model, mb_args = load_motionbert()
    base = mb_model.module if hasattr(mb_model, 'module') else mb_model
    feat_extractor = FeatureExtractor(base)
    if torch.cuda.is_available():
        feat_extractor = nn.DataParallel(feat_extractor).cuda()
    feat_extractor.eval()

    for split, logit_out, feat_out in [
        ('xsub_train', f'{LOGIT_DIR}/logits_train.npz',  f'{FEAT_DIR}/features_train.npz'),
        ('xsub_val',   f'{LOGIT_DIR}/logits_val.npz',    f'{FEAT_DIR}/features_val.npz'),
    ]:
        if os.path.exists(logit_out) and os.path.exists(feat_out):
            print(f"{split}: already done, skipping.")
            continue
        print(f"\n--- {split} ---")
        all_logits, all_feats, labels = [], [], None
        t0 = time.time()
        for deg in VIEW_ANGLES:
            tag = ANGLE_TAGS[deg]
            pkl = f"{DATA_DIR}/ntu60_gt_{tag}.pkl"
            print(f"  {deg:+4d}°  logits ...", end=' ', flush=True)
            logits, lbl = extract_one_view(mb_model, mb_args, pkl, split, BATCH_MB)
            print(f"{logits.shape}  feats ...", end=' ', flush=True)
            feats, _    = extract_one_view(feat_extractor, mb_args, pkl, split, BATCH_MB)
            print(f"{feats.shape}")
            all_logits.append(logits)
            all_feats.append(feats)
            if labels is None: labels = lbl
            else: assert (labels == lbl).all()

        np.savez(logit_out, logits=np.stack(all_logits, axis=1), labels=labels)
        np.savez(feat_out,  features=np.stack(all_feats, axis=1), labels=labels)
        print(f"  Saved  ({time.time()-t0:.0f}s)")


# ── Mode: logit ────────────────────────────────────────────────────────────
def run_logit(device):
    os.makedirs(LOGIT_DIR, exist_ok=True)
    os.makedirs(CKPT_LOGIT, exist_ok=True)
    extract_gt(device)

    tr = np.load(f'{LOGIT_DIR}/logits_train.npz')
    va = np.load(f'{LOGIT_DIR}/logits_val.npz')
    L_tr = torch.FloatTensor(tr['logits'])
    y_tr = torch.LongTensor(tr['labels'])
    L_va = torch.FloatTensor(va['logits'])
    y_va = torch.LongTensor(va['labels'])
    L_tr_z = zscore(L_tr); L_va_z = zscore(L_va)
    print(f'Logits — Train: {L_tr.shape}  Val: {L_va.shape}')

    results = {}
    for cid, degs in COMBOS.items():
        idx = [VIEW_ANGLES.index(d) for d in degs]
        nv  = len(idx)
        lbl = COMBO_LABELS[cid]

        # Raw argmax
        raw_acc = (L_va[:, idx, :].mean(1).numpy().argmax(1) == y_va.numpy()).mean() * 100.
        results[f'C{cid} Raw-Avg [{lbl}]'] = (raw_acc, None)

        print(f'\n--- C{cid} [{lbl}]  Logit-Avg ---')
        acc, f1 = train_model(MLP(LOGIT_DIM),
                              L_tr_z[:, idx, :].mean(1), y_tr,
                              L_va_z[:, idx, :].mean(1), y_va,
                              f'logit_c{cid}_avg', CKPT_LOGIT, device)
        results[f'C{cid} Logit-Avg [{lbl}]'] = (acc, f1)

        print(f'\n--- C{cid} [{lbl}]  Logit-Concat ---')
        acc, f1 = train_model(MLP(nv * LOGIT_DIM),
                              L_tr_z[:, idx, :].reshape(len(L_tr), -1), y_tr,
                              L_va_z[:, idx, :].reshape(len(L_va), -1), y_va,
                              f'logit_c{cid}_concat', CKPT_LOGIT, device)
        results[f'C{cid} Logit-Concat [{lbl}]'] = (acc, f1)

    _print_table(results, "Logit-level Fusion Results")


# ── Mode: feat ────────────────────────────────────────────────────────────
def run_feat(device):
    os.makedirs(FEAT_DIR, exist_ok=True)
    os.makedirs(CKPT_FEAT, exist_ok=True)
    extract_gt(device)

    tr = np.load(f'{FEAT_DIR}/features_train.npz')
    va = np.load(f'{FEAT_DIR}/features_val.npz')
    F_tr = torch.FloatTensor(tr['features'])
    y_tr = torch.LongTensor(tr['labels'])
    F_va = torch.FloatTensor(va['features'])
    y_va = torch.LongTensor(va['labels'])
    print(f'Features — Train: {F_tr.shape}  Val: {F_va.shape}')

    results = {}
    for cid, degs in COMBOS.items():
        idx = [VIEW_ANGLES.index(d) for d in degs]
        nv  = len(idx)
        lbl = COMBO_LABELS[cid]

        print(f'\n--- C{cid} [{lbl}]  Feat-Avg ---')
        acc, f1 = train_model(FusionMLP(FEAT_DIM),
                              F_tr[:, idx, :].mean(1), y_tr,
                              F_va[:, idx, :].mean(1), y_va,
                              f'c{cid}_avg', CKPT_FEAT, device)
        results[f'C{cid} Feat-Avg [{lbl}]'] = (acc, f1)

        print(f'\n--- C{cid} [{lbl}]  Feat-Concat ---')
        acc, f1 = train_model(FusionMLP(nv * FEAT_DIM),
                              F_tr[:, idx, :].reshape(len(F_tr), -1), y_tr,
                              F_va[:, idx, :].reshape(len(F_va), -1), y_va,
                              f'c{cid}_concat', CKPT_FEAT, device)
        results[f'C{cid} Feat-Concat [{lbl}]'] = (acc, f1)

    _print_table(results, "Feature-level Fusion Results")


# ── Mode: mixed ────────────────────────────────────────────────────────────
def _extract_cmas_train(mb_model, feat_extractor, mb_args):
    out_l = f'{LOGIT_DIR}/logits_cmas_train.npz'
    out_f = f'{FEAT_DIR}/features_cmas_train.npz'
    if os.path.exists(out_l) and os.path.exists(out_f):
        print("c-MAS train logits/features already extracted.")
        return
    print("Extracting c-MAS train logits/features ...")
    all_logits, all_feats, labels = [], [], None
    for deg in VIEW_ANGLES:
        tag = ANGLE_TAGS[deg]
        pkl = f"{DATA_DIR}/ntu60_cmas_train_{tag}.pkl"
        print(f"  {deg:+4d}°  logits ...", end=' ', flush=True)
        logits, lbl = extract_one_view(mb_model, mb_args, pkl, 'xsub_train', BATCH_MB)
        print(f"{logits.shape}  feats ...", end=' ', flush=True)
        feats, _    = extract_one_view(feat_extractor, mb_args, pkl, 'xsub_train', BATCH_MB)
        print(f"{feats.shape}")
        all_logits.append(logits); all_feats.append(feats)
        if labels is None: labels = lbl
    np.savez(out_l, logits=np.stack(all_logits, axis=1), labels=labels)
    np.savez(out_f, features=np.stack(all_feats, axis=1), labels=labels)
    print(f"Saved c-MAS train  logits {np.stack(all_logits,1).shape}")


def run_mixed(device):
    os.makedirs(LOGIT_DIR, exist_ok=True)
    os.makedirs(FEAT_DIR,  exist_ok=True)
    os.makedirs(CKPT_MIXED, exist_ok=True)
    extract_gt(device)

    print("Loading MotionBERT for c-MAS extraction ...")
    mb_model, mb_args = load_motionbert()
    base = mb_model.module if hasattr(mb_model, 'module') else mb_model
    feat_extractor = FeatureExtractor(base)
    if torch.cuda.is_available():
        feat_extractor = nn.DataParallel(feat_extractor).cuda()
    feat_extractor.eval()
    _extract_cmas_train(mb_model, feat_extractor, mb_args)

    # ── Load all data ──────────────────────────────────────────────────────
    gt_tr_l = torch.FloatTensor(np.load(f'{LOGIT_DIR}/logits_train.npz')['logits'])
    gt_tr_f = torch.FloatTensor(np.load(f'{FEAT_DIR}/features_train.npz')['features'])
    gt_va_l = torch.FloatTensor(np.load(f'{LOGIT_DIR}/logits_val.npz')['logits'])
    gt_va_f = torch.FloatTensor(np.load(f'{FEAT_DIR}/features_val.npz')['features'])
    cm_tr_l = torch.FloatTensor(np.load(f'{LOGIT_DIR}/logits_cmas_train.npz')['logits'])
    cm_tr_f = torch.FloatTensor(np.load(f'{FEAT_DIR}/features_cmas_train.npz')['features'])
    cm_va_l = torch.FloatTensor(np.load(f'{LOGIT_DIR}/logits_cmas_val.npz')['logits'])
    cm_va_f = torch.FloatTensor(np.load(f'{FEAT_DIR}/features_cmas_val.npz')['features'])
    y_tr_cm = torch.LongTensor(np.load(f'{LOGIT_DIR}/logits_cmas_train.npz')['labels'])
    y_va_cm = torch.LongTensor(np.load(f'{LOGIT_DIR}/logits_cmas_val.npz')['labels'])

    # ── Match frame_dirs ───────────────────────────────────────────────────
    with open(f'{DATA_DIR}/ntu60_gt_theta_000.pkl', 'rb') as f:
        gt_pkl = pickle.load(f)
    with open(f'{DATA_DIR}/ntu60_cmas_train_theta_000.pkl', 'rb') as f:
        cm_tr_pkl = pickle.load(f)
    with open(f'{DATA_DIR}/ntu60_cmas_theta_000.pkl', 'rb') as f:
        cm_va_pkl = pickle.load(f)

    def match(cm_fds, gt_fds):
        gt_map = {fd: i for i, fd in enumerate(gt_fds)}
        gi = np.array([gt_map[fd] for fd in cm_fds if fd in gt_map])
        ci = np.array([i for i, fd in enumerate(cm_fds) if fd in gt_map])
        return gi, ci

    tr_gi, tr_ci = match([a['frame_dir'] for a in cm_tr_pkl['annotations']],
                          gt_pkl['split']['xsub_train'])
    va_gi, va_ci = match([a['frame_dir'] for a in cm_va_pkl['annotations']],
                          gt_pkl['split']['xsub_val'])
    print(f"Matched — train: {len(tr_gi)}, val: {len(va_gi)}")

    # Build mixed tensors (view 0 = GT, rest = c-MAS)
    def make_mixed(cm_l, cm_f, gt_l, gt_f, ci, gi):
        ML = cm_l[ci].clone(); ML[:, 0, :] = gt_l[gi, 0, :]
        MF = cm_f[ci].clone(); MF[:, 0, :] = gt_f[gi, 0, :]
        return ML, MF

    ML_tr, MF_tr = make_mixed(cm_tr_l, cm_tr_f, gt_tr_l, gt_tr_f, tr_ci, tr_gi)
    ML_va, MF_va = make_mixed(cm_va_l, cm_va_f, gt_va_l, gt_va_f, va_ci, va_gi)
    y_tr = y_tr_cm[tr_ci]; y_va = y_va_cm[va_ci]
    ML_tr_z = zscore(ML_tr); ML_va_z = zscore(ML_va)
    N_tr, N_va = len(y_tr), len(y_va)
    print(f"Train: {N_tr}  Val: {N_va}")

    results = {}
    for cid, view_idx in MIXED_COMBOS.items():
        nv  = len(view_idx)
        lbl = MIXED_LABELS[cid]
        print(f"\n{'='*55}\n  C{cid}: {lbl}\n{'='*55}")

        for name, X_tr, X_va, ModelCls, in_dim in [
            ('Logit-Avg',    ML_tr_z[:, view_idx, :].mean(1),         ML_va_z[:, view_idx, :].mean(1),         MLP,       LOGIT_DIM),
            ('Logit-Concat', ML_tr_z[:, view_idx, :].reshape(N_tr,-1),ML_va_z[:, view_idx, :].reshape(N_va,-1),MLP,       nv*LOGIT_DIM),
            ('Feat-Avg',     MF_tr[:, view_idx, :].mean(1),           MF_va[:, view_idx, :].mean(1),           FusionMLP, FEAT_DIM),
            ('Feat-Concat',  MF_tr[:, view_idx, :].reshape(N_tr,-1),  MF_va[:, view_idx, :].reshape(N_va,-1),  FusionMLP, nv*FEAT_DIM),
        ]:
            acc, f1 = train_model(ModelCls(in_dim), X_tr, y_tr, X_va, y_va,
                                  f'c{cid}_{name.lower().replace("-","_")}',
                                  CKPT_MIXED, device, epochs=150, batch=128)
            results[f'C{cid} {name}'] = (acc, f1)

    _print_table(results, "Mixed GT+c-MAS Fusion Results")


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
    parser.add_argument('--mode', required=True, choices=['logit', 'feat', 'mixed'],
                        help='logit | feat | mixed')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Mode: {args.mode}  Device: {device}")

    {'logit': run_logit, 'feat': run_feat, 'mixed': run_mixed}[args.mode](device)
    print('\nDone.')


if __name__ == '__main__':
    main()
