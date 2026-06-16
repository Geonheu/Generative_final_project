"""
4개 표 전체 재현 스크립트.

실험 2 GT  : logits_val.npz + features_val.npz + checkpoints_logit_combo/ + checkpoints_feat/
실험 2 cMAS: logits_cmas_val.npz + features_cmas_val.npz + 동일 체크포인트 (zero-shot)
추가실험 1 : GT 0° logit/feat + c-MAS 나머지 뷰 → GT-trained MLP (zero-shot)
추가실험 2 : GT 0° + c-MAS → checkpoints_mixed/

Usage:
    cd MotionBERT
    python run_full_eval.py
"""

import os, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

LOGIT_DIR  = "view_attn_out/logits"
FEAT_DIR   = "view_attn_out/features"
CKPT_LOGIT = "view_attn_out/checkpoints_logit_combo"
CKPT_FEAT  = "view_attn_out/checkpoints_feat"
CKPT_MIXED = "view_attn_out/checkpoints_mixed"
DATA_DIR   = "data/action"

VIEW_ANGLES = [0, 30, -30, 45, -45, 90, -90]
N_CLASSES = 60; LOGIT_DIM = 60; FEAT_DIM = 2048; BATCH = 256

COMBOS = {1:[0], 2:[0,30,-30], 3:[0,45,-45], 4:[0,90,-90],
          8:[0,30,-30,45,-45,90,-90]}
COMBO_LABELS = {1:"0°", 2:"0°,±30°", 3:"0°,±45°", 4:"0°,±90°", 8:"All 7"}

MIXED_COMBOS = {1:[0], 2:[0,1,2], 3:[0,3,4], 4:[0,5,6], 8:[0,1,2,3,4,5,6]}
MIXED_LABELS = {1:"GT 0°", 2:"GT 0°+cMAS±30°", 3:"GT 0°+cMAS±45°",
                4:"GT 0°+cMAS±90°", 8:"GT 0°+cMAS all"}


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


# ── Utilities ─────────────────────────────────────────────────────────────
def zscore(X):
    m = X.mean(dim=-1, keepdim=True)
    s = X.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (X - m) / s

def load_ckpt(model, path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    key  = "state_dict" if "state_dict" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    return model.to(device).eval()

def evaluate(model, X, y, device):
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=False)
    preds_all = []
    with torch.no_grad():
        for xb, _ in loader:
            preds_all.append(model(xb.to(device)).argmax(1).cpu())
    preds  = torch.cat(preds_all).numpy()
    labels = y.numpy()
    acc = (preds == labels).mean() * 100.
    f1  = f1_score(labels, preds, average="macro") * 100.
    return acc, f1

def print_table(title, rows, methods=("Raw Avg","Logit-Avg","Logit-Concat","Feat-Avg","Feat-Concat")):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    hdr = f"  {'Combo':<22}" + "".join(f"{m:>14}" for m in methods)
    print(hdr)
    print("-"*90)
    for lbl, vals in rows:
        row = f"  {lbl:<22}" + "".join(f"{v:>13.2f}%" for v in vals)
        print(row)
    print("="*90)


# ── 실험 2: GT 기반 ───────────────────────────────────────────────────────
def eval_gt(device):
    L = torch.FloatTensor(np.load(f"{LOGIT_DIR}/logits_val.npz")["logits"])
    y = torch.LongTensor(np.load(f"{LOGIT_DIR}/logits_val.npz")["labels"])
    F = torch.FloatTensor(np.load(f"{FEAT_DIR}/features_val.npz")["features"])
    L_z = zscore(L)
    print(f"GT val  L={L.shape}  F={F.shape}  N={len(y)}")

    rows = []
    for cid, degs in COMBOS.items():
        idx = [VIEW_ANGLES.index(d) for d in degs]; nv = len(idx)
        lbl = f"C{cid} [{COMBO_LABELS[cid]}]"

        raw = (L[:, idx, :].mean(1).numpy().argmax(1) == y.numpy()).mean() * 100.

        m = load_ckpt(MLP(LOGIT_DIM),      f"{CKPT_LOGIT}/logit_c{cid}_avg.pth",    device)
        la, _ = evaluate(m, L_z[:, idx, :].mean(1), y, device)

        m = load_ckpt(MLP(nv*LOGIT_DIM),   f"{CKPT_LOGIT}/logit_c{cid}_concat.pth", device)
        lc, _ = evaluate(m, L_z[:, idx, :].reshape(len(L), -1), y, device)

        m = load_ckpt(FusionMLP(FEAT_DIM),    f"{CKPT_FEAT}/c{cid}_avg.pth",    device)
        fa, _ = evaluate(m, F[:, idx, :].mean(1), y, device)

        m = load_ckpt(FusionMLP(nv*FEAT_DIM), f"{CKPT_FEAT}/c{cid}_concat.pth", device)
        fc, _ = evaluate(m, F[:, idx, :].reshape(len(F), -1), y, device)

        rows.append((lbl, [raw, la, lc, fa, fc]))
    print_table("실험 2 — GT 데이터 기반 (xsub_val 16,487 샘플)", rows)
    return rows


# ── 실험 2: c-MAS 기반 (zero-shot) ───────────────────────────────────────
def eval_cmas(device):
    L = torch.FloatTensor(np.load(f"{LOGIT_DIR}/logits_cmas_val.npz")["logits"])
    y = torch.LongTensor(np.load(f"{LOGIT_DIR}/logits_cmas_val.npz")["labels"])
    F = torch.FloatTensor(np.load(f"{FEAT_DIR}/features_cmas_val.npz")["features"])
    L_z = zscore(L)
    print(f"\nc-MAS val  L={L.shape}  F={F.shape}  N={len(y)}")

    rows = []
    for cid, degs in COMBOS.items():
        idx = [VIEW_ANGLES.index(d) for d in degs]; nv = len(idx)
        lbl = f"C{cid} [{COMBO_LABELS[cid]}]"

        raw = (L[:, idx, :].mean(1).numpy().argmax(1) == y.numpy()).mean() * 100.

        m = load_ckpt(MLP(LOGIT_DIM),      f"{CKPT_LOGIT}/logit_c{cid}_avg.pth",    device)
        la, _ = evaluate(m, L_z[:, idx, :].mean(1), y, device)

        m = load_ckpt(MLP(nv*LOGIT_DIM),   f"{CKPT_LOGIT}/logit_c{cid}_concat.pth", device)
        lc, _ = evaluate(m, L_z[:, idx, :].reshape(len(L), -1), y, device)

        m = load_ckpt(FusionMLP(FEAT_DIM),    f"{CKPT_FEAT}/c{cid}_avg.pth",    device)
        fa, _ = evaluate(m, F[:, idx, :].mean(1), y, device)

        m = load_ckpt(FusionMLP(nv*FEAT_DIM), f"{CKPT_FEAT}/c{cid}_concat.pth", device)
        fc, _ = evaluate(m, F[:, idx, :].reshape(len(F), -1), y, device)

        rows.append((lbl, [raw, la, lc, fa, fc]))
    print_table("실험 2 — c-MAS 데이터 기반 (xsub_val 480 샘플, zero-shot)", rows)
    return rows


# ── 추가실험 1: GT 0° + c-MAS views (GT-trained MLP, zero-shot) ──────────
def eval_gt0_cmas(device):
    gt_l = torch.FloatTensor(np.load(f"{LOGIT_DIR}/logits_val.npz")["logits"])
    gt_f = torch.FloatTensor(np.load(f"{FEAT_DIR}/features_val.npz")["features"])
    cm_l = torch.FloatTensor(np.load(f"{LOGIT_DIR}/logits_cmas_val.npz")["logits"])
    cm_f = torch.FloatTensor(np.load(f"{FEAT_DIR}/features_cmas_val.npz")["features"])
    gt_la = torch.LongTensor(np.load(f"{LOGIT_DIR}/logits_val.npz")["labels"])
    cm_la = torch.LongTensor(np.load(f"{LOGIT_DIR}/logits_cmas_val.npz")["labels"])

    # frame_dir 매칭
    with open(f"{DATA_DIR}/ntu60_cmas_theta_000.pkl", "rb") as f:
        cmas_pkl = pickle.load(f)
    with open(f"{DATA_DIR}/ntu60_gt_theta_000.pkl", "rb") as f:
        gt_pkl = pickle.load(f)
    cmas_fds  = [a["frame_dir"] for a in cmas_pkl["annotations"]]
    gt_val_fds = gt_pkl["split"]["xsub_val"]
    gt_map = {fd: i for i, fd in enumerate(gt_val_fds)}
    gi = np.array([gt_map[fd] for fd in cmas_fds if fd in gt_map])
    ci = np.array([i for i, fd in enumerate(cmas_fds) if fd in gt_map])
    N  = len(gi)
    print(f"\n추가실험 1: 매칭 샘플 {N}개")
    assert (gt_la[gi].numpy() == cm_la[ci].numpy()).all(), "Label mismatch!"

    # 혼합 텐서: view 0 = GT, 나머지 = c-MAS
    ML = cm_l[ci].clone(); ML[:, 0, :] = gt_l[gi, 0, :]
    MF = cm_f[ci].clone(); MF[:, 0, :] = gt_f[gi, 0, :]
    ML_z = zscore(ML)
    labels = torch.LongTensor(gt_la[gi].numpy())

    rows = []
    for cid, view_idx in MIXED_COMBOS.items():
        nv  = len(view_idx); lbl = f"C{cid} [{MIXED_LABELS[cid]}]"

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

        raw = (raw_preds == gt_la[gi].numpy()).mean() * 100.

        m = load_ckpt(MLP(LOGIT_DIM),         f"{CKPT_LOGIT}/logit_c{cid}_avg.pth",    device)
        la, _ = evaluate(m, l_avg, labels, device)

        m = load_ckpt(MLP(nv*LOGIT_DIM),      f"{CKPT_LOGIT}/logit_c{cid}_concat.pth", device)
        lc, _ = evaluate(m, l_cat, labels, device)

        m = load_ckpt(FusionMLP(FEAT_DIM),    f"{CKPT_FEAT}/c{cid}_avg.pth",    device)
        fa, _ = evaluate(m, f_avg, labels, device)

        m = load_ckpt(FusionMLP(nv*FEAT_DIM), f"{CKPT_FEAT}/c{cid}_concat.pth", device)
        fc, _ = evaluate(m, f_cat, labels, device)

        rows.append((lbl, [raw, la, lc, fa, fc]))
    print_table(f"추가실험 1 — GT 0° + c-MAS views, GT-trained MLP, zero-shot (N={N})", rows)
    return rows


# ── 추가실험 2: Mixed training (GT 0° + c-MAS → checkpoints_mixed) ────────
def eval_mixed(device):
    gt_l = torch.FloatTensor(np.load(f"{LOGIT_DIR}/logits_val.npz")["logits"])
    gt_f = torch.FloatTensor(np.load(f"{FEAT_DIR}/features_val.npz")["features"])
    cm_l = torch.FloatTensor(np.load(f"{LOGIT_DIR}/logits_cmas_val.npz")["logits"])
    cm_f = torch.FloatTensor(np.load(f"{FEAT_DIR}/features_cmas_val.npz")["features"])
    gt_la = torch.LongTensor(np.load(f"{LOGIT_DIR}/logits_val.npz")["labels"])
    cm_la = torch.LongTensor(np.load(f"{LOGIT_DIR}/logits_cmas_val.npz")["labels"])

    with open(f"{DATA_DIR}/ntu60_cmas_theta_000.pkl", "rb") as f:
        cmas_pkl = pickle.load(f)
    with open(f"{DATA_DIR}/ntu60_gt_theta_000.pkl", "rb") as f:
        gt_pkl = pickle.load(f)
    cmas_fds  = [a["frame_dir"] for a in cmas_pkl["annotations"]]
    gt_val_fds = gt_pkl["split"]["xsub_val"]
    gt_map = {fd: i for i, fd in enumerate(gt_val_fds)}
    gi = np.array([gt_map[fd] for fd in cmas_fds if fd in gt_map])
    ci = np.array([i for i, fd in enumerate(cmas_fds) if fd in gt_map])
    N  = len(gi)
    print(f"\n추가실험 2: 매칭 샘플 {N}개")

    ML = cm_l[ci].clone(); ML[:, 0, :] = gt_l[gi, 0, :]
    MF = cm_f[ci].clone(); MF[:, 0, :] = gt_f[gi, 0, :]
    ML_z = zscore(ML)
    labels = torch.LongTensor(gt_la[gi].numpy())

    methods = ["Logit-Avg", "Logit-Concat", "Feat-Avg", "Feat-Concat"]
    rows = []
    for cid, view_idx in MIXED_COMBOS.items():
        nv  = len(view_idx); lbl = f"C{cid} [{MIXED_LABELS[cid]}]"

        if cid == 1:
            l_avg = zscore(gt_l[gi, 0:1, :]).squeeze(1)
            l_cat = l_avg; f_avg = gt_f[gi, 0, :]; f_cat = f_avg
        else:
            l_avg = ML_z[:, view_idx, :].mean(1)
            l_cat = ML_z[:, view_idx, :].reshape(N, -1)
            f_avg = MF[:, view_idx, :].mean(1)
            f_cat = MF[:, view_idx, :].reshape(N, -1)

        m = load_ckpt(MLP(LOGIT_DIM),         f"{CKPT_MIXED}/c{cid}_logit_avg.pth",    device)
        la, _ = evaluate(m, l_avg, labels, device)

        m = load_ckpt(MLP(nv*LOGIT_DIM),      f"{CKPT_MIXED}/c{cid}_logit_concat.pth", device)
        lc, _ = evaluate(m, l_cat, labels, device)

        m = load_ckpt(FusionMLP(FEAT_DIM),    f"{CKPT_MIXED}/c{cid}_feat_avg.pth",    device)
        fa, _ = evaluate(m, f_avg, labels, device)

        m = load_ckpt(FusionMLP(nv*FEAT_DIM), f"{CKPT_MIXED}/c{cid}_feat_concat.pth", device)
        fc, _ = evaluate(m, f_cat, labels, device)

        rows.append((lbl, [la, lc, fa, fc]))

    print(f"\n{'='*80}")
    print(f"  추가실험 2 — Mixed GT+c-MAS 학습 MLP (N={N})")
    print(f"{'='*80}")
    print(f"  {'Combo':<26}" + "".join(f"{m:>13}" for m in methods))
    print("-"*80)
    for lbl, vals in rows:
        print(f"  {lbl:<26}" + "".join(f"{v:>12.2f}%" for v in vals))
    print("="*80)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    eval_gt(device)
    eval_cmas(device)
    eval_gt0_cmas(device)
    eval_mixed(device)
    print("\n모든 평가 완료.")

if __name__ == "__main__":
    main()
