"""
c-MAS evaluation on NTU60.

Modes (--mode):
  full   : Run c-MAS on N stratified samples → project all angles → MotionBERT eval
           Reports per-angle Top-1/F1 + multi-view ensemble combos
  quick  : Run c-MAS on a few samples → visualize GT/c-MAS 3D + 2D projections
  time   : Benchmark c-MAS inference time for 1 sample (projects time for N samples)

Usage (from kinect/cmas_pipeline dir; model_path defaults to ../../models/cMAS checkpoint):
    python eval.py --mode full --n_eval 200 --diffusion_steps 20

    python eval.py --mode quick --num_samples 3

    python eval.py --mode time
"""

import sys, os, argparse, pickle, time, types
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..'))
_CMAS_ROOT    = os.path.join(_PROJECT_ROOT, 'models', 'cMAS')
_MB_ROOT      = os.path.join(_PROJECT_ROOT, 'models', 'MotionBERT')
sys.path.insert(0, _CMAS_ROOT)

# ── Paths ─────────────────────────────────────────────────────────────────
NTU_DATA_PATH  = os.path.join(_THIS_DIR, '..', 'data', 'ntu60_3danno.pkl')
MB_CONFIG      = os.path.join(_MB_ROOT, 'configs', 'action', 'MB_ft_NTU60_xsub.yaml')
MB_WEIGHTS     = os.path.join(_MB_ROOT, 'save', 'MB_ft_NTU60_xsub', 'best_epoch.bin')

# ── Constants ─────────────────────────────────────────────────────────────
NBA_NTU_SCALE = 2.905
KINECT_DEPTH  = 3.5
NBA_DISTANCE  = 7
KINECT_FX     = 1081.37
IMG_W, IMG_H  = 1920, 1080

EVAL_ANGLES = [0, 30, -30, 45, -45, 90, -90]
NBA_BONES   = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,9),(7,8),
               (7,10),(10,11),(11,12),(7,13),(13,14),(14,15)]
COCO_BONES  = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
               (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
BONE_COLORS = {'sp':'#FF6B6B','la':'#4ECDC4','ra':'#45B7D1',
               'to':'#DDA0DD','ll':'#96CEB4','rl':'#FFEAA7'}

NTU_ACTIONS = {0:'drink water',1:'eat meal',2:'brush teeth',3:'brush hair',4:'drop',
               5:'pick up',6:'throw',7:'sit down',8:'stand up',9:'clapping'}


# ── Joint mappings ────────────────────────────────────────────────────────
def kinect25_to_nba16_3d(kp3d):
    T = kp3d.shape[0]; nb = np.zeros((T, 16, 3), np.float32)
    nb[:, 0]  = (kp3d[:, 12] + kp3d[:, 16]) / 2
    for i, k in enumerate([16, 17, 18, 12, 13, 14]): nb[:, i+1] = kp3d[:, k]
    nb[:, 7]  = (kp3d[:, 4] + kp3d[:, 8]) / 2
    nb[:, 8]  = (kp3d[:, 2] + kp3d[:, 3]) / 2
    nb[:, 9]  = kp3d[:, 3]
    for i, k in zip([10, 11, 12, 13, 14, 15], [4, 5, 6, 8, 9, 10]): nb[:, i] = kp3d[:, k]
    return nb


def nba16_to_coco17_3d(nba3d):
    T = nba3d.shape[0]; co = np.zeros((T, 17, 3), np.float32)
    co[:, 0] = nba3d[:, 8]; co[:, 1] = nba3d[:, 8]; co[:, 2] = nba3d[:, 8]
    co[:, 3] = nba3d[:, 8]; co[:, 4] = nba3d[:, 8]
    co[:, 5] = nba3d[:, 10]; co[:, 6] = nba3d[:, 13]
    co[:, 7] = nba3d[:, 11]; co[:, 8] = nba3d[:, 14]
    co[:, 9] = nba3d[:, 12]; co[:, 10] = nba3d[:, 15]
    co[:, 11] = nba3d[:, 4]; co[:, 12] = nba3d[:, 1]
    co[:, 13] = nba3d[:, 5]; co[:, 14] = nba3d[:, 2]
    co[:, 15] = nba3d[:, 6]; co[:, 16] = nba3d[:, 3]
    return co


# ── Coordinate helpers ────────────────────────────────────────────────────
def prepare_ntu_for_cmas(kp3d, nba_mean, nba_std):
    nb  = kinect25_to_nba16_3d(kp3d)
    cen = (nb - nb[:, 0:1, :]) * NBA_NTU_SCALE
    Z   = cen[:, :, 2:3]
    r2d = np.concatenate([cen[:, :, 0:1] / (Z + NBA_DISTANCE) * NBA_DISTANCE,
                          cen[:, :, 1:2] / (Z + NBA_DISTANCE) * NBA_DISTANCE], axis=-1)
    n2d = ((r2d - nba_mean) / nba_std).transpose(1, 2, 0).astype(np.float32)
    return torch.from_numpy(n2d).unsqueeze(0)   # (1,16,2,T)


def _align_vectors(a, b):
    a = a / (np.linalg.norm(a) + 1e-8); b = b / (np.linalg.norm(b) + 1e-8)
    c = float(np.dot(a, b))
    if c >  1 - 1e-6: return np.eye(3, dtype=np.float64)
    if c < -1 + 1e-6:
        p = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
        p -= np.dot(p, a) * a; p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    v  = np.cross(a, b); s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)


def rotate_body_aligned(coco3d, theta):
    root  = (coco3d[..., 11, :] + coco3d[..., 12, :]) / 2.0
    cen   = coco3d - root[..., None, :]
    spine = ((cen[..., 5, :] + cen[..., 6, :]) / 2.0 -
             (cen[..., 11, :] + cen[..., 12, :]) / 2.0)
    spine = spine.mean(0) if coco3d.ndim == 3 else spine
    R_al  = _align_vectors(spine, np.array([0., 1., 0.]))
    c_t, s_t = np.cos(theta), np.sin(theta)
    R_az  = np.array([[c_t, 0, s_t], [0, 1, 0], [-s_t, 0, c_t]])
    rot   = (cen.astype(np.float64) @ R_al.T @ R_az.T) @ R_al
    return (rot + root[..., None, :]).astype(np.float32)


def project_to_2d(coco3d_nba, theta):
    rot = rotate_body_aligned(coco3d_nba, theta)
    m   = rot / NBA_NTU_SCALE; m[:, :, 2] += KINECT_DEPTH
    Z   = np.where(np.abs(m[:, :, 2]) < 1e-6, 1e-6, m[:, :, 2])
    return np.stack([m[:, :, 0] / Z * KINECT_FX + IMG_W / 2.,
                     -m[:, :, 1] / Z * KINECT_FX + IMG_H / 2.], axis=-1)


# ── c-MAS setup ───────────────────────────────────────────────────────────
def build_mas(model_path, num_views, diffusion_steps, opt_steps, out_dir='_eval_tmp'):
    import sample.mas as mas_module
    mas_module.OPTIMIZE_STEPS = opt_steps
    from sample.mas import MAS
    from utils.parser_utils import mas_args
    from utils.fixseed import fixseed

    saved = sys.argv[:]
    sys.argv = ['x',
        '--model_path', model_path,
        '--num_samples', '1', '--num_views', str(num_views),
        '--diffusion_steps', str(diffusion_steps),
        '--dataset', 'nba', '--output_dir', out_dir, '--overwrite']
    args = mas_args(); sys.argv = saved
    fixseed(args.seed); os.makedirs(out_dir, exist_ok=True)

    mas = MAS(args); mas.args.input_iterations = diffusion_steps - 1

    def _angles(self):
        n = self.num_views
        self.hor_angles = [0.] + [i * 2 * np.pi / n for i in range(1, n)]
        self.ver_angles = [0.] * n
    mas.sample_angles = types.MethodType(_angles, mas)
    return mas


def run_cmas_sample(mas, kp3d, nba_mean, nba_std):
    T  = kp3d.shape[0]
    mt = prepare_ntu_for_cmas(kp3d, nba_mean, nba_std)
    mk = {'y': {'lengths': torch.tensor([T], dtype=torch.long),
                'mask':    torch.ones(1, 16, 2, T, dtype=torch.bool),
                'uncond':  True}}
    mas.input_motions    = mt.to(mas.device)
    mas.args.num_samples = 1; mas.args.batch_size = 1
    out = mas(model_kwargs=mk, save=False, visualize=False)
    return out[0].detach().cpu().numpy()  # (T,16,3)


# ── MotionBERT input conversion ───────────────────────────────────────────
def kp2d_to_mb_input(kp2d, T, n_frames=243, scale_range=(2, 2)):
    from lib.utils.utils_data import crop_scale

    def make_cam(x):
        w = IMG_W
        return x / w * 2 - [1, IMG_H / IMG_W]

    def coco2h36m(x):
        y = np.zeros_like(x)
        y[:, :, 0,  :] = (x[:, :, 11, :] + x[:, :, 12, :]) * 0.5
        y[:, :, 1,  :] = x[:, :, 12, :];  y[:, :, 2,  :] = x[:, :, 14, :]
        y[:, :, 3,  :] = x[:, :, 16, :];  y[:, :, 4,  :] = x[:, :, 11, :]
        y[:, :, 5,  :] = x[:, :, 13, :];  y[:, :, 6,  :] = x[:, :, 15, :]
        y[:, :, 8,  :] = (x[:, :, 5, :] + x[:, :, 6, :]) * 0.5
        y[:, :, 7,  :] = (y[:, :, 0, :] + y[:, :, 8, :]) * 0.5
        y[:, :, 9,  :] = x[:, :, 0, :];   y[:, :, 10, :] = (x[:, :, 1, :] + x[:, :, 2, :]) * 0.5
        y[:, :, 11, :] = x[:, :, 5, :];   y[:, :, 12, :] = x[:, :, 7, :]
        y[:, :, 13, :] = x[:, :, 9, :];   y[:, :, 14, :] = x[:, :, 6, :]
        y[:, :, 15, :] = x[:, :, 8, :];   y[:, :, 16, :] = x[:, :, 10, :]
        return y

    idx   = np.linspace(0, T, num=n_frames, endpoint=False).astype(int)
    kp2d_m = kp2d[None]                          # (1,T,17,2)
    cam    = make_cam(kp2d_m)                     # (1,T,17,2)
    h36m   = coco2h36m(cam[:, idx])              # (1,n_frames,17,2)
    conf   = np.ones((1, n_frames, 17, 1), np.float32)
    motion = np.concatenate([h36m, conf], axis=-1)
    motion = np.concatenate([motion, np.zeros_like(motion)], axis=0)  # (2,n_frames,17,3)
    return crop_scale(motion[None], list(scale_range))[0].astype(np.float32)


# ── Visualization helpers ─────────────────────────────────────────────────
def _bone_col(s, e):
    if s <= 4 or e <= 4:              return BONE_COLORS['sp']
    if {s, e} <= {5, 6}:             return BONE_COLORS['to']
    if s in (5, 7) and e in (7, 9):  return BONE_COLORS['la']
    if s in (6, 8) and e in (8, 10): return BONE_COLORS['ra']
    if {s, e} & {5, 6, 11, 12}:      return BONE_COLORS['to']
    if s in (11, 13) and e in (13, 15): return BONE_COLORS['ll']
    return BONE_COLORS['rl']


def _draw_3d(ax, joints, title):
    ax.set_facecolor('#1A1A2E')
    for s, e in NBA_BONES:
        ax.plot([joints[s, 0], joints[e, 0]], [joints[s, 2], joints[e, 2]],
                [joints[s, 1], joints[e, 1]], c='#4ECDC4', lw=2)
    ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c='#FFF', s=16, linewidths=0)
    c = joints.mean(0); h = max((joints.max(0) - joints.min(0)).max() / 2 * 1.3, 0.4)
    ax.set_xlim(c[0]-h, c[0]+h); ax.set_ylim(c[2]-h, c[2]+h); ax.set_zlim(c[1]-h, c[1]+h)
    ax.set_title(title, fontsize=8, fontweight='bold', color='#EEE', pad=3)
    for a in [ax.xaxis, ax.yaxis, ax.zaxis]:
        a.pane.fill = False; a.pane.set_edgecolor('#333')
    ax.tick_params(labelsize=5, colors='#AAA'); ax.view_init(10, -65)


def _draw_2d(ax, j2d, title):
    ax.set_facecolor('#1A1A2E')
    for b in COCO_BONES:
        ax.plot([j2d[b[0], 0], j2d[b[1], 0]], [j2d[b[0], 1], j2d[b[1], 1]],
                color=_bone_col(*b), lw=2.2, solid_capstyle='round')
    ax.scatter(j2d[:, 0], j2d[:, 1], c='#FFF', s=22, zorder=5, linewidths=0)
    xc, yc = j2d[:, 0].mean(), j2d[:, 1].mean()
    h = max((j2d.max(0) - j2d.min(0)).max() / 2 * 1.3, 100)
    ax.set_xlim(xc-h, xc+h); ax.set_ylim(yc+h, yc-h); ax.set_aspect('equal')
    ax.set_title(title, fontsize=8, fontweight='bold', color='#EEE', pad=3)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')


def save_quick_vis(nba3d, kp3d, label, frame_dir, out_dir):
    T   = nba3d.shape[0]; mid = T // 2
    action = NTU_ACTIONS.get(label, f'label_{label}')
    gt_nba = kinect25_to_nba16_3d(kp3d)
    gt_c   = (gt_nba - gt_nba[:, 0:1, :]) * NBA_NTU_SCALE
    coco3d = nba16_to_coco17_3d(nba3d)

    n_cols = 2 + len(EVAL_ANGLES)
    fig = plt.figure(figsize=(3.2 * n_cols, 4.5)); fig.patch.set_facecolor('#12122A')
    fig.suptitle(f'c-MAS · {frame_dir} · {action} · frame {mid}/{T}',
                 fontsize=9, fontweight='bold', color='#EEE', y=1.01)

    _draw_3d(fig.add_subplot(1, n_cols, 1, projection='3d'), gt_c[mid], 'GT\n(NBA-16)')
    _draw_3d(fig.add_subplot(1, n_cols, 2, projection='3d'), nba3d[mid], 'c-MAS\n(NBA-16)')
    for ci, deg in enumerate(EVAL_ANGLES):
        j2d = project_to_2d(coco3d[mid:mid+1], np.deg2rad(deg))[0]
        lbl = '0°' if deg == 0 else f'{"+":"" [deg<0]}{deg}°'
        _draw_2d(fig.add_subplot(1, n_cols, 3 + ci), j2d, f'COCO-17\n({lbl})')

    legend = [mpatches.Patch(color=v, label=k) for k, v in
              [('Spine','#FF6B6B'),('L-Arm','#4ECDC4'),('R-Arm','#45B7D1'),
               ('Torso','#DDA0DD'),('L-Leg','#96CEB4'),('R-Leg','#FFEAA7')]]
    fig.legend(handles=legend, loc='lower center', ncol=6, fontsize=7,
               framealpha=0.3, facecolor='#1A1A2E', labelcolor='#EEE',
               edgecolor='#444', bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    out = os.path.join(out_dir, f'cmas_{frame_dir}_{action.replace(" ","_")}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig); print(f'  Saved: {out}')


# ── Mode: full ────────────────────────────────────────────────────────────
def run_full(opts):
    from sklearn.metrics import f1_score
    sys.path.insert(0, _MB_ROOT)

    nba_mean = np.load(os.path.join(_CMAS_ROOT, 'data_loaders/nba/Mean.npy')).astype(np.float32)
    nba_std  = np.load(os.path.join(_CMAS_ROOT, 'data_loaders/nba/Std.npy')).astype(np.float32)

    print('Loading NTU60 GT ...')
    with open(NTU_DATA_PATH, 'rb') as f:
        ntu = pickle.load(f)
    val_set = set(ntu['split']['xsub_val'])

    per_class = max(1, opts.n_eval // 60)
    from collections import defaultdict
    buckets = defaultdict(list)
    for ann in ntu['annotations']:
        if ann['frame_dir'] not in val_set: continue
        if len(buckets[ann['label']]) < per_class:
            buckets[ann['label']].append(ann)
    samples = [a for v in buckets.values() for a in v][:opts.n_eval]
    print(f'Selected {len(samples)} samples  ({per_class}/class × {len(buckets)} classes)')

    mas = build_mas(opts.model_path, opts.num_views, opts.diffusion_steps,
                    opts.opt_steps, '_eval_tmp')

    from lib.utils.tools import get_config
    from lib.utils.learning import load_backbone
    from lib.model.model_action import ActionNet
    import torch.nn as nn
    mb_args = get_config(MB_CONFIG)
    bb      = load_backbone(mb_args)
    mb_model = ActionNet(backbone=bb, dim_rep=mb_args.dim_rep,
                         num_classes=mb_args.action_classes,
                         dropout_ratio=mb_args.dropout_ratio,
                         version=mb_args.model_version,
                         hidden_dim=mb_args.hidden_dim,
                         num_joints=mb_args.num_joints)
    if torch.cuda.is_available():
        mb_model = nn.DataParallel(mb_model).cuda()
    ckpt = torch.load(MB_WEIGHTS, map_location='cpu')
    mb_model.load_state_dict(ckpt['model'], strict=True)
    mb_model.eval()
    print('MotionBERT loaded.')

    all_preds  = {deg: [] for deg in EVAL_ANGLES}
    all_labels = []
    t_total    = 0

    for si, ann in enumerate(tqdm(samples, desc='c-MAS eval')):
        kp3d  = ann['keypoint'].astype(np.float32)[0]; T = ann['total_frames']
        t0    = time.time()
        nba3d = run_cmas_sample(mas, kp3d, nba_mean, nba_std)
        t_total += time.time() - t0
        coco3d = nba16_to_coco17_3d(nba3d)
        all_labels.append(ann['label'])

        for deg in EVAL_ANGLES:
            proj2d = project_to_2d(coco3d, np.deg2rad(deg))
            inp    = kp2d_to_mb_input(proj2d, T,
                                      n_frames=mb_args.clip_len,
                                      scale_range=mb_args.scale_range_test)
            with torch.no_grad():
                x = torch.from_numpy(inp).unsqueeze(0)
                if torch.cuda.is_available(): x = x.cuda()
                all_preds[deg].append(mb_model(x).cpu().numpy()[0])

        if (si + 1) % 20 == 0:
            print(f'  [{si+1}/{len(samples)}] avg {t_total/(si+1):.1f}s/sample')

    labels = np.array(all_labels)
    print('\n' + '='*60)
    print(f'  {"Angle":<8} {"Top-1":>8} {"Macro F1":>10}  (n={len(samples)})')
    print('='*60)
    for deg in EVAL_ANGLES:
        logits = np.stack(all_preds[deg]); preds = logits.argmax(1)
        top1 = (preds == labels).mean() * 100
        f1   = f1_score(labels, preds, average='macro') * 100
        sign = '+' if deg > 0 else ''
        print(f'  {("" if deg==0 else sign+str(deg))+"°":<8} {top1:>7.2f}%  {f1:>9.2f}%')
    print('='*60)

    combos = {1:[0], 2:[0,30,-30], 3:[0,45,-45], 4:[0,90,-90]}
    print('\n── Multi-view ensemble ──')
    print(f'  {"Combo":<22} {"Top-1":>8} {"Macro F1":>10}')
    print('-'*44)
    for cid, degs in combos.items():
        ls = sum(np.stack(all_preds[d]) for d in degs) / len(degs)
        p  = ls.argmax(1)
        t1 = (p == labels).mean() * 100
        f1 = f1_score(labels, p, average='macro') * 100
        print(f'  C{cid} [{", ".join(str(d)+"°" for d in degs)}]{"":<4} {t1:>7.2f}%  {f1:>9.2f}%')

    print(f'\nTotal c-MAS time: {t_total/60:.1f} min  ({t_total/len(samples):.1f}s/sample)')


# ── Mode: quick ───────────────────────────────────────────────────────────
def run_quick(opts):
    nba_mean = np.load(os.path.join(_CMAS_ROOT, 'data_loaders/nba/Mean.npy')).astype(np.float32)
    nba_std  = np.load(os.path.join(_CMAS_ROOT, 'data_loaders/nba/Std.npy')).astype(np.float32)

    print('Loading NTU60 GT ...')
    with open(NTU_DATA_PATH, 'rb') as f:
        ntu = pickle.load(f)
    val_set = set(ntu['split']['xsub_val'])

    picked, seen = [], set()
    for ann in ntu['annotations']:
        if ann['frame_dir'] not in val_set: continue
        lb = ann['label']
        if lb not in seen and lb < 10:
            picked.append(ann); seen.add(lb)
        if len(picked) >= opts.num_samples: break
    print(f'Selected {len(picked)} samples')

    mas = build_mas(opts.model_path, opts.num_views, opts.diffusion_steps,
                    opts.opt_steps, opts.out_dir)
    os.makedirs(opts.out_dir, exist_ok=True)

    for si, ann in enumerate(picked):
        kp3d = ann['keypoint'].astype(np.float32)[0]; T = ann['total_frames']
        print(f'\n[{si+1}/{len(picked)}] {ann["frame_dir"]}  T={T}')
        nba3d = run_cmas_sample(mas, kp3d, nba_mean, nba_std)
        save_quick_vis(nba3d, kp3d, ann['label'], ann['frame_dir'], opts.out_dir)

    print(f'\nDone. Results in: {opts.out_dir}')


# ── Mode: time ────────────────────────────────────────────────────────────
def run_time(opts):
    nba_mean = np.load(os.path.join(_CMAS_ROOT, 'data_loaders/nba/Mean.npy')).astype(np.float32)
    nba_std  = np.load(os.path.join(_CMAS_ROOT, 'data_loaders/nba/Std.npy')).astype(np.float32)

    with open(NTU_DATA_PATH, 'rb') as f:
        ntu = pickle.load(f)
    ann  = ntu['annotations'][240]
    kp3d = ann['keypoint'].astype(np.float32)[0]; T = ann['total_frames']

    mas = build_mas(opts.model_path, opts.num_views, opts.diffusion_steps,
                    opts.opt_steps, '_timing_test')

    print(f'Benchmarking 1 sample (T={T}, steps={opts.diffusion_steps}, opt={opts.opt_steps}) ...')
    t0 = time.time()
    run_cmas_sample(mas, kp3d, nba_mean, nba_std)
    elapsed = time.time() - t0

    print(f'\n  1샘플 소요시간 : {elapsed:.1f}초')
    print(f'  200샘플 예상   : {elapsed*200/60:.0f}분')
    print(f'  500샘플 예상   : {elapsed*500/60:.0f}분')


# ── Entry point ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',            required=True, choices=['full', 'quick', 'time'])
    p.add_argument('--model_path',      default=os.path.join(_CMAS_ROOT, 'save/yoga_diffusion_model/checkpoint_200000.pth'))
    p.add_argument('--diffusion_steps', type=int, default=20)
    p.add_argument('--opt_steps',       type=int, default=100)
    p.add_argument('--num_views',       type=int, default=3)
    # full mode
    p.add_argument('--n_eval',          type=int, default=200,
                   help='Number of val samples to evaluate (full mode)')
    # quick mode
    p.add_argument('--num_samples',     type=int, default=3,
                   help='Number of samples to visualize (quick mode)')
    p.add_argument('--out_dir',         default='eval_out',
                   help='Output directory for quick mode visualizations')
    return p.parse_args()


def main():
    opts = parse_args()
    print(f"Mode: {opts.mode}  steps={opts.diffusion_steps}  opt={opts.opt_steps}")
    {'full': run_full, 'quick': run_quick, 'time': run_time}[opts.mode](opts)


if __name__ == '__main__':
    main()
