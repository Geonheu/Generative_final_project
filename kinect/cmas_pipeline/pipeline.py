"""
c-MAS NTU60 multi-view generation pipeline.

Modes (--split):
  val   : xsub_val subset  → Stage 1 (infer) → Stage 2 (visualize) → Stage 3 (build pkls)
  train : xsub_train subset → Stage 1 (infer)                       → Stage 3 (build pkls)

Stages (--stage):
  all   : run all applicable stages (default)
  1     : c-MAS inference only → save npy per sample
  2     : visualize GT vs c-MAS 3D + 2D projections  (val only)
  3     : build per-angle MotionBERT-compatible pkl files
  13    : Stage 1 + 3 (no visualization; useful for train)

Usage (from kinect/cmas_pipeline dir; model_path defaults to ../../models/cMAS checkpoint):
    python pipeline.py --split val \\
        --n_samples 500 --diffusion_steps 20 --out_dir pipeline_out

    python pipeline.py --split train \\
        --per_class 8 --diffusion_steps 20 --out_dir pipeline_train_out --stage 13
"""

import sys, os, argparse, pickle, time, types
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..'))
_CMAS_ROOT   = os.path.join(_PROJECT_ROOT, 'models', 'cMAS')
sys.path.insert(0, _CMAS_ROOT)

# ── Paths (relative to this project) ─────────────────────────────────────
NTU_DATA_PATH  = os.path.join(_THIS_DIR, '..', 'data', 'ntu60_3danno.pkl')
MB_ACTION_DIR  = os.path.join(_THIS_DIR, '..', 'motionbert_har', 'data', 'action')

# ── Constants ─────────────────────────────────────────────────────────────
NBA_NTU_SCALE = 2.905
KINECT_DEPTH  = 3.5
NBA_DISTANCE  = 7
KINECT_FX     = 1081.37
IMG_W, IMG_H  = 1920, 1080

PKL_ANGLES = [0, 30, -30, 45, -45, 90, -90]
ANGLE_TAGS = {0:'theta_000', 30:'theta_p030', -30:'theta_n030',
              45:'theta_p045', -45:'theta_n045', 90:'theta_p090', -90:'theta_n090'}

NBA_BONES  = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,9),(7,8),
              (7,10),(10,11),(11,12),(7,13),(13,14),(14,15)]
COCO_BONES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
              (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
BONE_COLORS = {'sp':'#FF6B6B','la':'#4ECDC4','ra':'#45B7D1',
               'to':'#DDA0DD','ll':'#96CEB4','rl':'#FFEAA7'}

NTU_ACTIONS = [
    'drink water','eat meal','brush teeth','brush hair','drop','pick up','throw',
    'sit down','stand up','clapping','reading','writing','tear up paper',
    'wear jacket','take off jacket','wear shoe','take off shoe','wear on glasses',
    'take off glasses','put on hat/cap','take off hat/cap','cheer up','hand waving',
    'kicking something','reach into pocket','hopping','jump up','make a phone call',
    'playing with phone','typing on keyboard','pointing to something','taking selfie',
    'check time','rub two hands','nod head/bow','shake head','wipe face','salute',
    'put palms together','cross hands in front','sneeze/cough','staggering','falling',
    'touch head','touch chest','touch back','touch neck','nausea/vomiting',
    'use a fan','punch other','kick other','push other','pat on back',
    'point finger at other','hug other','give something','touch other pocket',
    'handshaking','walking towards','walking apart',
]


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
    co[:, 0]  = nba3d[:, 8];  co[:, 1]  = nba3d[:, 8];  co[:, 2]  = nba3d[:, 8]
    co[:, 3]  = nba3d[:, 8];  co[:, 4]  = nba3d[:, 8]
    co[:, 5]  = nba3d[:, 10]; co[:, 6]  = nba3d[:, 13]
    co[:, 7]  = nba3d[:, 11]; co[:, 8]  = nba3d[:, 14]
    co[:, 9]  = nba3d[:, 12]; co[:, 10] = nba3d[:, 15]
    co[:, 11] = nba3d[:, 4];  co[:, 12] = nba3d[:, 1]
    co[:, 13] = nba3d[:, 5];  co[:, 14] = nba3d[:, 2]
    co[:, 15] = nba3d[:, 6];  co[:, 16] = nba3d[:, 3]
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
                     -m[:, :, 1] / Z * KINECT_FX + IMG_H / 2.], axis=-1)  # (T,17,2)


# ── c-MAS setup ───────────────────────────────────────────────────────────
def build_mas(model_path, num_views, diffusion_steps, out_dir, opt_steps):
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
    fixseed(args.seed)
    os.makedirs(out_dir, exist_ok=True)

    mas = MAS(args)
    mas.args.input_iterations = diffusion_steps - 1

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
    motions_3d = mas(model_kwargs=mk, save=False, visualize=False)
    return motions_3d[0].detach().cpu().numpy()  # (T,16,3)


# ── Visualization helpers ─────────────────────────────────────────────────
def _bone_col_coco(s, e):
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
                color=_bone_col_coco(*b), lw=2.2, solid_capstyle='round')
    ax.scatter(j2d[:, 0], j2d[:, 1], c='#FFF', s=22, zorder=5, linewidths=0)
    xc, yc = j2d[:, 0].mean(), j2d[:, 1].mean()
    h = max((j2d.max(0) - j2d.min(0)).max() / 2 * 1.3, 100)
    ax.set_xlim(xc-h, xc+h); ax.set_ylim(yc+h, yc-h); ax.set_aspect('equal')
    ax.set_title(title, fontsize=8, fontweight='bold', color='#EEE', pad=3)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')


def save_vis(nba3d, kp3d, label, frame_dir, vis_dir):
    T   = nba3d.shape[0]; mid = T // 2
    action  = NTU_ACTIONS[label] if label < len(NTU_ACTIONS) else f'label_{label}'
    coco3d  = nba16_to_coco17_3d(nba3d)
    gt_nba  = kinect25_to_nba16_3d(kp3d)
    gt_nba_c = (gt_nba - gt_nba[:, 0:1, :]) * NBA_NTU_SCALE

    n_cols = 2 + len(PKL_ANGLES)
    fig = plt.figure(figsize=(3.2 * n_cols, 4.5))
    fig.patch.set_facecolor('#12122A')
    fig.suptitle(f'c-MAS · {frame_dir} · {action} · frame {mid}/{T}',
                 fontsize=9, fontweight='bold', color='#EEE', y=1.01)

    ax0 = fig.add_subplot(1, n_cols, 1, projection='3d')
    _draw_3d(ax0, gt_nba_c[mid], 'GT\n(Kinect→NBA)')
    ax1 = fig.add_subplot(1, n_cols, 2, projection='3d')
    _draw_3d(ax1, nba3d[mid], 'c-MAS\n(NBA-16 3D)')

    for ci, deg in enumerate(PKL_ANGLES):
        ax  = fig.add_subplot(1, n_cols, 3 + ci)
        j2d = project_to_2d(coco3d[mid:mid+1], np.deg2rad(deg))[0]
        lbl = '0°' if deg == 0 else f'{"+":"" [deg<0]}{deg}°'
        _draw_2d(ax, j2d, f'COCO-17\n({lbl})')

    legend = [mpatches.Patch(color=v, label=k) for k, v in
              [('Spine', '#FF6B6B'), ('L-Arm', '#4ECDC4'), ('R-Arm', '#45B7D1'),
               ('Torso', '#DDA0DD'), ('L-Leg', '#96CEB4'), ('R-Leg', '#FFEAA7')]]
    fig.legend(handles=legend, loc='lower center', ncol=6, fontsize=7,
               framealpha=0.3, facecolor='#1A1A2E', labelcolor='#EEE',
               edgecolor='#444', bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    out = os.path.join(vis_dir, f'vis_{frame_dir}_{action.replace(" ","_")}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Stage 1: c-MAS inference ──────────────────────────────────────────────
def stage1(opts, split_key):
    save_dir = os.path.join(opts.out_dir, 'cmas_3d')
    os.makedirs(save_dir, exist_ok=True)

    nba_mean = np.load(os.path.join(_CMAS_ROOT, 'data_loaders/nba/Mean.npy')).astype(np.float32)
    nba_std  = np.load(os.path.join(_CMAS_ROOT, 'data_loaders/nba/Std.npy')).astype(np.float32)

    print('Loading NTU60 GT ...')
    with open(NTU_DATA_PATH, 'rb') as f:
        ntu = pickle.load(f)
    split_set = set(ntu['split'][split_key])

    # Stratified sampling
    from collections import defaultdict
    buckets = defaultdict(list)
    for ann in ntu['annotations']:
        if ann['frame_dir'] not in split_set: continue
        lb = ann['label']
        if len(buckets[lb]) < opts.per_class:
            buckets[lb].append(ann)
    samples = [a for v in buckets.values() for a in v]
    print(f'Selected {len(samples)} {split_key} samples ({opts.per_class}/class × {len(buckets)} classes)')

    done = {f.replace('.npy', '') for f in os.listdir(save_dir) if f.endswith('.npy')}
    todo = [a for a in samples if a['frame_dir'] not in done]
    print(f'Already done: {len(done)}, To process: {len(todo)}')
    if not todo:
        print('All samples already processed.'); return

    mas = build_mas(opts.model_path, opts.num_views, opts.diffusion_steps,
                    opts.out_dir, opts.opt_steps)
    t_total = 0
    for si, ann in enumerate(tqdm(todo, desc='Stage 1 — c-MAS')):
        kp3d = ann['keypoint'].astype(np.float32)[0]; T = ann['total_frames']
        t0   = time.time()
        nba3d = run_cmas_sample(mas, kp3d, nba_mean, nba_std)
        t_total += time.time() - t0
        np.save(os.path.join(save_dir, f"{ann['frame_dir']}.npy"),
                {'nba3d': nba3d, 'label': ann['label'],
                 'total_frames': T, 'gt_kp3d': kp3d})
        if (si + 1) % 20 == 0:
            print(f'  [{si+1}/{len(todo)}] avg {t_total/(si+1):.1f}s/sample')

    print(f'\nStage 1 done. {len(os.listdir(save_dir))} total samples.'
          f'  ({t_total/60:.1f} min, {t_total/max(len(todo),1):.1f}s/sample)')


# ── Stage 2: visualize (val only) ─────────────────────────────────────────
def stage2(opts, n_vis=8):
    save_dir = os.path.join(opts.out_dir, 'cmas_3d')
    vis_dir  = os.path.join(opts.out_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(save_dir) if f.endswith('.npy'))
    picked, seen = [], set()
    for f in files:
        d = np.load(os.path.join(save_dir, f), allow_pickle=True).item()
        lb = d['label']
        if lb not in seen:
            picked.append((f, d)); seen.add(lb)
        if len(picked) >= n_vis: break
    if not picked:
        picked = [(files[i], np.load(os.path.join(save_dir, files[i]),
                   allow_pickle=True).item()) for i in range(min(n_vis, len(files)))]

    for fname, d in picked:
        save_vis(d['nba3d'], d['gt_kp3d'], d['label'],
                 fname.replace('.npy', ''), vis_dir)
    print(f'Stage 2 done. {len(picked)} visualizations → {vis_dir}')


# ── Stage 3: build per-angle pkl files ────────────────────────────────────
def stage3(opts, split_key, pkl_prefix):
    save_dir = os.path.join(opts.out_dir, 'cmas_3d')
    pkl_dir  = os.path.join(opts.out_dir, 'pkls')
    os.makedirs(pkl_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(save_dir) if f.endswith('.npy'))
    if not files:
        print('No c-MAS outputs found. Run Stage 1 first.'); return

    print(f'Building pkls from {len(files)} samples ...')
    anns_per_angle = {deg: [] for deg in PKL_ANGLES}
    frame_dirs = []

    for fname in tqdm(files, desc='Stage 3 — build pkls'):
        d     = np.load(os.path.join(save_dir, fname), allow_pickle=True).item()
        nba3d = d['nba3d']; T = d['total_frames']; label = int(d['label'])
        fd    = fname.replace('.npy', '')
        frame_dirs.append(fd)
        coco3d = nba16_to_coco17_3d(nba3d)

        for deg in PKL_ANGLES:
            proj2d = project_to_2d(coco3d, np.deg2rad(deg))   # (T,17,2)
            anns_per_angle[deg].append({
                'frame_dir':      fd,
                'label':          label,
                'total_frames':   T,
                'img_shape':      (IMG_H, IMG_W),
                'keypoint':       proj2d[None].astype(np.float32),
                'keypoint_score': np.ones((1, T, 17), np.float32),
            })

    split = {split_key: frame_dirs}
    os.makedirs(MB_ACTION_DIR, exist_ok=True)

    for deg in PKL_ANGLES:
        tag  = ANGLE_TAGS[deg]
        data = {'split': split, 'annotations': anns_per_angle[deg]}
        for path in [os.path.join(pkl_dir,      f'{pkl_prefix}_{tag}.pkl'),
                     os.path.join(MB_ACTION_DIR, f'{pkl_prefix}_{tag}.pkl')]:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        print(f'  {tag}: {len(anns_per_angle[deg])} samples → saved')

    print(f'Stage 3 done. Pkls in {pkl_dir} and {MB_ACTION_DIR}')


# ── Entry point ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split',           default='val', choices=['val', 'train'],
                   help='val | train')
    p.add_argument('--stage',           default='all',
                   help='all | 1 | 2 | 3 | 13')
    p.add_argument('--model_path',      default=os.path.join(_CMAS_ROOT, 'save/yoga_diffusion_model/checkpoint_200000.pth'))
    p.add_argument('--diffusion_steps', type=int, default=20)
    p.add_argument('--opt_steps',       type=int, default=200)
    p.add_argument('--num_views',       type=int, default=3)
    # val-specific
    p.add_argument('--n_samples',       type=int, default=500,
                   help='Total samples to generate (val only, distributes across classes)')
    # train-specific
    p.add_argument('--per_class',       type=int, default=8,
                   help='Samples per class (train mode)')
    p.add_argument('--n_vis',           type=int, default=8,
                   help='Number of visualizations for Stage 2')
    p.add_argument('--out_dir',         default=None,
                   help='Output directory (default: pipeline_out or pipeline_train_out)')
    return p.parse_args()


def main():
    opts = parse_args()

    is_val   = opts.split == 'val'
    split_key = 'xsub_val' if is_val else 'xsub_train'
    pkl_prefix = 'ntu60_cmas' if is_val else 'ntu60_cmas_train'

    if opts.out_dir is None:
        opts.out_dir = 'pipeline_out' if is_val else 'pipeline_train_out'
    os.makedirs(opts.out_dir, exist_ok=True)

    # for val, per_class = n_samples // 60
    if is_val:
        opts.per_class = max(1, opts.n_samples // 60)

    stage = opts.stage
    run_all = (stage == 'all')

    print(f"split={opts.split}  stage={stage}  out_dir={opts.out_dir}")

    if run_all or '1' in stage:
        print('\n' + '='*60 + '\n  STAGE 1 — c-MAS inference\n' + '='*60)
        stage1(opts, split_key)

    if (run_all or '2' in stage) and is_val:
        print('\n' + '='*60 + '\n  STAGE 2 — Visualisation\n' + '='*60)
        stage2(opts, n_vis=opts.n_vis)
    elif '2' in stage and not is_val:
        print('Stage 2 (visualization) is only available for --split val, skipping.')

    if run_all or '3' in stage:
        print('\n' + '='*60 + '\n  STAGE 3 — Build per-angle pkls\n' + '='*60)
        stage3(opts, split_key, pkl_prefix)

    print('\nAll done.')


if __name__ == '__main__':
    main()
