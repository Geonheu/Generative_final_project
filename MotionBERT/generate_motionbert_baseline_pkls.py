import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone

def coco17_to_h36m_2d(coco_2d):
    """
    convert COCO17 2D joints to H36M order
    """
    T = coco_2d.shape[0]
    h36m = np.zeros((T, 17, 2), dtype=np.float32)
    
    # lower body
    h36m[:, 0] = (coco_2d[:, 11] + coco_2d[:, 12]) / 2.0  # 0: Pelvis 
    h36m[:, 1] = coco_2d[:, 12]                                   # 1: R_Hip
    h36m[:, 2] = coco_2d[:, 14]                                   # 2: R_Knee
    h36m[:, 3] = coco_2d[:, 16]                                   # 3: R_Ankle
    h36m[:, 4] = coco_2d[:, 11]                                   # 4: L_Hip
    h36m[:, 5] = coco_2d[:, 13]                                   # 5: L_Knee
    h36m[:, 6] = coco_2d[:, 15]                                   # 6: L_Ankle
    
    # spine and neck
    neck = (coco_2d[:, 5] + coco_2d[:, 6]) / 2.0
    h36m[:, 7] = (h36m[:, 0] + neck) / 2.0                # 7: Spine
    h36m[:, 8] = neck                                     # 8: Neck
    
    # head and upper body
    h36m[:, 9] = coco_2d[:, 0]                            # 9: Nose
    h36m[:, 10] = (coco_2d[:, 1] + coco_2d[:, 2] + coco_2d[:, 3] + coco_2d[:, 4]) / 4.0 # 10: Head
    h36m[:, 11] = coco_2d[:, 5]                                   # 11: L_Shoulder
    h36m[:, 12] = coco_2d[:, 7]                                   # 12: L_Elbow
    h36m[:, 13] = coco_2d[:, 9]                                   # 13: L_Wrist
    h36m[:, 14] = coco_2d[:, 6]                                   # 14: R_Shoulder
    h36m[:, 15] = coco_2d[:, 8]                                   # 15: R_Elbow
    h36m[:, 16] = coco_2d[:, 10]                                  # 16: R_Wrist
    return h36m

def h36m_to_coco17_2d(h36m_2d):
    """
    Mapping the projected H36M joints back to COCO17 order
    """
    T = h36m_2d.shape[0]
    coco = np.zeros((T, 17, 2), dtype=np.float32)
    
    coco[:, 0] = h36m_2d[:, 9]   # Nose
    coco[:, 1] = h36m_2d[:, 10]  # L_Eye
    coco[:, 2] = h36m_2d[:, 10]  # R_Eye
    coco[:, 3] = h36m_2d[:, 10]  # L_Ear
    coco[:, 4] = h36m_2d[:, 10]  # R_Ear
    coco[:, 5] = h36m_2d[:, 11]  # L_Shoulder
    coco[:, 6] = h36m_2d[:, 14]  # R_Shoulder
    coco[:, 7] = h36m_2d[:, 12]  # L_Elbow
    coco[:, 8] = h36m_2d[:, 15]  # R_Elbow
    coco[:, 9] = h36m_2d[:, 13]  # L_Wrist
    coco[:, 10] = h36m_2d[:, 16] # R_Wrist
    coco[:, 11] = h36m_2d[:, 4]  # L_Hip
    coco[:, 12] = h36m_2d[:, 1]  # R_Hip
    coco[:, 13] = h36m_2d[:, 5]  # L_Knee
    coco[:, 14] = h36m_2d[:, 2]  # R_Knee
    coco[:, 15] = h36m_2d[:, 6]  # L_Ankle
    coco[:, 16] = h36m_2d[:, 3]  # R_Ankle
    return coco

def rotate_y_axis(pose_3d, theta):
    """ Rotate 3D joints around the vertical axis """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]])
    return pose_3d @ R.T

def project_to_image(pose_3d, img_w=1920, img_h=1080):
    """ Project normalized 3D joints to image coordinates """
    scale = min(img_w, img_h) / 2.0
    cx, cy = img_w / 2.0, img_h / 2.0
    x_pixel = pose_3d[:, :, 0] * scale + cx
    y_pixel = pose_3d[:, :, 1] * scale + cy
    return np.stack([x_pixel, y_pixel], axis=-1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = "./lifting_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("load ntu60_hrnet.pkl")
    with open('data/action/ntu60_hrnet.pkl', 'rb') as f:
        src_data = pickle.load(f)

    xsub_val_set = set(src_data['split']['xsub_val'])
    annotations = src_data['annotations']

    print("Initialize MotionBERT 3D Pose Model")
    args = get_config('configs/pose3d/MB_ft_h36m.yaml')
    model = load_backbone(args)
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    checkpoint_path = 'checkpoint/pose3d/best_epoch.bin'
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_pos' in checkpoint:
        model.load_state_dict(checkpoint['model_pos'], strict=True)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()

    view_sets = {
        "set_0_30_m30": [0, np.pi/6, -np.pi/6],                  # 0, +30, -30
        "set_0_45_m45": [0, np.pi/4, -np.pi/4],                  # 0, +45, -45
        "set_0_90_m90": [0, np.pi/2, -np.pi/2],                  # 0, +90, -90
        "set_0_45_m45_90_m90": [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2],  # 0, +45, -45, +90, -90
    }

    for set_name, angles in view_sets.items():
        print(f"\nProcessing {set_name} ({len(angles)} views)")
        set_annotations = [[] for _ in range(len(angles))]

        with torch.no_grad():
            for anno in tqdm(annotations, desc=f"{set_name}"):
                frame_dir = anno['frame_dir']
                if frame_dir not in xsub_val_set:
                    continue

                label = int(anno['label'])
                kp_2d = anno['keypoint'][0]  # (T, 17, 2)
                T = kp_2d.shape[0]
                
                # Convert COCO joints to H36M order
                h36m_in = coco17_to_h36m_2d(kp_2d)
                
                # Normalize image coordinates [-1,1]
                h36m_normalized = (h36m_in - np.array([960.0, 540.0])) / 540.0
                
                # Add confidence channel (T, 17, 3)
                conf_channel = np.ones((T, 17, 1), dtype=np.float32)
                h36m_ready = np.concatenate([h36m_normalized, conf_channel], axis=-1)
                
                # Split sequences longer than 243 frames
                max_len = 243
                joints_3d_chunks = []
                
                for start_idx in range(0, T, max_len):
                    end_idx = min(start_idx + max_len, T)
                    chunk = h36m_ready[start_idx:end_idx]
                    
                    inp = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
                    out_3d_chunk = model(inp)
                    joints_3d_chunks.append(out_3d_chunk.squeeze(0).cpu().numpy())
                
                # Merge chunk outputs
                joints_3d = np.concatenate(joints_3d_chunks, axis=0) # (T, 17, 3)

                for view_idx, theta in enumerate(angles):
                    rotated_3d = rotate_y_axis(joints_3d, theta)
                    projected_h36m = project_to_image(rotated_3d)
                    projected_coco = h36m_to_coco17_2d(projected_h36m)
                    
                    conf = np.ones((projected_coco.shape[0], 17), dtype=np.float32)

                    ann = {
                        'frame_dir': f"{frame_dir}_view{view_idx}",
                        'label': label,
                        'img_shape': (1080, 1920),
                        'original_shape': (1080, 1920),
                        'total_frames': projected_coco.shape[0],
                        'keypoint': projected_coco[np.newaxis],
                        'keypoint_score': conf[np.newaxis],
                    }
                    set_annotations[view_idx].append(ann)

        for view_idx in range(len(angles)):
            view_anns = set_annotations[view_idx]
            split = {
                'xsub_train': [],
                'xsub_val': [ann['frame_dir'] for ann in view_anns]
            }
            pkl_data = {'split': split, 'annotations': view_anns}
            
            save_path = os.path.join(output_dir, f"ntu60_xsub_val_{set_name}_view{view_idx}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(pkl_data, f)
            print(f"saved: {save_path} ({len(view_anns)} samples)")

    print("\ndata processing finished")

if __name__ == '__main__':
    main()