#Evaluate view-set ensembles from the MotionBERT lifting -> rotation -> projection -> HAR pipeline
import os
import sys
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MB_ROOT  = os.path.normpath(os.path.join(_THIS_DIR, '..', '..', 'models', 'MotionBERT'))
sys.path.insert(0, _MB_ROOT)

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import ActionNet

class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def calculate_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():
    config_path = os.path.join(_MB_ROOT, "configs/action/MB_ft_NTU60_xsub.yaml")
    if not os.path.exists(config_path): config_path = os.path.join(_MB_ROOT, "configs/action/MB_train_NTU60_xsub.yaml")
    args = get_config(config_path)
    
    model_backbone = load_backbone(args)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, 
                      dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    checkpoint_path = os.path.join(_MB_ROOT, "save/MB_ft_NTU60_xsub/best_epoch.bin")
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    # outputs from the lifting baseline
    pkl_dir = "./lifting_outputs"
    
    # view groups used for the ensemble run
    view_sets = {
        "set_0_30_m30": [
            "ntu60_xsub_val_deg0.pkl", 
            "ntu60_xsub_val_deg30.pkl", 
            "ntu60_xsub_val_deg-30.pkl"
        ],
        "set_0_45_m45": [
            "ntu60_xsub_val_deg0.pkl", 
            "ntu60_xsub_val_deg45.pkl", 
            "ntu60_xsub_val_deg-45.pkl"
        ],
        "set_0_90_m90": [
            "ntu60_xsub_val_deg0.pkl", 
            "ntu60_xsub_val_deg90.pkl", 
            "ntu60_xsub_val_deg-90.pkl"
        ],
        "set_0_45_m45_90_m90": [
            "ntu60_xsub_val_deg0.pkl", 
            "ntu60_xsub_val_deg45.pkl", 
            "ntu60_xsub_val_deg-45.pkl", 
            "ntu60_xsub_val_deg90.pkl", 
            "ntu60_xsub_val_deg-90.pkl"
        ]
    }

    loader_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True}

    print("Evaluating multi-view ensemble")

    for set_name, file_list in view_sets.items():
        print(f"\n{set_name} with {len(file_list)} views")
        
        loaders = [DataLoader(NTURGBD(data_path=os.path.join(pkl_dir, f), data_split='xsub_val', n_frames=args.clip_len, random_move=False, scale_range=args.scale_range_test), **loader_params) for f in file_list]
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        set_logits_list = []
        set_gts_list = []
        
        with torch.no_grad():
            for batches in tqdm(zip(*loaders), desc=f"Ensemble {set_name}", total=len(loaders[0])):
                batch_size = len(batches[0][0])
                gt_labels = batches[0][1].cuda() if torch.cuda.is_available() else batches[0][1]
                
                ensemble_logits = []
                for batch_input, _ in batches:
                    if torch.cuda.is_available(): batch_input = batch_input.cuda()
                    output = model(batch_input) 
                    ensemble_logits.append(output)
                
                stacked_logits = torch.stack(ensemble_logits, dim=0) 
                mean_logits = torch.mean(stacked_logits, dim=0)      
                
                set_logits_list.append(mean_logits.cpu())
                set_gts_list.append(batches[0][1].cpu())
                
                loss = criterion(mean_logits, gt_labels)
                losses.update(loss.item(), batch_size)
                
                acc1, acc5 = calculate_accuracy(mean_logits, gt_labels, topk=(1, 5))
                top1.update(acc1[0].item(), batch_size)
                top5.update(acc5[0].item(), batch_size)

        print(f"[{set_name} Ensemble result]")
        print(f"Top-1 Acc: {top1.avg:.2f}% | Top-5 Acc: {top5.avg:.2f}% | Loss: {losses.avg:.4f}")
        
        final_set_logits = torch.cat(set_logits_list, dim=0).numpy()
        final_set_gts = torch.cat(set_gts_list, dim=0).numpy()
        
        # save logits for this view set
        save_filename = f"ensemble_{set_name.lower().replace(' ', '')}.pkl"
        save_path = os.path.join(pkl_dir, save_filename)
        
        with open(save_path, "wb") as f:
            pickle.dump({"logits": final_set_logits, "gt": final_set_gts}, f)
            
        print(f"saved logits to: {save_path}")
        print("-" * 50)

if __name__ == "__main__":
    main()