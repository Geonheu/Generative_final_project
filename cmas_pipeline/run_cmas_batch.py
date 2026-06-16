"""
Run cMAS (3D lifting diffusion model) on NTU60 xsub_val NBA16 samples.
Processes first half (index 0~8242). Run run_cmas_batch_gpu2.py in parallel for second half.
Input:  ntu_test_60class_nba/   (NBA16 .npy per sample)
Output: ntu_test_60class_3d/    (cMAS 3D output .npy per sample)
"""
import os
import glob

input_dir   = "ntu_test_60class_nba"
output_dir  = "ntu_test_60class_3d"
cmas_dir    = "."
model_path  = "save/yoga_diffusion_model/checkpoint_200000.pth"
motions_dir = os.path.join(cmas_dir, "dataset/nba/motions")

os.makedirs(output_dir, exist_ok=True)

files = sorted(os.listdir(input_dir))
files = files[:8243]
total = len(files)
print(f"GPU 0: processing {total} samples (index 0~8242)")

for i, fname in enumerate(files):
    src = os.path.join(input_dir, fname)
    sample_name = fname.replace(".npy", "")
    out = os.path.join(output_dir, sample_name)

    if os.path.exists(out + ".npy"):
        continue

    for f in glob.glob(os.path.join(motions_dir, "*.npy")):
        os.remove(f)
    os.system(f"cp {src} {os.path.join(motions_dir, fname)}")

    cmd = (
        f"cd {cmas_dir} && python -m sample.mas"
        f" --model_path {model_path}"
        f" --num_samples 1 --seed 42 --overwrite"
        f" --output_dir results/batch/{sample_name}"
        f" --use_data --num_views 7 --input_iterations 100"
        f" --dataset nba 2>/dev/null"
    )
    os.system(cmd)

    result_path = os.path.join(cmas_dir, f"results/batch/{sample_name}/{sample_name}.npy")
    if os.path.exists(result_path):
        os.system(f"cp {result_path} {out}.npy")

    if (i + 1) % 100 == 0:
        print(f"{i + 1}/{total}")

print("Done")
