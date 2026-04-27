#!/bin/bash
#SBATCH --job-name=street_nocn
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=results_street_nocn/gen_%j.out
#SBATCH --error=results_street_nocn/gen_%j.err

export PYTHONUNBUFFERED=1
export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env
NV_BASE=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$NV_BASE/cuda_cupti/lib:$NV_BASE/cuda_nvrtc/lib:$NV_BASE/cuda_runtime/lib:$NV_BASE/cudnn/lib:$NV_BASE/cufft/lib:$NV_BASE/cusolver/lib:$NV_BASE/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nccl/lib:/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export HF_TOKEN_PATH=/dev/null
cd /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl

mkdir -p results_street_nocn

python -c "
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import os

pipe = AutoPipelineForText2Image.from_pretrained(
    'stabilityai/sdxl-turbo',
    torch_dtype=torch.float16,
    variant='fp16',
).to('cuda')

output_dir = 'results_street_nocn'
prompt = 'a street portrait'
n = 200
seed = 42

for i in range(n):
    gen = torch.Generator(device='cuda').manual_seed(seed + i)
    img = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0, generator=gen).images[0]
    img.save(os.path.join(output_dir, f'img_{i:03d}.png'))
    if (i+1) % 20 == 0:
        print(f'  Generated {i+1}/{n}', flush=True)

# Make grid
from torchvision.utils import make_grid
import torchvision.transforms as T
imgs = [T.ToTensor()(Image.open(os.path.join(output_dir, f'img_{i:03d}.png'))) for i in range(n)]
grid = make_grid(imgs, nrow=20, padding=2)
T.ToPILImage()(grid).save(os.path.join(output_dir, 'grid.png'))
print('Saved grid.png')
"

echo "Done: $(date)"
