#!/bin/bash
#SBATCH -J te_kvlBGTSiT
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/te_kvlBGTSiT.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/te_kvlBGTSiT.err%j
#SBATCH --partition=tier2_gpu
#SBATCH --account=hcp
#SBATCH --gres=gpu:tesla_a100:1
# --mem=700G, tesla_a100
#SBATCH --mem-per-cpu 10G# 10G for base, 6.5G para small
#SBATCH --cpus-per-task 2
#SBATCH -t 2-00:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1000'

cd /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/tools
python3 ./BGT_SiT_VAE/te_kvlBGTswSiT.py ../config/BGT_SiT_VAE/hparams_kvlBGTswSiT.yml

# conda activate # not specified means back to (base)