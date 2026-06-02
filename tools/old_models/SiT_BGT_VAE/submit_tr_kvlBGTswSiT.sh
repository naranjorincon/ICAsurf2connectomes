#!/bin/bash
#SBATCH -J tr_unet
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/tr_unetVAEBGT_swMSSiT.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/tr_unetVAEBGT_swMSSiT.err%j
#SBATCH --partition=tier2_gpu
#SBATCH --account=hcp
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --mem-per-cpu 30G# 10G for base, 6.5G para small
#SBATCH --cpus-per-task 15
#SBATCH -t 4-10:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:10000'

cd /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/tools
python3 ./BGT_SiT_VAE/krakenloss_VAE_BGT_swMSSiT_train.py ../config/BGT_SiT_VAE/hparams_kvlBGTswSiT.yml

# conda activate # not specified means back to (base)