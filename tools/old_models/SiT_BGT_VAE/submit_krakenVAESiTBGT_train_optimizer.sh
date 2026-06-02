#!/bin/bash
#SBATCH -J 4621
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/tr_kvSiTBGT_opt.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/tr_kvSiTBGT_opt.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 15G# 10G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 4-00:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

cd /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/tools
python3 ./SiT_BGT_VAE/krakenloss_VAESiTBGT_train.py ../config/SiT_BGT_VAE/hparams_krakenSiTBGT_VAE_optimizer.yml

# conda activate # not specified means back to (base)