#!/bin/bash
#SBATCH -J 3121 #see notebook for naming convention
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/tr_kSiTBGT_corrEYE.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/tr_kSiTBGT_corrEYE.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 12G# 10G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 3-00:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

cd /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/tools
python3 ./SiT_BGT/krakenloss_SiTBGT_train.py ../config/SiT_BGT/hparams_krakenSiTBGT_corrEYE.yml

# conda activate # not specified means back to (base)