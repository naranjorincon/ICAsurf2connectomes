#!/bin/bash
#SBATCH -J tr_kSiTBGT
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/tr_kSiTBGT.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/tr_kSiTBGT.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 7G # 10G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 4-0:00:00  # might depend on epoch, approx 50epoch = 24 hours

. /home/naranjorincon/miniconda3/bin/activate
source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

cd /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/tools
python3 ./BGT_SiT/krakenloss_SiTBGT_train.py ../config/BGT_SiT/old_hparams_krakenSiTBGT.yml

# conda activate # not specified means back to (base)