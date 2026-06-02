#!/bin/bash
#SBATCH -J 3edlz
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/tr_kSiTBGT_latentsize.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/tr_kSiTBGT_latentsize.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 15G# small latent is like only 40GB
#SBATCH --cpus-per-task 10
#SBATCH -t 4-00:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

cd /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/tools
python3 ./SiT_BGT/krakenloss_SiTBGT_train.py ../config/SiT_BGT/hparams_krakenSiTBGT_latentsize.yml

# conda activate # not specified means back to (base)