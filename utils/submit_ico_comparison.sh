#!/bin/bash
#SBATCH -J icocomp
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/icocomp.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/icocomp.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 20G# 30G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 0-09:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
cd /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils

python3 ICO2_ICO5_comparison_corr.py