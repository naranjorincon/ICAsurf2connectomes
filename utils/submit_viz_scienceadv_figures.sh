#!/bin/bash
#SBATCH -J VIZ
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/VIZ_scienceadv.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/VIZ_scienceadv.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 8G# 30G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 0-12:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
utils_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/"
cd ${utils_path}

python3 ./viz_scienceadv_figures_new_010926.py
