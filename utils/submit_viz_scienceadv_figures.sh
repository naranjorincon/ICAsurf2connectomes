#!/bin/bash
#SBATCH -J V100pcIM
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/VIZ_100pcIM_scienceadv.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/VIZ_100pcIM_scienceadv.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 10G# 30G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 0-12:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
utils_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/"
# cd ${utils_path}

# python3 ./viz_scienceadv_figures_new_010926.py #arggs are--> 0glasser, 1scgf300, 2schf100, 3ICAnetmatFull/INFOMAPnetmat
# order is dataset so ABCD or infomap_prior_ABCDdr, then 0 or 1 for partial, and then 0,1,2,3 for glasser, both schaefer, then ICAnetmat/INFOMAPnetmat
python3 ${utils_path}/viz_scienceadv_figures_new_071026.py infomap_prior_ABCDdr 1 0
