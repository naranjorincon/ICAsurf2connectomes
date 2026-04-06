#!/bin/bash
#SBATCH -J TransposeABCD_tseries
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/TransposeABCD_tseries.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/TransposeABCD_tseries.err%j
#SBATCH --partition=tier2_cpu 
#SBATCH --account=janine_bijsterbosch
#SBATCH -t 0-23:00:00 

source activate neurotranslate

script_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/utils/transpose_ptseries.py"
# subjects_list_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/wapiaw26/subject_list_only5.txt"
subjects_list_path="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/qc/5min_pconn_subjects.txt"
untranspose_ts_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/new_ABCD_version/schaefer_d100/untranspose"
save_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/new_ABCD_version/schaefer_d100/transpose"
mkdir -p ${save_path}

python ${script_path} ${subjects_list_path} ${untranspose_ts_path} ${save_path}

chmod -R 771 /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/new_ABCD_version