#!/bin/bash
#SBATCH -J ts2netmats
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/logs/ts2netmats.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/logs/ts2netmats.err%j
#SBATCH --partition=tier2_cpu 
#SBATCH --account=janine_bijsterbosch 
#SBATCH -t 0-24:00:00 

module load fsl

script_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/utils/ts2netmats.py"
# subjects_list_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/wapiaw26/subject_list_only5.txt"
subjects_list_path="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/qc/5min_pconn_subjects.txt"

parcellation_type="schaefer_d100"
dir_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/new_ABCD_version/${parcellation_type}/transpose"
Fnetmats_output="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/new_ABCD_version/${parcellation_type}/netmats"
Pnetmats_output="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/new_ABCD_version/${parcellation_type}/partial_netmats"
mkdir -p ${Fnetmats_output}
mkdir -p ${Pnetmats_output}

fslipython ${script_path} ${dir_path} ${Fnetmats_output} ${Pnetmats_output} ${subjects_list_path}

chmod -R 771 /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/new_ABCD_version

#Visualize some people to qa check that this went well
source activate neurotranslate
script_path_python="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/create_netmats_from_ts"
python ${script_path_python}/qa_netmats.py
