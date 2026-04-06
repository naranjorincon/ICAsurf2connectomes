#!/bin/bash
#SBATCH -J "ExtractABCD_Timeseries"
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/ExtractABCD_Timeseries_Schaefer200.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/ExtractABCD_Timeseries_Schaefer200.err%j
#SBATCH --partition=tier2_cpu 
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 4G 
#SBATCH --cpus-per-task 6
#SBATCH -t 0-12:00:00 

module load workbench

# need subj ID file
file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/generate_ICA/ABCD_ICA_subjids.txt"

# Schaefer100
parcel_file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer100/Schaefer2018_100Parcels_17Networks_order.dlabel.nii"
dir_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer100/timeseries"

# Schaefer200
#parcel_file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer200/Schaefer2018_200Parcels_17Networks_order.dlabel.nii"
#dir_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer200/timeseries"

# Schaefer300
#parcel_file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer300/Schaefer2018_300Parcels_17Networks_order.dlabel.nii"
#dir_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer300/timeseries"

# Glasser360
#parcel_file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/glasser360/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors_210P_Orig.32k_fs_LR.dlabel.nii"
#dir_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/glasser360/timeseries"

while IFS= read -r subject_id; 
do
    subject_id=$(echo "$subject_id" | tr -d '\r')
    echo "Extracting Timeseries For: $subject_id"

    wb_command -cifti-parcellate \
        /ceph/chpc/rcif_datasets/abcd/abcd_collection3165/derivatives/abcd-hcp-pipeline/sub-${subject_id}/ses-baselineYear1Arm1/files/MNINonLinear/Results/task-rest_DCANBOLDProc_v4.0.0_Atlas.dtseries.nii \
        ${parcel_file} \
        COLUMN \
        ${dir_path}/${subject_id}.ptseries.nii

    wb_command -cifti-convert -to-text \
        ${dir_path}/${subject_id}.ptseries.nii \
        ${dir_path}/untranspose_${subject_id}.txt

done < "${file}"

chmod 771 ${dir_path}/*