#!/bin/bash 
tmp1=$1
tmp2=$2
tmp3=$3

# file="${1:-}"
session=${tmp1:='ses-00A'} #defaults
smoothing=${tmp2:='no_smooth'}
dim=${tmp3:='20'}

subjID_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/dr_scripts/neurotranslate_abcd_subject_list.txt"
subjID_list=$(cat ${subjID_fpath})
curr_script_file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/dr_scripts/submit_dr_template"
prior_type="functional" #spatial or functional

########################## Write the input and the script #########################
for subject_id in ${subjID_list};
do
	subject_id="sub-${subject_id}"
    # echo "Dual Regressing Subject ID: ${subject_id}" # already get this in a way so comment it out

	# Create scripts
    SLURM_out="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/ICA_logs"
	if [ ! -d "${SLURM_out}" ]; then
		mkdir -p "${SLURM_out}"
	fi

	# OUT PATHS
	DR_out="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/brain_reps_datasets/infomap_prior_ABCDdr/dr_infomap_priors/ABCD_infomap20_${smoothing}/${prior_type}/dr_${subject_id}"
	if [ ! -d "${DR_out}" ]; then
		mkdir -p "${DR_out}"
	fi

	FILE="${DR_out}/timecourse.csv"
    if [ -f "$FILE" ]; then
        # echo "File $FILE exists."
        continue
    fi
	
	echo "\
\
#!/bin/bash
#SBATCH -J "dr_${subject_id}"
#SBATCH --output=${SLURM_out}/${subject_id}_%j.out
#SBATCH --error=${SLURM_out}/${subject_id}_%j.err
#SBATCH --partition=tier2_cpu 
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 4G 
#SBATCH --cpus-per-task 6
#SBATCH -t 0-00:30:00 

# Constant Paths
grp_map_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/data/ABCD/infomap_20net_prior/infomap_${prior_type}_priors.dscalar.nii"
python_script_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/dr_scripts/run_pyder_subject_fyzeen.py"

if [ "${smoothing}" = "no_smooth" ]; then
    INPUT_CIFTI="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/brain_reps_datasets/ABCD/cortex_data/${session}/no_smooth/${subject_id}_cortex_only_demean.dtseries.nii"
elif [ "${smoothing}" = "2mm" ] || [ "${smoothing}" = "4mm" ] || [ "${smoothing}" = "6mm" ]; then
    INPUT_CIFTI="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/brain_reps_datasets/ABCD/cortex_data/${session}/${smoothing}_smooth/${subject_id}_cortex_only_demean_smooth_${smoothing}.dtseries.nii"
else
    echo "Error: unrecognized smoothing value"
    exit 1
fi

out_timecourse_fpath="${DR_out}/timecourse.csv" 
out_map_fpath="${DR_out}/surf.nii"
fixed_out_map_fpath="${DR_out}/surf.dscalar.nii"

# Load Environment
module load fsl
export FSLOUTPUTTYPE=NIFTI_GZ

source activate neurotranslate

export DISPLAY=:1

python3 \${python_script_path} -func \"\${INPUT_CIFTI}\" -map \"\${out_map_fpath}\" -timecourse \"\${out_timecourse_fpath}\" -grp_map \"\${grp_map_fpath}\"
# Use workbench to fix surf.nii
module load workbench
wb_command -cifti-convert-to-scalar \"\${out_map_fpath}\" ROW \"\${fixed_out_map_fpath}\"

chmod -R 771 "${DR_out}"
\
" > "${curr_script_file}"

		# Overwrite submission script# Make script executable
		chmod 771 "${curr_script_file}" || { echo "Error changing the script permission!"; exit 1; }

		# Submit script
		sbatch "${curr_script_file}" || { echo "Error submitting jobs!"; exit 1; }

done 


