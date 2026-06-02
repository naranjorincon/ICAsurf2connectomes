#!/bin/bash 
set -o nounset 

# =================================================
#          Parallel PYDR Submission Script
# =================================================
#adapted from Fyzeen's original script - path names changed

# DESCRIPTION
# -----------
# This Bash script automates and parallelizes the submission of 
# subject-level dual regression jobs to the CHPC. It loops 
# through a list of subject IDs and submits one SLURM job per subject 
# using the run_pydr_subject.py script. Each job performs dual regression 
# on the subject's fMRI data using a group ICA map.


# USAGE
# -----
# bash run_dr_parallel.sh [text_file_subj_list] [session] [smoothing] [run]
# 
# bash run_dr_parallel.sh [full_path] ses-all 4mm 14

# USER INPUTS
# --------------------
#file
#	full path to txt file with subject ids (one per line).

#session
#	concat or session (ex: ses-all, ses-00A)

#smoothing
#	smoothing level (ex: 2mm, no_smooth)

#dim
#	dimension of ICA run ( ex: 14, 20, 25)

# INPUT TO BASH SCRIPT
# --------------------
# job_name: 
#     A unique job name is generated per subject, used for tracking SLURM jobs.

# sbatch_fpath: 
#     Path where each subject’s SLURM submission script will be saved.

# DR_out: 
#     Output directory for storing subject-level results:
#         - Dual regression time series
#         - Subject-specific spatial maps
#         - Fixed CIFTI files


# INPUT TO EACH SBATCH SCRIPT
# ---------------------------
# func_fpath:
#     Path to the subject's functional MRI CIFTI time series file (.dtseries.nii).

# grp_map_fpath:
#     Path to group-level spatial ICA maps in CIFTI format for dual regression.

# python_script_path:
#     Path to the run_pydr_subject.py script.

# out_timecourse_fpath:
#     Output path for time series file (from stage 1 of dual regression). Should be a .csv or .txt file

# out_map_fpath:
#     Output path for subject-specific spatial maps. Should be a .nii file

# fixed_out_map_fpath:
#     Final output file in .dscalar.nii format, fixed with wb_command to ensure proper structure.


# OUTPUTS (PER SUBJECT)
# ---------------------
# - timecourse.csv:     Stage 1 regression output (time series).
# - surf.nii:           Stage 2 output (subject-specific spatial maps).
# - surf.dscalar.nii:   Fixed version of spatial maps using wb_command.

# Additional SLURM files per subject:
# - dr_ABCD_ICAd15.out<jobid>
# - dr_ABCD_ICAd15.err<jobid>
# - do_ABCD_pydr_Subj<subject_id> (submission script)


# NOTES
# -----
# - Existing SBATCH scripts for each subject are overwritten if they already exist.
# - The script ensures the output directory exists or creates it.
# - SLURM jobs are submitted with required modules (e.g., FSL, Workbench).
# - Ensure the CONDA environment you load includes the following packages: "click", "nibabel", "numpy", "pandas", "pyyaml", "scipy"

# file="${1:-}"
session="${1:-}"
smoothing="${2:-}"
dim="${3:-}"

# mkdir -p "/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/ABCD_ICAd${dim}_${smoothing}/ABCD_ICAd${dim}_${smoothing}.dr/scripts"
# mkdir -p "/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/ABCD_ICAd${dim}_${smoothing}/ABCD_ICAd${dim}_${smoothing}.dr/dr_output/neurotranslate_subset"

# sbatch_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/dr_scripts/do_template" #will get rewritten
# while IFS= read -r subject_id; 
subjID_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/dr_scripts/neurotranslate_abcd_subject_list.txt"
subjID_list=$(cat ${subjID_fpath})
curr_script_file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/dr_scripts/submit_dr_template"

########################## Write the input and the script #########################
for subject_id in ${subjID_list};
do

	subject_id="sub-${subject_id}"
    # echo "Dual Regressing Subject ID: ${subject_id}" # already get this in a way so comment it out

	# Create scripts
    SLURM_out="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/ICA_logs/"
	if [ ! -d "${SLURM_out}" ]; then
		mkdir -p "${SLURM_out}"
	fi

	# OUT PATHS
	DR_out="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/brain_reps_datasets/ABCD_v6/dr_infomap_priors/ABCD_infomap20_${smoothing}/spatial/dr_${subject_id}" 
	if [ ! -d "${DR_out}" ]; then
		mkdir -p "${DR_out}"
	fi

    # FILE="" #/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/gradient_maps/gradmaps_d14/ses-all/emb_${subject_id}_grad_smooth_4mm.dtseries.nii     
    # if [ -f "$FILE" ]; then
    #     echo "File $FILE exists."
    #     continue
    # fi

	# subject_id=$(echo "${subject_id}" | tr -d '\r')

	# job_name=ABCD_dr_sub-${subject_id}

	# batch_out_log_root='/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/ICA_logs'
	# if [ ! -d "${batch_out_log_root}" ]; then
	# 	mkdir -p "${batch_out_log_root}"
	# fi
	
	echo "\
\
#!/bin/bash
#SBATCH -J "dr_${subject_id}"
#SBATCH --output=${subject_id}_${SLURM_out}_%j.out
#SBATCH --error=${subject_id}_${SLURM_out}_%j.err
#SBATCH --partition=tier2_cpu 
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 4G 
#SBATCH --cpus-per-task 6
#SBATCH -t 0-01:00:00 

# Constant Paths
# base_dir="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA" 
grp_map_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/data/ABCD/infomap_20net_prior/infomap_spatial_priors.dscalar.nii"
python_script_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/dr_scripts/run_pyder_subject_fyzeen.py"

if [ "${smoothing}" = "no_smooth" ]; then
    INPUT_CIFTI="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/brain_reps_datasets/ABCD_v6/cortex_data/${session}/no_smooth/${subject_id}_cortex_only_demean.dtseries.nii"
elif [ "${smoothing}" = "2mm" ] || [ "${smoothing}" = "4mm" ] || [ "${smoothing}" = "6mm" ]; then
    INPUT_CIFTI="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/brain_reps_datasets/ABCD_v6/cortex_data/${session}/${smoothing}_smooth/${subject_id}_cortex_only_demean_smooth_${smoothing}.dtseries.nii"
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

done #< "${file}"


