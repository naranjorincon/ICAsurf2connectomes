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
#bash run_dr_parallel.sh [full_path] ses-all 4mm 14

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

file="${1:-}"
session="${2:-}"
smoothing="${3:-}"
dim="${4:-}"

mkdir -p "/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/ABCD_ICAd${dim}_${smoothing}/ABCD_ICAd${dim}_${smoothing}.dr/scripts"
mkdir -p "/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/ABCD_ICAd${dim}_${smoothing}/ABCD_ICAd${dim}_${smoothing}.dr/dr_output/neurotranslate_subset"

while IFS= read -r subject_id; 
do
	subject_id=$(echo "$subject_id" | tr -d '\r')
    echo "Dual Regressing Subject ID: $subject_id"

	job_name=ABCD_dr_${subject_id}
	sbatch_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/ABCD_ICAd${dim}_${smoothing}/ABCD_ICAd${dim}_${smoothing}.dr/scripts/do_${job_name}"

	if compgen -G ${sbatch_fpath} 
	then 
			rm $sbatch_fpath 
	fi 

	# Print DR_out for debugging
    DR_out="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/ABCD_ICAd${dim}_${smoothing}/ABCD_ICAd${dim}_${smoothing}.dr/dr_output/neurotranslate_subset/dr_${subject_id}"
    echo "DR_out: ${DR_out}"

	echo "\
\
#!/bin/bash
#SBATCH -J "dr-${subject_id}"
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/logs/dr_ABCD_ICAd${dim}_${smoothing}_out.%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/logs/dr_ABCD_ICAd${dim}_${smoothing}_err.%j
#SBATCH --partition=tier2_cpu 
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 4G 
#SBATCH --cpus-per-task 6
#SBATCH -t 0-12:00:00 

# Constant Paths
base_dir="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA" 
grp_map_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/ABCD_ICAd${dim}_${smoothing}/melodic_IC.nii"
python_script_path="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/DR/run_pyder_subject_fyzeen.py"


if [ "${smoothing}" = "no_smooth" ]; then
    INPUT_CIFTI="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/cortex_only_data/${session}/no_smooth/${subject_id}_cortex_only_demean.dtseries.nii"
elif [ "${smoothing}" = "2mm" ] || [ "${smoothing}" = "4mm" ] || [ "${smoothing}" = "6mm" ]; then
    INPUT_CIFTI="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/cortex_only_data/${session}/${smoothing}_smooth/${subject_id}_cortex_only_demean_smooth_${smoothing}.dtseries.nii"
else
    echo "Error: unrecognized smoothing value '${smoothing}'"
    exit 1
fi

# OUT PATHS
DR_out="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/ABCD_ICAd${dim}_${smoothing}/ABCD_ICAd${dim}_${smoothing}.dr/dr_output/neurotranslate_subset/dr_${subject_id}" 

if [ ! -d "${DR_out}" ]; then
	mkdir -p "${DR_out}"
fi

out_timecourse_fpath="${DR_out}/timecourse.csv" 
out_map_fpath="${DR_out}/surf.nii"
fixed_out_map_fpath="${DR_out}/surf.dscalar.nii"


# Load Environment
module load fsl
export FSLOUTPUTTYPE=NIFTI_GZ

#source ~/.bashrc
#. ~/miniconda3/bin/activate

source /scratch/crown/miniconda3/etc/profile.d/conda.sh
conda activate DR_conda

export DISPLAY=:1

python3 \${python_script_path} -func \"\${INPUT_CIFTI}\" -map \"\${out_map_fpath}\" -timecourse \"\${out_timecourse_fpath}\" -grp_map \"\${grp_map_fpath}\"
# Use workbench to fix surf.nii
module load workbench
wb_command -cifti-convert-to-scalar \"\${out_map_fpath}\" ROW \"\${fixed_out_map_fpath}\"

chmod -R 771 "${DR_out}"
\
" > "${sbatch_fpath}"

		# Overwrite submission script# Make script executable
		chmod 771 "${sbatch_fpath}" || { echo "Error changing the script permission!"; exit 1; }

		# Submit script
		sbatch "${sbatch_fpath}" || { echo "Error submitting jobs!"; exit 1; }

done < "${file}"


