#!/bin/bash

#This script was meant to submit multiple jobs calling the python file below for each subject on a subject list.
subjID_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/dr_scripts/neurotranslate_abcd_subject_list.txt"
subjID_list=$(cat ${subjID_fpath})

# Script directory
script_dir="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/wapiaw26/scripts/data_correction"
curr_script_file=${script_dir}"/submit_template_subject_cortex_extraction" #will overwrite per jobsubmission/subject
output_dir="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/brain_reps_datasets/ABCD_v6/cortex_data"

########################## Write the input and the script #########################
for subjID in ${subjID_list};
do
    # Create scripts
    SLURM_out="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/subj_logs_cortex_extraction/"${subjID}"_cortex"
    echo "\
\
#!/bin/sh
#SBATCH --output=${SLURM_out}.out
#SBATCH --error=${SLURM_out}.err
#SBATCH --job-name=subj_cortex_extraction
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --time=0-00:30:00
#SBATCH --mem=20GB

src_path=${script_dir}/cortex_extraction.py

source activate neurotranslate
python \${src_path} ${subjID} -o ${output_dir} --overwrite_flag True

\
" > "${curr_script_file}"  # Overwrite submission script
    # Make script executable
    chmod +x "${curr_script_file}" || { echo "Error changing the script permission!"; exit 1; }

    # Submit script
    sbatch "${curr_script_file}" || { echo "Error submitting jobs!"; exit 1; }
done

# chmod -R 771 "${output_dir}" # what it does is it writes the above sbatch job ONTO the temporary file you have, since its necessity. That is why we give it execute permisison and then sbatch it!!