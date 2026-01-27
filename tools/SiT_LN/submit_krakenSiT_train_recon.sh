#!/bin/bash
#SBATCH -J 1L_fulldemean
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/tr_kSiT_recon.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/tr_kSiT_recon.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 55G# 10G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 3-12:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

scratch_path=/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch
netmat2surf_path=${scratch_path}/NeuroTranslate/surf2netmat
cd ${netmat2surf_path} # go there

# where the config files are
model_type="SiT_LN"
yaml_loc="${netmat2surf_path}/config/${model_type}"
cd ${yaml_loc} #should be in config/../ path

condition="hparams_krakenSiTLN_recon.yml" # kept hparams at begining bc wildcard at start will include .hparams file(s)
chosen_param_config=$(find "$yaml_loc" -type f -name "$condition")
echo chosen param file is: ${chosen_param_config}

config_model_name=$(grep -A11 'transformer' ${chosen_param_config} | tail -n1 | awk '{ print $2}')
echo Model name and config file to freeze: ${config_model_name}

# date_time_stamp=$(date +"%Y%m%d_%Hh_%Mm_%Ss")
mkdir -p ${netmat2surf_path}/tmp_files/${model_type}
touch ${netmat2surf_path}/tmp_files/${model_type}/config_${config_model_name}.yml
cp ${chosen_param_config} ${netmat2surf_path}/tmp_files/${model_type}/config_${config_model_name}.yml

echo "param file created, copied, and saved at tmp path! If you want to submit another job, go ahead."

echo "Using --> ${netmat2surf_path}/tmp_files/${model_type}/config_${config_model_name}.yml" # ${chosen_param_config}

python3 ${netmat2surf_path}/tools/${model_type}/krakenloss_SiT_train.py ${netmat2surf_path}/tmp_files/${model_type}/config_${config_model_name}.yml #${chosen_param_config}

# after training and test, visualize it
python3 ${netmat2surf_path}/utils/viz_krakenBGT_outputs_EXAMmodels.py ${netmat2surf_path}/tmp_files/${model_type}/config_${config_model_name}.yml

# then look at downstream analyses
python3 ${netmat2surf_path}/utils/downstream_analyses.py ${netmat2surf_path}/tmp_files/${model_type}/config_${config_model_name}.yml

chmod -R 774 ${netmat2surf_path}/tmp_files/

echo DONE
