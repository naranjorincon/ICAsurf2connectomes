#!/bin/bash
#SBATCH -J te_top2netmat
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/te_kSiT_recon.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/te_kSiT_recon.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 16G# 18GB for others and 30GB for bilateral
#SBATCH --cpus-per-task 15
#SBATCH -t 0-02:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

scratch_path=/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch
netmat2surf_path=${scratch_path}/NeuroTranslate/surf2netmat
cd ${netmat2surf_path} # go there

# where the config files are
model_type="SiT_LN"
config_model_name="070426_d6h3_tiny_adamW_cosinedecay_1L_full_INFOMAPd20_schfd300_demean_wGelu_ico02"
config_path="${netmat2surf_path}/tmp_files/${model_type}/config_${config_model_name}.yml"
######
# config_model_name="recon"
# config_path="${netmat2surf_path}/config/${model_type}/hparams_SiTLN_${config_model_name}.yml" 
######

echo "Using ${config_model_name}"
python3 ${netmat2surf_path}/tools/${model_type}/top2conn_test.py ${config_path}
## after training and test, visualize it
python3 ${netmat2surf_path}/utils/viz_top2conn_outputs_EXAMmodels.py ${config_path}
# # then look at downstream analyses
python3 ${netmat2surf_path}/utils/downstream_analyses.py ${config_path}
