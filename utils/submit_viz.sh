#!/bin/bash
#SBATCH -J VIZ
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/VIZ.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/VIZ.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 3G# 30G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 0-02:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
surf2netmat_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/"
cd ${surf2netmat_path}

yaml_loc="${surf2netmat_path}/config/SiT_LN" #SiT_LN, SiT_LN_VAE
condition="hparams_krakenSiTLN_recon.yml" # kept hparams at begining bc wildcard at start will include .hparams file(s)
chosen_param_config=$(find "$yaml_loc" -type f -name "$condition")
echo chosen param file is: ${chosen_param_config}

python3 ./utils/viz_krakenBGT_outputs_EXAMmodels.py ${chosen_param_config}
# python3 viz_latentspace.py 
