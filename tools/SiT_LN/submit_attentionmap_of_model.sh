#!/bin/bash
#SBATCH -J extract_attn
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/extract_attn.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/extract_attn.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 30G# 10G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 1-12:00:00  # might depend on epoch, approx 50epoch = 24 hours

# module load workbench/1.5.0
module load workbench/2.0.1

# wb_command #lets see what is output

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

scratch_path=/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch
netmat2surf_path=${scratch_path}/NeuroTranslate/surf2netmat
# cd ${netmat2surf_path} # go there

# where the config files are
model_type="SiT_LN"
cd "${netmat2surf_path}/tools/${model_type}/"

python3 extract_attention_maps.py

cd "../../outputs/"

chmod -R 771 ./