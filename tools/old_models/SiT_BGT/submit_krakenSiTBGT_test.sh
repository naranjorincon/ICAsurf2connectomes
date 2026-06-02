#!/bin/bash
#SBATCH -J 3022
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/te_kSiTBGT.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/te_kSiTBGT.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 16G # 10G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 0-05:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

scratch_path=/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch

cd ${scratch_path}/NeuroTranslate/surf2netmat/tools
yaml_loc="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/config/SiT_BGT"
# Go to prep folder and for each settings file, run the prep for it
condition="hparams*recon*" # if hparams* then all config files are being tested
for yml_file in $(find "$yaml_loc" -type f -name "$condition.yml" -print); do
    echo Runnign This preprocessing settings file: "$yml_file"
    python3 ./SiT_BGT/krakenloss_SiTBGT_test.py "$yml_file"

    echo Finished with yml file: $yml_file

done

# find ../tools/SiT_* -type f -name "submit*test.sh" -exec sbatch {} \;

