#!/bin/bash
#SBATCH -J 1022
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/te_kSiT.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/te_kSiT.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 40G # 10G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 0-08:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

scratch_path=/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch
netmat2surf_path=${scratch_path}/NeuroTranslate/surf2netmat

cd ${scratch_path}/NeuroTranslate/surf2netmat/tools
model_type="SiT_LN"
yaml_loc="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/tmp_files/${model_type}/"
echo yaml_loc: $yaml_loc
# Go to prep folder and for each settings file, run the prep for it
condition="config_011426*ico04" # config_version_20251023_14h_55m_04s  config_version_20251023_14h_56m_24s
for yml_file in $(find "$yaml_loc" -type f -name "$condition.yml" -print); do
    echo All files match are: 
    find "$yaml_loc" -type f -name "$condition.yml"

    echo Runnign This preprocessing settings file: "$yml_file"
    python3 ./${model_type}/krakenloss_SiT_test.py "$yml_file"

    # echo Finished with yml file: $yml_file
    python3 ${netmat2surf_path}/utils/viz_krakenBGT_outputs_EXAMmodels.py "$yml_file"

    # also downstream analyses
    python3 ${netmat2surf_path}/utils/downstream_analyses.py "$yml_file"

    chmod -R 774 ${netmat2surf_path}/tmp_files/

done

# sed -i 's/\bRHO\b/MSE/g' ./*.yml # this command goes to current dir, changes any RHO->MSE in any yaml file, logic can be used for other changes

# find ../tools/SiT_* -type f -name "submit*test.sh" -exec sbatch {} \;

