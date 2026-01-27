#!/bin/bash
#SBATCH -J VIZICA
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/VIZICA.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/VIZICA.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 7G# 30G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 1-02:00:00  # might depend on epoch, approx 50epoch = 24 hours

module load workbench/2.0.1

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

scratch_path=/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch
netmat2surf_path=${scratch_path}/NeuroTranslate/surf2netmat
cd ${netmat2surf_path}

ico6_ica_viz_output=${netmat2surf_path}/utils/surfaces

model_type="SiT_LN"
yaml_loc="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/tmp_files/${model_type}/"
echo yaml_loc: ${yaml_loc}
# Go to prep folder and for each settings file, run the prep for it
condition="config*expICARECON*3*" # config_version_20251023_14h_55m_04s  config_version_20251023_14h_56m_24s
for yml_file in $(find "${yaml_loc}" -type f -name "${condition}.yml" -print); do
    echo All files match are: 
    find "${yaml_loc}" -type f -name "$condition.yml"

    # python3 ${netmat2surf_path}/utils/viz_krakenBGT_outputs_EXAMmodels.py "$yml_file"

    # viz output
    viz_output=${ico6_ica_viz_output}
    shape_file="test_pred*shape.gii"
    for outtt in $(find "${ico6_ica_viz_output}" -type f -name "${shape_file}" -print); do
        id_num="${outtt#*_sub-}"
        id_num="${id_num%%_*}"
        chnl_num="${outtt#*_chnl-}"
        chnl_num="${chnl_num%%_*}"
        echo "Doing channel test_pred: ${chnl_num}"
        echo "Pred ICA brains"
        echo ${id_num}

        wb_command -metric-resample ${outtt} "${ico6_ica_viz_output}/ico-6.L.surf.gii" "${ico6_ica_viz_output}/naranjo_ico.L.surf.gii" BARYCENTRIC "${ico6_ica_viz_output}/resamp_sub-${id_num}_chnl-${chnl_num}_test_pred_ico6.shape.gii"
        wb_command -set-structure "${ico6_ica_viz_output}/resamp_sub-${id_num}_chnl-${chnl_num}_test_pred_ico6.shape.gii" CORTEX_LEFT
        wb_command -metric-palette "${ico6_ica_viz_output}/resamp_sub-${id_num}_chnl-${chnl_num}_test_pred_ico6.shape.gii" MODE_USER_SCALE -pos-user 0.5 2.0 -neg-user -0.5 -2.0 -palette-name cool-warm -interpolate true -disp-pos true -disp-neg true -disp-zero true


    done

    shape_file="test_true*shape.gii"
    for outtt in $(find "${ico6_ica_viz_output}" -type f -name "${shape_file}" -print); do
        id_num="${outtt#*_sub-}"
        id_num="${id_num%%_*}"
        chnl_num="${outtt#*_chnl-}"
        chnl_num="${chnl_num%%_*}"
        echo "Doing channel test_true: ${chnl_num}"
        echo "True ICA brains"
        echo ${id_num}

        wb_command -metric-resample ${outtt} "${ico6_ica_viz_output}/ico-6.L.surf.gii" "${ico6_ica_viz_output}/naranjo_ico.L.surf.gii" BARYCENTRIC "${ico6_ica_viz_output}/resamp_sub-${id_num}_chnl-${chnl_num}_test_true_ico6.shape.gii"
        wb_command -set-structure "${ico6_ica_viz_output}/resamp_sub-${id_num}_chnl-${chnl_num}_test_true_ico6.shape.gii" CORTEX_LEFT
        wb_command -metric-palette "${ico6_ica_viz_output}/resamp_sub-${id_num}_chnl-${chnl_num}_test_true_ico6.shape.gii" MODE_USER_SCALE -pos-user 0.5 2.0 -neg-user -0.5 -2.0 -palette-name cool-warm -interpolate true -disp-pos true -disp-neg true -disp-zero true
        
    done
    
done

cd ${netmat2surf_path}/utils/surfaces
chmod -R 771 ./*