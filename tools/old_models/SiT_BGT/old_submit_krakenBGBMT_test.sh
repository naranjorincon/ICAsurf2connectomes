#!/bin/bash
#SBATCH -J te_kBGBMT
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/te_kBGBMT.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/te_kBGBMT.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 5G # for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 1-00:00:00  # might depend on epoch, approx 50epoch = 24 hours

. /home/naranjorincon/miniconda3/bin/activate
source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

cd /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/tools
python3 krakenloss_BGBMT_patch_test.py

# conda activate # not specified means back to (base)
# below is here to make sure we can resample everything to match a brain mesh dim instead of patches/spheres
version="normMATrawICA"
model_name="krakenBGBMT_patch" 
model_details="d6h5_small_enc_d6h6_dec_adam_demeanL2"
sphere_dir=/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/model_out/schfd100_ICAd15/recon_spheres/kraken/${model_name}/${version}/${model_details}/
sphere_natives=/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/model_out/
count=$(find "$sphere_dir" -type f -name "t*" | wc -l) # both train and test begin with "t" so, should only count those and ignore resample
count_resamp=$(find "$sphere_dir" -type f -name "resamp*" | wc -l)
echo "There are $count files in the $dir directory."
# if [ ${count} -eq 1814 ]; then # make sure they all got created
# post to upsample again reconstructed ico-6 spheres into native resolution to make into brain shapes
module load workbench/1.5.0

cd ${sphere_dir} #update path as needed, raw_mat, deman, norm, fihser_z

for file in train_pred_L*.shape.gii; do
    # echo "Resampling from og sphere to ico-6"
    id_num="${file#*_sub-}"
    id_num="${id_num%%_*}"

    wb_command -metric-resample "train_pred_L_sub-${id_num}_ico6.shape.gii" "${sphere_natives}/ico-6.L.surf.gii" "${sphere_natives}/naranjo_ico.L.surf.gii" BARYCENTRIC "resamp_train_pred_${id_num}.L.shape.gii"
    # wb_command -metric-resample "train_pred_R_sub-${id_num}_ico6.shape.gii" "../../ico-6.R.surf.gii" "../../naranjo_ico.R.surf.gii" BARYCENTRIC "resamp_train_pred_${id_num}.R.shape.gii"
    wb_command -metric-resample "train_true_L_sub-${id_num}_ico6.shape.gii" "${sphere_natives}/ico-6.L.surf.gii" "${sphere_natives}/naranjo_ico.L.surf.gii" BARYCENTRIC "resamp_train_true_${id_num}.L.shape.gii"
    # wb_command -metric-resample "train_true_R_sub-${id_num}_ico6.shape.gii" "../../ico-6.R.surf.gii" "../../naranjo_ico.R.surf.gii" BARYCENTRIC "resamp_train_true_${id_num}.R.shape.gii"

    # wb_command -metric-resample "test_pred_L_sub-${id_num}_ico6.shape.gii" "${sphere_natives}/ico-6.L.surf.gii" "${sphere_natives}/naranjo_ico.L.surf.gii" BARYCENTRIC "resamp_test_pred_${id_num}.L.shape.gii"
    # # wb_command -metric-resample "test_pred_R_sub-${id_num}_ico6.shape.gii" "../../ico-6.R.surf.gii" "../../naranjo_ico.R.surf.gii" BARYCENTRIC "resamp_test_pred_${id_num}.R.shape.gii"
    # wb_command -metric-resample "test_true_L_sub-${id_num}_ico6.shape.gii" "${sphere_natives}/ico-6.L.surf.gii" "${sphere_natives}/naranjo_ico.L.surf.gii" BARYCENTRIC "resamp_test_true_${id_num}.L.shape.gii"
    # wb_command -metric-resample "test_true_R_sub-${id_num}_ico6.shape.gii" "../../ico-6.R.surf.gii" "../../naranjo_ico.R.surf.gii" BARYCENTRIC "resamp_test_pred_${id_num}.R.shape.gii"
done

for file in test_pred_L*.shape.gii; do
    id_num="${file#*_sub-}"
    id_num="${id_num%%_*}"

    wb_command -metric-resample "test_pred_L_sub-${id_num}_ico6.shape.gii" "${sphere_natives}/ico-6.L.surf.gii" "${sphere_natives}/naranjo_ico.L.surf.gii" BARYCENTRIC "resamp_test_pred_${id_num}.L.shape.gii"
    # wb_command -metric-resample "test_pred_R_sub-${id_num}_ico6.shape.gii" "../../ico-6.R.surf.gii" "../../naranjo_ico.R.surf.gii" BARYCENTRIC "resamp_test_pred_${id_num}.R.shape.gii"
    wb_command -metric-resample "test_true_L_sub-${id_num}_ico6.shape.gii" "${sphere_natives}/ico-6.L.surf.gii" "${sphere_natives}/naranjo_ico.L.surf.gii" BARYCENTRIC "resamp_test_true_${id_num}.L.shape.gii"
    # wb_command -metric-resample "test_true_R_sub-${id_num}_ico6.shape.gii" "../../ico-6.R.surf.gii" "../../naranjo_ico.R.surf.gii" BARYCENTRIC "resamp_test_pred_${id_num}.R.shape.gii"

done

# elif [ ${count_resamp} -eq ${count} ]; then
#     echo papi your stuff did not finish, chekea de nuevo "-\_(o.o)_/-"

# fi
