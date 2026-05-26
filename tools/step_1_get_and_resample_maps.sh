#!/bin/bash
#SBATCH -J resamping_meshes
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/resamping_meshes.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/resamping_meshes.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch 
# --mem-per-cpu 10G
#SBATCH --cpus-per-task 10
#SBATCH -t 0-12:00:00  # might depend on epoch, approx 50epoch = 24 hours


# Adapted 05.29.2025 by Samuel Naranjo Rincon - orginally made for HCP but adapted to be for either HCP-YA or ABCD
#
# to use the transformer pipeline as in Dahan et al 2021 (https://arxiv.org/abs/2203.16414), I need to first resample our brain_maps 
# into a tessellated sphere with batches of equal size. Our brain maps are contained within the cifti file dtseries.nii
# which is a 'dense time series' cifti file - don't be confused by the nii, its NOT a volume file in the way regular nii or nii.gz
# files are. This script is meant ot be used to conver brain maps of ICA contained in the brain_representation folder
# into left and right spheres that represent left and right cortex, respectively. 
# info can be found in:
# 1. https://manpages.debian.org/testing/connectome-workbench/wb_command.1.en.html#Maps
# 2. https://www.humanconnectome.org/software/workbench-command/-surface-create-sphere
# 3. https://www.humanconnectome.org/software/workbench-command/-metric-resample
#

module load workbench  # /1.5.0, we don't have 1.5.0 anymore I think we now have the more updated one. Lets try anduse the updated one to see if it still works.

dataset="ABCD_v6" #ABCD or HCPYA or HCPYA_ABCDdr

# step 1: make a folder for where we will put our brian data
scratch_dir="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch" #technically not scratch, but no need to change variable name
surface_root="${scratch_dir}/NeuroTranslate/brain_reps_datasets/${dataset}"
mkdir -p "${surface_root}" #-p is option to make dir and anything along the way if it don't exist yet

# step 2: extract the L and R cortex data from the dense time series from each subj and move to the dir we made
if [ "${dataset}" == "HCPYA" ]; then
    mapdata="/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/ICA_reps/3T_HCP1200_MSMAll_d25_ts2_Z/component_maps"

    # comments to know what's happening
    echo "getting data from $mapdata and transferring to $surface_root"
    cd "$mapdata"
    count=0
    for file in *.dtseries*; do
        count=$[count +1 ];
        echo "Inside subject: ${count}"
        echo "extracting cortex from: $file"
        id_num=("${file[@]%.*.*}")

        wb_command -cifti-separate "$mapdata/$file" COLUMN -metric CORTEX_LEFT "$surface_root/${id_num}_subj_L_cortex.shape.gii"; 
        wb_command -cifti-separate "$mapdata/$file" COLUMN -metric CORTEX_RIGHT "$surface_root/${id_num}_subj_R_cortex.shape.gii"; 

    done

    # step 3: go back to the surface folder where the data was extracted to and create a sphere to project the cortex
    # data onto
    cd "${surface_root}"

    # now, lets make a sphere for each subject with 32492 verteces. Once we have this sphere, its easier to down sample or upsample
    # as is the case in the Dahan et al 2021 pipeline
    if test -e "./naranjo_ico.L.surf.gii"; then # if it DOES exist
        echo "Already made the ico6 template spheres"
    else
        wb_command -surface-create-sphere 32492 naranjo_ico.R.surf.gii
        wb_command -surface-flip-lr naranjo_ico.R.surf.gii naranjo_ico.L.surf.gii
        wb_command -set-structure naranjo_ico.R.surf.gii CORTEX_RIGHT
        wb_command -set-structure naranjo_ico.L.surf.gii CORTEX_LEFT
    fi

    # highly reccomend that you view 5 or 8 or 10 random subjects to make sure viz is good, you can do this 
    # with wb_view command on terminal and choose the surface file to project onto
    # and the data files to load in seperated by L and R hemispheres. I checked, and the 10 local subjects I have all ran correctly

    # now we have an R sphere and an L sphere of verteces num 32492, and we only need the one casue they all are in that surface space

    # step 4: map the brain_rep onto the 32k sphere and then upsample to the Dahan 2021 ico-6 sphere of 40962 verteces (ico-6)
    for file in *L_cortex.shape*; do
        id_num=("${file[@]%_*_*_*.*.*}")
        echo "Resampling from og sphere to ico-6: sub-${id_num}"

        wb_command -metric-resample "${id_num}_subj_L_cortex.shape.gii" naranjo_ico.L.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.L.surf.gii" BARYCENTRIC "resamp_${id_num}.L.shape.gii"
        wb_command -metric-resample "${id_num}_subj_R_cortex.shape.gii" naranjo_ico.R.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.R.surf.gii" BARYCENTRIC "resamp_${id_num}.R.shape.gii"

    done
    mkdir ./original_space
    mv *cortex* ./original_space

elif [ "${dataset}" == "ABCD" ]; then
    mapdata="${scratch_dir}/NeuroTranslate/generate_ICA/ABCD_ICA/ICAd15/groupICA15.dr/dr_output/" #dr_{SUBJID}/surf.dscalar.nii

    # comments to know what's happening
    echo "getting data from $mapdata and transferring to $surface_root"
    cd "$mapdata"
    # for all subjects makes their L and R data
    count=0
    for dd in dr*; do
        count=$[count +1 ];
        echo "Inside subject: ${count}"
        
        # cd ${dd} # inside subject
        dr_file="surf.dscalar.nii"
        id_num=("${dd#*_}") # subject file to transform
        wb_command -cifti-separate "${dd}/${dr_file}" COLUMN -metric CORTEX_LEFT "$surface_root/${id_num}_subj_L_cortex.shape.gii"; 
        wb_command -cifti-separate "${dd}/${dr_file}" COLUMN -metric CORTEX_RIGHT "$surface_root/${id_num}_subj_R_cortex.shape.gii"; 

    done

    # now we reformat from that space to ico6
    cd "$surface_root"
    if test -e "./naranjo_ico.L.surf.gii"; then # if it DOES exist
        echo "Already made the ico6 template spheres"
    else
        wb_command -surface-create-sphere 32492 naranjo_ico.R.surf.gii
        wb_command -surface-flip-lr naranjo_ico.R.surf.gii naranjo_ico.L.surf.gii
        wb_command -set-structure naranjo_ico.R.surf.gii CORTEX_RIGHT
        wb_command -set-structure naranjo_ico.L.surf.gii CORTEX_LEFT
    fi

    # step 4: map the brain_rep onto the 32k sphere and then upsample to the Dahan 2021 ico-6 sphere of 40962 verteces (ico-6)
    for file in *L_cortex.shape*; do
        id_num="${file%%_*}"
        echo "Resampling from og sphere to ico-6: sub-${id_num}"

        wb_command -metric-resample "${id_num}_subj_L_cortex.shape.gii" naranjo_ico.L.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.L.surf.gii" BARYCENTRIC "resamp_${id_num}.L.shape.gii"
        wb_command -metric-resample "${id_num}_subj_R_cortex.shape.gii" naranjo_ico.R.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.R.surf.gii" BARYCENTRIC "resamp_${id_num}.R.shape.gii"

    done
    mkdir ./original_space
    mv *cortex* ./original_space

elif [ "${dataset}" == "ABCD_v6" ]; then
    mapdata="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/ABCD_ICAd20_4mm/ABCD_ICAd20_4mm.dr/dr_output/neurotranslate_subset" #path to surface data
    map_dimension_icores="ICA_d20_ico06"
    output_path_resampled_spheres="${surface_root}/ICA_maps/${map_dimension_icores}"
    mkdir -p "${output_path_resampled_spheres}"

    # comments to know what's happening
    echo "getting data from ${mapdata} and transferring to ${surface_root}"
    cd "${mapdata}"
    # for all subjects makes their L and R data
    count=0
    for dd in dr*; do
        count=$[count +1 ];
        echo "Inside subject: ${count}"
        
        dr_file="surf.dscalar.nii"
        id_num=("${dd#*_}") # subject file to transform
        wb_command -cifti-separate "${dd}/${dr_file}" COLUMN -metric CORTEX_LEFT "${output_path_resampled_spheres}/original_${id_num}_subj_L_cortex.shape.gii"; 
        wb_command -cifti-separate "${dd}/${dr_file}" COLUMN -metric CORTEX_RIGHT "${output_path_resampled_spheres}/original_${id_num}_subj_R_cortex.shape.gii"; 

    done

    # now we reformat from that space to ico6
    template_ico6_spehre_path="${scratch_dir}/NeuroTranslate/surf2netmat/surfaces"
    if test -e "${template_ico6_spehre_path}/naranjo_ico.L.surf.gii"; then # if it DOES exist
        echo "Already made the ico6 template spheres"
    else #if it does not then make one
        wb_command -surface-create-sphere 32492 naranjo_ico.R.surf.gii
        wb_command -surface-flip-lr naranjo_ico.R.surf.gii naranjo_ico.L.surf.gii
        wb_command -set-structure naranjo_ico.L.surf.gii CORTEX_LEFT
        wb_command -set-structure naranjo_ico.R.surf.gii CORTEX_RIGHT
    fi

    cd "${output_path_resampled_spheres}"
    # step 4: map the brain_rep onto the 32k sphere and then upsample to the Dahan 2021 ico-6 sphere of 40962 verteces (ico-6)
    ico_res="6"
    for file in *L_cortex.shape*; do #for each left file, there is a right one so this should be good enough
        id_num="${file%%_*}"
        echo "Resampling from og sphere to ico-${ico_res}: sub-${id_num}"

        wb_command -metric-resample "${id_num}_subj_L_cortex.shape.gii" naranjo_ico.L.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.L.surf.gii" BARYCENTRIC "resamp_${id_num}.L.shape.gii"
        wb_command -metric-resample "${id_num}_subj_R_cortex.shape.gii" naranjo_ico.R.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.R.surf.gii" BARYCENTRIC "resamp_${id_num}.R.shape.gii"

    done

    #to not over load and have many copies per subject, lets reset and remove the original copies s.t. we only have the ico-version we created
    rm -rf "${output_path_resampled_spheres}/original*"
    #below not necessary anymore, its what I used to do but then data would accumulate and I wouldn't even use it.
    # mkdir ./original_space
    # mv *cortex* ./original_space

elif [ "${dataset}" == "HCPYA_ABCDdr" ]; then
    mapdata="${scratch_dir}/NeuroTranslate/generate_ICA/HCPYA_ICA/ICAd15_dr/dr_output/" #dr_{SUBJID}/surf.dscalar.nii

    # comments to know what's happening
    echo "getting data from $mapdata and transferring to $surface_root for dataset $dataset"
    cd "$mapdata"
    # for all subjects makes their L and R data
    count=0
    for dd in dr*; do
        count=$[count +1 ];
        echo "Inside subject: ${count}"
        
        # cd ${dd} # inside subject
        dr_file="surf.dscalar.nii"
        id_num=("${dd#*_}") # subject file to transform
        if test -e "$surface_root/${id_num}_subj_R_cortex.shape.gii"; then #left is done first, so if they have R then both. if not, redo.
            echo "${id_num} already has extracted surface. L and R."
        else
            wb_command -cifti-separate "${dd}/${dr_file}" COLUMN -metric CORTEX_LEFT "$surface_root/${id_num}_subj_L_cortex.shape.gii"; 
            wb_command -cifti-separate "${dd}/${dr_file}" COLUMN -metric CORTEX_RIGHT "$surface_root/${id_num}_subj_R_cortex.shape.gii"; 
        fi

    done

    # now we reformat from that space to ico6
    cd "$surface_root"
    if test -e "./naranjo_ico.L.surf.gii"; then # if it DOES exist
        echo "Already made the ico6 template spheres"
    else
        wb_command -surface-create-sphere 32492 naranjo_ico.R.surf.gii
        wb_command -surface-flip-lr naranjo_ico.R.surf.gii naranjo_ico.L.surf.gii
        wb_command -set-structure naranjo_ico.R.surf.gii CORTEX_RIGHT
        wb_command -set-structure naranjo_ico.L.surf.gii CORTEX_LEFT
    fi

    # step 4: map the brain_rep onto the 32k sphere and then upsample to the Dahan 2021 ico-6 sphere of 40962 verteces (ico-6)
    for file in *L_cortex.shape*; do
        id_num="${file%%_*}"
        echo "Resampling from og sphere to ico-6: sub-${id_num}"

        if test -e "./resamp_${id_num}.R.shape.gii"; then
            echo "${id_num} already has resampled spheres."       
        else
            wb_command -metric-resample "${id_num}_subj_L_cortex.shape.gii" naranjo_ico.L.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.L.surf.gii" BARYCENTRIC "resamp_${id_num}.L.shape.gii"
            wb_command -metric-resample "${id_num}_subj_R_cortex.shape.gii" naranjo_ico.R.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.R.surf.gii" BARYCENTRIC "resamp_${id_num}.R.shape.gii"
        fi

    done

    mkdir ./original_space
    mv *cortex* ./original_space    
else
    echo "That dataset is invalid/does not exit"
    # unknown_dataset_glaf=T
fi

# mkdir ./original_space
# mv *cortex* ./original_space
chmod -R 771 "${surface_root}"
