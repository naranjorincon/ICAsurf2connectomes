import os
import sys
sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

import torch
import numpy as np
import pandas as pd
import argparse
import yaml
from models import models
from utils.utils import *
from utils import utils
from utils import functions_train
import torch.optim as optim 
import glob

def fcn_validate(model, val_loader, mean_train_label, device, VAE_flag, netmat_prep_choice):
    model.eval()
    model.to(device)
    mse_val_list = []
    mae_val_list = []
    demean_corr_val_list = []
    orig_corr_val_list = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets = data[0].to(device), data[1].to(device)#.squeeze()
            
            if VAE_flag:
                pred, latent, log_latent = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                del latent, inputs, log_latent
            else:
                pred, latent = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                del latent, inputs

            pred = pred.detach().numpy() # picks first element of pred tuble, so preds which is all we care for validation
            targets = targets.detach().numpy()

            mae = np.mean(np.abs( (targets - pred) ), axis=0) # mean of this batch
            mae_val_list.append(mae)
            mse = np.mean(( (targets - pred)**2 ), axis=0)
            mse_val_list.append(mse)

            if "demean" in netmat_prep_choice: # if doing any demeaning, then predictions are of original and we must demean here
                # print("Validation, using demean version.")
                tr_corr_demean = np.corrcoef(targets, pred) #[subj*2 x subj*2] matrix where quadrant1 = target_target, quad2=target_pred, quad3=pred_target, quad4=pred_pred
                split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
                top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
                demean_corr_val_list.append(np.diag(top_right_quad))

                tr_corr_org = np.corrcoef((targets+mean_train_label), (pred+mean_train_label)) # going to be low-ish cause 256->mesh size sphere but curious
                split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
                top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
                orig_corr_val_list.append(np.diag(top_right_quad))
            else: # if data was preped by demeaning, then for original need to readd mean
                tr_corr_demean = np.corrcoef((targets-mean_train_label), (pred-mean_train_label))
                split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
                top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
                demean_corr_val_list.append(np.diag(top_right_quad))

                tr_corr_org = np.corrcoef(targets, pred)# going to be low-ish cause 256->mesh size sphere but curious
                split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
                top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
                orig_corr_val_list.append(np.diag(top_right_quad))
    
    # print(f"List length: {len(demean_corr_val_list)} I think it should be batches?")
    across_sub_mae_mean = np.mean(mae_val_list) # mean across batches
    across_sub_mae_std = np.std(mae_val_list)
    across_sub_mse_mean = np.mean(mse_val_list)
    across_sub_mse_std = np.std(mse_val_list)
    # because of batching, some in the list are different size so make into whole array
    upto_n_minus1 = np.asarray(demean_corr_val_list[:-1]).squeeze() # all upto last item, do that seperate then concat
    # print(f"upto_minus1_shape: {upto_n_minus1.shape}")
    upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
    n_minus_1 = np.asarray(demean_corr_val_list[-1])[np.newaxis,:] 
    # print(f"n_minus_1: {n_minus_1.shape}")
    val_corr_demean_flat = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
    across_sub_corr_demean = np.mean(val_corr_demean_flat)
    across_sub_corr_demean_std = np.std(val_corr_demean_flat)

    # same for original corr values
    upto_n_minus1 = np.asarray(orig_corr_val_list[:-1]).squeeze() # all upto last item, do that seperate then concat
    upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
    n_minus_1 = np.asarray(orig_corr_val_list[-1])[np.newaxis,:] # adding dim cause above reshape gives dim to index 0, so shape is 1xDIMS so adding this also makes this 1xDIMs
    val_corr_org_flat = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
    across_sub_corr_org = np.mean(val_corr_org_flat)
    across_sub_corr_org_std = np.std(val_corr_org_flat)

    return across_sub_mae_mean, across_sub_mae_std, across_sub_mse_mean, across_sub_mse_std, across_sub_corr_demean, across_sub_corr_demean_std, across_sub_corr_org, across_sub_corr_org_std

def whole_model_arch(config):
    model_out_root = config['logging']['model_out_root']
    fcn_train = getattr(functions_train, config['training']['fcn_train'])  
    fcn_model_module = getattr(models, config['training']['fcn_model_to_use']) 
    flag_experiment_ICArecon = config['training']['flag_experiment_ICArecon']  
    netmat_prep_choice = config['training']['netmat_prep_choice']
    surf_prep_choice = config['training']['surf_prep_choice']
    dataset_choice = config['training']['dataset_choice']
    bilateral_condition = config['training']['bilateral_condition'] # both hemispheres instead of 1
    channel_specific_condition = config['training']['channel_specific_condition']
    specific_channel = config['training']['specific_channel']
    specific_channel_end = config['training']['specific_channel_end']
    overfit_condition = config['training']['overfit_condition']
    train_epoch_range = config['training']['epochs']
    val_epoch = config['training']['val_epoch']
    LR = config['training']['LR']
    batch_size = config['training']['bs']
    if overfit_condition:
        batch_size = batch_size // 4 #overfit tends to need smaller batch so that the script works fully. Dont belive me? Try. 
    parcellation_corr_type = config['training']['parcellation_corr_type']
    VAE_flag = config['training']['VAE_flag']
    from_parcellation = config['data']['from_parcellation']
    to_icamap = config['data']['to_icamap']
    translation= config['data']['translation']
    version = config['data']['version']
    model_type = config['data']['model_type']
    krak_mse_weight = config['transformer']['krak_mse_weight']
    krak_latent_weight = config['transformer']['krak_latent_weight']
    krak_corrEYE_weight = config['transformer']['krak_corrEYE_weight']
    rho_loss_weight = config['transformer']['rho_loss_weight']
    # get ico resolution
    try: #since this is a new thing added, putting this here so that previous yml files without it can run assuming they were done with ico02
        icores = config['data']['icores']
    except:
        # write_to_file("YML file does not have ico resolution information. Defaulting to ico-02", filepath=write_fpath)
        icores="2"
    # else:
        # write_to_file(f"Using ico-{icores} surf data.", filepath=write_fpath)

    if flag_experiment_ICArecon:
        model_details = config['transformer']['model_details'] + f'_chnl{specific_channel}' # if YES specific channel, then per channel save model with that channel
        write_fpath = config['logging']['sanity_file_pth'] + f'_ico-0{icores}' +  f'_chnl{specific_channel}' + '.print'
    else:
        model_details = config['transformer']['model_details'] # if not specicfic channel then save as normal
        write_fpath = config['logging']['sanity_file_pth'] + f'_ico-0{icores}' + '.print'

    write_to_file(f"Using ICAd15_ico0{icores} surf data.", filepath=write_fpath)
    write_to_file(f"details are: {model_details}", filepath=write_fpath)
    
    device = "cpu"
    best_mae = 1e+9
    best_mse = 1e+9
    best_demean_rho = int(-1 * 1e+9)
    TEST_FLAG = config['testing']['immediate_test_flag']
    te_batch_size = config['testing']['bs_test']

    folder_to_save_model = f'/home/naranjorincon/neurotranslate/surf2netmat/logs/{translation}/{dataset_choice}/{model_type}/{version}'
    folder_to_save_losses = f'{model_out_root}/{translation}/{dataset_choice}/{model_type}/{version}/{model_details}'
    # make necessary folders
    if not os.path.exists(folder_to_save_model):
        # Create the directory
        os.makedirs(folder_to_save_model)
    if not os.path.exists(folder_to_save_losses):
        # Create the directory
        os.makedirs(folder_to_save_losses)

    data_root_path = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"

    # for TESTING #
    chosen_test_model = config['testing']['chosen_test_model']
    folder_to_save_test=f'{model_out_root}/{translation}/{dataset_choice}/{model_type}/{version}/{model_details}/{chosen_test_model}'
    if not os.path.exists(folder_to_save_test):
        # Create the directory
        os.makedirs(folder_to_save_test)

    ############################################# LOAD IN NETMATS AND SURFACE MESHES #############################################
    if bilateral_condition:
        hemi_cond = "2LR"
    else:
        hemi_cond = config['training']['hemi_cond']

    if dataset_choice == "HCPYA":
        if parcellation_corr_type == "full":
            train_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_train_labels.npy") 
            train_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_train_data.npy")#[:, np.newaxis, channel_testing, :] 
            write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)
            write_to_file(f"loading from:{hemi_cond}_train_netmat_clean.npy and {hemi_cond}_train_surf.npy", filepath=write_fpath)

            val_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_validation_labels.npy") 
            val_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_validation_data.npy")#[:, np.newaxis, channel_testing, :]
            write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)

            if TEST_FLAG:
                te_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_test_labels.npy")
                te_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_test_data.npy")#[:, np.newaxis, channel_testing, :]
                write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)

    elif dataset_choice == "ABCD":
        if parcellation_corr_type == "full":
            if translation == "ICAd15_glasserd360":
                train_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/glasser_mats/netmat_d{from_parcellation}/{hemi_cond}_train_netmat_clean.npy")
                train_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/glasser/ICAd15_ico0{icores}/{hemi_cond}_train_surf.npy")#[:, np.newaxis, channel_testing, :] 
                write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)
                write_to_file(f"loading from:{hemi_cond}_train_netmat_clean.npy and {hemi_cond}_train_surf.npy", filepath=write_fpath)
                
                val_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/glasser_mats/netmat_d{from_parcellation}/{hemi_cond}_val_netmat_clean.npy")
                val_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/glasser/ICAd15_ico0{icores}/{hemi_cond}_val_surf.npy")#[:, np.newaxis, channel_testing, :]
                write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)

                if TEST_FLAG:
                    te_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/glasser_mats/netmat_d{from_parcellation}/{hemi_cond}_test_netmat_clean.npy")
                    te_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/glasser/ICAd15_ico0{icores}/{hemi_cond}_test_surf.npy")#[:, np.newaxis, channel_testing, :]
                    write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)

            else:    
                train_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_train_netmat_clean.npy")
                train_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{icores}/{hemi_cond}_train_surf.npy")#[:, np.newaxis, channel_testing, :] 
                write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)
                write_to_file(f"loading from:{hemi_cond}_train_netmat_clean.npy and {hemi_cond}_train_surf.npy", filepath=write_fpath)
                
                val_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_val_netmat_clean.npy")
                val_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{icores}/{hemi_cond}_val_surf.npy")#[:, np.newaxis, channel_testing, :]
                write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)

                if TEST_FLAG:
                    te_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_test_netmat_clean.npy")
                    te_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{icores}/{hemi_cond}_test_surf.npy")#[:, np.newaxis, channel_testing, :]
                    write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)
        elif parcellation_corr_type == "partial":
            train_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/partialnetmat_d{from_parcellation}/train_netmat_clean.npy")
            # train_netmat_np = train_netmat_np.T.to_numpy()
            train_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{icores}/{hemi_cond}_train_surf.npy")#[:, np.newaxis, channel_testing, :] 
            write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)

            val_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/partialnetmat_d{from_parcellation}/val_netmat_clean.npy")
            # val_netmat_np = val_netmat_np.T.to_numpy()
            val_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{icores}/{hemi_cond}_val_surf.npy")#[:, np.newaxis, channel_testing, :]
            write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)

            if TEST_FLAG:
                te_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/partialnetmat_d{from_parcellation}/test_netmat_clean.npy")
                te_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{icores}/{hemi_cond}_test_surf.npy")#[:, np.newaxis, channel_testing, :]
                write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)
    
    # check if any nan or inf values to avoid exploding/vanishing grads
    assert train_surf_np.shape[0] == train_netmat_np.shape[0]
    assert val_surf_np.shape[0] == val_netmat_np.shape[0]
    assert te_surf_np.shape[0] == te_netmat_np.shape[0]

    surf_check_nan = np.isnan(train_surf_np).sum()
    surf_check_inf = np.isinf(train_surf_np).sum()
    netmat_check_nan = np.isnan(train_netmat_np).sum()
    netmat_check_inf = np.isinf(train_netmat_np).sum()
    total_train = surf_check_nan + surf_check_inf + netmat_check_nan + netmat_check_inf
    write_to_file(f'TRAINING COUNTS: {surf_check_nan} - {surf_check_inf} - {netmat_check_nan} - {netmat_check_inf}', filepath=write_fpath)

    surf_check_nan = np.isnan(val_surf_np).sum()
    surf_check_inf = np.isinf(val_surf_np).sum()
    netmat_check_nan = np.isnan(val_netmat_np).sum()
    netmat_check_inf = np.isinf(val_netmat_np).sum()
    total_val = surf_check_nan + surf_check_inf + netmat_check_nan + netmat_check_inf
    write_to_file(f'VALIDATION COUNTS: {surf_check_nan} - {surf_check_inf} - {netmat_check_nan} - {netmat_check_inf}', filepath=write_fpath)

    assert total_train == 0, "Nan/Inf in TRAIN. Check."
    assert total_val == 0, "Nan/Inf in VALIDATION. Check."
    if overfit_condition:
        n=config['training']['overfit_condition_sub_range'] # upto how many subjects
        write_to_file(f'Overfit CONDITION is true, using {n} subjects', filepath=write_fpath)
        train_netmat_np = train_netmat_np[:n] #random subject(s) to pick to over fit
        train_surf_np = train_surf_np[:n]
        # 10 percent of train N
        val_netmat_np = val_netmat_np[:int(n*0.1)]
        val_surf_np = val_surf_np[:int(n*0.1)]
        te_netmat_np = te_netmat_np[:int(n*0.1)]
        te_surf_np = te_surf_np[:int(n*0.1)] 

    # condition for specific channel not
    if channel_specific_condition:
        if type(specific_channel_end) is list:
            chnl_range = np.arange(0,to_icamap)
            mask = ~np.isin(chnl_range, specific_channel_end)
            final_chnls = chnl_range[mask]
            write_to_file(f'Channels chosen to stay: {final_chnls}', filepath=write_fpath)
            
            train_surf_np = train_surf_np[:,final_chnls,:,:]
            write_to_file(f'SHAPE: {train_surf_np.shape} -- should be NxPxV', filepath=write_fpath)
            val_surf_np = val_surf_np[:,final_chnls,:,:]
            te_surf_np = te_surf_np[:,final_chnls,:,:]
        elif specific_channel == specific_channel_end:
            cc = specific_channel
            write_to_file(f'SPECIFIC CHANNEL CHOSEN: {cc}', filepath=write_fpath)
            train_surf_np = train_surf_np[:,cc,:,:]
            write_to_file(f'SHAPE: {train_surf_np.shape} -- should be NxPxV', filepath=write_fpath)
            val_surf_np = val_surf_np[:,cc,:,:]
            te_surf_np = te_surf_np[:,cc,:,:]
            train_surf_np = np.expand_dims(train_surf_np, axis=1)
            val_surf_np = np.expand_dims(val_surf_np, axis=1) # channel axis is 1 so expand that to keep shape BxCxPxV ow you get BxPxV
            te_surf_np = np.expand_dims(te_surf_np, axis=1)
        else:
            cc = specific_channel
            # specific_channel_end=cc+1
            train_surf_np = train_surf_np[:,cc:specific_channel_end,:,:]
            write_to_file(f'SHAPE: {train_surf_np.shape} -- should be NxPxV', filepath=write_fpath)
            val_surf_np = val_surf_np[:,cc:specific_channel_end,:,:]
            te_surf_np = te_surf_np[:,cc:specific_channel_end,:,:]
            # train_surf_np = np.expand_dims(train_surf_np, axis=1)
            # val_surf_np = np.expand_dims(val_surf_np, axis=1) # channel axis is 1 so expand that to keep shape BxCxPxV ow you get BxPxV
            # te_surf_np = np.expand_dims(te_surf_np, axis=1)
        write_to_file(f'We expand on channel dim now. TRAIN SHAPE: {train_surf_np.shape} -- should be Nx1xPxV after expansion', filepath=write_fpath)
        write_to_file(f'We expand on channel dim now. VAL SHAPE: {val_surf_np.shape} -- should be Nx1xPxV after expansion', filepath=write_fpath)
        write_to_file(f'We expand on channel dim now. TEST SHAPE: {te_surf_np.shape} -- should be Nx1xPxV after expansion', filepath=write_fpath)

    # tr_loader, val_loader, mean_train_label = fcn_prep_data_get_loaders(train_netmat=train_netmat_np, train_surface=train_surf_np, validation_netmat=val_netmat_np, validation_surface=val_surf_np, parcellation_N=from_parcellation, netmat_prep_choice=netmat_prep_choice, b_sz=batch_size, write_fpath=write_fpath)
    # padding=50 #config['transformer']['padding'] #TODO if model does worse, might be because padding was once done BEFORE putting in fcn_prep_data_get_loaders so revert to test if so
    # upper_tri_sz = train_netmat_np.shape[1] # should be SUBx4950 (or node size upper tri count)
    if flag_experiment_ICArecon:
        write_to_file(f"CHOSEN TO RECONSTRUCT ICA MAPs! {flag_experiment_ICArecon}", filepath=write_fpath)
                      
        tr_loader, val_loader, mean_train_label = fcn_prep_data_get_loaders_ICAren(train_surface=train_surf_np, validation_surface=val_surf_np, b_sz=batch_size, write_fpath=write_fpath)
        if TEST_FLAG:
            tr_loader_for_test, te_loader, _ = fcn_prep_data_get_loaders_ICAren(train_surface=train_surf_np, validation_surface=te_surf_np, b_sz=te_batch_size, write_fpath=write_fpath)
    
    else:
        write_to_file(f"regular training, not ICA recon.", filepath=write_fpath)
        tr_loader, val_loader, mean_train_label = fcn_prep_data_get_loaders(train_netmat=train_netmat_np, train_surface=train_surf_np, validation_netmat=val_netmat_np, validation_surface=val_surf_np, parcellation_N=from_parcellation, netmat_prep_choice=netmat_prep_choice, surf_prep_choice=surf_prep_choice, b_sz=batch_size,encdec=False, write_fpath=write_fpath)
        if TEST_FLAG:
            tr_loader_for_test, te_loader, _ = fcn_prep_data_get_loaders(train_netmat=train_netmat_np, train_surface=train_surf_np, validation_netmat=te_netmat_np, validation_surface=te_surf_np, parcellation_N=from_parcellation, netmat_prep_choice=netmat_prep_choice, surf_prep_choice=surf_prep_choice, b_sz=te_batch_size,encdec=False, write_fpath=write_fpath)
        
    write_to_file(f"Loaded in data. Tunning on dataset: {dataset_choice}", filepath=write_fpath)
    
    # because any parcellation given is NxN symm matrix, no need to netmat.shape to get sizes, we already know them from "from_parcellation" variable
    hold, dim_c, dim_p, dim_v = train_surf_np.shape
    hold, upper_tri = train_netmat_np.shape
    
    del train_surf_np, val_surf_np

    dim = config['transformer']['sit_dim']
    dim_head = config['transformer']['dim_head']
    depth = config['transformer']['depth']
    heads = config['transformer']['heads']
    emb_dropout = config['transformer']['emb_dropout']
    dropout = config['transformer']['dropout']
    if VAE_flag:
        VAE_latent_dim = config['transformer']['vae_dim']
        latent_samples = config['transformer']['latent_samples']
    
        model = fcn_model_module(
                        dim=dim, 
                        depth=depth,
                        heads=heads,
                        num_patches = dim_p,
                        upper_tri = upper_tri, #parcellation
                        num_channels = dim_c,
                        num_vertices = dim_v,
                        dim_head = dim_head,
                        dropout = dropout,
                        emb_dropout = emb_dropout,
                        VAE_latent_dim=VAE_latent_dim,
                        latent_samples=latent_samples
                        )
    else: # not variational
        model = fcn_model_module(
                        dim=dim, 
                        depth=depth,
                        heads=heads,
                        num_patches = dim_p,
                        upper_tri = upper_tri, #parcellation
                        num_channels = dim_c,
                        num_vertices = dim_v,
                        dim_head = dim_head,
                        dropout = dropout,
                        emb_dropout = emb_dropout,
                        )
    
    # initialize optimizer / loss
    scheduler = False #default is false, unless otherwise specified by the yml configuration file
    if config['optimisation']['optimiser']=='Adam':
        write_to_file('using Adam optimiser',  filepath=write_fpath)
        optimizer = optim.Adam(model.parameters(),
                               lr=LR,
                               weight_decay=config['Adam']['weight_decay'])
        if config['Adam']['use_scheduler']:
            scheduler = True
            lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max = config['CosineDecay']['T_max'],
                                                                eta_min= config['CosineDecay']['eta_min']
                                                                )


    elif config['optimisation']['optimiser']=='SGD':
        write_to_file('using SGD optimiser',  filepath=write_fpath)
        optimizer = optim.SGD(model.parameters(), lr=LR, 
                                                weight_decay=config['SGD']['weight_decay'],
                                                momentum=config['SGD']['momentum'],
                                                nesterov=config['SGD']['nesterov'])
    elif config['optimisation']['optimiser']=='AdamW':
        write_to_file('using AdamW optimiser',  filepath=write_fpath)
        optimizer = optim.AdamW(model.parameters(),
                                lr=LR,
                                weight_decay=config['AdamW']['weight_decay'])
        if config['AdamW']['use_scheduler']:
            scheduler = True
            if config['AdamW']['scheduler']=='CosineDecay': # TODO currrently only set for CosineDecay bc that is what was used in swMSSiT paper from dahan
                lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max = config['CosineDecay']['T_max'],
                                                        eta_min= config['CosineDecay']['eta_min'],
                                                        last_epoch=-1
                                                        )


    # Find number of parameters
    model_params = sum(p.numel() for p in model.parameters())
    write_to_file(f"Model params: {model_params}", filepath=write_fpath)

    # reset params 
    model._reset_parameters()

    running_train_loss = 0
    # running_val_loss = 0
    df_train = pd.DataFrame(columns=['train_mae', 'train_mae_sigma', 'train_mse', 'train_mse_sigma', 'train_loss', 'train_demean_corr', 'train_demean_corr_sigma', 'train_orig_corr', 'train_orig_corr_sigma'])
    df_val = pd.DataFrame(columns=['val_mae', 'val_mae_sigma', 'val_mse', 'val_mse_sigma', 'val_loss', 'val_demean_corr', 'val_demean_corr_sgima', 'val_orig_corr', 'val_orig_corr_sigma'])

    write_to_file("Training has begun.", filepath=write_fpath)
    lr_list=[]
    for epoch in range(1, train_epoch_range):
        
        tr_epoch_loss, across_sub_mae_mean, across_sub_mae_std, across_sub_mse_mean, across_sub_mse_std, across_sub_corr_demean, across_sub_corr_demean_std, across_sub_corr_org, across_sub_corr_org_std = fcn_train(model, krak_mse_weight, krak_latent_weight, krak_corrEYE_weight, rho_loss_weight, tr_loader, mean_train_label, device, optimizer, VAE_flag, netmat_prep_choice)
        write_to_file(f"SCHEDULE IS: {scheduler}", filepath=write_fpath)
        if scheduler: # if you are using a scheduler, this should be TRUE o.w. FALSE so no need to do the "step" to change LR
            write_to_file("scheduler TRUE, so changing LR as needed.", filepath=write_fpath)
            lr_schedule.step() #after each epoch
            
            # write_to_file(f"{lr_schedule.get_last_lr()}", filepath=write_fpath)
            # write_to_file(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}", filepath=write_fpath)
            curr_lr = optimizer.param_groups[0]['lr'] #lr_schedule.get_last_lr()[-1]
        else:
            curr_lr = LR

        # Convert tensors to floats
        train_loss_value = float(tr_epoch_loss)
        running_train_loss += train_loss_value
        lr_list.append(curr_lr)
        
        write_to_file('| Training | Epoch - {} | LR - {:.4f}| Loss - {:.4f} | MAE - {:.4f} | MSE = {:.4f} | demeanCorr {:.4f}'.format(epoch, curr_lr, running_train_loss, across_sub_mae_mean, across_sub_mse_mean, across_sub_corr_demean), filepath=write_fpath)

        new_row = pd.DataFrame({'train_mae': [across_sub_mae_mean], 'train_mae_sigma': [across_sub_mae_std], 'train_mse': [across_sub_mse_mean], 'train_mse_sigma': [across_sub_mse_std], 
                                'train_loss': [train_loss_value], 'train_demean_corr': [across_sub_corr_demean], 'train_demean_corr_sigma': [across_sub_corr_demean_std], 'train_orig_corr': [across_sub_corr_org], 
                                'train_orig_corr_sigma': [across_sub_corr_org_std]
                                })
        
        df_train = pd.concat([df_train, new_row], ignore_index=True)
        df_train.to_csv(os.path.join(folder_to_save_losses, 'train_losses_patch.csv'))

        if epoch%val_epoch == 0:
            grpavg_val_mae, grpstd_val_mae, grpavg_val_mse, grpstd_val_mse, val_deman_corr, val_deman_corr_std, val_orig_corr, val_orig_corr_std = fcn_validate(model, val_loader, mean_train_label, device, VAE_flag, netmat_prep_choice)
            write_to_file('| Validation | Epoch - {} | MAE - {:.4f} | MSE = {:.4f} | demeanCorr {:.4f}'.format(epoch, grpavg_val_mae, grpavg_val_mse, val_deman_corr), filepath=write_fpath)

            # save model with best MSE - gives leeway to values around 0 so maybe betetr for correlation values?
            curr_val_mse = grpavg_val_mse
            if curr_val_mse < best_mse:
                best_mse = curr_val_mse
                write_to_file('saving MSE model...', filepath=write_fpath)
                torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}_MSE.pt'))
            # save model with best MAE - forces values closer to 0
            curr_val_mae = grpavg_val_mae
            if curr_val_mae < best_mae:
                best_mae = curr_val_mae
                write_to_file('saving MAE model...', filepath=write_fpath)
                torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}_MAE.pt'))
            # save model with best RHO_demean
            curr_val_demean_rho = val_deman_corr # prioritize model with best demean correlation performance with validation set
            if curr_val_demean_rho > best_demean_rho:
                best_demean_rho = curr_val_demean_rho
                write_to_file('saving RHO model...', filepath=write_fpath)
                torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}_RHO.pt'))

            new_row = pd.DataFrame({'val_mae': [grpavg_val_mae], 'val_mae_sigma': [grpstd_val_mae], 'val_mse': [grpavg_val_mse], 'val_mse_sigma': [grpstd_val_mse],
                                    'val_demean_corr': [val_deman_corr], 'val_demean_corr_sigma': [val_deman_corr_std], 'val_orig_corr': [val_orig_corr],
                                    'val_orig_corr_sigma': [val_orig_corr_std]
                                    })
            
            df_val = pd.concat([df_val, new_row], ignore_index=True)
            df_val.to_csv(os.path.join(folder_to_save_losses, 'val_losses_patch.csv'))


        write_to_file('saving LAST model...', filepath=write_fpath)
        torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}_LAST.pt'))

    df_version_lr_list = pd.DataFrame(lr_list)
    df_version_lr_list.to_csv(os.path.join(folder_to_save_test, 'model_lr_list.csv'))
    
    # TESTING #
    if TEST_FLAG:
        write_to_file(f'TEST FLAG ON. TESTING.', filepath=write_fpath)
        # see all models
        model_path = sorted(glob.glob(f"{folder_to_save_model}/*{model_details}_{chosen_test_model}.pt")) # look at training script for details, but all models saves as type_details_chosen: ex-kBGTLN_d6h5_demeanL2_skewloss_RHO.pt
        chosen_model = model_path[0]
        write_to_file(f'\n\nmodel loaded is {chosen_model}', filepath=write_fpath)
        model.load_state_dict(torch.load(chosen_model)) # most recent model

        # Find number of parameters
        model_params = sum(p.numel() for p in model.parameters())
        write_to_file(f"\n\nModel params: {model_params}", filepath=write_fpath)

        # Testing below
        model.eval()
        model.to(device)

        # lists to keep track
        mse_train_list = []
        mae_train_list = []
        mse_test_list = []
        mae_test_list = []

        if flag_experiment_ICArecon:
            # hold, dim_c, dim_p, dim_v = train_surf_np.shape
            hold_test = te_surf_np.shape[0]
            # reshaped_data = train_surf_np.reshape(hold, dim_c*dim_p*dim_v)
            reshaped_data_test = te_surf_np.reshape(hold_test, dim_c*dim_p*dim_v)
            # ss = hold # subj num
            tr_ground_truth = np.zeros((hold, dim_c*dim_p*dim_v))
            tr_pred = np.zeros((hold, dim_c*dim_p*dim_v))
            te_ground_truth = np.zeros(reshaped_data_test.shape)
            te_pred = np.zeros(reshaped_data_test.shape)
        else:
            if bilateral_condition:
                ss, nn = train_netmat_np.shape
                ss = 2 * ss
                tr_ground_truth = np.zeros((ss,nn))
                tr_pred = np.zeros((ss,nn))
                te_ground_truth = np.zeros((ss,nn))
                te_pred = np.zeros((ss,nn))
            else:
                tr_ground_truth = np.zeros(train_netmat_np.shape)
                tr_pred = np.zeros(train_netmat_np.shape) #SUBx4950 of zeros
                te_ground_truth = np.zeros(te_netmat_np.shape)
                te_pred = np.zeros(te_netmat_np.shape)

        with torch.no_grad():
            for i, data in enumerate(te_loader):
                inputs, targets = data[0].to(device), data[1].to(device)#.squeeze()
                # pred, latent = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                # del latent, inputs

                if VAE_flag:
                    pred, latent, log_latent = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                    del latent, inputs, log_latent
                else:
                    pred, latent = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                    del latent, inputs
                
                # just having some output to see while testing, otherwise terminal is silent. Nice to see progress IMO
                if i % 100 == 0:
                    write_to_file(f"checkpoint. Running test subject: {i}", filepath=write_fpath)

                pred = pred.detach().numpy()
                targets = targets.detach().numpy()
                
                mae = np.mean(np.abs(pred - targets))
                mae_test_list.append(mae)

                mse = np.mean( (pred - targets)**2 )
                mse_test_list.append(mse)

                te_ground_truth[i, :] = targets
                te_pred[i, :] = pred

            write_to_file(f"Done with TESTING loop.", filepath=write_fpath)

            # to optimize testing and data saving, will only get best, mid, and lowest corr
            across_sub_rho = np.corrcoef(te_ground_truth, te_pred) # gives sub_dim*2 x sub_dim*2 and will likely be two square clusters truth and pred
            write_to_file(f"SZ of bigg matrix: {across_sub_rho.shape}", filepath=write_fpath)
            np.save(f"{folder_to_save_test}/te_big_corr_matrix.npy", across_sub_rho) # save for viz later
            
            # # find best, and worst corr(truth,pred)
            # row_half = np.split(across_sub_rho,2, axis = 0) #split in half across rows
            # top_right_quad = np.split(row_half[0],2, axis = 1)[1] # again split by col, and top rigth quaf is corr(y,yhat) so choose 1 automatically == quad2
            # find_max_rho = np.argwhere(top_right_quad == np.max(np.diag(top_right_quad)))[0] # find max across diagonal
            # find_min_rho = np.argwhere(top_right_quad == np.min(np.diag(top_right_quad)))[0] #find min across diagonal
            # max_idx = find_max_rho[0] # which subject had the highest corr across diagonal in quad2
            # min_idx = find_min_rho[0] #0 is i, so subject index althougth same as j but keeping consistency
            # write_to_file(f"IDX in big TEST corr matrix for both best (max) and worst (min) performance: {max_idx} {min_idx}", filepath=write_fpath)

            # # save bet and worst netmat translation
            # te_max_netmat_translation = te_pred[max_idx]
            # te_min_netmat_translation = te_pred[min_idx]
            # np.save(f"{folder_to_save_test}/te_max_netmat_translation.npy", te_max_netmat_translation)
            # np.save(f"{folder_to_save_test}/te_min_netmat_translation.npy", te_min_netmat_translation)

            for i, data in enumerate(tr_loader_for_test):
                inputs, targets = data[0].to(device), data[1].to(device)#.squeeze()
                # pred, latent = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                # del latent, inputs

                if VAE_flag:
                    pred, latent, log_latent = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                    del latent, inputs, log_latent
                else:
                    pred, latent = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                    del latent, inputs

                # just having some output to see while testing, otherwise terminal is silent. Nice to see progress IMO
                if i % 100 == 0:
                    write_to_file(f"checkpoint. Running test subject: {i}", filepath=write_fpath)

                pred = pred.detach().numpy()
                targets = targets.detach().numpy()
                
                mae = np.mean(np.abs(pred - targets))
                mae_train_list.append(mae)

                mse = np.mean( (pred - targets)**2 )
                mse_train_list.append(mse)

                tr_ground_truth[i, :] = targets
                tr_pred[i, :] = pred

            write_to_file(f"Done with TRAINING loop.", filepath=write_fpath)

            # to optimize testing and data saving, will only get best, mid, and lowest corr
            across_sub_rho = np.corrcoef(tr_ground_truth, tr_pred) # gives sub_dim*2 x sub_dim*2 and will likely be two square clusters truth and pred
            np.save(f"{folder_to_save_test}/tr_big_corr_matrix.npy", across_sub_rho) # save for viz later
            
            # # find best, mid, and worst corr(truth,pred)
            # row_half = np.split(across_sub_rho,2, axis = 0) #split in half across rows
            # top_right_quad = np.split(row_half[0],2, axis = 1)[1] # again split by col, and top rigth quaf is corr(y,yhat) so choose 1 automatically == quad2
            # find_max_rho = np.argwhere(top_right_quad == np.max(np.diag(top_right_quad)))[0] # find max across daigonal
            # find_min_rho = np.argwhere(top_right_quad == np.min(np.diag(top_right_quad)))[0] #find min across diagonal
            # max_idx = find_max_rho[0] #0 is rows
            # min_idx = find_min_rho[0] #0 is i, so subject index althougth same as j but keeping consistency
            # write_to_file(f"IDX in big TRAIN corr matrix for both best (max) and worst (min) performance: {max_idx} {min_idx}", filepath=write_fpath)

            # # save bet and worst netmat translation
            # tr_max_netmat_translation = tr_pred[max_idx]
            # tr_min_netmat_translation = tr_pred[min_idx]
            # np.save(f"{folder_to_save_test}/tr_max_netmat_translation.npy", tr_max_netmat_translation)
            # np.save(f"{folder_to_save_test}/tr_min_netmat_translation.npy", tr_min_netmat_translation)
        
        # save training losses
        df_version_mae = pd.DataFrame(mae_train_list)
        df_version_mae.to_csv(os.path.join(folder_to_save_test, 'mae_train_model.csv'))
        df_version_mse = pd.DataFrame(mse_train_list)
        df_version_mse.to_csv(os.path.join(folder_to_save_test, 'mse_train_model.csv'))

        # save test losses
        df_version_mae = pd.DataFrame(mae_test_list)
        df_version_mae.to_csv(os.path.join(folder_to_save_test, 'mae_test_model.csv'))
        df_version_mse = pd.DataFrame(mse_test_list)
        df_version_mse.to_csv(os.path.join(folder_to_save_test, 'mse_test_model.csv'))

        write_to_file("TRAIN Mean MAE:", filepath=write_fpath)
        write_to_file(np.nanmean(mae_train_list), filepath=write_fpath)
        write_to_file("TEST Mean MAE:", filepath=write_fpath)
        write_to_file(np.nanmean(mae_test_list), filepath=write_fpath)

        write_to_file("TRAIN Mean MSE:", filepath=write_fpath)
        write_to_file(np.nanmean(mse_train_list), filepath=write_fpath)
        write_to_file("TEST Mean MSE:", filepath=write_fpath)
        write_to_file(np.nanmean(mse_test_list), filepath=write_fpath)

        np.save(f"{folder_to_save_test}/train_ground_truth.npy", tr_ground_truth)
        np.save(f"{folder_to_save_test}/train_pred.npy", tr_pred)
        np.save(f"{folder_to_save_test}/test_ground_truth.npy", te_ground_truth)
        np.save(f"{folder_to_save_test}/test_pred.npy", te_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='kSiT_tr')

    parser.add_argument(
                        'config',
                        type=str,
                        default='',
                        help='args from yaml file')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call training
    whole_model_arch(config)

