import os
import sys
import glob 
sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

import yaml
import argparse
import torch
import numpy as np   
import pandas as pd
from models import models
# from models.models import *
from utils.utils import *

def whole_model_arch(config):
    #some settings
    try: #since this is a new thing added, putting this here so that previous yml files without it can run assuming they were done with ico02
        icores = config['data']['icores']
    except:
        # write_to_file("YML file does not have ico resolution information. Defaulting to ico-02", filepath=write_fpath)
        icores="02"
    # else:
        # write_to_file(f"Using ico-{icores} surf data.", filepath=write_fpath)
    model_out_root = config['logging']['model_out_root']
    model_details = config['transformer']['model_details']
    netmat_prep_choice = config['training']['netmat_prep_choice']
    fcn_model_module = getattr(models, config['training']['fcn_model_to_use']) 
    flag_experiment_ICArecon = config['training']['flag_experiment_ICArecon']
    channel_specific_condition = config['training']['channel_specific_condition']
    if flag_experiment_ICArecon:
        to_icamap = config['data']['to_icamap']
        specific_channel = config['training']['specific_channel']
        specific_channel_end = config['training']['specific_channel_end']
        model_details = config['transformer']['model_details'] + f"_chnl{specific_channel}"
        write_fpath = config['logging']['sanity_file_pth'] + f'ico-{icores}' + f'_chnl{specific_channel}' + '.print'
    else:
        write_fpath = config['logging']['test_file_pth'] + f'ico-{icores}' + '.print'
    surf_prep_choice = config['training']['surf_prep_choice']
    dataset_choice = config['training']['dataset_choice']
    bilateral_condition = config['training']['bilateral_condition'] # both hemispheres instead of 1
    parcellation_corr_type = config['training']['parcellation_corr_type']
    from_parcellation = config['data']['from_parcellation']
    translation = config['data']['translation']
    version = config['data']['version'] #normICAdemeanfishzMAT
    model_type = config['data']['model_type']
    VAE_flag = config['training']['VAE_flag']
    device = "cpu"
    TEST_FLAG = config['testing']['immediate_test_flag']
    te_batch_size = config['testing']['bs_test']
    same_sample_test = config['testing']['same_sample_test']
    out_of_sample_test = config['testing']['out_of_sample_test']
    if out_of_sample_test:
        dataset_choice = "HCPYA_" + dataset_choice + "dr"
        assert dataset_choice == "HCPYA_ABCDdr", "out of distribution TRUE, but dataset for data is same distribution. Verify."
        write_to_file(f"Out of sample flag: {out_of_sample_test}", filepath=write_fpath)

    chosen_test_model = config['testing']['chosen_test_model']
    # saved_model_path = f'/home/naranjorincon/neurotranslate/surf2netmat/logs/{translation}/{dataset_choice}/{model_type}/{version}'
    # path_to_model=f'/home/naranjorincon/neurotranslate/surf2netmat/logs/{translation}/{dataset_choice}/{model_type}/{version}'
    path_to_model=f'/home/naranjorincon/neurotranslate/surf2netmat/logs/{translation}/ABCD/{model_type}/{version}'

    ############################################# LOAD IN NETMATS AND SURFACE MESHES #############################################
    if bilateral_condition:
        hemi_cond = "2LR"
    else:
        hemi_cond = config['training']['hemi_cond']

    data_root_path = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"
    # chosen_test_model = config['testing']['chosen_test_model']
    # if out_of_sample_test:
    #     folder_to_save_test=f'{model_out_root}/{translation}/{dataset_choice}/{model_type}/{version}/{model_details}/{chosen_test_model}'
    # else:
    folder_to_save_test=f'{model_out_root}/{translation}/{dataset_choice}/{model_type}/{version}/{model_details}/{chosen_test_model}'

    if not os.path.exists(folder_to_save_test):
        # Create the directory
        os.makedirs(folder_to_save_test)
        write_to_file("Directory created.", filepath=write_fpath)
    else:
        write_to_file("Directory already exists.", filepath=write_fpath)

    if dataset_choice == "HCPYA":
        train_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_train_labels.npy") 
        train_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_train_data.npy")#[:, np.newaxis, channel_testing, :] 
        write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)
        
        val_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_validation_labels.npy") 
        val_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_validation_data.npy")#[:, np.newaxis, channel_testing, :]
        write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)
    
    # elif dataset_choice == "ABCD":
    else:
        if parcellation_corr_type == "full":
            if translation == "ICAd15_glasserd360":
                
                if out_of_sample_test:
                    train_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/train_netmat_clean.npy")
                else:
                    train_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/glasser_mats/netmat_d{from_parcellation}/{hemi_cond}_train_netmat_clean.npy")
                
                train_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/glasser/ICAd15_ico0{icores}/{hemi_cond}_train_surf.npy")#[:, np.newaxis, channel_testing, :] 
                write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)
                
                if out_of_sample_test:
                    val_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/validation_netmat_clean.npy")
                else:
                    val_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/glasser_mats/netmat_d{from_parcellation}/{hemi_cond}_val_netmat_clean.npy")
    
                val_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/glasser/ICAd15_ico0{icores}/{hemi_cond}_val_surf.npy")#[:, np.newaxis, channel_testing, :]
                write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)

                if TEST_FLAG:
                    if out_of_sample_test:
                        te_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/test_netmat_clean.npy")
                    else:
                        te_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/glasser_mats/netmat_d{from_parcellation}/{hemi_cond}_test_netmat_clean.npy")

                    te_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/glasser/ICAd15_ico0{icores}/{hemi_cond}_test_surf.npy")#[:, np.newaxis, channel_testing, :]
                    write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)

            else:    

                if out_of_sample_test: # so far, HCPYA only has schff100 ready so forcing all tests on this to be on schaefer100
                    train_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/train_netmat_clean.npy")
                    write_to_file(f'SHOULD BE TRUE CHECK. {train_netmat_np.shape} ', filepath=write_fpath)
                else:
                    train_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_train_netmat_clean.npy")

                train_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{icores}/{hemi_cond}_train_surf.npy")#[:, np.newaxis, channel_testing, :] 
                write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)
                write_to_file(f"loading from:{hemi_cond}_train_netmat_clean.npy and {hemi_cond}_train_surf.npy", filepath=write_fpath)
                
                if out_of_sample_test:
                    val_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/validation_netmat_clean.npy")
                else:
                    val_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_val_netmat_clean.npy")
    
                val_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{icores}/{hemi_cond}_val_surf.npy")#[:, np.newaxis, channel_testing, :]
                write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)

                if TEST_FLAG:
                    if out_of_sample_test:
                        te_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/test_netmat_clean.npy")
                    else:
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
                      
        # tr_loader, val_loader, mean_train_label = fcn_prep_data_get_loaders_ICAren(train_surface=train_surf_np, validation_surface=val_surf_np, b_sz=batch_size, write_fpath=write_fpath)
        if TEST_FLAG:
            tr_loader_for_test, te_loader, _ = fcn_prep_data_get_loaders_ICAren(train_surface=train_surf_np, validation_surface=te_surf_np, b_sz=te_batch_size, write_fpath=write_fpath)
    
    else:
        # tr_loader, val_loader, mean_train_label = fcn_prep_data_get_loaders(train_netmat=train_netmat_np, train_surface=train_surf_np, validation_netmat=val_netmat_np, validation_surface=val_surf_np, parcellation_N=from_parcellation, netmat_prep_choice=netmat_prep_choice, b_sz=batch_size,encdec=False, write_fpath=write_fpath)
        if TEST_FLAG:
            tr_loader_for_test, te_loader, _ = fcn_prep_data_get_loaders(train_netmat=train_netmat_np, train_surface=train_surf_np, validation_netmat=te_netmat_np, validation_surface=te_surf_np, parcellation_N=from_parcellation, netmat_prep_choice=netmat_prep_choice, surf_prep_choice=surf_prep_choice, b_sz=te_batch_size,encdec=False, write_fpath=write_fpath)

    # write_to_file(f"test sizes: tr_loader_for_test:{tr_loader_for_test} {}", filepath=write_fpath)    
    write_to_file(f"Loaded in data. Tunning on dataset: {dataset_choice}", filepath=write_fpath)

    ############################################# Set up Test run and model configs #############################################  
    # because any parcellation given is NxN symm matrix, no need to netmat.shape to get sizes, we already know them from "from_parcellation" variable
    hold, dim_c, dim_p, dim_v = train_surf_np.shape
    hold, upper_tri = train_netmat_np.shape
    
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
    
    model_params = sum(p.numel() for p in model.parameters())
    write_to_file(f"\n\nModel params: {model_params}", filepath=write_fpath)

    write_to_file(f"{path_to_model}/*{model_details}_{chosen_test_model}.pt", filepath=write_fpath)
    model_path = sorted(glob.glob(f"{path_to_model}/*{model_details}_{chosen_test_model}.pt")) # look at training script for details, but all models saves as type_details_chosen: ex-kBGTLN_d6h5_demeanL2_skewloss_RHO.pt
    print(model_path)
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
        hold, dim_c, dim_p, dim_v = train_surf_np.shape
        hold_test = te_surf_np.shape[0]
        reshaped_data = train_surf_np.reshape(hold, dim_c*dim_p*dim_v)
        reshaped_data_test = te_surf_np.reshape(hold_test, dim_c*dim_p*dim_v)
        # ss = hold # subj num
        tr_ground_truth = np.zeros(reshaped_data.shape)
        tr_pred = np.zeros(reshaped_data.shape)
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
        # find_max_rho = np.argwhere(top_right_quad == np.max(np.diag(top_right_quad)))[0] # find max across daigonal
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
                write_to_file(f"checkpoint. Running train subject: {i}", filepath=write_fpath)

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
        
        # find best, mid, and worst corr(truth,pred)
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
    parser = argparse.ArgumentParser(description='kBGT_te')

    parser.add_argument(
                        'config',
                        type=str,
                        default='',
                        help='path where the data is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call training
    whole_model_arch(config)

    