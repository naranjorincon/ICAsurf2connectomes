import os
import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import yaml
from models.models import *
from utils.utils import *
from utils import utils
from utils import functions_train
from utils.functions_kraken_loss import *
import torch.optim as optim 
import glob 

# def fcn_validate(model, val_loader, mean_train_label, padding, device):
#     model.eval()
#     model.to(device)
#     mse_val_list = []
#     mae_val_list = []
#     demean_corr_val_list = []
#     orig_corr_val_list = []
#     with torch.no_grad():
#         for i, data in enumerate(val_loader):
#             mesh_indata, targets = data[0].to(device), data[1].to(device).squeeze() #, data[2].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
#             dec_input = targets
#             pred = model(src=mesh_indata, tgt=dec_input,  tgt_mask=generate_subsequent_mask(model.latent_length).to(device))

#             pred = pred[0].detach().numpy() # picks first element of pred tuble, so preds which is all we care for validation
#             targets = targets.detach().numpy()

#             pred = pred[:, padding:] 
#             targets = targets[:,padding:]    
#             # mean_train_label = np.mean(targets, axis=0)
            
#             mae = np.mean(np.abs( (targets - pred) ), axis=0)
#             mae_val_list.append(mae)
#             mse = np.mean(( (targets - pred)**2 ), axis=0)
#             mse_val_list.append(mse)

#             val_corr_demean = np.corrcoef((targets - mean_train_label), (pred - mean_train_label)) #[subj*2 x subj*2] matrix where quadrant1 = target_target, quad2=target_pred, quad3=pred_target, quad4=pred_pred
#             split_half_horizontal = np.split(val_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
#             top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
#             # lower_tri_of_qaudrant_demean = get_lower_tris(top_right_quad) # indeces are column, so you get upper right tri but ele idx is going down and exhasuting each column
#             demean_corr_val_list.append(np.diag(top_right_quad))

#             val_corr_org = np.corrcoef(targets, pred) # going to be low-ish cause 256->mesh size sphere but curious
#             split_half_horizontal = np.split(val_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
#             top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
#             # lower_tri_of_qaudrant_org = get_lower_tris(top_right_quad)
#             orig_corr_val_list.append(np.diag(top_right_quad))
    
#     across_sub_mae_mean = np.mean(mae_val_list) # across all elements, so no axis ==> mean of flatten mat->vector, so across all subs and channels and patches and verteces
#     across_sub_mae_std = np.std(mae_val_list)
#     across_sub_mse_mean = np.mean(mse_val_list)
#     across_sub_mse_std = np.std(mse_val_list)

#     # because of batching, some in the list are different size so make into whole array
#     upto_n_minus1 = np.asarray(demean_corr_val_list[:-1]).squeeze() # all upto last item, do that seperate then concat
#     upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
#     n_minus_1 = np.asarray(demean_corr_val_list[-1])[np.newaxis,:] 
#     val_corr_demean_flat = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
#     across_sub_corr_demean = np.mean(val_corr_demean_flat)
#     across_sub_corr_demean_std = np.std(val_corr_demean_flat)
#     # across_sub_corr_demean = np.mean(val_corr_demean_flat)
#     # across_sub_corr_demean_std = np.std(val_corr_demean_flat)

#     # same for original corr values
#     upto_n_minus1 = np.asarray(orig_corr_val_list[:-1]).squeeze() # all upto last item, do that seperate then concat
#     upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
#     n_minus_1 = np.asarray(orig_corr_val_list[-1])[np.newaxis,:] # adding dim cause above reshape gives dim to index 0, so shape is 1xDIMS so adding this also makes this 1xDIMs
#     val_corr_org_flat = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
#     # across_sub_corr_org = np.mean(val_corr_org_flat)
#     # across_sub_corr_org_std = np.std(val_corr_org_flat)
#     across_sub_corr_org = np.mean(val_corr_org_flat)
#     across_sub_corr_org_std = np.std(val_corr_org_flat)

#     return across_sub_mae_mean, across_sub_mae_std, across_sub_mse_mean, across_sub_mse_std, across_sub_corr_demean, across_sub_corr_demean_std, across_sub_corr_org, across_sub_corr_org_std

def fcn_validate_autoregressive(model, val_loader, mean_train_label, device):
    model.eval()
    model.to(device)

    mae_val_list = []
    mse_val_list = []
    demean_corr_val_list = []
    orig_corr_val_list = []

    with torch.no_grad():
        for i, (mesh_indata, full_target) in enumerate(val_loader):
            mesh_indata = mesh_indata.to(device)
            full_target = full_target.to(device)

            # Split target: [B, 5000] → [B, 100, 50]
            full_target_blocks = full_target.view(full_target.size(0), -1, 50)  # [B, 100, 50]
            start_tokens = full_target_blocks[:, 0, :]  # [B, 50]
            target = full_target[:, 50:]               # [B, 4950]

            # Encoder output (required for SiT_BGT_VAE)
            encoder_out = model.encode(mesh_indata)

            # Autoregressive generation (no teacher forcing)
            pred = model.autoregressive_decode(
                encoder_out=encoder_out,
                start_tokens=start_tokens,
                total_len=5000,
                block_size=50,
                teacher=None, # none gives it dummy target 
                teacher_forcing_ratio=0.0 #NOT using tearcher learning
            ).detach().cpu().numpy()  # [B, 4950]

            target = target.cpu().numpy()

            # MAE & MSE
            mae = np.mean(np.abs(target - pred), axis=0)
            mse = np.mean((target - pred) ** 2, axis=0)
            mae_val_list.append(mae)
            mse_val_list.append(mse)

            # Correlation (demeaned)
            corr_demean = np.corrcoef(target - mean_train_label, pred - mean_train_label)
            top_right_demean = np.split(np.split(corr_demean, 2, axis=0)[0], 2, axis=1)[1]
            demean_corr_val_list.append(np.diag(top_right_demean))

            # Correlation (original)
            corr_org = np.corrcoef(target, pred)
            top_right_org = np.split(np.split(corr_org, 2, axis=0)[0], 2, axis=1)[1]
            orig_corr_val_list.append(np.diag(top_right_org))

    # Aggregate metrics
    mae_mean = np.mean(mae_val_list)
    mae_std = np.std(mae_val_list)
    mse_mean = np.mean(mse_val_list)
    mse_std = np.std(mse_val_list)

    demean_corr_flat = np.concatenate(demean_corr_val_list).reshape(1, -1)
    org_corr_flat = np.concatenate(orig_corr_val_list).reshape(1, -1)

    corr_demean_mean = demean_corr_flat.mean()
    corr_demean_std = demean_corr_flat.std()
    corr_org_mean = org_corr_flat.mean()
    corr_org_std = org_corr_flat.std()

    return (
        mae_mean, mae_std,
        mse_mean, mse_std,
        corr_demean_mean, corr_demean_std,
        corr_org_mean, corr_org_std
    )


def train_SiTBGT_krakenonly_autoregressive(model, krak_mse_weight, krak_latent_weight, krak_corrI_weight,
                            train_loader, mean_train_label, device, optimizer, VAE_flag=False):
    model.train()

    tr_mae_subs = []
    tr_mse_subs = []
    tr_corr_subs_demean = []
    tr_corr_subs_org = []
    tr_epoch_loss = 0

    for i, (mesh_indata, full_target) in enumerate(train_loader):
        mesh_indata = mesh_indata.to(device)
        full_target = full_target.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Reshape target into blocks of 50
        full_target_blocks = full_target.view(full_target.size(0), -1, 50)  # [B, 100, 50]
        start_tokens = full_target_blocks[:, 0, :]  # [B, 50]
        target_blocks = full_target_blocks[:, 1:, :]  # [B, 99, 50]

        # Forward pass
        if VAE_flag:
            pred, mu, log_var = model(mesh_indata, full_target, teacher_forcing_ratio=1.0)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
            latent = mu
        else:
            pred, latent = model(mesh_indata, full_target, teacher_forcing_ratio=1.0)

        # Confirm pred is flattened to [B, 4950]
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target_blocks.view(target_blocks.size(0), -1)

        # Output Losses
        Lr_corrI = correye(target_flat, pred_flat)
        Lr_mse = F.mse_loss(pred_flat, target_flat)
        Lr_marg = distance_loss(target_flat, pred_flat, neighbor=True)

        # Latent Losses
        Lz_corrI = correye(latent, latent)
        Lz_dist = distance_loss(latent, latent, neighbor=False)

        Lr = krak_corrI_weight * Lr_corrI + Lr_marg + krak_mse_weight * Lr_mse
        Lz = krak_latent_weight * (Lz_corrI + Lz_dist)

        if VAE_flag:
            loss = Lr + Lz + Lr_mse + kld_loss
        else:
            loss = Lr + Lz + Lr_mse

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        optimizer.step()

        # Detach for evaluation
        pred_np = pred_flat.detach().cpu().numpy()
        target_np = target_flat.detach().cpu().numpy()

        # Metrics
        mae = np.mean(np.abs(target_np - pred_np), axis=0, keepdims=True)
        mse = np.mean((target_np - pred_np) ** 2, axis=0, keepdims=True)
        tr_mae_subs.append(mae)
        tr_mse_subs.append(mse)

        # Correlation metrics
        demean_corr = np.corrcoef(target_np - mean_train_label, pred_np - mean_train_label)
        org_corr = np.corrcoef(target_np, pred_np)

        for corr_matrix, acc_list in [(demean_corr, tr_corr_subs_demean), (org_corr, tr_corr_subs_org)]:
            top_right = np.split(np.split(corr_matrix, 2, axis=0)[0], 2, axis=1)[1]
            acc_list.append(np.diag(top_right))

        tr_epoch_loss += loss.item()

    # Aggregate metrics
    mae_mean = np.mean(tr_mae_subs)
    mae_std = np.std(tr_mae_subs)
    mse_mean = np.mean(tr_mse_subs)
    mse_std = np.std(tr_mse_subs)

    corr_demean = np.concatenate(tr_corr_subs_demean).reshape(1, -1)
    corr_org = np.concatenate(tr_corr_subs_org).reshape(1, -1)
    corr_demean_mean, corr_demean_std = corr_demean.mean(), corr_demean.std()
    corr_org_mean, corr_org_std = corr_org.mean(), corr_org.std()

    return (tr_epoch_loss, mae_mean, mae_std, mse_mean, mse_std, corr_demean_mean, corr_demean_std, corr_org_mean, corr_org_std)


def whole_model_arch(config):
    write_fpath = config['logging']['sanity_file_pth']
    model_out_root = config['logging']['model_out_root']
    model_details = config['transformer']['model_details']
    # fcn_train = getattr(functions_train, config['training']['fcn_train'])  
    netmat_prep_choise = config['training']['netmat_prep_choise']
    dataset_choice = config['training']['dataset_choice']
    bilateral_condition = config['training']['bilateral_condition'] # both hemispheres instead of 1
    channel_specific_condition = config['training']['channel_specific_condition']
    specific_channel = config['training']['channel_specific_condition']
    overfit_condition = config['training']['overfit_condition']
    train_epoch_range = config['training']['epochs']
    val_epoch = config['training']['val_epoch']
    LR = config['training']['LR']
    batch_size = config['training']['bs']
    parcellation_corr_type = config['training']['parcellation_corr_type']
    VAE_flag = config['training']['VAE_flag']
    from_parcellation = config['data']['from_parcellation']
    translation= config['data']['translation']
    version = config['data']['version']
    model_type = config['data']['model_type']
    krak_mse_weight = config['transformer']['krak_mse_weight']
    krak_latent_weight = config['transformer']['krak_latent_weight']
    krak_corrEYE_weight = config['transformer']['krak_corrEYE_weight']
    # skew_weight = config['transformer']['skew_weight']
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
        hemi_cond = "1L" # or alternatively, 2R

    if dataset_choice == "HCPYA":
        train_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_train_labels.npy") 
        train_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_train_data.npy")#[:, np.newaxis, channel_testing, :] 
        write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)
        
        val_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_validation_labels.npy") 
        val_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_validation_data.npy")#[:, np.newaxis, channel_testing, :]
        write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)

        if TEST_FLAG:
            te_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_test_labels.npy") 
            te_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_test_data.npy")#[:, np.newaxis, channel_testing, :]
            write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)
        
    elif dataset_choice == "ABCD":
        if parcellation_corr_type == "full":
            train_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/train_netmat_clean.npy")
            # train_netmat_np = train_netmat_np.T.to_numpy()
            train_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_train_surf.npy")#[:, np.newaxis, channel_testing, :] 
            write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)

            val_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/val_netmat_clean.npy")
            # val_netmat_np = val_netmat_np.T.to_numpy()
            val_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_val_surf.npy")#[:, np.newaxis, channel_testing, :]
            write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)

            if TEST_FLAG:
                te_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/test_netmat_clean.npy")
                te_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_test_surf.npy")#[:, np.newaxis, channel_testing, :]
                write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)
        elif parcellation_corr_type == "partial":
            train_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/partialnetmat_d100/train_netmat_clean.npy")
            # train_netmat_np = train_netmat_np.T.to_numpy()
            train_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_train_surf.npy")#[:, np.newaxis, channel_testing, :] 
            write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)

            val_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/partialnetmat_d100/val_netmat_clean.npy")
            # val_netmat_np = val_netmat_np.T.to_numpy()
            val_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_val_surf.npy")#[:, np.newaxis, channel_testing, :]
            write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)

            if TEST_FLAG:
                te_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/partialnetmat_d100/test_netmat_clean.npy")
                te_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_test_surf.npy")#[:, np.newaxis, channel_testing, :]
                write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)
    
    # check if any nan or inf values to avoid exploding/vanishing grads
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

    # condition for specific channel not
    if channel_specific_condition:
        cc = specific_channel
        train_surf_np = np.expand_dims(train_surf_np[:,cc,:,:], axis=1)
        val_surf_np = np.expand_dims(val_surf_np[:,cc,:,:], axis=1) # channel axis is 1 so expand that to keep shape BxCxPxV ow you get BxPxV

    if overfit_condition:
        n=config['training']['overfit_condition_sub_range'] # upto how many subjects
        train_netmat_np = train_netmat_np[:n] #random subject(s) to pick to over fit
        train_surf_np = train_surf_np[:n]
        val_netmat_np = val_netmat_np[:n]
        val_surf_np = val_surf_np[:n]

    # Below used to be padding=50 and immediately put into netmats as begining, 
    # but in fcn_prep_data_get_loaders we get the mean and std for preps, so token 
    # must be included after therefore it has been updated
    padding=config['transformer']['padding'] #TODO if model does worse, might be because padding was once done BEFORE putting in fcn_prep_data_get_loaders so revert to test if so
    upper_tri_sz = train_netmat_np.shape[1] # should be SUBx4950 (or node size upper tri count)
    # train_netmat_np = add_start_token_np(train_netmat_np, n=padding)
    # val_netmat_np = add_start_token_np(val_netmat_np, n=padding)

    tr_loader, val_loader, mean_train_label = fcn_prep_data_get_loaders(train_netmat=train_netmat_np, train_surface=train_surf_np, validation_netmat=val_netmat_np, validation_surface=val_surf_np, parcellation_N=from_parcellation, netmat_prep_choice=netmat_prep_choise, b_sz=batch_size, padding=padding, encdec=True, write_fpath=write_fpath)
    if TEST_FLAG:
        tr_loader_for_test, te_loader, _ = fcn_prep_data_get_loaders(train_netmat=train_netmat_np, train_surface=train_surf_np, validation_netmat=te_netmat_np, validation_surface=te_surf_np, parcellation_N=from_parcellation, netmat_prep_choice=netmat_prep_choise, b_sz=te_batch_size, padding=padding, encdec=True, write_fpath=write_fpath)
    
    write_to_file(f"Loaded in data. Tunning on dataset: {dataset_choice}", filepath=write_fpath)

    # hold, input_dim = train_netmat_np.shape # schf100 parcellation
    hold, dim_c, dim_p, dim_v =  train_surf_np.shape
    
    enc_sit_dim = config['transformer']['enc_sit_dim']
    enc_heads = config['transformer']['enc_heads']
    enc_depth = config['transformer']['enc_depth']
    # dec_nhead = config['transformer']['dec_heads']
    dec_depth = config['transformer']['dec_depth']
    dec_input_dim = int( upper_tri_sz + padding ) # should be 4950+50 #config['transformer']['dec_input_dim']
    # emb_dropout = config['transformer']['enc_emb_drop']
    dropout = config['transformer']['enc_drop']
    VAE_latent_dim = config['transformer']['vae_dim']
    latent_length = config['transformer']['latent_length']
    
    model = SiT_BGT_VAE(
        dim_model=enc_sit_dim, # lowkey, i think I can keep dim_model as anything I want! -- only latent_length and decoder_input_dim need compatability
        encoder_depth=enc_depth,
        nhead=enc_heads,
        decoder_input_dim=dec_input_dim, #4950 + 50 start tokens
        decoder_depth=dec_depth,
        VAE_latent_dim=VAE_latent_dim,
        latent_length=latent_length,
        num_channels=dim_c,
        num_patches=dim_p, 
        num_verteces=dim_v,
        dropout=dropout
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
            # if config['AdamW']['scheduler']=='CosineDecay': # TODO currrently only set for CosineDecay bc that is what was used in swMSSiT paper from dahan
            lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max = config['CosineDecay']['T_max'],
                                                        eta_min= config['CosineDecay']['eta_min']
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
    for epoch in range(1, train_epoch_range):
        
        # tr_epoch_loss, across_sub_mae_mean, across_sub_mae_std, across_sub_mse_mean, across_sub_mse_std, across_sub_corr_demean, across_sub_corr_demean_std, across_sub_corr_org, across_sub_corr_org_std = fcn_train(model, padding, krak_mse_weight, krak_latent_weight, krak_corrEYE_weight, tr_loader, mean_train_label, device, optimizer, VAE_flag)
        tr_epoch_loss, across_sub_mae_mean, across_sub_mae_std, across_sub_mse_mean, across_sub_mse_std, across_sub_corr_demean, across_sub_corr_demean_std, across_sub_corr_org, across_sub_corr_org_std = train_SiTBGT_krakenonly_autoregressive(model, krak_mse_weight, krak_latent_weight, krak_corrEYE_weight, tr_loader, mean_train_label, device, optimizer, VAE_flag)
        if scheduler: # if you are using a scheduler, this should be TRUE o.w. FALSE so no need to do the "step" to change LR
            lr_schedule.step() #after each epoch
            curr_lr = lr_schedule.get_lr()[0]
        else:
            curr_lr = LR

        # Convert tensors to floats
        train_loss_value = float(tr_epoch_loss)
        running_train_loss += train_loss_value

        write_to_file('| Training | Epoch - {} | LR - {:.4f}| Loss - {:.4f} | MAE - {:.4f} | MSE = {:.4f} | demeanCorr {:.4f}'.format(epoch, curr_lr, running_train_loss, across_sub_mae_mean, across_sub_mse_mean, across_sub_corr_demean), filepath=write_fpath)

        new_row = pd.DataFrame({'train_mae': [across_sub_mae_mean], 'train_mae_sigma': [across_sub_mae_std], 'train_mse': [across_sub_mse_mean], 'train_mse_sigma': [across_sub_mse_std], 
                                'train_loss': [train_loss_value], 'train_demean_corr': [across_sub_corr_demean], 'train_demean_corr_sigma': [across_sub_corr_demean_std], 'train_orig_corr': [across_sub_corr_org], 
                                'train_orig_corr_sigma': [across_sub_corr_org_std]
                                })
        
        df_train = pd.concat([df_train, new_row], ignore_index=True)
        df_train.to_csv(os.path.join(folder_to_save_losses, 'train_losses_patch.csv'))

        if epoch%val_epoch == 0:
            # grpavg_val_mae, grpstd_val_mae, grpavg_val_mse, grpstd_val_mse, val_deman_corr, val_deman_corr_std, val_orig_corr, val_orig_corr_std = fcn_validate(model, val_loader, mean_train_label, padding, device) 
            grpavg_val_mae, grpstd_val_mae, grpavg_val_mse, grpstd_val_mse, val_deman_corr, val_deman_corr_std, val_orig_corr, val_orig_corr_std = fcn_validate_autoregressive(model, val_loader, mean_train_label, device)
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

    # TESTING #
    if TEST_FLAG:
        # del model # restart
        # model = SiT_BGT_VAE(
        #     dim_model=enc_sit_dim, # lowkey, i think I can keep dim_model as anything I want! -- only latent_length and decoder_input_dim need compatability
        #     encoder_depth=enc_depth,
        #     nhead=enc_heads,
        #     decoder_input_dim=dec_input_dim, #4950 + 50 start tokens
        #     decoder_depth=dec_depth,
        #     VAE_latent_dim=VAE_latent_dim,
        #     latent_length=latent_length,
        #     num_channels=dim_c,
        #     num_patches=dim_p, 
        #     num_verteces=dim_v,
        #     dropout=dropout
        #     )
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
            for i, (mesh_indata, full_target) in enumerate(te_loader):
                mesh_indata = mesh_indata.to(device)
                full_target = full_target.to(device)

                # Split target: [B, 5000] → [B, 100, 50]
                full_target_blocks = full_target.view(full_target.size(0), -1, 50)  # [B, 100, 50]
                start_tokens = full_target_blocks[:, 0, :]  # [B, 50]
                target = full_target[:, 50:]               # [B, 4950]

                # Encoder output (required for SiT_BGT_VAE)
                encoder_out = model.encode(mesh_indata)

                # Autoregressive generation (no teacher forcing)
                pred = model.autoregressive_decode(
                    encoder_out=encoder_out,
                    start_tokens=start_tokens,
                    total_len=5000,
                    block_size=50,
                    teacher=None, # none gives it dummy target 
                    teacher_forcing_ratio=0.0 #NOT using tearcher learning
                ).detach().cpu().numpy()  # [B, 4950]

                target = target.cpu().numpy()

                # MAE & MSE
                mae = np.mean(np.abs(target - pred), axis=0)
                mse = np.mean((target - pred) ** 2, axis=0)
                mae_test_list.append(mae)
                mse_test_list.append(mse)

                te_ground_truth[i, :] = target
                te_pred[i, :] = pred

            write_to_file(f"Done with TESTING loop.", filepath=write_fpath)

            # to optimize testing and data saving, will only get best, mid, and lowest corr
            across_sub_rho = np.corrcoef(te_ground_truth, te_pred) # gives sub_dim*2 x sub_dim*2 and will likely be two square clusters truth and pred
            write_to_file(f"SZ of bigg matrix: {across_sub_rho.shape}", filepath=write_fpath)
            np.save(f"{folder_to_save_test}/te_big_corr_matrix.npy", across_sub_rho) # save for viz later
            
            # find best, and worst corr(truth,pred)
            row_half = np.split(across_sub_rho,2, axis = 0) #split in half across rows
            top_right_quad = np.split(row_half[0],2, axis = 1)[1] # again split by col, and top rigth quaf is corr(y,yhat) so choose 1 automatically == quad2
            find_max_rho = np.argwhere(top_right_quad == np.max(np.diag(top_right_quad)))[0] # find max across daigonal
            find_min_rho = np.argwhere(top_right_quad == np.min(np.diag(top_right_quad)))[0] #find min across diagonal
            max_idx = find_max_rho[0] # which subject had the highest corr across diagonal in quad2
            min_idx = find_min_rho[0] #0 is i, so subject index althougth same as j but keeping consistency
            write_to_file(f"IDX in big TEST corr matrix for both best (max) and worst (min) performance: {max_idx} {min_idx}", filepath=write_fpath)

            # save bet and worst netmat translation
            te_max_netmat_translation = te_pred[max_idx]
            te_min_netmat_translation = te_pred[min_idx]
            np.save(f"{folder_to_save_test}/te_max_netmat_translation.npy", te_max_netmat_translation)
            np.save(f"{folder_to_save_test}/te_min_netmat_translation.npy", te_min_netmat_translation)

            for i, (mesh_indata, full_target) in enumerate(tr_loader_for_test):
                mesh_indata = mesh_indata.to(device)
                full_target = full_target.to(device)

                # Split target: [B, 5000] → [B, 100, 50]
                full_target_blocks = full_target.view(full_target.size(0), -1, 50)  # [B, 100, 50]
                start_tokens = full_target_blocks[:, 0, :]  # [B, 50]
                target = full_target[:, 50:]               # [B, 4950]

                # Encoder output (required for SiT_BGT_VAE)
                encoder_out = model.encode(mesh_indata)

                # Autoregressive generation (no teacher forcing)
                pred = model.autoregressive_decode(
                    encoder_out=encoder_out,
                    start_tokens=start_tokens,
                    total_len=5000,
                    block_size=50,
                    teacher=None, # none gives it dummy target 
                    teacher_forcing_ratio=0.0 #NOT using tearcher learning
                ).detach().cpu().numpy()  # [B, 4950]

                target = target.cpu().numpy()

                # MAE & MSE
                mae = np.mean(np.abs(target - pred), axis=0)
                mse = np.mean((target - pred) ** 2, axis=0)
                mae_train_list.append(mae)
                mse_train_list.append(mse)

                tr_ground_truth[i, :] = target
                tr_pred[i, :] = pred

            write_to_file(f"Done with TRAINING loop.", filepath=write_fpath)

            # to optimize testing and data saving, will only get best, mid, and lowest corr
            across_sub_rho = np.corrcoef(tr_ground_truth, tr_pred) # gives sub_dim*2 x sub_dim*2 and will likely be two square clusters truth and pred
            np.save(f"{folder_to_save_test}/tr_big_corr_matrix.npy", across_sub_rho) # save for viz later
            
            # find best, mid, and worst corr(truth,pred)
            row_half = np.split(across_sub_rho,2, axis = 0) #split in half across rows
            top_right_quad = np.split(row_half[0],2, axis = 1)[1] # again split by col, and top rigth quaf is corr(y,yhat) so choose 1 automatically == quad2
            find_max_rho = np.argwhere(top_right_quad == np.max(np.diag(top_right_quad)))[0] # find max across daigonal
            find_min_rho = np.argwhere(top_right_quad == np.min(np.diag(top_right_quad)))[0] #find min across diagonal
            max_idx = find_max_rho[0] #0 is rows
            min_idx = find_min_rho[0] #0 is i, so subject index althougth same as j but keeping consistency
            write_to_file(f"IDX in big TRAIN corr matrix for both best (max) and worst (min) performance: {max_idx} {min_idx}", filepath=write_fpath)

            # save bet and worst netmat translation
            tr_max_netmat_translation = tr_pred[max_idx]
            tr_min_netmat_translation = tr_pred[min_idx]
            np.save(f"{folder_to_save_test}/tr_max_netmat_translation.npy", tr_max_netmat_translation)
            np.save(f"{folder_to_save_test}/tr_min_netmat_translation.npy", tr_min_netmat_translation)
        
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
    parser = argparse.ArgumentParser(description='kBGTSiT_vae')

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

