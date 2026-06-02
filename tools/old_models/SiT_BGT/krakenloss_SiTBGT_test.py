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
from models.models import *
from utils.utils import *

def whole_model_arch(config):
    model_out_root = config['logging']['model_out_root']
    model_details = config['transformer']['model_details']
    netmat_prep_choise = config['training']['netmat_prep_choise']
    dataset_choice = config['training']['dataset_choice']
    bilateral_condition = config['training']['bilateral_condition'] # both hemispheres instead of 1
    batch_size = config['testing']['bs_test']
    parcellation_corr_type = config['training']['parcellation_corr_type']
    from_parcellation = config['data']['from_parcellation']
    translation= config['data']['translation']
    version = config['data']['version']
    chosen_test_model = config['testing']['chosen_test_model']
    model_type = config['data']['model_type']
    device = "cpu"
    saved_model_path = f'/home/naranjorincon/neurotranslate/surf2netmat/logs/{translation}/{dataset_choice}/{model_type}/{version}'
    folder_to_save_model=f'{model_out_root}/{translation}/{dataset_choice}/{model_type}/{version}/{model_details}/{chosen_test_model}'
    folder_to_save_losses = f'{model_out_root}/{translation}/{dataset_choice}/{model_type}/{version}/{model_details}'
    # make necessary folders
    if not os.path.exists(folder_to_save_model):
        # Create the directory
        os.makedirs(folder_to_save_model)
    if not os.path.exists(folder_to_save_losses):
        # Create the directory
        os.makedirs(folder_to_save_losses)

    data_root_path = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"
    sanity_check_version = config['logging']['sanity_check_version']
    write_fpath = f"{data_root_path}/NeuroTranslate/surf2netmat/batch/{model_type}_{sanity_check_version}_{dataset_choice}_test.print"

    ############################################# LOAD IN NETMATS AND SURFACE MESHES #############################################
    if bilateral_condition:
        hemi_cond = "2LR"
    else:
        hemi_cond = "1L" # or alternatively, 2R

    if dataset_choice == "HCPYA":
        tr_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_train_labels.npy") 
        tr_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_train_data.npy")#[:, np.newaxis, channel_testing, :] 
        write_to_file(f'Loaded in TRAIN. They have shapes: {tr_netmat_np.shape} & {tr_surf_np.shape} respectively.', filepath=write_fpath)
        
        te_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_test_labels.npy") 
        te_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_test_data.npy")#[:, np.newaxis, channel_testing, :]
        write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)
    
    elif dataset_choice == "ABCD":
        if parcellation_corr_type == "full":
            tr_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/train_netmat_clean.npy")
            tr_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_train_surf.npy")#[:, np.newaxis, channel_testing, :] 
            write_to_file(f'Loaded in TRAIN. They have shapes: {tr_netmat_np.shape} & {tr_surf_np.shape} respectively.', filepath=write_fpath)

            te_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/test_netmat_clean.npy")
            te_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_test_surf.npy")#[:, np.newaxis, channel_testing, :]
            write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)

        elif parcellation_corr_type == "partial":
            tr_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/partialnetmat_d100/train_netmat_clean.npy")
            tr_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_train_surf.npy")#[:, np.newaxis, channel_testing, :] 
            write_to_file(f'Loaded in TRAIN. They have shapes: {tr_netmat_np.shape} & {tr_surf_np.shape} respectively.', filepath=write_fpath)

            te_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/partialnetmat_d100/test_netmat_clean.npy")
            te_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_test_surf.npy")#[:, np.newaxis, channel_testing, :]
            write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)
    
    # check if any nan or inf values to avoid exploding/vanishing grads
    surf_check_nan = np.isnan(tr_surf_np).sum()
    surf_check_inf = np.isinf(tr_surf_np).sum()
    netmat_check_nan = np.isnan(tr_netmat_np).sum()
    netmat_check_inf = np.isinf(tr_netmat_np).sum()
    total_train = surf_check_nan + surf_check_inf + netmat_check_nan + netmat_check_inf
    write_to_file(f'TRAINING COUNTS: {surf_check_nan} - {surf_check_inf} - {netmat_check_nan} - {netmat_check_inf}', filepath=write_fpath)

    surf_check_nan = np.isnan(te_surf_np).sum()
    surf_check_inf = np.isinf(te_surf_np).sum()
    netmat_check_nan = np.isnan(te_netmat_np).sum()
    netmat_check_inf = np.isinf(te_netmat_np).sum()
    total_test = surf_check_nan + surf_check_inf + netmat_check_nan + netmat_check_inf
    write_to_file(f'TEST COUNTS: {surf_check_nan} - {surf_check_inf} - {netmat_check_nan} - {netmat_check_inf}', filepath=write_fpath)

    assert total_train == 0, "Nan/Inf in TRAIN. Check."
    assert total_test == 0, "Nan/Inf in TEST. Check."

    # adds start token to *_label_np
    padding=50
    upper_tri_sz = tr_netmat_np.shape[1]
    # tr_netmat_np = add_start_token_np(tr_netmat_np, n=padding)
    # te_netmat_np = add_start_token_np(te_netmat_np, n=padding)

    tr_loader, te_loader, mean_train_label = fcn_prep_data_get_loaders(train_netmat=tr_netmat_np, train_surface=tr_surf_np, validation_netmat=te_netmat_np, validation_surface=te_surf_np, parcellation_N=from_parcellation, netmat_prep_choice=netmat_prep_choise, b_sz=batch_size, padding=padding, write_fpath=write_fpath)

    ############################################# Set up Test run and model configs #############################################

    # write to file    
    write_to_file("Loaded in data.", filepath=write_fpath)

    hold, dim_c, dim_p, dim_v =  tr_surf_np.shape
    
    enc_sit_dim = config['transformer']['enc_sit_dim']
    enc_heads = config['transformer']['enc_heads']
    enc_depth = config['transformer']['enc_depth']
    # dec_nhead = config['transformer']['dec_heads']
    dec_depth = config['transformer']['dec_depth']
    dec_input_dim = int( upper_tri_sz + padding )
    # emb_dropout = config['transformer']['enc_emb_drop']
    dropout = config['transformer']['enc_drop']
    # VAE_latent_dim = config['transformer']['vae_dim']
    latent_length = config['transformer']['latent_length']
    
    model = SiT_BGT(
        dim_model=enc_sit_dim, # lowkey, i think I can keep dim_model as anything I want! -- only latent_length and decoder_input_dim need compatability
        encoder_depth=enc_depth,
        nhead=enc_heads,
        decoder_input_dim=dec_input_dim, #4950 + 50 start tokens
        decoder_depth=dec_depth,
        latent_length=latent_length,
        num_channels=dim_c,
        num_patches=dim_p, 
        num_verteces=dim_v,
        dropout=dropout
        )
    
    # see all models
    model_path = sorted(glob.glob(f"{saved_model_path}/*{model_details}_{chosen_test_model}.pt")) # look at training script for details, but all models saves as type_details_chosen: ex-kBGTLN_d6h5_demeanL2_skewloss_RHO.pt
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
        ss, nn = tr_netmat_np.shape
        ss = 2 * ss
        tr_ground_truth = np.zeros((ss,nn))
        tr_pred = np.zeros((ss,nn))
        te_ground_truth = np.zeros((ss,nn))
        te_pred = np.zeros((ss,nn))
    else:
        tr_ground_truth = np.zeros(tr_netmat_np.shape)
        tr_pred = np.zeros(tr_netmat_np.shape)
        te_ground_truth = np.zeros(te_netmat_np.shape)
        te_pred = np.zeros(te_netmat_np.shape)

    with torch.no_grad():
        for i, data in enumerate(te_loader):
            mesh_indata, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0) #, data[2].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
            dec_input = targets
            allvals = model(src=mesh_indata, tgt=dec_input,  tgt_mask=generate_subsequent_mask(model.latent_length).to(device))
            pred = allvals[0]
            pred = pred[padding:]
            targets = targets[:, padding:]

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
        np.save(f"{folder_to_save_model}/te_big_corr_matrix.npy", across_sub_rho) # save for viz later
        
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
        np.save(f"{folder_to_save_model}/te_max_netmat_translation.npy", te_max_netmat_translation)
        np.save(f"{folder_to_save_model}/te_min_netmat_translation.npy", te_min_netmat_translation)

        for i, data in enumerate(tr_loader):
            mesh_indata, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0) #, data[2].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
            dec_input = targets
            allvals = model(src=mesh_indata, tgt=dec_input,  tgt_mask=generate_subsequent_mask(model.latent_length).to(device))
            pred = allvals[0]

            pred = pred[padding:]
            targets = targets[:, padding:]

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
        np.save(f"{folder_to_save_model}/tr_big_corr_matrix.npy", across_sub_rho) # save for viz later
        
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
        np.save(f"{folder_to_save_model}/tr_max_netmat_translation.npy", tr_max_netmat_translation)
        np.save(f"{folder_to_save_model}/tr_min_netmat_translation.npy", tr_min_netmat_translation)
    
    # save training losses
    df_version_mae = pd.DataFrame(mae_train_list)
    # write_to_file(f'check mae pd creating: {df_version_mae.head()}. \n being sent to {folder_to_save_model}', filepath=write_fpath)
    df_version_mae.to_csv(os.path.join(folder_to_save_model, 'mae_train_model.csv'))
    df_version_mse = pd.DataFrame(mse_train_list)
    df_version_mse.to_csv(os.path.join(folder_to_save_model, 'mse_train_model.csv'))
    # save test losses
    df_version_mae = pd.DataFrame(mae_test_list)
    # write_to_file(f'check mae pd creating: {df_version_mae.head()}. \n being sent to {folder_to_save_model}', filepath=write_fpath)
    df_version_mae.to_csv(os.path.join(folder_to_save_model, 'mae_test_model.csv'))
    df_version_mse = pd.DataFrame(mse_test_list)
    df_version_mse.to_csv(os.path.join(folder_to_save_model, 'mse_test_model.csv'))

    write_to_file("TRAIN Mean MAE:", filepath=write_fpath)
    write_to_file(np.nanmean(mae_train_list), filepath=write_fpath)
    write_to_file("TEST Mean MAE:", filepath=write_fpath)
    write_to_file(np.nanmean(mae_test_list), filepath=write_fpath)

    write_to_file("TRAIN Mean MSE:", filepath=write_fpath)
    write_to_file(np.nanmean(mse_train_list), filepath=write_fpath)
    write_to_file("TEST Mean MSE:", filepath=write_fpath)
    write_to_file(np.nanmean(mse_test_list), filepath=write_fpath)

    np.save(f"{folder_to_save_model}/train_ground_truth.npy", tr_ground_truth)
    np.save(f"{folder_to_save_model}/train_pred.npy", tr_pred)
    np.save(f"{folder_to_save_model}/test_ground_truth.npy", te_ground_truth)
    np.save(f"{folder_to_save_model}/test_pred.npy", te_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='kSiTBGT')

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

    