import os
import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import yaml

from models.models import *
from utils.utils import *


def train_demean(model, krak_mse_weight, krak_latent_weight, train_loader, mean_train_label, device, optimizer, krak_corrI_weight=1000, reset_params=True):
    model.train()

    targets_ = []
    preds_ = []
    train_mae_subs = []
    train_mse_subs = []
    train_corr_demean_subs = []
    train_corr_orig_subs = []

    for i, data in enumerate(train_loader):
        # inputs, in_targets, targets = data[0].to(device), data[1].to(device), data[2].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
        netmat_indata, mesh_decinput_data, mesh_target_data = data[0].to(device), data[1].to(device), data[2].to(device) #, data[2].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
        
        pred, latent = model(src=netmat_indata, tgt=mesh_decinput_data,  mesh_mask=generate_subsequent_mask(model.ico_patch).to(device))

        vec_tmp = pred
        train_num_sub, num_chnl, num_patches, num_ver = vec_tmp.shape
        vec_pred = vec_tmp.reshape(train_num_sub, num_chnl*num_patches*num_ver)

        # tensor the mean to subtract
        tensor_mean_train_label = torch.tensor(mean_train_label, dtype=torch.float32)
        demean_pred = torch.tensor(vec_pred) - tensor_mean_train_label
        demean_targets = mesh_target_data - tensor_mean_train_label # mesh_target_data already vectorized surface mesh for each subj

        # Output Losses
        Lr_corrI = correye(demean_targets, demean_pred) # corr mat of measured->predicted should be high along diagonal, loww off diagonal 
        Lr_mse = torch.FloatTensor(torch.nn.MSELoss()(demean_targets, demean_pred)) # MSE should be low
        Lr_marg = distance_loss(demean_targets, demean_pred, neighbor=True) # predicted X should be far from nearet ground truth X (for a different subject)

        # Latent Space Losses
        Lz_corrI = correye(latent, latent) # correlation matrix of latent space should be low off diagonal
        Lz_dist = distance_loss(latent, latent, neighbor=False) # mean intersubject altent space distances should be high

        Lr = krak_corrI_weight*Lr_corrI + Lr_marg + (krak_mse_weight * Lr_mse) # weighting MSE with 100,000 (1000 from Krakencoder), Fyzeen OG is 50k SDNR
        Lz = Lz_corrI + Lz_dist

        mae = torch.nn.L1Loss()(demean_targets, demean_pred)
        train_mae_subs.append(mae.detach().numpy())
        train_corr_demean = np.corrcoef(demean_targets.detach().numpy(), demean_pred.detach().numpy())[0,1] # going to be low-ish cause 256->mesh size sphere but curious
        train_corr_demean_subs.append(train_corr_demean)

        train_corr_orig = np.corrcoef(mesh_target_data.detach().numpy(), vec_pred.detach().numpy())[0,1] # going to be low-ish cause 256->mesh size sphere but curious
        train_corr_orig_subs.append(train_corr_orig)

        mse = np.mean( (demean_targets.detach().squeeze().numpy() - demean_pred.squeeze().detach().numpy())**2 )
        train_mse_subs.append(mse)

        loss = Lr + (krak_latent_weight * Lz) #+ train_corr_demean
        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        targets_.append(demean_targets.cpu().numpy())
        preds_.append(demean_pred.cpu().detach().numpy())

    across_sub_mae_mean = np.mean(train_mae_subs)
    across_sub_mse_mean = np.mean(train_mse_subs)
    across_sub_corr_demean = np.mean(train_corr_demean_subs)
    across_sub_corr_orig = np.mean(train_corr_orig_subs)
    
    return across_sub_mae_mean, across_sub_mse_mean, loss, across_sub_corr_demean, across_sub_corr_orig

def validation(model, val_loader, mean_train_label, device, reset_params=True):
    model.eval()
    model.to(device)

    mse_val_list = []
    mae_val_list = []
    demean_corr_val_list = []
    orig_corr_val_list = []

    with torch.no_grad():

        for i, data in enumerate(val_loader):
            netmat_indata, mesh_decinput_data, mesh_target_data = data[0].to(device), data[1].to(device), data[2].to(device) #, data[2].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
            
            pred, latent = model(src=netmat_indata, tgt=mesh_decinput_data,  mesh_mask=generate_subsequent_mask(model.ico_patch).to(device))

            vec_tmp = pred
            train_num_sub, num_chnl, num_patches, num_ver = vec_tmp.shape
            vec_pred = vec_tmp.reshape(train_num_sub, num_chnl*num_patches*num_ver)

            # tensor the mean to subtract
            tensor_mean_train_label = torch.tensor(mean_train_label, dtype=torch.float32)
            demean_pred = vec_pred - tensor_mean_train_label
            demean_targets = mesh_target_data - tensor_mean_train_label # mesh_target_data already vectorized surface mesh for each subj
        
            mae = torch.nn.L1Loss()(demean_targets, demean_pred)
            mae_val_list.append(mae)

            mse = np.mean( (demean_targets.squeeze().detach().numpy() - demean_pred.squeeze().detach().numpy())**2 )
            mse_val_list.append(mse)

            demean_corr = np.corrcoef(demean_targets, demean_pred)[0,1]
            demean_corr_val_list.append(demean_corr)
            orig_corr = np.corrcoef(mesh_target_data, vec_pred)[0,1]
            orig_corr_val_list.append(orig_corr)
    
    return np.mean(mae_val_list), np.mean(mse_val_list), np.mean(demean_corr_val_list), np.mean(orig_corr_val_list)


def whole_model_arch(config):
    ## configuration for model train lenght and type of translation
    write_fpath = config['logging']['sanity_file_pth']
    model_out_root = config['logging']['model_out_root']
    model_details = config['transformer']['model_details']
    train_epoch_range = config['training']['epochs']
    val_epoch = config['training']['val_epoch']
    LR = config['training']['LR']
    batch_size = config['training']['bs']
    from_parcellation = config['data']['from_parcellation']
    to_icamap = config['data']['to_icamap']
    translation= config['data']['translation']
    version = config['data']['version']
    model_type = config['data']['model_type']
    krak_mse_weight = config['transformer']['krak_mse_weight']
    krak_latent_weight = config['transformer']['krak_latent_weight']
    krak_corrEYE_weight = config['transformer']['krak_corrEYE_weight']
    device = "cpu"
    best_mae = 1e+9
    best_demean_rho = int(-1 * 1e+9)


    folder_to_save_model = f'/home/naranjorincon/neurotranslate/netmat2surf/logs/{translation}/{model_type}/{version}'
    folder_to_save_losses = f'{model_out_root}/{translation}/{model_type}/{version}/{model_details}'
    # make necessary folders
    if not os.path.exists(folder_to_save_model):
        # Create the directory
        os.makedirs(folder_to_save_model)
    if not os.path.exists(folder_to_save_losses):
        # Create the directory
        os.makedirs(folder_to_save_losses)

    data_root_path = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"

    # loads in np train data/labels
    train_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_labels.npy") # label = netmat, so TODO is fix these later
    train_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_data.npy") #data = sruf
    # train_surf_np = add_start_patch(train_surf_np) # adds patch for first prediction to sequence
    write_to_file(f'Loaded in data and labels. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)

    train_num_sub, num_chnl, num_patches, num_ver = train_surf_np.shape
    train_surf_chnlxpatchxver = train_surf_np.reshape(train_num_sub, num_chnl*num_patches*num_ver) # concats ver and chnls so [100 x 15*320*153]=[100x734,400]
    
    norm_netmats = (train_netmat_np - np.mean(train_netmat_np, axis=0))/ np.std(train_netmat_np, axis=0)
    train_z_transform_ele = norm_netmats #fisher_z_transform(norm_netmats)
    mean_train_label = np.mean(train_surf_chnlxpatchxver, axis=0)
    write_to_file(f'across subj mean shape: {mean_train_label.shape}', filepath=write_fpath)

    train_netmat_np = make_nemat_allsubj(train_z_transform_ele, from_parcellation) # turns vec into netmat for all subs, second variable is nodes in netmat
    write_to_file(f'Made netmat for each subject. Took label data and reformat to sym netmat. Has now shape: {train_netmat_np.shape}\nAnd surf is: {train_surf_np.shape}', filepath=write_fpath)

    #### LOAD VALIDATION DATA AND SURF
    # loads in np train data/labels
    val_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_validation_labels.npy") # label = netmat, so TODO is fix these later
    val_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_validation_data.npy") #data = sruf
    # val_surf_np = add_start_patch(val_surf_np) # adds patch for first prediction to sequence
    write_to_file(f'Loaded in data and labels. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)

    val_num_sub, _, _, _ = val_surf_np.shape
    val_surf_chnlxpatchxver = val_surf_np.reshape(val_num_sub, num_chnl*num_patches*num_ver) # concats ver and chnls so [100 x 15*320*153]=[100x734,400]

    norm_netmats = (val_netmat_np - np.mean(val_netmat_np, axis=0))/ np.std(val_netmat_np, axis=0)
    val_z_transform_ele = norm_netmats #fisher_z_transform(val_netmat_np)
    val_netmat_np = make_nemat_allsubj(val_z_transform_ele, from_parcellation) # turns vec into netmat for all subs, second variable is nodes in netmat
    write_to_file(f'Made netmat for each subject. Took label data and reformat to sym netmat. Has now shape: {val_netmat_np.shape}\nAnd surf is: {val_surf_np.shape}', filepath=write_fpath)

    #### MODEL DATALOADERS
    # make netmat and add start node(s) -- you need to have an EVEN number of NODES so that model_dim can be even
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_netmat_np).float(), torch.from_numpy(train_surf_np).float(), torch.from_numpy(train_surf_chnlxpatchxver).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=10)
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_netmat_np).float(), torch.from_numpy(val_surf_np).float(), torch.from_numpy(val_surf_chnlxpatchxver).float())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers=10)    

    # write to file
    write_to_file('Loaded in data.', filepath=write_fpath)

    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"

    place_hold, input_dim, conn_profile_num = train_netmat_np.shape # schf100 parcellation
    place_hold2, chnls, patches, verteces =  train_surf_np.shape

    assert chnls == to_icamap, "ICA maps mismatch between specified and derived from data. Make sure both are correct"

    # d_model=conn_profile_num # no self loops 
    write_to_file(f'holder:{place_hold} inputdim:{input_dim} conn_profile:{conn_profile_num}', filepath=write_fpath)

    ## configuration for model params
    enc_input = input_dim
    enc_model_dim = conn_profile_num
    enc_depth = config['transformer']['enc_depth'] #layers
    enc_heads = config['transformer']['enc_heads'] # attn heads
    enc_emb_drop = config['transformer']['enc_emb_drop']# drop out of embedding step
    enc_drop = config['transformer']['enc_drop']  # dropout at transformer layers
    dec_input_dim = config['transformer']['dec_input_dim'] #384, #192-tiny, 384-small, 768-base
    dec_heads = config['transformer']['dec_heads']
    decoder_depth = config['transformer']['decoder_depth']
    dec_channels = chnls
    dec_emb_drop = config['transformer']['dec_emb_drop']
    dec_drop = config['transformer']['dec_drop']
    ico_patch = patches #based on ico sphere patch num 320 is ico-2, our default. +1 because start token added
    ico_vertex = verteces

    # TriuGraphTransformer is OG
    model = BGBMT(enc_input = enc_input,
                 enc_model_dim = enc_model_dim,
                 enc_depth = enc_depth, #layers
                 enc_heads = enc_heads, # attn heads
                 enc_emb_drop = enc_emb_drop, # drop out of embedding step
                 enc_drop = enc_drop,  # dropout at transformer layers
                 dec_input_dim = dec_input_dim, #384, #192-tiny, 384-small, 768-base
                 dec_heads = dec_heads,
                 decoder_depth = decoder_depth,
                 dec_channels = dec_channels,
                 dec_emb_drop = dec_emb_drop,
                 dec_drop = dec_drop,
                 ico_patch = ico_patch, #based on ico sphere patch num 320 is ico-2, our default
                 ico_vertex = ico_vertex
                )
    
    # initialize optimizer / loss
    if config['optimisation']['optimiser']=='Adam':
        write_to_file('using Adam optimiser',  filepath=write_fpath)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=config['Adam']['weight_decay'])
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
        loss_fn = nn.MSELoss()

    # reset params 
    model._reset_parameters()

    running_train_loss = 0
    running_val_loss = 0
    df_train = pd.DataFrame(columns=['train_mae', 'train_mse', 'train_loss', 'train_demean_corr', 'train_orig_corr'])
    df_val = pd.DataFrame(columns=['val_mae', 'val_mse', 'val_loss', 'val_demean_corr', 'val_orig_corr'])

    write_to_file("Begining training.", filepath=write_fpath)

    for epoch in range(1, train_epoch_range):
        
        across_sub_mae_mean, across_sub_mse_mean, loss, train_deman_corr, train_orig_corr = train_demean(model, krak_mse_weight, krak_latent_weight, krak_corrEYE_weight, train_loader, mean_train_label, device, optimizer)
        
        # Convert tensors to floats
        train_loss_value = float(loss.detach().cpu().item())
        running_train_loss += train_loss_value

        write_to_file('| Training | Epoch - {} | Loss - {:.4f} | MAE - {:.4f} | MSE = {:.4f} | demeanCorr {:.4f}'.format(epoch, running_train_loss, across_sub_mae_mean, across_sub_mse_mean, train_deman_corr,), filepath=write_fpath)

        new_row = pd.DataFrame({'train_mae': [across_sub_mae_mean], 'train_mse': [across_sub_mse_mean], 'train_loss': [train_loss_value], 'train_demean_corr': [train_deman_corr], 'train_orig_corr': [train_orig_corr]})
        df_train = pd.concat([df_train, new_row], ignore_index=True)
        df_train.to_csv(os.path.join(folder_to_save_losses, 'train_losses_patch.csv'))

        if epoch%val_epoch == 0:
            grpavg_val_mae, grpavg_val_mse, val_deman_corr, val_orig_corr = validation(model, val_loader, mean_train_label, device)

            write_to_file('| Validation | Epoch - {} | MAE - {:.4f} | MSE = {:.4f} | demeanCorr {:.4f}'.format(epoch, grpavg_val_mae, grpavg_val_mse, val_deman_corr), filepath=write_fpath)

            # curr_val_mae = grpavg_val_mae
            # if curr_val_mae < best_mae:
            #     best_mae = curr_val_mae
            #     # best_epoch = epoch
            #     write_to_file('saving model checkpoint...', filepath=write_fpath)
            #     torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}.pt'))
            
            curr_val_demean_rho = val_deman_corr # prioritize model with best demean correlation performance with validation set
            if curr_val_demean_rho > best_demean_rho:
                best_demean_rho = curr_val_demean_rho

                write_to_file(f'epoch:{epoch} \nsaving model checkpoint...', filepath=write_fpath)
                torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}.pt'))

            new_row = pd.DataFrame({'val_mae': [grpavg_val_mae], 'val_mse': [grpavg_val_mse], 'val_demean_corr': [val_deman_corr], 'val_orig_corr': [val_orig_corr]})
            df_val = pd.concat([df_val, new_row], ignore_index=True)
            df_val.to_csv(os.path.join(folder_to_save_losses, 'val_losses_patch.csv'))

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