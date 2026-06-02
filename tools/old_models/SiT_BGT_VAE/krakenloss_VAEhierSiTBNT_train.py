import os
import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')
sys.path.append('../../../')

import torch
import numpy as np
import pandas as pd
import argparse
import yaml
import torch as nn
from models.models import *
from utils.utils import *
from models.ms_sit_unet_shifted import *

def train_demean_BGTswMSSiT(model, krak_mse_weight, krak_latent_weight, krak_corrI_weight, train_loader, mean_train_label, device, optimizer, write_fpath, reset_params=True):
    torch.cuda.empty_cache()
    model.train()
    # targets_ = []
    # preds_ = []
    train_mae_subs = []
    train_mse_subs = []
    train_corr_demean_subs = []
    train_corr_orig_subs = []
    running_loss = 0

    for i, data in enumerate(train_loader):
        inputs, mesh_target_data = data[0].to(device), data[1].to(device)
        # write_to_file(f'Loaded in data for Train. Shapes: inputs:{inputs.shape} targets:{mesh_target_data.shape}', filepath=write_fpath)
        
        pred, z_mu, z_variance = model(inputs)
        del inputs # not needed anymore
        # write_to_file(f'Model ran, outputs provided. Shapes: pred-{pred.shape} mu-{z_mu.shape} sigma-{z_variance.shape}', filepath=write_fpath)

        # vec_tmp = pred # for this architecture, its B x C x ico6_verteces
        train_num_sub, num_chnl, num_verteces = pred.shape

        # tensor the mean to subtract
        # tensor_mean_train_label = torch.tensor(mean_train_label, dtype=torch.float32)
        demean_pred = pred.reshape(train_num_sub, num_chnl*num_verteces) #- tensor_mean_train_label
        demean_targets = mesh_target_data #- tensor_mean_train_label # mesh_target_data already vectorized surface mesh for each subj
        
        # Output Losses
        Lr_corrI = correye(demean_targets, demean_pred) # corr mat of measured->predicted should be high along diagonal, loww off diagonal 
        Lr_mse = torch.nn.MSELoss()(demean_targets,demean_pred) # MSE should be low
        Lr_marg = distance_loss(demean_targets, demean_pred, neighbor=True) # predicted X should be far from nearet ground truth X (for a different subject)
        # kl_loss = -0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4]) #unet MSSiT KL-loss used
        kl_loss = -0.5 * torch.sum(1 + (torch.log(z_variance)) - (z_mu.pow(2)) - (z_variance), dim=0) #unet MSSiT KL-loss used
        KL_div_loss = torch.sum(kl_loss) / train_num_sub # batch size so we get mean over batch

        # Latent Space Losses
        Lz_corrI = correye(z_mu, z_mu) # correlation matrix of latent space should be low off diagonal
        Lz_dist = distance_loss(z_mu, z_mu, neighbor=False) # mean intersubject altent space distances should be high

        Lr = krak_corrI_weight*Lr_corrI + Lr_marg + (krak_mse_weight * Lr_mse) # weighting MSE with 100,000 (1000 from Krakencoder), Fyzeen OG is 50k SDNR
        # write_to_file(f'CHECK Lr: Lr_corrI:{Lr_corrI} Lr_marg:{Lr_marg} Lr_mse:{Lr_mse}', filepath=write_fpath)
        Lz = Lz_corrI + Lz_dist
        # reset mem space now that Lr and Lz are calculated
        del Lr_corrI, Lr_marg, Lr_mse, Lz_corrI, Lz_dist, z_mu, z_variance
        # write_to_file(f'LATENT loss defined.', filepath=write_fpath)

        mae = np.mean( np.abs(demean_targets.cpu().detach().numpy() -  demean_pred.cpu().detach().numpy()) )
        train_mae_subs.append(mae)
        mse = np.mean( (demean_targets.cpu().detach().numpy() - demean_pred.cpu().detach().numpy())**2 )
        train_mse_subs.append(mse)
        
        train_corr_demean = np.corrcoef(demean_targets.cpu().detach().numpy(), demean_pred.cpu().detach().numpy())[0,1] # going to be low-ish cause 256->mesh size sphere but curious
        train_corr_demean_subs.append(train_corr_demean)

        train_corr_orig = np.corrcoef((mesh_target_data.cpu().detach().numpy() + mean_train_label), (demean_pred.cpu().detach().numpy() + mean_train_label))[0,1] # going to be low-ish cause 256->mesh size sphere but curious
        train_corr_orig_subs.append(train_corr_orig)

        loss = Lr + (krak_latent_weight * Lz) + KL_div_loss #+ train_corr_demean
        del Lr, Lz, KL_div_loss
        # write_to_file(f"CUDA mem at train BEFORE loss. {torch.cuda.memory_allocated(device=device)}", filepath=write_fpath)
        
        loss.backward()
        running_loss += loss.item()
        del loss

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # write_to_file(f'Completed run of train for iteration {i}', filepath=write_fpath)

    across_sub_mae_mean = np.mean(train_mae_subs)
    across_sub_mse_mean = np.mean(train_mse_subs)
    across_sub_corr_demean = np.mean(train_corr_demean_subs)
    across_sub_corr_orig = np.mean(train_corr_orig_subs)
    
    return across_sub_mae_mean, across_sub_mse_mean, running_loss, across_sub_corr_demean, across_sub_corr_orig

def validation(model, val_loader, mean_train_label, device, write_fpath, reset_params=True):
    model.eval()
    model.to(device)

    mse_val_list = []
    mae_val_list = []
    demean_corr_val_list = []
    orig_corr_val_list = []

    with torch.no_grad():

        for i, data in enumerate(val_loader):                        
            inputs, mesh_target_data = data[0].to(device), data[1].to(device)

            pred, z_mu, z_variance = model(inputs)
            del inputs, z_mu, z_variance # fee up space
            
            train_num_sub, num_chnl, num_ver = pred.shape
            # tensor_mean_train_label = torch.tensor(mean_train_label, dtype=torch.float32)
            demean_pred = pred.reshape(train_num_sub, num_chnl*num_ver) #- tensor_mean_train_label
            demean_targets = mesh_target_data #- tensor_mean_train_label # mesh_target_data already vectorized surface mesh for each subj
        
            mae = np.mean( np.abs(demean_targets.cpu().detach().numpy() - demean_pred.cpu().detach().numpy()) )
            mae_val_list.append(mae)

            mse = np.mean( (demean_targets.cpu().detach().numpy() - demean_pred.cpu().detach().numpy())**2 )
            mse_val_list.append(mse)

            demean_corr = np.corrcoef(demean_targets.cpu().detach().numpy(), demean_pred.cpu().detach().numpy())[0,1]
            demean_corr_val_list.append(demean_corr)
            orig_corr = np.corrcoef((mesh_target_data.cpu().detach().numpy() + mean_train_label), (demean_pred.cpu().detach().numpy() + mean_train_label))[0,1]
            orig_corr_val_list.append(orig_corr)
    
    return np.mean(mae_val_list), np.mean(mse_val_list), np.mean(demean_corr_val_list), np.mean(orig_corr_val_list)


def whole_model_arch(config):

    data_root_path = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"
    write_fpath = config['logging']['sanity_file_pth']
    model_out_root = config['logging']['model_out_root']
    model_details = config['transformer']['model_details']
    train_epoch_range = config['training']['epochs']
    val_epoch = config['training']['val_epoch']
    LR = config['training']['LR']
    batch_size = config['training']['bs']
    from_parcellation = config['data']['from_parcellation']
    translation= config['data']['translation']
    version = config['data']['version']
    model_type = config['data']['model_type']
    krak_mse_weight = config['transformer']['krak_mse_weight']
    krak_latent_weight = config['transformer']['krak_latent_weight']
    krak_corrEYE_weight = config['transformer']['krak_corrEYE_weight']
    # gpu = config['training']['gpu']
    # if gpu is not None:
    #     write_to_file(f'Using GPU(s): {gpu}', filepath=write_fpath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #"cpu"
    write_to_file(f'Using: {device} and they are: {torch.cuda.device_count()}', filepath=write_fpath)
    best_mae = 1e+9
    best_mse = 1e+9
    best_demean_rho = int(-1 * 1e+9)

    folder_to_save_model = f'/home/naranjorincon/neurotranslate/surf2netmat/logs/{translation}/{model_type}/{version}'
    folder_to_save_losses = f'{model_out_root}/{translation}/{model_type}/{version}/{model_details}'
    # make necessary folders
    if not os.path.exists(folder_to_save_model):
        # Create the directory
        os.makedirs(folder_to_save_model)
    if not os.path.exists(folder_to_save_losses):
        # Create the directory
        os.makedirs(folder_to_save_losses)

    ############################################# LOAD IN NETMATS AND SURFACE MESHES #############################################
    channel_testing = 1 #1 is DMN, when indexing with 0
    train_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_labels.npy") # label = netmat, so TODO is fix these later
    train_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_data_ico6.npy")[:, np.newaxis, channel_testing, :] #adding np.newaxis when doing channel index to keep subk x 1 x verteces
    
    # train_surf_np = add_start_patch(train_surf_np) # adds patch for first prediction to sequence
    write_to_file(f'Loaded in data and labels. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)

    train_num_sub, num_chnl, num_ver = train_surf_np.shape

    train_surf_chnlxver = train_surf_np.reshape(train_num_sub, num_chnl*num_ver) # concats ver and chnls so [100 x 15*320*153]=[100x734,400]
    
    norm_netmats = (train_netmat_np - np.mean(train_netmat_np, axis=0))/ np.std(train_netmat_np, axis=0)
    train_z_transform_ele = norm_netmats #fisher_z_transform(norm_netmats)
    mean_train_label = np.mean(train_surf_chnlxver, axis=0)
    write_to_file(f'across subj mean shape: {mean_train_label.shape}', filepath=write_fpath)

    train_netmat_np = make_nemat_allsubj(train_z_transform_ele, from_parcellation) # turns vec into netmat for all subs, second variable is nodes in netmat
    write_to_file(f'Made netmat for each subject. Took label data and reformat to sym netmat. Has now shape: {train_netmat_np.shape}\nAnd surf is: {train_surf_np.shape}', filepath=write_fpath)

    #### LOAD VALIDATION DATA AND SURF
    # loads in np train data/labels
    val_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_validation_labels.npy") # label = netmat, so TODO is fix these later
    val_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_validation_data_ico6.npy")[:, np.newaxis, channel_testing, :]
    # val_surf_np = add_start_patch(val_surf_np) # adds patch for first prediction to sequence
    write_to_file(f'Loaded in data and labels. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)

    val_num_sub, _, _ = val_surf_np.shape
    val_surf_chnlxver = val_surf_np.reshape(val_num_sub, num_chnl*num_ver) # concats ver and chnls so [100 x 15*320*153]=[100x734,400]

    norm_netmats = (val_netmat_np - np.mean(val_netmat_np, axis=0))/ np.std(val_netmat_np, axis=0)
    val_z_transform_ele = norm_netmats #fisher_z_transform(val_netmat_np)
    val_netmat_np = make_nemat_allsubj(val_z_transform_ele, from_parcellation) # turns vec into netmat for all subs, second variable is nodes in netmat
    write_to_file(f'Made netmat for each subject. Took label data and reformat to sym netmat. Has now shape: {val_netmat_np.shape}\nAnd surf is: {val_surf_np.shape}', filepath=write_fpath)

    #### MODEL DATALOADERS
    # make netmat and add start node(s) -- you need to have an EVEN number of NODES so that model_dim can be even
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_netmat_np).float(), torch.from_numpy((train_surf_chnlxver - mean_train_label)).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=10)
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_netmat_np).float(), torch.from_numpy((val_surf_chnlxver - mean_train_label)).float())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers=10)    

    # remove large data
    del train_surf_chnlxver, val_surf_chnlxver

    # write to file
    write_to_file("Loaded in DATA.", filepath=write_fpath)

    place_hold, input_dim, conn_profile_num = train_netmat_np.shape # schf100 parcellation
    place_hold2, chnls, verteces =  train_surf_np.shape

    model = VAE_LNET_BGT_swMSSiT(
                enc_input = input_dim,
                enc_model_dim = conn_profile_num,
                enc_depth = config['transformer']['enc_depth'], #layers
                enc_heads = config['transformer']['enc_heads'], # attn heads
                enc_emb_drop = config['transformer']['enc_emb_drop'],# drop out of embedding step
                enc_drop = config['transformer']['enc_drop'],  # dropout at transformer layers
                decoder_model_dim = config['transformer']['decoder_model_dim'],
                VAE_latent_dim =  config['transformer']['vae_dim'],
                dec_input_dim = config['transformer']['dec_input_dim'], #384, #192-tiny, 384-small, 768-base
                channels=chnls,
                norm_layer = nn.LayerNorm,
                mlp_ratio=config['transformer']['mlp_ratio'],
                qkv_bias=config['transformer']['qkv_bias'], #default
                qk_scale=config['transformer']['qk_scale'], #default
                dropout=config['transformer']['dropout'],
                attention_dropout=config['transformer']['attention_dropout'],
                dropout_path=config['transformer']['dropout_path'], #default
                depths=config['transformer']['depth'],
                num_heads=config['transformer']['heads'],
                window_size=config['transformer']['window_size'],
                window_size_factor=config['transformer']['window_size_factor'],
                path_to_workdir=config['data']['path_to_workdir'],#default
                ico_init_resolution=config['mesh_resolution']['ico_grid'],#5=default for segmentation task
                reorder=config['mesh_resolution']['reorder'],
                device=device
    )
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model._reset_parameters()
    model.to(device)
    # Find number of parameters
    model_params = sum(p.numel() for p in model.parameters())
    write_to_file(f"Model PARAMS: {model_params}", filepath=write_fpath)

    write_to_file("Loaded in MODEL.", filepath=write_fpath)

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
        # loss_fn = nn.MSELoss()

    write_to_file('', filepath=write_fpath)
    write_to_file('#'*30, filepath=write_fpath)
    write_to_file('######## BEGINING TRAINING ########', filepath=write_fpath)
    write_to_file('#'*30, filepath=write_fpath)
    write_to_file('', filepath=write_fpath)
    if  config['MODEL'] == 'ms-sit':
        write_to_file('Mesh resolution - ico {}'.format(config['mesh_resolution']['ico_mesh']), filepath=write_fpath)
        write_to_file('Grid resolution - ico {}'.format(config['mesh_resolution']['ico_grid']), filepath=write_fpath)
        # write_to_file('Number of patches - {}'.format(patches), filepath=write_fpath)
        write_to_file('Number of vertices - {}'.format(verteces), filepath=write_fpath)
        write_to_file('Reorder patches: {}'.format(config['mesh_resolution']['reorder']), filepath=write_fpath)
        write_to_file('', filepath=write_fpath)


    running_train_loss = 0
    # running_val_loss = 0
    df_train = pd.DataFrame(columns=['train_mae', 'train_mse', 'train_loss', 'train_demean_corr', 'train_orig_corr'])
    df_val = pd.DataFrame(columns=['val_mae', 'val_mse', 'val_loss', 'val_demean_corr', 'val_orig_corr'])

    write_to_file("Begining training.", filepath=write_fpath)
    torch.cuda.empty_cache()
    for epoch in range(1, train_epoch_range):
        
        across_sub_mae_mean, across_sub_mse_mean, running_loss, train_deman_corr, train_orig_corr = train_demean_BGTswMSSiT(model, krak_mse_weight, krak_latent_weight, krak_corrEYE_weight, train_loader, mean_train_label, device, optimizer, write_fpath)
        # write_to_file(f"CUDA mem AFTER train:{epoch}. {torch.cuda.memory_allocated(device=device)}", filepath=write_fpath)
        # if epoch%10 == 0:
        #     write_to_file(f"Sumamry of mem usage AFTER TRAIN: \n\n{torch.cuda.memory_summary()}\n\n", filepath=write_fpath)

        # Convert tensors to floats
        # train_loss_value = float(loss.detach().cpu().item())
        running_train_loss += running_loss
        write_to_file('| Training | Epoch - {} | Loss - {:.4f} | MAE - {:.4f} | MSE = {:.4f} | demeanCorr {:.4f}'.format(epoch, running_train_loss, across_sub_mae_mean, across_sub_mse_mean, train_deman_corr), filepath=write_fpath)

        new_row = pd.DataFrame({'train_mae': [across_sub_mae_mean], 'train_mse': [across_sub_mse_mean], 'train_loss': [running_train_loss], 'train_demean_corr': [train_deman_corr], 'train_orig_corr': [train_orig_corr]})
        df_train = pd.concat([df_train, new_row], ignore_index=True)
        df_train.to_csv(os.path.join(folder_to_save_losses, 'train_losses_patch.csv'))
        del running_loss, running_train_loss

        if epoch%val_epoch == 0:
            grpavg_val_mae, grpavg_val_mse, val_deman_corr, val_orig_corr = validation(model, val_loader, mean_train_label, device, write_fpath)
            write_to_file('| Validation | Epoch - {} | MAE - {:.4f} | MSE = {:.4f} | demeanCorr {:.4f}'.format(epoch, grpavg_val_mae, grpavg_val_mse, val_deman_corr), filepath=write_fpath)

            # save model with best MSE - gives leeway to values around 0 so maybe betetr for correlation values?
            curr_val_mse = grpavg_val_mse
            if curr_val_mse < best_mse:
                best_mse = curr_val_mse
                write_to_file('saving MSE model checkpoint...', filepath=write_fpath)
                torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}_MSE.pt'))
            # save model with best MAE - forces values closer to 0
            curr_val_mae = grpavg_val_mae
            if curr_val_mae < best_mae:
                best_mae = curr_val_mae
                write_to_file('saving MAE model checkpoint...', filepath=write_fpath)
                torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}_MAE.pt'))
            # save model with best RHO_demean
            curr_val_demean_rho = val_deman_corr # prioritize model with best demean correlation performance with validation set
            if curr_val_demean_rho > best_demean_rho:
                best_demean_rho = curr_val_demean_rho
                write_to_file('saving RHO model checkpoint...', filepath=write_fpath)
                torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}_RHO.pt'))

            new_row = pd.DataFrame({'val_mae': [grpavg_val_mae], 'val_mse': [grpavg_val_mse], 'val_demean_corr': [val_deman_corr], 'val_orig_corr': [val_orig_corr]})
            df_val = pd.concat([df_val, new_row], ignore_index=True)
            df_val.to_csv(os.path.join(folder_to_save_losses, 'val_losses_patch.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='unet_VAE_BGT_swMSSiT')

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

