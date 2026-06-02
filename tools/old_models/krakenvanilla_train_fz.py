import os
import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

import torch
import numpy as np
import pandas as pd
from models.models import *
from models.kraken_latent_recon_losses import *
from utils.utils import *
from models.krakencoder_model import *
from sklearn.decomposition import PCA

# TEST #

def train_krak(model, train_loader, device, optimizer, reset_params=True):
    model.train()

    targets_ = []
    preds_ = []
    train_mae_subs = []
    train_mse_subs = []
    train_corr_subs = []

    for i, data in enumerate(train_loader):
        inputs, targets = data[0].to(device).squeeze(), data[1].to(device)#.unsqueeze(0) #.squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
        
        latent, pred = model(inputs, 0, 0)
        
        # Output Losses
        Lr_corrI = correye(targets, pred) # corr mat of measured->predicted should be high along diagonal, loww off diagonal 
        Lr_mse = torch.FloatTensor(torch.nn.MSELoss()(pred, targets)) # MSE should be low
        Lr_marg = distance_loss(targets, pred, neighbor=True) # predicted X should be far from nearet ground truth X (for a different subject)

        # Latent Space Losses
        Lz_corrI = correye(latent, latent) # correlation matrix of latent space should be low off diagonal
        Lz_dist = distance_loss(latent, latent, neighbor=False) # mean intersubject altent space distances should be high

        Lr = Lr_corrI + Lr_marg + (1000 * Lr_mse) # weighting MSE with 100,000 (1000 from Krakencoder), Fyzeen OG is 50k SDNR
        Lz = Lz_corrI + Lz_dist

        loss = Lr + (10 * Lz) # weighting Lz with 10 (from Krakencoder)

        mae = torch.nn.L1Loss()(pred, targets)
        train_mae_subs.append(mae.detach().numpy())
        train_corr = np.corrcoef(pred.detach().numpy(),targets.detach().numpy())[0,1] # going to be low-ish cause 256->mesh size sphere but curious
        train_corr_subs.append(train_corr)
        mse = np.mean( (targets.detach().squeeze().numpy() - pred.squeeze().detach().numpy())**2 )
        train_mse_subs.append(mse)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        targets_.append(targets.cpu().numpy())
        preds_.append(pred.cpu().detach().numpy())

    across_sub_mae_mean = np.mean(train_mae_subs)
    across_sub_mse_mean = np.mean(train_mse_subs)
    across_sub_corr_mean = np.mean(train_corr_subs)
    
    return targets_, preds_, loss, across_sub_mae_mean, across_sub_mse_mean, across_sub_corr_mean

def validate_demean(model, train_loader_fortesting, val_loader, device, pca, mean_train_label):
    model.eval()
    model.to(device)

    # mse_train_list = []
    # mae_train_list = []
    corr_train_list = []

    mse_val_list = []
    mae_val_list = []
    corr_val_list = []

    with torch.no_grad():
        for i, data in enumerate(train_loader_fortesting):
            inputs, targets_transform, targets = data[0].to(device), data[1].to(device).squeeze(), data[2].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            latent, pred = model(inputs, 0, 0)

            inverse_pca = pca.inverse_transform(np.expand_dims(pred.squeeze().detach().numpy(), 0))
            corr = np.corrcoef(targets.squeeze().numpy() - mean_train_label, inverse_pca.squeeze(0) - mean_train_label)[0,1]
            corr_train_list.append(corr) # correlation between actual labels and inverse PCA predictions
        
        for i, data in enumerate(val_loader):
            inputs, targets_transform, targets = data[0].to(device), data[1].to(device).squeeze(), data[2].to(device).squeeze()#.unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            latent, pred = model(inputs, 0, 0)

            # Output Losses
            Lr_corrI = correye(targets_transform, pred) # corr mat of measured->predicted should be high along diagonal, loww off diagonal 
            Lr_mse = torch.FloatTensor(torch.nn.MSELoss()(pred, targets_transform)) # MSE should be low
            Lr_marg = distance_loss(targets_transform, pred, neighbor=True) # predicted X should be far from nearet ground truth X (for a different subject)
            # Latent Space Losses
            Lz_corrI = correye(latent, latent) # correlation matrix of latent space should be low off diagonal
            Lz_dist = distance_loss(latent, latent, neighbor=False) # mean intersubject altent space distances should be high
            Lr = Lr_corrI + Lr_marg + (1000 * Lr_mse) # weighting MSE with 100,000 (1000 from Krakencoder), Fyzeen OG is 50k SDNR
            Lz = Lz_corrI + Lz_dist
            val_loss = Lr + (10 * Lz) # weighting Lz with 10 (from Krakencoder)

            mae = np.mean(np.abs(targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy()))
            mae_val_list.append(mae)

            mse = np.mean( (targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy())**2 )
            mse_val_list.append(mse)

            inverse_pca = pca.inverse_transform(np.expand_dims(pred.squeeze().detach().numpy(), 0))
            corr = np.corrcoef(targets.squeeze().numpy() - mean_train_label, inverse_pca.squeeze(0) - mean_train_label)[0,1]
            corr_val_list.append(corr)
    
    return np.mean(mae_val_list), np.mean(mse_val_list), val_loss, np.mean(corr_train_list), np.mean(corr_val_list)


if __name__ == "__main__":
    #some settings
    model_out_root = "/scratch/naranjorincon/NeuroTranslate/netmat2surf/model_out"
    netmat_parcellation_res = 100
    batch_size = 64
    translation="ICAd15_schfd100"
    model_type = "kraken_vanilla_patch"
    version = "fisher_z"
    write_fpath = f"/scratch/naranjorincon/NeuroTranslate/netmat2surf/batch/krakenvanilla_{version}.print"
    train_epoch_range = 501
    val_epoch = 5
    best_mae = 100000000
    chosen_pca_size = 256

############################################# LOAD IN NETMATS AND SURFACE MESHES #############################################
    # loads in np train data/labels
    train_netmat_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_labels.npy")
    train_z_transform_ele = fisher_z_transform(train_netmat_np)
    train_netmat_np = make_nemat_allsubj(train_z_transform_ele, netmat_parcellation_res)
    train_surf_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_data.npy")
    # transform surf to be subs x channels*patch_num*verteces
    train_num_sub, num_chnl, num_patches, num_ver = train_surf_np.shape
    # from what I understand, the reshape is such that its all vertex for each patch of channle 1, then same for chnl2 and so on. SO it is "ordered"
    train_surf_chnlxpatchxver = train_surf_np.reshape(train_num_sub, num_chnl*num_patches* num_ver) # concats ver and chnls so [100 x 15*320*153]=[100x734,400]
    mean_train_label = np.mean(train_surf_chnlxpatchxver, axis=0)
    write_to_file(f'Reformat surf label data shape: {train_surf_chnlxpatchxver.shape}', filepath=write_fpath)

    val_netmat_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/1L_validation_labels.npy")
    val_z_transform_ele = fisher_z_transform(val_netmat_np)
    val_netmat_np = make_nemat_allsubj(val_z_transform_ele, netmat_parcellation_res)
    val_surf_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/1L_validation_data.npy")
    # transform surf to be subs x channels*patch_num*verteces
    val_num_sub, _, _, _ = val_surf_np.shape
    # from what I understand, the reshape is such that its all vertex for each patch of channle 1, then same for chnl2 and so on. SO it is "ordered"
    val_surf_chnlxpatchxver = val_surf_np.reshape(val_num_sub, num_chnl*num_patches* num_ver) # concats ver and chnls so [100 x 15*320*153]=[100x734,400]
    write_to_file(f'Reformat surf label data shape: {val_surf_chnlxpatchxver.shape}', filepath=write_fpath)

    ############################################# compute PCA on data AND output for later use ###################################
    train_data_np = train_netmat_np
    train_label_np = train_surf_chnlxpatchxver
    mean_train_label = np.mean(train_label_np, axis=0)
    val_data_np = val_netmat_np
    val_label_np = val_surf_chnlxpatchxver

    # netmat PCAs
    mat_pca = PCA(n_components=chosen_pca_size)
    mat_pca.fit(train_z_transform_ele)
    train_mat_transform = mat_pca.transform(train_z_transform_ele)
    val_mat_transform = mat_pca.transform(val_z_transform_ele)

    # mesh PCAs
    # train_mesh_flat = train_label_np.reshape(train_label_np.shape[0], -1)  # makes into 2D subject x mesh_data
    # val_mesh_flat = val_label_np.reshape(val_label_np.shape[0], -1) 
    # train_mesh_flat = train_label_np # already 2D so don't think necessary
    # val_mesh_flat = val_label_np
    
    mesh_pca = PCA(n_components=chosen_pca_size)
    mesh_pca.fit(train_label_np)
    train_mesh_transform = mesh_pca.transform(train_label_np)
    val_mesh_transform = mesh_pca.transform(val_label_np)

    ############################################# Set up DataLoader and model params #############################################    
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_mat_transform).float(), torch.from_numpy(train_mesh_transform).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=10)
    
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_mat_transform).float(), torch.from_numpy(val_mesh_transform).float(), torch.from_numpy(val_label_np).float())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers=10)    

    train_dataset_fortesting = torch.utils.data.TensorDataset(torch.from_numpy(train_mat_transform).float(), torch.from_numpy(train_mesh_transform).float(), torch.from_numpy(train_label_np).float())
    train_loader_fortesting = torch.utils.data.DataLoader(train_dataset_fortesting, batch_size = batch_size, shuffle=False, num_workers=10)

    # write to file
    write_to_file("Loaded in data.", filepath=write_fpath)

    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"

    model = Krakencoder([chosen_pca_size])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, eps=1e-9)

    # reset params 
    model._reset_parameters()

    running_train_loss = 0
    running_val_loss = 0
    df_train = pd.DataFrame(columns=['train_mae', 'train_mse', 'train_loss', 'train_corr'])
    df_val = pd.DataFrame(columns=['val_mae', 'val_mse', 'val_loss', 'val_pcacorr', 'train_pcacorr'])

    for epoch in range(1, train_epoch_range):
        targets_, preds_, loss, across_sub_mae_mean, across_sub_mse_mean, across_sub_corr_mean = train_krak(model, train_loader, device, optimizer)

        # Convert tensors to floats
        train_loss_value = float(loss.detach().cpu().item())
        running_train_loss += train_loss_value

        write_to_file('| Training | Epoch - {} | Loss - {:.4f} | MAE - {:.4f} |'.format(epoch+1, running_train_loss, across_sub_mae_mean), filepath=write_fpath)

        folder_to_save_losses = f'{model_out_root}/{translation}/{model_type}/{version}'
        if not os.path.exists(folder_to_save_losses):
            # Create the directory
            os.makedirs(folder_to_save_losses)
            print("Directory for losses created.")
        else:
            print("Directory for losses output already exists.")

        new_row = pd.DataFrame({'train_mae': [across_sub_mae_mean], 'train_mse': [across_sub_mse_mean], 'train_loss': [train_loss_value], 'train_corr': [across_sub_corr_mean]})
        df_train = pd.concat([df_train, new_row], ignore_index=True)
        df_train.to_csv(os.path.join(folder_to_save_losses, 'train_losses_patch.csv'))

        if epoch%val_epoch == 0:
            grpavg_val_mae, grpavg_val_mse, val_loss, train_corr, val_corr = validate_demean(model, train_loader_fortesting, val_loader, device, mesh_pca, mean_train_label)

            loss_value = float(val_loss.detach().cpu().item())
            running_val_loss += loss_value

            write_to_file('| Validation | Epoch - {} | Loss - {:.4f} | MAE - {:.4f} |'.format(epoch, running_val_loss, grpavg_val_mae ), filepath=write_fpath)

            curr_val_mae = grpavg_val_mae
            if curr_val_mae < best_mae:
                best_mae = curr_val_mae
                # best_epoch = epoch
                write_to_file('saving model checkpoint...', filepath=write_fpath)

                folder_to_save_model = f'/scratch/naranjorincon/NeuroTranslate/netmat2surf/logs/{translation}/{model_type}/{version}'
                if not os.path.exists(folder_to_save_model):
                    # Create the directory
                    os.makedirs(folder_to_save_model)
                    print("Directory for model created.")
                else:
                    print("Directory for model output already exists.")

                torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'krakenvanilla_chk_{version}.pt'))

            new_row = pd.DataFrame({'val_mae': [grpavg_val_mae], 'val_mse': [grpavg_val_mse], 'val_loss': [loss_value], 'val_pcacorr': [val_corr], 'train_pcacorr': [train_corr]})
            df_val = pd.concat([df_val, new_row], ignore_index=True)
            df_val.to_csv(os.path.join(folder_to_save_losses, 'val_losses_patch.csv'))
        
