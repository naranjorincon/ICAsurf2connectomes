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
import glob
from models.models import *
from utils.utils import *
from models.ms_sit_unet_shifted import *

def whole_model_arch(config):
    data_root_path = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"
    write_fpath = config['logging']['sanity_file_pth_test']
    model_out_root = config['logging']['model_out_root']
    model_details = config['transformer']['model_details']
    batch_size = config['training']['bs']
    from_parcellation = config['data']['from_parcellation']
    translation= config['data']['translation']
    version = config['data']['version']
    model_type = config['data']['model_type']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #"cpu"
    write_to_file(f'Using: {device} and they are: {torch.cuda.device_count()}', filepath=write_fpath)

    saved_model_path = f'/home/naranjorincon/neurotranslate/surf2netmat/logs/{translation}/{model_type}/{version}'
    folder_to_save_model_testing=f'{model_out_root}/{translation}/{model_type}/{version}/{model_details}'
    folder_to_save_losses = f'{model_out_root}/{translation}/{model_type}/{version}/{model_details}'
    # make necessary folders
    if not os.path.exists(folder_to_save_model_testing):
        # Create the directory
        os.makedirs(folder_to_save_model_testing)
    if not os.path.exists(folder_to_save_losses):
        # Create the directory
        os.makedirs(folder_to_save_losses)

    ############################################# LOAD IN NETMATS AND SURFACE MESHES #############################################
    tr_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_labels.npy") # label = netmat, so TODO is fix these later
    tr_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_data_ico6.npy") #data = sruf
    
    write_to_file(f'Loaded in data and labels. They have shapes: {tr_netmat_np.shape} & {tr_surf_np.shape} respectively.', filepath=write_fpath)

    tr_num_sub, num_chnl, num_ver = tr_surf_np.shape

    tr_surf_chnlxver = tr_surf_np.reshape(tr_num_sub, num_chnl*num_ver) # concats ver and chnls so [100 x 15*320*153]=[100x734,400]
    
    norm_netmats = (tr_netmat_np - np.mean(tr_netmat_np, axis=0))/ np.std(tr_netmat_np, axis=0)
    tr_z_transform_ele = norm_netmats #fisher_z_transform(norm_netmats)
    mean_tr_label = np.mean(tr_surf_chnlxver, axis=0)
    write_to_file(f'across subj mean shape: {mean_tr_label.shape}', filepath=write_fpath)

    tr_netmat_np = make_nemat_allsubj(tr_z_transform_ele, from_parcellation) # turns vec into netmat for all subs, second variable is nodes in netmat
    write_to_file(f'Made netmat for each subject. Took label data and reformat to sym netmat. Has now shape: {tr_netmat_np.shape}\nAnd surf is: {tr_surf_np.shape}', filepath=write_fpath)

    #### LOAD TEST DATA AND SURF
    # loads in np train data/labels
    te_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_test_labels.npy") # label = netmat, so TODO is fix these later
    te_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_test_data_ico6.npy") #data = sruf
    # val_surf_np = add_start_patch(val_surf_np) # adds patch for first prediction to sequence
    write_to_file(f'Loaded in data and labels. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)

    te_num_sub, _, _ = te_surf_np.shape
    te_surf_chnlxver = te_surf_np.reshape(te_num_sub, num_chnl*num_ver) # concats ver and chnls so [100 x 15*320*153]=[100x734,400]

    norm_netmats = (te_netmat_np - np.mean(te_netmat_np, axis=0))/ np.std(te_netmat_np, axis=0)
    te_z_transform_ele = norm_netmats #fisher_z_transform(val_netmat_np)
    te_netmat_np = make_nemat_allsubj(te_z_transform_ele, from_parcellation) # turns vec into netmat for all subs, second variable is nodes in netmat
    write_to_file(f'Made netmat for each subject. Took label data and reformat to sym netmat. Has now shape: {te_netmat_np.shape}\nAnd surf is: {te_surf_np.shape}', filepath=write_fpath)

    #### MODEL DATALOADERS
    # make netmat and add start node(s) -- you need to have an EVEN number of NODES so that model_dim can be even
    tr_dataset = torch.utils.data.TensorDataset(torch.from_numpy(tr_netmat_np).float(), torch.from_numpy((tr_surf_chnlxver - mean_tr_label)).float())
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size = batch_size, shuffle=True, num_workers=10)
    te_dataset = torch.utils.data.TensorDataset(torch.from_numpy(te_netmat_np).float(), torch.from_numpy((te_surf_chnlxver - mean_tr_label)).float())
    te_loader = torch.utils.data.DataLoader(te_dataset, batch_size = batch_size, shuffle=True, num_workers=10)    

    # write to file
    write_to_file("Loaded in DATA.", filepath=write_fpath)

    place_hold, input_dim, conn_profile_num = tr_netmat_np.shape # schf100 parcellation
    place_hold2, chnls, verteces =  tr_surf_np.shape

    model = VAE_LNET_BGT_swMSSiT(
                enc_input = input_dim,
                enc_model_dim = conn_profile_num,
                enc_depth = config['transformer']['enc_depth'], #layers
                enc_heads = config['transformer']['enc_heads'], # attn heads
                enc_emb_drop = config['transformer']['enc_emb_drop'],# drop out of embedding step
                enc_drop = config['transformer']['enc_drop'],  # dropout at transformer layers
                VAE_latent_dim =  config['transformer']['vae_dim'],
                dec_input_dim = config['transformer']['dec_input_dim'], #384, #192-tiny, 384-small, 768-base
                # ico_patch = patches, #based on ico sphere patch num 320 is ico-2, our default
                ico_vertex = verteces,
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
                to_icamap=config['transformer']['to_icamap'],
                reorder=config['mesh_resolution']['reorder'],
                device=device
    )
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # see all models
    # all_models = sorted(glob.glob(f"{saved_model_path}/*.pt"))
    # write_to_file(f"All models in path: {all_models}", filepath=write_fpath)
    # get the one you want below
    te_versions = ["MSE", "MAE", "RHO"]
    for tt in range(len(te_versions)):
        te_version = te_versions[tt]

        model_path = sorted(glob.glob(f"{saved_model_path}/*{te_version}.pt"))
        write_to_file(f'\n\nmodel loaded path is {model_path[-1]}', filepath=write_fpath)
        model.load_state_dict(torch.load(model_path[-1])) # most recent model

        # Find number of parameters
        model_params = sum(p.numel() for p in model.parameters())
        write_to_file(f"\n\nModel params: {model_params}", filepath=write_fpath)

        # Testing below
        model.eval()
        model.to(device)

        write_to_file("Loaded in MODEL.", filepath=write_fpath)

        write_to_file("Begin TESTING.", filepath=write_fpath)
        torch.cuda.empty_cache()

        te_ground_truth = np.zeros(te_surf_chnlxver.shape)
        te_pred = np.zeros(te_surf_chnlxver.shape)
        tr_ground_truth = np.zeros(tr_surf_chnlxver.shape)
        tr_pred = np.zeros(tr_surf_chnlxver.shape)

        with torch.no_grad():
            mse_te_list = []
            mae_te_list = []
            demean_corr_te_list = []
            orig_corr_te_list = []

            mse_tr_list = []
            mae_tr_list = []
            demean_corr_tr_list = []
            orig_corr_tr_list = []
            for i, data in enumerate(te_loader):

                inputs, targets = data[0].to(device), data[1].to(device)

                pred, z_mu, z_variance = model(inputs)
                del inputs, z_mu, z_variance # fee up space

                if i % 100 == 0:
                    write_to_file(f"checkpoint. Running TEST subject: {i}", filepath=write_fpath)

                te_num_sub, num_chnl, num_ver = pred.shape
                # tensor_mean_train_label = torch.tensor(mean_train_label, dtype=torch.float32)
                demean_pred = pred.reshape(te_num_sub, num_chnl*num_ver).cpu().detach().numpy() #- tensor_mean_train_label
                demean_targets = targets.cpu().detach().numpy() #- tensor_mean_train_label # mesh_target_data already vectorized surface mesh for each subj
            
                mae = np.mean( np.abs(demean_targets - demean_pred) )
                mae_te_list.append(mae)

                mse = np.mean( (demean_targets - demean_pred)**2 )
                mse_te_list.append(mse)

                demean_corr = np.corrcoef(demean_targets, demean_pred)[0,1]
                demean_corr_te_list.append(demean_corr)
                orig_corr = np.corrcoef((demean_targets + mean_tr_label), (demean_pred + mean_tr_label))[0,1]
                orig_corr_te_list.append(orig_corr)

                te_ground_truth[i, :] = demean_targets
                te_pred[i, :] = demean_pred


            write_to_file(f"Done with TESTING loop.", filepath=write_fpath)

            for i, data in enumerate(tr_loader):
                inputs, targets = data[0].to(device), data[1].to(device)

                pred, z_mu, z_variance = model(inputs)
                del inputs, z_mu, z_variance # fee up space

                if i % 100 == 0:
                    write_to_file(f"checkpoint. Running TRAIN subject: {i}", filepath=write_fpath)

                tr_num_sub, num_chnl, num_ver = pred.shape
                # tensor_mean_train_label = torch.tensor(mean_train_label, dtype=torch.float32)
                demean_pred = pred.reshape(tr_num_sub, num_chnl*num_ver).cpu().detach().numpy() #- tensor_mean_train_label
                demean_targets = targets.cpu().detach().numpy() #- tensor_mean_train_label # mesh_target_data already vectorized surface mesh for each subj
            
                mae = np.mean( np.abs(demean_targets - demean_pred) )
                mae_tr_list.append(mae)

                mse = np.mean( (demean_targets - demean_pred)**2 )
                mse_tr_list.append(mse)

                demean_corr = np.corrcoef(demean_targets, demean_pred)[0,1]
                demean_corr_tr_list.append(demean_corr)
                orig_corr = np.corrcoef((demean_targets + mean_tr_label), (demean_pred + mean_tr_label))[0,1]
                orig_corr_tr_list.append(orig_corr)

                tr_ground_truth[i, :] = demean_targets
                tr_pred[i, :] = demean_pred

            write_to_file(f"Done with TRAINING loop.", filepath=write_fpath)


        # save training losses
        df_version_mae = pd.DataFrame(mae_tr_list)
        # write_to_file(f'check mae pd creating: {df_version_mae.head()}. \n being sent to {folder_to_save_model}', filepath=write_fpath)
        df_version_mae.to_csv(os.path.join(folder_to_save_model_testing, f'mae_train_model_{te_version}.csv'))
        df_version_mse = pd.DataFrame(mse_tr_list)
        df_version_mse.to_csv(os.path.join(folder_to_save_model_testing, f'mse_train_model_{te_version}.csv'))
        # save test losses
        df_version_mae = pd.DataFrame(mae_te_list)
        # write_to_file(f'check mae pd creating: {df_version_mae.head()}. \n being sent to {folder_to_save_model}', filepath=write_fpath)
        df_version_mae.to_csv(os.path.join(folder_to_save_model_testing, f'mae_test_model_{te_version}.csv'))
        df_version_mse = pd.DataFrame(mse_te_list)
        df_version_mse.to_csv(os.path.join(folder_to_save_model_testing, f'mse_test_model_{te_version}.csv'))

        write_to_file("TRAIN Mean MAE:", filepath=write_fpath)
        write_to_file(np.mean(mae_tr_list), filepath=write_fpath)
        write_to_file("TEST Mean MAE:", filepath=write_fpath)
        write_to_file(np.mean(mae_te_list), filepath=write_fpath)

        write_to_file("TRAIN Mean MSE:", filepath=write_fpath)
        write_to_file(np.mean(mse_tr_list), filepath=write_fpath)
        write_to_file("TEST Mean MSE:", filepath=write_fpath)
        write_to_file(np.mean(mse_te_list), filepath=write_fpath)

        np.save(f"{folder_to_save_model_testing}/train_ground_truth.npy", tr_ground_truth)
        np.save(f"{folder_to_save_model_testing}/train_pred_{te_version}.npy", tr_pred)
        np.save(f"{folder_to_save_model_testing}/test_ground_truth.npy", te_ground_truth)
        np.save(f"{folder_to_save_model_testing}/test_pred_{te_version}.npy", te_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='unet_VAE_BGT_swMSSiT_test')

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

