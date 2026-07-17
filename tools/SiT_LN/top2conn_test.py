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
    icores = config['data']['icores']   
    translation= config['data']['translation'] 
    model_out_root = config['logging']['model_out_root']
    version = config['data']['version'] #normICAdemeanfishzMAT
    model_type = config['data']['model_type']
    parcellation_name = config['data']['parcellation_name']
    from_parcellation = config['data']['from_parcellation']
    netmat_prep_choice = config['training']['netmat_prep_choice']
    fcn_model_module = getattr(models, config['training']['fcn_model_to_use']) 
    flag_experiment_ICArecon = config['training']['flag_experiment_ICArecon']
    channel_specific_condition = config['training']['channel_specific_condition']
    parcellation_corr_type = config['training']['parcellation_corr_type']
    hemi_cond = config['training']['hemi_cond']
    bilateral_condition = config['training']['bilateral_condition']
    if flag_experiment_ICArecon:
        to_icamap = config['data']['to_icamap']
        specific_channel = config['training']['specific_channel']
        specific_channel_end = config['training']['specific_channel_end']
        model_details = config['transformer']['model_details'].format(hemi_cond,parcellation_corr_type,translation,netmat_prep_choice)
        write_fpath = config['logging']['test_file_pth'].format(model_type, version, parcellation_name, from_parcellation,parcellation_corr_type)
    else:
        model_details = config['transformer']['model_details'].format(hemi_cond,parcellation_corr_type,translation,netmat_prep_choice)
        write_fpath = config['logging']['test_file_pth'].format(model_type, version, parcellation_name, from_parcellation,parcellation_corr_type)

    surf_prep_choice = config['training']['surf_prep_choice']
    dataset_choice = config['training']['dataset_choice']
    bilateral_condition = config['training']['bilateral_condition'] # both hemispheres instead of 1
    translation = config['data']['translation']
    VAE_flag = config['training']['VAE_flag']
    device = "cpu"
    overfit_condition = config['training']['overfit_condition']
    
    te_batch_size = config['testing']['bs_test']
    same_sample_test = config['testing']['same_sample_test']
    out_of_sample_test = config['testing']['out_of_sample_test']
    if out_of_sample_test:
        dataset_choice = "HCPYA_" + dataset_choice + "dr"
        assert dataset_choice == "HCPYA_ABCDdr", "out of distribution TRUE, but dataset for data is same distribution. Verify."
        write_to_file(f"Out of sample flag: {out_of_sample_test}", filepath=write_fpath)

    chosen_test_model = config['testing']['chosen_test_model']
    # path_to_model=f'/home/naranjorincon/neurotranslate/surf2netmat/logs/{translation}/ABCD/{model_type}/{version}'
    model_save_path=config['logging']['model_save_path'] #/home/naranjorincon/neurotranslate/surf2netmat/logs
    path_to_model=f'{model_save_path}/{translation}/{dataset_choice}/{model_type}/{version}'
    # path_to_model = f'{model_save_path}/{translation}/{dataset_choice}/{model_type}/{version}'

    ############################################# LOAD IN NETMATS AND SURFACE MESHES #############################################
    data_root_path = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"
    # for TESTING #
    chosen_test_model = config['testing']['chosen_test_model']
    folder_to_save_test=f'{model_out_root}/{translation}/{dataset_choice}/{model_type}/{version}/{model_details}/{chosen_test_model}'
    if not os.path.exists(folder_to_save_test):
        # Create the directory
        os.makedirs(folder_to_save_test)

    ############################################# LOAD IN NETMATS AND SURFACE MESHES #############################################
    if dataset_choice == "HCPYA":
        if parcellation_corr_type == "full":
            train_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_train_labels.npy") 
            train_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_train_data.npy")#[:, np.newaxis, channel_testing, :] 
            # val_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_validation_labels.npy") 
            # val_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_validation_data.npy")#[:, np.newaxis, channel_testing, :]
            te_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_test_labels.npy")
            te_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_test_data.npy")#[:, np.newaxis, channel_testing, :]
    elif dataset_choice == "ABCD":
        if translation == "ICAd15_glasserd360":
            main_brainrep_data_path_root=f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/maps_and_netmats/topo2glasserd360_{parcellation_corr_type}"
        elif translation == "ICAd15_ICAnetmatd15":
            main_brainrep_data_path_root=f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/maps_and_netmats/topo2toponetmat_{parcellation_corr_type}"
        else:     #translation == "INFOMAPd20_schaeferd100":
            main_brainrep_data_path_root=f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/maps_and_netmats/topo2schaeferd{from_parcellation}_{parcellation_corr_type}"
        # based on above path chosen
        train_netmat_np = np.load(f"{main_brainrep_data_path_root}/train_{hemi_cond}_vecnetmat_uppertri.npy")
        train_surf_np = np.load(f"{main_brainrep_data_path_root}/train_{hemi_cond}_surf.npy")
        # val_netmat_np = np.load(f"{main_brainrep_data_path_root}/validation_{hemi_cond}_vecnetmat_uppertri.npy")
        # val_surf_np = np.load(f"{main_brainrep_data_path_root}/validation_{hemi_cond}_surf.npy")
        # if TEST_FLAG is True:
        te_netmat_np = np.load(f"{main_brainrep_data_path_root}/test_{hemi_cond}_vecnetmat_uppertri.npy")
        te_surf_np = np.load(f"{main_brainrep_data_path_root}/test_{hemi_cond}_surf.npy")
        write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)
        # write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)
        write_to_file(f'Loaded in VALIDATION. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)
        
        #2LR option
        if bilateral_condition is True:
            train_netmat_np_R = np.load(f"{main_brainrep_data_path_root}/train_1R_vecnetmat_uppertri.npy")
            train_surf_np_R = np.load(f"{main_brainrep_data_path_root}/train_1R_surf.npy")
            # val_netmat_np_R = np.load(f"{main_brainrep_data_path_root}/validation_1R_vecnetmat_uppertri.npy")
            # val_surf_np_R = np.load(f"{main_brainrep_data_path_root}/validation_1R_surf.npy")
            # if TEST_FLAG is True:
            te_netmat_np_R = np.load(f"{main_brainrep_data_path_root}/test_1R_vecnetmat_uppertri.npy")
            te_surf_np_R = np.load(f"{main_brainrep_data_path_root}/test_1R_surf.npy")

            #concat them
            train_netmat_np = np.concatenate((train_netmat_np,train_netmat_np_R),axis=0)
            train_surf_np = np.concatenate((train_surf_np,train_surf_np_R),axis=0)
            # val_netmat_np = np.concatenate((val_netmat_np,val_netmat_np_R),axis=0)
            # val_surf_np = np.concatenate((val_surf_np,val_surf_np_R),axis=0)
            # if TEST_FLAG is True:
            te_netmat_np = np.concatenate((te_netmat_np,te_netmat_np_R),axis=0)
            te_surf_np = np.concatenate((te_surf_np,te_surf_np_R),axis=0)
            
            write_to_file('BILATERAL CONDITION IS TRUE. Loaded in R and concat such that its LLLLL...LRRR..R', filepath=write_fpath)
            write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)
            # write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)
            write_to_file(f'Loaded in VALIDATION. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)

    elif dataset_choice == "infomap_prior_ABCDdr":
        if translation == "INFOMAPd20_glasserd360":
            main_brainrep_data_path_root=f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/maps_and_netmats/topo2glasserd360_{parcellation_corr_type}"
        elif translation == "INFOMAPd20_INFOMAPnetmatd20":
            main_brainrep_data_path_root=f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/maps_and_netmats/topo2toponetmat_{parcellation_corr_type}"
        else:     #translation == "INFOMAPd20_schaeferd100":
            main_brainrep_data_path_root=f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/maps_and_netmats/topo2schaeferd{from_parcellation}_{parcellation_corr_type}"
        
        train_netmat_np = np.load(f"{main_brainrep_data_path_root}/train_{hemi_cond}_vecnetmat_uppertri.npy")
        train_surf_np = np.load(f"{main_brainrep_data_path_root}/train_{hemi_cond}_surf.npy")
        # val_netmat_np = np.load(f"{main_brainrep_data_path_root}/validation_{hemi_cond}_vecnetmat_uppertri.npy")
        # val_surf_np = np.load(f"{main_brainrep_data_path_root}/validation_{hemi_cond}_surf.npy")
        # if TEST_FLAG is True:
        te_netmat_np = np.load(f"{main_brainrep_data_path_root}/test_{hemi_cond}_vecnetmat_uppertri.npy")
        te_surf_np = np.load(f"{main_brainrep_data_path_root}/test_{hemi_cond}_surf.npy")
        write_to_file(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath)
        # write_to_file(f'Loaded in VALIDATION. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath)
        write_to_file(f'Loaded in VALIDATION. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)

    assert train_surf_np.shape[0] == train_netmat_np.shape[0]
    # assert val_surf_np.shape[0] == val_netmat_np.shape[0]
    assert te_surf_np.shape[0] == te_netmat_np.shape[0]

    # check if any nan or inf values to avoid exploding/vanishing grads
    if overfit_condition:
        n=config['training']['overfit_condition_sub_range'] # upto how many subjects
        write_to_file(f'Overfit CONDITION is true, using {n} subjects', filepath=write_fpath)
        train_netmat_np = train_netmat_np[:n] #random subject(s) to pick to over fit
        train_surf_np = train_surf_np[:n]
        # 10 percent of train N
        # val_netmat_np = val_netmat_np[:int(n*0.1)]
        # val_surf_np = val_surf_np[:int(n*0.1)]
        te_netmat_np = te_netmat_np[:int(n*0.1)]
        te_surf_np = te_surf_np[:int(n*0.1)] 

    # condition for specific channel not
    if channel_specific_condition is True:
        if type(specific_channel_end) is list:
            chnl_range = np.arange(0,to_icamap)
            mask = ~np.isin(chnl_range, specific_channel_end)
            final_chnls = chnl_range[mask]
            write_to_file(f'Channels chosen to stay: {final_chnls}', filepath=write_fpath)
            
            train_surf_np = train_surf_np[:,final_chnls,:,:]
            write_to_file(f'SHAPE: {train_surf_np.shape} -- should be NxPxV', filepath=write_fpath)
            # val_surf_np = val_surf_np[:,final_chnls,:,:]
            te_surf_np = te_surf_np[:,final_chnls,:,:]
        elif specific_channel == specific_channel_end:
            cc = specific_channel
            write_to_file(f'SPECIFIC CHANNEL CHOSEN: {cc}', filepath=write_fpath)
            train_surf_np = train_surf_np[:,cc,:,:]
            write_to_file(f'SHAPE: {train_surf_np.shape} -- should be NxPxV', filepath=write_fpath)
            # val_surf_np = val_surf_np[:,cc,:,:]
            te_surf_np = te_surf_np[:,cc,:,:]
            train_surf_np = np.expand_dims(train_surf_np, axis=1)
            # val_surf_np = np.expand_dims(val_surf_np, axis=1) # channel axis is 1 so expand that to keep shape BxCxPxV ow you get BxPxV
            te_surf_np = np.expand_dims(te_surf_np, axis=1)
        else:
            cc = specific_channel
            # specific_channel_end=cc+1
            train_surf_np = train_surf_np[:,cc:specific_channel_end,:,:]
            write_to_file(f'SHAPE: {train_surf_np.shape} -- should be NxPxV', filepath=write_fpath)
            # val_surf_np = val_surf_np[:,cc:specific_channel_end,:,:]
            te_surf_np = te_surf_np[:,cc:specific_channel_end,:,:]
    
        write_to_file(f'We expand on channel dim now. TRAIN SHAPE: {train_surf_np.shape} -- should be Nx1xPxV after expansion', filepath=write_fpath)
        # write_to_file(f'We expand on channel dim now. VAL SHAPE: {val_surf_np.shape} -- should be Nx1xPxV after expansion', filepath=write_fpath)
        write_to_file(f'We expand on channel dim now. TEST SHAPE: {te_surf_np.shape} -- should be Nx1xPxV after expansion', filepath=write_fpath)

    if flag_experiment_ICArecon:
        write_to_file(f"CHOSEN TO RECONSTRUCT ICA MAPs! {flag_experiment_ICArecon}", filepath=write_fpath)       
        tr_loader_for_test, te_loader, _, train_subjects_to_keep, test_subjects_to_keep = fcn_prep_data_get_loaders_ICAren(train_surface=train_surf_np, validation_surface=te_surf_np, b_sz=te_batch_size, write_fpath=write_fpath)
    else:
        write_to_file("regular training, not ICA recon.", filepath=write_fpath)
        tr_loader_for_test, te_loader, _, train_subjects_to_keep, test_subjects_to_keep = fcn_prep_data_get_loaders(train_netmat=train_netmat_np, train_surface=train_surf_np, validation_netmat=te_netmat_np, validation_surface=te_surf_np, parcellation_N=from_parcellation, netmat_prep_choice=netmat_prep_choice, surf_prep_choice=surf_prep_choice, b_sz=te_batch_size, write_fpath=write_fpath, bilateral_condition=bilateral_condition)
        
    write_to_file(f"Loaded in data. Tunning on dataset: {dataset_choice}", filepath=write_fpath)
    write_to_file(f"len of test subejct to keep {len(test_subjects_to_keep)}", filepath=write_fpath)
    # because any parcellation given is NxN symm matrix, no need to netmat.shape to get sizes, we already know them from "from_parcellation" variable
    _, dim_c, dim_p, dim_v = train_surf_np.shape
    _, upper_tri = train_netmat_np.shape
    # del train_surf_np #already converted to tensors for optimization, del to not take up so much mem anymore than necessary

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

    write_to_file('TEST FLAG ON. TESTING.', filepath=write_fpath)
    # see all models
    write_to_file(f"path is \n\n{path_to_model}/*{model_details}_{chosen_test_model}.pt \n\n", filepath=write_fpath)
    
    model_path = sorted(glob.glob(f"{path_to_model}/*{model_details}_{chosen_test_model}.pt")) # look at training script for details, but all models saves as type_details_chosen: ex-kBGTLN_d6h5_demeanL2_skewloss_RHO.pt
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

    N_test = len(te_loader.dataset)
    N_train = len(tr_loader_for_test.dataset)
    # print(f"\n\n TRAIN SIZE{N_train} TEST SIZE{N_test}\n\n")
    write_to_file(f"\n\n TRAIN SIZE{N_train} TEST SIZE{N_test}\n\n", filepath=write_fpath)
    if flag_experiment_ICArecon:
        reshaped_data_test = te_surf_np.reshape(N_test, dim_c*dim_p*dim_v)
        tr_ground_truth = np.zeros((N_train, dim_c*dim_p*dim_v))
        tr_pred = np.zeros((N_train, dim_c*dim_p*dim_v))
        te_ground_truth = np.zeros(reshaped_data_test.shape)
        te_pred = np.zeros(reshaped_data_test.shape)
    else:
        tr_ground_truth = np.zeros((N_train, upper_tri))
        tr_pred = np.zeros((N_train, upper_tri)) #SUBx4950 of zeros
        te_ground_truth = np.zeros((N_test, upper_tri))
        te_pred = np.zeros((N_test, upper_tri))

    with torch.no_grad():
        for i, data in enumerate(te_loader):
            inputs, targets = data[0].to(device), data[1].to(device)#.squeeze()
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

    #save subjects that were kept i.e. had good data and were not scrubbed during cleaning
    test_subjects_to_keep.to_csv(os.path.join(folder_to_save_test, 'test_subjects_to_keep.csv'))

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

    