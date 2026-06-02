
import os
import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

import torch
# import torch.nn as nn
# import torch.optim as optim
import numpy as np
import glob
# import pandas as pd

from models.models import *
from utils.utils import *

## configuration for model params

if __name__ == "__main__":
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.set_printoptions(threshold=10_000)
    translation="schfd100_ICAd15" # needs to be "" type of string
    model_type = "krakenBGBMT_patch"
    model_details = "d6h5_small_enc_d6h6_dec_adam_demeanL2" #"d6h5_small_enc_d6h6_dec_adam"
    version = "normMATrawICA"
    model_out_root = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/netmat2surf/model_out"
    write_fpath = f"/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/netmat2surf/batch/{model_type}_test.print"
    recon_ico_path = f"{model_out_root}/{translation}/recon_spheres/kraken/{model_type}/{version}/{model_details}"
    saved_model_path = f'/home/naranjorincon/neurotranslate/netmat2surf/logs/{translation}/{model_type}/{version}'
    
    folder_to_save_model=f'{model_out_root}/{translation}/{model_type}/{version}/{model_details}/'
    
    if not os.path.exists(folder_to_save_model):
        # Create the directory
        os.makedirs(folder_to_save_model)
        write_to_file("Directory for recon ico created.", filepath=write_fpath)
    else:
        write_to_file("Directory for recon ico already exists.", filepath=write_fpath)


    if not os.path.exists(recon_ico_path):
        # Create the directory
        os.makedirs(recon_ico_path)
        print("Directory for recon ico created.")
    else:
        print("Directory for recon ico already exists.")

    batch_size = 1
    netmat_parcellation_res = 100 # look at what translation is taking place. OG is formatted for schf100 to ICA 15

    # loads in np train/test data/labels
    test_netmat_np = np.load("/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/surface-vision-transformers/data/ICAd15_schfd100/template/1L_test_labels.npy")
    test_surf_np = np.load("/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/surface-vision-transformers/data/ICAd15_schfd100/template/1L_test_data.npy")
    sub_dim_test, chnl_dim, patch_dim, vert_dim = test_surf_np.shape

    train_netmat_np = np.load("/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_labels.npy") # label = netmat, so TODO is fix these later
    train_surf_np = np.load("/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_data.npy") #data = surf
    sub_dim_train, _, _, _ = train_surf_np.shape

    # old way was not fisherZ transform netmat so commented out
    # make netmat and add start node(s) -- you need to have an EVEN number of NODES so that model_dim can be even
    # train_data_np = make_nemat_allsubj(train_netmat_np, netmat_parcellation_res) # turns vec into netmat for all subs, second variable is nodes in netmat
    # test_data_np = make_nemat_allsubj(test_netmat_np, netmat_parcellation_res) # turns vec into netmat for all subs, second variable is nodes in netmat
    
    # fisher transform
    train_z_transform_ele = fisher_z_transform(train_netmat_np)
    train_data_np = make_nemat_allsubj(train_z_transform_ele, netmat_parcellation_res)
    
    test_z_transform_ele = fisher_z_transform(test_netmat_np)
    test_data_np = make_nemat_allsubj(test_z_transform_ele, netmat_parcellation_res)
    
    write_to_file('Made netmat for each subject. Took netmat nodes and reformat to sym netmat. Has now train shape: {} and test shape:{}'.format(train_data_np.shape, test_data_np.shape), filepath=write_fpath)

    # Creates out arrays
    train_surf_chnlxpatchxver = train_surf_np.reshape(sub_dim_train, chnl_dim*patch_dim* vert_dim)
    train_ground_truth = np.zeros(train_surf_chnlxpatchxver.shape)
    train_pred = np.zeros(train_surf_chnlxpatchxver.shape)

    test_surf_chnlxpatchxver = test_surf_np.reshape(sub_dim_test, chnl_dim*patch_dim* vert_dim)
    test_ground_truth = np.zeros(test_surf_chnlxpatchxver.shape)
    test_pred = np.zeros(test_surf_chnlxpatchxver.shape)

    # read numpy files into torch dataset and dataloader
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data_np).float(), torch.from_numpy(test_surf_np).float(), torch.from_numpy(test_surf_np).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=10)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_surf_np).float(), torch.from_numpy(train_surf_np).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=False, num_workers=10)

    write_to_file('Loaded in data.', filepath=write_fpath)

    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    place_hold, input_dim, conn_profile_num=train_data_np.shape # schf100 parcellation
    d_model=conn_profile_num # no self loops 
    write_to_file(f'holder:{place_hold} inputdim:{input_dim} conn_profile:{conn_profile_num}', filepath=write_fpath)

    enc_input = input_dim
    enc_model_dim = conn_profile_num
    enc_depth = 6 #layers
    enc_heads = 5 # attn heads
    enc_emb_drop = 0.1 # drop out of embedding step
    enc_drop = 0.3  # dropout at transformer layers
    dec_input_dim = 384 #384, #192-tiny, 384-small, 768-base
    dec_heads = 6
    dec_depth = 6
    dec_channels = 15
    dec_emb_drop = 0.1
    dec_drop = 0.3
    ico_patch = 320 #based on ico sphere patch num 320 is ico-2, our default
    ico_vertex = 153

    # TriuGraphTransformer is OG
    model = BGBMT(enc_input = enc_input,
                 enc_model_dim = enc_model_dim,
                 enc_depth = enc_depth, #layers
                 enc_heads = enc_heads, # attn heads
                 enc_emb_drop = enc_emb_drop, # drop out of embedding step
                 enc_drop = enc_drop,  # dropout at transformer layers
                 dec_input_dim = dec_input_dim, #384, #192-tiny, 384-small, 768-base
                 dec_heads = dec_heads,
                 decoder_depth = dec_depth,
                 dec_channels = dec_channels,
                 dec_emb_drop = dec_emb_drop,
                 dec_drop = dec_drop,
                 ico_patch = ico_patch, #based on ico sphere patch num 320 is ico-2, our default
                 ico_vertex = ico_vertex
                )

    
    # model_path = sorted(glob.glob(f"{saved_model_path}/kraken*_d{enc_depth}h{enc_heads}*d{dec_depth}h{dec_heads}_dec*.pt"))
    # model_path = sorted(glob.glob(f"{saved_model_path}/krakenBGBMT_patch_d12h05_small_enc_d12h05_dec_adam_chk_fisher_z.pt"))
    model_path = sorted(glob.glob(f"{saved_model_path}/*{model_details}*.pt"))
    
    write_to_file(f"All models in path: {model_path}", filepath=write_fpath)
    write_to_file(f'model loaded path is {model_path[-1]}', filepath=write_fpath)
    model.load_state_dict(torch.load(model_path[-1])) # most recent model

    # model_path = sorted(glob.glob(f"/scratch/naranjorincon/NeuroTranslate/netmat2surf/logs/{model_type}*.pt"))
    # write_to_file(f'model loaded path is {model_path[-1]}', filepath=write_fpath)
    # model.load_state_dict(torch.load(model_path[-1])) # most recent model

    model.eval()
    model.to(device)

    # Find number of parameters
    model_params = sum(p.numel() for p in model.parameters())
    write_to_file(f'Model params: {model_params}', filepath=write_fpath)

    train_targets = []
    test_targets = []

    train_preds_decoder = []
    test_preds_decoder = []

    train_maes = []
    train_mse = []
    test_maes = []
    test_mse = []

    # path_tosave_losses=f'{model_out_root}/{translation}/{model_type}/'
    with torch.no_grad():

        for i, data in enumerate(test_loader):
            netmat_indata, mesh_target_data, mesh_dec_input = data[0].to(device).squeeze().unsqueeze(0), data[1].to(device), data[2].to(device) #.unsqueeze(0)#, data[2].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1
            # write_to_file(f'Dataloader TEST shapes: {netmat_indata.shape}, {mesh_target_data.shape}', filepath=write_fpath)

            decoder_pred, out = BGBMT_mesh_greedy_decode(model=model, source=netmat_indata, dec_channels=model.dec_channels, ico_patch=model.ico_patch, ico_vertex=model.ico_vertex, device=device, b=1, target=mesh_dec_input)  # this function output is (1) batchx100x100 and (2) batchx4950

            # test_ground_truth[i, :, :, :] = mesh_target_data.numpy()
            # test_pred[i, :, :, :] = out.detach().numpy()

            if not os.path.isfile(f"{recon_ico_path}/test_pred_L_sub-{i+1:04d}_ico6.shape.gii"): # makes both so only need to check if one is there or not
                matrix_to_mesh(input_mat=out, tri_indices_ico6subico2_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/netmat2surf/utils/triangle_indices_ico_6_sub_ico_2.csv", out_fpath=f"{recon_ico_path}/test_pred_L_sub-{i+1:04d}_ico6")
                matrix_to_mesh(input_mat=mesh_target_data, tri_indices_ico6subico2_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/netmat2surf/utils/triangle_indices_ico_6_sub_ico_2.csv", out_fpath=f"{recon_ico_path}/test_true_L_sub-{i+1:04d}_ico6")

                write_to_file(f"NO EXIST, making it: test_pred_L_sub-{i+1:04d}_ico6", filepath=write_fpath)

            # need the additional index to skip first patch which is a token
            # matrix_to_mesh(input_mat=out, tri_indices_ico6subico2_fpath="/scratch/naranjorincon/NeuroTranslate/netmat2surf/utils/triangle_indices_ico_6_sub_ico_2.csv", out_fpath=f"{recon_ico_path}/test_pred_L_sub-{i+1:04d}_ico6")
            # matrix_to_mesh(input_mat=mesh_target_data, tri_indices_ico6subico2_fpath="/scratch/naranjorincon/NeuroTranslate/netmat2surf/utils/triangle_indices_ico_6_sub_ico_2.csv", out_fpath=f"{recon_ico_path}/test_true_L_sub-{i+1:04d}_ico6")

            if i % 20 == 0:
                write_to_file(f"checkpoint. Running TEST subject: {i}", filepath=write_fpath)
            
            out = out.squeeze().detach().numpy() 
            mesh_target_data = mesh_target_data.squeeze().numpy()

            mae = np.mean(np.abs(out - mesh_target_data))
            test_maes.append(mae)

            mse = np.mean((out - mesh_target_data)**2)
            test_mse.append(mse)

            test_ground_truth[i, :] = mesh_target_data.reshape(1, chnl_dim*patch_dim* vert_dim)
            test_pred[i, :] = out.reshape(1, chnl_dim*patch_dim* vert_dim)

            # save test losses
            df_version_mae = pd.DataFrame(test_maes)
            # write_to_file(f'check mae pd creating: {df_version_mae.head()}. \n being sent to {folder_to_save_model}', filepath=write_fpath)
            df_version_mae.to_csv(os.path.join(folder_to_save_model, 'mae_test_model.csv'))
            df_version_mse = pd.DataFrame(test_mse)
            df_version_mse.to_csv(os.path.join(folder_to_save_model, 'mse_test_model.csv'))
            np.save('{}/test_ground_truth.npy'.format(folder_to_save_model), test_ground_truth)
            np.save('{}/test_pred.npy'.format(folder_to_save_model), test_pred)

        write_to_file(f"Done with TESTING loop.", filepath=write_fpath)
        
        for i, data in enumerate(train_loader):
            netmat_indata, mesh_target_data, mesh_dec_input = data[0].to(device).squeeze().unsqueeze(0), data[1].to(device), data[2].to(device) #.unsqueeze(0)#, data[2].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1
            # write_to_file(f'Dataloader shapes: {netmat_indata.shape}, {mesh_target_data.shape}', filepath=write_fpath)

            decoder_pred, out = BGBMT_mesh_greedy_decode(model=model, source=netmat_indata, dec_channels=model.dec_channels, ico_patch=model.ico_patch, ico_vertex=model.ico_vertex, device=device, b=1, target=mesh_dec_input)  # this function output is (1) batchx100x100 and (2) batchx4950

            # train_ground_truth[i, :, :, :] = mesh_target_data.numpy()
            # train_pred[i, :, :, :] =  out#.detach().numpy()

            # need the additional index to skip first patch which is a token
            # below needs indexing of [:, :,1:,:] to skip first dummy patch, not necessary if no dummy patch
            if not os.path.isfile(f"{recon_ico_path}/train_pred_L_sub-{i+1:04d}_ico6.shape.gii"): # makes both so only need to check if one is there or not
                matrix_to_mesh(input_mat=out, tri_indices_ico6subico2_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/netmat2surf/utils/triangle_indices_ico_6_sub_ico_2.csv", out_fpath=f"{recon_ico_path}/train_pred_L_sub-{i+1:04d}_ico6")
                matrix_to_mesh(input_mat=mesh_target_data, tri_indices_ico6subico2_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/netmat2surf/utils/triangle_indices_ico_6_sub_ico_2.csv", out_fpath=f"{recon_ico_path}/train_true_L_sub-{i+1:04d}_ico6")

                write_to_file(f"NO EXIST, making it: train_pred_L_sub-{i+1:04d}_ico6", filepath=write_fpath)

            # matrix_to_mesh(input_mat=out, tri_indices_ico6subico2_fpath="/scratch/naranjorincon/NeuroTranslate/netmat2surf/utils/triangle_indices_ico_6_sub_ico_2.csv", out_fpath=f"{recon_ico_path}/train_pred_L_sub-{i+1:04d}_ico6")
            # matrix_to_mesh(input_mat=mesh_target_data, tri_indices_ico6subico2_fpath="/scratch/naranjorincon/NeuroTranslate/netmat2surf/utils/triangle_indices_ico_6_sub_ico_2.csv", out_fpath=f"{recon_ico_path}/train_true_L_sub-{i+1:04d}_ico6")

            if i % 100 == 0:
                write_to_file(f"checkpoint. Running TRAIN subject: {i}", filepath=write_fpath)
            
            out = out.squeeze().detach().numpy() 
            mesh_target_data = mesh_target_data.squeeze().numpy()

            mae = np.mean(np.abs(out - mesh_target_data))
            train_maes.append(mae)

            mse = np.mean((out - mesh_target_data)**2)
            train_mse.append(mse)

            train_ground_truth[i, :] = mesh_target_data.reshape(1, chnl_dim*patch_dim* vert_dim)
            train_pred[i, :] = out.reshape(1, chnl_dim*patch_dim* vert_dim)

            # save training losses
            df_version_mae = pd.DataFrame(train_maes)
            # write_to_file(f'check mae pd creating: {df_version_mae.head()}. \n being sent to {folder_to_save_model}', filepath=write_fpath)
            df_version_mae.to_csv(os.path.join(folder_to_save_model, 'mae_train_model.csv'))
            df_version_mse = pd.DataFrame(train_mse)
            df_version_mse.to_csv(os.path.join(folder_to_save_model, 'mse_train_model.csv'))

            np.save('{}/train_ground_truth.npy'.format(folder_to_save_model), train_ground_truth)
            np.save('{}/train_pred.npy'.format(folder_to_save_model), train_pred)
    
            
        write_to_file(f"Done with TRAINING loop.", filepath=write_fpath)
        
            
    # write_to_file(f'Shapes of training and test (truth,preds) output.\n{train_ground_truth.shape} \n{train_pred.shape} \n{test_ground_truth.shape} \n{test_pred.shape}', filepath=write_fpath)

