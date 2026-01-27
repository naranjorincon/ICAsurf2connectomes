import os
import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

import torch
import numpy as np
import pandas as pd
from models.models import *
from utils.utils import *
# from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

if __name__ == "__main__":
    #some settings
    write_fpath = "/scratch/naranjorincon/NeuroTranslate/netmat2surf/batch/CCA.print"
    model_out_root = "/scratch/naranjorincon/NeuroTranslate/netmat2surf/model_out"
    netmat_parcellation_res = 100
    batch_size = 64
    translation="ICAd15_schfd100"
    model_type = "krakenVAE_BGT"
    version = "fisher_z"
    train_epoch_range = 501
    val_epoch = 5
    best_mae = 100000000
    chosen_pca_size = 256
    cca_flag = 1

    ############################################# LOAD IN NETMATS AND SURFACE MESHES #############################################
    # loads in np train data/labels
    train_netmat_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_labels.npy")
    train_z_transform_ele = fisher_z_transform(train_netmat_np)
    train_cca_x = train_z_transform_ele
    
    train_netmat_np = make_nemat_allsubj(train_z_transform_ele, netmat_parcellation_res)
    train_surf_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_data.npy")
    # transform surf to be subs x channels*patch_num*verteces
    train_num_sub, num_chnl, num_patches, num_ver = train_surf_np.shape
    # from what I understand, the reshape is such that its all vertex for each patch of channle 1, then same for chnl2 and so on. SO it is "ordered"
    train_surf_chnlxpatchxver = train_surf_np.reshape(train_num_sub, num_chnl, num_patches* num_ver) # concats ver and chnls so [100 x 15*320*153]=[100x734,400]
    mean_train_label = np.mean(train_surf_chnlxpatchxver, axis=0) # mean cross subs
    write_to_file(f'Reformat surf label data shape: {train_surf_chnlxpatchxver.shape} \nAnd mean is shape: {mean_train_label.shape}', filepath=write_fpath)

    val_netmat_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/1L_validation_labels.npy")
    val_z_transform_ele = fisher_z_transform(val_netmat_np)
    val_cca_x = val_z_transform_ele
    val_netmat_np = make_nemat_allsubj(val_z_transform_ele, netmat_parcellation_res)
    val_surf_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/1L_validation_data.npy")
    # transform surf to be subs x channels*patch_num*verteces
    val_num_sub, _, _, _ = val_surf_np.shape
    # from what I understand, the reshape is such that its all vertex for each patch of channle 1, then same for chnl2 and so on. SO it is "ordered"
    val_surf_chnlxpatchxver = val_surf_np.reshape(val_num_sub, num_chnl, num_patches* num_ver) # concats ver and chnls so [100 x 15*320*153]=[100x734,400]
    write_to_file(f'Reformat surf label data shape: {val_surf_chnlxpatchxver.shape}', filepath=write_fpath)
    

    ############################################# compute PCA on data and output for later use ###################################
    if cca_flag == 1:
        train_cca_y = train_surf_np.reshape(train_num_sub, num_chnl*num_patches* num_ver)
        val_cca_y = val_surf_np.reshape(val_num_sub, num_chnl*num_patches* num_ver)

        # look at CCA of input and targets
        write_to_file(f'train cca x is shape: {train_cca_x.shape} \n and y is shape: {train_cca_y.shape}', filepath=write_fpath)
        train_cca = CCA(n_components=min(train_cca_x.shape[0], train_cca_y.shape[0])) # min of features cause second dim
        train_cca.fit(train_cca_x, train_cca_y)
        X_tc, Y_tc = train_cca.transform(train_cca_x, train_cca_y)
        correlations_tr = np.corrcoef(X_tc.T, Y_tc.T).diagonal(offset=X_tc.shape[0])
        write_to_file(f"TRAIN DATA CCA between mat and mesh: {correlations_tr}", filepath=write_fpath)

        val_cca = CCA(n_components=min(val_cca_x.shape[0], val_cca_y.shape[0]))
        val_cca.fit(val_cca_x, val_cca_y)
        X_vc, Y_vc = val_cca.transform(val_cca_x, val_cca_y)
        correlations_vl = np.corrcoef(X_vc.T, Y_vc.T).diagonal(offset=X_vc.shape[0])
        write_to_file(f"VAL DATA CCA between mat and mesh: {correlations_vl}", filepath=write_fpath)

        df_cca = pd.DataFrame(columns=['train_cca', 'val_cca'])
        new_row = pd.DataFrame({'train_cca': [correlations_tr], 'val_cca': [correlations_vl]})
        df_train = pd.concat([df_cca, new_row], ignore_index=True)
        df_train.to_csv(os.path.join(f"{model_out_root}/{translation}", 'mesh_netmat_CCA.csv'))