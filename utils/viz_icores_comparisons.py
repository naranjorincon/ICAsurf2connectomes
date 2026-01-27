# %%

'''
import time 
start_time = time.time()
from json import load
filename = 'viz_icores_comparisons.ipynb'

with open(filename) as fp:
    nb = load(fp)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(line for line in cell['source'] if not line.startswith('%'))
        exec(source, globals(), locals())


'''

# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import seaborn as sns
import pandas as pd
# import math
import os
import sys
sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')
from utils import * #make_netmat
import glob
# for later data viz
from_parcellation = 100
from_atlas="schf" #schf, glasser
translation= "ICAd15_ICAd15" #"ICAd15_ICAd15" #f"ICAd15_{from_atlas}d{from_parcellation}" # needs to be "" type of string
version = "normICAdemeanMAT" #normICAdemeanfishzMAT normICArawMAT. normICAdemeanMAT
print(translation)
if "/Users/snaranjo/" in os.getcwd():
    local_flag=True
    using_local_root="/Users/snaranjo/Desktop/neurotranslate/mount_point"
else:
    local_flag=False
    using_local_root=""
    
scratch_path=f"{using_local_root}/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"
# root=f"{scratch_path}/NeuroTranslate/surf2netmat"
# model_output_names = ["kSiTLN", "kvSiTLN", "kSiTBGT", "kvSiTBGT"]
# datasets = ["HCPYA", "ABCD"]
# chosen_one = 0  #corresponds to ["kSiTLN", "kvSiTLN", "kSiTBGT", "kvSiTBGT"]
# dataset_choice = 1 # 0==HCPYA, 1==ABCD, 2==HCPYA_ABCDdr
# ica_reconstuction=0
# ICAd15_ICAd15=1
# icores=2
# if translation == "ICAd15_ICAd15":
#     version = "normICAnormICA" #over ride what it was prior
#     ica_reconstuction=1
#     chnl_icarecon = 0 #0-14
#     type_of_model_train_fcn = f"120425_d6h3_tiny_adamW_cosinedecay_reconICA_MSEtrain_expICARECON_chnl{chnl_icarecon}"
#     # type_of_model_train_fcn = f"011426_d6h3_tiny_adamW_cosinedecay_recon_MSEtrain_1L_full_demean_exp100_wGelu_ico0{icores}"
# else:
#     type_of_model_train_fcn = f"120325_d6h3_tiny_adamW_cosinedecay_recon_MSEtrain_1L_Partial_demean_exp{from_parcellation}_wGelu" #"d6h3_adamW_cosinedecay_recon_krakenonly_1R_082925" #"d6h3_adamW_cosinedecay_recon_krakenonly_1R_090125"
#     # type_of_model_train_fcn = f"120325_d6h3_tiny_adamW_cosinedecay_recon_MSEtrain_1L_Partial_demean_exp100_wGelu"
    
# model_test_type_list = ["MSE"] #["MSE", "MAE", "RHO", "LAST"]
# img_extension = 'png' # png or eps #MICCAI file extension for images preferred
# val_step=1
# print(type_of_model_train_fcn)

# # %%
# for mm in model_test_type_list:
#     model_test_type = mm #model_test_type_list[mm]
#     model_type = model_output_names[chosen_one] # chosen index
#     print(model_type)
#     # print(f"{root}/model_out/{translation}/{datasets[dataset_choice]}/{model_type}/{version}/{type_of_model_train_fcn}")
#     list_of_details=np.sort(glob.glob(f"{root}/model_out/{translation}/{datasets[dataset_choice]}/{model_type}/{version}/{type_of_model_train_fcn}")) # should be all versions and their detial names
#     print(list_of_details)
    
# model_details = list_of_details[0].split('/')[-1]
# print(f"chosen: {model_details}")

# %%

# directory = root + '/images/' + datasets[dataset_choice] + '/' + translation  + '/' + model_type +'/' + version + '/'+ model_details + '/' + model_test_type #+ '/ABCD_train_HCPYA_test'# Replace with your target directory
# if not os.path.exists(directory):
#     # Create the directory
#     os.makedirs(directory)
#     print("Directory created.")
# else:
#     print("Directory already exists.")


# # %%
# out_of_sample_test = False #config['testing']['out_of_sample_test']
# train_empty_flag=True
# if out_of_sample_test:
#     curr_dataset = datasets[dataset_choice]
#     datasets[dataset_choice] = curr_dataset + "_HCPYAtest"
# print(f"{datasets[dataset_choice]}")

# #smaller_scale = 500 # subjects to laod here
# root_data=f"{root}/model_out/{translation}/{datasets[dataset_choice]}/{model_type}/{version}/{model_details}/{model_test_type}"
# print(root_data)

# test_truth_holder = np.load(f"{root_data}/test_ground_truth.npy")#[:smaller_scale]
# test_pred_holder = np.load(f"{root_data}/test_pred.npy")#[:smaller_scale]
# print(f"Test shapes: {test_truth_holder.shape} (Target)  {test_pred_holder.shape} (Pred)")

# train_truth_holder = np.zeros(test_truth_holder.shape) #np.load(f"{root_data}/train_ground_truth.npy")#[:smaller_scale]
# train_pred_holder = np.zeros(test_pred_holder.shape) #np.load(f"{root_data}/train_pred.npy")#[:smaller_scale]
# print(f"Train shapes: {train_truth_holder.shape} (Target)  {train_pred_holder.shape} (Pred)")

# if "demean" in version: # if version is already demeaned, need to get actual MEAN from original data
#     print("DATA already demeaned and predicted as such. For original versions, need to get unprep raw netmat data to get means.")
#     hemi_cond="1L"
#     #train is huge so if dont wanna load the whole thing and make zeros to make thing faster
#     # if train_empty_flag:
#     #     train_netmat_np = np.zeros((train_truth_holder.shape))
#     if translation == "ICAd15_glasserd360":
#         train_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/ABCD/glasser_mats/netmat_d{from_parcellation}/{hemi_cond}_train_netmat_clean.npy")
#     else:
#         train_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/ABCD/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_train_netmat_clean.npy")
#     # train_surf_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{datasets}/ICA_maps/ICAd15_ico02/{hemi_cond}_train_surf.npy")
#     print(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} respectively.')

#     # get the same for test
#     if out_of_sample_test:
#         outofsample_dataset_choice="HCPYA_ABCDdr" # choose which out of sample to use
        
#         if translation == "ICAd15_glasserd360":
#             # te_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/ABCD/glasser_mats/netmat_d{from_parcellation}/{hemi_cond}_test_netmat_clean.npy")
#             print("Glasser HCP not done yet. Can't run.")
            
#         else:
#             te_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{outofsample_dataset_choice}/schaefer_mats/netmat_d{from_parcellation}/train_netmat_clean.npy")
#         # te_surf_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{outofsample_dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_train_surf.npy")
#         print(f'OUT OF SAMPLE Loaded in TEST. They have shapes: {te_netmat_np.shape}.')
#     else:
#         if translation == "ICAd15_glasserd360":
#             te_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/ABCD/glasser_mats/netmat_d{from_parcellation}/{hemi_cond}_test_netmat_clean.npy")
#         else:                       
#             te_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/ABCD/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_test_netmat_clean.npy")
#         # te_surf_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{datasets}/ICA_maps/ICAd15_ico02/{hemi_cond}_test_surf.npy")

#     print(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} respectively.')

#     train_mean_flatten_true = np.mean(train_netmat_np, axis=0, keepdims=True)
#     train_mean_flatten_pred = np.zeros((train_mean_flatten_true.shape)) #none so empty

#     test_mean_flatten_true = np.mean(te_netmat_np, axis=0, keepdims=True)
#     test_mean_flatten_pred = np.zeros((test_mean_flatten_true.shape))

# else: #other wise, predicting standard netmats and can get means from those for later demeaning
#     print("DATA predicted as is, so need to get averages to subtract for DEMEAN figures.")
#     train_mean_flatten_pred = np.mean(train_pred_holder, axis=0, keepdims=True)
#     train_mean_flatten_true = np.mean(train_truth_holder, axis=0, keepdims=True)
#     test_mean_flatten_pred = np.mean(test_pred_holder, axis=0, keepdims=True)
#     test_mean_flatten_true = np.mean(test_truth_holder, axis=0, keepdims=True)

# print(train_mean_flatten_true.shape)

# %%
# look across icos and check similarity
# true ico stuff
data_root_path=f"{scratch_path}"
dataset_choice="ABCD"
from_parcellation="100"
hemi_cond="1L"
ico_res_true_list = []
netmat_true_list = []
icores_list = [1, 2, 3, 4]
for ico_idx in range(len(icores_list)):
    # ico_idx=0
    icores=icores_list[ico_idx]
    # val_surf_np = np.zeros((624, 15, 80, 561)) #np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{icores}/{hemi_cond}_val_surf.npy")
    # print(f"VALIDATION ico-0{icores} information\n surf:{val_surf_np.shape}")
    te_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{icores}/{hemi_cond}_test_surf.npy")
    print(f"TEST ico-0{icores} information\n surf:{te_surf_np.shape}")
    train_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{icores}/{hemi_cond}_train_surf.npy")
    print(f"TRAIN ico-0{icores} information\n surf:{train_surf_np.shape}")
    mean_train_surface = np.nanmean(train_surf_np, axis=0, keepdims=True) #1x15x320x153 TRAIN mean for that ICO resolution
    sigma_train_surface = np.nanstd(train_surf_np, axis=0, keepdims=True) #1x15x320x153
    norm_test_surf =  (te_surf_np-mean_train_surface) / (sigma_train_surface+1e-99)
    norm_train_surf = (train_surf_np-mean_train_surface) / (sigma_train_surface+1e-99)
    ico_res_true_list.append([te_surf_np,train_surf_np])
    # netmat_true_list.append()
    print(len(ico_res_true_list), ico_res_true_list[ico_idx][0].shape)
    
# netmats are the same across ico so keep here
train_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_train_netmat_clean.npy")
val_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_val_netmat_clean.npy")
te_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_test_netmat_clean.npy")
netmat_true_list.append([train_netmat_np, val_netmat_np, te_netmat_np])
print(len(netmat_true_list))

# %%
# ico_idx=0 #corresponds to ico-version so keep in mind 0 index: 0->1, 2->3, and so on.
all_ico_test_flattened = []
N, C, P, V = ico_res_true_list[0][0].shape #same N and C regardless of ICO so using ico1 w/e
test_ico1_surf = np.reshape(ico_res_true_list[0][0], (N, C, P*V))
all_ico_test_flattened.append(test_ico1_surf)
_, _, P, V = ico_res_true_list[1][0].shape
test_ico2_surf = np.reshape(ico_res_true_list[1][0], (N, C, P*V))
print(test_ico1_surf.shape, test_ico2_surf.shape) # checking and yes so far good
all_ico_test_flattened.append(test_ico2_surf)
_, _, P, V = ico_res_true_list[2][0].shape
test_ico3_surf = np.reshape(ico_res_true_list[2][0], (N, C, P*V))
print(test_ico3_surf.shape) # checking and yes so far good
all_ico_test_flattened.append(test_ico3_surf)

channel=0
#corr
corr1=np.corrcoef(test_ico1_surf[:,channel].squeeze())
corr2=np.corrcoef(test_ico2_surf[:,channel].squeeze())
corr3=np.corrcoef(test_ico3_surf[:,channel].squeeze())
print(corr1.shape,corr2.shape, corr3.shape)

corr1_fz_vec = fisher_z_transform(mat2vector(corr1))
corr2_fz_vec = fisher_z_transform(mat2vector(corr2))
corr3_fz_vec = fisher_z_transform(mat2vector(corr3))

ico1_ico2_corr = np.corrcoef(corr1_fz_vec,corr2_fz_vec)[0,1]
ico3_ico2_corr = np.corrcoef(corr3_fz_vec,corr2_fz_vec)[0,1]
ico1_ico3_corr = np.corrcoef(corr1_fz_vec,corr3_fz_vec)[0,1]
print(f"ico1ico2:{ico1_ico2_corr} \nico3ico2:{ico3_ico2_corr} \nico1ico3:{ico1_ico3_corr}")
#diff
# diff=(test_ico1_surf[:,channel].squeeze() - test_ico2_surf[:,channel].squeeze())


shrink_val=0.5
fig, axes = plt.subplots(3, 3, figsize=(80, 80))
axes = axes.flatten()

ax = sns.heatmap(corr1,cmap='afmhot_r',
    cbar_kws={"shrink": shrink_val}, linewidths=.5, square=True, xticklabels=False, yticklabels=False, ax=axes[0]) #vmin=-0.6, vmax=0.6
axes[0].set_title(f"Subject RHO Ico-01", fontsize=80)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=80)

ax = sns.heatmap(corr2,cmap='afmhot_r',
    cbar_kws={"shrink": shrink_val}, linewidths=.5, square=True, xticklabels=False, yticklabels=False, ax=axes[1]) #vmin=-0.6, vmax=0.6
axes[1].set_title(f"Subject RHO Ico-02", fontsize=80)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=80)

ax = sns.heatmap(corr2 - corr1,cmap='afmhot_r',
    cbar_kws={"shrink": shrink_val}, linewidths=.5, square=True, xticklabels=False, yticklabels=False, ax=axes[2]) #vmin=-0.6, vmax=0.6
axes[2].set_title(f"Ico-02 - Ico-01, rho:{ico1_ico2_corr:.2f}", fontsize=80)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=80)

ax = sns.heatmap(corr3,cmap='afmhot_r',
    cbar_kws={"shrink": shrink_val}, linewidths=.5, square=True, xticklabels=False, yticklabels=False, ax=axes[3]) #vmin=-0.6, vmax=0.6
axes[3].set_title(f"Subject RHO Ico-03", fontsize=80)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=80)

ax = sns.heatmap(corr2,cmap='afmhot_r',
    cbar_kws={"shrink": shrink_val}, linewidths=.5, square=True, xticklabels=False, yticklabels=False, ax=axes[4]) #vmin=-0.6, vmax=0.6
axes[4].set_title(f"Subject RHO Ico-02", fontsize=80)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=80)

ax = sns.heatmap(corr2 - corr3,cmap='afmhot_r',
    cbar_kws={"shrink": shrink_val}, linewidths=.5, square=True, xticklabels=False, yticklabels=False, ax=axes[5]) #vmin=-0.6, vmax=0.6
axes[5].set_title(f"Ico-02 - Ico-03, rho:{ico3_ico2_corr:.2f}", fontsize=80)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=80)


ax = sns.heatmap(corr3,cmap='afmhot_r',
    cbar_kws={"shrink": shrink_val}, linewidths=.5, square=True, xticklabels=False, yticklabels=False, ax=axes[6]) #vmin=-0.6, vmax=0.6
axes[3].set_title(f"Subject RHO Ico-03", fontsize=80)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=80)

ax = sns.heatmap(corr1,cmap='afmhot_r',
    cbar_kws={"shrink": shrink_val}, linewidths=.5, square=True, xticklabels=False, yticklabels=False, ax=axes[7]) #vmin=-0.6, vmax=0.6
axes[4].set_title(f"Subject RHO Ico-02", fontsize=80)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=80)

ax = sns.heatmap(corr3 - corr1,cmap='afmhot_r',
    cbar_kws={"shrink": shrink_val}, linewidths=.5, square=True, xticklabels=False, yticklabels=False, ax=axes[8]) #vmin=-0.6, vmax=0.6
axes[5].set_title(f"Ico-03 - Ico-01, rho:{ico1_ico3_corr:.2f}", fontsize=80)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=80)

# for ii in range(len(all_ico_test_flattened)):
#     ax = sns.heatmap(all_ico_test_flattened[ii][:,0,:],cmap='Spectral_r',
#                     cbar_kws={"shrink": shrink_val}, center=0, linewidths=.5, xticklabels=False, yticklabels=False, ax=axes[0]) #vmin=-0.6, vmax=0.6


# axes[0].set_title(f"Measured \nSUB:{subs2view[0]}", fontsize=40)
# axes[0].set_ylabel(f"Translation", fontsize=40)

plt.tight_layout()
plt.savefig('./ico_comparisons_true_input', format='png')
plt.show()
plt.close()



