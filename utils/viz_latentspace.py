# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import math
import os
import sys
sys.path.append('../') #models
sys.path.append('./') # utils folders
sys.path.append('../../')
# from utils import make_netmat, make_nemat_allsubj #mat2vector, fisher_z_transform, make_nemat_allsubj
# from utils.utils import *
from utils import *
from models.models import *
import glob
import torch
from PIL import Image

# for later data viz
# local_root="/Users/snaranjo/Desktop/neurotranslate/mount_point"
model_out_root = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/model_out" #config['logging']['model_out_root']
translation= "ICAd15_schfd100" #config['data']['translation']
model_type = "kvSiTBGT" #config['data']['model_type']
version = "normICAfishzMAT" #config['data']['version']
model_details = "d6h6_VAEd10k_d4h2_adamW_cosinedecay_scheduler_recon_krakenonly" #config['transformer']['model_details']
netmat_prep_choise = "fisherz" #config['training']['netmat_prep_choise']
dataset_choice = "ABCD" #config['training']['dataset_choice']
bilateral_condition = False #config['training']['bilateral_condition'] # both hemispheres instead of 1
batch_size = 1 #config['testing']['bs_test']
from_parcellation = 100 #config['data']['from_parcellation']
chosen_test_model = "MSE" #config['testing']['chosen_test_model']
device = "cpu"
model_test_type="MSE"
saved_model_path = f'/home/naranjorincon/neurotranslate/surf2netmat/logs/{translation}/{dataset_choice}/{model_type}/{version}'
folder_to_save_model=f'{model_out_root}/{translation}/{dataset_choice}/{model_type}/{version}/{model_details}/{chosen_test_model}'
folder_to_save_losses = f'{model_out_root}/{translation}/{dataset_choice}/{model_type}/{version}/old/{model_details}/outofsample'
data_root_path = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"
write_fpath = f"{data_root_path}/NeuroTranslate/surf2netmat/batch/{model_type}_{dataset_choice}_VIZ.print"

root="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/surf2netmat"
directory = f"{root}/images/{dataset_choice}/{model_type}/{version}/{model_details}/{model_test_type}" # Replace with your target directory
print(directory)
if not os.path.exists(directory):
    # Create the directory
    os.makedirs(directory)
    print("Directory for model created.")
else:
    print("Directory for model output already exists.")


# img_extension = 'png'
# filename = f'/model_losses.{img_extension}'
# write_to_file(f'Saving to path:{directory}')
# # Save the figure
# plt.savefig(directory + filename, format=img_extension)
# plt.show()
# plt.close()

# %%
hemi_cond = "1L"
if dataset_choice == "HCPYA":
    tr_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_train_labels.npy") 
    tr_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_train_data.npy")#[:, np.newaxis, channel_testing, :] 
    write_to_file(f'Loaded in TRAIN. They have shapes: {tr_netmat_np.shape} & {tr_surf_np.shape} respectively.', filepath=write_fpath)
    
    te_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_test_labels.npy") 
    te_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/{hemi_cond}_test_data.npy")#[:, np.newaxis, channel_testing, :]
    write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)

elif dataset_choice == "ABCD":
    # n=400
    tr_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/train_netmat_clean.npy")#[:n]
    tr_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_train_surf.npy")#[:n]
    write_to_file(f'Loaded in TRAIN. They have shapes: {tr_netmat_np.shape} & {tr_surf_np.shape} respectively.', filepath=write_fpath)

    te_netmat_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d100/test_netmat_clean.npy")#[:n]
    te_surf_np = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_test_surf.npy")#[:n]
    write_to_file(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} & {te_surf_np.shape} respectively.', filepath=write_fpath)

# overfit_condition = True
# if overfit_condition:
#     n=400 #config['training']['overfit_condition_sub_range'] # upto how many subjects
#     train_netmat_np = tr_netmat_np[:n] #random subject(s) to pick to over fit
#     train_surf_np = tr_surf_np[:n]
#     val_netmat_np = te_netmat_np[:n]
#     val_surf_np = te_surf_np[:n]

# adds start token to *_label_np
padding=50#config['transformer']['padding'] #TODO if model does worse, might be because padding was once done BEFORE putting in fcn_prep_data_get_loaders so revert to test if so
upper_tri_sz = tr_netmat_np.shape[1]
# tr_netmat_np = add_start_token_np(tr_netmat_np, n=padding)
# te_netmat_np = add_start_token_np(te_netmat_np, n=padding)

tr_loader, te_loader, mean_train_label = fcn_prep_data_get_loaders(train_netmat=tr_netmat_np, train_surface=tr_surf_np, validation_netmat=te_netmat_np, validation_surface=te_surf_np, parcellation_N=from_parcellation, netmat_prep_choice=netmat_prep_choise, b_sz=batch_size, padding=padding, encdec=True, write_fpath=write_fpath)

############################################# Set up Test run and model configs #############################################


# %%
# write to file    
write_to_file("Loaded in data.", filepath=write_fpath)

# hold, input_dim = train_netmat_np.shape # schf100 parcellation
hold, dim_c, dim_p, dim_v =  tr_surf_np.shape

enc_sit_dim = 102 #config['transformer']['enc_sit_dim']
enc_depth = 6#config['transformer']['enc_depth']
enc_heads = 6#config['transformer']['enc_heads']
# dec_nhead = config['transformer']['dec_heads']
dec_depth = 4#config['transformer']['dec_depth']
dec_input_dim = int( upper_tri_sz + padding )
# emb_dropout = config['transformer']['enc_emb_drop']
dropout = 0.3 #config['transformer']['enc_drop']
VAE_latent_dim = 10000 #config['transformer']['vae_dim']
latent_length = 100 #config['transformer']['latent_length']

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

# see all models
# model_path = sorted(glob.glob(f"{saved_model_path}/*{model_details}_{chosen_test_model}.pt")) # look at training script for details, but all models saves as type_details_chosen: ex-kBGTLN_d6h5_demeanL2_skewloss_RHO.pt
model_path = sorted(glob.glob(f"{saved_model_path}/*{model_details}_{chosen_test_model}.pt")) # look at training script for details, but all models saves as type_details_chosen: ex-kBGTLN_d6h5_demeanL2_skewloss_RHO.pt
chosen_model = model_path[0]
write_to_file(f'\n\nmodel loaded is {chosen_model}', filepath=write_fpath)
model.load_state_dict(torch.load(chosen_model)) # most recent model

# Find number of parameters
model_params = sum(p.numel() for p in model.parameters())
write_to_file(f"\n\nModel params: {model_params}", filepath=write_fpath)

# # Testing below
# model.eval()
# model.to(device)

# # lists to keep track
# mse_train_list = []
# mae_train_list = []
# mse_test_list = []
# mae_test_list = []

# if bilateral_condition:
#     ss, nn = tr_netmat_np.shape
#     ss = 2 * ss
#     tr_ground_truth = np.zeros((ss,nn))
#     tr_pred = np.zeros((ss,nn))
#     te_ground_truth = np.zeros((ss,nn))
#     te_pred = np.zeros((ss,nn))
# else:
#     tr_ground_truth = np.zeros(tr_netmat_np.shape)
#     tr_pred = np.zeros(tr_netmat_np.shape) #SUBx4950 of zeros
#     te_ground_truth = np.zeros(te_netmat_np.shape)
#     te_pred = np.zeros(te_netmat_np.shape)

# with torch.no_grad():
#     for i, data in enumerate(te_loader):
#         mesh_indata, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0) #, data[2].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
#         meshnan = torch.isnan(mesh_indata).sum()
#         write_to_file(f"MESH-NaNs: {meshnan}", filepath=write_fpath)
#         dec_input = targets
#         allvals = model(src=mesh_indata, tgt=dec_input,  tgt_mask=generate_subsequent_mask(model.latent_length).to(device))
#         pred = allvals[0]
#         pred = pred[padding:]
#         targets = targets[:, padding:]

#         pnan = torch.isnan(pred).sum()
#         tnan = torch.isnan(targets).sum()
#         write_to_file(f"Pred-NaNs: {pnan} \n Target-NaNs: {tnan}", filepath=write_fpath)
#         assert pnan == 0
#         assert tnan == 0

#         # just having some output to see while testing, otherwise terminal is silent. Nice to see progress IMO
#         if i % 100 == 0:
#             write_to_file(f"checkpoint. Running test subject: {i}", filepath=write_fpath)

#         pred = pred.detach().numpy()
#         targets = targets.detach().numpy()
        
#         mae = np.mean(np.abs(pred - targets))
#         mae_test_list.append(mae)

#         mse = np.mean( (pred - targets)**2 )
#         mse_test_list.append(mse)

#         te_ground_truth[i, :] = targets
#         te_pred[i, :] = pred

#     write_to_file(f"Done with TESTING loop.", filepath=write_fpath)

#     # # to optimize testing and data saving, will only get best, mid, and lowest corr
#     # across_sub_rho = np.corrcoef(te_ground_truth, te_pred) # gives sub_dim*2 x sub_dim*2 and will likely be two square clusters truth and pred
#     # write_to_file(f"SZ of bigg matrix: {across_sub_rho.shape}", filepath=write_fpath)
#     # np.save(f"{folder_to_save_model}/te_big_corr_matrix.npy", across_sub_rho) # save for viz later
    
#     # # find best, and worst corr(truth,pred)
#     # row_half = np.split(across_sub_rho,2, axis = 0) #split in half across rows
#     # top_right_quad = np.split(row_half[0],2, axis = 1)[1] # again split by col, and top rigth quaf is corr(y,yhat) so choose 1 automatically == quad2
#     # find_max_rho = np.argwhere(top_right_quad == np.max(np.diag(top_right_quad)))[0] # find max across daigonal
#     # find_min_rho = np.argwhere(top_right_quad == np.min(np.diag(top_right_quad)))[0] #find min across diagonal
#     # max_idx = find_max_rho[0] # which subject had the highest corr across diagonal in quad2
#     # min_idx = find_min_rho[0] #0 is i, so subject index althougth same as j but keeping consistency
#     # write_to_file(f"IDX in big TEST corr matrix for both best (max) and worst (min) performance: {max_idx} {min_idx}", filepath=write_fpath)

#     # # save bet and worst netmat translation
#     # te_max_netmat_translation = te_pred[max_idx]
#     # te_min_netmat_translation = te_pred[min_idx]
#     # np.save(f"{folder_to_save_model}/te_max_netmat_translation.npy", te_max_netmat_translation)
#     # np.save(f"{folder_to_save_model}/te_min_netmat_translation.npy", te_min_netmat_translation)

#     for i, data in enumerate(tr_loader):
#         mesh_indata, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0) #, data[2].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
#         dec_input = targets
#         allvals = model(src=mesh_indata, tgt=dec_input,  tgt_mask=generate_subsequent_mask(model.latent_length).to(device))
#         pred = allvals[0]

#         pred = pred[padding:]
#         targets = targets[:, padding:]

#         # just having some output to see while testing, otherwise terminal is silent. Nice to see progress IMO
#         if i % 100 == 0:
#             write_to_file(f"checkpoint. Running test subject: {i}", filepath=write_fpath)

#         pred = pred.detach().numpy()
#         targets = targets.detach().numpy()
        
#         mae = np.mean(np.abs(pred - targets))
#         mae_train_list.append(mae)

#         mse = np.mean( (pred - targets)**2 )
#         mse_train_list.append(mse)

#         tr_ground_truth[i, :] = targets
#         tr_pred[i, :] = pred

#     write_to_file(f"Done with TRAINING loop.", filepath=write_fpath)

#     # # to optimize testing and data saving, will only get best, mid, and lowest corr
#     # across_sub_rho = np.corrcoef(tr_ground_truth, tr_pred) # gives sub_dim*2 x sub_dim*2 and will likely be two square clusters truth and pred
#     # np.save(f"{folder_to_save_model}/tr_big_corr_matrix.npy", across_sub_rho) # save for viz later
    
#     # # find best, mid, and worst corr(truth,pred)
#     # row_half = np.split(across_sub_rho,2, axis = 0) #split in half across rows
#     # top_right_quad = np.split(row_half[0],2, axis = 1)[1] # again split by col, and top rigth quaf is corr(y,yhat) so choose 1 automatically == quad2
#     # find_max_rho = np.argwhere(top_right_quad == np.max(np.diag(top_right_quad)))[0] # find max across daigonal
#     # find_min_rho = np.argwhere(top_right_quad == np.min(np.diag(top_right_quad)))[0] #find min across diagonal
#     # max_idx = find_max_rho[0] #0 is rows
#     # min_idx = find_min_rho[0] #0 is i, so subject index althougth same as j but keeping consistency
#     # write_to_file(f"IDX in big TRAIN corr matrix for both best (max) and worst (min) performance: {max_idx} {min_idx}", filepath=write_fpath)

#     # # save bet and worst netmat translation
#     # tr_max_netmat_translation = tr_pred[max_idx]
#     # tr_min_netmat_translation = tr_pred[min_idx]
#     # np.save(f"{folder_to_save_model}/tr_max_netmat_translation.npy", tr_max_netmat_translation)
#     # np.save(f"{folder_to_save_model}/tr_min_netmat_translation.npy", tr_min_netmat_translation)

# %% [markdown]
# # lookoing at latent space
# #### got below from: https://medium.com/@outerrencedl/a-simple-autoencoder-and-latent-space-visualization-with-pytorch-568e4cd2112a

# %%
def plotting(model2viz, loader2viz, step:int=0, show=True):
    model = model2viz
    te_loader = loader2viz
    model.eval() # Switch the model to evaluation mode
    
    points = []
    label_idcs = []
    
    path = "./latent_viz_check"
    if not os.path.exists(path): os.mkdir(path)
    
    for i, data in enumerate(te_loader):
        # img, label = [d.to(device) for d in data]
        img, label = data[0].to(device), data[1].to(device)
        # We only need to encode the validation images
        # proj = model.encoder(img) # original code for model but our model is different
        pred, mu, logvar = model.encode(img)
        # proj = pred
        write_to_file(f"plotting - {pred.shape} - {label.shape}", filepath=write_fpath)

        points.extend(pred.detach().cpu().numpy())
        label_idcs.extend(label.detach().cpu().numpy())
        del img, label
    
    points = np.array(points)
    
    # Creating a scatter plot
    fig, ax = plt.subplots(figsize=(10, 10) if not show else (8, 8))
    # scatter = ax.scatter(x=points[:, 0], y=points[:, 1], s=2.0, 
    #             c=label_idcs, cmap='tab10', alpha=0.9, zorder=2)
    scatter = ax.scatter(x=points[:, 0], y=points[:, 1], s=2.0,
                         cmap='tab10', alpha=0.9, zorder=2)
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    if show: 
        ax.grid(True, color="lightgray", alpha=1.0, zorder=0)
        plt.show()
    else: 
        # Do not show but only save the plot in training
        plt.savefig(f"{path}/Step_{step:03d}.png", bbox_inches="tight")
        plt.close() # don't forget to close the plot, or it is always in memory
        # model.train()

# convert image sequence to a gif file
def save_gif(path):
  
  frames = []
  imgs = sorted(os.listdir(path))

  for im in imgs:
      new_frame = Image.open(f"{path}/" + im)
      frames.append(new_frame)
  
  frames[0].save("latentspace.gif", format="GIF",
                 append_images=frames[1:],
                 save_all=True,
                 duration=200, loop=0)
  
plotting(model, te_loader, show=False) # if show is false, it saves the image


