import os
import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')
sys.path.append('../../../')

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import yaml
import torch as nn
from models.models import *
from utils.utils import *
from models.ms_sit_unet_shifted import *
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from torch.cuda.amp import autocast#, GradScaler

# from generative.networks.nets import (
#     DiffusionModelUNet
# )

def train_UnetLDM(model, train_loader, device, optimizer, inferer, scheduler, latent_batch, write_fpath):
    torch.cuda.empty_cache()
    model.train()
    running_loss = 0

    for i, data in enumerate(train_loader):
        with autocast(enabled=True):

            # latents = next(latent_iter)#.to(device) #latent_batch.to(device) #* scale_factor, 1/std(z) but not sure yet
            latents = torch.from_numpy( latent_batch[:, i, :].squeeze())
            write_to_file(f"{type(latents)}, \n{latents.shape}", filepath=write_fpath, also_print=True)

            n = latents.shape[0] # supposed to be subject in batch yeah?
            with torch.set_grad_enabled(True):
                noise = torch.randn_like(latents).to(device) # make noise of same shape/dims
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=device).long()

                noise_pred = inferer(
                                inputs=latents, 
                                diffusion_model=model, 
                                noise=noise, 
                                timesteps=timesteps,
                                # condition=context, # trying no context
                                mode='crossattn'
                )

                loss = F.mse_loss( noise.float(), noise_pred.float() )
            
            loss.backward()
            running_loss += loss.item()
            del loss, noise, noise_pred, timesteps, latents

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return running_loss

def validation(model, val_loader, device, inferer, scheduler, latent_batch, write_fpath):
    torch.cuda.empty_cache()
    model.eval()
    running_loss = 0 
    with torch.no_grad():
        for i, data in enumerate(val_loader):   
            with autocast(enabled=True): #automizes precision reductions of floats i assume 
                latents = latent_batch.to(device) #* scale_factor, 1/std(z) but not sure yet
                n = latents.shape[0]        

                with torch.set_grad_enabled(False):
                    noise = torch.randn_like(latents).to(device)
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=device).long()

                    noise_pred = inferer(
                        inputs=latents, 
                        diffusion_model=model, 
                        noise=noise, 
                        timesteps=timesteps,
                        # condition=context,
                        mode='crossattn'
                    )         
                    loss = F.mse_loss( noise.float(), noise_pred.float() )  
                
            running_loss += loss.item() 
            
    return running_loss


def whole_model_arch(config):
    torch.cuda.empty_cache()

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
    get_latent_flag = config['data']['get_latent_flag']
    print_condition = True 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #"cpu"
    write_to_file(f'Using: {device} and they are: {torch.cuda.device_count()}', filepath=write_fpath, also_print=print_condition)
    best_loss = -1e+9

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
    lim = 5
    train_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_labels.npy")[0:lim] # label = netmat, so TODO is fix these later
    train_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_train_data_ico6.npy")[0:lim] #data = sruf
    write_to_file(f'Loaded in data and labels. They have shapes: {train_netmat_np.shape} & {train_surf_np.shape} respectively.', filepath=write_fpath, also_print=print_condition)

    train_num_sub, chnls, verteces = train_surf_np.shape
    train_surf_chnlxver = train_surf_np.reshape(train_num_sub, chnls*verteces) # concats ver and chnls so [100 x 15*320*153]=[100x734,400]
    
    norm_netmats = (train_netmat_np - np.mean(train_netmat_np, axis=0))/ np.std(train_netmat_np, axis=0)
    mean_train_label = np.mean(train_surf_chnlxver, axis=0)
    write_to_file(f'across subj mean shape: {mean_train_label.shape}', filepath=write_fpath, also_print=print_condition)
    train_netmat_np = make_nemat_allsubj(norm_netmats, from_parcellation) # turns vec into netmat for all subs, second variable is nodes in netmat
    write_to_file(f'Made netmat for each subject. Took label data and reformat to sym netmat. Has now shape: {train_netmat_np.shape}\nAnd surf is: {train_surf_np.shape}', filepath=write_fpath, also_print=print_condition)

    #### LOAD VALIDATION DATA AND SURF
    val_netmat_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_validation_labels.npy")[0:lim] # label = netmat, so TODO is fix these later
    val_surf_np = np.load(f"{data_root_path}/surface-vision-transformers/data/ICAd15_schfd100/template/1L_validation_data_ico6.npy")[0:lim] #data = sruf
    write_to_file(f'Loaded in data and labels. They have shapes: {val_netmat_np.shape} & {val_surf_np.shape} respectively.', filepath=write_fpath, also_print=print_condition)

    val_num_sub, _, _ = val_surf_np.shape
    val_surf_chnlxver = val_surf_np.reshape(val_num_sub, chnls*verteces) # concats ver and chnls so [100 x 15*320*153]=[100x734,400]

    norm_netmats = (val_netmat_np - np.mean(val_netmat_np, axis=0))/ np.std(val_netmat_np, axis=0)
    val_netmat_np = make_nemat_allsubj(norm_netmats, from_parcellation) # turns vec into netmat for all subs, second variable is nodes in netmat
    write_to_file(f'Made netmat for each subject. Took label data and reformat to sym netmat. Has now shape: {val_netmat_np.shape}\nAnd surf is: {val_surf_np.shape}', filepath=write_fpath, also_print=print_condition)

    # combine them here, to get and save vae latents (z_mean)
    netmats_fused = np.concatenate((train_netmat_np,val_netmat_np), axis=0)
    surf_fused = np.concatenate((train_surf_chnlxver,val_surf_chnlxver), axis=0)
    surf_fused_demean = surf_fused - mean_train_label
    train_demean_surf = train_surf_chnlxver - mean_train_label
    validation_demean_surf = val_surf_chnlxver - mean_train_label

    # below is for VAE model
    place_hold, input_dim, conn_profile_num = train_netmat_np.shape # schf100 parcellation

    # data loader for fused data
    full_dataset = torch.utils.data.TensorDataset(torch.from_numpy(netmats_fused).float(), torch.from_numpy((surf_fused_demean)).float())
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size = batch_size, shuffle=False, num_workers=10)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_netmat_np).float(), torch.from_numpy((train_demean_surf)).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=10)
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_netmat_np).float(), torch.from_numpy((validation_demean_surf)).float())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers=10)    
    write_to_file("Loaded in DATA.", filepath=write_fpath, also_print=print_condition)

    # remove large data
    del train_surf_chnlxver, val_surf_chnlxver, mean_train_label, surf_fused

    variational_autoencoder = VAE_LNET_BGT_swMSSiT(
                enc_input = input_dim,
                enc_model_dim = conn_profile_num,
                enc_depth = config['transformer']['enc_depth'], #layers
                enc_heads = config['transformer']['enc_heads'], # attn heads
                enc_emb_drop = config['transformer']['enc_emb_drop'],# drop out of embedding step
                enc_drop = config['transformer']['enc_drop'],  # dropout at transformer layers
                VAE_latent_dim = config['transformer']['vae_dim'],
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

    variational_autoencoder.to(device)

    # get latent mean from trained vae model
    save_path_latent = f'{model_out_root}/{translation}/{model_type}/{version}/{model_details}'
    if get_latent_flag:
        with torch.no_grad():
            for i, data in enumerate(full_loader): # should be a z for each subject
                inputs = data[0].to(device) #[b 100 100]
                # write_to_file(f"full input shape {inputs.shape}", filepath=write_fpath, also_print=print_condition)
                latent_mu, latent_var = variational_autoencoder.encode(inputs)
                latent_mu = latent_mu.cpu().squeeze().numpy()
                # write_to_file(f"full latent shape {latent_mu.shape}", filepath=write_fpath, also_print=print_condition)
                np.savez_compressed(f"{save_path_latent}/{i:04d}_latent.npz", data=latent_mu)
                del latent_mu, latent_var

    
    # npz_reader = NumpyReader(npz_keys=['data'])
    npz_all = []
    for i, data in enumerate(full_loader):
        curr_load = np.load(f"{save_path_latent}/{i:04d}_latent.npz")
        npz_all.append(curr_load['data']) # each subject latent extracted from saved path, shape is [858,10000] because 10k is latent dim sz
    latent_batch_np = np.asarray(npz_all, dtype=np.float32)
    write_to_file(latent_batch_np.shape, filepath=write_fpath, also_print=print_condition)
    allsub, latent_dim = latent_batch_np.shape
    latent_batch_np = latent_batch_np.reshape(batch_size, (allsub // batch_size), latent_dim)
    write_to_file(f"Loaded in latent npz: {latent_batch_np.shape}", filepath=write_fpath, also_print=print_condition)
    torch.cuda.empty_cache()
    model = diff_LNET(
                enc_input = input_dim // input_dim,
                enc_model_dim = conn_profile_num*input_dim,
                enc_depth = config['transformer']['enc_depth'], #layers
                enc_heads = config['transformer']['enc_heads'], # attn heads
                enc_emb_drop = config['transformer']['enc_emb_drop'],# drop out of embedding step
                enc_drop = config['transformer']['enc_drop'],  # dropout at transformer layers
                VAE_latent_dim =  config['transformer']['vae_dim'],
                ####################

                ####################
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
    
    # reset params, xavier uniform
    model.to(device)
    model._reset_parameters()
    
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        schedule='scaled_linear_beta', 
        beta_start=0.0015, 
        beta_end=0.0205
    )

    def sample_using_diffusin(
        autoencoder: nn.Module, 
        diffusion: nn.Module, 
        context: torch.Tensor,
        device: str, 
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015, 
        beta_end: float = 0.0205, 
        verbose: bool = True
        ) -> torch.Tensor:

        scheduler = DDPMScheduler(
                    num_train_timesteps=1000, 
                    schedule='scaled_linear_beta', 
                    beta_start=0.0015, 
                    beta_end=0.0205,
                    clip_sample=False)
        
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        
        # context = context.unsqueeze(0).to(device).to(device)
        # drawing a random z_T ~ N(0,I)
        LATENT_SHAPE_DM = (3, 16, 20, 16) # this is the latent shape? actual shape is (3, 15, 18, 15) but added pads to be divisible by 2**2 as 2 is downsample layers in OG
        # Adjusting the latent space (with constant padding) to be divisible by 4 (2^2 where 2 are the downsampling layers of U-Net)
        z = torch.randn(LATENT_SHAPE_DM).unsqueeze(0).to(device)
        
        progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
        for t in progress_bar:
            with torch.no_grad():
                with autocast(enabled=True):

                    timestep = torch.tensor([t]).to(device)
                    
                    # predict the noise
                    noise_pred = diffusion(
                        x=z.float(), 
                        timesteps=timestep, 
                        context=context.float(), 
                    )

                    # the scheduler applies the formula to get the 
                    # denoised step z_{t-1} from z_t and the predicted noise
                    z, _ = scheduler.step(noise_pred, t, z)
        
        # decode the latent
        z = z / scale_factor
        # z = utils.to_vae_latent_trick( z.squeeze(0).cpu() )
        x = autoencoder.decode_stage_2_outputs( z.unsqueeze(0).to(device) )
        # x = utils.to_mni_space_1p5mm_trick( x.squeeze(0).cpu() ).squeeze(0)
        return x

    inferer = DiffusionInferer(scheduler=scheduler)
    
    if torch.cuda.device_count() > 1:
        write_to_file("Let's use", torch.cuda.device_count(), "GPUs!", filepath=write_fpath, also_print=print_condition)
        model = nn.DataParallel(model)

    model_params = sum(p.numel() for p in model.parameters())
    write_to_file(f"Model PARAMS: {model_params}", filepath=write_fpath, also_print=print_condition)

    write_to_file("Loaded in MODEL.", filepath=write_fpath, also_print=print_condition)

    # initialize optimizer / loss
    if config['optimisation']['optimiser']=='Adam':
        write_to_file('using Adam optimiser',  filepath=write_fpath, also_print=print_condition)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=config['Adam']['weight_decay'])
    elif config['optimisation']['optimiser']=='SGD':
        write_to_file('using SGD optimiser',  filepath=write_fpath, also_print=print_condition)
        optimizer = optim.SGD(model.parameters(), lr=LR, 
                                                weight_decay=config['SGD']['weight_decay'],
                                                momentum=config['SGD']['momentum'],
                                                nesterov=config['SGD']['nesterov'])
    elif config['optimisation']['optimiser']=='AdamW':
        write_to_file('using AdamW optimiser',  filepath=write_fpath, also_print=print_condition)
        optimizer = optim.AdamW(model.parameters(),
                                lr=LR,
                                weight_decay=config['AdamW']['weight_decay'])

    write_to_file('', filepath=write_fpath, also_print=print_condition)
    write_to_file('#'*30, filepath=write_fpath, also_print=print_condition)
    write_to_file('######## BEGINING TRAINING ########', filepath=write_fpath, also_print=print_condition)
    write_to_file('#'*30, filepath=write_fpath, also_print=print_condition)
    write_to_file('', filepath=write_fpath, also_print=print_condition)
    if  config['MODEL'] == 'ms-sit':
        write_to_file('Mesh resolution - ico {}'.format(config['mesh_resolution']['ico_mesh']), filepath=write_fpath, also_print=print_condition)
        write_to_file('Grid resolution - ico {}'.format(config['mesh_resolution']['ico_grid']), filepath=write_fpath, also_print=print_condition)
        # write_to_file('Number of patches - {}'.format(patches), filepath=write_fpath)
        write_to_file('Number of vertices - {}'.format(verteces), filepath=write_fpath, also_print=print_condition)
        write_to_file('Reorder patches: {}'.format(config['mesh_resolution']['reorder']), filepath=write_fpath, also_print=print_condition)
        write_to_file('', filepath=write_fpath, also_print=print_condition)

    running_train_loss = 0
    running_validation_loss = 0
    df_train = pd.DataFrame(columns=['train_loss'])
    df_val = pd.DataFrame(columns=['val_loss'])

    write_to_file("Begining training.", filepath=write_fpath, also_print=print_condition)
    torch.cuda.empty_cache()
    
    for epoch in range(1, train_epoch_range):
        
        running_loss = train_UnetLDM(model, train_loader, device, optimizer, inferer, scheduler, latent_batch_np, write_fpath)
        running_train_loss += (running_loss.detach().cpu().numpy())
        del running_loss
        write_to_file('| Training | Epoch - {} | Loss - {:.4f}'.format(epoch, running_train_loss), filepath=write_fpath, also_print=print_condition)

        new_row = pd.DataFrame({'train_loss': [running_train_loss]})
        df_train = pd.concat([df_train, new_row], ignore_index=True)
        df_train.to_csv(os.path.join(folder_to_save_losses, 'train_losses_patch.csv'))

        if epoch%val_epoch == 0:
            running_val_loss = validation(model, val_loader, device, inferer, scheduler, latent_batch_np, write_fpath)
            write_to_file('| Validation | Epoch - {} | Loss - {:.4f}'.format(epoch, running_val_loss), filepath=write_fpath, also_print=print_condition)

            # save model with best loss
            curr_val_loss = running_val_loss
            if curr_val_loss < best_loss:
                best_loss = curr_val_loss
                write_to_file('saving LOSS model checkpoint...', filepath=write_fpath, also_print=print_condition)
                torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}_LOSS.pt'))
            
            new_row = pd.DataFrame({'val_loss': [running_val_loss]})
            df_val = pd.concat([df_val, new_row], ignore_index=True)
            df_val.to_csv(os.path.join(folder_to_save_losses, 'val_losses_patch.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='kUnet_LDM')

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

