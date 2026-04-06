# %%
'''
To run this shole jupyter notebook as a python script follow this:

(1) activate conda environemt
(2) go to where this notebook is located in your computer
(3) use `python` to enter python with in your shell/terminal
(4) follow the above syntax


from json import load
filename = 'extract_attention_maps.ipynb'
with open(filename) as fp:
    nb = load(fp)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(line for line in cell['source'] if not line.startswith('%'))
        exec(source, globals(), locals())

'''

# %%


# %%
import os 
import torch
import argparse
import yaml
import datetime
import subprocess
import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')
sys.path.append('../../../')

import numpy as np
import nibabel as nb
import pandas as pd

from einops import rearrange
import torch.nn.functional as F

import fnmatch
from typing import Union, Optional, List

from models import models
from tools.datasets import dataset_cortical_surfaces

def save_gifti(data, filename):
    gifti_file = nb.gifti.gifti.GiftiImage()
    gifti_file.add_gifti_data_array(nb.gifti.gifti.GiftiDataArray(data))
    nb.save(gifti_file,filename)

class AttentionExtract(torch.nn.Module):
    # defaults should cover a significant number of timm models with attention maps.
    default_module_names = ['*attend']
    def __init__(
            self,
            model: Union[torch.nn.Module],
            names: Optional[List[str]] = None,
            mode: str = 'eval',
            method: str = 'fx',
            hook_type: str = 'forward',
    ):
        """ Extract attention maps (or other activations) from a model by name.
        Args:
            model: Instantiated model to extract from.
            names: List of concrete or wildcard names to extract. Names are nodes for fx and modules for hooks.
            mode: 'train' or 'eval' model mode.
            method: 'fx' or 'hook' extraction method.
            hook_type: 'forward' or 'forward_pre' hooks used.
        """
        super().__init__()
        assert mode in ('train', 'eval')
        print(mode)
        if mode == 'train':
            model = model.train()
        else:
            model = model.eval()
        assert method in ('fx', 'hook')
        print(method)
        if method == 'hook':
            # names are module names
            assert hook_type in ('forward', 'forward_pre')
            print(hook_type)
            # from timm.models._features import FeatureHooks
            from timm.models.features import FeatureHooks
            module_names = [n for n, m in model.named_modules()]
            print(module_names)
            matched = []
            names = names or self.default_module_names
            print(names)
            for n in names:
                matched.extend(fnmatch.filter(module_names, n))
                print(n)
            if not matched:
                raise RuntimeError(f'No module names found matching {names}.')
            print(matched)
            hooks = [{'module': m} for m in matched]
            print(hooks)
            self.model = model
            self.hooks = FeatureHooks(
                hooks,
                # matched, 
                model.named_modules(), 
                default_hook_type=hook_type)
        self.names = matched
        self.mode = mode
        self.method = method
    def forward(self, x):
        if self.hooks is not None:
            self.model(x)
            output = self.hooks.get_output(device=x.device)
        return output
    

class AttentionMaps(torch.nn.Module):
    # defaults should cover a significant number of timm models with attention maps.
    def __init__(
            self,
            model,
            config,
            device,
                    ):
        super().__init__()
        self.model = model
        self.config = config
        self.device = device
    def forward(self, x):
        out = self.model(x)
        attention_maps_np = []
        for n, t in out.items():
            #print(n, t.shape)
            attention_maps_np.append(t.cpu().detach().numpy())
        return np.concatenate(attention_maps_np,axis=0)

def fisher_z_transform(correlation_values):
    """
    Apply Fisher Z-transform to correlation values.
    Args:
    - correlation_values (torch.Tensor): Tensor of correlation values.
    Returns:
    - torch.Tensor: Tensor of Fisher Z-transformed values.
    """
    z_output = 0.5 * np.log((1 + correlation_values) / (1 - correlation_values))
    return z_output

def fcn_prep_data_get_loaders_formaps(train_netmat, train_surface, validation_netmat, validation_surface, test_netmat, test_surface, netmat_prep_choice="demean", surf_prep_choice="norm", b_sz=32, write_fpath=''):
    '''
    Preprocessing function that takes in netmat parcellation of size N and suface maps for some component, like ICA. For netmats, input data is a subx[(N*(N-1))/2] matrix, 
    so upper triangle elements. parcellation = 100, then N=4950 and so on. Provided a tranformation condition is given, norm or fisherZ, ir applies both or either. Then, makes full connectome
    matrix for all subjects => data loader input for netmat becomes SUBxNxN = SUBx100x100 for example. For surface maps, it takes in their data BxCxPxV and makes into vertex for easier prediction
    and for kraken coder, because craken needs SUBxFEAT sapce to do latent and MSE reconstructions. 
    They all use this so make into a fcn all can call!
    '''
    tr_sub_dim, c_dim, p_dim, v_dim = train_surface.shape
    # write_to_file(f'regular surface shpae{train_surface.shape}', filepath=write_fpath)
    mean_train_netmat = np.mean(train_netmat, axis=0)
    sigma_train_netmat = np.std(train_netmat, axis=0)
    if surf_prep_choice=="norm":
        mean_train_surface = np.nanmean(train_surface, axis=0, keepdims=True) #1x15x320x153
        sigma_train_surface = np.nanstd(train_surface, axis=0, keepdims=True) #1x15x320x153
        normalized_train_surface = (train_surface - mean_train_surface) / (sigma_train_surface + 1e-99)
        normalized_val_surface = (validation_surface - mean_train_surface) / (sigma_train_surface  + 1e-99)
        normalized_test_surface = (test_surface - mean_train_surface) / (sigma_train_surface + 1e-99)
    else:
        # write_to_file('ALERT!!! USING RAW ICA MAPS', filepath=write_fpath)
        normalized_train_surface = train_surface
        normalized_val_surface = validation_surface
        normalized_test_surface = test_surface
    if netmat_prep_choice == "norm_fisherz":
        # write_to_file('NetMat prep chosen is both FISHERZ and NORM ', filepath=write_fpath)
        train_netmat_fz = fisher_z_transform(train_netmat) #fisheZ transform first
        validation_netmat_fz = fisher_z_transform(validation_netmat)
        test_netmat_fz = fisher_z_transform(test_netmat) 
        mean_train_netmat_fz = np.mean(train_netmat_fz, axis=0) #norm based on training data
        sigma_train_netmat_fz = np.std(train_netmat_fz, axis=0)
        tr_transformed_netmats = (train_netmat_fz - mean_train_netmat_fz) / (sigma_train_netmat_fz  + 10e-99)
        val_transformed_netmats = (validation_netmat_fz - mean_train_netmat_fz) / (sigma_train_netmat_fz  + 10e-99)
        te_transformed_netmats = (test_netmat_fz - mean_train_netmat_fz) / (sigma_train_netmat_fz  + 10e-99)
    elif netmat_prep_choice == "demean_fisherz":
        # write_to_file('NetMat prep chosen is both FISHERZ and DEMEAN ', filepath=write_fpath)
        train_netmat_fz = fisher_z_transform(train_netmat) #fisheZ transform first
        validation_netmat_fz = fisher_z_transform(validation_netmat)
        mean_train_netmat_fz = np.mean(train_netmat_fz, axis=0) #norm based on training data
        sigma_train_netmat_fz = np.std(train_netmat_fz, axis=0)
        tr_transformed_netmats = (train_netmat_fz - mean_train_netmat_fz) #/ (sigma_train_netmat_fz  + 10e-99)
        val_transformed_netmats = (validation_netmat_fz - mean_train_netmat_fz)# / (sigma_train_netmat_fz  + 10e-99)
    elif netmat_prep_choice == "norm":
        # write_to_file('NetMat prep chosen is NORM ', filepath=write_fpath)
        tr_transformed_netmats = (train_netmat - mean_train_netmat)/ (sigma_train_netmat  + 10e-99) # normalize r values?
        val_transformed_netmats = (validation_netmat - mean_train_netmat)/ (sigma_train_netmat  + 10e-99)
    elif netmat_prep_choice == "fisherz":
        # write_to_file('NetMat prep chosen is FISHERZ ', filepath=write_fpath)
        tr_transformed_netmats = fisher_z_transform(train_netmat)
        val_transformed_netmats = fisher_z_transform(validation_netmat)
    elif netmat_prep_choice == "demean":
        # write_to_file('NetMat prep chosen is DEMEAN ', filepath=write_fpath)
        tr_transformed_netmats = (train_netmat - mean_train_netmat)
        val_transformed_netmats = (validation_netmat - mean_train_netmat)
        te_transformed_netmats = (test_netmat - mean_train_netmat)
    else:
        tr_transformed_netmats = train_netmat
        val_transformed_netmats = validation_netmat
        te_transformed_netmats = test_netmat
    train_netmat_np = tr_transformed_netmats
    val_netmat_np = val_transformed_netmats
    test_netmat_np = te_transformed_netmats
    #### MODEL DATALOADERS
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized_train_surface).float(), torch.from_numpy((train_netmat_np)).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = b_sz, shuffle=True, num_workers=10)
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized_val_surface).float(), torch.from_numpy((val_netmat_np)).float())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = b_sz, shuffle=True, num_workers=10)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized_test_surface).float(), torch.from_numpy((test_netmat_np)).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle=True, num_workers=10)
    return train_loader, val_loader, test_loader, mean_train_netmat

# %%

if "/Users/snaranjo/" in os.getcwd():
    local_flag=True
    using_local_root="/Users/snaranjo/Desktop/neurotranslate/mount_point"
else:
    local_flag=False
    using_local_root=""

specific_trained_model = f"kSiTLN_120325_d6h3_tiny_adamW_cosinedecay_recon_MSEtrain_1L_full_demean_exp100_wGelu_MSE"
scratch_path=f"{using_local_root}/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"
root=f"{scratch_path}/NeuroTranslate/surf2netmat"
path_to_config = f"{root}/config/SiT_LN"
path_to_ckpt= f"{using_local_root}/ceph/chpc/home/naranjorincon/neurotranslate/surf2netmat/logs/ICAd15_schfd100/ABCD/kSiTLN/normICAdemeanMAT"
device="cpu"
# config_file_chosen="config_120325_d6h3_tiny_adamW_cosinedecay_recon_MSEtrain_1L_full_demean_exp100_wGelu.yml"
config_file_chosen = "hparams_SiTLN_recon_getmaps.yml"

# once paths are set, we move to loading that config file and then using its contents to populate our variables herein
config_file_path = os.path.join(path_to_config,config_file_chosen)
print(config_file_path)
with open(config_file_path) as f:
        config_ckpt = yaml.safe_load(f)
        config = config_ckpt#yaml.safe_load(f)

# get shapes and sizes of ico data
sub_ico = config_ckpt['mesh_resolution']['ico_grid']
ico = config_ckpt['mesh_resolution']['ico_mesh']
num_patches = config_ckpt['sub_ico_{}'.format(sub_ico)]['num_patches']
num_verteces = config_ckpt['sub_ico_{}'.format(sub_ico)]['num_vertices']
num_channels = config_ckpt['data']['to_icamap']
path_to_template = f"{root}/surfaces" #config_ckpt['data']['path_to_template']
mask = np.array(nb.load(f'{path_to_template}/week-40_hemi-left_space-dhcpSym_dens-40k_desc-medialwallsymm_mask.shape.gii').agg_data())

triangle_indices = pd.read_csv(f'{root}/patch_extraction/triangle_indices_ico_{ico}_sub_ico_{sub_ico}.csv')   

#Now, we have ico/subico shape and size, a mask, and corresponding trinagle indeces
config_ckpt['training']['bs']=1
config_ckpt['training']['bs_val']=1
config_ckpt['data']['loader']='metrics' #metrics or numpy

# data_path = config_ckpt['data']['path_to_metrics'].format(config_ckpt['data']['dataset'],config_ckpt['data']['configuration'])
# train_loader, val_loader, test_loader = get_dataloaders_metrics(config_ckpt,data_path)
# get train, val, test loader
dataset_choice=config_ckpt['training']['dataset_choice']
from_parcellation=config['data']['from_parcellation']
hemi_cond=config_ckpt['training']['hemi_cond']
train_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_train_netmat_clean.npy")
train_surf_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{sub_ico}/{hemi_cond}_train_surf.npy")
val_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_val_netmat_clean.npy")
val_surf_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{sub_ico}/{hemi_cond}_val_surf.npy")
test_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_test_netmat_clean.npy")
test_surf_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico0{sub_ico}/{hemi_cond}_test_surf.npy")
#now make them into loaders
train_loader, val_loader, test_loader, mean_train_label = fcn_prep_data_get_loaders_formaps(train_netmat_np, train_surf_np, val_netmat_np, val_surf_np, test_netmat_np, test_surf_np)
    
# config['split_to_test']='test'
if config['split_to_test'] == 'train':
    print('running segmentation on the train set')
    loader = train_loader
elif config['split_to_test'] == 'val':
    print('running segmentation on the validation set')
    loader = val_loader
elif config['split_to_test'] == 'test':
    print('running segmentation on the test set')
    loader = test_loader
else:
    raise('Not implemented')

fcn_model_module = getattr(models, config['training']['fcn_model_to_use']) 
dim = config['transformer']['sit_dim']
depth = config['transformer']['depth']
heads = config['transformer']['heads']
dim_p = num_patches
upper_tri = ((from_parcellation * (from_parcellation-1)) // 2) #gives integer of upper triangle element count for parcellation
dim_c = num_channels
dim_v = num_verteces
dim_head = config['transformer']['dim_head']
dropout = config['transformer']['dropout']
emb_dropout = config['transformer']['emb_dropout']
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

# %%
# get and load weights
# weights = torch.load(os.path.join(path_to_ckpt,'checkpoint.pth'),map_location=device)
weights = torch.load(os.path.join(path_to_ckpt,f'{specific_trained_model}.pt'),map_location=device)
print('** Loading checkpoint **')
model.load_state_dict(weights,strict=True)
model.to(device)

# %%
##############################################################
# Instatiate the Extract Attention class
hook_model = AttentionExtract(model, method='hook')

##############################################################
# Instantiate the Attention Maps class
hook_attention_maps = AttentionMaps(hook_model, config_ckpt, device)

##############################################################
# Logging
output_dir = f'{root}/outputs/attention_maps'
current_time = datetime.datetime.now().strftime("%m_%d_%H%M%S")
output_dir = os.path.join(output_dir, path_to_ckpt.split('/')[-1],specific_trained_model, current_time)
print(output_dir)
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# %%
##############################################################
# Extract Attention

list_ids = config['subject_id'] #[2]?

with torch.no_grad():
    for id, data in enumerate(loader):
        if id in list_ids:
            print('**** subject {} ****'.format(id))
            inputs, targets = data[0].to(device), data[1].to(device)
            out_maps = hook_attention_maps(inputs) # this is after the softmax
            # print(out_maps[1].shape,len(out_maps))
            nh = out_maps.shape[1] # number of head
            nl = out_maps.shape[0] # number of layers
            # we keep only the output patch attention
            # attentions = out_maps[:, :, 1:, 0].reshape(nh, -1,nl)
            attentions = out_maps[:, :, :, 0].reshape(nh, -1,nl)
            print(attentions.shape)

            ### saving all maps
            reconstructed_sphere_am = np.zeros((40962,nl,nh),dtype=np.float32)
            new_inputs = np.transpose(attentions,(1,2,0))
            for i in range(num_patches):
                indices_to_extract = triangle_indices[str(i)].values
                reconstructed_sphere_am[indices_to_extract,:,:] = new_inputs[i,:,:]

            # import pdb;pdb.set_trace()
            reconstructed_sphere_am = np.multiply(np.expand_dims(mask,axis=(1,2)),reconstructed_sphere_am)
            reconstructed_sphere_am = rearrange(reconstructed_sphere_am, 'a b c -> a (b c)')
            save_gifti(reconstructed_sphere_am, os.path.join(output_dir, 'subject_{}_attention_all_layers.shape.gii'.format(id)))
            p1 = subprocess.Popen(['wb_command', '-set-structure',os.path.join(output_dir, 'subject_{}_attention_all_layers.shape.gii'.format(id)), 'CORTEX_LEFT'])
            p1.wait()

            ### averaging maps per layer
            reconstructed_sphere_am = np.zeros((40962,1,nh),dtype=np.float32)
            new_inputs = np.transpose(attentions,(1,2,0))
            new_inputs =np.mean(new_inputs, axis=1, keepdims=True)
            for i in range(num_patches):
                indices_to_extract = triangle_indices[str(i)].values
                reconstructed_sphere_am[indices_to_extract,:,:] = new_inputs[i,:,:]

            reconstructed_sphere_am = np.multiply(np.expand_dims(mask,axis=(1,2)),reconstructed_sphere_am)
            reconstructed_sphere_am = rearrange(reconstructed_sphere_am, 'a b c -> a (b c)')
            save_gifti(reconstructed_sphere_am, os.path.join(output_dir, 'subject_{}_attention_avg_all_layers.shape.gii'.format(id)))
            p1 = subprocess.Popen(['wb_command', '-set-structure',os.path.join(output_dir, 'subject_{}_attention_avg_all_layers.shape.gii'.format(id)), 'CORTEX_LEFT'])
            p1.wait()

            ### MAX maps per layer
            reconstructed_sphere_am = np.zeros((40962,1,nh),dtype=np.float32)
            new_inputs = np.transpose(attentions,(1,2,0))
            new_inputs =np.max(new_inputs, axis=1, keepdims=True)
            for i in range(num_patches):
                indices_to_extract = triangle_indices[str(i)].values
                reconstructed_sphere_am[indices_to_extract,:,:] = new_inputs[i,:,:]

            reconstructed_sphere_am = np.multiply(np.expand_dims(mask,axis=(1,2)),reconstructed_sphere_am)
            reconstructed_sphere_am = rearrange(reconstructed_sphere_am, 'a b c -> a (b c)')
            save_gifti(reconstructed_sphere_am, os.path.join(output_dir, 'subject_{}_attention_max_all_layers_per_head.shape.gii'.format(id)))
            p1 = subprocess.Popen(['wb_command', '-set-structure',os.path.join(output_dir, 'subject_{}_attention_max_all_layers_per_head.shape.gii'.format(id)), 'CORTEX_LEFT'])
            p1.wait()

            ### avg maps per head
            reconstructed_sphere_am = np.zeros((40962,nl,1),dtype=np.float32)
            new_inputs = np.transpose(attentions,(1,2,0))
            new_inputs =np.max(new_inputs, axis=2, keepdims=True)
            for i in range(num_patches):
                indices_to_extract = triangle_indices[str(i)].values
                reconstructed_sphere_am[indices_to_extract,:,:] = new_inputs[i,:,:]

            reconstructed_sphere_am = np.multiply(np.expand_dims(mask,axis=(1,2)),reconstructed_sphere_am)
            reconstructed_sphere_am = rearrange(reconstructed_sphere_am, 'a b c -> a (b c)')
            save_gifti(reconstructed_sphere_am, os.path.join(output_dir, 'subject_{}_attention_avg_all_layers_per_head.shape.gii'.format(id)))
            p1 = subprocess.Popen(['wb_command', '-set-structure',os.path.join(output_dir, 'subject_{}_attention_avg_all_layers_per_head.shape.gii'.format(id)), 'CORTEX_LEFT'])
            p1.wait()

            ### averaging everything
            reconstructed_sphere_am = np.zeros((40962,1,1),dtype=np.float32)
            new_inputs = np.transpose(attentions,(1,2,0))
            new_inputs =np.mean(new_inputs, axis=1, keepdims=True)
            new_inputs =np.mean(new_inputs, axis=2, keepdims=True)
            for i in range(num_patches):
                indices_to_extract = triangle_indices[str(i)].values
                reconstructed_sphere_am[indices_to_extract,:,:] = new_inputs[i,:,:]

            reconstructed_sphere_am = np.multiply(np.expand_dims(mask,axis=(1,2)),reconstructed_sphere_am)
            reconstructed_sphere_am = rearrange(reconstructed_sphere_am, 'a b c -> a (b c)')
            save_gifti(reconstructed_sphere_am, os.path.join(output_dir, 'subject_{}_attention_avg_all.shape.gii'.format(id)))
            p1 = subprocess.Popen(['wb_command', '-set-structure',os.path.join(output_dir, 'subject_{}_attention_avg_all.shape.gii'.format(id)), 'CORTEX_LEFT'])
            p1.wait()

            ### maxing everything
            reconstructed_sphere_am = np.zeros((40962,1,1),dtype=np.float32)
            new_inputs = np.transpose(attentions,(1,2,0))
            new_inputs =np.max(new_inputs, axis=1, keepdims=True)
            new_inputs =np.max(new_inputs, axis=2, keepdims=True)
            for i in range(num_patches):
                indices_to_extract = triangle_indices[str(i)].values
                reconstructed_sphere_am[indices_to_extract,:,:] = new_inputs[i,:,:]

            reconstructed_sphere_am = np.multiply(np.expand_dims(mask,axis=(1,2)),reconstructed_sphere_am)
            reconstructed_sphere_am = rearrange(reconstructed_sphere_am, 'a b c -> a (b c)')
            save_gifti(reconstructed_sphere_am, os.path.join(output_dir, 'subject_{}_attention_max_all.shape.gii'.format(id)))
            p1 = subprocess.Popen(['wb_command', '-set-structure',os.path.join(output_dir, 'subject_{}_attention_max_all.shape.gii'.format(id)), 'CORTEX_LEFT'])
            p1.wait()

            ### save input data
            new_inputs =  np.transpose(inputs.detach().cpu().numpy(),(0,2,1,3))
            input_sphere = np.zeros((40962,num_channels),dtype=np.float32)
            for i in range(num_patches):
                indices_to_extract = triangle_indices[str(i)].values
                input_sphere[indices_to_extract,:] = new_inputs[0,i,:,:].transpose()
            save_gifti(input_sphere, os.path.join(output_dir, 'subject_{}_input_metrics.shape.gii'.format(id)) )
            # import pdb;pdb.set_trace()
            p1 = subprocess.Popen(['wb_command', '-set-structure', os.path.join(output_dir, 'subject_{}_input_metrics.shape.gii'.format(id)), 'CORTEX_LEFT'])
            p1.wait()

            p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_all_layers.shape.gii'.format(id)),'{}/ico-6.L.surf.gii'.format(path_to_template),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches), 'BARYCENTRIC', os.path.join(output_dir, 'resample_subject_{}_attention_all_layers{}.shape.gii'.format(id,num_patches))])
            p1.wait()
            # import pdb;pdb.set_trace()

            p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_all_layers{}.shape.gii'.format(id,num_patches)),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches),'{}/ico-6.L.surf.gii'.format(path_to_template), 'BARYCENTRIC', os.path.join(output_dir, 'resample_subject_{}_attention_all_layers_resamp.shape.gii'.format(id))])
            p1.wait()

            p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_avg_all_layers.shape.gii'.format(id)),'{}/ico-6.L.surf.gii'.format(path_to_template),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches), 'BARYCENTRIC', os.path.join(output_dir, 'resample_subject_{}_attention_avg_all_layers{}.shape.gii'.format(id,num_patches))])
            p1.wait()

            p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_avg_all_layers{}.shape.gii'.format(id,num_patches)),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches),'{}/ico-6.L.surf.gii'.format(path_to_template), 'BARYCENTRIC', os.path.join(output_dir, 'resample_subject_{}_attention_avg_all_layers_resamp.shape.gii'.format(id))])
            p1.wait()

            p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_max_all_layers_per_head.shape.gii'.format(id)),'{}/ico-6.L.surf.gii'.format(path_to_template),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches), 'BARYCENTRIC', os.path.join(output_dir, 'resample_subject_{}_attention_max_all_layers_per_head{}.shape.gii'.format(id,num_patches))])
            p1.wait()

            p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_max_all_layers_per_head{}.shape.gii'.format(id,num_patches)),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches),'{}/ico-6.L.surf.gii'.format(path_to_template), 'BARYCENTRIC', os.path.join(output_dir, 'resample_subject_{}_attention_max_all_layers_per_head_resamp.shape.gii'.format(id))])
            p1.wait()

            p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_avg_all_layers_per_head.shape.gii'.format(id)),'{}/ico-6.L.surf.gii'.format(path_to_template),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches), 'BARYCENTRIC', os.path.join(output_dir, 'resample_subject_{}_attention_avg_all_layers_per_head{}.shape.gii'.format(id,num_patches))])
            p1.wait()

            p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_avg_all_layers_per_head{}.shape.gii'.format(id,num_patches)),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches),'{}/ico-6.L.surf.gii'.format(path_to_template), 'BARYCENTRIC', os.path.join(output_dir, 'resample_subject_{}_attention_avg_all_layers_per_head_resamp.shape.gii'.format(id))])
            p1.wait()

            p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_avg_all.shape.gii'.format(id)),'{}/ico-6.L.surf.gii'.format(path_to_template),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches), 'BARYCENTRIC', os.path.join(output_dir, 'resample_subject_{}_attention_avg_all{}.shape.gii'.format(id,num_patches))])
            p1.wait()

            p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_avg_all{}.shape.gii'.format(id,num_patches)),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches),'{}/ico-6.L.surf.gii'.format(path_to_template), 'BARYCENTRIC', os.path.join(output_dir, 'resample_subject_{}_attention_avg_all_resamp.shape.gii'.format(id))])
            p1.wait()

            p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_max_all.shape.gii'.format(id)),'{}/ico-6.L.surf.gii'.format(path_to_template),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches), 'BARYCENTRIC', os.path.join(output_dir, 'resample_subject_{}_attention_max_all{}.shape.gii'.format(id,num_patches))])
            p1.wait()

            p1 = subprocess.Popen(['wb_command', '-metric-resample',os.path.join(output_dir, 'subject_{}_attention_max_all{}.shape.gii'.format(id,num_patches)),'{}/sphere.{}.L.surf.gii'.format(path_to_template,num_patches),'{}/ico-6.L.surf.gii'.format(path_to_template), 'BARYCENTRIC', os.path.join(output_dir, 'resample_subject_{}_attention_max_all.shape.gii'.format(id))])
            p1.wait()


