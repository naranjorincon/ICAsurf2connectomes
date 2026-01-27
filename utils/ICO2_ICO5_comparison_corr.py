# %%
import numpy as np
from utils import * 

# %%
write_fpath = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/ico2_ICO5_comparison_corr"
data_root_path = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch" #/Users/snaranjo/Desktop/neurotranslate/mount_point
dataset_choice = "ABCD"
# write_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/kSiTLN_recon_normICAdemeanfishzMAT.write_to_file"
hemi_cond="1L"

# load in data
train_surf_np_ico2 = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico02/{hemi_cond}_train_surf.npy")
write_to_file(f'Loaded in TRAIN shape: {train_surf_np_ico2.shape} respectively.', filepath=write_fpath)
train_surf_np_ico5 = np.load(f"{data_root_path}/NeuroTranslate/brain_reps_datasets/{dataset_choice}/ICA_maps/ICAd15_ico05/{hemi_cond}_train_surf_ico5.npy")
write_to_file(f'Loaded in TRAIN shape: {train_surf_np_ico5.shape} respectively.', filepath=write_fpath)

N, C2, P2, V2 = train_surf_np_ico2.shape
_, C5, P5, V5 = train_surf_np_ico5.shape

reshape_ico2 = train_surf_np_ico2.reshape(N, C2, P2*V2)
reshape_ico5 = train_surf_np_ico5.reshape(N, C5, P5*V5)
# channel_to_corr = 1 #1=DMN i think
# reshape_ico2 = reshape_ico2[:,channel_to_corr,:].squeeze()
# reshape_ico5 = reshape_ico5[:,channel_to_corr,:].squeeze()
write_to_file(f"What goes into the corrcoeff: {reshape_ico2.shape}/{reshape_ico5.shape}", filepath=write_fpath)

#### SAME SUBJECT RESULTS FOR VIOLIN PLOTS ####
# corr1_perchannel = []
# corr2_perchannel = []
# for cc in range(C2):
#     corr1_persubj = []
#     corr2_persubj = []
#     write_to_file(f"calculating corr across subjects for channel: {cc}", filepath=write_fpath)
        
#     for ii in range(N):
#         calc_corr1 = np.corrcoef((reshape_ico2[ii, cc], reshape_ico2[ii, cc]))[0,1]
#         calc_corr2 = np.corrcoef(reshape_ico5[ii, cc], reshape_ico5[ii, cc])[0,1]
#         corr1_persubj.append(calc_corr1)
#         corr2_persubj.append(calc_corr2) 

#     write_to_file(f"Length of corr_perchannels are: {len(corr1_persubj)}", filepath=write_fpath)

#     corr1_perchannel.append(corr1_persubj) # should end be len()==15, where each I=N
#     corr2_perchannel.append(corr2_persubj) # should end be len()==15, where each i=N


# np.save('/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/train_ico2_corr_mat.npy', corr1_perchannel)
# np.save('/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/train_ico5_corr_mat.npy', corr2_perchannel)

#### ACROSS SUBJECTS RESULTS FOR VIOLIN PLOTS ####
corr1_perchannel_ico2_same = []
corr2_perchannel_ico2_other = []
corr1_perchannel_ico5_same = []
corr2_perchannel_ico5_other = []
for cc in range(C2):
    # corr1_persubj_ico2 = [] # resets after every channel
    # corr2_persubj_ico2 = []
    # corr1_persubj_ico5 = []
    # corr2_persubj_ico5 = []
    write_to_file(f"calculating corr across subjects for channel: {cc}", filepath=write_fpath)
        
    # for ii in range(N):
    calc_corr_ico2 = np.corrcoef(reshape_ico2[:,cc,:]).squeeze()
    same_subj = np.diag(calc_corr_ico2)
    other_subj_idx = np.tril_indices(calc_corr_ico2.shape[0], -1)
    other_subj = calc_corr_ico2[other_subj_idx] # should be lower triangle not including diag ==> 
    # write_to_file(f"ii shape: {same_subj.shape}==N, ij shape: {other_subj.shape}==0.5*N*(N-1)", filepath=write_fpath) # ideally its 1xN cause then we geat each subject across ech channel with everyone else ow will need to corrcoeff then upper trinagle idx
    
    # corr1_persubj_ico2.append(same_subj)
    # corr2_persubj_ico2.append(other_subj) 
    corr1_perchannel_ico2_same.append(same_subj) # should end be len()==15, where each I=N
    corr2_perchannel_ico2_other.append(other_subj) # should end be len()==15, where each i=N

    calc_corr_ico5 = np.corrcoef(reshape_ico5[:,cc,:]).squeeze()
    same_subj = np.diag(calc_corr_ico5)
    other_subj_idx = np.tril_indices(calc_corr_ico5.shape[0], -1)
    other_subj = calc_corr_ico5[other_subj_idx] # should be lower triangle not including diag ==> 
    
    corr1_perchannel_ico5_same.append(same_subj) # should end be len()==15, where each I=N
    corr2_perchannel_ico5_other.append(other_subj) # should end be len()==15, where each i=N

    # corr1_persubj_ico5.append(same_subj)
    # corr2_persubj_ico5.append(other_subj) 


#ico2
np.save('/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/train_ico2_corr_samesubj_channels.npy', corr1_perchannel_ico2_same)
np.save('/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/train_ico2_corr_othersubj_channels.npy', corr2_perchannel_ico2_other)

# ico5
np.save('/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/train_ico5_corr_samesubj_channels.npy', corr1_perchannel_ico5_same)
np.save('/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/train_ico5_corr_othersubj_channels.npy', corr2_perchannel_ico5_other)

