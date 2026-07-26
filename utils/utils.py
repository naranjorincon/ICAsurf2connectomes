import torch
# import torch.nn as nn
# import torch.optim as optim # all models will use this, so defined here so all call same script
import numpy as np
import pandas as pd
import nibabel as nib
# import numpy.linalg as LA

# =========================================== WHAT WE ACTUALLY USE =========================================== #
def write_to_file(content, filepath="", also_print=True):
    with open(filepath, 'a') as file:
        file.write(str(content) + '\n')
    if also_print:
        print(content)
    return
        
def generate_subsequent_mask(size):
    """
    Generate a mask to ensure that each position in the sequence can only attend to
    positions up to and including itself. This is a upper triangular matrix filled with ones.
    
    :param size: int, the length of the sequence
    :return: tensor of shape (size, size), where element (i, j) is False if j <= i, and True otherwise (See attn_mask option here: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
    """
    # i comment below because if you make siagonal=+1 as below, you get the same resutls with one line of code
    mask = torch.triu(torch.ones(size, size),1).bool() #on and above diagonal included
    # mask.diagonal().fill_(False) # only upper tri has values
    return mask#.bool()

def make_netmat(data, netmat_dim=100):
    '''
    Makes netmat from upper triangle in numpy
    '''
    sing_sub = int((netmat_dim * (netmat_dim-1))/2)
    # get indeces of upptri cause all these vec netmats are upper trinagles. 
    # ones is nice cause makes diagonal auto==1
    out_mat_init = np.zeros(2*sing_sub+netmat_dim).reshape(netmat_dim,netmat_dim)
    # inds_lowtri = np.triu_indices_from(out_mat_init,k=1) # k=1 means no diagonal?
    inds_lowtri = np.tril_indices_from(out_mat_init,k=-1) # k=1 means no diagonal?

    out_mat_init[inds_lowtri] = data #presumably, data is from upper triangle from matlab but needs index lower trinagle to have good visuals not sure why...
    out_mat_init = out_mat_init + out_mat_init.T
    np.fill_diagonal(out_mat_init, 1)
    return out_mat_init # returns netmat version from vector form

def make_netmat_into_trinagle(data, netmat_dim: int=100, upper_trinagle: bool=True):
    '''
    takes vectorized triangle data and turns it into a netmat with 0 mask depending on which is chosen
    '''
    sing_sub = int((netmat_dim * (netmat_dim-1))/2) #parcellation upper/lower trinagle elements
    out_mat_init = np.zeros(2*sing_sub+netmat_dim).reshape(netmat_dim,netmat_dim)
    if upper_trinagle is True:
        # inds_tri = np.triu_indices_from(out_mat_init,k=1) # k=1 means no diagonal
        inds_tri = np.tril_indices_from(out_mat_init,k=-1) 
        out_mat_init[inds_tri] = data #presumably, data is from upper triangle from matlab but needs index lower trinagle to have good visuals not sure why...
        out_mat_init = out_mat_init.T
        # np.fill_diagonal(out_mat_init, 0) # no diagonal, that can be added another time. For now, used to compare true/translation so 0 diagonals
    else:
        inds_tri = np.tril_indices_from(out_mat_init,k=-1) # k=-1 means no diagonal
   
        out_mat_init[inds_tri] = data #presumably, data is from upper triangle from matlab but needs index lower trinagle to have good visuals not sure why...
        out_mat_init = out_mat_init
        # np.fill_diagonal(out_mat_init, 0)
    return out_mat_init # returns netmat version from vector form

def make_nemat_allsubj(data, num_nodes):
    '''
    Takes a numpy array of size [num_subj, size_vectorized_netmat] and reshapes to [num_subj, num_nodes, num_nodes]
    '''
    out = np.zeros([data.shape[0], num_nodes, num_nodes]) # init with ones of same size as output
    for i in range(data.shape[0]): # for each sub
        out[i, :, :] = make_netmat(data[i], num_nodes)
    return out

def make_nemat_allsubj_triangle_only(data, num_nodes, upper_trinagle):
    '''
    Takes a numpy array of size [num_subj, size_vectorized_netmat] and reshapes to [num_subj, num_nodes, num_nodes]
    '''
    out = np.zeros([data.shape[0], num_nodes, num_nodes]) # init with ones of same size as output
    for i in range(data.shape[0]): # for each sub
        out[i, :, :] = make_netmat_into_trinagle(data[i], num_nodes, upper_trinagle=upper_trinagle)
    return out

def mat2vector(mat, diagonal_flag: bool=False):
    '''
    Right now, expects input bxpxp cause channel gets squeezed in curr verison of diff train function below
    '''
    og_mat_shape = mat.shape
    parcellation_sz = mat.shape[-1] # last one will always be parcellatino size if its bxpxp
    if diagonal_flag:
        upper_tri_indices = np.triu_indices(parcellation_sz, k=0)
    else:
        upper_tri_indices = np.triu_indices(parcellation_sz, k=1)
    # when runing train, we only use one subject so .squeeze() 
    # removes batch x channel and leaves us w pxp so indx is different
    if len(og_mat_shape) == 2: # if onyl 2 dim then @ test or batch = 1
        upper_triangle_vectors = mat[upper_tri_indices[0], upper_tri_indices[1]]
    elif len(og_mat_shape) > 2:
        upper_triangle_vectors = mat[:, upper_tri_indices[0], upper_tri_indices[1]]
    return upper_triangle_vectors

def BGBMT_mesh_greedy_decode(model, source, dec_channels, ico_patch, ico_vertex, device, b=1, target=None):
    encoder_output, latent_post_trasnformer, latent_post_projection2mesh_lowdim,  = model.encode(source) # output is bx321x384
    decoder_input = torch.ones(b, dec_channels, ico_patch, ico_vertex).to(device) # ones of bx15x320x153
    #decoder_input[:, :, :] = target.to(device)[:, :, :]
    decoder_mask = generate_subsequent_mask(ico_patch).to(device) #makes a masked upper tri.. 320x320 is patches==320

    for i in range(ico_patch):
        # out = model.decode(encoder_out=encoder_output, tgt=decoder_input, mesh_mask=decoder_mask)
        out = model.decode(encoder_output=encoder_output, tgt=decoder_input, mesh_mask=decoder_mask) # out is bx15x320x153
        # print(out.shape)
        decoder_input[:, :, :, :] = torch.tensor(out[:, :, :, :])
        subj_dim, chnll_dim, patch_dim, vert_dim = out.shape
        vec_out_for_test =  out.reshape(subj_dim,chnll_dim*patch_dim*vert_dim)

    return decoder_input, out, vec_out_for_test

def matrix_to_mesh(input_mat, tri_indices_ico6subico2_fpath=None, out_fpath=None):
    '''
    This function will take a numpy array of size [num_channels, 320, 153] and transform it into a shape.gii (GIFTI) file to overlay on an ico6 surface.

    Inputs 
    ----------
    input_mat: np.ndarray
        Array of shape [num_channels, 320, 153] containing the surface information for a single subject you want to make into a shape.gii file

    num_channels: int
        Number of channels in the surface (e.g., number if ICA dims)

    tri_indices_ico6subico2_fpath: str
        Path to file with mapping from 320x153 matrix to ico6 sphere vertices. This csv is also available here: https://github.com/metrics-lab/surface-vision-transformers/blob/main/utils/triangle_indices_ico_6_sub_ico_2.csv

    out_fpath: str 
        Path to which you'd like to save the generated GIFTI file. Default: None will NOT save the GIFTI file

    Outputs
    ----------
    out: nib.GiftiImage
        GIFTI image filled in with surface information stored in the input matrix
    '''

    batch_sz = 1
    num_channels, num_patches, num_vert = input_mat.shape
    print(f"C:{num_channels}, P:{num_patches} V:{num_vert}")
    if tri_indices_ico6subico2_fpath is None:
        assert 1==2, "No ICO file provided."

    indices_mesh_triangles = pd.read_csv(tri_indices_ico6subico2_fpath)
    # write_to_file(f'Read in ico6-2 path, has this shape:{indices_mesh_triangles.shape} \nAnd head: {indices_mesh_triangles.head()}', filepath=write_fpath)
    mesh_vec = np.zeros([batch_sz, num_channels, 40962]) # forced to be 40962 to make ico6 verteces

    for b in range(batch_sz):
        for i in range(num_channels):
            for j in range(num_patches):
                indices_to_insert = indices_mesh_triangles[str(j)].to_numpy()
                mesh_vec[b, i, indices_to_insert] = input_mat[i, j, :] #orig is input_mat[b, i, j, :]

    mesh_ico6=mesh_vec

    out = nib.GiftiImage()
    for b in range(batch_sz):
        for i in range(num_channels):
            out.add_gifti_data_array(nib.gifti.GiftiDataArray(mesh_vec[b, i, :].astype("float32")))

    # if out_fpath is not None:
    out.to_filename(out_fpath + '.shape.gii')

    return mesh_ico6

def matrix_to_mesh_keepico(input_mat, tri_indices_ico6subico2_fpath=None, out_fpath=None):

    batch_sz = 1
    num_channels, num_patches, num_vert = input_mat.shape
    # indices_mesh_triangles = pd.read_csv(tri_indices_ico6subico2_fpath)
    # mesh_vec = np.zeros([batch_sz, num_channels, num_patches*num_vert]) # only one hemi?
    mesh_vec = np.reshape(input_mat, (batch_sz,num_channels,42))

    # for b in range(batch_sz):
    #     for j in range(num_patches):
    #         indices_to_insert = indices_mesh_triangles[str(j)].to_numpy()
    #         mesh_vec[b, :, indices_to_insert] = input_mat[:, j, :] #orig is input_mat[b, i, j, :]

    out = nib.GiftiImage()
    for i in range(num_channels):
        out.add_gifti_data_array(nib.gifti.GiftiDataArray(mesh_vec[0, i, :].astype("float32")))
    series = out.agg_data()
    print(f"Series shape: {series.shape}")
    # if out_fpath is not None:
    out.to_filename(out_fpath + '.shape.gii')

    return out

def all_matrix_to_mesh(input_mat, tri_indices_ico6subico2_fpath, out_fpath=None):
    '''
    adapting matrix_to_mesh for many subjects
    '''

    N, num_channels, num_patches, num_vert = input_mat.shape
    batch_sz=N
    indices_mesh_triangles = pd.read_csv(tri_indices_ico6subico2_fpath)
    mesh_vec = np.zeros([batch_sz, num_channels, 40962]) # only one hemi
    out = nib.GiftiImage()
    for b in range(batch_sz):
        for i in range(num_channels):
            out.add_gifti_data_array(nib.gifti.GiftiDataArray(mesh_vec[b, i, :].astype("float32")))
            for j in range(num_patches):
                indices_to_insert = indices_mesh_triangles[str(j)].to_numpy()
                mesh_vec[b, i, indices_to_insert] = input_mat[i, j, :] #orig is input_mat[b, i, j, :]
    
    
    if out_fpath is not None:
        out.to_filename(out_fpath + '.shape.gii')

    return out

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

def fcn_prep_data_get_loaders_ICAren(train_surface, validation_surface, b_sz=32, write_fpath='',train_subject_list=None, validation_subject_list=None):
    '''
    Preprocessing function that takes in netmat parcellation of size N and suface maps for some component, like ICA. For netmats, input data is a subx[(N*(N-1))/2] matrix, 
    so upper triangle elements. parcellation = 100, then N=4950 and so on. Provided a tranformation condition is given, norm or fisherZ, ir applies both or either. Then, makes full connectome
    matrix for all subjects => data loader input for netmat becomes SUBxNxN = SUBx100x100 for example. For surface maps, it takes in their data BxCxPxV and makes into vertex for easier prediction
    and for kraken coder, because craken needs SUBxFEAT sapce to do latent and MSE reconstructions. 
    They all use this so make into a fcn all can call!
    '''
    write_to_file(f'regular surface TRAIN shape: {train_surface.shape}', filepath=write_fpath)
    write_to_file('Cleaning data first if needed! Training data.', filepath=write_fpath)
    train_surface, _, train_subjects_to_keep = fcn_get_clean_data(surf=train_surface, netmat=None, subject_list_path=train_subject_list, write_fpath=write_fpath, ICA_surf_only=True)
    write_to_file('Cleaning data first if needed! For validation/test now.', filepath=write_fpath)
    validation_surface, _, validation_subjects_to_keep = fcn_get_clean_data(surf=validation_surface, netmat=None, subject_list_path=validation_subject_list, write_fpath=write_fpath, ICA_surf_only=True)
    
    tr_sub_dim, c_dim, p_dim, v_dim = train_surface.shape
    mean_train_surface = np.nanmean(train_surface, axis=0, keepdims=True) #1x15x320x153
    sigma_train_surface = np.nanstd(train_surface, axis=0, keepdims=True) #1x15x320x153
    normalized_train_surface = (train_surface - mean_train_surface) / (sigma_train_surface + 10e-99)
    normalized_train_surface_reshaped = normalized_train_surface.reshape(tr_sub_dim, c_dim*p_dim*v_dim)

    write_to_file(f'regular surface VAL shape: {validation_surface.shape}', filepath=write_fpath)
    val_num_sub, _, _, _ = validation_surface.shape
    normalized_val_surface = (validation_surface - mean_train_surface) / (sigma_train_surface  + 10e-99)
    normalized_val_surface_reshaped = normalized_val_surface.reshape(val_num_sub, c_dim*p_dim*v_dim)

    #### MODEL DATALOADERS
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized_train_surface).float(), torch.from_numpy((normalized_train_surface_reshaped)).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = b_sz, shuffle=True, num_workers=10)
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized_val_surface).float(), torch.from_numpy((normalized_val_surface_reshaped)).float())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = b_sz, shuffle=True, num_workers=10)

    # only when doing reconstuction
    # if model.__class__.__name__ == "SurfaceImageTransformer_ICArecon":
    mean_train_surface = mean_train_surface.reshape(1, c_dim*p_dim*v_dim)
    del normalized_train_surface, normalized_train_surface_reshaped, normalized_val_surface, normalized_val_surface_reshaped
    return train_loader, val_loader, mean_train_surface, train_subjects_to_keep, validation_subjects_to_keep

def fcn_get_clean_data(surf: None, netmat: None, subject_list_path: str="", write_fpath: str="", bilateral_condition: bool=False, ICA_surf_only: bool=False):
    '''removes subjects with NaNs in either surf or netmats.'''
    if ICA_surf_only is True:
        netmat = np.zeros((surf.shape[0],1)) #forced to not have any nans cause not real in this case
    no_nan_condition=False
    no_inf_condition=False #assume there are inf/nans
    surf_nan=np.isnan(surf); netmat_nan=np.isnan(netmat)
    surf_inf=np.isinf(surf); netmat_inf=np.isinf(netmat)
    if np.sum((surf_nan.sum(), netmat_nan.sum())) == 0:
        write_to_file("No NaNs in data.", filepath=write_fpath)
        no_nan_condition = True
    if np.sum((surf_inf.sum(), netmat_inf.sum())) == 0:
        write_to_file("No INFs in data.", filepath=write_fpath)
        no_inf_condition = True
    if no_nan_condition is True and no_inf_condition is True:
        clean_surf, clean_netmat = surf, netmat

        if subject_list_path is None:
            sample_size = surf.shape[0] #should be first dim
            main_subj_ID_root="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/subj_ids"
            if sample_size > 1000:
                subject_list_path=f"{main_subj_ID_root}/ABCD_train_IDs.txt"
            elif sample_size < 1000 and sample_size > 700:
                subject_list_path=f"{main_subj_ID_root}/ABCD_validation_IDs.txt"
            elif sample_size < 700:
                subject_list_path=f"{main_subj_ID_root}/ABCD_test_IDs.txt"

        subjects_to_keep = pd.read_csv(subject_list_path) #same as what is given
        if bilateral_condition is True:
            subjects_to_keep = pd.concat([subjects_to_keep, subjects_to_keep], ignore_index=True)
        return clean_surf, clean_netmat, subjects_to_keep
    else:
        write_to_file("NaNs or INFs detected. Cleaning...", filepath=write_fpath)
        find_inf_mask_surf=(np.sum(np.sum(np.sum(surf_inf,axis=-1),axis=-1), axis=-1)).astype(bool)
        find_nan_mask_surf=(np.sum(np.sum(np.sum(surf_nan,axis=-1),axis=-1), axis=-1)).astype(bool) #add backwards to get subject sums of NaNs
        full_mask_surf = (find_inf_mask_surf+find_nan_mask_surf)
        write_to_file(f"Total surf mask len and count: {len(full_mask_surf)} {full_mask_surf.sum()}", filepath=write_fpath)
        # same for netmat
        find_inf_mask_netmat=(np.sum(netmat_inf,axis=-1)).astype(bool)
        find_nan_mask_netmat=(np.sum(netmat_nan,axis=-1)).astype(bool) #add backwards to get subject sums of NaNs
        full_mask_netmat = (find_inf_mask_netmat+find_nan_mask_netmat)
        write_to_file(f"Total netmat mask len and count: {len(full_mask_netmat)} {full_mask_netmat.sum()}", filepath=write_fpath)
        full_mask_all = (full_mask_surf+full_mask_netmat)
        clean_netmat = netmat[~full_mask_all]
        clean_surf = surf[~full_mask_all] #only keep does NOT in the mask, so inverse it
        subj_mask = np.where(full_mask_all==0)[0] #false means good here, same as before with ~full_mask_all
        write_to_file(f"Final subject mask of good ones is: {len(subj_mask)} {subj_mask.sum()}", filepath=write_fpath)
        
        #subjects to keep so we can trac which have NaNs/INF for later
        if subject_list_path is None:
            sample_size = surf.shape[0] #should be first dim
            main_subj_ID_root="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/subj_ids"
            
            if bilateral_condition is True:
                if sample_size == 14242: #then bilateral
                    write_to_file("USING TRAIN SUBJECT IDS", filepath=write_fpath)
                    sub_ids1 = pd.read_csv(f"{main_subj_ID_root}/ABCD_train_IDs.txt")
                    sub_ids2 = pd.read_csv(f"{main_subj_ID_root}/ABCD_train_IDs.txt")
                    sub_ids = pd.concat([sub_ids1, sub_ids2], ignore_index=True) #row wise by default
                elif sample_size == 1856:
                    write_to_file("USING VALIDATION SUBJECT IDS", filepath=write_fpath)
                    sub_ids1 = pd.read_csv(f"{main_subj_ID_root}/ABCD_validation_IDs.txt")
                    sub_ids2 = pd.read_csv(f"{main_subj_ID_root}/ABCD_validation_IDs.txt")
                    sub_ids = pd.concat([sub_ids1, sub_ids2], ignore_index=True)
                elif sample_size == 1248:
                    write_to_file("USING VALIDATION SUBJECT IDS", filepath=write_fpath)
                    sub_ids1 = pd.read_csv(f"{main_subj_ID_root}/ABCD_test_IDs.txt")
                    sub_ids2 = pd.read_csv(f"{main_subj_ID_root}/ABCD_test_IDs.txt")
                    sub_ids = pd.concat([sub_ids1, sub_ids2], ignore_index=True)
                else:
                    write_to_file("UNRECOGNIZED SAMPLE SIZE.", filepath=write_fpath)
            else:
                if sample_size > 5000:
                    write_to_file("USING TRAIN SUBJECT IDS", filepath=write_fpath)
                    sub_ids = pd.read_csv(f"{main_subj_ID_root}/ABCD_train_IDs.txt")
                elif sample_size > 800 and sample_size < 1000:
                    write_to_file("USING VALIDATION SUBJECT IDS", filepath=write_fpath)
                    sub_ids = pd.read_csv(f"{main_subj_ID_root}/ABCD_validation_IDs.txt")
                elif sample_size < 700:
                    write_to_file("USING VALIDATION SUBJECT IDS", filepath=write_fpath)
                    sub_ids = pd.read_csv(f"{main_subj_ID_root}/ABCD_test_IDs.txt")
                else:
                    write_to_file("UNRECOGNIZED SAMPLE SIZE.", filepath=write_fpath)
                    
            subject_list = sub_ids["subID"].to_numpy() #needs to be numpy for later indecing based on mask
        else: #if one is given, just use what it is, load it and make it numpy
            sub_ids = pd.read_csv(subject_list_path)
            subject_list = sub_ids["subID"].to_numpy()
        
        subjects_to_keep = pd.DataFrame({
            "subID": subject_list[subj_mask]
            })

    assert np.isnan(clean_surf).sum()==np.isnan(clean_netmat).sum()==0, "Cleanning unsuccessful."

    return clean_surf, clean_netmat, subjects_to_keep

def fcn_prep_data_get_loaders(train_netmat, train_surface, validation_netmat, validation_surface, parcellation_N, netmat_prep_choice=None, surf_prep_choice=None, b_sz: int=32, mvae: bool=False, write_fpath: str='', train_subject_list=None, validation_subject_list=None,bilateral_condition: bool=False):
    '''
    Preprocessing function that takes in netmat parcellation of size N and suface maps for some component, like ICA. For netmats, input data is a subx[(N*(N-1))/2] matrix, 
    so upper triangle elements. parcellation = 100, then N=4950 and so on. Provided a tranformation condition is given, norm or fisherZ, ir applies both or either. Then, makes full connectome
    matrix for all subjects => data loader input for netmat becomes SUBxNxN = SUBx100x100 for example. For surface maps, it takes in their data BxCxPxV and makes into vertex for easier prediction
    and for kraken coder, because craken needs SUBxFEAT sapce to do latent and MSE reconstructions. 
    They all use this so make into a fcn all can call!
    '''
    tr_sub_dim, c_dim, p_dim, v_dim = train_surface.shape
    # write_to_file(f'Regular surface shape:{train_surface.shape}', filepath=write_fpath)

    write_to_file('Cleaning data first if needed! Training data.', filepath=write_fpath)
    train_surface, train_netmat, train_subjects_to_keep = fcn_get_clean_data(surf=train_surface, netmat=train_netmat, subject_list_path=train_subject_list, write_fpath=write_fpath, bilateral_condition=bilateral_condition)
    write_to_file('Cleaning data first if needed! For validation/test now.', filepath=write_fpath)
    validation_surface, validation_netmat, validation_subjects_to_keep = fcn_get_clean_data(surf=validation_surface, netmat=validation_netmat, subject_list_path=validation_subject_list, write_fpath=write_fpath, bilateral_condition=bilateral_condition)

    mean_train_netmat = np.mean(train_netmat, axis=0)
    if surf_prep_choice=="norm":
        sigma_train_netmat = np.std(train_netmat, axis=0)   
        mean_train_surface = np.nanmean(train_surface, axis=0, keepdims=True) #1x15x320x153
        sigma_train_surface = np.nanstd(train_surface, axis=0, keepdims=True) #1x15x320x153
        normalized_train_surface = (train_surface - mean_train_surface) / (sigma_train_surface + 10e-99)
        # val_num_sub, _, _, _ = validation_surface.shape
        normalized_val_surface = (validation_surface - mean_train_surface) / (sigma_train_surface  + 10e-99)
    else:
        write_to_file('ALERT!!! USING RAW ICA MAPS', filepath=write_fpath)
        normalized_train_surface = train_surface
        normalized_val_surface = validation_surface

    if netmat_prep_choice == "norm_fisherz":
        write_to_file('NetMat prep chosen is both FISHERZ and NORM ', filepath=write_fpath)
        train_netmat_fz = fisher_z_transform(train_netmat) #fisheZ transform first
        validation_netmat_fz = fisher_z_transform(validation_netmat)

        mean_train_netmat_fz = np.mean(train_netmat_fz, axis=0) #norm based on training data
        sigma_train_netmat_fz = np.std(train_netmat_fz, axis=0)

        tr_transformed_netmats = (train_netmat_fz - mean_train_netmat_fz) / (sigma_train_netmat_fz  + 10e-99)
        val_transformed_netmats = (validation_netmat_fz - mean_train_netmat_fz) / (sigma_train_netmat_fz  + 10e-99)
        
    elif netmat_prep_choice == "demean_fisherz":
        write_to_file('NetMat prep chosen is both FISHERZ and DEMEAN ', filepath=write_fpath)
        train_netmat_fz = fisher_z_transform(train_netmat) #fisheZ transform first
        validation_netmat_fz = fisher_z_transform(validation_netmat)

        mean_train_netmat_fz = np.mean(train_netmat_fz, axis=0) #norm based on training data
        sigma_train_netmat_fz = np.std(train_netmat_fz, axis=0)

        tr_transformed_netmats = (train_netmat_fz - mean_train_netmat_fz) #/ (sigma_train_netmat_fz  + 10e-99)
        val_transformed_netmats = (validation_netmat_fz - mean_train_netmat_fz)# / (sigma_train_netmat_fz  + 10e-99)

    elif netmat_prep_choice == "norm":
        write_to_file('NetMat prep chosen is NORM ', filepath=write_fpath)
        tr_transformed_netmats = (train_netmat - mean_train_netmat)/ (sigma_train_netmat  + 10e-99) # normalize r values?
        val_transformed_netmats = (validation_netmat - mean_train_netmat)/ (sigma_train_netmat  + 10e-99)

    elif netmat_prep_choice == "fisherz":
        write_to_file('NetMat prep chosen is FISHERZ ', filepath=write_fpath)
        tr_transformed_netmats = fisher_z_transform(train_netmat)
        val_transformed_netmats = fisher_z_transform(validation_netmat)
    
    elif netmat_prep_choice == "demean":
        write_to_file('NetMat prep chosen is DEMEAN ', filepath=write_fpath)
        tr_transformed_netmats = (train_netmat - mean_train_netmat)
        val_transformed_netmats = (validation_netmat - mean_train_netmat)
    else:
        tr_transformed_netmats = train_netmat
        val_transformed_netmats = validation_netmat
    
    if mvae:
        train_netmat_np = make_nemat_allsubj(tr_transformed_netmats, parcellation_N) # turns vec into netmat for all subs, second variable is nodes in netmat
        val_netmat_np = make_nemat_allsubj(val_transformed_netmats, parcellation_N)
    else:
        train_netmat_np = tr_transformed_netmats
        val_netmat_np = val_transformed_netmats

    write_to_file(f'Netmat Shape:{train_netmat_np.shape} \nSurface Shape: {train_surface.shape}', filepath=write_fpath)

    #### MODEL DATALOADERS
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized_train_surface).float(), torch.from_numpy((train_netmat_np)).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = b_sz, shuffle=True, num_workers=10)
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized_val_surface).float(), torch.from_numpy((val_netmat_np)).float())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = b_sz, shuffle=True, num_workers=10)

    return train_loader, val_loader, mean_train_netmat, train_subjects_to_keep, validation_subjects_to_keep

def fcn_prep_swintrans_data_get_loaders(train_netmat, train_surface, validation_netmat, validation_surface, parcellation_N, netmat_prep_choice=None, b_sz=32, padding=50, encdec=True, write_fpath=''):
    '''
    Preprocessing function that takes in netmat parcellation of size N and suface maps for some component, like ICA. For netmats, input data is a subx[(N*(N-1))/2] matrix, 
    so upper triangle elements. parcellation = 100, then N=4950 and so on. Provided a tranformation condition is given, norm or fisherZ, ir applies both or either. Then, makes full connectome
    matrix for all subjects => data loader input for netmat becomes SUBxNxN = SUBx100x100 for example. For surface maps, it takes in their data BxCxPxV and makes into vertex for easier prediction
    and for kraken coder, because craken needs SUBxFEAT sapce to do latent and MSE reconstructions. 
    They all use this so make into a fcn all can call!
    '''
    tr_sub_dim, p_dim, c_dim, v_dim = train_surface.shape
    write_to_file(f'regular surface shape{train_surface.shape}', filepath=write_fpath)

    # upper_tri_sz = train_netmat_np.shape[1] # should be SUBx4950 (or node size upper tri count)
    mean_train_netmat = np.mean(train_netmat, axis=0)
    sigma_train_netmat = np.std(train_netmat, axis=0)   
    mean_train_surface = np.mean(train_surface, axis=0, keepdims=True) #1xCxV
    sigma_train_surface = np.std(train_surface, axis=0, keepdims=True) #1xCxV
    normalized_train_surface = (train_surface - mean_train_surface) / (sigma_train_surface + 10e-99)

    val_num_sub,_, _, _ = validation_surface.shape
    normalized_val_surface = (validation_surface - mean_train_surface) / (sigma_train_surface  + 10e-99) # adding small epsillon for stability cause ow we get NaN
    del mean_train_surface, sigma_train_surface # large so deleting to save mem we only want the norm versions anyway

    if netmat_prep_choice == "norm_fisherz":
        write_to_file('NetMat prep chosen is both FISHERZ and NORM ', filepath=write_fpath)
        train_netmat_fz = fisher_z_transform(train_netmat) #fisheZ transform first
        validation_netmat_fz = fisher_z_transform(validation_netmat)

        mean_train_netmat_fz = np.mean(train_netmat_fz, axis=0) #norm based on training data
        sigma_train_netmat_fz = np.std(train_netmat_fz, axis=0)

        tr_transformed_netmats = (train_netmat_fz - mean_train_netmat_fz) / (sigma_train_netmat_fz  + 10e-99)
        val_transformed_netmats = (validation_netmat_fz - mean_train_netmat_fz) / (sigma_train_netmat_fz  + 10e-99)
        del train_netmat, validation_netmat, train_netmat_fz, validation_netmat_fz
    
    elif netmat_prep_choice == "demean_fisherz":
        write_to_file('NetMat prep chosen is both FISHERZ and DEMEAN ', filepath=write_fpath)
        train_netmat_fz = fisher_z_transform(train_netmat) #fisheZ transform first
        validation_netmat_fz = fisher_z_transform(validation_netmat)

        mean_train_netmat_fz = np.mean(train_netmat_fz, axis=0) #norm based on training data
        sigma_train_netmat_fz = np.std(train_netmat_fz, axis=0)

        tr_transformed_netmats = (train_netmat_fz - mean_train_netmat_fz) #/ (sigma_train_netmat_fz  + 10e-99)
        val_transformed_netmats = (validation_netmat_fz - mean_train_netmat_fz)# / (sigma_train_netmat_fz  + 10e-99)
        del train_netmat, validation_netmat, train_netmat_fz, validation_netmat_fz
    
    elif netmat_prep_choice == "norm":
        write_to_file('NetMat prep chosen is NORM ', filepath=write_fpath)
        tr_transformed_netmats = (train_netmat - mean_train_netmat)/ (sigma_train_netmat  + 10e-99) # normalize r values?
        val_transformed_netmats = (validation_netmat - mean_train_netmat)/ (sigma_train_netmat  + 10e-99)
        del train_netmat, validation_netmat#, train_netmat_fz, validation_netmat_fz

    elif netmat_prep_choice == "fisherz":
        write_to_file('NetMat prep chosen is FISHERZ ', filepath=write_fpath)
        tr_transformed_netmats = fisher_z_transform(train_netmat)
        val_transformed_netmats = fisher_z_transform(validation_netmat)
        del train_netmat, validation_netmat#, train_netmat_fz, validation_netmat_fz

    elif netmat_prep_choice == "demean":
        write_to_file('NetMat prep chosen is DEMEAN ', filepath=write_fpath)
        tr_transformed_netmats = (train_netmat - mean_train_netmat)
        val_transformed_netmats = (validation_netmat - mean_train_netmat)
        del train_netmat, validation_netmat#, train_netmat_fz, validation_netmat_fz
    else:
        tr_transformed_netmats = train_netmat
        val_transformed_netmats = validation_netmat
    
    if encdec:
        train_netmat_np = add_start_token_np(tr_transformed_netmats, n=padding) #make_nemat_allsubj(tr_transformed_netmats, parcellation_N) # turns vec into netmat for all subs, second variable is nodes in netmat
        val_netmat_np = add_start_token_np(val_transformed_netmats, n=padding) #make_nemat_allsubj(val_transformed_netmats, parcellation_N)
    else:
        train_netmat_np = tr_transformed_netmats
        val_netmat_np = val_transformed_netmats

    write_to_file(f'Netmat Shape:{train_netmat_np.shape} \nSurface Shape: {train_surface.shape}', filepath=write_fpath)

    #### MODEL DATALOADERS
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized_train_surface).float(), torch.from_numpy((train_netmat_np)).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = b_sz, shuffle=True, num_workers=10)
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized_val_surface).float(), torch.from_numpy((val_netmat_np)).float())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = b_sz, shuffle=True, num_workers=10)

    return train_loader, val_loader, mean_train_netmat

def fcn_prep_data_get_loaders_forLINEAR(train_netmat, train_surface, validation_netmat, validation_surface, parcellation_N, netmat_prep_choice=None, surf_prep_choice=None, b_sz=32, padding=50, encdec=True, write_fpath=''):
    '''
    Preprocessing function that takes in netmat parcellation of size N and suface maps for some component, like ICA. For netmats, input data is a subx[(N*(N-1))/2] matrix, 
    so upper triangle elements. parcellation = 100, then N=4950 and so on. Provided a tranformation condition is given, norm or fisherZ, ir applies both or either. Then, makes full connectome
    matrix for all subjects => data loader input for netmat becomes SUBxNxN = SUBx100x100 for example. For surface maps, it takes in their data BxCxPxV and makes into vertex for easier prediction
    and for kraken coder, because craken needs SUBxFEAT sapce to do latent and MSE reconstructions. 
    They all use this so make into a fcn all can call!
    '''
    tr_sub_dim, c_dim, p_dim, v_dim = train_surface.shape
    val_num_sub, _, _, _ = validation_surface.shape
    write_to_file(f'regular surface shpae{train_surface.shape}', filepath=write_fpath)
   
    if surf_prep_choice == "norm":
        mean_train_surface = np.nanmean(train_surface, axis=0, keepdims=True) #1x15x320x153
        sigma_train_surface = np.nanstd(train_surface, axis=0, keepdims=True) #1x15x320x153

        normalized_train_surface = (train_surface - mean_train_surface) / (sigma_train_surface + 10e-99)
        normalized_val_surface = (validation_surface - mean_train_surface) / (sigma_train_surface  + 10e-99)
    elif surf_prep_choice == "norm_linear_enc":
        mean_train_surface = np.nanmean(train_surface, axis=0, keepdims=True) #1x15x320x153
        sigma_train_surface = np.nanstd(train_surface, axis=0, keepdims=True) #1x15x320x153

        normalized_train_surface = (train_surface - mean_train_surface) / (sigma_train_surface + 10e-99)
        normalized_val_surface = (validation_surface - mean_train_surface) / (sigma_train_surface  + 10e-99)
        # reshape to BxFEATURES
        normalized_train_surface = normalized_train_surface.reshape(tr_sub_dim, c_dim*p_dim*v_dim)
        normalized_val_surface = normalized_val_surface.reshape(val_num_sub, c_dim*p_dim*v_dim)
    elif surf_prep_choice == "linear_enc":
        # reshape to BxFEATURES
        normalized_train_surface = train_surface.reshape(tr_sub_dim, c_dim*p_dim*v_dim)
        normalized_val_surface = validation_surface.reshape(val_num_sub, c_dim*p_dim*v_dim)
    elif surf_prep_choice == "norm_linear_PCA":
        from sklearn.decomposition import PCA
        write_to_file('Prepping surfmaps with norm PCA reduction!!!', filepath=write_fpath)
        # reshape to BxFEATURES
        mean_train_surface = np.nanmean(train_surface, axis=0, keepdims=True) #1x15x320x153
        sigma_train_surface = np.nanstd(train_surface, axis=0, keepdims=True) #1x15x320x153

        normalized_train_surface = (train_surface - mean_train_surface) / (sigma_train_surface + 10e-99)
        normalized_val_surface = (validation_surface - mean_train_surface) / (sigma_train_surface  + 10e-99)
        # reshape into vectors
        normalized_train_surface = normalized_train_surface.reshape(tr_sub_dim, c_dim*p_dim*v_dim)
        normalized_val_surface = normalized_val_surface.reshape(val_num_sub, c_dim*p_dim*v_dim)
        # PCA dim reduction
        pca_components = int(600) #normalized_train_surface.shape[0]
        pca = PCA(n_components=pca_components) #reduces surfmap 734k --> ABCD/HCPYA N size | 7k/760
        normalized_train_surface = pca.fit_transform(normalized_train_surface) #NxN matrix because originally Nx734k but reduced dimensionality
        pca = PCA(n_components=pca_components)
        normalized_val_surface = pca.fit_transform(normalized_val_surface)
    else:
        normalized_train_surface = train_surface
        normalized_val_surface = validation_surface

    if netmat_prep_choice == "norm_fisherz":
        mean_train_netmat = np.mean(train_netmat, axis=0)
        sigma_train_netmat = np.std(train_netmat, axis=0)   
        write_to_file('NetMat prep chosen is both FISHERZ and NORM ', filepath=write_fpath)
        train_netmat_fz = fisher_z_transform(train_netmat) #fisheZ transform first
        validation_netmat_fz = fisher_z_transform(validation_netmat)

        mean_train_netmat_fz = np.mean(train_netmat_fz, axis=0) #norm based on training data
        sigma_train_netmat_fz = np.std(train_netmat_fz, axis=0)

        tr_transformed_netmats = (train_netmat_fz - mean_train_netmat_fz) / (sigma_train_netmat_fz  + 10e-99)
        val_transformed_netmats = (validation_netmat_fz - mean_train_netmat_fz) / (sigma_train_netmat_fz  + 10e-99)
        
    elif netmat_prep_choice == "demean_fisherz":
        mean_train_netmat = np.mean(train_netmat, axis=0)
        write_to_file('NetMat prep chosen is both FISHERZ and DEMEAN ', filepath=write_fpath)
        train_netmat_fz = fisher_z_transform(train_netmat) #fisheZ transform first
        validation_netmat_fz = fisher_z_transform(validation_netmat)

        mean_train_netmat_fz = np.mean(train_netmat_fz, axis=0) #norm based on training data
        sigma_train_netmat_fz = np.std(train_netmat_fz, axis=0)

        tr_transformed_netmats = (train_netmat_fz - mean_train_netmat_fz) #/ (sigma_train_netmat_fz  + 10e-99)
        val_transformed_netmats = (validation_netmat_fz - mean_train_netmat_fz)# / (sigma_train_netmat_fz  + 10e-99)

    elif netmat_prep_choice == "norm":
        mean_train_netmat = np.mean(train_netmat, axis=0)
        sigma_train_netmat = np.std(train_netmat, axis=0)
        write_to_file('NetMat prep chosen is NORM ', filepath=write_fpath)
        tr_transformed_netmats = (train_netmat - mean_train_netmat)/ (sigma_train_netmat  + 10e-99) # normalize r values?
        val_transformed_netmats = (validation_netmat - mean_train_netmat)/ (sigma_train_netmat  + 10e-99)

    elif netmat_prep_choice == "fisherz":
        write_to_file('NetMat prep chosen is FISHERZ ', filepath=write_fpath)
        tr_transformed_netmats = fisher_z_transform(train_netmat)
        val_transformed_netmats = fisher_z_transform(validation_netmat)
    
    elif netmat_prep_choice == "demean":
        mean_train_netmat = np.mean(train_netmat, axis=0)
        write_to_file('NetMat prep chosen is DEMEAN ', filepath=write_fpath)
        tr_transformed_netmats = (train_netmat - mean_train_netmat)
        val_transformed_netmats = (validation_netmat - mean_train_netmat)
    else:
        tr_transformed_netmats = train_netmat
        val_transformed_netmats = validation_netmat
    
    if encdec:
        train_netmat_np = add_start_token_np(tr_transformed_netmats, n=padding) #make_nemat_allsubj(tr_transformed_netmats, parcellation_N) # turns vec into netmat for all subs, second variable is nodes in netmat
        val_netmat_np = add_start_token_np(val_transformed_netmats, n=padding) #make_nemat_allsubj(val_transformed_netmats, parcellation_N)
    else:
        train_netmat_np = tr_transformed_netmats
        val_netmat_np = val_transformed_netmats

    write_to_file(f'Netmat Shape:{train_netmat_np.shape} \nSurface Shape: {train_surface.shape}', filepath=write_fpath)

    #### MODEL DATALOADERS
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized_train_surface).float(), torch.from_numpy((train_netmat_np)).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = b_sz, shuffle=True, num_workers=10)
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized_val_surface).float(), torch.from_numpy((val_netmat_np)).float())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = b_sz, shuffle=True, num_workers=10)

    return train_loader, val_loader, mean_train_netmat

def get_lower_tris(mat):
    trils = []
    if mat.ndim == 3: # then BatchxROWxCOL
        for i in range(mat.shape[0]):
            tril = mat[i, :, :][np.tril_indices_from(mat[i, :, :], k=-1)]
            trils.append(tril)
    elif mat.ndim == 2: # so a single matrix
        tril = mat[np.tril_indices_from(mat, k=-1)] #-1 means lower trinagle
        trils.append(tril)
    return np.array(trils)

def get_upper_tris(mat):
    trius = []
    if mat.ndim == 3: # then BatchxROWxCOL
        for i in range(mat.shape[0]):
            triu = mat[i, :, :][np.triu_indices_from(mat[i, :, :], k=1)]
            trius.append(triu)
    elif mat.ndim == 2: # so a single matrix
        triu = mat[np.triu_indices_from(mat, k=1)] #-1 means lower trinagle
        trius.append(triu)
    return np.array(trius)

def add_start_token_torch(tensor, n=1, start_value=1):
    """
    Add a new column with a start value to the beginning of each sequence in the input tensor.
    
    :param tensor: Tensor of shape (batch_size, seq_length), input tensor
    :param start_value: int, value to add at the start of each sequence
    :return: Tensor of shape (batch_size, seq_length + 1), tensor with a new column added to the start of each sequence
    """
    batch_size, seq_length = tensor.size()
    new_column = torch.full((batch_size, n), start_value, dtype=tensor.dtype, device=tensor.device)  # Create a new column with the start value
    out = torch.cat([new_column, tensor], dim=1)  # Concatenate the new column with the input tensor
    return out

def add_start_token_np(array, n=1, start_value=1):
    """
    Add a new column with a start value to the beginning of each sequence in the input array.
    
    :param array: Array of shape (batch_size, seq_length), input array
    :param start_value: int, value to add at the start of each sequence
    :return: Array of shape (batch_size, seq_length + 1), array with a new column added to the start of each sequence
    """
    batch_size, seq_length = array.shape
    new_column = np.full((batch_size, n), start_value, dtype=array.dtype)  # Create a new column with the start value
    out = np.concatenate((new_column, array), axis=1)  # Concatenate the new column with the input array
    return out


# loading pretrained transformer ViT on imagenet1K
def load_weights_imagenet(state_dict,state_dict_imagenet,nb_layers):

    state_dict['mlp_head.0.weight'] = state_dict_imagenet['norm.weight'].data
    state_dict['mlp_head.0.bias'] = state_dict_imagenet['norm.bias'].data

    # transformer blocks
    for i in range(nb_layers):
        state_dict['transformer.layers.{}.0.norm.weight'.format(i)] = state_dict_imagenet['blocks.{}.norm1.weight'.format(i)].data
        state_dict['transformer.layers.{}.0.norm.bias'.format(i)] = state_dict_imagenet['blocks.{}.norm1.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.norm.weight'.format(i)] = state_dict_imagenet['blocks.{}.norm2.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.norm.bias'.format(i)] = state_dict_imagenet['blocks.{}.norm2.bias'.format(i)].data

        state_dict['transformer.layers.{}.0.fn.to_qkv.weight'.format(i)] = state_dict_imagenet['blocks.{}.attn.qkv.weight'.format(i)].data

        state_dict['transformer.layers.{}.0.fn.to_out.0.weight'.format(i)] = state_dict_imagenet['blocks.{}.attn.proj.weight'.format(i)].data
        state_dict['transformer.layers.{}.0.fn.to_out.0.bias'.format(i)] = state_dict_imagenet['blocks.{}.attn.proj.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.fn.net.0.weight'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc1.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.fn.net.0.bias'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc1.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.fn.net.3.weight'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc2.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.fn.net.3.bias'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc2.bias'.format(i)].data

    return state_dict
