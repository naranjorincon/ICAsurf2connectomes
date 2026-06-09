# %%
'''
To run this shole jupyter notebook as a python script follow this:

(1) activate conda environemt
(2) go to where this notebook is located in your computer
(3) use `python` to enter python with in your shell/terminal
(4) follow the above syntax

from json import load
filename='step_2_prep_maps_and_netmats.ipynb'
with open(filename) as fp:
    nb = load(fp)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(line for line in cell['source'] if not line.startswith('%'))
        exec(source, globals(), locals())
'''

def main(config):
    if "/Users/snaranjo" in os.getcwd():
        local_pc_flag = "/Users/snaranjo/Desktop/neurotranslate/mount_point"
    else:
        local_pc_flag=""

    ico = config['resolution']['ico']
    sub_ico = config['resolution']['sub_ico']
    dataset = config['data']['dataset']
    parcellation_name=config['data']['parcellation_name']
    # translation = 'INFOMAPd20_schfd100' #'ICAd15_schfd100'
    parcellation_size=config['data']['parcellation_nodes']
    parcellation_type='full'
    sub_ids_path=config['data']['sub_ids_path']
    netmats_paths=config['data']['netmats_paths']
    patch_indeces_path=config['data']['patch_indeces_path']
    chosen_hemi=config['data']['hemisphere'] #1L or 1R or 2 for both
    num_vertices=config['sub_ico_{}'.format(sub_ico)]['num_vertices'] # sub_ico_2
    num_patches=config['sub_ico_{}'.format(sub_ico)]['num_patches'] # sub_ico_2
    path_to_data=config['data']['path_to_data']
    path_to_subject_lists=config['data']['path_to_subject_lists']
    skipped_subject_path=config['data']['skipped_subject_path']
    output_maps_netmats_path=config['data']['output_maps_netmats_path']

    # helper functions
    def fcn_generate_subject_split(single_df: pd.DataFrame, twins_df: pd.DataFrame, triplets_df: pd.DataFrame, train_split_portion:float=0.8, random_seed: int=123):
        assert train_split_portion < 1 and train_split_portion > 0, "Train split must be 0<x<1."

        '''Goal here is to have a train/val/test split where subject relationship is respected.
        If a twin/triplet is in a split, the other siblings must also be in that split.'''
        total_N = single_df.shape[0] + twins_df.shape[0] + triplets_df.shape[0]
        import math
        train_N= math.floor(total_N * train_split_portion) #if its even we're good
        # added_part_cond = 1 if train_N % 2 != 0 else 0
        left_over = total_N-train_N
        validation_N = left_over // 2
        test_N = left_over - validation_N
        print(f"Based on train_split_proportion ({train_split_portion}): \nTrain:{train_N} \nValidation:{validation_N} \nTest:{test_N}")
        #ensure that these are ordered by family ID where same ID means TWINS/TRIPLETS
        name_of_famID_column="ab_g_stc__design_id__fam__gen"
        name_of_subID_column="participant_id"
        single_df=single_df.sort_values(by=[name_of_famID_column]).reset_index(drop=True)
        twins_df=twins_df.sort_values(by=[name_of_famID_column]).reset_index(drop=True)
        triplets_df=triplets_df.sort_values(by=[name_of_famID_column]).reset_index(drop=True)
        # now that sorted we can split and use that they are ordered for our purposes
        family_to_subjects: dict[str, list] = {} #init dictionary
        for df in [single_df, twins_df, triplets_df]: 
            for fam_id, group in df.groupby(name_of_famID_column): #organize by group 
                subjects = group[name_of_subID_column].to_list() #group size depends on fam, so singletons==1,twins==2,triplets==3
                if fam_id in family_to_subjects:
                    family_to_subjects[fam_id].extend(subjects)
                else:
                    family_to_subjects[fam_id] = subjects #every fam_id is a key and corresponding list of subjects for that key
        
        #shuffle family
        family_ids = list(family_to_subjects.keys())
        pre_family_ids=family_ids
        from random import shuffle
        shuffle(family_ids)
        post_family_ids=family_ids
        # print(f"Post-Shuffle: {post_family_ids}")
        assert pre_family_ids != post_family_ids, "Shuffle did not work for some reason. Check this."

        # assign families to splits
        train_ids, validation_ids, test_ids = [],[],[]
        train_count, validation_count, test_count = 0,0,0

        for fam_id in family_ids:
            subjects = family_to_subjects[fam_id] #list of subjects for that famID key
            family_size = len(subjects)

            #starting with train, fill the train/validation/test splits as needed
            if train_count < train_N:
                train_ids.extend(subjects) #assign that list of subjects to train
                train_count += family_size
            elif validation_count < validation_N:
                validation_ids.extend(subjects)
                validation_count += family_size
            else:
                test_ids.extend(subjects)
                test_count += family_size
        
        #report and make sure no overlap in assignment
        print(
            f"Subject counts after family-aware assignment:\n"
            f"Train:{len(train_ids)} \nValidation:{len(validation_ids)} \Test:{len(test_ids)}\n"
            f"Total:{len(train_ids) + len(validation_ids) + len(test_ids)}"
        )
        all_assigned   = train_ids + validation_ids + test_ids
        all_subjects   = (
            single_df[name_of_subID_column].tolist()
            + twins_df[name_of_subID_column].tolist()
            + triplets_df[name_of_subID_column].tolist()
        )
        assert set(all_assigned) == set(all_subjects), \
            "Mismatch: some subjects were lost or duplicated during splitting!"
        assert len(set(train_ids) & set(validation_ids)) == 0,   "Overlap between train and val!"
        assert len(set(train_ids) & set(test_ids)) == 0,  "Overlap between train and test!"
        assert len(set(validation_ids)   & set(test_ids)) == 0,  "Overlap between val and test!"

        return train_ids, validation_ids, test_ids

    # %%
    if dataset == "ABCD_v6":
        get_subject_list = pd.read_csv(f"{local_pc_flag}/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/qc/individual_subjects/individual_subject_list.txt", sep=' ', header=None)[0].values.tolist()
        print(f"Number of subjects: {len(get_subject_list)}, {get_subject_list[-5:-1]}")

        fam_id_files=pd.read_csv(f"{local_pc_flag}/ceph/chpc/rcif_datasets/abcd/ABCD-6.0/phenotypes/ab/ab_g_stc.tsv", sep='\t')
        get_relevant_columns = fam_id_files[["participant_id", "ab_g_stc__design_id__fam__gen","ab_g_stc__design_id__birth__gen", "ab_g_stc__design_id__group"]]

        print(get_relevant_columns.shape)
        mask_to_match_original_subIDlist = np.isin(get_relevant_columns["participant_id"], get_subject_list)
        print(mask_to_match_original_subIDlist.shape, mask_to_match_original_subIDlist.sum())

        get_only_subjlist_match_subjs=get_relevant_columns[mask_to_match_original_subIDlist]
        get_only_subjlist_match_subjs.sort_values(by=["ab_g_stc__design_id__fam__gen"]).dropna() #order by family id to match singletons and twins/triplets+
        assert get_only_subjlist_match_subjs.shape[0] == len(get_subject_list)
        # print(get_only_subjlist_match_subjs.head())

        counts = get_only_subjlist_match_subjs["ab_g_stc__design_id__fam__gen"].value_counts() #each value and its frequency
        single_counts = counts[counts == 1].index #index of where this is true
        two_counts = counts[counts == 2].index
        three_counts = counts[counts == 3].index
        more_counts = counts[counts > 3].index
        print(f"Only once:{len(single_counts)}, Twice:{len(two_counts)}, Thrice:{len(three_counts)}, More:{len(more_counts)}")
        singletons = get_only_subjlist_match_subjs[get_only_subjlist_match_subjs["ab_g_stc__design_id__fam__gen"].isin(single_counts)]
        print(three_counts)
        # print(get_only_subjlist_match_subjs["ab_g_stc__design_id__fam__gen"])
        twins = get_only_subjlist_match_subjs[get_only_subjlist_match_subjs["ab_g_stc__design_id__fam__gen"].isin(two_counts)]
        triplets = get_only_subjlist_match_subjs[get_only_subjlist_match_subjs["ab_g_stc__design_id__fam__gen"].isin(three_counts)]

        #now make train/val/test split respecting this
        subj_list_singles=singletons["participant_id"].to_list()
        subj_list_twins=twins["participant_id"].to_list()
        subj_list_triplets=triplets["participant_id"].to_list()

        #order them and save
        subject_list_subset=subj_list_twins
        sibling_type="twins"

        from_main_df_get_subsets_mask = np.isin(get_only_subjlist_match_subjs["participant_id"], subject_list_subset)
        order_them = get_only_subjlist_match_subjs[from_main_df_get_subsets_mask]
        order_them=order_them.sort_values(by="ab_g_stc__design_id__fam__gen")
        directory=f"{local_pc_flag}/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/subj_ids"

        flag_seperate_by_famID=False
        if flag_seperate_by_famID:
            order_them.to_csv(f"{directory}/ABCDv6_{sibling_type}.csv")
        # print(order_them)

        generate_new_subject_list=False
        if generate_new_subject_list:
            single_df = pd.read_csv(f"{directory}/ABCDv6_singles.csv")
            twins_df = pd.read_csv(f"{directory}/ABCDv6_twins.csv")
            triplets_df = pd.read_csv(f"{directory}/ABCDv6_triplets.csv")
            total_N = single_df.shape[0] + twins_df.shape[0] + triplets_df.shape[0]
            # print(single_df.shape, twins_df.shape, triplets_df.shape,total_N)

            train_ids, validation_ids, test_ids = fcn_generate_subject_split(single_df, twins_df, triplets_df)
            #save generated subject lists for train/validation/test split. Then, begin preprocessing of those participants
            ABCDv6_split_ids = pd.DataFrame({
                "participant_ids": np.concatenate((train_ids,validation_ids,test_ids)),
                "assignment": ["train"]*len(train_ids) + ["validation"]*len(validation_ids) + ["test"]*len(test_ids)
            })
            from datetime import date
            todays_date = date.today()
            ABCDv6_split_ids.to_csv(f"{directory}/ABCDv6_split_ids_{todays_date}.csv")

    # %%
    print('#'*30)
    print('Starting: preprocessing script')
    print('#'*30)

    # Get meshes and paired netmats
    ids = pd.read_csv(f"{sub_ids_path}/neurotranslate_abcd_subject_list.txt", sep=' ', header=None)[0].values.tolist() #reads all subjects, should be the train/val/test split.
    print(f"Number of subjects for prep is {len(ids)}")
    # ids = ids[:2]

    if parcellation_name == "schaefer":
        if parcellation_type == "full":
            path_to_netmat_specific=f"{netmats_paths}/schaefer{parcellation_size}/netmats"
        elif parcellation_type == "partial":
            path_to_netmat_specific=f"{netmats_paths}/schaefer{parcellation_size}/partial_netmats"

    elif parcellation_name == "glasser":
        if parcellation_type == "full":
            path_to_netmat_specific=f"{netmats_paths}/glasser360/netmats"
        elif parcellation_type == "partial":
            path_to_netmat_specific=f"{netmats_paths}/glasser360/partial_netmats"

    #making sure everyone has a mesh is most important so lets do that then pair them with ther respective netmats
    data = [] # list of numpy arrays each is a numpy array version of the shape.gii info
    netmat_data_list=[]
    print(f'Dataset is {dataset}')
    subject_list_skipped=[]
    netmat_subject_list_skipped=[]
    # usable_IDs_total=[]
    usable_mesh_IDs=[]
    # usable_netmat_IDs=[]
    if chosen_hemi == '2LR':
        print("Not ready yet. Needs fixing. Will raise error")
        raise ValueError('Error from chosen hemisphere.')
    
    # indeces for netmats later, so no need to compute everytime. Should be based on parcellation size and only upper triangle
    lr, lc = np.triu_indices(parcellation_size, k=1) #only upper triangle, ignore diagonal
    for i, id in enumerate(ids): # reads in actual id num with 'id' inside the pandas column from the read csv, see above ids variable
        if chosen_hemi == '1L':
            print('LEFT hemisphere was chosen.')
            hemisphere_chosen='L'
        elif chosen_hemi == '1R':
            print('RIGHT hemisphere was chosen.')
            hemisphere_chosen='R'

        filename=f'{path_to_data}/resamp_sub-{id}.{hemisphere_chosen}.shape.gii'
        if not os.path.isfile(filename):
            print(f"sub {id}/{i} does not have mesh file.")
            subject_list_skipped.append(i)
            get_mesh_data=np.zeros((20,40962)) #nothing tmp element
        else:
            get_mesh_data=nb.load(filename).agg_data()
            usable_mesh_IDs.append(id)
            print(f"mesh shape: {np.array(get_mesh_data).shape}")
            # print(f'sub {id}/{i} doesnt have mesh, skipping.')
            # subject_list_skipped.append(i)
            # passed so loaded well and can be part of list
        data.append(np.array(get_mesh_data))
        # same for netmats         
        filename=f"{path_to_netmat_specific}/NDARINV{id}.csv"
        if not os.path.isfile(filename):
            print(f'sub {id}/{i} doesnt have netmat, skipping.')
            netmat_subject_list_skipped.append(i)
            vec_netmat=np.zeros((4950)).squeeze()
        else:
            get_sub_netmat=pd.read_csv(filename, header=None).to_numpy()
            # passed so loaded well and can be part of list
            vec_netmat = get_sub_netmat[lr,lc]
            print(f"netmatshape triangle: {vec_netmat.shape}")
            # usable_netmat_IDs.append(id)
        # save data
        netmat_data_list.append(vec_netmat)

        if i%500==0: #every 300 subjects
            print(f'Done matching mesh with netmat for: {id},{i}')
            # from sanity checks, I see now that our data values are dim C x TS, our in our case inputdim x TS for each vertex in the sphere

    assert len(data) == len(netmat_data_list), "Neet to make sure that same amount of surface data and netmats! Soemthing went wrong here."

    # save list of subject skipped in case needed for the future
    # now that we have data for all subjects
    subjects_to_remove=subject_list_skipped+netmat_subject_list_skipped
    # usable_IDs_total=usable_mesh_IDs#+usable_netmat_IDs
    data=np.asarray(data)
    data = np.delete(data, subjects_to_remove, axis=0)
    assert len(usable_mesh_IDs) == data.shape[0], "Something went wrong in saving usable subjects and mesh data. Should be equal."
    netmat_data=np.asarray(netmat_data_list)
    netmat_data = np.delete(netmat_data, subjects_to_remove, axis=0) #subjects with no mesh data
    assert len(data) == len(netmat_data),"Mesh data and netmat data must match (same subject). Some error occurred."
    print(f"Mesh data ready shape is {data.shape} and netmats are {netmat_data.shape}")
    if len(subjects_to_remove) > 0:
        subjects_skipped_nomesh = pd.DataFrame({
            "subIDs_remove": subjects_to_remove,
            "missing_type": "mesh"*len(subject_list_skipped) + "netmat"*len(netmat_subject_list_skipped)
        })
        subjects_skipped_nomesh.to_csv(f"{skipped_subject_path}/subjects_skipped_nomesh_nomat_hemi{hemisphere_chosen}.csv")


    # %%
    def fcn_split_datas(data: np.ndarray, netmat_data: np.ndarray, path_to_subject_lists: str, subject_ids: list):
        train_ids = pd.read_csv(f"{path_to_subject_lists}/ABCD_train_IDs.txt")
        validation_ids = pd.read_csv(f"{path_to_subject_lists}/ABCD_validation_IDs.txt")
        test_ids = pd.read_csv(f"{path_to_subject_lists}/ABCD_test_IDs.txt")
        # get matches with match masks
        train_match = np.isin(subject_ids, train_ids)
        validation_match = np.isin(subject_ids, validation_ids)
        test_match = np.isin(subject_ids, test_ids) #TRUE where test subjects are

        # index with above masks for ease
        data_train=data[train_match]
        data_validation=data[validation_match]
        data_test=data[test_match]
        netmat_data_train=netmat_data[train_match]
        netmat_data_validation=netmat_data[validation_match]
        netmat_data_test=netmat_data[test_match]
        return data_train, data_validation, data_test, netmat_data_train, netmat_data_validation, netmat_data_test

    indices_mesh_triangles=pd.read_csv(f'{patch_indeces_path}/triangle_indices_ico_{ico}_sub_ico_{sub_ico}.csv')
    num_subjects, num_channels = data.shape[0], data.shape[1]
    if chosen_hemi == '2LR':
        print("Not ready yet.")
    else:
        print('\nBecause one hemisphere chosen, data is num_subj C P V')
        data_ico_lowres = np.zeros((num_subjects, num_channels, num_patches, num_vertices))
        print(f'ICO-{sub_ico} data shape: {data_ico_lowres.shape}')
        for i, id in enumerate(usable_mesh_IDs): # subjects?
            if i%500==0:
                print('Preping patches for sub: {}'.format(id))
            for j in range(num_patches): # for each columns
                indices_to_extract = indices_mesh_triangles[str(j)].to_numpy()
                data_ico_lowres[i,:,j,:] = data[i][:,indices_to_extract] #will be subXmapsX320X153 for ico2
                #data[i+num_subjects,:,:,j,:] = normalised_data[2*i+1][:,indices_to_extract]

    print('#'*30)
    print('#Saving: data')
    print('#'*30)
    del data

    (data_train, 
    data_validation, 
    data_test, 
    netmat_data_train, 
    netmat_data_validation, 
    netmat_data_test) = fcn_split_datas(data_ico_lowres, netmat_data, path_to_subject_lists, usable_mesh_IDs)
    # print(data_train.shape, data_validation.shape, data_test.shape) #correct shapes

    if not os.path.isdir(output_maps_netmats_path):
        os.makedirs(output_maps_netmats_path)

    if chosen_hemi == '2LR':
        print("Not ready yet.")
    else:
        # save surfaces
        filename = os.path.join(f"{output_maps_netmats_path}/train_1{hemisphere_chosen}_surf.npy")
        np.save(filename,data_train)

        filename = os.path.join(f"{output_maps_netmats_path}/validation_1{hemisphere_chosen}_surf.npy")
        np.save(filename,data_validation)

        filename = os.path.join(f"{output_maps_netmats_path}/test_1{hemisphere_chosen}_surf.npy")
        np.save(filename,data_test)

    # no need to be in if statement b/c L and R use same netmats per subject
    # same for netmats
    filename = os.path.join(f"{output_maps_netmats_path}/train_vecnetmat_uppertri.npy")
    np.save(filename,netmat_data_train)

    filename = os.path.join(f"{output_maps_netmats_path}/validation_vecnetmat_uppertri.npy")
    np.save(filename,netmat_data_validation)

    filename = os.path.join(f"{output_maps_netmats_path}/test_vecnetmat_uppertri.npy")
    np.save(filename,netmat_data_test)

if __name__ == '__main__':
    import pandas as pd
    import nibabel as nb
    import numpy as np
    import yaml
    import os
    import argparse
    import sys
    # sys.path.append('../')
    # sys.path.append('../../')

    # Set up argument parser        
    parser = argparse.ArgumentParser(description='preprocessing cortical sheet maps for patching.')
    
    parser.add_argument('--config_path',
                        type=str,
                        default='./config/hparams_surf2mat.yml',
                        help='Path to YAML file containing parameter information.')
    # parser.add_argument('--type',
    #                     default='train',
    #                     help='Preprocessing type: train, validation, test.')
    
    options = parser.parse_args()
    with open(options.config_path) as f:
        config = yaml.safe_load(f)

    # Call training
    main(config)