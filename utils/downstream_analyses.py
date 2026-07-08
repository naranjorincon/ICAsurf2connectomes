# %%
'''
To run this shole jupyter notebook as a python script follow this:

(1) activate conda environemt
(2) go to where this notebook is located in your computer
(3) use `python` to enter python with in your shell/terminal
(4) follow the above syntax


from json import load

filename = 'downstream_analyses.ipynb'
with open(filename) as fp:
    nb = load(fp)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(line for line in cell['source'] if not line.startswith('%'))
        exec(source, globals(), locals())


'''

# %%
# PCA first
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import *
import glob
import os
import math
import argparse
import yaml
# from scipy.stats import * #false_discovery_control


# %%
## HELPER FUCNTIONS FOR LARET
def fdr_bhmethod(p_vals):
    '''
    BH==Benjamin-Hochberg method. Entails first ranking all p values from smallest (1t)
    to largets (last). So, ranking them. Then, adjusting each p-value accoring to rank.
    '''
    from scipy.stats import rankdata
    ps = np.asfarray(p_vals)
    ranked_p_values = rankdata(ps)
    print(ranked_p_values)
    fdr = ps * (len(ps) / ranked_p_values)
    fdr[fdr > 1] = 1

    return fdr

def bonferroni_adj(p_vals):
    ps = np.asfarray(p_vals)
    num_of_tests = len(ps)
    print(ps)
    adj_ps = ps * num_of_tests
    print(adj_ps) # should be same nums but times constant 4950 (from_parcellation)
    adj_ps[adj_ps > 1] = 1
    return adj_ps 

# %%
def whole_model_arch(config):
    # for later data viz
    from_parcellation = config['data']['from_parcellation']
    translation = config['data']['translation'] #f"ICAd15_schfd{from_parcellation}" # needs to be "" type of string
    matrix_type = "topography_maps" if "netmatd" in translation else "connectome"
    parcellation_name=config['data']['parcellation_name']
    hemi_cond=config['training']['hemi_cond']
    parcellation_corr_type=config['training']['parcellation_corr_type']

    if translation == "ICAd15_ICAd15":
        assert 0==1, "Forced FALSE to stop downstream analyses. No such thing for ICA-->ICA testing."

    version = config['data']['version'] #"normICAdemeanMAT" #normICAdemeanfishzMAT normICArawMAT
    local_flag=False
    if local_flag:
        using_local_root="/Users/snaranjo/Desktop/neurotranslate/mount_point"
    else:
        using_local_root=""
    scratch_path=f"{using_local_root}/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"
    root=f"{scratch_path}/NeuroTranslate/surf2netmat"
    # model_output_names = ["kSiTLN", "kvSiTLN", "kSiTBGT", "kvSiTBGT"]
    datasets = config['training']['dataset_choice'] #"ABCD" #["HCPYA", "ABCD"]
    # chosen_one = 0  #corresponds to ["kSiTLN", "kvSiTLN", "kSiTBGT", "kvSiTBGT"]
    # dataset_choice = 1 # 0==HCPYA, 1==ABCD, 2==HCPYA_ABCDdr 
    model_details = config['transformer']['model_details'] #"d12h6_adamW_cosinedecay_recon_MSEtrain_1L_full_demean_expDeepBase_wGelu_102825" #"d6h3_adamW_cosinedecay_recon_krakenonly_1R_082925"  #"d6h3_adamW_cosinedecay_recon_krakenonly_1L_full_100425" 
    model_test_type = config['testing']['chosen_test_model'] #"MSE" #["MSE", "MAE", "RHO", "LAST"]
    img_extension = 'png' # png or eps #MICCAI file extension for images preferred
    model_type = config['data']['model_type'] #model_output_names[chosen_one]

    out_of_sample_test = config['testing']['out_of_sample_test']
    if out_of_sample_test:
        assert 0==1, "Forced FALSE to stop downstream analyses. No such thing for HCPYA out of distribution testing."
        datasets = datasets #+ "_HCPYAtest"

    root_data=f"{root}/model_out/{translation}/{datasets}/{model_type}/{version}/{model_details}/{model_test_type}" #/ABCD_train_HCPYA_test"
    print(f"Root path: {root_data}")
    try:
        test_truth_holder = np.load(f"{root_data}/test_ground_truth.npy")#[:smaller_scale]
        test_pred_holder = np.load(f"{root_data}/test_pred.npy")#[:smaller_scale]
    except:
        print(f"Model does not have TEST results. Either was not ran, cancelled, or NaNs so exited. Verify as needed. \nMODEL IS: {model_details}")
        # continue
    print(f"Test shapes: {test_truth_holder.shape} (Target)  {test_pred_holder.shape} (Pred)")

    try:
        train_truth_holder = np.load(f"{root_data}/train_ground_truth.npy")#[:smaller_scale]
        train_pred_holder = np.load(f"{root_data}/train_pred.npy")#[:smaller_scale]
    except:
        print(f"Model does not have TEST results. Either not ran, or cancelled, or NaNs so exited. Verify as needed. \nMODEL IS: {model_details}")
        # continue
    print(f"Train shapes: {train_truth_holder.shape} (Target)  {train_pred_holder.shape} (Pred)")

    directory = root + '/images/' + datasets + '/' + translation  + '/' + model_type +'/' + version + '/'+ model_details + '/' + model_test_type + '/downstream_analyses' #
    print(f"DIRECTORY PATH: {directory}")
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)

    # netmat_id_version=["glasser_mats", "schaefer_mats", "schaefer_mats"]
    # if translation == "ICAd15_glasserd360":
    #     cc=0
    # else:
    #     cc=1
    
    train_test_true_fused = np.concatenate((train_truth_holder,test_truth_holder), axis=0)
    train_test_pred_fused = np.concatenate((train_pred_holder,test_pred_holder), axis=0)
    print(train_test_true_fused.shape)
    true_train_true_test_rho = np.corrcoef(train_test_true_fused)
    print(true_train_true_test_rho.shape)
    pred_train_pred_test_rho = np.corrcoef(train_test_pred_fused)
    print(pred_train_pred_test_rho.shape)

    # print(f"Train-Test_true_rho: {true_train_true_test_groundtruth_rho}")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes = axes.flatten()
    # img0 = axes[0].imshow(true_train_true_test_rho, aspect='auto', vmax=1, cmap="Spectral_r")
    axes[0].hist(mat2vector(true_train_true_test_rho, diagonal_flag=1), bins=100) #including diagonal 1s
    axes[0].set_title("rho TRAIN,TEST TRUE")
    # plt.colorbar(img0, ax=axes[0])
    # img1 = axes[1].imshow(pred_train_pred_test_rho, aspect='auto', vmax=1, cmap="Spectral_r")
    # plt.colorbar(img1, ax=axes[1])
    axes[1].hist(mat2vector(pred_train_pred_test_rho, diagonal_flag=1), bins=100) #including diagonal 1s
    axes[1].set_title("rho TRAIN,TEST PRED")
    plt.tight_layout()

    plt.close()

    # demean when needed
    print(train_pred_holder)
    if "demean" in version: #pred demean data already, so no need to demean further
        demean_flag = False
        print(f"DEMEAN POST ANALYSES FLAG IS: {demean_flag}")
        train_true_mean = 0
        train_pred_mean = 0
        test_true_mean = 0
        test_pred_mean = 0
    else:
        demean_flag = True
        print(f"DEMEAN POST ANALYSES FLAG IS: {demean_flag}")
        train_true_mean = np.nanmean(train_truth_holder, axis=0, keepdims=True)
        train_pred_mean = np.nanmean(train_pred_holder, axis=0, keepdims=True)
        test_true_mean = np.nanmean(test_truth_holder, axis=0, keepdims=True)
        test_pred_mean = np.nanmean(test_pred_holder, axis=0, keepdims=True)

    train_truth_holder = train_truth_holder - train_true_mean
    train_pred_holder = train_pred_holder - train_pred_mean
    test_truth_holder = test_truth_holder - test_true_mean
    test_pred_holder = test_pred_holder - test_pred_mean

    # %%
    # visualize to again make sure all looks good
    train_true_netmats = train_truth_holder
    train_pred_netmats = train_pred_holder
    test_true_netmats = test_truth_holder
    test_pred_netmats = test_pred_holder
    a = make_nemat_allsubj(train_true_netmats,from_parcellation)
    b = make_nemat_allsubj(train_pred_netmats,from_parcellation)
    c = make_nemat_allsubj(test_true_netmats,from_parcellation)
    d = make_nemat_allsubj(test_pred_netmats,from_parcellation)

    plt.rcParams.update({'font.size': 18})

    subs2view=[]
    for kk in range(6):
        x = np.random.randint(0,a.shape[0])
        subs2view.append(x)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    xx = 0
    for i in range(6):
        axes[xx].imshow(a[subs2view[i]].squeeze(), aspect="auto", vmin=-0.5, vmax=0.5, cmap="Spectral_r")
        axes[xx].set_title(f"True{subs2view[i]}")
        axes[xx+1].imshow(b[subs2view[i]].squeeze(), aspect="auto", vmin=-0.5, vmax=0.5, cmap="Spectral_r")
        axes[xx+1].set_title(f"Pred{subs2view[i]}")

        xx += 2
    plt.suptitle("TRAIN")
    plt.tight_layout()
    # plt.close()
    filename = f"/examples_netmats_train.{img_extension}"
    # print(f"Saving to path:{directory}")
    # Save the figure
    plt.savefig(directory + filename, format=img_extension)
    plt.close()
    plt.close()

    subs2view=[]
    for kk in range(6):
        x = np.random.randint(0,c.shape[0])
        subs2view.append(x)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    xx = 0
    for i in range(6):
        axes[xx].imshow(c[subs2view[i]].squeeze(), aspect="auto", vmin=-0.5, vmax=0.5, cmap="Spectral_r")
        axes[xx].set_title(f"True{subs2view[i]}")
        axes[xx+1].imshow(d[subs2view[i]].squeeze(), aspect="auto", vmin=-0.5, vmax=0.5, cmap="Spectral_r")
        axes[xx+1].set_title(f"Pred{subs2view[i]}")

        xx += 2
    plt.suptitle("TEST")
    plt.tight_layout()

    # directory = root + '/images/' + 'ABCD' + '/' + translation  + '/' + model_type +'/' + version + '/'+ model_details + '/' + model_test_type + '/' + 'downstream_analyses' #
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory for model created.")
    else:
        print("Directory for model output already exists.")

    filename = f"/examples_netmats_test.{img_extension}"
    plt.savefig(directory + filename, format=img_extension)
    plt.close()


    # %%
    # ABCD behavioral measures
    beh_path = f"{scratch_path}/NeuroTranslate/ABCD_behv/nc_y_nihtb.tsv"
    composite_scores = pd.read_csv(beh_path)
    # behv_of_interest_list = ["nc_y_nihtb__comp__cryst__uncor_score",
    #                         "nc_y_nihtb__comp__fluid__uncor_score",
    #                         "nc_y_nihtb__comp__tot__uncor_score",
    #                         "nc_y_nihtb__flnkr__uncor_score",
    #                         "nc_y_nihtb__lswmt__uncor_score",
    #                         "nc_y_nihtb__readr__uncor_score"]
    # behv_type_list=["Crystal",
    #                 "Fluid",
    #                 "Total",
    #                 "Flnkr",
    #                 "LSWMT",
    #                 "Readoral"]
    behv_of_interest_list = ["nc_y_nihtb__comp__cryst__uncor_score", "nc_y_nihtb__comp__fluid__uncor_score"]
    behv_type_list=["Crystal", "Fluid"]
    for pp in range(len(behv_of_interest_list)):
        behv_of_interest=behv_of_interest_list[pp]
        print(f"Behv path file is: {beh_path}")
        print(f"Current behv of interest being visualized: {behv_of_interest}")
        behv_type=behv_type_list[pp]

        cols_to_use_list = ["participant_id", f"{behv_of_interest}"] #nc_y_nihtb__comp__fluid__uncor_score, nc_y_nihtb__comp__cryst__uncor_score
        composite_scores = pd.read_csv(beh_path, sep='\t', usecols=cols_to_use_list) #nc_y_nihtb__comp__cryst__uncor_score, nc_y_nihtb__comp__fluid__uncor_score
        print(composite_scores.shape)
        # clean of NaNs
        composite_scores_clean = composite_scores.dropna()
        print(composite_scores_clean.shape)
        print(f"Removed {(composite_scores.shape[0] - composite_scores_clean.shape[0])} subjects with NaNs in either crystal or fluid intelligence.")

        unique_ids = composite_scores_clean["participant_id"].unique()
        print(f"Unique IDs: {len(unique_ids)}")
        unique_ids_clean = unique_ids
        for ii in range(unique_ids.shape[0]):
            unique_ids_clean[ii] = unique_ids[ii][4:]
        print(len(unique_ids_clean), unique_ids_clean) #clean here means removed the "sub-""

        #do the same for the IDs we use in our analyses
        main_ids_path=f"{scratch_path}/NeuroTranslate/surf2netmat/utils/subj_ids/{datasets}/full_IDs_train_val_test_{hemi_cond}_{matrix_type}_{parcellation_name}_{from_parcellation}_{parcellation_corr_type}.csv"
        get_subIDs_groups = pd.read_csv(main_ids_path)
        ABCD_main_ids_clean=list(get_subIDs_groups[get_subIDs_groups['group'] == 'train']['sub_IDs'].unique())
        print(len(ABCD_main_ids_clean))
        # repeat for test
        ABCD_main_ids_test_clean=list(get_subIDs_groups[get_subIDs_groups['group'] == 'test']['sub_IDs'].unique())
        print(len(ABCD_main_ids_test_clean))
        
        # only get subjects that are in the main IDs for reference
        isin_check_mask = np.isin(unique_ids_clean, ABCD_main_ids_clean)
        print(len(isin_check_mask))
        unique_ids_clean_unified = unique_ids_clean[isin_check_mask]
        print((unique_ids_clean_unified), unique_ids_clean_unified.shape)

        isin_check_mask = np.isin(unique_ids_clean, ABCD_main_ids_test_clean)
        print(len(isin_check_mask)) # can use same mask even if overwrite 
        unique_ids_clean_unified_test = unique_ids_clean[isin_check_mask] # same composite IDs file but now only using subjects that are in test
        print((unique_ids_clean_unified_test), unique_ids_clean_unified_test.shape)

        bilateral_condition_flag=config['training']['bilateral_condition']
        if bilateral_condition_flag is True:
            unique_ids_clean_unified = np.concatenate((unique_ids_clean_unified,unique_ids_clean_unified), axis=0)
            unique_ids_clean_unified_test = np.concatenate((unique_ids_clean_unified_test,unique_ids_clean_unified_test), axis=0)
            print(len(unique_ids_clean_unified),len(unique_ids_clean_unified_test))
            ABCD_main_ids_clean = ABCD_main_ids_clean+ABCD_main_ids_clean
            ABCD_main_ids_test_clean= ABCD_main_ids_test_clean+ABCD_main_ids_test_clean
            print(len(ABCD_main_ids_clean), len(ABCD_main_ids_test_clean))

        subj_list_scores = []
        # below gets crystal and fluid scores for all subjects in the large composite_scores file that ARE in the train_IDs AND have data (no nans)
        for ii in range(unique_ids_clean_unified.shape[0]):
            str_version_with_sub = "sub-" + f"{unique_ids_clean_unified[ii]}"
            # print(str_version_with_sub)
            curr_ii = np.where(composite_scores_clean["participant_id"] == str_version_with_sub)
            # print(curr_ii) #idx of where its at in larger composite_scores_clean file
            idx_max_last = np.max(curr_ii) # choose the last one whatever
            df_for_subj = (composite_scores_clean.iloc[idx_max_last,:]).to_numpy()
            # print(f"PPP: {(df_for_subj)}")
            subj_list_scores.append(df_for_subj) #subjID, crystal, fluid

        ABCD_original_ABCD_behv_match_df = pd.DataFrame(subj_list_scores, columns=cols_to_use_list) #nc_y_nihtb__comp__fluid__uncor_score
        print(ABCD_original_ABCD_behv_match_df.shape)
        ABCD_original_ABCD_behv_match_df["participant_id"] = unique_ids_clean_unified # replace column with version without sub-
        ABCD_original_ABCD_behv_match_df_clean = ABCD_original_ABCD_behv_match_df.dropna() # remove any NaNs
        print(ABCD_original_ABCD_behv_match_df_clean)
        print(f"Removed {(ABCD_original_ABCD_behv_match_df.shape[0] - ABCD_original_ABCD_behv_match_df_clean.shape[0])} subjects with NaNs in either crystal or fluid intelligence.")

        ## same for test
        subj_list_scores_test = []
        # below gets crystal and fluid scores for all subjects in the large composite_scores_clean file that ARE in the train_IDs AND have data (no nans)
        for ii in range(unique_ids_clean_unified_test.shape[0]):
            str_version_with_sub = "sub-" + f"{unique_ids_clean_unified_test[ii]}"
            curr_ii = np.where(composite_scores_clean["participant_id"] == str_version_with_sub)
            idx_max_last = np.max(curr_ii) # choose the last one whatever
            df_for_subj = (composite_scores_clean.iloc[idx_max_last,:]).to_numpy()
            subj_list_scores_test.append(df_for_subj) #subjID, crystal, fluid

        ABCD_original_test_ABCD_behv_match_df = pd.DataFrame(subj_list_scores_test, columns=cols_to_use_list) #nc_y_nihtb__comp__fluid__uncor_score
        print(ABCD_original_test_ABCD_behv_match_df.shape)
        ABCD_original_test_ABCD_behv_match_df["participant_id"] = unique_ids_clean_unified_test # replace column with version without sub-
        ABCD_original_test_ABCD_behv_match_df_clean = ABCD_original_test_ABCD_behv_match_df.dropna() # remove any NaNs
        print(ABCD_original_ABCD_behv_match_df_clean)
        print(f"Removed {(ABCD_original_test_ABCD_behv_match_df.shape[0] - ABCD_original_test_ABCD_behv_match_df_clean.shape[0])} subjects with NaNs in either crystal or fluid intelligence.")


        # take the time to match these new subjects with BEHV data and the imaging data we have
        print(len(ABCD_original_ABCD_behv_match_df_clean["participant_id"]), ABCD_original_ABCD_behv_match_df_clean["participant_id"][0])
        subjs_with_behv_and_netmat_mask = np.isin(ABCD_main_ids_clean, ABCD_original_ABCD_behv_match_df_clean["participant_id"])
        print((subjs_with_behv_and_netmat_mask).sum()) # should be the same as before N=2822
        # netmats_match = ABCD_main_ids[subjs_with_behv_and_netmat_mask]
        iix_netmats = np.where(subjs_with_behv_and_netmat_mask == 1)[0] # gives tuple, so choose first element
        print(len(iix_netmats))
        crystal_match = ABCD_original_ABCD_behv_match_df_clean[f"{behv_of_interest}"].to_numpy()        

        # repeat for test
        print(len(ABCD_original_test_ABCD_behv_match_df_clean["participant_id"]), ABCD_original_test_ABCD_behv_match_df_clean["participant_id"][0])
        subjs_test_with_behv_and_netmat_mask = np.isin(ABCD_main_ids_test_clean, ABCD_original_test_ABCD_behv_match_df_clean["participant_id"])
        print((subjs_test_with_behv_and_netmat_mask).sum()) # should be the same as before N=2822
        # netmats_match_test = ABCD_main_ids_test[subjs_test_with_behv_and_netmat_mask]
        iix_netmats_test = np.where(subjs_test_with_behv_and_netmat_mask == 1)[0] # gives tuple, so choose first element
        print(iix_netmats_test.shape, type(iix_netmats_test))
        crystal_match_test = ABCD_original_test_ABCD_behv_match_df_clean[f"{behv_of_interest}"].to_numpy()


        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes=axes.flatten()
        axes[0].hist((crystal_match), bins=50, color='red', label=f"{behv_type}", alpha=0.5)
        # axes[0].hist((fluid_match), bins=50, color='blue', label="fluid", alpha=0.5)
        axes[0].legend()

        axes[1].hist((crystal_match_test), bins=50, color='red', label=f"{behv_type}", alpha=0.5)
        # axes[1].hist((fluid_match_test), bins=50, color='blue', label="fluid", alpha=0.5)
        axes[1].legend()

        filename = f"/histogram_of_behv.{img_extension}"
        plt.savefig(directory + filename, format=img_extension)
        plt.close()


        ####### TEST ######
        numpy_corr_flag=False
        scipy_corr_flag=True
        print(f"C/D shapes -->. {c.shape}/{d.shape}")
        print(f"iix_netmats_test details. N={len(iix_netmats_test)}, shape={iix_netmats_test.shape}, example={iix_netmats_test[:5]}")
        test_true_netmats_clean = get_lower_tris(c[iix_netmats_test]) #only get netmats with behavioral data to compare
        test_pred_netmats_clean = get_lower_tris(d[iix_netmats_test]) #only get netmats with behavioral data to compare
        print(test_true_netmats_clean.shape, test_pred_netmats_clean.shape)

        brain_cyrstal_rho_test = np.zeros((1, test_true_netmats_clean.shape[1]))
        brain_cyrstal_rho_pred_test = np.zeros((1, test_pred_netmats_clean.shape[1]))
        print(brain_cyrstal_rho_test.shape)

        brain_cyrstal_rho_pval_test = np.zeros((1, test_true_netmats_clean.shape[1]))
        brain_cyrstal_rho_pred_pval_test= np.zeros((1, test_pred_netmats_clean.shape[1]))

        for ee in range(test_true_netmats_clean.shape[1]):
            if numpy_corr_flag:
                brain_cyrstal_rho_test[:,ee] = np.corrcoef(test_true_netmats_clean[:,ee], crystal_match_test)[0,1]

                # pred associations too
                brain_cyrstal_rho_pred_test[:,ee] = np.corrcoef(test_pred_netmats_clean[:,ee], crystal_match_test)[0,1]
            elif scipy_corr_flag:
                # using scipy instead for also quick p-value --- corr, p-val_cal from beta distribution
                brain_cyrstal_rho_test[:,ee] = stats.pearsonr(test_true_netmats_clean[:,ee], crystal_match_test)[0]
                brain_cyrstal_rho_pred_test[:,ee] = stats.pearsonr(test_pred_netmats_clean[:,ee], crystal_match_test)[0]

                brain_cyrstal_rho_pval_test[:,ee] = stats.pearsonr(test_true_netmats_clean[:,ee], crystal_match_test)[1]
                brain_cyrstal_rho_pred_pval_test[:,ee] = stats.pearsonr(test_pred_netmats_clean[:,ee], crystal_match_test)[1]

        # print(brain_fluid_rho_test, brain_fluid_rho_test.shape)
        print(brain_cyrstal_rho_test, brain_cyrstal_rho_pred_test.shape)

        filename = f"/TRUE_test_brain_{behv_type}_rho.npy"
        np.save(directory+f'{filename}', brain_cyrstal_rho_test)

        filename = f"/PRED_test_brain_{behv_type}_rho.npy"
        np.save(directory+f'{filename}', brain_cyrstal_rho_pred_test)

        #make into netmats to show
        corr_netmat_crystal_test = make_netmat(fisher_z_transform(brain_cyrstal_rho_test), from_parcellation)
        # corr_netmat_fluid_test = make_netmat(fisher_z_transform(brain_fluid_rho_test), from_parcellation)
        corr_netmat_crystal_pred_test = make_netmat(fisher_z_transform(brain_cyrstal_rho_pred_test), from_parcellation)
        # corr_netmat_fluid_pred_test = make_netmat(fisher_z_transform(brain_fluid_rho_pred_test), from_parcellation)

        if scipy_corr_flag:
            fdr_correction_flag=False
            bonferroni_correction_flag=True

            if fdr_correction_flag:
                correction_method="FDR"
                pval_threshold= 0.05*(0.5**8) # how many times to dive by 2 or times 1/2=0.5 
                brain_cyrstal_rho_pval_adj_test = fdr_bhmethod(brain_cyrstal_rho_pval_test)
                # brain_fluid_rho_pval_adj = fdr_bhmethod(brain_fluid_rho_pval_test)
                brain_cyrstal_rho_pred_pval_adj_test = fdr_bhmethod(brain_cyrstal_rho_pred_pval_test)
                # brain_fluid_rho_pred_pval_adj = fdr_bhmethod(brain_fluid_rho_pred_pval_test)
                print(brain_cyrstal_rho_pred_pval_adj_test)

                brain_cyrstal_rho_pval_adj_test[brain_cyrstal_rho_pval_adj_test > pval_threshold] = 0
                print(brain_cyrstal_rho_pval_adj_test)
                # brain_fluid_rho_pval_adj[brain_fluid_rho_pval_adj > pval_threshold] = 0
                brain_cyrstal_rho_pred_pval_adj_test[brain_cyrstal_rho_pred_pval_adj_test > pval_threshold] = 0
                # brain_fluid_rho_pred_pval_adj[brain_fluid_rho_pred_pval_adj > pval_threshold] = 0

                brain_cyrstal_rho_pval_adj_test[brain_cyrstal_rho_pval_adj_test > 0] = 1
                # brain_fluid_rho_pval_adj[brain_fluid_rho_pval_adj > 0] = 1
                brain_cyrstal_rho_pred_pval_adj_test[brain_cyrstal_rho_pred_pval_adj_test > 0] = 1
                # brain_fluid_rho_pred_pval_adj[brain_fluid_rho_pred_pval_adj > 0] = 1

            if bonferroni_correction_flag:
                correction_method="BONF"
                pval_threshold= 0.05*(0.5**3) # how many times to dive by 2 or times 1/2=0.5 
                brain_cyrstal_rho_pval_adj_test = bonferroni_adj(brain_cyrstal_rho_pval_test)
                brain_cyrstal_rho_pred_pval_adj_test = bonferroni_adj(brain_cyrstal_rho_pred_pval_test)

                brain_cyrstal_rho_pval_adj_test[brain_cyrstal_rho_pval_adj_test > pval_threshold] = 0
                brain_cyrstal_rho_pred_pval_adj_test[brain_cyrstal_rho_pred_pval_adj_test > pval_threshold] = 0

                brain_cyrstal_rho_pval_adj_test[brain_cyrstal_rho_pval_adj_test > 0] = 1
                brain_cyrstal_rho_pred_pval_adj_test[brain_cyrstal_rho_pred_pval_adj_test > 0] = 1

            corr_netmat_crystal_pval_test = make_netmat(brain_cyrstal_rho_pval_adj_test, from_parcellation)
            print(corr_netmat_crystal_pval_test)
            corr_netmat_crystal_pred_pval_test = make_netmat(brain_cyrstal_rho_pred_pval_adj_test, from_parcellation)
            
            #diagonal should be 0
            # rho values
            np.fill_diagonal(corr_netmat_crystal_test, 0)
            np.fill_diagonal(corr_netmat_crystal_pred_test, 0)
            # pvalues of rho above
            np.fill_diagonal(corr_netmat_crystal_pval_test, 0)
            np.fill_diagonal(corr_netmat_crystal_pred_pval_test, 0)
            
        true_pred_corr_crystal_test = np.corrcoef(brain_cyrstal_rho_test, brain_cyrstal_rho_pred_test)[0,1]
        true_pred_spear_crystal_obj_test = stats.spearmanr(brain_cyrstal_rho_test.squeeze(), brain_cyrstal_rho_pred_test.squeeze())
        true_pred_spear_crystal_test = true_pred_spear_crystal_obj_test.correlation

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes = axes.flatten()
        img0 = axes[0].imshow(corr_netmat_crystal_test, aspect="auto", vmin=-0.2, vmax=0.2, cmap="Spectral_r")
        axes[0].set_title(f"True-{behv_type}")
        plt.colorbar(img0, ax=axes[0])

        img1 = axes[1].imshow(corr_netmat_crystal_pred_test, aspect="auto", vmin=-0.2, vmax=0.2, cmap="Spectral_r")
        axes[1].set_title(f"Pred-(rho:{true_pred_corr_crystal_test:.2f},spr:{true_pred_spear_crystal_test:.2f})")
        plt.colorbar(img1, ax=axes[1])
        plt.suptitle("TEST DATASET")

        plt.tight_layout()

        filename = f"/test_downstream_brainbehv.{img_extension}"
        plt.savefig(directory + filename, format=img_extension)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='viz')

    parser.add_argument(
                        'config',
                        type=str,
                        default='',
                        help='args from yaml file')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call training
    whole_model_arch(config)