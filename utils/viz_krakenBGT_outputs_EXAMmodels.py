# %%
import numpy as np
import matplotlib.pyplot as plt
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
import argparse
import yaml

def plot_training_losses(ax, data, data_val, validation_step=1, err=int, err_val=int, loss=str, title_str=str):
                xvals_consecutive = np.arange(0,len(data),1)
                make_data_np = np.asarray(data)
                # make_data_val_np = np.asarray(data_val)
                # plot train
                ax.plot(xvals_consecutive, make_data_np, marker='.', linestyle='-', label="Train")
                # ax.fill_between(xvals_consecutive, make_data_np-err, make_data_np+err) #adding std across batcehs per epoch

                # plot validation
                if data_val is not None:
                    ax.plot(range(0, len(data_val)*validation_step, validation_step), data_val, color='red', label="Validation",  alpha=0.5)
                # ax.fill_between(range(0, len(data_val)*validation_step, validation_step), make_data_val_np-err_val, make_data_val_np+err_val)

                ax.set_xlabel('Epoch')
                ax.set_ylabel(loss)
                # ax.set_title(title_str)
                ax.grid(True)
                
def whole_model_arch(config):
    # for later data viz
    translation= config['data']['translation']
    local_flag=False #always correct in CHPC

    version = config['data']['version'] #"normICAnormICA" #"normICArawMAT" # "normICAdemeanfishzMAT" #normICAdemeanfishzMAT
    scratch_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"
    root=f"{scratch_path}/NeuroTranslate/surf2netmat"
    datasets = config['training']['dataset_choice'] #["HCPYA", "ABCD", "HCPYA_ABCDdr"]
    out_of_sample_test = config['testing']['out_of_sample_test'] = config['testing']['out_of_sample_test']
    if out_of_sample_test:            
        datasets = "HCPYA_" + datasets + "dr" #"HCPYA_ABCDdr"
    print(f"dataset is: {datasets}")

    # specific_channel = config['training']['specific_channel']
    # if ica_reconstuction:
    #     type_of_model_train_fcn = config['transformer']['model_details'] + f'_chnl{specific_channel}'
    type_of_model_train_fcn = config['transformer']['model_details']
    
    model_test_type_list = [ config['testing']['chosen_test_model'] ] #["MSE"] #["MSE", "MAE", "RHO", "LAST"]
    img_extension = 'png' # png or eps #MICCAI file extension for images preferred
    val_step = 1 #5 or 1, depends. 
    model_type = config['data']['model_type'] #model_output_names[chosen_one] # chosen index
    from_parcellation = config['data']['from_parcellation']

    if translation == "ICAd15_ICAd15": #over ride in this case
        ica_reconstuction = True
        version = "normICAnormICA" #over ride what it was prior
        chnl_icarecon = config['training']['specific_channel']
        # type_of_model_train_fcn = f"120425_d6h3_tiny_adamW_cosinedecay_reconICA_MSEtrain_expICARECON_chnl{chnl_icarecon}"
        type_of_model_train_fcn = type_of_model_train_fcn + f"_chnl{chnl_icarecon}"
    else:
        ica_reconstuction = False

    for mm in model_test_type_list:
        model_test_type = mm #model_test_type_list[mm]
        
        list_of_details=np.sort(glob.glob(f"{root}/model_out/{translation}/{datasets}/{model_type}/{version}/{type_of_model_train_fcn}")) # should be all versions and their detial names
        print(f"{root}/model_out/{translation}/{datasets}/{model_type}/{version}/{type_of_model_train_fcn}")
        print(list_of_details)
        
        print("All detials to choose from")
        actual_details_to_choose = []
        for ll in list_of_details:  
            get_only_details = ll.split('/')
            print(get_only_details[-1])
            actual_details_to_choose.append(get_only_details[-1])  

        print(f"Full list: {actual_details_to_choose}")
        # if you want to run a single model, give here the index of that model you want to run from the list 
        # if single_model_flag:
        #     idx = chosen_model_details
        #     actual_details_to_choose = [actual_details_to_choose[idx]]

        print("Entering loop")
        # then it will loop thorugh the list
        for ll in actual_details_to_choose:
            model_details_list = ll #actual_details_to_choose[0] #"d12h10_adamW_cosinedecay_scheduler_skewloss"
            model_details = model_details_list #model_details_list[-1]
            print(f"Current ModDetails:{model_details}")

            # %%
            folder_to_save_model=f"{root}/model_out/{translation}/{datasets}/{model_type}/{version}/{model_details}/"

            directory = root + '/images/' + datasets + '/' + translation  + '/' + model_type +'/' + version + '/'+ model_details + '/' + model_test_type #+ '/ABCD_train_HCPYA_test'# Replace with your target directory
            if not os.path.exists(directory):
                # Create the directory
                os.makedirs(directory)
                print("Directory for model created.")
            else:
                print("Directory for model output already exists.")
            # if out_of_sample_test:
            #     datasets = datasets #+ "_HCPYAtest"

            try:
                train_loss_file=pd.read_csv(folder_to_save_model + "/train_losses_patch.csv")
                val_loss_file=pd.read_csv(folder_to_save_model + "/val_losses_patch.csv")
                print(train_loss_file.head(),'\n', val_loss_file.head())

                # Training curve
                epoch_step_so_far, _  = train_loss_file.shape
                # epoch_step_so_far2, _ = val_loss_file.shape
                print(f"Epochs so far on model are: {epoch_step_so_far}")

                # losses
                MAEs = train_loss_file['train_mae'].tolist()
                # MAEs_sigma = train_loss_file['train_mae_sigma'].tolist()
                MSEs = train_loss_file['train_mse'].tolist() #[]
                # MSEs_sigma = train_loss_file['train_mse_sigma'].tolist() #[]

                train_loss = train_loss_file['train_loss'].tolist()
                train_rho = train_loss_file['train_demean_corr'].tolist()
                # train_rho_sigma = train_loss_file['train_demean_corr_sigma'].tolist()
                train_origrho = train_loss_file['train_orig_corr'].tolist()
                # train_origrho_sigma = train_loss_file['train_orig_corr_sigma'].tolist()

                val_mae = val_loss_file['val_mae'].tolist()
                # val_mae_sigma = val_loss_file['val_mae_sigma'].tolist()
                val_mse = val_loss_file['val_mse'].tolist() # []
                # val_mse_sigma = val_loss_file['val_mse_sigma'].tolist() # []
                # val_loss = val_loss_file['val_loss'].tolist()
                val_rho = val_loss_file['val_demean_corr'].tolist()
                # val_rho_sigma = val_loss_file['val_demean_corr_sigma'].tolist()
                val_origrho = val_loss_file['val_orig_corr'].tolist()
                # val_origrho_sigma = val_loss_file['val_orig_corr_sigma'].tolist()

                # fig, axes = plt.subplots(2, 3, figsize=(20, 10)) 
                # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
                # plt.rcParams.update({'font.size': 18})
                plt.rcParams.update({'font.size': 18})
                fig, axes = plt.subplots(2, 3, figsize=(25, 8)) 
                ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
                    # first plot
                plot_training_losses(ax1, MAEs, val_mae, val_step, loss="MAE")
                ax1.legend(loc = 'upper right')

                # second plot
                plot_training_losses(ax2, MSEs, val_mse, val_step, loss="MSE")
                ax2.legend(["Train", "Val"], loc = 'upper right')

                # third plot
                plot_training_losses(ax3, train_loss, None, val_step, loss="KrakenLoss")
                ax3.legend(["Train", "Val"], loc = 'upper right')

                # fourth plot
                plot_training_losses(ax4, train_rho, val_rho, val_step, loss="demeanRHO")
                ax4.legend(["Train", "Val"], loc = 'upper right')

                # fifth plot
                plot_training_losses(ax5, train_origrho, val_origrho, val_step, loss="origRHO")
                ax5.legend(["Train", "Val"], loc = 'upper right')
                plt.tight_layout()

                filename = f'/model_losses.{img_extension}'
                print(f'Saving to path:{directory}')
                # Save the figure
                plt.savefig(directory + filename, format=img_extension)
                # plt.show()
                plt.close()
            except:
                print(f"Model does not have train/val losses. Either not ran, or cancelled, or NaNs so exited. Verify as needed. \nMODEL IS{model_details}")
                # continue

            # %%
            # smaller_scale = 500 # subjects to laod here
            root_data=f"{root}/model_out/{translation}/{datasets}/{model_type}/{version}/{model_details}/{model_test_type}"#/ABCD_train_HCPYA_test"

            try:
                test_truth_holder = np.load(f"{root_data}/test_ground_truth.npy")#[:smaller_scale]
                test_pred_holder = np.load(f"{root_data}/test_pred.npy")#[:smaller_scale]
            except:
                print(f"Model does not have TEST results Either not ran, or cancelled, or NaNs so exited. Verify as needed. \nMODEL IS{model_details}")
                continue
        
            print(f"Test shapes: {test_truth_holder.shape} (Target)  {test_pred_holder.shape} (Pred)")

            try:
                train_truth_holder = np.load(f"{root_data}/train_ground_truth.npy")#[:smaller_scale]
                train_pred_holder = np.load(f"{root_data}/train_pred.npy")#[:smaller_scale]
            except:
                print(f"Model does not have TEST results. Either not ran, or cancelled, or NaNs so exited. Verify as needed. \nMODEL IS{model_details}")
                continue
            print(f"Train shapes: {train_truth_holder.shape} (Target)  {train_pred_holder.shape} (Pred)")

            
            ### OLD BELOW
            # # train_mean_flatten_pred = np.mean(train_pred_holder, axis=0)
            # train_mean_flatten_true = np.mean(train_truth_holder, axis=0, keepdims=True) # mean across all vals even across channel
            # test_mean_flatten_true = np.mean(test_truth_holder, axis=0, keepdims=True)
            # # also get predicted values
            # train_mean_flatten_pred = np.mean(train_pred_holder, axis=0, keepdims=True)
            # test_mean_flatten_pred = np.mean(test_pred_holder, axis=0, keepdims=True)
            # print(test_mean_flatten_pred)
            ### NEW BELOW
            if "demean" in version: # if version is already demeaned, need to get actual MEAN from original data
                print("DATA already demeaned and predicted as such. For original versions, need to get unprep raw netmat data to get means.")
                hemi_cond="1L"
                if translation == "ICAd15_glasserd360":
                    train_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/ABCD/glasser_mats/netmat_d{from_parcellation}/{hemi_cond}_train_netmat_clean.npy")
                else:
                    train_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/ABCD/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_train_netmat_clean.npy")
                # train_surf_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{datasets}/ICA_maps/ICAd15_ico02/{hemi_cond}_train_surf.npy")
                print(f'Loaded in TRAIN. They have shapes: {train_netmat_np.shape} respectively.')

                # get the same for test
                if out_of_sample_test:
                    # outofsample_dataset_choice="HCPYA_ABCDdr" # choose which out of sample to use
                    
                    if translation == "ICAd15_glasserd360":
                        # te_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/ABCD/glasser_mats/netmat_d{from_parcellation}/{hemi_cond}_test_netmat_clean.npy")
                        print("Glasser HCP not done yet. Can't run.")
                        break
                    else:
                        te_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{datasets}/schaefer_mats/netmat_d100/train_netmat_clean.npy")
                    # te_surf_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{datasets}/ICA_maps/ICAd15_ico02/{hemi_cond}_train_surf.npy")
                    print(f'OUT OF SAMPLE Loaded in TEST. They have shapes: {te_netmat_np.shape}.')
                else:
                    if translation == "ICAd15_glasserd360":
                        te_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/ABCD/glasser_mats/netmat_d{from_parcellation}/{hemi_cond}_test_netmat_clean.npy")
                    else:                       
                        te_netmat_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{datasets}/schaefer_mats/netmat_d{from_parcellation}/{hemi_cond}_test_netmat_clean.npy")
                    # te_surf_np = np.load(f"{scratch_path}/NeuroTranslate/brain_reps_datasets/{datasets}/ICA_maps/ICAd15_ico02/{hemi_cond}_test_surf.npy")

                print(f'Loaded in TEST. They have shapes: {te_netmat_np.shape} respectively.')

                train_mean_flatten_true = np.mean(train_netmat_np, axis=0, keepdims=True)
                train_mean_flatten_pred = train_mean_flatten_true #same in this case

                test_mean_flatten_true = np.mean(te_netmat_np, axis=0, keepdims=True)
                test_mean_flatten_pred = test_mean_flatten_true

            else: #other wise, predicting standard netmats and can get means from those for later demeaning
                print("DATA predicted as is, so need to get averages to subtract for DEMEAN figures.")
                train_mean_flatten_pred = np.mean(train_pred_holder, axis=0, keepdims=True)
                train_mean_flatten_true = np.mean(train_truth_holder, axis=0, keepdims=True)
                test_mean_flatten_pred = np.mean(test_pred_holder, axis=0, keepdims=True)
                test_mean_flatten_true = np.mean(test_truth_holder, axis=0, keepdims=True)
            
            # variability (SD) across channels
            if translation == "ICAd15_ICAd15":
                sd_channel_allsubs=np.std(test_truth_holder,axis=1)
                print(f"Across channel sd to show variance: {sd_channel_allsubs.shape}")

                filename = f'/std_channel_{chnl_icarecon}.npy'
                print(f'Saving to path:{directory}')
                if not local_flag:
                    np.save(directory + filename, sd_channel_allsubs)
                
            train_correlation_matrix = np.corrcoef(train_truth_holder, train_pred_holder)
            test_correlation_matrix = np.corrcoef(test_truth_holder, test_pred_holder)

            ########### FINGER PRINT PERFORMANCE ##########
            # correct identification success rate
            test_split_half_horizontal = np.split(test_correlation_matrix, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            test_top_right_quad = np.split(test_split_half_horizontal[0], 2, axis = 1)[1]
            ii_test_performance = np.diag(test_top_right_quad)
            ij_test_comparison = mat2vector(test_top_right_quad)
            print(ii_test_performance.shape, ij_test_comparison.shape)

            ########################## train now
            train_split_half_horizontal = np.split(train_correlation_matrix, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            train_top_right_quad = np.split(train_split_half_horizontal[0], 2, axis = 1)[1]
            ii_train_performance = np.diag(train_top_right_quad)
            ij_train_comparison = mat2vector(train_top_right_quad)
            print(ii_train_performance.shape, ij_train_comparison.shape)

            test_success_count = 0
            test_fail_count = 0
            test_subj_id_success = np.zeros((ii_test_performance.shape[0],1))
            for ii in range(ii_test_performance.shape[0]):
                curr_subj = ii_test_performance[ii]
                check_comparison = curr_subj > ij_test_comparison # should be boolean of Trues and False
                # print((check_comparison.sum() / ij_test_comparison.shape[0]))    
                if np.all(check_comparison):
                    test_success_count += 1
                    test_subj_id_success[ii] = 1
                else:
                    test_fail_count += 1
                    test_subj_id_success[ii] = 0
            test_success_rate = (test_success_count / ii_test_performance.shape[0])
            test_fail_rate = (test_fail_count / ii_test_performance.shape[0])
            print(test_success_rate, test_fail_rate)

            train_success_count = 0
            train_fail_count = 0
            train_subj_id_success = np.zeros((ii_train_performance.shape[0],1))
            for ii in range(ii_train_performance.shape[0]):
                curr_subj = ii_train_performance[ii]
                check_comparison = curr_subj > ij_train_comparison # should be boolean of Trues and False
                # print((check_comparison.sum() / ij_train_comparison.shape[0]))    
                if np.all(check_comparison):
                    train_success_count += 1
                else:
                    train_fail_count += 1
            train_success_rate = (train_success_count / ii_train_performance.shape[0])
            train_fail_rate = (train_fail_count / ii_train_performance.shape[0])
            print(train_success_rate, train_fail_rate)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes = axes.flatten()
            img0 = axes[0].imshow(test_top_right_quad, vmin=-0.4, vmax=0.4, cmap="Spectral_r")
            plt.colorbar(img0, ax=axes[0])
            img1 = axes[1].imshow(train_top_right_quad, vmin=-0.4, vmax=0.4, cmap="Spectral_r")
            plt.colorbar(img1, ax=axes[1])

            # img2 = axes[2].imshow(corr_crystal_matching_true_pred_test, aspect="auto", vmin=0, vmax=2, cmap="afmhot_r")
            plt.tight_layout()
            filename = f"/FingerPrint_matrix_illustration.{img_extension}"
            if not local_flag:
                    plt.savefig(directory + filename, format=img_extension)
            plt.show()

            table_of_test_performance = pd.DataFrame({
                "Values": np.concatenate([test_subj_id_success.squeeze(), train_subj_id_success.squeeze()]),
                "Groups": ["Test"]*(test_subj_id_success.shape[0]) + ["Train"]*(train_subj_id_success.shape[0])
            })
            table_of_test_performance.to_csv(directory+'/FingerPrintTable.csv')

            ### aaverage rank version
            curr_frac_rank_test=[]
            for ii in range(ii_test_performance.shape[0]): #for each subject
                curr_subj = ii_test_performance[ii]
                check_comparison = curr_subj > ij_test_comparison # should be boolean of Trues and False
                rank_test_percentage = check_comparison.sum() / ij_test_comparison.shape
                # print(rank_test_percentage)
                curr_frac_rank_test.append(rank_test_percentage) 

            print(np.mean(curr_frac_rank_test))
            print(len(curr_frac_rank_test))

            curr_frac_rank_train=[]
            for ii in range(ii_train_performance.shape[0]):
                curr_subj = ii_train_performance[ii]
                check_comparison = curr_subj > ij_train_comparison # should be boolean of Trues and False
                rank_train_percentage = check_comparison.sum() / ij_train_comparison.shape
                # print(rank_train_percentage)
                curr_frac_rank_train.append(rank_train_percentage) 

            print(np.mean(curr_frac_rank_train))
            print(len(curr_frac_rank_train))

            df = pd.DataFrame({
                "Values": np.concatenate([np.asarray(curr_frac_rank_test).squeeze(), np.asarray(curr_frac_rank_train).squeeze()]),
                "Groups": ["Test AvgRank"] * (len(curr_frac_rank_test)) + ["Train AvgRank"] * len(curr_frac_rank_train)
            })
            fig= plt.figure(figsize=(8, 4))
            sns.histplot(df,x="Values", hue="Groups", bins=10, common_norm=False, log_scale=(False, True))
            filename = f"/histogram_avgrank_finger.{img_extension}"
            plt.tight_layout()
            if not local_flag:
                    plt.savefig(directory + filename, format=img_extension)
            plt.close()

            table_of_test_performance = pd.DataFrame(np.concatenate([np.asarray(curr_frac_rank_test).squeeze(), np.asarray(curr_frac_rank_train).squeeze()]),
                                    columns=["FingerPrintPerformance_avgrank"]
                                )

            # table_of_test_performance
            table_of_test_performance.to_csv(directory+'/FingerPrintTable_avgrank.csv')

            if not ica_reconstuction:
                train_mean_flatten_pred_mat = make_netmat(train_mean_flatten_pred,from_parcellation)
                train_mean_flatten_true_mat = make_netmat(train_mean_flatten_true,from_parcellation)
                test_mean_flatten_pred_mat = make_netmat(test_mean_flatten_pred,from_parcellation)
                test_mean_flatten_true_mat = make_netmat(test_mean_flatten_true,from_parcellation)

                # check correlation of corr(pred_mean,true_mean)
                corr_train_pred_true = (np.corrcoef(train_mean_flatten_pred, train_mean_flatten_true)[0,1])
                corr_test_pred_true = np.corrcoef(test_mean_flatten_pred, test_mean_flatten_true)[0,1]

                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                axes = axes.flatten()
                img0 = axes[0].imshow(train_mean_flatten_pred_mat, aspect='auto', vmin=-0.8, vmax=0.8, cmap="Spectral_r")
                axes[0].set_title("TRAIN PRED MEAN")
                img1 = axes[1].imshow(train_mean_flatten_true_mat, aspect='auto', vmin=-0.8, vmax=0.8, cmap="Spectral_r")
                axes[1].set_title("TRAIN TRUE MEAN")
                img2 = axes[2].imshow((train_mean_flatten_pred_mat - train_mean_flatten_true_mat), aspect='auto', cmap="Spectral_r")
                axes[2].set_title(f"PRED - TRUE, rho={corr_train_pred_true:.3f}")

                img3 = axes[3].imshow(test_mean_flatten_pred_mat, aspect='auto', vmin=-0.8, vmax=0.8, cmap="Spectral_r")
                axes[3].set_title("TEST PRED MEAN")
                img4 = axes[4].imshow(test_mean_flatten_true_mat, aspect='auto', vmin=-0.8, vmax=0.8, cmap="Spectral_r")
                axes[4].set_title("TEST TRUE MEAN")
                img5 = axes[5].imshow((test_mean_flatten_pred_mat - test_mean_flatten_true_mat), aspect='auto', cmap="Spectral_r")
                axes[5].set_title(f"PRED - TRUE, rho={corr_test_pred_true:.3f}")

                plt.colorbar(img0, ax=axes[0])
                plt.colorbar(img1, ax=axes[1])
                plt.colorbar(img2, ax=axes[2])
                plt.colorbar(img3, ax=axes[3])
                plt.colorbar(img4, ax=axes[4])
                plt.colorbar(img5, ax=axes[5])
                plt.tight_layout
                # directory = root + '/images/' + datasets + '/' + translation  + '/' + model_type +'/' + version + '/'+ model_details + '/' + model_test_type #+ '/ABCD_train_HCPYA_test'# Replace with your target directory
                filename = f'/train_test_pred_true_averageNetmats_comparison.{img_extension}'
                print(f'Saving to path:{directory}')
                plt.savefig(directory + filename, format=img_extension)
                # plt.show()
                plt.close()

            # %%
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes = axes.flatten()

            axes[0].hist((train_truth_holder).flatten(), bins=100, color='red', label="target", alpha=0.5)
            axes[0].hist((train_pred_holder).flatten(), bins=100, color='blue', label="pred", alpha=0.5)
            # sns.kdeplot(data=train_truth_holder.flatten(), ax=axes[0])
            # sns.kdeplot(data=train_pred_holder.flatten(), ax=axes[0])
            axes[0].set_title("TRAIN Hist: target, pred")
            axes[0].legend()

            axes[1].hist((test_truth_holder).flatten(), bins=100, color='red', label="target", alpha=0.5)
            axes[1].hist((test_pred_holder).flatten(), bins=100, color='blue', label="pred", alpha=0.5)
            axes[1].set_title("TEST Hist: target, pred")
            axes[1].legend()

            filename = f"/hist_realVals_truth_pred_schff100.{img_extension}"
            print(f'Saving to path:{directory}')
            plt.suptitle('Hist. of schf values across channels')
            plt.savefig(directory + filename, format=img_extension)
            plt.tight_layout()
            # plt.show()
            plt.close()

            # %%
            # %%
            plot_name = ["trainavg", "testavg"]
            for gg in range(len(plot_name)):
                ''' Below Plots across train and test Predi Targeti '''
                train_mean = train_mean_flatten_pred
                if gg == 0:
                    test_mean = train_mean # first, demean/add_mean from train data
                else:
                    test_mean = test_mean_flatten_pred #second, demean/add_mean from test data
                    
                sub_dim = 0
                chnl_dim = 1 # leaving as "channel" but refers to netmat upper tri
                if "demean" in version: # predictions are demeaned, so no need to demean further. Instead, add to original.
                    print("VERSION has demean, so predictions already are demenaed. No need to subtract mean. Instead, adding it to original.")
                    flag_corr1_mean_condition = 0 # subtract nothing
                    flag_corr2_mean_condition = train_mean # add mean
                    flag_corr3_mean_condition = 0 # subtract nothing
                    flag_corr4_mean_condition = test_mean # add mean
                    flag_corr5_mean_condition = 0
                    flag_corr6_mean_condition = train_mean_flatten_true
                    flag_corr7_mean_condition = 0
                    flag_corr8_mean_condition = test_mean_flatten_true
                elif ica_reconstuction:
                    print("VERSION is ICA reconstructions")
                    flag_corr1_mean_condition = 0 # subtract mean
                    flag_corr2_mean_condition = 0 # add nothing
                    flag_corr3_mean_condition = 0 # subtract nothing
                    flag_corr4_mean_condition = 0
                    flag_corr5_mean_condition = 0
                    flag_corr6_mean_condition = 0
                    flag_corr7_mean_condition = 0
                    flag_corr8_mean_condition = 0
                else: # predictions are raw/not_demean so need to demean to visualize
                    print("VERSION is raw/not_demean so needs to be subtracted.")
                    flag_corr1_mean_condition = train_mean # subtract mean
                    flag_corr2_mean_condition = 0 # add nothing
                    flag_corr3_mean_condition = test_mean # subtract nothing
                    flag_corr4_mean_condition = 0
                    flag_corr5_mean_condition = train_mean_flatten_true
                    flag_corr6_mean_condition = 0
                    flag_corr7_mean_condition = test_mean_flatten_true
                    flag_corr8_mean_condition = 0

                train_rho_chnl = np.zeros((train_truth_holder.shape[sub_dim], 1), dtype=float)
                train_rho_chnl_org = np.zeros((train_truth_holder.shape[sub_dim], 1), dtype=float)
                for i in range(train_truth_holder.shape[sub_dim]): # for each subj
                    corr = np.corrcoef((train_truth_holder[i, :] - flag_corr1_mean_condition), (train_pred_holder[i, :] - flag_corr1_mean_condition))[0,1]
                    corr2 = np.corrcoef((train_truth_holder[i, :] + flag_corr2_mean_condition), (train_pred_holder[i, :] + flag_corr2_mean_condition))[0,1]
                    # print(corr2)
                    train_rho_chnl[i] = corr
                    train_rho_chnl_org[i] = corr2

                test_rho_chnl = np.zeros((test_truth_holder.shape[sub_dim], 1), dtype=float)
                test_rho_chnl_org = np.zeros((test_truth_holder.shape[sub_dim], 1), dtype=float)
                for i in range(test_truth_holder.shape[sub_dim]):
                    corr = np.corrcoef((test_truth_holder[i, :] - flag_corr3_mean_condition), (test_pred_holder[i, :] - flag_corr3_mean_condition))[0,1]
                    corr2 = np.corrcoef((test_truth_holder[i, :] + flag_corr4_mean_condition), (test_pred_holder[i, :] + flag_corr4_mean_condition))[0,1]
                    # print(corr)
                    test_rho_chnl[i] = corr
                    test_rho_chnl_org[i] = corr2

                print(train_rho_chnl.shape)

                #save for later use
                full_predtarget_corr = [train_rho_chnl, train_rho_chnl_org, test_rho_chnl, test_rho_chnl_org]
                full_predtarget_corr = np.asanyarray(full_predtarget_corr, dtype=object)
                # np.save(f'{directory}/list_of_models_predtarget_rho.npy', full_predtarget_corr)

                ''' Below Plots across train and test PrediPredj '''
                sub_uptri = ((train_truth_holder.shape[sub_dim] * (train_truth_holder.shape[sub_dim]-1)) / 2)
                same_pred_train_rho_chnl = np.zeros((int(sub_uptri), 1), dtype=float) #[sub*sub upper tri x 1]
                same_pred_train_rho_chnl_org = np.zeros((int(sub_uptri), 1), dtype=float)
                print(f"Same Pred rho shape: {same_pred_train_rho_chnl.shape}")
                corr = np.corrcoef(((train_pred_holder - flag_corr1_mean_condition))) #it does subtract every subjevct by same constant avg vector
                corr2 = np.corrcoef(((train_pred_holder + flag_corr2_mean_condition)))
                same_pred_train_rho_chnl = corr[np.triu_indices_from(corr, k=1)] #upptri of subjxsubj correlation mat
                same_pred_train_rho_chnl_org = corr2[np.triu_indices_from(corr2, k=1)] # gets squeeze but same size
                print(f"Same Pred rho shape: {same_pred_train_rho_chnl.shape}")

                sub_uptri = ((test_truth_holder.shape[sub_dim] * (test_truth_holder.shape[sub_dim]-1)) / 2)
                same_pred_test_rho_chnl = np.zeros((int(sub_uptri), 1), dtype=float)
                same_pred_test_rho_chnl_org = np.zeros((int(sub_uptri), 1), dtype=float)
                print(same_pred_test_rho_chnl.shape)
                corr = np.corrcoef((test_pred_holder - flag_corr3_mean_condition))#[0,1]
                corr2 = np.corrcoef(((test_pred_holder + flag_corr4_mean_condition)))
                same_pred_test_rho_chnl = corr[np.triu_indices_from(corr, k=1)]
                same_pred_test_rho_chnl_org = corr2[np.triu_indices_from(corr2, k=1)]
                print(same_pred_test_rho_chnl.shape)

                ''' Now we plot Truei Truej '''
                sub_uptri = ((train_truth_holder.shape[sub_dim] * (train_truth_holder.shape[sub_dim]-1)) / 2)
                same_true_train_rho_chnl = np.zeros((int(sub_uptri), 1), dtype=float) #[sub*sub upper tri x 1]
                same_true_train_rho_chnl_org = np.zeros((int(sub_uptri), 1), dtype=float)
                print(f"Same True rho shape: {same_true_train_rho_chnl.shape}")
                corr = np.corrcoef((train_truth_holder - flag_corr5_mean_condition))#[0,1]
                corr2 = np.corrcoef(((train_truth_holder + flag_corr6_mean_condition)))
                same_true_train_rho_chnl = corr[np.triu_indices_from(corr, k=1)] #upptri of subjxsubj correlation mat
                same_true_train_rho_chnl_org = corr2[np.triu_indices_from(corr2, k=1)] # gets squeeze but same size
                print(f"Same True rho shape: {same_true_train_rho_chnl.shape}")

                sub_uptri = ((test_truth_holder.shape[sub_dim] * (test_truth_holder.shape[sub_dim]-1)) / 2)
                same_true_test_rho_chnl = np.zeros((int(sub_uptri), 1), dtype=float)
                same_true_test_rho_chnl_org = np.zeros((int(sub_uptri), 1), dtype=float)
                print(same_true_test_rho_chnl.shape)
                corr = np.corrcoef((test_truth_holder - flag_corr7_mean_condition))#[0,1]
                corr2 = np.corrcoef(((test_truth_holder + flag_corr8_mean_condition)))
                same_true_test_rho_chnl = corr[np.triu_indices_from(corr, k=1)]
                same_true_test_rho_chnl_org = corr2[np.triu_indices_from(corr2, k=1)]
                print(same_true_test_rho_chnl.shape)

                ## Below is looking into demean rho(target,pred)
                # %%
                fig, axes = plt.subplots(3, 1, figsize=(12, 24))
                axes=axes.flatten()
                # Flatten arrays if needed
                tr_demean_vals = train_rho_chnl.flatten()
                tr_orig_vals = train_rho_chnl_org.flatten()
                te_demean_vals = test_rho_chnl.flatten()
                te_orig_vals = test_rho_chnl_org.flatten()

                list_tr_demean = [np.mean(tr_demean_vals), np.std(tr_demean_vals)]
                list_tr_orig = [np.mean(tr_orig_vals), np.std(tr_orig_vals)]
                list_te_demean = [np.mean(te_demean_vals), np.std(te_demean_vals)]
                list_te_orig = [np.mean(te_orig_vals), np.std(te_orig_vals)]

                # print(f"Mean(std): \nTrain_dmean:{list_tr_demean[0]:.3f}({list_tr_demean[1]:.3f}) Train_Orig:{list_tr_orig[0]:.3f}({list_tr_orig[1]:.3f}) \nTest_dmean:{list_te_demean[0]:.3f}({list_te_demean[1]:.3f}) Test_Orig:{list_te_orig[0]:.3f}({list_te_orig[1]:.3f})")
                table_of_violin_plot = pd.DataFrame([list_tr_demean, list_tr_orig, list_te_demean, list_te_orig],
                                    index=["Train_dmean", "Train_original", "Test_dmean", "Test_original"],
                                    columns=["mean_of_violin", "std_of_violin"])
                print(table_of_violin_plot)
                table_of_violin_plot.to_csv(directory+f'/ViolinPlotTable_true_pred_{plot_name[gg]}.csv') 

                # Build long-form DataFrame
                df = pd.DataFrame({ # its values per row and col==1 and its group and is different based on concat
                    "value": np.concatenate([tr_demean_vals, tr_orig_vals, te_demean_vals, te_orig_vals]),
                    "group": ["TRAIN_dmn"] * len(tr_demean_vals) + ["TRAIN_org"] * len(tr_orig_vals) + ["TEST_dmn"] * len(te_demean_vals) + ["TEST_org"] * len(te_orig_vals)
                })
                df.to_csv(directory+f'/allsubjects_individual_true_pred_{plot_name[gg]}.csv')
                # sns.violinplot(data=df, x="group", y="value", inner="point", ax=axes[0])
                # axes[0].set_title("Violin Plot of Rho by Group $Pred_i True_i$")
                # axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
                i=0
                group_stats = df.groupby("group")["value"].agg(["mean", "std"]).reset_index()
                custome_colors=["crimson","lightcoral","steelblue","lightskyblue"]
                # Plot bars with error bars
                axes[i].bar(
                    group_stats["group"],
                    group_stats["mean"],
                    yerr=group_stats["std"],
                    capsize=5,               # small caps at the end of error bars
                    color=custome_colors,
                    edgecolor="black"
                )
                axes[i].set_title("Demeaned and Original Matrix, $True_i Pred_j$")
                axes[i].axhline(0, color='black', linestyle='--', linewidth=1)
                axes[i].set_ylabel("Correlation")
                # axes[i].set_xlabel("group")

                '''Below is for plotting pred i pred j'''
                tr_demean_vals = same_pred_train_rho_chnl.flatten()
                tr_orig_vals = same_pred_train_rho_chnl_org.flatten()
                te_demean_vals = same_pred_test_rho_chnl.flatten()
                te_orig_vals = same_pred_test_rho_chnl_org.flatten()

                list_tr_demean = [np.mean(tr_demean_vals), np.std(tr_demean_vals)]
                list_tr_orig = [np.mean(tr_orig_vals), np.std(tr_orig_vals)]
                list_te_demean = [np.mean(te_demean_vals), np.std(te_demean_vals)]
                list_te_orig = [np.mean(te_orig_vals), np.std(te_orig_vals)]

                # print(f"Mean(std): \nTrain_dmean:{list_tr_demean[0]:.3f}({list_tr_demean[1]:.3f}) Train_Orig:{list_tr_orig[0]:.3f}({list_tr_orig[1]:.3f}) \nTest_dmean:{list_te_demean[0]:.3f}({list_te_demean[1]:.3f}) Test_Orig:{list_te_orig[0]:.3f}({list_te_orig[1]:.3f})")
                table_of_violin_plot = pd.DataFrame([list_tr_demean, list_tr_orig, list_te_demean, list_te_orig],
                                    index=["Train_dmean", "Train_original", "Test_dmean", "Test_original"],
                                    columns=["mean_of_violin", "std_of_violin"])
                print(table_of_violin_plot)
                table_of_violin_plot.to_csv(directory+f'/ViolinPlotTable_pred_pred_{plot_name[gg]}.csv') 
                
                # Build long-form DataFrame
                df = pd.DataFrame({ # its values per row and col==1 and its group and is different based on concat
                    "value": np.concatenate([tr_demean_vals, tr_orig_vals, te_demean_vals, te_orig_vals]),
                    "group": ["$TRAIN_{dmn}$"] * len(tr_demean_vals) + ["$TRAIN_{org}$"] * len(tr_orig_vals) + ["$TEST_{dmn}$"] * len(te_demean_vals) + ["$TEST_{org}$"] * len(te_orig_vals)
                })
                # sns.violinplot(data=df, x="group", y="value", inner="point", ax=axes[1])
                # axes[1].set_title("Violin Plot of Rho by Group $Pred_i Pred_j$")
                # axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
                group_stats = df.groupby("group")["value"].agg(["mean", "std"]).reset_index()
                # Plot bars with error bars
                axes[i+1].bar(
                    group_stats["group"],
                    group_stats["mean"],
                    yerr=group_stats["std"],
                    capsize=5,               # small caps at the end of error bars
                    color=custome_colors,
                    edgecolor="black"
                )
                axes[i+1].set_title("Demeaned and Original Matrix, $Pred_i Pred_j$")
                axes[i+1].axhline(0, color='black', linestyle='--', linewidth=1)
                axes[i+1].set_ylabel("Correlation")
                # axes[i+1].set_xlabel("group")

                '''We do the same but this time True i True j'''
                '''Below is for plotting pred i pred j'''
                tr_demean_vals = same_true_train_rho_chnl.flatten()
                tr_orig_vals = same_true_train_rho_chnl_org.flatten()
                te_demean_vals = same_true_test_rho_chnl.flatten()
                te_orig_vals = same_true_test_rho_chnl_org.flatten()

                list_tr_demean = [np.mean(tr_demean_vals), np.std(tr_demean_vals)]
                list_tr_orig = [np.mean(tr_orig_vals), np.std(tr_orig_vals)]
                list_te_demean = [np.mean(te_demean_vals), np.std(te_demean_vals)]
                list_te_orig = [np.mean(te_orig_vals), np.std(te_orig_vals)]

                # print(f"Mean(std): \nTrain_dmean:{list_tr_demean[0]:.3f}({list_tr_demean[1]:.3f}) Train_Orig:{list_tr_orig[0]:.3f}({list_tr_orig[1]:.3f}) \nTest_dmean:{list_te_demean[0]:.3f}({list_te_demean[1]:.3f}) Test_Orig:{list_te_orig[0]:.3f}({list_te_orig[1]:.3f})")
                table_of_violin_plot = pd.DataFrame([list_tr_demean, list_tr_orig, list_te_demean, list_te_orig],
                                    index=["Train_dmean", "Train_original", "Test_dmean", "Test_original"],
                                    columns=["mean_of_violin", "std_of_violin"])
                print(table_of_violin_plot)
                table_of_violin_plot.to_csv(directory+f'/ViolinPlotTable_true_true_{plot_name[gg]}.csv')

                # Build long-form DataFrame
                df = pd.DataFrame({ # its values per row and col==1 and its group and is different based on concat
                    "value": np.concatenate([tr_demean_vals, tr_orig_vals, te_demean_vals, te_orig_vals]),
                    "group": ["$TRAIN_{dmn}$"] * len(tr_demean_vals) + ["$TRAIN_{org}$"] * len(tr_orig_vals) + ["$TEST_{dmn}$"] * len(te_demean_vals) + ["$TEST_{org}$"] * len(te_orig_vals)
                })
                # sns.violinplot(data=df, x="group", y="value", inner="point", ax=axes[2])
                # axes[2].set_title("Bar Plot of Rho by Group $True_i True_j$")
                # axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
                group_stats = df.groupby("group")["value"].agg(["mean", "std"]).reset_index()
                # Plot bars with error bars
                axes[i+2].bar(
                    group_stats["group"],
                    group_stats["mean"],
                    yerr=group_stats["std"],
                    capsize=5,               # small caps at the end of error bars
                    color=custome_colors,
                    edgecolor="black"
                )
                axes[i+2].set_title("Demeaned and Original Matrix, $True_i True_j$")
                axes[i+2].axhline(0, color='black', linestyle='--', linewidth=1)
                axes[i+2].set_ylabel("Correlation")
                # axes[i+2].set_xlabel("group")

                filename = f'/model_rho_performance_across_all_{plot_name[gg]}.{img_extension}'
                print(f'Saving to path:{directory}')
                # Save the figure
                plt.savefig(directory + filename, format=img_extension)
                # plt.show()
                plt.close()

            # %%
            # correlation squares
            tr_big_corr_mat = np.load(f"{root_data}/tr_big_corr_matrix.npy")
            te_big_corr_mat = np.load(f"{root_data}/te_big_corr_matrix.npy")
            print(f"Shapes of matrices. tr:{tr_big_corr_mat.shape} te:{te_big_corr_mat.shape}")

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes = axes.flatten()

            img0 = axes[0].imshow(tr_big_corr_mat, aspect='auto', vmin=0, vmax=1, cmap="Spectral_r")
            axes[0].set_title("TRAIN across Subj RHO")
            img1 = axes[1].imshow(te_big_corr_mat, aspect='auto', vmin=0, vmax=1, cmap="Spectral_r")
            axes[1].set_title("TEST across Subj RHO")

            plt.colorbar(img0, ax=axes[0])
            plt.colorbar(img1, ax=axes[1])
            plt.tight_layout
            filename = f'/acrosssubj_train_test_mean_bigRHOmat.{img_extension}'
            print(f'Saving to path:{directory}')
            # Save the figure
            plt.savefig(directory + filename, format=img_extension)
            # plt.show()
            plt.close()

            # %%
            if not ica_reconstuction:
                # parcellation = 100

                # find MAX rho performance DEMEAN, for original use train_rho_chnl_org
                max_ii = np.argmax(train_rho_chnl)  # best performance idx, this is a 1d vector with corr per subject
                max_truth_values = train_truth_holder[max_ii]
                max_pred_values = train_pred_holder[max_ii]
                # find MIN rho performance
                min_ii = np.argmin(train_rho_chnl)  # worst performance idx
                min_truth_values = train_truth_holder[min_ii]
                min_pred_upper_values = train_pred_holder[min_ii]
                # now make into netmats to visualize
                tr_truth_matrix_max = make_netmat(max_truth_values, from_parcellation)
                tr_pred_matrix_max = make_netmat(max_pred_values, from_parcellation)
                tr_truth_matrix_min = make_netmat(min_truth_values, from_parcellation)
                tr_pred_matrix_min = make_netmat(min_pred_upper_values, from_parcellation)

                print(f"TRAIN correlations: MAX:{np.max(train_rho_chnl)} | MIN:{np.min(train_rho_chnl)}")
                print(f"TEST correlations: MAX:{np.max(test_rho_chnl)} | MIN:{np.min(test_rho_chnl)}")

                # find MAX rho performance DEMEAN, for original use train_rho_chnl_org
                max_ii = np.argmax(test_rho_chnl)  # best performance idx
                max_truth_values = test_truth_holder[max_ii]
                max_pred_values = test_pred_holder[max_ii]
                # find MIN rho performance
                min_ii = np.argmin(test_rho_chnl)  # worst performance idx
                min_truth_values = test_truth_holder[min_ii]
                min_pred_upper_values = test_pred_holder[min_ii]
                # now make into netmats to visualize
                te_truth_matrix_max = make_netmat(max_truth_values, from_parcellation)
                te_pred_matrix_max = make_netmat(max_pred_values, from_parcellation)
                te_truth_matrix_min = make_netmat(min_truth_values, from_parcellation)
                te_pred_matrix_min = make_netmat(min_pred_upper_values, from_parcellation)

                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                list_of_matrices = [
                    tr_truth_matrix_max,
                    tr_pred_matrix_max,
                    tr_truth_matrix_min,
                    tr_pred_matrix_min,
                ]
                list_of_titles = ["True MAX", f"Pred MAX(rho={np.max(train_rho_chnl):.2f})", "True MIN", f"Pred MIN(rho={np.min(train_rho_chnl):.2f})"]
                for ii in range(0, 4):
                    img0 = axes[ii].imshow(
                        list_of_matrices[ii], aspect="auto", vmin=-0.8, vmax=0.8, cmap="Spectral_r")
                    axes[ii].set_title(f"{list_of_titles[ii]}")
                    plt.colorbar(img0, ax=axes[ii])

                plt.suptitle("TRAIN Subjects")

                plt.tight_layout
                filename = f"/max_min_example_netmats_train.{img_extension}"
                print(f"Saving to path:{directory}")
                # Save the figure
                plt.savefig(directory + filename, format=img_extension)
                # plt.show()
                plt.close()

                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                list_of_matrices = [
                    te_truth_matrix_max,
                    te_pred_matrix_max,
                    te_truth_matrix_min,
                    te_pred_matrix_min,
                ]
                list_of_titles = ["True MAX", f"Pred MAX(rho={np.max(test_rho_chnl):.2f})", "True MIN", f"Pred MIN(rho={np.min(test_rho_chnl):.2f})"]
                for ii in range(0, 4):
                    img0 = axes[ii].imshow(
                        list_of_matrices[ii], aspect="auto", vmin=-0.8, vmax=0.8, cmap="Spectral_r")
                    axes[ii].set_title(f"{list_of_titles[ii]}")
                    plt.colorbar(img0, ax=axes[ii])

                plt.suptitle("TEST Subjects")

                plt.tight_layout
                filename = f"/max_min_example_netmats_test.{img_extension}"
                print(f"Saving to path:{directory}")
                # Save the figure
                plt.savefig(directory + filename, format=img_extension)
                # plt.show()
                plt.close()

                train_true_netamts = train_truth_holder #+ train_mean_flatten_pred
                train_pred_netmats = train_pred_holder #+ train_mean_flatten_true
                test_true_netamts = test_truth_holder #+ test_mean_flatten_pred
                test_pred_netamts = test_pred_holder #+ test_mean_flatten_true

                a = make_nemat_allsubj(train_true_netamts,from_parcellation)
                b = make_nemat_allsubj(train_pred_netmats,from_parcellation)
                c = make_nemat_allsubj(test_true_netamts,from_parcellation)
                d = make_nemat_allsubj(test_pred_netamts,from_parcellation)

                # concat_train = np.concatenate((a,b), axis=0)
                # print(concat_train.shape)
                # if datasets == "HCYA":
                # subs2view = [0, 100, 200, 300, 500, 700]
                subs2view=[]
                for kk in range(6):
                    x = np.random.randint(0,a.shape[0])
                    subs2view.append(x)
                fig, axes = plt.subplots(3, 4, figsize=(20, 12))
                axes = axes.flatten()
                xx = 0
                for i in range(6):
                    axes[xx].imshow(a[subs2view[i]], aspect="auto", vmin=-0.5, vmax=0.5, cmap="Spectral_r")
                    axes[xx].set_title(f"True{i}")
                    axes[xx+1].imshow(b[subs2view[i]], aspect="auto", vmin=-0.5, vmax=0.5,cmap="Spectral_r")
                    axes[xx+1].set_title(f"Pred{i}")

                    xx += 2
                plt.suptitle("TRAIN")
                plt.tight_layout()
                # plt.show()
                filename = f"/examples_netmats_train.{img_extension}"
                print(f"Saving to path:{directory}")
                # Save the figure
                plt.savefig(directory + filename, format=img_extension)
                # plt.show()
                plt.close()

                # subs2view = [0, 5, 10, 20, 30, 40]
                subs2view=[]
                for kk in range(6):
                    x = np.random.randint(0,c.shape[0])
                    subs2view.append(x)
                    
                fig, axes = plt.subplots(3, 4, figsize=(20, 12))
                axes = axes.flatten()
                xx = 0
                for i in range(6):
                    axes[xx].imshow(c[subs2view[i]], aspect="auto", vmin=-0.5, vmax=0.5,cmap="Spectral_r")
                    axes[xx].set_title(f"True{i}")
                    axes[xx+1].imshow(d[subs2view[i]], aspect="auto", vmin=-0.5, vmax=0.5,cmap="Spectral_r")
                    axes[xx+1].set_title(f"Pred{i}")

                    xx += 2
                plt.suptitle("TEST")
                plt.tight_layout()
                # plt.show()
                filename = f"/examples_netmats_test.{img_extension}"
                print(f"Saving to path:{directory}")
                # Save the figure
                plt.savefig(directory + filename, format=img_extension)
                # plt.show()
                plt.close()
            
            if ica_reconstuction:
                N, all_v  = test_truth_holder.shape
                # C=1
                P=320
                V=153
                overwrite_recon_sphere_flag=True
                numss = np.asarray([8, 1, 454, 183, 3, 9]) #chosen post verification, got from viz_scienceadv_figures.ypynb script for VIZ and DMN channel
                # print(numss) # order is top(viz,dmn), med(viz, dmn), bottom(viz, dmn)
                for ii in range(len(numss)): #first five no order most are really good reconstruction so all good
                    #get real one and pred one
                    idx_sub = numss[ii]
                    ica_component_i_pred = test_pred_holder[idx_sub]
                    ica_component_i_true = test_truth_holder[idx_sub]
                    # exapnd dims to include channel
                    # ica_component_i_pred=np.expand_dims(ica_component_i_pred,1) #add channel dim
                    # ica_component_i_true=np.expand_dims(ica_component_i_true,1) #add channel dim
                    pred_reshaped = np.reshape(ica_component_i_pred, (1, P, V))
                    true_reshaped = np.reshape(ica_component_i_true, (1, P, V))
                    print(f"{numss[ii]:04d}")
                    if overwrite_recon_sphere_flag:
                        matrix_to_mesh(input_mat=pred_reshaped, tri_indices_ico6subico2_fpath=f"{root}/utils/surfaces/triangle_indices_ico_6_sub_ico_2.csv", out_fpath=f"{root}/utils/surfaces/test_pred_L_chnl-{chnl_icarecon+1:02d}_sub-{idx_sub:04d}_ico6")
                        matrix_to_mesh(input_mat=true_reshaped, tri_indices_ico6subico2_fpath=f"{root}/utils/surfaces/triangle_indices_ico_6_sub_ico_2.csv", out_fpath=f"{root}/utils/surfaces/test_true_L_chnl-{chnl_icarecon+1:02d}_sub-{idx_sub:04d}_ico6")    
                    else: # don't overwrite, only do ones that have not been done if any
                        if not os.path.isfile(f"{root}/utils/test_pred_L_sub-{i+1:04d}_ico6.shape.gii"): # makes both so only need to check if one is there or not
                            matrix_to_mesh(input_mat=pred_reshaped, tri_indices_ico6subico2_fpath=f"{root}/utils/surfaces/triangle_indices_ico_6_sub_ico_2.csv", out_fpath=f"{root}/utils/surfaces/test_pred_L_chnl-{chnl_icarecon+1:02d}_sub-{idx_sub:04d}_ico6")
                            matrix_to_mesh(input_mat=true_reshaped, tri_indices_ico6subico2_fpath=f"{root}/utils/surfaces/triangle_indices_ico_6_sub_ico_2.csv", out_fpath=f"{root}/utils/surfaces/test_true_L_chnl-{chnl_icarecon+1:02d}_sub-{idx_sub:04d}_ico6")





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