import sys
sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

import numpy as np
import torch
from utils.utils import *
from utils.functions_kraken_loss import *

# =========================================== BGTLN / BGTLN_VAE - ecn only then Linear layer =========================================== #
'''
Below are train functions used to test and explore which training regimes help our translation task. We have skewloss which is taken from this: https://ieeexplore.ieee.org/abstract/document/9997546/citations#full-text-header
Idea is that when using a loss function like MSE, both over predictions and under predictions are equal and fixed similarly. But there are times where you want to skew that s.t. you penalize some values and maybe not others. Example,
if used on somethign like a correlation, then ideally shifts that to +1 correlation between target and predictions. We also have kraken losses taken from: https://www.biorxiv.org/content/10.1101/2024.04.12.589274v1
which attempts to translate across netmat -> netmat where they can be the same (reconstrcution) or different (translation) by having loss functions that capture seperate parts of the model. There are latent LOSS and recon LOSS, s.t. kraken hopes
to optimize the latent space and make subject netmats across flavors (kidns of netmats) more similar for a single subject than across i.e. learn to make same subject info more like itself than with others thereby trying to make a correlation matrix
1 on the diagonal (perfect target with pred relation ship) and the off-diagional close to 0 or less than diagonal. Example: I have ICAd15, so I and you have 15 channels in our dataset. Even though ICA1 is the same for you and me (assumingf we are from
same dataset and dual regression was done with us both and all other participants), my ICA1 and my ICA2 ... and so on should be more similar than to your ICA1 and ICA2 and so on because they are from my individual brain. Of course, ICA1 me and ICA1 you will be more
similar than ICA1 me ICA5 you, but this big diagonal matrix 
'''

# def train_skewloss(model, krak_mse_weight, krak_latent_weight, krak_corrI_weight, train_loader, mean_train_label, device, optimizer, lamda_value, skew_weight, VAE_flag=False):
#     '''
#         This one does kraken losses + skewloss. Kraken loss is done by subjxsuvj matrices so need BxVEC where VEC is the flatten vector of all your dimensions.
#         In this case dims are Channel Patches and Verteces.
#     '''
    
#     assert 1 < lamda_value < 2, "Skewloss lambda values needs to be between [+1, +2]"
#     model.train()

#     # targets_ = [] # comment for now because useful for how preds evolve in time, but a huge vallue to "store" PER EPOCH nah not right now
#     # preds_ = []
#     tr_mae_subs = []
#     tr_mse_subs = []
#     tr_corr_subs_demean = []
#     tr_corr_subs_org = []
#     tr_epoch_loss = 0
#     for i, data in enumerate(train_loader): # for loop that goes over each batch in training loop
#         inputs, targets = data[0].to(device), data[1].to(device).squeeze() #mesh,uppertri_vectorized
#         optimizer.zero_grad(set_to_none=True) # True by default anyway

#         if VAE_flag:
#             pred, latent, latent_logvar = model(inputs) #latent = z_mu, latent_sigma=sigma_mu
#             kld_loss = -0.5 * torch.sum(1 + latent_logvar - (latent ** 2) - latent_logvar.exp(), dim = 1) # dim1 bc each subject has own loss
#             kld_loss = torch.mean(kld_loss, dim=0) #now average across subjecrs to get single loss for all in batch
#         else:
#             pred, latent = model(inputs)

#         # Output Losses
#         Lr_corrI = correye(targets, pred) # identity matrix where diagonal is optimized and off-daig is reduced (more within, less between)
#         Lr_mse = torch.nn.MSELoss()(targets, pred) # MSE should be low
#         Lr_marg = distance_loss(targets, pred, neighbor=True) # predicted X should be far from nearet ground truth X (for a different subject)

#         # Latent Space Losses
#         Lz_corrI = correye(latent, latent) # correlation matrix of latent space should be low off diagonal
#         Lz_dist = distance_loss(latent, latent, neighbor=False) # mean intersubject altent space distances should be high

#         Lr =  krak_corrI_weight*Lr_corrI + Lr_marg + (krak_mse_weight * Lr_mse) # weighting MSE with 100,000 (1000 from Krakencoder), Fyzeen OG is 50k SDNR
#         Lz = Lz_corrI + Lz_dist
        
#         # skew loss
#         # lamda = lamda_value # ideal lambda is between 1 and 2, so 1.5 is therefor arbitrary for now can be optimized for our purposes       
#         skewed_loss = torch.exp(torch.sign(targets - pred) * lamda_value)
#         skewed_loss = skew_weight * torch.mean(skewed_loss) #mean skew, then that is coefficient to original loss for batch mean

#         # make into CPU numpy variables and detach from pytorch graph, notice no .cpu() needed because these modes are expected to be run w CPUs, might need change if done on GPU        
#         pred = pred.detach().numpy()
#         targets = targets.detach().numpy()

#         # pred_nan = np.isnan(pred).sum()
#         # pred_inf = np.isinf(pred).sum()
#         # print(f"PRED CHECK INF/NAN: {pred_nan} -- {pred_inf}")

#         # MAE and MSE metrics per training iteration/batch
#         mae = np.mean( np.abs((targets - pred)), axis=0, keepdims=True) #LA.norm((targets - pred), ord=1, axis=0) #np.abs( (targets - pred) ) # |y - y_hat|
#         tr_mae_subs.append(mae)
#         mse = np.mean( (targets - pred)**2 , axis=0, keepdims=True) #(LA.norm((targets - pred), ord=2, axis=0)) ** 2 #needs to be **2 becasue norm-2 squareroots result. np.mean( (targets - pred)**2 ) #(y - y_hat)^2
#         tr_mse_subs.append(mse) # 1xCxPxV, MSE for this batch

#         # important points!! saving predicted values and same values plus training mean, demean_corr and orig_corr respectively
#         tr_corr_demean = np.corrcoef(targets, pred)
#         # print(f"tr_corr_demean shape: {tr_corr_demean.shape}")
#         split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
#         top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
#         lower_tri_of_qaudrant_demean = get_lower_tris(top_right_quad)
#         # print(f"tril shape: {lower_tri_of_qaudrant_demean.shape}")
#         tr_corr_subs_demean.append(lower_tri_of_qaudrant_demean) #upper tri rho values for this batch (correlating targets and pred)

#         tr_corr_org = np.corrcoef((targets + mean_train_label), (pred + mean_train_label))# going to be low-ish cause 256->mesh size sphere but curious
#         split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
#         top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
#         lower_tri_of_qaudrant_org = get_lower_tris(top_right_quad)
#         tr_corr_subs_org.append(lower_tri_of_qaudrant_org)

#         # update loss and use optimizer, direct connection of LR_MSE like a skip connection
#         if VAE_flag:
#             loss = ((Lr + (krak_latent_weight * Lz) + Lr_mse + kld_loss) * skewed_loss) # loss uses demean so add that
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
#         else:
#             loss = ((Lr + (krak_latent_weight * Lz) + Lr_mse) * skewed_loss) # loss uses demean so add that
#             # torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)

#         loss.backward()
#         tr_epoch_loss += loss.item()

#         optimizer.step()

#         # targets_.append(demean_targets.cpu().numpy())
#         # preds_.append(demean_pred.cpu().detach().numpy())

#     across_sub_mae_mean = np.mean(tr_mae_subs) # across all elements, so no axis ==> mean of flatten mat->vector, so across all subs and channels and patches and verteces
#     across_sub_mae_std = np.std(tr_mae_subs)
#     across_sub_mse_mean = np.mean(tr_mse_subs)
#     across_sub_mse_std = np.std(tr_mse_subs)
#     # because of batching, some in the list are different size so make into whole array
#     upto_n_minus1 = np.asarray(tr_corr_subs_demean[:-1]).squeeze() # all upto last item, do that seperate then concat
#     upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
#     n_minus_1 = np.asarray(tr_corr_subs_demean[-1])
#     tr_corr_demean_flat = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
#     across_sub_corr_demean = np.mean(tr_corr_demean_flat)
#     across_sub_corr_demean_std = np.std(tr_corr_demean_flat)

#     # same for original corr values
#     upto_n_minus1 = np.asarray(tr_corr_subs_org[:-1]).squeeze() # all upto last item, do that seperate then concat
#     upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
#     n_minus_1 = np.asarray(tr_corr_subs_org[-1])
#     tr_corr_org_flat = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
#     across_sub_corr_org = np.mean(tr_corr_org_flat)
#     across_sub_corr_org_std = np.std(tr_corr_org_flat)
    
#     return tr_epoch_loss, across_sub_mae_mean, across_sub_mae_std, across_sub_mse_mean, across_sub_mse_std, across_sub_corr_demean, across_sub_corr_demean_std, across_sub_corr_org, across_sub_corr_org_std

def train_MSE_skewloss(model, krak_mse_weight, krak_latent_weight, krak_corrI_weight, rho_loss_weight, train_loader, mean_train_label, device, optimizer, VAE_flag=False, netmat_prep_choice=None):
    '''
    Train function only using MSE and Skewloss (NO KRAKEN) to see effects of skewloss on original basic loss optimization. Function inputs are not relevant and not used for kraken, but for
    simplicity kept as function inputs. That way, all train scripts can stay the same and I just have to specify which train function is being used.
    '''
    model.train()

    # targets_ = [] # comment for now because useful for how preds evolve in time, but a huge vallue to "store" PER EPOCH nah not right now
    # preds_ = []
    tr_mae_subs = []
    tr_mse_subs = []
    tr_corr_subs_demean = []
    tr_corr_subs_org = []
    tr_epoch_loss = 0
    for i, data in enumerate(train_loader): # for loop that goes over each batch in training loop
        # i is iteration, data is batch x dimensions, probably BxCxPxV = batch x channel x patch x verteces
        inputs, targets = data[0].to(device), data[1].to(device).squeeze() # inputs = graph, output=tr_demean_mesh ico-n
        
        optimizer.zero_grad(set_to_none=True) # True by default anyway

        if VAE_flag:
            pred, latent, latent_logvar = model(inputs) #latent = z_mu, latent_sigma=sigma_mu
            kld_loss = -0.5 * torch.sum(1 + latent_logvar - (latent ** 2) - latent_logvar.exp(), dim = 1) # dim1 bc each subject has own loss
            kld_loss = torch.mean(kld_loss, dim=0) #now average across subjecrs to get single loss for all in batch
            del inputs#, latent
        else:
            pred, latent = model(inputs)
            del inputs#, latent

        
        # Output Losses
        # Lr_corrI = correye(targets, pred) # identity matrix where diagonal is optimized and off-daig is reduced (more within, less between)
        Lr_mse = torch.FloatTensor(torch.nn.MSELoss()(targets, pred)) # MSE should be low
        # Lr_marg = distance_loss(targets, pred, neighbor=True) # predicted X should be far from nearet ground truth X (for a different subject)

        # # Latent Space Losses
        # Lz_corrI = correye(latent, latent) # correlation matrix of latent space should be low off diagonal
        # Lz_dist = distance_loss(latent, latent, neighbor=False) # mean intersubject altent space distances should be high

        # Lr = krak_corrI_weight*Lr_corrI + Lr_marg + (krak_mse_weight * Lr_mse) # weighting MSE with 100,000 (1000 from Krakencoder), Fyzeen OG is 50k SDNR
        # Lz = Lz_corrI + Lz_dist
        
        # make into CPU numpy variables and detach from pytorch graph, notice no .cpu() needed because these modes are expected to be run w CPUs, might need change if done on GPU        
        pred = pred.detach().numpy()
        targets = targets.detach().numpy()

        # MAE and MSE metrics per training iteration/batch
        mae = np.mean( np.abs((targets - pred)), axis=0, keepdims=True) #LA.norm((targets - pred), ord=1, axis=0) #np.abs( (targets - pred) ) # |y - y_hat|
        tr_mae_subs.append(mae)
        mse = np.mean( (targets - pred)**2 , axis=0, keepdims=True) #(LA.norm((targets - pred), ord=2, axis=0)) ** 2 #needs to be **2 becasue norm-2 squareroots result. np.mean( (targets - pred)**2 ) #(y - y_hat)^2
        tr_mse_subs.append(mse) # 1xCxPxV, MSE for this batch

        if "demean" in netmat_prep_choice: # if doing any demeaning, then predictions are of original and we must demean here
            tr_corr_demean = np.corrcoef(targets, pred) #[subj*2 x subj*2] matrix where quadrant1 = target_target, quad2=target_pred, quad3=pred_target, quad4=pred_pred
            split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
            tr_corr_subs_demean.append(np.diag(top_right_quad))

            tr_corr_org = np.corrcoef((targets+mean_train_label), (pred+mean_train_label)) # going to be low-ish cause 256->mesh size sphere but curious
            split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
            tr_corr_subs_org.append(np.diag(top_right_quad))
        else: # if data was preped by demeaning, then for original need to readd mean
            tr_corr_demean = np.corrcoef((targets-mean_train_label), (pred-mean_train_label))
            split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
            tr_corr_subs_demean.append(np.diag(top_right_quad))

            tr_corr_org = np.corrcoef(targets, pred)# going to be low-ish cause 256->mesh size sphere but curious
            split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
            tr_corr_subs_org.append(np.diag(top_right_quad))

        # update loss and use optimizer, direct connection of LR_MSE like a skip connection
        #adding demean rho as a loss term explictly, should be sum of tr_corr_subs_demean cause that is diagonals for that batch
        # and kept at 1-rho to skew loss to be bigger at worse performance and smaller at larger performance. 
        # print(f"checkkkkkkkkkk: {len(tr_corr_subs_demean)} {tr_corr_subs_demean[i].shape}")
        B = tr_corr_subs_demean[i].shape # Batch for current iter, dynamic here because sometimes batches dont even split the main num
        L_rho =  (B - (np.asarray(tr_corr_subs_demean[i]).sum())) + 1 #plus 1 because if pefect corr, then 32(batch)-32(sum of all true,pred correlations) and so L_rho=0 then weighting cant happen
        
        if L_rho > 1:
            weight_lrho = torch.tensor(rho_loss_weight, dtype=torch.int32) #torch.tensor(rho_loss_weight, dtype=torch.int32) # -1:65
            L_rho = torch.tensor(L_rho, dtype=torch.int32)
        else:
            weight_lrho = torch.tensor(0, dtype=torch.int32) # no penalty
            L_rho = torch.tensor(L_rho, dtype=torch.int32)
    
        if VAE_flag:
            # loss = ((Lr + (krak_latent_weight * Lz) + weight_lrho*L_rho + kld_loss)) # loss uses demean so add that
            loss = Lr_mse
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        else:
            loss = Lr_mse
            # loss = ((Lr + (krak_latent_weight * Lz) + weight_lrho*L_rho)) # loss uses demean so add that
    
        loss.backward()
        tr_epoch_loss += loss.item()

        optimizer.step()

    across_sub_mae_mean = np.mean(tr_mae_subs) # across all elements, so no axis ==> mean of flatten mat->vector, so across all subs and channels and patches and verteces
    across_sub_mae_std = np.std(tr_mae_subs)
    across_sub_mse_mean = np.mean(tr_mse_subs)
    across_sub_mse_std = np.std(tr_mse_subs)
    # # because of batching, some in the list are different size so make into whole array
    # print(f"Length of tr_corr_subs_demean: {len(tr_corr_subs_demean)}, should be batches=32*N+lastbatch=Train_size")
    upto_n_minus1 = np.asarray(tr_corr_subs_demean[:-1]).squeeze() # all upto last item, do that seperate then concat
    # print(f"upto_n_minus1: {upto_n_minus1.shape}")
    upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
    n_minus_1 = np.asarray(tr_corr_subs_demean[-1])[np.newaxis,:] 
    tr_corr_subs_demean = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
    across_sub_corr_demean = np.mean(tr_corr_subs_demean)
    across_sub_corr_demean_std = np.std(tr_corr_subs_demean)

    # same for original corr values
    upto_n_minus1 = np.asarray(tr_corr_subs_org[:-1]).squeeze() # all upto last item, do that seperate then concat
    upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
    n_minus_1 = np.asarray(tr_corr_subs_org[-1])[np.newaxis,:] 
    tr_corr_subs_org = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col    across_sub_corr_org = np.mean(tr_corr_subs_org)
    across_sub_corr_org = np.mean(tr_corr_subs_org)
    across_sub_corr_org_std = np.std(tr_corr_subs_org)
    
    return tr_epoch_loss, across_sub_mae_mean, across_sub_mae_std, across_sub_mse_mean, across_sub_mse_std, across_sub_corr_demean, across_sub_corr_demean_std, across_sub_corr_org, across_sub_corr_org_std

def train_krakenonly(model, krak_mse_weight, krak_latent_weight, krak_corrI_weight, rho_loss_weight, train_loader, mean_train_label, device, optimizer, VAE_flag=False, netmat_prep_choice=None):
    '''
    Explicitly seperating here and making our kraken loss optimization its own function. This was the main one we have been useing, but with SKEWLOSS tests I need to also speerate and test KRAKEN losses alone.
    As emnitoned in function train_MSE_skewloss, some function inputs are not relevant (skew or corrI) but kept for simplicity of training models. All models give their training fcn those params, so kept to avoid 
    having to erase and re write train functions in main train scripts. CorrI is an experiment I have with the subjxsubj corr identity matrix, where ideally daigonal is 1 and off are clsoe to 0.5 correlation which is as expected.
    But so far have used it on reconstructed corrI maybe need to run on latent space and optimize to make subjects mroe independent (1 diag, 0 off diag)??
    '''
    model.train()

    # targets_ = [] # comment for now because useful for how preds evolve in time, but a huge vallue to "store" PER EPOCH nah not right now
    # preds_ = []
    tr_mae_subs = []
    tr_mse_subs = []
    tr_corr_subs_demean = []
    tr_corr_subs_org = []
    tr_epoch_loss = 0
    for i, data in enumerate(train_loader): # for loop that goes over each batch in training loop
        # i is iteration, data is batch x dimensions, probably BxCxPxV = batch x channel x patch x verteces
        inputs, targets = data[0].to(device), data[1].to(device).squeeze() # inputs = graph, output=tr_demean_mesh ico-n
        
        optimizer.zero_grad(set_to_none=True) # True by default anyway

        if VAE_flag: #kl loss is -1/2 * sum(1+ logvar - mu**2 - var)
            pred, latent, latent_logvar = model(inputs) #latent = z_mu, latent_sigma=sigma_mu
            kld_loss = -0.5 * torch.sum(1 + latent_logvar - (latent ** 2) - latent_logvar.exp(), dim = 1) # dim1 bc each subject has own loss
            kld_loss = torch.mean(kld_loss, dim=0) #now average across subjecrs to get single loss for all in batch
            del inputs#, latent, latent_logvar
        else:
            pred, latent = model(inputs)

            del inputs#, latent

        # Output Losses
        Lr_corrI = correye(targets, pred) # identity matrix where diagonal is optimized and off-daig is reduced (more within, less between)
        Lr_mse = torch.FloatTensor(torch.nn.MSELoss()(targets, pred)) # MSE should be low
        Lr_marg = distance_loss(targets, pred, neighbor=True) # predicted X should be far from nearet ground truth X (for a different subject)

        # Latent Space Losses
        Lz_corrI = correye(latent, latent) # correlation matrix of latent space should be low off diagonal
        Lz_dist = distance_loss(latent, latent, neighbor=False) # mean intersubject altent space distances should be high

        Lr = krak_corrI_weight*Lr_corrI + Lr_marg + (krak_mse_weight * Lr_mse) # weighting MSE with 100,000 (1000 from Krakencoder), Fyzeen OG is 50k SDNR
        Lz = Lz_corrI + Lz_dist
        
        # make into CPU numpy variables and detach from pytorch graph, notice no .cpu() needed because these modes are expected to be run w CPUs, might need change if done on GPU        
        pred = pred.detach().numpy()
        targets = targets.detach().numpy()

        # MAE and MSE metrics per training iteration/batch
        mae = np.mean( np.abs((targets - pred)), axis=0, keepdims=True) #LA.norm((targets - pred), ord=1, axis=0) #np.abs( (targets - pred) ) # |y - y_hat|
        tr_mae_subs.append(mae)
        mse = np.mean( (targets - pred)**2 , axis=0, keepdims=True) #(LA.norm((targets - pred), ord=2, axis=0)) ** 2 #needs to be **2 becasue norm-2 squareroots result. np.mean( (targets - pred)**2 ) #(y - y_hat)^2
        tr_mse_subs.append(mse) # 1xCxPxV, MSE for this batch

        if "demean" in netmat_prep_choice: # if doing any demeaning, then predictions are of original and we must demean here
            tr_corr_demean = np.corrcoef(targets, pred) #[subj*2 x subj*2] matrix where quadrant1 = target_target, quad2=target_pred, quad3=pred_target, quad4=pred_pred
            split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
            tr_corr_subs_demean.append(np.diag(top_right_quad))

            tr_corr_org = np.corrcoef((targets+mean_train_label), (pred+mean_train_label)) # going to be low-ish cause 256->mesh size sphere but curious
            split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
            tr_corr_subs_org.append(np.diag(top_right_quad))
        else: # if data was preped by demeaning, then for original need to readd mean
            tr_corr_demean = np.corrcoef((targets-mean_train_label), (pred-mean_train_label))
            split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
            tr_corr_subs_demean.append(np.diag(top_right_quad))

            tr_corr_org = np.corrcoef(targets, pred)# going to be low-ish cause 256->mesh size sphere but curious
            split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
            tr_corr_subs_org.append(np.diag(top_right_quad))

        # update loss and use optimizer, direct connection of LR_MSE like a skip connection
        #adding demean rho as a loss term explictly, should be sum of tr_corr_subs_demean cause that is diagonals for that batch
        # and kept at 1-rho to skew loss to be bigger at worse performance and smaller at larger performance. 
        # print(f"checkkkkkkkkkk: {len(tr_corr_subs_demean)} {tr_corr_subs_demean[i].shape}")
        B = tr_corr_subs_demean[i].shape # Batch for current iter, dynamic here because sometimes batches dont even split the main num
        L_rho =  (B - (np.asarray(tr_corr_subs_demean[i]).sum())) + 1 #plus 1 because if pefect corr, then 32(batch)-32(sum of all true,pred correlations) and so L_rho=0 then weighting cant happen
        
        if L_rho > 1:
            weight_lrho = torch.tensor(rho_loss_weight, dtype=torch.int32) #torch.tensor(rho_loss_weight, dtype=torch.int32) # -1:65
            L_rho = torch.tensor(L_rho, dtype=torch.int32)
        else:
            weight_lrho = torch.tensor(0, dtype=torch.int32) # no penalty
            L_rho = torch.tensor(L_rho, dtype=torch.int32)
    
        if VAE_flag:
            loss = ((Lr + (krak_latent_weight * Lz) + weight_lrho*L_rho + kld_loss)) # loss uses demean so add that
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        else:
            loss = ((Lr + (krak_latent_weight * Lz) + weight_lrho*L_rho)) # loss uses demean so add that
    
        loss.backward()
        tr_epoch_loss += loss.item()

        optimizer.step()

        # targets_.append(demean_targets.cpu().numpy())
        # preds_.append(demean_pred.cpu().detach().numpy())

    across_sub_mae_mean = np.mean(tr_mae_subs) # across all elements, so no axis ==> mean of flatten mat->vector, so across all subs and channels and patches and verteces
    across_sub_mae_std = np.std(tr_mae_subs)
    across_sub_mse_mean = np.mean(tr_mse_subs)
    across_sub_mse_std = np.std(tr_mse_subs)
    # # because of batching, some in the list are different size so make into whole array
    # print(f"Length of tr_corr_subs_demean: {len(tr_corr_subs_demean)}, should be batches=32*N+lastbatch=Train_size")
    upto_n_minus1 = np.asarray(tr_corr_subs_demean[:-1]).squeeze() # all upto last item, do that seperate then concat
    # print(f"upto_n_minus1: {upto_n_minus1.shape}")
    upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
    n_minus_1 = np.asarray(tr_corr_subs_demean[-1])[np.newaxis,:] 
    tr_corr_subs_demean = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
    across_sub_corr_demean = np.mean(tr_corr_subs_demean)
    across_sub_corr_demean_std = np.std(tr_corr_subs_demean)

    # same for original corr values
    upto_n_minus1 = np.asarray(tr_corr_subs_org[:-1]).squeeze() # all upto last item, do that seperate then concat
    upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
    n_minus_1 = np.asarray(tr_corr_subs_org[-1])[np.newaxis,:] 
    tr_corr_subs_org = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col    across_sub_corr_org = np.mean(tr_corr_subs_org)
    across_sub_corr_org = np.mean(tr_corr_subs_org)
    across_sub_corr_org_std = np.std(tr_corr_subs_org)
    
    return tr_epoch_loss, across_sub_mae_mean, across_sub_mae_std, across_sub_mse_mean, across_sub_mse_std, across_sub_corr_demean, across_sub_corr_demean_std, across_sub_corr_org, across_sub_corr_org_std


# =========================================== BGTSiT/BGTSiT_VAE - enc+dec model training =========================================== #
'''
    Below are trainign functions for enc-dec architectures. Encoder will be our graph transformer and decoder will be sit, however, masking is required at this step to ensure model does not cheat
    when taking in input to decoder. Normally, decoder takes in the labels, does regular MHSA, THEN its cross attention with enc and THAT is output to FFN then sequence begins where each patch is predicted. 
    However, we also have U-NET or more like L-NETs where enc is same as always but decoder actually upsamples and does not need masking. See this for architectural specifics: https://arxiv.org/abs/2303.11909
'''
# train_kraken_encdec
# def train_SiTBGT_skew_kraken(model, padding, krak_mse_weight, krak_latent_weight, krak_corrI_weight, train_loader, mean_train_label, device, optimizer, lamda_value, skew_weight, VAE_flag=False):
#     model.train()
#     # targets_ = []
#     # preds_ = []
#     tr_mae_subs = []
#     tr_mse_subs = []
#     tr_corr_subs_demean = []
#     tr_corr_subs_org = []
#     tr_epoch_loss = 0
#     for i, data in enumerate(train_loader):
#         # inputs, in_targets, targets = data[0].to(device), data[1].to(device), data[2].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
#         mesh_indata, targets = data[0].to(device), data[1].to(device).squeeze() #.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
#         # print(f"MESH NAN?:\n{torch.isnan(mesh_indata).sum()} \n\nTARGETS NAN?:{torch.isnan (targets).sum()}")
#         optimizer.zero_grad(set_to_none=True) # True by default anyway

#         if VAE_flag: # -1/2 SUM*(1 + logvar -mu**2 - var)
#             pred, latent, latent_logvar = model(src=mesh_indata, tgt=targets,  tgt_mask=generate_subsequent_mask(model.latent_length).to(device))
#             # print(f"PRED:\n{torch.isnan(pred).sum()}\n\n MU:{torch.isnan(latent).sum()}\n\n Sigma:{torch.isnan(latent_sigma).sum()}")
#             kld_loss = -0.5 * torch.sum(1 + latent_logvar - (latent ** 2) - latent_logvar.exp(), dim = 1) # dim1 bc each subject has own loss
#             kld_loss = torch.mean(kld_loss, dim=0) #now average across subjecrs to get single loss for all in batch
#         else:
#             pred, latent = model(src=mesh_indata, tgt=targets,  tgt_mask=generate_subsequent_mask(model.latent_length).to(device))
#             # print(f"PRED:\n{torch.isnan(pred).sum()} \n\n MU:{torch.isnan(latent).sum()}")
#             # pred, mu, log_var = model(src=netmat_indata, tgt=mesh_decinput_data,  mesh_mask=generate_subsequent_mask(model.ico_patch).to(device))

#         pred = pred[:, padding:] 
#         targets = targets[:,padding:]
#         # mean_train_label = np.mean(targets.detach().numpy(), axis=0) #mean_train_label[padding:]    

#         # Output Losses
#         Lr_corrI = correye(targets, pred) # identity matrix where diagonal is optimized and off-daig is reduced (more within, less between)
#         Lr_mse = torch.nn.MSELoss()(targets, pred) # MSE should be low
#         Lr_marg = distance_loss(targets, pred, neighbor=True) # predicted X should be far from nearet ground truth X (for a different subject)

#         # Latent Space Losses
#         Lz_corrI = correye(latent, latent) # correlation matrix of latent space should be low off diagonal
#         Lz_dist = distance_loss(latent, latent, neighbor=False) # mean intersubject altent space distances should be high

#         Lr =  krak_corrI_weight*Lr_corrI + Lr_marg + (krak_mse_weight * Lr_mse) # weighting MSE with 100,000 (1000 from Krakencoder), Fyzeen OG is 50k SDNR
#         Lz = Lz_corrI + Lz_dist
#         # print(f"lr:{torch.isnan(Lr_corrI).sum()}{torch.isnan(Lr_mse).sum()}{torch.isnan(Lr_marg).sum()} \n\n lz:{torch.isnan(Lz_corrI).sum()}{torch.isnan(Lz_dist).sum()}")
#         # skew loss
#         # lamda = lamda_value # ideal lambda is between 1 and 2, so 1.5 is therefor arbitrary for now can be optimized for our purposes       
#         skewed_loss = torch.exp(torch.sign(targets - pred) * lamda_value)
#         skewed_loss = skew_weight * torch.mean(skewed_loss) #mean skew, then that is coefficient to original loss for batch mean

#         # make into CPU numpy variables and detach from pytorch graph, notice no .cpu() needed because these modes are expected to be run w CPUs, might need change if done on GPU        
#         pred = pred.detach().numpy()
#         targets = targets.detach().numpy()

#         # MAE and MSE metrics per training iteration/batch
#         mae = np.mean( np.abs((targets - pred)), axis=0, keepdims=True) #LA.norm((targets - pred), ord=1, axis=0) #np.abs( (targets - pred) ) # |y - y_hat|
#         tr_mae_subs.append(mae)
#         mse = np.mean( (targets - pred)**2 , axis=0, keepdims=True) #(LA.norm((targets - pred), ord=2, axis=0)) ** 2 #needs to be **2 becasue norm-2 squareroots result. np.mean( (targets - pred)**2 ) #(y - y_hat)^2
#         tr_mse_subs.append(mse) # 1xCxPxV, MSE for this batch

#         # important points!! saving predicted values and same values plus training mean, demean_corr and orig_corr respectively
#         tr_corr_demean = np.corrcoef((targets-mean_train_label), (pred-mean_train_label))
#         # print(f"tr_corr_demean shape: {tr_corr_demean.shape}")
#         split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
#         top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
#         lower_tri_of_qaudrant_demean = get_lower_tris(top_right_quad)
#         # print(f"tril shape: {lower_tri_of_qaudrant_demean.shape}")
#         tr_corr_subs_demean.append(lower_tri_of_qaudrant_demean) #upper tri rho values for this batch (correlating targets and pred)

#         tr_corr_org = np.corrcoef(targets,pred)# going to be low-ish cause 256->mesh size sphere but curious
#         split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
#         top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
#         lower_tri_of_qaudrant_org = get_lower_tris(top_right_quad)
#         tr_corr_subs_org.append(lower_tri_of_qaudrant_org)

#         # update loss and use optimizer, direct connection of LR_MSE like a skip connection
#         if VAE_flag:
#             loss = ((Lr + (krak_latent_weight * Lz) + Lr_mse + kld_loss) * skewed_loss) # loss uses demean so add that
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
#         else:
#             loss = ((Lr + (krak_latent_weight * Lz) + Lr_mse) * skewed_loss) # loss uses demean so add that
    
#         loss.backward()
#         tr_epoch_loss += loss.item()

#         optimizer.step()

#         # targets_.append(demean_targets.cpu().numpy())
#         # preds_.append(demean_pred.cpu().detach().numpy())

#     across_sub_mae_mean = np.mean(tr_mae_subs) # across all elements, so no axis ==> mean of flatten mat->vector, so across all subs and channels and patches and verteces
#     across_sub_mae_std = np.std(tr_mae_subs)
#     across_sub_mse_mean = np.mean(tr_mse_subs)
#     across_sub_mse_std = np.std(tr_mse_subs)

#     # # because of batching, some in the list are different size so make into whole array
#     # upto_n_minus1 = np.asarray(tr_corr_subs_demean[:-1]).squeeze() # all upto last item, do that seperate then concat
#     # upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
#     # n_minus_1 = np.asarray(tr_corr_subs_demean[-1])
#     # tr_corr_demean_flat = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
#     # across_sub_corr_demean = np.mean(tr_corr_demean_flat)
#     # across_sub_corr_demean_std = np.std(tr_corr_demean_flat)
#     # # same for original corr values
#     # upto_n_minus1 = np.asarray(tr_corr_subs_org[:-1]).squeeze() # all upto last item, do that seperate then concat
#     # upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
#     # n_minus_1 = np.asarray(tr_corr_subs_org[-1])
#     # tr_corr_org_flat = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
#     # across_sub_corr_org = np.mean(tr_corr_org_flat)
#     # across_sub_corr_org_std = np.std(tr_corr_org_flat)
#     across_sub_corr_demean = np.mean(tr_corr_subs_demean)
#     across_sub_corr_demean_std = np.std(tr_corr_subs_demean)
#     across_sub_corr_org = np.mean(tr_corr_subs_org)
#     across_sub_corr_org_std = np.std(tr_corr_subs_org)
    
#     # print(f"FINAL TRAIN OUTPUT: \n{np.isnan(tr_epoch_loss).sum()} \n{np.isnan(across_sub_mae_mean).sum()} \n{np.isnan(across_sub_mse_mean).sum()} \n{np.isnan(across_sub_corr_demean).sum()} \n{np.isnan(across_sub_corr_org).sum()}")

#     return tr_epoch_loss, across_sub_mae_mean, across_sub_mae_std, across_sub_mse_mean, across_sub_mse_std, across_sub_corr_demean, across_sub_corr_demean_std, across_sub_corr_org, across_sub_corr_org_std
    
def train_SiTBGT_krakenonly(model, padding, krak_mse_weight, krak_latent_weight, krak_corrI_weight, train_loader, mean_train_label, device, optimizer, VAE_flag=False):
    model.train()
    # targets_ = []
    # preds_ = []
    tr_mae_subs = []
    tr_mse_subs = []
    tr_corr_subs_demean = []
    tr_corr_subs_org = []
    tr_epoch_loss = 0
    device = "cpu"
    
    for i, data in enumerate(train_loader):
        # inputs, in_targets, targets = data[0].to(device), data[1].to(device), data[2].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
        mesh_indata, targets = data[0].to(device), data[1].to(device).squeeze() #.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
        optimizer.zero_grad(set_to_none=True) # True by default anyway

        if VAE_flag: # -1/2 SUM*(1 + logvar -mu**2 - var)
            pred, latent, latent_logvar = model(src=mesh_indata, tgt=targets,  tgt_mask=generate_subsequent_mask(model.latent_length).to(device))
            # print(f"PRED:\n{torch.isnan(pred).sum()}\n\n MU:{torch.isnan(latent).sum()}\n\n Sigma:{torch.isnan(latent_sigma).sum()}")
            kld_loss = -0.5 * torch.sum(1 + latent_logvar - (latent ** 2) - latent_logvar.exp(), dim = 1) # dim1 bc each subject has own loss
            kld_loss = torch.mean(kld_loss, dim=0) #now average across subjects to get single loss for all in batch
        else:
            pred, latent = model(src=mesh_indata, tgt=targets,  tgt_mask=generate_subsequent_mask(model.latent_length).to(device))
            # print(f"PRED:\n{torch.isnan(pred).sum()} \n\n MU:{torch.isnan(latent).sum()}")
            # pred, mu, log_var = model(src=netmat_indata, tgt=mesh_decinput_data,  mesh_mask=generate_subsequent_mask(model.ico_patch).to(device))

        pred = pred[:, padding:] 
        targets = targets[:,padding:]
        # mean_train_label = np.mean(targets.detach().numpy(), axis=0) #mean_train_label[padding:]    

        # Output Losses
        Lr_corrI = correye(targets, pred) # identity matrix where diagonal is optimized and off-daig is reduced (more within, less between)
        Lr_mse = torch.nn.MSELoss()(targets, pred) # MSE should be low
        Lr_marg = distance_loss(targets, pred, neighbor=True) # predicted X should be far from nearet ground truth X (for a different subject)

        # Latent Space Losses
        Lz_corrI = correye(latent, latent) # correlation matrix of latent space should be low off diagonal
        Lz_dist = distance_loss(latent, latent, neighbor=False) # mean intersubject altent space distances should be high

        Lr =  krak_corrI_weight*Lr_corrI + Lr_marg + (krak_mse_weight * Lr_mse) # weighting MSE with 100,000 (1000 from Krakencoder), Fyzeen OG is 50k SDNR
        Lz = Lz_corrI + Lz_dist

        # make into CPU numpy variables and detach from pytorch graph, notice no .cpu() needed because these modes are expected to be run w CPUs, might need change if done on GPU        
        pred = pred.detach().numpy()
        targets = targets.detach().numpy()

        # MAE and MSE metrics per training iteration/batch
        mae = np.mean( np.abs((targets - pred)), axis=0, keepdims=True) #LA.norm((targets - pred), ord=1, axis=0) #np.abs( (targets - pred) ) # |y - y_hat|
        tr_mae_subs.append(mae)
        mse = np.mean( (targets - pred)**2 , axis=0, keepdims=True) #(LA.norm((targets - pred), ord=2, axis=0)) ** 2 #needs to be **2 becasue norm-2 squareroots result. np.mean( (targets - pred)**2 ) #(y - y_hat)^2
        tr_mse_subs.append(mse) # 1xCxPxV, MSE for this batch

        # important points!! saving predicted values and same values plus training mean, demean_corr and orig_corr respectively
        # tr_corr_demean = np.corrcoef((targets-mean_train_label), (pred-mean_train_label))
        # split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
        # top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
        # tr_corr_subs_demean.append(np.diag(top_right_quad)) #upper tri rho values for this batch (correlating targets and pred)

        # tr_corr_org = np.corrcoef(targets,pred)# going to be low-ish cause 256->mesh size sphere but curious
        # split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
        # top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
        # tr_corr_subs_org.append(np.diag(top_right_quad))
        tr_corr_demean = np.corrcoef(targets, pred)
        split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
        top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
        tr_corr_subs_demean.append(np.diag(top_right_quad)) #upper tri rho values for this batch (correlating targets and pred)

        tr_corr_org = np.corrcoef((targets+mean_train_label), (pred+mean_train_label))# going to be low-ish cause 256->mesh size sphere but curious
        split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
        top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
        tr_corr_subs_org.append(np.diag(top_right_quad))

        # update loss and use optimizer, direct connection of LR_MSE like a skip connection
        if VAE_flag:
            loss = ((Lr + (krak_latent_weight * Lz) + Lr_mse + kld_loss)) # loss uses demean so add that
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        else:
            loss = ((Lr + (krak_latent_weight * Lz) + Lr_mse)) # loss uses demean so add that
    
        loss.backward()
        tr_epoch_loss += loss.item()

        optimizer.step()

        # targets_.append(demean_targets.cpu().numpy())
        # preds_.append(demean_pred.cpu().detach().numpy())
        
    across_sub_mae_mean = np.mean(tr_mae_subs) # across all elements, so no axis ==> mean of flatten mat->vector, so across all subs and channels and patches and verteces
    across_sub_mae_std = np.std(tr_mae_subs)
    across_sub_mse_mean = np.mean(tr_mse_subs)
    across_sub_mse_std = np.std(tr_mse_subs)

    upto_n_minus1 = np.asarray(tr_corr_subs_demean[:-1])#.squeeze() # all upto last item, do that seperate then concat
    upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
    n_minus_1 = np.asarray(tr_corr_subs_demean[-1])[np.newaxis,:]
    tr_corr_subs_demean = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
    across_sub_corr_demean = np.mean(tr_corr_subs_demean)
    across_sub_corr_demean_std = np.std(tr_corr_subs_demean)

    # same for original corr values
    upto_n_minus1 = np.asarray(tr_corr_subs_org[:-1]).squeeze() # all upto last item, do that seperate then concat
    upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
    n_minus_1 = np.asarray(tr_corr_subs_org[-1])[np.newaxis,:] 
    tr_corr_subs_org = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col    across_sub_corr_org = np.mean(tr_corr_subs_org)
    across_sub_corr_org = np.mean(tr_corr_subs_org)
    across_sub_corr_org_std = np.std(tr_corr_subs_org)
    
    return tr_epoch_loss, across_sub_mae_mean, across_sub_mae_std, across_sub_mse_mean, across_sub_mse_std, across_sub_corr_demean, across_sub_corr_demean_std, across_sub_corr_org, across_sub_corr_org_std
    

def train_CNN(model, train_loader, mean_train_label, device, optimizer, krak_mse_weight, krak_latent_weight, krak_corrI_weight, krak_lzdist_weight, lamda_value, grad_clip=False, VAE_flag=False, epsilon=0):
    '''
    Train function for cnn training. Uses MSE + Krakencoder Losses
    Written by Fyzeen Ahmad - May 30 2025
    '''
    model.train()

    # targets_ = [] # comment for now because useful for how preds evolve in time, but a huge vallue to "store" PER EPOCH nah not right now
    # preds_ = []
    tr_mae_subs = []
    tr_mse_subs = []
    tr_corr_subs_demean = []
    tr_corr_subs_org = []
    tr_epoch_loss = 0
    for i, data in enumerate(train_loader): # for loop that goes over each batch in training loop
        # i is iteration, data is batch x dimensions, probably BxCxPxV = batch x channel x patch x verteces
        inputs, targets = data.x.to(device), data.y.to(device).view(-1, 4950) # inputs = graph, output=tr_demean_mesh ico-n
        
        optimizer.zero_grad(set_to_none=True) # True by default anyway

        pred, latent = model(data)
        pred = pred.squeeze()
        latent = latent.squeeze()
        
        # Compute Reconstruction Losses
        Lr_corrI = correye(targets, pred, epsilon) # corr mat of measured->predicted should be high along diagonal, loww off diagonal 
        Lr_mse = torch.nn.functional.mse_loss(targets, pred) # MSE should be low
        Lr_marg = distance_loss(targets, pred, neighbor=True, epsilon=epsilon) # predicted X should be far from nearet ground truth X (for a different subject)

        # Compute Latent Losses
        Lz_corrI = correye(latent, latent, epsilon) # correlation matrix of latent space should be low off diagonal
        Lz_dist = distance_loss(latent, latent, neighbor=False, epsilon=epsilon) # mean intersubject altent space distances should be high

        # Compute Total Loss
        Lr = (krak_corrI_weight * Lr_corrI) + Lr_marg + (krak_mse_weight * Lr_mse) 
        Lz = Lz_corrI + (Lz_dist * krak_lzdist_weight)
        Lz_final = krak_latent_weight * Lz

        loss = Lr + Lz_final

        # Backprop and Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        print("+++++ PRE CLIP +++++")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"[NaN in Gradients] {name}")
                    NaNExists = True
                elif torch.isinf(param.grad).any():
                    print(f"[Inf in Gradients] {name}")
                elif param.grad.abs().max() > 1e4:
                    print(f"[Exploding Gradients] {name} - Max: {param.grad.abs().max().item()}")

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        tr_epoch_loss += loss.item()
        
        # make into CPU numpy variables and detach from pytorch graph, notice no .cpu() needed because these modes are expected to be run w CPUs, might need change if done on GPU        
        pred = pred.detach().numpy()
        targets = targets.detach().numpy()

        # MAE and MSE metrics per training iteration/batch
        mae = np.mean( np.abs((targets - pred)), axis=0, keepdims=True) #LA.norm((targets - pred), ord=1, axis=0) #np.abs( (targets - pred) ) # |y - y_hat|
        tr_mae_subs.append(mae)
        mse = np.mean( (targets - pred)**2 , axis=0, keepdims=True) #(LA.norm((targets - pred), ord=2, axis=0)) ** 2 #needs to be **2 becasue norm-2 squareroots result. np.mean( (targets - pred)**2 ) #(y - y_hat)^2
        tr_mse_subs.append(mse) # 1xCxPxV, MSE for this batch

        tr_corr_demean = np.corrcoef(targets - mean_train_label, pred - mean_train_label)
        split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
        top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
        lower_tri_of_qaudrant_demean = get_lower_tris(top_right_quad)
        tr_corr_subs_demean.append(lower_tri_of_qaudrant_demean) #upper tri rho values for this batch (correlating targets and pred)

        tr_corr_org = np.corrcoef(targets, pred)# going to be low-ish cause 256->mesh size sphere but curious
        split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
        top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
        lower_tri_of_qaudrant_org = get_lower_tris(top_right_quad)
        tr_corr_subs_org.append(lower_tri_of_qaudrant_org)

        print(f"===== Loss Breakdown, Batch {i} =====")
        print("Lr_corrI    : ", Lr_corrI)
        print("Lr_mse      : ", Lr_mse)
        print("Lr_marg     : ", Lr_marg)
        print("Lz_corrI    : ", Lz_corrI)
        print("Lz_dist     : ", Lz_dist)
        print("Lr (total)  : ", Lr)
        print("Lz (total)  : ", Lz)
        print("Lz_final    : ", Lz_final)
        print("Total loss  : ", loss)

        print("MAE         : ", np.mean(mae))
        print("MSE         : ", np.mean(mse))
        print("Corr        : ", np.mean(lower_tri_of_qaudrant_org))
        print("Demean Corr : ", np.mean(lower_tri_of_qaudrant_demean))

        if torch.isnan(torch.tensor(pred)).any():
            nan_indices = torch.isnan(torch.tensor(pred)).nonzero(as_tuple=True)
            print("NaN indices:", nan_indices)
            raise ValueError("NaNs detected in model output!")

        # targets_.append(demean_targets.cpu().numpy())
        # preds_.append(demean_pred.cpu().detach().numpy())

    across_sub_mae_mean = np.mean(np.concatenate([arr.ravel() for arr in tr_mae_subs]))
    across_sub_mae_std = np.std(np.concatenate([arr.ravel() for arr in tr_mae_subs]))

    across_sub_mse_mean = np.mean(np.concatenate([arr.ravel() for arr in tr_mse_subs]))
    across_sub_mse_std = np.std(np.concatenate([arr.ravel() for arr in tr_mse_subs]))

    across_sub_corr_demean = np.mean(np.concatenate([arr.ravel() for arr in tr_corr_subs_demean]))
    across_sub_corr_demean_std = np.std(np.concatenate([arr.ravel() for arr in tr_corr_subs_demean]))

    across_sub_corr_org = np.mean(np.concatenate([arr.ravel() for arr in tr_corr_subs_org]))
    across_sub_corr_org_std = np.std(np.concatenate([arr.ravel() for arr in tr_corr_subs_org]))

    # OLD CODE BELOW WHICH COULD NOT DEAL WITH INHOMOGENOUS DIMENSION (due to final batch being smaller)
    # across_sub_mae_mean = np.mean(tr_mae_subs) # across all elements, so no axis ==> mean of flatten mat->vector, so across all subs and channels and patches and verteces
    # across_sub_mae_std = np.std(tr_mae_subs)
    # across_sub_mse_mean = np.mean(tr_mse_subs)
    # across_sub_mse_std = np.std(tr_mse_subs)

    # across_sub_corr_demean = np.mean(tr_corr_subs_demean)
    # across_sub_corr_demean_std = np.std(tr_corr_subs_demean)
    # across_sub_corr_org = np.mean(tr_corr_subs_org)
    # across_sub_corr_org_std = np.std(tr_corr_subs_org)
    
    return tr_epoch_loss, across_sub_mae_mean, across_sub_mae_std, across_sub_mse_mean, across_sub_mse_std, across_sub_corr_demean, across_sub_corr_demean_std, across_sub_corr_org, across_sub_corr_org_std
