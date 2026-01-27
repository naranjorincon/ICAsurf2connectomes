# this script is used tp get the mean and std across all channels 

import pandas as pd
import nibabel as nb
import numpy as np

import yaml
import os
import argparse

def main(config):
    split = config['data']['split']

    def write_to_file(content):
        with open("/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/chnll_mean_std.print", 'a') as file:
            file.write(str(content) + '\n')

    write_to_file('')
    write_to_file('#'*30)

    #### PARAMETERS #####
    num_channels = config['data']['channels'] # 4
    task = config['data']['task'] # surf2mat #scan_age #birth_age
    path_to_data = config['data']['data_path'] #  /data/
    nm_fs_data = config['data']['fs_data_path'] # d15_fs_LR
    label_path = config['data']['label_path'] # ../labels/HCPdb/
    chosen_hemi = config['data']['hemisphere'] #1L or 1R or 2 for both

    write_to_file('')
    ####

    ids_train = pd.read_csv(os.path.join(label_path, '{}/train_upt.csv'.format(task)))['ids']
    ids_validation = pd.read_csv(os.path.join(label_path, '{}/validation_upt.csv'.format(task)))['ids']
    ids_test = pd.read_csv(os.path.join(label_path, '{}/test_upt.csv'.format(task)))['ids']
    # concat them
    ids = pd.concat([ids_train,ids_validation,ids_test],0) # now stacked as rows onto eachother
    ids = ids.to_numpy().reshape(-1) # shape into expected dim

    write_to_file('')
    # transpose labels so that its formatted to subj x triu array    
    num_subjects = ids.shape[0]
    write_to_file('Num of subjects:'+str(num_subjects))    
    write_to_file('')

    data = [] # list of numpy arrays each is a numpy array version of the shape.gii info
    if chosen_hemi == '1L':
        write_to_file('Left hemisphere was chosen.')
        for i, id in enumerate(ids): # reads in actual id num with 'id' inside the pandas column from the read csv, see above ids variable
            filename = os.path.join(path_to_data,'ICA_fs_LR_32',nm_fs_data,'resamp_{}.L.shape.gii'.format(id)) # shape.gii is a metric file for values at each vertex, in our case the ICA values
            # check if real, ow skip to next
            check_exists = os.path.exists(filename)
            if check_exists is False:
                write_to_file('This path did not exists, skipping:{}'.format(filename))
                continue

            # data is then an array where each element is a shape.gii image
            # agg based on channels, bc if you look below at print, data is [15,49k] so resamp gray ordinates 
            # for each ICA!! I think this then aggregates based on channels? See below if changes
            # from nibabel website, agg_data() is "Aggregate GIFTI data arrays into an ndarray or tuple of ndarray", assuming that makes conversion to numpy easy
            data.append(np.array(nb.load(filename).agg_data())[:num_channels,:]) # not sure why add specifically the index of only channels cause that is all there is, leaving it tho -\(*_*)/-
            if i%100==0:
                check_load = nb.load(filename)
                check_agg_data = nb.load(filename).agg_data()
                write_to_file('\nLoading GIFTI for subject: {}'.format(i))
                write_to_file('\nChecking data metrics. \n MRI data has loaded number of data arrays: {} \n After Using agg_data(), looking into first tuple: {}'.format(len(check_load.darrays),len(check_agg_data[0])))
                write_to_file('\nActual stored data value, presumably aggregated through channels=inputdim:{}'.format(len(data[i])))
            # from sanity checks, I see now that our data values are dim C x TS, our in our case inputdim x TS for each vertex in the sphere
    elif chosen_hemi == '1R':
        write_to_file('Right hemisphere was chosen.')
        for i, id in enumerate(ids):
            filename = os.path.join(path_to_data,'ICA_fs_LR_32',nm_fs_data,'resamp_{}.R.shape.gii'.format(id))
            # check if real, ow skip to next
            check_exists = os.path.exists(filename)
            if check_exists is False:
                write_to_file('This path did not exists, skipping:{}'.format(filename))
                continue

            data.append(np.array(nb.load(filename).agg_data())[:num_channels,:])
            if i%100==0:
                check_load = nb.load(filename)
                check_agg_data = nb.load(filename).agg_data()
                write_to_file('\nLoading GIFTI for subject: {}'.format(i))
                write_to_file('\nChecking data metrics. \n MRI data has loaded number of data arrays: {} \n After Using agg_data(), looking into first tuple: {}'.format(len(check_load.darrays),len(check_agg_data[0])))
                write_to_file('\nActual stored data value, presumably aggregated through channels=inputdim:{}'.format(len(data[i])))

    elif chosen_hemi == '2LR':
        write_to_file('Both hemispheres were chosen.')
        for i,id in enumerate(ids):
            # data here becomes [iiL,iiR, ii+1L,ii+1R .... NL,NR] so shape is list of num_subs*2?
            filename = os.path.join(path_to_data,'ICA_fs_LR_32',nm_fs_data,'resamp_{}.L.shape.gii'.format(id))
            # check if real, ow skip to next
            check_exists = os.path.exists(filename)
            if check_exists is False:
                write_to_file('This path did not exists, skipping:{}'.format(filename))
                continue

            data.append(np.array(nb.load(filename).agg_data()))
            filename = os.path.join(path_to_data,'ICA_fs_LR_32',nm_fs_data,'resamp_{}.R.shape.gii'.format(id))
            data.append(np.array(nb.load(filename).agg_data())[:num_channels,:])
            if i%100==0:
                check_load = nb.load(filename)
                check_agg_data = nb.load(filename).agg_data()
                write_to_file('\nLoading GIFTI for subject: {}'.format(i))
                write_to_file('\nChecking data metrics. \n MRI data has loaded number of data arrays: {} \n After Using agg_data(), looking into first tuple: {}'.format(len(check_load.darrays),len(check_agg_data[0])))
                write_to_file('\nActual stored data value, presumably aggregated through channels=inputdim:{}'.format(len(data[i])))

    write_to_file('raw resamp loaded data, order is L1R1,L2R2,...,LNRN, where N = subject. Size of data should be 2*length(subjects) if bilateral, o.w. len(subs). It is length:{}'.format(len(data)))        
    write_to_file('looking at ONE of the values:\n Has shape: {} \n len of a single value inside the list: \n {}'.format(data[10].shape, len(data[10])))

    # mean and std of each channel for normalization of raw data
    # from script, I see its reshaped into a 1xchannelsx1 numpy mat, so 1x4x1 in OG SiT script. It takes in a numpy array and reshapres it into 1x4x1
    # so I think OG numpy array is 4 numbers so 1x4 or 4x1. Same with std.
    data = np.asarray(data)
    write_to_file(f"Finding means and std. \nData shape:{data.shape}")

    means_ch = np.mean(data, axis=1) # shape is subjxica_channels x allvertex
    stds_ch = np.std(data, axis=1)
    write_to_file(f"Finding output shapes. \nmean & std shape:{means_ch.shape}, {stds_ch.shape}")

    # conver to numpy array for later rehape
    means = means_ch
    stds = stds_ch
    write_to_file('Saving means and stds, should not be used again if so. Keep as same for training.')
    filename_means = os.path.join(config['data']['label_path'],'{}/{}_means.npy'.format(task,chosen_hemi))
    filename_stds = os.path.join(config['data']['label_path'],'{}/{}_stds.npy'.format(task, chosen_hemi))

    np.save(filename_means,means)
    np.save(filename_stds,stds)
    
    # from above, it is true that the data list has N elements where each element is a indim x resampdim numpy 
    # array. But its gray ordinates i'm sure, numes are floats and seem to range from negative 6 to positive 6
    

if __name__ == '__main__':

    # Set up argument parser
        
    parser = argparse.ArgumentParser(description='preprocessing ICAd50 data for patching')
    
    parser.add_argument(
                        'config',
                        type=str,
                        default='./config/hparams_surf2mat.yml',
                        help='path where the data is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call training
    main(config)