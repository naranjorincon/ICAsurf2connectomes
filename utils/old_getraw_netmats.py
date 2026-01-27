
# importing modules for script that looks for the netmats of subj and vectorizes them. 
# Some netmats are in a weird format, so they have to be fixed.
import os
import os.path as op

import pandas as pd
import numpy as np
import torch

def ListDirNoHidden(path):
        '''
        Lists all the files non-hidden files in a directory provided by 'path'
        '''
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

def ListRawNetMats(path, type):
    if type == 'profumo':
        for f in os.listdir(path):
            if not f.startswith('.'):
                if f.endswith('.csv') and f.startswith("sub-"):
                    yield f
    elif type == "ICA":
        for f in os.listdir(path):
            if not f.startswith('.'):
                if f.endswith('.csv') and f.startswith("sub-"):
                    yield f
    elif type == "schaefer":
        for f in os.listdir(path):
            if not f.startswith('.'):
                if f.endswith('.csv') and f.startswith("sub-"):
                    yield f
    elif type == "glasser":
        for f in os.listdir(path):
            if not f.startswith('.'):
                if f.endswith('csv') and f.startswith("subj-"):
                    yield f
    elif type == "yeo":
        for f in os.listdir(path):
            if not f.startswith('.'):
                if f.endswith('.csv') and f.startswith("sub-"):
                    yield f
    else:
        raise ValueError("Type MUST be 'profumo', 'ICA', 'scahefer', 'yeo', or 'glasser'")


def findNetMatDir(root, type):
    if type == "profumo":
        path = op.join(root, "profumo_reps", "netmats")    
    elif type == "ICA":
        path = op.join(root, "ICA_reps", "netmats")
    elif type == "schaefer":
        path = op.join(root, "schaefer", "netmats")
    elif type == "glasser":
        path = op.join(root, "glasser", "netmats")
    elif type == "yeo":
        path = op.join(root, "yeo", "netmats")
    else:
        raise ValueError("Type MUST be 'profumo', 'ICA', 'scahefer', 'yeo', or 'glasser'")
    return path

def process_pfm_netmat(df):
    '''
    Processes PROFUMO NetMat.
    
    Inputs
    ===========
    df: pandas.DataFrame
        DataFrame read in from raw .csv file 
    
    Outputs
    ===========
    out: pandas.DataFrame
        DataFrame with cleaned values
    '''
    # Grabbing first 49 columns; the last column had nomral float values for some reason
    df1 = df.iloc[:,:49]
    
    # Applying lambda function to transform strings (scientific notation) to floats
    df2 = df1.applymap(lambda x: float(x[:7]) * 10**float(x[-4:-1])) 
    
    # Appending 49th column
    df2[49] = df[49]
    
    # All diagonal elements are -1, the below code makes them +1
    matrix = df2.to_numpy()
    diagonal_indices = np.diag_indices_from(matrix)
    matrix[diagonal_indices] = 1
    out = pd.DataFrame(matrix, columns=df.columns, index=df.index)
    
    return out

def readRawNetMat(path, type):
    if type in ["ICA", "schaefer", "yeo"]:
        df = pd.read_csv(path, header=None, sep=" ")
    elif type == "glasser":
        df = pd.read_csv(path, header=None, sep=",")
    elif type == "profumo":
        df = pd.read_csv(path, header=None, sep=" ")
        df = process_pfm_netmat(df)
    else:
        raise ValueError("Type MUST be 'profumo', 'ICA', 'scahefer', 'yeo', or 'glasser'")
    
    vectorized = df.to_numpy()[np.triu_indices(df.shape[0], k=1)] # k=1 option ensures diagonal is ignored

    return vectorized

def read_Subj(y=None, numFeatures_y=None, y_type = None):    
    # Second, process your NetMats #
    y_vectorized = readRawNetMat(y, y_type)
    y_tensor = torch.tensor(y_vectorized)

    out = torch.utils.data.TensorDataset(y=y_tensor)

    return out

class LocalTranslationsData(torch.utils.data.TensorDataset):
    def __init__(self, root="/Users/snaranjo/Desktop/neurotranslate/", raw_root="/Users/snaranjo/Desktop/neurotranslate/brain_representations/",
                 y_type=None, numFeatures_y=None, transform=None, pre_transform=None, pre_filter=None):
        
        self.root = root
        self.raw_root = raw_root
        self.y_type = y_type # 'profumo', 'ICA' or 'gradients'
        self.numFeatures_y = numFeatures_y #int
        
        self.y_raw_path = findNetMatDir(raw_root, y_type)
                        
        self.subj_list = list(np.genfromtxt(op.join(root, "local_subj_list.csv"), delimiter=",", dtype=int))
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        return {'y':sorted(list(ListRawNetMats(self.y_raw_path, self.y_type)))}
    
    @property
    def processed_file_names(self):
        return sorted(list(ListDirNoHidden(self.processed_dir)))

    @property
    def processed_dir(self):
        return op.join(self.root, "processed", "transformers","surfmap2netmat", f"{self.x_type}_d{self.numFeatures_x}_to_{self.y_type}_d{self.numFeatures_y}")
        
    def process(self):
        for i, y_fname in enumerate(self.raw_file_names['y']):
            y_path = op.join(self.y_raw_path, y_fname)
            
            subjData = read_Subj(y=y_path, numFeatures_y=self.numFeatures_y, y_type=self.y_type)
                                    
            torch.save(subjData, op.join(self.processed_dir, f"subj{self.subj_list[i]}.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(op.join(self.processed_dir, f"subj{self.subj_list[idx]}.pt"))
