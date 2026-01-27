import torch 
import numpy as np

# =========================================== Kraken Train fcns =========================================== #
def xycorr(x,y,axis=1,epsilon=0):
    """
    **FROM KRAKENCODER.loss.py**

    Compute correlation between all pairs of rows in x and y (or columns if axis=0)
    
    x: torch tensor or numpy array (Nsubj x M), generally the measured data for N subjects
    y: torch tensor or numpy array (Nsubj x M), generally the predicted data for N subjects
    axis: int (optional, default=1), 1 for row-wise, 0 for column-wise
    
    Returns: torch tensor or numpy array (Nsubj x Nsubj)
    
    NOTE: in train.py we always call cc=xycorr(Ctrue, Cpredicted)
    which means cc[i,:] is cc[true subject i, predicted for all subjects]
    and thus top1acc, which uses argmax(xycorr(true,predicted),axis=1) is:
    for every TRUE output, which subject's PREDICTED output is the best match
    """
    if epsilon != 0:
        # Adds clipping (added by Fyzeen)
        if torch.is_tensor(x):
            cx = x - x.mean(keepdim=True, dim=axis)
            cy = y - y.mean(keepdim=True, dim=axis)
            norm_cx = torch.sqrt(torch.sum(cx ** 2, dim=axis, keepdim=True))
            norm_cy = torch.sqrt(torch.sum(cy ** 2, dim=axis, keepdim=True))
            norm_cx = torch.clamp(norm_cx, min=epsilon)
            norm_cy = torch.clamp(norm_cy, min=epsilon)
            cx = cx / norm_cx
            cy = cy / norm_cy
            cc = torch.matmul(cx, cy.t())
        else:
            cx = x - np.mean(x, axis=axis, keepdims=True)
            cy = y - np.mean(y, axis=axis, keepdims=True)
            norm_cx = np.sqrt(np.sum(cx ** 2, axis=axis, keepdims=True))
            norm_cy = np.sqrt(np.sum(cy ** 2, axis=axis, keepdims=True))
            norm_cx = np.clip(norm_cx, a_min=epsilon, a_max=None)
            norm_cy = np.clip(norm_cy, a_min=epsilon, a_max=None)
            cx = cx / norm_cx
            cy = cy / norm_cy
            cc = np.matmul(cx, cy.T)
    else:
        if torch.is_tensor(x):
            cx=x-x.mean(keepdims=True,axis=axis) #gets mean of features per subj and demans each with their respective mean
            cy=y-y.mean(keepdims=True,axis=axis)
            cx=cx/torch.sqrt(torch.sum(cx ** 2,keepdims=True,axis=axis)) # normalizes based on sqrt(sum_of_squares_demean)
            cy=cy/torch.sqrt(torch.sum(cy ** 2,keepdims=True,axis=axis))
            cc=torch.matmul(cx,cy.t()) # similarity scaler?
        else:
            cx=x-x.mean(keepdims=True,axis=axis)
            cy=y-y.mean(keepdims=True,axis=axis)
            cx=cx/np.sqrt(np.sum(cx ** 2,keepdims=True,axis=axis))
            cy=cy/np.sqrt(np.sum(cy ** 2,keepdims=True,axis=axis))
            cc=np.matmul(cx,cy.T)
    return cc

def correye(x,y, epsilon=0):
    """
    **FROM KRAKENCODER.loss.py**

    Loss function: mean squared error between pairwise correlation matrix for xycorr(x,y) and identity matrix
    (i.e., want diagonal to be near 1, off-diagonal to be near 0)
    you have identity matrix with 0 at off-daig and 1 at diag, and we want to reduce frobenius norm between apprx cc and that I mat
    """
    cc=xycorr(x,y, epsilon=epsilon)
    #need keepdim for some reason now that correye and enceye are separated
    loss=torch.linalg.matrix_norm((cc-torch.eye(cc.shape[0],device=cc.device)), ord='fro', keepdim=True) #og device = cc.device
    return loss


def distance_loss(x,y, margin=None, neighbor=False, epsilon=0):
    """
    **FROM KRAKENCODER.loss.py**

    Loss function: difference between self-distance and other-distance for x and y, with optional margin
    If neighbor=True, reconstruction loss applies only to nearest neighbor distance, otherwise to mean distance between all
        off-diagonal pairs.
    
    Inputs:
    x: torch tensor (Nsubj x M), generally the measured data
    y: torch tensor (Nsubj x M), generally the predicted data
    margin: float, optional margin for distance loss (distance above margin is penalized, below is ignored)
    neighbor: bool, (optional, default=False), True for maximizing nearest neighbor distance, False for maximizing mean distance
    
    fcn values:
    d: SUBxSUB L2 matrix! diagonal is (true_i,pred_i) off-diagonal is (true_i,pred_j) i!=j
    dtrace: scalar and sum of all d_ii values
    dself: mean L2 norm of self (true_i,pred_i)
    dnei: for recon loss and nearest neighbor seperation. SUBxSUB L2 mat (d) + I*d.max() so diagonal mat with only the max distance.
    So adds maximum distance from all SUBxSUB elements and adds them to the diagonal of "d
    dother: "

    Returns: 
    loss: torch FloatTensor, difference between self-distance and other-distance
    """
    # d is subxsub because l2 norm is on self to self and self to others, ideally distance is larger in off diagonal and smaller in diagonal
    if epsilon != 0:
        d = torch.clamp(torch.cdist(x, y), min=epsilon)
    else:
        d=torch.cdist(x,y) # euclidean distance, L2 norm in paper for latent space (BLUE) pnorm=2 by default output is subxsub
    dtrace=torch.trace(d) # summ of diag ele, small means distance(x,y) is low 
    dself=dtrace/d.shape[0] #mean predicted->true distance -- avg distance x_subja to y_subja
    
    if neighbor:
        dnei=d+torch.eye(d.shape[0],device=d.device)*d.max() #subj with biggest L2 norm err?
        #mean of row-wise min and column-wise min
        dother=torch.mean((dnei.min(axis=0)[0]+dnei.min(axis=1)[0])/2) # divided by 2? then mean of that..
    else:
        dother=(torch.sum(d)-dtrace)/(d.shape[0]*(d.shape[0]-1)) #mean predicted->other distance
    
    if margin is not None:
        #dother=torch.min(dother,margin)
        #dother=-torch.nn.ReLU()(dother-margin) #pre 4/5/2024
        #if dother<margin, penalize (lower = more penalty).
        #if dother>=margin, ignore
        #standard triplet loss: torch.nn.ReLU()(dself-dother+margin) or torch.clamp(dself-dother+margin,min=0)
        dother=-torch.nn.ReLU()(margin-dother) #new 4/5/2024
    
    loss=dself-dother
    return loss