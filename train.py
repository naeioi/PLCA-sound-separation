# training (decomposing) module

from numpy.linalg import norm
import numpy as np
from spectrogram import reconstruct

def cross_entropy(U, V):
    """
    Cross entropy of U and V. i.e. H(U, V)
        When U is fixed, H(U, V) decreases as V approximates to U
    """
    return -np.sum(U * np.log(V))

def train_given_Pf(Po, pPf, zn, maxstep=500, progress_step=50):
    """ 
    Given P(f, t), part of fixed trained Pf and number of classification |Z|, compute P(f|z), P(t|z) for all z
    P(f, t) is not necessarility normalized.
    """
    return train(Po, zn, maxstep, progress_step, pPf=pPf)    
    
def train(Po, zn, maxstep=500, progress_step=50, pPf=None):
    """
    Given P(f, t) and number of classification |Z|, compute P(f|z), P(t|z) for all z
    P(f, t) is not necessarility normalized.
    
    Parameters:
        Po : 2-D narray
            P(t, f).
        zn : int
            |Z|, number of possible value of latent variable Z.

    Return: (Pf, Pt, Pz)
        Pf : 2-D narray
            Pf[z][f] = P(f|z)
        Pt : 2-D array
            Pz[z][t] = P(t|z)
        Pz : 1-D array
            P(z)
    """ 
    eps = 1e-8
    step = 0
    
    if pPf is not None:
        zn1, _ = np.shape(pPf)
    
    P = np.abs(Po)
    P = P / np.sum(P)
    tn, fn = np.shape(P)
    Pf = np.random.random((zn, fn))
    
    if pPf is not None:
        Pf[:zn1, :] = pPf
        
    Pt = np.random.random((zn, tn))
    Pz = np.ones(zn) / zn
    # normailize row of Pf, Pt so that sum(P(f|z) | all z) = 1
    Pf /= np.sum(Pf, axis=1).reshape(-1, 1)
    Pt /= np.sum(Pt, axis=1).reshape(-1, 1)
    
    # P(z|f,t)
    Pztf = np.zeros((zn, tn, fn))
    
    oldentropy = cross_entropy(P, reconstruct(Pf, Pt, Pz))
    
    while True:
        # E-step. compute P(z|f,t) given P(f|z), P(t|z)
        # first compute P(z)*P(f|z)*P(t|z) for all (z,f,t)
        for z in range(zn):
            Pztf[z,:,:] = Pz[z] * Pt[z, :].reshape(-1, 1) * Pf[z, :]
        # normalize over z
        Pztf /= np.sum(Pztf, axis=0)
        
        # M-step. update P(z), P(f|z), P(t|z)
        for z in range(zn):
            PP = P * Pztf[z,:,:]
            Pz[z] = np.sum(PP)
            Pf[z,:] = np.sum(PP, axis=0) / Pz[z]
            Pt[z,:] = np.sum(PP, axis=1) / Pz[z]
            
        if pPf is not None:
            Pf[:zn1, :] = pPf
        
        entropy = cross_entropy(P, reconstruct(Pf, Pt, Pz))
        
        impr = oldentropy - entropy
        if (step > maxstep or (impr > 0 and impr < eps)):
            break
        else:            
            if step % progress_step == 0:
                print('Step %d: Entropy = %e, D(Entropy) = %e.\n' % (step, entropy, impr))
            oldentropy = entropy
    
        step += 1
        
    return (Pf, Pt, Pz)
