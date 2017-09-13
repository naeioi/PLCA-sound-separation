# extract and reconstruct module

import numpy as np
from train import train_given_Pf, train
from spectrogram import relative_restore, reconstruct

def extract_given_Pf(frames, Pf, extraz=None, maxstep=300, progress_step=50):
    """ Given original spectrogram and pretrained Pf, extract conforming and complementary wave """
    zn1, _ = np.shape(Pf)
    if extraz is None:
        extraz = zn1
    zn = zn1 + extraz
    
    Pf2, Pt, Pz = train_given_Pf(frames, Pf, zn, maxstep, progress_step)
    return (extract_partial(frames, Pf2, Pt, Pz, range(zn1)), \
            extract_partial(frames, Pf2, Pt, Pz, range(zn1, zn)))

def extract_partial(frames, Pf, Pt, Pz, partial=None):
    """ Given original spectrogram, Pf, Pt, Pz, extract partial features and return restored wave """
    if partial is None:
        partial = range(len(Pz))
    z = np.zeros_like(Pz)
    z[partial] = Pz[partial]
    extracted = reconstruct(Pf, Pz, z)
    return relative_restore(frames, extracted)
