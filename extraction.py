        
        
# ====================== #



# ==================== #

# extract and reconstruct module

def extract_given_Pf(frames, Pf, extraz=None, maxstep=300, progress_step=50):
    """ Given original spectrogram and pretrained Pf, extract conforming and complementary wave """
    zn1, _ = np.shape(Pf)
    if extraz is None:
        extraz = zn1
    zn = zn1 + extraz
    
    Pf2, Pt, Pz = train_given_Pf(frames, Pf, zn, maxstep, progress_step)
    return (extract_partial(frames, Pf2, Pt, Pz, range(zn1)), \
            extract_partial(frames, Pf2, Pt, Pz, range(zn1, zn)))

def reconstruct(Pf, Pt, Pz):
    """Reconstruct P[t][f] from Pf, Pt, Pz"""
    return np.matmul(np.transpose(Pt), Pz[:, np.newaxis] * Pf)

def extract_partial(frames, Pf, Pt, Pz, partial=None):
    """ Given original spectrogram, Pf, Pt, Pz, extract partial features and return restored wave """
    if partial is None:
        partial = range(len(Pz))
    z = np.zeros_like(Pz)
    z[partial] = Pz[partial]
    extracted = reconstruct(Pf, Pz, z)
    return relative_restore(frames, extracted)

def relative_restore(fframes, rframes, ratio=1):
    """
    Restore wave from PLCA approximated spectrogram given original spectrogram
    
    Note that original spectrogram is necessary for it retains phase information, which is dropped when calculating absolute frequency.
    Assume spectrogram derived from 50% hanning window.
    
    fframes : 2-D narray
        narray of original spectrogram, whose first dimension is number of frames.
        Not neccessarily normalized. 
    
    rframes : 2-D narray
        narray of relative spectrogram. 
        Not neccessarily normalized. 
        
    ratio : float
        Sum of magnitude in restore spectrogram / sum of magnitude in original spectrogram.
    """
    """ TODO Possible caveat: is scale in freq domain linear? """
    # import pdb; pdb.set_trace();
    omag = np.abs(fframes)
    rmag = np.abs(rframes)
    scale = np.max(omag) / np.max(rmag) 
    rmag = ratio * scale * rmag
    frames = fframes * rmag / omag
    
    return restore(frames)
    
def restore(fframes):
    """
    Restore wave from spectrogram
    
    Assuming spectrogram derived from 50% hanning window
    
    fframes : 2-D narray
        narray of spectrogram, whose first dimension is number of frames.
    """
    frames = np.fft.irfft(fframes)
    # assume fframes derived from 50% overlap hanning window
    col, binSize = np.shape(frames)
    # align second half of first frame with first half of second frame
    flatted = frames.flat[int(np.ceil(binSize/2.0)):-binSize//2].reshape(col-1, binSize)
    flatted_shift = frames.flat[binSize:].reshape(col-1, binSize)
    # with hanning window's nice property: h(t) + h(1/2-t), where t in [0,1]
    # we can safely align then add frame to reconstruct the wave
    restored = flatted + flatted_shift
    # drop the redundant half of frame
    restored_wave = restored.reshape(-1, 2, binSize//2)[:,0,:].flat[:]
    return restored_wave

# ============== # 