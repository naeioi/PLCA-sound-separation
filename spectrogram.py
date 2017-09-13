import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

""" short time fourier transform of audio signal """
def stft(sig, frameSize=2**10, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = int(np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale).astype(int))
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:   
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs    

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)
    
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    
    timebins, freqbins = np.shape(ims)
    
    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

def reconstruct(Pf, Pt, Pz):
    """Reconstruct P[t][f] from Pf, Pt, Pz"""
    return np.matmul(np.transpose(Pt), Pz[:, np.newaxis] * Pf)

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
