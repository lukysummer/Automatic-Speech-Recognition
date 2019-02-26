import numpy as np
import soundfile
from numpy.lib.stride_tricks import as_strided


### taken from Udacity's NLP Nanodegree ###
def spectrogram(samples, 
                fft_length = 256,   # number of elements in fft window
                sample_rate = 2, 
                hop_length = 128):  # relative offset between neighboring fft windows
    """
    Computes spectrogram for a real signal; parameters follow naming convention of matplotlib.mlab.specgram
    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x
    Note:
        This is a truncating computation- 
        e.g. if fft_length=10, hop_length=5 & the signal has 23 elements, then the last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of matplotlib.mlab.specgram which is the same as matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs



def spectrogram_from_file(filename, 
                          frame_stride = 10,  # Step size in milliseconds between windows
                          frame_window = 20,  # FFT window size in milliseconds
                          max_freq = None,  # Only FFT bins corresponding to frequencies between [0, max_freq] are returned
                          eps = 1e-14):  # Small value to ensure numerical stability (for ln(x))
    """ Calculate the log of linear spectrogram from FFT energy  """
    
    with soundfile.SoundFile(filename) as sound_file:
        audio = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate # number of sampels/second (in Hz)
        
        if audio.ndim >= 2:   # if there is more than 1 channels
            audio = np.mean(audio, 1)
            
        if max_freq is None:
            max_freq = sample_rate / 2
            
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             " sample rate")
        if frame_stride > frame_window:
            raise ValueError("step size must not be greater than window size")
            
        hop_length = int(0.001 * frame_stride * sample_rate)   # [# of samples]
        
        fft_length = int(0.001 * frame_window * sample_rate)   # [# of samples]
        print(hop_length, fft_length)
        pxx, freqs = spectrogram(audio, 
                                 fft_length = fft_length, 
                                 sample_rate = sample_rate,
                                 hop_length = hop_length)
        
        ind = np.where(freqs <= max_freq)[0][-1] + 1
        
    return np.transpose(np.log(pxx[:ind, :] + eps)), pxx, freqs


out, x, freqs = spectrogram_from_file("sample.wav",
                                         frame_stride = 10,
                                         frame_window = 20,
                                         max_freq = 8000)
print(out.shape)
print(x.shape)
print(freqs.shape)
print(freqs[-10:])