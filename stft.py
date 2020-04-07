import warnings
import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import scipy.interpolate
import six
from numpy.lib.stride_tricks import as_strided
# from numba import jit
#
# from . import time_frequency
from fft import get_fftlib

MAX_MEM_BLOCK = 2**8 * 2**10

def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):


    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    fft = get_fftlib()

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.rfft(fft_window *
                                             y_frames[:, bl_s:bl_t],
                                             axis=0)
    return stft_matrix

def get_window(window, Nx, fftbins=True):

    if six.callable(window):
        return window(Nx)

    elif (isinstance(window, (six.string_types, tuple)) or
          np.isscalar(window)):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)
    else:
        print('Invalid window specification: {}'.format(window))

def pad_center(data, size, axis=-1, **kwargs):

    kwargs.setdefault('mode', 'constant')
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    return np.pad(data, lengths, **kwargs)

def frame(x, frame_length=2048, hop_length=512, axis=-1):

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:

        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]

    elif axis == 0:

        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)

    return as_strided(x, shape=shape, strides=strides)

def amplitude_to_db(S, ref=1.0, amin=1e-5, top_db=80.0):

    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('amplitude_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call amplitude_to_db(np.abs(S)) instead.')

    magnitude = np.abs(S)

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    power = np.square(magnitude, out=magnitude)

    return power_to_db(power, ref=ref_value**2, amin=amin**2,
                       top_db=top_db)

def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):

    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('power_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call power_to_db(np.abs(D)**2) instead.')
        magnitude = np.abs(S)
    else:
        magnitude = S

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec