##################################################
# COMPUTE THE CROSS-CORRELATION BETWEEN LINES IN FREQUENCY
# Author: Suk Yee Yong
# Usage: python xcorrelation.py
#   Set Config
##################################################

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
import pypsrfits


# Config
basedatadir = os.path.join(os.path.expanduser('~'), 'Desktop/sparkesX_data')
survey = 'multi'
simsignal = 'combo'
filename = 'combo+rfi_multi_01'
iframe = 99
resize_dim = 512 # Original W: 4096
corr_mode = 'same' # 'full', 'valid', 'same'
nfreq_slide = None # None for cross-correlation of consecutive lines, int for index of sliding frequency slice

# TODO change datadir
datadir = os.path.join(basedatadir, 'sparkesX', survey, simsignal)
imshow_kwargs = {'aspect': 'auto', 'origin': 'lower', 'interpolation': 'nearest', 'rasterized': True, 'cmap': 'BuPu'}

# ------------------------------ #
# Average data across time
# ------------------------------ #
image_avgt = lambda image, resize_dim: np.mean(image.reshape(image.shape[0], resize_dim, int(image.shape[1]//resize_dim)), axis=-1) # (HxW_resize)
bdata = np.array(pypsrfits.PSRFITS(os.path.join(datadir, f"{filename}.sf")).getData(iframe, iframe, get_ft=False, squeeze=True, transpose=True, print_info=False), dtype=np.float32) # (HxW)
image = image_avgt(bdata, resize_dim=resize_dim)

# Plot comparison between original and resized
fig, axes = plt.subplots(1, 2, figsize=(12,4), sharey=True, gridspec_kw={'wspace': 0.})
axes[0].set_title(f"Image size={bdata.shape}")
axes[0].imshow(bdata, **imshow_kwargs)
axes[1].set_title(f"Resized={image.shape}")
axes[1].imshow(image, **imshow_kwargs)
fig.savefig(f"avgtime.pdf", bbox_inches='tight')
plt.show()

# ------------------------------ #
# Zero normalized cross-correlation
# ------------------------------ #
znorm_data = lambda data: (data - np.mean(data))/np.std(data)

# Relative to nfreq_slide
if isinstance(nfreq_slide, int):
    image_sliced = np.concatenate((image[slice(0, nfreq_slide)], image[slice(nfreq_slide+1, None)]))
    xcorr = [signal.correlate(znorm_data(fi), znorm_data(image[nfreq_slide]), mode=corr_mode)/min(len(image[nfreq_slide]), len(fi)) for fi in image_sliced]
    lags_indices = signal.correlation_lags(image[nfreq_slide].shape[0], image_sliced[0].shape[0], mode=corr_mode)
# Between consecutive slice
else:
    xcorr = [signal.correlate(znorm_data(fi), znorm_data(fj), mode=corr_mode)/min(len(fj), len(fi)) for fi, fj in zip(image, image[1:])]
    lags_indices = signal.correlation_lags(image[0].shape[0], image[1].shape[0], mode=corr_mode)
lags = lags_indices[np.argmax(xcorr, axis=1)]
bin_lags = np.histogram(lags, bins=lags_indices)[0] # TODO: can increase the bin size to sum signals along x

# ------------------------------ #
# Plots
# ------------------------------ #
# For one cross-correlation
# xcorr_i = 20
# plt.plot(lags_indices, xcorr[xcorr_i])
# plt.xlabel('Lags')
# plt.ylabel('Cross-correlation')
# plt.show()

# Max shift
fig, axes = plt.subplots(2, 1, figsize=(6,8), sharex=True, gridspec_kw={'hspace': 0})
axes[0].set_title("Time shift")
axes[0].scatter(lags, np.arange(lags.shape[0]))
axes[0].set_ylabel('Frequency')
axes[1].plot(lags_indices[:-1], bin_lags)
axes[1].set_xlabel('Lag Indices')
axes[1].set_ylabel('Sum')
fig.savefig(f"lag.pdf", bbox_inches='tight')
plt.show()

# Cross-correlation
fig, axes = plt.subplots(2, 1, figsize=(6,8), sharex=True, gridspec_kw={'hspace': 0})
axes[0].set_title("Cross-correlation" + f"{f'at reference frequency index={nfreq_slide}' if isinstance(nfreq_slide, int) else ''}")
axes[0].imshow(xcorr, extent=[lags_indices[0], lags_indices[-1], 0, bdata.shape[0]-1], **imshow_kwargs)
axes[0].set_ylabel('Frequency')
axes[1].plot(lags_indices, np.sum(xcorr, axis=0))
axes[1].set_xlabel('Lag Indices')
axes[1].set_ylabel('Sum')
fig.savefig(f"xcorr.pdf", bbox_inches='tight')
plt.show()
