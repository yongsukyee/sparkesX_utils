import numpy as np
import pandas as pd
from pypsrfits import PSRFITS


def bin_label(data, bins, clip=False):
    """
    Bin data for label, 0 for normal and 1 for event
    
    Parameters
    ----------
        data: array
        bins: Bins
        clip: If True, then clip value from 0 to 1. Default without clipping.
    
    Returns
    ----------
        array of label counts
    """
    
    hist = np.histogram(data, bins=bins)[0]
    
    return hist if not clip else np.clip(hist, 0, 1, out=hist)


def df_frame(df, frame):
    """Set DataFrame index for event in corresponding frame"""
    
    return df.set_index(frame).sort_index()


def read_simlabel(filename, datadir='../data/'):
    """
    Read PSRFITS simulation file and get labels
    
    Parameters
    ----------
        filename: Name of file
        datadir: Data directory
    
    Returns
    ----------
        sim: DataFrame of simulation labels with added columns [log10(AMP), log10(T1)]
        y_sim: array of simulation labels in tframe
        sim_frame: Same as sim with labels indexed by corresponding frame
        tframe: Bins in time frame with NSBLK*TBIN per time frame (SUBINT)
    """
    
    psrfile = PSRFITS(f'{datadir}{filename}')
    sim = pd.DataFrame(np.array(psrfile.fits['EVENTS'][:]).byteswap().newbyteorder())
    print(f"Read simulation file >> {filename}")
    
    # Create log10 columns for small values
    for param in ['AMP', 'T1']:
        if sim[param].median() < 1.:
            sim[f'log10({param})'] = np.log10(sim[param]).values
    
    # Bin time frame
    tperframe = psrfile.nsblk*psrfile.tbin
    total_time = tperframe*psrfile.nrows_file
    tframe = np.arange(0, total_time+tperframe, tperframe)
    
    # Create labels
    y_sim = bin_label(sim['T0'].values, bins=tframe, clip=False)
    sim_frame = df_frame(sim, frame=np.where(y_sim)[0])
    
    return sim, y_sim, sim_frame, tframe
