import pypsrfits
import evalmetricsimv0 as evalmetricsim
from pathlib import Path

DIR = '/datasets/work/mlaifsp-sparkes/work/sparkesX/multi/simplepulse/'
FNAME = 'simplepulse_multi_01.sf'
OUTPUT_DIR = '../outputs/'


def main(fname, plot_inputs = False, plot_predictions=False):
    Path(OUTPUT_DIR).mkdir(parents = True, exist_ok=True)
    
    # Load file
    psrfile = pypsrfits.PSRFITS(DIR + fname)
    
    fout = open(OUTPUT_DIR + FNAME[:-3] + '.out', 'w')
    fout.write('frame,label,filename\n')
    
    for nrow in range(psrfile.nrows_file):
        bdata, times, _ = psrfile.getData(nrow, None, get_ft=True, 
            squeeze=True, transpose=True)
        #####
        # RUN ALGORITHM
        # Input bdata: 2d array with dimension [nfrequency, ntime]
        #####
        
        # Save output for all frames
        #ypred = 0 # Prediction: 0 for no event or 1 for event
        #fout.write(f"{nrow},{ypred},{FNAME[:-3]}\n")
    
    fout.close()
    
    # Get sim labels
    _, y_sim, sim, tframe = evalmetricsim.read_simlabel(DIR + fname, datadir='')


if __name__ == '__main__':
    main(FNAME)
