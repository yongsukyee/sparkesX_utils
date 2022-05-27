#! /usr/bin/env python

# Simple code to read search-mode PSRFITS data arrays into python
# Adopted from Paul Demorest: https://github.com/demorest/pypsrfits
# Updated on October 2020 by Suk Yee Yong
# Description: Read 1-, 2-, 4-, 8-bit PSRFITS data
# - python3 syntax
# - itertools for loop
# - removed extra dimension in result
# - added transpose dimension option in getData()


import fitsio
import itertools
import numpy

class PSRFITS:
    def __init__(self, filename):
        # Reference: https://www.atnf.csiro.au/research/pulsar/psrfits_definition/PsrfitsDocumentation.html
        self.filename = filename
        self.fits = fitsio.FITS(filename,'r')
        self.hdr = self.fits[0].read_header()
        self.subhdr = self.fits['SUBINT'].read_header()
        
        # Read primary header info
        self.srcname = str(self.hdr['SRC_NAME']) # Source or scan ID
        self.ra = str(self.hdr['RA']) # Right ascension (hh:mm:ss.ssss)
        self.dec = str(self.hdr['DEC']) # Declination (-dd:mm:ss.sss)
        
        # Read SUBINT info
        self.nrows_file = int(self.subhdr['NAXIS2']) # Number of rows in table
        self.poltype = self.subhdr['POL_TYPE'] # Polarisation identifier
        self.npol = int(self.subhdr['NPOL']) # Number of polarisations
        self.tbin = float(self.subhdr['TBIN']) # Time per bin [s]
        self.nbits = int(self.subhdr['NBITS']) # Number of bits
        self.nchan = int(self.subhdr['NCHAN']) # Number of frequency channels
        self.nsblk = int(self.subhdr['NSBLK']) # Samples

    def getFreqs(self, row=0):
        """Return the frequency array from the specified subint."""
        return self.fits['SUBINT']['DAT_FREQ'][row]

    def getData(self, start_row=0, end_row=None, downsamp=1, fdownsamp=1, apply_scales=False, get_ft=False, squeeze=False, transpose=False):
        """Read the data from the specified rows and return it as a
        single array.  Dimensions are [NSBLK, NPOL, NCHAN].

        options:
          start_row: first subint to read (0-based index)

          end_row: final subint to read.  None implies end_row=start_row.
            Negative values imply offset from the end, i.e.
            get_data(0,-1) would read the entire file.  (Don't forget 
            that PSRFITS files are often huge so this might be a bad idea).

          downsamp: downsample the data in time as they are being read in.
            The downsample factor should evenly divide the number of spectra
            per row.  downsamp=0 means integrate each row completely.

          fdownsamp: downsample the data in freq as they are being read in.
            The downsample factor should evenly divide the number of channels.

          apply_scales: set to False to avoid applying the scale/offset
            data stored in the file.

          get_ft: if True return time and freq arrays as well.

          squeeze: if True, "squeeze" the data array (remove len-1 
            dimensions).

          transpose: if True, "transpose" the order of data axes (flip first and 
            last dimensions).
        """

        if self.hdr['OBS_MODE'].strip() != 'SEARCH':
            raise RuntimeError("get_data() only works on SEARCH-mode PSRFITS")

        if downsamp == 0:
            downsamp = 1

        if downsamp > self.nsblk:
            downsamp = self.nsblk

        if fdownsamp == 0:
            downsamp = self.nchan

        if fdownsamp > self.nchan:
            fdownsamp = self.nchan

        if end_row == None:
            end_row = start_row

        if end_row < 0:
            end_row = self.nrows_file + end_row

        if self.nsblk % downsamp > 0:
            raise RuntimeError(f"downsamp does not evenly divide NSBLK={self.nsblk}.")

        if self.nchan % fdownsamp > 0:
            raise RuntimeError(f"fdownsamp does not evenly divide NCHAN={self.nchan}.")

        nrows_tot = end_row - start_row + 1
        nsblk_ds = self.nsblk // downsamp
        nchan_ds = self.nchan // fdownsamp
        tbin_ds = self.tbin * downsamp

        # Data types of the signed and unsigned
        if self.nbits <= 8:
            signed_type = numpy.int8
            unsigned_type = numpy.uint8
        # elif self.nbits == 16:
        #     signed_type = numpy.int16
        #     unsigned_type = numpy.uint16
        # elif self.nbits == 32:
        #     signed_type = numpy.float32
        #     unsigned_type = numpy.float32
        else:
            raise RuntimeError(f"Unhandled number of bits={self.nbits}")

        # Allocate the result array
        sampresult = numpy.zeros(self.nchan, dtype=numpy.float32)
        result = numpy.zeros((nrows_tot * nsblk_ds, self.npol, nchan_ds), dtype=numpy.float32)
        if get_ft:
            # freqs = numpy.zeros(nchan_ds)
            times = numpy.zeros(nrows_tot * nsblk_ds)

        signpol = 1
        if 'AABB' in self.poltype:
            signpol = 2

        # Assume frequency is the same across time
        if get_ft:
            freqs = self.getFreqs(row=start_row)[:nchan_ds]
            if fdownsamp != 1:
                freqs = freqs.reshape((-1, fdownsamp)).mean(1)

        # Iterate over rows
        for irow in range(nrows_tot):
            print(f"Reading subint {irow+start_row}")

            if apply_scales:
                offsets = self.fits['SUBINT']['DAT_OFFS'][irow + start_row]
                scales = self.fits['SUBINT']['DAT_SCL'][irow + start_row]
                scales = scales.reshape((self.npol, self.nchan))
                offsets = offsets.reshape((self.npol, self.nchan))

            if get_ft:
                t0_row = self.fits['SUBINT']['OFFS_SUB'][irow + start_row] - (self.fits['SUBINT']['TSUBINT'][irow + start_row] / 2.0)
                # freqs_row = self.fits['SUBINT']['DAT_FREQ'][irow + start_row]

            dtmp = self.fits['SUBINT']['DATA'][irow + start_row]

            if nsblk_ds//dtmp.shape[0] != 1:
                if self.nbits == 1:
                    dtmp = numpy.unpackbits(dtmp, axis=0, bitorder='big')

            # Fix up 16 bit data type
            if (self.nbits == 16):
                dtmp = numpy.fromstring(dtmp.tostring(), dtype=numpy.int16)
                dtmp = dtmp.reshape((self.nsblk, self.npol, self.nchan))

            # Iterate over samples and polarisations
            for isamp, ipol in itertools.product(range(nsblk_ds), range(self.npol)):
            # for isamp in range(nsblk_ds):
            #     for ipol in range(self.npol):
                    if ipol < signpol: 
                        data_type = unsigned_type
                    else: 
                        data_type = signed_type

                    sampresult = dtmp[isamp * downsamp:(isamp + 1) * downsamp, ipol, :].astype(data_type).mean(0)

                    if get_ft: 
                        times[irow * nsblk_ds + isamp] = t0_row + (isamp + 0.5) * tbin_ds

                    if apply_scales:
                        sampresult *= scales[ipol, :]
                        sampresult += offsets[ipol, :]

                    if fdownsamp == 1:
                        result[irow * nsblk_ds + isamp, ipol, :] = sampresult
                        # Assumes freqs don't change:
                        # if get_ft:
                        #     freqs[:] = freqs_row[:]
                    else:
                        result[irow * nsblk_ds + isamp, ipol, :] = sampresult.reshape((-1, fdownsamp)).mean(1)
                        # if get_ft: 
                        #     freqs[:] = freqs_row.reshape((-1, fdownsamp)).mean(1)

        if squeeze: result = result.squeeze()

        if transpose: result = result.transpose()

        if get_ft:
            return (result, times, freqs)
        else:
            return result

