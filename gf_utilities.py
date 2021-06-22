# Utilities for Gaussian fit program

import math
import scipy.stats
import numpy as np
import pickle

from custom_exceptions import DimensionError
from pypulse.singlepulse import SinglePulse


def get_data_from_asc( asc_file, duty = None ):
    data = np.loadtxt( asc_file )
    x = data[:, 0]
    y = data[:, 1]
    if duty is not None:
        y = remove_base( y, duty )
    return x, y

def RMS( array ):

    '''
    Returns the RMS of a data array
    '''

    return np.sqrt( np.mean( np.power( array, 2 ) ) )

def get_1D_OPW_mask( vector, **kwargs ):

    if vector.ndim != 1:
        raise DimensionError( "Input data must be 1 dimensional to create an OPW mask." )

    mask = []
    sp_dat = SinglePulse( vector, **kwargs )

    while len( mask ) < len( vector ):
        if len( mask ) in sp_dat.opw:
            mask.append( False )
        else:
            mask.append( True )

    mask = np.asarray( mask )
    return mask

def get_base( prof_data, duty ):
     # get profile mask to determine off-pulse bins. May need tweaking for the user's needs
     if len( prof_data ) > 100:
         mask = get_1D_OPW_mask( prof_data, windowsize = (len( prof_data ) - 100) )
     else:
         mask = get_1D_OPW_mask( prof_data, windowsize = (len( prof_data ) - 10) )
     # select those with mask==0, i.e. baseline bins
     baseline = prof_data[mask == 0]
     # get mean and rms of baseline
     base_mean = np.mean( baseline )
     #base_rms = np.std( baseline )
     base_rms = RMS( baseline )

     # return tuple consisting of mean and rms of baseline
     return base_mean, base_rms


# Returns the profile data minus the baseline
def remove_base( prof_data, duty ):

     baseline, base_rms = get_base( prof_data, duty )

     # remove baseline mean from profile in place
     prof_data = prof_data - baseline

     return prof_data

def multi_norm( x, *args ):
    ret = None
    n_gauss = len( args )//3

    if len( args ) % 3 != 0:
        print( "Args supplied must be a multiple of 3 of form: mu, sig, amp" )
    else:
        ret = 0
        for i in np.arange( 0, 3*n_gauss, 3 ):
            ret += args[i + 2]*scipy.stats.norm.pdf( x, loc = args[i], scale = args[i + 1] )
    return ret


def norm( x, m, s, k ):
    ret = k*scipy.stats.norm.pdf( x, loc = m, scale = s )
    return ret
