#!/usr/local/bin/python3.9
# Advanced mulit-Gaussian fitting

import sys, os, ast
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats, scipy.optimize
from numpy import exp, pi, sqrt
import gf_utilities as gfu
from custom_exceptions import DimensionError, ArgumentError

################## USER DEFINITIONS ################

PSR = "J0000+0000"
FE = "lbw"
NCHAN = 1
NBIN = 2048

FILE_PREFIX = ""
SAVE_DIRECTORY = os.getcwd()



###################### DEFAULTS ####################
GUESS = [ NBIN//2, 5, 2000 ]
NGAUSS = 3
####################################################

def print_parameters( p0, p1, n ):
    for i in np.arange( 0, 3*n, 3 ):
        print( p0[i], " -> ", p1[i], "\t", p0[i + 1], " -> ", p1[i + 1], "\t", p0[i + 2], " -> ", p1[i + 2] )

def get_gaussians( p1, n ):
    '''
    Obtains the list of individual Gaussians that comprise the complete fit.

    Inputs:
                p1: the fitted parameter output from scipy
                n:  number of Gaussians used

    Returns:
                ind_gaussians: a list of the individual Gaussians that comprise the fit
    '''

    ind_gaussians = []
    for i in np.arange( 0, 3*n, 3 ):
        ind_gaussians.append( [ p1[i], p1[i + 1], p1[i + 2] ] )
    ind_gaussians = np.array( ind_gaussians )
    return ind_gaussians

def save_all( n, ig, m, abs_dir = os.getcwd(), file_prefix = "" ):
    '''
    Saves both the full fit and the individual Gaussian list as numpy arrays.

    Inputs:
                n:              number of Gaussians used
                ig:             list containing the individual Gaussians
                m:              Gaussian output from the multinorm fit
                abs_dir:        the directory to save to
                file_prefix:    anything you would like before the main filename
    '''

    np.save( os.path.join( abs_dir, f"{file_prefix}_{PSR}_{FE}_nchan{NCHAN}_nbin{NBIN}_{n}gaussianfit_individualgaussians.npy" ), ig )
    np.save( os.path.join( abs_dir, f"{file_prefix}_{PSR}_{FE}_nchan{NCHAN}_nbin{NBIN}_{n}gaussianfit.npy" ), m )
    print( "Save successful!" )


def get_best_gaussian_fit( x, y, remove_base = True, m_gauss = 8, bp = 15, p_wid = 150, guess = [ 1024, 5, 800 ], plot_chisq = False, is_masked = True, verbose = True ):

    '''
    Obtains the best Gaussian fit from the initial guesses.

    Inputs:
                x, y:           data to fit
                remove_base:    remove baseline before fitting (boolean)
                m_gauss:        maximum number of Gaussians to try
                bp:             breakpoint (if m_gauss is never reached)
                p_wid:          window size to take the mean over
                guess:          initial guess for the Gaussians
                plot_chisq:     choose whether to plot the chi2 values (boolean)
                is_masked:      let the fitting algorithm know if your array is masked
                verbose:        print more information to the console

    Returns:
                m:              the Gaussian output constructed from the individual fits
                c:              the array of chi2 values for each fit
                ind_gaussians:  the individual Gaussian profiles
                mask:           any mask that was used, or 'm' if none were used.
    '''

    if remove_base:
        y = gfu.remove_base( y, 0.05 )

    if not isinstance( guess, np.ndarray ):
        guess_shape = np.array( guess ).shape
    else:
        guess_shape = guess.shape

    n_gauss = 0
    params, c = [], []
    while ( len(c) < m_gauss ) and ( n_gauss < bp ):

        if len( guess_shape ) == 1:
            params.extend( guess )
        elif len( guess_shape ) == 2:
            try:
                params.extend( guess[ n_gauss ] )
            except IndexError:
                params.extend( guess[0] )
        else:
            raise DimensionError( f"Initial guess parameters must be a Nx3 array. Current shape of array is: {guess_shape}" )

        n_gauss = len( params )//3

        try:
            fitted_params,_ = scipy.optimize.curve_fit( gfu.multi_norm, x, y, p0 = params )
            if ( n_gauss == bp ) and verbose:
                print( f"Maximum number of tries reached ({n_gauss})" )
        except RuntimeError:
            if len( guess_shape ) == 1:
                fitted_params = np.append( fitted_params, guess )
            elif len( guess_shape ) == 2:
                try:
                    fitted_params = np.append( fitted_params, guess[ n_gauss ] )
                except IndexError:
                    fitted_params = np.append( fitted_params, guess[0] )
            else:
                raise DimensionError( "You definitely shouldn't be able to see this error message." )

            if verbose:
                print( f"No fit for {n_gauss} gaussians" )
                if n_gauss == bp:
                    print( f"Maximum number of tries reached ({n_gauss})" )
            continue

        m = gfu.multi_norm( x, *fitted_params )

        if is_masked:
            mask = gfu.get_1D_OPW_mask( m, windowsize = ( len(m) - p_wid )  )
            for i, elem in enumerate( m ):
                if mask[i] == False:
                    m[i] = 0
            chi2, p = scipy.stats.chisquare( y[mask == 1], f_exp = m[mask == 1] )
            if verbose:
                print( f"Chi-sq for {n_gauss} gaussians: ", chi2 )
            c.append( chi2 )

            if plot_chisq:
                plt.plot( c[1:] )
                plt.show()
                plt.close()
        else:
            mask = m

    if verbose:
        print_parameters( params, fitted_params, n_gauss )

    ind_gaussians = get_gaussians( fitted_params, n_gauss )

    return m, c, ind_gaussians, mask


# TESTING
if __name__ == "__main__":

    if len(sys.argv) == 1 or sys.argv[1] == "-h" or sys.argv[1] == "help":
        print(f'''
                Gaussian fitting
               Henryk T. Haniewicz
        --------------------------------

Usage: gaussian_fit.py file [max_num_gauss = 3] [guesses]

where [] indicates that the argument is optional.

file:           the ASCII data file
                should have two columns: Index and Y-value

max_num_gauss:  maximum number of Gaussians to try (default = 3)

guesses:        list of guesses (default = {GUESS})
                format: "[[mid_point, FWHM, amplitude], ..., []]"
        ''')
        sys.exit(1)
    elif len(sys.argv) == 3:
        NGAUSS = int(sys.argv[2])
    elif len(sys.argv) == 4:
        NGAUSS = int(sys.argv[2])
        GUESS = ast.literal_eval( sys.argv[3] )
    elif len(sys.argv) > 4:
        raise ArgumentError( f"Too many arguments. Maximum should be 3. You have {len(sys.argv) - 1}" )

    asc = sys.argv[1]

    # Read in the ASCII data
    x, y = gfu.get_data_from_asc( asc )

    #y -= min(y)

    # Get the multinorm fit with the best chi2
    m, c, ig, msk = get_best_gaussian_fit( x, y, remove_base = False, m_gauss = NGAUSS, p_wid = NBIN//16, guess = GUESS )

    # Plots individual Gaussians. Comment out if this isn't necessary
    #for p in ig:
    #    plt.plot( x, norm( x, p[0], p[1], p[2] ) )

    # Build the full Gaussian profile
    out = gfu.norm( x, ig[0][0], ig[0][1], ig[0][2] )
    for i in np.arange(1, len(ig)):
        out = gfu.norm( x, ig[i][0], ig[i][1], ig[i][2] )


    # Plot the Gaussian (red) and the original data (grey) together
    plt.plot( x, out, color = 'r' )
    plt.plot( x, y, 'k', alpha = 0.7 )
    #plt.plot( x[ msk == 1 ], m[ msk == 1 ], 'k' ) # Uncomment if output is masked
    plt.show()

    # Save both the full profile and the Gaussian breakdown as Numpy arrays. Comment out for testing features
    #save_all( NGAUSS, ig, out, abs_dir = SAVE_DIRECTORY, file_prefix = FILE_PREFIX )
