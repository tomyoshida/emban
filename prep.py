import casatools as ctool
import numpy as np
import matplotlib.pyplot as plt

import astropy.constants as cst
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.io import fits

import dsharp_opac as opacity
import pickle

import os


import casatasks as ct
c = cst.c.cgs.value


import glob
from scipy.stats import binned_statistic_2d

# 存在するspwを全部書き出す



def splitallspws(vis, dryrun=True):
    
    if dryrun:
        
        tb = ctool.table()

        tb.open(vis + '/DATA_DESCRIPTION')

        spw_id = tb.getcol('SPECTRAL_WINDOW_ID')
        tb.close()

        print(spw_id)
        
        visname = os.path.splitext(os.path.basename(vis))[0]
        
        outputvis_arr = []
        
        for spw in spw_id:

            outputvis = f'./working_{visname}/spw_id_{spw}.ms'
            outputvis_arr.append(outputvis)
        
        return outputvis_arr
        
    else:
    
        tb = ctool.table()

        tb.open(vis + '/DATA_DESCRIPTION')

        spw_id = tb.getcol('SPECTRAL_WINDOW_ID')
        tb.close()

        print(spw_id)

        visname = os.path.splitext(os.path.basename(vis))[0]

        os.system('rm -rf ' + f'./working_{visname}')
        os.system(f'mkdir ./working_{visname}')

        outputvis_arr = []

        for spw in spw_id:

            outputvis = f'./working_{visname}/spw_id_{spw}.ms'

            print(outputvis)

            os.system('rm -rf ' + outputvis)

            ct.split(
                vis = vis,
                outputvis = outputvis,
                spw = f'{spw}',
                keepflags = False,
                datacolumn = 'DATA'
            )

            outputvis_arr.append(outputvis)

    
        return outputvis_arr
    
    
def load_visibility( vis ):
    
    tb = ctool.table()
    tb.open(vis)
    u,v,w = tb.getcol('UVW')
    data = tb.getcol('DATA')
    _flag = tb.getcol('FLAG')
    
    tb.close()
    
    XX_flag = _flag[0]
    YY_flag = _flag[1]

    flag = XX_flag | YY_flag
    
    
    tb.open(vis + '/SPECTRAL_WINDOW')
    chan_freq = tb.getcol('CHAN_FREQ')
    tb.close()
    
    Re = 0.5*(data[0] + data[1]).real
    
    return u, v, Re, chan_freq, flag

    

def deprojected_visibility( vis, pa, incl ):

    u, v, Re, chan_freq, flag = load_visibility( vis )

    pa = np.deg2rad(pa)
    incl = np.deg2rad(incl)

    u_rot = u * np.cos(pa) - v * np.sin(pa)
    v_rot = u * np.sin(pa) + v * np.cos(pa)

    v_deproj = v_rot 
    u_deproj = u_rot * np.cos(incl)

    q = np.sqrt( v_deproj**2 + u_deproj**2 )*1e2
    
    lam = c/chan_freq
    
    q_lam = q/lam
    
    
    return q_lam, Re, chan_freq, flag



def multiple_visibilities( vis_arr, pa, incl ):
    
    
    q_all = np.array([])
    Re_all = np.array([])
    freq_all = np.array([])
    
    for vis in vis_arr:
        
        print('loading: ', vis)
        
        q_lam, Re, chan_freq, flag = deprojected_visibility( vis, pa, incl )
        
        
        Re_m = np.ma.array(Re, mask=flag)
        q_lam_m = np.ma.array(q_lam, mask=flag)


        q_flat = q_lam_m.compressed()
        Re_flat = Re_m.compressed()

        freq_flat = np.broadcast_to(
            chan_freq, Re.shape
        )[~flag]


        q_all = np.append( q_all, q_flat )
        Re_all = np.append( Re_all, Re_flat )
        freq_all = np.append( freq_all, freq_flat )
        
    
    return q_all, Re_all, freq_all




def binning_visibility( q_all, Re_all, freq_all, FoV, dnu ):
    

    dq = 1/np.deg2rad( FoV / 3600 )

    q_min = q_all.min()
    q_max = q_all.max()

    q_bins = np.arange(q_min, q_max + dq, dq)


    df = dnu

    f_min = freq_all.min()
    f_max = freq_all.max()

    f_bins = np.arange(f_min, f_max + df, df)
    
    

    Re_bin, q_edge, f_edge, binnumber = binned_statistic_2d(
        q_all,
        freq_all,
        Re_all,
        statistic="mean",
        bins=[q_bins, f_bins],
    )

    Re2_mean, _, _, _ = binned_statistic_2d(
        q_all,
        freq_all,
        Re_all**2,
        statistic="mean",
        bins=[q_bins, f_bins],
    )


    Re_count, _, _, _ = binned_statistic_2d(
        q_all,
        freq_all,
        Re_all,
        statistic="count",
        bins=[q_bins, f_bins],
    )



    Re_sigma = np.sqrt(Re2_mean - Re_bin**2) / np.sqrt( Re_count )


    q_cent = 0.5 * (q_edge[:-1] + q_edge[1:])
    f_cent = 0.5 * (f_edge[:-1] + f_edge[1:])


    Qc, Fc = np.meshgrid(q_cent, f_cent, indexing="ij")


    
    Re_dict = {}
    s_dict = {}
    nu_dict = {}
    q_dict = {}


    inu_valid = 0

    for inu in range(len(f_cent)):

        mask1 = ~np.isnan(Re_bin[:, inu])
        mask2 = ~np.isnan(Re_sigma[:, inu])
        mask3 = Re_count[:, inu] > 2

        mask = mask1 * mask2 * mask3
        
        if mask.any():
            Re_dict[inu_valid] = Re_bin[mask, inu]
            s_dict[inu_valid] = Re_sigma[mask, inu]

            nu_dict[inu_valid] = Fc[mask, inu][0]
            q_dict[inu_valid] = Qc[mask,inu]

            inu_valid += 1
            
    return Re_dict, s_dict, nu_dict, q_dict, inu_valid


def process_vis( vis_arr, pa, incl, FoV, dnu ):
    
    _q, _Re, _nu = multiple_visibilities( vis_arr, pa, incl )
       
    Re, s, nu, q, Nnu = binning_visibility( _q, _Re/np.cos(np.deg2rad(incl)), _nu, FoV, dnu )
    
    return Re, s, nu, q, Nnu


def pickle_vis( vis_arr, pa, incl, FoV, dnu, filename ):
    '''
    Process and pickle the visibility data.
    vis_arr: list of visibility measurement set paths 
    pa: position angle in degrees
    incl: inclination angle in degrees
    FoV: field of view in arcseconds
    dnu: frequency bin size in Hz
    filename: output pickle file name (without .pkl extension)
    '''
    
    Re, s, nu, q, Nnu = process_vis( vis_arr, pa, incl, FoV, dnu )
    
    data = { 'q':q, 'V':Re, 's':s, 'nu':nu, 'Nch':Nnu }
    
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(data, f)

    print(f"Visibility data saved to {filename}.pkl")