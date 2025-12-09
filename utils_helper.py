# Slightly modified utils_helper.py at https://github.com/chl8856/Dynamic-DeepHit

import import_data as impt
import numpy       as np
import pandas      as pd
import random

from scipy.interpolate       import BSpline
from scipy.integrate         import quad

##### USER-DEFINED FUNCTIONS

def f_get_minibatch(mb_size, x, x_mi, label, time, time_last, num_Bspline, degree_Bspline, T_max, mask1, mask2, mask3, mask5):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb     = x[idx, :, :].astype(float)
    x_mi_mb  = x_mi[idx, :, :].astype(float)
    k_mb     = label[idx, :].astype(float) # censoring(0)/event(1,2,..) label
    t_mb     = time[idx, :].astype(float)
    l_mb     = time_last[idx, :].astype(float)
    m1_mb    = mask1[idx, :, :].astype(float) #fc_mask1
    m2_mb    = mask2[idx, :, :].astype(float) #fc_mask2
    m3_mb    = mask3[idx, :, :].astype(float) #fc_mask3
    m5_mb    = mask5[idx, :].astype(float) #fc_mask5
    
    return x_mb, x_mi_mb, k_mb, t_mb, l_mb, m1_mb, m2_mb, m3_mb, m5_mb