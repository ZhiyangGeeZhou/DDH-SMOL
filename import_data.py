# Slightly modified import_data.py at https://github.com/chl8856/Dynamic-DeepHit

import pandas as pd
import numpy as np

from scipy.interpolate       import BSpline
from scipy.integrate         import quad

##### USER-DEFINED FUNCTIONS
def f_get_Normalization(X, norm_mode):    
    num_Patient, num_Feature = np.shape(X)
    
    if norm_mode == 'standard': #zero mean unit variance
        for j in range(num_Feature):
            if np.nanstd(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.nanmean(X[:, j]))/np.nanstd(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.nanmean(X[:, j]))
    elif norm_mode == 'normal': #min-max normalization
        for j in range(num_Feature):
            X[:,j] = (X[:,j] - np.nanmin(X[:,j]))/(np.nanmax(X[:,j]) - np.nanmin(X[:,j]))
    else:
        print("INPUT MODE ERROR!")
    
    return X

def f_get_fc_mask12(time, num_Event, num_Bspline, degree_Bspline, T_max):
    '''
        mask1 is required to get BSpline evaluated at time-to-event
        mask2 is required to get the 1st derivative of BSpline evaluated at time-to-event
    '''
    mask1 = np.zeros((time.shape[0], num_Event, np.max(num_Bspline)), dtype=float)
    mask2 = np.zeros((time.shape[0], num_Event, np.max(num_Bspline)), dtype=float)
    time_norm  = time
    time_norm  = time_norm.astype(float)
    time_norm  = time_norm.flatten()
    for k in range(num_Event):
        # create knots within [0,T_max] for BSplines
        num_int_knots   = int(num_Bspline[k] - degree_Bspline[k] + 1)
        tt_step         = float(T_max/(num_int_knots - 1)) # step length of internal knots
        tt = np.concatenate(( # all knots                                         
          np.linspace(0.-degree_Bspline[k]*tt_step, 0.-tt_step, num=degree_Bspline[k]), 
          np.linspace(0., T_max, num = num_int_knots), 
          np.linspace(T_max+tt_step, T_max+degree_Bspline[k]*tt_step, num=degree_Bspline[k])
        ))
        for l in range(num_Bspline[k]):
            b = BSpline.basis_element(tt[l:(l+degree_Bspline[k]+2)])
            tmp1 = b.__call__(time_norm) * (time_norm >= tt[l]) * (time_norm < tt[l+degree_Bspline[k]+1])
            tmp2 = b.derivative(nu=1).__call__(time_norm) * (time_norm >= tt[l]) * (time_norm < tt[l+degree_Bspline[k]+1])
            mask1[:, k, l] = tmp1
            mask2[:, k, l] = tmp2
    
    return mask1, mask2
    
def f_get_fc_mask3(time_last, num_Event, num_Bspline, degree_Bspline, T_max):
    '''
        mask3 is required to get BSpline evaluated at last measurement time points
    '''
    mask = np.zeros((time_last.shape[0], num_Event, np.max(num_Bspline)))
    time_norm  = time_last
    time_norm  = time_norm.astype(float)
    time_norm  = time_norm.flatten()
    for k in range(num_Event):
        # create knots within [0,T_max] for BSplines
        num_int_knots   = int(num_Bspline[k] - degree_Bspline[k] + 1)
        tt_step         = float(T_max/(num_int_knots - 1)) # step length of internal knots
        tt = np.concatenate(( # all knots                                         
          np.linspace(0.-degree_Bspline[k]*tt_step, 0.-tt_step, num=degree_Bspline[k]), 
          np.linspace(0., T_max, num = num_int_knots), 
          np.linspace(T_max+tt_step, T_max+degree_Bspline[k]*tt_step, num=degree_Bspline[k])
        ))
        for l in range(num_Bspline[k]):
            b = BSpline.basis_element(tt[l:(l+degree_Bspline[k]+2)])
            tmp = b.__call__(time_norm) * (time_norm >= tt[l]) * (time_norm < tt[l+degree_Bspline[k]+1])
            mask[:, k, l] = tmp
            
    return mask
  
def f_get_fc_mask4(num_Event, num_Bspline, degree_Bspline, T_max):
    '''
        mask4 is required to get the smoothness penalty
    '''
    
    def integrand(x, l1, l2, tt, degree_Bspline):
        b1 = BSpline.basis_element(tt[l1:(l1+degree_Bspline+2)])
        b2 = BSpline.basis_element(tt[l2:(l2+degree_Bspline+2)])
        return b1.derivative(nu=2).__call__(x)* b2.derivative(nu=2).__call__(x)
        
    mask = np.zeros((np.max(num_Bspline), np.max(num_Bspline), num_Event), dtype=float)
    for k in range(num_Event):
        # create knots within [0,T_max] for BSplines
        num_int_knots   = int(num_Bspline[k] - degree_Bspline[k] + 1)
        tt_step         = float(T_max/(num_int_knots - 1)) # step length of internal knots
        tt = np.concatenate(( # all knots                                         
          np.linspace(0.-degree_Bspline[k]*tt_step, 0.-tt_step, num=degree_Bspline[k]), 
          np.linspace(0., T_max, num = num_int_knots), 
          np.linspace(T_max+tt_step, T_max+degree_Bspline[k]*tt_step, num=degree_Bspline[k])
        ))
        for l1 in range(num_Bspline[k]):
            for l2 in range(l1, num_Bspline[k]):
                if l2-l1<=degree_Bspline[k]:
                    I = quad(integrand, tt[l2], tt[l1+degree_Bspline[k]+1], args=(l1, l2, tt, degree_Bspline[k]))
                    mask[l1, l2, k] = I[0]
                    mask[l2, l1, k] = I[0]
      
    return mask
    
def f_get_fc_mask5(label):
    '''
        mask5 assign one to the element of the kth column if the kth event is encountered
    '''

    num_Event       = len(np.unique(label)) - 1
    mask            = np.zeros((label.shape[0], num_Event), dtype = float)
    for k in range(num_Event):
        mask[np.where(label == k+1), k] = 1.
      
    return mask 

##### TRANSFORMING DATA
def f_construct_dataset(df, feat_list):
    '''
        id   : patient indicator
        tte  : time-to-event or time-to-censoring
            - must be synchronized based on the reference time
        times: time at which observations are measured
            - must be synchronized based on the reference time (i.e., times start from 0)
        label: event/censoring information
            - 0: censoring
            - 1: event type 1
            ...
    '''

    grouped  = df.groupby(['id'])
    id_list  = pd.unique(df['id'])
    max_meas = np.max(grouped.count())[0]

    data     = np.zeros((len(id_list), max_meas, len(feat_list)+1))
    pat_info = np.zeros((len(id_list), 5))

    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)

        pat_info[i,4] = tmp.shape[0]       #number of measurement
        pat_info[i,3] = np.max(tmp['times'])     #last measurement time
        pat_info[i,2] = tmp['label'][0]      #cause
        pat_info[i,1] = tmp['tte'][0]         #time_to_event
        pat_info[i,0] = tmp['id'][0]      

        data[i, :int(pat_info[i, 4]), 1:]  = tmp[feat_list]
        data[i, :int(pat_info[i, 4]-1), 0] = np.diff(tmp['times'])
    
    return pat_info, data

def import_dataset_ASCVD(study, num_Bspline, degree_Bspline, norm_mode = 'standard'):
    if isinstance(study, list):
        df_        = pd.read_csv("PATH TO YOUR CSV DATA FILE")           
    else:
        print ('ERROR: ILLEGAL SETUP OF STUDY !!!')
        
    bin_list           = [] # names of binary covariates
    cont_list          = []  # names of continuous covariates
    feat_list          = cont_list + bin_list
    df_                = df_[['id', 'tte', 'times', 'label']+feat_list]
    df_org_            = df_.copy(deep=True)

    df_[cont_list]     = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)

    pat_info, data     = f_construct_dataset(df_, feat_list)
    _, data_org        = f_construct_dataset(df_org_, feat_list)

    data_mi                  = np.zeros(np.shape(data))
    data_mi[np.isnan(data)]  = 1
    data_org[np.isnan(data)] = 0
    data[np.isnan(data)]     = 0 

    x_dim           = np.shape(data)[2] # 1 + x_dim_cont + x_dim_bin (including delta)
    x_dim_cont      = len(cont_list)
    x_dim_bin       = len(bin_list) 

    time_last       = pat_info[:,[3]]  #the last measurement time
    label           = pat_info[:,[2]]  #competing risks
    time            = pat_info[:,[1]]  #age when event occurred
    T_max           = float(round(np.max(time) * 1.2)) # right boundry of time interval
    
    num_Event       = len(np.unique(label)) - 1
    
    for k, tmp_label in enumerate(np.unique(label)):
        label[np.where(label == tmp_label)] = k # relabel risks by 0,...,num_Event

    mask1, mask2    = f_get_fc_mask12(time, num_Event, num_Bspline, degree_Bspline, T_max)
    mask3           = f_get_fc_mask3(time_last, num_Event, num_Bspline, degree_Bspline, T_max)
    mask4           = f_get_fc_mask4(num_Event, num_Bspline, degree_Bspline, T_max)
    mask5           = f_get_fc_mask5(label)

    DIM             = (x_dim, x_dim_cont, x_dim_bin)
    DATA            = (data, label, time, time_last)
    MASK            = (mask1, mask2, mask3, mask4, mask5)

    return DIM, DATA, MASK, data_mi
    
def import_dataset_simu(study, scenario, seed, num_Bspline, degree_Bspline, norm_mode = 'standard'):
    if isinstance(study, list):
        df_        = pd.read_csv("PATH TO YOUR CSV DATA FILE")
    else:
        print ('ERROR: ILLEGAL SETUP OF STUDY !!!')
        
    bin_list           = [] # names of binary covariates
    cont_list          = [] # names of continuous covariates
    feat_list          = cont_list + bin_list
    df_                = df_[['id', 'tte', 'times', 'label']+feat_list]
    df_org_            = df_.copy(deep=True)

    df_[cont_list]     = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)

    pat_info, data     = f_construct_dataset(df_, feat_list)
    _, data_org        = f_construct_dataset(df_org_, feat_list)

    data_mi                  = np.zeros(np.shape(data))
    data_mi[np.isnan(data)]  = 1
    data_org[np.isnan(data)] = 0
    data[np.isnan(data)]     = 0 

    x_dim           = np.shape(data)[2] # 1 + x_dim_cont + x_dim_bin (including delta)
    x_dim_cont      = len(cont_list)
    x_dim_bin       = len(bin_list) 

    time_last       = pat_info[:,[3]]  #the last measurement time
    label           = pat_info[:,[2]]  #competing risks
    time            = pat_info[:,[1]]  #age when event occurred
    T_max           = float(round(np.max(time) * 1.2)) # right boundry of time interval
    
    num_Event       = len(np.unique(label)) - 1
    
    for k, tmp_label in enumerate(np.unique(label)):
        label[np.where(label == tmp_label)] = k # relabel risks by 0,...,num_Event

    mask1, mask2    = f_get_fc_mask12(time, num_Event, num_Bspline, degree_Bspline, T_max)
    mask3           = f_get_fc_mask3(time_last, num_Event, num_Bspline, degree_Bspline, T_max)
    mask4           = f_get_fc_mask4(num_Event, num_Bspline, degree_Bspline, T_max)
    mask5           = f_get_fc_mask5(label)

    DIM             = (x_dim, x_dim_cont, x_dim_bin)
    DATA            = (data, label, time, time_last)
    MASK            = (mask1, mask2, mask3, mask4, mask5)

    return DIM, DATA, MASK, data_mi