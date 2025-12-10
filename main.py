# Modified from https://github.com/chl8856/Dynamic-DeepHit

_EPSILON = 1e-08

import import_data as impt
import lifelines
import math
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf

from class_DeepLongitudinal  import Model_Longitudinal_Attention
from scipy.interpolate       import BSpline
from sklearn.model_selection import train_test_split
from utils_eval              import c_index, brier_score, brier_score_new
from utils_helper            import f_get_minibatch
from utils_log               import save_logging, load_logging


def _f_get_pred(sess, model, data, data_mi, pred_horizon):
    '''
        predictions based on the prediction time.
        create new_data and new_mask2 that are available previous or equal to the prediction time (no future measurements are used)
    '''
    new_data    = np.zeros(np.shape(data))
    new_data_mi = np.zeros(np.shape(data_mi))
    time_last   = np.zeros([np.shape(data)[0], 1])

    meas_time = np.concatenate([np.zeros([np.shape(data)[0], 1]), np.cumsum(data[:, :, 0], axis=1)[:, :-1]], axis=1)
    
    for i in range(np.shape(data)[0]):
        last_meas       = np.sum(meas_time[i, :] <= pred_horizon)
        time_last[i, 0] = last_meas

        new_data[i, :last_meas, :]    = data[i, :last_meas, :]
        new_data_mi[i, :last_meas, :] = data_mi[i, :last_meas, :]

    return model.predict(new_data, new_data_mi), time_last


def f_get_risk_predictions(sess, model, data_, data_mi_, pred_time, eval_time, network_settings, input_dims):
    
    num_Event       = input_dims['num_Event']
    
    # create knots for BSplines
    T_max           = input_dims['T_max']
    num_Bspline     = network_settings['num_Bspline']
    degree_Bspline  = network_settings['degree_Bspline']
          
    risk_all = {}
    N = np.shape(data_)[0]
    for k in range(num_Event):
        risk_all[k] = np.zeros([N, len(pred_time), len(eval_time)])
    for p, p_time in enumerate(pred_time):
        ### PREDICTION
        pred_horizon    = p_time
        pred, time_last = _f_get_pred(sess, model, data_, data_mi_, pred_horizon)
        for t, t_time in enumerate(eval_time):
            eval_horizon = t_time + pred_horizon

            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            mask3_1 = impt.f_get_fc_mask3(
                np.array([pred_horizon]*N).reshape([N,1]), num_Event, num_Bspline, degree_Bspline, T_max)
            mask3_2 = impt.f_get_fc_mask3(
                np.array([eval_horizon]*N).reshape([N,1]), num_Event, num_Bspline, degree_Bspline, T_max)
        
            part1 = np.exp(-np.sum(mask3_1 * pred, axis=2))
            part2 = np.exp(-np.sum(mask3_2 * pred, axis=2))
            risk  = (part1 - part2)/np.sum(part1, axis=1, keepdims=True)
            
            for k in range(num_Event):
                risk_all[k][:, p, t] = risk[:, k]
                     
    return risk_all


# ### 1. Set Hyper-Parameters
# ##### - Play with your own hyper-parameters!

data_mode                   = 'simu'
study                       = ['black', 'male', 'year']
burn_in_mode                = 'ON' #{'ON', 'OFF'}
pars_combn_num              = 12 # number of hyper-parametric combinations
seed                        = 1
gpu                         = str(seed % 3) # originally '0'

random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

if data_mode == 'CVD':
    num_Event = 2
if data_mode == 'ASCVD':
    num_Event = 1
if data_mode == 'simu':  
    num_Event = 1
    scenario = 1

##### HYPER-PARAMETERS
new_parser = {'mb_size'          : [32, 64, 128],
              
              'iteration_burn_in': [3000],
              'iteration'        : [5000],

              'keep_prob'        : [.4],
              'lr_train'         : [1e-4],
              
              'h_dim_RNN'        : [50, 100, 200, 300],
              'h_dim_FC'         : [50, 100, 200, 300],
              'num_layers_RNN'   : [2, 3, 5], 
              'num_layers_ATT'   : [2],
              'num_layers_CS'    : [2, 3, 5],

              'RNN_type'         : ['GRU', 'LSTM'],

              'FC_active_fn'     : [tf.nn.tanh],
              'RNN_active_fn'    : [tf.nn.relu],
              'SOL_active_fn'    : ['relu'],

              'reg_W'            : 1e-5,
              'reg_W_out'        : 0.,

              'lambda1'          : [round(pow(10, random.uniform(-1,1)),2) for _ in range(pars_combn_num)],
              'lambda2'          : [round(pow(10, random.uniform(-3,1)),5) for _ in range(pars_combn_num)],
              'lambda3'          : [round(pow(10, random.uniform(0,4)),2) for _ in range(pars_combn_num)]
}

for k in range(num_Event):
    new_parser["_".join(['num_Bspline', str(k+1)])] = [10, 30, 50, 100, 300, 500]
    new_parser["_".join(['degree_Bspline', str(k+1)])] = [3, 5, 7, 9] 

file_path = '{}'.format("".join(['output/',data_mode]))
if data_mode == 'simu':
    file_path = '{}'.format("".join(['output/',data_mode,'_scenario',str(scenario)]))

if not os.path.exists(file_path):
    os.makedirs(file_path)

# SAVE HYPERPARAMETERS
log_name = file_path + '/hyperparameters_log'+'_'+study[0]+'_'+study[1]+'_'+study[2]+'_seed'+str(seed)+'.txt'
save_logging(new_parser, log_name)

# ### 2. Construct Combinations of Hyperparameters
pars_tune_Bspline = ['num_Bspline', 'degree_Bspline']
pars_tune = ['h_dim_RNN',
             'h_dim_FC',
             'num_layers_RNN',
             'num_layers_ATT',
             'num_layers_CS',
             'mb_size', 
             'iteration', 
             'iteration_burn_in', 
             'RNN_type',
             'FC_active_fn',
             'RNN_active_fn',
             'SOL_active_fn',
             'keep_prob', 
             'lr_train', 
             'lambda1',  
             'lambda2',
             'lambda3'] + ["_".join([pars_tune_Bspline[i],str(j+1)]) for i in range(len(pars_tune_Bspline)) for j in range(num_Event)]

pars_combn = pd.DataFrame(
    data = np.zeros(shape=(pars_combn_num, len(pars_tune))),
    columns = pars_tune
)

for pars_curr in pars_combn.columns:
    for i in range(pars_combn_num//len(new_parser[pars_curr])):
        pars_combn[pars_curr][(i*len(new_parser[pars_curr])):((i+1)*len(new_parser[pars_curr]))] = np.random.choice(new_parser[pars_curr], len(new_parser[pars_curr]), replace = False)
    if pars_combn_num%len(new_parser[pars_curr]) != 0:
        pars_combn[pars_curr][-(pars_combn_num%len(new_parser[pars_curr])):] = np.random.choice(new_parser[pars_curr], pars_combn_num%len(new_parser[pars_curr]), replace = False)

# ### 3. Traning, Tuning & Testing

min_valid = 0. # initialize min_valid which instores the lowest average C index value
for pars_combn_idx in range(pars_combn_num):
    
    renew_valid = 0 # indicate whehther the lowest average C index value has been renewed for the current pars_combn_idx
        
    # NETWORK HYPER-PARMETERS
    network_settings = {'h_dim_RNN'         : int(pars_combn['h_dim_RNN'][pars_combn_idx]),
                        'h_dim_FC'          : int(pars_combn['h_dim_FC'][pars_combn_idx]),
                        'num_layers_RNN'    : int(pars_combn['num_layers_RNN'][pars_combn_idx]),
                        'num_layers_ATT'    : int(pars_combn['num_layers_ATT'][pars_combn_idx]),
                        'num_layers_CS'     : int(pars_combn['num_layers_CS'][pars_combn_idx]),
                        
                        'num_Bspline'       : list(map(int, pars_combn[["_".join(['num_Bspline',str(k+1)]) for k in range(num_Event)]].iloc[pars_combn_idx,:].tolist())),
                        'degree_Bspline'    : list(map(int, pars_combn[["_".join(['degree_Bspline',str(k+1)]) for k in range(num_Event)]].iloc[pars_combn_idx,:].tolist())),
                                                  
                        'RNN_type'          : pars_combn['RNN_type'][pars_combn_idx],
                                                  
                        'FC_active_fn'      : pars_combn['FC_active_fn'][pars_combn_idx],
                        'RNN_active_fn'     : pars_combn['RNN_active_fn'][pars_combn_idx],
                        'SOL_active_fn'     : pars_combn['SOL_active_fn'][pars_combn_idx],
                        
                        'initial_W'         : tf.contrib.layers.xavier_initializer(),

                        'reg_W'             : float(new_parser['reg_W']),
                        'reg_W_out'         : float(new_parser['reg_W_out'])
                     }

    mb_size           = int(pars_combn['mb_size'][pars_combn_idx])
    iteration         = int(pars_combn['iteration'][pars_combn_idx])
    iteration_burn_in = int(pars_combn['iteration_burn_in'][pars_combn_idx])

    keep_prob         = float(pars_combn['keep_prob'][pars_combn_idx])
    lr_train          = float(pars_combn['lr_train'][pars_combn_idx])

    lambda1           = float(pars_combn['lambda1'][pars_combn_idx])
    lambda2           = float(pars_combn['lambda2'][pars_combn_idx])
    lambda3           = float(pars_combn['lambda3'][pars_combn_idx])
    
    ##### IMPORT DATASET
    ## Users must prepare dataset in csv format and modify 'import_data.py' following our examplar 'PBC2'
    '''
        T_max                   = max event/censoring time * 1.2
        num_Event               = number of evetns i.e., len(np.unique(label))-1
        max_length              = maximum number of measurements
        x_dim                   = data dimension including delta (1 + num_features)
        x_dim_cont              = dim of continuous features
        x_dim_bin               = dim of binary features
        mask1,...,5             = used for loss computation
    '''

    if data_mode == 'ASCVD':
        (x_dim, x_dim_cont, x_dim_bin), (data, label, time, time_last), (mask1, mask2, mask3, mask4, mask5), (data_mi) = impt.import_dataset_ASCVD(
            study, network_settings['num_Bspline'], network_settings['degree_Bspline'], 'standard')

        # This must be changed depending on the datasets, prediction/evaliation times of interest
        if study[2] == 'year':
            multi = 1
        elif study[2] == 'month':
            multi = 12
        elif study[2] == 'week':
            multi = 52
        elif study[2] == 'day':
            multi = 365
        pred_time = [5*multi, 10*multi] # prediction time (in year/month/week/day)
        eval_time = [5*multi, 10*multi] # evaluation time (in year/month/week/day; for C-index and Brier-Score)
    elif data_mode == 'CVD':
        (x_dim, x_dim_cont, x_dim_bin), (data, label, time, time_last), (mask1, mask2, mask3, mask4, mask5), (data_mi) = impt.import_dataset_CVD(
            study, network_settings['num_Bspline'], network_settings['degree_Bspline'], 'standard')

        # This must be changed depending on the datasets, prediction/evaliation times of interest
        if study[2] == 'year':
            multi = 1
        elif study[2] == 'month':
            multi = 12
        elif study[2] == 'week':
            multi = 52
        elif study[2] == 'day':
            multi = 365
        pred_time = [5*multi, 10*multi] # prediction time (in year/month/week/day)
        eval_time = [5*multi, 10*multi] # evaluation time (in year/month/week/day; for C-index and Brier-Score)
    elif data_mode == 'simu':
        (x_dim, x_dim_cont, x_dim_bin), (data, label, time, time_last), (mask1, mask2, mask3, mask4, mask5), (data_mi) = impt.import_dataset_simu(
            study, scenario, seed, network_settings['num_Bspline'], network_settings['degree_Bspline'], 'standard')
        
        # This must be changed depending on the datasets, prediction/evaliation times of interest
        if study[2] == 'year':
            multi = 1
        elif study[2] == 'month':
            multi = 12
        elif study[2] == 'week':
            multi = 52
        elif study[2] == 'day':
            multi = 365
        pred_time = [5*multi, 10*multi] # prediction time (in year/month/week/day)
        eval_time = [5*multi, 10*multi] # evaluation time (in year/month/week/day; for C-index and Brier-Score)
    else:
        print ('ERROR:  DATA_MODE NOT FOUND !!!')

    num_Event                   = len(np.unique(label))-1
    T_max                       = float(round(np.max(time) * 1.2))
    max_length                  = np.shape(data)[1]

    # INPUT DIMENSIONS
    input_dims                  = { 'x_dim'         : x_dim,
                                    'x_dim_cont'    : x_dim_cont,
                                    'x_dim_bin'     : x_dim_bin,
                                    'num_Event'     : num_Event,
                                    'T_max'         : T_max,
                                    'max_length'    : max_length
                                  }
    
    ##### TRAINING-VALIDATION-TESTING SPLIT
    (tr_data, te_data, tr_data_mi, te_data_mi, 
     tr_label, te_label, tr_time, te_time, tr_time_last, te_time_last,  
     tr_mask1, te_mask1, tr_mask2, te_mask2, tr_mask3, te_mask3, tr_mask5, te_mask5) = train_test_split(
        data, data_mi, label, time, time_last, mask1, mask2, mask3, mask5,
        test_size=0.2, random_state=seed) 

    (tr_data, va_data, tr_data_mi, va_data_mi, 
     tr_label, va_label, tr_time, va_time, tr_time_last, va_time_last,
     tr_mask1, va_mask1, tr_mask2, va_mask2, tr_mask3, va_mask3, tr_mask5, va_mask5) = train_test_split(
        tr_data, tr_data_mi, tr_label, tr_time, tr_time_last, tr_mask1, tr_mask2, tr_mask3, tr_mask5,
        test_size=0.2, random_state=seed) 

    ##### CREATE DYNAMIC-DEEPFHT NETWORK
    tf.reset_default_graph()

    #### SPECIFY GPU NUMBER
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config = tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model_Longitudinal_Attention(sess, "Dyanmic-DeepHit", input_dims, network_settings)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
 
    ### TRAINING - BURN-IN
    if burn_in_mode == 'ON':
        print(pars_combn.iloc[pars_combn_idx])
        print("BURN-IN TRAINING for Hyper-Parametric Combination", pars_combn_idx)

        for itr in range(iteration_burn_in):
            x_mb, x_mi_mb, k_mb, t_mb, l_mb, m1_mb, m2_mb, m3_mb, m5_mb = f_get_minibatch(
                mb_size, tr_data, tr_data_mi, tr_label, tr_time, tr_time_last, 
                network_settings['num_Bspline'], network_settings['degree_Bspline'], input_dims['T_max'],
                tr_mask1, tr_mask2, tr_mask3, tr_mask5)
            DATA = (x_mb, k_mb, t_mb, l_mb)
            MISSING = (x_mi_mb)

            _, loss_curr = model.train_burn_in(DATA, MISSING, keep_prob, lr_train)
            loss_curr = float(loss_curr)

            if (itr+1)%1000 == 0:
                print('itr: {:04d} | loss3: {:.4f}'.format(itr+1, loss_curr))


    ### TRAINING - MAIN
    if burn_in_mode == 'OFF':
        print(pars_combn.iloc[pars_combn_idx])
    
    print("MAIN TRAINING for Hyper-Parametric Combination", pars_combn_idx)       

    for itr in range(iteration):
        x_mb, x_mi_mb, k_mb, t_mb, l_mb, m1_mb, m2_mb, m3_mb, m5_mb = f_get_minibatch(
            mb_size, tr_data, tr_data_mi, tr_label, tr_time, tr_time_last, 
            network_settings['num_Bspline'], network_settings['degree_Bspline'], input_dims['T_max'],
            tr_mask1, tr_mask2, tr_mask3, tr_mask5)
        DATA = (x_mb, k_mb, t_mb, l_mb)
        MASK = (m1_mb, m2_mb, m3_mb, mask4, m5_mb)
        MISSING = (x_mi_mb)
        PARAMETERS = (lambda1, lambda2, lambda3)

        _, loss_curr, loss1_curr, loss2_curr, loss3_curr, loss4_curr = model.train(
            DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train)
        loss_curr  = float(loss_curr)
        loss1_curr = float(loss1_curr)
        loss2_curr = float(loss2_curr)
        loss3_curr = float(loss3_curr)
        loss4_curr = float(loss4_curr)
        
        if (itr+1)%1000 == 0:
            print('itr: {:04d} | loss: {:.4f} | Lkd: {:.4f} | Rkg: {:.4f} | Pred: {:.4f} | Smooth: {:.4f}'.format(
                itr+1, loss_curr, loss1_curr, loss2_curr, loss3_curr, loss4_curr))

        ### VALIDATION  (based on average C-index of our interest)
        if (itr+1)%1000 == 0:
            risk_all = f_get_risk_predictions(
                sess, model, va_data, va_data_mi, pred_time, eval_time, network_settings, input_dims)

            for p, p_time in enumerate(pred_time):
                pred_horizon = int(p_time)
                val_result1 = np.zeros([num_Event, len(eval_time)])

                for t, t_time in enumerate(eval_time):                
                    eval_horizon = int(t_time) + pred_horizon
                    for k in range(num_Event):
                        val_result1[k, t] = c_index(
                            risk_all[k][:, p, t], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon
                        ) #-1 for no event (not comparable)

                if p == 0:
                    val_final1 = val_result1
                else:
                    val_final1 = np.append(val_final1, val_result1, axis=0)

            tmp_valid = np.mean(val_final1)
                        
            if min_valid < tmp_valid:
                renew_valid        = 1
                min_valid          = tmp_valid
                pars_combn_idx_opt = pars_combn_idx
                val_final1_opt     = val_final1
                saver.save(sess, file_path + '/model'+'_'+study[0]+'_'+study[1]+'_'+study[2]+'_seed'+str(seed))
                print( 'updated.... average c-index = ' + str('%.4f' %(tmp_valid)))
    
    ### Testing
    if renew_valid == 1:
        saver.restore(sess, file_path + '/model'+'_'+study[0]+'_'+study[1]+'_'+study[2]+'_seed'+str(seed))
        risk_all = f_get_risk_predictions(
            sess, model, te_data, te_data_mi, pred_time, eval_time, network_settings, input_dims)
        for p, p_time in enumerate(pred_time):
            pred_horizon = int(p_time)
            result1, result2 = np.zeros([num_Event, len(eval_time)]), np.zeros([num_Event, len(eval_time)])
            for t, t_time in enumerate(eval_time):                
                eval_horizon = int(t_time) + pred_horizon
                for k in range(num_Event):
                    result1[k, t] = c_index(risk_all[k][:, p, t], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                    result2[k, t] = brier_score(risk_all[k][:, p, t], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
    
            if p == 0:
                final1, final2 = result1, result2
            else:
                final1, final2 = np.append(final1, result1, axis=0), np.append(final2, result2, axis=0)


# ### 4. Printing

# In[ ]:


col_header1 = ['Seed']
for p_time in pred_time:
    for e_time in eval_time:
        for k in range(num_Event):
            col_header1.append(''.join(['event',str(k+1),'_',str(p_time),'->', str(p_time+e_time)]))
            
col_header2 = ['Seed']
for p_time in pred_time:
    for e_time in eval_time:
        for k in range(num_Event):
            col_header2.append(''.join(['event',str(k+1),'_',str(p_time),'->', str(p_time+e_time)]))
        
# C-index result
df1 = pd.DataFrame(np.append(int(seed), final1.flatten()).reshape([1,len(col_header1)]), columns = col_header1)
df1_val = pd.DataFrame(np.append(int(seed), val_final1_opt.flatten()).reshape([1,len(col_header1)]), columns = col_header1)

# Brier-score result
df2 = pd.DataFrame(np.append(int(seed), final2.flatten()).reshape([1,len(col_header2)]), columns = col_header2)

### PRINTING
print(pars_combn.iloc[pars_combn_idx_opt]) # CURRENT OPTIMAL HYPERPARAMETERS
print('========================================================')
print('--------------------------------------------------------')
print('- C-INDEX-VALIDATION: ')
print(df1_val.to_string(index=False))
print('--------------------------------------------------------')
print('- C-INDEX-TESTING: ')
print(df1.to_string(index=False))
print('--------------------------------------------------------')
print('- BRIER-SCORE: ')
print(df2.to_string(index=False))
print('========================================================')

### CSV OUTPUT
if not os.path.exists(file_path+'/'+'cindex_'+study[0]+'_'+study[1]+'_'+study[2]+'_DDHSMOL.csv'):
    df1 = pd.concat([
        df1.reset_index(drop=True), 
        pars_combn.iloc[[pars_combn_idx_opt]].reset_index(drop=True)
    ], axis=1)
else:
    df1 = pd.concat([
        pd.read_csv(file_path+'/'+'cindex_'+study[0]+'_'+study[1]+'_'+study[2]+'_DDHSMOL.csv').reset_index(drop=True), 
        pd.concat([
            df1.reset_index(drop=True), 
            pars_combn.iloc[[pars_combn_idx_opt]].reset_index(drop=True)
        ], axis=1).reset_index(drop=True)
    ])
df1.to_csv(file_path+'/'+'cindex_'+study[0]+'_'+study[1]+'_'+study[2]+'_DDHSMOL.csv', sep=',', index=False)
    
if not os.path.exists(file_path+'/'+'brier_'+study[0]+'_'+study[1]+'_'+study[2]+'_DDHSMOL.csv'):
    df2 = pd.concat([
        df2.reset_index(drop=True), 
        pars_combn.iloc[[pars_combn_idx_opt]].reset_index(drop=True)
    ], axis=1)
else:
    df2 = pd.concat([
        pd.read_csv(file_path+'/'+'brier_'+study[0]+'_'+study[1]+'_'+study[2]+'_DDHSMOL.csv').reset_index(drop=True), 
        pd.concat([
            df2.reset_index(drop=True), 
            pars_combn.iloc[[pars_combn_idx_opt]].reset_index(drop=True)
        ], axis=1).reset_index(drop=True)
    ])
df2.to_csv(file_path+'/'+'brier_'+study[0]+'_'+study[1]+'_'+study[2]+'_DDHSMOL.csv', sep=',', index=False)

