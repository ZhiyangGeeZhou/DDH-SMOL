# Simplified version of utils_eval.py at https://github.com/chl8856/Dynamic-DeepHit

import numpy as np
import pandas as pd

from lifelines import AalenJohansenFitter, KaplanMeierFitter

### C(t)-INDEX CALCULATION
def c_index(Prediction, Time_survival, Death, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
  
        if (Time_survival[i]<=Time and Death[i]==1):
            N_t[i,:] = 1

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result

### BRIER-SCORE
def brier_score(Prediction, Time_survival, Death, Time):
    N = len(Prediction)
    y_true = ((Time_survival <= Time) * Death).astype(float)
    
    N = len(Prediction)
    ind1 = ((Time_survival < Time).reshape([N,]) * Death == 1)
    ind2 = (Time_survival < Time).reshape([N,])
    ind3 = ((Time_survival < Time).reshape([N,]) * Death == 0)

    events = np.nansum((Prediction - ind1)**2)
    no_events = np.nansum(Prediction[ind2]**2)
    censored = 0
    if any(ind3):
        weights = Prediction[ind3]
        censored = np.nansum(weights * ((1 - Prediction[ind3])**2) + (1 - weights) * (Prediction[ind3]**2))

    return (events + no_events + censored)/N