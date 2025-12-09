# Python code for DDH-SMOL
## Description
Zhou et al. (2025) introduced a novel output layer for deep neural networks termed the Smooth Monotonic Output Layer (SMOL).
For illustration, 
SMOL is integrated into Dynamic-DeepHit (DDH; Lee et al., 2020), 
resulting in DDH-SMOL.
This Python code provides an implementation of DDH-SMOL, built largely upon the original DDH source code available at https://github.com/chl8856/Dynamic-DeepHit.

## Environment
Python==2.7.15 tensorflow==1.15.0 numpy==1.16.5 pandas==1.0.1 scikit-learn==0.22.1 lifelines==0.24.9 termcolor==1.1.0 scikit-survival==0.12.0

## References
C. Lee, J. Yoon and M. van der Schaar (2020). Dynamic-DeepHit: A deep learning approach for dynamic survival analysis with competing risks based on longitudinal data. 
_IEEE Transactions on Biomedical Engineering_, **67**, 122--133. 
doi: 10.1109/TBME.2019.2909027. 
