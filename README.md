# Python code for DDH-SMOL
## Description
Zhou et al. (2025) introduced a novel output layer for deep neural networks termed the Smooth Monotonic Output Layer (SMOL).
For illustration, 
SMOL is integrated into Dynamic-DeepHit (DDH; Lee et al., 2020), 
resulting in DDH-SMOL.
This Python code provides an implementation of DDH-SMOL, built largely upon the source code of DDH available at https://github.com/chl8856/Dynamic-DeepHit.

## Instruction

  1. Make input at import_data.py.
  2. Excecute main.py.

## Environment
Python==2.7.15 tensorflow==1.15.0 numpy==1.16.5 pandas==1.0.1 scikit-learn==0.22.1 lifelines==0.24.9 termcolor==1.1.0 scikit-survival==0.12.0

## References
Lee, C., Yoon, J., & Van Der Schaar, M. (2020). 
Dynamic-deephit: A deep learning approach for dynamic survival analysis with competing risks based on longitudinal data. _IEEE Transactions on Biomedical Engineering_, 67, 122-133.
[doi:10.1109/TBME.2019.2909027](https://dx.doi.org/10.1109/TBME.2019.2909027)

Zhou, Z., Deng, Y., Liu, L., Jiang, H., Peng, Y., Yang, X., Zhao, Y., Ning, H., Allen, N., Wilkins, J., Liu, K., Lloyd-Jones, D., and Zhao, L. (2025).
Deep neural network with a smooth monotonic output layer for dynamic risk prediction. 
