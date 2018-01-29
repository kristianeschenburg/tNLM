# tNLM

Code to apply temporal non-local means filtering to fMRI as described in Bushan et al, 2016.

**Note #1:**  tNLM method is currently is not parallelized, and will loop over every surface vertex.  I'm working on incorporating Python's **multiprocessing** library to facillitate parallelized vertex filtering.

**Note #2:**  Manjón et al. described a method to adapt the smoothing strength based on the local signal noise -- this adaptive smoothing has not yet been applied to fMRI.  Will be updated soon.

Non-local means applied to MRI data in several papers:

Coupé, Pierrick et al. “An Optimized Blockwise Nonlocal Means Denoising Filter for 3-D Magnetic Resonance Images.” Ieee Transactions on Medical Imaging 27.4 (2008): 425–441. PMC. Web. 26 Jan. 2018.

Manjón, J. V., Coupé, P., Martí-Bonmatí, L., Collins, D. L. and Robles, M. (2010), Adaptive non-local means denoising of MR images with spatially varying noise levels. J. Magn. Reson. Imaging, 31: 192–203.

Bhushan C, Chong M, Choi S, Joshi AA, Haldar JP, et al. (2016) Temporal Non-Local Means Filtering Reveals Real-Time Whole-Brain Cortical Interactions in Resting fMRI. PLOS ONE 11(7): e0158504.
