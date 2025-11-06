# Matrix Analysis Tools
Basic tool kit for analysing root files containing numerically labelled sequences of matrices.

# Installing
Package can be install with `pip install .` which will install the package and all dependencies

# Usage
The package contains a single macro that can be run with
```sh
analyse_matrix  -i  /path/to/file/containing/matrix
                -s /suffix/common/to/all/matrices # [i.e xsec_cov_throw_matrix]
                --start #Start point for your matrix sequence 
                --stop #End point of your matrix sequence
                --step #Steps between matrix updates 
                --output_pdf /path/to/output/pdf
```
For example for a file `~/scratch/FinalFits/SaveMatrixEveryIter_2MStep.root` corresponding to Adaptive MCMC starting throws at step `100000`, ending throws at step `2000000` with steps between updates of `10000` saving the plots to `my_plots.pdf` this would be

```sh
analyse_matrix -i ~/scratch/FinalFits/SaveMatrixEveryIter_2MStep.root -s xsec_cov_throw_matrix --start 100000 --stop 2000000 --step 10000 --output_pdf better_x_lab.pdf
```

# Current Analysis Methods
Currently we have two convergence metrics

## Norm analysis
Here we simply look at how at how the Frobenius norm, $\lvert\lvert M \rvert\rvert_{frob}= \sqrt{\sum_{i,j} m^{2}_{ij} }$, changes for each matrix, 

**_NOTE:_** This should tend to a constant value as the adaptive MCMC process continues

## Eigenvalue analysis
Here we simply look at how at how the norm of the eigenvalues for each matrix change over time.

**_NOTE:_** This should tend to a constant value as the adaptive MCMC process continues

