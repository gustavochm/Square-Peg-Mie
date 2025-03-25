# Supplementary information: "Fitting a Square Peg in a Round Hole: parameterisation of quasi-spherical molecules employing the Mie potential" by Gustavo Chaparro and Erich A. Müller.

This repository is part of the supplementary data of the article "Fitting a Square Peg in a Round Hole: parameterisation of quasi-spherical molecules employing the Mie potential". This repository includes scripts and ANN-based models to reproduce the results of the article.

The following folders are included:
1. [Calculation scripts](./1_calculation_scripts). This folder includes the Jupyter notebooks used to parameterise the Mie potential for several quasi-spherical molecules. The results are ultimately saved in the [computed_files](./computed_files/) folder.
1. [Plotting scripts](./2_plotting_scripts/). This folder includes the scripts to reproduce the figures of the article. These scripts read the data available in the [computed_files](./computed_files/) folder and save the figures in the [figures](./figures/) folder.
1. [python_helpers](./python_helpers/). This folder includes Python functions to use the FE-ANN EoS and the transport properties models.
1. [ann_models](./ann_models). This folder includes pre-trained ANN parameters for the FE-ANN EoS, FE-ANN(s) EoS and transport properties models. These models are updated versions of the original publications. Further details can be obtained from Gustavo's PhD thesis and this [repository](https://github.com/gustavochm/Chaparro-PhD-Thesis).

### Dependencies

These scripts were tested using the following packages.
- `jax==0.4.4`
- `flax==0.6.6`
- `numpy==1.24.2`

### License information

See ``LICENSE`` for information on the terms and conditions for usage of this software and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the license, if it is convenient for you, please cite this if used in your work. Please also consider contributing any changes you make back, and benefit the community.