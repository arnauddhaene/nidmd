# Dynamic Mode Decomposition

Based on [Casorso et al., 2019][2], the dynamic mode decomposition (DMD) algorithm allows for a dynamic analysis of cortical neurological activation. Here, a Python module is developed facilitating both analysis and visualization aspects of the DMD.

## Installation

To install the package, simply run the following command::

    pip install nidmd

## Usage

### Dashboard

In parallel to this Python module, a dashboard called [nidmd-dashboard](https://github.com/arnauddhaene/nidmd-dashboard) has been developed to facilitate analysis, comparison, and mode matching of the DMD of time-series fMRI data.

### Input data

This dashboard handles preprocessed data as described in [Casorso et al., 2019 - Methods][2].
The input needed for a successful visualization is one or multiple files containing time-series data. Each file corresponds to an fMRI run and should contain one matrix of size `N x T`, with `N` being the number of ROIs in the cortical parcellation and `T` being the observational timepoints.

In the current version, two parcellations are supported:

* [Glasser et al., 2016][1], containing `N = 360` regions.
* [Schaefer et al., 2018][2], containing `N = 400` regions.

### Examples

A Jupyter Notebook can be found in the `examples` directory. It complements the [documentation](arnauddhaene.github.io/nidmd).

## References


[1] M. F. Glasser et al., “A multi-modal parcellation of human cerebral cortex,” Nature, vol. 536, no. 7615, pp. 171–178, 11 2016, doi: 10.1038/nature18933.

[2] J. Casorso, X. Kong, W. Chi, D. Van De Ville, B. T. T. Yeo, and R. Liégeois, “Dynamic mode decomposition of resting-state and task fMRI,” NeuroImage, vol. 194, pp. 42–54, Jul. 2019, doi: 10.1016/j.neuroimage.2019.03.019.

[3] A. Schaefer et al., “Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI,” Cerebral Cortex, vol. 28, no. 9, pp. 3095–3114, Sep. 2018, doi: 10.1093/cercor/bhx179.


[2]: http://www.sciencedirect.com/science/article/pii/S1053811919301922
[1]: https://pubmed.ncbi.nlm.nih.gov/27437579/
[3]: https://academic.oup.com/cercor/article/28/9/3095/3978804
[4]: https://build-system.fman.io/
