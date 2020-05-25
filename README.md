
# Dynamic Mode Decomposition Dashboard

Based on [Casorso et al., 2019][2], the dynamic mode decomposition algorithm allows for a dynamic analysis of cortical neurological activation. Here, a dashboard is developed to allow users to analyse their data using standardized and publication-quality visualizations of the dynamic modes.

## Installation

If you have Anaconda installed, skip to "Installation with conda"

First, make sure you have a working installation of Python 3.5, 3.6, or 3.7 (Python 3.8 is not yet supported).
Run ```python --version``` in your Terminal, which should return your current Python version.

The second step is installing dependencies. To do so in a safe manner, create a virtual environment at the base of the project and install all needed libraries in it

```
pip install virtualenv
virtualenv venv/
source venv/bin/activate
pip install -r requirements/base.txt
```

### Installation with conda

If you have Anaconda installed, navigate to the project's root folder and run the following commands.

```
conda create -n venv python=3.5
source activate venv
pip install -r requirements/base.txt
```

## Running the program

Assuming this installation runs without errors, you should be able to launch the Dynamic Mode Decomposition Dashboard using the following command

```
fbs run
```

#### Segmentation fault on Linux

If you get the following error on Linux,

```
[1:1:0100/000000.386942:ERROR:broker_posix.cc(41)] Invalid node channel message
Segmentation fault (core dumped)
```

Run the following command before using `fbs run` again:

```
export QT_XCB_GL_INTEGRATION=xcb_egl
```


### Setting

When the window appears, you should be greeted with a short description, an input setting choice, and a card with selection information and settings.

Once you select a setting, the description is modified and instructs users how to proceed. Usually, files need to be added using the upload bar and selection settings need to be updated. When files are loaded and settings chosen, users can click on the`Run Decomposition` button and the visualization is launched.

#### Analysis

The analysis setting allows users to analyse the dynamic mode decomposition of one or multiple runs. This is the most straightforward way to use the dashboard.

#### Comparison

The comparison setting is useful when users want to compare two different groups of runs (or two individual runs). It simply displays the informations regarding both decompositions on the same figures.

#### Mode Matching

The mode matching setting is a bit more complex. Users are encouraged to read [Casorso et al., 2019][2] for more information regarding the mathematical aspect of mode matching.

In short, the data from the Match Group is used and modes are matched to the Reference group to allow for a better temporal comparison of the dynamic decomposition.

In this implementation, only the top performing modes (those having the highest damping times) are approximated. The reason for this is the heavy computation of mode matching.

### Exporting your visualizations

The dashboard is built using Dash by Plotly. Therefore, each figure comes with a toolbar which includes a camera icon. When pressed, a dialog appears and asks users for the location where the image should be saved.

To allow for flexibility after export and to ensure high quality, the images are exported in Scalable Vector Graphics format (`.svg`).

### Debugging

A tab called `log` is included at all times to give users the ability to see what the program is doing in the back-end. If any errors are encountered, the detailed error-log should be displayed within this tab. A file version `log.log` can also be found in the `cache/` directory in the project root.

## Input data

This dashboard handles preprocessed data as described in [Casorso et al., 2019 - Methods][2].
The input needed for a successful visualization is one or multiple `.mat` file(s). Each file corresponds to an fMRI run and should contain one matrix of size `N x T`, with `N` being the number of ROIs in the cortical parcellation and `T` being the observational timepoints.

In the current version, two parcellations are supported:

* [Glasser et al., 2016][1], containing `N = 360` regions.
* [Schaefer et al., 2018][3], containing `N = 400` regions.


## References


[1] M. F. Glasser et al., “A multi-modal parcellation of human cerebral cortex,” Nature, vol. 536, no. 7615, pp. 171–178, 11 2016, doi: 10.1038/nature18933.

[2] J. Casorso, X. Kong, W. Chi, D. Van De Ville, B. T. T. Yeo, and R. Liégeois, “Dynamic mode decomposition of resting-state and task fMRI,” NeuroImage, vol. 194, pp. 42–54, Jul. 2019, doi: 10.1016/j.neuroimage.2019.03.019.

[3] A. Schaefer et al., “Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI,” Cerebral Cortex, vol. 28, no. 9, pp. 3095–3114, Sep. 2018, doi: 10.1093/cercor/bhx179.


[2]: http://www.sciencedirect.com/science/article/pii/S1053811919301922
[1]: https://pubmed.ncbi.nlm.nih.gov/27437579/
[3]: https://academic.oup.com/cercor/article/28/9/3095/3978804
[4]: https://build-system.fman.io/
