# This Python file uses the following encoding: utf-8
import scipy.io as scp
import numpy as np
import math, cmath
import os, shutil
import json
from nilearn import datasets, plotting, surface
import nibabel.freesurfer as fs
from plotting import RadarPlot

PI = math.pi

class Decomposition:
    def __init__(self, filenames, how_many_modes=5):
        # Read Global Variables which include paths to ATLASes
        self.gv = self.readGlobalVariables()
        self.filenames = filenames

        # Fetch surface mesh
        self.fsaverage = datasets.fetch_surf_fsaverage('fsaverage', 'resources')

        # Get Dynamic Modes from filenames
        self.extractData()  # this line defines data
        self.checkData()  # this line defines X, Y, N, T
        self.getDynamicModes()  # this line defines A, eigVal, eigVec, eigIdx

        # Identify atlas depending on number of ROIs in data
        self.atlas = self.gv['atlas'][str(self.N)]

        # Create modes dict {1: 'path1', 2: 'path2', ...}
        self.modes = self.createModeDirs(how_many_modes)

        # List of Dictionaries [ {'left':'pathL1', 'right':'pathR1'}, ...]
        self.annots = []

        # Create annot files for BrainViews
        for mode in range(len(self.modes)):
            self.annots.append(self.writeLabels(self.modes[mode], np.real(self.eigVec[:, mode + 1])))

    def radar_plot(self, modes=5):
        """
        Create RadarPlot instance displaying active networks.

        :param modes: modes to plot (int)
        :return: RadarPlot instance
        """

        labels = [self.gv['networks'][self.atlas][network]['name'] for network in self.gv['networks'][self.atlas]]
        idx = [self.gv['networks'][self.atlas][network]['index'] for network in self.gv['networks'][self.atlas]]
        # Global Variables contain MATLAB (1->) vs. Python (0->) indices
        index = [np.add(np.asarray(idx[i]), -1) for i in range(len(idx))]

        data = {}
        for axis in ['real','complex']:

            to_axis = (np.real if axis == 'real' else np.imag)

            data[axis] = [[
                np.mean(to_axis(self.eigVec[index[network], mode + 1])) for network in range(len(index))
            ] for mode in range(modes)]

        return RadarPlot(data, labels)

    def reset(self):
        """
        Reset
        """
        self.clear_cache()

    def readGlobalVariables(self):
        """
        Read JSON file containing all global variables.

        :return gv: global variables
        """
        with open('resources/GLOBAL.JSON') as json_file:
            return json.load(json_file)

    def clear_cache(self):
        """
        Clear cache directory.
        """
        for filename in os.listdir('cache'):
            file_path = os.path.join('cache', filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def clear_target(self):
        """
        Clear target directory.
        """
        for filename in os.listdir('target'):
            file_path = os.path.join('target', filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def createModeDirs(self, howMany):
        """
        Make directories to store created files.

        :param howMany: how many modes should be created
        :return modes: list of paths
        """
        self.clear_cache()

        modes = ['cache/mode-%s' % mode for mode in range(howMany)]
        for mode in modes: os.mkdir(mode)

        return modes

    def saveHTML(self, surf = 'inflated', colorbar = 'RdYlGn', shadow = True):
        """
        Save 3D brain surface plots to html files.

        :return htmls: list of dict of hemispheric .html filepaths
        """
        htmls = []

        for mode in range(len(self.annots)):
            
            htmls.append(mode)
            htmls[mode] = {}
            
            for hemi in ['left', 'right']:
    
                    dir, filename = os.path.split(self.annots[mode][hemi])
    
                    # TODO ADD FILE EXTENSION PROBLEM
                    htmls[mode][hemi] = os.path.join(dir, hemi + '.html')

                    labels = surface.load_surf_data(self.annots[mode][hemi])

                    args = {'surf_mesh': self.fsaverage['{0}_{1}'.format(surf[:4], hemi)],
                            'surf_map': labels,
                            'cmap': colorbar,
                            'bg_map': self.fsaverage['sulc_%s' % hemi] if shadow else None,
                            'symmetric_cmap': False,
                            'colorbar_fontsize': 10,
                            'title': 'Dynamic Mode ' + str(mode) + ': ' + hemi + ' hemisphere',
                            'title_fontsize': 12}

                    view = plotting.view_surf(**args)

                    view.save_as_html(htmls[mode][hemi])
        
        return htmls

    def writeLabels(self, dirpath, eigenVector):
        """
        Write Data mapping .annot file for Dynamic Modes.

        :param dirpath: Path to new where the .annot files will be written
        :param atlas: Accepted parcellations are : {'glasser','schaefer'}
        :param eigenVector: eigenvector of selected Dynamic Mode.
        :return writepath: dict of 'left' and 'right' hemispheric .annot filepaths
        """
        writepath = {}

        for hemi in ['left', 'right']:
            # extract parcellation
            labels, ctab, names = fs.io.read_annot(self.gv['label'][self.atlas][hemi])

            # initialize new color tab
            atlasLength = ctab.shape[0]
            newCtab = np.empty((atlasLength, 5))

            # first row is always the same as old colortab
            newCtab[0] = ctab[0]

            # change color according to eigenvector values
            scale = 127 / (np.max(eigenVector) - np.min(eigenVector))
            mean = np.mean(eigenVector)

            for roi in range(1, atlasLength):
                eigVecIdx = (roi - 1) if (hemi == 'left') else (roi + (atlasLength - 2))
                color = 127 + scale * (eigenVector[eigVecIdx] - mean)
                newCtab[roi] = np.array([color, color, color, 0, labels[roi]])

            writepath[hemi] = os.path.join(dirpath, hemi + '.annot')

            fs.io.write_annot(writepath[hemi], labels, newCtab, names, fill_ctab = True)

        return writepath


    def getDynamicModes(self):
        """
        Get dynamic modes by Least Squares optimization. To use the index simply use eigVal[idx] and eigVec[:,idx]
        """
        self.A = (self.X @ self.Y.T) @ (np.linalg.inv(self.Y @ self.Y.T))

        # extract eigenvalues and eigenvectors
        self.eigVal, self.eigVec = np.linalg.eig(self.A)

        # sort descending - from https://stackoverflow.com/questions/8092920/
        # simply use index for later use
        absEigVal = abs(self.eigVal)
        self.eigIdx = absEigVal.argsort()[::-1]

        # adjust eigenvectors' phases for orthogonality
        self.eigVec = self.adjustPhase(self.eigVec)

    def checkData(self):
        """
        Check and format data into autoregressive model.
        """
        # 'empty' arrays for creating X and Y
        self.X = np.array([]).reshape(self.data[0].shape[0], 0)
        self.Y = np.array([]).reshape(self.data[0].shape[0], 0)

        for matrice in self.data:
            # check for zero rows
            # indices of rows that are zero (full zero ROIs)
            zIdx = np.where(~matrice.any(axis=1))[0]
            logMsg = "ROIError. Matrice contains " + str(zIdx.shape) + " zero rows. "
            # TODO: remove row ? how will that be shown in the graph then ????

            # normalize matrices
            matriceN, _, _ = self.normalize(matrice, direction=1, demean=True, destandard=False)

            # concatenate matrices
            Xn = matriceN[:, 1:  ]
            Yn = matriceN[:,  :-1]
            self.X = np.concatenate((self.X, Xn), axis=1)
            self.Y = np.concatenate((self.Y, Yn), axis=1)

        self.N, self.T = self.X.shape

#        if not self.N > self.T:
#            logMsg = 'DimensionError. Not enough time points to use auto-regressive model. More than ' +  ATLAS[N] + ' time points are needed.'
#            # TODO: find a way to stop program at this point in time


#        logMsg = 'Data matrice succesfully extracted with ' + ATLAS[N] + ' atlas and ' + str(T) + ' time points.'

    def extractData(self):
        """
        Extracts fMRI data from files.
        Supported formats are: .mat
        """
        logMsg = ''
        self.data = []

        if isinstance(self.filenames, list):
            for filename in self.filenames:
                if (self.fileFormat(filename) == '.mat'):
                    mat = scp.loadmat( filename )
                    self.data.append(mat['TCSnf'])
#                    else:
                    #, "FileError. Filetype not supported. If multiple files are processed, make sure they all have the same file extension."

            #, "Data successfully extracted."
#            else:

            #, "TypeError. Parameter filenames must be a list of strings."

    def fileFormat(self, filename):
        """
        Find file format by extracting what comes after the dot.

        :param filename: filename
        :return format: str of the file format
        """
        return os.path.splitext(filename)[-1]

    def normalize(self, x, direction=1, demean=True, destandard=False):
        """
        Normalize the original data set.

        :param x: data
        :param direction: 0 for columns, 1 for lines, None for global
        :param demean: demean
        :param destandard: remove standard deviation (to 1)
        :return x: standardized data
        :return mean: mean of original data
        """
        r, c = x.shape
        std_x = np.std(x, axis=direction)
        mean = np.mean(x, axis=direction)

        # removal of mean
        if demean:
            x -= mean.reshape( (mean.shape[0]), 1 )
            # different implementation
            # x -= mean[:, None]

        # removal of standard deviation
        if destandard:
            # not sure about formulation - not needed in this protocol should usually be F
            # x = x / std_x
            print('Standardization formula not too sure')

        return x, mean, std_x

    def adjustPhase(self, x):
        """
        Adjust phase as to have orthogonal eigenVectors

        :param x: original eigenvectors
        :return ox: orthogonalized eigenvectors
        """

        # create empty instance for ox
        ox = np.empty(shape = x.shape, dtype=complex)

        for j in range(1, x.shape[1]):

            # seperate real and imaginary parts
            a = np.real(x[:,j])
            b = np.imag(x[:,j])

            # phase calculation
            phi = 0.5 * np.arctan( 2 * (a @ b) / (b.T @ b - a.T @ a) )

            # compute normalised a, b
            anorm = np.linalg.norm( math.sin(phi) * a + math.cos(phi) * b )
            bnorm = np.linalg.norm( math.cos(phi) * a - math.sin(phi) * b )

            if bnorm > anorm:
                if phi < 0:
                    phi -= PI / 2
                else:
                    phi += PI / 2

            ox[:,j] = np.multiply(x[:,j], cmath.exp( complex(0,1) * phi ))

        return ox