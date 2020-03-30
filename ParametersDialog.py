# This Python file uses the following encoding: utf-8
from PyQt5 import QtWidgets, QtCore
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from nilearn.plotting import show
import logging

class ParametersDialog(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)

        self.setWindowTitle("Set Decomposition Parameters")

        self.setWindowModality(QtCore.Qt.WindowModal)
        self.resize(363, 187)
        self.setSizeGripEnabled(True)
        self.setModal(False)

        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(10, 140, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")

        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 340, 121))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.paramLayoutShadow = QtWidgets.QHBoxLayout()
        self.paramLayoutShadow.setObjectName("paramLayoutShadow")

        self.labelShadow = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelShadow.sizePolicy().hasHeightForWidth())
        self.labelShadow.setSizePolicy(sizePolicy)
        self.labelShadow.setObjectName("labelShadow")
        self.labelShadow.setText("Sulcal Shading")

        self.paramLayoutShadow.addWidget(self.labelShadow)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.paramLayoutShadow.addItem(spacerItem)

        self.comboShadow = QtWidgets.QComboBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboShadow.sizePolicy().hasHeightForWidth())
        self.comboShadow.setSizePolicy(sizePolicy)
        self.comboShadow.setObjectName("comboShadow")
        self.comboShadow.addItems(['Yes', 'No'])

        self.paramLayoutShadow.addWidget(self.comboShadow)

        self.verticalLayout.addLayout(self.paramLayoutShadow)

        self.paramLayoutColobar = QtWidgets.QHBoxLayout()
        self.paramLayoutColobar.setObjectName("paramLayoutColobar")

        self.labelColorbar = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelColorbar.sizePolicy().hasHeightForWidth())
        self.labelColorbar.setSizePolicy(sizePolicy)
        self.labelColorbar.setObjectName("labelColorbar")
        self.labelColorbar.setText("Colorbar")

        self.paramLayoutColobar.addWidget(self.labelColorbar)

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.paramLayoutColobar.addItem(spacerItem1)

        self.comboColorbar = QtWidgets.QComboBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboColorbar.sizePolicy().hasHeightForWidth())
        self.comboColorbar.setSizePolicy(sizePolicy)
        self.comboColorbar.setObjectName("comboColorbar")

        # Fetch colormaps
        deprecated_cmaps = ['Vega10', 'Vega20', 'Vega20b', 'Vega20c', 'spectral']
        m_cmaps = []
        for m in plt.cm.datad:
            if not m.endswith("_r") and m not in deprecated_cmaps:
                m_cmaps.append(m)
        m_cmaps.sort()
        colorMaps = []
        for _, cmap in enumerate(m_cmaps):
            colorMaps.append(cmap)

        self.comboColorbar.addItems(colorMaps)

        self.paramLayoutColobar.addWidget(self.comboColorbar)

        self.verticalLayout.addLayout(self.paramLayoutColobar)

        self.paramLayoutSurfMesh = QtWidgets.QHBoxLayout()
        self.paramLayoutSurfMesh.setObjectName("paramLayoutSurfMesh")

        self.labelSurfMesh = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelSurfMesh.sizePolicy().hasHeightForWidth())
        self.labelSurfMesh.setSizePolicy(sizePolicy)
        self.labelSurfMesh.setObjectName("labelSurfMesh")
        self.labelSurfMesh.setText("Surface Mesh")

        self.paramLayoutSurfMesh.addWidget(self.labelSurfMesh)

        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.paramLayoutSurfMesh.addItem(spacerItem2)

        self.comboSurfMesh = QtWidgets.QComboBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboSurfMesh.sizePolicy().hasHeightForWidth())
        self.comboSurfMesh.setSizePolicy(sizePolicy)
        self.comboSurfMesh.setObjectName("comboSurfMesh")
        self.comboSurfMesh.addItems(['inflated', 'pial', 'sulcal'])

        self.paramLayoutSurfMesh.addWidget(self.comboSurfMesh)

        self.verticalLayout.addLayout(self.paramLayoutSurfMesh)

        QtCore.QMetaObject.connectSlotsByName(self)