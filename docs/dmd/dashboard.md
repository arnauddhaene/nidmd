Module dmd.dashboard
====================

Classes
-------

`Dashboard()`
:   The Dashboard is where the Dash web-app lives.
    
    Constructor.

    ### Ancestors (in MRO)

    * PyQt5.QtWebEngineWidgets.QWebEngineView
    * PyQt5.QtWidgets.QWidget
    * PyQt5.QtCore.QObject
    * sip.wrapper
    * PyQt5.QtGui.QPaintDevice
    * sip.simplewrapper

    ### Methods

    `on_download_requested(self, download)`
    :

    `run_dash(self, address='127.0.0.1', port=8000)`
    :   Run Dash
        
        Parameters
        ----------
        address: str
            address of localhost
        port: int
            port of localhost