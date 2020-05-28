import io
import base64
import threading

import scipy.io as sio

import dash
from dash_table import DataTable
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import (Input, Output, State)

from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5 import QtWebEngineWidgets, QtCore, QtWidgets

from dmd.decomposition import Decomposition
from dmd.utils import *
from dmd.plotting import *


class Dashboard(QWebEngineView):
    """
    The Dashboard is where the Dash web-app lives.
    """

    def __init__(self):
        """ Constructor. """

        QWebEngineView.__init__(self)

        # Display full screen
        self.showMaximized()

        # Handle download requests for Plotly image saving
        QtWebEngineWidgets.QWebEngineProfile.defaultProfile().downloadRequested.connect(
            self.on_download_requested
        )

        # Define local host address
        host = dict(address='127.0.0.1', port=8000)

        # Initialize decompositions
        self.dcp1 = None
        self.dcp2 = None
        self.match_group = None
        self.match_df = None
        self.match_x = None
        self.match_y = None
        self.atlas = None

        # Used for cortical surface representation loading
        self.progress = 0

        # Input data valid and visualization can be launched
        self.valid = False

        # Display imaginary values in visualization
        self.imag = False

        # Fetch and add Dash Bootstrap Component theme
        self.app = dash.Dash(
            external_stylesheets=[dbc.themes.COSMO]
        )

        # Initialize log file
        self.logfile = open(CACHE_DIR.joinpath('log.log').as_posix(), 'r')

        # Fetch app layout
        self.app.layout = self._get_app_layout()

        # Use threading to launch Dash App
        threading.Thread(target=self.run_dash, daemon=True).start()

        self.load(QUrl("http://{0}:{1}".format(host['address'], host['port'])))

    @QtCore.pyqtSlot("QWebEngineDownloadItem*")
    def on_download_requested(self, download):
        old_path = download.url().path()  # download.path()
        suffix = QtCore.QFileInfo(old_path).suffix()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", old_path, "*.svg" + suffix
        )
        if path:
            download.setPath(path)
            download.accept()

    def run_dash(self, address='127.0.0.1', port=8000):
        """
        Run Dash.

        :param address: [str] address
        :param port: [int] port
        """

        @self.app.callback(
            Output('log', 'children'),
            [Input('log-update', 'n_intervals')],
            [State('log', 'children')]
        )
        def update_logs(interval, console):

            if console is not None:
                for line in self.logfile.read().split('\n'):
                    console.append(html.Tbody(line))

                return console

            else:
                return None

        @self.app.callback([
            Output('upload-1-div', 'className'),
            Output('upload-1', 'style'),
            Output('upload-2', 'style'),
            Output('selected-files-group-2-t', 'style'),
            Output('selected-files-group-2-p', 'style'),
            Output('selected-files-group-1-t', 'children'),
            Output('selected-files-group-2-t', 'children'),
            Output('upload-1', 'children'),
            Output('upload-2', 'children')
        ], [
            Input('setting', 'value')
        ])
        def input_setting(value):
            """ Modify input setting. """

            show = {}
            hide = dict(display="none")

            style = dict(height="60px", lineHeight="60px", borderWidth="1px", borderStyle="dashed", borderRadius="5px",
                         textAlign="center", margin="10px")

            if value is None:
                return "row", hide, hide, hide, hide, "Selected files", \
                       None, None, None
            elif value == 1:  # Analysis
                return "col-12", style, hide, hide, hide, "Selected files", None, \
                       html.Div(['Drag and Drop or ', html.A('Select Files')]), None
            elif value == 2:  # Comparison
                return "col-6", style, style, show, show, "Group 1", "Group 2", \
                       html.Div(['Group 1: Drag and Drop or ', html.A('Select Files')]), \
                       html.Div(['Group 2: Drag and Drop or ', html.A('Select Files')])
            elif value == 3:  # Matching Modes
                return "col-6", style, style, show, show, "Reference Group", "Match Group",  \
                       html.Div(['Reference Group: Drag and Drop or ', html.A('Select Files')]), \
                       html.Div(['Match Group: Drag and Drop or ', html.A('Select Files')])

        @self.app.callback([
            Output('table-1-tab', 'label'),
            Output('table-2-tab', 'label')
        ], [
            Input('setting', 'value')
        ])
        def update_tab_labels(setting):

            if setting is None:
                raise PreventUpdate
            elif setting == 1:  # Analysis
                return 'Data', ''
            elif setting == 2:  # Comparison
                return 'Group 1', 'Group 2'
            elif setting == 3:  # Mode Matching
                return 'Reference', 'Match'

        @self.app.callback(
            [Output('description', 'children'),
             Output('warning', 'children')],
            [Input('setting', 'value')]
        )
        def update_description(value):
            """ Descriptions. """

            if value is None:
                return "Based on 'Dynamic mode decomposition of resting-state and task fMRI' by Casorso et al, \
                        the dmd dashboard allows you to analyse, compare, and display the decomposition of your \
                        fMRI time-series data. Click on the radio buttons below to get started!", ""
            elif value == 1:  # Analysis
                return "Analysis: this setting allows you to analyse the decomposition of one or multiple time-series \
                       files. Simply input the sampling time, select the one or multiple files you want to analyse, \
                       and the rest is done automatically.", ""
            elif value == 2:  # Comparison
                return "Comparison: this setting allows you to compare the decomposition of two groups of one or \
                       multiple time-series files. Simply input the sampling time, select the groups of one or \
                       multiple files you want to compare, and the rest is done automatically.", ""
            elif value == 3:  # Matching Modes
                return "Matching Modes: this setting allow you to match one group's modes to anothers. The selection \
                        toolbar on the left will take the reference group files, while the one on the right will have \
                        its time-series data matched to the spatial modes of the reference group.", \
                       "Please upload the Reference group before the Match group."

        @self.app.callback([
            Output('animated-progress-1', 'style'),
            Output('animated-progress-2', 'style'),
        ], [
            Input('upload-1', 'contents'),
            Input('upload-2', 'contents')
        ], [
            State('setting', 'value')
        ])
        def progress_file(contents1, contents2, setting):
            """ Progress bar for file upload. """

            if setting is None:
                raise PreventUpdate
            else:
                show = {}
                hide = {'display': 'none'}

                if setting == 1:
                    return show if self.dcp1 is None else hide, hide
                elif setting == 2:
                    return show if self.dcp1 is None and contents1 is not None else hide, \
                           show if self.dcp2 is None and contents2 is not None else hide
                elif setting == 3:
                    return show if self.dcp1 is None and contents1 is not None else hide, \
                           show if self.match_df is None and contents2 is not None else hide
                else:
                    return hide, hide

        @self.app.callback([
            Output('selected-files-group-1-p', 'children'),
            Output('selected-files-group-2-p', 'children'),
            Output('table-1-tab', 'children'),
            Output('table-2-tab', 'children'),
            Output('table-2-tab', 'disabled'),
            Output('animated-progress-1-div', 'style'),
            Output('animated-progress-2-div', 'style'),
            Output('import-alert', 'children'),
            Output('import-alert', 'style'),
        ], [
            Input('upload-1', 'contents'),
            Input('upload-2', 'contents'),
            Input('sampling-time', 'value')
        ], [
            State('upload-1', 'filename'),
            State('upload-2', 'filename'),
            State('setting', 'value')
        ])
        def upload(contents1, contents2, time, names1, names2, setting):

            df1 = None
            df2 = None
            tab1 = None
            tab2 = None
            disabled = True
            error = False
            message = ""

            table_config = dict(
                fixed_rows=dict(headers=True, data=0),
                style_cell=dict(padding="5px"),
                style_header=dict(backgroundColor="white",
                                  fontWeight="bold",
                                  fontFamily="Helvetica",
                                  padding="0px 5px"),
                style_data=dict(fontFamily="Helvetica"),
                style_data_conditional=[{'if': dict(row_index="odd"),  # if is a keyword
                                        **dict(backgroundColor="rgb(248, 248, 248)")}],
                style_as_list_view=True
            )

            columns = [dict(name='#', id='mode'), dict(name='Value', id='value'),
                       dict(name='Damping Time', id='damping_time'), dict(name='Period', id='period')]

            if [contents1, contents2, names1, names2].count(None) == 4:
                raise PreventUpdate
            else:

                try:

                    if contents1 is not None:

                        logging.info("Adding contents to {}.".format("Group 1" if setting == 2 else "Reference Group"))

                        self.dcp1 = _parse_files(contents1, names1, float(time))
                        self.dcp1.run()
                        df1 = self.dcp1.df[['mode', 'value', 'damping_time', 'period']]

                        logging.info("Creating Data Table for {}.".format("Group 1" if setting == 2 else "Reference Group"))

                        tab1 = _create_table(df1, id="table-1", columns=columns, config=table_config)

                    if setting != 1:
                        if contents2 is not None and (self.dcp2 is None or self.match_group is None):
                            if setting == 2:  # Comparison

                                logging.info("Adding contents to Group 2.")

                                self.dcp2 = _parse_files(contents2, names2, float(time))
                                self.dcp2.run()
                                df2 = self.dcp2.df[['mode', 'value', 'damping_time', 'period']]

                                tab2 = _create_table(df2, id="table-2", columns=columns, config=table_config)
                                disabled = False

                            if setting == 3:  # Mode Matching

                                logging.info("Adding contents to Match Group.")

                                self.match_group = _parse_files(contents2, names2, float(time))
                                self.match_group.run()

                                if self.dcp1 is None:
                                    raise ImportError("Please upload Reference group before Match group.")
                                else:
                                    self.match_df, self.match_x, self.match_y = self.dcp1.compute_match(self.match_group, 20)
                                    match_df = self.match_df.copy()

                                    tab2 = _create_table(match_df, id="table-2", columns=columns, config=table_config)
                                    disabled = False

                except Exception as e:
                    message = str(e)
                    error = True

            deb = "Types = Group 1: {0}, Group 2: {1}, Match: {2}".format(type(self.dcp1), type(self.dcp2),
                                                                          type(self.match_group))

            logging.debug(deb)

            hide = {'display': 'none'}
            show = {}

            def indent(lines):
                if isinstance(lines, list):
                    return html.P(', \n'.join(lines))
                else:
                    return html.P(lines)

            return indent(names1), indent(names2), tab1, tab2, disabled, hide if contents1 is not None else show, \
                   hide if contents2 is not None else show, message, show if error else hide

        @self.app.callback(
            Output('imag-setting', 'options')
        , [
            Input('imag-setting', 'value')
        ])
        def imag_switch(switch):
            """ Switch between displaying and not displaying imaginary values. """

            self.imag = switch == [1]

            message = "Plotting imaginary values" if self.imag else "Not plotting imaginary values"

            logging.info(message)

            options = [{"label": message, "value": 1}]

            return options

        @self.app.callback([
            Output('message-alert', 'children'),
            Output('message-alert', 'style'),
            Output('upload-row', 'style')
        ], [
            Input('run', 'n_clicks')
        ], [
            State('setting', 'value')
        ])
        def validate(n, setting):
            """
            Validate input

            :param n:  [int] onClick() "Run Decomposition" button
            :param setting: [int] value of setting radio
            :return message: [str] error message
            :return style: [dict] show or hide error message
            :return style: [dict] show or hide upload row
            """

            message = ""
            error = False

            if n is None:
                raise PreventUpdate
            if setting is not None:
                if setting == 1:
                    if self.dcp1 is None:
                        message += "No file(s) chosen. "
                        error = True
                elif setting == 2:
                    if self.dcp1 is None or self.dcp2 is None:
                        message += "Group {} missing. ".format(2 if self.dcp2 is None else 1)
                        error = True
                elif setting == 3:
                    if self.dcp1 is None:
                        message += "Reference group missing. "
                        error = True
                    if self.match_df is None:
                        message += "Match group is loading. Please wait. "
                        error = True
                elif self.atlas is None:
                    message += "Parsing unsuccessful, no atlas found. Be sure to use a .mat file. "
                    error = True
            else:
                message += "No setting chosen. "
                error = True

            if error:
                message += "Check log for more info."
            else:
                self.valid = True

            hide = dict(display="none")

            return message, {} if error else hide, {} if error else hide

        @self.app.callback([
            Output('app-layout', 'children')
        ], [
            Input('reset', 'n_clicks')
        ])
        def rst(n):
            """ Reset Application. """

            if n is None:
                raise PreventUpdate
            else:
                self.dcp1 = None
                self.dcp2 = None
                self.match_group = None
                self.match_df = None
                self.match_x = None
                self.match_y = None
                self.atlas = None
                self.progress = 0
                self.valid = False
                self.imag = False

                return [self._get_app_layout()]

        @self.app.callback(
            Output('spectre', 'figure')
        ,[
            Input('run', 'n_clicks')
        ], [
            State('number-of-modes', 'value')
        ])
        def compute_spectre(n, nom):
            """
            Compute Spectre figure

            :param n: [int] onClick() "Run Decomposition" button
            :param nom: [int] number of modes
            :return: [go.Figure] figure
            """
            if n is None or not self.valid:
                raise PreventUpdate
            else:

                if self.match_group is None:

                    logging.info("Computing spectre of dynamical modes")
                    s = Spectre(_filter_spectre())
                    return s.figure()

                else:

                    assert self.match_x is not None
                    assert self.match_y is not None
                    logging.info("Computing spectre of dynamical modes")
                    return Spectre.correlation(pd.DataFrame({'Approximated': self.match_x, 'Real': self.match_y}))

        @self.app.callback(
            Output('timeplot', 'figure')
            , [
                Input('run', 'n_clicks')
            ], [
            State('number-of-modes', 'value')
        ])
        def compute_timeplot(n, nom):
            """
            Compute Timeplot figure

            :param n: [int] onClick() "Run Decomposition" button
            :param nom: [int] number of modes
            :return: [go.Figure] figure
            """
            if n is None or not self.valid:
                raise PreventUpdate
            else:

                logging.info("Computing time series activation of dominant modes")

                t = TimePlot(_filter_time())

                return t.figure(nom + 1)

        @self.app.callback(
            Output('radar', 'figure')
        , [
            Input('run', 'n_clicks')
        ], [
            State('number-of-modes', 'value')
        ])
        def compute_radar(n, nom):
            """
            Compute Radar figure

            :param n: [int] onClick() "Run Decomposition" button
            :param nom: [int] number of modes
            :return: [go.Figure] figure
            """
            if n is None or not self.valid:
                raise PreventUpdate
            else:

                logging.info("Computing cortical network activation")

                r = Radar(*_filter_radar())

                return r.figure(self.imag, nom + 1)

        @self.app.callback([
            Output('brains', 'children'),
            Output('progress-div', 'style')
        ], [
            Input('run', 'n_clicks')
        ], [
            State('number-of-modes', 'value')
        ])
        def compute_brain(n, nom):
            """
            Compute brain figures

            :param n: [int] onClick() "Run Decomposition" button
            :param nom: [int] number of modes
            :return brains: [list of html.Div] figures
            :return style: [dict] hide progress Div
            """
            if n is None or not self.valid:
                raise PreventUpdate
            else:

                logging.info("Computing cortical surface representations")

                brains = []

                self.progress += 10.0

                for mode in range(1, nom + 1):

                    b = Brain(*_filter_brain(mode))

                    brains.append(html.Div(children=[
                        dcc.Graph(figure=b.figure(self.imag), config=dict(displaylogo=False,
                                                                          toImageButtonOption=dict(
                                                                              width=None, height=None,
                                                                              format="svg",
                                                                              filename="mode {}".format(mode))))
                    ]))

                    self.progress += 90.0 / nom

                return brains, dict(display="none")

        @self.app.callback([
            Output("progress", "value"),
            Output("progress", "children")
        ], [
            Input("progress-interval", "n_intervals")
        ])
        def progress(n):
            """
            Update progress bar
            Inspired from - https://stackoverflow.com/questions/59241705/dash-progress-bar-for-reading-files
            """
            # check progress of some background process, in this example we'll just
            # use n_intervals constrained to be in 0-100
            prog = min(self.progress % 110, 100)
            # only add text after 5% progress to ensure text isn't squashed too much
            return prog, "{} %".format(prog if prog >= 5 else "")

        # UTILITY FUNCTIONS
        # placed in run_dash because of daemon=True

        def _parse_files(contents, files, sampling_time):
            """
            Parse incoming .mat files.

            :param contents: list of Base64 encoded contents
            :param files: list of names
            :param sampling_time: sampling time of data
            :return decomposition: Decomposition instance
            """

            logging.info("Parsing {0} file{1} with sampling time {2}".format(len(files),
                                                                             's' if len(files) > 1 else '',
                                                                             sampling_time))

            dcp = Decomposition()
            d = None

            for content, name in zip(contents, files):
                type, string = content.split(',')

                mat = io.BytesIO(base64.b64decode(string))

                data = sio.loadmat(mat)

                key = list(data.keys())[-1]
                if key[:2] != '__':
                    d = data[key]
                    logging.info("Extracting matrix from file {} from key {}".format(name, key))
                else:
                    logging.error(".mat file incorrectly formatted.")

                if d is None:
                    logging.error("Invalid .mat file, no matrices inside.")

                dcp.add_data(d, sampling_time)


            dcp.run()
            self.atlas = dcp.atlas

            return dcp

        def _filter_spectre():
            """ Filter df for Spectre Figure. """

            logging.info("Filtering Spectre data")

            df1 = pd.DataFrame({'Mode': self.dcp1.df['mode'], 'Value': np.abs(self.dcp1.df['value']),
                                'Group': ['Group 1' if self.match_group is None else 'Reference' for i in range(self.dcp1.df.shape[0])]}) \
                if self.dcp1 is not None else None

            if self.dcp2 is not None:
                df2 = pd.DataFrame({'Mode': self.dcp2.df['mode'], 'Value': np.abs(self.dcp2.df['value']),
                                    'Group': ['Group 2' for i in range(self.dcp2.df.shape[0])]})
            elif self.match_group is not None:
                df2 = pd.DataFrame({'Mode': self.match_df['mode'], 'Value': np.abs(self.match_df['value']),
                                    'Group': ['Match' for i in range(self.match_df.shape[0])]})
            else:
                df2 = None

            return pd.concat([df1, df2])

        def _filter_time():
            """ Filter df for Timeplot Figure. """

            logging.info("Filtering TimePlot data")

            df1 = pd.DataFrame({'Mode': self.dcp1.df['mode'], 'Activity': self.dcp1.df['activity'],
                                'Group': ['Group 1' for i in range(self.dcp1.df.shape[0])]}) \
                if self.dcp1 is not None else None
            df2 = pd.DataFrame({'Mode': self.dcp2.df['mode'], 'Activity': self.dcp2.df['activity'],
                                'Group': ['Group 2' for i in range(self.dcp2.df.shape[0])]}) \
                if self.dcp2 is not None else None

            return pd.concat([df1, df2])

        def _filter_radar():
            """ Filter df for Radar Figure. """

            logging.info("Filtering Radar data")

            df1 = pd.DataFrame({'mode': self.dcp1.df['mode'], 'group': [1 for i in range(self.dcp1.df.shape[0])],
                                'strength_real': self.dcp1.df['strength_real'], 'strength_imag': self.dcp1.df['strength_imag']}) \
                if self.dcp1 is not None else None
            df2 = pd.DataFrame({'mode': self.dcp2.df['mode'], 'group': [2 for i in range(self.dcp2.df.shape[0])],
                                'strength_real': self.dcp2.df['strength_real'], 'strength_imag': self.dcp2.df['strength_imag']}) \
                if self.dcp2 is not None else None

            networks = self.dcp1.df['networks'][0] if self.dcp1 is not None else None
            if networks is None:
                networks = self.dcp2.df['networks'][0] if self.dcp2 is not None else None

            return pd.concat([df1, df2]), networks

        def _filter_brain(order):
            """
            Filter brain information.

            :param order: [int] mode order
            :return atlas: [str] cortical atlas
            :return mode1: [pd.DataFrame] for mode 1
            :return mode2: [pd.DataFrame] for mode 2
            :return order: [int] mode order
            """

            logging.info("Filtering Brain data for Mode {}".format(order))

            mode1 = self.dcp1.df.loc[order - 1][['intensity', 'conjugate']] if self.dcp1 is not None else None
            mode2 = self.dcp2.df.loc[order - 1][['intensity', 'conjugate']] if self.dcp2 is not None else None

            return self.atlas, mode1, mode2, order

        def _format_table(df):
            """
            Format table

            :param df: [pd.DataFrame] data
            :return: [pd.DataFrame]
            """

            def _set_precision(number):
                if number == np.inf:
                    return 'inf'

                if type(number) != str:
                    number = str(number)
                splat = number.split('.')
                splat[1] = splat[1][:5] if len(splat[1]) > 5 else splat[1]
                return ".".join([splat[0], splat[1]])

            def _handle_complex(number):
                splat = [str(number.real), str(number.imag)]
                sett = [_set_precision(splat[0]), _set_precision(splat[1])]
                return "{0} +/- {1} j".format(sett[0], sett[1])

            def _format_list(p):
                f = _handle_complex if type(p[0]) == np.complex128 else _set_precision
                return [f(e) for e in p]

            df['value'] = _format_list(df['value'])
            df['damping_time'] = _format_list(df['damping_time'])
            df['period'] = _format_list(df['period'])

            return df

        def _create_table(df, id=None, columns=None, config=None):
            """
            Create table.

            :param df: [pd.DataFrame] data
            :param id: [str] id in app.layout
            :param columns: [list] following Dash rules
            :param config: [dict] of config elements for table
            :return: [html.Div] containing DataTable
            """
            if df is None:
                return None
            else:
                data = _format_table(df).to_dict('records')

                return html.Div(className="container mt-4", children=[DataTable(id=id, data=data,
                                                                                columns=columns, **config)])

        self.app.run_server(debug=False, port=port, host=address)

    @staticmethod
    def _get_app_layout():

        logging.info("Setting Application Layout")

        return html.Div(id="app-layout", children=[
            html.Div(id="dummy", style=dict(display="none")),
            # ####################### #
            # HEADER + FILE SELECTION
            # ####################### #
            html.Div(className="row", children=[
                # HEADER = TITLE + DESCRIPTION + SETTING
                html.Div(className="col-5", style={'margin-top': '25px'}, children=[
                    dbc.FormGroup(row=True, children=[
                        html.Div(className="ml-5 mt-2", children=[
                            # TITLE
                            html.H4("Dynamic Mode Toolbox"),
                            # DESCRIPTION
                            html.P(id="description"),
                            html.P(id="warning", className="text-danger mt-1"),
                        ]),
                        dbc.Col(className="ml-5 mt-4", children=[
                            # SETTING
                            dbc.RadioItems(id="setting", options=[
                                    {"label": "Analysis", "value": 1},
                                    {"label": "Comparison", "value": 2},
                                    {"label": "Mode Matching", "value": 3}
                            ])
                        ])
                    ])
                ]),
                # FILE SELECTION
                html.Div(className="col-7", children=[
                    html.Div(className="col-12 mt-2 mb-2 mr-2 ml-2", id="file-selection-card", children=[
                        dbc.Card(dbc.CardBody(children=[
                            # TITLE
                            html.H5("Selection", className="card-title"),
                            html.Div(className="row", children=[
                                # SELECTION INPUT
                                html.Div(className="col-6", children=[
                                    dbc.FormGroup(children=[
                                        dbc.Label("Sampling Time (seconds)", className="mt-2"),
                                        dbc.Input(id="sampling-time", type="number", placeholder="0.72",
                                                  value=0.72, className="col-6"),
                                        dbc.Label("Number of modes to plot", className="mt-2"),
                                        dbc.Input(id="number-of-modes", type="number", placeholder="5",
                                                  value=5, className="col-6"),
                                        dbc.Checklist(className="mt-2", id="imag-setting", value=[], switch=True,
                                                      options=[dict(label="Plot Imaginary Values", value=1)]),
                                    ]),
                                ]),
                                # SELECTED FILES
                                html.Div(className="col-6", children=[
                                    dbc.Label(id="selected-files-group-1-t"),
                                    html.P(id="selected-files-group-1-p"),
                                    html.Div(className="mb-2", id="animated-progress-1-div", children=[
                                        dbc.Progress(value=80, id="animated-progress-1", striped=True,
                                                     animated="animated", style={'display': 'none'})
                                    ]),
                                    dbc.Label(id="selected-files-group-2-t"),
                                    html.P(id="selected-files-group-2-p"),
                                    html.Div(id="animated-progress-2-div", className="mb-2", children=[
                                        dbc.Progress(value=80, id="animated-progress-2", striped=True,
                                                     animated="animated", style={"display": "none"})
                                    ]),
                                    html.Div(id="import-alert", className="text-danger mt-2")
                                ]),
                            ]),
                            # BUTTONS + ALERT MESSAGE
                            html.Div(children=[
                                dbc.Button("Run Decomposition", color="primary", id="run", className="mr-2"),
                                dbc.Button("Reset", color="secondary", id="reset"),
                                html.Div(id="message-alert", className="text-danger mt-2")
                            ]),
                        ]))
                    ]),
                ]),
            ]),
            # ########## #
            # UPLOAD ROW
            # ########## #
            html.Div(className="row", id="upload-row", children=[
                # UPLOAD 1
                html.Div(className="col-6", id="upload-1-div", children=[
                    dcc.Upload(id="upload-1", multiple=True)
                ]),
                # UPLOAD 2
                html.Div(className="col-6", id="upload-2-div", children=[
                    dcc.Upload(id="upload-2", multiple=True)
                ])
            ]),
            # ####### #
            # CONTENT #
            # ####### #
            # TABS = GRAPHS + TABLE 1 + TABLE 2 + LOG
            dbc.Tabs(children=[
                # GRAPHS
                dbc.Tab(label="Graphs", children=[
                    html.Div(className="row", children=[
                        # LEFT PANEL = RADAR + SPECTRE + TIMEPLOT
                        html.Div(className="col-5", children=[
                            # RADAR
                            html.Div(className="row", children=[
                                html.Div(className="col-12", children=[
                                    dcc.Graph(id="radar",
                                              config=dict(displaylogo=False,
                                                          toImageButtonOption=dict(
                                                              width=None, height=None,
                                                              format="svg", filename="radar")))
                                ]),
                            ]),
                            # SPECTRE
                            html.Div(className="row", children=[
                                html.Div(className="col-12", children=[
                                    dcc.Graph(id="spectre",
                                              config=dict(displaylogo=False,
                                                          toImageButtonOption=dict(
                                                              width=None, height=None,
                                                              format="svg", filename="spectre")))
                                ]),
                            ]),
                            # TIMEPLOT
                            html.Div(className="row", children=[
                                html.Div(className="col-12", children=[
                                    dcc.Graph(id="timeplot",
                                              config=dict(displaylogo=False,
                                                          toImageButtonOption=dict(
                                                              width=None, height=None,
                                                              format="svg", filename="timeplot")))
                                ]),
                            ]),
                        ]),
                        # RIGHT PANEL = BRAINS + PROGRESS
                        html.Div(className="col-7", children=[
                            # BRAINS
                            html.Div(className="col-12", id="brains"),
                            # PROGRESS
                            html.Div(className="col-12 my-4 mx-4", id="progress-div", children=[
                                html.P('Loading cortical surface graphs...', className="mt-4"),
                                dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
                                dbc.Progress(id="progress", style=dict(width='70%', align='center')),
                            ]),
                        ]),
                    ]),
                ]),
                # TABLE 1
                dbc.Tab(label="Group A", id="table-1-tab"),
                # TABLE 2
                dbc.Tab(label="Group B", disabled=True, id="table-2-tab"),
                # LOG
                dbc.Tab(label="Log", id='log-tab', children=[
                    html.Div(className="col-12", id="log-div", children=[
                        dcc.Interval(id='log-update', interval=1000),  # interval in milliseconds
                        html.Div(id='log', children=[
                            html.P("———— APP START ————"),
                        ]),
                    ]),
                ]),
            ]),
        ])
