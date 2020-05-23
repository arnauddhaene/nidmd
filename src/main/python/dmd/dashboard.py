import io
import base64
import threading
import scipy.io as sio
import dash
import plotly.express as px
from dash_table import DataTable
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from flask_caching import Cache
from dash.dependencies import (Input, Output, State)
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5 import QtWebEngineWidgets, QtCore, QtWidgets
from decomposition import Decomposition
from utils import *
from plotting import *


class Dashboard(QWebEngineView):

    def __init__(self):
        """
        Constructor.

        """
        # the figure is now accessible via self.figure
        QWebEngineView.__init__(self)
        self.showMaximized()

        QtWebEngineWidgets.QWebEngineProfile.defaultProfile().downloadRequested.connect(
            self.on_downloadRequested
        )

        address = {'address': '127.0.0.1',
                   'port': 8000
        }

        self.dcp1 = None
        self.dcp2 = None
        self.match_group = None
        self.match_df = None
        self.atlas = None
        self.progress = 0
        self.valid = False  # put to true if decomposition can run
        self.imag = False

        self.app = dash.Dash(
            external_stylesheets=[dbc.themes.COSMO]
        )

        self.logfile = open(CACHE_DIR.joinpath('log.log').as_posix(), 'r')

        self.app.layout = self._set_app_layout()

        threading.Thread(target=self.run_dash, daemon=True).start()
        # self.run_dash()

        self.load(QUrl("http://{0}:{1}".format(address['address'], address['port'])))

    @QtCore.pyqtSlot("QWebEngineDownloadItem*")
    def on_downloadRequested(self, download):
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
        Run Dash

        :param address: address (str)
        :param port: port (int)
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

            uploadStyle = {
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            }

            if value is None:
                return "row", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, "Selected files", \
                       None, None, None
            elif value == 1:  # Analysis
                return "col-12", uploadStyle, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, None, None, \
                       html.Div(['Drag and Drop or ', html.A('Select Files')]), None
            elif value == 2:  # Comparison
                return "col-6", uploadStyle, uploadStyle, {}, {}, "Group 1", "Group 2", \
                       html.Div(['Group 1: Drag and Drop or ', html.A('Select Files')]), \
                       html.Div(['Group 2: Drag and Drop or ', html.A('Select Files')])
            elif value == 3:  # Matching Modes
                return "col-6", uploadStyle, uploadStyle, {}, {}, "Reference Group", "Match Group",  \
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
                return 'Data', 'Disabled'
            elif setting == 2:  # Comparison
                return 'Group 1', 'Group 2'
            elif setting == 3:  # Mode Matching
                return 'Reference', 'Match'

        @self.app.callback(
            Output('description', 'children')
        , [
            Input('setting', 'value')
        ])
        def update_description(value):

            if value is None:
                return "Based on 'Dynamic mode decomposition of resting-state and task fMRI' by Casorso et al, \
                        the dmd dashboard allows you to analyse, compare, and display the decomposition of your \
                        fMRI time-series data. Click on the radio buttons on the right to get started!"
            elif value == 1: # Analysis
                return "Analysis: this setting allows you to analyse the decomposition of one or multiple time-series \
                       files. Simply input the sampling time, select the one or multiple files you want to analyse, \
                       and the rest is done automatically."
            elif value == 2: # Comparison
                return "Comparison: this setting allows you to compare the decomposition of two groups of one or multiple time-series \
                       files. Simply input the sampling time, select the groups of one or multiple files you want to compare, \
                       and the rest is done automatically."
            elif value == 3: # Matching Modes
                return "Matching Modes: this setting allow you to match one group's modes to anothers. The selection \
                        toolbar on the left will take the reference group files, while the one on the right will have \
                        its time-series data matched to the spatial modes of the reference group."



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
            Output('animated-progress-2-div', 'style')
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
            message = ""
            error = False

            table_config = dict(
                fixed_rows={'headers': True, 'data': 0},
                style_cell={'padding': '5px'},
                style_header={
                    'backgroundColor': 'white',
                    'fontWeight': 'bold',
                    'font-family': 'Helvetica',
                    'padding': '0px 5px'
                },
                style_data={'font-family': 'Helvetica'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'Mode'},
                        'textAlign': 'left'
                    },
                    {'if': {'column_id': 'Mode'},
                     'width': '15%'},
                    {'if': {'column_id': 'Value'},
                     'width': '30%'},
                    {'if': {'column_id': 'Damping Time'},
                     'width': '35%'},
                    {'if': {'column_id': 'Period'},
                     'width': '20%'},
                ],
                style_as_list_view=True
            )

            if [contents1, contents2, names1, names2].count(None) == 4:
                raise PreventUpdate
            else:

                if contents1 is not None:

                    logging.info("Adding contents to Analysis / Group 1 / Reference Group.")

                    self.dcp1 = _parse_files(contents1, names1, float(time))
                    self.dcp1.run()
                    df1 = self.dcp1.df[['mode', 'value', 'damping_time', 'period']]
                    data1 = _format_table(df1).to_dict('records') if df1 is not None else dict()
                    columns1 = [dict(name='#', id='mode'), dict(name='Value', id='value'),
                                dict(name='Damping Time', id='damping_time'),
                                dict(name='Period', id='period')] if df1 is not None else [
                        {"name": "none", "id": "none"}]
                    tab1 = html.Div(DataTable(
                        id="table-1", data=data1, columns=columns1, **table_config
                    ), className="container") if self.dcp1 is not None else None

                if setting != 1:
                    if contents2 is not None:
                        if setting == 2:  # Comparison

                            logging.info("Adding contents to Group 2.")

                            self.dcp2 = _parse_files(contents2, names2, float(time))
                            self.dcp2.run()
                            df2 = self.dcp2.df[['mode', 'value', 'damping_time', 'period']]
                            data2 = _format_table(df2).to_dict('records') if df2 is not None else dict()
                            columns2 = [dict(name='#', id='mode'), dict(name='Value', id='value'),
                                        dict(name='Damping Time', id='damping_time'),
                                        dict(name='Period', id='period')] if df2 is not None else [
                                {"name": "none", "id": "none"}]
                            tab2 = html.Div(DataTable(
                                id="table-2", data=data2, columns=columns2, **table_config
                            ), className="container") if self.dcp2 is not None else None
                            disabled = False

                        if setting == 3:  # Mode Matching

                            logging.info("Adding contents to Match Group.")

                            self.match_group = _parse_files(contents2, names2, float(time))
                            self.match_group.run()

                            if self.dcp1 is not None:

                                self.match_df = self.dcp1.compute_match(self.match_group, 20)

                                match_df = self.match_df.copy()

                                match_data = _format_table(match_df).to_dict('records') if match_df is not None else dict()

                                match_columns = [dict(name='#', id='mode'), dict(name='Value', id='value'),
                                                 dict(name='Damping Time', id='damping_time'),
                                                 dict(name='Period', id='period')] if match_df is not None else [
                                                {"name": "none", "id": "none"}]
                                tab2 = html.Div(DataTable(
                                    id="table-2", data=match_data, columns=match_columns, **table_config
                                ), className="container") if self.match_group is not None else None

                                disabled = False

            dumbo = "Types = Group 1: {0}, Group 2: {1}, Match: {2}".format(type(self.dcp1), type(self.dcp2), type(self.match_group))

            logging.debug(dumbo)

            hide = {'display': 'none'}
            show = {}

            def format(list):
                if type(list) == list:
                    return html.P('\n'.join(list))
                else:
                    return html.P(list)

            return format(names1), format(names2), tab1, tab2, \
                   disabled, hide if contents1 is not None else show, hide if contents2 is not None else show

        @self.app.callback(
            Output('imag-setting', 'options')
        , [
            Input('imag-setting', 'value')
        ])
        def imag_switch(switch):

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
                    if self.match_group is None:
                        message += "Match group missing. "
                        error = True
                elif self.atlas is None:
                    message += "Parsing unsuccessful, no atlas found. Please use a .mat file with the data under key 'TCS' or 'TCSnf'. "
                    error = True
            else:
                message += "No setting chosen. "
                error = True

            if error:
                message += "Check log for more info."
            else:
                self.valid = True

            return message, {} if error else {'display':'none'}, {} if error else {'display':'none'}

        @self.app.callback([
            Output('app-layout', 'children')
        ], [
            Input('reset', 'n_clicks')
        ])
        def rst(n):

            if n is None:
                raise PreventUpdate
            else:
                self.dcp1 = None
                self.dcp2 = None
                self.match_group = None
                self.match_df = None
                self.atlas = None
                self.progress = 0
                self.valid = False  # put to true if decomposition can run
                self.imag = False

                return [self._set_app_layout()]

        @self.app.callback(
            Output('spectre', 'figure')
        ,[
            Input('run', 'n_clicks')
        ], [
            State('number-of-modes', 'value')
        ])
        def compute_spectre(n, nom):

            if n is None or not self.valid:
                raise PreventUpdate
            else:

                logging.info("Computing spectre of dynamical modes")

                s = Spectre(_filter_spectre())

                return s.figure()

        @self.app.callback(
            Output('timeplot', 'figure')
            , [
                Input('run', 'n_clicks')
            ], [
            State('number-of-modes', 'value')
        ])
        def compute_timeplot(n, nom):

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

            if n is None or not self.valid:
                raise PreventUpdate
            else:

                logging.info("Computing cortical surface representations")

                brains = []

                self.progress += 10.0

                for mode in range(1, nom + 1):

                    b = Brain(*_filter_brain(mode))

                    brains.append(html.Div([dcc.Graph(figure=b.figure(self.imag),
                                                      config={"toImageButtonOptions": {"width": None,
                                                                                       "height": None,
                                                                                       "format": "svg",
                                                                                       "filename": "mode {}".format(mode)},
                                                              "displaylogo": False})]))

                    self.progress += 90.0 / nom

                return brains, {'display': 'none'}

        @self.app.callback([
            Output("progress", "value"),
            Output("progress", "children")
        ], [
            Input("progress-interval", "n_intervals")
        ])
        def progress(n):
            # check progress of some background process, in this example we'll just
            # use n_intervals constrained to be in 0-100
            progress = min(self.progress % 110, 100)
            # only add text after 5% progress to ensure text isn't squashed too much
            return progress, "{} %".format(progress if progress >= 5 else "")

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

            for content, name in zip(contents, files):
                type, string = content.split(',')

                mat = io.BytesIO(base64.b64decode(string))

                data = sio.loadmat(mat)

                for key in mat.keys():
                    if key[:2] != '__':
                        d = mat[key]
                        logging.info("Extracted matrice from file {} from key {}".format(filename, key))
                        continue

                dcp.add_data(d, sampling_time)

            dcp.run()
            self.atlas = dcp.atlas

            return dcp

        def _filter_spectre():

            logging.info("Filtering Spectre data")

            # Filter data for Spectre
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

            logging.info("Filtering TimePlot data")

            df1 = pd.DataFrame({'Mode': self.dcp1.df['mode'], 'Activity': self.dcp1.df['activity'],
                                'Group': ['Group 1' for i in range(self.dcp1.df.shape[0])]}) \
                if self.dcp1 is not None else None
            df2 = pd.DataFrame({'Mode': self.dcp2.df['mode'], 'Activity': self.dcp2.df['activity'],
                                'Group': ['Group 2' for i in range(self.dcp2.df.shape[0])]}) \
                if self.dcp2 is not None else None

            return pd.concat([df1, df2])

        def _filter_radar():


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

            logging.info("Filtering Brain data for Mode {}".format(order))

            mode1 = self.dcp1.df.loc[order - 1][['intensity', 'conjugate']] if self.dcp1 is not None else None
            mode2 = self.dcp2.df.loc[order - 1][['intensity', 'conjugate']] if self.dcp2 is not None else None

            return self.atlas, mode1, mode2, order

        def _format_table(df):

            def _set_precision(number):
                if number == np.inf:
                    return 'inf'

                if type(number) != str:
                    number = str(number)
                splat = number.split('.')
                splat[1] = splat[1][:5] if len(splat[1]) > 5 else splat[1]
                return ".".join([splat[0], splat[1]])

            def _handle_complex(number):
                print("HELLO")
                splat = [str(number.real), str(number.imag)]
                set = [_set_precision(splat[0]), _set_precision(splat[1])]
                return "{0} +/- {1} j".format(set[0], set[1])

            def _format_list(p):
                f = _handle_complex if type(p[0]) == np.complex128 else _set_precision
                return [f(e) for e in p]

            df['value'] = _format_list(df['value'])
            df['damping_time'] = _format_list(df['damping_time'])
            df['period'] = _format_list(df['period'])

            return df

        self.app.run_server(debug=False, port=port, host=address)

    def _set_app_layout(self):

        logging.info("Setting Application Layout")

        return html.Div([
            html.Div(id="dummy", style=dict(display="none")),
            html.Div(children=[
                # SETTING CHOICE RADIO ROW
                html.Div([dbc.FormGroup([
                    # dbc.Label("Decomposition Setting", html_for="setting", className="col-4"),
                    html.Div(children=[
                        html.H4("Dynamic Mode Toolbox"),
                        html.P(id="description")],
                        className="ml-5 mt-2"),
                    dbc.Col(
                        dbc.RadioItems(
                            id="setting",
                            options=[
                                {"label": "Analysis", "value": 1},
                                {"label": "Comparison", "value": 2},
                                {"label": "Mode Matching", "value": 3}
                            ],
                        ), className="ml-5 mt-4")], row=True)],
                    className="col-5", style={'margin-top': '25px'}
                ),
                # FILE SELECTION + LOGGER
                html.Div(children=[
                    # file selection info
                    html.Div(children=[dbc.Card(
                                dbc.CardBody([
                                    html.H5("Selection", className="card-title"),
                                    html.Div(className="row", children=[
                                        html.Div(className="col-6", children=[
                                            dbc.FormGroup(
                                                [
                                                    dbc.Label("Sampling Time (seconds)", className="mt-2"),
                                                    dbc.Input(id="sampling-time", type="number", placeholder="0.72",
                                                              value=0.72, className="col-3"),
                                                    dbc.Label("Number of modes to plot", className="mt-2"),
                                                    dbc.Input(id="number-of-modes", type="number", placeholder="5",
                                                              value=5, className="col-3"),
                                                    dbc.Checklist(
                                                        options=[
                                                            {"label": "Plot imaginary values", "value": 1},
                                                        ],
                                                        value=[],
                                                        id="imag-setting",
                                                        switch=True, className="mt-2"
                                                    ),
                                                ]
                                            ),
                                        ]),
                                        html.Div(className="col-6", children=[
                                            dbc.Label(id="selected-files-group-1-t"),
                                            html.P(id="selected-files-group-1-p"),
                                            html.Div(id="animated-progress-1-div", className="mb-2", children=[
                                                dbc.Progress(value=80, id="animated-progress-1", striped=True,
                                                             animated="animated", style={'display': 'none'})
                                            ]),
                                            dbc.Label(id="selected-files-group-2-t"),
                                            html.P(id="selected-files-group-2-p"),
                                            html.Div(id="animated-progress-2-div", className="mb-2", children=[
                                                dbc.Progress(value=80, id="animated-progress-2", striped=True,
                                                             animated="animated", style={'display': 'none'})
                                            ]),
                                        ]),
                                    ]),
                                        dbc.Button("Run Decomposition", color="primary", id="run", className="mr-2"),
                                        dbc.Button("Reset", color="secondary", id="reset"),
                                        html.Div(id="message-alert", className="text-danger mt-2")
                                ]))],
                             id="file-selection-card",
                             className="col-12 mt-2 mb-2 mr-2 ml-2")
                ], className="col-7"),
            ], className="row"),
            # UPLOAD ROW
            html.Div([
                # UPLOAD 1
                html.Div([dcc.Upload(
                    id='upload-1',
                    # Allow multiple files to be uploaded
                    multiple=True)],
                    className="col-6", id='upload-1-div',
                ),
                # UPLOAD 2
                html.Div([dcc.Upload(
                    id='upload-2',
                    # Allow multiple files to be uploaded
                    multiple=True)],
                    className="col-6")
            ], className="row", id="upload-row"),
            # TABS
            dbc.Tabs(
                [
                    ## PLOTS
                    dbc.Tab(
                        html.Div(children=[
                            # left panel - radar, spectre, temporal information
                            html.Div(className="col-5", children=[
                                # radar plots
                                html.Div(className="row", children=[
                                    html.Div(
                                        [dcc.Graph(id="radar",
                                                   config={"toImageButtonOptions": {"width": None,
                                                                                    "height": None,
                                                                                    "format": "svg",
                                                                                    "filename": "radar"},
                                                           "displaylogo": False})],
                                         className="col-12")
                                ]),
                                # spectre
                                html.Div(className="row", children=[
                                    html.Div(
                                        [dcc.Graph(id="spectre",
                                                   config={"toImageButtonOptions": {"width": None,
                                                                                    "height": None,
                                                                                    "format": "svg",
                                                                                    "filename": "spectre"},
                                                           "displaylogo": False})],
                                        className="col-12")
                                ]),
                                # timeplot
                                html.Div(className="row", children=[
                                    html.Div(
                                        [dcc.Graph(id="timeplot",
                                                   config={"toImageButtonOptions": {"width": None,
                                                                                    "height": None,
                                                                                    "format": "svg",
                                                                                    "filename": "timeplot"},
                                                           "displaylogo": False})],
                                        className="col-12")
                                ]),
                            ]),
                            # right panel - brains
                            # brains
                            html.Div(className="col-7", children=[
                                html.Div(className="col-12", id="brains"),
                                # progress
                                html.Div(
                                    [
                                        html.P('Loading cortical surface graphs...', className="mt-4"),
                                        dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
                                        dbc.Progress(id="progress", style={'width': '70%', 'align':'center'}),
                                    ], className="col-12 my-4 mx-4", id="progress-div")
                            ])
                        ], className="row"),
                        label="Graphs"),

                    ## TABLE 1
                    dbc.Tab(label="Group A", id="table-1-tab"),

                    ## TABLE 2
                    dbc.Tab(label="Group B", disabled=True, id="table-2-tab"),

                    ## LOG
                    dbc.Tab(
                        html.Div(children=[
                            dcc.Interval(
                                id='log-update',
                                interval=1000  # in milliseconds
                            ),
                            html.Div(children=[html.P("———— APP START ————")], id='log')],
                            className="col-6", id="log-div"),
                        label="Log", id='log-tab')
                ]
            ),
        ], id="app-layout")
