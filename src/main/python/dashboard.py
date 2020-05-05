import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import io
import time
import os
import sys
import base64
import datetime
import threading
import scipy.io as sio
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from nibabel.freesurfer.io import (read_annot, write_annot)
import dash
from dash_table import DataTable
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from flask_caching import Cache
from dash.dependencies import (Input, Output, State)
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
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

        address = {'address': '127.0.0.1',
                   'port': 8000
        }

        self.df1 = None
        self.df2 = None
        self.atlas = None
        self.progress = 0

        self.logfilename = TARGET_DIR.joinpath('log.log').as_posix()
        handler = RotatingFileHandler(self.logfilename, maxBytes=10000, backupCount=1)
        self.logfile = open(TARGET_DIR.joinpath(self.logfilename).as_posix(), 'r')
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        self.app = dash.Dash(
            external_stylesheets=[dbc.themes.FLATLY]
        )

        self.app.server.logger.addHandler(handler)

        self._set_app_layout()

        threading.Thread(target=self.run_dash, daemon=True).start()
        # self.run_dash()

        self.load(QUrl("http://{0}:{1}".format(address['address'], address['port'])))

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

                console.append(html.P(self.logfile.read()))
                return console

            else:
                return None

        @self.app.callback([
            Output('upload-1-div', 'className'),
            Output('upload-1', 'style'),
            Output('upload-2', 'style'),
            Output('selected-files-group-2-t', 'style'),
            Output('selected-files-group-2-p', 'style'),
        ], [
            Input('setting', 'value')
        ])
        def input_setting(value):

            upload = {
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            }

            if value is None:
                return "row", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
            elif value == 1:
                return "col-12", upload, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
            elif value == 2:
                return "col-6", upload, upload, {}, {}

        @self.app.callback([
            Output('selected-files-group-1-p', 'children'),
            Output('selected-files-group-2-p', 'children'),
            Output('table-1-tab', 'children'),
            Output('table-2-tab', 'children'),
            Output('table-2-tab', 'disabled')
        ], [
            Input('upload-1', 'contents'),
            Input('upload-2', 'contents')
        ], [
            State('upload-1', 'filename'),
            State('upload-2', 'filename')
        ])
        def upload(contents1, contents2, names1, names2):

            df1 = None
            df2 = None
            tab1 = None
            tab2 = None
            disabled = True

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
                     'width': '10%'},
                    {'if': {'column_id': 'Value'},
                     'width': '30%'},
                    {'if': {'column_id': 'Damping Time'},
                     'width': '40%'},
                    {'if': {'column_id': 'Period'},
                     'width': '20%'},
                ],
                style_as_list_view=True
            )

            if [contents1, contents2, names1, names2].count(None) == 4:
                raise PreventUpdate
            else:
                if contents1 is not None:
                    self.df1 = _parse_files(contents1, names1)
                    df1 = self.df1[['mode', 'value', 'damping_time', 'period']]
                    df1['value'] = [str(value) for value in list(df1['value'])]
                    data1 = df1.to_dict('records') if df1 is not None else dict()
                    columns1 = [{"name": i, "id": i} for i in df1.columns] if df1 is not None else [
                        {"name": "none", "id": "none"}]
                    tab1 = html.Div(DataTable(
                        id="table-1", data=data1, columns=columns1, **table_config
                    ), className="container") if self.df1 is not None else None
                if contents2 is not None:
                    self.df2 = _parse_files(contents2, names2)
                    df2 = self.df2[['mode', 'value', 'damping_time', 'period']]
                    df2['value'] = [str(value) for value in list(df2['value'])]
                    data2 = df2.to_dict('records') if df2 is not None else dict()
                    columns2 = [{"name": i, "id": i} for i in df2.columns] if df2 is not None else [
                        {"name": "none", "id": "none"}]
                    tab2 = html.Div(DataTable(
                        id="table-2", data=data2, columns=columns2, **table_config
                    ), className="container") if self.df2 is not None else None
                    disabled = False

            return names1, names2, tab1, tab2, disabled

        @self.app.callback(
            Output('spectre', 'figure')
        ,[
            Input('run', 'n_clicks')
        ])
        def compute_spectre(n):

            if n is None or self.atlas is None:
                raise PreventUpdate
            else:

                self.app.logger.info("Computing spectre of dynamical modes")

                s = Spectre(_filter_spectre())

                return s.figure()

        @self.app.callback(
            Output('timeplot', 'figure')
            , [
                Input('run', 'n_clicks')
            ])
        def compute_timeplot(n):

            if n is None or self.atlas is None:
                raise PreventUpdate
            else:

                self.app.logger.info("Computing time series activation of dominant modes")

                t = TimePlot(_filter_time())

                return t.figure()

        @self.app.callback(
            Output('radar', 'figure')
        , [
            Input('run', 'n_clicks')
        ])
        def compute_radar(n):

            if n is None or self.atlas is None:
                raise PreventUpdate
            else:

                self.app.logger.info("Computing cortical network activation")

                r = Radar(*_filter_radar())

                return r.figure()

        @self.app.callback([
            Output('brains', 'children'),
            Output('progress-div', 'style')
        ], [
            Input('run', 'n_clicks')
        ])
        def compute_brain(n):

            if n is None or self.atlas is None:
                raise PreventUpdate
            else:

                self.app.logger.info("Computing cortical surface representations")

                brains = []

                self.progress += 10

                for mode in range(1, 4):

                    b = Brain(*_filter_brain(mode))

                    brains.append(html.Div([dcc.Graph(figure=b.figure())]))

                    self.progress += 30

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
            return progress, f"{progress} %" if progress >= 5 else ""

        def _parse_files(contents, files):
            """
            Parse incoming .mat files.

            :param contents: list of Base64 encoded contents
            :param files: list of names
            """

            self.app.logger.info("Parsing {} files".format(len(files)))

            dcp = Decomposition()

            for content, name in zip(contents, files):
                type, string = content.split(',')

                mat = io.BytesIO(base64.b64decode(string))

                data = sio.loadmat(mat)

                if 'TCSnf' in data.keys():
                    data = data['TCSnf']
                else:
                    data = data['TCS']

                dcp.add_data(data)

            dcp.run()
            self.atlas = dcp.atlas

            return dcp.df

        def _filter_spectre():

            self.app.logger.info("Filtering Spectre data")

            # Filter data for Spectre
            df1 = pd.DataFrame({'Mode': self.df1['mode'], 'Value': np.abs(self.df1['value']),
                                'Group': ['Group 1' for i in range(self.df1.shape[0])]}) \
                if self.df1 is not None else None
            df2 = pd.DataFrame({'Mode': self.df2['mode'], 'Value': np.abs(self.df2['value']),
                                'Group': ['Group 2' for i in range(self.df2.shape[0])]}) \
                if self.df2 is not None else None

            return pd.concat([df1, df2])

        def _filter_time():

            self.app.logger.info("Filtering TimePlot data")

            df1 = pd.DataFrame({'Mode': self.df1['mode'], 'Activity': self.df1['activity'],
                                'Group': ['Group 1' for i in range(self.df1.shape[0])]}) \
                if self.df1 is not None else None
            df2 = pd.DataFrame({'Mode': self.df2['mode'], 'Activity': self.df2['activity'],
                                'Group': ['Group 2' for i in range(self.df2.shape[0])]}) \
                if self.df2 is not None else None

            return pd.concat([df1, df2])

        def _filter_radar():


            self.app.logger.info("Filtering Radar data")

            df1 = pd.DataFrame({'mode': self.df1['mode'], 'group': [1 for i in range(self.df1.shape[0])],
                                'strength_real': self.df1['strength_real'], 'strength_imag': self.df1['strength_imag']}) \
                if self.df1 is not None else None
            df2 = pd.DataFrame({'mode': self.df2['mode'], 'group': [2 for i in range(self.df2.shape[0])],
                                'strength_real': self.df2['strength_real'], 'strength_imag': self.df2['strength_imag']}) \
                if self.df2 is not None else None

            networks = self.df1['networks'][0] if self.df1 is not None else None
            if networks is None:
                networks = self.df2['networks'][0] if self.df2 is not None else None

            return pd.concat([df1, df2]), networks

        def _filter_brain(order):

            self.app.logger.info("Filtering Brain data for Mode {}".format(order))

            mode1 = self.df1.loc[order - 1][['intensity', 'conjugate']] if self.df1 is not None else None
            mode2 = self.df2.loc[order - 1][['intensity', 'conjugate']] if self.df2 is not None else None

            return self.atlas, mode1, mode2, order

        self.app.run_server(debug=False, port=port, host=address)

    def _set_app_layout(self):

        self.app.logger.info("Setting Application Layout")

        self.app.layout = html.Div([
            # SETTING CHOICE RADIO ROW
            html.Div([dbc.FormGroup([
                dbc.Label("Decomposition Setting", html_for="setting", className="col-4"),
                dbc.Col(
                    dbc.RadioItems(
                        id="setting",
                        options=[
                            {"label": "Analysis", "value": 1},
                            {"label": "Comparision", "value": 2},
                        ],
                    ), className="col-8")], row=True)],
                className="col-12", style={'margin-top': '25px'}
            ),
            # UPLOAD ROW
            html.Div([
                # UPLOAD 1
                html.Div([dcc.Upload(
                    id='upload-1',
                    children=html.Div([
                        'Group 1: Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=True)],
                    className="col-6", id='upload-1-div',
                ),
                # UPLOAD 2
                html.Div([dcc.Upload(
                    id='upload-2',
                    children=html.Div([
                        'Group 2: Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=True)],
                    className="col-6")
                ], className="row"),
            # FILE SELECTION + LOGGER
            html.Div(children=[
                # file selection info
                html.Div(children=[dbc.Card(
                            dbc.CardBody([
                                    html.H5("Selected Files", className="card-title"),
                                    html.H6("Group 1"),
                                    html.P(id="selected-files-group-1-p"),
                                    html.H6("Group 2", id="selected-files-group-2-t"),
                                    html.P(id="selected-files-group-2-p"),
                                    dbc.Button("Run Decomposition", color="primary", id="run")
                            ]))],
                         id="file-selection-card",
                         className="col-6"),
                # logger
                html.Div(children=[
                            dcc.Interval(
                                id='log-update',
                                interval=1 * 1000  # in milliseconds
                            ),
                            html.Div(children=[html.P("———— APP START ————")], id='log')],
                         className="col-6", id="log-div", style={"maxHeight": "200px", "overflow": "scroll"})
            ], className="row"),
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
                                        [dcc.Graph(id="radar")],
                                         className="col-12")
                                ]),
                                # spectre
                                html.Div(className="row", children=[
                                    html.Div(
                                        [dcc.Graph(id="spectre")],
                                        className="col-12")
                                ]),
                                # timeplot
                                html.Div(className="row", children=[
                                    html.Div(
                                        [dcc.Graph(id="timeplot")],
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
                                    ], className="col-12", id="progress-div")
                            ])
                        ], className="row"),
                        label="Graphs"),

                    ## TABLE 1
                    dbc.Tab(label="Group 1", id="table-1-tab"),

                    ## TABLE 2
                    dbc.Tab(label="Group 2", disabled=True, id="table-2-tab")]
            ),
        ])
