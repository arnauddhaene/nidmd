import numpy as np
import logging
import io
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

        address = {'address': '127.0.0.1',
                   'port': 8000
        }

        self.df1 = None
        self.df2 = None
        self.atlas = None
        self.progress = 0

        self.app = dash.Dash(
            external_stylesheets=[dbc.themes.FLATLY]
        )

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

        # cache = Cache(self.app.server, config={
        #     'CACHE_TYPE': 'redis',
        #     # Note that filesystem cache doesn't work on systems with ephemeral
        #     # filesystems like Heroku.
        #     'CACHE_TYPE': 'filesystem',
        #     'CACHE_DIR': 'cache-directory',
        #
        #     # should be equal to maximum number of users on the app at a single time
        #     # higher numbers will store more data in the filesystem / redis cache
        #     'CACHE_THRESHOLD': 200
        # })
        #
        # def get_dataframe(session_id, decompositions):
        #     @cache.memoize()
        #     def query_and_serialize_data(session_id, decompositions):
        #         # expensive or user/session-unique data processing step goes here
        #
        #
        #
        #         # simulate an expensive data processing task by sleeping
        #         time.sleep(5)
        #
        #
        #
        #         return df.to_json()
        #
        #     return pd.read_json(query_and_serialize_data(session_id))

        @self.app.callback([
            Output('upload-1-div', 'className'),
            Output('upload-2', 'style'),
            Output('selected-files-group-2-t', 'style'),
            Output('selected-files-group-2-p', 'style'),
        ], [
            Input('setting', 'value')
        ])
        def input_setting(value):

            if value is None:
                raise PreventUpdate
            elif value == 1:
                return "col-12", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
            elif value == 2:
                return "col-6", {
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                }, {}, {}

        @self.app.callback([
            Output('selected-files-group-1-p', 'children'),
            Output('selected-files-group-2-p', 'children')
        ], [
            Input('upload-1', 'contents'),
            Input('upload-2', 'contents')
        ], [
            State('upload-1', 'filename'),
            State('upload-2', 'filename')
        ])
        def upload(contents1, contents2, names1, names2):

            if [contents1, contents2, names1, names2].count(None) == 4:
                raise PreventUpdate
            else:
                if contents1 is not None:
                    self.df1 = _parse_files(contents1, names1)
                if contents2 is not None:
                    self.df2 = _parse_files(contents2, names2)

            return names1, names2

        @self.app.callback(
            Output('spectre', 'figure')
        ,[
            Input('run', 'n_clicks')
        ])
        def compute_spectre(n):

            if n is None or self.atlas is None:
                raise PreventUpdate
            else:

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

            logging.info('_parse_files called')
            print('_parse_files called')

            dcp = Decomposition()

            for content, name in zip(contents, files):
                type, string = content.split(',')

                mat = io.BytesIO(base64.b64decode(string))

                data = sio.loadmat(mat)['TCSnf']

                dcp.add_data(data)

            dcp.run()
            self.atlas = dcp.atlas

            return dcp.df

        def _filter_spectre():
            # Filter data for Spectre
            df1 = pd.DataFrame({'Mode': self.df1['mode'], 'Value': np.abs(self.df1['value']),
                                'Group': ['Group 1' for i in range(self.df1.shape[0])]}) \
                if self.df1 is not None else None
            df2 = pd.DataFrame({'Mode': self.df2['mode'], 'Value': np.abs(self.df2['value']),
                                'Group': ['Group 2' for i in range(self.df2.shape[0])]}) \
                if self.df2 is not None else None

            return pd.concat([df1, df2])

        def _filter_time():

            df1 = pd.DataFrame({'Mode': self.df1['mode'], 'Activity': self.df1['activity'],
                                'Group': ['Group 1' for i in range(self.df1.shape[0])]}) \
                if self.df1 is not None else None
            df2 = pd.DataFrame({'Mode': self.df2['mode'], 'Activity': self.df2['activity'],
                                'Group': ['Group 2' for i in range(self.df2.shape[0])]}) \
                if self.df2 is not None else None

            return pd.concat([df1, df2])

        def _filter_radar():

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

            mode1 = self.df1.loc[order - 1][['intensity', 'conjugate']] if self.df1 is not None else None
            mode2 = self.df2.loc[order - 1][['intensity', 'conjugate']] if self.df2 is not None else None

            return self.atlas, mode1, mode2, order

        # if self.dcp:
                #
                #     self.dcp.run()
                #
                #     fig_radar_real, fig_radar_imag = self._create_radar()
                #     fig_spectre = self._create_spectre()
                #     modes_df = self.dcp.modes_df
                #
                #     children = [
                #         html.Div(className="container",
                #                  # style={'max-width': 1920},
                #                  children=[
                #             # main row
                #             html.Div(className="row", children=[
                #                 # left panel - radar, spectre, temporal information
                #                 html.Div(className="col-5", children=[
                #                     # radar plots
                #                     html.Div(className="row", children=[
                #                         html.Div(
                #                             [dcc.Graph(id="radar-real", figure=fig_radar_real)],
                #                              className="col-6"),
                #                         html.Div(
                #                             [dcc.Graph(id="radar-imag", figure=fig_radar_imag)],
                #                             className="col-6"
                #                         )
                #                     ]),
                #                     # spectre and table
                #                     html.Div([dcc.Graph(id="spectre", figure=fig_spectre)], className="row"),
                #                     html.Div(DataTable(
                #                         id="name",
                #                         columns=[{"name": i, "id": i} for i in modes_df.columns],
                #                         data=modes_df.to_dict('records'),
                #                         fixed_rows={'headers': True, 'data': 0},
                #                         style_cell={'padding': '5px'},
                #                         style_header={
                #                             'backgroundColor': 'white',
                #                             'fontWeight': 'bold',
                #                             'font-family': 'Helvetica',
                #                             'padding': '0px 5px'
                #                         },
                #                         style_data={'font-family': 'Helvetica'},
                #                         style_data_conditional=[
                #                             {
                #                                 'if': {'row_index': 'odd'},
                #                                 'backgroundColor': 'rgb(248, 248, 248)'
                #                             }
                #                         ],
                #                         style_cell_conditional=[
                #                             {
                #                                 'if': {'column_id': 'Mode'},
                #                                 'textAlign': 'left'
                #                             },
                #                             {'if': {'column_id': 'Mode'},
                #                              'width': '10%'},
                #                             {'if': {'column_id': 'Value'},
                #                              'width': '30%'},
                #                             {'if': {'column_id': 'Damping Time'},
                #                              'width': '40%'},
                #                             {'if': {'column_id': 'Period'},
                #                              'width': '20%'},
                #                         ],
                #                         style_as_list_view=True,
                #                         style_table={
                #                             'overflowY': 'scroll',
                #                             'maxHeight': '200px',
                #                             'width': '500px',
                #                             'align': 'center',
                #                             'margin': '25px'
                #                         }
                #                     ), className="row")
                #                 ]),
                #                 # right panel - brain views
                #                 html.Div(className="col-7", children=[
                #                     # brains
                #                     html.Div(className="col-10", children=[
                #                         html.Div([dcc.Graph(id="mode1",
                #                                             figure=self._create_brain(self.dcp.modes[0]))],
                #                                  style={'height': '300px'}),
                #                         html.Div([dcc.Graph(id="mode2",
                #                                             figure=self._create_brain(self.dcp.modes[1]))],
                #                                  style={'height': '300px'}),
                #                         html.Div([dcc.Graph(id="mode3",
                #                                             figure=self._create_brain(self.dcp.modes[2]))],
                #                                  style={'height': '300px'})
                #                     ])
                #                 ])
                #             ])
                #         ])
                #     ]

            # return children

        self.app.run_server(debug=False, port=port, host=address)

    def _set_app_layout(self):

        logging.info('_set_app_layout called')
        print('_set_app_layout called')

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
            # FILE SELECTION CARD
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
                     className="col-12"),
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

                # html.Div(
                #     [dcc.Graph(figure=self._get_cbar())],
                #     className="col-12", id="colorbar")
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
            ], className="row")
        ])

    def _get_cbar(self):
        """
        Get colorbar Figure.

        :return:
        """
        # cbar = go.Figure(data=[go.Mesh3d(x=[1, 0, 0],
        #                                  y=[0, 1, 0],
        #                                  z=[0, 0, 1],
        #                                  i=[0], j=[1], k=[2],
        #                                  intensity=[0.1],
        #                                  opacity=0,
        #                                  colorbar={'tickfont': {'size': 14}, 'len': 1},
        #                                  colorscale=self._mpl_to_plotly(cm.coolwarm, 255))])
        #
        axis_config = {
            'visible': False,
            'showgrid': False,
            'showline': False,
            'ticks': '',
            'title': '',
            'showticklabels': False,
            'zeroline': False,
            'showspikes': False
        }
        #
        # cbar.update_layout(scene={'xaxis': axis_config,
        #                           'yaxis': axis_config,
        #                           'zaxis': axis_config,
        #                           'bgcolor': '#fff'})

        a = [-0.1, 0.1]
        b = [-0.1, 0.1]
        c = [-0.1, 0.1]

        fig = go.Figure(go.Heatmap(
            z=[a, b, c],
            colorscale='RdBu',
            zmid=0))

        fig.layout['xaxis'].update(axis_config)
        fig.layout['yaxis'].update(axis_config)


        return fig

    @staticmethod
    def _mpl_to_plotly(cmap, pl_entries):
        h = 1.0 / (pl_entries - 1)
        pl_colorscale = []

        for k in range(pl_entries):
            C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
            pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

        return pl_colorscale

    @staticmethod
    def _color(x):
        return plt.cm.coolwarm(int(x))

    @staticmethod
    def _compute_vertexcolor(eigenvector, hemi):

        # TODO : make it work for Schaefer as well -> integrate a Dashboard having a Decomposition element
        atlas = read_annot(RES_DIR.joinpath(ATLAS['label']['glasser'][hemi]).as_posix())[0]

        vertexcolor = np.empty(atlas.shape[0], dtype=object)
        m = np.max(np.abs(eigenvector))

        lut = list(map(Dashboard._color,
                       list((255 / (2 * np.max(np.abs(eigenvector)))) * eigenvector + 127.5)))

        vertexcolor = [(0.0, 0.0, 0.0, 1.0) if atlas[vertex] == 0 else lut[atlas[vertex] - 1]
                       for vertex in range(atlas.shape[0])]

        return vertexcolor