import numpy as np
import logging
import threading
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
from dash.dependencies import (Input, Output)
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from utils import *

class Dashboard(QWebEngineView):

    def __init__(self, dcp):
        """
        Constructor.

        :param dcp: Decomposition
        """
        # the figure is now accessible via self.figure
        QWebEngineView.__init__(self)

        self.decomposition = dcp

        address = {'address': '127.0.0.1',
                   'port': 8000
        }

        threading.Thread(target=self.run_dash, daemon=True).start()

        self.load(QUrl("http://{0}:{1}".format(address['address'], address['port'])))

    def run_dash(self, address='127.0.0.1', port=8000):
        """
        Run Dash

        :param address: address (str)
        :param port: port (int)
        """

        fig_radar_real, fig_radar_imag = self._create_radar()
        fig_spectre = self._create_spectre()
        modes_df = self.decomposition.modes_df

        app = dash.Dash(
            external_stylesheets=[dbc.themes.YETI]
        )

        app.layout = html.Div(className="container", style={'max-width': 1920}, children=[
            # main row
            html.Div(className="row", children=[
                # left panel - radar, spectre, temporal information
                html.Div(className="col-5", children=[
                    # radar plots
                    html.Div(className="row", children=[
                        html.Div(
                            [dcc.Graph(id="radar-real", figure=fig_radar_real)],
                             className="col-6"),
                        html.Div(
                            [dcc.Graph(id="radar-imag", figure=fig_radar_imag)],
                            className="col-6"
                        )
                    ]),
                    # spectre and temporal informations
                    html.Div(
                        [dcc.Graph(id="spectre", figure=fig_spectre)],
                        className="row"
                    ),
                    html.Div(
                        [DataTable(
                            id="name",
                            columns=[{"name": i, "id": i} for i in modes_df.columns],
                            data=modes_df.to_dict('records'),
                            style_table={
                                'overflowY': 'scroll'
                            }
                        )],
                        className="table", style={'height': '300px'}
                    )
                ]),
                # right panel - brain views
                html.Div(className="col-7", children=[
                    # brains
                    html.Div(className="col-10", children=[
                        html.Div([dcc.Graph(id="mode1",
                                            figure=self._create_brain(self.decomposition.modes[0]))],
                                 style={'height': '300px'}),
                        html.Div([dcc.Graph(id="mode2",
                                            figure=self._create_brain(self.decomposition.modes[1]))],
                                 style={'height': '300px'}),
                        html.Div([dcc.Graph(id="mode3",
                                            figure=self._create_brain(self.decomposition.modes[2]))],
                                 style={'height': '300px'})
                    ])
                ])
            ])
        ])

        app.run_server(debug=False, port=port, host=address)

    def _create_radar(self):
        """
        Create radar plot.

        :return:
        """

        labels = [ATLAS['networks'][self.decomposition.atlas][network]['name'] for network in
                  ATLAS['networks'][self.decomposition.atlas]]
        idx = [ATLAS['networks'][self.decomposition.atlas][network]['index'] for network in
               ATLAS['networks'][self.decomposition.atlas]]
        # Global Variables contain MATLAB (1->) vs. Python (0->) indices
        index = [np.add(np.asarray(idx[i]), -1) for i in range(len(idx))]

        df_radar = pd.DataFrame(columns=['mode', 'network', 'strength', 'complex'])

        for axis in ['real', 'imag']:

            to_axis = (np.real if axis == 'real' else np.imag)

            for mode in range(4):

                for n, network in enumerate(labels):

                    strength = np.mean(np.abs(to_axis(self.decomposition.modes[mode].intensity[index[n]])))

                    df_radar = pd.concat([df_radar,
                                          pd.DataFrame({'Mode': ['Mode {}'.format(mode + 1)],
                                                        'Network': [network],
                                                        'Strength': [strength],
                                                        'complex': [axis]}
                                                       )
                                          ]
                                         )

        fig_radar_real = px.line_polar(df_radar[df_radar['complex'] == 'real'],
                                       r="Strength",
                                       theta="Network",
                                       color="Mode",
                                       line_close=True,
                                       title="Real Part",
                                       color_discrete_sequence=px.colors.diverging.Portland)

        fig_radar_imag = px.line_polar(df_radar[df_radar['complex'] == 'imag'],
                                       r="Strength",
                                       theta="Network",
                                       color="Mode",
                                       line_close=True,
                                       title="Imaginary Part",
                                       color_discrete_sequence=px.colors.diverging.Portland)

        fig_radar_real.update_layout(legend_orientation="h")
        fig_radar_imag.update_layout(legend_orientation="h")

        return fig_radar_real, fig_radar_imag

    def _create_spectre(self):
        """
        Create spectre plot.

        :return:
        """

        df_spectre = pd.DataFrame(
            data={'Mode': range(len(self.decomposition.modes)),
                  'Absolute Value of Eigenvalue': [np.abs(mode.value) for mode in self.decomposition.modes]
                  }
        )

        return px.line(df_spectre, x="Mode", y="Absolute Value of Eigenvalue", title="Spectre")

    def _create_brain(self, mode):
        """
        Create 2D brain plot.

        :param mode: Mode instance
        :return:
        """

        # Load atlas 2D pandas data frame
        atlas = ATLAS2D[self.decomposition.atlas]

        if not mode.is_complex_conjugate:
            rows = [1]
            # normalize from -0.1 -> 0.1 in modes to 0.0 -> 1.0 for colormap
            # real valued eigenvalue carry imaginary valued eigenvectors
            intensity = [5 * np.imag(mode.intensity) + 0.5]
        else:
            rows = [1, 2]
            # normalize from -0.1 -> 0.1 in modes to 0.0 -> 1.0 for colormap
            intensity = list(map(lambda s: 5 * s + 0.5, [np.real(mode.intensity), np.imag(mode.intensity)]))

        fig = make_subplots(rows=len(rows), cols=1,
                            shared_xaxes=True, specs=[[{}], [{}]] if len(rows) == 2 else [[{}]],
                            horizontal_spacing=0.05,
                            vertical_spacing=0.15)

        for row in rows:

            for id in range(1, list(atlas['.id'])[-1] + 1):
                roi = atlas[atlas['.id'] == id]
                x = roi['.long']
                y = roi['.lat']
                region = roi['region'].iloc[0] if roi['region'].size != 0 else None

                if type(region) == str:

                    indice = roi['.roi'].iloc[0] - 1

                    if roi['hemi'].iloc[0] == 'right':
                        indice += int(intensity[0].shape[0] / 2)

                    val = intensity[row - 1][indice]

                    col = 'rgb({0},{1},{2})'.format(*cm.coolwarm(val)[:3])
                else:
                    col = 'black'

                fig.add_trace(go.Scatter(x=x,
                                         y=y,
                                         fill='toself',
                                         mode='lines',
                                         line=dict(color='black', width=0.5),
                                         fillcolor=col,
                                         name=region if region else None,
                                         hoverinfo=None),
                              row=row, col=1)

            fig.add_trace(go.Scatter(x=[300, 1100], y=[-25, -25],
                                     text=['left', 'right'],
                                     mode='text'),
                          row=row, col=1)

            fig.update_xaxes(showgrid=False, showline=False, visible=False, ticks='',
                             showticklabels=False, zeroline=False, showspikes=False, constrain='domain')
            fig.update_yaxes(showgrid=False, showline=False, visible=True, ticks='',
                             title='Real' if row == 1 else 'Imaginary',
                             showticklabels=False, zeroline=False, showspikes=False, constrain='domain',
                             scaleanchor='x', scaleratio=1,
                             row=row, col=1)

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          showlegend=False, title_text='Mode {}'.format(mode.order))

        return fig

    def _get_cbar(self):
        """
        Get colorbar Figure.

        :return:
        """
        cbar = go.Figure(data=[go.Mesh3d(x=[1, 0, 0],
                                         y=[0, 1, 0],
                                         z=[0, 0, 1],
                                         i=[0], j=[1], k=[2],
                                         intensity=[0.],
                                         opacity=0,
                                         colorbar={'tickfont': {'size': 14}, 'len': .1},
                                         colorscale=self._mpl_to_plotly(cm.coolwarm, 255))])

        axis_config_x = {
            'visible': False,
            'showgrid': False,
            'showline': False,
            'ticks': '',
            'title': '',
            'showticklabels': False,
            'zeroline': False,
            'showspikes': False,
            'constrain': 'domain'
        }

        cbar.update_layout(scene={'xaxis': self.axisConfig,
                                  'yaxis': self.axisConfig,
                                  'zaxis': self.axisConfig,
                                  'bgcolor': '#fff'})

        return cbar

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