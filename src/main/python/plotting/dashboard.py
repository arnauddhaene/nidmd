import numpy as np
import logging
import sys
import threading
from PyQt5 import QtWidgets
import dash
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication


class Dashboard(QWebEngineView):

    def __init__(self, df_radar, df_spectre):
        """
        Constructor.

        :param df_radar: pandas dataframe with columns=['mode','network','strength','complex']
        :param df_spectre: pandas dataframe with columns=['mode','value']
        """
        # the figure is now accessible via self.figure
        QWebEngineView.__init__(self)

        self.df_radar = df_radar
        self.df_spectre = df_spectre

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

        fig_radar_real = px.line_polar(self.df_radar[self.df_radar['complex'] == 'real'],
                                       r="Strength",
                                       theta="Network",
                                       color="Mode",
                                       line_close=True,
                                       title="Real Part",
                                       color_discrete_sequence=px.colors.diverging.Portland)

        fig_radar_imag = px.line_polar(self.df_radar[self.df_radar['complex'] == 'imag'],
                                       r="Strength",
                                       theta="Network",
                                       color="Mode",
                                       line_close=True,
                                       title="Imaginary Part",
                                       color_discrete_sequence=px.colors.diverging.Portland)

        fig_radar_imag.update_layout(legend_orientation="h")

        fig_spectre = px.line(self.df_spectre,
                              x="Mode",
                              y="Absolute Value of Eigenvalue",
                              title="Spectre")

        app = dash.Dash()
        app.layout = html.Div(children=[
            html.H4(children='Dynamic Mode Network Dashboard'),
            # html.Div(children='''
            #     Use the toggle below to switch between real and complex values.
            # '''),
            # html.Div([
            #     dcc.RadioItems(
            #         id='type',
            #         options=[{'label': i, 'value': i} for i in ['Real', 'Complex']],
            #         value='Real',
            #         labelStyle={'display': 'inline-block'}
            #     )
            # ], style={'float': 'left', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='radar-chart-real',
                    figure=fig_radar_real
                )
            ], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([
                dcc.Graph(
                    id='radar-chart-imag',
                    figure=fig_radar_imag
                )
            ], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([
                dcc.Graph(
                    id='spectre-plot',
                    figure=fig_spectre
                )
            ], style={'display': 'inline-block', 'width': '49%'})
        ])

        # app callback for radio items
        #
        # @app.callback(
        #     dash.dependencies.Output('radar-chart', 'figure'),
        #     [dash.dependencies.Input('type', 'value')])
        # def update_graph(typetype):
        #
        #     categories = ['Dorsal Attention', 'Default', 'Visual', 'Ventral Attention',
        #                   'Somatomotor', 'Limbic', 'Frontoparietal',
        #                   'Dorsal Attention']  # add last value do close circle
        #
        #     m1 = list(np.random.normal(5, 3, 7))
        #     m1.append(m1[0])
        #     m2 = list(np.random.normal(5, 3, 7))
        #     m2.append(m2[0])
        #
        #     fig = go.Figure()
        #
        #     if typetype == 'Real':
        #
        #         fig.add_trace(go.Scatterpolar(
        #             name="Mode 1",
        #             r=m1,  # add last value to close circle !!!
        #             theta=categories,
        #         ))
        #         fig.add_trace(go.Scatterpolar(
        #             name="Mode 2",
        #             r=m2,  # add last value to close circle !!!
        #             theta=categories,
        #         ))
        #
        #     else:
        #         fig.add_trace(go.Scatterpolar(
        #             name="Mode 1",
        #             r=m1,
        #             theta=categories,
        #         ))
        #         fig.add_trace(go.Scatterpolar(
        #             name="Mode 2",
        #             r=m2,  # add last value to close circle !!!
        #             theta=categories,
        #         ))
        #
        #     fig.update_traces(fill='toself')
        #
        #     return fig

        app.run_server(debug=False, port=port, host=address)