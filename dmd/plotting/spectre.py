# This Python file uses the following encoding: utf-8
import plotly.express as px
import pandas as pd
import numpy as np


class Spectre:

    def __init__(self, df1: pd.DataFrame, groups, df2=None):
        """
        Constructor

        :param df1: [pd.DataFrame] should be Decomposition.df
        :param groups: [list of str]
        :param df2: [pd.DataFrame] should be Decomposition.df
        """
        df1_ = df1.copy()
        df1_['group'] = groups[0]
        df1_['value'] = np.abs(df1_['value'])
        if df2 is not None:
            df2_ = df2.copy()
            df2_['group'] = groups[1]
            df2_['value'] = np.abs(df2_['value'])
        else:
            df2_ = None

        self.df = pd.concat([df1_, df2_])

    def figure(self):
        """
        Get Figure

        :return: [go.Figure]
        """
        fig = px.line(self.df, x='mode', y='value', color='group', title='Spectre',
                      custom_data=['damping_time', 'period'])

        fig.update_traces(hovertemplate='Value: %{y:.2f}<br>' +
                                        'Damping Time: %{customdata[0]:.2f} s<br>' +
                                        'Period: %{customdata[1]:.2f} s' +
                                        '<extra></extra>')

        fig.update_layout(hovermode="x unified", legend_orientation="v", legend_title_text='',
                          xaxis_title='Mode',
                          yaxis_title='Value (absolute value of eigenvalue)')

        return fig

    @staticmethod
    def correlation(df):
        """
        Get linear trendline figure.

        :param df: [pd.DataFrame] with 'x', 'y' columns
        :return: [go.Figure]
        """
        return px.scatter(df, x="Approximated", y="Real", trendline="ols",
                         title="Reference Group Regression Approximation")
