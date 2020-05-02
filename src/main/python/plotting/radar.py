# This Python file uses the following encoding: utf-8
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import cm
from utils import *


class Radar:

    def __init__(self, df, networks):
        """


        :param df: dataframe containing 'mode', 'group', 'strength_real', 'strength_imaginary'
        """

        self.df = df
        self.networks = networks

    def figure(self, amount=4):
        """
        Get Figure

        :param amount: number of modes to plot
        :return: Plotly Figure instance
        """

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}] * 2],
                            subplot_titles=("Real", "Imaginary"))

        colors = COLORS

        for mode in range(1, amount + 1):

            for group in list(np.unique(self.df.group)):

                self._add_trace(fig, mode, group, colors[mode])

        fig.update_layout(polar1=dict(radialaxis_range=[0, 0.1]),
                          polar2=dict(radialaxis_range=[0, 0.1]),
                          legend_orientation="h")

        return fig

    def _add_trace(self, fig, mode, group, color):
        """
        Adds adequate trace to figure

        :param fig: Plotly Figure
        :param mode: mode (1, 2, ...)
        :param comp: 'real' or 'imag'
        :param group: group (1, 2, )
        :param color: line color in css standards
        """

        r_r = self.df.loc[(self.df['mode'] == mode) & (self.df['group'] == group)].strength_real.to_list()[0]
        r_i = self.df.loc[(self.df['mode'] == mode) & (self.df['group'] == group)].strength_imag.to_list()[0]

        fig.add_trace(go.Scatterpolar(
            r=self._close(r_r),
            theta=self._close(self.networks),
            mode="lines",
            legendgroup='Mode {0} Group {1}'.format(mode, group),
            name='Mode {0} Group {1}'.format(mode, group),
            showlegend=True,
            line=dict(color=color, dash='dash' if group != 1 else None),
            subplot="polar1"
        ), row=1, col=1)

        fig.add_trace(go.Scatterpolar(
            r=self._close(r_i),
            theta=self._close(self.networks),
            mode="lines",
            legendgroup='Mode {0} Group {1}'.format(mode, group),
            name='Mode {0} Group {1}'.format(mode, group),
            showlegend=False,
            line=dict(color=color, dash='dash' if group != 1 else None),
            subplot="polar2"
        ), row=1, col=2)

    @staticmethod
    def _close(l):
        l.append(l[0])
        return l
