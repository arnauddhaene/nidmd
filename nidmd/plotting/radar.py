# This Python file uses the following encoding: utf-8
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .colors import colorlist


class Radar:

    def __init__(self, df1, atlas, df2=None):
        """
        Constructor.

        :param df1: [pd.DataFrame] df gotten from Decomposition.df
        :param atlas: [dmd.datasets.Atlas] from Decomposition
        :param df2: [pd.DataFrame] df gotten from Decomposition.df
        """

        df1_ = df1.copy()
        df1_['group'] = [1] * df1_.shape[0]
        if df2 is not None:
            df2_ = df2.copy()
            df2_['group'] = [2] * df2_.shape[0]
            self.analysis = False
        else:
            df2_ = None
            self.analysis = True

        self.df = pd.concat([df1_, df2_])
        self.networks = list(atlas.networks.keys())

    def figure(self, imag=False, amount=6):
        """
        Get Figure

        :param imag: [boolean] add imaginary values
        :param amount: [int] number of modes to plot
        :return: [go.Figure]
        """

        fig = make_subplots(rows=1, cols=2 if imag else 1, specs=[[{'type': 'polar'}] * (2 if imag else 1)],
                            subplot_titles=("Real", "Imaginary"))

        for mode in range(1, amount):

            for group in list(np.unique(self.df.group)):

                self._add_trace(fig, mode, group, colorlist[mode], imag)

        fig.update_layout(polar1=dict(radialaxis_range=[0, 0.1]),
                          polar2=dict(radialaxis_range=[0, 0.1]) if imag else None,
                          legend_orientation="v")

        return fig

    def _add_trace(self, fig, mode, group, color, imag=False):
        """
        Adds adequate trace to figure

        :param fig: [go.Figure]
        :param mode: [int] mode (1, 2, ...)
        :param comp: [str] 'real' or 'imag'
        :param group: [int] group (1, 2, )
        :param color: [str] line color in css standards
        """

        r_r = self.df.loc[(self.df['mode'] == mode) & (self.df['group'] == group)].strength_real.to_list()[0]
        r_i = self.df.loc[(self.df['mode'] == mode) & (self.df['group'] == group)].strength_imag.to_list()[0]

        verbalise = 'Mode {}'.format(mode) + ' Group {}'.format(group) if not self.analysis else ''

        fig.add_trace(go.Scatterpolar(
            r=self._close(r_r), theta=self._close(self.networks), mode="lines",
            legendgroup=verbalise, showlegend=True, name=verbalise, subplot="polar1",
            line=dict(color=color, dash='dash' if group != 1 else None)
        ), row=1, col=1)

        if imag:
            fig.add_trace(go.Scatterpolar(
                r=self._close(r_i), theta=self._close(self.networks), mode="lines",
                legendgroup=verbalise, showlegend=False, name=verbalise, subplot="polar2",
                line=dict(color=color, dash='dash' if group != 1 else None),
            ), row=1, col=2)

    @staticmethod
    def _close(li):
        assert isinstance(li, list)
        li.append(li[0])
        return li
