# This Python file uses the following encoding: utf-8
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils import *


class Radar:

    def __init__(self, df, networks):
        """
        Constructor.

        :param df: [pd.DataFrame] containing 'mode', 'group', 'strength_real', 'strength_imaginary'
        """

        self.df = df
        self.networks = networks

    def figure(self, imag=False, amount=6):
        """
        Get Figure

        :param imag: [boolean] add imaginary values
        :param amount: [int] number of modes to plot
        :return: [go.Figure]
        """

        fig = make_subplots(rows=1, cols=2 if imag else 1, specs=[[{'type': 'polar'}] * (2 if imag else 1)],
                            subplot_titles=("Real", "Imaginary"))
        colors = COLORS

        for mode in range(1, amount):

            for group in list(np.unique(self.df.group)):

                self._add_trace(fig, mode, group, colors[mode], imag)

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

        fig.add_trace(go.Scatterpolar(
            r=self._close(r_r), theta=self._close(self.networks), mode="lines",
            legendgroup='Mode {0} Group {1}'.format(mode, group), showlegend=True,
            name='Mode {0} Group {1}'.format(mode, group), subplot="polar1",
            line=dict(color=color, dash='dash' if group != 1 else None)
        ), row=1, col=1)

        if imag:
            fig.add_trace(go.Scatterpolar(
                r=self._close(r_i), theta=self._close(self.networks), mode="lines",
                legendgroup='Mode {0} Group {1}'.format(mode, group), showlegend=False,
                name='Mode {0} Group {1}'.format(mode, group), subplot="polar2",
                line=dict(color=color, dash='dash' if group != 1 else None),
            ), row=1, col=2)

    @staticmethod
    def _close(li):
        assert isinstance(li, list)
        li.append(li[0])
        return li
