# This Python file uses the following encoding: utf-8
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

from .colors import colorlist


class TimePlot:

    def __init__(self, df1, df2=None):
        """
        Constructor

        :param df1: [pd.DataFrame] from Decomposition.df
        :param df2: [pd.DataFrame] from Decomposition.df
        """

        df1_ = df1.copy()
        df1_['group'] = ['Group 1'] * df1_.shape[0]
        if df2 is not None:
            df2_ = df2.copy()
            df2_['group'] = ['Group 2'] * df2_.shape[0]
        else:
            df2_ = None

        self.df = pd.concat([df1_, df2_])

    def figure(self, amount=6):
        """
        Get Figure

        :return: [go.Figure]
        """
        groups = np.unique(self.df['group']).shape[0]

        fig = make_subplots(rows=groups, cols=1,
                            subplot_titles=['Group 1', 'Group 2'] if groups == 2 else None,
                            shared_xaxes=True, vertical_spacing=0.08)

        for g in range(1, groups + 1):

            df = self.df[self.df['group'] == 'Group {}'.format(g)]

            for mode in range(1, amount):

                activity = df[df['mode'] == mode].activity.to_list()[0]

                fig.add_trace(go.Scatter(x=list(np.array(range(activity.shape[0]))), y=activity,
                                         legendgroup='Mode {}'.format(mode), showlegend=True if g == 1 else False,
                                         name='Mode {}'.format(mode),
                                         line=dict(color=colorlist[mode])),
                              row=g, col=1)

        fig.update_traces(hovertemplate=None)
        fig.update_yaxes(title_text='Activity', nticks=10)
        fig.layout['xaxis1' if groups == 1 else 'xaxis2'].update(dict(title_text='Time/Sampling Time (s)', nticks=10))

        fig.update_layout(hovermode="x unified", legend_orientation="v", height=150+groups*250)

        return fig
