# This Python file uses the following encoding: utf-8
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dmd.utils import *


class TimePlot:

    def __init__(self, df):
        """
        Constructor

        :param df: [pd.DataFrame] containing 'Mode', 'Activity', 'Group'
        """
        self.df = df

    def figure(self, amount=6):
        """
        Get Figure

        :return: [go.Figure]
        """
        groups = np.unique(self.df['Group']).shape[0]

        colors = COLORS

        fig = make_subplots(rows=groups, cols=1,
                            subplot_titles=['Group 1', 'Group 2'] if groups == 2 else None,
                            shared_xaxes=True, vertical_spacing=0.08)

        for group in range(1, groups + 1):

            df = self.df[self.df.Group == 'Group {}'.format(group)]

            for mode in range(1, amount):

                activity = df[df.Mode == mode].Activity.to_list()[0]

                fig.add_trace(go.Scatter(x=list(np.array(range(activity.shape[0]))), y=activity,
                                         legendgroup='Mode {}'.format(mode), showlegend=True if group == 1 else False,
                                         name='Mode {}'.format(mode),
                                         line=dict(color=colors[mode])),
                              row=group, col=1)

        fig.update_traces(hovertemplate=None)
        fig.update_yaxes(title_text='Activity', nticks=10)
        fig.layout['xaxis1' if groups == 1 else 'xaxis2'].update(dict(title_text='Time/Sampling Time (s)', nticks=10))

        fig.update_layout(hovermode="x unified", legend_orientation="v", height=150+groups*250)

        return fig
