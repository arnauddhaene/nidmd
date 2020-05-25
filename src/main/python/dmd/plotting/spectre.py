# This Python file uses the following encoding: utf-8
import plotly.express as px


class Spectre:

    def __init__(self, df):
        """
        Constructor

        :param df: [pd.DataFrame] containing 'Mode', 'Absolute Value of Eigenvalue', 'Group'
        """
        self.df = df

    def figure(self):
        """
        Get Figure

        :return: [go.Figure]
        """
        fig = px.line(self.df, x='Mode', y='Value', color='Group', title='Spectre')

        fig.update_traces(hovertemplate=None)
        fig.update_layout(hovermode="x unified", legend_orientation="v", legend_title_text='')

        return fig

    @staticmethod
    def correlation(df):
        """
        Get linear trendline figure.

        :param df: [pd.DataFrame] with 'x', 'y' columns
        :return: [go.Figure]
        """
        fig = px.scatter(df, x="Approximated", y="Real", trendline="ols", color="Group",
                         title="Reference Group Regression Approximation")

        fig.update_traces(hovertemplate=None)
        fig.update_layout(hovermode="x unified", legend_orientation="v", legend_title_text='', )

        return fig
