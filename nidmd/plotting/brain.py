# This Python file uses the following encoding: utf-8
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.cm as cm

from .colors import colorscale

class Brain:

    def __init__(self, df1, order, coords_2d, df2=None):
        """
        Brain Constructor.

        Parameters
        ----------
        df1 : pd.DataFrame
            Pandas DataFrame containing at least :code:`intensity` and :code:`conjugate` columns.
            The :code:`intensity` column must contain Array-like values of the length of the cortical atlas.
        order : int
            Mode order
        coords_2d : pd.DataFrame
            Pandas DataFrame containing the 2D corticial parcellation coordinates. These can be fetched from
            Decomposition.atlas.coords_2d
        df2 : pd.DataFrame, optional
            Pandas DataFrame containing at least :code:`intensity` and :code:`conjugate` columns
            The :code:`intensity` column must contain Array-like values of the length of the cortical atlas.
        """
        self.coords_2d = coords_2d
        self.mode1 = df1.loc[order - 1][['intensity', 'conjugate']]
        if df2 is not None:
            self.mode2 = df2.loc[order - 1][['intensity', 'conjugate']]
        else:
            self.mode2 = None
        self.order = order

    @staticmethod
    def intensities(modes, imag=False):
        """
        Returns activity intensities of modes.

        Parameters
        ----------
        modes : pd.DataFrame
            Pandas DataFrame containing at least :code:`intensity` and :code:`conjugate` columns.
            The :code:`intensity` column must contain Array-like values of the length of the cortical atlas.
        imag : boolean, optional
            Retrieve the imaginary values from the activity intensities (default False)

        Returns
        -------
        rows : list
            list of range from 0 number of figure subplots
        intensity : list of Array-like
            Activity intensities for each mode
        """

        rows = []
        intensity = []

        for mode in modes:

            if not mode.conjugate or not imag:
                # extend rows list
                rows.append(rows[-1] + 1 if len(rows) != 0 else 1)
                # normalize from -0.1 -> 0.1 in modes to 0.0 -> 1.0 for colormap
                # real valued eigenvalue carry imaginary valued eigenvectors
                intensity.append(5 * np.real(mode.intensity) + 0.5)
            else:
                # extend rows list
                rows.extend(
                    list(
                        map(lambda x: rows[-1] + x, [1, 2])
                    ) if len(rows) != 0 else [1, 2]
                )
                # normalize from -0.1 -> 0.1 in modes to 0.0 -> 1.0 for colormap
                intensity.extend(
                    list(
                        map(lambda s: 5 * s + 0.5,
                            [np.real(mode.intensity), np.imag(mode.intensity)])
                    )
                )

        return rows, intensity

    def figure(self, imag=False, colormap='coolwarm'):
        """
        Returns Plotly Figure.

        Parameters
        ----------
        imag : boolean, optional
            Incorporate brain visualizations for imaginary activity values (default False)
        colormap : str
            Colormap supported by matplotlib, which can be found on the 
            `official reference <https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html>`_

        Returns
        -------
        fig : go.Figure
            Plotly figure of brain visualizations
        """

        # Analysis
        if self.mode2 is None:

            rows, intensity = self.intensities([self.mode1], imag)
            labels = ['Real'] if len(rows) == 1 else ['Real', 'Imaginary']

        # Comparison
        else:

            rows, intensity = self.intensities([self.mode1, self.mode2], imag)

            if len(rows) == 2:
                labels = ['Group 1 \n Real', 'Group 2 \n Real']
            elif len(rows) == 3:
                if self.mode1.conjugate:
                    labels = ['Group 1 \n  Real', 'Group 1 \n Imaginary', 'Group 2 \n Real']
                else:
                    labels = ['Group 1 \n Real', 'Group 2 \n  Real', 'Group 2 \n  Imaginary']
            else:
                labels = ['Group 1 \n Real', 'Group 1 \n Imaginary', 'Group 2 \n  Real', 'Group 2 \n  Imaginary']

        fig = make_subplots(rows=len(rows), cols=1, horizontal_spacing=0.05, vertical_spacing=0.05)

        for row in rows:

            for id in range(1, list(self.coords_2d['.id'])[-1] + 1):
                roi = self.coords_2d[self.coords_2d['.id'] == id]
                x = roi['.long']
                y = roi['.lat']
                region = roi['region'].iloc[0] if roi['region'].size != 0 else None

                if type(region) == str:

                    indice = roi['.roi'].iloc[0] - 1

                    if roi['hemi'].iloc[0] == 'right':
                        indice += int(intensity[0].shape[0] / 2)

                    val = intensity[row - 1][indice]

                    col = 'rgb({0},{1},{2})'.format(*cm.get_cmap(colormap)(val)[:3])
                else:
                    col = 'black'

                fig.add_trace(go.Scatter(x=x, y=y, fill='toself', mode='lines', line=dict(color='black', width=0.5),
                                         fillcolor=col, name=region if region else None, hoverinfo=None),
                              row=row, col=1)

            fig.add_trace(go.Scatter(x=[300, 1100], y=[-25, -25], text=['left', 'right'], mode='text'),
                          row=row, col=1)

        axis_config = dict(showgrid=False, showline=False, visible=False, ticks='',
                           showticklabels=False, zeroline=False, showspikes=False)

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
                          title_text='Mode {}'.format(self.order if self.order is not None else ''),
                          height=150 + len(rows) * 200)

        for i, label in enumerate(labels):

            fig.layout['xaxis' + str(i + 1)].update(axis_config)
            fig.layout['yaxis' + str(i + 1)].update({**axis_config, **dict(scaleanchor='x' + str(i + 1), scaleratio=1,
                                                                           title=labels[i], visible=True)})

        # Hack to get a colorbar
        fig.add_trace(go.Scatter(x=[200, 200], y=[-20, -20],
                                 marker=dict(size=0.01, opacity=1, cmax=0.1, cmin=-0.1, color=[-0.1, 0.1],
                                             colorbar=dict(title="Activity", len=.7, nticks=3),
                                             colorscale=colorscale(colormap)),
                                 mode="markers"))

        return fig
