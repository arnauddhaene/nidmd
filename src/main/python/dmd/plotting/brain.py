# This Python file uses the following encoding: utf-8
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils import *


class Brain:

    def __init__(self, atlas, mode1, mode2=None, order=None):
        """
        Constructor

        :param atlas: [str] cortical parcellation { 'schaefer', 'glasser' }
        :param mode1: intensity information (object with 'intensity', 'conjugate')
        :param mode2: intensity information for comparison (object with 'intensity', 'conjugate')
        """

        self.atlas = ATLAS2D[atlas]

        self.mode1 = mode1
        self.mode2 = mode2
        self.order = order

    @staticmethod
    def _get_intensity(modes, imag=False):
        """
        Add mode

        :param modes: [pd.DataFrame] with columns 'conjugate' and 'intensity'
        :param imag: [boolean] get imaginary values
        :return rows: [list] index list of number of figure rows
        :return intensity: [list of Array-like]
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

    def figure(self, imag=False):
        """
        Get Figure.

        :param imag: [boolean] Plot imaginary values
        :return: [go.Figure]
        """

        # Analysis
        if self.mode2 is None:

            rows, intensity = self._get_intensity([self.mode1], imag)
            labels = ['Real'] if len(rows) == 1 else ['Real', 'Imaginary']

        # Comparison
        else:

            rows, intensity = self._get_intensity([self.mode1, self.mode2], imag)

            if len(rows) == 2:
                labels = ['G1 Real', 'G2 Real']
            elif len(rows) == 3:
                if self.mode1.conjugate:
                    labels = ['G1 Real', 'G1 Imaginary','G2 Real']
                else:
                    labels = ['G1 Real', 'G2 Real', 'G2 Imaginary']
            else:
                labels = ['G1 Real', 'G1 Imaginary', 'G2 Real', 'G2 Imaginary']

        fig = make_subplots(rows=len(rows), cols=1, horizontal_spacing=0.05, vertical_spacing=0.05)

        for row in rows:

            for id in range(1, list(self.atlas['.id'])[-1] + 1):
                roi = self.atlas[self.atlas['.id'] == id]
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
                                             colorscale=COOLWARM),
                                 mode="markers"))

        return fig
