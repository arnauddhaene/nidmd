# This Python file uses the following encoding: utf-8
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib import cm
from utils import *


class Brain:

    def __init__(self, atlas, mode1, mode2=None):
        """
        Constructor

        :param atlas: cortical parcellation { 'schaefer', 'glasser' } (string)
        :param mode1: intensity information (Mode)
        :param mode2: intensity information for comparison (Mode)
        """

        self.atlas = ATLAS2D[atlas]

        # if not mode1.order == mode2.order:
            # TODO: RAISE ERROR
        self.mode1 = mode1
        self.mode2 = mode2

    def _get_intensity(self, modes):
        """
        Add mode

        :param mode: mode (list of Mode)
        :return: rows, intensity
        """

        rows = []
        intensity = []

        for mode in modes:

            if not mode.is_complex_conjugate:
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


    def figure(self, complex=None):
        """
        Get Plotly figure

        :param complex: { 'real', 'imag' }, default both (string)
        :return: Plotly figure instance
        """

        # Analysis
        if not self.mode2:

            rows, intensity = self._get_intensity([self.mode1])

            labels = ['Real'] if len(rows) == 1 else ['Real', 'Imaginary']

        # Comparison
        else:

            rows, intensity = self._get_intensity([self.mode1, self.mode2])

            if len(rows) == 2:
                labels = ['G1 Real', 'G2 Real']
            elif len(rows) == 3:
                if self.mode1.is_complex_conjugate():
                    labels = ['G1 Real', 'G1 Imaginary','G2 Real']
                else:
                    labels = ['G1 Real', 'G2 Real','G2 Imaginary']
            else:
                labels = ['G1 Real', 'G1 Imaginary', 'G2 Real','G2 Imaginary']

        fig = make_subplots(rows=len(rows), cols=1,
                            # shared_xaxes=True,  # specs=[[{}], [{}]] if len(rows) == 2 else [[{}]],
                            horizontal_spacing=0.05,
                            vertical_spacing=0.05)

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

                fig.add_trace(go.Scatter(x=x,
                                         y=y,
                                         fill='toself',
                                         mode='lines',
                                         line=dict(color='black', width=0.5),
                                         fillcolor=col,
                                         name=region if region else None,
                                         hoverinfo=None),
                              row=row, col=1)

            fig.add_trace(go.Scatter(x=[300, 1100], y=[-25, -25],
                                     text=['left', 'right'],
                                     mode='text'),
                          row=row, col=1)

            fig.update_xaxes(showgrid=False, showline=False, visible=False, ticks='',
                             showticklabels=False, zeroline=False, showspikes=False)
            fig.update_yaxes(showgrid=False, showline=False, visible=True, ticks='',
                             title=labels[row - 1],
                             showticklabels=False, zeroline=False, showspikes=False, constrain='domain',
                             scaleanchor='x', scaleratio=1,
                             row=row, col=1)

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          showlegend=False, title_text='Mode {}'.format(self.mode1.order))

        return fig
