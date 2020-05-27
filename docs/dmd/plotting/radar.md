Module dmd.plotting.radar
=========================

Classes
-------

`Radar(df, networks)`
:   Constructor.
    
    :param df: [pd.DataFrame] containing 'mode', 'group', 'strength_real', 'strength_imaginary'

    ### Methods

    `figure(self, imag=False, amount=6)`
    :   Get Figure
        
        :param imag: [boolean] add imaginary values
        :param amount: [int] number of modes to plot
        :return: [go.Figure]