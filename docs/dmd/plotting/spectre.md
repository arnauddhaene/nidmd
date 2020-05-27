Module dmd.plotting.spectre
===========================

Classes
-------

`Spectre(df)`
:   Constructor
    
    :param df: [pd.DataFrame] containing 'Mode', 'Absolute Value of Eigenvalue', 'Group'

    ### Static methods

    `correlation(df)`
    :   Get linear trendline figure.
        
        :param df: [pd.DataFrame] with 'x', 'y' columns
        :return: [go.Figure]

    ### Methods

    `figure(self)`
    :   Get Figure
        
        :return: [go.Figure]