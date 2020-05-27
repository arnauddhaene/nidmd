Module dmd.plotting.brain
=========================

Classes
-------

`Brain(atlas, mode1, mode2=None, order=None)`
:   Constructor
    
    :param atlas: [str] cortical parcellation { 'schaefer', 'glasser' }
    :param mode1: intensity information (object with 'intensity', 'conjugate')
    :param mode2: intensity information for comparison (object with 'intensity', 'conjugate')

    ### Methods

    `figure(self, imag=False)`
    :   Get Figure.
        
        :param imag: [boolean] Plot imaginary values
        :return: [go.Figure]