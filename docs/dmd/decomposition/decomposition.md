Module dmd.decomposition.decomposition
======================================

Classes
-------

`Decomposition(data=None, filenames=None)`
:   Decomposition constructor
    
    :param data: [list of Array-like]
    :param filenames: [list of str]

    ### Static methods

    `normalize(data, direction=1, demean=True, destandard=True)`
    :   Normalize the original data set.
        
        :param x: [Array-like] data
        :param direction: [int] 0 for columns, 1 for rows, None for global
        :param demean: [boolean] demean
        :param destandard: [boolean] remove standard deviation (to 1)
        :return x: [Array-like] standardized data
        :return mean: [float] mean of original data
        :return std: [float] standard deviation of original data

    ### Methods

    `add_data(self, data, sampling_time=None)`
    :   Add data to decomposition. Also defines Decomposition.atlas
        
        :param data: [Array-like] data matrix (N x T)
        :param sampling_time: [float] sampling time (s)

    `compute_match(self, other, m)`
    :   Get matched modes for match group with self (reference group).
        Predicts amplification of approximated modes using linear regression.
        
        :param other: [Decomposition]
        :param m: [int] number of modes approximated
        :return: [pd.DataFrame] containing 'mode', 'value', 'damping_time', 'period', 'conjugate'
        :return x: [Array-like] vector containing absolute value of top 10 approximated eigenvalues of self
        :return y: [Array-like] vector containing absolute value of real 10 eigenvalues of self

    `run(self)`
    :   Run decomposition.