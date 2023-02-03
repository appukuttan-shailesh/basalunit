import sciunit

#==============================================================================

class Provides_FiringFreqVsCurrent_Info(sciunit.Capability):
    """
    Indicates that the model returns F-I curve, namely:
    the firing-frequency response of neurons as a function of
    the input current.
    """

    def get_FreqCurrent(self):
        """
        Must return a 2D-list with two arrays containing information of the type:
            [I_amp, SpikeCount]
        """
        raise NotImplementedError()
