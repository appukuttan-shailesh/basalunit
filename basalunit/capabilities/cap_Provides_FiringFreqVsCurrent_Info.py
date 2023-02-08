import sciunit

#==============================================================================

class Provides_FiringFreqVsCurrent_Info(sciunit.Capability):
    """
    Indicates that the model returns a F-I curve, namely:
    the firing-frequency response of neurons as a function of
    the input current.
    """

    def get_FreqCurrent(self):
        """
        Must return a 2D-list with two Numpy arrays, containing information about
        the neuronal firing rate for each current amplitude, in the form:
            [ numpy.array(I_amp), numpy.array(Firing_rate) ]
        """
        raise NotImplementedError()
