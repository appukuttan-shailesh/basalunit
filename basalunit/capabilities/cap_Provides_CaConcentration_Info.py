import sciunit

#==============================================================================

class Provides_CaConcentration_Info(sciunit.Capability):
    """
    Indicates that the model returns Calcium concetration, namely:
    the local change in calcium concentration as a function of somatic distance
    following a backpropagating action potential
    """

    def get_Ca_bAP(self):
        """
        Must return a 2D-list with two Numpy arrays containing information about
        the dendritic distance from the soma and the Calcium concentrations:
        [ numpy.array(soma_distances), numpy.array(calcium_concentrations) ]
        """
        raise NotImplementedError()
