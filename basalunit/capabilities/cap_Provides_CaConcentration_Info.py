import sciunit

#==============================================================================

class Provides_CaConcentration_Info(sciunit.Capability):
    """
    Indicates that the model returns Calcium concetration, namely:
    the local change in calcium concentration as a function of somatic distance
    following a backpropagating action potential
    """

    def get_Ca(self):
        """
        Must return a 2D-tupple with arrays containing information of the type:
            ([soma_distance_array], [calcium_concentration_array])
        """
        raise NotImplementedError()
