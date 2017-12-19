import sciunit

#==============================================================================

class ProvidesDensityInfo(sciunit.Capability):
    """
    Indicates that the model returns morphological information, namely:
    1) density of cells in a specfic layer of the model (1000/mm3)
    """

    def get_density_info(self):
        """
        Must return a dictionary of the form:
            {"density": {"value": "XX 1000/mm3"}}
        """
        raise NotImplementedError()

    # not used currently
    def get_cell_density(self):
        """ returns the cell density in model """
        density_info = self.get_density_info()
        return density_info["density"]["value"]
