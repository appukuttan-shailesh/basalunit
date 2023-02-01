import sciunit
import sciunit.utils as utils

import numpy as np
import similaritymeasures

class AreaInBetweenScore(sciunit.Score):
    """
    An Area in between two curves score. A float giving the value of
    a Area in between them between two 2D-curves.

    Two curves are generally discretized into a time-series of ordered data points.
    The algorithm constructs quadrilaterals between two curves and calculates
    the area for each quadrilateral. Two curves that are being compared require
    the same number of points in order to construct quadrilaterals to approximate
    the total area between curves. If one curve has fewer data points than the
    other curve, data points are added until both curves have the same number of
    data points. It was chosen to add, rather than remove points to avoid any
    loss of information for general problems.

    Source: Jekel, C. F., Venter, G., Venter, M. P., Stander, N., & Haftka,
    R. T. (2018). Similarity measures for identifying material parameters from
    hysteresis loops using inverse analysis. International Journal of Material
    Forming. https://doi.org/10.1007/s12289-018-1421-8
    """

    _allowed_types = (float,)

    _description = ('A Area in between them score. A float giving the similarity between'
                    'two 2D-curves, measured as a Area in between them.')

    @classmethod
    def compute(cls, observation, prediction):
        """
        Computes a Area in between them score from an observation and a prediction.
        """
        value = similaritymeasures.area_between_two_curves(observation, prediction)
        value = utils.assert_dimensionless(value)
        return AreaInBetweenScore(value)

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return 'Area in between two profiles Score = %.5f' % self.score
