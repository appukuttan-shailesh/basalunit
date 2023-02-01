import sciunit
import sciunit.utils as utils

import numpy as np
import similaritymeasures

class CurveLengthScore(sciunit.Score):
    """
    A Curve Length score. A float giving the value of
    a Curve Length between two 2D-curves.

    A corresponding data point on one curve is calculated at the
    equivalent curve length location of the other curve. Squared residual
    values are then calculated as function of both coordinates. The sum of
    these squared residuals is used to quantify the difference between
    the two curves.

    Source: Jekel, C. F., Venter, G., Venter, M. P., Stander, N., & Haftka,
    R. T. (2018). Similarity measures for identifying material parameters from
    hysteresis loops using inverse analysis. International Journal of Material
    Forming. https://doi.org/10.1007/s12289-018-1421-8
    """

    _allowed_types = (float,)

    _description = ('A Curve Length score. A float giving the similarity between'
                    'two 2D-curves, measured as a Curve Length.')

    @classmethod
    def compute(cls, observation, prediction):
        """
        Computes a Curve Length score from an observation and a prediction.
        """
        value = similaritymeasures.curve_length_measure(observation, prediction)
        value = utils.assert_dimensionless(value)
        return CurveLengthScore(value)

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return 'Curve Length score = %.5f' % self.score
