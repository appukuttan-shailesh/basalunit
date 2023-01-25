import sciunit
import sciunit.utils as utils

import numpy as np
import similaritymeasures

class DynamicTimeWarpingScore(sciunit.Score):
    """
    A Dynamic Time Warping score. A float giving the value of
    a Dynamic Time Warping between two 2D-curves.
    """

    _allowed_types = (float,)

    _description = ('A Dynamic Time Warping score. A float giving the similarity between'
                    'two 2D-curves, measured as a Dynamic Time Warping.')

    @classmethod
    def compute(cls, observation, prediction):
        """
        Computes a Dynamic Time Warping score from an observation and a prediction.
        """
        value, d = similaritymeasures.dtw(observation, prediction)
        value = utils.assert_dimensionless(value)
        return DynamicTimeWarpingScore(value)

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return 'Dynamic Time Warping score = %.5f' % self.score
