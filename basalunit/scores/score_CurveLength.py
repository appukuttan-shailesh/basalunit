import sciunit
import sciunit.utils as utils

import numpy as np
import similaritymeasures

class CurveLengthScore(sciunit.Score):
    """
    A Curve Length score. A float giving the value of
    a Curve Length between two 2D-curves.
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
