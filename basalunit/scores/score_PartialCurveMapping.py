import sciunit
import sciunit.utils as utils

import numpy as np
import similaritymeasures

class PartialCurveMappingScore(sciunit.Score):
    """
    A Partial Curve Mapping score. A float giving the value of
    a Partial Curve Mapping between two 2D-curves.
    """

    _allowed_types = (float,)

    _description = ('A Partial Curve Mapping score. A float giving the similarity between'
                    'two 2D-curves, measured as a Partial Curve Mapping.')

    @classmethod
    def compute(cls, observation, prediction):
        """
        Computes a Partial Curve Mapping score from an observation and a prediction.
        """
        value = similaritymeasures.pcm(observation, prediction)
        value = utils.assert_dimensionless(value)
        return PartialCurveMappingScore(value)

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return 'Partial Curve Mapping score = %.5f' % self.score
