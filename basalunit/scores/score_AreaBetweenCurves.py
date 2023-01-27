import sciunit
import sciunit.utils as utils

import numpy as np
import similaritymeasures

class AreaInBetweenScore(sciunit.Score):
    """
    An Area in between two curves score. A float giving the value of
    a Area in between them between two 2D-curves.
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
