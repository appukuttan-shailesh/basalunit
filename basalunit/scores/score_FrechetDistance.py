import sciunit
import sciunit.utils as utils

import numpy as np
import similaritymeasures

class FrechetDistanceScore(sciunit.Score):
    """
    A Discrete Frechet Distance score. A float giving the value of
    a discrete Frechet distance between two 2D-curves.
    """

    _allowed_types = (float,)

    _description = ('A Discrete Frechet Distance score. A float giving the similarity between'
                    'two 2D-curves, measured as a discrete Frechet distance.')

    @classmethod
    def compute(cls, observation, prediction):
        """
        Computes a Discrete Frechet Distance score from an observation and a prediction.
        """
        value = similaritymeasures.frechet_dist(observation, prediction)
        value = utils.assert_dimensionless(value)
        return FrechetDistanceScore(value)

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return 'Discrete Frechet Distance score = %.5f' % self.score
