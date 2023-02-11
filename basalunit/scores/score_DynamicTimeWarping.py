import sciunit
import sciunit.utils as utils

import numpy as np
import similaritymeasures

class DynamicTimeWarpingScore(sciunit.Score):
    """
    A Dynamic Time Warping (DTW) score. A float giving the value of
    a Dynamic Time Warping between two 2D-curves.

    In general, DTW is a method that calculates an optimal match between two
    given sequences (e.g. time series) with certain restriction and rules:

    - Every index from the first sequence must be matched with one or more
    indices from the other sequence, and vice versa
    - The first index from the first sequence must be matched with the first
    index from the other sequence (but it does not have to be its only match)
    - The last index from the first sequence must be matched with the last index
     from the other sequence (but it does not have to be its only match)
    - The mapping of the indices from the first sequence to indices from the
    other sequence must be monotonically increasing, and vice versa.

    The optimal match is denoted by the match that satisfies all the restrictions
    and the rules and that has the minimal cost, where the cost is computed as
    the sum of absolute differences, for each matched pair of indices,
    between their values.

    Source: https://en.wikipedia.org/wiki/Dynamic_time_warping
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
