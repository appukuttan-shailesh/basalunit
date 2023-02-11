import sciunit
import sciunit.utils as utils

import numpy as np
import similaritymeasures

class FrechetDistanceScore(sciunit.Score):
    """
    A Discrete Frechet Distance score. A float giving the value of
    a discrete Frechet distance between two 2D-curves.

    In mathematics, the Fréchet distance is a measure of similarity between
    curves that takes into account the location and ordering of the points along
     the curves. It is named after Maurice Fréchet.

    Intuitive definition:
    ---------------------
    Imagine a person traversing a finite curved path while walking their dog on
    a leash, with the dog traversing a separate finite curved path. Each can
    vary their speed to keep slack in the leash, but neither can move backwards.
     The Fréchet distance between the two curves is the length of the shortest
     leash sufficient for both to traverse their separate paths from start to
     finish.

    Source: https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance
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
