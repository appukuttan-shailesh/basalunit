import sciunit
import sciunit.utils as utils

import numpy as np
import similaritymeasures

class PartialCurveMappingScore(sciunit.Score):
    """
    A Partial Curve Mapping (PCM) score. A float giving the value of
    a Partial Curve Mapping between two 2D-curves.

    PCM uses a combination of arc-length and area to determine the similarity
    between curves. First the arc-length of the shorter curve is imposed onto a
    section on the longer curve. Then trapezoids are constructed between the
    curves, and the areas of the trapezoids are summed. This is repeated for 200
     or so iterations, as various offsets of the short arc-length are imposed on
    the curve with the longer arc-length. The final PCM value is the minimum area
    from all attempted arc-length offsets.

    Source: Jekel, C. F., Venter, G., Venter, M. P., Stander, N., & Haftka,
    R. T. (2018). Similarity measures for identifying material parameters from
    hysteresis loops using inverse analysis. International Journal of Material
    Forming. https://doi.org/10.1007/s12289-018-1421-8
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
