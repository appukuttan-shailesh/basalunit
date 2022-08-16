import sciunit
from numpy import nan
import math

#==============================================================================

class KLdivMeanStd(sciunit.Score):
    """
    A Kullback-Leibler divergence score. A float giving the Kullback-Leibler divergence (KLdiv),
    a measure indicating how much a predicted (model's) probability distribution P_mod
    diverges from an experimental one (observation's) P_obs. In the simple case, a KLdiv with value of 0
    indicates that almost similar behavior of both probabilities can be expected.
    The contrary holds when the divergence's value is 1.

    Based on simplification for two univariate normal distributions:
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """

    _allowed_types = (float, int, None,)

    _description = ('The divergence from the probability P_mod to the probability P_obs, being computed '
                    'as the expectation of the logarithmic difference between P_mod and P_obs, '
                    'where the expectation is taken using the probabilities P_obs.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dont_hide = ["related_data"]
    
    @classmethod
    def compute(cls, obs, pred):
        """
        Computes a KLdiv-score from an observation and a prediction, both in form of mean, std.
        """
        if pred["mean"] == None:
            # No prediction available from test
            score = nan
        elif pred["std"] == 0:
            # use abs Z-score instead
            score = abs((float(pred["mean"]) - float(obs["mean"])) / float(obs["std"]))
        else:
            # use KLdiv score
            score = math.log(float(obs["std"])/float(pred["std"])) + (((float(pred["std"])**2) + ((float(pred["mean"])-float(obs["mean"]))**2))/(2*(float(obs["std"])**2))) - 0.5
        return KLdivMeanStd(score)

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return '%.5f' % self.score