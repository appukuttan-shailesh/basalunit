import sciunit
from numpy import nan
import math

#==============================================================================

class CohenDScore(sciunit.Score):
    """
    Cohen’s D, or standardized mean difference, is one of the most common ways to measure effect size. 
    It is a measure of the difference between two means, normalized by the standard deviation of the data.
    
    Based on: https://github.com/scidash/sciunit/blob/master/sciunit/scores/complete.py
    """

    _allowed_types = (float, int, None,)

    _description = ('The measure of the difference between two means, normalized by the standard deviation of the data')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dont_hide = ["related_data"]
    
    @classmethod
    def compute(cls, obs, pred):
        """
        Computes Cohen’s D score from an observation and a prediction, both in form of mean, std.
        """
        if pred["mean"] == None:
            # No prediction available from test
            score = nan
        elif pred["std"] == 0:
            # use abs Z-score instead
            score = abs((float(pred["mean"]) - float(obs["mean"])) / float(obs["std"]))
        else:
            # use KLdiv score
            p_mean = pred["mean"]  # Use the prediction's mean.
            p_std = pred["std"]
            o_mean = obs["mean"]
            o_std = obs["std"]
            try:  # Try to pool taking samples sizes into account.
                p_n = pred["n"]
                o_n = obs["n"]
                s = (
                    ((p_n - 1) * (p_std ** 2) + (o_n - 1) * (o_std ** 2)) / (p_n + o_n - 2)
                ) ** 0.5
            except KeyError:  # If sample sizes are not available.
                s = (p_std ** 2 + o_std ** 2) ** 0.5
            score = (p_mean - o_mean) / s
        return CohenDScore(score)

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return '%.2f' % self.score