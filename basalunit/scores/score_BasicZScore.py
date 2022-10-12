import sciunit
from sciunit.scores.incomplete import NAScore
import math

#==============================================================================

class BasicZScore(sciunit.Score):
    """
    The difference between the means of the observation and prediction 
    divided by the standard deviation of the observation"
    """

    _allowed_types = (float, int, None,)

    _description = ('Basic Z-score. Float indicating standardized difference')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dont_hide = ["related_data"]
    
    @classmethod
    def compute(cls, obs, pred):
        """
        Computes a Z-score from an observation (mean, std) and a prediction (mean)
        """
        print(pred["mean"])
        print(obs["mean"])
        print(obs["std"])
        score = ((float(pred["mean"]) - float(obs["mean"])) / float(obs["std"]))
        return BasicZScore(score)

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return '%.5f' % self.score