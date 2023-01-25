import math
import sciunit

#==============================================================================

class CombineScores(sciunit.Score):
    """
    Custom implementation for combining multiple Scores into a single value
    Approach: Mean of absolute Scores
    """

    def __init__(self, score, related_data={}):
        if not isinstance(score, Exception) and not isinstance(score, float):
            raise sciunit.InvalidScoreError("Score must be a float.")
        else:
            super(CombineScores,self).__init__(score, related_data=related_data)

    @classmethod
    def compute(cls, input_scores):
        """
        Accepts a sequence of Scores (e.g. computed via sciunit.scores.ZScore)
        and combines them into a single score.

        To keep this generic, the input data format (supplied by the test) are:
        input_scores = [S1, S2, ..., Sn]
            where Si corresponds to the individual Scores
        """
        score = sum(map(abs,input_scores)) / len(input_scores)
        return CombineScores(score)

    _description = ("Combining Scores between observation and prediction")

    @property
    def sort_key(self):
        """
        ## Copied from sciunit.scores.ZScore ##
        Returns 1.0 for a score of 0, falling to 0.0 for extremely positive
        or negative values.
        """
        cdf = (1.0 + math.erf(self.score / math.sqrt(2.0))) / 2.0
        return 1 - 2*math.fabs(0.5 - cdf)

    def __str__(self):
        return 'Combined score = %.2f' % self.score
