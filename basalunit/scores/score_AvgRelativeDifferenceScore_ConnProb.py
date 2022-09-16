import math
import sciunit.scores
import quantities
import numpy as np

#==============================================================================

class AvgRelativeDifferenceScore_ConnProb(sciunit.Score):
    """
	Returns average of absolute relative differences
    """

    _allowed_types = (float,)

    _description = ('Returns average of absolute relative differences')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dont_hide = ["related_data"]

    @classmethod
    def compute(cls, observation, prediction):
        """
        Computes average of absolute relative differences
        """
        score_list = []
        for item in observation:
            max_dist = item["max_dist"]
            obs_val = item["value"] if "value" in item.keys() else item["num"]/item["total"]
            pred_val = prediction[max_dist]
            rel_diff_score = (obs_val-pred_val)/obs_val
            score_list.append({ "max_dist" : max_dist,
                                "obs": obs_val,
                                "pred": pred_val,
                                "rel_diff_score": rel_diff_score})
        score = sum([abs(x["rel_diff_score"]) for x in score_list])/len(score_list)
        return AvgRelativeDifferenceScore_ConnProb(score), score_list

    def __str__(self):
        return '%.2f' % self.score
