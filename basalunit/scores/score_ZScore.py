import math
import sciunit.scores
import quantities
import numpy as np

#==============================================================================

class BU_ZScore(sciunit.Score):
    """
	Returns Z-score
    (not using sciunit's Z-score currently owing to complex structure of
    observation and prediction; could merge/change later on)
    """

    _allowed_types = (float,)

    _description = ('The difference between the means of the observation and '
                    'prediction divided by the standard deviation of the '
                    'observation')

    @classmethod
    def compute(cls, observation, prediction):
        """
        Computes score based on whether value is within the range or not.
        """

        def nested_set(dic, keys, value):
            for key in keys[:-1]:
                dic = dic.setdefault(key, {})
            dic[keys[-1]] = value

        assert isinstance(prediction,dict)
        assert isinstance(observation,dict)

        score_dict = {}
        mean_score = 0
        ctr = 0
        prob_list = []

        for key_0 in prediction:
            for key_1 in prediction[key_0]:
                for key_2 in prediction[key_0][key_1]:
                    o_mean = observation[key_0][key_1][key_2][0]
                    o_std = observation[key_0][key_1][key_2][1]
                    p_value = prediction[key_0][key_1][key_2]
                    value = (p_value - o_mean)/o_std
                    nested_set(score_dict, [key_0, key_1, key_2], value)
                    if not np.isnan(value) and not np.isinf(value):
                        mean_score += abs(value)
                        ctr += 1
                    else:
                        prob_list.append("{}.{}.{}".format(key_0, key_1, key_2))
        score_dict = dict(score_dict)
        score = mean_score/float(ctr)
        return score, score_dict, prob_list

    def __str__(self):
        return '%.2f' % self.score
