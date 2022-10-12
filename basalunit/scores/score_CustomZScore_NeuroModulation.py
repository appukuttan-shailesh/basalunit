import sciunit.scores
import numpy as np

#==============================================================================

class CustomZScore_NeuroModulation(sciunit.Score):
    """
	Returns abs Mean of Z-scores; each abs score above 2.0 gets 50.0 as penalty
    Minimum of 5 scores; else a penalty of 50.0 per missing score (i.e. 50 * [5 - len(scores)])
    """

    _allowed_types = (float,)

    _description = ('Returns custom mean Z-scores')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dont_hide = ["related_data"]

    @classmethod
    def compute(cls, observation, prediction):
        f = prediction["experiment"]
        fc = prediction["control"]
        
        total_score = 0.0
        total_penalty = 0.0
        ind_z_scores = {}
        for i, n in enumerate(f["neurons"].keys()):
            if prediction["neuron_types"][i] == observation["neuron_type"]:
                control_spikes = fc["neurons"][n]["spikes"]["data"][()]
                experiment_spikes = f["neurons"][n]["spikes"]["data"][()]
                if experiment_spikes.size > 0 or control_spikes.size>0:
                    diff = experiment_spikes.size - control_spikes.size
                    z_score = (diff - observation["mean"]) / observation["std"]
                    ind_z_scores[i] = z_score
                    z_score = abs(z_score)
                    if z_score > 2.0:
                        total_penalty += 50.0
                    total_score += z_score

        if len(ind_z_scores) < 5:
            total_penalty += 50.0 * (5 - len(ind_z_scores))
        total_score += total_penalty

        score = total_score/float(len(ind_z_scores) if len(ind_z_scores) > 5 else 5)
        return CustomZScore_NeuroModulation(score), ind_z_scores, total_penalty

    def __str__(self):
        return '%.2f' % self.score
