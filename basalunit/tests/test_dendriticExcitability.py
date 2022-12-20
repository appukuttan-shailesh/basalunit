import sciunit
import basalunit.capabilities as basalunit_cap
import basalunit.scores as basalunit_scores
import json
import os
import numpy as np


class DendriticExcitability_Test(sciunit.Test):

    """Tests dendritic excitability as the local change in calcium concentration
    as a function of somatic distance following a backpropagating
    action potential (Day et al., 2008)"""
    # score_type = basalunit_scores.CombineZScores

    def __init__(self, model_path=None, observation=None, name="DendriticExcitability_Test", base_directory=None):

        self.description = "Tests dendritic excitability as the local change in \
        calcium concentration following a backpropagating action potential"
        require_capabilities = (basalunit_cap.Provides_CaConcentration_Info,)

        if not model_path:
            raise ValueError("Please specify the path to the model directory!")
        if not os.path.isdir(model_path):
            raise ValueError("Specified path to the model directory is invalid!")
        self.model_path = model_path

        if not observation:
            # Use the path to experimental data, inside Lindroos et al. (2018) model directory
            observation_path=os.path.join(self.model_path, 'Exp_data/bAP/bAP-DayEtAl2006-D1.csv')
            self.observation = self.get_observation(obs_path=observation_path)
        else:
            self.observation = observation

        if not base_directory:
            base_directory = "./validation_results"
        self.path_test_output = base_directory
        # create output directory
        if not os.path.exists(self.path_test_output):
            os.makedirs(self.path_test_output)
        self.figures = []


    def get_observation(self, obs_path):
        if obs_path == None:
            raise ValueError("Please specify the path to the observation file!")
        if not os.path.isfile(obs_path):
            raise ValueError("Specified path to the observation file is invalid!")

        [x1,y1] = np.loadtxt(obs_path, unpack=True)

        return [x1,y1]


    def generate_prediction(self, model, verbose=False):
        self.model = model
        self.model_name = model.name
        model.dendrite_BAP()

        [dend_dist, Ca_mean_amp] = model.get_Ca()
        Ca_mean_amp = np.divide(Ca_mean_amp, Ca_mean_amp[3])

        return [dend_dist, Ca_mean_amp]


    def compute_score(self, observation, prediction, verbose=True):
        """Implementation of sciunit.Test.score_prediction"""

        self.observation = observation
        self.prediction = prediction

        # Computing the scores
        self.score = 1.0
        # self.score.description = "A mean distance between two profiles"

        # ---------------------- Saving relevant results ----------------------
        fig_Ca_model_obs = self.model.plot_Ca()
        self.figures.extend(fig_Ca_model_obs)

        return self.score


    def bind_score(self):
        self.score.related_data["figures"] = self.figures
        return self.score
