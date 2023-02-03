import sciunit
import basalunit.capabilities as basalunit_cap
import basalunit.scores as basalunit_scores
import basalunit.plots as basalunit_plots
import json
import os
import numpy as np
import math

class SomaticExcitability_Test(sciunit.Test):

    """Tests the firing-frequency response of neurons as a function of
    the input current. More specifically, it compares the current-frequency
    curve of the model with experimental curves from Planert et al. (2013)"""
    score_type = basalunit_scores.CombineScores

    def __init__(self, model_path=None, observation=None, name="SomaticExcitability_Test", base_directory=None):

        self.description = "Tests the firing-frequency response of neurons as a \
                            function of the input current."
        require_capabilities = (basalunit_cap.Provides_FiringFreqVsCurrent_Info,)

        if not model_path:
            raise ValueError("Please specify the path to the model directory!")
        if not os.path.isdir(model_path):
            raise ValueError("Specified path to the model directory is invalid!")
        self.model_path = model_path

        if not observation:
            # Use the path to experimental data, inside Lindroos et al. (2018) model directory
            observation_path=os.path.join(self.model_path, 'Exp_data/FI/Planert2013-D1-FI-trace1.csv')
            dict_observation = self.get_observation(obs_path=observation_path)
        else:
            raw_observation = dict_observation
        self.observation = dict_observation

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

        return { "Frequency (Hz)": x1, "Current (pA)": y1 }


    def format_data(self, data):
        """
        This accepts data input in the form of a 2D-list of numpy arrays:
        [ numpy.ndarray([ X1, X2, ...., Xn ]), numpy.ndarray([Y1, Y2, ...., Yn]) ]

        Returns the transpose of the 2D-data input, after discarding NaN values
        """
        data_vals = [ coord_v[~np.isnan(coord_v)] for coord_v in data]

        return np.transpose(data_vals)


    def generate_prediction(self, model, verbose=False):
        self.model = model
        self.model_name = model.name
        model.somatic_excitability()

        raw_prediction = model.get_FreqCurrent()[0:2]

        prediction = self.format_data([ raw_prediction[0], raw_prediction[1] ])

        return prediction


    def compute_score(self, observation, prediction, verbose=True):
        """Implementation of sciunit.Test.score_prediction"""

        raw_observation = list(observation.values())
        observation = self.format_data(raw_observation)

        scores_dict = dict()
        # quantify the difference between the two curves using Partial Curve Mapping
        scores_dict['Partial Curve Mapping'] = \
            basalunit_scores.PartialCurveMappingScore.compute(observation, prediction).score

        # quantify the difference between the two curves using
        # Discrete Frechet distance
        scores_dict['Discrete Frechet distance'] = \
            basalunit_scores.FrechetDistanceScore.compute(observation, prediction).score

        # quantify the difference between the two curves using
        # area between two curves
        scores_dict['Area in between'] = \
            basalunit_scores.AreaInBetweenScore.compute(observation, prediction).score

        # quantify the difference between the two curves using
        # Curve Length based similarity measure
        scores_dict['Curve Length'] = \
            basalunit_scores.CurveLengthScore.compute(observation, prediction).score

        # quantify the difference between the two curves using
        # Dynamic Time Warping distance
        scores_dict['Dynamic Time Warping'] = \
            basalunit_scores.DynamicTimeWarpingScore.compute(observation, prediction).score

        self.scores_dict = scores_dict

        # Computing the score
        # Taking the average of the similarity measures, as the overall score for the Test
        scores_list = list(self.scores_dict.values())
        self.score = basalunit_scores.CombineScores.compute(scores_list)
        self.score.description = "A mean value of similarity measure between two curves"

        # Saving figure with with scores in the form of bar-plot
        ylabel = 'Curves similarity measures'
        score_label = r'|Score value| in $\mathit{log}$ scale'
        fig_title = 'FreqCurrent'
        plt_title = "Frequency-Current curves: \n Model vs Experiment"
        barplot_figure = basalunit_plots.ScoresBars(testObj=self, score_label=score_label, ylabel=ylabel,
                                                    fig_title=fig_title, plt_title=plt_title,
                                                    score_scale='symlog')
        barplot_files = barplot_figure.create()
        self.figures.extend(barplot_files)

        # ---------------------- Saving other relevant results ----------------------
        fig_FI_model_obs = self.model.plot_vm()
        self.figures.extend(fig_FI_model_obs)

        return self.score


    def bind_score(self):
        self.score.related_data["figures"] = self.figures
        return self.score
