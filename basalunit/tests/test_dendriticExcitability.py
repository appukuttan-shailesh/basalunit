import sciunit


class DendriticExcitability(sciunit.test):

    """Tests dendritic excitability as the local change in calcium concentration
    as a function of somatic distance following a backpropagating
    action potential (Day et al., 2008)"""
    # score_type = basalunit_scores.CombineZScores

    def __init__(self, observation=None, name="DendriticExcitability_Test", base_directory=None):

        self.description = "Tests dendritic excitability as the local change in \
        calcium concentration following a backpropagating action potential"
        # require_capabilities = (mph_cap.ProvidesMorphFeatureInfo,)

        if not base_directory:
            base_directory = "."
        self.path_test_output = base_directory
        # create output directory
        if not os.path.exists(self.path_test_output):
            os.makedirs(self.path_test_output)

        # Checks raw observation data compliance with NeuroM's nomenclature
        self.check_observation(observation)
        self.raw_observation = observation

        json.dumps(observation, sort_keys=True, indent=3)

        self.figures = []
        observation = self.format_data(observation)
        sciunit.Test.__init__(self, observation, name)

    def format_data(self, data):
        """
        This accepts data input in the form:
        ***** (observation) *****
        """
        return data

    def validate_observation(self, observation):

    def raw_model_prediction(self, model):
        return mod_prediction_all

    def generate_prediction(self, model, verbose=False):
        return prediction

    def compute_score(self, observation, prediction, verbose=True):
        return self.score

    def bind_score(self, score, model, observation, prediction):
        score.related_data["figures"] = self.figures
        return score
