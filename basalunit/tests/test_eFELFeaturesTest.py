import os
import json
import shutil
import sciunit
import sciunit.scores
import pkg_resources
from pprint import pprint

from basalunit.utils import CellEvaluator
from basalunit.scores import BU_ZScore
import basalunit.plots as plots

class eFELfeaturesTest(sciunit.Test):
    "Tests somatic features under current injection of varying amplitudes."
    score_type = BU_ZScore

    def __init__(self,
                 observation={},
                 name="Basal Ganglia - Somatic Features",
                 observation_dir=None,
                 cell_type=None,
                 base_directory=None,
                 junction_potential=0):
        cell_types = {"msn_d1":"YJ150915_c67D1ch01D2ch23-c6-protocols",
                      "msn_d2":"YJ150915_c67D1ch01D2ch23-c7-protocols"}
        if not cell_type in cell_types.keys():
            raise TypeError("Invalid cell_type for SomaticFeaturesTest!")
        self.stim_file = pkg_resources.resource_filename("basalunit", "tests/somafeat_stim/" + cell_types[cell_type] + ".json")
        #with open(stim_file, 'r') as f:
		#	self.config = json.load(f)

        description = ("Tests somatic features under current injection of varying amplitudes.")
        #XXXXXXXXXXXXXXXXXXXXXself.required_capabilities = (cap.ProvidesDensityInfo,)

        self.figures = []
        """
        ##### Testing - Reduce Tests
        for key in observation.keys():
            if not key.startswith(("IDrest_", "IDthresh_", "IV_")):
                observation.pop(key)
        ##### Testing - Reduce Tests
        """
        sciunit.Test.__init__(self, observation, name)
        self.base_directory = base_directory
        self.observation_dir = observation_dir
        self.junction_potential = junction_potential

    #----------------------------------------------------------------------

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""

        def nested_set(dic, keys, value):
            for key in keys[:-1]:
                dic = dic.setdefault(key, {})
            dic[keys[-1]] = value

        self.model_name = model.name
        if not self.base_directory:
            self.base_directory = model.base_path
        self.directory_output = os.path.join(self.base_directory, 'validation_results')

        self.cell_evaluator = CellEvaluator(cell_model=model.cell,
                                            protocols_path=self.stim_file,
                                            features=self.observation,
                                            params=model.params)
        self.pred_traces = self.cell_evaluator.run_protocols(params=model.params)
        prediction = {}
        # construct prediction with structure similar to observation
        for objective in self.cell_evaluator.calculator.objectives:
            for feature in objective.features:
                # e.g. recording name = `APWaveform_548.soma.v`
                # this should be changed to {APWaveform_548: {soma: {efel_feature_name: feature_value} } }
                feat_name_parts = feature.recording_names.values()[0].split('.')
                nested_set(prediction, [feat_name_parts[0], feat_name_parts[1], feature.efel_feature_name], feature.calculate_feature(self.pred_traces))
        # Saving current model "hall of fame" parameters to `test instance`
        self.model_params = model.params

        # print "======================================1"
        # print "self.observation  = ", self.observation # 61
        # print "self.pred_traces  = ", self.pred_traces # 61
        # print "prediction  = ", prediction             # 61
        # print "======================================2"
        return prediction

    #----------------------------------------------------------------------

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        self.score, self.score_dict, self.prob_list = BU_ZScore.compute(observation, prediction)
        # print "======================================3"
        # print "self.score  = ", self.score
        # print "self.score_dict  = ", self.score_dict
        # print "self.prob_list  = ", self.prob_list
        # print "======================================4"

        # create output directory
        self.path_test_output = os.path.join(self.directory_output, 'efel_feat', self.model_name)
        if not os.path.exists(self.path_test_output):
            os.makedirs(self.path_test_output)

        self.observation = observation
        self.prediction = prediction

        # create relevant output files
        # 1. Simulation (and experimental) traces
        with open(os.path.join(self.observation_dir, "trace_protocol_mapping.json")) as data_file:
            self.expdata = json.load(data_file)
        self.exp_trace_dir = os.path.join(self.observation_dir, os.path.basename(self.observation_dir))
        trace_plot = plots.ModelExpTraces(self)
        trace_file = trace_plot.create()
        self.figures.append(trace_file)
        # 2. Saving model "hall of fame" parameters
        hof_params_file = os.path.join(self.path_test_output, "hof_params.json")
        with open(hof_params_file, 'w') as outfile:
            pprint(self.model_params, stream=outfile)
        self.figures.append(hof_params_file)
        # 3. Text Table
        txt_table = plots.TxtTable(self)
        table_file = txt_table.create()
        self.figures.append(table_file)
        # 4. JSON data
        json_data = plots.JsonData(self)
        json_file = json_data.create()
        self.figures.append(json_file)
        # 5. Plotting relative errors
        rel_err_plot = plots.ErrorRelative(self)
        rel_err_file = rel_err_plot.create()
        self.figures.append(rel_err_file)
        # 6. Plotting relative errors
        rel_abs_plot = plots.ErrorAbsolute(self)
        rel_abs_file = rel_abs_plot.create()
        self.figures.append(rel_abs_file)

        return BU_ZScore(self.score)

    #----------------------------------------------------------------------

    def bind_score(self, score, model, observation, prediction):
        score.related_data["figures"] = self.figures
        shutil.rmtree(os.path.dirname(self.observation_dir))
        return score
