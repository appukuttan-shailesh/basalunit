import os
import json
import shutil
import sciunit
import sciunit.scores
import pkg_resources
from pprint import pprint
import os.path
import copy
from datetime import datetime
import pickle
import tarfile
import zipfile
import StringIO

from basalunit.utils import CellEvaluator
from basalunit.scores import BU_ZScore
import basalunit.plots as plots

class eFELfeaturesTest(sciunit.Test):
    "Tests somatic features under current injection of varying amplitudes."
    score_type = BU_ZScore

    def __init__(self,
                 observation=None,
                 name="Basal Ganglia - Somatic Features",
                 cell_type=None,
                 base_directory=None,
                 junction_potential=0,
                 use_cache=True):

        # convert he observation data format into zip format
        zipDoc = zipfile.ZipFile(StringIO.StringIO(observation))

        # filename = name.replace(" ", "") + "_{}.zip".format(datetime.now().strftime("%Y-%m-%d"))
        obs_base_dir = os.path.abspath("./temp")
        if not os.path.exists(obs_base_dir):
            os.makedirs(obs_base_dir)

        resp = zipDoc.extractall(obs_base_dir)
        filename_without_ext = zipDoc.namelist()[0][:-1] # remove trailing slash
        zipDoc.close()

        self.observation_dir = os.path.join(obs_base_dir, filename_without_ext)

        # load JSON from the zip file contents
        with open(os.path.join(self.observation_dir, filename_without_ext+".json")) as data_file:
            observation = json.load(data_file)

        cell_types = {"msn_d1":"YJ150915_c67D1ch01D2ch23-c6-protocols",
                      "msn_d2":"YJ150915_c67D1ch01D2ch23-c7-protocols",
                      "fs":"str-fs-161205_FS1-protocols"}
        if not cell_type in cell_types.keys():
            raise TypeError("Invalid cell_type for SomaticFeaturesTest!")
        stim_file = pkg_resources.resource_filename("basalunit", "tests/somafeat_stim/" + cell_types[cell_type] + ".json")
        with open(stim_file, 'r') as f:
			self.protocol_definitions = json.load(f)

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
        self.junction_potential = junction_potential
        self.use_cache = use_cache

    #----------------------------------------------------------------------

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""

        def nested_set(dic, keys, value):
            for key in keys[:-1]:
                dic = dic.setdefault(key, {})
            dic[keys[-1]] = value

        self.model_name = model.model_name
        self.model_version = model.version
        if not self.base_directory:
            self.base_directory = model.base_path

        # Create output directory
        self.path_test_output = os.path.join(self.base_directory, 'validation_results', 'efel_feat', self.model_version, datetime.now().strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(self.path_test_output):
            os.makedirs(self.path_test_output)

        # Verify that observation data for all requested protocols is available, and vice versa
        no_observations = list(set(self.protocol_definitions) - set(self.observation))
        if len(no_observations) > 0:
            raise ValueError("No observation data found for following protocols: ".format(no_observations))
        no_protocols = list(set(self.observation) - set(self.protocol_definitions))
        for key in no_protocols:
            self.observation.pop(key)
        observations_new = copy.deepcopy(self.observation)
        protcols_new = copy.deepcopy(self.protocol_definitions)

        model.model_hash = model.hash
        cache_path = os.path.abspath(os.path.join(self.path_test_output, "../cache", model.model_hash))
        cached_traces = {}
        cached_features = {}
        if self.use_cache:
            if not os.path.exists(os.path.join(cache_path)):
                print("Note: no cached data for this model specification (hash)!")
            else:
                print("***** Using cache to retrieve relevant model data *****")
                print("Cached data found for following protocols: ")
                for key in self.observation.keys():
                    if (os.path.exists(os.path.join(cache_path, key+"_feats.json")))  and (os.path.exists(os.path.join(cache_path, key+"_trace.pkl"))):
                        print("\t"+key)
                        with open(os.path.join(cache_path, key+"_feats.json"), 'r') as f1, open(os.path.join(cache_path, key+"_trace.pkl"), 'r') as f2:
                            cached_features[key] = json.load(f1)
                            cached_traces[key+".soma.v"] = pickle.load(f2)
                        observations_new.pop(key)
                        protcols_new.pop(key)

        self.cell_evaluator = CellEvaluator(cell_model=model.cell,
                                            protocol_definitions=protcols_new,
                                            features=observations_new,
                                            params=model.params)
        self.pred_traces = self.cell_evaluator.run_protocols(params=model.params)
        prediction = {}
        # construct prediction with structure similar to observation
        for objective in self.cell_evaluator.calculator.objectives:
            for feature in objective.features:
                # e.g. recording name = `APWaveform_548.soma.v`
                # this is changed to {APWaveform_548: {soma: {efel_feature_name: feature_value} } }
                feat_name_parts = feature.recording_names.values()[0].split('.')
                nested_set(prediction, [feat_name_parts[0], feat_name_parts[1], feature.efel_feature_name], feature.calculate_feature(self.pred_traces))

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        for key in observations_new.keys():
            with open(os.path.join(cache_path, key+"_feats.json"), 'w') as f:
                json.dump(prediction[key], f, indent=4, sort_keys=True)
            with open(os.path.join(cache_path, key+"_trace.pkl"), 'w') as f:
                pickle.dump(self.pred_traces[key+".soma.v"], f)
        prediction.update(cached_features)
        self.pred_traces.update(cached_traces)

        # Saving current model "hall of fame" parameters to `test instance`
        self.model_params = model.params
        return prediction

    #----------------------------------------------------------------------

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        self.score, self.score_dict, self.prob_list = BU_ZScore.compute(observation, prediction)
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
