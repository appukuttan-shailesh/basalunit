# Original files:
#   l5pc_model.py by Werner Van Geit at EPFL/Blue Brain Project
#   l5pc_evaluator.py by Werner Van Geit at EPFL/Blue Brain Project
# Modified by Alexander Kozlov <akozlov@kth.se>
# Restructured and modified by Shailesh Appukuttan <shailesh.appukuttan@unic.cnrs-gif.fr>

import os
import glob
import json
import bluepyopt.ephys as ephys
import tarfile
import zipfile
import sciunit
import multiprocessing
import functools
from neuron import h

try:
    import copy_reg
except:
    import copyreg

from types import MethodType

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

try:
	copyreg.pickle(MethodType, _pickle_method, _unpickle_method)
except:
	copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)

# ==============================================================================

class CellModel(sciunit.Model):
    def __init__(self, model_path=None, cell_type=None, hof_index=None, model_name=None):
        # `model_path` is the path to the model's zip file
        if not os.path.isfile(model_path):
            raise IOError("Invalid file path: {}".format(model_path))

        root_path = os.path.dirname(model_path)
        if not model_name:
            file_name = os.path.basename(model_path)
            model_name = file_name.split(".")[0]
        self.model_name = model_name
        self.base_path = os.path.join(root_path, self.model_name)
        self.owd = os.getcwd()     # original working directory saved to return later

        if (model_path.endswith(".zip")):
            file_ref = zipfile.ZipFile(model_path, 'r')
        elif (model_path.endswith(".tar.gz")):
            file_ref = tarfile.open(model_path, "r:gz")
        elif (model_path.endswith(".tar.bz2")):
            file_ref = tarfile.open(model_path, "r:bz2")
        elif (model_path.endswith(".tar")):
            file_ref = tarfile.open(model_path, "r:")
        else:
            raise Exception("Only .zip, .tar, .tar.gz files supported.")
        file_ref.extractall(root_path)
        file_ref.close()

        valid_cell_types = ["msn_d1", "msn_d2", "fs"]
        if cell_type not in valid_cell_types:
            raise ValueError("cell_type has to be from: {}".format(valid_cell_types))

        morphology_path = None
        for myfile in os.listdir(os.path.join(self.base_path, 'morphology')):
            if myfile.endswith(".swc"):
                print("Morphology used: ", myfile)
                morphology_path = os.path.join(self.base_path, 'morphology', myfile)
        if not morphology_path:
            raise Exception("No .swc file found in model's morphology directory.")

        mechanisms_path = os.path.join(self.base_path, 'config/mechanisms.json')
        if not os.path.isfile(mechanisms_path):
            raise Exception("mechanisms.json missing in model's config directory.")

        parameters_path = os.path.join(self.base_path, 'config/parameters.json')
        if not os.path.isfile(parameters_path):
            raise Exception("parameters.json missing in model's config directory.")

        self.load_mod_files()

        self.create(cell_type=cell_type, morph_path=morphology_path,
                    mechs_path=mechanisms_path, params_path=parameters_path)

        super(CellModel, self).__init__()

    def define_mechanisms(self, filename):
        mech_definitions = json.load(open(filename))
        mechanisms = []
        for sectionlist, channels in mech_definitions.iteritems():
            seclist_loc = ephys.locations.NrnSeclistLocation(
                sectionlist,
                seclist_name=sectionlist)
            for channel in channels:
                mechanisms.append(ephys.mechanisms.NrnMODMechanism(
                    name='%s.%s' % (channel, sectionlist),
                    mod_path=None,
                    prefix=channel,
                    locations=[seclist_loc],
                    preloaded=True))
        return mechanisms


    def define_parameters(self, filename):
        param_configs = json.load(open(filename))
        parameters = []
        for param_config in param_configs:
            if 'value' in param_config:
                frozen = True
                value = param_config['value']
                bounds = None
            elif 'bounds':
                frozen = False
                bounds = param_config['bounds']
                value = None
            else:
                raise Exception(
                    'Parameter config has to have bounds or value: %s'
                    % param_config)
            if param_config['type'] == 'global':
                parameters.append(
                    ephys.parameters.NrnGlobalParameter(
                        name=param_config['param_name'],
                        param_name=param_config['param_name'],
                        frozen=frozen,
                        bounds=bounds,
                        value=value))
            elif param_config['type'] in ['section', 'range']:
                if param_config['dist_type'] == 'uniform':
                    scaler = ephys.parameterscalers.NrnSegmentLinearScaler()
                elif param_config['dist_type'] == 'exp':
                    scaler = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
                        distribution=param_config['dist'])
                seclist_loc = ephys.locations.NrnSeclistLocation(
                    param_config['sectionlist'],
                    seclist_name=param_config['sectionlist'])
                name = '%s.%s' % (param_config['param_name'],
                                  param_config['sectionlist'])
                if param_config['type'] == 'section':
                    parameters.append(
                        ephys.parameters.NrnSectionParameter(
                            name=name,
                            param_name=param_config['param_name'],
                            value_scaler=scaler,
                            value=value,
                            frozen=frozen,
                            bounds=bounds,
                            locations=[seclist_loc]))
                elif param_config['type'] == 'range':
                    parameters.append(
                        ephys.parameters.NrnRangeParameter(
                            name=name,
                            param_name=param_config['param_name'],
                            value_scaler=scaler,
                            value=value,
                            frozen=frozen,
                            bounds=bounds,
                            locations=[seclist_loc]))
            else:
                raise Exception(
                    'Param config type has to be global, section or range: %s' %
                    param_config)
        return parameters


    def define_morphology(self, morph_path=None):
        """Define morphology"""
        return ephys.morphologies.NrnFileMorphology(morph_path,
                                                    do_replace_axon=True)


    def create(self, cell_type=None, morph_path=None, mechs_path=None, params_path=None):
        """Create cell model"""
        self.cell = ephys.models.CellModel(
            cell_type,
            morph=self.define_morphology(morph_path),
            mechs=self.define_mechanisms(mechs_path),
            params=self.define_parameters(params_path))

    def load_mod_files(self):
        os.chdir(self.base_path)
        libpath = "x86_64/.libs/libnrnmech.so.0"
        #if not os.path.isfile(os.path.join(self.base_path, libpath)):
        os.system("nrnivmodl mechanisms")   # do nrnivmodl in mechanisms directory
        if not os.path.isfile(os.path.join(self.base_path, libpath)):
            raise IOError("Error in compiling mod files!")
        h.nrn_load_dll(os.path.join(self.base_path, libpath))
        os.chdir(self.owd)

# ==============================================================================


class CellEvaluator(object):
    def __init__(self, cell_model=None, protocol_definitions=None, features=None, params=None):
        opt_params = [p.name for p in cell_model.params.values() if not p.frozen]
        self.protocols = self.define_protocols(protocol_definitions)
        self.calculator = self.define_fitness_calculator(self.protocols, features)

        self.evaluator = ephys.evaluators.CellEvaluator(
            cell_model=cell_model,
            param_names=opt_params,
            fitness_protocols=self.protocols,
            fitness_calculator=self.calculator,
            sim=ephys.simulators.NrnSimulator())

    """
    def load_protocols(self, filename):
        protocol_definitions = json.load(open(filename))
        return protocol_definitions
    """

    def define_protocols(self, protocol_definitions):
        #protocol_definitions = json.load(open(filename))
        protocols = {}
        soma_loc = ephys.locations.NrnSeclistCompLocation(
            name='soma',
            seclist_name='somatic',
            sec_index=0,
            comp_x=0.5)
        for protocol_name, protocol_definition in protocol_definitions.iteritems():
            # By default include somatic recording, could be any
            somav_recording = ephys.recordings.CompRecording(
                name='%s.soma.v' %
                protocol_name,
                location=soma_loc,
                variable='v')
            recordings = [somav_recording]
            if 'extra_recordings' in protocol_definition:
                for recording_definition in protocol_definition['extra_recordings']:
                    if recording_definition['type'] == 'somadistance':
                        location = ephys.locations.NrnSomaDistanceCompLocation(
                            name=recording_definition['name'],
                            soma_distance=recording_definition['somadistance'],
                            seclist_name=recording_definition['seclist_name'])
                        var = recording_definition['var']
                        recording = ephys.recordings.CompRecording(
                            name='%s.%s.%s' % (protocol_name, location.name, var),
                            location=location,
                            variable=recording_definition['var'])
                        recordings.append(recording)
                    else:
                        raise Exception(
                            'Recording type %s not supported' %
                            recording_definition['type'])
            stimuli = []
            for stimulus_definition in protocol_definition['stimuli']:
                stimuli.append(ephys.stimuli.NrnSquarePulse(
                    step_amplitude=stimulus_definition['amp'],
                    step_delay=stimulus_definition['delay'],
                    step_duration=stimulus_definition['duration'],
                    location=soma_loc,
                    total_duration=stimulus_definition['totduration']))
            protocols[protocol_name] = ephys.protocols.SweepProtocol(
                protocol_name,
                stimuli,
                recordings)
        return protocols

    def define_fitness_calculator(self, protocols, features):
        feature_definitions = features
        objectives = []
        for protocol_name, locations in feature_definitions.iteritems():
            for location, features in locations.iteritems():
                for efel_feature_name, meanstd in features.iteritems():
                    feature_name = '%s.%s.%s' % (
                        protocol_name, location, efel_feature_name)
                    recording_names = {'': '%s.%s.v' % (protocol_name, location)}
                    stimulus = protocols[protocol_name].stimuli[0]
                    stim_start = stimulus.step_delay
                    if location == 'soma':
                        threshold = -20
                    elif 'dend' in location:
                        threshold = -55
                    stim_end = stimulus.step_delay + stimulus.step_duration
                    feature = ephys.efeatures.eFELFeature(
                        feature_name,
                        efel_feature_name=efel_feature_name,
                        recording_names=recording_names,
                        stim_start=stim_start,
                        stim_end=stim_end,
                        exp_mean=meanstd[0],
                        exp_std=meanstd[1],
                        threshold=threshold)
                    objective = ephys.objectives.SingletonObjective(
                        feature_name,
                        feature)
                    objectives.append(objective)
        fitcalc = ephys.objectivescalculators.ObjectivesCalculator(objectives)
        return fitcalc

    def run_protocols(self, params):
        """
        npool = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(npool, maxtasksperchild=1)
        run_protocols_ = functools.partial(self.evaluator.run_protocols, param_values=params)
        test_responses = pool.map(run_protocols_, self.protocols.values(), chunksize=1)
        """
        traces = self.evaluator.run_protocols(protocols=self.protocols.values(),
                                              param_values=params)
        return traces
