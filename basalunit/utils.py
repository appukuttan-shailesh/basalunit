# Original files:
#   l5pc_model.py by Werner Van Geit at EPFL/Blue Brain Project
#   l5pc_evaluator.py by Werner Van Geit at EPFL/Blue Brain Project
# Modified by Alexander Kozlov <akozlov@kth.se>
# Restructured and modified by Shailesh Appukuttan <shailesh.appukuttan@unic.cnrs-gif.fr>

from __future__ import print_function, division

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

from joblib import Parallel, delayed
import multiprocessing
import numpy                as np
import matplotlib.pyplot    as plt
from math import exp

try:
    import copy_reg
except:
    import copyreg

from types import MethodType

def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = method.__self__.__class__
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
        for sectionlist, channels in mech_definitions.items():
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
        libpath = "x86_64/.libs/libnrnmech.so"
        #if not os.path.isfile(os.path.join(self.base_path, libpath)):
        os.system("nrnivmodl mechanisms")   # do nrnivmodl in mechanisms directory
        if not os.path.isfile(os.path.join(self.base_path, libpath)):
            raise IOError("Error in compiling mod files!")
        h.nrn_load_dll(str(os.path.join(self.base_path, libpath)))
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
        for protocol_name, protocol_definition in protocol_definitions.items():
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
        for protocol_name, locations in feature_definitions.items():
            for location, features in locations.items():
                for efel_feature_name, meanstd in features.items():
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

# ==============================================================================

# Original files by R. Lindroos and A. Koslov at https://github.com/ModelDBRepository/237653:
#   MSN_builder.py
#   fig2_validation.py
#   plot_functions.py
# Restructured and modified by Pedro Garcia-Rodriguez <pedro.garcia@cnrs.fr>

h.load_file('stdlib.hoc')
h.load_file('import3d.hoc')

class CellModel_Lindroos2018(sciunit.Model):

    def __init__(self, model_path=None, morph_file=None, params_file=None):

        if not model_path:
            raise ValueError("Please specify the path to the model directory!")
        if not os.path.isdir(model_path):
            raise ValueError("Specified path to the model directory is invalid!")
        self.model_path = model_path
        try:
            h.nrn_load_dll(os.path.join(self.model_path, 'x86_64/.libs/libnrnmech.so'))
        except:
            pass

        if not params_file:
            params_file = "params_dMSN.json"
            self.params_file = os.path.join(self.model_path, params_file)
        if not morph_file:
            morph_file = 'latest_WT-P270-20-14ak.swc'
            self.morph_file = os.path.join(self.model_path, morph_file)


    def save_vector(self, t, v, outfile):

        with open(outfile, "w") as out:
            for time, y in zip(t, v):
                out.write("%g %g\n" % (time, y))


    def load_obj(name ):
        with open(name, 'rb') as f:
            return pickle.load(f)


    def main(self,  sim='vm',       \
                    amp=0.265,      \
                    run=None,       \
                    simDur=1000,    \
                    stimDur=900     ):

        # initiate cell
        cell    =   MSN( params=self.params_file, morphology=self.morph_file )

        # set cascade--not activated in this script,
        # but used for setting pointers needed in the channel mechnisms
        casc    =   h.D1_reduced_cascade2_0(0.5, sec=cell.soma)


        # set pointer target in cascade
        pointer =   casc._ref_Target1p


        # set edge of soma as reference for dendritic distance
        h.distance(1, sec=h.soma[0])


        # set current injection
        stim        =   h.IClamp(0.5, sec=cell.soma)
        stim.amp    =   amp
        stim.delay  =   100
        stim.dur    =   stimDur


        # record vectors
        tm  = h.Vector()
        tm.record(h._ref_t)
        vm  = h.Vector()
        vm.record(cell.soma(0.5)._ref_v)

        tstop       = simDur
        # dt = default value; 0.025 ms (25 us)


        # set pointers; need since same mechanisms are used for dynamic modulation of channels.
        # Modulation of channels is not used in this script
        for sec in h.allsec():

            for seg in sec:


                # naf and kas is in all sections
                h.setpointer(pointer, 'pka', seg.kas )
                h.setpointer(pointer, 'pka', seg.naf )

                if sec.name().find('axon') < 0:


                    # these channels are not in the axon section
                    h.setpointer(pointer, 'pka', seg.kaf )
                    h.setpointer(pointer, 'pka', seg.cal12 )
                    h.setpointer(pointer, 'pka', seg.cal13 )
                    h.setpointer(pointer, 'pka', seg.kir )

                    if sec.name().find('soma') >= 0:


                        # N-type Ca (can) is only distributed to the soma section
                        h.setpointer(pointer, 'pka', seg.can )


        # configure simulation to record from both calcium pools.
        # the concentration is here summed, instead of averaged.
        # This doesn't matter for the validation fig, since relative concentration is reported.
        # For Fig 5B, where concentration is reported, this is fixed when plotting.
        # -> see the plot_Ca_updated function in plot_functions.
        if sim == 'ca':

            print('configure', sim, 'simulation')

            for i,sec in enumerate(h.allsec()):

                if sec.name().find('axon') < 0: # don't record in axon

                    for j,seg in enumerate(sec):

                        sName = sec.name().split('[')[0]


                        # N, P/Q, R Ca pool
                        cmd = 'ca_%s%s_%s = h.Vector()' % (sName, str(i), str(j))
                        exec(cmd)
                        cmd = 'ca_%s%s_%s.record(seg._ref_cai)' % (sName, str(i), str(j))
                        exec(cmd)

                        # the L-type Ca
                        cmd = 'cal_%s%s_%s = h.Vector()' % (sName, str(i), str(j))
                        exec(cmd)
                        cmd = 'cal_%s%s_%s.record(seg._ref_cali)' % (sName, str(i), str(j))
                        exec(cmd)


                        # uncomment here if testing kaf blocking effect on bAP
                        #block_fraction = 0.2
                        #gbar           = seg.kaf.gbar
                        #seg.kaf.gbar   = (1 - block_fraction) * gbar



        # solver------------------------------------------------------------------------------
        cvode = h.CVode()

        h.finitialize(cell.v_init)

        # run simulation
        while h.t < tstop:

            h.fadvance()


        # save output ------------------------------------------------------------------------

        if sim == 'ca':

            print('saving', sim, 'simulation')

            # vm
            self.save_vector(tm, vm, ''.join([ self.model_path, '/Results/Ca/vm_', sim, '_', str(int(amp*1e3)), '.out']) )

            # ca
            for i,sec in enumerate(h.allsec()):

                if sec.name().find('axon') < 0:

                    for j,seg in enumerate(sec):


                        sName       =   sec.name().split('[')[0]
                        vName       =   'ca_%s%s_%s'  %  ( sName, str(i), str(j)  )
                        v2Name      =   'cal_%s%s_%s' %  ( sName, str(i), str(j)  )
                        fName       =   '%s/Results/Ca/ca_%s_%s.out'  %  ( self.model_path, str(int(np.round(h.distance(cell.soma(0.5), sec(seg.x))))), vName )

                        cmd = 'self.save_vector(tm, np.add(%s, %s), %s)' % (vName, v2Name, 'fName' ) # this is were concentrations are summed (see above)

                        exec(cmd)

        elif sim == 'vm':

            print('saving', sim, 'simulation', str(int(amp*1e3)))

            # vm
            self.save_vector(tm, vm, ''.join([self.model_path, '/Results/FI/vm_', sim, '_', str(int(amp*1e3)), '.out']) )


    def somatic_excitability(self):

        print('starting somatic excitability simulation')

        # somatic excitability (validated against FI curves in Planert et al., 2013)
        currents    = np.arange(-100,445,40)
        num_cores   = multiprocessing.cpu_count()

        Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(self.main)(   \
                                                    amp=current*1e-3,           \
                                                    run=1,                      \
                                                    simDur=1000,                \
                                                    stimDur=900                 \
                                                ) for current in currents)
        currents    = np.arange(320,445,40)
        Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(self.main)(   \
                                                    amp=current*1e-3,           \
                                                    run=1,                      \
                                                    simDur=1000,                \
                                                    stimDur=900                 \
                                                ) for current in currents)

        print('all simulations done!')


    def plot_vm(self):

        '''
        Reproduces the base plots (validation: figure 2 B and C) for the frontiers paper.

        '''

        files  = glob.glob(os.path.join(self.model_path, 'Results/FI/vm_vm_*'))

        f1,a1  = plt.subplots(1,1)
        fig,ax = plt.subplots(1,1)

        res = {}

        num_cores = multiprocessing.cpu_count() #int(np.ceil(multiprocessing.cpu_count() / 2))
        M = Parallel(n_jobs=num_cores)(delayed(self.loadFile)( f ) for f in files)

        least_spikes = 100
        last_subThresh = 0

        inj = np.arange(-100,345,40)

        # sort traces in spiking vs non spiking
        for i,m in enumerate(M):

            if len(m[0]) > 0:

                res[m[1]] = (len(m[0]) -1) * ( 1000 / (m[0][-1]-m[0][0]) )

                if len(m[0]) < least_spikes:

                    least_spikes = len(m[0])
                    rheob_index = i

            else:

                if m[1] in inj:
                    a1.plot(m[2][0], m[2][1], 'grey')

                if m[1] > last_subThresh:

                    last_subThresh = m[1]

        # add last trace without spike
        res[last_subThresh] = 0

        # plot first trace to spike
        a1.plot(M[rheob_index][2][0], M[rheob_index][2][1], 'k', lw=2)
        a1.plot([10,110], [-20, -20], 'k')
        a1.plot([10, 10], [-20, -10], 'k')
        a1.axis('off')

        # FI curve ---------------------------------------------------------------------------------------
        exp_data_path = os.path.join(self.model_path, 'Exp_data/FI/Planert2013-D1-FI-trace*')
        for i,f in enumerate(glob.glob(exp_data_path)):

            [x_i,y_i] = np.loadtxt(f, unpack=True)

            if i == 0:
                label = 'D1planert'
            else:
                label = ''

            ax.plot(x_i, y_i, 'brown', lw=5, alpha=1, label=label, clip_on=False)
            #plt.legend()


        AMP = []
        inj = np.arange(last_subThresh,480,40)

        for I in inj:

            AMP.append(res[I])

        ax.plot(inj, AMP, color='k', lw=7, label='neuron', clip_on=False)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # set x-range
        ax.set_xlim([150, 800])

        # set ticks
        yticks = np.arange(10,31,10)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=30)
        xticks = np.arange(150,760,150)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=30)

        ax.tick_params(width=3, length=6)

        # size of frame
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(4)

        plt.show()
        fig_path = os.path.join(self.model_path, 'Figures', 'FI_profiles.png' )
        fig.savefig(fig_path)

        return [fig_path]


    def dendrite_BAP(self):
        # dendritic validation: change in [Ca] following a bAP (validated against Day et al., 2008)
        current = 2000
        self.main(  amp=current*1e-3,           \
                    simDur=200,                 \
                    stimDur=2,                  \
                    sim='ca'                    )


    def get_Ca(self, fString='Results/Ca/ca*.out'):

        '''
        gets resulting Ca as distance dependent using all traces matching fString
        '''

        files_path = os.path.join(self.model_path, fString)
        files       = glob.glob(files_path)
        N           = len(files)
        res = {}
        distances = np.arange(0,200, 10)

        num_cores = multiprocessing.cpu_count() #int(np.ceil(multiprocessing.cpu_count() / 2))
        M = Parallel(n_jobs=num_cores)(delayed(self.get_max)( f ) for f in files)
        for m in M:
            for d in distances:
                if m[1] > d-5 and m[1] < d+5:
                    if d not in res:
                        res[d] = []
                    res[d].append(m[0])
                    break

        mean_amp    = []
        x           = []
        for d in distances:
            if d in res:
                mean_amp.append( np.mean(res[d]) )
                x.append(d)

        return [ np.array(x), mean_amp ]


    def plot_Ca(self, fString='Results/Ca/ca*.out'):

        '''
        plots resulting Ca as distance dependent using all traces matching fString
        '''
        files_path = os.path.join(self.model_path, fString)

        files       = glob.glob(files_path)

        fig, ax = plt.subplots(1,1, figsize=(6,8))

        [x, mean_amp] = self.get_Ca(files_path)

        num_cores = multiprocessing.cpu_count()
        M = Parallel(n_jobs=num_cores)(delayed(self.get_max)( f ) for f in files)
        for m in M:
            if m[1] >= 40:
                ax.plot(m[1], np.divide(m[0], mean_amp[3]), '.', ms=20, color='k', alpha=0.2)

        mean_amp = np.divide(mean_amp, mean_amp[3])
        ax.plot(x[3:], mean_amp[3:], lw=6, color='k', label='Model prediction')
        ax.legend(fontsize=18)
        # ax.legend(['Experimental data', 'Model prediction'])
        # ax.legend(['Experimental data'])

        #day et al 2008
        exp_data_file = os.path.join(self.model_path, 'Exp_data/bAP/bAP-DayEtAl2006-D1.csv')
        [x1,y1] = np.loadtxt(exp_data_file, unpack=True)
        ax.plot(x1, y1, 'brown', lw=6, label='Experimental data')
        ax.legend(fontsize=18)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # set ticks
        yticks = [0,1]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=20)
        xticks = [40, 120, 200]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=20)

        ax.tick_params(width=2, length=4)

        ax.set_ylabel('Normalized Ca amplitude', fontsize=25)
        ax.set_xlabel('Somatic distance (µm)', fontsize=25)
        ax.set_title(r'$\Delta$Ca concentration' + '\n following a bAP', fontsize=30)
        # size of frame
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(4)

        f,a = plt.subplots(1,1, figsize=(2,4))

        path = os.path.dirname(files[0])

        path_file = ''.join([path, '/vm_ca_2000.out'])

        f = glob.glob( path_file )

        lw=4
        a.plot(*np.loadtxt(f[0], unpack=True), color='k', lw=lw )

        # construct I curve

        base = -110
        a.plot([0,100], [base, base], 'k', lw=lw)
        a.plot([100,100],[base, -90], 'k', lw=lw)
        a.plot([100,102], [-90, -90], 'k', lw=lw)
        a.plot([102,102],[-90, base], 'k', lw=lw)
        a.plot([102,155], [base, base], 'k', lw=lw)


        a.set_ylim([base-1, 50])
        a.set_xlim([95, 120])

        a.axis('off')
        plt.show()

        fig_path = os.path.join(self.model_path, 'Figures', 'Ca_BPA.png' )
        fig.savefig(fig_path)

        return [fig_path]


    def get_max(self, f):

        x,y = np.loadtxt(f, unpack=True)

        m   = max(y[3300:-1])

        path_file = os.path.split(f)
        dist = int(path_file[1].split('_')[1])

        return [m, dist, x[3300:-1], y[3300:-1]]


    def loadFile(self, f):

        x,y = np.loadtxt(f, unpack=True)

        amp = int(f.split('/')[-1].split('_')[2].split('.')[0])
        #print (f, amp)

        return [self.getSpikedata_x_y(x,y), amp, [x,y] ]


    def getSpikedata_x_y(self, x,y):

        '''
        getSpikedata_x_y(x,y) -> return count

        Extracts and returns the number of spikes from spike trace data.

        # arguments
        x = time vector
        y = vm vector

        # returns
        count = number of spikes in trace (int)

        # extraction algorithm
        -threshold y and store list containing index for all points larger than 0 V
        -sorts out and counts the index that are the first one(s) crossing the threshold, i.e.
            the first index of each spike. This is done by looping over all index and check if
            the index is equal to the previous index + 1. If not it is the first index of a
            spike.

            If no point is above threshold in the trace the function returns 0.

        '''

        count = 0
        spikes = []

        # pick out index for all points above zero potential for potential trace
        spikeData = [i for i,v in enumerate(y) if v > 0]

        # if no point above 0
        if len(spikeData) == 0:

            return spikes

        else:
            # pick first point of each individaul transient (spike)...
            for j in range(0, len(spikeData)-1):
                if j==0:

                    count += 1
                    spikes.append(x[spikeData[j]])

                # ...by checking above stated criteria
                elif not spikeData[j] == spikeData[j-1]+1:
                    count += 1
                    spikes.append(x[spikeData[j]])

        return spikes

# ======================= the MSN class ==================================================

class MSN:
    def __init__(self,  params=None, \
                        morphology='WT-P270-20-14ak_1.03_SGA2-m12.swc'):

        Import = h.Import3d_SWC_read()
        Import.input(morphology)
        imprt = h.Import3d_GUI(Import, 0)
        imprt.instantiate(None)
        h.define_shape()
        # h.cao0_ca_ion = 2  # default in nrn
        h.celsius = 35
        self._create_sectionlists()
        self._set_nsegs()
        self.v_init = -80

        for sec in self.somalist:
            for mech in [
                    "naf",
                    "kaf",
                    "kas",
                    "kdr",
                    "kir",
                    "cal12",
                    "cal13",
                    "can",
                    "car",
                    "cadyn",
                    "caldyn",
                    "sk",
                    "bk"
                ]:
                sec.insert(mech)

        for sec in self.axonlist:
            for mech in [
                    "naf",
                    "kas"
                ]:
                sec.insert(mech)

        for sec in self.dendlist:
            for mech in [
                    "naf",
                    "kaf",
                    "kas",
                    "kdr",
                    "kir",
                    "cal12",
                    "cal13",
                    "car",
                    "cat32",
                    "cat33",
                    "cadyn",
                    "caldyn",
                    "sk",
                    "bk"
                ]:
                sec.insert(mech)

        for sec in self.allseclist:
            sec.Ra = 150
            sec.cm = 1.0
            sec.insert('pas')
            #sec.g_pas = 1e-5 # set using json file
            sec.e_pas = -70 # -73
            sec.ena = 50
            sec.ek = -85 # -90

        with open(params) as file:
            par = json.load(file)

        self.distribute_channels("soma", "g_pas", 0, 1, 0, 0, 0, float(par['g_pas_all']['Value']))
        self.distribute_channels("axon", "g_pas", 0, 1, 0, 0, 0, float(par['g_pas_all']['Value']))
        self.distribute_channels("dend", "g_pas", 0, 1, 0, 0, 0, float(par['g_pas_all']['Value']))

        self.distribute_channels("soma", "gbar_naf", 0, 1, 0, 0, 0, float(par['gbar_naf_somatic']['Value']))
        self.distribute_channels("soma", "gbar_kaf", 0, 1, 0, 0, 0, float(par['gbar_kaf_somatic']['Value']))
        self.distribute_channels("soma", "gbar_kas", 0, 1, 0, 0, 0, float(par['gbar_kas_somatic']['Value']))
        self.distribute_channels("soma", "gbar_kdr", 0, 1, 0, 0, 0, float(par['gbar_kdr_somatic']['Value']))
        self.distribute_channels("soma", "gbar_kir", 0, 1, 0, 0, 0, float(par['gbar_kir_somatic']['Value']))
        self.distribute_channels("soma", "gbar_sk",  0, 1, 0, 0, 0, float(par['gbar_sk_somatic']['Value']))
        self.distribute_channels("soma", "gbar_bk",  0, 1, 0, 0, 0, float(par['gbar_bk_somatic']['Value']))

        self.distribute_channels("axon", "gbar_naf", 3, 1, 1.1, 30, 500, float(par['gbar_naf_axonal']['Value']))
        self.distribute_channels("axon", "gbar_kas", 0, 1, 0, 0, 0,      float(par['gbar_kas_axonal']['Value']))
        self.distribute_channels("dend", "gbar_naf", 1, 0.1, 0.9,   60.0,   10.0, float(par['gbar_naf_basal']['Value']))
        self.distribute_channels("dend", "gbar_kaf", 1,   1, 0.5,  120.0,  -30.0, float(par['gbar_kaf_basal']['Value']))

        self.distribute_channels("dend", "gbar_kas", 2,   1, 9.0,  0.0, -5.0, float(par['gbar_kas_basal']['Value']))
        self.distribute_channels("dend", "gbar_kdr", 0, 1, 0, 0, 0, float(par['gbar_kdr_basal']['Value']))
        self.distribute_channels("dend", "gbar_kir", 0, 1, 0, 0, 0, float(par['gbar_kir_basal']['Value']))
        self.distribute_channels("dend", "gbar_sk",  0, 1, 0, 0, 0, float(par['gbar_sk_basal']['Value']))
        self.distribute_channels("dend", "gbar_bk",  0, 1, 0, 0, 0, float(par['gbar_bk_basal']['Value']))

        self.distribute_channels("soma", "pbar_cal12", 0, 1, 0, 0, 0, 1e-5)
        self.distribute_channels("soma", "pbar_cal13", 0, 1, 0, 0, 0, 1e-6)
        self.distribute_channels("soma", "pbar_car",   0, 1, 0, 0, 0, 1e-4)
        self.distribute_channels("soma", "pbar_can",   0, 1, 0, 0, 0, 3e-5)
        self.distribute_channels("dend", "pbar_cal12", 0, 1, 0, 0, 0, 1e-5)
        self.distribute_channels("dend", "pbar_cal13", 0, 1, 0, 0, 0, 1e-6)
        self.distribute_channels("dend", "pbar_car",   0, 1, 0, 0, 0, 1e-4)
        self.distribute_channels("dend", "pbar_cat32", 1, 0, 1.0, 120.0, -30.0, 1e-7)
        self.distribute_channels("dend", "pbar_cat33", 1, 0, 1.0, 120.0, -30.0, 1e-8)


    def _create_sectionlists(self):
        self.allsecnames = []
        self.allseclist  = h.SectionList()
        for sec in h.allsec():
            self.allsecnames.append(sec.name())
            self.allseclist.append(sec=sec)
        self.nsomasec = 0
        self.somalist = h.SectionList()
        for sec in h.allsec():
            if sec.name().find('soma') >= 0:
                self.somalist.append(sec=sec)
                if self.nsomasec == 0:
                    self.soma = sec
                self.nsomasec += 1
        self.axonlist = h.SectionList()
        for sec in h.allsec():
            if sec.name().find('axon') >= 0:
                self.axonlist.append(sec=sec)
        self.dendlist = h.SectionList()
        for sec in h.allsec():
            if sec.name().find('dend') >= 0:
                self.dendlist.append(sec=sec)


    def _set_nsegs(self):
        for sec in self.allseclist:
            sec.nseg = 2*int(sec.L/40.0)+1
        for sec in self.axonlist:
            sec.nseg = 2  # two segments in axon initial segment


    def distribute_channels(self, as1, as2, d3, a4, a5, a6, a7, g8):
        h.distance(sec=self.soma)

        for sec in self.allseclist:

            # if right cellular compartment (axon, soma or dend)
            if sec.name().find(as1) >= 0:
                for seg in sec:
                    dist = h.distance(seg.x, sec=sec) - 7.06 + 5.6
                    val = self.calculate_distribution(d3, dist, a4, a5, a6, a7, g8)
                    cmd = 'seg.%s = %g' % (as2, val)
                    exec(cmd)


    def calculate_distribution(self, d3, dist, a4, a5,  a6,  a7, g8):
        '''
        Used for setting the maximal conductance of a segment.
        Scales the maximal conductance based on somatic distance and distribution type.

        Parameters:
        d3   = distribution type:
             0 linear,
             1 sigmoidal,
             2 exponential
             3 step function
        dist = somatic distance of segment
        a4-7 = distribution parameters
        g8   = base conductance (similar to maximal conductance)

        '''
        # Distributions:
        '''
        T-type Ca: g = 1.0/( 1 +np.exp{(x-70)/-4.5} )
        naf (den): (0.1 + 0.9/(1 + np.exp((x-60.0)/10.0)))

        '''

        if   d3 == 0:
            value = a4 + a5*dist
        elif d3 == 1:
            value = a4 + a5/(1 + exp((dist-a6)/a7) )
        elif d3 == 2:
            value = a4 + a5*exp((dist-a6)/a7)
        elif d3 == 3:
            if (dist > a6) and (dist < a7):
                value = a4
            else:
                value = a5

        if value < 0:
            value = 0

        value = value*g8
        return value
