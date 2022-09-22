import os
import json
import h5py
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from snudda.utils.load import SnuddaLoad
from snudda.neuromodulation.modulation_network import Neuromodulation
from snudda.neuromodulation.neuromodulation import SnuddaSimulateNeuromodulation

from sciunit import Test
import basalunit.capabilities as cap
from basalunit.scores import CustomZScore_NeuroModulation
try:
	from sciunit import ObservationError
except:
	from sciunit.errors import ObservationError
import multiprocessing
import contextlib

class NeuroModulationTest(Test):
	"""
	Evaluates the connection probabilities and compares it to experimental data.

	Parameters
	----------
	observation : dict
		JSON file containing the experimental mean and std values of connectivity for various cell pairs
	"""

	def __init__(self,
				 observation=[],
				 name=None,
				 neuron_type=None,
				 log_file=None):

		self.neuron_type = neuron_type
		observation = self.format_data(observation)
		if not name:
			name = "NeuroModulationTest"
		Test.__init__(self, observation, name)

		self.required_capabilities += (cap.SnuddaBasedModel,)

		self.log_file = log_file

		self.figures = []
		description = "Evaluates the connection probabilities and compares it to experimental data."

	score_type = CustomZScore_NeuroModulation

	def format_data(self, observation):
		"""
		Formats the data to be used in the test
		Adds the neuron_type to the observation
		"""
		return { **observation, "neuron_type": self.neuron_type }

	def validate_observation(self, observation):
		try:
			assert type(observation) is dict
			assert all(key in observation.keys()
					   for key in ["mean", "std"])
		except Exception as e:
			raise ObservationError(("Observation must be of the form "
									"{'mean': X, 'std': Y}"))

	def generate_current_injection(self, network_path):
		# source: https://github.com/Hjorthmedh/Snudda/blob/synaptic_fitting2/examples/notebooks/validation/neuromodulation/neuromodulation_sim.py
		self.sl = SnuddaLoad(os.path.join(network_path, "network-synapses.hdf5"))
		tmp = dict()
		for n in self.sl.data["neurons"]:
			p = os.path.join(n["neuronPath"], "if_info.json")
			import json
			with open(p, "r") as f:
				pdata = json.load(f)
			p = n['parameterKey']
			m = n['morphologyKey']
			nid = n["neuronID"]
			
			idx = np.argmin(np.array(pdata[p][m]['frequency']) - 10)
			current = pdata[p][m]['current'][idx]
			tmp.update({str(nid): current})
		with open(os.path.join(network_path, "current_injection.json"), "w") as f:
			json.dump(tmp,f)
			
		sim_conf = {"sampleDt":0.5e-3}
		with open(os.path.join(network_path, "simulation_config.json"), "w") as f:
			json.dump(sim_conf,f)

	def create_modulation(self, network_path):
		# source: https://github.com/Hjorthmedh/Snudda/blob/synaptic_fitting2/examples/notebooks/validation/neuromodulation/neuromodulation_sim.py

		nl = Neuromodulation()
		nl.set_timestep(dt=0.025)
		nl.set_modulation(neurotransmitter="dopamine", neurotransmitter_key="DA")
		nl.transient(neurotransmitter="dopamine", method="bath_application", duration=3000,
					parameters={"gmax": 0})

		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="dSPN",
								section="soma",
								ion_channels=["kas_ms", "kaf_ms", "can_ms"])
		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="dSPN",
								section="dendrite",
								ion_channels=["kas_ms", "kaf_ms"])

		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="iSPN",
								section="soma",
								ion_channels=["kir_ms", "kas_ms", "kaf_ms", "naf_ms", "cal12_ms", "cal13_ms",
												"can_ms", "car_ms"])
		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="iSPN",
								section="dendrite",
								ion_channels=["kir_ms", "kas_ms", "kaf_ms", "naf_ms", "cal12_ms", "cal13_ms",
												"can_ms", "car_ms"])
		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="iSPN",
								section="axon",
								ion_channels=["kir_ms", "kas_ms", "kaf_ms", "naf_ms", "cal12_ms", "cal13_ms",
												"can_ms", "car_ms"])

		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="FSN",
								section="soma",
								ion_channels=["kir_fs", "kas_fs", "kaf_fs", "naf_fs"])
		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="FSN",
								section="dendrite",
								ion_channels=["kir_fs"])

		nl.save(dir_path=network_path, name="dopamine_modulation_control.json")

		nl = Neuromodulation()
		nl.set_timestep(dt=0.025)
		nl.set_modulation(neurotransmitter="dopamine", neurotransmitter_key="DA")
		nl.transient(neurotransmitter="dopamine", method="bath_application", duration=3000,
					parameters={"gmax": 1})

		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="dSPN",
								section="soma",
								ion_channels=["kas_ms", "kaf_ms", "can_ms"])
		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="dSPN",
								section="dendrite",
								ion_channels=["kas_ms", "kaf_ms"])

		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="iSPN",
								section="soma",
								ion_channels=["kir_ms", "kas_ms", "kaf_ms", "naf_ms", "cal12_ms", "cal13_ms",
												"can_ms", "car_ms"])
		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="iSPN",
								section="dendrite",
								ion_channels=["kir_ms", "kas_ms", "kaf_ms", "naf_ms", "cal12_ms", "cal13_ms",
												"can_ms", "car_ms"])
		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="iSPN",
								section="axon",
								ion_channels=["kir_ms", "kas_ms", "kaf_ms", "naf_ms", "cal12_ms", "cal13_ms",
												"can_ms", "car_ms"])

		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="FSN",
								section="soma",
								ion_channels=["kir_fs", "kas_fs", "kaf_fs", "naf_fs"])
		nl.ion_channel_modulation(neurotransmitter="dopamine",
								cell_type="FSN",
								section="dendrite",
								ion_channels=["kir_fs"])

		nl.save(dir_path=network_path, name="dopamine_modulation.json")

	def simulate(network_path):
		# source: https://github.com/Hjorthmedh/Snudda/blob/synaptic_fitting2/examples/notebooks/validation/neuromodulation/neuromodulation_sim.py
		with open(os.path.join(network_path, "dopamine_modulation.json"), "r") as neuromod_f:
			neuromodulation_dict = json.load(neuromod_f, object_pairs_hook=OrderedDict)
			
		with open(os.path.join(network_path, "current_injection.json"), "r") as f:
			cell_ids_current_injection = json.load(f)
		
		network_file = os.path.join(network_path, "network-synapses.hdf5")
		output_file = os.path.join(network_path, "simulation", "test.hdf5")
		log_file = os.path.join(network_path, "log", "network-simulation-log.txt")
		simulation_config = os.path.join(network_path, "simulation_config.json")
		tSim = 2000
				
		sim = SnuddaSimulateNeuromodulation(network_file=network_file,
														input_file=None,
														output_file=output_file,
														log_file=log_file,
														verbose=True,
														simulation_config=simulation_config)

		sim.setup()
		sim.apply_neuromodulation(neuromodulation_dict)
		sim.neuromodulation_network_wide()
		sim.check_memory_status()
		sim.add_volt_recording_soma()
		
		for neuron_id, data in cell_ids_current_injection.items():

			print(f"Adding to {neuron_id} the amplitude {data}")
			print(f"Within the function to {neuron_id} the amplitude {data*1e9}")
			sim.add_current_injection(neuron_id=int(neuron_id), start_time=0.5, end_time=1.5, amplitude=data)
		
		sim.run(tSim)
		sim.write_output()

	def simulate_control(network_path):
		# source: https://github.com/Hjorthmedh/Snudda/blob/synaptic_fitting2/examples/notebooks/validation/neuromodulation/neuromodulation_sim.py
		with open(os.path.join(network_path, "dopamine_modulation_control.json"), "r") as neuromod_f:
			neuromodulation_dict = json.load(neuromod_f, object_pairs_hook=OrderedDict)
			
		with open(os.path.join(network_path, "current_injection.json"), "r") as f:
			cell_ids_current_injection = json.load(f)
		
		network_file = os.path.join(network_path, "network-synapses.hdf5")
		output_file = os.path.join(network_path, "simulation", "test_control.hdf5")
		log_file = os.path.join(network_path, "log", "network-simulation-log.txt")
		simulation_config = os.path.join(network_path, "simulation_config.json")
		tSim = 2000
				
		sim = SnuddaSimulateNeuromodulation(network_file=network_file,
														input_file=None,
														output_file=output_file,
														log_file=log_file,
														verbose=True,
														simulation_config=simulation_config)

		sim.check_memory_status()
		sim.distribute_neurons()
		sim.pc.barrier()
		sim.setup_neurons()
		sim.apply_neuromodulation(neuromodulation_dict)
		sim.neuromodulation_network_wide()
		sim.check_memory_status()
		sim.add_volt_recording_soma()
		
		for neuron_id, data in cell_ids_current_injection.items():

			print(f"Adding to {neuron_id} the amplitude {data}")
			print(f"Within the function to {neuron_id} the amplitude {data*1e9}")
			sim.add_current_injection(neuron_id=int(neuron_id), start_time=0.5, end_time=1.5, amplitude=data)
		
		sim.run(tSim)
		sim.write_output()

	def analyse(self, network_path):
		# source: https://github.com/Hjorthmedh/Snudda/blob/synaptic_fitting2/examples/notebooks/validation/neuromodulation/neuromodulation_sim.py
		validation = dict(dSPN=dict(mean=6,std=2.8),iSPN=dict(mean=-6,std=3))
		neuron_types = [n["type"] for n in self.sl.data["neurons"]]
		f = h5py.File(os.path.join(network_path, "simulation", "test.hdf5"))
		fc = h5py.File(os.path.join(network_path, "simulation", "test_control.hdf5"))
		tmp = dict()
		for i, n in enumerate(f["neurons"].keys()):
			control_spikes = fc["neurons"][n]["spikes"]["data"][()]
			experiment_spikes = f["neurons"][n]["spikes"]["data"][()]
			neurontype = neuron_types[i]
			if experiment_spikes.size > 0 or control_spikes.size>0:
				diff = experiment_spikes.size - control_spikes.size
				z_score = (diff - validation[neurontype]["mean"]) / validation[neurontype]["std"]
				tmp.update({i : z_score})
		return tmp

	def generate_prediction(self, model):
		"""Implementation of sciunit.Test.generate_prediction."""
		if not self.log_file:
			self.log_file = os.path.join(
				model.network_path, "log", "connectivity.log")

		# with open(self.log_file, "w") as o:
			# with contextlib.redirect_stdout(o):
		self.generate_current_injection(network_path=model.network_path)
		self.create_modulation(network_path=model.network_path)

		pool = multiprocessing.Pool(multiprocessing.cpu_count())
		# necessary to run both in isolation to avoid NEURON errors
		task1 = pool.apply_async(self.simulate_control, args=(self, model.network_path,))
		task2 = pool.apply_async(self.simulate, args=(self, model.network_path,))
		[task.wait() for task in [task1, task2]]
		# self.simulate_control(network_path=model.network_path)
		# self.simulate(network_path=model.network_path)

		f = h5py.File(os.path.join(model.network_path, "simulation", "test.hdf5"))
		fc = h5py.File(os.path.join(model.network_path, "simulation", "test_control.hdf5"))

		# generate figures
		self.figures = []
		fig_dir = os.path.join(model.network_path, "figures")
		if not os.path.exists(fig_dir):
			os.makedirs(fig_dir)
		time = f["time"][()]
		for i, n in enumerate(f["neurons"].keys()):
			name = self.sl.data["neurons"][i]["type"]
			v = f["neurons"][n]["voltage"]["data"][()][0]
			vc = fc["neurons"][n]["voltage"]["data"][()][0]
			plt.figure()
			plt.title(f"{name}")
			plt.plot(time,v)
			plt.plot(time,vc, c="black")
			fig_path = os.path.join(fig_dir, f"{name}_{str(i)}.png")
			plt.savefig(fig_path, dpi=300)
			self.figures.append(fig_path)

		# control_spikes = []
		# experiment_spikes = []
		# for i, n in enumerate(f["neurons"].keys()):
		# 	if neuron_types[i] == self.neuron_type:
		# 		control_spikes.append(fc["neurons"][n]["spikes"]["data"][()])
		# 		experiment_spikes.append(f["neurons"][n]["spikes"]["data"][()])
		
		neuron_types = [n["type"] for n in self.sl.data["neurons"]]
		prediction = {"control": fc, "experiment": f, "neuron_types": neuron_types}
		self.network_path = model.network_path
		return prediction

	def compute_score(self, observation, prediction):
		"""Implementation of sciunit.Test.score_prediction."""
		# Evaluate the score
		# Based on: https://github.dev/Hjorthmedh/Snudda/blob/synaptic_fitting2/examples/notebooks/validation/neuromodulation/neuromodulation_sim.py

		print(observation)
		print(prediction)
		score, ind_z_scores, total_penalty = CustomZScore_NeuroModulation.compute(observation, prediction)
		print(score)
		print(ind_z_scores)
		print(total_penalty)
		summary = {
			"individual_scores_without_penalty": ind_z_scores,
			"total_penalty": total_penalty,
			"final_score": score
		}
		file_name_summary = os.path.join(self.network_path, 'test_summary.json')
		json.dump(summary, open(file_name_summary, "w"), default=str, indent=4)
		return score

	def bind_score(self, score, model, observation, prediction):
		score.observation = observation
		score.prediction = prediction
		score.related_data["figures"] = self.figures
		self.figures.append(os.path.join(self.network_path, 'test_summary.json'))
		return score
