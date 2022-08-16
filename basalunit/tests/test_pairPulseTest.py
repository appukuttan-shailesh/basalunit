from array import typecodes
import os
import json
from sciunit import Test
import basalunit.capabilities as cap
from basalunit.scores import KLdivMeanStd
from quantities import mV
from quantities.quantity import Quantity
try:
	from sciunit import ObservationError
except:
	from sciunit.errors import ObservationError
from snudda.simulate.network_pair_pulse_simulation import SnuddaNetworkPairPulseSimulation


class PairPulseTest(Test):
	"""
	Evaluates statistics of connections in the striatal microcircuitry and compares it to experimental data.

	Parameters
	----------
	observation : dict
		JSON file containing the experimental mean and std values of connectivity for various cell pairs
	"""

	def __init__(self,
				 observation=[],
				 name=None,
				 pre_type=None,
				 post_type=None,
				 hold_v=None,
				 max_dist=None,
				 GABA_rev=None,
				 cur_inj=None,
				 log_file=None):

		observation = self.format_data(observation)
		if not name:
			name = "PairPulseTest: {} -> {}".format(pre_type, post_type)
		Test.__init__(self, observation, name)

		self.required_capabilities += (cap.SnuddaBasedModel,)

		self.pre_type = pre_type
		self.post_type = post_type
		self.hold_v = hold_v
		self.max_dist = max_dist
		self.GABA_rev = GABA_rev
		self.curr_inj = cur_inj
		self.log_file = log_file

		self.figures = []
		description = "Evaluates statistics of connections in the striatal microcircuitry and compares it to experimental data."

	score_type = KLdivMeanStd

	def format_data(self, observation):
		# target format:
		# {"mean": X * mV, "std": Y * mV}
		for key, val in observation.items():
			if type(val) is Quantity:
				observation[key] = val
			elif type(val) is float or type(val) is int:
				observation[key] = val * mV
			else:
				quantity_parts = val.split(" ")
				number = float(quantity_parts[0])
				units = " ".join(quantity_parts[1:])
				observation[key] = Quantity(number, units)
		return observation

	def validate_observation(self, observation):
		try:
			assert type(observation) is dict
			assert all(key in observation.keys()
					   for key in ["mean", "std"])
			for key, val in observation.items():
				assert type(val) is Quantity
		except Exception as e:
			raise ObservationError(("Observation must be of the form "
									"{'mean': X * mV, 'std': Y * mV}"))

	def generate_prediction(self, model, verbose=False):
		"""Implementation of sciunit.Test.generate_prediction."""
		if not self.log_file:
			self.log_file = os.path.join(
				model.network_path, "log", "pair-pulse.log")

		pps = SnuddaNetworkPairPulseSimulation(network_path=model.network_path,
											   exp_type="Planert2010",
											   pre_type=self.pre_type,
											   post_type=self.post_type,
											   max_dist=self.max_dist,
											   hold_voltage=self.hold_v,
											   current_injection=self.curr_inj)

		pps.run_sim(gaba_rev=self.GABA_rev)
		model_mean, model_std, trace_fig, hist_fig = pps.analyse(post_type=self.post_type)

		prediction = {"mean": model_mean, "std": model_std}
		if trace_fig:
			self.figures.append(trace_fig)
		if hist_fig:
			self.figures.append(hist_fig)
		return prediction

	def compute_score(self, observation, prediction, verbose=False):
		"""Implementation of sciunit.Test.score_prediction."""

		# Evaluate the score
		print(observation)
		print(prediction)
		score = KLdivMeanStd.compute(observation, prediction)
		print(score)
		return score

	def bind_score(self, score, model, observation, prediction):
		score.observation = observation
		score.prediction = prediction
		score.related_data["figures"] = self.figures
		return score
