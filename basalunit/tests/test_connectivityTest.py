from array import typecodes
import os
import json
from sciunit import Test
import basalunit.capabilities as cap
from basalunit.scores import AvgRelativeDifferenceScore_ConnProb
from quantities import mV
try:
	from sciunit import ObservationError
except:
	from sciunit.errors import ObservationError
from snudda.analyse.analyse import SnuddaAnalyse
import contextlib

class ConnectivityTest(Test):
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
				 pre_type=None,
				 post_type=None,
				 dist_3d=True,
				 exp_max_dist=None,
				 exp_data=None,
				 exp_data_detailed=None,
				 y_max=None,
				 log_file=None):

		if not name:
			name = "ConnectivityTest: {} -> {}".format(pre_type, post_type)
		Test.__init__(self, observation, name)

		self.required_capabilities += (cap.SnuddaBasedModel,)

		self.pre_type = pre_type
		self.post_type = post_type
		self.dist_3d = dist_3d
		self.y_max = y_max
		self.log_file = log_file

		self.figures = []
		description = "Evaluates the connection probabilities and compares it to experimental data."

	score_type = AvgRelativeDifferenceScore_ConnProb

	def format_data(self, observation):
		# input observation format:
		# keys: max_dist, num, total, value(=num/total)
		# e.g.
		# observation = [{"max_dist": 50e-6,  "num": 5.0, "total": 19.0, "value": 5.0/19.0},
		# 							 {"max_dist": 100e-6, "num": 3.0, "total": 43.0, "value": 3.0/43.0}]
		#
		# required format:
		# self.exp_max_dist = [50e-6, 100e-6]
		# self.exp_data_detailed = [(5, 19), (3, 43)]
		# self.exp_data = [5/19.0, 3/43.0]

		self.exp_max_dist = []
		self.exp_data = []
		self.exp_data_detailed = []

		for item in observation:
			self.exp_max_dist.append(item["max_dist"])
			if "num" in item.keys():
				self.exp_data_detailed.append((item["num"], item["total"]))
			if "value" in item.keys():
				self.exp_data.append(item["value"])
			else:
				self.exp_data.append(item["num"]/item["total"])

		if len(self.exp_max_dist) == 0:
			self.exp_max_dist = None
		if len(self.exp_data) == 0:
				self.exp_data = None
		if len(self.exp_data_detailed) == 0:
				self.exp_data_detailed = None

	def validate_observation(self, observation):
		try:
			assert type(observation) is list
			for item in observation:
				assert type(item) is dict
				assert "max_dist" in item.keys()
				assert (
								all(key in item.keys() for key in ["num", "total"])
								or ("value" in item.keys())
							)
			self.format_data(observation)
		except Exception as e:
			raise ObservationError("Observation not in the required form!")

	def generate_prediction(self, model, verbose=False):
		"""Implementation of sciunit.Test.generate_prediction."""
		if not self.log_file:
			self.log_file = os.path.join(
				model.network_path, "log", "connectivity.log")

		hdf5_file = os.path.join(model.network_path, "network-synapses.hdf5")
		with open(self.log_file, "w") as o:
			with contextlib.redirect_stdout(o):
				sa = SnuddaAnalyse(hdf5_file=hdf5_file)
				model_probs, plot_fig = sa.plot_connection_probability( pre_type=self.pre_type,
																		post_type=self.post_type,
																		exp_max_dist=self.exp_max_dist,
																		exp_data=self.exp_data,
																		exp_data_detailed=self.exp_data_detailed,
																		dist_3d=self.dist_3d,
																		y_max=self.y_max)
		prediction = model_probs
		if plot_fig:
			self.figures.append(plot_fig)
		self.figures.append(self.log_file)
		# self.figures.append(hdf5_file)
		return prediction

	def compute_score(self, observation, prediction, verbose=False):
		"""Implementation of sciunit.Test.score_prediction."""

		# Evaluate the score
		print(observation)
		print(prediction)
		score, score_list = AvgRelativeDifferenceScore_ConnProb.compute(observation, prediction)
		print(score)
		print(score_list)
		return score

	def bind_score(self, score, model, observation, prediction):
		score.observation = observation
		score.prediction = prediction
		score.related_data["figures"] = self.figures
		return score
