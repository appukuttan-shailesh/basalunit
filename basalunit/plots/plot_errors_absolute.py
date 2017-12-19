import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

#==============================================================================

class ErrorAbsolute:
    """
    Plots figure with experimental data (mean, std) and prediction (value)
    for each of the features
    """

    def __init__(self, testObj):
        self.testObj = testObj
        self.filename = "absolute_errors"

    def create(self):
        pred_data = OrderedDict(sorted(self.testObj.pred_feature_dict.items(), reverse=True))
        yvals, pred_xvals = map(list, zip(*pred_data.items()))
        yinds = range(len(yvals))

        obs_mean = []
        obs_std = []
        for key in self.testObj.observation.keys():
            for key2 in self.testObj.observation[key]["soma"].keys():
                temp_obs = self.testObj.observation[key]["soma"][key2]
                obs_mean.append(temp_obs[0])
                obs_std.append(temp_obs[1])

        fig = plt.figure(figsize=(8,16))
        plt.errorbar(obs_mean, yinds, xerr=obs_std, ecolor='black', elinewidth=2,
                     capsize=5, capthick=2, fmt='ob', markersize='5', mew=5)
        plt.plot(pred_xvals, yinds, 'rx', markersize='8', mew=2)
        ax = plt.gca()
        ax.yaxis.grid()
        plt.margins(0.02)
        ttl = fig.suptitle('Absolute Feature Errors', fontsize=20)
        ttl.set_position([0.5, 0.925])
        plt.xlabel("(Units)")
        plt.yticks(yinds, yvals)
        filepath = os.path.join(self.testObj.path_test_output, self.filename + ".pdf")
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
        return filepath
