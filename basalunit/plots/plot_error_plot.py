import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

#==============================================================================

class ErrorPlot:
    """
    Creates error plot for specified data
    Note: should be extended in future to provide more flexibility, such as
    provision for specifying various graph parameters (e.g. xlim, ylim)

    Caution: Currently this presumes that terminal dictionary keys for
    observation are ('mean' and 'std') or ('min' and 'max'), and for prediction
    is 'value'; with identical keys at all non-terminal levels.
    """

    def __init__(self, testObj):
        self.testObj = testObj
        self.filename = "error_plot"
        self.xlabels = ["(not specified)"] # self.testObj.observation.keys()
        self.ylabel = "(not specified)"

    def traverse_dicts(self, obs, prd, output = []):
        # output will contain list with elements in the form:
        # [("type", obs_mean, obs_std, prd_value), ... ] or
        # [("type", obs_min, obs_max, prd_value), ... ]
        # where "type" specifies whether observation is in the form of
        # (mean,std) -> type="mean_sd", or (min,max) -> type="min_max"
        od_obs = OrderedDict(sorted(obs.items(), key=lambda t: t[0]))
        od_prd = OrderedDict(sorted(prd.items(), key=lambda t: t[0]))
        flag = True

        for key in od_obs.keys():
            if flag is True:
                if isinstance(od_obs[key], dict):
                    self.traverse_dicts(od_obs[key], od_prd[key])
                else:
                    if "mean" in od_obs.keys():
                        output.append(("mean_sd",od_obs["mean"],od_obs["std"],od_prd["value"]))
                    elif "min" in od_obs.keys():
                        output.append(("min_max",od_obs["min"],od_obs["max"],od_prd["value"]))
                    else:
                        print("Error in terminal keys!")
                        raise
                    flag = False
        return output

    def create(self):
        output = self.traverse_dicts(self.testObj.observation, self.testObj.prediction)
        fig = plt.figure()
        ix = 0
        for (obs_type, obs_var1, obs_var2, prd_value) in output:
            if obs_type == "mean_sd":
                ax_o = plt.errorbar(ix, obs_var1, yerr=obs_var2, ecolor='black', elinewidth=2,
                                    capsize=5, capthick=2, fmt='ob', markersize='5', mew=5)
            elif obs_type == "min_max":
                ax_o = plt.plot([ix, ix],[obs_var1, obs_var2],'_b-', markersize=8, mew=8, linewidth=2.5)
            else:
                # should never be executed
                print("ERROR! Unknown type of observation data.")
            ax_p = plt.plot(ix, prd_value, 'rx', markersize='8', mew=2)
            ix = ix + 1
        plt.xticks(range(len(output)), self.xlabels, rotation=20)
        plt.tick_params(labelsize=11)
        plt.figlegend((ax_o,ax_p[0]), ('Observation', 'Prediction',), 'upper right')
        plt.margins(0.1)
        plt.ylabel(self.ylabel)
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        filepath = self.testObj.path_test_output + self.filename + '.pdf'
        plt.savefig(filepath, dpi=600,)
        return filepath
