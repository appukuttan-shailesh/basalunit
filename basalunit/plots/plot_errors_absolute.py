import matplotlib
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import math
import matplotlib.backends.backend_pdf

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
        data = []
        # creating 'data' list of tuples in the format:
        # [('feat_name', obs_mean, obs_std, prd_val), ...]
        for key_0 in self.testObj.observation:
            for key_1 in self.testObj.observation[key_0]:
                for key_2 in self.testObj.observation[key_0][key_1]:
                    if "{}.{}.{}".format(key_0, key_1, key_2) not in self.testObj.prob_list:
                        temp_obs = self.testObj.observation[key_0][key_1][key_2]
                        prd_val = self.testObj.prediction[key_0][key_1][key_2]
                        feat_name = "{}.{}.{}".format(key_0, key_1, key_2)
                        entry = (feat_name, temp_obs[0], temp_obs[1], prd_val)
                        data.append(entry)

        data = sorted(data)
        feat_names, obs_mean, obs_std, pred_val = map(list, zip(*data))
        yinds = range(len(feat_names))

        MAX_FEATS_PER_PAGE = 50
        total_pages = int(math.ceil(len(feat_names)/MAX_FEATS_PER_PAGE))

        fig_list = []
        for page_ctr in range(total_pages):
            start_ind = page_ctr*MAX_FEATS_PER_PAGE
            end_ind = (page_ctr+1)*MAX_FEATS_PER_PAGE
            if end_ind > len(feat_names):
                end_ind = len(feat_names)
            fig = plt.figure(figsize=(8,16))
            plt.errorbar(obs_mean[start_ind:end_ind], yinds[start_ind:end_ind],
                         xerr=obs_std[start_ind:end_ind], ecolor='black', elinewidth=2,
                         capsize=5, capthick=2, fmt='ob', markersize='5', mew=5, zorder=1)
            plt.plot(pred_val[start_ind:end_ind], yinds[start_ind:end_ind], 'rx', markersize='8', mew=2, zorder=100)
            ax = plt.gca()
            ax.yaxis.grid()
            plt.margins(0.02)
            ttl = fig.suptitle('Absolute Feature Errors', fontsize=20)
            ttl.set_position([0.5, 0.925])
            plt.xlabel("(Units)")
            plt.yticks(yinds[start_ind:end_ind], feat_names[start_ind:end_ind])
            fig_list.append(fig)
        filepath = os.path.join(self.testObj.path_test_output, self.filename + ".pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(filepath)
        for fig in fig_list:
            pdf.savefig(fig, dpi=600, bbox_inches='tight')
        pdf.close()
        plt.close('all')
        return filepath
