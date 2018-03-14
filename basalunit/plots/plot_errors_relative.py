import matplotlib
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import math
import matplotlib.backends.backend_pdf

#==============================================================================

class ErrorRelative:
    """
    Plots figure with Z-scores for each of the features
    """

    def __init__(self, testObj):
        self.testObj = testObj
        self.filename = "relative_errors"

    def create(self):
        data = []
        # creating 'data' list of tuples in the format:
        # [('feat_name', score), ...]
        for key_0 in self.testObj.score_dict:
            for key_1 in self.testObj.score_dict[key_0]:
                for key_2 in self.testObj.score_dict[key_0][key_1]:
                    if "{}.{}.{}".format(key_0, key_1, key_2) not in self.testObj.prob_list:
                        feat_name = "{}.{}.{}".format(key_0, key_1, key_2)
                        entry = (feat_name, self.testObj.score_dict[key_0][key_1][key_2])
                        data.append(entry)

        data = sorted(data)
        feat_names, scores = map(list, zip(*data))
        yinds = range(len(feat_names))
        scores = map(abs, scores)

        MAX_FEATS_PER_PAGE = 50
        total_pages = int(math.ceil(len(feat_names)/MAX_FEATS_PER_PAGE))

        fig_list = []
        for page_ctr in range(total_pages):
            start_ind = page_ctr*MAX_FEATS_PER_PAGE
            end_ind = (page_ctr+1)*MAX_FEATS_PER_PAGE
            fig = plt.figure(figsize=(8,16))
            plt.plot(scores[start_ind:end_ind], yinds[start_ind:end_ind], 'or', markersize='8', mew=2)
            ax = plt.gca()
            ax.yaxis.grid()
            ax.axvline(color='b', linestyle='--')
            plt.margins(0.02)
            ttl = fig.suptitle('Relative Feature Errors', fontsize=20)
            ttl.set_position([0.5, 0.925])
            plt.xlabel("abs(Z-scores)")
            plt.yticks(yinds[start_ind:end_ind], feat_names[start_ind:end_ind])
            fig_list.append(fig)
        filepath = os.path.join(self.testObj.path_test_output, self.filename + ".pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(filepath)
        for fig in fig_list:
            pdf.savefig(fig, dpi=600, bbox_inches='tight')
        pdf.close()
        plt.close('all')
        return filepath
