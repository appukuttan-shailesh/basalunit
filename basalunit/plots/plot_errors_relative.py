import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

#==============================================================================

class ErrorRelative:
    """
    Plots figure with Z-scores for each of the features
    """

    def __init__(self, testObj):
        self.testObj = testObj
        self.filename = "relative_errors"

    def create(self):
        data = OrderedDict(sorted(self.testObj.score_dict.items(), reverse=True))
        yvals, xvals = map(list, zip(*data.items()))
        yinds = range(len(yvals))
        fig = plt.figure(figsize=(8,16))
        plt.plot(xvals, yinds, 'or', markersize='8', mew=2)
        ax = plt.gca()
        ax.yaxis.grid()
        plt.margins(0.02)
        ttl = fig.suptitle('Relative Feature Errors', fontsize=20)
        ttl.set_position([0.5, 0.925])
        plt.xlabel("abs(Z-scores)")
        plt.yticks(yinds, yvals)
        filepath = os.path.join(self.testObj.path_test_output, self.filename + ".pdf")
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
        return filepath
