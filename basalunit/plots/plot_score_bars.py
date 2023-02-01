# For data manipulation
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.

from matplotlib import pyplot as plt
import seaborn as sns
import os


class ScoresBars:
    """
    Displays data in table inside text file
    in the form of barplots
    """

    def __init__(self, testObj, score_label= 'Scoring metrics', xlabel='|Score value|', \
                fig_title='Model_scores', plt_title='Model_scores'):
        self.testObj = testObj
        self.prefix_filename = "score_barPlots_"
        self.score_label = score_label
        self.xlabel = xlabel
        self.fig_title = fig_title
        self.plt_title = plt_title
        self.filepath_list = list()

    def score_barplot(self, filepath=None, scores_floats={}, score_label=None,
                      xlabel=None, x_fontsize=5, ylabel=None, y_fontsize=5, title=None):

        fig = plt.figure()

        # pal = sns.cubehelix_palette(len(scores_floats))
        pal = sns.color_palette('Reds', len(scores_floats))

        scores_floats_df = pd.DataFrame(scores_floats, index=[score_label]).transpose()

        rank = [int(value) - 1 for value in scores_floats_df[score_label].rank()]
        axis_obj = sns.barplot(x=scores_floats_df[score_label], y=scores_floats_df.index, palette=np.array(pal)[rank])

        plt.subplots_adjust(left=0.3)
        axis_obj.set(xlabel=xlabel, ylabel=ylabel)
        # axis_obj.set_ylabel(ylabel, fontsize=y_fontsize)
        # axis_obj.set_xlabel(xlabel, fontsize=x_fontsize)
        axis_obj.set_yticklabels(axis_obj.get_yticklabels(), fontsize=y_fontsize)
        # axis_obj.set_xticklabels(axis_obj.get_xticklabels(), fontsize=x_fontsize)
        axis_obj.axes.set_title(title, fontsize=11)

        # sns.despine()

        plt.savefig(filepath, dpi=600, )
        self.filepath_list.append(filepath)

        plt.close(fig)

        return self.filepath_list

    def create(self):

        # -------------------------- Plotting feature scores ------------------------------------------------
        scores_dict = self.testObj.scores_dict
        filepath_scores_float = \
            os.path.join(self.testObj.path_test_output, self.prefix_filename + self.fig_title + '.pdf')

        plt.close('all')
        self.score_barplot(filepath=filepath_scores_float, scores_floats=scores_dict,
                            score_label=self.score_label, ylabel=self.score_label, xlabel=self.xlabel,
                            x_fontsize=9, y_fontsize=9, title=self.plt_title)

        return self.filepath_list
