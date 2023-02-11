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

    def __init__(self, testObj, score_label= 'Scoring metrics', ylabel='Scoring metrics', \
                fig_title='Model_scores', plt_title='Model_scores', score_scale = 'symlog'):
        self.testObj = testObj
        self.prefix_filename = "score_barPlots_"
        self.score_label = score_label
        self.ylabel = ylabel
        self.scale = score_scale
        self.fig_title = fig_title
        self.plt_title = plt_title
        self.filepath_list = list()

    def score_barplot(self, filepath=None, scores_floats={}, score_label=None,
                      ylabel=None, x_fontsize=5, y_fontsize=5, title=None, score_scale='linear'):

        fig = plt.figure()

        # pal = sns.cubehelix_palette(len(scores_floats))
        pal = sns.color_palette('Reds', len(scores_floats))

        scores_floats_df = pd.DataFrame(scores_floats, index=[score_label]).transpose()

        rank = [int(value) - 1 for value in scores_floats_df[score_label].rank()]
        axis_obj = sns.barplot(x=scores_floats_df[score_label], y=scores_floats_df.index, palette=np.array(pal)[rank])

        plt.subplots_adjust(left=0.3)
        axis_obj.set_ylabel(self.ylabel, fontsize=y_fontsize)
        axis_obj.set_yticklabels(axis_obj.get_yticklabels(), fontsize=y_fontsize-2, rotation=30)
        # axis_obj.set_xticklabels(axis_obj.get_xticklabels(), fontsize=x_fontsize-1)
        axis_obj.set(xscale=score_scale)
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
                            score_label=self.score_label, ylabel=self.ylabel,
                            x_fontsize=10, y_fontsize=10, title=self.plt_title, score_scale = self.scale)

        return self.filepath_list
