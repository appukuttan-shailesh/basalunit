#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

#==============================================================================

class ModelExpTraces:
    """
    Plots traces of both simulation and experimental data on same graph
    """

    def __init__(self, testObj=None):
        self.testObj = testObj
        self.filename = "model_exp_traces"

    def create(self):
        # TODO: expdata and junction_potential to be implemented after
        # mapping of expdata to protocols obtained
        responses = self.testObj.pred_traces
        junction_potential = self.testObj.junction_potential

        MAX_FIGS_PER_PAGE = 5
        plot_ctr = 0

        fig_list = []
        for (name, response) in sorted(responses.items()):
            axes_num = plot_ctr%MAX_FIGS_PER_PAGE
            if axes_num == 0:
                fig = plt.figure(figsize=(8,12))
            fig_num = int(plot_ctr/MAX_FIGS_PER_PAGE)
            ax = plt.subplot2grid((MAX_FIGS_PER_PAGE, 1), (axes_num, 0))
            if self.testObj.expdata and name in self.testObj.expdata:
                data = np.loadtxt(os.path.join(self.testObj.exp_trace_dir, self.testObj.expdata[name]))
                time = data[:,0]
                voltage = data[:,1] - junction_potential
                ax.plot(time, voltage, color='lightgrey', linewidth=3)
            ax.plot(response['time'], response['voltage'])
            ax.set_title(name)
            plot_ctr += 1

            if (plot_ctr%MAX_FIGS_PER_PAGE) == 0 or plot_ctr==len(responses):
                fig.tight_layout()
                fig_list.append(fig)

        filepath = os.path.join(self.testObj.path_test_output, self.filename + ".pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(filepath)
        for fig in fig_list:
            pdf.savefig(fig, dpi=600)
        pdf.close()
        plt.close('all')
        return filepath
