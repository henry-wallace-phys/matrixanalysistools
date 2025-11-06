'''
Simple plotting util to ensure consistent plot styling
'''

from typing import Dict, Sequence, TypedDict
from matplotlib.backends.backend_pdf import PdfPages
from typing import Sequence
import matplotlib.pyplot as plt
import logging

class PlotInfo(TypedDict):
    ''' Simple data struct for dealing with plotting info
    '''
    data: Sequence[float]
    title: str
    x_label: str
    y_label: str

class Plotter:
    def __init__(self):
        logging.info("[bold green] Initialised Plotting Handler")
        self._plots: Dict[str, PlotInfo] = {}

    def add_plot(self, id: str, data: Sequence[float], x_lab: str, y_lab: str, title: str):
        logging.info(f"[blue]   - Added {id} to plotter")
        plot_info: PlotInfo = {
            'data': data,
            'title': title,
            'x_label': x_lab,
            'y_label': y_lab
        }
        self._plots[id] = plot_info

    def __call__(self, output_file_name: str):
        print(f"Saving plots to {output_file_name}")
        with PdfPages(output_file_name) as out_file:
            for plot_id, plot_info in self._plots.items():
                logging.info(f"[blue]   - Making {plot_id} plot")
                self.make_plot(plot_info['data'], plot_info['x_label'], plot_info['y_label'], plot_info['title'], out_file)
                

    @classmethod
    def make_plot(cls, data: Sequence[float], x_label: str, y_label: str, title: str, outfile: PdfPages | str="output.pdf"):
        plt.plot(data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        if isinstance(outfile, PdfPages):
            outfile.savefig()
        else:
            plt.savefig(outfile)

        plt.close()