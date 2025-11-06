'''
Simple plotting util to ensure consistent plot styling
'''

from typing import Dict, Sequence, TypedDict, Optional
import logging


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

class PlotInfo(TypedDict):
    ''' Simple data struct for dealing with plotting info
    '''
    data: Sequence[float]
    title: str
    y_label: str
    x_override: Optional[Sequence[float]]

class Plotter:
    def __init__(self, start: int, stop: int, step: int):
        logging.info("[bold green] Initialised Plotting Handler")
        self._plots: Dict[str, PlotInfo] = {}
        self._x_axis = np.arange(start, stop, step)
        

    def add_plot(self, id: str, data: Sequence[float], y_lab: str, title: str, x_override: Optional[Sequence[float]]=None):
        logging.info(f"[blue]   - Added {id} to plotter")
        plot_info: PlotInfo = {
            'data': data,
            'title': title,
            'y_label': y_lab,
            'x_override': x_override
        }
        self._plots[id] = plot_info

    def __call__(self, output_file_name: str):
        print(f"Saving plots to {output_file_name}")
        with PdfPages(output_file_name) as out_file:
            for plot_id, plot_info in self._plots.items():
                logging.info(f"[blue]   - Making {plot_id} plot")
                self.make_plot(plot_info['data'], plot_info['y_label'], plot_info['title'], out_file, plot_info['x_override'])
                

    def make_plot(self, data: Sequence[float], y_label: str, title: str, outfile: PdfPages | str="output.pdf", x_axis_override: Optional[Sequence[int]]=None):
        if x_axis_override is not None:
            x_axis = x_axis_override
        else:
            x_axis = self._x_axis
        
        if len(x_axis)!=len(data):
            raise ValueError("Error! X axis labels not the same length as data")
        
        plt.plot(x_axis, data)
        plt.xlabel("Step")
        plt.ylabel(y_label)
        plt.title(title)

        if isinstance(outfile, PdfPages):
            outfile.savefig()
        else:
            plt.savefig(outfile)

        plt.close()