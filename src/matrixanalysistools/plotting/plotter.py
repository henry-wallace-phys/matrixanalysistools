'''
Simple plotting util to ensure consistent plot styling
'''

from typing import Dict, Sequence, TypedDict, Optional
import logging
from dataclasses import dataclass

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class PlotInfo:
    ''' Simple data struct for dealing with plotting info
    '''
    data: Sequence[float]
    title: str
    y_label: str
    x_override: Optional[Sequence[float]] = None
    
    log_y: bool = False

class Plotter:
    def __init__(self, start: int, stop: int, step: int):
        logging.info("[bold green] Initialised Plotting Handler")
        self._plots: Dict[str, PlotInfo] = {}
        self._x_axis = np.arange(start, stop, step)
        

    def add_plot(self, id: str, plot_info: PlotInfo):
        logging.info(f"[blue]   - Added {id} to plotter")
        self._plots[id] = plot_info

    def __call__(self, output_file_name: str):
        print(f"Saving plots to {output_file_name}")
        with PdfPages(output_file_name) as out_file:
            for plot_id, plot_info in self._plots.items():
                logging.info(f"[blue]   - Making {plot_id} plot")
                self.make_plot(plot_info, out_file)
                

    def make_plot(self, plot_info: PlotInfo, outfile: PdfPages | str="output.pdf"):
        if plot_info.x_override is not None:
            x_axis = plot_info.x_override
        else:
            x_axis = self._x_axis
        
        if len(x_axis)!=len(plot_info.data):
            raise ValueError("Error! X axis labels not the same length as data")
        
        plt.plot(x_axis, plot_info.data)
        plt.xlabel("Step")
        plt.ylabel(plot_info.y_label)
        plt.title(plot_info.title)
        
        if plot_info.log_y:
            plt.yscale('log')

        if isinstance(outfile, PdfPages):
            outfile.savefig()
        else:
            plt.savefig(outfile)

        plt.close()