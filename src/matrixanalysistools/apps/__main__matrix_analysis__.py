import logging

import click
import numpy as np

from matrixanalysistools.matrix_handler.root_matrix import MatrixFileHandler
from matrixanalysistools.analysis.convergence_analyser import ConvergenceAnalyser
from matrixanalysistools.plotting.plotter import Plotter, PlotInfo
from matrixanalysistools.utils.logging import setup_logging

@click.command()
@click.option('--input_file', "-i", help="Input root file containing matrices")
@click.option('--matrix_stem', "-s", help="Matrix stem in file")
@click.option('--output_pdf', "-o", help="Output PDF contianing plots", default="analysis_plots.pdf")
@click.option('--start', default=0, help="Iteration to start finding plots for")
@click.option('--stop', default=1000, help="Iteration to stop finding plots for")
@click.option('--step', default=1000, help="Steps between matrix creation")
@click.option('--make_norm', default=True, help="Make the norm plot?")
@click.option('--make_eigen', default=True, help="Plot |eigen values|?")
@click.option('--make_trace', default=True, help="Steps between matrix creation")
@click.option('--make_suboptimality', default=False, help="Steps between matrix creation")
@click.option('--n_threads', default=1, help="Number of threads to run")
def main(input_file: str, output_pdf: str, matrix_stem: str, start: int, stop: int, step: int,
         make_norm: bool, make_eigen: bool, make_trace: bool, make_suboptimality: bool, n_threads: int):
    setup_logging()
    logging.info(f"[green]Making diagnostic plots from [/][bold blue]{input_file}[/][green] saving to [/][bold blue]{output_pdf}[/][green] using [/][bold blue]{n_threads}[/][green] threads")
    
    # Make the handler object
    handler = MatrixFileHandler(input_file, matrix_stem, (start, stop, step))
    # Analyse converge properties
    convergence_analyser = ConvergenceAnalyser(handler, n_threads)
    
    # Initialise plotter
    plotter = Plotter(start, stop, step)
    
    # Add plots
    if make_norm:
        plotter.add_plot("Norm", PlotInfo(data=convergence_analyser.matrix_norms,
                                     y_label="Covariance Matrix Frob. Norm",
                                    title="Norm of matrix/step")
                        ) 
    if make_eigen:
        plotter.add_plot("Change in |Eigenvalue|", PlotInfo(data=convergence_analyser.eigenvalue_roc,
                                                            y_label=r"Relative Change in Eigenvalue: $|\Delta\lambda|$",
                                                            title=r"Change in $|\Delta\lambda|$ per step",
                                                            x_override=np.arange(start+step, stop-step, step))
                        )
    if make_trace:
        plotter.add_plot("Trace", PlotInfo(data=convergence_analyser.trace,
                                            y_label="Trace",
                                            title=r"Trace")
                        )


    if make_suboptimality:
        plotter.add_plot("Suboptimality", PlotInfo(data=convergence_analyser.suboptimality,
                                                    y_label="Suboptimality",
                                                    title=r"Suboptimality of matrix/step",
                                                    log_y=True)
                        )

    
    # Write
    plotter(output_pdf)

if __name__=="__main__":
    main()