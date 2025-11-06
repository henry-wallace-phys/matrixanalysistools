import click
import numpy as np

from matrixanalysistools.matrix_handler.root_matrix import MatrixFileHandler
from matrixanalysistools.analysis.convergence_analyser import ConvergenceAnalyser
from matrixanalysistools.plotting.plotter import Plotter
from matrixanalysistools.utils.logging import setup_logging

@click.command()
@click.option('--input_file', "-i", help="Input root file containing matrices")
@click.option('--matrix_stem', "-s", help="Matrix stem in file")
@click.option('--output_pdf', "-o", help="Output PDF contianing plots", default="analysis_plots.pdf")
@click.option('--start', default=0, help="Iteration to start finding plots for")
@click.option('--stop', default=1000, help="Iteration to stop finding plots for")
@click.option('--step', default=1000, help="Steps between matrix creation")
def main(input_file, output_pdf, matrix_stem, start, stop, step):
    setup_logging()
    # Make the handler object
    handler = MatrixFileHandler(input_file, matrix_stem, (start, stop, step))
    # Analyse converge properties
    convergence_analyser = ConvergenceAnalyser(handler)
    
    # Initialise plotter
    plotter = Plotter(start, stop, step)
    
    # Add plots
    plotter.add_plot(id="Norm",
                     data=convergence_analyser.matrix_norms,
                     y_lab="Covariance Matrix Frob. Norm",
                     title="Norm of matrix/iteration"
                    ) 

    plotter.add_plot(id="Change in |Eigenvalue|",
                     data=convergence_analyser.eigenvalue_roc,
                     y_lab=r"Relative Change in Eigenvalue: $|\Delta\lambda|$",
                     title=r"Change in $|\Delta\lambda|$ over iterations",
                     x_override=np.arange(start+step, stop-step, step)
                    )

    plotter.add_plot(id="Suboptimality",
                     data=convergence_analyser.suboptimality,
                     y_lab="Suboptimality",
                     title=r"Suboptimality of matrix"
                    )

    
    # Write
    plotter(output_pdf)

if __name__=="__main__":
    main()