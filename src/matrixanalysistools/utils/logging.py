from tqdm import TqdmExperimentalWarning
from rich.traceback import install as rich_install
import logging
from matplotlib import pyplot as plt
from rich.logging import RichHandler
import warnings

def setup_logging():
    rich_install(show_locals=True)
    logging.basicConfig(format="[{filename}:{lineno}]  {message}", style="{", handlers=[RichHandler(markup=True, rich_tracebacks=True)], level=logging.INFO)
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    plt.rcParams['text.usetex'] = True