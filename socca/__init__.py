import warnings
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings("ignore", category=FITSFixedWarning)

import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from . import data
from . import priors
from . import noise
from . import models

from .fitting import fitter, load

__all__ = ["data", "priors", "noise", "models", "fitter", "load"]
