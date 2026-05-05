import numpy as np
import socca

import corner
import pickle
fit = socca.load('stefan_nautilus_fit_results.pickle')

fit.parameters()

fit.plot.corner(name="stefan_cornerplot_nautilus", sigma=None)