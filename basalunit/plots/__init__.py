"""Loads MorphoUnit plot classes for NeuronUnit"""

from os.path import dirname, basename, isfile
import glob

"""
NOTE: All plot files must have a prefix "plot_" and extension ".py".
Only these would be loaded.
"""
files = glob.glob(dirname(__file__)+"/plot_*.py")
modules = [ basename(f)[:-3] for f in files if isfile(f)]

for module in modules:
    exec("from %s import *" % module)
