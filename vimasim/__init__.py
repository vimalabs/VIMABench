"""
Due to historical reason, the dataset was generated with package name "vimasim". To avoid package not found error
when loading pickled data, we add an alias here.
"""
import sys
import vima_bench
from vima_bench import *

sys.modules["vimasim"] = vima_bench
