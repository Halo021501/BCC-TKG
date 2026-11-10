"""Run offline correctness tests; no formal training or provider calls."""
from _common import ROOT
import sys
import unittest

if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),top_level_dir=str(ROOT))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
