from main import *
import main

import sys
modulenames = set(sys.modules) & set(globals())
allmodules = [sys.modules[name] for name in modulenames]

for s in allmodules:
    print(s)
