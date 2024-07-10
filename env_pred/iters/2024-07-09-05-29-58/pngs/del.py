import os
import sys
import shutil

args = sys.argv[1:]

for arg in args:
    shutil.rmtree(arg)
    os.system('echo "Deleted ' + arg + '"')
    os.system('mkdir '+arg)