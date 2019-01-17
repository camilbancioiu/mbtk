import sys
import os
import subprocess


from utilities import cd
sys.path.insert(1, os.path.join(sys.path[0], '..'))

process = subprocess.run(['./exds.py', 'list', 'all'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(process.stdout.decode('utf-8'))
print(process.stderr.decode('utf-8'))

