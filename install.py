import sys
import subprocess

# install dependency
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PySimpleGUI'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'joblib'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'seaborn'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
