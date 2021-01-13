'''
Module used to construct lp file from discrete model and stocastic add ons.
'''



import subprocess

subprocess.call(['C:/Users/21512438/Anaconda3/Scripts/runef.exe', '--traceback', '--symbolic-solver-labels',
                 '--verbose', '-mReferenceModel.py', '--solve', '--solver=glpk'])
