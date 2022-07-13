Running
=======

Overview
---------
AFO requires Python to be installed on your device and can be run using any IDLE or through
the teminal. The model is controlled via the Exp1.py module and exp.xlsx spreadsheet. AFO is run from Exp1.py
but the trials which are run and the application of any sensitivities are controlled using exp.xlsx.
To customise the execution you can pass in optional args to Exp1.py <experiment number> <processors>
which control the experiment to be run and the number of processors used. If no arguments are passed
experiment type 0 will be executed using 1 processor. Once the model has been run two output files per trial are
saved to the pkl folder. To generate an excel report run ReportControl.py. The execution can also be
customised by passing in some optional args <experiment number> <processors> <report number>.
If no arguments are passed experiment type 0 will be reported using 1 processor and the default report settings.

IDLE
----
Scripts can be run by any python interpreter. Although the necessary packages must be installed (this can be done using ``pip``).
Anaconda comes with the spyder IDLE. Spyder has its strong points but it becomes tediously slow when
debugging the livestock module due to the large arrays. So we tend to used Pycharm.
Pycharm IDLE handles the debugging efficiently and has many useful capabilities built in. For more information
check out the documentation or youtube help videos. To run AFO from an IDLE, simply navigate to the Exp1.py module and
and apply the run command (usually a play button or the like).

Run using cmd/terminal
----------------------
The model can be run through the cmd/terminal. This is used for cloud computing.
To run the script on terminal; navigate to the directory with the model and run the script
using the following command: ``python Exp1.py <experiment number> <processors>``
The python used is the one listed in PATH but you can use others by specifying
the path when running the command eg ``/PATH/python Exp1.py <experiment number> <processors>``.
A similar process can be done using the anaconda prompt.

Exp.xlsx
--------
Firstly some Terminology.

Analysis:
    An analysis is all the lines of Exp.xlsx and may also include the same Exp.xlsx being used on multiple regional models.
Experiment:
    An experiment is a block of Exp.xlsx that is examining an individual question pertaining to that analysis
Trial:
    A trial is a single line within an experiment.

Exp.xlsx is used to control the sensitivity values for each trial. It also controls if the
trial is run, the level of output and if the trial is reported. Trials are grouped into
experiments in the ‘Exp Group’ column. Trials with the same number are in the same experiment.
The user can then select which experiment to run by passing in a number when running the
scripts from terminal (e.g. python Exp.py 2 this will run all the trials with a 2 in the
‘Exp Group’ column). If no argument is passed in then all trials are included. In Pycharm
you can pass in an argument by right clicking on the module name, then there is an option to
'Create' (it changes to Edit once yo have used it). In there is a parameter box which you put
a value in. Within an experiment you can use True/False in the ‘Run Trials’ column to
control if a given trial is executed.

Reporting
    Running the reports is much like running the model (see above), simply just run
    ReportControl.py inplace of Exp.py. As with running the core model an argument can be
    passed into the script when running via the terminal which controls which experiment to report.
    For further information on reporting see the report section.





