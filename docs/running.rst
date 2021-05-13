Running
=======

Overview
---------
AFO requires Python to be installed on your device and can be run using any IDLE or through
the teminal. To customise the execution you can pass in optional args <experiment number> <processors>
which control the experiment to be run and the number of processors used. If no arguments are passed
experiment type 0 will be executed using 1 processor. Once the model has been run two output files per trial will have been
saved to the pkl folder. To generate an excel report run ReportControl.py. The execution can also
customised by passing in some optional args <experiment number> <processors> <report number>.
If no arguments are passed experiment type 0 will be reported using 1 processor and the default report settings.

IDLE
----
Scripts can be run by any python interpreter. Although the necessary packages must be installed (this can be done using ``pip``).
Anaconda comes with the spyder idle. Spyder has its strong points but it becomes tediously slow when
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
you can pass in an argument by right click on the module name then there is an option to
Create (it changes to Edit once yo have used it). In there is a parameter box which you put
a value in. Within an experiment you can use True/False in the ‘Run Trials’ column to
control if a given trial is executed.

Reporting
    Running the reports is much like running the model (see above), simply just run
    ReportControl.py inplace of Exp.py. As with running the core model an argument can be
    passed into the script when running via the terminal which controls which experiment to report.
    For further information on reporting see the report section.

Size: RAM & Speed
-----------------
This table is a record of the resources used for different trials. The number of slices in the stock axes for both dams and offspring, and the number of rotations are the main variables that will alter resources

.. list-table:: Running
   :header-rows: 1

   * - Trial details
     - Scenario description CPU/RAM (gb)/# of processors
     - Amount used RAM, pkl file size, time per loop

   * - FVP 4,3 N 5,4 (1875, 192) scan=0, everything else small
     - 48 / 384 / 1
     - ~50%, , 60 mins
   * - FVP 5,3 N 3,4 (375, 192) scan=0, everything else small
     - 64GB ubuntu
     - Max ~33GB, , 20 mins
   * - FVP 5,3 N 3,5 scan=0, everything else small
     - 64GB ubuntu
     - Max 39GB,
   * - FVP 4,3 N 4,5 scan=0, everything else small
     - 64GB ubuntu
     - Max 39GB,
   * - FVP 4,3 N 3,4 scan=0, everything else small
     - Uni Multiprocess with 5 proccessors
     - 6mins ish. Didnt check RAM but it handled 5 without crashing
   * - FVP 3,3 N 5,6 scan=0, everything else small
     - T7600 with 64GB 1 processor used
     - 26 mins. Only possible w/o the full report information
   * - FVP 3,3 N 5,5 scan=0, everything else small
     - T7600 with 64GB 1 processor used
     - 20 mins. Handled storing the full report information
   * - FVP 3,3 N 3,3 scan=1, everything else small
     - 64GB ubuntu Multiprocess with 12 proccessors
     - Handled no problem
   * - FVP 3,3 N 3,3 scan=2, everything else small
     - 64GB ubuntu Multiprocess with 12 proccessors
     - Handled sitting around 50GB RAM
   * - FVP 3,3 N 3,3 scan=3, everything else small
     - 64GB ubuntu Multiprocess with 12 proccessors
     - Killed one processor
   * - FVP3,3 N1,6
     - 32GB HP 1 processor
     - Completed in 900sec. Used most memory (about 26GB)
   * - FVP4,3 N5,4 LWi2 Scan=0 without large reports
     - GoogleCloud 786GB N2D 8 processors
     - Max memory 624gb 40mis per cycle
   * - FVP4,3 N5,4 LWi2 Scan=0 without large reports
     - GoogleCloud 256GB N2D Highmem ($2.20/hr)
     - With 3 processors one was killed. With 2 processors max memory 79.8%, pkl files 0.3GB, 35mins per cycle






.. list-table:: Reporting
   :header-rows: 1

   * - Trial details
     - Scenario description CPU/RAM (gb)/# of processors
     - Amount used RAM, pkl file size, time per loop

   * - FVP 4,3 N 3,4 Large reports included
     - 64GB ubuntu Multiprocess with 16 proccessors
     - Slower and ran out of RAM. Cut down to 3 processors and it seemed to be okay. But close to limit.


 



