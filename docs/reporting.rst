Output & Reporting
==================

Model output
---------------
Definitions:

* Slacks - this is the slack on each constraint
* Duals - this is the effect on the objective if the RHS of the constraint is increase by one
* Rc - this is the effect on the objective if one extra of a given variable is included

Output
^^^^^^
Printed on screen at the beginning of the execution:

* Time when exp.xlsx was last saved
* Number of trials to be run
* Number of full solutions

Printed on screen for each trial:

* Profit
* Solver status
* Time for current loop
* Eta to completion

To check during execution:

* Variable summary: Text file with each activity level - this file is overwritten each iteration but provides a summary of all the variables, only positive variables are included.

If the user selects that they want the ‘full output’ in the exp.xlsx the model will return additional
information:

* A summary of all activity levels.
* a lp file: this is a file containing the linear equations of the whole model ie what the solver sees. This can be read via the lpviewer.xlsx document to make it easier to navigate.
* a text file with the full model print out: this contains the all the activities and constraints, you can also see any slack in the constraints
* A text file with the reduced costs (effect on profit if one more variable forced into solution) of the variables, the duals (effect on profit if RHS is increased by one) of each constraint and the slack on each constraint.

For reporting:

* A dict (lp_vars) with the model variable results determined from the solver for each trial is saved as a pickle file. The pickle file is loaded each time before the execution therefore if the trial has been run already it is overwritten however if this is the first time the trial has run it is added to the existing dict. This dict can be accessed for post run calculations.
* A dict (r_vals) with report variables from the pre-calcs. These will include:

    * A selection of the parameter values
    * Some specifically defined report variables.

Using output
^^^^^^^^^^^^
There are 4 fairly distinct phases in which the model results are used. Each requires different output
and is carried out differently. There will be some overlapping between the output required for each
steps. Eg. when understanding the farm system it might be necessary to check if there is an error in
the model definition, and likewise when reporting scenarios it might be necessary to understand the farm
system.

#. Debugging model operation (is the maths what we think it is).

    When a new component has been added it is necessary to check that it is operating correctly and the LP is
    defined correctly. Checking the model definition is the voluminous part. Model definition is the matrix:
    what constraints are defined for what variables (activities) and what is the parameter (coefficient).
    This can be done using:

        a. The ‘full output’ method which generates full lp file, variable summary, RC, duals and slacks.
        b. You can use LPviewer to aid in viewing the lp file.

#. Understanding the results of the analysis (why does the farm optimise the way it does).

    Once confident that the model is solving correctly, the next step is understanding the ‘why’ of the farming
    system. This is usually achieved by looking at details of why a particular solution is the optimum and not
    another that was expected to be more profitable. This can be done by:

        a. examining detailed reports (discussed in more detail in the next topic) associated with the
           optimal solution.
        b. examining the reduced costs of the model variables and shadow prices of the constraints.
        c. bounding individual variables and checking the impact on the optimal solution.

#. Reporting values for inclusion in reports (what scenarios need to be described for the reader).
   This can be done by:

    a. Generating reports using lp_vars & r_vals (discussed in more detail in the next topic).

#. Viewing the results whilst the model is running.

    This is for large analyses to check that the model is solving satisfactorily, or whether the analysis should be aborted prior to normal completion. This can be done by:

    a. Inspecting the console print out.
    b. Inspecting the variable summary which is saved each iteration.

Reporting
------------
How to use
^^^^^^^^^^
Running the reports is much like running the model (see section or running), simply just run
ReportControl.py inplace of Exp.py. As with running the core model an argument can be passed into the
script when running via the terminal which controls which experiment to report. ReportControl.py calls the
necessary report functions in ReportFunctions.py. For more information on the process to add reports and information regarding the existing
reports see below.

About
^^^^^
The detailed budgets are created by multiplying the optimal LP primal variable levels (lp_vars) by values from
the pre-calcs (r_vals), although in some cases it can be using the slacks from the constraints.
An example of using the slacks is when following the fate of a product (pasture, stubble, grain) in the
production chain. In this case slack on a row is a ‘sink’ for the product and understanding that is where
it is going helps understand the output (it may also identify an error, if slack is not expected...ungrazed
pasture and stubble (slack) are common outcomes, unsold grain (slack) would indicate an error.

.. toctree::
   :maxdepth: 2

   ReportFunctions
   



