Tips
=======
.. include:: <isonum.txt>

Model structure overview
------------------------

The general flow of data in the model is;

Inputs |rarr| apply sensitivities |rarr| adjustments calculations (functions and methods) |rarr|
pyomo (compiles linear model) |rarr| solve.

The core units of the model are:
   	i. Inputs.
  	ii. Precalcs.
 	iii. Pyomo and solver.

Inputs
    The model inputs are stored in three excel spread sheets. The first contains inputs that are likely to
    change for different regions or properties such as farm area and the second, contains inputs that are
    'universal' and likely to be consistent for different regions or properties such as prices. Thirdly,
    the structural inputs, which are typically fixed and only changed by experienced users.
Precalcs
    The precalcs are the calculations applied to the input data to generate the data for the Pyomo
    parameters (in other terms, the coefficients for the matrix).
Pyomo
    This is the ‘guts’ of the model, it defines all the variables and parameters then utilised them to
    construct the model equations (constraints). Pyomo formulates all the equations into a linear program
    format and passes the file to a solver. Many solvers are compatible with the Pyomo, currently GLPK
    (GNU Linear Programming Kit) (Makhorin 2014) is used.

Firstly, the inputs are read in from excel, they are then adjusted according to the user defined sensitivities.
For example, the user may be interested in the impact of increasing prices hence the price inputs are
adjusted. Secondly, each module containing precalcs is executed. The produced parameters are stored in a
python data structure called a dictionary, which is passed to the Pyomo modules.
The pyomo section of AFO creates the variables, creates parameters, populates the parameters with the
coefficients passed in from the precalcs, constructs the model constraints and passes the information to
a linear solver. The results from the solver reveal the maximum farm profit and the optimal levels of each
variable to maximise the profit. From here the user can create
any number of different reports.

Execution flow:
    #. Exp1.py is run
    #. Inputs and exp.xlsx read
    #. Determine which trials require running. Trials are only run if set to True in exp.xlsx and the trial
       is out of date (code has changed, inputs have changed or the sensitivity values for that trial have
       changed since it was last run)
    #. Apply the sensitivities.
    #. Run the precalcs.
    #. Run Pyomo.
    #. Solve.
    #. Pickle and save output files

Multiprocessing
---------------
Python’s global interpreter lock (GIL), allows only one thread to carry the Python interpreter at any given
time. The GIL was implemented to handle a memory management issue, but as a result, Python is limited to
using a single processor. This also limits the ability to utilise the multithreading module in python.

AFO has been designed such that each trial is independant to all the other trials. Thus to utilise all
processing power of the computer, AFO can be run using multiple processes that are executed
simultaneously. This is made possible by the multiprocessing package available in python. The number of processors
used is determined by passing in an argument when executing the program. If the number of trials to be run is less
than the number of processors specified the number of processors used will automtically scale to match the
number of trials.

Rotation
----------

Seasonal Variation
------------------

Pyomo
-----
Speed
Parameters: initialization of large parameters can take time (for example in the stock module). To save time mask out the 0 values when creating the param dict and set default argument in the parameter to 0. This make it quicker to build the dictionary in the first place and makes it much faster to initilise the parameter in pyomo (see stock pyomo for examples of this)

Constraints: building constraints can be slow however efficiency can be improved by using if statement which reduce the summing required. Additionally in certain circumstances (eg stock numbers) you can alter the parameters so that duplicate constraints are not built and then use the constraint.skip method to jump over.
It takes longer than I would have expected to evaluate if statements when building a pyomo constraint so time can be saved when summing by only evaluating required items in the if statement/s.

Slacks/duals/rc
^^^^^^^^^^^^^^^
Duals, reduced costs and constraints slacks are not reported by default in pyomo.
You must explicity specify them. By default AFO does this. However, they are only processed
if ``full output == True``.

Access duals, rc and duals:

    #. Tell pyomo to store the information (pre solving):

        * ``model.rc = pe.Suffix(direction=pe.Suffix.IMPORT)``
        * ``model.slack = pe.Suffix(direction=pe.Suffix.IMPORT)``
        * ``model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)``

    #. Report the information (post solving):

        * ``model.rc[model.variable]``
        * ``constraint.uslack()``
        * ``model.dual[model.constraint]``

        As a full example:

        .. code-block::

            for c in model.component_objects(pe.Constraint, active=True):
                for index in c:
                    print("      ", index, model.dual[c[index]], file=f)


Code snippets
^^^^^^^^^^^^^^
Print a param to file:

.. code-block::

 textbuffer = StringIO()
 model.p_numbers_req_dams.pprint(textbuffer)
 textbuffer.write('\n')
 with open('number_prov.txt', 'w') as outputfile:
     outputfile.write(textbuffer.getvalue())



   



