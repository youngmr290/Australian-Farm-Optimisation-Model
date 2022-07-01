Debugging
=========

Possible model errors
---------------------
1. Two trials that should be the same are giving different results.

    - Symptoms:
        - Run one trial and it works fine, run multiple trials that are the same and they do not result in the same answer.
    - Solution:
        - You may be altering the original inputs because you have operated on a view not a copy.
          Create a copy() the problematic input or array. In some cases with co plex data structures you may need a deepcopy()

2. The model is not selecting any livestock.

    - Symptoms:
        - Reducing the number of nutrition options causes no sheep or changing the feedsupply causes no sheep.
        - The model becomes infeasible if you force in some livestock without allowing purchasing.
    - Solution:
        - This problem is likely caused by dams not being able to provide enough prog to replace themselves.
          This is because the weight distributions don't align, and prog will be getting lost (eg initial dam
          weights are not similar enough to the prog weights so the prog are not able to provide dams).
          To fix this you can alter the initial lw of the dams (structural inputs: i_adjp_lw_initial_w1).

3. Infeasible model for no apparent reason.

    - Symptoms:
        - Model not solving. Solver exit status returned 'other'
        - Other very similar trials solve.
    - Solution:
        - In the past we have had issue with no solution being found but the model is not infeasible.
          This could be due to very small numbers getting passed to the solver.
          To fix this tweak a sensitivity value slightly (remove some decimals if possible).

4. KeyError: 1247829999 (key can be any bunch of numbers).

    - Symptoms:
        - Model throwing key error in presolve stage.
        - If you run one trial it works.
    - Solution:
        - This occurs when an object in pyomo is being called that doesnt exist.
        - This may be because a bound or constraint is still active from the last trial, however
          the variable that it is referencing has been deleted or changed.
          You need to make sure constrains are being deleted between trials or being re-built correctly.
        - This may be because a variable has been deleted and rebuilt but a constraint referencing that
          variable was not rebuilt. To fix this, the constraint will also need to be deleted and rebuilt.

5. CPLEX> CPLEX Error  1615: Line 33271: Expected number, found 'n'

    - Symptoms:
        - CPLEX throws an error and doesnt solve.
    - Solution:
        - A parameter with nan value has been introduced.
        - Search the .lp file to find which parameter.

Process to debug
----------------
#. Check the variable summary. This will point out any obvious issues and help to determine where to start looking.

#. Turn off bounds (overdraw, erosion limit, labour, sr, rotation area).

#. Simplify the model as much as possible to make it easier to debug (reduce TOL options, reduce lw patterns, reduce genotypes etc).

#. To locate where the problem is, you can often just comment out constraints until the model solves. This can further indicate where to look.

#. Go back to the last working version (using the power of git) and compare outputs with current. If you have some idea where the issue is you can even compare the params between the two versions to see if something jumps out.

#. Once you have an idea where the problem is you can look in more detail using:
    - The lp file.
    - The params dictionary.
    - The arrays (assessable in debug mode).

Solving an infeasible model
---------------------------
If the model is infeasible then the typical execution of the model will not return a solution.
To force the model to create an output run the .lp file through the command prompt and specify the
option ``--nopresol``. This can be done as follows:

    - Create the .lp file. This can be done by including ``model.write('Output/test.lp',io_options={'symbolic_solver_labels':True})``
      in CorePyomo prior to solving.
    - Open a command prompt, change to the AFO directory (where the .lp file is located)
    - Run ``C:\Models\AFO>glpsol --lp --nopresol test.lp --output resultfile``
    - This will generate a screen report of solver info and create a file resultfile
      (or whatever name was specified) in the AFO directory.

Pyomo errors
-------------
#. ERROR: evaluating object as numeric value: v_phase_area[GAANw,lmu1] (object: <class 'pyomo.core.base.var._GeneralVarData'>) No value for uninitialized NumericValue object v_phase_area[GAANw,lmu1]
    You are most likely trying to evaluate an equation with a variable, variables don’t
    have a value at this time hence the error ie trying to do a conditional statement on an equation

#. Error: Constraint resulting in boolean
    Often because param is default to 0 and that is multiplying by other stuff in the equation therefore overall results is 0.
    Can fix by using if statement to cut out that constrain and use Constraint.Skip (example of this is con_stubble_a in coremodel.py)

#. ERROR: Solver (glpk) returned non-zero return code (1) ERROR: See the solver log above for diagnostic information.
        This may be due to an inf number as a param

#. Error: cant evaluate a quadratic
        GLPK can’t evaluate a formula where one variable is being multiplied by another variable.
        Often to solve this you may need to add a variable and make a transfer constraint.

#. No value for uninitialized NumericValue object v_credit[ND$FLOW]
        This means the solver returned a no feasible solution
        This was caused once by having a very small negative number (-2e-16) returned from the crop sim (sometimes python returns a small negitive number instead of 0), the solver clearly didn’t like it (even though I thought it would just be treated as a 0 or as a small value)

#. Writing full solution: No value for uninitialized NumericValue object
        This means a variable has None as its value. This can happen for some variables eg sheep which
        have been masked out and hence are not really included in the model. When writing the full model the
        constraints are evaluated. This can cause errors if variables have None value. To fix this error, you
        should skip building constraints which are not required and/or use if statements when summing variables.
        An example is the mating dams propn bound.