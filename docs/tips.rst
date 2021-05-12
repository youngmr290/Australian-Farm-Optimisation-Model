Tips
=======


Rotation
^^^^^^^^

Seasonal Variation
^^^^^^^^^^^^^^^^^^

Pyomo
^^^^^
Speed
Parameters: initialization of large parameters can take time (for example in the stock module). To save time mask out the 0 values when creating the param dict and set default argument in the parameter to 0. This make it quicker to build the dictionary in the first place and makes it much faster to initilise the parameter in pyomo (see stock pyomo for examples of this)

Constraints: building constraints can be slow however efficiency can be improved by using if statement which reduce the summing required. Additionally in certain circumstances (eg stock numbers) you can alter the parameters so that duplicate constraints are not built and then use the constraint.skip method to jump over.
It takes longer than I would have expected to evaluate if statements when building a pyomo constraint so time can be saved when summing by only evaluating required items in the if statement/s.

   



