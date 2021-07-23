Sensitivity
==================

General
--------
AFO uses sensitivity to allow the user to examine various scenarios. For example, price. Sensitivities are
controlled in exp.xlsx. The sensitivity values for a trial are read from the relevant trial in exp.xlsx and
stored in a dictionary. They are then applied in the relevant place in the code.

The sensitivity analyses are implemented in 2 distinct places during the execution of the model.

#. At the input level - the SA is directly applied to an input variable that have been read
   from one of the <input>.xlsx workbooks. This is done in one of the input modules (PropertyInputs.py, UniversalInputs.py
   or StructuralInputs.py). This is the preferred sensitivity method because it is simple and easy to check.
#. At a function level during the precalcs - the SA is carried out on intermediate variables
   in the body of the code.



How to use:
-----------
#. Check sensitivity.py to ensure the SA variable does not already exist.
#. Add the SA variable to the sensitivity.py module. This requires allocating the variable to a sensitivity dictionary, setting a default value and giving the variable a description.
#. Add it to the relevant module - if the SA is acting on an input (most common) the applying is done in the one of the input modules. if the SA is acting on an intermediate calculation the applying is done in the given module. To make this process easy there is a SA function fun.f_sa. Note: if you are applying multiple SA to the same variable (eg applying saa and sam) you must consider the order. The recommended order is  sai, sav, saa, sap, sam, sat, sar.
#. The sensitivity variable is not limited to being a single number. You can also initialize an array. In exp.xlsx you can assign to different slices of the array by specifying the indices.
#. Add the SA variable to the exp.xlsx document. The model will run without this step however this step is required inorder to use SA.

.. note::
    If multiple sensitivity adjustments (sam or sap and saa) are applied to a single input it must be done in the following way. This is the simplest method and follows the standard rules of maths.

    * .. math:: y = y * sam * (1+sap) + saa

Sensitivity Analysis variables
------------------------------

There are 6 types of sensitivity variables:

.. note:: The above formulas are used in that situation where the final value ``y`` is a different variable than
            the input value ``x``

#. (default) sam_name: sensitivity multiplier.

    .. math:: y = sam_x * x

#. sap_name: sensitivity proportion.

    .. math:: y = (1 + sap_x) * x

#. saa_name: sensitivity addition.

    .. math:: y = saa_x + x

#. sat_name: sensitivity towards (or away from) a target value.

    The SA is adjusting the gap between the current value and the specified target.
    sat = 0 is no change. sat = 1 is 'reach target'. sat = 0.5 is halfway to the target. sat >1 continues
    and overshoots the target. sat = -1 is twice as far from the target.

    .. math:: y = sat * (t - x) + x

    Default is 0

#. sar_name: sensitivity on a range.

    This is for variables that are confined to the range 0 to 1. Eg survival or mortality, frequency of
    reseeding a pasture in a continuous rotation. A positive sar increases the value towards to 1
    (and reaches 1 when the sar = 1), a negative sar reduces the value towards 0 (and reaches 0 when
    sar = -1). sar is the same as sat if  0 <= sar <= 1 and target = 1. But operates differently when sar <0.
    sar is like sat working in both directions with a target of 1 for sat >= 0 and a target of 0 with sat <= 0.
    There is a different formula depending if the sar is <0 or >0.
    sar_x < 0 (which is the same as sap):

    .. math:: y = (1 + sar_x) * x

    sar_x > 0 (which is the same as sat with a target of 1):

    .. math:: y = sar_x * (1 - x) + x

    The formula without an if statement:

    .. math:: y = x * (1 - abs(sar_x)) + max(0 , sar_x)

    If sar_x = 0 then y = x, if sar_x = -1 then y = 0, if sar_x = 1 then y = 1. If sar_x is outside the
    range -1 to 1 then y moves outside the range 0 to 1. This could be constrained with ``max(0,min(1,formula))``

    The default value is 0.

#. sav_name: sensitivity value, that is when the sensitivity value replaces the input value.

    .. math:: x = sav_x

   The default value is ‘-’ which results in no change.


