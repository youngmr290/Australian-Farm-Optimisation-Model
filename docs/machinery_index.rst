Machinery
==================

Background
----------

There is great variation in the type, age and investment in machinery between farms :cite:p:`RN2`.
To account for this, the model accommodates a range of machinery options. The model user can then
select which machinery complement is appropriate for their analysis. The machinery option selected
determines the fixed cost, variable cost and machinery work rate. For the seeding activity,
the land management unit can also impact the variable cost rate and work rate. For example,
work rates are slower on heavy clay soils.

A machinery cost applies to all farm activities that require machinery. However, for both
seeding and harvest, the work rate of the machinery affects the timelines of completion.
For example, with smaller machinery, seeding takes longer, potentially incurring a late
seeding yield penalty. Additionally, the model can hire contract services
for seeding and harvest, although this can be limited by the user.

There are operating costs and depreciation costs associated with machinery. Operating costs refer
to expenses incurred during usage such as for fuel, oil, grease, repairs and maintenance. Depreciation
costs represent the decline in the value of the asset. Depreciation is made up of two components.
Firstly, a fixed component which represents depreciation even if the machine is not used and secondly,
a variable component which represents that asset value reduces faster with increased usage.



Precalcs
--------

.. toctree::
   :maxdepth: 1

   Mach

Pyomo
-----

.. toctree::
   :maxdepth: 1

   MachPyomo



