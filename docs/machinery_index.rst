Machinery
==================

Background
----------

There is great variation in the type, age and investment in machinery between farms :cite:p:`RN2`.
To account for this, the model accommodates a range of machine options. The model user can then
select which machine complement is appropriate for their analysis. The machine option selected
determines the fixed cost, variable cost and machinery work rate. For the seeding activity,
the land management unit can also impact the variable cost rate and work rate. For example,
work rates are slower on heavy clay soils.

A machinery cost applies to all farm activities that require machinery. However, for both
seeding and harvest, the work rate of the machinery affects the timelines of completion.
For example, with smaller machinery, seeding takes longer, potentially incurring a late
seeding yield penalty. Additionally however, the model can hire contractual services
for seeding and harvest.

There are operating costs and depreciation costs associated with machinery. Operating costs
refer to expenses incurred through use and include the cost of fuel, oil, grease, repairs
and maintenance. Depreciation costs refer to the decline in value of the asset and consists
of two components, fixed and variable.


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



