Feed supply
==================

Summary
-------
The feed supply for livestock is represented by changes in the type, amount and quality of feed
throughout the year. The year is partitioned into 10 feed periods. The feed periods in AFO are
equivalent to the time-step in a simulation model, however, they are much longer than a typical
simulation model as a requirement of computing capacity. The dates of the feed periods
during the growing season are selected to group periods that have a similar response of pasture
growth to defoliation. During the dry feed phase the dates are selected to minimise feed variation
within each period and are shorter after pasture senescence and the break of season. The selection
of the period definitions is likely to alter depending on the region being modelled.

Energy is the primary nutritional constraint for extensive ruminant livestock enterprises
:cite:p:`RN2` (maybe Phil would have a better nutritionist reference to back that comment) as such,
energy is the only nutritional element that is constrained in the model to ensure that supply is
greater than demand.
The volume [#]_ of each feed is also constrained to ensure that the diet selection is feasible and
the voluntary feed intake capacities of sheep are sufficient to consume the quantity of feed
selected. The volume of each feed source (kg of intake capacity / kg of feed DM) varies depending on
the feed quality (relative ingestibility) and feed availability (relative availability) :cite:p:`Freer2007`.

The main sources of feed considered in the model are; pasture, stubble and supplement. See below
for more information.

.. [#]  Volume - livestock intake capacity required to consume specific mass of a given feed.

Nutritive pools
^^^^^^^^^^^^^^^^
The feed requirement (as measured by metabolisable energy) of each animal decision variable is a minimum
constraint in the matrix and sufficient feed must be available for each animal that is part of
the optimal solution. The feed selected in the optimal solution must also be of sufficient quality
that the quantity required to meet the animal's energy needs is within the intake capacity of
that animal. These requirements are represented by 2 constraints and together constitute a constraint
on nutritive value [#]_:

con_me: ME supplied from diet - ME required by stock <= 0

con_vol: Intake capacity of stock - Volume of feed <= 0

If animal with different nutritive value requirements are in the same nutritive value pool
(same constraints) cross-subsidisation of
volume from animals that require a low quality feed to animals that require a high quality feed
can occur. For example, consider two animals, one that is losing 100 g/hd/d and one that is
gaining 150 g/hd/d. The first animal can achieve it's target on low quality feed whereas the
second animal would have to consume
high quality feed.  However, if both of these animals were constrained using a single nutritive
value pool with one energy and one volume
constraints then the energy requirement and intake capacity is combined, such that feeding
medium quality feed to both animals meets the constraints. This is likely to be the optimal solution
because the cost of feed by quality is a convex function and
therefore the cost minimising solution is to provide an average quality to both classes of
stock. However, this is not a technically feasible solution.

To minimise model size there is not an energy and volume constraint for
each animal activity. Instead, to reduce cross subsidisation animal activities are allocated
into pools based on their nutritive value during that feed period. Each pool has an energy
and volume constraint.

This system can represent confinement feeding explicitly by including a 'confinement' NV pool from which
pasture and dry paddock residues are excluded (unless they incur a cost and labour requirement
for "cut and carry"). This allows the reduced energy requirement for confined animals to be
represented.

The M/D of the paddock feed is scaled if the feed quality is above a threshold. This is to represent
the situation when the quality of paddock feed is high and voluntary feed intake would lead
to greater LW performance than the target LW pattern. To achieve the target LW pattern the
animals would have to be offered the feed for a period of time, then placed on a restricted
diet to bring them back to the target pattern before being returned to the high quality
paddock feed. An example of this situation is wool growing ewes with access to high quality green lucerne during
summer (potential growth rate 150 g/hd/d) while the target profile is maintenance. The optimal
solution could be those animals grazing a combination of lucerne and other low quality
stubbles (the lucerne is providing the energy and the stubble is providing some energy
with no opportunity cost). The practical implementation would require the animals grazing
the lucerne for a while, then grazing the stubble for a while because they are unlikely
to volunteer a diet of 50% lucerne and 50% stubble to achieve maintenance. This grazing
strategy means that the animals would gain weight then lose weight then gain weight then
lose weight. The overall effect is to maintain but gaining and then losing is not as
energetically efficient as maintenance, which means the diet quantities selected in the optimal
solution are not technically feasible in reality. Calculations based on efficiency of maintenance
(:math:`k_m`) and efficiency of storing energy (:math:`k_g`) show that the energy consumed
above the target level (or maintenance, whichever is greater) is only 50% as effective as if
the energy intake could have been rationed to the target level.

To represent this the nutritive value of the feed that is above the average animal requirement
in that NV pool is reduced by a efficiency scalar that is input by the user. The logic is that
the ME intake above the target quality will be consumed and stored as fat and later mobilised.

.. [#] Nutritive value – megajoule of metabolisable energy per kg of livestock intake capacity.

Feed supply equations
^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   FeedsupplyFunctions


Pasture
-------------

Precalcs
^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   Pasture
   PastureFunctions

Pyomo
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   PasturePyomo


Stubble
-------------

Precalcs
^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   Stubble

Pyomo
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   StubblePyomo

Supplement
-------------

Precalcs
^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   SupFeed

Pyomo
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   SupFeedPyomo