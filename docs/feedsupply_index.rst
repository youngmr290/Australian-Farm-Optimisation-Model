Feed supply
==================

Summary
-------
The feed supply for livestock is represented by changes in the type, amount and quality of feed
throughout the year. The year is partitioned into 10 feed periods. The dates of the feed periods
are selected to minimise feed variation within each period. Hence the period definitions may
alter depending on the farm being modelled. Energy is the main nutritional limitation for the
livestock enterprise :cite:p:`RN2` thus, energy is the only nutritional value
tracked in the model. The volume [#]_ of each feed is also track because the voluntary feed intake
capacities of sheep vary depending on the feed quality (relative ingestibility) and feed
availability (relative availability) :cite:p:`Freer2007`. The main sources of feed considered
in the model are; pasture, stubble and supplement. See below for more information.

.. [#]  Volume - livestock intake capacity required to consume specific mass of a given feed.

Nutritive pools
^^^^^^^^^^^^^^^^
The feed requirement (as measured by metabolisable energy) of each animal activity is a minimum
constraint in the matrix and sufficient feed must be available for each animal that is part of
the optimal solution. The feed selected in the optimal solution must also be of sufficient quality
that the quantity required to meet the animal's energy needs is within the intake capacity of
that animal. These requirements are represented by 2 constraints:

Con_me: ME supplied from diet - ME required by stock <= 0

Con_vol: Intake capacity of stock - Volume of feed <= 0

If animal with different nutritive values [#]_ are in the same constraint cross-subsidisation of
volume from animals that require a low quality feed to animals that require a high quality feed
can occur. For example, consider two animals, one that is maintaining weight and one that is
gaining 150 g/hd/d. The first animal can maintain weight on medium quality feed and still have
intake capacity left over. The second animal can not consume enough medium quality feed to meet
its energy requirement to gain 150 g/hd/day. Therefore, the second animal must consume some
high quality feed.  However, if both of these animals were bound by the same energy and volume
constraints then the leftover intake capacity from animal one could be essentially be used by
animal two. Meaning that animal two can now meet its energy requirement. Cross subsidisation
is selected in the optimisation because the cost of feed by quality is a convex function and
therefore the cost minimising solution is to provide an average quality to both classes of
stock. Which is not a technically feasible solution.

To keep the model size as small as possible there is not an energy and volume constraint for
each animal activity. Instead, to reduce cross subsidisation animal activities are allocated
into pools based on their nutritive value during that feed period. Each pool has an energy
and volume constraint.

This system can represent feedlotting explicitly by including a feed lot NV pool from which
pasture and dry paddock residues are excluded (unless they incur a cost and labour requirement
for “cut and carry”). This allows the reduced energy requirement for confined animals to be
represented.

The M/D of the feed is scaled if the feed quality is above a threshold. This is to represent
the situation when the quality of paddock feed is high and voluntary feed intake would lead
to greater LW performance than the target LW pattern. To achieve the target LW pattern the
animals would have to be offered the feed for a period of time, then placed on a restricted
diet to bring them back to the target pattern before being returned to the high quality
paddock feed. For example, wool growing ewes with access to high quality green lucerne during
summer (potential growth rate 150 g/hd/d) while the target profile is maintenance. The optimal
solution could be those animals grazing a combination of lucerne and other low quality
stubbles (the lucerne is providing the energy and the stubble is providing some energy
with no opportunity cost). The practical implementation would require the animals grazing
the lucerne for a while, then grazing the stubble for a while because they are unlikely
to volunteer a diet of 50% lucerne and 50% stubble to achieve maintenance. This grazing
strategy means that the animals would gain weight then lose weight then gain weight then
lose weight. The overall effect is to maintain but gaining and then losing is not as
energetically efficient as maintenance, which means the diet quantities selected in the optimal
solution are not technically feasible in reality. Calculations show that the energy consumed
above the target level (or maintenance, whichever is greater) is only 50% as effective as if
the energy intake could have been rationed to the target level.

The feed quality required to achieve the target LW pattern varies for each feed quality pool.
This quality is calculated in the sheep simulation and is increased to the quality required
for maintenance based on a user input. The user also inputs the efficiency which the extra
energy is utilised. Reasons to adjust the efficiency value include:

#. A sensitivity analysis to determine the importance of inclusion of this assumption.

#. LW gain is desired to turn the animals off for sale. In this case the extra energy gained will
   not be mobilised in order to achieve the target pattern. In fact, if the animals are to be sold
   a target LW pattern that gains weight more quickly is likely to be more profitable, so
   penalising the efficiency would take the solution further from the ‘best’ solution.

#. The feed can be effectively rationed so that LW gain beyond the target does not occur.
   Note: the system of rationing may need to be included in the assumptions depending whether
   (in the pyomo optimisation) the unutilised intake capacity of the animals can be utilised to
   consume low quality ‘cheap’ dry residues.

The calculation is carried out with a concept of ‘energy of the feed per kg of intake capacity required’
rather than per kg of DM. This is termed the feed energy concentration (FEC), this is compared with
the FEC required by the animal to achieve the target LW pattern (or maintenance) and the surplus
is scaled by the efficiency. The logic is that the quality above the target quality will be
consumed and stored as fat and later mobilised. That energy has a lower efficiency.

.. [#] Nutritive value – megajoule of metabolisable energy per kg of intake capacity.

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