Feed supply
==================

Feed budget
-----------
Energy is the primary nutritional constraint for extensive ruminant livestock enterprises (Rickards and Passmore, 1977).
As such, energy is the only nutritional element that is constrained in AFO to ensure that feed supply is greater
than or equal to the feed demand of the livestock (as measured by metabolisable energy). There is also a volume [#]_
constraint that limits the minimum diet quality to ensure that the voluntary feed intake capacities of the
livestock are sufficient to consume the quantity of feed selected. The volume of each feed source
(kg of intake capacity / kg of feed dry matter) varies depending on the feed quality (relative ingestibility)
and feed availability (relative availability) using relationships from :cite:p:`Freer2007`.

The feed supply from pastures, crop residues and supplementary feeds, is represented by changes in the type,
amount and quality of feed available during the year. The feed demand of livestock is represented as the
requirement for metabolisable energy and the feed intake capacity. The year is partitioned into 10 feed
periods. A feed budget is carried out for each feed period to ensure that the feed demand of the flock
can be met from the feed available on the farm. The dates of the feed periods during the growing season
are selected to group periods that have similar supply and demand characteristics. During the growing
season this is driven by the response of pasture growth to defoliation and the periods are shorter after
the break of season and just prior to senescence. During the dry feed phase the dates are selected to
minimise feed variation within each period and are shorter after pasture senescence and prior to the break
of season. The selection of the period definitions is likely to alter depending on the region being modelled.

Any of the 10 periods can be the period that limits the farm carrying capacity. This is representing far more
detail than underpins a typical gross margin analysis that considers a pre-defined feed limiting period of
the year. Furthermore, AFO includes the capacity to alter the live weight (LW) profile and hence feed
demand of any class of stock in any feed period with a concomitant change in production per head. This
links to the capacity for supplementary feeding the livestock to optimise the number of livestock carried
on the farm. As such AFO is much more detailed than a typical gross margins analysis of livestock
profitability. If AFO is compared to a simulation model the feed periods are equivalent to the
time-steps in the simulation model, however, they are much longer than a typical simulation model
that often considers daily time-steps. As such AFO represents the feeding options in less detail
than is possible in a dynamic simulation model, however, AFO has the advantage of optimising the
grazing management of the pastures and crop residues and optimising the target nutrition for each
class of stock during the year.

.. [#]  Volume - livestock intake capacity required to consume specific mass of a given feed.

Nutritive pools
^^^^^^^^^^^^^^^^
Cross subsidisation of volume is a problem that can occur in the feed budgets of linear programming models.
Cross subsidisation occurs if animals with divergent quality requirements are constrained by single
energy and volume constraints; the single constraint is termed a feed pool. For example, consider two
animals, one losing 100 g/hd/d and one gaining 150 g/hd/d. The first animal can achieve its target on
low quality feed whereas the second animal needs high quality feed. However, if both of these animals
were constrained using a single feed pool, then the total energy requirement and total intake capacity
is combined, such that feeding medium quality feed to both animals meets the constraints.
This is likely to be the optimal solution because the cost of feed by quality is a convex function
and therefore the cost-minimising solution is to provide an average quality to both classes of stock.
However, this is not a technically feasible solution. To reduce the possibility of cross-subsidisation
of volume while still limiting model size, the energy requirement and maximum volume constraints are
applied in multiple nutritive value pools, each spanning a small range of nutritive value (where nutritive
value = ME requirement / volume capacity). This is more efficient in reducing model size and complexity
than having a feed pool for each animal class.

The feed requirement (as measured by metabolisable energy) of each animal decision variable is a minimum
constraint in the matrix and sufficient feed must be available for each animal that is part of
the optimal solution. The feed selected in the optimal solution must also be of sufficient quality
that the quantity required to meet the animal's energy needs is within the intake capacity of
that animal. These requirements are represented by 2 constraints and together constitute a constraint
on nutritive value [#]_:

con_me: ME supplied from diet - ME required by stock <= 0

con_vol: Intake capacity of stock - Volume of feed <= 0

This system can represent confinement feeding explicitly by including a 'confinement' NV pool from which
pasture and dry paddock residues are excluded (unless they incur a cost and labour requirement
for "cut and carry"). This allows the reduced energy requirement for confined animals to be
represented.

The M/D of the paddock feed is scaled if the feed quality is above a threshold. This is to represent
the situation when the quality of paddock feed is high and voluntary feed intake would lead
to greater LW performance than the target LW pattern. To achieve the target LW pattern the
animals would have to be offered the feed for a period of time, then placed on a restricted
diet to bring them back to the target pattern, before being returned to the high quality
paddock feed. An example of this situation is wool growing ewes with access to high quality green lucerne during
summer (potential growth rate 150 g/hd/d) while the target profile is maintenance. The optimal
solution could be those animals grazing a combination of lucerne and other low quality
stubbles (the lucerne is providing the energy and the stubble is providing some energy
with no opportunity cost). The practical implementation would require the animals grazing
the lucerne for a while, then grazing the stubble for a while because they are unlikely
to volunteer a diet of 50% lucerne and 50% stubble to achieve maintenance. This grazing
strategy means that the animals would gain weight then lose weight then gain weight then
lose weight. The overall effect is to maintain, but gaining and then losing is not as
energetically efficient as maintenance, which means the diet quantities selected in the optimal
solution are not technically feasible in reality. Calculations based on efficiency of maintenance
(:math:`k_m`) and efficiency of storing energy (:math:`k_g`) show that the energy consumed
above the target level (or maintenance, whichever is greater) is only 50% as effective as if
the energy intake could have been rationed to the target level.

To represent this, the nutritive value of the feed that is above the average animal requirement
in that NV pool is reduced by a efficiency scalar that is input by the user. The logic is that
the ME intake above the target quality will be consumed and stored as fat and later mobilised.

.. [#] Nutritive value - megajoule of metabolisable energy per kg of livestock intake capacity.

Feed supply summary
-------------------
The main sources of feed considered in the model are; pasture (annual and/or perennial), crop residue (stubble)
and supplement (grain concentrates and conserved fodder). AFO also includes some novel feed sources such as early
season crop grazing, grazing standing fodder crops and salt land pastures.

The feed management decisions that are optimised can include:

    1.	Area of each pasture variety on each soil type.
    2.	Area of reseeded pasture based on paddock history.
    3.	Area of pasture manipulated and/or spray-topped based on paddock history and setting up for future land uses.
    4.	Grazing intensity of different pasture varieties on different soil types at different times of the year which manifests as a FOO profile of the pasture.
    5.	Timing and extent of pasture deferment.
    6.	Level and timing of supplementary feeding of hay or grain to each class of stock.
    7.	Grazing management of stubbles.

        a.	The time to start grazing of each stubble.
        b.	The class of stock that grazes the stubble.
        c.	The duration of grazing.
        d.	The amount of supplementary required in addition to stubble (to meet alternative LW profiles).
    8.	Area of fodder crops established and their grazing management.
    9.	Tactical grazing of standing crops in place of harvesting.
    10.	Amount of early season crop grazing.
    11.	Salt land pasture grazing management.
    12.	Conserving surplus pasture as hay or silage.
    13.	The level of growth modifier (e.g. nitrogen fertiliser) applied to pasture.

The model can also represent and compare (but not optimise in a single model solution):

    1.	The level of phosphate fertiliser application to pastures.
    2.	The Impact of varying pasture conservation limits.
    3.	Altering pasture cultivars on different land management units.

Feed supply equations
---------------------

.. toctree::
   :maxdepth: 1

   FeedsupplyFunctions


Pasture
-------------

Precalcs
^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   PastureFunctions

Pyomo
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   PasturePyomo


Crop grazing
-------------

Precalcs
^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   CropGrazing

Pyomo
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   CropGrazingPyomo


Crop residue (fodder & stubble)
-------------------------------

Precalcs
^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   CropResidueSim
   CropResidue

Pyomo
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   CropResiduePyomo

Salt land pasture
-------------------------------

Precalcs
^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   Saltbush

Pyomo
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   SaltbushPyomo

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