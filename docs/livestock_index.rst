Livestock
==================

Summary
--------
The stock module generates production parameters for livestock under a range of different conditions. These are
represented as different decision variables which allows the optimisation of a wide range of management decisions.

Data is generated for the following components:

    * Animal live weight and sale values.
    * Fleece production data, both quantity and quality (including the impact of ewe live weight profile during
      pregnancy on the lifetime performance of the progeny).
    * Dam reproduction levels.
    * Mortality rates of dams, progeny and dry animals.
    * Energy requirement profile and the nutritive value to achieve the target whole body energy profiles.
    * Methane emissions.
    * Foetal growth rates and birth weights for progeny.
    * Milk production and progeny weaning weights.
    * Husbandry cost and labour requirements.

.. note:: The livestock component of the sub-model currently only represents sheep, however, the option of
          representing cattle exists with little code alteration, simply addition of management inputs for the cattle.

Structure
---------
The livestock data generating sub-model is a simulation model that predicts feed intake, energy requirements,
production, emissions, husbandry and labour requirements during the lifetime of different classes of stock offered
different amounts of feed. The above values are calculated for each ‘class’ of stock represented. The feed supply
offered to the animals is not based on simulating a growing pasture, rather the FOO, digestibility and supplement
offered are inputs to the data generator. The outcome is the parameters required to define the production
possibilities that are included in the pyomo matrix of the AFO model.

The data generator model simulates the sires, dams and offspring from weaning to their latest possible sale age
and simulates the young at foot from birth to weaning. The initial animal for the sires, dams and offspring are
based on input values for liveweight, cfw and fibre diameter.

The prediction equations included in the data generator can be selected from any of a number of equation sources:

    #. GrazPlan equations as documented in Freer et al. 2012, which are an improved version of the
       Australian Feed Standards (SCA 1990).
    #. Research trials carried out by Murdoch University, DPIRD & DPI Victoria that have quantified
       the impact of changing nutrition on production. This research began with the Lifetime Wool Trial (refs here) but has continued with a suite of other projects (refs here)
    #. A selection of other sources that have developed equations to predict animal performance. Including:

        a. Blaxter & Clapperton for enteric methane emissions
        b. NGGI for emissions including methane and nitrous oxide
        c. Hutton Oddy’s group for alternative equations for heat production associated with maintenance and liveweight gain.

The relationships are calculated for a range of age groups for each of the following groups:

    #. Sires - entire males used for breeding purposes.
    #. Dams - females of reproductive age, mated or unmated.
    #. Young at foot - male and female progeny prior to weaning.
    #. Offspring - female or male progeny of the dams from post weaning until sale.

For each animal group there are a range of classes represented and for each class the range of optimised
management decisions are included. The classes and the management decisions are represented as axes (or dimensions)
of a multi-dimensional numpy array. Note: numpy, like python, works on a base 0 system so the numbering of the
slices of each axis begins at 0. The array axes are:

The management optimisation structure
-------------------------------------
To disaggregate the matrix the livestock management options are grouped into decision variable periods (DVP). At a
DVP junction the animals are distributed from the decision variables of the previous period to the decision
variables of the next generator period. This is necessary when the sheep change clusters, for example at scanning
the pregnancy status of the dam is identified and if it is a separate activity it can then be managed differently
to a dry dam, whereas prior to scanning they are clustered and must be managed the same. The timing of the DVPs
is based on periods when the management groups change, this includes pre-joining, scanning and birth.
Animal sale options can be defined for each DVP. The selling opportunity is represented by the t axis and
(in version 1) there is opportunity for a maximum of 2 sale time per DVP for dams and offspring. Although
these selling opportunities can be masked from the matrix if sales in a particular DVP is not desired.

For the dams the annual cycle starts with prejoining and the other DVPs are based on the reproduction calendar:
early pregnancy (period_between_prejoinscan), late pregnancy (period_between_scanbirth), lactation and post weaning
recovery (period_between_birthprejoin). The Dams sale time is just after the main shearing, and additionally dry
ewes can be sold after scanning.

For the offspring (dry stock) the annual cycle starts with shearing. The offspring only require one DVP
for the year because the animals do not change class and require distributing during the year as do the dams.
The 2 sale options for Offspring can be input as a combination of either a target date or a target liveweight
and the animals can be shorn at sale (retained animals are shorn at the main shearing).

The nutrition optimisation structure
------------------------------------
Representing different nutrition levels is useful in analyses where it is expected that the technique or
technology being tested will alter the optimal feed supply to the animals. This may be because the technology
alters the feed supply or the production efficiency of the animals, either of which may alter optimal feed allocation.

Variable liveweight patterns also allows realistic evaluation of nutrition strategies that don’t include the
implicit assumption that managers can ration the intake of their livestock. The model can be constrained to
represent animals grazing to their voluntary feed intake (by making the volume constraint an “E” constraint)
and feasible options exist that include increasing animal intake.

The range of nutrition levels are represented by profiles that are continuous for the entire year. At the end
of the nutrition cycle (year) the range of final liveweights are ‘condensed’ back to a range of starting weights
for the start of the next nutrition cycle.

A feed variation period (FVP) is a user defined period at the start of which the nutrition profiles can
diverge to different levels (eg. medium, high or low). The feed variation periods are defined to
represent time points when either:

    #. An alternative nutrition level may be optimal for different classes of stock.
    #. Quantifying the value of altering nutrition is desired for an analysis report.

Further information regarding the feedsuppy patterns used in the livestock generator can be found at the
following link:

.. toctree::
   :maxdepth: 1

   StockFeedSupply


Precalcs
---------

.. toctree::
   :maxdepth: 1

   StockFunctions

Pyomo
-----

.. toctree::
   :maxdepth: 1

   StockPyomo