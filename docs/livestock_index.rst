.. _livestock_ref:

Livestock
==================

Summary
--------
The stock module generates production parameters for livestock under a range of different conditions. These are
represented as different decision variables which allows the optimisation of a wide range of management decisions.

AFO includes a livestock data generator that generates the production parameters for livestock for a user
specified number of nutritional profiles. It is based on the relationships that underpin the GrazPlan suite
of models as described by (Freer et al., 2007) and updated with production relationships developed in other
research projects. Data is generated for the following components:

    •	Animal liveweight and sale values.
    •	Energy requirement profile and the nutritive value to achieve the target whole body energy profiles.
    •	Fleece production data; both quantity and quality (including the impact of the ewe liveweight profile during pregnancy on the lifetime performance of the progeny).
    •	Dam reproductive rate; represented by the proportion of ewes that are empty, single-, twin- and triplet-bearing.
    •	Perinatal survival of single, twin and triplet born lambs.
    •	Perinatal ewe survival associated with pregnancy toxaemia, dystocia and lambing difficulties.
    •	Mortality rates of dams, progeny and dry animals related to nutrition level.
    •	Foetal growth rates and birth weights for progeny.
    •	Milk production and progeny weaning weights.
    •	Husbandry cost and labour requirements.
    •	Methane emissions.

The values above are calculated for each 'class' of stock, for its lifetime. The feed supply offered to the animals
is not based on simulating a growing pasture, rather the FOO, digestibility and supplement offered are inputs to
the data generator. The outcome is the parameters required to define the production possibilities that are included
in the matrix of the AFO model.

The data generator model simulates the sires, dams and offspring from weaning to their latest possible sale age
and simulates the young at foot from birth to weaning. The initial animal for the sires, dams and offspring are
based on input values for liveweight, clean fleece weight and fibre diameter.

The prediction equations included in the data generator can be selected from a range of equation sources. Currently:

    1.	GrazPlan equations as documented in Freer et al. (2012), which are an improved version of the Australian
        Feed Standards (SCA 1990).
    2.	Research trials carried out by Murdoch University, DPIRD and DPI Victoria that have quantified
        the impact of changing nutrition on production. This research began with the Lifetime Wool
        Trial (Oldham et al., 2011, Thompson et al., 2011)  but has continued with a suite of other projects
        including Lifetime Maternals (Behrendt et al., 2019) and Mob size (Lockwood et al., 2019).
    3.	A selection of other sources that have developed equations to predict animal performance including:

        a.	Blaxter and Clapperton for enteric methane emissions (Blaxter and Clapperton, 1965).
        b.	NGGI for emissions including methane and nitrous oxide (DISER, 2021).
        c.	Hutton Oddy's group (NSW DPI) for alternative equations for heat production associated with maintenance and liveweight gain (Oddy et al., 2019).

.. note:: The livestock component of AFO currently only represents sheep, however, with some work cattle could be added.

In the full model the livestock management decisions that are optimised can include:

    1.	Number of animals carried (i.e. stocking rate) based on whole flock, whole year feed requirements and whole farm feed supply.
    2.	Sale age and weight of each animal group.
    3.	The proportion of the ewe flock mated to different sire genotypes (pure bred, maternal type or terminal)
    4.	The proportion of the ewe flock that is a first cross dam  mated to a terminal sire (the dam cross is between the purebred and the maternal genotype)
    5.	The reproductive life of dams in the flock (based on whole flock feed requirements, value of wool variation by age, reproduction variation by age, the value of CFA dams at different ages, the selection pressure that can be applied on replacement ewes).
    6.	Whether to mate ewe lambs and the optimal proportion to mate.
    7.	A trading operation for dry animals. This can be either a short term trade with an aim to fatten animals or a multi-year trade to produce wool.
    8.	A ewe flock based on buying in ewes and mating all ewes to a terminal sire to produce first-cross lambs for sale. The age at purchase and sale can be optimised.
    9.	Diet selection for the animals based on the feed base options represented in the model including supplementary feeding.
    10.	Time in confinement and/or feed lot (note, this reduces the animals' energy requirements due to reduced walking).
    11.	Nutrition profile of the animals during the year which is related to reproduction status, wool value, sale objectives and unfolding climate conditions.
    12.	Differential feeding of dams based on litter size, lactation number and foetal age, provided the dams are pregnancy-scanned or assessed for 'gave birth and lost'
    13.	Optimal replacement policy based on:

        a.	the change in reproduction and production over the animal's lifetime,
        b.	the potential to increase per head production through culling and a response in the current generation.
    14.	Optimal weaning age for each dam age group.

Furthermore, constraints can be applied to the model to limit:

    1.	Level of enteric greenhouse gas emissions and emissions of nitrous oxide from faeces and urine.
    2.	Bare ground during the summer/autumn period
    3.	Animal mortality or liveweight loss during the feed limiting period of the year
    4.	Animals to graze at their voluntary feed intake level (i.e. that intake reflects the FOO and DMD
        offered to the animals i.e. feed is not rationed through active management of the stock. This has
        little effect unless the pasture management is also constrained to limit variation of the FOO and quality profile)

The model can also represent and compare (but not optimise in a single model solution):

    1.	The length of the joining period (measured in the number of cycles mated); including the trade-off between the number of ewes conceiving and the distribution of size and energy requirements of the later born progeny.
    2.	The age that the young ewes are mated. For example, a 7 month mating versus a 8.5 month mating for ewe lambs.
    3.	Accelerated lambing where ewes are mated every 8 months and therefore have 3 lambing opportunities in 2 years.
    4.	Variation in timing of lamb, hogget and adult shearing.
    5.	More frequent shearing. For example, adopting a shearing interval of 6, 8 or 12 months.

The relationships are calculated for a range of age groups for each of the following groups:

    #. Sires - entire males used for breeding purposes.
    #. Dams - females of reproductive age, mated or unmated.
    #. Young at foot - male and female progeny prior to weaning.
    #. Offspring - female or male progeny of the dams from post weaning until sale.

For each animal group there are a range of classes represented, and for each class the range of optimised
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
(in version 1) there is opportunity for a maximum of 2 sale times per DVP for dams and offspring. Although
these selling opportunities can be masked from the matrix if sales in a particular DVP is not desired.

For the dams, the annual cycle starts with prejoining and the other DVPs are based on the reproduction calendar:
early pregnancy (period_between_prejoinscan), late pregnancy (period_between_scanbirth), lactation and post weaning
recovery (period_between_birthprejoin). The Dams sale time is just after the main shearing, and additionally dry
ewes can be sold after scanning.

For the offspring (dry stock), the annual cycle starts with shearing. The offspring only require one DVP
for the year because the animals do not change class and require distributing during the year as do the dams.
The 2 sale options for offspring can be input as a combination of either a target date or a target liveweight,
and the animals can be shorn at sale (retained animals are shorn at the main shearing).

The nutrition optimisation structure
------------------------------------
A powerful and advanced feature of AFO is its ability to optimise livestock liveweight/nutrition profiles.
AFO does this by generating production parameters for animals following a range of nutrition profiles (up to
2000 profiles for each class of sheep can be concurrently evaluated). These are represented as different
decision variables which allows the optimisation of a wide range of management decisions. The total feed
requirement and the minimum diet quality can vary for each feed period for each livestock decision variable.
The range of nutrition levels are represented by profiles that are continuous for the entire year. At the
end of the nutrition cycle (year) the range of final live weights are 'condensed' back to a range of starting
weights for the start of the next nutrition cycle. This capacity allows AFO to differentially feed animals
based on reproduction, sale goals and feed supply based on land use selection while minimising model size
and computing resources required.

Representing different nutrition levels is useful in analyses where it is expected that the technique or
technology being tested will alter the optimal feed supply to the animals. This may be because the technology
alters the feed supply or the production efficiency of the animals, either of which may alter optimal feed allocation.

Variable liveweight patterns also allows realistic evaluation of nutrition strategies that don't include the
implicit assumption that managers can ration the intake of their livestock. The model can be constrained to
represent animals grazing to their voluntary feed intake (by making the volume constraint an "E" constraint)
and feasible options exist that include increasing animal intake.

A feed variation period (FVP) is a user defined period at the start of which the nutrition profiles can
diverge to different levels (eg. medium, high or low). The feed variation periods are defined to
represent time points when either:

    #. An alternative nutrition level may be optimal for different classes of stock.
    #. Quantifying the value of altering nutrition is desired for an analysis report.

Further information regarding the feed suppy patterns used in the livestock generator can be found at the
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