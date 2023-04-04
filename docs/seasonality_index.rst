Uncertainty and short term tactics
==================================
The two main sources of uncertainty in Australian farming systems are the high variance of world
prices for most agricultural commodities (e.g. Hazell et al., 1990) and climate variability, which
results in significant production variability (Feng et al., 2022, Laurie et al., 2018). To deal with
short-term variability within the system, farmers implement tactical adjustments that deviate from
the long-term strategic plan. Tactical adjustments are applied in response to unfolding opportunities
or threats and aim to generate additional income or to avoid losses (Pannell et al., 2000). In AFO
price and weather variation is represented as a number of discrete options along with a range of
relevant tactical adjustment options.

In AFO the user has the capacity to included or exclude price and weather uncertainty.

Weather uncertainty
-------------------
Weather uncertainty in AFO can be included or excluded and the representation of uncertainty can be more or
less detailed. Variability or uncertainty is represented using the modelling approach of discrete stochastic
programming (Cocks, 1968, Rae, 1971, Crean et al., 2013). Discrete stochastic programming is a formulation
of a decision tree. It requires the explicit specification of management choices and their possible consequences.
The nodes or event forks are usually represented by a relatively small number of discrete outcomes. The
inclusion of uncertainty allows management decisions to be made as the year unfolds
(Norton and Hazell, 1986, Hardaker et al., 1991), which has been noted as an important aspect of farm
management (Pannell et al., 2000, McCown et al., 2006). The three different AFO frameworks are:

    (i)	Deterministic static equilibrium programming (SE) (e.g. MIDAS - (e.g. Kingwell and Pannell, 1987)).
        SE represents the farming system with a single discrete state. Representing a farm system as a single
        state requires use of expected inputs and outputs (e.g. the wheat yield is the average of all years).
        It assumes every year is the same and the finishing state equals the starting state. Thus, only strategic
        (long term) management is represented and management does not change between years because there is only
        one branch of the decision tree being represented.
    (ii) Single year discrete stochastic programming (DSP) (e.g. MUDAS - (e.g. Kingwell et al., 1991)). DSP
         represents the farm system with multiple discrete states where each state represents a different
         weather-year that can have separate inputs and management decisions to reflect different prices and weather
         conditions. All states begin from a common point that is determined by the weighted average of the end of
         all the weather-years, but then separate at various nodes during the production year to unveil the particular nature
         of that weather-year. Once a weather-year has been identified, subsequent decisions can be differentiated based
         on the known information about that given weather-year. For example, in AFO one node is the start of the growing
         season or 'break of season'. If that start is what is known colloquially as an 'early break', then after that
         starting point those types of weather-years can be managed differently to weather-years where the break occurs
         later. For example, in an early break it may be optimal to crop more area and run a higher stocking rate and
         vice-versa for a late break, although these decisions can only be made after the break of season is known. However,
         at the break of the season the subsequent conditions are uncertain (e.g. 30% chance of a poor spring and a 70%
         chance of a good spring). Thus, the decisions made at the break of season must factor in future uncertainty
         about the spring conditions. DSP examines each possible outcome and its probability to determine the optimal
         decisions. These decisions are a suite of tactical adjustments made at each node that complement or adjust
         an overarching farm management strategy.
    (iii) Multi-year discrete stochastic programming (SQ) (Xie and Huang, 2018). SQ is similar to DSP with
          the difference being that the discrete states represent a sequence of weather-years in equilibrium rather
          than a single year in equilibrium. Optimisation of management within the sequence of weather years fully
          accounts for the temporal effects of management change between years. In AFO, the production data in the
          SQ is the same as the DSP for the individual weather-years. The difference is that the SQ framework more
          accurately represents carryover management implications from the previous year. For example, if stock
          were sold in the previous year the current year would start from a destocked position.


Defining the seasons
^^^^^^^^^^^^^^^^^^^^
The seasons (weather years) that will be modelled need to be selected based on:

    - Indicators of the season (points in the year when a season can be identified)
    - Tactics that can be implemented
    - Outcome of the season (productivity)

The aim is to select seasons with large production variations that have clear indicators that give the most potential
for implementing tactical management. This is achieved by selecting indicators that maximise the
differentiation of outcome between the season groups. But it also needs to account for the timing of the indicator
and the capacity to make tactical decisions (eg. there is no value in finding out it is a good season at harvest).
For example, break of season timing is a clear indicator which can be used to classify season types (e.g. season 1
could be an early break and season 2 could be a late break). The break of season influences production and it is
early enough in the year that farmers can adjust their strategy accordingly.
There can be multiple indicators used to define the season types. For example:

    - Date of season break
    - Summer rainfall level
    - Winter temperature
    - Spring rainfall
    - Crop growth during the previous spring may affect disease build up and carry into the next season which could
      alter crop yields and therefore rotation choice.
    - Rainfall during summer (in the previous season) may affect the soil moisture profile which could affect the
      crop yield outlook for the next season. In the wheatbelt, summer rain may increase the yield prospects
      whereas in the high rainfall areas it may reduce yields due to extra waterlogging.
    - Season forecast might also be used as a season indicator if the forecast is associated with a change
      in probability of the otherwise identified weather-years.

The timing of the weather-years is defined by nodes when they differentiate from their parents. At these nodes more
information becomes available and it is then possible to identify a weather-year from years that have been similar
to that point. This can be visualised as a tree that is branching at each node towards the final amount of detail
for the total number of season types that will be described (Figure 1). This is termed the parent and child at that node.

.. image:: season_tree.png
    :height: 400

Figure 1: A visualisation of 6 weather-years that evolve as information becomes available during the year.

The convention used in AFO, in order to minimise model size, is that at each node where new weather-years are
initiated one of the child weather-years retains the parent description (which is either a slice in a parameter
array or an element of a pyomo set) and the other(s) take on a new description. An alternative visualisation
that is more similar to the design in AFO is Figure 2. In the convention of AFO weather-year0 is the parent of
weather-year 2 at the early break, and is the parent of weather-year1 at the spring node.

.. image:: afo_season_representation.png

Figure 2: Alternative visualisation of 6 weather-years with 3 season breaks and 2 different spring outcomes.

The weather-years that might need more thought to represent are:

    - False break. Pasture germinates and then a portion dies prior to receiving follow-up rainfall. This could be
      represented as:

        - the break that turns out to be false is the start of the next weather-year

      Is there data on the proportion of death in a false break based on the defoliation. If there is this could be
      built into the greenha decision variables with different senescence by FOO level, or a fixed senescence in kg/ha
      (so less proportion of death with lower grazing). This relationship would determine the optimum grazing in a false break.
    - Summer rainfall affecting the quality of the dry pasture from the previous growing season and also affecting
      the crop yields in the next growing season (at least in the wheatbelt on the heavier soils that store moisture).

Each season type is allocated a probability based on the historical data (or other method) of occurrence. A season type
with 0 probability could be excluded from the model to reduce size.

Current definition
^^^^^^^^^^^^^^^^^^
The key tactical decisions are (see below for the full list):

    #. Stock liveweight
    #. Pasture deferment
    #. Stock sales
    #. Rotation choice
    #. Biomass usage (ie harvest, fodder or hay)

Determining season types for crop is quite simple. Decisions about rotation are made at the
time of seeding and decisions about harvest are made at the end of the season.
Statistical modelling indicated that crop production is primarily
determined by time of season break and spring rainfall. Thus, for crop, the break of season
and spring conditions are good seasonal indicators that capture the majority of the production variance
whilst allowing realistic tactical decisions to be made.

Pasture/livestock is more complicated because their decisions are more continuous. However, discussion with consultants
and statistical modelling has identified the key points in the season where decisions are made.
The break of season timing was identified as most important indicator for pasture and livestock management
because time of break is a good indicator of the amount of early feed available. Feed is most limiting early in the
season, and thus the timing of break is a key indicator for pasture management. For example, in an early break
season a decision could be to stop feedlotting and graze pastures earlier than usual.
There is a possibility of the season breaking without any follow up rain causing germinated plants to die. False breaks
tend to have a greater impact on pasture productivity because some false breaks occur before crops have been sown
and crops tend to be more resilient (farmers have suggested the following explanations as to why: (i) because they
are not being grazed (ii) because the crop furrow has greater moisture retention).
A false break is most likely to occur after an early break thus, to save model size, a false break is only
represented for early break seasons.
Winter productivity is less variable between seasons due to the
lower temperatures and therefore no winter decision points were included. Spring conditions largely explain
total pasture production and dry summer feed quality and availability which may impact management decisions
such as livestock liveweight pattern and sale timing.

Overall for both crop and pasture/livestock time of sowing, follow up rains (only for the early break season)
and spring rainfall were determined as key indicators of seasonal management and outcome.
The season definition described above was defined for the Great Southern
region of Western Australia. The season definition may vary for different regions.


Season details
--------------

Season start
^^^^^^^^^^^^
The season start is the point when the previous season ends and the new season begins. Choice of season start must be
thought through carefully. From a crop and pasture perspective a changeover in autumn (season break) seems most
sensible because that aligns with a change of crop year and the beginning of the new pasture growth cycle. However,
season start is further complicated by livestock because liveweight needs to be averaged and distributed (such that
lw essentially has a singleton z axis at the beginning and end of each year) at the start of each season. This is
required so that each season starts and finishes in a common position. The live weights also need to be condensed
at prejoining because there are 81 patterns each year which start from 3 common points (this is required to save
space). Difference in production (wool and repo) due to how an animal got to a given live weight (e.g. gain then
lose vs maintain) is lost when averaging/condensing the liveweights. Thus, from the perspective of livestock, it would
be optimal to match the season start with prejoining which occurs at the beginning of the sheep reproduction cycle
so that minimal reproduction information is lost.

This leaves two reasonable options for season start to occur:

    #. Start as close to the beginning of reproduction cycle as possible. This tracks production more accurately
       particularly if only one TOL is used (because the prejoining will be similar time for all animals)
       but means important aspects of season type may not be represented accurately e.g. season start in December
       means you don't capture the impacts of a poor spring on the following year because at season start the
       seasons are averaged.
    #. Start at the break of season. This means some production variation between livestock in each season is lost
       but the crop and pasture will be much more accurate.

Dry seeding start has been selected as the point all seasons start from. This maximises the accuracy of pasture
supply in different seasons. It also means that dry seeding in the earlier breaks occurs in all the subsequent breaks.

Periods
^^^^^^^
When the seasons are unclustered (identified) the parent season type needs to transfer all the 'starting' info to the
newly created season type. Thus, season nodes need to be added to all period arrays used by model activities (if the
season nodes are not periods then there is not a new activity and thus the model can’t manage the seasons differently).
For the stock this means each season node has to be a DVP so that the animals can be transferred. For the pasture it
means each season node needs to be a feed period to allow the transfer of feed. Furthermore, season start node is
required so that activity levels in each season can start from a common point (weighted average of the end condition).

Price Variation
---------------

.. toctree::
   :maxdepth: 1

   PriceVariation

Tactics
--------
There are many tactical or adjustment options represented in AFO that reflect a farmer’s reality. The
tactics are similar to, but an expansion of those represented by Kingwell et al. (1992) and revolve
around land use area adjustment, land use inputs, whether a crop is harvested, baled or grazed as a
standing crop, intensity of machinery use, labour utilisation, seasonal sheep liveweight patterns,
tactical sale of sheep, grazing management of pasture and stubble, and supplementary feeding. The
same tactical adjustments are made to all weather-years that are indistinguishable from one another
at the time a tactical decision is implemented. Such weather-years are clustered at that decision
point, as the node that later differentiates these weather-years is still in the future. By illustration,
tactical adjustments selected at the early season break node have to be the same for all weather-years
that have an early break, because at the time of making the break of season tactical decision the
occurrence of follow-up rain and the spring conditions are unknown. Typical tactical adjustments include:

    •	Rotation phase - The area of each land use can be adjusted depending on the date of season break or other early indicators such as residual soil moisture from summer rainfall. Choice of rotation phase can also be delayed at the break of season, for example waiting to ensure it is not a false break. During this period of delay, pasture will germinate on these paddocks and is able to be grazed (the level of germination is dependent on the rotation history of the paddock). The potential for tactical adjustment of rotation phases depends on the land use history on each LMU because the choice for current land use is constrained by the land use history. Likewise, tactical adjustment affects subsequent rotation phase choice through its impact on altering the land use history provided.
    •	Land use inputs – In favourable weather-years additional chemicals and fertiliser can be applied to maximise yields and vice versa in poor weather-years. Note, in this analysis the input level for each land use on each land management unit in each weather-year is optimised by the user externally to the model, reliant on expert agronomist advice for the study region. The optimisation accounted for the clustering of the weather-years.
    •	Fodder crops - In adverse weather-years where either livestock feed is limiting or crops are frosted or are not worth harvesting, saleable crops can be turned into standing fodder. That is, instead of harvesting a crop it is grazed by livestock as summer feed.
    •	Bale crops - Crops planted with the expectation of being harvested for grain can be baled as hay. This may occur in adverse weather-years where either livestock feed is limiting or crops are frosted or are not worth harvesting.
    •	Labour supply - Permanent and manager labour is fixed (I.e. must be the same for all weather-years). However, casual labour can be optimised for each weather-year as it unfolds.
    •	Machinery contracting - If the timeliness of an activity is an issue, contract services can be selected to improve the work rate. This could be valuable in a late break weather-year to ensure the crops get the maximum possible growing season. Note, the assumption that contracting services are available can be changed.
    •	Dry seeding - A useful tactic to improve timeliness of seeding is to sow into dry soil, before the opening rains, to ensure crops experience the maximum possible growing season. If dry seeding is selected it is implemented for all weather-years that have yet to have the season break.
    •	Confinement feeding - Confinement feeding can be a good tactic to allow pasture deferment at the beginning of a growing season or to keep ground cover on paddocks in the late summer and autumn.
    •	Supplement feeding – In-paddock supplement feeding can be used as a tactic to help finish lambs for sale, ensure ewes reach target conditions for reproduction or to help meet energy requirements during weather-years with poor pasture growth.
    •	Changing liveweight - Altering livestock liveweight targets can be used as a tactic to handle varying feed availability due to seasonal variation e.g. animals can lose weight in poor feed years but this is associated with lower production per head.
    •	Not mating ewes - If the feed supply is sufficiently poor prior to joining then there is the option of not mating ewes. This might be most relevant if mating ewe lambs.
    •	Selling scanned dry ewes or other ewes at scanning – Sale of dry sheep can be a useful tactic if the year is unfolding unfavourably.
    •	Retain dry ewes - If the strategy is to sell dry ewes, and the weather-year is favourable, a tactical adjustment can be to retain the dry ewes until shearing, thereby generating wool income and then a further decision is to retain them for mating the following year.
    •	Selling at other times – The ewes and lambs’ sale time can be adjusted with the value received depending on the liveweight and condition of the animals at sale. In AFO there are ten selling opportunities throughout the year for ewes and eight sale opportunities for lambs and wethers.
