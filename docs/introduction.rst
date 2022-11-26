Introduction
=====================

Background
----------
Farming systems often involve large areas with a range of soil types, several
crop, pasture and livestock enterprise choices, relatively inflexible
infrastructure, past decisions that influence current resource status and feasible
future actions, and price and climate variability :cite:p:`RN13`. The
combination of some or all these factors means farming systems can be complex to
analyse :cite:p:`RN34,RN38`. This complexity complicates
decision-making, making it difficult to determine the optimal farm management
strategy for a current production year or over several years. Even with access to
information about each aspect of the farming system, decision-making remains a
challenge due to the interactions between the various parts of the farming system
:cite:p:`RN1`.

The intricacies of the farming system, combined with the farmer’s desire to meet their objective,
causes whole farm modelling to be a helpful tool to aid decision making
:cite:p:`RN76`. Whole farm modelling involves a detailed representation of the whole
farming system, capturing the biological and economic interactions within a farming system.
This is an important precursor for assessing farm strategy because many aspects of a farming
system are interrelated, such that changing one factor may affect another :cite:p:`RN76`.

Agricultural systems are most frequently modelled by dynamic simulation, for example APSIM :cite:p:`RN8` or
mathematical programming, for example MIDAS :cite:p:`RN2` techniques. Dynamic simulation (DS) is
frequently applied to represent biological systems encompassing the whole farm :cite:p:`RN105`
or a subsection of the farm :cite:p:`keating2003, RN23`. Mathematical programming (MP)
encompasses a group of optimisation techniques and is commonly used for whole farm modelling
:cite:p:`RN9, RN2, RN76`. Both DS and MP often achieve
more than their simple categorisation implies, as it is feasible to specify an objective in a
simulation model and search for an optimal solution and MP techniques can represent simulated biological
details :cite:p:`RN2`. Heuristic techniques are a branch of commonly used optimisation procedures, including genetic
algorithms :cite:p:`RN109` and simulated annealing :cite:p:`RN113`. These methods use various
computational algorithms, often inspired by physical processes, to identify solutions in complex search
spaces :cite:p:`RN111`. Such procedures are valuable for the optimisation of simulation
models in which analytical gradients cannot be efficiently computed. However, these techniques are
not mathematically guaranteed to find the optima, and can be limited in their capacity to consistently
incorporate resource constraints :cite:p:`RN112`. For example, :cite:t:`RN112`
combined an agricultural weed simulation called Ryegrass and Integrated Management (RIM) with the
heuristic technique of compressed-annealing and determined that compressed-annealing was a suitable
algorithm to identify near-optimal configurations in constrained simulation models of weed populations.
RIM is a simulation model encompassing around 500 parameters. However, :cite:t:`RN112` note that including
additional detail would result in a much larger solution time. Therefore, although heuristic techniques
such as those used by :cite:t:`RN112` are conceptually interesting, it is likely that they may be
computationally challenging if applied to the representation of detailed whole farming systems.
The lack of optimisation capability can be a significant limitation of simulation modelling for
evaluating the economics of on-farm decision making. For example, the profitability of
mating ewe lambs is dependent on many factors such as ewe live weight before, during and
after mating, pasture supply, time of lambing and relative prices. Thus, even for a skilled
simulation user it would be easy to land at a local optimum resulting in inaccurate economic
advice regarding optimal management of ewe lambs.

While MP is not as flexible as DS in representing biological and
dynamic features, it does provide a more powerful and efficient optimisation method. Although MP is not
as sound at representing biological and dynamic features, this limitation should not be overstated.
Firstly, at the whole farm level, representing precise biological and dynamic relationships is often
not of high importance and the overall relationships can be represented at a higher level still
capturing the necessary detail. Secondly, in the hands of skilled practitioners, it is possible to
represent or closely approximate the complex nonlinear biological and dynamic features using MP
techniques :cite:p:`RN134`. Thirdly, DS and MP are somewhat complementary because they are suited
to different tasks. For example, simulation models developed to imitate the biological features of
a farm sub system may generate data for use in whole farm MP models (e.g. :cite:p:`young2010, young2014`).

Evaluating farming systems is a dynamic problem where current actions impact future productivity
(e.g. crops in a rotation, livestock breeding or equipment purchases); price and weather states
change over time; there is future uncertainty and the resource base is exhaustible :cite:p:`RN115`.
Fundamentally, one must determine the extent to which dynamic situations are represented. A prevalent
whole farm mathematical program used to examine broadacre farming systems principally in Western Australia
is MIDAS (Model of an Integrated Dryland Agricultural System)
:cite:p:`RN42, RN41, RN11, young2011, RN33, RN76`.
MIDAS is a static equilibrium model encompassing the key assumption that the same
management decisions are made repeatedly each year, with that year being an average or modal climatic
year. However, as discussed by :cite:t:`young2022` MIDAS fails to represent inter-year variation, which has
been noted as a significant limitation by many of the MIDAS model developers and users
:cite:p:`Kingwelletal1992, RN76`. A model called MUDAS (Model of an Uncertain Dryland
Agricultural System) was developed to analyse the impact of risk associated with weather-year
variation and price variation :cite:p:`RN83`. MUDAS is a discrete stochastic programming (DSP)
model that considers nine season types and five price states. The model represents the farmer’s risk
aversion and possible tactics to implement as the various seasons unfold :cite:p:`Kingwelletal1992`.
Representing risk and tactics associated with farm management enabled MUDAS to overcome the key
steady-state limitation of MIDAS and provide more accurate findings concerning farm management.
However, the added detail increased the size and complexity of the model. Such a large model resulted
in long solution times and an arduous error-checking and calibration processes. Due to this, MUDAS has
not been maintained or updated since the 1990’s. Moreover, since the development of MUDAS technological
innovations and practice changes have occurred in Australian
broadacre farming (e.g. :cite:p:`RN146`), adding to the complexity of farm management, but simultaneously
computing power and more versatile computing languages have become available. A possible limitation of
the MUDAS framework is its exclusion of weather-year sequences i.e. the assumption that the probability
of incurring a given weather-year is independent of the previous weather-year, and the assumption that
the optimal farm strategy and the tactics employed in a given weather-year is only altered by the starting
position based on the weighted average of all weather-years. Adding the ability of a model to represent
weather-year sequence exponentially increases the size of the model, commonly known as the curse of
dimensionality, which adds to the computational power required to build and solve the model, enlarges
the task of error-trapping and potentially complicates the interpretation of modelling results.

The whole farm model described in this documentation uses MP, more specifically linear programming (LP). LP was
chosen because it is well established, reliable and efficient for optimising large problems with thousands
of activities and constraints. Furthermore, LP has been used successfully to model farming systems in
Australia :cite:p:`RN2, RN83` and overseas :cite:p:`Annetts2002, Schäfer2017`. To maximise
the accuracy of representing biological relationships in LP, non-linear relationships such as pasture
growth are represented by piece-wise linearization :cite:p:`RN134`.

The following documentation assumes a basic level of LP
understanding such as outlined by :cite:t:`RN134` in *Introduction to practical linear programming*.

Model summary
-------------

A model called **A**\ustralian **F**\arm **O**\ptimisation Model (AFO) is described in detail below. In summary,
AFO is a whole farm linear programming model. The model represents the economic and
biological details of a farming system including components of rotations, crops, pastures, sheep, stubble,
supplementary feeding, machinery, labour and finance. Furthermore, it includes land heterogeneity by considering
enterprise rotations on any number of soil classes. The detail captured in the modules allows various
management tactics to be represented and, if the user opts, seasonal uncertainty can also be represented.

The model is built and calibrated such that
it represents current farm management technology insofar as the types of machinery complements,
herbicides used and rates applied. Tasks contracted and crop and livestock options considered are all
consistent with those used by farmers in the modelled region.
AFO has been built with flexibility in mind and allows the user to select the level of dynamic representation.
Depending on the problem being examined, the user can select to include or not include seasonal
variation. Excluding seasonal variation results in a steady-state equilibrium model which represents an average year.
Including seasonal variation results in a larger discrete stochastic model.

AFO has been developed with the aim of maximum flexibility to support future model development. The
model is built in Python, a popular open source programming language. There are several reasons why
Python was chosen over a more typical algebraic modelling language (AML) such as GAMS or Matlab. Firstly,
Python is open source and heavily documented making it easier to access and learn. Secondly, Python
is a general-purpose programming language with over 200 000 available packages with a wide range of
functionality :cite:p:`RN137`. Packages such as NumPy and Pandas :cite:p:`RN138` provide powerful
methods for data manipulation and analysis, highly useful in constructing AFO which contains large
multi-dimensional arrays (see sheep section). Packages such as Multiprocessing :cite:p:`RN139`
provide the ability to run the model over multiple processors taking advantage of the full computational
power of computers and significantly reducing the execution time of the model. Thirdly, Python supports
a package called Pyomo which provides a platform for specifying optimization models that embodies the
central ideas found in modern AMLs :cite:p:`RN106`. Python’s clean syntax enables Pyomo to express
mathematical concepts in an intuitive and concise manner. Furthermore, Python’s expressive programming
environment can be used to formulate complex models and to define high-level solvers that customize
the execution of high-performance optimization libraries. Python provides extensive scripting
capabilities, allowing users to analyse Pyomo models and solutions, leveraging Python’s rich set of
third-party libraries designed with an emphasis on usability and readability :cite:p:`RN140`.

The core units of AFO are:

    #. Inputs: The model inputs are stored in three Excel spread sheets. The first contains inputs
       likely to change for different regions or properties such as farm area. The second file contains
       inputs that are “universal” and likely to be consistent for different regions or properties
       such as global prices of exported farm products. The third file contains structural inputs
       that control the core structure of the model.

    #. Precalcs: The precalcs are the calculations applied to the input data to generate the data for
       the Pyomo parameters (in other terms, the coefficients for the LP matrix).

    #. Pyomo and solver: This is the ‘guts’ of the model. It defines all the variables and parameters
       then utilises them to construct the model equations (e.g. resource constraints). Pyomo formulates
       all the equations into a linear program format and passes the file to a solver. Many solvers
       are compatible with the Pyomo. Currently GLPK (GNU Linear Programming Kit) :cite:p:`RN107`
       is used.

The procedure for building and solving AFO is that firstly, the inputs are read in from the Excel files.
They are then adjusted according to the user’s application of AFO. For example, the user may be
interested in the impact of increasing prices, hence the price inputs are increased. Secondly, each
module containing precalcs is executed. The produced parameters are stored in a python data structure
called a dictionary. The Pyomo section of the model then creates the variables and parameters, populates
the parameters with the coefficients from the precalcs, constructs the model constraints and passes
the information to a linear solver. The results from the solver reveal the maximum farm profit and
the optimal levels of each variable that maximises the farm profit (or some other objective function).
From here the user can create any number of different reports. A loop is used to allow the user to
adjust the inputs, solve the model and report the outcomes multiple times in a single execution of
the model.


Summary of Model Capacity
-------------------------

.. note:: The model is not limited to the listed capabilities on this page.

Stock
^^^^^
In the full model the livestock management decisions that are optimised can include:

#. The proportion of the ewe flock mated to different sire genotypes (pure bred, maternal type or 
   terminal)
#. The inclusion of a cross bred dam component mated to a terminal sire (the dam cross is between
   the purebred and the maternal genotype)
#. The reproductive life of dams in the flock (based on whole flock feed requirements, reproduction 
   variation by age, the value of CFA dams at different ages, the selection pressure that can be applied on replacement ewes).
#. Whether to mate ewe lambs and the optimal proportion to mate.
#. Number of animals carried based on whole of flock, whole year feed requirements and whole
   farm feed supply (the feed budget is divided into 10 periods with 4 feed quality pools in each period)
#. Sale age of each animal group, up to 3 options each year (or each shearing cycle if more 
   frequent than annually).
#. Purchase options for each animal group to include a trading operation for dry animals or a longer 
   term buy in and sell operation for ewes.
#. Diet selection for the animals based on the feedbase represented in the model (likely to 
   include green or dry pasture, crop residues and supplementary grain feeding).
#. Confinement feeding of adults and feedlotting or confinement feeding of young animals (which 
   reduces the animals energy requirements due to reduced distance walked).
#. Nutrition profile of the animals during the year, based on feed variation periods which relate 
   to reproduction and the time of the seasonal break for dams and seasonal break and 
   turn-off time for the dry animals.
#. Differential feeding of dams based on litter size, lactation number and foetal age,
   provided the dams are scanned or assessed for ‘gave birth and lost’
#. Optimal replacement policy including:

    #. The change in reproduction and production over the animals lifetime,
    #. The potential to increase per head production through culling and a response in the
       current generation.

#. Sale value of animals of different ages and different weights
#. Optimal weaning age for each dam age group

Furthermore, constraints can be applied to the model to limit:

#. Level of enteric greenhouse gas emissions and emissions of nitrous oxide from faeces and urine.
#. Bare ground during the summer/autumn period
#. Maximum levels of animal mortality or maximum level of liveweight loss during the feed limiting period of the year
#. Animals to graze at their voluntary feed intake level (i.e. that intake reflects the FOO & DMD offered to the animals i.e. feed is not rationed through active management of the stock, although pasture management can be altered to vary the FOO & quality profile)

If desired the model can also represent and compare (but not optimise in a single model solution)

#. The length of the joining period (measured in number of cycles mated). Includes the tradeoff between the number of
   ewes conceiving and the distribution of size and energy requirements of the later born progeny
#. Altering the age that the young ewes are mated, for example a 7 month mating versus a 8.5 month mating for ewe lambs.
#. Accelerated lambing, where ewes are mated every 8 months and therefore have 3 lambs in 2 years.
#. Variation in timing of lamb, hogget and adult shearing
#. More frequent shearing. Shearing interval of 6, 8 or 12 months can be assessed.

Feed supply
^^^^^^^^^^^
The following feed management decisions that are optimised can include:

#. Value of confinement feeding
#. Pasture variety 
#. Pasture deferment
#. Level of supplementary feeding
#. Timing of stubble consumption
#. Crop grazing
#. Fodder crops
#. Pasture area and rotation selection

Additional aspects that can be tested but not optimised include:

#. Fertiliser application
#. Impact of varying pasture conservation limits

Cropping
^^^^^^^^
The following crop management decisions that are optimised can include:

#. Rotation/crop selection
#. Contractual services
#. Labour allocation
#. Time of sowing

Additional aspects that can be tested but not optimised include:

#. Fertiliser application
#. Chemical application
#. Seeding rate
