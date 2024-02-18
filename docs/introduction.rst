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

The intricacies of the farming system, combined with the farmer's desire to meet their objective,
causes whole farm modelling to be a helpful tool to aid decision making
:cite:p:`RN76`. Whole farm modelling involves a detailed representation of the whole
farming system, capturing the biological and economic interactions within a farming system.
This is an important precursor for assessing farm strategy because many aspects of a farming
system are interrelated, such that changing one factor may affect another :cite:p:`RN76`.

Agricultural systems are most frequently modelled by dynamic simulation, for example APSIM :cite:p:`RN8` or
by mathematical programming, for example MIDAS :cite:p:`RN2` techniques. Dynamic simulation (DS) is
frequently applied to represent biological systems encompassing the whole farm :cite:p:`RN105`
or a subsection of the farm :cite:p:`keating2003, RN23`. Mathematical programming (MP)
encompasses a group of optimisation techniques and is commonly used for whole farm modelling
:cite:p:`RN9, RN2, RN76`. Both DS and MP often achieve
more than their simple categorisation implies, as it is feasible to specify an objective in a
simulation model and search for an optimal solution whilst MP techniques can represent simulated biological
details :cite:p:`RN2`.

Heuristic techniques are a branch of commonly used optimisation procedures, including genetic
algorithms :cite:p:`RN109` and simulated annealing :cite:p:`RN113`. These methods use various
computational algorithms, often inspired by physical processes, to identify solutions in complex search
spaces :cite:p:`RN111`. Such procedures are valuable for the optimisation of simulation
models in which analytical gradients cannot be efficiently computed. However, these techniques are
not mathematically guaranteed to find the optima, and can be limited in their capacity to consistently
incorporate resource constraints :cite:p:`RN112`. For example, :cite:t:`RN112`
combined an agricultural weed simulation called Ryegrass and Integrated Management (RIM) with the
heuristic technique of compressed-annealing and determined that compressed-annealing was a suitable
algorithm to identify near-optimal configurations in constrained simulation models of weed populations.
RIM is a simulation model encompassing around 500 parameters. However, :cite:t:`RN112` noted that including
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
as efficient at representing biological and dynamic features, this limitation should not be overstated.
Firstly, at the whole farm level, representing precise biological and dynamic relationships is often
not of high importance and the overall relationships can be represented at a higher level still
capturing the necessary detail. Secondly, in the hands of skilled practitioners, it is possible to
represent or closely approximate the complex nonlinear biological and dynamic features using MP
techniques :cite:p:`RN134`. Thirdly, DS and MP are somewhat complementary because they are suited
to different tasks. For example, simulation models developed to imitate the biological features of
a farm sub system may generate data for use in whole farm MP models (e.g. :cite:p:`young2010, young2014`).

Due to lack of available computing power, software, time and knowledge, previous MP models that represented
farming systems were developed with a fixed, inflexible modelling framework and were simplified depictions
of reality. For example, MIDAS (Model of an Integrated Dryland Agricultural system), a prevalent whole farm
MP model used to examine broadacre farming systems principally in Australia :cite:p:`RN42, RN41, RN11, young2011, RN33, RN76`
, excludes price and weather uncertainty. Yet the farming system of the Western Australian region is a
dryland farming system in which variation in rainfall between weather-years can cause dramatic changes
in crop and pasture yields (Feng et al., 2022). Failure to represent this variation in MIDAS weakens the
credibility of some of its results. Furthermore, these previously developed models were built in Microsoft Excel,
which although easy to learn, has large computational overheads, size restrictions and its tabular structure
makes scalability challenging. Although it is unlikely that a computer program will ever fully reflect
reality, frequent improvements in computing power and a greater ease of coding enable increasingly
sophisticated models to now be constructed. Moreover, generating and capturing farm-level data is
increasingly feasible, and cost-effective, which allows more detailed farm models to be constructed.

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

The **A**\ustralian **F**\arm **O**\ptimisation Model (AFO) is described in detail below. In summary,
AFO is a whole farm LP model. AFO leverages a powerful algebraic modelling add-on package called Pyomo (Hart et al., 2011)
and IBMs CPLEX solver to efficiently build and solve the model. The model represents the economic and biological
details of a farming system including components of rotations, crops, pastures, livestock, stubble, supplementary
feeding, machinery, labour and finance. Furthermore, it includes land heterogeneity by considering enterprise
rotations on any number of soil classes. The detail included in the modules facilitates evaluation of a large
array of management strategies and tactics.

AFO has been built with the aim of maximising flexibility. Accordingly, depending on the problem being examined,
the user has the capacity to:

•	Change the region or property.
•	Select the level of dynamic representation. For example, the user controls the number of discrete options for seasonal variation and price variation.
•	Add or remove model components such as the number of land management units, land uses, novel feed sources such as salt land pasture, times of lambing for the flock and flock types (pure bred, 1st cross or 2nd cross).
•	Adjust the detail in linearising the production functions (e.g. the number of livestock nutrition profiles).
•	Make temporary changes to production parameters and relationships. For example, altering the impact of livestock condition at joining on reproductive rate.
•	Constrain management. For example, fix the stocking rate or crop area.
•	Include or exclude farmer risk aversion.

To facilitate user flexibility and support future development, AFO is built in Python, a popular open source
programming language. Python was chosen over a more typical algebraic modelling language (AML) such as
GAMS or Matlab for several reasons. Firstly, Python is open source and widely
documented making it easier to access and learn. Secondly, Python
is a general-purpose programming language with over 200 000 available packages with a wide range of
functionality :cite:p:`RN137`. Packages such as NumPy and Pandas :cite:p:`RN138` provide powerful
methods for data manipulation and analysis, highly useful in constructing AFO which contains large
multi-dimensional arrays (see sheep section). Packages such as Multiprocessing :cite:p:`RN139`
provide the ability to run the model over multiple processors taking advantage of the full computational
power of computers to significantly reduce the execution time of the model. Thirdly, Python supports
a package called Pyomo which provides a platform for specifying optimization models that embody the
central ideas found in modern AMLs :cite:p:`RN106`. Python's clean syntax enables Pyomo to express
mathematical concepts in an intuitive and concise manner. Furthermore, Python's expressive programming
environment can be used to formulate complex models and to define high-level solvers that customize
the execution of high-performance optimization libraries. Python provides extensive scripting
capabilities, allowing users to analyse Pyomo models and solutions, leveraging Python's rich set of
third-party libraries designed with an emphasis on usability and readability :cite:p:`RN140`.

The core units of AFO are:

    #. Inputs: The model inputs are stored in three Excel spreadsheets. The first contains inputs
       likely to change for different regions or properties such as farm area. The second file contains
       inputs that are "universal" and likely to be consistent for different regions or properties
       such as global prices of exported farm products. The third file contains structural inputs
       that control the core structure of the model.

    #. Precalcs: The precalcs are the calculations applied to the input data to generate the data for
       the Pyomo parameters (in other terms, the conversion of the inputs to the parameters for the LP matrix).
       The precalcs for each individual trial (trial is the name for a single model solution) can be controlled
       by the user with the 'experiment' spreadsheet which allows inputs from the three input spreadsheets to
       be temporarily adjusted, or the intermediate calculations in the precalcs to be temporarily adjusted.

    #. Pyomo and solver: This is the LP component of the model. It defines all the decision variables, the
       objective function, the constraints and parameters then utilises them to construct the model's equations
       (i.e. constraints). Components of the LP model can also be temporarily adjusted by the user via the 'experiment'
       spreadsheet. Pyomo formulates all the equations into a linear program format and passes the file to a solver.
       AFO has multiple compatible solver options. Most frequently used are CPLEX (Cplex, 2009) and GLPK :cite:p:`RN107`.
       When tested both solvers resulted in the same answer. CPLEX has some advanced features unavailable in GLPK.
       However, GLPK is open source whereas CPLEX is costly proprietary software (Cplex, 2009).

The procedure for building and solving AFO is that firstly, the inputs are read in from the Excel files.
The experiment spreadsheet is read that includes the temporary adjustments for the model parameters.
Furthermore, the spreadsheet allows the user to group trials into an experiment to be run as a batch.
For example, the user may be interested in the impact of increasing prices, hence an experiment examines
several price levels. Secondly, each module containing precalcs is executed. The parameters produced are
stored in a python data structure called a dictionary. Then the Pyomo section of the model creates the
decision variables, formulates the model constraints, populates the parameters with the coefficients
from the precalcs and passes the information to a linear solver. The results from the solver reveal
the maximum farm profit and the optimal levels of each decision variable that maximises the farm
profit (or some other objective function). From here the user can create a range of reports.

Key improvements
^^^^^^^^^^^^^^^^
Some of the key improvements of AFO over previous optimisation models include;

i.	Inclusion of price and weather uncertainty, the associated short-term management tactics and farmer risk attitude.
ii.	Increased rotation options.
iii. Extra detail on the biology of livestock production that allows:

    a. Inclusion of optimisation of the nutrition profile of livestock during the year.
    b. A larger array of livestock management options such as time of lambing and time of sale.

iv. Improved pasture representation that includes production effects of varying grazing intensity.
v. More detailed representation of crop residue that includes multiple feed pools based on quality and quantity.

Additionally, developing AFO in Python has resulted in a flexible framework that overcomes many previous
structural challenges such as scalability. The structure allows the user to alter the biological detail to
balance computer resource requirement against model realism in different aspects of the farm system.
For example, the user of AFO can easily alter the number of discrete options represented in different
sections of the model so that detail can be added to aspects that are important for a particular analysis
while simplifying the less important. Furthermore, AFOs usability and detailed representation of the farm
system means it can be applied to a plethora of current and future farming system opportunities and problems.

