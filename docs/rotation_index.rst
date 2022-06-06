Rotation
========

Rotation background
-------------------

Modelling of cropping or crop-pasture rotations to date has primarily been based on a predetermined,
restricted set of rotations as “activities” of a LP matrix :cite:p:`RN78`. For example,
MIDAS applies this same framework. This approach, however, limits the potential rotations that can be
selected by the model and does not support the flexible nature of real-life rotation selection. For
example, if the start to a growing season is late, farmers may opt to reduce the area of crop in
their rotations. It also results in the necessity to build entirely new modules for each agro-climatic
region due to differences in crop and rotation choices that are available and applicable to each region.

In AFO we adopt an alternative method proposed by Wimalasuriya and Eigenraam :cite:p:`RN78`, where
the model solves for the optimal rotation from all possibilities. Each land use [#lmu]_ in the optimal
solution is determined based on the paddock history and the productivity of the land use that
follows the given history. This is an unrestricted approach that supports a large range of possible
rotations and allows greater flexibility for adding new land uses. Additionally, the approach aligns
more closely with reality, facilitating a more detailed and accurate representation of effects of
weather-year type on rotation choice.

We define a rotation phase as a land use (‘current land use’) with a specific sequence of prior
land uses (‘history required’). Each rotation phase has a level of production (grain and stubble
production from crops and a pattern and magnitude of pasture production), a level of costs and
provides a history (‘history provided’). The history provided is the sequence of previous land uses.
As a simple example, consider the rotation phase Canola: Barley: Wheat. Canola, the current land use.
Barley – Wheat is the history required and Canola – Wheat is the history provided. To utilise this
rotation structure in LP requires introducing a constraint to ensure that for the model to select
a given rotation phase the history required must match a history provide from another rotation.

The terminology used in AFO to distinguish each land use in a rotation is that the current land
use is termed year 0. The prior year land use is year 1 and so on working backwards through the
rotation phase. For example, in the rotation Canola: Barley: Wheat; Canola is year 0, Barley is
year 1 and Wheat is year 2.

The rotation phases are designed to be as simple or general as possible, covering all potential
performance and management variants. A myriad of rotation phases are considered involving land
uses over a set number of years, and then the infeasible options are removed.

The length of the rotation phases and the level of generalisation possible is determined such
that the impacts of the history required on the current land use production and costs are captured.
These can be summarised by:

#. The need to track the number of crop phases to determine if an annual pasture needs reseeding.
#. The need to track the effect of a land use on the productivity or costs of subsequent land uses.
   This can be either:

    a. Fixing of soil nitrogen and the effect on following crops. This requires tracking:

        - The number of years of the legume as it affects the quantity of organic nitrogen.

        - The number of years since the legume to determine the remaining nitrogen.

    b. Impacts on disease levels

    c. Impact on weed seed levels

#. The impact of cropping on subsequent annual pasture germination.

The impacts and assumptions of land use history on production and costs that are being captured
in the rotation phases developed are:

#. Annual pasture will be resown if the four most recent land uses in the history are crops.
   Resowing impacts the current year and the succeeding year.

#. Lucerne (or Tedera) will be resown if the immediately preceding land use is not Lucerne (or Tedera)

#. The impacts of spray-topping and manipulating pastures lasts for two years.

#. Germination of annual pasture is affected by

    a. The two most recent land uses in the history.

    b. The crop type immediately prior to the resown annual pasture. Specifically:

        - Oat fodder crop increases pasture germination

        - a pulse crop increases growth of annual pastures (which is represented by an increase in
          germination)

#. A history of legume pasture (annual, Lucerne and Tedera) provides organic nitrogen for subsequent
   non-legume crops (cereal or Canola).

    a. The amount of organic nitrogen increases up to 5 years of consecutive legume pasture.

    b. The utilisation of the organic nitrogen lasts for 2 years for a non-legume crop if 1 or 2 years
       of preceding legume pastures occur. It lasts for 3 years if there is 3+ years of legume pasture.

#. Pulse crops provide organic nitrogen for subsequent non-legume crops.

    a. The impact of the organic nitrogen lasts for a maximum of 3 years.

#. Leaf disease and root disease builds up for each land use and reduces productivity of subsequent
   repeats of the land use. There is variation in the length of the break required and the duration
   of the benefit.

    a. It is assumed that the maximum level of disease is reached after 4 consecutive years of a land use.

To capture all the factors listed above the length of the rotation phases represented in AFO is
defined to a maximum of 6 years, allowing a history of 5 pastures to be tracked. To reduce the number
of rotation phases, land uses in the history that are assumed to have the same impact on the production
and cost of the current land use are grouped into ‘land use sets’ (see `Table 2`_). Whilst still capturing
the factors listed above, the following generalisations apply to history in rotations:

    #. All cereal crops can be treated the same, except for fodder which must be tracked in year 1.
    #. All pulses can be treated the same.
    #. All crops can be treated the same after year 3.
    #. Resown annuals can be treated as any annual pasture after year 1.
    #. Manipulated and spray-topped pastures can be treated as any pasture after year 2.

After accounting for these generalisations, the following land use options as listed in `Table 1`_ are
represented in each year of a rotation phase:

* Year 0: all the land use options need to be included

* Year 1: E1, N, OF, P, A, S, U, X, T, J, ar, sr, m

* Year 2: E, N, P, A, S, M, U, X, T, J

* Year 3: the sets E, N, P, A, U, T

* Year 4 → year X [#x]_ : the land use sets Y, A, U, T

Some of the rotation phases constructed will be illogical and must be removed. For example, annual
pasture is only resown after 4 years of continuous crop therefore, any rotation phase that are
generated with resown annual that do not have 4 years of crops preceding it must be removed. See RotGeneration_
for the full list of illogical rules.

To further reduce the possible number of rotation phases in the model, unprofitable and unused land
sequences are removed. See RotGeneration_ for the full list of rules.

.. _Table 1:

.. list-table:: Table 1: Land uses represented by the rotation phases.
   :header-rows: 1

   * - Key
     - Land use
   * - a
     - Annual (no chemical)
   * - ar
     - Annual resown
   * - b
     - Barley
   * - bd
     - Dry sown Barley
   * - f
     - Faba
   * - h
     - Hay
   * - i
     - Lentil
   * - j
     - Tedera (manipulated)
   * - jc
     - Continuous Tedera (manipulated)
   * - jr
     - Tedera resown (manipulated)
   * - k
     - Chickpea
   * - l
     - Lupins
   * - m
     - Annual (manipulated in winter)
   * - o
     - Oats
   * - od
     - Dry sown Oats
   * - of
     - Oats fodder crop
   * - r
     - Canola (RoundUp Ready)
   * - rd
     - Dry sown Canola (RoundUp Ready)
   * - s
     - Annual (spray-topped)
   * - sp
     - Salt land pasture (saltbush plus pasture understory)
   * - sr
     - Annual (spray-topped & resown)
   * - t
     - Tedera (mixed sward)
   * - tc
     - Continuous Tedera (resown every 10yrs) [#tc]_
   * - tr
     - Tedera resown (mixed sward)
   * - u
     - Lucerne (mixed sward)
   * - uc
     - Continuous Lucerne (mixed sward)
   * - ur
     - Lucerne resown (mixed sward)
   * - v
     - Vetch
   * - w
     - Wheat
   * - wd
     - Dry sown Wheat
   * - x
     - Lucerne (monoculture)
   * - xc
     - Continuous Lucerne (monoculture)
   * - xr
     - Lucerne resown (monoculture)
   * - z
     - Canola (Triazine Tolerant)
   * - zd
     - Dry sown Canola (Triazine Tolerant)


.. _Table 2:

.. list-table:: Table 2: Land use sets used in the rotation phase histories.
   :header-rows: 1

   * - Key
     - Land use set
   * - A
     - Annual (a, ar)
   * - E
     - Cereal (b, h, o, of, w)
   * - E1
     - Cereal without fodder (b, h, o, w)
   * - J
     - Manipulated Tedera (j, jr)
   * - N
     - Canola (r, z)
   * - P
     - Pulse (f, i, k, l, v)
   * - S
     - Spray-topped annual pasture (s, sr)
   * - T
     - Tedera (t, tr)
   * - U
     - Lucerne (u, ur)
   * - X
     - Manipulated Lucerne (x, xr)
   * - Y
     - Anything not annual (E, E1, P, N, J, T, U, X)


Land heterogeneity
^^^^^^^^^^^^^^^^^^^

Land quality can vary significantly across a farm, predominantly due to variations in soil type. This
can impact:

#. The efficiency of machinery use and its rate of wear.
#. The proportion of arable area.
#. The level of inputs applied.
#. The level of production.

AFO represents land variation by splitting the farm area into a specified number of land management
units. Each land management unit has its own parameters reflecting the factors as listed above.
Phases of rotations are possible on each land management unit. The model can solve how to best utilize
the area of each land management unit. Each LMU has a specified proportion of arable area. Non arable
area can not be cropped however it is accessible by livestock.

.. note:: If different crop management prior to a pasture phase (e.g. reduced expenditure on herbicide
    because weed seed control is not important) is to be represented then this will require extra landuse
    options for the previous crop and extra landuse options for the germinating pasture.

.. note:: The FOO level at the end of spring is not carried into the next year as a weed burden (e.g.
    the weed burden in the current year is the same independent of grazing/season last year). This could be
    represented by adding pasture phases that have spring FOO level as a part of the definition e.g. EEEEah
    which is annual pasture with high spring FOO following multiple cereals.


.. [#lmu] Use of paddock in a given year.
.. [#x] Year X is the final year of the rotation phase. This is set by the user.
.. [#tc] Continuous Tedera/Lucerne are separate land use so that resowing every 10 years can be included.


Rotation generation
-------------------

.. _RotGeneration:

.. toctree::
   :maxdepth: 1

   RotGeneration

Precalcs
---------

.. toctree::
   :maxdepth: 1

   RotationPhases

Pyomo
-----

.. toctree::
   :maxdepth: 1

   RotationPyomo

