Mathematical summary
=====================

A simplified mathematical representation of the model is below.

where:

* :math:`n` is node n∈N[0,1], 1 decision point during the year at the break of the season (just using one node for the
  example, in the model there are more nodes).
* :math:`s` is scenario s∈S, 3 season types (Good, average and poor)
* :math:`x_ins` are the decision variables in the DSP, xin in the deterministic model (with only 1 scenario).
* :math:`c_ins` are the objective function weights for the decision variables
* :math:`J` is the set of constraints and j are the individual constraints j∈J
* :math:`a_ijn` are the coefficients/parameters in the matrix
* :math:`b_j` are the RHS coefficients
* :math:`p_s` is the probability of scenario s    (:math:`∑_s p_s == 1`)

Deterministic model:
----------------------
The nodes exist in the deterministic model representing the temporal nature of some decision variables and the
transfers during the year, but there are no scenarios.

:math:`max∑_i ∑_n c_in x_in`

st.

Production constraints

:math:`∑_i a_ijn x_in ≤ b_j, ∀ j,∀ n`

transfer constraints (within the year). E.g. the number of sheep in the second stage is a function of the
number in the first stage. Where :math:`w_in` is just a special case of :math:`a_in` (separated out to make it clear).

:math:`x_in ≤ ∑_i w_{in-1} x_{in-1}, ∀ i,∀ n`

equilibrium constraints (between years). E.g. the number of sheep at the start of next year (which is also
start of this year) is the number at the end this year shifted one age group.
:math:`x_{i0} ≤ ∑_i w_iN x_iN`

DSP model
---------

:math:`max∑_i ∑_n ∑_s c_ins x_ins`

st.

non-anticipativity

:math:`x_{i0s} == root_is, ∀ i, ∀ s`

production constraints

:math:`∑_i a_ijns x_ins ≤ b_js, ∀ jns`

transfer constraints (within the year). E.g. the number of sheep in the second stage is a function of the number
in the first stage.

:math:`x_ins ≤ ∑_i w_{in-1s} x_{in-1s}, ∀ ins`

equilibrium constraints (between years). E.g. the number at the start of next year (which is also start of this
year) is the number at the end this year shifted one age group and weighted by the scenario probability.

:math:`x_{i0s} ≤ ∑_i ∑_s w_iNs x_iNs p_s,  ∀ i`, N is the last node
