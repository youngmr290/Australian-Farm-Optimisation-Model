'''
Stock feedsupply construction

author: young

Feedsupply is represented as a nutritive value (NV). There is also an input which controls if the animal
is in confinement or not.

Feedsupply can be generated in two ways

    #. from inputs in Property.xl and Structural.xl
    #. from pkl file that is generated from running the model with multiple w options.

Feedsupply inputs contain a base feedsupply (i_feedoptions_r1pj0) for different animals. The base feed supply
used for each animal in the generator is selected using ia_r1_zig?.

For dams (has not been hooked up for offs) there is possible further adjustment (i_feedsupply_adj_options_r2p)
for LSLN (litter size lactation number - b1 axis) and weaning
age (a1 axis). Typically this is not used. Instead the user runs AFO with multiple w and allows the model to
generate the optimal feedsupply which is used for subsequent runs.

Generating the feedsupply from the optimal nutrition pattern selected by the model means that the feedsupply
can easily be optimised for different axis that are not included in the inputs (eg t axis and z axis).
This is carried out in the feedsupply creation experiment (in exp.xl). Depending on the number of w that
can be included in the model it may take several trials to converge on the optimal feedsupply.
Optimising the duration of confinement feeding is imprecise because the optimisation occurs at the DVP level
and this timestep is likely too long. Therefore, the user needs to test including confinement in a range of feed
periods for different stock classes. This is controlled using the input i_confinement_r1p6z.

When using the pkl file as the source of the feedsupply inputs it is possible to control whether the inclusion
of confinement is controlled by the previous optimum solution (from the pkl file) or from the
input i_confinement_r1p6z. This allows the inclusion of confinement feeding in subsequent trials even if it wasn't
selected in the previous optimum solution.
This also allows confinement to be included in the N1 model without forcing it to occur all year.

Confinement feeding is good to represent separately for a few reason:

    #. Feeding a given quantity of energy per day in confinement results in better production because less energy
       is expended grazing.
    #. Confinement feeding is represented in a separate feed pool because a restricted intake of supplement that meets
       the energy requirements of an animal results in slack volume. If in the same feed pool this slack volume could
       be used to consume greater quantities of poor quality feed such as dry pasture or stubble.
    #. The confinement pool is not adjusted for effective MEI, which is to allow for the lower efficiency of feeding
       if feed quality is greater than the quality required to meet the target for the given pool.
    #. Confinement feeding should incur a capital cost of the feed lot and the labour requirement should reflect
       feedlot conditions rather than paddock feeding conditions.

Likely process to calibrate feedsupply: feedsupply can be calibrated prior to an analysis using multiple n
slices and once the optimum is identified the model can be set back to N1. This might take multiple iterations
using the big model. The feedsupply will need to be generated for each group of trials that have different
active sheep axis (ie if the analysis is comparing scanning options then there would need to be a feedsupply
for scan=0 and scan=1 because the clustering is different). A further challenge is to optimise when to confinement
feed. Optimising the NV in confinement is similar to optimising in the paddock and requires multiple confinement n.





'''

#todo supp feeding in confinement incurs the same costs as paddock feeding. this should be changed. it should also incur some capital cost.
