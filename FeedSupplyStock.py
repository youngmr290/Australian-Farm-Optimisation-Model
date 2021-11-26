'''
Stock feedsupply construction

author: young

Feedsupply is represented as a nutritive value (NV). There is also an input which controls if the animal
is in confinement or not.

Feedsupply can be generated in two ways

    #. from inputs
    #. from pkl file that is generated from running the model with multiple w options.

Feedsupply inputs contain a base feedsupply (i_feedoptions_r1pj0) for different animals. The base feed supply
selected used for each animal in the generator is selected using ia_r1_zig?.

For dams (has not been hooked up for offs) there is possible further adjustment (i_feedsupply_adj_options_r2p)
for lsln (b1 axis) and weaning
age (a1 axis). Typically this is not used. Instead the user runs AFO with multiple w and allows the model to
generate the optimal feedsupply which is used for subsequent runs.

Generating the feedsupply by selecting the optimal nutrition pattern selected by the model means that the feedsupply
can easily be optimised for different axis that are not included in the inputs (eg t axis and z axis). However,
depending on the number of w it may take a few creation runs to converge on the optimal feedsupply.
Optimising if the stock are in confinement or not is more complicated because the level of optimisation
can only be made to the dvp level. The user can increase the precision by specifying the generator periods
when confinement feeding can occur (i_confinement_r1p).

Confinement feeding is good to represent separately for a few reason:

    #. Feeding a given quantity of energy per day in confinement results in better production because less energy
       is expended grazing.
    #. Confinement feeding is represented in a separate feed pool because a restricted intake of supplement that meets
       the energy requirements of an animal results in slack volume. If in the same feed pool this slack volume could
       be used to consume other poor quality feed such as dry pasture or stubble.
    #. The confinement pool is not adjusted for effective MEI, which is to allow for the lower efficiency of feeding
       if feed quality is greater than the quality required to meet the target for the given pool.
    #. Confinement feeding should incur a capital cost of the feed lot and the labour requirement should reflect
       feedlot conditions rather than paddock feeding conditions.

Likely process to calibrate feedsupply: feedsupply can be calibrated prior to an analysis using multiple n
slices and once the optimum is identified the model can be set back to 1n. This might take multiple itterations
using the big model. The a feedsupply will need to be generated for each group of trials that have different
active sheep axis (ie if the analysis is comparing scanning options then there would need to be a feedsupply
for scan=0 and scan=1 because the clustering is different). A further challenge is to optimise when to confinement
feed. Optimising the NV in confinement is similar to optimising in the paddock and requires multiple confinement n.





'''

#todo supp feeding in confinement incurs the same costs as paddock feeding. this should be changed. it should also incur some capital cost.
