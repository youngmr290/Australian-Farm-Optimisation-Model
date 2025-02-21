


def f1_tree_shape():
    area_trees = 10 #area of trees themselves
    number_of_belts = 5
    belt_spacing = 100
    tree_rows = 2
    row_width = 2
    propn_paddock_adj = 1
    lmu_area=100

    propn_lmu_trees #proportion of paddocks within LMU that have tree belts


    tree_width = (tree_rows + 1) * row_width #plus 1 to account for spacing on either edge of plantation.
    tree_length = area_trees * 10000 / tree_width / number_of_belts

    #can i make a simple plot of the expected tree layout based on this info???? GPT??

    #assuming that all belts within the LMU are same distance apart or they are far enough apart to realise all costs/benefits
    #assuming square LMU
    #if trees belts are closer together than the interaction distance this would mean final belt should have more impact - this will be ignored.
    import matplotlib.pyplot as plt

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 5))  # Further increased figure width

    # Define section boundaries
    sections = {
        "Open paddock": (0, 20),  # Left Open paddock
        "Competition zone": (20, 28),  # Left Competition zone
        "Belt": (28, 38),  # Tree belt
        "Competition zone (Right)": (38, 46),  # Right Competition zone
        "Open paddock (Right)": (46, 66)  # Right Open paddock
    }

    # Add section labels
    for section, (start, end) in sections.items():
        ax.text((start + end) / 2, 8, section.replace(" (Right)", ""), ha='center', fontsize=14, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Draw vertical dashed lines to separate sections
    for start, end in sections.values():
        ax.axvline(x=start, color='black', linestyle="dashed")

    # # Add "No crop zone" labels in both Competition Zones
    # for x_pos in [24, 42]:  # Left and right competition zones
    #     ax.annotate("No crop\nzone", xy=(x_pos, 7), xytext=(x_pos, 8.5),
    #                 ha='center', fontsize=12, arrowprops=dict(arrowstyle='->'))

    # Add distance labels
    # ax.text(10, 0.5, "Up to 20m", ha='center', fontsize=12)
    # ax.text(24, 0.5, "8 to 8m", ha='center', fontsize=12)
    # ax.text(33, 0.5, "5-10m", ha='center', fontsize=12)
    # ax.text(42, 0.5, "8 to 8m", ha='center', fontsize=12)
    # ax.text(56, 0.5, "Up to 20m", ha='center', fontsize=12)

    # Draw trees in the belt section (Two sets of two trees, 2m apart)
    tree_positions = [[32, 5], [34, 5], [32, 3], [34, 3]]
    for group in tree_positions:
        ax.scatter(group[0], group[1], s=600, color="green", label="Tree")

    # Add inter-row space labels
    ax.text(33, 6, "Inter-row space", ha='center', fontsize=12, fontweight='bold')
    ax.text(33, 5.5, "|---|", ha='center', fontsize=14, fontweight='bold')  # Indicator below the label

    # Add within-row space labels
    ax.text(33, 4, "Within-row space", ha='center', fontsize=12, fontweight='bold')
    ax.text(34, 3.5, "|---|", ha='center', fontsize=18, rotation=90)

    # Add buffer space labels
    ax.text(33, 2.5, "Side buffer", ha='center', fontsize=12, fontweight='bold')
    ax.text(36, 2.5, "|---|", ha='center', fontsize=18)

    # Add "No crop zone" labels in both Competition Zones
    # for x_pos in [24, 42]:  # Left and right competition zones
    ax.annotate("Wind direction", xy=(50, 9), xytext=(33, 9),
                ha='center', fontsize=12, arrowprops=dict(arrowstyle='->'))

    # # Add y-axis label
    # ax.text(-5, 5, "0 to 10m", ha='center', fontsize=12, rotation=90)

    # Adjust axes
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 66)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # Show figure
    plt.show()

    return tree_length, tree_width

def f_costs():
    '''
    The main production costs are:

        - Establishment and maintenance costs for the trees
        - Costs for harvest and on-farm transport
        - Road transport to the bioenergy plant
        - Costs for auditing and administration.

    Note, the opportunity cost of the land that is affected by the trees is considered in the other functions.
    '''

    #i think there will be certain if statemtns in here eg if the farmer is worried about carbon credits then there will be an auditing cost.

    #will need to add an input thta is expected date costs are incured (probs near the start of the gs).

    #labour will be hooked up correctly???? maybe only for the maint

    return

def f_crop_production_scalar():
    '''
    This function calculates an average yield scalar based on the costs (resources competition)
    and benefits (wind protection) of tree plantations.

    The logic used in the function is based on the assumption that the land adjacent to the
    trees is the same LMU as the trees and is wide enough to realise all the costs/benefits of the trees.

    '''
    area_trees = 10
    tree_rows = 2
    row_width = 2
    propn_paddock_adj = 1
    lmu_area=100


    tree_width = (tree_rows + 1) * row_width #plus 1 to account for spacing on either edge of plantation.
    tree_length = area_trees * 10000 / tree_width

    ##assumes the LMU is a rectangle where the length is equal to the length of the tree plantation.
    ## The shape of the LMU doesnt matter assuming that on farm the land adjacent to the LMU is wide enough
    ## to realise all the costs/benefits of the trees.
    lmu_width = lmu_area * 10000 / (tree_length * propn_paddock_adj)


    ##calc yield scalar
    #todo pass lmu_width into the yield by distance from trees function.

    #todo how to represent that belt spacing will impact production function because of water competition on both side of trees.
    #have a yield scalar for 0 - -x m from trees and one for 0 - x m
    #data needs to come from plantation with wide spread belts or a single belt so as not to double count



    #todo graph out these functions








def f_windspeed_adj():
    '''
    This function calculates the average windspeed in paddocks adjacent to trees. Used to calculate livestock chill.

    This assumes that livestock do not seek shelter. or should this retun a windspeed function
    Assumes that benefits of shelter only exist in the paddock adjacent to trees



    '''

    #inputs are the coeficients to the windspeed function (windspeed by distance from trees)
    #some how need to adjust for tree configuration. either by scalar or having inputs for several tree configs then just slicing based on the user inputted shape.

    #lLook at the ws inputs and decide if it is worth having more c slices to represent distance from trees.
    # if not then make a note that could expand the axis to consider ws close and further away from trees in the same paddock.
    # only required if the chill function is not linear which it isnt.
    #this will complicate the livestock side of it a bit with the b1 allocation because cant allocate sheep to the sheltered part of a paddock.

    width_paddock = 200 #width of paddock adjacent to trees #todo or this could be distance between tree belts...

    ##calc ws scalar
    #todo pass lmu_width into the ws by distance from trees function.

    ##calc temp scalar
    #todo pass lmu_width into the ws by distance from trees function.



    #todo ask dad about adjusting temp. Just adjust max temp i guess???
    tree_length, tree_width = f1_tree_shape()




    #width of paddock with trees


    #average windspeed in paddock with trees



def f_harvestable_biomass():
    '''

    '''
    #get growth/yr then convert to $/yr then probably discount? (hard part about discounting is assuming that the sale price wont rise)

    #need to see if this changes a lot between year to see if we need a yr axis on the input

    #see if can develop growth scalars for different shaped plantations.




def f_sequestration():
    '''
    Carbon sequestration from trees
    '''


    #need to conside fuel emissions

    #maybe need to consider time value of money like in MIDAS (ask dad about this)

    #consider tree plantatin shape on this.