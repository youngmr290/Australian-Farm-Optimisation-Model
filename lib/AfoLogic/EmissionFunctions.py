"""
This module contains functions that parametrise the emission components of the model.


author: Young



"""
import numpy as np

from . import UniversalInputs as uinp
from . import FeedsupplyFunctions as fsfun

def f_stock_ch4_animal_bc(ch, intake_f, intake_s, md_solid, level):
    '''
    Animal component of the Blaxter and Clapperton method for CH4 production - linked to the animal DVs.
    This is best linked to the animal DVs because the value varies with the level of feeding which is not known
    if it were connected to the feed-stuff. However, it does require an estimate of the M/D of the feed being consumed
    which is an optimised variable. However, a reasonable approximation is available from the M/D of the feed that
    generated the animal profile.

    Note, in Blaxter and Clapperton 1965 there is an error in table 5 (it should be b=2.37-0.05D (required to make figure 3 work))
    and there is an error CH4 equation derived from table 4 and 5. It should be CH4 (kcal/100kcal feed) = 1.30 + 0.112D + L (2.37 -0.05D).
    In the 2012 tech paper these error were fixed. This is the equation used below.

    Note 2: When they are lactating the ewes intake doubles (or more) and there is a corresponding increase in the
    energy expended on producing milk. So should the extra intake be counted as higher L and therefore reduce
    emissions per unit of intake. Or if there is no change in weight are they still considered to be at maintenance
    and hence high emissions per unit of intake? This is not mentioned in B&C. Currently assuming that level of feeding
    is relative to energy used for maintenance functions and doesn't include the energy used for milk production.


    :param ch: methane parameters
    :param intake_f: herb intake (kg)
    :param intake_s: supplement intake (kg)
    :param md_solid:
    :param level: the amount of feed consumed divided by the amount required when energy retention is zero, that
                  is the amount required at maintenance (level=0 at maint).
    :return: kg of methane produced per day - animal component
    '''

    ##Methane production total - this includes a feed component that is calculated elsewhere and hooked to feed activities
    ## not used just here to show the full equation
    # ch4_total_mj = ch[1, ...] * (intake_f + intake_s) * ((ch[2, ...] + ch[3, ...] * md_solid) + (level + 1) * (ch[4, ...] - ch[5, ...] * md_solid))

    ##Methane production animal component
    ch4_animal_mj = ch[1, ...] * (intake_f + intake_s) * (level + 1) * (ch[4, ...] - ch[5, ...] * md_solid)

    ##convert from mj of methane to kg
    ch4_energy_density = 52.5 #energy density of methane is 50–55.5 MJ/kg
    ch4_animal_kg = ch4_animal_mj / ch4_energy_density
    return ch4_animal_kg


def f_stock_ch4_feed_bc(intake, md):
    '''
    Feed component of the Blaxter and Clapperton method for CH4 production - linked to the feed-stuff DVs.
    This is best linked to the feed-stuff DVs because the value varies with M/D and this is only an estimate when
    emissions are calculated for the animal component of the emissions.

    Note, in Blaxter and Clapperton 1965 there is an error in table 5 (it should be b=2.37-0.05D (required to make figure 3 work))
    and there is an error CH4 equation derived from table 4 and 5. It should be CH4 (kcal/100kcal feed) = 1.30 + 0.112D + L (2.37 -0.05D).
    In the 2012 tech paper these error were fixed. This is the equation used below.

    :param intake: dry matter intake of the feed-stuff decision variable (kg).
    :param md:
    :return: kg of methane produced per x intake - feed component
    '''
    ch = uinp.parameters['i_ch_c2'][:,0] #slice c2 use the first genotype (note the methane params are currently the same for all genotypes)
    ch4_feed_mj = ch[1] * (intake) * (ch[2] + ch[3] * md)

    ##convert from mj of methane to kg
    ch4_energy_density = 52.5 #energy density of methane is 50–55.5 MJ/kg
    ch4_feed_kg = ch4_feed_mj / ch4_energy_density
    return ch4_feed_kg


def f_stock_ch4_feed_nir(intake, dmd):
    '''
    Calculates the component of livestock methane emissions linked to feed activities, using the methods documented
    in the National Greenhouse Gas Inventory Report.

    livestock produce methane from both enteric fermentation and anaerobic digestion of manure.
    The amount of emissions are effected by both
    livestock factors (e.g. age, relative size, EBG) and feed factors (e.g. quality, protein content, intake).
    Thus, in AFO the NIR equations are split between livestock and feed activities for improve accuracy.

    The NIR equations for livestock methane emissions are as follows:
    
    - Methane production from enteric fermentation (M): M = I x 0.0188 + 0.00158
    - Methane production from manure (M): M = I x (1 - DMD) x EFT
    
    :param intake: dry matter intake
    :param dmd:
    :return:
    '''
    ##inputs
    temperate_emission_factor = uinp.emissions['i_eft'] #temperate emission factor

    ##methan production
    ###Enteric methane production per kg of feed eaten (M): M = I x 0.0188 + 0.00158
    ### note the fixed (0.00158) component is accounted for in the stock part of the equation.
    ch4_entric = intake * 0.0188
    ###Methane production from manure per kg of feed eaten (M): M = I x (1 - DMD) x EFT
    ch4_manure = intake * (1 - dmd) * temperate_emission_factor
    ###total methane
    ch4 = ch4_manure + ch4_entric

   ##return methane emissions. These are converted to co2 equivalents at a later stage.
    return ch4


def f_stock_ch4_animal_nir():
    '''
    Calculates the component of livestock methane emissions linked to stock activities, using the methods documented
    in the National Greenhouse Gas Inventory Report.

    livestock produce methane from both enteric fermentation and anaerobic digestion of manure.
    The amount of emissions are effected by both
    livestock factors (e.g. age, relative size, EBG) and feed factors (e.g. quality, protein content, intake).
    Thus, in AFO the NIR equations are split between livestock and feed activities for improve accuracy.

    The NIR equations for livestock methane emissions are as follows:

    - Methane production from enteric fermentation (M): M = I x 0.0188 + 0.00158
    - Methane production from manure (M): M = I x (1 - DMD) x EFT

    At the moment milk consumption does not contribute to methane from either enteric fermentation or
    manure decomposition (likely not a big issue since it would be of a small magnitude).

    :return: methane production kg/d that is linked to the livestock decision variable.
    '''

    ##Fixed daily methane production. This part of the equation is not linked to feed intake. Essentially the methane emitted by an animal each day irrelevant of feed intake.
    ##Methane production from feed intake e.g. pasture is accounted for in the feed emission function.
    ch4_fixed = 0.00158

    return ch4_fixed


def f_stock_n2o_feed_nir(intake, dmd, cp):
    '''
    Calculates the component of livestock nitrous oxide emissions linked to feed activities, using the methods documented
    in the National Greenhouse Gas Inventory Report.

    livestock produce nitrous oxide emissions from the combined nitrification-denitrification process that
    occurs on the nitrogen in manure. The amount of emissions are effected by both
    livestock factors (e.g. age, relative size, EBG) and feed factors (e.g. quality, protein content, intake).
    Thus, in AFO the NIR equations are split between livestock and feed activities for improve accuracy.

    The NIR equations for livestock nitrous oxide emissions are as follows:

    - Nitrogen retained in the body(NR): NR = {(0.045 x MP) + (WP x 0.84) + {[(212 - 4 x {[(EBG x 1000) / (4 x SRW ^ 0.75)] - 1}) - (140 - 4 x {[(EBG x 1000) / (4 x SRW ^ 0.75)] - 1}) / {1 + exp(-6 x(Z - 0.4))}] x EBG} / 1000 / 6.25
    - Nitrogen excreted in faeces (F): F = {0.3 x (CPI x (1 - [(DMD + 10) / 100])) + 0.105 x (ME x I x 0.008) + 0.08 x (0.045 x MC) + 0.0152 x I} / 6.25
    - Nitrogen excreted in urine (U): U = (CPI / 6.25) - NR - F
    - Nitrous oxide production from animal waste (N): N = ((F x EFf x Cg) + (U x EFu x Cg))

    Note: Freer 2007: Crude protein, being total N × 6.25

    :param intake:
    :param dmd:
    :param cp:
    :return: kilograms of n2o emissions linked to the consumption of x amount of feed - feed component of equation
    '''
    ##inputs
    EFf = uinp.emissions['i_eff']  # emision factor faeces - Emission factors are used to convert a unit of activity into its emissions equivalent.
    EFu = uinp.emissions['i_efu']  # emision factor urine - Emission factors are used to convert a unit of activity into its emissions equivalent.
    Cg = uinp.emissions['i_cf_n2o']  # 44/28 - weight conversion factor of Nitrogen (molecular weight 28) to Nitrous oxide (molecular weight 44)
    me = fsfun.f1_dmd_to_md(dmd) #Metabolisable energy MJ/kg DM
    ##Nitrous oxide production - feed component of equation
    ###crude protein intake
    cpi_solids = intake * cp
    ###Nitrogen excreted in faeces (F): F = {0.3 x (CPI x (1 - [(DMD + 10) / 100])) + 0.105 x (ME x I x 0.008) + 0.08 x (0.045 x MC) + 0.0152 x I} / 6.25
    ###milk component is accounted for in the animal emission function because milk consumed is calculated in sgen.
    NF_solids = (0.3 * (cpi_solids * (1 - ((dmd + 10) / 100))) + 0.105 * (me * intake * 0.008) + 0.0152 * intake) / 6.25
    ###N excreted in urine (U): U = (CPI / 6.25) - NR - F
    ### NR is accounted for in the animal emission function.
    NU_solids = (cpi_solids / 6.25) - NF_solids  # feed component
    ###convert to nitrous oxide
    n2o = (NF_solids * EFf * Cg) + (NU_solids * EFu * Cg)

    ##return nitrous oxide emissions. These are converted to co2 equivalents at a later stage.
    return n2o


def f_stock_n2o_animal_nir(cl, d_cfw, relsize, srw, ebg, mp=0, mc=0):
    '''
    Calculates the component of livestock nitrous oxide emissions linked to animal activities, using the methods documented
    in the National Greenhouse Gas Inventory Report.

    livestock produce nitrous oxide emissions from the combined nitrification-denitrification process that
    occurs on the nitrogen in manure. The amount of emissions are effected by both
    livestock factors (e.g. age, relative size, EBG) and feed factors (e.g. quality, protein content, intake).
    Thus, in AFO the NIR equations are split between livestock and feed activities for improve accuracy.

    The NIR equations for livestock nitrous oxide emissions are as follows:

    - Nitrogen retained in the body(NR): NR = {(0.045 x MP) + (WP x 0.84) + {[(212 - 4 x {[(EBG x 1000) / (4 x SRW ^ 0.75)] - 1}) - (140 - 4 x {[(EBG x 1000) / (4 x SRW ^ 0.75)] - 1}) / {1 + exp(-6 x(Z - 0.4))}] x EBG} / 1000 / 6.25
    - Nitrogen excreted in faeces (F): F = {0.3 x (CPI x (1 - [(DMD + 10) / 100])) + 0.105 x (ME x I x 0.008) + 0.08 x (0.045 x MC) + 0.0152 x I} / 6.25
    - Nitrogen excreted in urine (U): U = (CPI / 6.25) - NR - F
    - Nitrous oxide production from animal waste (N): N = ((F x EFf x Cg) + (U x EFu x Cg))

    Note: Freer 2007: Crude protein, being total N × 6.25

    :param d_cfw: daily growth of clean fleece
    :param mp: milk production i.e mp2_dams
    :param mc: milk consumption i.e mp2_yatf
    :param relsize: relative size of animal
    :param srw: standard reference weight of animal
    :param ebg: daily empty body gain
    :return: kilograms of n2o emissions per day linked to the animal activity
    '''


    ##inputs
    ##inputs
    EFf = uinp.emissions['i_eff']  # emision factor faeces - Emission factors are used to convert a unit of activity into its emissions equivalent.
    EFu = uinp.emissions['i_efu']  # emision factor urine - Emission factors are used to convert a unit of activity into its emissions equivalent.
    Cg = uinp.emissions['i_cf_n2o']  # 44/28 - weight conversion factor of Nitrogen (molecular weight 28) to Nitrous oxide (molecular weight 44)
    me_milk = cl[6] #ME / kg used to convert mp2 to kg of wet milk
    milk_dmd = 85 #dmd of milk

    MP = mp/me_milk #milk production kg/d - need to convert mp2 from Mj/d to kg by dividing by ME
    MC = mc/me_milk#milk intake - not the same as MP because of multiples - need to convert mp2 from Mj/d to kg by dividing by ME
    WP = d_cfw #clean wool production per day
    Z = relsize
    SRW = srw
    EBG = ebg

    ##Nitrous oxide production - feed component of equation based on milk consumed and nitrogen retained in the body
    ###crude protein of milk intake
    cpi_milk = 0.045 * MC
    ###Nitrogen excreted in faeces (F): F = {0.3 x (CPI x (1 - [(DMD + 10) / 100])) + 0.105 x (ME x I x 0.008) + 0.08 x (0.045 x MC) + 0.0152 x I} / 6.25
    ###milk component - solids component is accounted for in the animal emission function because milk consumed is calculated in sgen.
    NF_milk = (0.3 * (cpi_milk * (1 - ((milk_dmd + 10) / 100))) + 0.08 * cpi_milk) / 6.25
    ###Nitrogen retained in the body(NR): NR = {(0.045 x MP) + (WP x 0.84) + {[(212 - 4 x {[(EBG x 1000) / (4 x SRW ^ 0.75)] - 1}) - (140 - 4 x {[(EBG x 1000) / (4 x SRW ^ 0.75)] - 1}) / {1 + exp(-6 x(Z - 0.4))}] x EBG} / 1000 / 6.25
    NR = ((0.045 * MP) + (WP * 0.84) + (((212 - 4 * (((EBG * 1000) / (4 * SRW ** 0.75)) - 1)) - (140 - 4 * (((EBG * 1000) / (4 * SRW ** 0.75)) - 1)) / (1 + np.exp(-6 * (Z - 0.4)))) * EBG) / 1000) / 6.25
    ###N excreted in urine (U): U = (CPI / 6.25) - NR - F
    NU_milk = (cpi_milk / 6.25) - NR - NF_milk #animal component
    ###convert to nitrous oxide
    n2o = (NF_milk * EFf * Cg) + (NU_milk * EFu * Cg)

    ##return nitrous oxide emissions per day. These are converted to co2 equivalents at a later stage.
    return n2o