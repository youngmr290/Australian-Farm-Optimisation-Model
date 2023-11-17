"""
This module contains functions that parametrise the emission components of the model.


author: Young



"""
import numpy as np

from . import UniversalInputs as uinp
from . import FeedsupplyFunctions as fsfun

def f_ch4_animal_bc(ch, intake_f, intake_s, md_solid, level):
    #todo these formulas need to be reviewed, then connected to emissions in Pyomo.
    # Compare with original Blaxter & Clapperton to check if error in the sign in Tech 2012 paper.
    # Note comment made in Wilkerson etal 1995
    # Compare the original MIDAS derivation of animal and feed components
    ##Methane production total - this includes a feed component that is calculated elsewhere and hooked to feed activities
    ch4_total = ch[1, ...] * (intake_f + intake_s)*((ch[2, ...] + ch[3, ...] * md_solid) + (level + 1) * (ch[4, ...] - ch[5, ...] * md_solid))
    ##Methane production animal component
    ch4_animal = ch[1, ...] * (intake_f + intake_s) * (level + 1) * (ch[4, ...] - ch[5, ...] * md_solid)
    return ch4_animal


def f_ch4_feed_bc():
    #todo needs to be built
    return


def f_ch4_feed_nir(intake, dmd):
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
    
    :param intake:
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
    ###total methan
    ch4 = ch4_manure + ch4_entric

   ##return methane emissions. These are converted to co2 equivalents at a later stage.
    return ch4


def f_ch4_animal_nir(mc=0):
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

    :param mc: milk consumption i.e mp2_yatf
    :return:
    '''
    ##inputs
    dm_milk = 20.6 #milk dry matter content (MJ/kg) source:Stankov etal 2022. Dry matter was 21.36% in April and 19.85% in August at the end of lactation.

    MC = mc/dm_milk#milk intake kg DM

    ##Methane production due to milk intake (methane from other feed intake e.g. pasture is accounted for in the feed emission function)
    ch4_milk = MC * 0.0188

    ##Fixed daily methane production. This part of the equation is not linked to feed intake. Essentially the methane emitted by an animal each day irrelevant of feed intake.
    ch4_fixed = 0.00158

    ##return methane emissions per day. This is converted to co2 equivalents at a later stage.
    ch4 = ch4_milk + ch4_fixed
    return ch4


def f_n2o_feed_nir(intake, dmd, cp):
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
    :return:
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


def f_n2o_animal_nir(cl, d_cfw, relsize, srw, ebg, mp=0, mc=0):
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
    :return:
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