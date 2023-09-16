'''
General feed supply equations used throughout AFO.
'''


import numpy as np
import math

from . import Functions as fun
from . import SeasonalFunctions as zfun
from . import PropertyInputs as pinp
from . import UniversalInputs as uinp
from . import Periods as per


#######################################
#function for feed budget & livestock #
#######################################
def f1_dmd_to_md(dmd):
    '''
    Calculation of megajoules energy per kg dry matter (MD) from dry matter digestibility (DMD).

    Where:

        * MD is megajoules of energy per unit of dry matter,
        * DMD is dry matter digestibility.

    :param dmd: dmd can be a numpy array or a scalar it can be either a percentage or a decimal.
    :returns: M/D in MJ of ME per kg of DM

    Note: this could be expanded to include supplement (M/D = 0.134 * dmd + 0.235 ee + 1.23)  Freer et al 2007 eqn 1.11A
          using an extra 'type' input that is default 'herbage'
    '''
    try:
        if (dmd > 1).all() : dmd /= 100 # if dmd is a list or an array and is not a decimal then convert to decimal (in excel 80% is 0.8 in python)
    except:
        if dmd > 1:          dmd /= 100 # if dmd is a scalar and is not a decimal then convert to decimal   ^ alternative would be to convert scalar values to a list (if dmd isinstance not list: dmd=[dmd]) or perhaps type is float]
    return np.maximum(0, 17.2 * dmd - 1.707)                # formula 1.12A from Freer et al 2007 pg 7.

def f1_md_to_dmd(md):
    '''
    Calculation of dry matter digestibility (dmd) from megajoule per kg of dry matter (MD).

    Where:

        * DMD is dry matter digestibility.
        * MD is megajoules of energy per unit of dry matter

    :param md: MJ of ME per kg of DM
    :returns: dmd as a decimal
    '''
    return (md + 1.707) / 17.2              # from formula 1.12A from Freer et al 2007 pg 7.


def f_effective_mei(dmi, md, threshold_f, confinement_inc, ri=1, eff_above=0.5):
    """According to the Australian Feeding Standards (Freer et al 2007) an animal that is gaining weight then
       losing weight is less efficient than an animal that is maintaining weight on a constant diet. Therefore,
       switching between a high quality diet and a low quality diet to maintain weight requires more MJ of ME
       than using a consistent medium quality diet.
       This is represented by scaling the MEI that is above that required to meet the animal target with the
       animal target being based on the mid-point of the feed pool (hence effective mei varies for each feed
       pool (v) because the target profile varies between pools).
       This would be important to represent in the model for animals with a liveweight target set to reduce MEI and
       maximise SR. This occurs when animals are being retained on-farm over summer, and extra MEI with weight
       gain would not increase profit. However, for animals that are being fed to gain weight to achieve a target
       for turnoff, better quality feed with increased weight gain would increase profit due to faster growth with
       earlier turnoff.
       In situations where a animal decision variable with faster growth would be selected, if it were available, it
       would be better to not represent the reduced efficiency, whereas if that decision variable would not be selected
       then the reduction should be represented.
       Working conclusion: All feedstuffs should go through f_effective_mei. However, the highest nv pool should not
       be reduced, even if the feed quality is greater than then animal demand, on the assumption that the target for
       these animals is growth and that faster growth would be better.
       The exception to the above is supplement because the quantity of supplement can be controlled so as not to
       result in LWG followed by a period of LWL.

    :param dmi: value or array - Dry matter intake of the feed decision variable (kg).
    :param md: value or array - M/D of the feed (MJ of ME / kg of DM).
    :param threshold_f: value or array - Diet quality (ME/Vol) required by animals (with a f axis). Below the threshold: effective m/d == m/d
    :param confinement_inc: boolean stating if the confinement feed pool is included.
    :param ri: value or array, optional (1.0) - Relative intake of the feed (quality and quantity).
    :param eff_above: value or array, optional (0.5) - Efficiency that energy is used if above required quality, and
                      animals are gaining then losing weight.

    If inputs are provided in arrays then they must be broadcastable.

    :return: ME available to the animal to meet their ME requirements for the target profile, from the quantity of DM consumed in the feed decision variable.

    """
    nv = md * ri
    nv_effective_f  = np.minimum(nv, threshold_f + (nv - threshold_f) * eff_above)
    ## no reduction in efficiency in the highest nv pool (excluding confinement pool) on the assumption that LWG is the target for these animals.
    f_high_idx = -1 - confinement_inc #highest pool is -1 unless confinement is included then it is -2.
    nv_effective_f[f_high_idx,...] = nv
    md_effective_f = nv_effective_f / ri
    mei_effective_f = dmi * md_effective_f
    return mei_effective_f


def f_hf(hr, cr=None):
    '''
    Calculate height factor of feed.

    :param hr: the height of feed relative to 3cm per t/ha.
    :param cr: cr12 effect of pasture height
    :return: Height factor
    '''
    ##create scalar cr if not passed in
    if cr is None:
        ###Scalar version of cr[1,…] using c2[0] (finewool merino)
        cr12 = uinp.parameters['i_cr_c2'][12,0]
    else:
        cr12=cr[12, ...]

    ##calc hf
    hf = 1 + cr12 * (hr - 1)
    return hf


def f_foo_convert(cu3, cu4, foo, pasture_stage, legume=0, hr_scalar = 1, cr=None, z_pos=-1, treat_z=False):
    '''
    Adjust FOO for measurement method.

    Depending on the region, FOO can be measured differently. For example, in WA when measuring FOO
    the pasture is cut to ground level using a scalpel versus NSW where it is cut at a higher level
    with shears. This results in the same amount of feed being valued at a higher FOO in WA.
    Furthermore, there are differences in sward morphology when comparing WA pasture with Victoria and NSW.
    WA has a higher clover content and for any level of FOO are more prostrate i.e. shorter. These conversion
    equations also make an adjustment for pasture height, which is returned in the hf variable.
    The FOO is adjusted in this function to the method which is used in the livestock production equations.
    This was the shears method but now appears to be total above ground biomass which is as used in WA.

    FOO only needs to be adjusted for the intake equations. It does not need to be adjusted for the pasture
    growth calculations because the FOO in the inputs is consistent with the FOO in the pasture growth calculations.

    The minimum grazing limit FOO level (level which grazing can not occur below) is specified in FOO units
    of the livestock grazing equations (this value is removed from available FOO in f_ra().
    The base level in the pasture grazing intensity calculations is in units of FOO that have been used to define
    the PGR by FOO inputs. Thus, in WA, the base level FOO would be higher than NSW.

    :param cu3: this parameter should already be slice on the c4 axis.
    :param cu4: this parameter should already be slice on the c4 axis.
    :param treat_z: boolean to control if z axis is treated. Pasture stage is used as an index so in the weighted average
                    season model the z axis should get treated at the end of this function. This happens for stock but
                    not for pasture because it would require doubling up lots of code.
    '''
    ##pasture conversion scenario (convert the region and pasture stage to an index
    ### because the second axis of cu3 is a combination of region & stage)
    ### To allow entry of a decimal for the stage the calculations are carried out for 2 stage and then weighted
    pasture_stage1 = np.trunc(pasture_stage).astype(np.int)
    pasture_stage2 = np.minimum(pasture_stage1 + 1, uinp.pastparameters['i_n_pasture_stage'] - 1).astype(np.int)
    proportion_1 = 1 - pasture_stage % 1
    conversion_scenario1 = pinp.sheep['i_region'] * uinp.pastparameters['i_n_pasture_stage'] + pasture_stage1
    conversion_scenario2 = pinp.sheep['i_region'] * uinp.pastparameters['i_n_pasture_stage'] + pasture_stage2
    ##select cu3&4 params for the specified region and stage. Remaining axes are season and formula coefficient (intercept & slope)
    cu3_1=cu3[..., conversion_scenario1]
    cu4_1=cu4[..., conversion_scenario1]
    cu3_2=cu3[..., conversion_scenario2]
    cu4_2=cu4[..., conversion_scenario2]
    ##Convert FOO to hand shears measurement
    foo_grazplan1 = np.maximum(0, np.minimum(foo, cu3_1[2] + cu3_1[0] * foo + cu3_1[1] * legume))
    foo_grazplan2 = np.maximum(0, np.minimum(foo, cu3_2[2] + cu3_2[0] * foo + cu3_2[1] * legume))
    ##Estimate height of pasture
    height1 = np.maximum(0, np.exp(cu4_1[3] + cu4_1[0] * foo + cu4_1[1] * legume + cu4_1[2] * foo * legume)
                        + cu4_1[5] + cu4_1[4] * foo)
    height2 = np.maximum(0, np.exp(cu4_2[3] + cu4_2[0] * foo + cu4_2[1] * legume + cu4_2[2] * foo * legume)
                        + cu4_2[5] + cu4_2[4] * foo)
    ##Calculate the weighted average for foo & height based on the pasture_stage
    foo_grazplan = foo_grazplan1 * proportion_1 + foo_grazplan2 * (1 - proportion_1)
    height = height1 * proportion_1 + height2 * (1 - proportion_1)
    ##Height density (height per unit FOO)
    hd = fun.f_divide(height, foo_grazplan) #handles div0 (e.g. if in feedlot with no pasture or adjusted foo is less than 0)
    ##height ratio - height of this feed per unit foo vs standard pasture (value great than 1 means increased availability (more up right), value less than 1 is less available (prostrate))
    hr = hr_scalar * hd / uinp.pastparameters['i_hd_std']
    hr = np.clip(hr, 0.333, 3) #clip availability of current pasture within a factor of 3 relative to the grazplan standard pasture.
    ##calc hf
    hf = f_hf(hr, cr)
    ##apply z treatment
    if treat_z:
        foo_grazplan = zfun.f_seasonal_inp(foo_grazplan,numpy=True,axis=z_pos)
        hf = zfun.f_seasonal_inp(hf,numpy=True,axis=z_pos)
    return foo_grazplan, hf

def f_ra_cs(foo, hf, cr=None, zf=1):
    '''
    CSIRO relative availability of a feed.

    Relative availability is the availability of a feed to livestock. It is the product of two
    components; relative rate of eating (RR) and relative time spent grazing (RT).

    NOTE: Only pass cr parameter if called from Stock_generator that have a g axis

    :param foo: feed on offer (kg/ha dry matter)
    :param hf: height factor
    :param cr: parameters for prediction of relative intake
    :param zf: mouth size factor. Accommodates the smaller mouth size of young animals, allowing them to achieve
               their potential intake at a lower level of herbage availability than would be needed by adults.
    :return: Relative availability
    '''
    ##create scalar cr if not passed in
    if cr is None:
        ###Scalar version of cr[…] using c2[0] (finewool merino)
        cr4 = uinp.parameters['i_cr_c2'][4, 0]
        cr5 = uinp.parameters['i_cr_c2'][5, 0]
        cr6 = uinp.parameters['i_cr_c2'][6, 0]
        cr13 = uinp.parameters['i_cr_c2'][13, 0]
    else:
        cr4 = cr[4, ...]
        cr5 = cr[5, ...]
        cr6 = cr[6, ...]
        cr13 = cr[13, ...]
    ##Adjust the FOO level to allow for the ungrazable limit measured in the units of FOO defined by GrazPlan.
    ### This is so RI=0 when foo == i_min_grazing_limit and just above 0 is foo is just above the limit.
    foo = np.maximum(0, foo - uinp.pastparameters['i_min_grazing_limit'])
    ##Relative rate of eating (rr) & Relative time spent grazing (rt)
    try:
        ###Scalar version
        rr = 1 - math.exp(-(1 + cr13 * 1) * cr4 * hf * zf * foo) #todo *1 is a reminder that this formula could be improved in a future version
        rt = 1 + cr5 * math.exp(-(1 + cr13 * 1) * (cr6 * hf * zf * foo)**2)
    except:
        ###Numpy version
        rr = 1 - np.exp(-(1 + cr13 * 1) * cr4 * hf * zf * foo) #todo *1 is a reminder that this formula could be improved in a future version
        rt = 1 + cr5 * np.exp(-(1 + cr13 * 1) * (cr6 * hf * zf * foo)**2)
    ##Relative availability
    ra = rr * rt
    return ra


def f_rq_cs(dmd, legume, cr=None, sf=0):
    '''
    CSIRO relative ingestibility (rq) of a feed source.

    NOTE: Only pass cr parameter if called from Stock_generator that have a g axis
    NOTE 2: sf is currently not used. Will be required if using tropical species

    :param dmd: feed digestibility
    :param legume: legume proportion.
    :param cr: parameters for prediction of relative intake
    :param sf: species factor. It is making an adjustment for the intake of C4 grasses being higher than C3 grasses if measured at the same digestibility.
    :return: Relative ingestibility
    '''
    ##To work for DMD as a % or a proportion
    try:
        if (dmd >= 1).any() : dmd /= 100
    except:
        if dmd >= 1:          dmd /= 100
    ##create scalar cr if not passed in
    if cr is None:
        ###Scalar version of cr[…] using c2[0] (finewool merino)
        cr1 = uinp.parameters['i_cr_c2'][1, 0]
        cr3 = uinp.parameters['i_cr_c2'][3, 0]
    else:
        cr1 = cr[1, ...]
        cr3 = cr[3, ...]
    ##Relative ingestibility
    try:
        ###Scalar version of formula
        rq = max(0.01, min(1, 1 - cr3 * (cr1 - (dmd +  sf * (1 - legume))))) #(1-legume) because sf is actually a factor related to the grass component of the sward
    except:
        ###Numpy version of formula
        rq = np.maximum(0.01, np.minimum(1, 1 - cr3 * (cr1 - (dmd +  sf * (1 - legume))))) #(1-legume) because sf is actually a factor related to the grass component of the sward
    return rq


def f_ra_mu(foo, hf, zf=1, cu0=None):
    '''
    Murdoch relative availability of a feed.

    Relative availability is the availability of a feed to livestock.

    NOTE: Only pass cr parameter if called from Stock_generator that have a g axis

    :param foo: feed on offer (kg/ha dry matter)
    :param hf: height factor
    :param zf: mouth size factor. Accommodates the smaller mouth size of young animals, allowing them to achieve
               their potential intake at a lower level of herbage availability than would be needed by adults.
    :param cu0:
    :return: Relative availability
    '''
    ##create scalar cr if not passed in
    if cu0 is None:
        ###Scalar version of cr[…] using c2[0] (finewool merino)
        cu0 = uinp.parameters['i_cu0_c2'][0, 0]
    else:
        cu0 = cu0[0, ...]
    ##Adjust the FOO level to allow for the ungrazable limit measured in the units of FOO defined by GrazPlan.
    ### This is so RI=0 when foo == i_min_grazing_limit and just above 0 is foo is just above the limit.
    ### The same minimum limit is used for this equation system as for the GrazPlan equations
    foo = np.maximum(0, foo - uinp.pastparameters['i_min_grazing_limit'])
    ##Relative availability
    try:
        ###Scalar version
        ra = 1 - cu0 ** (hf * zf * foo)
    except:
        ###Numpy version (exact same so not required atm)
        ra = 1 - cu0 ** (hf * zf * foo)
    return ra


def f_rel_intake(ra, rq, legume, cr=None):
    '''
    Relative intake – The proportion of livestock potential intake required to consume 1kg of
    feed given the availability and digestibility.

    Calculation of relative intake includes the effect of feed availability, feed quality and the interaction.
    This function is not called for feeds (such as supplements) that do not have an 'availability' characteristic.
    The calculated RI can be greater than 1 - which implies that actual intake can be greater than the potential intake
    This can occur if rq is greater than 1, due to the 'legume' effect on the intercept or if DMD is greater than cr1.

    NOTE: Only pass cr parameter if called from Stock_generator that have a g axis
    '''
    ##create scalar cr if not passed in
    if cr is None:
        ###Scalar version of cr[…] using c2[0] (finewool merino)
        cr2 = uinp.parameters['i_cr_c2'][2, 0]
    else:
        cr2=cr[2, ...]

    ##Relative intake

    try:
        ###Scalar version of formula
        ri = max(0.05, ra * rq * (1 + cr2 * ra**2 * legume))
    except:
        ###Numpy version of formula
        ri = np.maximum(0.05, ra * rq * (1 + cr2 * ra**2 * legume))
    return ri
