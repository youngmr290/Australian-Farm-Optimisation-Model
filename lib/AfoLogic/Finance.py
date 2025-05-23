
"""
author: young

The financial components of the model includes:

    - interest
    - cashflow
    - a limit on capital borrowings
    - minimum return on expenditure
    - opportunity cost of assets

Each module tracks its relevant financial components. The finance module provides a common location to describe the
finance section of the model and is also home to key finance functions that calculate the interest and working
capital of each cashflow item.

To support the sporadic nature of farming income, finance is often drawn from the bank throughout the year to fund
costly operations such as purchasing fertiliser and chemicals. In AFO the total capital required for the given farm structure is tallied and can be
constrained to a user specified level. This allows the user to examine how the business structure would change if
finance is limited. This can also be used to ensure the model doesn't overdraw an unrealistic/undesired level of capital
from the bank. Total farm capital required is calculated from the value of starting assets plus the sum of all the
expenses minus any income, between the previous 'main' income (e.g. harvest or shearing) and the peak debt date.
Peak debt is typically expected for an enterprise just before the main income is received for that enterprise, ensuring
the main income for the enterprise is not included in the working
capital constraint. The aim of the working capital constraint is to allow the user to constrain management
practices which have high costs. If the main income was included in this constraint there would be no way to
constrain high cost high reward management practices.
The default is to have one peak debt date per enterprise, just before the main income for that enterprise is received.

In an equilibrium model there is no start and end point. This complicates the calculation of interest because
interest must be calculated for a given period. In AFO the interest period starts and finishes after the main income
for the enterprise is received. This is logical from an expense point of view because the expense accumulates
interest from the date it was incurred through to when the income associated with that cost is received. This ensures that
expenditure is only incurred if the return exceeds the cost of interest.

Asset value is the value of all assets at the beginning of the interest period. The opportunity cost of investing
in farm assets including livestock, machinery and infrastructure (sheds, yards etc) is captured in AFO. Asset
value operates in conjunction with the interest to represent the opportunity cost of holding assets. Its role
is to ensure that all assets that are selected have a return more than the interest cost, this ensures the
optimal solution does not include assets that return less than investing the same money in a savings account
(or to reduce core debt). This structure makes an equilibrium model generate a result similar to a multi-period
model that accounts for the interest cost of money. Livestock flock structure is the main ‘decision’ that is
altered by the inclusion of an asset value. For livestock, it ensures that the flock structure optimisation
accounts for the opportunity cost of interest foregone from holding an animal till it is sold.
Note: Main income is not receiving any interest because it is occurring at the end of the cashflow
period (this is by definition because we are assuming that expenses are aimed at generating the "main income" -
so the cashflow period starts just after the main income period and ends with the main income period). So the
main income is earning interest for 0 days. Interest on Tactical income is just offsetting the asset cost,
which can be thought of 'stopping' the asset cost at the point that the animal is sold. This means that the
sale of the animal only needs to have generated enough income to justify the asset cost for a portion of the year.

The interest rate for credit and debit are different for farmers ‘real money’ in the bank.
However, in AFO the same interest rate is used to represent debit and credit.
The reasons are:

#. Many farmers often have a core debt, so the farm cash position is usually negative even though
   their short term operating account may occasionally be positive. The differential interest
   rates are only justified if the farmer does not operate with a sweep facility to pay down
   core debt and then redraw when required later.
#. As discussed above, the asset value and the cashflow operate together in the optimisation of
   flock structure. This implies that the interest rate for the cash flow should be the same as
   the discount rate for the asset value.

AFO tallies the total farm expenditure, adjusts it by a user defined return on expense factor and
includes it in the objective to ensure the model achieves a minimum return on expenditure. The
purpose of this is to represent farmer behaviour. It can also be used in the static equilibrium
version to 'fudge' the risk associated with seasonal variation and reduce the optimal stocking rate
to better align with on-farm values.
The minimum rate of return on expenditure (MinROE) is specified by the user and can be turned off.
The current rate in the static equilibrium model (25%) was calibrated by a comparison of the
model output with on-farm benchmarking (e.g. Planfarm, 2022).

There is no representation of a starting cash balance. If it is included the model just selects the
highest amount because that earns the most interest. The model can overdraw the working account if it needs
cash, so this does not affect the model solution.

Tax is also not represented for several reasons:

    #. There are several mechanisms by which farmers seek to lessen their tax liabilities. Not all
       are 'economically rational' and not all are easily represented in a LP model.

    #. Many farmers nowadays invest in farm management deposits and income-averaging as a means
       of taxable income averaging and to smooth working capital borrowings. if the FMDs could be used
       'perfectly' then each year would have the same taxable profit. Thus, the optimal farm management
       is unaffected by the inclusion of tax.

    #. AFO is a bioeconomic model with the aim of optimising farm management. It is not a finance model.

"""
##python modules
import pandas as pd
import numpy as np

##AFO modules
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import PropertyInputs as pinp
from . import Sensitivity as sen
from . import Periods as per
from . import Functions as fun
from . import SeasonalFunctions as zfun

na = np.newaxis
#######################
# cashflow & interest #
#######################
def f_cashflow_allocation(date_incurred,enterprise=None,z_pos=-1, c0_inc=False, is_phase_cost=False, interest_only=False):
    '''
    Allocates cashflow and wc to a season period and accounts for an interest component.

    Cashflow allocation always has a length of 1. Meaning that cost is allocated based on the start date when it is
    incurred. Interest is calculated from this date until the end of the cashflow periods. The reason for not
    including a length is that cashflow for a give decision variable can not cross a node
    otherwise the child weather years do not incur the cashflow (because the variable might not exist).

    Working capital is tallied for each 'main' enterprise (controlled by user inputs).
    The working capital is accumulated from the most recent main income (across all enterprises) until the peak debt
    date for the given enterprise. E.g. in a typical mix farm system the stock working constraint tallys up
    the cashflow from just after harvest (most recent main income) until just before shearing (stock peak debt).

    Note: For params linked to v_phase activity the timing of an item is adjusted (in fun.f_range_allocation)
    so that no cost is incurred between season start and break of season. This stops the model getting double costs in medium/late breaks where
    phases are carried over past the start of the season to provide dry pas and stubble area (because it is also
    accounted for by v_phase_increment).

    :param date_incurred: week of year when cashflow is incurred (must include z axis)
    :param enterprise: enterprise. If no enterprise is passed in the cashflow is averaged across the c0 axis.
    :param z_pos: axis position of z (must be negative e.g. reference from the end).
    :param c0_inc: boolean stating if c0 axis is included in date_incurred
    :param is_phase_cost: boolean stating if the cost is related to v_phase.
    :param interest_only: boolean stating if the cost is allocated to p7 in this function (if True this function only calcs interest).
    '''

    ##inputs
    rate = uinp.finance['i_interest']
    cashflow_date_c0 = per.f_cashflow_date()
    peakdebt_date_c0 = per.f_peak_debt_date()
    peakdebt_date_c0 = peakdebt_date_c0 + 364 * (cashflow_date_c0>peakdebt_date_c0) # peak debt is after the start of the cashflow

    ##expand cashflow and debt date to the same shape as date_incurred
    ndims = -date_incurred.ndim + c0_inc
    cashflow_date_c0 = fun.f_expand(cashflow_date_c0, left_pos=ndims-1)
    peakdebt_date_c0 = fun.f_expand(peakdebt_date_c0, left_pos=ndims-1)

    ##adjust yr of cashflow occurrence so it occurs within the cashflow periods for the interest calc below
    start_of_cash_c0 = cashflow_date_c0
    end_of_cash_c0 = start_of_cash_c0 + 363 #use 363 (364 is 1 yr in AFO) because end date is the day before the start of following yr otherwise can get item that starts on the last day of periods.
    add_yrs_c0 = np.ceil(np.maximum(0,(start_of_cash_c0 - date_incurred) / 364))
    sub_yrs_c0 = np.ceil(np.maximum(0,(date_incurred - end_of_cash_c0) / 364))
    date_incurred_c0 = date_incurred + add_yrs_c0 * 364 - sub_yrs_c0 * 364

    ##calc interest
    cashflow_incur_days_c0 = (end_of_cash_c0 - date_incurred_c0)
    wc_incur_days_c0 = (peakdebt_date_c0 - date_incurred_c0) #date incurred is set to be after the cashflow date
    cashflow_interest_c0 = (1 + rate / 364) ** cashflow_incur_days_c0
    wc_interest_c0 = (1 + rate / 364) ** wc_incur_days_c0 * (wc_incur_days_c0>=0) #bool to make wc 0 if the cashflow item occurs between peak debt and cashflow date (this stops an enterprises main income being included in wc constraint).

    ##adjust cashflow for enterprise - this essentially selects which interest to use.
    ## if no enterprise is provided the interest from all enterprise dates are averaged.
    if enterprise is not None:
        idx = list(sinp.general['i_enterprises_c0']).index(enterprise)
        cashflow_interest = cashflow_interest_c0[idx,...]
    else:
        cashflow_interest = np.average(cashflow_interest_c0, axis=0)


    ##Allocate wc to the correct c0 slice (note if cashflow falls between peak debt and the start of the cashflow period (i.e. the main income) it doesnt get allocated to any wc period.
    ##If both stk and crp are both big enterprises then wc is basically reset at the point of main income for both enterprises.
    ## Therefore, in real life, stk wc accumulates from just after harv (main income for crop) until just before shearing
    ## (peak debt for stk). And crp wc accumulates from just after shearing until just before harvest.
    ##If only one enterprise is big then wc accumulates from just after the main income for that enterprise until
    ### just before the main income for that enterprise.
    ### Note: cashflow items that fall between peak debt and main income (i.e they are the main cashflow) therefore doesn't get allocated to any wc constraint.
    ##mask c0 included - enterprises are only included if they are main enterprises (if it is a small enterprise
    ### it doesn't need its own wc constraint because the income will be insufficient to affect peak debt)
    crop_c0_inc = np.array([pinp.crop['i_crp_c0_inc']])
    stk_c0_inc = np.array([pinp.sheep['i_stk_c0_inc']])
    mask_c0_inc = np.concatenate([stk_c0_inc, crop_c0_inc]) #order of concat is important - needs to be the same as the c0 order in periods.py
    ###date of the most recent main income relative to stk peak debt
    start_of_cash_c0 = start_of_cash_c0 + 364*(start_of_cash_c0<peakdebt_date_c0%364) # make sure previous cashflow date is greater than peak debt date
    stk_previous_main_cashflow = np.max(start_of_cash_c0[mask_c0_inc] * np.logical_or(start_of_cash_c0<peakdebt_date_c0[0] % 364,
                                                                                      np.all(peakdebt_date_c0[0] % 364 < start_of_cash_c0))
                                        , axis=0, keepdims=True)
    crp_previous_main_cashflow = np.max(start_of_cash_c0[mask_c0_inc] * np.logical_or(start_of_cash_c0<peakdebt_date_c0[1] % 364,
                                                                                       np.all(peakdebt_date_c0[1] % 364 < start_of_cash_c0))
                                        , axis=0, keepdims=True)
    previous_main_cashflow_c0 = np.concatenate([stk_previous_main_cashflow, crp_previous_main_cashflow]) #order of concat is important - needs to be the same as the c0 order in periods.py
    ###calculate which wc constraint (stk or crop) the cashflow item falls into ie check if date incurred falls between last main income (from either enterprise) and peak debt date
    incur_between = np.logical_and(date_incurred_c0 % 364 >= previous_main_cashflow_c0
                                              , date_incurred_c0 % 364 <= peakdebt_date_c0 % 364)
    incur_before = np.logical_and(np.all(date_incurred_c0 % 364 <= peakdebt_date_c0 % 364, axis=0)
                                              , np.max(previous_main_cashflow_c0) == previous_main_cashflow_c0)
    incur_after = np.logical_and(np.all(date_incurred_c0 % 364 > previous_main_cashflow_c0, axis=0)
                                              , np.max(previous_main_cashflow_c0) == previous_main_cashflow_c0)
    ###handle the exception - when peak debt is the end of the year and main cashflow date is start of the year the maincashflow gets adjusted by adding 364 so that it is a bigger number than peak debt date. This works for most cases except if date incured occurs at the start of the year technically between peak debt and main income.
    ### is_exception == true means that the cashflow item fals between peak debt and main income therefore doesn't get allocated to any wc constraint.
    is_exception = np.any(np.logical_and(previous_main_cashflow_c0 > 364, previous_main_cashflow_c0 % 364 > date_incurred_c0 % 364), axis=0)
    ###combine all logic to identify the wc constraint that the cashflow item falls into.
    mask_wc_c0 = np.logical_and(np.logical_or(np.logical_or(incur_between, incur_before), incur_after), np.logical_not(is_exception))
    wc_interest_c0 = wc_interest_c0 * mask_wc_c0

    ##return before allocating to p7 if required
    if interest_only:
        return cashflow_interest, wc_interest_c0

    ##allocate to cashflow period
    p7_alloc_p7 = zfun.f1_z_period_alloc(date_incurred[na,...], z_pos=z_pos, is_phase_param=is_phase_cost)
    final_cashflow_p7 = cashflow_interest * p7_alloc_p7
    final_wc_p7c0 = wc_interest_c0 * p7_alloc_p7[:,na,...]

    ##get axis back into correct order - because all the other code was done before this function so rest of code expects different order
    final_wc_c0p7 = np.swapaxes(final_wc_p7c0, 0, 1)

    return final_cashflow_p7, final_wc_c0p7

#################
#overheads      #
#################
def overheads(params, r_vals):
    '''
    Calculate overhead costs in each cashflow period and the associated interest.

    Overheads are ongoing business expenses that are not directly attributed to creating a product
    or service. In AFO the user has the discretion to add, remove or alter the overheads that are
    including. Examples of overhead costs include; electricity, gas, shire rates, licenses,
    professional services, insurance and household expense.
    '''
    ##cost allocation - incurred at the beginning of each cash period
    overhead_start = np.array([182]) #Overheads are incurred in the middle of the year and incur half a yr interest (in attempt to represent the even spread of fixed costs over the yr)
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    ###call allocation/interset function - needs to be numpy
    ### no enterprise is passed because fixed cost are for both enterprise and thus the interest is the average of both enterprises
    overhead_cost_allocation_p7z, overhead_wc_allocation_c0p7z = f_cashflow_allocation(overhead_start, z_pos=-1)

    ##cost - overheads are incurred in the middle of the year and incur half a yr interest (in attempt to represent the even spread of fixed costs over the yr).
    overheads = pinp.general['i_overheads']
    overheads = overheads.sum()
    overhead_cost_p7z = overhead_cost_allocation_p7z * overheads
    overhead_wc_c0p7z = overhead_wc_allocation_c0p7z * overheads

    ##convert to df
    new_index_p7z = pd.MultiIndex.from_product([keys_p7,keys_z])
    overhead_cost_p7z = pd.Series(overhead_cost_p7z.ravel(),index=new_index_p7z)
    new_index_c0p7z = pd.MultiIndex.from_product([keys_c0,keys_p7,keys_z])
    overhead_wc_c0p7z = pd.Series(overhead_wc_c0p7z.ravel(),index=new_index_c0p7z)

    params['overheads_cost'] = overhead_cost_p7z.to_dict()
    params['overheads_wc'] = overhead_wc_c0p7z.to_dict()

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, overhead_cost_p7z, 'overheads', mask_season_p7z, z_pos=-1)

#################
#Min ROE        #
#################
def f1_min_roe():
    ##the default inputs for min roe are different for steady-state and stochastic version.
    ##but one SAV controls both inputs. So steady-state and stochastic can fairly be compared.
    if sinp.structuralsa['steady_state'] or np.count_nonzero(pinp.general['i_mask_z'])==1:
        min_roe = uinp.finance['minroe']
    else:
        min_roe = uinp.finance['minroe_dsp']
    return min_roe


#########################
# params & report vals  #
#########################
def f1_fin_params(params, r_vals):
    ##overheads
    overheads(params, r_vals)

    ##store params which are inputs
    params['prob_c1'] = uinp.price_variation['prob_c1'].to_dict()
    params['capital_limit'] = pinp.finance['capital_limit']

    ##store report
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_c1 = np.array(['c1_%d' % i for i in range(uinp.price_variation['len_c1'])])
    fun.f1_make_r_val(r_vals,keys_p7,'keys_p7')
    fun.f1_make_r_val(r_vals,keys_c0,'keys_c0')
    fun.f1_make_r_val(r_vals,keys_c1,'keys_c1')
    fun.f1_make_r_val(r_vals,uinp.price_variation['prob_c1'],'prob_c1')
    fun.f1_make_r_val(r_vals,uinp.finance['opportunity_cost_capital'],'opportunity_cost_capital')
    fun.f1_make_r_val(r_vals,uinp.finance['i_interest'],'interest_rate')



