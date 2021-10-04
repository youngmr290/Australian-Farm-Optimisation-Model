
"""
author: young

To support the sporadic nature of farming income
finance is often drawn from the bank throughout the year to fund costly operations such as seeding. This
is also represented by working capital in AFO. A working capital constraint tracks the bank balance throughout the year
and ensures that the maximum overdraw is below a user specified limit. This ensure the model doesnt overdraw an unrealistic
level of capital from the bank.

The interest cost of working capital is calculated from when the expense is incurred through to when the income associated
with that cost is received. This has an impact on total objective function value but also ensures that
expenditure is only incurred if the return exceeds the cost of interest.

There is no representation of a starting cash balance. If it is included the model just selects the
highest amount because that earns the most interest. The model can get loan from bank if it needs
cash, so this does not affect model solution.

Tax is also not represented for several reasons:

#. There are several mechanisms by which farmers seek to lessen their tax liabilities. Not all
   are 'economically rational' and not all are easily represented in an LP model.

#. Many farmers nowadays invest in farm management deposits and income-averaging as a means
   of taxable income averaging and smoothing working capital borrowings. if the FMDs could be used
   'perfectly' then each year would have the same taxable profit. Thus, the optimal farm management
   is unaffected by the inclusion of tax.

#. AFO is a bioeconomic model with the aim of optimising farm management. It is not a finance model.

Asset value
    Asset value is the value of all assets on the first day of the cashflow year. It captures the opportunity
    cost of investing in farm assets including livestock, machinery and infrastructure (sheds, yards etc).
    The role of the asset value is to ensure that all assets that are selected have a return more than the
    interest cost, this ensures the optimal solution does not include assets that returns
    less than investing the same money in high interest savings (or to reduce core debt).
    This structure makes the static equilibrium model generate a result similar to a multi-period optimisation
    that accounts for interest cost of money.
    For livestock this ensures that the flock structure optimisation accounts for the opportunity cost
    of interest foregone from holding an animal for an extra year.

The cashflow operates in conjunction with the asset value in representing the opportunity cost of holding assets.
Livestock flock structure is the main 'decision' that is altered by the inclusion of an asset value. Without
interest if animals are sold early in the year there would not be an offsetting value that would
make early sale a ‘reasonable’ option. For example, selling the day after the asset is valued for the price
that the animal was valued should be an ‘equal’ outcome solution, but this only occurs if there is
interest ‘earned’ in the cashflow.

The interest rate for credit & debit are different for farmers ‘real money’ in the bank.
However, in the AFO the same interest rate is used to represent debit and credit.
The reasons are:

#. Many farmers often have a core debt, so the farm cash position is usually negative even though
   their short term operating account may occasionally be positive. The differential interest
   rates are only justified if the farmer does not operate with a sweep facility to pay down
   core debt and then redraw when required later.
#. As discussed above the asset value and the cashflow operate together in the optimisation of
   flock structure. This implies that the interest rate for the cash flow should be the same as
   the discount rate for the asset value.


Minimum return on expenditure
    AFO tallies the total farm expenditure, adjusts it by a user defined return on expense factor and
    includes it in the objective to ensure the model achieves a minimum return on expenditure. The
    purpose of this is to represent farmer behaviour. It can also be used in the static equilibrium
    version to 'fudge' the risk associated with seasonal variation and reduce the optimal stocking rate
    to better align with on-farm values.
    The rate of MinROE is specified by the user and can be turned off. Comparison of the model output
    with on-farm benchmarking has been used to calibrate the selected value.

"""
##python modules
import pandas as pd
import numpy as np

##AFO modules
import UniversalInputs as uinp
import StructuralInputs as sinp
import PropertyInputs as pinp
import Periods as per
import Functions as fun
import SeasonalFunctions as zfun

na = np.newaxis
#######################
# cashflow & interest #
#######################
def f_cashflow_allocation(date_incurred,p_dates_c0p7,peakdebt_date,z8mask_c0p7,enterprise=None):
    '''
    Allocates cashflow and wc to a cashflow period with an additional interest component.

    Cashflow allocation always has a length of 1. Meaning that cost is allocated based on the start date when it is
    incurred. Interest is calculated from this date until the end of the cashflow periods. The reason for not
    including a length is that cashflow for a give decision variable can not cross a season junction
    otherwise some seasons do not incur the cashflow.

    All inputs must be broadcastable.

    :param date_incurred: datetime64 date when cashflow is incurred
    :param enterprise: enterprise
    :param peak_debt_date_c0: datetime64 date peak debt occurs
    :param p_dates_c0p7: datetime64 cashflow dates with c0 and p7 in position 0 and 1 respectively. Other axis can be included after p7.
    '''

    ##inputs
    rate = uinp.finance['i_interest']
    keys_c0 = np.expand_dims(sinp.general['i_enterprises_c0'], tuple(range(1, p_dates_c0p7.ndim)))
    peakdebt_date = peakdebt_date + np.timedelta64(365,'D') * (p_dates_c0p7[:,0:1,...]>peakdebt_date[:,0:1,...]) # peak debt is after the start of the cashflow

    ##adjust yr of cashflow occurence so it occurs within the cashflow periods
    start_of_cash = p_dates_c0p7[:,0:1,...]
    end_of_cash = start_of_cash + np.timedelta64(364,'D') #use 364 because end date is the day before before the end otherwise can get item that starts on the last day of periods.
    add_yrs = np.ceil(np.maximum(0,(start_of_cash - date_incurred).astype('timedelta64[D]').astype(int) / 365))
    sub_yrs = np.ceil(np.maximum(0,(date_incurred - end_of_cash).astype('timedelta64[D]').astype(int) / 365))
    date_incurred = date_incurred + add_yrs * np.timedelta64(365, 'D') - sub_yrs * np.timedelta64(365, 'D')

    ##calc interest
    cashflow_incur_days = (end_of_cash - date_incurred).astype('timedelta64[D]').astype(int)
    wc_incur_days = (peakdebt_date - date_incurred).astype('timedelta64[D]').astype(int)
    cashflow_interest = (1 + rate / 365) ** cashflow_incur_days
    wc_interest = (1 + rate / 365) ** wc_incur_days

    ##allocate to cashflow period
    p_dates_p7c0 = np.swapaxes(p_dates_c0p7, 0, 1) #period axis need to be first for allocation function
    date_incurred_p7c0 = np.swapaxes(date_incurred, 0, 1) #period axis need to be first for allocation function
    # date_incurred = np.expand_dims(date_incurred, tuple(range(-p_dates_c0p7.ndim,-date_incurred.ndim)))
    date_incurred_shape = date_incurred_p7c0.shape
    p_dates_p7c0_shape = list(p_dates_p7c0.shape) #has to be a list because cant change tuple.
    p_dates_p7c0_shape[0] = p_dates_p7c0_shape[0] -1 #remove the last cashflow peirod because it is not a real period. It is just the end date.
    shape = np.maximum.reduce([date_incurred_shape, p_dates_p7c0_shape]) #create shape which has the max size, this is used for alloc array
    p7_alloc_p7c0 = fun.range_allocation_np(p_dates_p7c0, date_incurred_p7c0, opposite=True, shape=tuple(shape))
    p7_alloc_c0p7 = np.swapaxes(p7_alloc_p7c0, 0, 1) #get axis back into correct order

    ##add interest adjustment
    final_cashflow = cashflow_interest * p7_alloc_c0p7
    final_wc = wc_interest * p7_alloc_c0p7

    ##adjust for enterprise
    if enterprise is not None:
        final_cashflow = final_cashflow * (keys_c0==enterprise)

    return final_cashflow * z8mask_c0p7, final_wc * z8mask_c0p7


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
    ##cost allocation
    p_dates_c0p7z = per.f_cashflow_periods()
    overhead_length = 365 #overheads are incurred equally each day
    overhead_start_c0p7z = p_dates_c0p7z[:,0:1,:]
    keys_p7 = per.f_cashflow_periods(return_keys_p7=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    peakdebt_date_c0p7z = per.f_peak_debt_date()[:,na,na]
    p7_start_dates_c0p7z = p_dates_c0p7z[:,:-1,:]  # slice off the end date slice
    mask_cashflow_z8var_c0p7z = zfun.f_season_transfer_mask(p7_start_dates_c0p7z, z_pos=-1, mask=True)
    ###call allocation/interset function - needs to be numpy
    overhead_cost_allocation_c0p7z, overhead_wc_allocation_c0p7z = f_cashflow_allocation(overhead_start_c0p7z,
                                                                                  p_dates_c0p7z, peakdebt_date_c0p7z,
                                                                                  mask_cashflow_z8var_c0p7z)
    ###convert to df
    new_index_c0p7z = pd.MultiIndex.from_product([keys_c0,keys_p7,keys_z])
    overhead_cost_allocation_c0p7z = pd.Series(overhead_cost_allocation_c0p7z.ravel(),index=new_index_c0p7z)
    overhead_wc_allocation_c0p7z = pd.Series(overhead_wc_allocation_c0p7z.ravel(),index=new_index_c0p7z)

    ##cost
    overheads = pinp.general['i_overheads']
    overheads_c0_alloc_c0 = pinp.finance['i_fixed_cost_enterprise_allocation_c0']
    overheads = overheads.sum()
    overheads_c0 = overheads * overheads_c0_alloc_c0
    overheads_c0 = pd.Series(overheads_c0, index=keys_c0)
    overhead_cost_c0p7z = overhead_cost_allocation_c0p7z.mul(overheads_c0, level=0)
    overhead_wc_c0p7z = overhead_wc_allocation_c0p7z.mul(overheads_c0, level=0)

    params['overheads_cost'] = overhead_cost_c0p7z.to_dict()
    params['overheads_wc'] = overhead_wc_c0p7z.to_dict()
    r_vals['overheads'] = overhead_cost_c0p7z

#################
#Min ROE        #
#################
def f_min_roe():
    ##the default inputs for min roe are different for steady-state and stochastic version.
    ##but one SAV controls both inputs. So steady-state and stochastic can fairly be compared.
    if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z'])==1:
        min_roe = uinp.finance['minroe']
    else:
        min_roe = uinp.finance['minroe_dsp']
    return min_roe

###################
#Season transfer  #
###################
def f1_cashflow_z8z9_transfer(params):
    '''
    Create pyomo param whcih masks cashflow transfer within a given season.

    Seasons are masked out until the point in the year when they are identified. At the point of identification
    the parent season provides the transfer parameters to the child season. This transfering method ensures the
    model has the same management across seasons until they are identified. For example, if there are two seasons, a
    good and a bad, that are identified in spring. Both seasons must have the same management through the beginning of
    the year until spring (because the farmer doesnt know if they are having the good or bad year until spring).
    '''
    ##get param
    p7_start_dates_c0p7z = per.f_cashflow_periods()[:,:-1,:]  # slice off the end date slice
    mask_cashflow_provz8z9_c0p7z8z9 = zfun.f_season_transfer_mask(p7_start_dates_c0p7z, period_axis_pos=1, z_pos=-1)

    ##build param
    keys_p7 = per.f_cashflow_periods(return_keys_p7=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()

    arrays = [keys_c0, keys_p7, keys_z, keys_z]
    index_c0p7z8z9 = fun.cartesian_product_simple_transpose(arrays)
    tup_c0p7z8z9 = tuple(map(tuple,index_c0p7z8z9))

    # arrays = [keys_z, keys_z]
    # index_z8z9 = fun.cartesian_product_simple_transpose(arrays)
    # tup_z8z9 = tuple(map(tuple,index_z8z9))

    # params['p_childz_req_cashflow'] =dict(zip(tup_z8z9, mask_cashflow_reqz8z9_z8z9.ravel()*1))
    params['p_parentchildz_transfer_cashflow'] =dict(zip(tup_c0p7z8z9, mask_cashflow_provz8z9_c0p7z8z9.ravel()*1))

#################
# report vals   #
#################

def finance_rep(r_vals):
    keys_p7 = per.f_cashflow_periods(return_keys_p7=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    r_vals['keys_p7'] = keys_p7
    r_vals['keys_c0'] = keys_c0
    r_vals['opportunity_cost_capital'] = uinp.finance['opportunity_cost_capital']
    r_vals['interest_rate'] = uinp.finance['i_interest']



