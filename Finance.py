
"""
author: young
#todo need to update this.
To capture debit and credit interest the year is split into bi-monthly cashflow periods. Activities with a
cost or income add to or remove from the cashflow balance. To support the sporadic nature of farming income
finance is often drawn from the bank throughout the year to fund costly operations such as seeding. This
is also represented in AFO however a user input exist which sets the maximum overdraw limit. At the end
of a period the positive or negative bank balance plus interest is transferred to the next period.

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

.. note:: To ensure accurate representation of asset opportunity cost the date when the assets are valued
    must be the date when the cashflow periods begin.

The cashflow operates in conjunction with the asset value in representing the opportunity cost of holding assets.
Livestock flock structure is the main 'decision' that is altered by the inclusion of an asset value. Without
interest if animals are sold early in the year there would not be an offsetting value that would
make early sale a ‘reasonable’ option. For example, selling the day after the asset is valued for the price
that the animal was valued should be an ‘equal’ outcome solution, but this only occurs if there is
interest ‘earned’ in the cashflow.

The interest rate for credit & debit are different for farmers ‘real money’ in the bank.
However, in the model very similar debit and credit interest rates are used (this can be changed by the user).
The reason equal interest rates are set as the default are:

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

'''
interest
'''


def f_cashflow_allocation(amount,start,enterprise,length=1):
    '''
    This function calculate the interest earned on cashflow item, tallies up the total cashflow including interest
    and tallies up the working capital from a cashflow item.

    This function is complicated by the fact that some cashflow items occur over a date range. This is treated such that
    an proportion of the cashflow is received each day.

    :param amount:
    :param start: date when cashflow starts
    :param enterprise:
    :param peak_debt_date_c0:
    :param p_dates_c0p7:
    :param rate:
    :param length: time over which the cashflow is incurred
    :return:
    '''

    pandas = isinstance(amount,pd.DataFrame) or isinstance(amount,pd.Series)
    p_dates_c0p7z = f_cashflow_periods(pandas)

    ##inputs
    date_peakdebt_stock = np.average(pinp.sheep['i_date_peakdebt_stock_i'][i_mask_i])
    date_peakdebt_crop = np.array([pinp.crop['i_date_peakdebt_crop']])
    peakdebt_date_c0 = np.concatenate([date_peakdebt_stock,date_peakdebt_crop])

    rate = uinp.finance['interest']

    ##adjust yr of cashflow occurence #todo check this is working
    add_yrs = np.ceil(np.maximum(0,(p_dates_c0p7z[:,0] - start) / 365))
    sub_yrs = np.ceil(np.maximum(0,(start - p_dates_c0p7z[:,-1]) / 365))
    start = start + add_yrs - sub_yrs

    ##create final array
    if pandas:
        new_index = pd.MultiIndex.from_product([keys_c0,keys_p7,amount.index])
        cols = amount.columns
        amount = amount.values
        final_cashflow = np.zeros((len(new_index),len(cols)))
        final_wc = np.zeros((len(new_index),len(cols)))
    else:
        final_cashflow = np.zeros(p_dates_c0p7.shape + amount.shape)
        final_wc = np.zeros(p_dates_c0p7.shape + amount.shape)

    principal_end = 0
    daily_payment = amount / length

    ##start and end dates for the cashflow periods
    for p in len_p7:
        ##cashflow period dates
        date_start_c0z = p_dates_c0p7z[...,p]
        date_end_c0z = p_dates_c0p7z[...,p + 1]

        ##princiapal at the begining of the p7 period (amount of cash at the begining of the period, inc interest from previous periods)
        principal_start = total_principal_end

        #
        # cashflow - principal and interest
        #
        ##calculate interest druing incur period (incur period is the time between the cashflow start and end)
        ###end date of incur period
        incur_end = start + length
        ###length of incur period (days)
        incur_days = np.minimum(date_end_c0,incur_end) - np.maximum(date_start_c0,start)
        ###interest on starting balance during the incur period - using formula: A = P (1 + r/n)**(t)
        principal_interest = principal_start * ((1 + rate / 365) ** incur_days - 1)
        ###daily payments plus interest on daily payments during the incur period - Using formula: Payment × ( ( ( (1 + r/n)^(t) ) - 1 ) / (r/n) )
        daily_interest = daily_payment * (((1 + rate / 365) ** incur_days - 1) / (rate / 365))

        ##calc interest over the period after payment incur has finished
        ###days from the end of incur to the end of the period
        post_incur_days = np.maximum(date_start_c0,incur_end) - date_end_c0
        ###balance at start of non incur period
        post_incurum_start_balance = principal_start + principal_interest + daily_interest
        ###interest on balance after the incur period - using formula: A = P (1 + r/n)**(t)
        post_incurum_interest = post_incurum_start_balance * ((1 + rate / 365) ** post_incur_days - 1)

        ##cash at the end of the period
        cashflow_n_interest = post_incurum_start_balance + post_incurum_interest
        total_principal_end = principal_start + post_incurum_start_balance + post_incurum_interest

        #
        # working capital constraint except that income or expense incurred after the peak debt date is excluded
        #
        ##calculate interest druing incur period (incur period is the time between the cashflow start and end)
        ###end date of incur period
        incur_end = start + length
        ###length of incur period (days)
        incur_days = np.minimum(date_end_c0,np.minimum(incur_end,peak_debt_date_c0)) - np.maximum(date_start_c0,start)
        ###interest on starting balance during the incur period - using formula: A = P (1 + r/n)**(t)
        wc_principal_interest = principal_start * ((1 + rate / 365) ** incur_days - 1)
        ###daily payments plus interest on daily payments during the incur period - Using formula: Payment × ( ( ( (1 + r/n)^(t) ) - 1 ) / (r/n) )
        wc_daily_interest = daily_payment * (((1 + rate / 365) ** incur_days - 1) / (rate / 365))

        ##calc interest over the period after payment incur has finished
        ###days from the end of incur to the end of the period
        post_incur_days = np.maximum(date_start_c0,incur_end) - np.minimum(date_end_c0,peak_debt_date_c0)
        ###balance at start of non incur period
        wc_post_incurum_start_balance = principal_start + wc_principal_interest + wc_daily_interest
        ###interest on balance after the incur period - using formula: A = P (1 + r/n)**(t)
        wc_post_incurum_interest = wc_post_incurum_start_balance * ((1 + rate / 365) ** post_incur_days - 1)

        ##cash at the end of the period
        wc = wc_post_incurum_start_balance + wc_post_incurum_interest

        ##assign to final array
        final_cashflow[:,p,...] = cashflow_n_interest
        final_wc[:,p,...] = wc

    if pandas:
        final_cashflow = pd.DataFrame(final_cashflow,index=new_index,columns=cols)
    return final_cashflow,final_wc


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
    length = 365 #overheads are incurred equally each day
    start_cashflow_stock = np.average(pinp.sheep['i_date_cashflow_stock_i'][i_mask_i])
    start_cashflow_crop = np.array([pinp.crop['i_date_cashflow_crop']])

    overheads = pinp.general['i_overheads']
    overheads = overheads.sum()
    overheads_stock = overheads/2
    overheads_crop = overheads/2

    f_cashflow_allocation(overheads_stock,start_cashflow_stock,'stk', length)
    f_cashflow_allocation(overheads_crop,start_cashflow_crop,'crp', length)
#need to add c0
    overheads = dict.fromkeys(sinp.general['cashflow_periods'], overheads)
    params['overheads'] = overheads
    r_vals['overheads'] = pd.Series(overheads)

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



#################
# report vals   #
#################

def finance_rep(r_vals):
    keys_c = sinp.general['cashflow_periods']
    r_vals['keys_c'] = keys_c
    r_vals['opportunity_cost_capital'] = uinp.finance['opportunity_cost_capital']
    r_vals['interest_rate'] = debit_interest()



