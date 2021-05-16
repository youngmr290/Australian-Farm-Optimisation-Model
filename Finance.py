
"""
author: young

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

#. AFO is a bioeconomical model with the aim of optimising farm management. It is not a finance model.

Asset value
    Asset value is the value of all assest on a specific day of the year specified by the model administrator.
    The asset value is used to capture the opportunity cost of investing in farm assets including livestock,
    machinery and infrastructure (sheds, yards etc). To be selected an asset must return more than the
    interest cost on the asset which ensures the optimal solution does not include an asset that returns
    less than investing the same money in high interest savings.
    For livestock this ensures that the flock structure optimisation accounts for the opportunity cost
    of interest foregone from holding an animal for an extra year.

.. note:: To ensure accurate representation of asset opportunity cost the date when the assets are valued
    must be the date when the cashflow periods begin.

The cashflow operates in conjunction with the asset value in representing the opportunity cost of holding assets.
Livestock flock structure is the main 'decision' that is altered by the inclusion of an asset value. Without
interest if animals are sold early in the year there would not be an offsetting value that would
make early sale a ‘reasonable’ option. For example, selling the day after the asset is valued for the price
that the animal was valued should be an ‘equal’ outcome solution, but this will only occur if there is
interest ‘earned’ in the cashflow.

The interest rate for credit & debit are different for farmers ‘real money’ in the bank.
However, in the model equal debit and credit interest rates are used (this caon be changed by the user).
The reason equal interest rates are set as the default are:

#. Many farmers have a core debt, so the farm cash position is always negative even though
   their short term operating account may occasionally be positive. The differential interest
   rates are only justified if the farmer does not operate with a sweep facility to pay down
   core debt and then redraw when required later.
#. As discussed above the asset value and the cashflow operate together in the optimisation of
   flock structure. This implies that the interest rate for the cash flow should be the same as
   the discount rate for the asset value.


Minimum return on expenditure
    AFO tallies the total farm expenditure, adjusts it by a user defined return on expense factor and
    includes it in the objective to ensure the model achieves a minimum return on expenditure. The
    purpose of this is to represent farmer behaviour. The rate of MinROE is specified by the user and
    can be turned off.

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

##########################
#debit & credit interest #
##########################

#If it's compound interest, which it generally is, take the annual interest rate (r) and raise it to the reciprocal of 12 to get your monthly rate.
#Why? Because there are 12 months in a year, and compound interest means exponential growth. Taking an exponent accounts for this.
#Converting yeary compound r to some shorter period m, use the following formula:
#[(1 + r)^(1/m)] - 1

#convert pa interest into per cashflow period
def debit_interest():
    return (1 + uinp.finance['debit_interest']) ** (1 / len(sinp.general['cashflow_periods']))


def credit_interest():
    return (1 + uinp.finance['credit_interest']) ** (1 / len(sinp.general['cashflow_periods']))


#################
#overheads      #
#################
def overheads(params, r_vals):
    '''
    Calculate overhead costs in each cashflow period.

    Overheads are ongoing business expenses that are not directly attributed to creating a product
    or service. In AFO the user has the discretion to add, remove or alter the overheads that are
    including. Examples of overhead costs include; electricity, gas, shire rates, licenses,
    professional services, insurance and household expense.
    '''
    overheads=pinp.general['overheads'] 
    overheads = overheads.squeeze().sum()/ len(sinp.general['cashflow_periods'])
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



