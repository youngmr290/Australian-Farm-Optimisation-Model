'''
Generate price scenario scalars.

Price scalars have two main purposes

    #. To account for variation in prices across weather years.
    #. To account for price variation within a year that is important for examining farm risk.

Notwithstanding weather variation agricultural commodity price is a major source of farm risk. Price variation is
approximated in AFO using user defined number of price states scalars for meat, wool and grain.
The price states were determined from a multivariate price distribution so that correlation between commodities
was captured. To reduce model complexity and simplify future input calibration all meat classes (lamb, shipper,
mutton, etc) received the same meat price scalar, and the same for wool and grain. Furthermore, price variation
was not included for input costs because input costs tend to vary less and therefore the additional model size
was not justified. The resulting assumptions are that all animal classes are 100% correlated,
all wool microns are 100% correlated, all grains are 100% correlated and all input comodities have no variation
If these assumptions become limiting it is possible to add the extra detail in the price generation.

The c1 axis is averaged for both the asset constraint and the working capital. This saves space without loosing
much/any information.

Generation of discrete price states:

The need to form discrete approximations of a continuous distributions is a necessary requirement for developing a
discrete stochastic programming model of farm management responses to price and weather-year states, where the price
states are affected by marketing schemes.  By their nature discrete stochastic programming models cannot consider
all possible price states as described by continuous distributions.  Rather continuous variables such as price and
yield need to be approximated by discrete states.

In AFO the user inputs the expected price this is then scalled by a factor for each price scenario. Price scenario
scalars exist for meat, wool and grain and are generated from historical prices. A historical price series is
inputted for meat, wool and grain from there a multivariate log normal distribution is fitted to the data and
multiple price states are selected. The price at each point is compared to the average to determine the magnitude
of the scalar.

'''
import numpy as np
import pandas as pd

import PropertyInputs as pinp

##need to generate price senario scalars with a z axis. To start there will only be scalar for woo, meat and grain (all grains get the same)
len_z = len(pinp.general['i_mask_z'])
len_c1 = 1
keys_z = pinp.general['i_z_idx']
keys_c1 = np.array(['c1_%s' % i for i in range(len_c1)])

##c1 probability
prob_c1 = pd.Series(np.ones(len_c1)/len_c1, index=keys_c1)

##grain scalar
grain_price_scalar_c1z = np.ones((len_c1, len_z))
##animal sale scalar
meat_price_scalar_c1z = np.ones((len_c1, len_z))

##wool sale scalar
wool_price_scalar_c1z = np.ones((len_c1, len_z))

##convert to df - if the arrays ever become more than 2d possible it would be better to save them as pkl files.
## they are saved as excel file so the user can manually change the value or look at the arrays easily.
grain_price_scalar_c1z = pd.DataFrame(grain_price_scalar_c1z, index=keys_c1, columns=keys_z)
meat_price_scalar_c1z = pd.DataFrame(meat_price_scalar_c1z, index=keys_c1, columns=keys_z)
wool_price_scalar_c1z = pd.DataFrame(wool_price_scalar_c1z, index=keys_c1, columns=keys_z)


##write to xl - it would be good to have it with a named range  and have all the tables in the same sheet so that the read in process could be simplified.
writer = pd.ExcelWriter('PriceScenarios.xlsx',engine='xlsxwriter')
grain_price_scalar_c1z.to_excel(writer, sheet_name='grain')
meat_price_scalar_c1z.to_excel(writer, sheet_name='meat')
wool_price_scalar_c1z.to_excel(writer, sheet_name='wool')
prob_c1.to_excel(writer, sheet_name='prob')

##finish writing and save
writer.save()


