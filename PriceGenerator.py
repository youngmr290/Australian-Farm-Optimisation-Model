'''
Generate price scenario scalars.
'''
import numpy as np
import pandas as pd

import PropertyInputs as pinp
import SeasonalFunctions as zfun

##need to generate price senario scalars with a z axis. To start there will only be scalar for woo, meat and grain (all grains get the same)
len_z = len(pinp.general['i_mask_z'])
len_c1 = 5
keys_z = pinp.general['i_z_idx']
keys_c1 = np.array(['c1_%s' % i for i in range(len_c1)])
grain_price_scalar_c1z = np.ones((len_c1, len_z))

##convert to df - if the arrays ever become more than 2d possible it would be better to save them as pkl files.
## they are saved as excel file so the user can manually change the value or look at the arrays easily.
grain_price_scalar_c1z = pd.DataFrame(grain_price_scalar_c1z, index=keys_c1, columns=keys_z)

writer = pd.ExcelWriter('PriceScenarios.xlsx',engine='xlsxwriter')
grain_price_scalar_c1z.to_excel(writer, sheet_name='grain')

##finish writing and save
writer.save()
