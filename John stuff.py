# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 10:44:53 2020

@author: John
"""
#################################################################################################################
#################################################################################################################

# PastureFunctions

#################################################################################################################
#################################################################################################################

########################
# calculation of pgr   #
########################
''' calculate pgr in the calc_foo_start method without looping through lmu
Developed to do the calculation with numpy functions, to perhaps help with @jit implementation
Requires extra calculations at the bottom of the read_inputs_from_excel method and replace the loop code
Removes the requirement to predefine the pgr_daily array which is required in the loop method'''
## goes at end of read_inputs_from_excel
        # self.fxg_bi_oflt[0,...] = self.fxg_b_oflt[0,...]                          # bi is increment of b (used in the bool index version of the calculation of pgr)
        # self.fxg_bi_oflt[1,...] = self.fxg_b_oflt[1,...] - self.fxg_b_oflt[0,...]
        # self.fxg_bi_oflt[2,...] = self.fxg_b_oflt[2,...] - self.fxg_b_oflt[1,...]
        # self.fxg_ai_oflt[0,...] = self.fxg_a_oflt[0,...]                          # ai is increment of a (to be used in a cumulative sum)
        # self.fxg_ai_oflt[1,...] = self.fxg_a_oflt[1,...] - self.fxg_a_oflt[0,...]
        # self.fxg_ai_oflt[2,...] = self.fxg_a_oflt[2,...] - self.fxg_a_oflt[1,...]


            ## alternative approach (a1) within a loop on f. Could this be done with a loop on f either (because now there is a t loop too)
            ## for pgr using a boolean filter and cumulative sum. Requires the incremental array for a & b (ai & bi which is created when a & b are created in read_inputs_from_excel)
            # bool                = (self.fxg_foo[:,f,:]<foo_start[..., np.newaxis][:,f,:])         # create a boolean filter to apply to pgr = a + b FOO so that a & b don't have to be calculated in a loop on lmu
            # bool                = np.insert(bool,  0, True, axis=1)                               # insert a True at the start because all values are greater than 0 (ai[0] & bi[0] are relevant for all)
            # bool                = np.delete(bool, -1,       axis=1)                               # delete the last entry that is always False because no FOO is greater than the max FOO
            # pgr_daily           = sam_pgr * ( np.sum((self.fxg_ai_oflt[:,f,:,:] * bool),axis=0)   # daily pgr for the pasture based on start foo
            #                                  +np.sum((self.fxg_bi_oflt[:,f,:,:] * bool),axis=0)
            #                                  *               foo_start[f,:,:])                    # calculated from pgr = a + b foo, but using a cumulative approach to foo level


#################################################################################################################
#################################################################################################################

# General information

#################################################################################################################
#################################################################################################################

''' Broadcasting arrays
    When arrays are broadcast the aligning starts at the highest axis (dimension) a [2,5,6,3] array
    will broadcast with a [3], [6,3] or [5,6,3] array but not a [2] or a [2,5] array.
    The methods to convert a [2] aray to be broadcastable are:
    1. array.reshape[-1,1,1,1] where -1 means any levels not accounted for yet.
    Works best if there is only one dimension and -1 can be that dimension
    but could do array.reshape[2,5,1,1] if starting with a [2,5]
    2. array(:,np.newaxis,np.newaxis,np.newaxis,np.newaxis) adds 4 axes to [2]
    3. array(:,None,None,None,None) exact same as above because newaxis is a synonym for None
    (not sure if None requires np.None)

    Note: .reshape is quicker than np.newaxis so I have decided to use that to reshape 1D vectors and
    use np.newaxis if the original array has multiple axes.

    if it just changing a 1D vector eg[2] to 2D then can use
    4. np.c_[1D_vector] which converts it to a 2D column array [n,1]. If there are multiple axes it puts the 2 array side by side on the last axes.
    5. np.r_[1D_vector] which converts it to a 2D row array [1,n].    If multiple axes, side by side on the first axis
          np.r_ seems to be the default if a 1D_vector is passed as an 'array like' variable
          (eg a list or dataframe column) without any alteration.
    '''

''' Converting Pandas to DataFrame
    df.to_numpy() for a view or pd.to_numpy(copy=True) for a copy seems to be the recommended way (stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array)
    np.asarray(df) also seems to work
        Stackoverflow doesn't mention the asarray option
'''
