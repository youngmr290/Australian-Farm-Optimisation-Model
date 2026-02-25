"""
Functions used across the model (multi module functions). These functions don't import other AFO modules
(hence don't initialise inputs inside the function). All function parameters are passed in when calling the function.

@author: young
"""
import pandas as pd
import numpy as np
import pickle as pkl
from dateutil import relativedelta as rdelta
import os.path
import pyomo.environ as pe
import copy

#this module shouldn't import other AFO modules
from . import Exceptions as exc #can import exceptions because exceptions imports no modules


na = np.newaxis

def f_convert_to_inf(input):
    input=input.astype('object') #have to convert to object so that when the stuff below is assigned it is not assigned as a string
    ##convert -- to -inf
    mask = input=='--'
    input[mask]=-np.inf
    ##convert ++ to inf
    mask = input=='++'
    input[mask]=np.inf
    ##convert 'True' to True (string to bool) - because array is read in as string
    mask = input=='True'
    input[mask]=True
    ##convert 'False' to False (string to bool) - because array is read in as string
    mask = input=='False'
    input[mask]=False
    return input.astype('float')

###########################
#general functions        #
###########################

#this is the fastest function for building cartesian products. Doesn't make much diff for small ones but up to 50% faster for big ones
def cartesian_product_simple_transpose(arrays):
    la = len(arrays)
    try:
        dtype = np.result_type(*arrays)
        arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    except TypeError:
        arr = np.empty([la] + [len(a) for a in arrays], dtype='U25')
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T

def searchsort_multiple_dim(a, v, axis_a0, axis_v0, axis_a1=None, axis_v1=None, side='left'):
    '''
    Find the indices into a sorted array 'a' such that, if the corresponding elements in 'v' were inserted before the indices, the order of 'a' would be preserved.
    It does this iteratively down the specified axis (therefore the specified axis must be present in both 'a' and 'v'

    Parameters:
        a: 3-D array_like
        Input array. Must be sorted in ascending order.

        v: array_like
        Values to insert into a.

        axis_a0: int
        The position of axis to iterate along. a1 & v1 axis must be same length
        axis_a1: int
        The position of axis to iterate along. a1 & v1 axis must be same length

    '''
    ##check a0 and v0 are the same length
    if a.shape[axis_a0] != v.shape[axis_v0]:
        raise ValueError('axis a0 and axis v0 are not the same length. This indicates an error.')
    if axis_a1 is not None:
        if a.shape[axis_a1] != v.shape[axis_v1]:
            raise ValueError('axis a1 and axis v1 are not the same length. This indicates an error.')

    ##do search sorted
    final = np.zeros_like(v).astype(int)
    slc_a = [slice(None)] * len(a.shape)
    slc_v = [slice(None)] * len(v.shape)
    for i in range(a.shape[axis_a0]):
        if axis_a1 is not None:
            for j in range(a.shape[axis_a1]):
                slc_a[axis_a0] = slice(i, i+1)
                slc_a[axis_a1] = slice(j, j+1)
                slc_v[axis_v0] = slice(i, i+1)
                slc_v[axis_v1] = slice(j, j+1)
                final[tuple(slc_v)] = np.searchsorted(np.squeeze(a[tuple(slc_a)]), v[tuple(slc_v)], side)
        else:
            slc_a[axis_a0] = slice(i, i+1)
            slc_v[axis_v0] = slice(i, i+1)
            final[tuple(slc_v)] = np.searchsorted(np.squeeze(a[tuple(slc_a)]), v[tuple(slc_v)], side)
    return final

#print(timeit.timeit(phases2,number=100)/100)

def f1_searchsorted_looped(a, v, axis, side='left'):
    """
    Apply np.searchsorted along the given axis of multidim arrays.
    Done by flattening the other dimensions and looping along the flattened axis.

    a and v must be broadcast-compatible except along `axis`.
    a must be sorted along 'axis'.

    Parameters (See np.searchsorted)
    ----------
    a : ndarray  Array to be searched (must be sorted along `axis`)
    v : ndarray  Values whose insertion points are sought
    axis : int  Axis along which to search
    side : {'left', 'right'}, optional

    Returns
    -------
    ndarray of int: Same shape as v, containing insertion indices
    """
    if a.ndim == 0 or v.ndim == 0:
        raise ValueError("Inputs must be at least 1-D")

    # Normalize axes to handle negative indices and test dimensions
    axis = a.ndim + axis if axis < 0 else axis
    if not (0 <= axis < a.ndim):
        raise ValueError(f"axis {axis} out of bounds for array of dimension {a.ndim}")

    # Move the search axis to the end → much easier broadcasting & indexing
    a_moved = np.moveaxis(a, axis, -1)  # shape:  ..., M
    v_moved = np.moveaxis(v, axis, -1)  # shape:  ..., K   (K may differ from M)

    # Make sure leading dimensions match (broadcasting)
    if a_moved.shape[:-1] != v_moved.shape[:-1]:
        try:
            v_moved = np.broadcast_to(v_moved, (*a_moved.shape[:-1], v_moved.shape[-1]))
        except ValueError:
            raise ValueError(f"Leading dimensions of v ({v_moved.shape[:-1]}) "
                             f"cannot be broadcast to those of a ({a_moved.shape[:-1]})")

    # Now flatten all dimensions except the last one
    a_flat = a_moved.reshape(-1, a_moved.shape[-1])  # (N_slices, M)
    v_flat = v_moved.reshape(-1, v_moved.shape[-1])  # (N_slices, K)

    # Output container
    result_flat = np.empty(v_flat.shape, dtype=np.int64)

    # Loop over all slices
    for i in range(a_flat.shape[0]):
        result_flat[i] = np.searchsorted(a_flat[i], v_flat[i], side=side)

    # Reshape back and move axis to original position
    result = result_flat.reshape(v_moved.shape)

    if axis != -1:
        result = np.moveaxis(result, -1, axis)

    return result


def f1_unique_count(a, axes, weights=None, threshold=0.0):
    """
    Count unique values along given axes.
    Values whose relative weight < threshold are excluded.

    Parameters
    ----------
    a : ndarray (float)
    axes : int or tuple of int
        Axes to collapse (result size becomes 1).
    weights : ndarray or None
        Broadcastable to a. If None, equal weights assumed.
    threshold : float
        Mask out entries with relative weight < threshold.

    Returns
    -------
    counts : ndarray
        Same shape as a but with collapsed axes size = 1.
    """
    a = np.asarray(a, dtype=float)

    if isinstance(axes, int):
        axes = (axes,)
    axes = tuple(ax % a.ndim for ax in axes)

    if weights is None:
        weights = np.ones_like(a, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    # relative weights along reduction axes
    denom = np.sum(weights, axis=axes, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_w = np.where(denom != 0, weights / denom, 0.0)

    # mask low-weight values
    a = np.where(rel_w < threshold, np.nan, a)

    # move axes to front
    moved = np.moveaxis(a, axes, range(len(axes)))

    front = np.prod(moved.shape[:len(axes)])
    rest_shape = moved.shape[len(axes):]

    flat = moved.reshape(front, -1)

    counts = np.empty(flat.shape[1], dtype=int)

    for i in range(flat.shape[1]):
        col = flat[:, i]
        col = col[~np.isnan(col)]  # ignore masked values
        counts[i] = np.unique(col).size

    counts = counts.reshape(rest_shape)

    # expand reduced axes back
    for ax in sorted(axes):
        counts = np.expand_dims(counts, axis=ax)

    return counts


def f1_percentile_weighted(values, weights, axes):
    """
    Compute weighted percentile rank within slices defined by the `axes` tuple.
    Weighted percentile rank definition:
        (cumulative of sorted weights - individual weight) / total of weights * 100

    Parameters
    ----------
    values : array_like: Input array to rank
    weights : array_like: Weights corresponding to each value (same shape as values)
    axes : int or tuple of ints: Axis or axes along which to compute percentile ranks. These are the axes being collapsed plus w

    Returns
    -------
    percentile : ndarray[float]
        Same shape as values.
        ~100 is the highest value and ~0 is the lowest value in the packed axes, based on average weighted position.
        Percentile is the centre of the weighting for the animal, therefore not 0 to 100.
    """
    # Normalize axes to handle negative indices
    axes = tuple(a % values.ndim for a in np.atleast_1d(axes))
    k = len(axes)

    # Move packed axes to end
    values_moved = np.moveaxis(values, axes, range(values.ndim - k, values.ndim))
    weights_moved = np.moveaxis(weights, axes, range(weights.ndim - k, weights.ndim))

    rest_shape = values_moved.shape[:-k]
    packed_shape = values_moved.shape[-k:]
    n = int(np.prod(packed_shape))

    # Flatten packed dimensions
    values_flat = values_moved.reshape(*rest_shape, n)
    weights_flat = weights_moved.reshape(*rest_shape, n)

    # # argsort values along packed axis
    # order = np.argsort(values_flat, axis=-1)

    #lexsort on both criteria, values (weight of the animals) ascending and separate ties based on weights (number of animals)
    ##The tie breaker is largest groups first for lighter animals (< median weight) and largest groups last for the heavier animals
    ##This means that the largest groups will be at the lower and higher end of the sort on values
    median_v = np.median(values_flat)    #note: this is the median over all the groups, not just within the tuple of axes
    direction = np.where(values_flat < median_v, -1, 1)
    tie_breaker = direction * weights_flat

    # Now sort by (weight ascending, tie_breaker ascending)
    order = np.lexsort((tie_breaker, values_flat))

    # reorder weights
    weights_sorted = np.take_along_axis(weights_flat, order, axis=-1)

    # cumulative weights
    weights_cum = np.cumsum(weights_sorted, axis=-1)

    # total weights (with keepdims for broadcasting). Note: same as np.sum(weights, axis=-1, keepdims) but more efficient
    weights_total = weights_cum[..., -1:]

    # compute percentile in sorted order
    # Handles edge case where only one valid entry exists and returns 50
    percentile_sorted = (weights_cum - weights_sorted/2) / weights_total * 100.0

    # unsort back to original order
    inverse_order = np.argsort(order, axis=-1)
    percentile_flat = np.take_along_axis(percentile_sorted, inverse_order, axis=-1)

    # Reshape and Move axes back to original positions
    percentile = percentile_flat.reshape(*rest_shape, *packed_shape)
    percentile = np.moveaxis(percentile, range(values.ndim - k, values.ndim), axes)

    return percentile
#
def f_expand(array, left_pos=0, swap=False, ax1=0, ax2=1, right_pos=0, left_pos2=0, right_pos2=0
                     , left_pos3=0, right_pos3=0, condition = None, axis = 0, swap2=False, ax1_2=1, ax2_2=2,
                     condition2=None, axis2=0, condition3=None, axis3=0, left_pos4=0, right_pos4=0, move=False, source=0, dest=1):
    '''
    *note: if adding two sets of new axis add from right to left (then the pos variables align)
    *note: mask applied last (after expanding and reshaping)

    Parameters
    ----------
    array : array
        parameter array - input from excel.
    left_pos : int
        position of axis to the left of where the new axis will be added.
    swap : boolean, optional
        do you want to swap the first two axis?. The default is False.
    ax1, ax2 : axes to swap. Position in the input array because this is carried out prior to expanding
    right_pos : int, optional
        the position of the axis to the right of the singleton axis being added. The default is -1, for when the axis to the right is g?.
    left_pos2 : int
        position of axis to the left of where the new axis will be added.
    right_pos2 : int, optional
        the position of the axis to the right of the singleton axis being added. The default is -1, for when the axis to the right is g?.
    condition: boolean, optional
        mask used to slice given axis.
    axis: int, optional
        axis to apply mask to.

    Returns
    -------
    expands, swaps axis if required and apply a mask to a given axis if required.
    '''
    ##convert int to 1d array if required
    if type(array) == int:
        array = np.array([array])
    ##swap axis if necessary
    if swap:
        array = np.swapaxes(array, ax1, ax2)
    ##swap axis if necessary
    if swap2:
        array = np.swapaxes(array, ax1_2, ax2_2)
    ##move axis if necessary
    if move:
        array = np.moveaxis(array, source=source, destination=dest)
    ##get axis into correct position 1
    if left_pos != 0:
        extra_axes = tuple(range((left_pos + 1), right_pos))
    else: extra_axes = ()
    array = np.expand_dims(array, axis = extra_axes)
    ##get axis into correct position 2 (some arrays need singleton axis added in multiple places ie separated by a used axis)
    if left_pos2 != 0:
        extra_axes = tuple(range((left_pos2 + 1), right_pos2))
    else: extra_axes = ()
    array = np.expand_dims(array, axis = extra_axes)
    ##get axis into correct position 3 (some arrays need singleton axis added in multiple places ie separated by a used axis)
    if left_pos3 != 0:
        extra_axes = tuple(range((left_pos3 + 1), right_pos3))
    else: extra_axes = ()
    array = np.expand_dims(array, axis = extra_axes)
    ##get axis into correct position 4 (some arrays need singleton axis added in multiple places ie separated by a used axis)
    if left_pos4 != 0:
        extra_axes = tuple(range((left_pos4 + 1), right_pos4))
    else: extra_axes = ()
    array = np.expand_dims(array, axis = extra_axes)
    ##apply mask if required
    if condition is not None: #see if condition exists
        if type(condition) == bool: #check if array or single value - note array of T & F is not type bool (it is array)
            condition= np.asarray([condition]) #convert to numpy if it is singular input
            array = np.compress(condition, array, axis)
        else:
            array = np.compress(condition, array, axis)
    ##apply mask if required
    if condition2 is not None: #see if condition exists
        if type(condition2) == bool: #check if array or single value - note array of T & F is not type bool (it is array)
            condition2= np.asarray([condition2]) #convert to numpy if it is singular input
            array = np.compress(condition2, array, axis2)
        else:
            array = np.compress(condition2, array, axis2)
    ##apply mask if required
    if condition3 is not None: #see if condition exists
        if type(condition3) == bool: #check if array or single value - note array of T & F is not type bool (it is array)
            condition3= np.asarray([condition3]) #convert to numpy if it is singular input
            array = np.compress(condition3, array, axis3)
        else:
            array = np.compress(condition3, array, axis3)
    return array

def f_update(existing_value, new_value, mask_for_new):
    '''
    Parameters
    ----------
    existing_value : numpy array or float or int
        values you want when mask = false.
    new_value : numpy array or float or int
        values you want when mask = true.
    mask_for_new : boolean mask
        boolean mask for the final axis of the array (typically the g axis).

    Returns
    -------
    Numpy array
        returns a combination of the two input arrays determined by the mask. Note: multiplying by true return the original number and multiplying by false results in 0.

    '''
    if isinstance(existing_value, pd.DataFrame) or isinstance(new_value, pd.DataFrame):
        print('using pandas in f_update. This should be changed. use .values to temporarily convert to numpy.')
    ##dtype for output (primarily needed for pp when int32 and float32 create float64 which we don't want)
    ##if the new value is an object (e.g. contains '-') then we want to return the original dtype otherwise return the biggest dtype

    if isinstance(new_value,np.ndarray) and isinstance(existing_value,np.ndarray):
        if new_value.dtype == object:
            if np.any(new_value == '-'):  # needs to be an object to perform elementwise comparison
                new_value[new_value == '-'] = 0
            is_float = np.any(np.mod(new_value, 1)!=0)
            ###incase int array is existing but sav has floats
            if is_float:
                if existing_value.dtype==int: #need this because max(int,float) returns int.
                    dtype = new_value.astype('float32').dtype #if existing value is int but new value has floats returns float dtype.
                else:
                    dtype = max(existing_value.dtype, new_value.astype('float32').dtype) #keeps it as float64 if existing value is float64
            else:
                dtype = existing_value.dtype
        else:
            #few steps required because int>float
            if (existing_value.dtype==int and new_value.dtype==int) or (existing_value.dtype==float and new_value.dtype==float):
                dtype = max(existing_value.dtype, new_value.dtype)
            elif existing_value.dtype==float:
                dtype = existing_value.dtype
            elif new_value.dtype==float:
                dtype = new_value.dtype
            else:
                dtype = existing_value.dtype #to handle bool
    elif isinstance(new_value,np.ndarray):
        dtype = new_value.dtype
    elif isinstance(existing_value,np.ndarray):
        dtype = existing_value.dtype
    elif isinstance(mask_for_new,np.ndarray):
        pass #if both values are int/float and mask is numpy then just ignore dtype
    elif type(existing_value)==type(new_value):
        dtype = type(existing_value)
    elif isinstance(new_value, str): #if it is '-' (used in exp) then retain the original dtype
        dtype = type(existing_value)


    ##convert '-' to 0 (because '-' * False == '' which causes and error when you add to existing value)
    ##need a try and except in case the new value is not a numpy array (ie it is a single value)
    try:
        if new_value.dtype==object and np.any(new_value=='-'): #needs to be an object to perform elementwise comparison
                new_value[new_value=='-'] = 0
                new_value = new_value.astype(float) #need to convert to number because if str it chucks error below
    except AttributeError:
        if new_value=='-':
            new_value = 0
    updated = np.where(mask_for_new, new_value, existing_value)
    # if result is a 0-D array, return a Python scalar
    if isinstance(updated, np.ndarray) and updated.shape == ():
        updated = updated.item()

    ##convert back to original dtype because adding float32 and int32 returns float64. And sometimes we don't want this e.g. postprocessing
    ###use try except because sometimes a single int is update e.g. in the first iteration on generator. this causes error because only numpy arrays have .dtype.
    try:
        if isinstance(updated, np.ndarray):
            updated = updated.astype(dtype)
        else:
            ###used for core python dtype e.g. floats/int/str
            updated = dtype(updated)
    except TypeError:
        pass
    except UnboundLocalError: #dtype not defined (i.e. if both values are int/float and mask is numpy then just ignore dtype)
        pass
    ###error check
    try:
        if updated.dtype == object:
            print('dtype error in f_update (type object is being returned)') #will give warning if ever returning a numpy object
    except: pass

    return updated

def f_weighted_average(array, weights, axis, keepdims=False, non_zero=False, den_weights=1, den_assoc=None, assoc_axis=0):
    '''
    Calculates weighted average (similar to np.average however this will handle:
        if the sum of the weights is 0 (np.average doesn't handle this)
        keeping the axis (using the keepdims argument)
    'non-zero' handles how the average is calculated
    Note: if non-zero is false then when sum weights = 0 the numbers being averaged also = 0 (so can divide by 1 instead of 0)
    The function is also called from the reporting module with den_weights. den_weights can be 0, in which case 'non-zero' handles how the average is calculated
    axis averaged along can be retained - default it is dropped.


    :param array:
    :param weights:
    :param axis:
    :param keepdims:
    :param non_zero: how to handle a weight of 0. True returns the numerator, False (default) returns 0
    :param den_weights: array: broadcastable to weights. This is used to weight the denominator (used in reporting)
    :return:
    '''
    if non_zero:
        ##for some situations (production) if numbers are 0 we don't want to return 0 we want to return the original value
        weights=f_update(weights,1,np.all(weights==0, axis=axis, keepdims=True))
    weighted_array = np.sum(array * weights, axis=axis, keepdims=keepdims)
    ##denom
    weights = weights * den_weights
    if den_assoc is not None:
        weights = np.take_along_axis(weights, den_assoc, axis=assoc_axis)
    weights = np.broadcast_to(np.sum(weights, axis=axis, keepdims=keepdims), weighted_array.shape)
    ##take average
    averaged_array = np.zeros_like(weighted_array)
    mask = weights!=0
    averaged_array[mask] = weighted_array[mask] / weights[mask]
    return averaged_array


def f_divide_float(numerator, denominator):
    '''this is the version when dividing singel values. Use f_divide() below for numpy.'''
    return numerator / denominator if denominator else 0


def f_divide(numerator, denominator, dtype='float64', option=0):
    '''
    Elementwise divides two arrays.
    If the denominator = 0 then return value depends on 'option'
     option == 0 then return 0
     option == 1 then return 1 if the numerator is also 0

     option == 1 will also return 1 if both the numerator and denominator are np.inf

    '''
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.zeros(numerator.shape, dtype=dtype) #make it a float in case the numerator is int
    ##use ~np.isclose to capture when the denominator is 0 within rounding tolerances
    mask = ~np.isclose(denominator.astype(float), 0) #astype float to handle timedeltas. timedelta / timedelta is a float so the final product needs to be a float anyway
    result[mask] = numerator[mask]/denominator[mask]

    ##If option is 1 then return 1 if the numerator and the denominator are the same (both 0 or both inf)
    #todo if useful sign could be included and np.inf / (-np.inf) could calculate to -1
    if option == 1:
        mask = np.isclose(denominator, numerator)
        result[mask] = 1

    ##If option is 2 then return the numerator if the denominator is 0
    if option == 2:
        mask = np.isclose(denominator.astype(float), 0)
        result[mask] = numerator[mask]

    return result

def f_bilinear_interpolate(im, x_im, y_im, x, y):
    ##get the index of x and y within the x_im and y_im arrays
    x= np.interp(x, x_im, np.arange(len(x_im)))
    y= np.interp(y, y_im, np.arange(len(y_im)))
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def np_extrap(x, xp, yp):
    ## np.interp function with linear extrapolation if x is beyond the input date (xp)
    ### from https://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range"""
    x = np.array(x) #convert x to array so that it can be masked
    y = np.array(np.interp(x, xp, yp))  #convert y to array so that it can be masked (required if x is a scalar)
    ##use a mask to adjust values if extrapolating x below the lowest input value in xp
    y[x < xp[0]] = yp[0] + (x[x<xp[0]]-xp[0]) * (yp[0]-yp[1]) / (xp[0]-xp[1])
    ##use a mask to adjust values if extrapolating x above the highest input value in xp
    y[x > xp[-1]]= yp[-1] + (x[x>xp[-1]]-xp[-1])*(yp[-1]-yp[-2])/(xp[-1]-xp[-2])
    return y

def f_norm_cdf(x, mu, cv=0.2, sd=None):
    '''
    ## returns the probability of the value being less than or equal to x
    ## based on a normal distribution with mean mu and either a
    ## coefficient of variation cv (default 20%) or standard deviation sd.
    ## If provided the sd is used.
    ## Calculated using an approximation of the normal probability function
    '''

    ##sd - standard deviation - maximum to stop div0 errors in next step.
    if sd is None:
        sd = mu * cv
    ##standardise x. f_divide in case SD is 0 (either mu is 0 or CV is 0)
    xstd = f_divide(x - mu,  sd)
    ##probability (<=x)
    prob = f_back_transform(358 / 23 * xstd - 111 * np.arctan(37 / 294 * xstd))
#    prob = 1 / (np.exp(-358 / 23 * xstd + 111 * np.arctan(37 / 294 * xstd)) + 1)
    return prob


def f_npv_nd(cashflows, discount_rate, axis=0):
    """
    Calculate the Net Present Value (NPV) along a specified axis for an N-dimensional array.

    Parameters:
        cashflows (np.ndarray): N-dimensional array of cashflows.
        discount_rate (float): Discount rate (e.g., 0.05 for 5%).
        axis (int): Axis along which time varies (default is 0).

    Returns:
        np.ndarray: Array of NPVs with the time axis reduced.
    """
    cashflows = np.asarray(cashflows)
    num_periods = cashflows.shape[axis]

    # Create discount factors
    years = np.arange(num_periods)
    discount_factors = 1 / (1 + discount_rate) ** years

    # Reshape discount_factors to broadcast along the correct axis
    shape = [1] * cashflows.ndim
    shape[axis] = num_periods
    discount_factors = discount_factors.reshape(shape)

    # Apply discounting and sum along the specified axis
    discounted = cashflows * discount_factors
    return np.sum(discounted, axis=axis)


def f_distribution7(mean, sd=None, cv=None):
    '''
    ##create a distribution around the mean for a variable that can be applied in any non-linear relationships
    ##Create 7 intervals with equal probability
    ## Equal probability allows the non-linear result to be averaged with equal weighting
    '''

    if sd is None:
        sd = cv * mean
    ## The distribution of standardised x based on the mid point of 7 intervals of 14.3%
    dist7_p1 = np.array([-1.535, -0.82, -0.375, 0, 0.375, 0.82, 1.535])
    ## Apply the distribution to the mean using the std deviation
    var_p1 = mean[..., na] + sd[..., na] * dist7_p1
    return var_p1

def f_find_closest(A, target):
    ##info here: https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def f_reduce_skipfew(ufunc, foo, preserveAxis=None):
    '''performs function on each axis except the axis that a specified as preserveAxis'''
    r = np.arange(foo.ndim)
    if preserveAxis is not None:
        preserveAxis = tuple(np.delete(r, preserveAxis))
    return ufunc(foo, axis=preserveAxis)

##check if two param dicts are the same.
def findDiff(d1, d2):
    a=False
    ##check if all the keys in d2 exist in d1
    for k in d2:
        if (k not in d1):  # check if the key in current params is in previous params dict.
            # print('DIFFERENT')
            a = True
            return a

    ##checks if all the keys in d1 exist in d2. if so it check the value stored is the same
    for k in d1:
        if a != True: #this stops it looping through the rest of the keys once it finds a difference
            if (k not in d2): #check if the key in current params is in previous params dict.
                # print('DIFFERENT')
                a = True
                return a
            else:
                if type(d1[k]) is dict and type(d2[k]) is dict: #need to check both are dicts in case extra level was added.
                    # print('going level deeper',k)
                    a=findDiff(d1[k],d2[k])
                    # print(k,a)
                else:
                    try: #have to try both ways because sometimes param is array and other times it is scalar
                        if np.any(d1[k] != d2[k]): #if keys are the same, check if the values are the same
                            # print('DIFFERENT',k)
                            a=True
                            return a #only return if true
                    except ValueError: #if the array is a different length then we get value error
                        a = True
                        return a  # only return if true
                    except TypeError:
                        if d1[k] != d2[k]: #if keys are the same, check if the values are the same
                            a=True
                            return a #only return if true
    return a

def f_clean_dict(d):
    '''Replace None values with 0 in a dict.'''
    for k in d:
        if type(d[k]) is dict:  # check if value is a dict. if so go a level deeper
            f_clean_dict(d[k])
        else:
            if d[k] == None:
                d[k] = 0
    return d

def f_dict_reset(used_dict, base_dict):
    '''

    :param used_dict: dictionary that is being reset (must have same keys as base_dict)
    :param base_dict: dictionary with base values used to reset (must have same keys as used_dict).
    :return: None
    '''

    for key in base_dict:
        used_dict[key] = copy.deepcopy(base_dict[key])

def f_produce_df(data, rows, columns, row_names=None, column_names=None):
    """rows is a list of lists that will be used to build a MultiIndex
    columns is a list of lists that will be used to build a MultiIndex"""
    ##if either cols or rows don't exist then add a default 0 as name
    if len(rows) == 0:  #check if no index
        row_index=[0]
    elif not any(isinstance(i, (list, np.ndarray,object)) for i in rows): #check if nested list
        row_index = rows
    elif len(rows)==1: #check if nested list with one element e.g. don't need to create multiindex
        row_index = rows[0]
    else:
        row_index = pd.MultiIndex.from_product(rows, names=row_names)
    if len(columns) == 0: #check if no index
        col_index=[0]
    elif not any(isinstance(i, (list, np.ndarray,object)) for i in columns): #check if nested list
        col_index = columns
    elif len(columns)==1: #check if nested list with one element e.g. don't need to create multiindex
        col_index = columns[0]
    else:
        col_index = pd.MultiIndex.from_product(columns, names=column_names)
    return pd.DataFrame(data, index=row_index, columns=col_index)

def f1_get_value(series, key):
    """
    Retrieve the value(s) associated with a given key or keys in a Series.
    If multiple keys are provided, their corresponding values are summed.

    Parameters:
    series (pd.Series): A Series with keys as the index.
    key (any or list or set): A single key, or multiple keys in a list or set to search in the Series index.

    Returns:
    int or float: The sum of the values associated with the key(s) if found, otherwise 0.
    """
    # Normalize input to a list for consistency
    if not isinstance(key, (list, set)):
        key = [key]

    # Use isin and sum only matching values
    return series.loc[series.index.isin(key)].sum()

def f_back_transform(x):
    ''' Back transform a value using a derivation of exp(x) / (1 + exp(x))'''
    return 1 / (1 + np.exp(-x))

def f_sig(x,a,b):
    ''' Sig function CSIRO equation 124 ^the equation below is the sig function from SheepExplorer'''
    return f_back_transform(2 * (np.log(0.95) - np.log(0.05)) / (b-a) * (x - (a+b)/2))
#    return  1/(1+np.exp(-((2*(np.log(0.95) - np.log(0.05))/(b-a))*(x-(a+b)/2))))

def f_ramp(x,a,b):
    ''' RAMP function CSIRO equation 125a'''
    return  np.minimum(1,np.maximum(0,(a-x)/(a-b)))

def f_dim(x,y):
    '''a function that returns the difference between the 2 inputs with a minimum value of zero'''
    return np.maximum(0,x-y)

def f_comb(n,k):
    # ##Create an array of factorial values up to n
    # factorial = np.cumprod(np.arange(np.max(n))+1)
    # ##Combination
    # combinations = factorial[n-1]/(factorial[k-1]*factorial[n-k-1])
    ##Create an array of factorial values up to n
    factorial_range = np.arange(np.max(n)+1)
    factorial_range[0] = 1
    factorial = np.cumprod(factorial_range)
    ##Combination
    combinations = factorial[n]/(factorial[k]*factorial[n-k])
    return combinations

def f_approach_asymptote(day0, p, step):
    '''For a measure that is approaching an asymptote during the generator period. Convert an estimate of change on
    day 0 to an average change across the days of the generator period. See Generator9:p14-15 for more detail derivation
        if d(0) = p(x* - x(0))
        then d(i) = d(0) * (1 - p)**i
        and then sum the geometric series (let q = 1-p)
        sum(d(i) (for i = 0 to n-1)) = d(0) (1-q**n) / (1-q)
    :param day0 - the estimate of the change on day0 of the generator period
    :param p -  the rate constant that the measure approaches the asymptote
    :param step - the length of the generator period in days'''

    average = f_divide(day0 * (1 - (1 - p) ** step) / p, step)   #f_divide because length of period can be 0
    return average

def solve_cubic_for_logistic(a, b, c, d):
    ''' Solve a general cubic equation of the form ax3 + bx2 + cx + d = 0
    Select the maximum value to identify a positive root that can be transformed with natural log

    To solve using the numpy.polynomial package requires looping through all elements of a,b,c,d
    or if this is too slow could be done with a vectorised calculation as done in 'Components combined - latest v2.xlsx'
    '''
    a, b, c, d = np.broadcast_arrays(a, b, c, d)
    shape = a.shape
    ##loop through axes of a,b,c,d
    roots=[]
    for i in range(len(a.ravel())):
        ###create cubic & solve
        cubic = np.polynomial.Polynomial([d.ravel()[i], c.ravel()[i], b.ravel()[i], a.ravel()[i]])
        root = np.max(np.polynomial.Polynomial.roots(cubic))
        ###save the single identified root in the array structure and repeat loop
        roots.append(root)
    roots.reshape(shape)
    cut_off01 = np.log(roots)
    return cut_off01


def f_solve_cubic_for_logistic_multidim(a, b, c, d):
    ''' Solve a general cubic equation of the form ax3 + bx2 + cx + d = 0
    The maximum value of the roots is selected to identify a positive root that can be transformed with natural log
    With correct specification of the b1[24 & 25] parameters in Universal.xlsx there should always be
    at least one positive real root

    Solved using a vectorised calculation as done in 'Components combined - latest v2.xlsx'
    Steps are:
    1. Convert to a depressed cubic
    2. Trig calculation of the roots
    3. Convert roots from depressed to general cubic
    4. Back transform the selected root to the logistic function cutoff
    '''
    ###Convert to a depressed cubic of the form t^3 + pt + q = 0
    ####where t = x + b/3a
    p = (3*a*c - b**2) / (3*a**2)
    q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)

    ###Identify multiple roots with a Trig approach with k axis in pos [-1]
    ####This method works for the type of cubic equation likely to be encountered but testing has not been exhaustive.
    k = np.array([0,1,2])
    t_roots_k = 2*np.sqrt(-p[...,na] / 3) * np.cos(1/3 * np.arccos(3 * q[...,na] / (2 * p[...,na])
                                                                 * np.sqrt(-3 / p[...,na])) - 2*np.pi * k /3)
    ###Transform the roots of the depressed cubic to the general cubic
    x_roots_k = t_roots_k - (b[...,na] / (3 * a[...,na]))
    ###Select the maximum value across the k axis (which is likely the only +ve root)
    x_roots = np.nanmax(x_roots_k, axis=-1)
    ###Back transform the roots from the conversion that created the cubic equation. This is not part of solving the roots
    ####but is done here rather than in the calling function to highlight that the problem is associated with the roots.
    cut_off01 = np.log(x_roots)   #todo an error here is most likely due to incorrect specification of the b1[24 or 25] parameters in Universal.xlsx or RR == 0 or LS == 1.0
    return cut_off01

def f_logistic_integral(x, L, k, x0, offset):
    """
    Indefinite integral of the logistic function:
        Y(x) = offset + (L - offset) / [1 + exp(-k*(x - x0))]
    
    The antiderivative is:
        F(x) = offset*x + (L - offset)/k * ln(1 + exp(k*(x - x0))) + C
    
    Parameters
    ----------
    x : float or array-like
        The x-value(s) at which to evaluate the antiderivative.
    L : float
        Upper plateau of the logistic.
    k : float
        Steepness of the logistic transition.
    x0 : float
        Midpoint (x-value where logistic is halfway between offset and L).
    offset : float
        Lower plateau of the logistic.
    
    Returns
    -------
    F(x) : float or array-like
        Value of the indefinite integral at x (up to a constant).
    """
    return offset*x + (L - offset)/k * np.log(1 + np.exp(k*(x - x0)))

def f_combined_integral(x, offset, k, x0, a, mu, sigma, x_anchor):
    """
    Indefinite integral (from 0 to x) of the anchored-and-clamped production function.

    Definition:
      Let g(x) = offset + (1 - offset)/(1 + exp(-k*(x - x0))) + a*exp(-((x - mu)/sigma)^2)
      Let h    = g(x_anchor) - 1
      Define the anchored function       f_anchor(x) = g(x) - h so that f_anchor(x_anchor) = 1
      Define the clamped production      f(x) = f_anchor(x) for x <= x_anchor, and f(x) = 1 for x > x_anchor

    This function returns:
        F(x) = ∫_0^x f(t) dt

    Notes
    -----
    - For x <= x_anchor:
        F(x) = [A(x) - A(0)] - h * x
      where A is the antiderivative of g.

    - For x > x_anchor:
        F(x) = ([A(x_anchor) - A(0)] - h * x_anchor) + 1 * (x - x_anchor)

    - Uses log1p(exp(.)) for numerical stability in the logistic integral.

    Parameters
    ----------
    x : float
        Upper limit of integration (lower limit fixed at 0).
    offset, k, x0, a, mu, sigma : floats
        Parameters of the raw model g(x) from `fit_gauss_tail`.
    x_anchor : float
        Distance at which the curve equals 1 and remains exactly 1 thereafter.

    Returns
    -------
    float
        Value of ∫_0^x f(t) dt.
    """
    from scipy.special import erf

    x = float(x)

    # Antiderivative of g(x) = logistic + Gaussian bump
    # ∫ logistic dx  +  ∫ Gaussian dx
    def A(t):
        logistic_int = offset * t + (1 - offset) / k * np.log1p(np.exp(k * (t - x0)))
        bump_int = a * (np.sqrt(np.pi) / 2) * sigma * erf((t - mu) / sigma)
        return logistic_int + bump_int

    # Compute anchor shift h = g(x_anchor) - 1
    g_anchor = offset + (1 - offset) / (1 + np.exp(-k * (x_anchor - x0))) \
               + a * np.exp(-((x_anchor - mu) / sigma) ** 2)
    h = g_anchor - 1.0

    if x <= x_anchor:
        return (A(x) - A(0.0)) - h * x
    else:
        head = (A(x_anchor) - A(0.0)) - h * x_anchor
        tail = 1.0 * (x - x_anchor)
        return head + tail

def f_dynamic_slice(arr, axis, start, stop, step=1, axis2=None, start2=None, stop2=None, step2=1):
    ##check if arr is int - this is the case for the first loop because arr may be initialised as 0
    if type(arr)==int:
        return arr
    else:
        ##first axis slice if it is not singleton
        if arr.shape[axis]!=1:
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice( start, stop, step)
            arr = arr[tuple(sl)]
        if axis2 is not None:
            ##second axis slice if required and not singleton
            if arr.shape[axis2] != 1:
                sl = [slice(None)] * arr.ndim
                sl[axis2] = slice( start2, stop2, step2)
                arr = arr[tuple(sl)]
        return arr

def f_nD_interp(x, xp, yp, axis):
    '''
    Interp with multi-D this is essentially the same as looping through axis and applying np.interp

    All inputs must be broadcastable.

    :param x: The x-coordinates at which to evaluate the interpolated values.
    :param xp: The x-coordinates of the data points, must be increasing
    :param yp: The y-coordinates of the data points, same length as xp
    :param axis: Axis to interp along
    :return: y - the interpolated values, same shape as x.
    '''
    ##add new axis where arrays have diff number of dims
    n_dims = max(x.ndim,xp.ndim,yp.ndim)
    extra_axes = tuple(range(n_dims-x.ndim))
    x = np.expand_dims(x,axis=extra_axes)
    extra_axes = tuple(range(n_dims-xp.ndim))
    xp = np.expand_dims(xp,axis=extra_axes)
    extra_axes = tuple(range(n_dims-yp.ndim))
    yp = np.expand_dims(yp,axis=extra_axes)

    ##move axis to interp along into pos=0
    x = np.moveaxis(x, source=axis, destination=0)
    xp = np.moveaxis(xp, source=axis, destination=0)
    yp = np.moveaxis(yp, source=axis, destination=0)
    ##broadcast all arrays to be the same along all axis except the interp axis
    shape = tuple(np.maximum.reduce([xp.shape[1:], yp.shape[1:], x.shape[1:]]))
    x = np.broadcast_to(x, x.shape[0:1]+shape)
    xp = np.broadcast_to(xp, xp.shape[0:1]+shape)
    yp = np.broadcast_to(yp, yp.shape[0:1]+shape)
    ##store shape of final array so it can be reshaped back
    final_shape = x.shape
    ##reshape
    x = x.reshape(x.shape[0],-1)
    xp = xp.reshape(xp.shape[0],-1)
    yp = yp.reshape(yp.shape[0],-1)
    ##loop and do interp
    y = np.zeros(x.shape)
    for i in range(x.shape[-1]):
        y[:,i] = np.interp(x[:,i], xp[:,i], yp[:,i])
    ##reshape to normal
    y = y.reshape(final_shape)
    y = np.moveaxis(y, source=0, destination=axis)
    return y

def f_merge_axis(a, source_axis=0, target_axis=1):
    '''
    This function merges two axis into one. This is basically reshaping but works if the two axis are not side by side.
    :param a: numpy array
    :param source_axis: position of axis being merged with target_axis (this axis wont exist in the new array)
    :param target_axis: position axis to be kept (this is merged with source axis)
    :return:
    '''
    ##determine new shape
    shp = np.array(a.shape)
    shp[target_axis] *= shp[source_axis]
    out_shp = np.delete(shp,source_axis)
    ##If source and target axes are adjacent ones, we can skip moveaxis and simply reshape
    if target_axis==source_axis+1:
        return a.reshape(out_shp)
    else:
        return np.moveaxis(a,source_axis,target_axis-1).reshape(out_shp)

def f_split_axis(a, len_a, axis):
    '''
    This function splits an axis into two axis.

    This is basically reshaping.

    :param a: numpy array
    :param len_a: length of the first of the two new axis
    :return:
    '''
    ##determine new shape
    shp = np.array(a.shape)
    shp[axis] /= len_a
    out_shp = np.insert(shp,axis,len_a)
    return a.reshape(out_shp)


#######################
#Specific AFO function#
#######################
def f_sa(value, sa, sa_type=0, target=0, value_min=-np.inf,pandas=False, axis=0):
    '''applies SA. Function can handle numpy or pandas'''
    ##Type 0 is sam (sensitivity multiplier) - default
    if sa_type == 0:
        if pandas:
            value = np.maximum(value_min, value.mul(sa, axis=axis))
        else:
            value  = np.maximum(value_min, value * sa)
    ##Type 1 is sap (sensitivity proportion, sa = 0 no change, sa = 1 doubles the value)
    elif sa_type == 1:
        if pandas:
            value = np.maximum(value_min, value.mul(1 + sa, axis=axis))
        else:
            value  = np.maximum(value_min, value * (1 + sa))
    ##Type 2 is saa (sensitivity addition)
    elif sa_type == 2:
        try:  #in case value array is datearray and value_min is np.inf
            value = np.maximum(value_min, value + sa)
        except TypeError:
            value = value + sa

    ##Type 3 is sat (sensitivity target)
    ### sa=0 returns existing value, sa=1 returns the target, sa=-1 returns value that is same distance but opposite direction to target)
    elif sa_type == 3:
        if pandas:
            value = np.maximum(value_min, value + (target - value).mul(sa, axis=axis))
        else:
            value  = np.maximum(value_min, value + (target - value) * sa)
    ##Type 4 is sar (sensitivity range. sa=-1 returns value_min, sa=0 returns value, sa=1 returns target)
    ### Note: sa less than -1 or sa greater than +1 continues beyond the range on the same slope
    elif sa_type == 4:
        if pandas:
            #todo add Pandas code to replace the target with value_min where sa<0 (see numpy code below)
            value = value + (target - value).mul(sa.abs(), axis=axis)
        else:
            #if the sa is less than zero then swap the target value and use the absolute value of sa
            f_update(target, value_min, sa < 0)
            value = value + (target - value) * np.abs(sa)
    ##Type 5 is sav (return the SA value, '-' is no change)
    elif sa_type == 5:
        try:
            sa=sa.copy()#have to copy the np arrays so that the original sa is not changed
        except:
            pass
        ###convert to numpy if pandas so f_update works correctly (so dtype is handled correctly)
        if isinstance(value, pd.DataFrame):
            value.iloc[:,:] = f_update(value.values, sa, sa != '-') #sa has to be object or this give FutureWarning
        elif isinstance(value, pd.Series):
            value.iloc[:] = f_update(value.values, sa, sa != '-') #sa has to be object or this give FutureWarning
        else:
            value = f_update(value, sa, sa != '-') #sa has to be object or this give FutureWarning

    return value

def f_update_sen(user_sa, sam, saa, sap, sar, sat, sav):
    '''
    Update default SA arrays from sensitivity.py with values from user.

    Each trial sensitivity.py rebuilds the SA arrays so nothing from the previous trial can carry over.
    '''
    for sa in user_sa:
        dic = sa["operation"]
        key1 = sa["key1"]
        key2 = sa["key2"]
        indices = sa["indices"]
        value = sa["value"]

        ##checks if both slice and key2 exists
        if indices is not None and key2 is not None:
            if dic == 'sam':
                sam[(key1,key2)][indices] = sam[(key1,key2)][indices] * value  # if there are multiple instances of the same SA in exp.xlsx they accumulate
            elif dic == 'sap':
                sap[(key1,key2)][indices] = (1 + sap[(key1,key2)][indices]) * (1 + value) - 1  # if there are multiple instances of the same SA in exp.xlsx they accumulate
            elif dic == 'saa':
                saa[(key1,key2)][indices] = saa[(key1,key2)][indices] + value  # if there are multiple instances of the same SA in exp.xlsx they accumulate
            elif dic == 'sat':
                sat[(key1,key2)][indices] = value   # last entry in exp.xlsx is used
            elif dic == 'sar':
                sar[(key1,key2)][indices] = sar[(key1,key2)][indices] + value  # if there are multiple instances of the same SA in exp.xlsx they accumulate
            elif dic == 'sav':
                try:
                    if value != "-": #SAV entries with '-' do not update the SAV. This means that if slices of a SAV overlap in Exp.xl the last non '-' is the value used.
                        update_sav=True
                    else:
                        update_sav=False
                except ValueError:   #try and except required for web app because "value" is an array (so the if statement causes error).
                    update_sav=True
                if update_sav:
                    sav[(key1,key2)][indices] = value   # last entry in exp.xlsx that is not "-" is used

        ##checks if just slice exists
        elif indices is not None:
            if dic == 'sam':
                sam[key1][indices] = sam[key1][indices] * value  # if there are multiple instances of the same SA in exp.xlsx they accumulate
            elif dic == 'sap':
                sap[key1][indices] = (1 + sap[key1][indices]) * (1 + value) - 1  # if there are multiple instances of the same SA in exp.xlsx they accumulate
            elif dic == 'saa':
                saa[key1][indices] = saa[key1][indices] + value  # if there are multiple instances of the same SA in exp.xlsx they accumulate
            elif dic == 'sat':
                sat[key1][indices] = value
            elif dic == 'sar':
                sar[key1][indices] = sar[key1][indices] + value  # if there are multiple instances of the same SA in exp.xlsx they accumulate
            elif dic == 'sav':
                try:
                    if value != "-": #SAV entries with '-' do not update the SAV. This means that if slices of a SAV overlap in Exp.xl the last non '-' is the value used.
                        update_sav=True
                    else:
                        update_sav=False
                except ValueError:   #try and except required for web app because "value" is an array (so the if statement causes error).
                    update_sav=True
                if update_sav:
                    sav[key1][indices] = value
        ##checks if just key2 exists
        elif key2 is not None:
            if dic == 'sam':
                sam[(key1, key2)] #checks the keys exist. Not required for SA that have indicies.
                sam[(key1,key2)] = sam[(key1,key2)] * value
            elif dic == 'sap':
                sap[(key1, key2)]  # checks the keys exist. Not required for SA that have indicies.
                sap[(key1,key2)] = (1 + sap[(key1,key2)]) * ( 1+ value) -1
            elif dic == 'saa':
                saa[(key1, key2)]  # checks the keys exist. Not required for SA that have indicies.
                saa[(key1,key2)] = saa[(key1,key2)] + value
            elif dic == 'sat':
                sat[(key1, key2)]  # checks the keys exist. Not required for SA that have indicies.
                sat[(key1,key2)] = value
            elif dic == 'sar':
                sar[(key1, key2)]  # checks the keys exist. Not required for SA that have indicies.
                sar[(key1, key2)] = sar[(key1,key2)] + value
            elif dic == 'sav':
                sav[(key1, key2)]  # checks the keys exist. Not required for SA that have indicies.
                if np.isscalar(sav[(key1,key2)]) and np.isscalar(value):
                    sav[(key1,key2)] = value if value != '-' else sav[(key1,key2)]
                elif isinstance(sav[(key1,key2)], np.ndarray) and np.isscalar(value):
                    if value != '-':
                        sav[(key1,key2)][...] = value
                else: #required for web app because "value" is an array
                    sliced_sav = sav[(key1,key2)][tuple(slice(0, s) for s in value.shape)] #have to slice to handle cases where sen is initiated with large number
                    sav[(key1,key2)] = np.where(value != '-', value, sliced_sav)
        ##if just key1 exists
        else:
            if dic == 'sam':
                sam[key1]  # checks the keys exist. Not required for SA that have indicies.
                sam[key1] = sam[key1] * value
            elif dic == 'sap':
                sap[key1]  # checks the keys exist. Not required for SA that have indicies.
                sap[key1] = (1 + sap[key1]) * (1 + value) - 1
            elif dic == 'saa':
                saa[key1]  # checks the keys exist. Not required for SA that have indicies.
                saa[key1] = saa[key1] + value
            elif dic == 'sat':
                sat[key1]  # checks the keys exist. Not required for SA that have indicies.
                sat[key1] = value
            elif dic == 'sar':
                sar[key1]  # checks the keys exist. Not required for SA that have indicies.
                sar[key1] = sar[key1] + value
            elif dic == 'sav':
                sav[key1]  # checks the keys exist. Not required for SA that have indicies.
                if np.isscalar(sav[key1]) and np.isscalar(value):
                    sav[key1] = value if value != '-' else sav[key1]
                elif isinstance(sav[key1], np.ndarray) and np.isscalar(value):
                    if value != '-':
                        sav[key1][...] = value
                else: #required for web app because "value" is an array
                    sliced_sav = sav[key1][tuple(slice(0, s) for s in value.shape)] #have to slice to handle cases where sen is initiated with large number
                    sav[key1] = np.where(value != '-', value, sliced_sav)

def f1_make_r_val(r_vals, param, name, maskz8=None, z_pos=0, shape=None):
    '''
    This function saves a variable in the r_vals dict so it can be accessed in the reporting stage.

    The majority of this function concerns unclustering the z axis. This is required for two reasons:

        1. By the time the r_val is save it would have likely been masked by mask_z8.
        2. The user may have incorrectly clustered the inputs in excel (e.g. seasons had different inputs before they
           were identified). This doesn't effect the actual model because z8 is masked until it is identified
           however if the r_val didn't get z8 treatment the reports could contain errors.

    :param r_vals: r_vals dict
    :param param: param to be stored
    :param name: name of r_val
    :param maskz8: season identification mask
    :param z_pos: position of z axis from the end (ie this should be negative)
    :param shape: optional - shape of r_val - this can be used to remove singleton axis.

    Note 1: Arrays must broadcast.
    Note 2: if no z axis then param is simply stored in r_vals no need to pass in the mask arg.
    '''
    if maskz8 is not None:
        df=False
        series=False
        ##convert df to series
        if isinstance(param,pd.DataFrame):
            df = True
            n_cols = param.columns.nlevels
            param = param.stack(list(range(n_cols)))
        ##convert pd.Series to numpy
        if isinstance(param,pd.Series):
            series = True
            ##store index
            index = param.index
            ##reshape array to be numpy
            reshape_size = param.index.remove_unused_levels().levshape # create a tuple with the rights dimensions
            param = np.reshape(param.values,reshape_size)

        ##uncluster z so that each season gets complete information
        index_z = f_expand(np.arange(maskz8.shape[z_pos]), z_pos)
        a_zcluster = np.maximum.accumulate(index_z * maskz8, axis=z_pos)
        a_zcluster = np.broadcast_to(a_zcluster, param.shape)
        param = np.take_along_axis(param, a_zcluster, axis=z_pos)

        ##add index if pandas
        if series:
            param = pd.Series(param.ravel(), index=index)

        ##unstack back to a df if required
        if df:
            param = param.unstack(list(range(-n_cols,0)))

    ##reshape if required
    if shape is not None:
        param = param.reshape(shape)

    ##store param
    r_vals[name] = param

def f1_make_pyomo_dict(param, index, loop_axis_pos=None, index_loop_axis_pos=None, dtype='float32'):
    '''
    Convert numpy array into dict for pyomo. A loop can be used to reduce memory if required.

    0 values are removed to reduce time (when creating the param in pyomo) and space.

    :param param: numpy array
    :param index: list of index arrays
    :param loop_axis_pos: optional: position of axis that is being looped on (arg not required if no loop)
    :param index_loop_axis_pos: optional: position of axis that is being looped on in the index array (arg not required if no loop)
    :return: dict for pyomo
    '''
    ##build in loop to reduce memory for some big params
    if loop_axis_pos:
        param_masked = np.array([],dtype=dtype)
        index_masked = np.array([])
        for i in range(param.shape[loop_axis_pos]):
            ###mask out values=0
            param_cut = f_dynamic_slice(param, loop_axis_pos, start=i, stop=i+1)
            mask = param_cut != 0
            param_masked = np.concatenate([param_masked,param_cut[mask]],0).astype(dtype)  # applying the mask does the raveling and squeezing of singleton axis
            mask = mask.ravel() #needs to be 1d to mask the index
            ###build index
            ####adjust if the position given is negative (e.g. cant use pos=-1)
            if index_loop_axis_pos<0:
                index_loop_axis_pos = len(index) + index_loop_axis_pos
            index_cut = [index[x] if x != index_loop_axis_pos else index[x][i:i+1] for x in range(len(index))]
            index_cut = cartesian_product_simple_transpose(index_cut)
            index_masked = np.vstack([index_masked,index_cut[mask,:]]) if index_masked.size else index_cut[mask,:]
    else:
        ###mask out values=0
        mask = param!=0
        ###build index
        index = cartesian_product_simple_transpose(index)
        ###mask param and index
        param_masked = param[mask]  # applying the mask does the raveling and squeezing of array
        mask = mask.ravel() #needs to be 1d to mask the index
        index_masked = index[mask,:]

    ##error check - index and param should be same length but zip() doesn't throw error if they are different length
    if len(index_masked) != len(param_masked):
        raise exc.ParamError('''Index and param must be the same length''')

    ##make index a tuple and zip with param and make dict
    tup = tuple(map(tuple,index_masked))
    return dict(zip(tup, param_masked))

import numpy as np
na = np.newaxis


def build_active_index(
    masks,
    axis_keys,
    reduce_extra_axes="any",
    active_condition="!=0",
):
    """
    Build a list of active index tuples from one or more masks.

    Parameters
    ----------
    masks : np.ndarray or list of np.ndarray
        Single mask or list of masks. They will be broadcast to a common
        shape and combined with logical AND.

        Example shapes:
            (k,p6,p5,z,l)
            (p6,z)  -> will be broadcast if needed

    axis_keys : list of 1D arrays
        Keys for each axis you want in the final index (left-most axes
        after broadcasting).

        Example:
            axis_keys = [keys_k, keys_p6, keys_p5, keys_z, keys_l]

    reduce_extra_axes : "any" | "all" | None | list[int]
        How to collapse extra dimensions beyond len(axis_keys).

        - "any": active if any extra axis is True.
        - "all": active if all extra axes are True.
        - list[int]: explicit axes to reduce over.
        - None: require mask.ndim == len(axis_keys).

    active_condition : str
        How to treat non-bool masks (if masks are numeric):

        - "!=0": mask != 0
        - ">0":  mask > 0

    Returns
    -------
    list[tuple]
        A list of tuples (key0, key1, ..., keyN) where the combined mask is active.
    """

    # --- 1. Normalise to a combined boolean mask ---
    if isinstance(masks, (list, tuple)):
        arrs = []
        for m in masks:
            m = np.asarray(m)
            if m.dtype == bool:
                arrs.append(m)
            else:
                if active_condition == "!=0":
                    arrs.append(m != 0)
                elif active_condition == ">0":
                    arrs.append(m > 0)
                else:
                    raise ValueError("unsupported active_condition")
        combined = np.logical_and.reduce(arrs)
    else:
        m = np.asarray(masks)
        if m.dtype == bool:
            combined = m
        else:
            if active_condition == "!=0":
                combined = (m != 0)
            elif active_condition == ">0":
                combined = (m > 0)
            else:
                raise ValueError("unsupported active_condition")

    # --- 2. Reduce extra axes if needed ---
    n_keep = len(axis_keys)
    if combined.ndim < n_keep:
        raise ValueError("mask has fewer dims than axis_keys")

    if combined.ndim > n_keep:
        if reduce_extra_axes is None:
            raise ValueError("mask has extra dims and reduce_extra_axes is None")

        if isinstance(reduce_extra_axes, str):
            extra_axes = tuple(range(n_keep, combined.ndim))
            if reduce_extra_axes == "any":
                combined = np.any(combined, axis=extra_axes)
            elif reduce_extra_axes == "all":
                combined = np.all(combined, axis=extra_axes)
            else:
                raise ValueError("unknown reduce_extra_axes string")
        else:
            # explicit list of axes
            combined = np.any(combined, axis=tuple(reduce_extra_axes))

    # Now combined.shape must match the len of each axis_keys[i]
    # --- 3. Extract active indices and map to keys ---
    idx = np.nonzero(combined)
    active = []

    for indices in zip(*idx):
        # indices is like (ik, ip6, ip5, iz, il)
        key_tuple = tuple(axis_keys[ax][i] for ax, i in enumerate(indices))
        active.append(key_tuple)

    return active


def write_variablesummary(model, trial_name, obj, option=0, property_id=''):
    '''

    :param model: pyomo model
    :param trial_name: Trial name
    :param obj: objective value
    :param option: 0: trial name and property ID will be included in file name
                   1: file name will be generic
    :return:
    '''
    ##This writes variables with value greater than 0.0001 to txt file
    ### written with trial description in file name if full solution is requested (option 0)
    ### written every iteration with generic name (option 1) - can be used to check progress of analysis each iteration
    from . import relativeFile #import here because function.py shouldnt import other modules if possible
    if option == 0:
        output_path = relativeFile.find(__file__, "../../Output", f'Variable summary {trial_name} - {property_id}.txt')
        file = open(output_path,'w')  # file name has to have capital
    else:
        output_path = relativeFile.find(__file__, "../../Output", 'Variable summary.txt')
        file = open(output_path,'w')  # file name has to have capital
    file.write('Trial: %s\n' % trial_name)  # the first line is the name of the trial
    file.write('{0} profit: {1}\n'.format(trial_name,obj))  # the second line is profit
    for v in model.component_objects(pe.Var,active=True):
        file.write("Variable %s\n" % v)  # \n makes new line
        for index in v:
            try:
                val = v[index].value
                if val is None:
                    continue

                r = round(val, 4)  # round to 4 decimal places
                if r != 0.0:  # drop values that become 0.0000 (or -0.0000)
                    file.write("   %s %.4f\n" % (index, r))  # always print 4 d.p.
            except:
                pass
    file.close()


##########################
# period calculators     #
##########################

def f_range_allocation_np(period_dates, item_start, length=np.array([1]), method=1, shape=None, is_phase_param=False,
                          break_z=None, season_start=None, z_pos=None):
    ''' Numpy version - The proportion of a date range that falls within each period or proportion of each period that falls in the tested date range.

    Where possible use the default option (method=1). When using method 2 the date range (item start to
    item end) must fall within the period array (the user must make this happen but a warning will be thrown if not).

    Note: if a date range that falls partly in p[0] and partly in p[-1] will get 100% allocated to p[-1].
          This is because we cant have allocation that crosses the season junction. In some cases this may result in over
          allocation to the last period.

    Note 2: For params linked to v_phase activity the timing of an item is adjusted so that no cost/labour/depn is incurred
    between season start and break of season. This stops the model getting double costs in medium/late breaks where
    phases are carried over past the start of the season to provide dry pas and stubble area (because it is also
    accounted for by v_phase_increment).

    :param period_dates: the start of the periods (including end date of last period). This array must be broadcastable with start
                  (therefore may need to add new axis if start has a dimension). Period axis must be in pos=0.
    :param item_start: the date of the beginning of the date range to test - a numpy array of dates
    :param length: the length of the date range to test (days). Must be broadcastable to start.
    :param method: Controls the proportion calculated. Method 1 returns the proportion of date range in each period.
                     Method 2 returns the proportion of the period in the date range.
    :param shape: this is the shape of returned array, required if both period_dates & start have more than 1 dim
    :param is_phase_param: boolean to flag if the item being allocated will be liked to v_phase activity. If True the timing gets adjusted so that no cost/labour/depn is incurred between season start and break of season. Otherwise double counting can occur with v_phase and v_phase_increment.

    :return: Numpy array with shape(period_dates, start array). Containing the proportion of the
             respective period for that test date.
    '''
    ##make length at least 1 to stop div 0 if seeding/harv periods are set to 0
    min = np.array([1])
    length = np.maximum(min,length)

    #start empty array to assign to
    if shape==None:
        allocation_period=np.zeros((period_dates.shape[:-1] + item_start.shape),dtype='float64')
    else:
        allocation_period=np.zeros(shape,dtype='float64')

    ###adjust the timing of items linked to phases so that no cost/labour/depn is incurred between season start and break of season.
    ### this stops the model getting double costs in medium/late breaks where phases are carried over past the
    ### start of the season to provide dry pas and stubble area (because it is also accounted for by v_phase_increment).
    if is_phase_param:
        date_break_z = f_expand(break_z, z_pos) #adjust to get z axis in the correct position.
        between_seasonstart_brkseason = np.logical_and(item_start%364>=season_start, item_start%364<date_break_z)
        item_start = f_update(item_start%364, date_break_z, between_seasonstart_brkseason)

    ## adjust dates.
    if method==1:
        ##adjust yr of item occurrence
        start_of_periods = period_dates[0,...]
        end_of_periods = start_of_periods + 363 #use 363 because end date is the day before the end otherwise can get item that starts on the last day of periods.
        add_yrs = np.ceil(np.maximum(0,(start_of_periods - item_start) / 364))
        sub_yrs = np.ceil(np.maximum(0,(item_start - end_of_periods) / 364))
        item_start = item_start + add_yrs * 364 - sub_yrs * 364
        ##handle cases where date + length is after the end of periods. in this situation length gets reduced - this is the easiest method and reduces possible errors due to crossing season junction.
        length = np.minimum(length, (period_dates[-1,...] - item_start))
    elif method==2:
        ###adjust the period dates (leave item date the same)
        period_end_dates = period_dates[1:, ...]
        period_start_dates = period_dates[:-1, ...]
        item_end = item_start + length
        add_yrs = np.ceil(np.maximum(0,(item_start - period_end_dates) / 364))
        sub_yrs = np.ceil(np.maximum(0,(period_start_dates - item_end) / 364))
        period_start_dates = period_start_dates + add_yrs * 364 - sub_yrs * 364
        period_end_dates = period_end_dates + add_yrs * 364 - sub_yrs * 364

    ##calc end of period
    item_end = np.minimum(item_start + length, period_dates[-1]) #minimum ensures that the assigned date range is within the period date range.

    ##checks if user wants the proportion of each period that falls in the tested date range or proportion of date range in each period
    if method==1:
        #check how much of each date range falls within the period (allocate start dates into periods)
        for i in range(len(period_dates)-1):
            per_start = period_dates[i, ...] #[i:i+1] #to keep dim
            per_end = period_dates[i+1, ...]
            calc_start = np.maximum(per_start,item_start)       #select the later of the period start or the start of the range
            calc_end = np.minimum(per_end,item_end)             #select earlier of the period end and the end of the range
            allocation_period[i,...] = np.maximum(0, (calc_end - calc_start) / (item_end - item_start)) #days between calc_end and calc_start (0 if end before start) divided by length of the range
    else:
        ###check date range falls within periods - period adjustment happens above so this should never be an error unless the inputs are bad.
        if not np.all(np.logical_and(np.min(period_start_dates,axis=0) < item_start, item_end < np.max(period_end_dates, axis=0))):
            raise Warning('Trying to allocate periods into a date array but date array range is greater than the period range.'
                          'This indicates poor inputs. You may need to alter the inputs')
        ###check how much of each period falls within the date range (allocate periods into start dates)
        for i in range(len(period_dates)-1):
            per_start = period_start_dates[i, ...] #[i:i+1]
            per_end = period_end_dates[i, ...]
            calc_start = np.maximum(per_start,item_start)       #select the later of the period start or the start of the range
            calc_end = np.minimum(per_end,item_end)            #select earlier of the period end and the end of the range
            allocation_period[i,...] = np.maximum(0, f_divide(calc_end - calc_start
                                                              , per_end - per_start)) #days between calc_end and calc_start (0 if end before start) divided by length of the period, use f_divide in case any period lengths are 0 (this is likely to occur in season version)

    return allocation_period

def period_proportion_np(period_dates, date_array):
    ''' Numpy version - The period that a given date falls in. and the proportion of the way through the period the date occurs.

    Parameters.
    period_dates: The dates of the periods to search within. Must contain the end date of the last period. If multi-D period axis must be pos 0.
    date_array:  The dates to allocate.

    Note: period_dates and date_array must be broadcastable.

    Returns.
    Two Numpy arrays with shape(date_array).
        1 period_array - the period which the values in date_array occur.
        2 proportion_array - how far through the period the date occurs.
    '''

    ##broadcast period_dates so that it has same size axis as date_array - so slicing works
    shape = (period_dates.shape[0],) + date_array.shape
    period_dates = np.broadcast_to(period_dates, shape)

    ##get dtype consistent
    period_dates = period_dates
    date_array = date_array

    ##dates
    dates_start = period_dates[:-1]
    dates_end = period_dates[1:].copy() #so original date array isn't altered when updating year in next step

    ##adjust yr of item occurrence
    start_of_periods = period_dates[0,...]
    end_of_periods = period_dates[-1,...]
    add_yrs = np.ceil(np.maximum(0,(start_of_periods - date_array) / 364))
    sub_yrs = np.ceil(np.maximum(0,(date_array - end_of_periods) / 364))
    date_array = date_array + add_yrs * 364 - sub_yrs * 364
    ###little check to ensure that all cashflow is all starting at least 1 day before the end cashflow date
    date_array = date_array - np.maximum(0, (date_array - (period_dates[-1,...] - 1)))

    ##calc the period each value in the date array falls within (can't use np.searchsorted because date array has z axis)
    ###occur is bool array which is true for the period that the date array fall into
    occur = np.logical_and(dates_start <= date_array, date_array < dates_end)
    ###period index
    p_idx = np.arange(period_dates[:-1].shape[0])
    ###mul occur and idx to return the period number the date falls in else a 0. then sum the period axis to return the period array
    occur = np.moveaxis(occur,0,-1) #so that period axis is at end
    period_array = np.sum(occur * p_idx, axis=-1)

    ##calc proportion
    per_start = np.take_along_axis(period_dates,period_array[None,...],0)[0]
    per_end = np.take_along_axis(period_dates,period_array[None,...]+1,0)[0]
    # per_start = period_dates[period_array, np.arange(date_array.shape[0])[:,None], np.arange(date_array.shape[1])] #problem is that this is fixed to 3d
    # per_end   = period_array[period_array + 1]
    proportion_array = (date_array - per_start) / (per_end - per_start)
    # print('propn, date, stat, end, start', proportion_array,date_array,per_start,per_end,per_start)
    return period_array, proportion_array

##################
#timing functions#
##################

def f_daylength(dayOfYear, lat):
    """Computes the length of the day (the time between sunrise and
    sunset) given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
    For more information see, for example,
    Forsythe et al., "A model comparison for daylength as a
    function of latitude and day of year", Ecological Modelling,
    1995.
    Parameters
    ----------
    dayOfYear : int
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.
    Returns
    -------
    d : float
        Daylength in hours.
    """
    dl=np.zeros_like(dayOfYear, dtype='float64')
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+dayOfYear)/365.0))
    p1 = (-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))) <= -1.0
    p2 = (-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))) >= 1.0
    hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
    daylen = 2.0*hourAngle/15.0
    dl[p1] = 24
    dl[p2] = 0
    #todo Should this be logical_or? JMY thinks it should be.
    dl[~np.logical_and(p2, p1)] = daylen[~np.logical_and(p2, p1)]
    return dl

def f_next_prev_association(datearray_slice,*args):
    '''
    Depending on the inputs this function will return the next or previous association.
    e.g. it can be used to determine the next lambing opportunity for each period.
    See john stuff.py for alternative methods.

    Parameters
    ----------
    datearray_slice : any
        This is 1d array which the second array is sorted into (this must be sorted).
    *args : 1 - array, 2 - int
        Arg 1: the period array 1d that is the index is being found for, note the index is based off the start date therefore must do [1:-1] if you want idx based on end date.
        Arg 2: period offset (this may be needed if evaluating the end of the period.

    Returns
    -------
    Array
        The function finds the index value of the datearray which is either the next or previous date for a given input date.
        ## The previous opportunity is the latest opportunity date that is less than the date at the end of the period
        ## e.g. ('end of the period' so that if joining occurs during the period it is the previous
        ## The next opportunity is the earliest joining date that is greater than the date at the start of the period
        ## e.g. So it is the prev + 1 except if the joining is occurring within the period, in which case it points to this one.


    '''
    date=args[0]
    offset=args[1] #offset is used to get the previous datearray period
    side=args[2]
    idx_next = np.searchsorted(datearray_slice, date, side)
    idx = np.clip(idx_next - offset, 0, len(datearray_slice)-1) #makes the max value equal to the length of joining array, because if the period date is after the last lambing opportunity there is no 'next'
    return idx

def f1_lmuregion_to_lmufarmer(dict, key1, a_lmuregion_lmufarmer, lmu_axis, lmu_flag):
    lmu_flag[key1] = True #set flag to true to say that the lmu input has been adjusted. This gets checked when applying lmu mask. This is to catch user error if the user doesn't update f_farmer_lmu_adj.
    ##add axes to a_lmuregion_lmufarmer
    ndims = dict[key1].ndim
    lmu_pos = lmu_axis - ndims
    left_pos2 = -ndims
    a_lmuregion_lmufarmer = f_expand(a_lmuregion_lmufarmer, left_pos=lmu_pos, right_pos2=lmu_pos, left_pos2=left_pos2-1)

    if isinstance(dict[key1], pd.DataFrame):
        dict[key1].iloc[:,:] = np.take_along_axis(dict[key1].values, a_lmuregion_lmufarmer, lmu_axis)
    else:
        dict[key1] = np.take_along_axis(dict[key1], a_lmuregion_lmufarmer, lmu_axis)

def f1_slices_to_str(array_name, slices):
    """
    Convert a list of slice objects into an indexing string like arr1[2:5,1:4,:]
    Useful for reporting action carried out on blocks within multi-dimensional arrays
    """

    def fmt(s):
        if s == slice(None):  # the most common case
            return ':'

        start = s.start if s.start is not None else ''
        stop = s.stop if s.stop is not None else ''
        step = s.step if s.step is not None else 1

        if step == 1:  # step=1 is the default → omit it
            if start == 0 and stop == '':
                return ':'
            elif start == 0:
                return f':{stop}'
            elif stop == '':
                return f'{start}:'
            else:
                return f'{start}:{stop}'
        else:  # explicit step
            start_str = str(start) if start != 0 else ''
            return f'{start_str}:{stop}:{step}'.strip(':') or ':'

    parts = [fmt(sl) for sl in slices]
    return f"{array_name}[{','.join(parts)}]"

