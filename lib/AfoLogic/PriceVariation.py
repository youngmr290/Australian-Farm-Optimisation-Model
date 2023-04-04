'''
Summary
-------
Notwithstanding weather variation, agricultural commodity price is a major source of farm risk.
There are two main methods to include price variation in whole farm LP.

    1.	Expected price variation (e.g. Kingwell, 1994): Expected price variation represents price
        variation by applying a discrete distribution to cashflow items after management decisions have been
        made. This method of representing price variation assumes that there is no knowledge of the
        price state, prior to purchasing or selling a commodity. The only known information is the
        expected price (i.e. a farmer does not know if they are in a high or low price year until
        they purchase or sell). Therefore, price variation has no impact on farm management for a
        risk neutral farmer. However, for a risk averse farmer price variation can alter their
        management. For example, if the grain price is more variable than livestock prices, it
        may be optimal for a risk averse farmer to have a higher livestock focus because it will
        reduce the variation in farm profit between years.
    2.	Forecasted price variation (Apland and Hauer, 1993): Forecasted price variation is a more
        realistic method achieved by including discrete states based on forecast information, allowing
        decision-making to change based on the forecasted conditions. The forecasted states are
        adjusted using a discrete distribution to reflect the actual prices received at purchase
        or sale. This requires a stochastic programming approach that increases model size and complexity.

AFO currently uses method 1 because price variation has not been a major focus as yet.
Nonetheless, a likely valuable future improvement for AFO would be to include forecasted
price variation. AFO’s flexible structure would facilitate inclusion of such price variation.

Currently, price variation is approximated in AFO using a range of discrete price states
for meat, wool and grain. The need to form discrete approximations of a continuous distributions
is a necessary requirement for developing a LP model of farm management responses to price and
weather-year states. By their nature, discrete stochastic programming models cannot consider
all possible price states as described by continuous distributions. Rather continuous variables
such as price need to be approximated by discrete states.

Price scalars have two main purposes:

    #. To account for variations in the price received for a given year due to external market conditions (c1 axis).
    #. To account for variation in prices due to season type.

Within year price cycles are accounted for in AFO for products such as sale sheep that can be sold at
different times during the year. Including the within year price cycles ensures that optimisation of the
nutrition of sale sheep represents that sale data has an effect on expected price. Representing the
annual price cycle also ensures that strategic management such as time of lambing is also evaluated
correctly given the impact of time of lambing on likely turn-off dates.

Generation of discrete price states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The price state scalars and their probabilities are calculated by fitting a multivariate normal
distribution to historical prices, then summarises as discrete states by dividing the multi-dimensional
probability density distribution into segments. A multivariate distribution is used so that
correlations between commodities are accurately represented in the resulting price states.
Grain and wool prices are better represented by log-normal distributions :cite:p:`kingwell1996`.
Thus, before fitting the distribution, grain and wool data were subject to a log transformation.
Additionally, the historical prices were CPI adjusted and detrended using a long-term moving average.
The reason for detrending the price data was that the price states represented in AFO serve the
purpose of capturing yearly price variation (i.e. variations around the expected price for that
year) rather than capturing within year price cycles.

To reduce model size and simplify input calibration, all meat classes (lamb, shipper, mutton, etc)
receive the same meat price scalar. The same thing happens for classes of wool and types of grain.
This simplification should not compromise the accuracy of the results because subclasses of a given
commodity tend to have a high correlation (e.g. between 2000 and 2021 the correlation between light
lamb and mutton was 96%). A further simplification was excluding price variation for input costs
because input costs tend to vary less :cite:p:`kingwell1996` and therefore the additional model size
was not justified. The resulting assumptions are that all animal classes are 100% correlated,
all wool microns are 100% correlated, all grains are 100% correlated and all input commodities
have no variation. This assumption is not entirely accurate (e.g. canola and wheat prices are
not 100% correlated) however, if in future analysis, price variation is of high importance this
can easily be rectified by expanding the inputs.

The c1 axis is averaged for both the asset constraint and the working capital. This saves space without losing
much/any information.

'''
import os
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from . import PropertyInputs as pinp

na=np.newaxis

##do this so the module doesnt run when building the docs
if __name__=="__main__":
    ##########
    ##inputs #
    ##########
    len_z = len(pinp.general['i_mask_z'])
    len_c1 = 8
    keys_z = pinp.general['i_z_idx']
    keys_c1 = np.array(['c1_%s' % i for i in range(len_c1)])
    method = 1 #0=use raw data, 1=use moving average to detrend data

    ##read in CPI adjusted price series
    directory_path = os.path.dirname(os.path.abspath(__file__))
    price_data = pd.read_excel(os.path.join(directory_path, "Raw Price Series.xlsx"), sheet_name="Python", index_col=0)

    ##plot to confirm if the relationship is normal or log normal
    # fig = px.histogram(price_data, x="APW wheat")
    # fig = px.histogram(np.log(price_data), x="APW wheat")
    # fig = px.histogram(price_data, x="Mutton")
    # fig = px.histogram(np.log(price_data), x="Mutton")
    # fig.show()

    ##create price scalars
    if method==0:
        ##select prices used in the distribution and log any that are best fit with log-normal dist
        adj_prices = price_data[["APW wheat","Mutton","S 21 MPG"]].div(price_data[["APW wheat","Mutton","S 21 MPG"]].mean()).copy()
        adj_prices.loc[:,"APW wheat"] = np.log(adj_prices["APW wheat"]) #log fits the grain data better (as seen by the plots above)
        adj_prices.loc[:,"S 21 MPG"] = np.log(adj_prices["S 21 MPG"]) #log fits the wool data better (as seen by the plots above)
        ##graph to check normal
        # fig = px.histogram(adj_prices, x="APW wheat")
        # fig = px.histogram(adj_prices, x="S 21 MPG")
        # fig.show()
    elif method==1:
        ##detrend raw prices using moving average.
        adj_prices = price_data.rolling(window=52).mean().div(price_data).dropna()
        adj_prices = adj_prices[["APW wheat","Mutton","S 21 MPG"]].copy()
        ##graph to check normal
        # fig = px.histogram(adj_prices, x="APW wheat")
        # fig = px.histogram(adj_prices, x="S 21 MPG")
        # fig.show()

    ###############################
    # summarise into price states #
    ###############################
    '''
    Using the mean and covariance of multiple variables (currently 3 - wool, meat & grain) a multivariate normal 
    distribution is fitted. The distribution is used to return the PDF (probability density function)of different prices. 
    
    Interpretation of the PDF:
    On its own the value returned from a PDF function doesnt mean much. If you were to plot the whole PDF the area under
    the curve would equal 1. 
    It is impossible to provide the probability of obtaining one singular price because there are infinite possibilities.
    Therefore interpreting a PDF is essentially like saying "what is the probability of a price between x and y".
    The answer is equal to the area under the graph. Therefore, in the following code we generate the pdf for a large number
    of prices and then determine the probability of a price within each little chunk.
     
    To generate the PDF for a range of values required builiding an array with the desired number of slices.
    The min and max value for each axis is determined by the 3rd standard deviation. This should capture 99.9% of the distribution.
    In the post processing the probability of each chunk is added together to provide a given number of price states.
    
    Note: This is inflexible - if you want to add more variables into the distribution you will need to add code.
    '''
    ##build the multivariate distribution function. This function returns the probablility of a given value of each independent variable.
    mu = np.mean(adj_prices)
    cov = np.cov(adj_prices.T) #rows are variables columns are observations.
    rv = multivariate_normal(mu, cov)

    ##build arrays to pass to the distribution function
    ##builds a probability density distribution with 100*100 chunks.
    n_chunks = 100
    w_min = adj_prices["APW wheat"].mean() - 3 * adj_prices["APW wheat"].std()
    w_max = adj_prices["APW wheat"].mean() + 3 * adj_prices["APW wheat"].std()
    w = np.linspace(w_min, w_max, n_chunks)
    w_step = w[1]-w[0]
    x_min = adj_prices["Mutton"].mean() - 3 * adj_prices["Mutton"].std()
    x_max = adj_prices["Mutton"].mean() + 3 * adj_prices["Mutton"].std()
    x = np.linspace(x_min, x_max, n_chunks)
    x_step = x[1]-x[0]
    y_min = adj_prices["S 21 MPG"].mean() - 3 * adj_prices["S 21 MPG"].std()
    y_max = adj_prices["S 21 MPG"].mean() + 3 * adj_prices["S 21 MPG"].std()
    y = np.linspace(y_min, y_max, n_chunks)
    y_step = y[1]-y[0]
    W, X, Y = np.meshgrid(w, x, y)
    pos = np.stack([W,X, Y], axis=-1)
    Z = rv.pdf(pos)

    ##plot returned values - this only works for 2 variables
    rv = multivariate_normal(mu[0:2], cov[0:2,0:2])
    W, X = np.meshgrid(w, x)
    pos = np.dstack((W, X))
    Z_plot = rv.pdf(pos)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W, X, Z_plot)
    fig.show()

    #################
    #post processing#
    #################
    ##calc the probability of the price falling within each chunk (this is basically the area under the graph for a given chunk. Using small chunks means you are essentially calculating the area like a rectangle)
    prob_wxy = w_step*x_step*y_step*Z

    ##calc the weighted price of each block - price is calculated like a weighted average because each price has a weight depending on the prob of the chunk.
    ##weighted average is done in 3 parts 1. weight price, sum prices in a given block, divide by prob of given block.
    ###part (a) of weighted average
    weighted_w_price_wxy = prob_wxy * w[:,na,na]
    weighted_x_price_wxy = prob_wxy * x[:,na]
    weighted_y_price_wxy = prob_wxy * y

    ##add chunks together to summarize into fewer bigger blocks
    n_blocks = 2 # number of chunks each variable is broken into. The number of price states = n_variable^n_states
    prob_blocks = np.zeros([n_blocks, n_blocks, n_blocks])
    w_price_blocks = np.zeros([n_blocks, n_blocks, n_blocks])
    x_price_blocks = np.zeros([n_blocks, n_blocks, n_blocks])
    y_price_blocks = np.zeros([n_blocks, n_blocks, n_blocks])
    for w_chunk in range(n_blocks):
        w_start = int(w_chunk * n_chunks/n_blocks) #start chunk
        w_end = int(w_start + n_chunks/n_blocks) #end chunk
        for x_chunk in range(n_blocks):
            x_start = int(x_chunk * n_chunks/n_blocks) #start chunk
            x_end = int(x_start + n_chunks/n_blocks) #end chunk
            for y_chunk in range(n_blocks):
                y_start = int(y_chunk * n_chunks/n_blocks)
                y_end = int(y_start + n_chunks/n_blocks) #end chunk
                ###sum the prob of each chunk in the block
                prob_blocks[w_chunk,x_chunk,y_chunk] = np.sum(prob_wxy[w_start:w_end, x_start:x_end, y_start:y_end])
                ###sum the price of each chunk in the block then scale by the total prob of the block (part (b) of weighted average)
                w_price_blocks[w_chunk,x_chunk,y_chunk] = np.sum(weighted_w_price_wxy[w_start:w_end, x_start:x_end, y_start:y_end])
                x_price_blocks[w_chunk,x_chunk,y_chunk] = np.sum(weighted_x_price_wxy[w_start:w_end, x_start:x_end, y_start:y_end])
                y_price_blocks[w_chunk,x_chunk,y_chunk] = np.sum(weighted_y_price_wxy[w_start:w_end, x_start:x_end, y_start:y_end])
    ###scale the price based on the total prob of each block (part (c) of weighted average) & take exp to convert from log to absolute.
    if method==0:
        w_price = np.exp(w_price_blocks / prob_blocks)
        x_price = np.exp(x_price_blocks / prob_blocks)
        y_price = y_price_blocks / prob_blocks
    elif method==1:
        w_price = w_price_blocks / prob_blocks #dont need exp() because grain price is not skewed when using moving average.
        x_price = x_price_blocks / prob_blocks #dont need exp() because grain price is not skewed when using moving average.
        y_price = y_price_blocks / prob_blocks

    ##convert to c1 by flattening
    prob_c1 = prob_blocks.ravel()
    grain_price_scalar_c1 = w_price.ravel()
    meat_price_scalar_c1 = x_price.ravel()
    wool_price_scalar_c1 = y_price.ravel()

    ##error check
    if np.sum(prob_c1)<0.99:
        raise ValueError('c1 prob doesnt add to 1. This can be because the min or max value used to build the distribution is not wide enough.')
    ##adjust prob so it adds to exactly 1 (if it only adds to 0.995 there is about 6k randomness)
    prob_c1 = prob_c1 / np.sum(prob_c1)

    ##adjust scalars so that weighted average is 1. (the price scalars shouldnt change the average price across all price states)
    grain_price_scalar_c1 = grain_price_scalar_c1 / np.sum(grain_price_scalar_c1 * prob_c1)
    meat_price_scalar_c1 = meat_price_scalar_c1 / np.sum(meat_price_scalar_c1 * prob_c1)
    wool_price_scalar_c1 = wool_price_scalar_c1 / np.sum(wool_price_scalar_c1 * prob_c1)

    ##add z scalar - for now this is just singleton (ie price is the same along z)
    index_z = np.arange(len_z)
    grain_price_scalar_c1z = grain_price_scalar_c1[:,na] * (index_z==index_z)
    meat_price_scalar_c1z = meat_price_scalar_c1[:,na] * (index_z==index_z)
    wool_price_scalar_c1z = wool_price_scalar_c1[:,na] * (index_z==index_z)

    ##convert to df - if the arrays ever become more than 2d possible it would be better to save them as pkl files.
    ## they are saved as excel file so the user can manually change the value or look at the arrays easily.
    grain_price_scalar_c1z = pd.DataFrame(grain_price_scalar_c1z, index=keys_c1, columns=keys_z)
    meat_price_scalar_c1z = pd.DataFrame(meat_price_scalar_c1z, index=keys_c1, columns=keys_z)
    wool_price_scalar_c1z = pd.DataFrame(wool_price_scalar_c1z, index=keys_c1, columns=keys_z)
    prob_c1 = pd.Series(prob_c1, index=keys_c1)


    ##write to xl - it would be good to have it with a named range  and have all the tables in the same sheet so that the read in process could be simplified.
    writer = pd.ExcelWriter('PriceScenarios.xlsx',engine='xlsxwriter')
    grain_price_scalar_c1z.to_excel(writer, sheet_name='grain')
    meat_price_scalar_c1z.to_excel(writer, sheet_name='meat')
    wool_price_scalar_c1z.to_excel(writer, sheet_name='wool')
    prob_c1.to_excel(writer, sheet_name='prob')

    ##finish writing and save
    writer.save()


