'''
Generate price scenario scalars.

Price scalars have two main purposes

    #. To account for variations in the price received for a given year due to external market conditions (c1 axis).
    #. To account for variation in prices due to season type.

Notwithstanding weather variation, agricultural commodity price is a major source of farm risk
and must be represented when including farmer risk attitude. For example, if grain price is more variable
than livestock prices, it may be optimal for a risk averse farmer to have a higher livestock focus
because it will reduce variation in the level of income. Price variation is
approximated in AFO using user defined number of price states for meat, wool and grain.
The price states were determined from a multivariate price distribution so that correlation between commodities
was captured.


Generation of discrete price states:

The need to form discrete approximations of a continuous distributions is a necessary requirement for developing a
discrete stochastic programming model of farm management responses to price and weather-year states.
By their nature discrete stochastic programming models cannot consider
all possible price states as described by continuous distributions.  Rather continuous variables such as price
need to be approximated by discrete states.

In AFO the user inputs the average price for each commodity. This is then adjusted by a price state scalar
which returns the price of each commodity in each discrete price state. The price state
scalars and their probability are calculated by fitting a multivariate normal distribution to historical price variation
scalars. A multivariate distribution is used so that correlations between commodities are accurately
represented in the resulting price states. The price variation scalars are calculated using two different methods. Both
methods use historical price data for each commodity.
Note: Grain prices are better represented by
a log-normal distribution (e.g. :cite:p:`kingwell1996`) thus before fitting the distribution grain data undergoes
a log transformation.

For method 1, the weekly price scalars are calculated by dividing the CPI adjusted historical prices by the
average price for the series. For method 2, the weekly price scalars are calculated by dividing the CPI adjusted historical
prices by the medium term moving average. A moving average is used to detrend the price series (remove long term price trends).
The logic behind method 2 is that the price states represented in AFO serve the purpose of capturing yearly price
variation (i.e variations around the expected price for that year). Thus, including long term price trends may overestimate
variation within a given year.

The multivariate normal distribution resulting from either method 1 or 2, is then summarised into discrete
states by dividing up the probability density distribution and calculating the probability (area under the curve)
and weighted average price of each section of the distribution.
The price at each point is compared to the average to determine the magnitude
of the scalar.

To reduce model size and simplify input calibration all meat classes (lamb, shipper,
mutton, etc) received the same meat price scalar. The same thing happen for classes of wool and types of grain.
This simplification should not compromise the accuracy of the results because subclasses of a given commodity
tend to have a high correlation (e.g. between 2000 and 2021 the correlation between light lamb and mutton was 96%).
A further simplification was not to include price variation
for input costs because input costs tend to vary less and therefore the additional model size
was not justified. The resulting assumptions are that all animal classes are 100% correlated,
all wool microns are 100% correlated, all grains are 100% correlated and all input commodities have no variation.
If these assumptions become limiting it is possible to add the extra detail in the price generation.

The c1 axis is averaged for both the asset constraint and the working capital. This saves space without loosing
much/any information.

'''
import os
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import PropertyInputs as pinp

na=np.newaxis

##do this so the module doesnt run when building the docs
if __name__=="__main__":
    ##########
    ##inputs #
    ##########
    len_z = len(pinp.general['i_mask_z'])
    len_c1 = 4
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
        adj_prices = price_data[["APW wheat","Mutton"]].div(price_data[["APW wheat","Mutton"]].mean()).copy()
        adj_prices.loc[:,"APW wheat"] = np.log(adj_prices["APW wheat"]) #log fits the grain data better (as seen by the plots above)
        ##graph to check normal
        # fig = px.histogram(adj_prices, x="APW wheat")
        # fig.show()
    elif method==1:
        ##detrend raw prices using moving average.
        adj_prices = price_data.rolling(window=52).mean().div(price_data).dropna()
        adj_prices = adj_prices[["APW wheat","Mutton"]].copy()
        ##graph to check normal
        # fig = px.histogram(adj_prices, x="APW wheat")
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
    x_min = adj_prices["APW wheat"].mean() - 3 * adj_prices["APW wheat"].std()
    x_max = adj_prices["APW wheat"].mean() + 3 * adj_prices["APW wheat"].std()
    x = np.linspace(x_min, x_max, n_chunks)
    x_step = x[1]-x[0]
    y_min = adj_prices["Mutton"].mean() - 3 * adj_prices["Mutton"].std()
    y_max = adj_prices["Mutton"].mean() + 3 * adj_prices["Mutton"].std()
    y = np.linspace(y_min, y_max, n_chunks)
    y_step = y[1]-y[0]
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    Z = rv.pdf(pos)

    ##plot returned values - this only works for 2 variables
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    fig.show()

    #################
    #post processing#
    #################
    ##calc the probability of the price falling within each chunk (this is basically the area under the graph for a given chunk. Using small chunks means you are essentially calculating the area like a rectangle)
    prob_xy = x_step*y_step*Z

    ##calc the weighted price of each block - price is calculated like a weighted average because each price has a weight depending on the prob of the chunk.
    ##weighted average is done in 3 parts 1. weight price, sum prices in a given block, divide by prob of given block.
    ###part (a) of weighted average
    weighted_x_price_xy = prob_xy * x[:,na]
    weighted_y_price_xy = prob_xy * y

    ##add chunks together to summarize into fewer bigger blocks
    n_blocks = 2 # number of chunks each variable is broken into. The number of price states = n_variable^n_states
    prob_blocks = np.zeros([n_blocks, n_blocks])
    x_price_blocks = np.zeros([n_blocks, n_blocks])
    y_price_blocks = np.zeros([n_blocks, n_blocks])
    for x_chunk in range(n_blocks):
        x_start = int(x_chunk * n_chunks/n_blocks) #start chunk
        x_end = int(x_start + n_chunks/n_blocks) #end chunk
        for y_chunk in range(n_blocks):
            y_start = int(y_chunk * n_chunks/n_blocks)
            y_end = int(y_start + n_chunks/n_blocks) #end chunk
            ###sum the prob of each chunk in the block
            prob_blocks[x_chunk,y_chunk] = np.sum(prob_xy[x_start:x_end, y_start:y_end])
            ###sum the price of each chunk in the block then scale by the total prob of the block (part (b) of weighted average)
            x_price_blocks[x_chunk,y_chunk] = np.sum(weighted_x_price_xy[x_start:x_end, y_start:y_end])
            y_price_blocks[x_chunk,y_chunk] = np.sum(weighted_y_price_xy[x_start:x_end, y_start:y_end])
    ###scale the price based on the total prob of each block (part (c) of weighted average) & take exp to convert from log to absolute.
    if method==0:
        x_price = np.exp(x_price_blocks / prob_blocks)
        y_price = y_price_blocks / prob_blocks
    elif method==1:
        x_price = x_price_blocks / prob_blocks #dont need exp() because grain price is not skewed when using moving average.
        y_price = y_price_blocks / prob_blocks

    ##convert to c1 by flattening
    prob_c1 = prob_blocks.ravel()
    grain_price_scalar_c1 = x_price.ravel()
    meat_price_scalar_c1 = y_price.ravel()

    ##error check
    if np.sum(prob_c1)<0.995:
        raise ValueError('c1 prob doesnt add to 1. This can be because the min or max value used to build the distribution is not wide enough.')
    ##adjust prob so it adds to exactly 1 (if it only adds to 0.995 there is about 6k randomness)
    prob_c1 = prob_c1 / np.sum(prob_c1)

    ##adjust scalars so that weighted average is 1. (the price scalars shouldnt change the average price across all price states)
    grain_price_scalar_c1 = grain_price_scalar_c1 / np.sum(grain_price_scalar_c1 * prob_c1)
    meat_price_scalar_c1 = meat_price_scalar_c1 / np.sum(meat_price_scalar_c1 * prob_c1)

    ##add z scalar - for now this is just singleton (ie price is the same along z)
    index_z = np.arange(len_z)
    grain_price_scalar_c1z = grain_price_scalar_c1[:,na] * (index_z==index_z)
    meat_price_scalar_c1z = meat_price_scalar_c1[:,na] * (index_z==index_z)
    wool_price_scalar_c1z = meat_price_scalar_c1z #todo hook this up

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


