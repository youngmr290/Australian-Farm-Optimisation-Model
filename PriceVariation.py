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
which returns the commodity price in each discrete price state. The price state
scalars and the probability of each state are generated from historical prices. A historical price series is
inputted for each commodity and a multivariate normal distribution is fitted to the data. Grain prices are better
represented by a log-normal distribution (e.g. :cite:p:`kingwell1996`) thus before fitting the distribution
grain data undergoes a log transformation. The multivariate normal distribution is then summarised into discrete
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
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import PropertyInputs as pinp

na=np.newaxis
##need to generate price senario scalars with a z axis. To start there will only be scalar for woo, meat and grain (all grains get the same)
len_z = len(pinp.general['i_mask_z'])
len_c1 = 4
keys_z = pinp.general['i_z_idx']
keys_c1 = np.array(['c1_%s' % i for i in range(len_c1)])

##read in CPI adjusted price series
price_data = pd.read_excel("Raw Price Series.xlsx", sheet_name="Python", index_col=0)

##plot to confirm that the relationship is log normal
# fig = px.histogram(price_data, x="APW wheat")
# fig = px.histogram(np.log(price_data), x="APW wheat")
# fig = px.histogram(price_data, x="Mutton")
# fig = px.histogram(np.log(price_data), x="Mutton")
# fig.show()

##select prices used in the distribution and log any that are best fit with log-normal dist 
adj_prices = price_data[["APW wheat","Mutton"]].copy()
adj_prices.loc[:,"APW wheat"] = np.log(adj_prices["APW wheat"]) #log fits the grain data better (as seen by the plots above)
mu = np.mean(adj_prices)
cov = np.cov(adj_prices.T) #rows are variables columns are observations.

#########################################
# summarise raw price into price states #
#########################################
##this is inflexible if you want to add more variables into the distribution
##builds a probability density distribution with 100*100 chunks.
##the min and max value for each axis is determined by the 3rd standard deviation. This should capture 99.9% of the distribution.
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
rv = multivariate_normal(mu, cov)
Z = rv.pdf(pos)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
fig.show()

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
x_price = np.exp(x_price_blocks / prob_blocks)
y_price = y_price_blocks / prob_blocks

##convert to c1 by flattening
prob_c1 = prob_blocks.ravel()
x_price_c1 = x_price.ravel()
y_price_c1 = y_price.ravel()

##error check
if np.sum(prob_c1)<0.995:
    raise ValueError('c1 prob doesnt add to 1. This can be because the min or max value used to build the distribution is not wide enough.')
##adjust prob so it adds to exactly 1 (if it only adds to 0.995 there is about 6k randomness)
prob_c1 = prob_c1 / np.sum(prob_c1)

##convert to a price state scalar - divide by average price
grain_price_scalar_c1 = x_price_c1 / np.sum(x_price_c1 * prob_c1)
meat_price_scalar_c1 = y_price_c1 / np.sum(y_price_c1 * prob_c1)

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


