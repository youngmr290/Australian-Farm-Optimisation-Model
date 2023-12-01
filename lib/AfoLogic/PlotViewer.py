import numpy as np
import matplotlib.pyplot as plt
import pickle as pl
from . import Functions as fun
from . import relativeFile


def read_spreadsheet():
    pv_xl_path = relativeFile.findExcel("PlotViewer.xlsm")
    plot_inp = fun.xl_all_named_ranges(pv_xl_path, "Parameters")
    plot_axes = plot_inp["Axes"]
    dimensions = plot_inp["Dimensions"]
    verticals = plot_inp["Verticals"]
    yvar1 = plot_axes.iloc[0][0]
    yvar2 = plot_axes.iloc[0][4]
    xlabels = plot_axes.iloc[0][8]
    wvar = plot_axes.iloc[0][12]
    xvar = plot_axes.iloc[0][16]

    if yvar2 is None or yvar2 != yvar2: yvar2 = yvar1
    if xlabels is None or xlabels != xlabels: xlabels = yvar1
    if wvar is None or wvar != wvar : wvar = yvar1
    if xvar is None or xvar != xvar : xvar = yvar1

    return yvar1, yvar2, xlabels, wvar, xvar, plot_axes, dimensions, verticals


def create_plots(yvar1, yvar2, xlabels, wvar, xvar, plot_axes, dimensions, verticals):
    yvar1, yvar2, wvar, xvar = np.broadcast_arrays(yvar1, yvar2, wvar, xvar)
    # yvar1, yvar2, xlabels, wvar, xvar = np.broadcast_arrays(yvar1, yvar2, xlabels, wvar, xvar);

    for datacol in dimensions.itertuples():  #for columns in each row
        try:                        #check if file exists
            f = open(datacol[23])
            f.close()
        except IOError:             #process row
            yvar1sliced = yvar1
            yvar2sliced = yvar2
            xlabelssliced = xlabels
            wvarsliced = wvar
            xvarsliced = xvar
            # label_list = []         #for plot labels
            agg_list = []           #for aggregation
            #read columns from current row
            for i in range (24,45):                     #process columns for each row in dataframe
                if datacol[i] != "``.":                 #data exists for column
                    component = datacol[i].split('`')   #split column into components

                    # label_list.append(component[0])     #extract label component from column
                    #extract aggregation component from column
                    # if component[0] == "": agg_list.append("l")            #leave axis
                    # else: agg_list.append(component[1])
                    agg_list.append(component[1])
                    if agg_list[-1] == "x" and np.array_equal(xlabels, yvar1):  # xlabels not set and processing x axis
                        xlabelssliced = np.arange(0, yvar1sliced.shape[len(agg_list)-1], 1)  # use sequential numbers for labels

                    #extract slicing component from column
                    slice_list = []
                    component = component[2].split('.') #split off trailing .
                    component = component[0].split(',') #split by comma - must be csv
                    for j in range(len(component)):
                        if component[j] != "":
                            slice_list.append(int(component[j]))

                    # delete unneeded slices
                    slice_list.sort(reverse=True)       #reverse the list and delete from last slice
                    curr_axis = len(agg_list)-1         #length of list gives the number of axes read so far
                    for slice in range(yvar1sliced.shape[curr_axis]-1,-1,-1):   #process deletion backwards to maintain index
                        if slice_list == []:  # blank means keep all slices
                            break
                        elif slice not in slice_list:                             #slice not listed for keeping
                            yvar1sliced = np.delete(yvar1sliced, slice, curr_axis)  #delete slice on current axis
                            yvar2sliced = np.delete(yvar2sliced, slice, curr_axis)  #delete slice on current axis
                            wvarsliced = np.delete(wvarsliced, slice, curr_axis)    #delete corresponding weightings
                            xvarsliced = np.delete(xvarsliced, slice, curr_axis)    #delete corresponding weightings
                            if agg_list[-1]=="x":                                   #if currently slicing x axis
                                xlabelssliced = np.delete(xlabelssliced, slice, 0)  #delete corresponding labels

            #process aggregation now that all slicing is complete
            for i in range(len(agg_list)-1,-1,-1):  #process dimensions backwards to maintain index
                if agg_list[i] == "a":
                    yvar1sliced = np.average(yvar1sliced, i)
                    yvar2sliced = np.average(yvar2sliced, i)
                    xvarsliced = np.average(xvarsliced, i)
                    wvarsliced = np.average(wvarsliced, i)
                    del agg_list[i]         #list is used to indicate how many dimensions exist, so this accounts for the lost dimension
                elif agg_list[i] == "s":
                    yvar1sliced = np.sum(yvar1sliced, i)
                    yvar2sliced = np.sum(yvar2sliced, i)
                    xvarsliced = np.average(xvarsliced, i)
                    wvarsliced = np.sum(wvarsliced, i)
                    del agg_list[i]         #list is used to indicate how many dimensions exist, so this accounts for the lost dimension
                elif agg_list[i] == "w":
                    yvar1sliced = np.ma.average(yvar1sliced, i, wvarsliced)
                    yvar2sliced = np.ma.average(yvar2sliced, i, wvarsliced)
                    xvarsliced = np.ma.average(xvarsliced, i, wvarsliced)
                    del agg_list[i]                 #list is used to indicate how many dimensions exist, so this accounts for the lost dimension

            #plot

            #vertical lines
            for col in verticals.itertuples():  # for columns in each row
                if col[1] == col[1]: plt.axvline(x=col[1], label =col[2], c=col[3])  # if a value is set plot a vertical line
                # if col[1] is not None and col[1] == col[1]: plt.axvline(x=col[1], label =col[2], c=col[3])  # vertical lines

            line_colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "pink"]

            # x axis
            plt.xticks(rotation=30, ha='right')
            if np.array_equal(xvar, yvar1): #line plot
                #reshape
                i = agg_list.index("x")
                yvar1sliced = np.moveaxis(yvar1sliced,i,-1)
                yvar1sliced = yvar1sliced.reshape(int(np.prod(yvar1sliced.shape)/yvar1sliced.shape[-1]),yvar1sliced.shape[-1])
                yvar2sliced = np.moveaxis(yvar2sliced,i,-1)
                yvar2sliced = yvar2sliced.reshape(int(np.prod(yvar2sliced.shape)/yvar2sliced.shape[-1]),yvar2sliced.shape[-1])

                try:  # check if start:stop:step is set and process axis
                    foo = plot_axes.iloc[1][8] + plot_axes.iloc[2][8] + plot_axes.iloc[3][8]
                    plt.xlim(xlabelssliced[plot_axes.iloc[1][8]], xlabelssliced[plot_axes.iloc[2][8]])
                    plt.xticks(xlabelssliced[plot_axes.iloc[1][8]: plot_axes.iloc[2][8]: plot_axes.iloc[3][8]])
                except: pass
                for p in range(yvar1sliced.shape[0]):   # plot yvar1
                    plt.plot(xlabelssliced, yvar1sliced[p], color=line_colors[p % (len(line_colors))], label=datacol[45 + p])
                if not np.array_equal(yvar1, yvar2):    # only ploy yvar2 if != yvar1
                    for p in range(yvar2sliced.shape[0]):
                        plt.plot(xlabelssliced, yvar2sliced[p], color=line_colors[p % (len(line_colors))], label=datacol[45 + p])

            else:   #scatter plot
                try:  # check if start:stop:step is set and process axis
                    foo = plot_axes.iloc[1][16] + plot_axes.iloc[2][16] + plot_axes.iloc[3][16]
                    plt.xlim(plot_axes.iloc[1][16], plot_axes.iloc[2][16])
                    plt.xticks(np.arange(plot_axes.iloc[1][16], plot_axes.iloc[2][16], step=plot_axes.iloc[3][16]))
                except: pass
                plt.scatter(xvarsliced, yvar1sliced, s=2, label=plot_axes.iloc[0][0], color=line_colors[0])
                # line of best fit and rsquared
                clofb = np.polyfit(xvarsliced.flatten(), yvar1sliced.flatten(), 1)  #get coefficients of line of best fit
                xlofb = np.array([xvarsliced.min(), xvarsliced.max()])              #find min and max x
                ylofb = np.array([clofb[1]+clofb[0]*xvarsliced.min(), clofb[1]+clofb[0]*xvarsliced.max()])  #calculate corresponding y
                rsquared = np.corrcoef(xvarsliced.flatten(), yvar1sliced.flatten())[0,1] ** 2  #calculate rsquared
                plt.plot(xlofb, ylofb, color="black", label="%.4f rsq" % rsquared)  #plot lobf and label with rsquared
                if not np.array_equal(yvar1, yvar2):    #only plot yvar2 if != yvar1
                    plt.scatter(xvarsliced, yvar2sliced, s=2, color=line_colors[1], label=plot_axes.iloc[0][16])
                    clofb = np.polyfit(xvarsliced.flatten(), yvar2sliced.flatten(), 1)  #get coefficients of line of best fit
                    xlofb = np.array([xvarsliced.min(), xvarsliced.max()])              #find min and max x
                    ylofb = np.array([clofb[1]+clofb[0]*xvarsliced.min(), clofb[1]+clofb[0]*xvarsliced.max()])  #calculate corresponding y
                    rsquared = np.corrcoef(xvarsliced.flatten(), yvar2sliced.flatten())[0, 1] ** 2  #calculate rsquared
                    plt.plot(xlofb, ylofb, color="gray", label="%.4f rsq" % rsquared)   #plot lobf and label with rsquared

            # y axis
            try:  # check if start:stop:step is set and process axis
                foo = plot_axes.iloc[1][0] + plot_axes.iloc[2][0] + plot_axes.iloc[3][0]
                plt.ylim(plot_axes.iloc[1][0], plot_axes.iloc[2][0])
                plt.yticks(np.arange(plot_axes.iloc[1][0], plot_axes.iloc[2][0], step=plot_axes.iloc[3][0]))
            except:
                pass

            #misc
            plt.title(datacol[23])
            plt.legend(loc='center left', bbox_to_anchor=(-0.2, 1))
            plt.savefig(datacol[23])
            #pl.dump(fig_handle, open(datacol[23]+".pkl", 'wb'))
            plt.show()
            plt.close()
            return  # only creates one plot and then reads spreadsheet again for any changes

    # raise exception to ask if user wants to continue
    raise UserWarning("No more rows to plot!")