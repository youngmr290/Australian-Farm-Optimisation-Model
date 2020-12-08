import numpy as np
import matplotlib.pyplot as plt
import pickle as pl
import Functions as fun

def read_spreadsheet():
    plot_inp = fun.xl_all_named_ranges("PlotViewer.xlsm", "Parameters")
    plot_axes = plot_inp["Axes"]
    dimensions = plot_inp["Dimensions"]
    verticals = plot_inp["Verticals"]
    yvar1 = plot_axes.iloc[0][0]
    yvar2 = plot_axes.iloc[0][5]
    xvar = plot_axes.iloc[0][10]
    wvar = plot_axes.iloc[0][15]
    return yvar1, yvar2, xvar, wvar, plot_axes, dimensions, verticals

def create_plots(yvar1, yvar2, xvar, wvar, plot_axes, dimensions, verticals):
    yvar1, yvar2, wvar = np.broadcast_arrays(yvar1, yvar2, wvar);
    #print(xvar.shape, yvar1.shape, yvar2.shape, wvar.shape)

    for datacol in dimensions.itertuples():  #for columns in each row
        try:                        #check if file exists
            f = open(datacol[23])
            f.close()
        except IOError:             #process row
            yvar1sliced = yvar1
            yvar2sliced = yvar2
            xvarsliced = xvar
            wvarsliced = wvar
            label_list = []         #for plot labels
            agg_list = []           #for aggregation
            #read columns from current row
            for i in range (24,45):                     #process columns for each row in dataframe
                if datacol[i] != "``.":                 #data exists for column
                    component = datacol[i].split('`')   #split column into components

                    #extract label component from column
                    label_list.append(component[0])

                    #extract aggregation component from column
                    if component[0] == "":
                        agg_list.append("l")            #leave axis
                    else:
                        agg_list.append(component[1])

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

                    if slice_list != []:                                            #blank means keep all slices
                        for slice in range(yvar1sliced.shape[curr_axis]-1,-1,-1):   #from last slice to first in current axis
                                if slice not in slice_list:                         #slice not listed for keeping
                                    yvar1sliced = np.delete(yvar1sliced, slice, curr_axis)  #delete slice on current axis
                                    yvar2sliced = np.delete(yvar2sliced, slice, curr_axis)  #delete slice on current axis
                                    wvarsliced = np.delete(wvarsliced, slice, curr_axis)    #delete corresponding weightings
                                    if agg_list[-1]=="x":                   #if currently slicing x axis
                                        xvarsliced = np.delete(xvarsliced, slice, 0)  #delete corresponding labels

            #process aggregation now that all slicing is complete
            for i in range(len(agg_list)-1,-1,-1):  #process dimensions backwards
                if agg_list[i] == "a":
                    yvar1sliced = np.average(yvar1sliced, i)
                    yvar2sliced = np.average(yvar2sliced, i)
                    wvarsliced = np.average(wvarsliced, i)
                    del agg_list[i]         #list is used to indicate how many dimensions exist, so this accounts for the lost dimension
                if agg_list[i] == "s":
                    yvar1sliced = np.sum(yvar1sliced, i)
                    yvar2sliced = np.sum(yvar2sliced, i)
                    wvarsliced = np.sum(wvarsliced, i)
                    del agg_list[i]         #list is used to indicate how many dimensions exist, so this accounts for the lost dimension
                elif agg_list[i] == "w":
                    yvar1sliced = np.ma.average(yvar1sliced, i, wvarsliced)
                    yvar2sliced = np.ma.average(yvar2sliced, i, wvarsliced)
                    del agg_list[i]         #list is used to indicate how many dimensions exist, so this accounts for the lost dimension

            #reshape for plotting
            for i in range(len(agg_list)):
                if agg_list[i] == "x":        # if x axis move axis to end and reshape for plotting
                    lastaxis = len(agg_list)-1
                    yvar1sliced = np.moveaxis(yvar1sliced,i,lastaxis)
                    lastaxislen = yvar1sliced.shape[lastaxis]
                    yvar1sliced = yvar1sliced.reshape(int(np.prod(yvar1sliced.shape)/lastaxislen),lastaxislen)
                    yvar2sliced = np.moveaxis(yvar2sliced,i,lastaxis)
                    lastaxislen = yvar2sliced.shape[lastaxis]
                    yvar2sliced = yvar2sliced.reshape(int(np.prod(yvar2sliced.shape)/lastaxislen),lastaxislen)

            #plot
            line_colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "pink"]
            fig_handle = plt.figure()
            plt.title(datacol[23])
            plt.xlabel("X axis")
            plt.ylabel("Y axis")
            try:  # check for numbers in dataframe and set y-axis labels
                foo = plot_axes.iloc[1][0] + plot_axes.iloc[2][0] + plot_axes.iloc[3][0]
                plt.ylim(plot_axes.iloc[1][0], plot_axes.iloc[2][0])
                plt.yticks(np.arange(plot_axes.iloc[1][0], plot_axes.iloc[2][0], step=plot_axes.iloc[3][0]))
            except:
                pass
            try:  # check for numbers in dataframe and set x-axis labels
                foo = plot_axes.iloc[1][10] + plot_axes.iloc[2][10] + plot_axes.iloc[3][10]
                plt.xlim(xvarsliced[plot_axes.iloc[1][10]],xvarsliced[plot_axes.iloc[2][10]])
                plt.xticks(xvarsliced[plot_axes.iloc[1][10] : plot_axes.iloc[2][10] : plot_axes.iloc[3][10]])
            except:
                pass
            plt.xticks(rotation=30, ha='right')
            for p in range(yvar1sliced.shape[0]):
                plt.plot(xvarsliced, yvar1sliced[p], color = line_colors[p%(len(line_colors))], label=datacol[45+p])
            for p in range(yvar2sliced.shape[0]):
                plt.plot(xvarsliced, yvar2sliced[p], color = line_colors[p%(len(line_colors))], label=datacol[45+p])
            for col in verticals.itertuples():  # for columns in each row
                plt.axvline(x=col[1], label =col[2], c=col[3])  # vertical lines
            plt.legend(loc='center left', bbox_to_anchor=(-0.2, 1))
            plt.savefig(datacol[23])
            #pl.dump(fig_handle, open(datacol[23]+".pkl", 'wb'))
            plt.show()
            #plt.clf()
            plt.close()
            return

    # no more row to plot, so raise exception to ask if user wants to continue
    raise UserWarning("No    more rows to plot!")