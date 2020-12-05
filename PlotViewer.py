import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pl
import Functions as fun

def read_spreadsheet():
    plot_inp = fun.xl_all_named_ranges("PlotViewer.xlsm", "Parameters")
    plot_axes = plot_inp["Axes"]
    dimensions = plot_inp["Dimensions"]
    yvar = plot_axes.iloc[0][0]
    yvar2 = yvar                            #this line needs deleting when coding for yvar2
    yvar2 = plot_axes.iloc[0][5]
    xvar = plot_axes.iloc[0][10]
    wvar = plot_axes.iloc[0][15]
    return yvar, yvar, xvar, wvar, plot_axes, dimensions

def create_plots(yvar, yvar2, xvar, wvar, plot_axes, dimensions):
    replot = True   #control variable to process spreadsheet data
    while replot:
        plot_created = False
        for datacol in dimensions.itertuples():  #for columns in each row

            try:                        #check if file exists
                f = open(datacol[23])
                f.close()
            except IOError:             #process row
                yvarsliced = yvar
                xvarsliced = xvar
                wvarsliced = wvar
                label_list = []                #for plot labels
                avg_list = []                  #for averaging
                slice_list = []                #for slicing


                #read columns from current row
                for i in range (24,45):                     #process columns for each row in dataframe
                    if datacol[i] != "``.":                 #data exists for column
                        component = datacol[i].split('`')   #split column into components

                        #extract label component from column
                        label_list.append(component[0])

                        #extract averaging component from column
                        if component[0] == "":
                            avg_list.append("l")            #leave axis
                        else:
                            avg_list.append(component[1])

                        #extract slicing component from column
                        slice_list = []
                        component = component[2].split('.') #split off trailing .
                        component = component[0].split(',') #split by comma - must be csv
                        for j in range(len(component)):
                            if component[j] != "":
                                slice_list.append(int(component[j]))

                        # delete unneeded slices
                        slice_list.sort(reverse=True)     #reverse the list and delete from last slice
                        curr_axis = len(avg_list)-1       #length of list gives the number of axes read so far

                        if slice_list != []:                                            #blank means keep all slices
                            for slice in range(yvarsliced.shape[curr_axis]-1,-1,-1):   #from last slice to first in current axis
                                    if slice not in slice_list:                         #slice not listed for keeping
                                        yvarsliced = np.delete(yvarsliced, slice, curr_axis)  #delete slice on current axis
                                        if avg_list[-1]=="x":                   #if currently slicing x axis
                                            xvarsliced = np.delete(xvarsliced, slice, 0)  #delete corresponding labels
                                        elif avg_list[-1]=="w":                 #if currently slicing w axis
                                            wvarsliced = np.delete(wvarsliced, slice, 0)  #delete corresponding weightings

                #process averaging now that all slicing is complete
                for i in range(len(avg_list)-1,-1,-1):  #process dimensions backwards
                    if avg_list[i] == "a":
                        yvarsliced = np.average(yvarsliced, i)
                        del avg_list[i]         #to account for lost dimension
                    elif avg_list[i] == "w":
                        yvarsliced = np.average(yvarsliced, i, wvarsliced)
                        del avg_list[i]         #to account for lost dimension

                #reshape for plotting
                for i in range(len(avg_list)):
                    if avg_list[i] == "x":        # if x axis move axis to end and reshape for plotting
                        lastaxis = len(avg_list)-1
                        yvarsliced = np.moveaxis(yvarsliced,i,lastaxis)
                        lastaxislen = yvarsliced.shape[lastaxis]
                        yvarsliced = yvarsliced.reshape(int(np.prod(yvarsliced.shape)/lastaxislen),lastaxislen)

                #plot
                line_colors = ["orange", "purple", "pink", "b", "g", "r", "c", "m", "y", "k"]
                fig_handle = plt.figure()
                plt.title(datacol[23])
                plt.xlabel("X axis")
                plt.ylabel("Y axis")
                try:  # check for numbers
                    foo = plot_axes.iloc[1][0] + plot_axes.iloc[2][0] + plot_axes.iloc[3][0]
                    plt.ylim(plot_axes.iloc[1][0], plot_axes.iloc[2][0])
                    plt.yticks(np.arange(plot_axes.iloc[1][0], plot_axes.iloc[2][0], step=plot_axes.iloc[3][0]))
                except:
                    pass
                for p in range(yvarsliced.shape[0]):
                    plt.plot(xvarsliced, yvarsliced[p], color = line_colors[p%(len(line_colors))], label=datacol[45+p])
                plt.legend()
                plt.savefig(datacol[23])
                #pl.dump(fig_handle, open(datacol[23]+".pkl", 'wb'))
                plt.show()
                plt.clf()
                plt.close()

                #once plotted check for new items by rereading and breaking for loop
                yvardiscard, yvar2discard, xvardiscard, wvardiscard, plot_axes, dimensions = read_spreadsheet()  # this line only used dimensions df
                plot_created = True
                break

        #get input and read spreadsheet again if replotting
        if not plot_created:
            replot = input( "Enter 1 to replot: ")
        if replot:
            yvardiscard, yvar2discard, xvardiscard, wvardiscard, plot_axes, dimensions = read_spreadsheet() #this line only used dimensions df