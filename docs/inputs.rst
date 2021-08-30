Inputs
=======

AFOs inputs are stored in excel. All input cells in the spreadsheets have their style defined as Input. The
user can then modify this style as they like to highlight these input cells. Currently the style is blue
background & Unlocked. Any input cell that is made unnecessary as a result of a value in another cell has
a conditional formatting which greys the background and the font.
Inputs are split between the inputs that are property specific (Property), inputs that are universal
across properties and regions (Universal) and inputs which affect the structure of the model (Structural).
These names correspond to the spreadsheet names (Property.xlsx, Universal.xlsx and Structural.xlsx) and to
modules in the python code (PropertyInputs.py, UniversalInputs.py and StructuralInputs.py). Within the
spreadsheet the inputs are stored in relevant sheets for example all the price inputs are stored in the
‘Price’ sheet. Each input is given a range name which python uses to locate all the inputs in excel. The
range names are defined local to each sheet (not as workbook level names), this allows different inputs
to have the same name for example the inputs for each pasture type needs to have the same name. The range
names must not be changed because the python model will cause an error if any names cannot be found.

In python the input data is stored inside dictionaries, the advantage is everything is grouped which makes
it simple to locate and consistent to access.

Excel can only contain 2d arrays, in some cases the 2d arrays represent more dimensions and must be
reshaped in python. The reshaping must be done prior to applying the sensitivity because in some cases
the SA is only applied to a given axis. The reshaping occurs in the input.py module straight after
reading in the input.

The inputs can be adjusted by the user through sensitivities (described in the following section).
Allowing variation between the different trials. Note; only a subsection of the structural inputs can be
adjusted by sensitivity. These can be found in the Structural SA sheet. All other structural inputs are
fixed and can not be adjusted.

There are three options for inputs to the model

    #. Typical year - this is user inputs for a “typical” year. Used in the static model.
       These are useful because in some cases the weighted average of the seasonal inputs
       may not represent the typical year. Eg for pasture the weighted average could result
       in a small amount of green pasture early in the year which the model could then defer
       and utilise with high benefit. Although this can happen in some years it may be
       unrealistic to call this the average season.
    #. Weighted average - this is the weighted average of all the season type inputs. Used in the static model
    #. Seasonal inputs - input for each season type. Used for the DSP model.

All inputs with a z axis must be passed through f_seasonal_inp function. This masks the seasons to be included
and takes the weighted average when required. The exception to this is inputs that are used as associations/indexes.
Inputs like this (e.g. pasture_stage and a_r_zida0e0b0xyg) are applied with a full z axis and the resulting array
is then passed through f_seasonal_inp. This is because associations/indexes must be integers so after applying
f_seasonal_inp the result would need to be rounded to an integer. Where the association/index is then applied
will not necessarily result in a correct weighting across seasons.

.. note:: When generating seasonal inputs you need to make sure the inputs are consistent with being able to identify the
    season. i.e. until a season is distinguished it must have the same inputs as other seasons (e.g. two seasons that
    are distinguished by spring must get the same amount of fertiliser at seeding because at seeding spring conditions
    are unknown).

For version 1 only the key inputs (e.g. yield, variable fert requirement, variable chem requirement, pasture growth,
etc) have a season axis in an attempt to reduce complexity and execution time where possible. Season axis can be added to
other inputs in the future if deemed necessary.


.. toctree::
   :maxdepth: 1

   PropertyInputs
   StructuralInputs
   UniversalInputs


