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


.. toctree::
   :maxdepth: 1

   PropertyInputs
   StructuralInputs
   UniversalInputs


