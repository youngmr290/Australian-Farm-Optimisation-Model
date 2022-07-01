Simulation integration
=======================

Currently in AFO the crop and pasture modules do not represent the plant life cycle at a biological level (sun, leaf area and soil properties).
Rather, the modules require inputs of a higher level.

For example:

    - The crop module requires the user to specify the yield obtained for a given crop on a given soil type with a certain
      level of fertiliser and chemical application.
    - The pasture module requires inputs which specify the growth curve at different FOO levels throughout the year.

.. note::
   AFO does represent livestock at a detailed biological level and thus already captures the detail included in
   a simulation model.

Generating inputs for AFO's pasture and crop modules can be challenging particularly when you are examining
unknown scenarios. To aid with this process detailed simulation models that represent plant life cycle at a biological
level can be used. Both AusFarm :cite:p:`Moore2007` and APSIMX :cite:p:`Holzworth2018` have previously been used for this.
Both models can represent the whole
farm system and capture a similar level of detail however, APSIMX is the newest of the two models and it is
being actively developed. Thus, APSIMX is the recommended tool.
APSIM is a detailed whole farm simulation model that can be used to generate inputs for AFO.
Full documentation can be found here: https://apsimnextgeneration.netlify.app/.



Pasture and crop input generation
---------------------------------
Here we provide information/guidance on generating crop and pasture inputs using simulation modeling.
This is a somewhat advanced topic and requires an understanding of the pasture structure in AFO.

Pasture
^^^^^^^
What is required:

    - Final FOO and pasture consumed during each AFO feed period and how it is affected by FOO at
      the start of the period and the intensity of grazing during the period.
    - DMD of the pasture consumed and the how it varies with initial FOO and intensity of grazing.
    - Pasture germination in each rotation.

Possible method:

The act of grazing can be simulated without actually including livestock in the simulation. Thus, the
simulating can be done using just pasture paddock/s. In AFO, each feed period has a range of starting FOO and a
range of grazing intensities. The simulation modelling is used to calculate the corresponding end FOO, the quantity
of feed consumed, and the quality of feed consumed.

Generating the starting FOO (on the date when the AFO feed period starts) in the simulation model is
complicated by the fact that the path taken to get to the starting FOO will impact future productivity.
For example, a pasture that was deferred and then grazed heavily to get to a low starting FOO would have
a bigger root system, and hence greater future productivity than a pasture that was consistently lightly
grazed to get to a low starting FOO. To most accurately reflect the typical farming system it is recommended that
the pasture is consistently grazed at the required intensity such that it reaches the starting FOO target.
The level of grazing required to reach the start FOO in each AFO feed period is unknown, therefore
multiple grazing intensities must be simulated (either by using multiple paddock or using a factorial
simulation design). Once the AFO feed period start date is reached the grazing level can be altered to
reflect the levels used in AFO for the given period. This process must be repeated for each AFO feed period.

The process above will generate all the required results but they will need to extracted.
All the trial results where the starting FOO in the simulation did not match the starting FOO in the AFO
feed period can be disregarded. From the remaining
results the end FOO, feed consumed and feed DMD for the AFO period need to be extracted. This process will
be completed outside of the simulation model (e.g. in Python or excel).

Cropping
^^^^^^^^^
What is required:

    - For each phase (landuse with history):

        - Yield
        - Herbicide and fungicide applications
        - Fertiliser application

Possible method:

Rotation inputs have previously been calibrated using farmer data or data from farm consultants, however,
these methods are limiting when rotations included in AFO are novel or a large number are being included.
Simulation modeling also makes it easier to capture seasonal effects. Each rotation included in AFO can be
simulated over a 30-50 year time period. The simulation needs to include a paddock per year in the rotation
(e.g. if the rotations are 6yrs long, 6 paddocks need to be included in the simulation) so that the outputs
are generated for each year. The results can be saved to an excel file and directly read into AFO.

AFO can handle a large number of rotations (>5000). This quickly becomes a big simulation task.
A possibly way to overcome this challenge, as done by previous users, is to develop external code
that interacts with the simulation model changing the required rotation information an then executing
the simulation engine.
To get realistic results using this method requires the simulation to be built with generic
management rules that capture variations in farmer management in each rotation.
For example, fertiliser application is likely to vary depending on rotation because different crops
have different soil interactions (e.g. if legumes are in the rotation, less nitrogen will be required
by following crops due to legumes ability to fix atmospheric nitrogen).

Examples of generic rules:

    - Spray herbicide when weed population reaches x plants/ha. If the application is before seeding then
      the 98% of germinated weeds are killed. If spraying is after seeding only 95% of weeds are killed.
      The assumption being that selective herbicides are not as effective.
    - Apply fertilise so that the resulting soil nutrients are x kg/ha.

The development of these generic rotation rules are best done in collaboration with farm consultants or
agronomists to ensure they are realistic. Furthermore, additional simulation can be conducted to optimise
the rules. For example, to determine the best level of soil nitrogen, a simulation could be run that tests
different levels.



