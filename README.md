# dressmaker
Create new images with properties like #theDress. Content:

- dressmaker.m: Matlab function that maps the color distribution of new images onto the major hue dimension (principal component of chromaticities) of #theDress; according to the study, the resulting images yield individual differences similar to #theDress. Run without input to see the example from Figure 1 (for this, dressmaker.m needs to be in the same folder as image.mat).

- image.mat: A Matlab data file that contains the original images and masks that distinguish the objects in the images from their background. The images and their mask are used as input to dressmaker.m to create the stimulus images used in the study.

- example_dressmaking.m: Matlab script that illustrates how dressmaker.m can be used with the 5 new images used in the study and provided by images.mat. Of course, the function can be used with still other images...

For data and survey see https://doi.org/10.5281/zenodo.3659308
