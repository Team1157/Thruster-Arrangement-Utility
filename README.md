# Thruster-Arrangement-Utility
TAU is a tool for generating 3d surface plots indicating max thrust at zero torque of a group of thrusters

To use, run tau.py with a thruster layout JSON file in your working directory. The program should output a 3d surface indicating your maximum thrust in each direction that produces zero torque.

This program was initially created with the design of underwater ROVs in mind, but could easily be expanded to many other things.

Note that the coordinate system for this program uses aircraft-style coordinates rather than conventional ones. ![image](https://user-images.githubusercontent.com/43499473/129992017-ad34299f-88f0-4ae0-800b-cbe1d22d72d5.png)

TODO:
 - add mode for max torque output at zero thrust
 - add a GUI
 - make the output surface less jagged
 - add a way to show individual thruster values at a given point