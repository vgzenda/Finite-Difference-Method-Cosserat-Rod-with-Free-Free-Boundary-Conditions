# Finite-Difference-Method-Cosserat-Rod-with-Free-Free Boundary Conditions
A simple python implementation of a finite difference simulation of Cosserat rod dynamics for soft robotic locomotion



## Free-Free Boundary Conditions

This is a simple implementation of a finite difference method simulation of the dynamics of a Cosserat rod based on notes found in the additionalMaterials folder. 

The rod dynamics may be simulated in the test_free_free.py 

The physical parameters of the rod may be modified in the Cosserat_Rod class. The simulator works for Young’d modulus greater that 5e6


## TODO
 - More robust solver for time stepping
 - Add actuation forces
 - Fix solver for more elastic materials


## Requirements:
    numpy, matplotlib, scipy
