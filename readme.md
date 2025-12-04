{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11740\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Finite-Difference-Method-Cosserat-Rod-with-Free-Free Boundary Conditions\
A simple python implementation of a finite difference simulation of Cosserat rod dynamics for soft robotic locomotion\
\
\
\
## Free-Free Boundary Conditions\
\
This is a simple implementation of a finite difference method simulation of the dynamics of a Cosserat rod based on notes found in the additionalMaterials folder. \
\
The rod dynamics may be simulated in the test_free_free.py \
\
The physical parameters of the rod may be modified in the Cosserat_Rod class. The simulator works for Young\'92d modulus greater that 5e6\
\
\
## TODO\
 - More robust solver for time stepping\
 - Add actuation forces\
 - Fix solver for more elastic materials\
\
\
## Requirements:\
    numpy, matplotlib, scipy}