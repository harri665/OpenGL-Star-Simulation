to run run final.exe 
or run make then then run final.exe

This is a simulation of a star using real data as simular to this project done in Houdini ( [Here](https://harrison-martin.com/#/projects/8Bvxdn))

i opted to use a vector field stored in a vdb as this is something i aimed to better understand. To impliment VDB it would require using alote more than the scope of this project and likely would not be able to run on yoru computer. The VDB files are converted to bin files so they fall within the scope of this project. 

Colors: 
the colors range in a blackbody from red to blue and represent velocity if a particle is blue its moving very fash and red if its moving very slow. 

FOR A BETTER RESULT INCREASE THE MAX AMOUNT OF PARTICLES to as high as your computer can handle. 

Controls: 

Mouse:  to move camera - Drag zoom etc ... 

SPACE: will pause/play the simulation

TILT: DEPRECATED

, and . : will increase and decrease the field strength respectively

### 1 and 2: will increase and decrease the MAX amount of particles becuase the emission rate is so high by 
default this is basically the same as increasing the emission rate. 

3 and 4: will Increase the emission rate this should only be used if you have a really good computer

7 and 8 Will increase and decrease the star radius - this is fun to test and see how the vector field reacts to different sized stars. ( NOT ACCURATE to real life just fun to play with)

9, 0 mostly for testing resizes the vector veclocity field 

V DEPRECATED 

### U: Toggle the velocity vector display shows the VDB velocity vector

### T: toggle particle trails on or off 

 5 and 6: lets you control the length of the particle trails (MOSTLY FOR TESTING )

### B: IF particle trails are enable this will set the length to be from when that specific particle was spawned so you can see the entire path it took. 

### R: will reset all the particles 

### ESC/Q will kill the program VERY usefull if you increase the simulation beyond your computers limits 
