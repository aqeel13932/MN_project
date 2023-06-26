# MN_project
This is the code base for "Emergence of Adaptive Circadian Rhythms in Deep Reinforcement Learning".

## Base code
First you need to clonse this repository
`
git clone https://github.com/aqeel13932/MN_project.git
`

`
cd MN_project
`

Clone the environment repository

`
git clone https://github.com/aqeel13932/APES.git
`

The required packages can be installed via conda:

`
conda env create -f mn_keras.yml
`

## Training
For training you need to use `morning_night.py` for full environment input or `morning_night_minimized.py` They both accept the same arguements. 
For example: 
`
python morning_night.py 1 --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed 1337 --batch_size 16 --totalsteps 6000000 --details "sample training" --rwrdschem 0 1 -2.5 --clue --max_timesteps 160 

`

`
python morning_night_minimized.py 2 --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed 1337 --batch_size 16 --totalsteps 6000000 --details "sample training" --rwrdschem 0 1 -2.5 --clue --max_timesteps 160 
`

