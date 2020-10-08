# We plot only the first record for each model.

#get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import listdir
from os.path import isfile, join
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--neuron',type=int,default=0)
parser.add_argument('--simulation',type=int,default=0)
parser.add_argument('--plot',action='store_true')
parser.add_argument('--depth',type=int,default=1)
parser.add_argument('--day',type=int,default=3)
parser.add_argument('--mean',action='store_true')
args = parser.parse_args()
def Get_LSTM_Files(model,simulation):
    episode=100
    lstm_files = []
    while isfile('output/{}/PRC_data/lstm_states_{}_model_eps:{}.h5.npy'.format(model,simulation,episode)):
        #lstm_files.append('lstm_states_{}_model_eps:{}.h5.npy'.format(simulation,episode))
        lstm_files.append(episode)
        episode+=100
    return lstm_files

def calculate_PRC_mean(model,simulation,neuron,version=None,mean=True,depth=1):
    if version is not None:
        x = np.load('output/{}/PRC_data/lstm_states_{}_model_eps:{}.h5.npy'.format(model,simulation,version))
    else:
        x = np.load('output/{}/lstm_states_{}.npy'.format(model,simulation))
    if mean:
        x = x.mean(axis=0)
    else:
        x = x[0]
    yt = x[:,neuron]
    
    difference = yt[depth:]-yt[:-depth]
    # Filling the z axis with the model number (100,200,1000 etc) this number represent 
    # the number of training cycles. Every 100 training cycling * 4 the model was saved once.
    if version is not None:
        z= np.zeros(difference.shape[0])
        z.fill(version)
    else:
        z = np.zeros(320)
        for j in range(1,8):
            z[j*40:j*40+40]=j
        z = z[:-1]
        z = z*40
    return yt[depth:],difference,z

def calculate_PRC(model,simulation,neuron):
    x = np.load('output/{}/lstm_states_{}.npy'.format(model,simulation))
    
    yt = x[:,:,neuron]
    print(yt.shape)
    difference = yt[:,1:]-yt[:,:-1]
    
    z = np.zeros((yt.shape[0],320))
    for j in range(1,8):
        z[:,j*40:j*40+40]=j
    z = z[:,:-1]
    z = z*40
    return yt[:,1:],difference,z

def Plot_PRC_versions(model,simulation,day,neuron):
    #Getting available mid-point
    versions = Get_LSTM_Files(model,simulation)
    v = len(versions)
    exp_length = 320-args.depth
    yt = np.zeros((v,exp_length))
    difference = np.zeros((v,exp_length))
    z = np.zeros((v,exp_length))
    number = np.zeros(v)
    for i in range(v):
        yt[i],difference[i],z[i] =calculate_PRC_mean(64,0,neuron,version=versions[i],mean=args.mean,depth = args.depth)
        number[i]= versions[i]
    
    #Generate 8 colors
    colors = plt.cm.jet(np.linspace(0,1,v))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #calculate the start and the end of the day
    # remember that the first day is short due to the substraction y[t]-y[t-depth]
    wday_start = max(0,40*day-args.depth)
    wday_end= 40*(day+1)-args.depth

    for i in range(v):
        if args.plot:
            ax.plot(yt[i,wday_start:wday_end],difference[i,wday_start:wday_end],z[i,wday_start:wday_end],color=colors[i])
        else:
            ax.scatter(yt[i,wday_start:wday_end],difference[i,wday_start:wday_end],z[i,wday_start:wday_end],color=colors[i])
    plt.show()

Plot_PRC_versions(64,args.simulation,args.day,args.neuron)
