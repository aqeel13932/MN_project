import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas.api.types as ptypes
from sklearn import preprocessing
from Scenarios import *
min_max_scaler = preprocessing.MinMaxScaler()
from sklearn.manifold import TSNE

def Normalizedata(data,ids):
    for i in ids:
        x2v = data[i].values.reshape(-1, 1)
        x2v = min_max_scaler.fit_transform(x2v)
        data[i] = x2v
        
def calculate_mean_window(c,vector):
    if vector.shape[0]%c !=0:
        rem = vector.shape[0]%c
        needed = c-rem
        needed = np.zeros(c-rem)
        vector = np.concatenate([vector,needed])
    vector = np.reshape(vector,(c,-1))
    vector = np.mean(vector,axis=0)
    return vector

def seperate_dataset(data):
    return data[2].mean(),data[8].mean(),data[9].mean(),data[10].mean()


def Process_data(data,window):
    times = int(data.shape[0]/window)
    draw_data = np.zeros((times+1,4))
    for i in range(times+1):
        draw_data[i] = seperate_dataset(data[i*window:(i+1)*window])
    return draw_data

def plotdata(data,window,i=None):
    draw_data1 = Process_data(data[data[5]=='Test'],window)
    
    plt.figure(figsize=(12,6))
    plt.plot(draw_data1[:,0],c='green',label='Average reward')
    plt.plot(draw_data1[:,1],c='red',label='Average eaten food')
    plt.plot(draw_data1[:,2],c='blue',label='Stps Hom N')
    plt.plot(draw_data1[:,3],c='purple',label='Stps Hom M')
    plt.xlabel('Sum of {} episode'.format(window))
    plt.ylabel('Count')
    plt.title('Over Testing Episodes,Model:{}'.format(i))
    plt.legend()
    
    draw_data = Process_data(data[data[5]=='train'],window)
    plt.figure(figsize=(12,6))
    plt.plot(draw_data[:,0],c='green',label='Average reward')
    plt.plot(draw_data[:,1],c='red',label='Average eaten food')
    plt.plot(draw_data[:,2],c='blue',label='Stps Hom N')
    plt.plot(draw_data[:,3],c='purple',label='Stps Hom M')
    plt.xlabel('Sum of {} episode'.format(window))
    plt.ylabel('Count')
    plt.title('Over training Episodes, Model:{}'.format(i))
    plt.legend()
    #draw_data(data[data[5]=='train'],window)
    return draw_data,draw_data1


    
def Calculate_TSNE(data,seed=10,n_comp=2,counter=80,ranges=[0,1,2,3,4,5,6,7,8,9]):
    for j in ranges:
        y = TSNE(n_components=n_comp, perplexity=j*5,random_state=seed).fit_transform(data)
        #print(y.shape)

        fig, ax = plt.subplots()
        fig.set_figheight(7)
        fig.set_figwidth(7)
        ax.scatter(x=y[:,0],y=y[:,1])
        for i in range(data.shape[0]):
            #if i ==77:
            #    ax.annotate(str(i), (y[i,0], y[i,1]),color='r')
            #else:
            ax.annotate(str(i), (y[i,0], y[i,1]))
        plt.title('perplixty:{}'.format(j*5))
        #plt.ylim(0,1)
        #plt.xlim(-3,-2)
        #plt.show()
        
def Plot_neurons_activations(neurons,plot_neurons=[0,1,2],merge=False):
    """
    plot specific neurons activations using a provided data set. The function will draw 2x * plots
    each represent a neuron.
    Input:
        neurons: dataset (320,128) (timestep,neuron)
        plot_neurons: a list of neurons to be plotted between [0,127]
        merge: draw all activations on same plot. Default: false.
    """
    if merge:
        plt.figure(figsize=(13,7))
        for i in plot_neurons:
            plt.plot(neurons[:,i],label=i)

        plt.legend(title="Neurons")
        plt.title('neurons:{}'.format(plot_neurons))
        return
    x=13
    neurons_number = len(plot_neurons)
    y=3*neurons_number
    plt.figure(figsize=(x,y))
    counter=1
    for i in plot_neurons:
        ax = plt.subplot(neurons_number,2,counter)
        ax.plot(neurons[:,i])
        counter +=1
        ax.set_title(i)
    plt.tight_layout()
    
def Get_simulations_data(model,simulation):
    return pd.read_csv('output/{}/states_{}.csv'.format(model,simulation),header=None)

def Get_rewards_data(model,simulation):
    return np.load('output/{}/rewards_{}.npy'.format(model,simulation))

def Get_lstm_states(model,simulation):
    return np.load('output/{}/lstm_states_{}.npy'.format(model,simulation))

def Baseline_shadow(figsize=(13,7)):
    sim = Construct_Scenario(Scenarios[0])
    factor=1
    i=0
    count=0
    while i <len(sim):
        if sim[i] =='0':
            count+=1
            i+=1
            continue
        else:
            if count>0:
                start= (i-count)*factor
                ## we used i-1 because the current step is not "1 or morning"
                end = (i-1)*factor+factor
                plt.axvspan(start,end,color='black',alpha=0.1)
                count=0
        i+=1
        
    if count>0:
        start= (i-count)*factor
        ## we used i-1 because the current step is not "1 or morning"
        end = (i-1)*factor+factor
        plt.axvspan(start,end,color='black',alpha=0.1)
        count=0
        
def neuron_activity(model,simulation,cycles=2,day_duration=40,figsize=(13,7),label='',shadow=False):
    data = Get_lstm_states(model,simulation).sum(axis=(2)).mean(axis=0)
    #plt.figure(figsize=figsize)

    plt.plot(data,label=label)
    if not shadow:
        return
    sim = str(simulation)
    factor =10
    if len(sim)<4:
        sim = Construct_Scenario(Scenarios[simulation])
        print(Scenarios_desc[simulation])
        factor=1
    i=0
    count=0
    while i <len(sim):
        if sim[i] =='0':
            count+=1
            i+=1
            continue
        else:
            if count>0:
                start= (i-count)*factor
                ## we used i-1 because the current step is not "1 or morning"
                end = (i-1)*factor+factor
                plt.axvspan(start,end,color='black',alpha=0.1)
                count=0
        i+=1
        
    if count>0:
        start= (i-count)*factor
        ## we used i-1 because the current step is not "1 or morning"
        end = (i-1)*factor+factor
        plt.axvspan(start,end,color='black',alpha=0.1)
        count=0
        
def rose_plot(model,simulation,cycles=2,combined=False,length=80):
    x= Get_simulations_data(model,simulation)
    print('Model:{}'.format(model))
    XH = (x=='XH').values.sum(axis=0)
    EH = (x=='EH').values.sum(axis=0)
    XF = (x=='XF').values.sum(axis=0)
    EF = (x=='EF').values.sum(axis=0)
    sim = str(simulation)

    rose_neurons={'Exit Home':XH,'Enter Home':EH,'Exit Food':XF,'Enter Food Area':EF}
    plt.rcParams['figure.figsize'] = (8, 8)

    bins = int(length/cycles if combined else length)
    factor = 360/bins
    if combined:
        for j in rose_neurons.keys():
            rose_neurons[j] = rose_neurons[j].reshape((cycles,-1)).sum(axis=0)
    #Array with the angles of bins
    Angels = [i*factor for i in range(bins)]
    RadAngels = np.deg2rad(Angels)
    labels = [str(binn) for binn in range(bins)]
    counter=0

    #Loop over the picked neurons
    for j in rose_neurons.keys():

        #Making the plot polar.
        ax = plt.subplot(2,2,1+counter,projection='polar')
        
        #Draw the rose table using matplotlib
        ax.bar(RadAngels,rose_neurons[j],width=2*np.pi/bins,edgecolor='blue', color='None')
        ax.set_xlabel('Time step')
        if combined:
            tick = length/(cycles*8)
        else:
            tick = length/8
            
            
        ax.set_xticklabels([0,tick,'Morning',tick*3,tick*4,tick*5,'Night',tick*7])
            
        ax.set_title('{}'.format(j),pad=15)
        counter+=1

        if not combined:
            step= 360/len(sim)
            #Set the center of the span to mark as night.
            start=step/2
            for mn in range(len(sim)):
                if sim[mn]=='0':
                    ax.bar(np.deg2rad(start),max(rose_neurons[j]),width=np.deg2rad(step),facecolor='black',alpha=0.1)
                start=start+step    
    plt.tight_layout()
    
def rose_plot_compare(model,simulations,cycles=2,combined=False,length=80,condition='EH',names=None):
    rose_neurons={}
    for i in simulations:
        rose_neurons[i]= (Get_simulations_data(model,i)==condition).values.sum(axis=0)
    plt.rcParams['figure.figsize'] = (16, 16)
    bins = int(length/cycles if combined else length)
    factor = 360/bins
    if combined:
        for j in rose_neurons.keys():
            rose_neurons[j] = rose_neurons[j].reshape((cycles,-1)).sum(axis=0)
    #Array with the angles of bins
    Angels = [i*factor for i in range(bins)]
    RadAngels = np.deg2rad(Angels)
    labels = [str(binn) for binn in range(bins)]
    counter=0
    n_bins = cycles*2
    n_bins = [i*(360/n_bins) for i in range(bins)]
    n_nights = 2 if combined else cycles*2
    
    
    #Loop over the picked neurons
    for j in rose_neurons.keys():
        #Making the plot polar.
        ax = plt.subplot(2,3,1+counter,projection='polar')
        #Draw the rose table using matplotlib
        ax.bar(RadAngels,rose_neurons[j],width=2*np.pi/bins,edgecolor='blue', color='None')
        ax.set_xlabel('Time step')
        if combined:
            tick = length/(cycles*8)
        else:
            tick = length/8
            sim=str(j)
            step= 360/len(sim)
            #Set the center of the span to mark as night.
            start=step/2
            
            for mn in range(len(sim)):
                if sim[mn]=='0':
                    ax.bar(np.deg2rad(start),max(rose_neurons[j]),width=np.deg2rad(step),facecolor='black',alpha=0.1)
                start=start+step 
            
            
        ax.set_xticklabels([0,tick,names[j],tick*3,tick*4,tick*5,tick*6,tick*7])
        ax.set_title('{}'.format(j),pad=15)
        counter+=1
    plt.tight_layout()
    return rose_neurons
