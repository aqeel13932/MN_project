import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas.api.types as ptypes
from sklearn import preprocessing
from Scenarios import *
from math import sqrt
min_max_scaler = preprocessing.MinMaxScaler()
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from os import listdir
from os.path import isfile, join
from PyPDF2 import PdfFileReader

# def Get_LSTM_Files(model,simulation):
#     '''
#     Return a list of progressive training models where a simulation exist.
#     Input:
#         - model: integer of the model, usually 64
#         - simulation: integer of the simulation. You can find the number in Scenarios.py.
#     output:
#         - List of progressive lstm activation for specific simulation available.
#     '''
#     episode=100
#     lstm_files = []
#     while isfile('output/{}/PRC_data/lstm_states_{}_model_eps:{}.h5.npy'.format(model,simulation,episode)):
#         #lstm_files.append('lstm_states_{}_model_eps:{}.h5.npy'.format(simulation,episode))
#         lstm_files.append(episode)
#         episode+=100
#     return lstm_files

def TSNE_scaled(model,simulation,ranges=[1]):
    '''
    Function to scale activations of the LSTM activation for specific model,simulation and preplixety.
    - Model: integer that specify the model.
    - simulation: integer that specify the simulation.
    - Range: a list of integers that will determine the perplixty (each number will be multipled with 5)
    '''
    
    x = Get_lstm_states(model,simulation)
    #transpose
    x_m = x.mean(axis=0).T
    ## Mean 0,std 1
    x_scaled = scale(x_m,axis=1)
    print(x.shape,x_m.shape,x_scaled.shape)
    Calculate_TSNE(x_scaled,ranges=ranges) 
    
def plot_model_performance(models,Normalized=False,Train=True,Test=True,window=100,wanted_columns=[2,8,9,10]):
    """
    Plot model performance for 
    """
    for i in models:
        x = pd.read_csv('output/{}/exp_details.csv'.format(i),header=None)
        if Normalized:
            Normalizedata(x,wanted_columns)
        plotdata(x,window,i,Train=Train,Test=Test)  
        
def Get_LSTM_Files(model,simulation):
    '''
    Return a list of progressive training models where a simulation exist.
    Input:
        - model: integer of the model, usually 64
        - simulation: integer of the simulation. You can find the number in Scenarios.py.
    output:
        - List of progressive lstm activation for specific simulation available.
    '''
    episode=100
    lstm_files = []
    for episode in range(0,37600):
        if isfile('output/{}/PRC_data/lstm_states_{}_model_eps:{}.h5.npy'.format(model,simulation,episode)):
        #lstm_files.append('lstm_states_{}_model_eps:{}.h5.npy'.format(simulation,episode))
            lstm_files.append(episode)
    return lstm_files

def All_in_one7seeds(simulation,cycles=2,length=80,label=['','{}','',''],combined=False,condition='XF',size=(13,7),xlim=None):
    data1=[]
    data2=[]
    for i in [64,65,66,67,68,69,70]:
        data1.append((Get_simulations_data(i,simulation)==condition).values.sum(axis=0))
        data2.append(Get_lstm_states(i,simulation).sum(axis=(2)).mean(axis=0))
    label[3]=Scenarios_desc[simulation]+' 7 seeds, ' + condition
    data1 = np.array(data1)
    data2 = np.array(data2)
    data1_m = data1.mean(axis=0)
    data2_m = data2.mean(axis=0)
    
    print(data1.shape)
    
    data1_std = data1.std(axis=0)/sqrt(data1.shape[0])
    data2_std = data2.std(axis=0)/sqrt(data2.shape[0])
    
    t = np.arange(data1.shape[1])
    
    fig, ax1 = plt.subplots(figsize=size)

    color = 'tab:red'
    ax1.set_xlabel(label[0])
    ax1.set_ylabel(label[1].format(condition), color=color)
    print(data1_m.shape,data2_m.shape,t.shape)
    ax1.bar(t, data1_m,yerr=data1_std, color=color,alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    sim = str(simulation)
    factor =10
    if len(sim)<3:
        sim = Construct_Scenario(Scenarios[simulation])
        print(Scenarios_desc[simulation])
        factor=1
    ###### Draw dark span over night code ########
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
                ax1.axvspan(start,end,color='black',alpha=0.1,label='night')
                count=0
        i+=1
    ### this is to handle if night in the end.
    if count>0:
        start= (i-count)*factor
        ## we used i-1 because the current step is not "1 or morning"
        end = (i-1)*factor+factor
        ax1.axvspan(start,end,color='black',alpha=0.1,label='night')
        count=0
    ###############################################
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(label[2], color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2_m,color=color)
    ax2.fill_between(t,data2_m+data2_std,data2_m-data2_std,alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(label[3])
    if xlim:
        plt.xlim(xlim)
    else:
        print(data1.shape[0])
        plt.xlim(0,data1.shape[0])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
def All_in_one(model,simulation,cycles=2,label=('','{}','',''),combined=False,condition='XF',size=(13,7),xlim=None,specific_model='',info_in_title=False,tiks_size=8):

    # Create some mock data
    data1= (Get_simulations_data(model,simulation,specific_model)==condition).values.sum(axis=0)
    data2= Get_lstm_states(model,simulation,specific_model).sum(axis=(2)).mean(axis=0)
    data3= Get_rewards_data(model,simulation,specific_model)

 
    all_average = data3.sum(axis=1).mean()
    first4days = data3[:,:160].sum(axis=1).mean()
    last4average = data3[:,160:].sum(axis=1).mean()
    average = ',Avg rewards:(All:{},1st4days:{},last4days:{})'.format(all_average,first4days,last4average)
    data3_std= data3.std(axis=0)/sqrt(data3.shape[0])
    data3 = data3.mean(axis=0)
    t = np.arange(data1.shape[0])

    fig, ax1 = plt.subplots(figsize=size)

    color = 'tab:red'
    ax1.set_xlabel(label[0],fontsize=tiks_size+2)
    ax1.set_ylabel(label[1].format(condition), color=color,fontsize=tiks_size+2)
    ax1.bar(t, data1, color=color,alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    sim = str(simulation)
    factor =10
    if len(sim)<4:
        sim = Construct_Scenario(Scenarios[simulation])
        print(Scenarios_desc[simulation])
        factor=1
    ###### Draw dark span over night code ########
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
                ax1.axvspan(start,end,color='black',alpha=0.1,label='night')
                count=0
        i+=1
    ### this is to handle if night in the end.
    if count>0:
        start= (i-count)*factor
        ## we used i-1 because the current step is not "1 or morning"
        end = (i-1)*factor+factor
        ax1.axvspan(start,end,color='black',alpha=0.1,label='night')
        count=0
    ###############################################
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(label[2], color=color,fontsize=tiks_size+2)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.plot(t, data3, color='tab:green')
    ax2.fill_between(t,data3+data3_std,data3-data3_std,alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)
    if info_in_title:
        plt.title(label[3]+average+',Model:{}'.format(model))
    
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim(0,data1.shape[0])
    
    # Set ticks size
    ax1.tick_params(labelsize=tiks_size)
    ax2.tick_params(labelsize=tiks_size)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.show()
    return ax1,ax2



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

def All_in_one(ax1,model,simulation,label=('','{}','',''),condition='XF',
               xlim=None,specific_model='',info_in_title=False,tiks_size=8,
              plot_char="",x_ticks=True,y1_ticks=True,y2_ticks=True,char_x=1,char_y=20):

    data1= (Get_simulations_data(model,simulation,specific_model)==condition).values.mean(axis=0)
    data2= Get_lstm_states(model,simulation,specific_model).sum(axis=(2)).mean(axis=0)
    data3 = Get_rewards_data(model,simulation).mean(axis=0)
    division = 8
    mround = int(data3.shape[0]/division)
    lst = []
    for i in range(1,division+1):
        start = (i-1)*mround
        end = i*mround
        lst.append(data3[start:end].mean())
        
    t = np.arange(data1.shape[0])

    #fig, ax1 = plt.subplots(figsize=size)

    color = 'blue'
    ax1.plot(t, data2, color=color,lw=1)
    #ax1.plot(np.arange(len(lst))*mround+(mround/2),lst,color='green')
    ax1.set_xlabel(label[0],fontsize=tiks_size+2)
    ax1.set_ylabel(label[1], color=color,fontsize=tiks_size+2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks([0,40,80,120,160,200,240,280,320])
    if not x_ticks:
        ax1.set_xticklabels(["" for _ in range(9)])
        
    ax1.text(char_x,char_y,plot_char,color="k",horizontalalignment="left",verticalalignment="baseline",fontweight="bold")
    sim = str(simulation)

    factor =10
    if len(sim)<4:
        sim = Construct_Scenario(Scenarios[simulation])
        print(Scenarios_desc[simulation])
        factor=1
    ###### Draw dark span over night code ########
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
                ax1.axvspan(start,end,lw=0, color="lightgrey")
                count=0
        i+=1
    ### this is to handle if night in the end.
    if count>0:
        start= (i-count)*factor
        ## we used i-1 because the current step is not "1 or morning"
        end = (i-1)*factor+factor
        ax1.axvspan(start,end,lw=0, color="lightgrey")
        
        count=0
    ###############################################
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'red'
    ax2.set_ylabel(label[2], color=color,fontsize=tiks_size+2)  # we already handled the x-label with ax1

    ax2.bar(t, data1, color=color,width=1)
    ax2.tick_params(axis='y', labelcolor=color)
    if info_in_title:
        plt.title(label[3]+average+',Model:{}'.format(model))
    
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim(0,data1.shape[0])
        
    ax1.set_yticks([-15,-10,-5,0,5,10,15])
    ax2.set_yticks([0,0.2,0.4,0.6,0.8])
    
    # Set ticks size
    ax1.tick_params(labelsize=tiks_size)
    if not y1_ticks:
        ax1.set_yticklabels(["" for _ in range(7)])
    ax2.tick_params(labelsize=tiks_size)
    if not y2_ticks:
        ax2.set_yticklabels(["" for _ in range(5)])

    return ax1,ax2


def postprocess(file):
    
    pdffile = PdfFileReader(open(file, mode='rb'))
    pdfsize = np.array([float(pdffile.getPage(0).mediaBox[2]),
               float(pdffile.getPage(0).mediaBox[3])])
    pdfdim = pdfsize*25.4/72. # points to mm
    print("input plot breite:", pdfdim[0], "mm")
    print("input plot hoehe:", pdfdim[1], "mm")
    
def plot_data_seeds(ax,window,models=[64],wd=[0,1,2,3],c=['g','r','b','purple'],dtype='train',
                   legend=['Reward','#Consumed food','#Steps at home during night',
                           '#Steps at home during morning'],title=''):
    '''
    Plot informatoin about the trianing or testing results of the experiments.
    Inputs: 
        - window: the number of episodes per point.
        - models: the models to be averaged.
        - wd (wanted data): list of numbers for wanted data. (0:reward,1:Consumed food,2:steps at home during night,3:steps at home during morning
        - c: color of each data line.
        - dtype: "train" or "Test" based on wanted data query.
        - legend: data legend.
        - title: plot title.
    output:
        - final: the processed data.
    '''
    for idx,m in enumerate(models):
        #print('Read model {}'.format(m))
        data = pd.read_csv('output/{}/exp_details.csv'.format(m),header=None)
        if idx==0:
            tmp = Process_data(data[data[5]==dtype],window)
            final=np.zeros((len(models),tmp.shape[0],4))
            final[idx] = tmp
            continue

        final[idx] = Process_data(data[data[5]==dtype],window)
    mn = final.mean(axis=0)
    std = final.std(axis=0)#/np.sqrt(final.shape[0])
    

    xvalues = np.arange(0,mn.shape[0])
    for i in wd:
        ax.plot(xvalues,mn[:,i],color=c[i],label=legend[i])
        ax.fill_between(xvalues,mn[:,i]+std[:,i],mn[:,i]-std[:,i],color=c[i],alpha=0.2,
                       label=legend[i]+' std')
    ax.axhline(y=0.0, color='k', linestyle='--',lw=1)
    ax.set_xlabel("Training episodes [thousands]")
    ax.set_ylabel("Average reward")
    #ax.set_xlim(0,episodes_thousands[-1])
    ax.set_ylim(-150,50)  # -186.114, 31.66
    ax.set_xticks([0,50,100,150,200,250,300,350],[0,5,10,15,20,25,30,35])
    ax.set_yticks([-150,-100,-50,0,50])

def plotdata(data,window,i=None,Test=True,Train=True):
    if Test:
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
    if Train:
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
    #return draw_data,draw_data1
    
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
        
def Plot_neurons_activations(model,simulation,plot_neurons=[0,1,2],merge=False):
    """
    plot specific neurons activations using a provided data set. The function will draw 2x * plots
    each represent a neuron.
    Input:
        neurons: dataset (320,128) (timestep,neuron)
        plot_neurons: a list of neurons to be plotted between [0,127]
        merge: draw all activations on same plot. Default: false.
    """
    neurons = Get_lstm_states(model,simulation)
    neurons = neurons.mean(axis=(0))
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
    
def Get_simulations_data(model,simulation,model_ver='',static=False):
    if static:
        return np.load('output/{}/static/states{}_{}_static.csv'.format(model,model_ver,simulation))
    return pd.read_csv('output/{}/simulations/states{}_{}.csv'.format(model,model_ver,simulation))

def Get_rewards_data(model,simulation,model_ver='',static=False):
    if static:
        return np.load('output/{}/static/rewards{}_{}_static.npy'.format(model,model_ver,simulation))
    return np.load('output/{}/simulations/rewards{}_{}.npy'.format(model,model_ver,simulation))

def Get_lstm_states(model,simulation,model_ver='',static=False):
    if static:
        return np.load('output/{}/static/lstm_states{}_{}_static.npy'.format(model,model_ver,simulation))
    return np.load('output/{}/simulations/lstm_states{}_{}.npy'.format(model,model_ver,simulation))

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
        
def neuron_activity(model,simulation,figsize=(13,7),label='',shadow=False):
    """
    Plot the average neuron activity for specific model,simulation.
    """
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
        
def rose_plot(model,simulation,cycles=2,combined=False):
    sim = Construct_Scenario(Scenarios[simulation])
    length=len(sim)
    x= Get_simulations_data(model,simulation)
    print('Model:{} simulation:{}'.format(model,Scenarios_desc[simulation]))
    XH = (x=='XH').values.sum(axis=0)
    EH = (x=='EH').values.sum(axis=0)
    XF = (x=='XF').values.sum(axis=0)
    EF = (x=='EF').values.sum(axis=0)

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
            Rose_plot_shadow(ax,rose_neurons[j],sim)   
    plt.tight_layout()

def Rose_plot_shadow(ax,r_n,sim):
    length=len(r_n)
    step= 360/length#len(sim)
    #Set the center of the span to mark as night.
    start=step/2
    for mn in range(length):#len(sim)):
        if sim[mn]=='0':
            ax.bar(np.deg2rad(start),max(r_n),width=np.deg2rad(step),facecolor='black',alpha=0.1)
        start=start+step
        
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
            Rose_plot_shadow(ax,rose_neurons[j],Construct_Scenario(Scenarios[j]))
            tick = length/8
            
        ax.set_xticklabels([0,tick,names[j],tick*3,tick*4,tick*5,tick*6,tick*7])
        ax.set_title('{}'.format(j),pad=15)
        counter+=1
    plt.tight_layout()
    return rose_neurons
