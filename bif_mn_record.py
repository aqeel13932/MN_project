# Record simulations for bifurcations based on trained models. The trained models must be located in  output/#Model_number/MOD/  and the simulations will be stored in a folder called simulations in #Model_nmber directory
import argparse
from Scenarios import Construct_Scenario,Scenarios,Scenarios_desc
from miscellaneous import *
parser = argparse.ArgumentParser()
parser.add_argument('--num_eps', type=int, default=1000)
parser.add_argument('--episode_length', type=int, default=320)# 1000 
parser.add_argument('--rwrdschem',nargs='+',default=[0,1,-2.5],type=float) #(calculated should be (1000 reward , -0.1 punish per step)
parser.add_argument('--svision',type=int,default=360)
parser.add_argument('--details',type=str,default='')
parser.add_argument('--train_m',type=str,default='')
parser.add_argument('--naction',type=int,default=0)
parser.add_argument('--clue',action='store_true')
parser.add_argument('--nofood', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--Scenario',type=int,default=0 ,help='Between 0-19. Check Scenarios.py for more information.')
args = parser.parse_args()
import numpy as np
import skvideo.io
from keras.models import Model,load_model
from keras.layers import Input, Dense, Lambda,LSTM,TimeDistributed,convolutional,Flatten,merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam,rmsprop
from keras import backend as K
from APES import *
from time import time
import os
import pandas as pd
from os import listdir
from os.path import isfile, join


def load_specific_model(model):
    print(model)
    X = load_model('{}/{}/MOD/{}'.format(EF,args.train_m,model))
    model = Model(inputs=X.inputs,outputs=[X.get_layer(index=9).output,X.get_layer(index=7).output])
    #We need to specifiy the batch size 
    c,x,z = createLayers((1,None,5),(1,None, 5, 5, 4), Settings.PossibleActions.shape[0])
    model2 = Model(inputs=[c,x], outputs=z)
    mweights = model.get_weights()
    model2.set_weights(mweights)
    X = model2
    model = Model(inputs=X.inputs,outputs=[X.get_layer(index=9).output,X.get_layer(index=7).output])
    return model

def New_Reward_Function(agents,foods,rwrdschem,world,AES,Terminated):
    """Calculate All agents rewards
    Args:
        * agents: dictionary of agents contain all agents by ID
        * foods: dictionary of all foods
        * rwrdschem: Reward Schema (More info in World __init__)
        * world: World Map
        * AES: one element array
    TODO:
        * copy this function to class or __init__ documentation as example of how to build customer reward function
        * Assign Reward To Agents
        * Impelent the Food Reward Part depending on the decision of who take the food reward if two 
          agent exist in food range in same time
        * Change All Ranges to .ControlRange not (-1) it's -1 only for testing purpuse
        * Change Punish per step to not punish when agent do nothing"""
    def ResetagentReward(ID):
        # We will penalize for steps in dark out of home from main loop not here.
        agents[ID].CurrentReward=0 # rwrdschem[2] if len(agents[ID].NextAction)>0 else 0

    for x in agents:
        ResetagentReward(x)

    AvailableFoods = world[(world>2000)&(world<=3000)]
    if len(AvailableFoods)==0:
        print('no Foods available')
        Terminated[0]= True if AES[0]<=0 else Terminated[0]
    for ID in agents.keys():
        if agents[ID].IAteFoodID >-1:
            agents[ID].CurrentReward+= foods[agents[ID].IAteFoodID].Energy* rwrdschem[1]
        agntcenter = World._GetElementCoords(ID,agents[ID].FullEgoCentric)
        aborder = World._GetVisionBorders(agntcenter,agents[ID].ControlRange,agents[ID].FullEgoCentric.shape)

def _FindAgentOutput(self,ID,array,agents):
    """ Generate the desired output for agent.NNFeed which will be provided later to neural network
    Args:
        * ID: agent ID
        * array: the world in the agent prospective
        * agents: Dictionary (Key: agent ID, Value: Agent Object)"""

    def _agentdirection(direction):
        """ Get a vector of current agent direction
        Args:
            * direction: direction ('N'or 'S' or 'W' or 'E')
        return:
            array (1,4) example [1,0,0,0]
        """
        return np.array([direction=='N',direction=='S',direction=='E',direction=='W'])
    #Used Ordered here so the output keys will be always in the same manner in case the values
    # feed to some network they will always be in same order.
    ls = OrderedDict()

    #observed (True)/unobeserved(False) layer
    ls['observed']= (array!=-1)

    #My Place
    ls['mypos']= (array==ID)
    ls['myori']= _agentdirection( agents[ID].Direction)
    ls['obstacles'] = (array>3000)
    ls['food'] = np.logical_and(array>2000,array<3001)
    #Get list of only observed agents.
    observedagents = array[(array>1000)&(array<2000)]
    for oID in agents.keys():
        if oID == ID:
            continue
        if oID in observedagents:
            ls['agentpos{}'.format(oID)]= (array==oID)
            ls['agentori{}'.format(oID)]= _agentdirection(agents[oID].Direction)
        else:
            ls['agentpos{}'.format(oID)]= np.zeros(array.shape,dtype=bool)# (array==oID)
            ls['agentori{}'.format(oID)]=np.array([0,0,0,0],dtype=bool)
    return ls

def SetupEnvironment():
    Start = time()
    #Add Pictures
    Settings.SetBlockSize(20)
    Settings.AddImage('Wall','APES/Pics/wall.jpg')
    Settings.AddImage('Food','APES/Pics/food.jpg')
    #Specify World Size
    Settings.WorldSize=(5,5)
    #Create Probabilities
    blue_Ag_PM = np.zeros(Settings.WorldSize)
    blue_Ag_PM[Settings.WorldSize[0]-1,Settings.WorldSize[1]-1]=1
    food_PM = np.zeros(Settings.WorldSize)
    food_PM[0:3,0:3] = 1
    #Add Probabilities to Settings
    Settings.AddProbabilityDistribution('PM',blue_Ag_PM)
    Settings.AddProbabilityDistribution('food_PM',food_PM)
    #Create World Elements
    food = Foods('Food',PdstName='food_PM')

    blue_Ag = Agent(Fname='APES/Pics/blue.jpg',
                    Power=3,
                    VisionAngle=args.svision,Range=-1,
                    PdstName='PM',
                    ActionMemory=args.naction)
    print(blue_Ag.ID)
    game=World(RewardsScheme=args.rwrdschem,StepsLimit=args.episode_length,RewardFunction=New_Reward_Function)
    #Agents added first has priority of executing there actions first.
    #game.AddAgents([ragnt])
    game.AddAgents([blue_Ag])
    #game.AddObstacles([obs])
    game.AddFoods([food])
    Start = time()-Start
    print ('Taken:',Start)
    return game

def trace_tracker(prev_h,cur_h,prev_f,cur_f):
    status='--'
    #Previously was in home, but not now
    if prev_h and (not cur_h):
        status='XH' #Agent Exited home
    #Previously was out, but now in home
    elif not prev_h and cur_h:
        status='EH' #Agent Entered home

    #Previously was in field, but not now
    if prev_f and (not cur_f):
        status='XF' #Agent Exited field
    #Previously was out, but now in field
    elif not prev_f and cur_f:
        status='EF' #Agent Entered field

    return status

def createLayers(insize,in_conv,naction):
    c = Input(batch_shape=in_conv)
    con_process = c
    con_process = TimeDistributed(convolutional.Conv2D(filters=6,kernel_size=(3,3),activation="relu",padding="same",strides=1))(con_process)
    con_process = TimeDistributed(Flatten())(con_process)
    x = Input(batch_shape=insize)#env.observation_space.shape)
    #h = merge.Concatenate(axis=1)([con_process,x])
    h = merge([con_process,x],mode="concat")
    h = TimeDistributed(Dense(32, activation='tanh'))(h)
    h = TimeDistributed(Dense(32, activation='tanh'))(h)
    h = LSTM(128,return_sequences=True,stateful=True)(h)
    y = TimeDistributed(Dense(naction + 1))(h)
    z = TimeDistributed(Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:,1:] - K.max(a[:, 1:], keepdims=True), output_shape=(naction,)))(y)
    return c,x, z


def TryModel(model,game):
    global AIAgent,TestingCounter
    TestingCounter+=1
    if args.render:
        writer = skvideo.io.FFmpegWriter("{}/{}/VID/{}_Test.avi".format(EF,args.train_m,TestingCounter))
        writer2 = skvideo.io.FFmpegWriter("{}/{}/VID/{}_TestAG.avi".format(EF,args.train_m,TestingCounter))
    game.GenerateWorld()
    #AIAgent.Direction='E'
    game.Step()
    img = game.BuildImage()
    Start = time()
    episode_reward=0
    cnn,rest = AIAgent.Convlutional_output()
    day = False
    if args.clue:
        rest = np.concatenate([rest,[not day]])
    all_activity=[]
    if args.render:
        writer.writeFrame(np.array(img*255,dtype=np.uint8))
        writer2.writeFrame(np.array(game.AgentViewPoint(AIAgent.ID)*255,dtype=np.uint8))
    eaten = 0
    morning_home=0
    night_home=0
    prev_home=True
    prev_field=False
    lstm_episode_data = np.zeros((args.episode_length,128))
    episode_reward=np.zeros(args.episode_length)
    for t in range(args.episode_length):
        day = bool(int(manipulation[t]))
        q,h = model.predict([cnn[np.newaxis,np.newaxis],rest[np.newaxis,np.newaxis]], batch_size=1)
        lstm_episode_data[t]=h[0,0]
        action = np.argmax(q[0,0])

        # Only subordinate moves, dominant is static
        AIAgent.NextAction = Settings.PossibleActions[action]
        AIAgent.AddAction(action)
        game.Step()
        
        if args.render:
            writer.writeFrame(np.array(game.BuildImage()*255,dtype=np.uint8))
            writer2.writeFrame(np.array(game.AgentViewPoint(AIAgent.ID)*255,dtype=np.uint8))

        #Remove any trace of food during night.
        if (not day) and (args.nofood):
            #When its night hide the food.
            AIAgent.NNFeed['food'] =np.logical_and(AIAgent.NNFeed['food'],False)
            #Set the agent reward to zero since it might eat it even it can't see it. This will make the agent totally blind about the existence of food during night.
            AIAgent.CurrentReward=0
        cnn,rest = AIAgent.Convlutional_output()
        if args.clue:
            rest = np.concatenate([rest,[day]])
        if AIAgent.CurrentReward>0:
            eaten+=1
        athome = game.world[4,4]==AIAgent.ID
        atfield= np.argwhere(game.world[:3,:3]==1001).shape[0]>0
        data[t].append(trace_tracker(prev_home,athome,prev_field,atfield))
        #print('s:{},h:{},f:{},d:{},result:{}'.format(t,athome,atfield,day,data[i][-1]))
        #print(trace_tracker(prev_home,athome,prev_field,atfield))
        prev_home = athome
        prev_field=atfield
        if (not day) and (not athome):
            AIAgent.CurrentReward+=args.rwrdschem[2]
            night_home+=1
        if day and athome:
            morning_home+=1
        reward = AIAgent.CurrentReward
        done = game.Terminated[0]

        #observation, reward, done, info = env.step(action)
        episode_reward[t]= reward

        #print "reward:", reward
        if done:
            break
    #exit()
    if args.render:
        writer.close()
        writer2.close()

    print(eaten)
    Start = time()-Start
    return eaten,Start,lstm_episode_data,episode_reward

#The folder name were we created subfolder to store each experiments.  
EF = 'output'
# Convert the Scenario number to specific morning/night patteren.
manipulation = Construct_Scenario(Scenarios[args.Scenario])
print(manipulation)
TestingCounter=0

game = SetupEnvironment()

AIAgent = game.agents[1001]
'''
input size :
Worldsize*(Agents Count+3)+Agents Count *4
worldsize*(Agents count +3(food,observed,obstacles)) + Agents count *4 (orintation per agent)
'''
naction =  Settings.PossibleActions.shape[0]
# Get all available models
def Get_availablemodels(model):
    mypath='{}/{}/MOD'.format(EF,model)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    wanted_models = []
    for i in onlyfiles:
        if i.find('model_eps')==0:
            wanted_models.append(i)
            
    mod = []
    for m in wanted_models:
            mod.append(int(m[10:-3]))
    lstms = Get_LSTM_Files(model,args.Scenario)

    for lstm in lstms:
        mod.remove(lstm)
    mod = sorted(mod)
    mod = ['model_eps:' + str(sub) for sub in mod]
    mod = [sub + '.h5' for sub in mod]
    return mod

if args.train_m=='':
    print('train_m is required')
    exit()
#Create a folder to save all the results.
if not os.path.exists('{}/{}/bif_data'.format(EF,args.train_m)):
        os.makedirs('{}/{}/bif_data'.format(EF,args.train_m))

wanted_models = Get_availablemodels(args.train_m)

for wm in wanted_models:
    TestingCounter=0
    #Prepare dataset container.
    data = {}
    for i in range(args.episode_length):
        data[i]=[]
    model =load_specific_model(wm)
    fs = (Settings.WorldSize[0]*Settings.BlockSize[0],Settings.WorldSize[1]*Settings.BlockSize[1])
    lstm_data = np.zeros((args.num_eps,args.episode_length,128))
    episodes_rewards= np.zeros((args.num_eps,args.episode_length))
    for i in range(args.num_eps):
        ate,sptime,lstm_data[i],episodes_rewards[i] = TryModel(model,game)
        print('episode:{},ate:{},spent:{} seconds'.format(i,ate,round(sptime,3)))
        model.reset_states()

    df= pd.DataFrame(data=data)
    df.to_csv('{}/{}/bif_data/states_{}_{}.csv'.format(EF,args.train_m,args.Scenario,wm),index=False)
    np.save('{}/{}/bif_data/lstm_states_{}_{}'.format(EF,args.train_m,args.Scenario,wm),lstm_data)
    np.save('{}/{}/bif_data/rewards_{}_{}'.format(EF,args.train_m,args.Scenario,wm),episodes_rewards)
