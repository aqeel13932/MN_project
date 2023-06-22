# Record simulations based on trained models. The trained models must be located in  output/#Model_number/MOD/  and the simulations will be stored in a folder called simulations in #Model_nmber directory
import argparse
from Scenarios import Construct_Scenario,Scenarios,Scenarios_desc
parser = argparse.ArgumentParser()
parser.add_argument('--num_eps', type=int, default=1000,help="Number of episodes when trying a simulation")
parser.add_argument('--hidden_size', type=int, default=32,help="FC layer hidden size" )
parser.add_argument('--rwrdschem',nargs='+',default=[0,1,-2.5],type=float,help="Reward scheme [,eating food reward, step outside of home at night reward]")
parser.add_argument('--train_m',type=str,default='',help="Identity number of the model")
parser.add_argument('--file_m',type=str,default='',help="full model name (used for intermediate saved models to check the progress of a model)")
parser.add_argument('--naction',type=int,default=0)
parser.add_argument('--clue',action='store_true',help="Add a clue to the network input or not")
parser.add_argument('--optimizer', choices=['adam', 'rmsprop','sgd'], default='adam')
parser.add_argument('--nofood', action='store_true',help="Remove food at night")
parser.add_argument('--render', action='store_true',help="Generate videos")
parser.add_argument('--L1L2', action='store_true',help="add L1L2 regularization to reccurent layer")
parser.add_argument('--Scenario',type=int,default=0 ,help='Between 0-19. Check Scenarios.py for more information.')
parser.add_argument('--reccurent_type', choices=['LSTM', 'GRU','simple'], default='LSTM')
parser.add_argument('--reccurent_size',type=int,default=128)
parser.add_argument('--static', action='store_true',help="Generate static environments between different runs")
args = parser.parse_args()
import numpy as np
import skvideo.io
from keras.models import Model,load_model
from keras.layers import Input, Dense, Lambda,TimeDistributed,convolutional,Flatten,merge


if args.reccurent_type=='GRU':
    from keras.layers import GRU as LSTM
elif args.reccurent_type=='simple':
    from keras.layers import SimpleRNN as LSTM
else:
    from keras.layers import LSTM 

from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam,rmsprop
from keras import backend as K
from APES import *
from time import time
import os
import pandas as pd

if args.static:
    np.random.seed(13357)



def New_Reward_Function(agents,foods,rwrdschem,world,AES,Terminated):
    """Calculate All agents rewards
    Args:
        * agents: dictionary of agents contain all agents by ID
        * foods: dictionary of all foods
        * rwrdschem: Reward Schema (More info in World __init__)
        * world: World Map
        * AES: one element array
        """
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
    """
    Create the environment (World)
    Return:
        * game: World instance
    """
    global episode_length
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
                    VisionAngle=360,Range=-1,
                    PdstName='PM',
                    ActionMemory=args.naction)
    print(blue_Ag.ID)
    game=World(RewardsScheme=args.rwrdschem,StepsLimit=episode_length,RewardFunction=New_Reward_Function)
    #Agents added first has priority of executing there actions first.
    #game.AddAgents([ragnt])
    game.AddAgents([blue_Ag])
    #game.AddObstacles([obs])
    game.AddFoods([food])
    Start = time()-Start
    print ('Taken:',Start)
    return game

def trace_tracker(prev_h,cur_h,prev_f,cur_f):
    """
    Return the events of existing or entering the food field or home.
    Args:
        * prev_h: boolean of agent being home at previous step.
        * cur_h:  boolean of agent being at home currently.
        * prev_f: boolean of agent being at food field at previous step.
        * cur_f:  boolean of agent being at food fielad at current step.
    Return
        * state: string of two characters, First character either "X" eXit or "E" Entered. Second character "H" Home or "F" field". "--" means nothing happened.
    """
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
    """
    Create the neural network
    Args:
        * insize: the size of  orientation+clue.
        * in_conv: the size of convloutional branch.
        * naction: output numer of actions.
    Return:
        * c: Convolutional layer input
        * x: Fully connected layer input
        * z: The network output layer.
    """
    c = Input(batch_shape=in_conv)
    con_process = c
    con_process = TimeDistributed(convolutional.Conv2D(filters=6,kernel_size=(3,3),activation="relu",padding="same",strides=1))(con_process)
    con_process = TimeDistributed(Flatten())(con_process)
    x = Input(batch_shape=insize)#env.observation_space.shape)
    #h = merge.Concatenate(axis=1)([con_process,x])
    h = merge([con_process,x],mode="concat")
    h = TimeDistributed(Dense(args.hidden_size, activation='tanh'))(h)
    h = TimeDistributed(Dense(args.hidden_size, activation='tanh'))(h)
    if args.L1L2:
        h = LSTM(args.reccurent_size,return_sequences=True,stateful=True, kernel_regularizer='l1_l2')(h)
    else:
        h = LSTM(args.reccurent_size,return_sequences=True,stateful=True)(h)
    y = TimeDistributed(Dense(naction + 1))(h)
    z = TimeDistributed(Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:,1:] - K.max(a[:, 1:], keepdims=True), output_shape=(naction,)))(y)
    return c,x, z
seeds=list(range(args.num_eps))

def TryModel(model,game,i):
    """
    Testing the model on a game.
    Args:
        * model: the model to be tested.
        * game: the world instance to test the model on.
    Return:
        * eaten: the count of eaten food.
        * Start: duration of the episode.
        * lstm_episode: reccurent activations.
        * episode_reward: the episode reward.
    """
    global AIAgent,TestingCounter,manipulation, episode_length,seeds
    if args.static:
        np.random.seed(seeds[i])
    TestingCounter+=1
    if args.render:
        writer = skvideo.io.FFmpegWriter("{}/{}/VID/{}_Test.avi".format(EF,args.train_m,TestingCounter))
    game.GenerateWorld()
    AIAgent.Direction='E'
    game.Step()
    Start = time()
    episode_reward=0
    cnn,rest = AIAgent.Convlutional_output()
    day = bool(int(manipulation[0]))
    if args.clue:
        rest = np.concatenate([rest,[not day]])
    all_activity=[]
    if args.render:
        img = game.BuildImage()
        writer.writeFrame(np.array(img*255,dtype=np.uint8))
    eaten = 0
    morning_home=0
    night_home=0
    prev_home=True
    prev_field=False
    lstm_episode_data = np.zeros((episode_length,args.reccurent_size))
    episode_reward=np.zeros(episode_length)
    for t in range(episode_length):
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

    print(eaten)
    Start = time()-Start
    return eaten,Start,lstm_episode_data,episode_reward

#The folder name were we created subfolder to store each experiments.  
EF = 'output'
data = {}
manipulation = Construct_Scenario(Scenarios[args.Scenario])
print(manipulation)
episode_length = len(manipulation)

for i in range(episode_length):
    data[i]=[]
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

if args.train_m=='':
    print('train_m is required')
    exit()
else:
    mod_dir='{}/{}/MOD/model.h5'.format(EF,args.train_m)
    if args.file_m!='':
        mod_dir = '{}/{}/MOD/{}'.format(EF,args.train_m,args.file_m)
    print("Loading this model:")
    print(mod_dir)
    X = load_model(mod_dir)
    model = Model(inputs=X.inputs,outputs=[X.get_layer(index=9).output,X.get_layer(index=7).output])
    #We need to specifiy the batch size 
    c,x,z = createLayers((1,None,5),(1,None, 5, 5, 4), Settings.PossibleActions.shape[0])
    model2 = Model(inputs=[c,x], outputs=z)
    mweights = model.get_weights()
    model2.set_weights(mweights)
    X = model2
    model = Model(inputs=X.inputs,outputs=[X.get_layer(index=9).output,X.get_layer(index=7).output])

fs = (Settings.WorldSize[0]*Settings.BlockSize[0],Settings.WorldSize[1]*Settings.BlockSize[1])
lstm_data = np.zeros((args.num_eps,episode_length,args.reccurent_size))
episodes_rewards= np.zeros((args.num_eps,episode_length))
for i in range(args.num_eps):
    ate,sptime,lstm_data[i],episodes_rewards[i] = TryModel(model,game,i)
    print('episode:{},ate:{},spent:{} seconds'.format(i,ate,round(sptime,3)))
    model.reset_states()

df= pd.DataFrame(data=data)
stor_dir= '{}/{}/simulations/'.format(EF,args.train_m)
print('Storing results in {}'.format(stor_dir))
if not os.path.exists(stor_dir):
    os.makedirs(stor_dir)
static=''
if args.static:
    static='_static'
if args.file_m!='':
    suffix1= '_{}_{}{}.csv'.format(args.Scenario,args.file_m,static)
    suffix2= '_{}_{}{}'.format(args.Scenario,args.file_m,static)
else:
    suffix1= '_{}{}.csv'.format(args.Scenario,static)
    suffix2= '_{}{}'.format(args.Scenario,static)

df.to_csv(stor_dir+'states'+suffix1,index=False)
np.save(stor_dir+'lstm_states'+suffix2,lstm_data)
np.save(stor_dir+'rewards'+suffix2,episodes_rewards)
