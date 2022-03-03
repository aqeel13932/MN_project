#In this file:
    # Food spawn in top left 5X5 square.
    # Agent spawn in bottom right square.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filesignature',type=int)
parser.add_argument('--batch_size', type=int, default=16)#100 ( 100, 16,32,64,128) priority 3
parser.add_argument('--seed',type=int,default=1337)#4(CH)9(JAP)17(ITAL)
parser.add_argument('--hidden_size', type=int, default=32)#priority 2
parser.add_argument('--batch_norm', action="store_true", default=False)#priority 5 , keep turned off
parser.add_argument('--no_batch_norm', action="store_false", dest='batch_norm')
parser.add_argument('--replay_size', type=int, default=1000)# try increasing later  , priority 3.1
parser.add_argument('--train_repeat', type=int, default=4)#(2^2) , priority 1
parser.add_argument('--gamma', type=float, default=0.99)# (calculated should be 0.99) (0.99)
parser.add_argument('--tau', type=float, default=0.001)# priority 0.9 (0.001 , 0.01 , 0.1) the one that work expeirment in the domain.
parser.add_argument('--totalsteps', type=int, default=1000000)# much more ( 1000 -> 10,000) (should be around 1 million steps)
parser.add_argument('--max_timesteps', type=int, default=80)# 1000 
parser.add_argument('--activation', choices=['tanh', 'relu'], default='relu')# experiment ( relu , tanh) priority 0.7
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')# priority 4.9
parser.add_argument('--optimizer_lr', type=float, default=0.001)#could be used later priority 4.5
parser.add_argument('--exploration', type=float, default=0.1)# priority (0.8) it should decrease over time to reach 0.001 or even 0
parser.add_argument('--vanish',type=float,default=0.75)#Decide when the exploration should stop in percentage (75%)
parser.add_argument('--advantage', choices=['naive', 'max', 'avg'], default='avg')# priority 2 maybe done once and stike with one 
parser.add_argument('--rwrdschem',nargs='+',default=[0,1,-1.5],type=float) #(calculated should be (1000 reward , -0.1 punish per step)
parser.add_argument('--svision',type=int,default=360)
parser.add_argument('--details',type=str,default='')
parser.add_argument('--train_m',type=str,default='')
parser.add_argument('--naction',type=int,default=0)
parser.add_argument('--lstm_size',type=int,default=128)
parser.add_argument('--night_length',type=int,default=20)
parser.add_argument('--clue', action='store_true')
parser.add_argument('--nofood', action='store_true')
args = parser.parse_args()
import numpy as np
import tensorflow as tf
import random as python_random
np.random.seed(args.seed)
python_random.seed(args.seed)
tf.set_random_seed(args.seed)
import skvideo.io
from keras.models import Model,load_model
from keras.layers import Input, Dense, Lambda,LSTM,TimeDistributed,convolutional,Flatten,merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam,rmsprop
from keras import backend as K
from APES import *
from time import time
from buffer import BufferLSTM as Buffer
from copy import deepcopy
import os
#File_Signature = int(round(time()))
File_Signature = args.filesignature
#The folder name were we created subfolder to store each experiments.  
EF = 'output'
def GenerateSettingsLine():
    global args
    line = []
    line.append(args.replay_size)
    line.append(args.tau)
    line.append(args.optimizer)
    line.append(args.advantage)
    line.append(args.max_timesteps)
    line.append(args.activation)
    line.append(args.batch_size)
    line.append(args.totalsteps)
    line.append(args.exploration)
    line.append(args.vanish)
    line.append(args.gamma)
    line.append(args.hidden_size)
    line.append(args.train_repeat)
    line.append(args.batch_norm)
    line.append(args.seed)
    line.append(args.rwrdschem)
    line.append(args.svision)
    line.append(args.lstm_size)
    line.append("\""+args.details+"\"")
    return ','.join([str(x) for x in line])

line = GenerateSettingsLine()
with open ('{}/features.results.out'.format(EF),'a') as f:
    f.write('{}\n{}\n'.format(File_Signature,line))

def WriteInfo(epis,t,epis_rwrd,start,rwsc,eptype,trqavg,tsqavg,eaten_num,night_home,morning_home):
    global File_Signature
    with open('{}/{}/exp_details.csv'.format(EF,File_Signature),'a') as outp:
        outp.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(epis,t,epis_rwrd,start,rwsc,eptype,trqavg,tsqavg,eaten_num,night_home,morning_home))

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
    game=World(RewardsScheme=args.rwrdschem,StepsLimit=args.max_timesteps,RewardFunction=New_Reward_Function)
    #Agents added first has priority of executing there actions first.
    #game.AddAgents([ragnt])
    game.AddAgents([blue_Ag])
    #game.AddObstacles([obs])
    game.AddFoods([food])
    Start = time()-Start
    print ('Taken:',Start)
    return game

def createLayers(insize,in_conv,naction):
    c = Input(shape=in_conv)
    con_process = c
    con_process = TimeDistributed(convolutional.Conv2D(filters=6,kernel_size=(3,3),activation="relu",padding="same",strides=1))(con_process)
    con_process = TimeDistributed(Flatten())(con_process)
    x = Input(shape=insize)#env.observation_space.shape)
    #h = merge.Concatenate(axis=1)([con_process,x])
    h = merge([con_process,x],mode="concat")
    h = TimeDistributed(Dense(args.hidden_size, activation=args.activation))(h)
    h = TimeDistributed(Dense(args.hidden_size, activation=args.activation))(h)
    if args.batch_norm and i != args.layers - 1:
        h = BatchNormalization(axis=1)(h)
    h = LSTM(args.lstm_size,return_sequences=True,stateful=False)(h)
    y = TimeDistributed(Dense(naction + 1))(h)
    if args.advantage == 'avg':
      z = TimeDistributed(Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:,1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(naction,)))(y)
    elif args.advantage == 'max':
      z = TimeDistributed(Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:,1:] - K.max(a[:, 1:], keepdims=True), output_shape=(naction,)))(y)
    elif args.advantage == 'naive':
      z = TimeDistributed(Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:,1:], output_shape=(naction,)))(y)
    else:
      assert False
    return c,x, z

def train_model():
    qpremean = 0
    qpostmean =0
    if len(mem.buffer)>=args.batch_size*2: 
        for k in range(args.train_repeat):
            pre_cnn,pre_rest,lst_action,post_cnn,post_rest,lst_reward,lst_done = mem.sample(args.batch_size)
            qpre = model.predict_on_batch([pre_cnn,pre_rest])
            qpost = target_model.predict_on_batch([post_cnn,post_rest])
            seq = np.arange(qpre.shape[1])
            for i in range(qpre.shape[0]):
                qpre[i,seq,lst_action[i,:,0]] = lst_reward[i,:,0]+ args.gamma * np.amax(qpost[i,:],axis=1) * np.logical_not(lst_done[i,:,0])
            model.train_on_batch([pre_cnn,pre_rest], qpre)
            weights = model.get_weights()
            target_weights = target_model.get_weights()
            for i in range(len(weights)):
                target_weights[i] = args.tau * weights[i] + (1 - args.tau) * target_weights[i]
        qpremean = qpre.mean()
        qpostmean = qpost.mean()
        target_model.set_weights(target_weights)
    return qpremean,qpostmean

TestingCounter=0
def TryModel(model,game):
    #print('Testing Target Model')
    global AIAgent,File_Signature,TestingCounter
    TestingCounter+=1
    #if TestingCounter%10==0:
    #    writer = skvideo.io.FFmpegWriter("output/{}/VID/{}_Test.avi".format(File_Signature,TestingCounter))
    #    writer2 = skvideo.io.FFmpegWriter("output/{}/VID/{}_TestAG.avi".format(File_Signature,TestingCounter))
    game.GenerateWorld()
    AIAgent.Direction='E'
    game.Step()
    img = game.BuildImage()
    rwtc =0# RandomWalk(game)
    Start = time()
    episode_reward=0
    cnn,rest = AIAgent.Convlutional_output()
    day = False
    if args.clue:
        rest = np.concatenate([rest,[not day]])
    all_cnn = np.zeros(conv_size,dtype=np.int8)
    all_rest = np.zeros(rest_size,dtype=np.int8)
    #if TestingCounter%10==0:
    #    writer.writeFrame(np.array(img*255,dtype=np.uint8))
    #    writer2.writeFrame(np.array(game.AgentViewPoint(AIAgent.ID)*255,dtype=np.uint8))
    eaten = 0
    morning_home=0
    night_home=0
    for t in range(args.max_timesteps):
        if (t%args.night_length)==0:
            day = not day
        all_cnn[t]=cnn
        all_rest[t]=rest
        q = model.predict([all_cnn[None,:],all_rest[None,:]], batch_size=1)
        action = np.argmax(q[0,t])

        # Only subordinate moves, dominant is static
        AIAgent.NextAction = Settings.PossibleActions[action]
        AIAgent.AddAction(action)
        game.Step()
        
        #if TestingCounter%10==0:
        #    writer.writeFrame(np.array(game.BuildImage()*255,dtype=np.uint8))
        #    writer2.writeFrame(np.array(game.AgentViewPoint(AIAgent.ID)*255,dtype=np.uint8))

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
        if (not day) and (not athome):
            AIAgent.CurrentReward+=args.rwrdschem[2]
        if athome:
            if day:
                morning_home+=1
            else:
                night_home+=1
            
        reward = AIAgent.CurrentReward
        done = game.Terminated[0]

        #observation, reward, done, info = env.step(action)
        episode_reward += reward

        #print "reward:", reward
        if done:
            break

    #if TestingCounter%10==0:
    #    writer.close()
    #    writer2.close()
    #if t>=999:
    #    plt.imsave('output/{}/PNG/{}_Test.png'.format(File_Signature,TestingCounter),img)
    #else:
        #os.remove("output/{}/VID/{}_Test.avi".format(File_Signature,TestingCounter))
        #os.remove("output/{}/VID/{}_TestAG.avi".format(File_Signature,TestingCounter))

    Start = time()-Start
    print(t)
    WriteInfo(TestingCounter,t+1,episode_reward,Start,rwtc,'Test','0','0',eaten,night_home,morning_home)
def Remove_Uneeded():
    global AIAgent
    AIAgent.NNFeed['observed']=''
    AIAgent.NNFeed['obstacles']=''
        
game = SetupEnvironment()

AIAgent = game.agents[1001]
'''
input size :
Worldsize*(Agents Count+3)+Agents Count *4
worldsize*(Agents count +3(food,observed,obstacles)) + Agents count *4 (orintation per agent)
'''
game.GenerateWorld()
game.Step()
conv_size=(args.max_timesteps,Settings.WorldSize[0],Settings.WorldSize[1],4,)
naction =  Settings.PossibleActions.shape[0]
if args.clue:
    rest_size=(args.max_timesteps,args.naction*5+5,)
else:
    rest_size=(args.max_timesteps,args.naction*5+4,)
print(conv_size,naction,rest_size)
if args.train_m=='':
    print('train default')
    c,x, z = createLayers(rest_size,conv_size,naction)
    model = Model(inputs=[c,x], outputs=z)
    model.summary()
    optimizer = adam(lr=args.optimizer_lr) if args.optimizer=='adam' else rmsprop(lr=args.optimizer_lr)
    model.compile(optimizer=optimizer, loss='mse')

    print('test from scractch')
    c,x, z = createLayers(rest_size,conv_size,naction)

    target_model = Model(inputs=[c,x], outputs=z)
    target_model.set_weights(model.get_weights())
else:
    model = load_model('morning_night/{}/MOD/model.h5'.format(args.train_m))
    target_model = load_model('morning_night/{}/MOD/target_model.h5'.format(args.train_m))
mem = Buffer(args.replay_size)
#Exploration decrease amount:
EDA = args.exploration/(args.totalsteps*args.vanish)
#Framse Size
fs = (Settings.WorldSize[0]*Settings.BlockSize[0],Settings.WorldSize[1]*Settings.BlockSize[1])
total_reward = 0
#Create Folder to store the output
if not os.path.exists('{}/{}'.format(EF,File_Signature)):
        os.makedirs('{}/{}'.format(EF,File_Signature))
        os.makedirs('{}/{}/PNG'.format(EF,File_Signature))
        os.makedirs('{}/{}/VID'.format(EF,File_Signature))
        os.makedirs('{}/{}/MOD'.format(EF,File_Signature))

progress=0
i_episode=0
while progress<args.totalsteps:
    i_episode+=1
    game.GenerateWorld()
    rwtc=0# = RandomWalk(game)
    Start = time()
    #First Step only do the calculation of the current observations for all agents
    game.Step()
    img =game.BuildImage()
    episode_reward=0
    cnn,rest = AIAgent.Convlutional_output()
    day=False
    morning_home=0
    night_home=0
    if args.clue:
        rest = np.concatenate([rest,[not day]])
    pre_cnn = np.zeros(conv_size,dtype=np.int8)
    post_cnn = np.zeros(conv_size,dtype=np.int8)
    pre_rest = np.zeros(rest_size,dtype=np.int8)
    post_rest = np.zeros(rest_size,dtype=np.int8)
    lst_action = np.zeros((args.max_timesteps,1),dtype=np.int8)
    lst_reward = np.zeros((args.max_timesteps,1))
    lst_done = np.zeros((args.max_timesteps,1),dtype=np.int8)
    episode_buffer = np.array([pre_cnn[0],post_cnn[0],pre_rest[0],post_rest[0],lst_action[0],lst_reward[0],lst_done[0]])
    eaten=0
    for t in range(args.max_timesteps):
        if (t%args.night_length)==0:
            day = not day

        args.exploration = max(args.exploration-EDA,0.1)
        pre_cnn[t]=cnn
        pre_rest[t]=rest
        #Generate action for the agent.
        if np.random.random() < args.exploration:
            action =AIAgent.RandomAction()
        else:
            q = model.predict([pre_cnn[None,:],pre_rest[None,:]], batch_size=1)
            action = np.argmax(q[0,t])
        lst_action[t] = action
        AIAgent.NextAction = Settings.PossibleActions[action]
        AIAgent.AddAction(action)
        game.Step()
        #Remove any trace of food during night if no food flag was raised.
        if (not day) and (args.nofood):
            #When its night hide the food.
            AIAgent.NNFeed['food'] =np.logical_and(AIAgent.NNFeed['food'],False)
            #Set the agent reward to zero since it might eat it even it can't see it. This will make the agent totally blind about the existence of food during night.
            AIAgent.CurrentReward=0
        #Give a clue to the network of morning/night situation.
        if args.clue:
            post_cnn[t],tmp_rest = AIAgent.Convlutional_output()
            tmp_rest = np.concatenate([tmp_rest,[day]])
            post_rest[t] = tmp_rest
        else:
            post_cnn[t],post_rest[t] = AIAgent.Convlutional_output()
        #this data is the pre for next step.
        cnn = post_cnn[t]
        rest = post_rest[t]
        if AIAgent.CurrentReward>0:
            eaten+=1
        athome = game.world[4,4]==AIAgent.ID
        if (not day) and (not athome):
            AIAgent.CurrentReward+=args.rwrdschem[2]
        if athome:
            if day:
                morning_home+=1
            else:
                night_home+=1
        lst_reward[t] = AIAgent.CurrentReward
        lst_done[t] = game.Terminated[0]
        #observation, reward, done, info = env.step(action)
        episode_reward += lst_reward[t]

        if lst_done[t]:
            break
    mem.add((pre_cnn,pre_rest,lst_action,post_cnn,post_rest,lst_reward,lst_done))
    #train the model every episode
    qpr,qpo = train_model()
    Start = time()-Start
    t = t+1
    progress+=t
    
    WriteInfo(i_episode,t,episode_reward[0],Start,rwtc,'train',qpr,qpo,eaten,night_home,morning_home)
    print("Episode {} finished after {} timesteps, episode reward {} Tooks {}s,eaten:{},night_home:{},morning_home:{}, Total Progress:{}".format(i_episode, t, episode_reward,Start,eaten,night_home,morning_home,progress))
    print(len(lst_reward))
    total_reward += episode_reward
    if i_episode%10==0:
        TryModel(model,game)
        print("Average reward per episode {}".format(total_reward /i_episode))
    if ((args.batch_size*2)<i_episode<400) or i_episode%100==0:
        model.save('{}/{}/MOD/model_eps:{}.h5'.format(EF,File_Signature,i_episode))
        print('only one model was saved')
        exit()
model.save('{}/{}/MOD/model.h5'.format(EF,File_Signature))
target_model.save('{}/{}/MOD/target_model.h5'.format(EF,File_Signature))
TryModel(model,game)
