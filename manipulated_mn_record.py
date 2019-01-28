#In this file:
    # Food spawn in top left 5X5 square.
    # Agent spawn in bottom right square.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_eps', type=int, default=100)
parser.add_argument('--max_timesteps', type=int, default=80)# 1000 
parser.add_argument('--rwrdschem',nargs='+',default=[0,1,-2.5],type=float) #(calculated should be (1000 reward , -0.1 punish per step)
parser.add_argument('--svision',type=int,default=360)
parser.add_argument('--details',type=str,default='')
parser.add_argument('--train_m',type=str,default='')
parser.add_argument('--naction',type=int,default=0)
parser.add_argument('--night_length',type=int,default=20)
parser.add_argument('--clue',action='store_true')
parser.add_argument('--nofood', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--manipulation',type=str,default='10101010',help='1 marks morning, 0 marks night. E.x 1000 means 1 morning, then 3 nights in arrow')
args = parser.parse_args()
import numpy as np
import skvideo.io
from keras.models import Model,load_model
from APES import *
from time import time
import os
import pandas as pd
#The folder name were we created subfolder to store each experiments.  
EF = 'output'
data = {}
for i in range(args.max_timesteps):
    data[i]=[]

def WriteInfo(epis,t,epis_rwrd,start,rwsc,eptype,trqavg,tsqavg,eaten_num,night_home,morning_home):
    global File_Signature
    with open('{}/{}/exp_details_test.csv'.format(EF,args.train_m),'a') as outp:
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

        
TestingCounter=0
def TryModel(model,game):
    global AIAgent,TestingCounter
    TestingCounter+=1
    if args.render:
        writer = skvideo.io.FFmpegWriter("{}/{}/VID/{}_Test.avi".format(EF,args.train_m,TestingCounter))
        writer2 = skvideo.io.FFmpegWriter("{}/{}/VID/{}_TestAG.avi".format(EF,args.train_m,TestingCounter))
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
    all_activity=[]
    if args.render:
        writer.writeFrame(np.array(img*255,dtype=np.uint8))
        writer2.writeFrame(np.array(game.AgentViewPoint(AIAgent.ID)*255,dtype=np.uint8))
    eaten = 0
    morning_home=0
    night_home=0
    prev_home=True
    prev_field=False
    lstm_episode_data = np.zeros((args.max_timesteps,128))
    mn_counter=0
    for t in range(args.max_timesteps):
        if (t%args.night_length)==0:
            day = bool(int(args.manipulation[mn_counter]))
            mn_counter+=1
        all_cnn[t]=cnn
        all_rest[t]=rest
        q,h = model.predict([all_cnn[None,:],all_rest[None,:]], batch_size=1)
        lstm_episode_data[t]=h[0,t]
        action = np.argmax(q[0,t])

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
        episode_reward += reward

        #print "reward:", reward
        if done:
            break

    if args.render:
        writer.close()
        writer2.close()

    print(eaten)
    Start = time()-Start
    return eaten,Start,lstm_episode_data
    #WriteInfo(TestingCounter,t+1,episode_reward,Start,rwtc,'Test','0','0',eaten,night_home,morning_home)

game = SetupEnvironment()

AIAgent = game.agents[1001]
'''
input size :
Worldsize*(Agents Count+3)+Agents Count *4
worldsize*(Agents count +3(food,observed,obstacles)) + Agents count *4 (orintation per agent)
'''
conv_size=(args.max_timesteps,Settings.WorldSize[0],Settings.WorldSize[1],4,)
naction =  Settings.PossibleActions.shape[0]
if args.clue:
    rest_size=(args.max_timesteps,args.naction*5+5,)
else:
    rest_size=(args.max_timesteps,args.naction*5+4,)
print(conv_size,naction,rest_size)
if args.train_m=='':
    print('train_m is required')
    exit()
else:
    X = load_model('{}/{}/MOD/model.h5'.format(EF,args.train_m))
    model = Model(inputs=X.inputs,outputs=[X.get_layer(index=9).output,X.get_layer(index=7).output])
fs = (Settings.WorldSize[0]*Settings.BlockSize[0],Settings.WorldSize[1]*Settings.BlockSize[1])
lstm_data = np.zeros((args.num_eps,args.max_timesteps,128))
for i in range(args.num_eps):
    ate,sptime,lstm_data[i] = TryModel(model,game)
    print('episode:{},ate:{},spent:{} seconds'.format(i,ate,round(sptime,3)))

df= pd.DataFrame(data=data)
df.to_csv('{}/{}/states_{}.csv'.format(EF,args.train_m,args.manipulation),index=False)
np.save('{}/{}/lstm_states_{}'.format(EF,args.train_m,args.manipulation),lstm_data)
