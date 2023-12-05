import tensorflow as tf
from ACNet import ACNet
import numpy as np
import json
import os
import mapf_gym_cap as mapf_gym
import time
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError
import random

results_path="primal_results"
environment_path="saved_environments"
warehouse_environment_path="warehouse_environments"
if not os.path.exists(results_path):
    os.makedirs(results_path)

class PRIMAL(object):
    '''
    This class provides functionality for running multiple instances of the 
    trained network in a single environment
    '''
    def __init__(self,model_path,grid_size):
        self.grid_size=grid_size
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        self.sess=tf.Session(config=config)
        self.network=ACNet("global",5,None,False,grid_size,"global")
        #load the weights from the checkpoint (only the global ones!)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver()
        saver.restore(self.sess,ckpt.model_checkpoint_path)
        
    def set_env(self,gym):
        self.num_agents=gym.num_agents
        self.agent_states=[]
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)
        self.size=gym.SIZE
        self.env=gym
        
    def step_all_parallel(self):
        action_probs=[None for i in range(self.num_agents)]
        '''advances the state of the environment by a single step across all agents'''
        #parallel inference
        actions=[]
        inputs=[]
        goal_pos=[]
        for agent in range(1,self.num_agents+1):
            o=self.env._observe(agent)
            inputs.append(o[0])
            goal_pos.append(o[1])
        #compute up to LSTM in parallel
        h3_vec = self.sess.run([self.network.h3], 
                                         feed_dict={self.network.inputs:inputs,
                                                    self.network.goal_pos:goal_pos})
        h3_vec=h3_vec[0]
        rnn_out=[]
        #now go all the way past the lstm sequentially feeding the rnn_state
        for a in range(0,self.num_agents):
            rnn_state=self.agent_states[a]
            lstm_output,state = self.sess.run([self.network.rnn_out,self.network.state_out], 
                                         feed_dict={self.network.inputs:[inputs[a]],
                                                    self.network.h3:[h3_vec[a]],
                                                    self.network.state_in[0]:rnn_state[0],
                                                    self.network.state_in[1]:rnn_state[1]})
            rnn_out.append(lstm_output[0])
            self.agent_states[a]=state
        #now finish in parallel
        policy_vec=self.sess.run([self.network.policy], 
                                         feed_dict={self.network.rnn_out:rnn_out})
        policy_vec=policy_vec[0]
        for agent in range(1,self.num_agents+1):
            action=np.argmax(policy_vec[agent-1])
            self.env._step((agent,action))
          
    def find_path(self,max_step=256):
        '''run a full environment to completion, or until max_step steps'''
        solution=[]
        step=0
        while((not self.env._complete()) and step<max_step):
            timestep=[]
            for agent in range(1,self.env.num_agents+1):
                timestep.append(self.env.world.getPos(agent))
            solution.append(np.array(timestep))
            self.step_all_parallel()
            step+=1
            #print(step)
        if step==max_step:
            raise OutOfTimeError
        for agent in range(1,self.env.num_agents):
            timestep.append(self.env.world.getPos(agent))
        return np.array(solution)
    
def make_name(n,s,id,extension,dirname,extra=""):
    if extra=="":
        return dirname+'/'+"{}_agents_{}_size_{}{}".format(n,s,id,extension)
    else:
        return dirname+'/'+"{}_agents_{}_size_{}_{}{}".format(n,s,id,extra,extension)
    
def run_simulations(next,primal, verbose=False):
    #txt file: planning time, crash, nsteps, finished
    (n,s,id) = next   # num_agents,size,density,iter
    # environment_data_filename=make_name(n,s,id,".npy",environment_path,extra="environment")
    # gym=mapf_gym.WarehouseEnv(num_agents=n)

    # pass world as argument
    warehouse_env_path = os.path.join(warehouse_environment_path, 'simple_warehouse_env.npy')
    simple_warehouse_world, open_list = np.load(warehouse_env_path, allow_pickle=True)  # loads world and open_list
    # generate agents start position and goals
    world, goals = generate_start_and_goals(simple_warehouse_world, n, open_list)
    gym = mapf_gym.WarehouseEnv(num_agents=n, world0=world, goals0=goals)
    primal.set_env(gym)
    solution_filename=make_name(n,s,id,".npy",results_path,extra="solution")
    txt_filename=make_name(n,s,id,".txt",results_path)

    world=gym.getObstacleMap()
    start_positions=tuple(gym.getPositions())
    goals=tuple(gym.getGoals())
    start_time=time.time()
    results=dict()
    start_time=time.time()
    try:
        # print('Starting test ({},{},{})'.format(n,s,id))
        max_time_step = 256 + 128*int(s>=80) + 128*int(s>=160)
        # max_time_step = 512
        path=primal.find_path(max_time_step)
        results['finished']=True
        results['time']=time.time()-start_time
        results['length']=len(path)
        if verbose:
            np.save(solution_filename,path)
    except OutOfTimeError:
        results['time']=time.time()-start_time
        results['finished']=False
    results['crashed']=False
    if verbose:
        f=open(txt_filename,'w')
        f.write(json.dumps(results))
        f.close()
    return results

def generate_start_and_goals(world, num_agents, open_list):
    #RANDOMIZE THE POSITIONS OF AGENTS
    agent_counter = 1
    agent_locations=[]
    while agent_counter<=num_agents:
        agent_rand_pos = random.choice(open_list)
        open_list.remove(agent_rand_pos)
        x = agent_rand_pos[0]
        y = agent_rand_pos[1]
        if(world[x,y] == 0):
            world[x,y]=agent_counter
            agent_locations.append((x,y))
            agent_counter += 1        
    
    #RANDOMIZE THE GOALS OF AGENTS
    goals = np.zeros(world.shape).astype(int)
    goal_counter = 1   
    while goal_counter<=num_agents:
        goal_rand_pos = random.choice(open_list)
        open_list.remove(goal_rand_pos)
        x = goal_rand_pos[0]
        y = goal_rand_pos[1]
        if(goals[x,y]==0 and world[x,y]!=-1):
            goals[x,y]    = goal_counter
            goal_counter += 1

    return world, goals

if __name__ == "__main__":
#    import sys
#    num_agents = int(sys.argv[1])

    primal=PRIMAL('model_primal',10)
    
    num_agents = 4
    size = 15
    n_tests = 100

    results = dict()
    results['finished'] = []
    results['time'] = []
    results['length'] = []
    results['crashed'] = []
    print("Starting tests for %d agents" % num_agents)

    for iter in range(n_tests):
        print("Starting test %d" % iter)
        res = run_simulations((num_agents,size,iter),primal)
        print(res)
        results['finished'].append(res['finished'])
        results['time'].append(res['time'])
        if res['finished']:
            results['length'].append(res['length'])
        results['crashed'].append(res['crashed'])
    final_results = dict()
    final_results['finished'] = np.sum(results['finished'])/len(results['finished'])
    final_results['time'] = np.mean(results['time'])
    final_results['length'] = np.mean(results['length'])
    final_results['crashed'] = np.sum(results['crashed'])/len(results['crashed'])
    print(final_results)    
        

print("finished all tests!")
