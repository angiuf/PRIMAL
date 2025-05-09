import tensorflow as tf
from ACNet import ACNet
import numpy as np
import json
import os
import mapf_gym_cap as mapf_gym
import time
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
import random
import csv
import datetime
from tqdm import tqdm
from pathlib import Path

def count_collisions(solution, obstacle_map_bool):
    agent_agent_collisions = 0
    obstacle_collisions = 0
    num_agents = 0
    if len(solution) > 0:
        num_agents = len(solution[0])

    for timestep in range(len(solution)):
        positions_at_timestep = solution[timestep]
        current_agent_positions_for_aa_check = []

        for agent_idx in range(num_agents):
            agent_pos_tuple = tuple(positions_at_timestep[agent_idx])
            
            # Check for obstacle collisions
            if obstacle_map_bool[agent_pos_tuple[0], agent_pos_tuple[1]] == -1:
                obstacle_collisions += 1
            
            # Prepare for agent-agent collision check
            current_agent_positions_for_aa_check.append(agent_pos_tuple)

        # Agent-agent collision check for this timestep
        for agent_idx in range(num_agents):
            agent_pos_tuple = current_agent_positions_for_aa_check[agent_idx]
            # Check against other agents at the same timestep
            for other_agent_idx in range(agent_idx + 1, num_agents):
                if agent_pos_tuple == current_agent_positions_for_aa_check[other_agent_idx]:
                    agent_agent_collisions += 2 # Count one for each agent involved
                    
    return agent_agent_collisions, obstacle_collisions


def get_csv_logger(model_dir, default_model_name):
    model_dir_path = Path(model_dir)
    csv_path = model_dir_path / f"log-{default_model_name}.csv"
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

def create_folders_if_necessary(path: Path):
    dirname = path.parent
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=True)

class PRIMAL(object):
    '''
    This class provides functionality for running multiple instances of the 
    trained network in a single environment
    '''
    def __init__(self, model_path, grid_size):
        self.grid_size = grid_size
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.network = ACNet("global", 5, None, False, grid_size, "global")
        # load the weights from the checkpoint (only the global ones!)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        
    def set_env(self, gym):
        self.num_agents = gym.num_agents
        self.agent_states = []
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)
        self.env = gym
        
    def step_all_parallel(self):
        action_probs = [None for i in range(self.num_agents)]
        '''advances the state of the environment by a single step across all agents'''
        # parallel inference
        actions = []
        inputs = []
        goal_pos = []
        for agent in range(1, self.num_agents + 1):
            o = self.env._observe(agent)
            inputs.append(o[0])
            goal_pos.append(o[1])
        # compute up to LSTM in parallel
        h3_vec = self.sess.run([self.network.h3], 
                                         feed_dict={self.network.inputs: inputs,
                                                    self.network.goal_pos: goal_pos})
        h3_vec = h3_vec[0]
        rnn_out = []
        # now go all the way past the lstm sequentially feeding the rnn_state
        for a in range(0, self.num_agents):
            rnn_state = self.agent_states[a]
            lstm_output, state = self.sess.run([self.network.rnn_out, self.network.state_out], 
                                         feed_dict={self.network.inputs: [inputs[a]],
                                                    self.network.h3: [h3_vec[a]],
                                                    self.network.state_in[0]: rnn_state[0],
                                                    self.network.state_in[1]: rnn_state[1]})
            rnn_out.append(lstm_output[0])
            self.agent_states[a] = state
        # now finish in parallel
        policy_vec = self.sess.run([self.network.policy], 
                                         feed_dict={self.network.rnn_out: rnn_out})
        policy_vec = policy_vec[0]
        for agent in range(1, self.num_agents + 1):
            action = np.argmax(policy_vec[agent - 1])
            self.env._step((agent, action))
    
    def compute_number_of_steps_and_cost(self, step, timestep, prev_timestep, total_step, total_cost):
        '''
        Computes the number of steps that the agent has taken
        '''
        goals = self.env.getGoals()
        for agent in range(self.num_agents):
            if prev_timestep is None:
                prev_timestep = timestep
            if timestep[agent][0] != prev_timestep[agent][0] or timestep[agent][1] != prev_timestep[agent][1]:
                total_step[agent] += 1
            if timestep[agent][0] != goals[agent][0] or timestep[agent][1] != goals[agent][1]:
                total_cost[agent] = step
        return total_step, total_cost


    def find_path(self, max_step=256):
        '''run a full environment to completion, or until max_step steps'''
        solution = []
        step = 0
        agent_steps = np.zeros(self.num_agents)
        agent_costs = np.zeros(self.num_agents)
        prev_timestep = None
        while(step < max_step):
            timestep = []
            for agent in range(1, self.env.num_agents + 1):
                timestep.append(self.env.world.getPos(agent))
            solution.append(np.array(timestep))
            self.step_all_parallel()
            agent_steps, agent_costs = self.compute_number_of_steps_and_cost(step, timestep, prev_timestep, agent_steps, agent_costs)
            step += 1
            prev_timestep = timestep
            if (self.env._complete()):
                break
        if step == max_step:
            raise OutOfTimeError
        for agent in range(1, self.env.num_agents):
            timestep.append(self.env.world.getPos(agent))

        return np.array(solution), np.array(agent_steps), np.array(agent_costs)
    
def run_simulations(dataset_dir, map_name, next_item, primal):
    # txt file: planning time, crash, nsteps, finished
    (n, s, id) = next_item   # num_agents, size, density, iter

    dataset_dir_path = Path(dataset_dir)

    # pass world as argument
    warehouse_env_path = dataset_dir_path / map_name / "input" / "map" / f"{map_name}.npy"
    warehouse_world = np.load(warehouse_env_path)  # loads world
    
    warehouse_world[warehouse_world == 1] = -1  # set obstacles to -1

    # generate agents start position and goals
    world, goals = generate_start_and_goals(dataset_dir_path, map_name, warehouse_world, n, id)
    gym = mapf_gym.WarehouseEnv(num_agents=n, world0=world, goals0=goals)
    primal.set_env(gym)

    world = gym.getObstacleMap()
    start_positions = tuple(gym.getPositions())
    goals = tuple(gym.getGoals())
    start_time = time.time()
    results = dict()
    solution = []
    start_time = time.time()
    try:
        max_time_step = 256 + 128 * int(s >= 80) + 128 * int(s >= 160)
        path, agent_steps, agent_costs = primal.find_path(max_time_step)
        results['finished'] = True
        results['time'] = time.time() - start_time
        results['episode_length'] = len(path) - 1 
        results['total_steps'] = np.sum(agent_steps)
        results['avg_steps'] = np.mean(agent_steps)
        results['max_steps'] = np.max(agent_steps)
        results['min_steps'] = np.min(agent_steps)
        results['total_costs'] = np.sum(agent_costs)
        results['avg_costs'] = np.mean(agent_costs)
        results['max_costs'] = np.max(agent_costs)
        results['min_costs'] = np.min(agent_costs)
        
        if path is not None and len(path) > 0 and results['episode_length'] > 0 and n > 0:
            agent_coll, obs_coll = count_collisions(path, primal.env.initial_world)
            results['crashed'] = (agent_coll + obs_coll) > 0
            results['agent_collisions'] = agent_coll # agent-agent collisions
            results['obstacle_collisions'] = obs_coll
            results['collisions'] = agent_coll + obs_coll
            results['agent_coll_rate'] = agent_coll / (results['episode_length'] * n)
            results['obstacle_coll_rate'] = obs_coll / (results['episode_length'] * n)
            results['total_coll_rate'] = (agent_coll + obs_coll) / (results['episode_length'] * n)
        else:
            results['crashed'] = False
            results['agent_collisions'] = None
            results['obstacle_collisions'] = None
            results['collisions'] = None
            results['agent_coll_rate'] = None
            results['obstacle_coll_rate'] = None
            results['total_coll_rate'] = None

        # Save solution
        solution = [[] for i in range(num_agents)]
        for i in range(num_agents):
            for timestep in range(len(path)):
                solution[i].append(tuple(path[timestep][i]) + (timestep,))
    except OutOfTimeError:
        results['time'] = time.time() - start_time
        results['finished'] = False
        results['agent_collisions'] = None
        results['obstacle_collisions'] = None
        results['crashed'] = False
        results['agent_coll_rate'] = None
        results['obstacle_coll_rate'] = None
        results['total_coll_rate'] = None
        results['min_steps'] = None
        results['min_costs'] = None
    
    return results, solution

def generate_start_and_goals(dataset_dir: Path, map_name: str, world: np.ndarray, num_agents: int, id: int):
    # Load case
    filepath = dataset_dir / map_name / "input" / "start_and_goal" / f"{num_agents}_agents"
    case_name = filepath / f"{map_name}_{num_agents}_agents_ID_{str(id).zfill(3)}.npy"
    pos = np.load(case_name, allow_pickle=True)
    
    # RANDOMIZE THE POSITIONS OF AGENTS
    agent_counter = 1
    agent_locations = []

    while agent_counter <= num_agents:
        agent_rand_pos = pos[agent_counter - 1][0]
        x = agent_rand_pos[0]
        y = agent_rand_pos[1]
        if(world[x, y] == 0):
            world[x, y] = agent_counter
            agent_locations.append((x, y))
            agent_counter += 1        
    
    # RANDOMIZE THE GOALS OF AGENTS
    goals = np.zeros(world.shape).astype(int)
    goal_counter = 1   
    while goal_counter <= num_agents:
        goal_rand_pos = pos[goal_counter - 1][1]
        x = goal_rand_pos[0]
        y = goal_rand_pos[1]
        if(goals[x, y] == 0 and world[x, y] != -1):
            goals[x, y] = goal_counter
            goal_counter += 1

    return world, goals

if __name__ == "__main__":
    primal_dir = Path(__file__).resolve().parent

    results_path = primal_dir / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    model_primal_dir = primal_dir / "model_primal"
    primal = PRIMAL(str(model_primal_dir), 10)

    baseline_dir = primal_dir.parent
    dataset_dir = baseline_dir / "Dataset"
    model_save_name = "PRIMAL"

    map_configurations = [
        {
            "map_name": "15_15_simple_warehouse",
            "size": 15,
            "n_tests": 200,
            "list_num_agents": [4, 8, 12, 16, 20, 22]
        },
        {
            "map_name": "50_55_simple_warehouse",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8, 16, 32, 64, 128, 256]  # Default, adjust if needed
        },
        {
            "map_name": "50_55_long_shelves",
            "size": 50,  # Assuming size based on map name, adjust if needed
            "n_tests": 200,
            "list_num_agents": [4, 8, 16, 32, 64, 128, 256]  # Default, adjust if needed
        },
    ]

    header = ["n_agents", 
              "success_rate", "time", "time_std", "time_min", "time_max",
              "episode_length", "episode_length_std", "episode_length_min", "episode_length_max",
              "total_step", "total_step_std", "total_step_min", "total_step_max",
              "avg_step", "avg_step_std", "avg_step_min", "avg_step_max",
              "max_step", "max_step_std", "max_step_min", "max_step_max",
              "min_step", "min_step_std", "min_step_min", "min_step_max",
              "total_costs", "total_costs_std", "total_costs_min", "total_costs_max",
              "avg_costs", "avg_costs_std", "avg_costs_min", "avg_costs_max",
              "max_costs", "max_costs_std", "max_costs_min", "max_costs_max",
              "min_costs", "min_costs_std", "min_costs_min", "min_costs_max",
              "agent_collision_rate", "agent_collision_rate_std", "agent_collision_rate_min", "agent_collision_rate_max",
              "obstacle_collision_rate", "obstacle_collision_rate_std", "obstacle_collision_rate_min", "obstacle_collision_rate_max",
              "total_collision_rate", "total_collision_rate_std", "total_collision_rate_min", "total_collision_rate_max"]

    for config in map_configurations:
        map_name = config["map_name"]
        size = config["size"]
        n_tests = config["n_tests"]
        list_num_agents = config["list_num_agents"]

        print(f"\nProcessing map: {map_name}")

        output_dir = dataset_dir / map_name / "output" / model_save_name
        output_dir.mkdir(parents=True, exist_ok=True)

        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        sanitized_map_name = map_name.replace("/", "_").replace("\\", "_")
        csv_filename_base = f'{model_save_name}_{sanitized_map_name}_{date}'
        csv_file, csv_logger = get_csv_logger(str(results_path), csv_filename_base)

        csv_logger.writerow(header)
        csv_file.flush()

        for num_agents in list_num_agents:
            output_agent_dir = output_dir / f"{num_agents}_agents"
            output_agent_dir.mkdir(parents=True, exist_ok=True)

            results = dict()
            results['finished'] = []
            results['time'] = []
            results['episode_length'] = []
            results['total_steps'] = []
            results['avg_steps'] = []
            results['max_steps'] = []
            results['total_costs'] = []
            results['avg_costs'] = []
            results['max_costs'] = []
            results['min_steps'] = []
            results['min_costs'] = []
            results['crashed'] = []
            results['agent_coll_rate'] = []
            results['obstacle_coll_rate'] = []
            results['total_coll_rate'] = []
            print(f"Starting tests for {num_agents} agents on map {map_name}")

            for iter_idx in tqdm(range(n_tests), desc=f"Map: {map_name}, Agents: {num_agents}"):
                res, solution = run_simulations(str(dataset_dir), map_name, (num_agents, size, iter_idx), primal)
                results['finished'].append(res['finished'])
                if res['finished']:
                    results['time'].append(res['time'])
                    results['episode_length'].append(res['episode_length'])
                    results['total_steps'].append(res['total_steps'])
                    results['avg_steps'].append(res['avg_steps'])
                    results['max_steps'].append(res['max_steps'])
                    results['min_steps'].append(res['min_steps'])
                    results['total_costs'].append(res['total_costs'])
                    results['avg_costs'].append(res['avg_costs'])
                    results['max_costs'].append(res['max_costs'])
                    results['min_costs'].append(res['min_costs'])
                    results['agent_coll_rate'].append(res['agent_coll_rate'])
                    results['obstacle_coll_rate'].append(res['obstacle_coll_rate'])
                    results['total_coll_rate'].append(res['total_coll_rate'])
                    results['crashed'].append(res['crashed'])

                solution_filepath = output_agent_dir / f"solution_{model_save_name}_{map_name}_{num_agents}_agents_ID_{str(iter_idx).zfill(3)}.txt"
                with open(solution_filepath, 'w') as f:
                    f.write("Metrics:\n")
                    json.dump(res, f, indent=4)
                    f.write("\n\nSolution:\n")
                    if solution:
                        for agent_path in solution:
                            f.write(f"{agent_path}\n")
                    else:
                        f.write("No solution found or OutOfTimeError.\n")

            final_results = dict()
            final_results['finished'] = np.sum(results['finished']) / len(results['finished']) if len(results['finished']) > 0 else 0
            final_results['time'] = np.mean(results['time']) if results['time'] else 0
            final_results['episode_length'] = np.mean(results['episode_length']) if results['episode_length'] else 0
            final_results['total_steps'] = np.mean(results['total_steps']) if results['total_steps'] else 0
            final_results['avg_steps'] = np.mean(results['avg_steps']) if results['avg_steps'] else 0
            final_results['max_steps'] = np.mean(results['max_steps']) if results['max_steps'] else 0
            final_results['min_steps'] = np.mean(results['min_steps']) if results['min_steps'] else 0
            final_results['total_costs'] = np.mean(results['total_costs']) if results['total_costs'] else 0
            final_results['avg_costs'] = np.mean(results['avg_costs']) if results['avg_costs'] else 0
            final_results['max_costs'] = np.mean(results['max_costs']) if results['max_costs'] else 0
            final_results['min_costs'] = np.mean(results['min_costs']) if results['min_costs'] else 0
            final_results['crashed'] = np.sum(results['crashed']) / len(results['crashed']) if len(results['crashed']) > 0 else 0
            final_results['agent_coll_rate'] = np.mean(results['agent_coll_rate']) if results['agent_coll_rate'] else 0
            final_results['obstacle_coll_rate'] = np.mean(results['obstacle_coll_rate']) if results['obstacle_coll_rate'] else 0
            final_results['total_coll_rate'] = np.mean(results['total_coll_rate']) if results['total_coll_rate'] else 0
            print(final_results)

            data = [num_agents,
                final_results['finished'] * 100,
                final_results['time'],
                np.std(results['time']) if results['time'] else 0,
                np.min(results['time']) if results['time'] else 0,
                np.max(results['time']) if results['time'] else 0,
                final_results['episode_length'],
                np.std(results['episode_length']) if results['episode_length'] else 0,
                np.min(results['episode_length']) if results['episode_length'] else 0,
                np.max(results['episode_length']) if results['episode_length'] else 0,
                final_results['total_steps'],
                np.std(results['total_steps']) if results['total_steps'] else 0,
                np.min(results['total_steps']) if results['total_steps'] else 0,
                np.max(results['total_steps']) if results['total_steps'] else 0,
                final_results['avg_steps'],
                np.std(results['avg_steps']) if results['avg_steps'] else 0,
                np.min(results['avg_steps']) if results['avg_steps'] else 0,
                np.max(results['avg_steps']) if results['avg_steps'] else 0,
                final_results['max_steps'],
                np.std(results['max_steps']) if results['max_steps'] else 0,
                np.min(results['max_steps']) if results['max_steps'] else 0,
                np.max(results['max_steps']) if results['max_steps'] else 0,
                final_results['min_steps'],
                np.std(results['min_steps']) if results['min_steps'] else 0,
                np.min(results['min_steps']) if results['min_steps'] else 0,
                np.max(results['min_steps']) if results['min_steps'] else 0,
                final_results['total_costs'],
                np.std(results['total_costs']) if results['total_costs'] else 0,
                np.min(results['total_costs']) if results['total_costs'] else 0,
                np.max(results['total_costs']) if results['total_costs'] else 0,
                final_results['avg_costs'],
                np.std(results['avg_costs']) if results['avg_costs'] else 0,
                np.min(results['avg_costs']) if results['avg_costs'] else 0,
                np.max(results['avg_costs']) if results['avg_costs'] else 0,
                final_results['max_costs'],
                np.std(results['max_costs']) if results['max_costs'] else 0,
                np.min(results['max_costs']) if results['max_costs'] else 0,
                np.max(results['max_costs']) if results['max_costs'] else 0,
                final_results['min_costs'],
                np.std(results['min_costs']) if results['min_costs'] else 0,
                np.min(results['min_costs']) if results['min_costs'] else 0,
                np.max(results['min_costs']) if results['min_costs'] else 0,
                final_results['agent_coll_rate'] * 100,
                np.std(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                np.min(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                np.max(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                final_results['obstacle_coll_rate'] * 100,
                np.std(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                np.min(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                np.max(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                final_results['total_coll_rate'] * 100,
                np.std(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,
                np.min(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,
                np.max(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0
                ]
            csv_logger.writerow(data)
            csv_file.flush()
        
        csv_file.close()

print("finished all tests!")
