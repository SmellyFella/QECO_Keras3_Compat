from MEC_Env import MEC
from D3QN import DuelingDoubleDeepQNetwork
from Config import Config
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil


def normalize(parameter, minimum, maximum):
    normalized_parameter = (parameter - minimum) / (maximum - minimum)
    return normalized_parameter


def QoE_Function(delay, max_delay, unfinish_task, ue_energy_state, ue_comp_energy, ue_trans_energy, edge_comp_energy, ue_idle_energy):
    
    edge_energy  = next((e for e in edge_comp_energy if e != 0), 0)
    idle_energy = next((e for e in ue_idle_energy if e != 0), 0)

    energy_cons = ue_comp_energy + ue_trans_energy #+ edge_energy + idle_energy
    #print(ue_comp_energy , ue_trans_energy , edge_energy , idle_energy)
    #print(ue_energy_state, delay, energy_cons)
    
    scaled_energy = normalize(energy_cons, 0, 20)*10
    cost = 2 * ((ue_energy_state * delay) + ((1 - ue_energy_state) * scaled_energy))

    Reward = max_delay*4

    if unfinish_task:
        QoE = - cost
    else:
        QoE = Reward - cost

    return QoE

def Drop_Count(ue_RL_list, episode):
    drrop_delay10 = 0 
    drrop = 0

    # Sum unfinished tasks from environment
    for time_index in range(min(100, env.n_time)):   
        drrop += sum(env.unfinish_task[time_index])

    # Count delays equal to 10 from each UE's delay_store
    for ue in ue_RL_list:
        if episode < len(ue.delay_store):
            for val in ue.delay_store[episode]:
                if val == 10:
                    drrop_delay10 += 1

    # Optional: you can return both if you want
    # return drrop, drrop_delay10
    return drrop

def Cal_QoE(ue_RL_list, episode):
    episode_rewards = []

    for ue in ue_RL_list:
        # Only process if the episode exists
        if episode < len(ue.reward_store):
            episode_data = ue.reward_store[episode]
            if len(episode_data) > 0:
                episode_rewards.append(sum(episode_data))
            else:
                episode_rewards.append(0.0)
        else:
            episode_rewards.append(0.0)

    if len(episode_rewards) == 0:
        return 0.0

    avg_episode_sum_reward = sum(episode_rewards) / len(ue_RL_list)
    return avg_episode_sum_reward

def Cal_Delay(ue_RL_list, episode):
    avg_delay_in_episode = []

    for i in range(len(ue_RL_list)):
        # Check if this episode exists in delay_store
        if episode < len(ue_RL_list[i].delay_store):
            for j in range(len(ue_RL_list[i].delay_store[episode])):
                if ue_RL_list[i].delay_store[episode][j] != 0:
                    avg_delay_in_episode.append(ue_RL_list[i].delay_store[episode][j])

    # Avoid division by zero if no delays were recorded
    if len(avg_delay_in_episode) > 0:
        return sum(avg_delay_in_episode) / len(avg_delay_in_episode)
    else:
        return 0

def Cal_Energy(ue_RL_list, episode):
    energy_ue_list = []

    for ue_RL in ue_RL_list:
        # Only process if the episode exists
        if episode < len(ue_RL.energy_store):
            energy_ue_list.append(sum(ue_RL.energy_store[episode]))
        else:
            energy_ue_list.append(0.0)  # Default if episode missing

    if len(energy_ue_list) == 0:
        return 0.0

    return sum(energy_ue_list) / len(energy_ue_list)
"""
def Cal_Successful_Offloads(ue_RL_list, episode):
    offloads = []

    for ue in ue_RL_list:
        # Only process if the episode exists
        if episode < len(ue.offload_store):
            offloads.append(sum(ue.offload_store[episode]))
        else:
            offloads.append(0.0)  # Default if episode missing

    if len(offloads) == 0:
        return 0.0

    return sum(offloads) / len(ue_RL_list)  # average per UE
"""
def train(ue_RL_list, NUM_EPISODE):
    avg_QoE_list = []
    avg_delay_list = []
    energy_cons_list = []
    num_drop_list = []
    avg_reward_list = []
    avg_reward_list_2 = []
    avg_delay_list_in_episode = []
    avg_energy_list_in_episode = []
    num_task_drop_list_in_episode = []
    RL_step = 0
    a = 1

    #adding a list to track successful offloads and list to store number of tasks arriving
    #avg_offload_success_list = []
    tasks_arrived_list = []

    offload_success_list = []

    for episode in range(NUM_EPISODE):

        print("\n-*-**-***-*****-********-*************-********-*****-***-**-*-")
        print("Episode  :", episode, )
        print("Epsilon  :", ue_RL_list[0].epsilon)

        # BITRATE ARRIVAL
        #Below replaced with periodic instead of probabilistic bit rate to better emulate smart meters
        ##bitarrive_size = np.random.uniform(env.min_arrive_size, env.max_arrive_size, size=[env.n_time, env.n_ue])
        bitarrive_size = np.zeros([env.n_time, env.n_ue])
        for ue in range(env.n_ue):
            for t in range(0, env.n_time, env.smart_meter_period):
                bitarrive_size[t, ue] = np.random.uniform(env.min_arrive_size, env.max_arrive_size)

        
        task_prob = env.task_arrive_prob
        bitarrive_size = bitarrive_size * (np.random.uniform(0, 1, size=[env.n_time, env.n_ue]) < task_prob)
        bitarrive_size[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_ue])

        bitarrive_dens = np.zeros([env.n_time, env.n_ue])
        for i in range(len(bitarrive_size)):
            for j in range(len(bitarrive_size[i])):
                if bitarrive_size[i][j] != 0:
                    bitarrive_dens[i][j] = Config.TASK_COMP_DENS[np.random.randint(0, len(Config.TASK_COMP_DENS))]


        test = 0 
        for i in range(len(bitarrive_size)):
            for j in range(len(bitarrive_size[i])):
                if bitarrive_size[i][j] != 0: 
                    test = test + 1

        print("Num_Task_Arrive: ", test)

        tasks_arrived_list.append(test)

        Check = []
        for i in range(len(bitarrive_size)):
            Check.append(sum(bitarrive_size[i]))

        #print("Sum_Arrived_Task_Size:", int(sum(Check)))

        #print(bitarrive_dens) = [Config.TASK_COMP_DENS[np.random.randint(0, len(Config.TASK_COMP_DENS))] ]

        # OBSERVATION MATRIX SETTING
        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for ue_index in range(env.n_ue):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_ue])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = env.reset(bitarrive_size, bitarrive_dens)
        #print(observation_all)
        #print(lstm_state_all)


        # TRAIN DRL
        while True:
    
    
            # PERFORM ACTION
            action_all = np.zeros([env.n_ue])
            for ue_index in range(env.n_ue):
                observation = np.squeeze(observation_all[ue_index, :])
                if np.sum(observation) == 0:
                    # if there is no task, action = 0 (also need to be stored)
                    action_all[ue_index] = 0
                else:
                    action_all[ue_index] = ue_RL_list[ue_index].choose_action(observation)
                    if observation[0] != 0:
                        ue_RL_list[ue_index].do_store_action(episode, env.time_count, action_all[ue_index])

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            observation_all_, lstm_state_all_, done = env.step(action_all)

            #print("+++___+++")
            #print(observation_all_)
            #print(lstm_state_all_)


            # should store this information in EACH time slot
            for ue_index in range(env.n_ue):
                ue_RL_list[ue_index].update_lstm(lstm_state_all_[ue_index,:])

            process_delay = env.process_delay
            unfinish_task = env.unfinish_task

            """

            #Track the successful offloads:
            for ue_index in range(env.n_ue):
                # Example: action==1 means "offload to fog"
                processed_bits = env.edge_bit_processed[env.time_count -1, ue_index]
                if action_all[ue_index] > 0: # and np.sum(processed_bits) > 0:
                    success_flag = 1
                else:
                    success_flag = 0
            
                ue_RL_list[ue_index].do_store_offload(
                    episode,
                    env.time_count-1,   # current timestep
                    success_flag
                )
                #print(f"Episode {episode}, t={env.time_count-1}, UE {ue_index}, action={action_all[ue_index]}, processed={processed_bits}")
                
           """ 

            # STORE MEMORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            for ue_index in range(env.n_ue):

                history[env.time_count - 1][ue_index]['observation'] = observation_all[ue_index, :]
                history[env.time_count - 1][ue_index]['lstm'] = np.squeeze(lstm_state_all[ue_index, :])
                history[env.time_count - 1][ue_index]['action'] = action_all[ue_index]
                history[env.time_count - 1][ue_index]['observation_'] = observation_all_[ue_index]
                history[env.time_count - 1][ue_index]['lstm_'] = np.squeeze(lstm_state_all_[ue_index,:])

                update_index = np.where((1 - reward_indicator[:,ue_index]) * process_delay[:,ue_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]
                        ue_RL_list[ue_index].store_transition(history[time_index][ue_index]['observation'],
                                                                history[time_index][ue_index]['lstm'],
                                                                history[time_index][ue_index]['action'],
                                                                QoE_Function(process_delay[time_index, ue_index],
                                                                                env.max_delay,
                                                                                unfinish_task[time_index, ue_index],
                                                                                env.ue_energy_state[ue_index],
                                                                                env.ue_comp_energy[time_index, ue_index],
                                                                                env.ue_tran_energy [time_index, ue_index],
                                                                                env.edge_comp_energy[time_index, ue_index],
                                                                                env.ue_idle_energy[time_index, ue_index]),
                                                                history[time_index][ue_index]['observation_'],
                                                                history[time_index][ue_index]['lstm_'])
                        ue_RL_list[ue_index].do_store_reward(episode, time_index,
                                                               QoE_Function(process_delay[time_index, ue_index],
                                                                                env.max_delay,
                                                                                unfinish_task[time_index, ue_index],
                                                                                env.ue_energy_state[ue_index],
                                                                                env.ue_comp_energy[time_index, ue_index],
                                                                                env.ue_tran_energy [time_index, ue_index],
                                                                                env.edge_comp_energy[time_index, ue_index],
                                                                                env.ue_idle_energy[time_index, ue_index]))
                        ue_RL_list[ue_index].do_store_delay(episode, time_index,
                                                              process_delay[time_index, ue_index])

                        ue_RL_list[ue_index].do_store_energy(
                            episode,
                            time_index,
                            env.ue_comp_energy[time_index, ue_index],
                            env.ue_tran_energy [time_index, ue_index],
                            env.edge_comp_energy[time_index, ue_index],
                            env.ue_idle_energy[time_index, ue_index])

                        reward_indicator[time_index, ue_index] = 1


            # ADD STEP (one step does not mean one store)
            RL_step += 1

            # UPDATE OBSERVATION
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            # CONTROL LEARNING START TIME AND FREQUENCY
            if (RL_step > 200) and (RL_step % 10 == 0):
                for ue in range(env.n_ue):
                    ue_RL_list[ue].learn()

            # GAME ENDS

            if done:
                with open("Delay.txt", 'a') as f:
                            f.write('\n' + str(Cal_Delay(ue_RL_list, episode)))

                with open("Energy.txt", 'a') as f:
                            f.write('\n' + str(Cal_Energy(ue_RL_list, episode)))

                with open("QoE.txt", 'a') as f:
                            f.write('\n' + str(Cal_QoE(ue_RL_list, episode)))

                with open("Drop.txt", 'a') as f:
                            f.write('\n' + str(Drop_Count(ue_RL_list, episode)))

                #with open("Success.txt", 'a') as f:
               #             f.write('\n' + str(Cal_Successful_Offloads(ue_RL_list, episode)))



                '''

                for task in env.task_history:
                    cmpl = drp = 0
                    for t in task:
                        d_states = t['d_state']
                        if any(d < 0 for d in d_states):
                            t['state'] = 'D'
                            drp += 1
                        elif all(d > 0 for d in d_states):
                            t['state'] = 'C'
                            cmpl += 1
                full_complete_task = 0
                full_drop_task = 0
                complete_task = 0
                drop_task = 0
                for history in env.task_history:
                    for task in history:
                        if task['state'] == 'C':
                            full_complete_task += 1
                        elif task['state'] == 'D':
                            full_drop_task += 1
                        for component_state in task['d_state']:
                            if component_state == 1:
                                complete_task += 1
                            elif component_state == -1:
                                drop_task += 1
                cnt = len(env.task_history) * len(env.task_history[0]) * env.n_component

                #a = Drop_Count(ue_RL_list, episode)
                '''

                
                if episode % 200 == 0 and episode != 0:
                    os.makedirs("models" + "/" + str(episode))
                    for ue in range(env.n_ue):
                        ue_RL_list[ue].saver.save(ue_RL_list[ue].sess, "models/" + str(episode) +'/'+ str(ue) + "_X_model" +'/model.ckpt', global_step=episode)
                        print("UE", ue, "Network_model_seved\n")
                
                
                if episode % 999 == 0 and episode != 0:
                    os.makedirs("models" + "/" + str(episode))
                    for ue in range(env.n_ue):
                        ue_RL_list[ue].saver.save(ue_RL_list[ue].sess, "models/" + str(episode) +'/'+ str(ue) + "_X_model" +'/model.ckpt', global_step=episode)
                        print("UE", ue, "Network_model_seved\n")


                    

                # Process energy
                ue_bit_processed = sum(sum(env.ue_bit_processed))
                ue_comp_energy = sum(sum(env.ue_comp_energy))

                # Transmission energy
                ue_bit_transmitted = sum(sum(env.ue_bit_transmitted))
                ue_tran_energy = sum(sum(env.ue_tran_energy))

                # edge energy
                edge_bit_processed = sum(sum(env.edge_bit_processed))
                edge_comp_energy = sum(sum(env.edge_comp_energy))
                ue_idle_energy = sum(sum(env.ue_idle_energy))

                avg_delay  = Cal_Delay(ue_RL_list, episode)
                avg_energy = Cal_Energy(ue_RL_list, episode)
                avg_QoE   = Cal_QoE(ue_RL_list, episode)
                
                #avg_success = Cal_Successful_Offloads(ue_RL_list, episode)


                avg_QoE_list.append(avg_QoE)
                avg_delay_list.append(avg_delay)
                energy_cons_list.append(avg_energy)
                num_drop_list.append(env.drop_trans_count+env.drop_edge_count+env.drop_ue_count)

                #avg_offload_success_list.append(avg_success)

                avg_reward_list.append(-(Cal_QoE(ue_RL_list, episode)))

                offload_success_list.append(env.successful_offloads)

                # Append metrics to tracking lists
                if episode % 10 == 0:
                    avg_reward_list_2.append(sum(avg_reward_list[episode-10:episode]) / 10)
                    avg_delay_list_in_episode.append(Cal_Delay(ue_RL_list, episode))
                    avg_energy_list_in_episode.append(Cal_Energy(ue_RL_list, episode))

                    # Create a figure with 4 vertically stacked subplots
                    fig, axs = plt.subplots(7, 1, figsize=(10, 20))
                    fig.suptitle('Performance Metrics Over Episodes', fontsize=16, y=0.92)

                    # Subplot for Average QoE
                    axs[0].plot(avg_QoE_list, marker='o', linestyle='-', color='b', label='Avg QoE')
                    axs[0].set_title('', fontsize=14)
                    axs[0].set_ylabel('Average QoE')
                    axs[0].set_xlabel('Episode')
                    axs[0].grid(True, linestyle='--', alpha=0.7)
                    axs[0].legend()

                    # Subplot for Average Delay
                    axs[1].plot(avg_delay_list, marker='s', linestyle='-', color='g', label='Avg Delay')
                    axs[1].set_title('', fontsize=14)
                    axs[1].set_ylabel('Average Delay')
                    axs[1].set_xlabel('Episode')
                    axs[1].grid(True, linestyle='--', alpha=0.7)
                    axs[1].legend()

                    # Subplot for Energy Consumption
                    axs[2].plot(energy_cons_list, marker='^', linestyle='-', color='r', label='Energy Cons.')
                    axs[2].set_title('', fontsize=14)
                    axs[2].set_ylabel('Energy Consumption')
                    axs[2].set_xlabel('Episode')
                    axs[2].grid(True, linestyle='--', alpha=0.7)
                    axs[2].legend()

                    # Subplot for Number of Drops
                    axs[3].plot(num_drop_list, marker='x', linestyle='-', color='m', label='Num Drops')
                    axs[3].set_title('', fontsize=14)
                    axs[3].set_ylabel('Number Drops')
                    axs[3].set_xlabel('Episode')
                    axs[3].grid(True, linestyle='--', alpha=0.7)
                    axs[3].legend()

                    # Subplot for Successful Offloads
                    axs[4].plot(offload_success_list, marker='x', linestyle='-', color='y', label='Successes')
                    axs[4].set_title('', fontsize=14)
                    axs[4].set_ylabel('Successful Offloads')
                    axs[4].set_xlabel('Episode')
                    axs[4].grid(True, linestyle='--', alpha=0.7)
                    axs[4].legend()

                    # Subplot for tasks arrived vs energy consumption
                    energy_per_task = [
                      e/t if t > 0 else 0 
                      for e, t in zip(energy_cons_list, tasks_arrived_list)
                    ]

                    axs[5].plot(energy_per_task, marker='x', linestyle='-', color='g', label='Tasks vs Energy')
                    axs[5].set_title('', fontsize=14)
                    axs[5].set_ylabel('Energy Consumption / task arrived')
                    axs[5].set_xlabel('Episode')
                    axs[5].grid(True, linestyle='--', alpha=0.7)
                    axs[5].legend()

                    # Subplot for task vs time 
                    delay_per_task = [
                      e/t if t > 0 else 0 
                      for e, t in zip(avg_delay_list, tasks_arrived_list)
                    ]

                    axs[6].plot(delay_per_task, marker='x', linestyle='-', color='g', label='Task vs Time')
                    axs[6].set_title('', fontsize=14)
                    axs[6].set_ylabel('Delay / task arrived')
                    axs[6].set_xlabel('Episodes')
                    axs[6].grid(True, linestyle='--', alpha=0.7)
                    axs[6].legend()

                    # Save the figure to a file
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.9)
                    plt.savefig('Performance_Chart.png', dpi=100)
                    #plt.show()



                print("SystemPerformance: ---------------------------------------------------------------------")
                #print("Num_Completed :  ", )
                print("Num_Dropped   :  ", env.drop_trans_count+env.drop_edge_count+env.drop_ue_count, "[Trans_Drop: ", env.drop_trans_count, "Edge_Drop: ", env.drop_edge_count, "UE_Drop: ", env.drop_ue_count,"]")
                print("Avg_Delay     :  ", "%0.1f" %avg_delay)
                print("Avg_Energy    :  ", "%0.1f" %avg_energy)
                print("Avg_QoE       :  ", "%0.1f" %avg_QoE)
                print("EnergyCosumption: ----------------------------------------------------------------------")
                print("Local         :  ", "%0.1f" %ue_comp_energy, "[ue_bit_processed:", int(ue_bit_processed),"]")
                print("Trans         :  ", "%0.1f" %ue_tran_energy, "[ue_bit_transmitted:", int(ue_bit_transmitted),"]")
                print("Edges         :  ", "%0.1f" % sum(ue_idle_energy), "[edge_bit_processed :", int(sum(edge_bit_processed)),"]")
                #print("--------------------------------------------------------------------------------------------------------")
                #print("Trans_Drop: ", env.drop_trans_count, "Edge_Drop: ", env.drop_edge_count, "UE_Drop: ", env.drop_ue_count)
                #print("Drop_Count: ",Drop_Count(ue_RL_list, episode))

                break # Training Finished


if __name__ == "__main__":

    # GENERATE ENVIRONMENT
    env = MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)

    # GENERATE MULTIPLE CLASSES FOR RL
    ue_RL_list = list()
    for ue in range(Config.N_UE):
        ue_RL_list.append(DuelingDoubleDeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                                    learning_rate       = Config.LEARNING_RATE,
                                                    reward_decay        = Config.REWARD_DECAY,
                                                    e_greedy            = Config.E_GREEDY,
                                                    replace_target_iter = Config.N_NETWORK_UPDATE,  
                                                    memory_size         = Config.MEMORY_SIZE,  
                                                    ))



    # LOAD Trained MODEL 
    '''
    for ue in range(Config.N_UE):
        ue_RL_list[ue].Initialize(ue_RL_list[ue].sess, ue)
        ue_RL_list[ue].epsilon = 1
    '''

    Delay  = open("Delay.txt" , 'w')
    Energy = open("Energy.txt", 'w')
    QoE    = open("QoE.txt"   , 'w')
    Drop   = open("Drop.txt"  , 'w')
                           

    # TRAIN THE SYSTEM
    train(ue_RL_list, Config.N_EPISODE)




