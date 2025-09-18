from SmartGrid_env import SmartGrid
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


def QoE_Function(delay, max_delay, unfinish_task, meter_energy_state, meter_comp_energy, meter_trans_energy, substation_comp_energy, meter_idle_energy):
    
    substation_energy  = next((e for e in substation_comp_energy if e != 0), 0)
    idle_energy = next((e for e in meter_idle_energy if e != 0), 0)

    energy_cons = meter_comp_energy + meter_trans_energy #+ substation_energy + idle_energy
    #print(meter_comp_energy , meter_trans_energy , substation_energy , idle_energy)
    #print(meter_energy_state, delay, energy_cons)
    
    scaled_energy = normalize(energy_cons, 0, 20)*10
    cost = 2 * ((meter_energy_state * delay) + ((1 - meter_energy_state) * scaled_energy))

    Reward = max_delay*4

    if unfinish_task:
        QoE = - cost
    else:
        QoE = Reward - cost

    return QoE

def Drop_Count(meter_RL_list, episode):
    drrop_delay10 = 0 
    drrop = 0

    # Sum unfinished tasks from environment
    for time_index in range(min(100, env.n_time)):   
        drrop += sum(env.unfinish_task[time_index])

    # Count delays equal to 10 from each UE's delay_store
    for meter in meter_RL_list:
        if episode < len(meter.delay_store):
            for val in meter.delay_store[episode]:
                if val == 10:
                    drrop_delay10 += 1

    # Optional: you can return both if you want
    # return drrop, drrop_delay10
    return drrop

def Cal_QoE(meter_RL_list, episode):
    episode_rewards = []

    for meter in meter_RL_list:
        # Only process if the episode exists
        if episode < len(meter.reward_store):
            episode_data = meter.reward_store[episode]
            if len(episode_data) > 0:
                episode_rewards.append(sum(episode_data))
            else:
                episode_rewards.append(0.0)
        else:
            episode_rewards.append(0.0)

    if len(episode_rewards) == 0:
        return 0.0

    avg_episode_sum_reward = sum(episode_rewards) / len(meter_RL_list)
    return avg_episode_sum_reward

def Cal_Delay(meter_RL_list, episode):
    avg_delay_in_episode = []

    for i in range(len(meter_RL_list)):
        # Check if this episode exists in delay_store
        if episode < len(meter_RL_list[i].delay_store):
            for j in range(len(meter_RL_list[i].delay_store[episode])):
                if meter_RL_list[i].delay_store[episode][j] != 0:
                    avg_delay_in_episode.append(meter_RL_list[i].delay_store[episode][j])

    # Avoid division by zero if no delays were recorded
    if len(avg_delay_in_episode) > 0:
        return sum(avg_delay_in_episode) / len(avg_delay_in_episode)
    else:
        return 0

def Cal_Energy(meter_RL_list, episode):
    energy_meter_list = []

    for meter_RL in meter_RL_list:
        # Only process if the episode exists
        if episode < len(meter_RL.energy_store):
            energy_meter_list.append(sum(meter_RL.energy_store[episode]))
        else:
            energy_meter_list.append(0.0)  # Default if episode missing

    if len(energy_meter_list) == 0:
        return 0.0

    return sum(energy_meter_list) / len(energy_meter_list)

def Cal_Total_Offloads(meter_RL_list, episode):
    offloads = []

    for meter in meter_RL_list:
        # Only process if the episode exists
        if episode < len(meter.offload_store):
            offloads.append(sum(meter.offload_store[episode]))
        else:
            offloads.append(0.0)  # Default if episode missing

    if len(offloads) == 0:
        return 0.0

    return sum(offloads) / len(meter_RL_list)  # average per UE

def train(meter_RL_list, NUM_EPISODE):
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
    total_offload_attempt_list = []
    tasks_arrived_list = []

    offload_success_list = []

    for episode in range(NUM_EPISODE):

        print("\n-*-**-***-*****-********-*************-********-*****-***-**-*-")
        print("Episode  :", episode, )
        print("Epsilon  :", meter_RL_list[0].epsilon)

        """
        # BITRATE ARRIVAL
        #Below replaced with periodic instead of probabilistic bit rate to better emulate smart meters
        ##bitarrive_size = np.random.uniform(env.min_arrive_size, env.max_arrive_size, size=[env.n_time, env.n_meter])
        bitarrive_size = np.zeros([env.n_time, env.n_meter])
        for meter in range(env.n_meter):
            for t in range(0, env.n_time, env.smart_meter_period):
                bitarrive_size[t, meter] = np.random.uniform(env.min_arrive_size, env.max_arrive_size)

        
        task_prob = env.task_arrive_prob
        bitarrive_size = bitarrive_size * (np.random.uniform(0, 1, size=[env.n_time, env.n_meter]) < task_prob)
        bitarrive_size[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_meter])

        bitarrive_dens = np.zeros([env.n_time, env.n_meter])
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

        # OBSERVATION MATRIX SETTING
        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for meter_index in range(env.n_meter):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_meter])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = env.reset(bitarrive_size, bitarrive_dens)
        #print(observation_all)
        #print(lstm_state_all)

        """
        ###BELOW IS GPT GENERATED SMARTGRID STYLE TASK GENERATION:
        # Generate smart-grid-like demand patterns instead of purely random
        bitarrive_size = np.zeros([env.n_time, env.n_meter])
        bitarrive_dens = np.zeros([env.n_time, env.n_meter])
        task_criticality = np.zeros([env.n_time, env.n_meter], dtype=int)
        
        hours = np.arange(env.n_time) % 24
        # Base demand curve: peaks around 8 AM and 7 PM
        base_demand = 0.5 + 0.5 * (
            np.sin((np.pi / 12) * (hours - 8))**2 + np.sin((np.pi / 12) * (hours - 19))**2
        )
        
        for meter in range(env.n_meter):
            for t in range(env.n_time):
                # Scale demand to your min/max arrival sizes
                scaled = env.min_arrive_size + base_demand[t] * (env.max_arrive_size - env.min_arrive_size)
                # Add small noise for variability
                noise = np.random.normal(0, 0.05 * scaled)
                size = max(0, scaled + noise)
                bitarrive_size[t, meter] = size

                if size > 0:
                    # Mark some percentage of tasks as critical (e.g., 20%)
                    if np.random.rand() < 0.2:
                        task_criticality[t, meter] = 1  # Critical
                    else:
                        task_criticality[t, meter] = 0  # Non-critical
        
                    # Critical tasks → higher computational density
                    if task_criticality[t, meter] == 1:
                        bitarrive_dens[t, meter] = max(Config.TASK_COMP_DENS)
                    else:
                        bitarrive_dens[t, meter] = np.random.choice(Config.TASK_COMP_DENS)
                
        # Optional: enforce reporting only at smart meter intervals
        mask = np.zeros_like(bitarrive_size)
        mask[::env.smart_meter_period, :] = 1
        bitarrive_size *= mask
        bitarrive_dens *= mask
        task_criticality *= mask
        
        # Compute computational density (urgency) based on demand level
        bitarrive_dens = np.zeros([env.n_time, env.n_meter])
        for i in range(env.n_time):
            for j in range(env.n_meter):
                if bitarrive_size[i, j] > 0:
                    # Higher demand → pick a denser requirement
                    if bitarrive_size[i, j] > (env.max_arrive_size + env.min_arrive_size) / 2:
                        bitarrive_dens[i, j] = max(Config.TASK_COMP_DENS)
                    else:
                        bitarrive_dens[i, j] = np.random.choice(Config.TASK_COMP_DENS)
        
        # Count arrivals for debugging
        test = np.count_nonzero(bitarrive_size)
        print("Num_Task_Arrive:", test)
        tasks_arrived_list.append(test)
        critical_count = np.count_nonzero(task_criticality)
        print(f"Critical tasks: {critical_count}, Total tasks: {np.count_nonzero(bitarrive_size)}")

        # OBSERVATION MATRIX SETTING
        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for meter_index in range(env.n_meter):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_meter])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = env.reset(bitarrive_size, bitarrive_dens)
        #print(observation_all)
        #print(lstm_state_all)
        
        ###END OF GPT PROVIDED CODE
        
        # TRAIN DRL
        while True:
    
    
            # PERFORM ACTION
            action_all = np.zeros([env.n_meter])
            for meter_index in range(env.n_meter):
                observation = np.squeeze(observation_all[meter_index, :])
                if np.sum(observation) == 0:
                    # if there is no task, action = 0 (also need to be stored)
                    action_all[meter_index] = 0
                else:
                    action_all[meter_index] = meter_RL_list[meter_index].choose_action(observation)
                    if observation[0] != 0:
                        meter_RL_list[meter_index].do_store_action(episode, env.time_count, action_all[meter_index])

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            observation_all_, lstm_state_all_, done = env.step(action_all)

            #print("+++___+++")
            #print(observation_all_)
            #print(lstm_state_all_)


            # should store this information in EACH time slot
            for meter_index in range(env.n_meter):
                meter_RL_list[meter_index].update_lstm(lstm_state_all_[meter_index,:])

            process_delay = env.process_delay
            unfinish_task = env.unfinish_task

            

            #Track the successful offloads:
            for meter_index in range(env.n_meter):
                # Example: action==1 means "offload to fog"
                processed_bits = env.meter_bit_processed[env.time_count -1, meter_index]
                if action_all[meter_index] > 0: # and np.sum(processed_bits) > 0:
                    success_flag = 1
                else:
                    success_flag = 0
            
                meter_RL_list[meter_index].do_store_offload(
                    episode,
                    env.time_count-1,   # current timestep
                    success_flag
                )
                #print(f"Episode {episode}, t={env.time_count-1}, UE {meter_index}, action={action_all[meter_index]}, processed={processed_bits}")
                
          

            # STORE MEMORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            for meter_index in range(env.n_meter):

                history[env.time_count - 1][meter_index]['observation'] = observation_all[meter_index, :]
                history[env.time_count - 1][meter_index]['lstm'] = np.squeeze(lstm_state_all[meter_index, :])
                history[env.time_count - 1][meter_index]['action'] = action_all[meter_index]
                history[env.time_count - 1][meter_index]['observation_'] = observation_all_[meter_index]
                history[env.time_count - 1][meter_index]['lstm_'] = np.squeeze(lstm_state_all_[meter_index,:])

                update_index = np.where((1 - reward_indicator[:,meter_index]) * process_delay[:,meter_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]
                        meter_RL_list[meter_index].store_transition(history[time_index][meter_index]['observation'],
                                                                history[time_index][meter_index]['lstm'],
                                                                history[time_index][meter_index]['action'],
                                                                QoE_Function(process_delay[time_index, meter_index],
                                                                                env.max_delay,
                                                                                unfinish_task[time_index, meter_index],
                                                                                env.meter_energy_state[meter_index],
                                                                                env.meter_comp_energy[time_index, meter_index],
                                                                                env.meter_tran_energy [time_index, meter_index],
                                                                                env.substation_comp_energy[time_index, meter_index],
                                                                                env.meter_idle_energy[time_index, meter_index]),
                                                                history[time_index][meter_index]['observation_'],
                                                                history[time_index][meter_index]['lstm_'])
                        meter_RL_list[meter_index].do_store_reward(episode, time_index,
                                                               QoE_Function(process_delay[time_index, meter_index],
                                                                                env.max_delay,
                                                                                unfinish_task[time_index, meter_index],
                                                                                env.meter_energy_state[meter_index],
                                                                                env.meter_comp_energy[time_index, meter_index],
                                                                                env.meter_tran_energy [time_index, meter_index],
                                                                                env.substation_comp_energy[time_index, meter_index],
                                                                                env.meter_idle_energy[time_index, meter_index]))
                        meter_RL_list[meter_index].do_store_delay(episode, time_index,
                                                              process_delay[time_index, meter_index])

                        meter_RL_list[meter_index].do_store_energy(
                            episode,
                            time_index,
                            env.meter_comp_energy[time_index, meter_index],
                            env.meter_tran_energy [time_index, meter_index],
                            env.substation_comp_energy[time_index, meter_index],
                            env.meter_idle_energy[time_index, meter_index])

                        reward_indicator[time_index, meter_index] = 1


            # ADD STEP (one step does not mean one store)
            RL_step += 1

            # UPDATE OBSERVATION
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            # CONTROL LEARNING START TIME AND FREQUENCY
            if (RL_step > 200) and (RL_step % 10 == 0):
                for meter in range(env.n_meter):
                    meter_RL_list[meter].learn()

            # GAME ENDS

            if done:
                with open("Delay.txt", 'a') as f:
                            f.write('\n' + str(Cal_Delay(meter_RL_list, episode)))

                with open("Energy.txt", 'a') as f:
                            f.write('\n' + str(Cal_Energy(meter_RL_list, episode)))

                with open("QoE.txt", 'a') as f:
                            f.write('\n' + str(Cal_QoE(meter_RL_list, episode)))

                with open("Drop.txt", 'a') as f:
                            f.write('\n' + str(Drop_Count(meter_RL_list, episode)))

                
               



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

                #a = Drop_Count(meter_RL_list, episode)
                '''

                
                if episode % 200 == 0 and episode != 0:
                    os.makedirs("models" + "/" + str(episode))
                    for meter in range(env.n_meter):
                        meter_RL_list[meter].saver.save(meter_RL_list[meter].sess, "models/" + str(episode) +'/'+ str(meter) + "_X_model" +'/model.ckpt', global_step=episode)
                        print("UE", meter, "Network_model_seved\n")
                
                
                if episode % 999 == 0 and episode != 0:
                    os.makedirs("models" + "/" + str(episode))
                    for meter in range(env.n_meter):
                        meter_RL_list[meter].saver.save(meter_RL_list[meter].sess, "models/" + str(episode) +'/'+ str(meter) + "_X_model" +'/model.ckpt', global_step=episode)
                        print("UE", meter, "Network_model_seved\n")


                    

                # Process energy
                meter_bit_processed = sum(sum(env.meter_bit_processed))
                meter_comp_energy = sum(sum(env.meter_comp_energy))

                # Transmission energy
                meter_bit_transmitted = sum(sum(env.meter_bit_transmitted))
                meter_tran_energy = sum(sum(env.meter_tran_energy))

                # substation energy
                substation_bit_processed = sum(sum(env.substation_bit_processed))
                substation_comp_energy = sum(sum(env.substation_comp_energy))
                meter_idle_energy = sum(sum(env.meter_idle_energy))

                avg_delay  = Cal_Delay(meter_RL_list, episode)
                avg_energy = Cal_Energy(meter_RL_list, episode)
                avg_QoE   = Cal_QoE(meter_RL_list, episode)
                
                total_offloads = Cal_Total_Offloads(meter_RL_list, episode)


                avg_QoE_list.append(avg_QoE)
                avg_delay_list.append(avg_delay)
                energy_cons_list.append(avg_energy)
                num_drop_list.append(env.drop_trans_count+env.drop_substation_count+env.drop_meter_count)

                total_offload_attempt_list.append(total_offloads)

                avg_reward_list.append(-(Cal_QoE(meter_RL_list, episode)))

                offload_success_list.append(env.successful_offloads)

                # Append metrics to tracking lists
                if episode % 10 == 0:
                    avg_reward_list_2.append(sum(avg_reward_list[episode-10:episode]) / 10)
                    avg_delay_list_in_episode.append(Cal_Delay(meter_RL_list, episode))
                    avg_energy_list_in_episode.append(Cal_Energy(meter_RL_list, episode))

                    # Create a figure with 4 vertically stacked subplots
                    fig, axs = plt.subplots(8, 1, figsize=(10, 20))
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

                    #Subplot for task vs offload ratio
                    offload_ratio = [
                        e/t if t > 0 else 0 
                        for e, t in zip(offload_success_list, total_offload_attempt_list)
                    ]
                    
                    print(offload_ratio)
                    offload_ratio_per_task = [
                      e/t if t > 0 else 0 
                      for e, t in zip(offload_ratio, tasks_arrived_list)
                    ]

                    axs[7].plot(offload_ratio_per_task, marker='x', linestyle='-', color='g', label='Task vs Offload ratio')
                    axs[7].set_title('', fontsize=14)
                    axs[7].set_ylabel('offload ratio / task arrived')
                    axs[7].set_xlabel('Episodes')
                    axs[7].grid(True, linestyle='--', alpha=0.7)
                    axs[7].legend()
                    

                    # Save the figure to a file
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.9)
                    plt.savefig('Performance_Chart.png', dpi=100)
                    #plt.show()



                print("SystemPerformance: ---------------------------------------------------------------------")
                #print("Num_Completed :  ", )
                print("Num_Dropped   :  ", env.drop_trans_count+env.drop_substation_count+env.drop_meter_count, "[Trans_Drop: ", env.drop_trans_count, "Substation_Drop: ", env.drop_substation_count, "UE_Drop: ", env.drop_meter_count,"]")
                print("Avg_Delay     :  ", "%0.1f" %avg_delay)
                print("Avg_Energy    :  ", "%0.1f" %avg_energy)
                print("Avg_QoE       :  ", "%0.1f" %avg_QoE)
                print("EnergyCosumption: ----------------------------------------------------------------------")
                print("Local         :  ", "%0.1f" %meter_comp_energy, "[meter_bit_processed:", int(meter_bit_processed),"]")
                print("Trans         :  ", "%0.1f" %meter_tran_energy, "[meter_bit_transmitted:", int(meter_bit_transmitted),"]")
                print("Substations         :  ", "%0.1f" % sum(meter_idle_energy), "[substation_bit_processed :", int(sum(substation_bit_processed)),"]")
                #print("--------------------------------------------------------------------------------------------------------")
                #print("Trans_Drop: ", env.drop_trans_count, "Substation_Drop: ", env.drop_substation_count, "UE_Drop: ", env.drop_meter_count)
                #print("Drop_Count: ",Drop_Count(meter_RL_list, episode))

                break # Training Finished


if __name__ == "__main__":

    # GENERATE ENVIRONMENT
    env = SmartGrid(Config.N_METER, Config.N_SUBSTATION, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)

    # GENERATE MULTIPLE CLASSES FOR RL
    meter_RL_list = list()
    for meter in range(Config.N_METER):
        meter_RL_list.append(DuelingDoubleDeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                                    learning_rate       = Config.LEARNING_RATE,
                                                    reward_decay        = Config.REWARD_DECAY,
                                                    e_greedy            = Config.E_GREEDY,
                                                    replace_target_iter = Config.N_NETWORK_UPDATE,  
                                                    memory_size         = Config.MEMORY_SIZE,  
                                                    ))



    # LOAD Trained MODEL 
    '''
    for meter in range(Config.N_UE):
        meter_RL_list[meter].Initialize(meter_RL_list[meter].sess, meter)
        meter_RL_list[meter].epsilon = 1
    '''

    Delay  = open("Delay.txt" , 'w')
    Energy = open("Energy.txt", 'w')
    QoE    = open("QoE.txt"   , 'w')
    Drop   = open("Drop.txt"  , 'w')
                           

    # TRAIN THE SYSTEM
    train(meter_RL_list, Config.N_EPISODE)











