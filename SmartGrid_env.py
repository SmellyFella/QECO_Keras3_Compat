from Config import Config
import numpy as np
import random
import math
import queue

class SmartGrid:
    def __init__(self, num_meter, num_substation, num_time, num_component, max_delay):
        # Initialize variables
        self.n_meter          = num_meter
        self.n_substation        = num_substation
        self.n_time        = num_time
        self.n_component   = num_component
        self.max_delay     = max_delay
        self.duration      = Config.DURATION
        self.meter_p_comp     = Config.METER_PROC_LOSS
        self.meter_p_tran     = Config.METER_LINE_LOSS
        self.meter_p_idle     = Config.METER_IDLE_ENERGY
        self.substation_p_comp   = Config.SUBSTATION_COMP_ENERGY

        self.smart_meter_period = Config.SMART_METER_PERIOD
        

        self.time_count      = 0
        self.task_count_meter   = 0
        self.task_count_substation = 0
        self.n_actions       = 1 + self.n_substation
        self.n_features      = 1 + 1 + 1 + 1 + self.n_substation
        self.n_lstm_state    = self.n_substation

        self.drop_trans_count = 0
        self.drop_substation_count = 0
        self.drop_meter_count = 0
        
        #Added successful offloads count to track how many tasks were successfully offloaded
        self.successful_offloads = 0
        #Criticality store for critical tasks
        self.task_criticality = np.zeros([self.n_time, self.n_meter], dtype=int)
        #deadline store for task deadlines
        self.task_deadlines = np.zeros([self.n_time, self.n_meter], dtype=int)
        #meter capacity utilisation store for reward function
        self.meter_capacity_util = np.zeros(self.n_meter)
        
        # Tansmission line and Substation feeder capacity
        self.line_cap_meter   = Config.METER_GEN_CAP_KW * np.ones(self.n_meter) * self.duration
        self.line_cap_substation = Config.SUBSTATION_TRANS_CAP_KW * np.ones([self.n_substation]) * self.duration
        self.feeder_cap_meter   = Config.METER_LINE_CAP_KW * np.ones([self.n_meter, self.n_substation]) * self.duration
        self.n_cycle = 1
        self.task_arrive_prob = Config.TASK_ARRIVE_PROB
        self.max_arrive_size   = Config.TASK_MAX_SIZE
        self.min_arrive_size   = Config.TASK_MIN_SIZE
        self.arrive_task_size_set    = np.arange(self.min_arrive_size, self.max_arrive_size, 0.1)
        #self.energy_state_set   = np.arange(0.25,1, 0.25) 
        self.meter_energy_state = [Config.METER_ENERGY_STATE[np.random.randint(0,len(Config.METER_ENERGY_STATE))] for ue in range(self.n_meter)]
        self.arrive_task_size   = np.zeros([self.n_time, self.n_meter])    #This has been changed to represent the amount of energy (kWh)
        self.arrive_task_dens   = np.zeros([self.n_time, self.n_meter])    #This has been changed to represent the urgency/cost

        self.n_task = int(self.n_time * self.task_arrive_prob)

        # Task delay and energy-related arrays
        self.process_delay = np.zeros([self.n_time, self.n_meter])
        self.meter_bit_processed = np.zeros([self.n_time, self.n_meter])
        self.substation_bit_processed = np.zeros([self.n_time, self.n_meter, self.n_substation])
        self.meter_bit_transmitted = np.zeros([self.n_time, self.n_meter])
        self.meter_comp_energy = np.zeros([self.n_time, self.n_meter])
        self.substation_comp_energy = np.zeros([self.n_time, self.n_meter, self.n_substation])
        self.meter_idle_energy = np.zeros([self.n_time, self.n_meter, self.n_substation])
        self.meter_tran_energy = np.zeros([self.n_time, self.n_meter])
        self.unfinish_task = np.zeros([self.n_time, self.n_meter])
        self.process_delay_trans = np.zeros([self.n_time, self.n_meter])
        self.substation_drop = np.zeros([self.n_meter, self.n_substation])

        # Queue information initialization
        self.t_meter_comp = -np.ones([self.n_meter])
        self.t_meter_tran = -np.ones([self.n_meter])
        self.b_substation_comp = np.zeros([self.n_meter, self.n_substation])

        # Queue initialization
        self.meter_computation_queue = [queue.Queue() for _ in range(self.n_meter)]
        self.meter_transmission_queue = [queue.Queue() for _ in range(self.n_meter)]
        self.substation_computation_queue = [[queue.Queue() for _ in range(self.n_substation)] for _ in range(self.n_meter)]
        self.substation_meter_m = np.zeros(self.n_substation)
        self.substation_meter_m_observe = np.zeros(self.n_substation)

        # Task indicator initialization
        self.local_process_task = [{'DIV': np.nan, 'METER_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, '': np.nan, 'REMAIN': np.nan} for _ in range(self.n_meter)]
        self.local_transmit_task = [{'DIV': np.nan, 'METER_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                     'TIME': np.nan, 'SUBSTATION': np.nan, 'REMAIN': np.nan} for _ in range(self.n_meter)]
        self.substation_process_task = [[{'DIV': np.nan, 'METER_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'REMAIN': np.nan} for _ in range(self.n_substation)] for _ in range(self.n_meter)]

        self.task_history = [[] for _ in range(self.n_meter)]

    def current_load(self):
      """
      Compute current load as total number of tasks in all queues.
      Could also be weighted by task size.
      """
      total_load = 0
      # Add meter computation queues
      for q in self.meter_computation_queue:
          total_load += q.qsize()
      # Add meter transmission queues
      for q in self.meter_transmission_queue:
          total_load += q.qsize()
      # Add substation queues
      for meter_idx in range(self.n_meter):
          for sub_idx in range(self.n_substation):
              total_load += self.substation_computation_queue[meter_idx][sub_idx].qsize()
      return total_load


    def reset(self, arrive_task_size, arrive_task_dens, task_criticality = None):
    
        self.drop_trans_count = 0
        self.drop_substation_count = 0
        self.drop_meter_count = 0
        #Added the reset for successful offloads
        self.successful_offloads = 0

        #reset the meter capacity tracker
        self.meter_capacity_util = np.zeros(self.n_meter)

        # Reset variables and queues
        self.task_history = [[] for _ in range(self.n_meter)]
        self.METER_TASK = [-1] * self.n_meter
        self.drop_substation_count = 0

        self.arrive_task_size = arrive_task_size
        self.arrive_task_dens = arrive_task_dens

        self.time_count = 0

        self.local_process_task = []
        self.local_transmit_task = []
        self.substation_process_task = []



        self.meter_computation_queue = [queue.Queue() for _ in range(self.n_meter)]
        self.meter_transmission_queue = [queue.Queue() for _ in range(self.n_meter)]
        self.substation_computation_queue = [[queue.Queue() for _ in range(self.n_substation)] for _ in range(self.n_meter)]
        
        self.t_meter_comp = -np.ones([self.n_meter])
        self.t_meter_tran = -np.ones([self.n_meter])
        self.b_substation_comp = np.zeros([self.n_meter, self.n_substation])

        self.process_delay = np.zeros([self.n_time, self.n_meter])
        self.meter_bit_processed = np.zeros([self.n_time, self.n_meter])
        self.substation_bit_processed = np.zeros([self.n_time, self.n_meter, self.n_substation])
        self.meter_bit_transmitted = np.zeros([self.n_time, self.n_meter])
        self.meter_comp_energy = np.zeros([self.n_time, self.n_meter])
        self.substation_comp_energy = np.zeros([self.n_time, self.n_meter, self.n_substation])
        self.meter_idle_energy = np.zeros([self.n_time, self.n_meter, self.n_substation])
        self.meter_tran_energy = np.zeros([self.n_time, self.n_meter])
        self.unfinish_task = np.zeros([self.n_time, self.n_meter])
        self.process_delay_trans = np.zeros([self.n_time, self.n_meter])
        self.substation_drop = np.zeros([self.n_meter, self.n_substation])

        self.local_process_task = [{'DIV': np.nan, 'METER_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'SUBSTATION': np.nan, 'REMAIN': np.nan} for _ in range(self.n_meter)]
        self.local_transmit_task = [{'DIV': np.nan, 'METER_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                     'TIME': np.nan, 'SUBSTATION': np.nan, 'REMAIN': np.nan} for _ in range(self.n_meter)]
        self.substation_process_task = [[{'DIV': np.nan, 'METER_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'REMAIN': np.nan} for _ in range(self.n_substation)] for _ in range(self.n_meter)]

        # Initial observation and LSTM state
        Meters_OBS = np.zeros([self.n_meter, self.n_features])
        for meter_index in range(self.n_meter):
            if self.arrive_task_size[self.time_count, meter_index] != 0:
                
                Meters_OBS[meter_index, :] = np.hstack([
                    self.arrive_task_size[self.time_count, meter_index], self.t_meter_comp[meter_index],
                    self.t_meter_tran[meter_index],
                    np.squeeze(self.b_substation_comp[meter_index, :]),
                    self.meter_energy_state[meter_index]])

        Meters_lstm_state = np.zeros([self.n_meter, self.n_lstm_state])

        # Store criticality for later use
        if task_criticality is not None:
            self.task_criticality = task_criticality
        else:
            # Default: assume all tasks are non-critical if not provided
            self.task_criticality = np.zeros_like(bitarrive_size, dtype=int)

        return Meters_OBS, Meters_lstm_state

   
    # perform action, observe state and delay (several steps later)
    def step(self, action):
    

        meter_action_local = np.zeros([self.n_meter], np.int32)
        meter_action_offload = np.zeros([self.n_meter], np.int32)

        for meter_index in range(self.n_meter):
            meter_action = action[meter_index]
            meter_action_offload[meter_index] = int(meter_action - 1)
            if meter_action == 0:
                meter_action_local[meter_index] = 1

        #Capacity utilisation code
        for meter_idx in range(self.n_meter):
            # actual work processed (energy or data)
            processed = self.meter_bit_processed[self.time_count-1, meter_idx]
            
            # maximum meter capacity (already in Config)
            max_cap = Config.METER_GEN_CAP_KW * self.duration
            
            if max_cap > 0:
                self.meter_capacity_util[meter_idx] = processed / max_cap
            else:
                self.meter_capacity_util[meter_idx] = 0
                
        ##criticality store
        t = self.time_count
    
        # Get criticality for tasks arriving at this timestep
        current_criticality = self.task_criticality[t, :]
        
        #meter_action_offload = np.zeros([self.n_meter], np.int32)
        #meter_action_component = np.zeros([self.n_meter], np.int32)-1
        #random_list  = []
        #for i in range(self.n_component):
            #random_list.append(i)

        # COMPUTATION QUEUE UPDATE ===================
        for meter_index in range(self.n_meter):

            meter_line_cap = np.squeeze(self.line_cap_meter[meter_index])
            meter_arrive_task_size = np.squeeze(self.arrive_task_size[self.time_count, meter_index])
            meter_arrive_task_dens = np.squeeze(self.arrive_task_dens[self.time_count, meter_index])
        
            tmp_dict = {
                'DIV' : 0 , 
                'METER_ID': meter_index,
                'TASK_ID': self.METER_TASK[meter_index],
                'SIZE': meter_arrive_task_size,
                'DENS': meter_arrive_task_dens,
                'TIME': self.time_count,
                'SUBSTATION': meter_action_offload[meter_index],
            }

            if meter_action_local[meter_index] == 1:
                self.meter_computation_queue[meter_index].put(tmp_dict)


            for cycle in range(self.n_cycle):    
                # TASK ON PROCESS
                if math.isnan(self.local_process_task[meter_index]['REMAIN']) \
                        and (not self.meter_computation_queue[meter_index].empty()):
                    while not self.meter_computation_queue[meter_index].empty():
                        get_task = self.meter_computation_queue[meter_index].get()
                        #print(get_task)
                        if get_task['SIZE'] != 0:
                            if self.time_count - get_task['TIME'] + 1 <= self.max_delay:
                                self.local_process_task[meter_index]['METER_ID']    = get_task['METER_ID']
                                self.local_process_task[meter_index]['TASK_ID']  = get_task['TASK_ID']
                                self.local_process_task[meter_index]['SIZE']     = get_task['SIZE']
                                self.local_process_task[meter_index]['DENS']     = get_task['DENS']
                                self.local_process_task[meter_index]['TIME']     = get_task['TIME']
                                self.local_process_task[meter_index]['REMAIN']   = self.local_process_task[meter_index]['SIZE']
                                self.local_process_task[meter_index]['DIV']      = get_task['DIV']

                                break
                            else:
                                #self.task_history[meter_index][get_task['TASK_ID']]['d_state'][get_task['DIV']] = -1
                                
                                self.process_delay[get_task['TIME'], meter_index] = self.max_delay
                                self.unfinish_task[get_task['TIME'], meter_index] = 1

                             
                # PROCESS
                if self.local_process_task[meter_index]['REMAIN'] > 0:

                    if self.local_process_task[meter_index]['REMAIN'] >= (meter_line_cap / self.local_process_task[meter_index]['DENS']):
    
                        self.meter_bit_processed[self.local_process_task[meter_index]['TIME'], meter_index] += meter_line_cap / self.local_process_task[meter_index]['DENS']
                        self.meter_comp_energy[self.local_process_task[meter_index]['TIME'], meter_index] += (meter_line_cap / self.local_process_task[meter_index]['DENS']) * (1 ** (-27) * (meter_line_cap / self.local_process_task[meter_index]['DENS'])) 
                    else:
                        self.meter_bit_processed[self.local_process_task[meter_index]['TIME'], meter_index] += self.local_process_task[meter_index]['REMAIN']/ self.local_process_task[meter_index]['DENS']
                        self.meter_comp_energy[self.local_process_task[meter_index]['TIME'], meter_index] += self.local_process_task[meter_index]['REMAIN']/ self.local_process_task[meter_index]['DENS'] * (1 ** (-27) * (meter_line_cap / self.local_process_task[meter_index]['DENS']))


                    self.local_process_task[meter_index]['REMAIN'] = \
                        self.local_process_task[meter_index]['REMAIN'] - meter_line_cap / self.local_process_task[meter_index]['DENS']

                    #print(self.local_process_task[meter_index]['REMAIN'])
                    #print(meter_line_cap, self.local_process_task[meter_index]['DENS'])

                    # if no remain, compute processing delay
                    if self.local_process_task[meter_index]['REMAIN'] <= 0: 
                        self.process_delay[self.local_process_task[meter_index]['TIME'], meter_index] \
                            = self.time_count - self.local_process_task[meter_index]['TIME'] + 1
                        self.local_process_task[meter_index]['REMAIN'] = np.nan
                        #print("hi")

                        #self.task_history[meter_index][self.local_process_task[meter_index]['TASK_ID']]['d_state'][self.local_process_task[meter_index]['DIV']] = 1 
                        #if sum(self.task_history[meter_index][self.local_process_task[meter_index]['TASK_ID']]['d_state']) > self.n_component-1:


                    elif self.time_count - self.local_process_task[meter_index]['TIME'] + 1 == self.max_delay:
                        #self.task_history[meter_index][self.local_process_task[meter_index]['TASK_ID']]['d_state'][self.local_process_task[meter_index]['DIV']] = -1
                        self.local_process_task[meter_index]['REMAIN'] = np.nan
                        self.process_delay[self.local_process_task[meter_index]['TIME'], meter_index] = self.max_delay
                        self.unfinish_task[self.local_process_task[meter_index]['TIME'], meter_index] = 1
                        self.drop_meter_count = self.drop_meter_count + 1

                    # OTHER INFO self.t_meter_comp[meter_index]
                    # update self.t_meter_comp[meter_index] only when meter_bitrate != 0
                if meter_arrive_task_size != 0:
                    tmp_tilde_t_meter_comp = np.max([self.t_meter_comp[meter_index] + 1, self.time_count])
                    self.t_meter_comp[meter_index] = np.min([tmp_tilde_t_meter_comp
                                                                    + math.ceil(meter_arrive_task_size * meter_action_local[meter_index]
                                                                    / (meter_line_cap / meter_arrive_task_dens)) - 1,
                                                                    self.time_count + self.max_delay - 1])

        # substation QUEUE UPDATE =========================
        for meter_index in range(self.n_meter):
            #meter_comp_density = self.comp_density

            for substation_index in range(self.n_substation):
                substation_cap = self.line_cap_substation[substation_index]/self.n_cycle
  
                for cycle in range(self.n_cycle): 
                    # TASK ON PROCESS
                    if math.isnan(self.substation_process_task[meter_index][substation_index]['REMAIN']) \
                            and (not self.substation_computation_queue[meter_index][substation_index].empty()):
                        while not self.substation_computation_queue[meter_index][substation_index].empty():
                            get_task = self.substation_computation_queue[meter_index][substation_index].get()

                                            

                            if self.time_count - get_task['TIME'] + 1 <= self.max_delay:
                                self.substation_process_task[meter_index][substation_index]['METER_ID']   = get_task['METER_ID']
                                self.substation_process_task[meter_index][substation_index]['TASK_ID'] = get_task['TASK_ID']
                                self.substation_process_task[meter_index][substation_index]['SIZE']    = get_task['SIZE']
                                self.substation_process_task[meter_index][substation_index]['DENS']    = get_task['DENS']
                                self.substation_process_task[meter_index][substation_index]['TIME']    = get_task['TIME']
                                self.substation_process_task[meter_index][substation_index]['REMAIN']  = self.substation_process_task[meter_index][substation_index]['SIZE']
                                self.substation_process_task[meter_index][substation_index]['DIV']     = get_task['DIV']
                                break
                            else:
                                
                                #self.task_history[get_task['METER_ID']][get_task['TASK_ID']]['d_state'][get_task['DIV']] = -1
                                self.process_delay[get_task['TIME'], meter_index] = self.max_delay
                                self.unfinish_task[get_task['TIME'], meter_index] = 1


                #    print(self.substation_process_task[meter_index][substation_index], "f_________")
                    # PROCESS
                    self.substation_drop[meter_index, substation_index] = 0

                    if self.substation_process_task[meter_index][substation_index]['REMAIN'] > 0:
    
                        if self.substation_process_task[meter_index][substation_index]['REMAIN'] >= (substation_cap / self.substation_process_task[meter_index][substation_index]['DENS'] / self.substation_meter_m[substation_index]):
                            self.substation_comp_energy[self.substation_process_task[meter_index][substation_index]['TIME'], meter_index, substation_index] += (substation_cap/ self.substation_process_task[meter_index][substation_index]['DENS']) * (self.substation_p_comp * self.duration)
                            self.substation_bit_processed[self.substation_process_task[meter_index][substation_index]['TIME'], meter_index, substation_index] += (substation_cap/ self.substation_process_task[meter_index][substation_index]['DENS'] / self.substation_meter_m[substation_index])                      
                            self.meter_idle_energy[self.substation_process_task[meter_index][substation_index]['TIME'], meter_index, substation_index] += (substation_cap / self.substation_process_task[meter_index][substation_index]['DENS'] / self.substation_meter_m[substation_index]) 
                        else:
                            self.substation_bit_processed[self.substation_process_task[meter_index][substation_index]['TIME'], meter_index, substation_index] += self.substation_process_task[meter_index][substation_index]['REMAIN'] / self.substation_meter_m[substation_index]
                            self.substation_comp_energy[self.substation_process_task[meter_index][substation_index]['TIME'], meter_index, substation_index] += (self.substation_process_task[meter_index][substation_index]['REMAIN']) * (self.substation_p_comp * self.duration)
                            self.meter_idle_energy[self.substation_process_task[meter_index][substation_index]['TIME'], meter_index, substation_index] += (self.substation_process_task[meter_index][substation_index]['REMAIN'] / self.substation_meter_m[substation_index]) * self.meter_p_idle  

                        self.substation_process_task[meter_index][substation_index]['REMAIN'] = self.substation_process_task[meter_index][substation_index]['REMAIN'] - substation_cap/ self.substation_process_task[meter_index][substation_index]['DENS'] / self.substation_meter_m[substation_index]
                        
                        

                        if self.substation_process_task[meter_index][substation_index]['REMAIN'] <= 0:
                            self.process_delay[self.substation_process_task[meter_index][substation_index]['TIME'],meter_index] \
                                = self.time_count - self.substation_process_task[meter_index][substation_index]['TIME'] + 1

                            #Added increment to count successful task offloads
                            self.successful_offloads += 1
                            
                            #self.task_history[self.substation_process_task[meter_index][substation_index]['METER_ID']][self.substation_process_task[meter_index][substation_index]['TASK_ID']]['d_state'][self.substation_process_task[meter_index][substation_index]['DIV']] = 1
                            self.substation_process_task[meter_index][substation_index]['REMAIN'] = np.nan
                            '''
                            if sum(self.task_history[meter_index][self.substation_process_task[meter_index][substation_index]['TASK_ID']]['d_state']) > self.n_component-1:
                                self.process_delay[self.substation_process_task[meter_index][substation_index]['TIME'],meter_index] \
                                    = self.time_count - self.substation_process_task[meter_index][substation_index]['TIME'] + 1
                            '''


                        elif self.time_count - self.substation_process_task[meter_index][substation_index]['TIME'] + 1 == self.max_delay:
                            #self.task_history[self.substation_process_task[meter_index][substation_index]['METER_ID']][self.substation_process_task[meter_index][substation_index]['TASK_ID']]['d_state'][self.substation_process_task[meter_index][substation_index]['DIV']] = -1
                            self.substation_drop[meter_index, substation_index] = self.substation_process_task[meter_index][substation_index]['REMAIN']
                            self.process_delay[self.substation_process_task[meter_index][substation_index]['TIME'], meter_index] = self.max_delay
                            self.unfinish_task[self.substation_process_task[meter_index][substation_index]['TIME'], meter_index] = 1
                            self.substation_process_task[meter_index][substation_index]['REMAIN'] = np.nan
                            self.drop_substation_count = self.drop_substation_count + 1


                        #self.TASK_log[meter_index][self.substation_process_task[meter_index][substation_index]['TASK_ID']]['state'] = 2

                # OTHER INFO
                    if self.substation_meter_m[substation_index] != 0:
                        self.b_substation_comp[meter_index, substation_index] \
                            = np.max([self.b_substation_comp[meter_index, substation_index]
                                        - self.line_cap_substation[substation_index]/ meter_arrive_task_dens / self.substation_meter_m[substation_index]
                                        - self.substation_drop[meter_index, substation_index], 0])

        # TRANSMISSION QUEUE UPDATE ===================
        for meter_index in range(self.n_meter):
            #meter_feeder_cap = np.squeeze(self.feeder_cap_meter[meter_index,:])[1]/self.n_cycle

            meter_feeder_cap = np.squeeze(self.feeder_cap_meter[meter_index,:])
            meter_arrive_task_size = np.squeeze(self.arrive_task_size[self.time_count, meter_index])
            meter_arrive_task_dens = np.squeeze(self.arrive_task_dens[self.time_count, meter_index])
        
            tmp_dict = {
                'DIV' : 0 , 
                'METER_ID': meter_index,
                'TASK_ID': self.METER_TASK[meter_index],
                'SIZE': meter_arrive_task_size,
                'DENS': meter_arrive_task_dens,
                'TIME': self.time_count,
                'SUBSTATION': meter_action_offload[meter_index],
            }


            if meter_action_local[meter_index] == 0:
                self.meter_transmission_queue[meter_index].put(tmp_dict)

          


            
            for cycle in range(self.n_cycle):

                # TASK ON PROCESS
                if math.isnan(self.local_transmit_task[meter_index]['REMAIN']) \
                        and (not self.meter_transmission_queue[meter_index].empty()):
                    while not self.meter_transmission_queue[meter_index].empty():
                        get_task = self.meter_transmission_queue[meter_index].get()
                        #print("trans", get_task)
                        if get_task['SIZE'] != 0:
                            
                            #self.TASK_log[meter_index][get_task['TASK_ID']] = get_task
                            #self.task_history[meter_index].append(get_task)

                            if self.time_count - get_task['TIME'] + 1 <= self.max_delay:
                                self.local_transmit_task[meter_index]['METER_ID'] = get_task['METER_ID']
                                self.local_transmit_task[meter_index]['TASK_ID'] = get_task['TASK_ID']
                                self.local_transmit_task[meter_index]['SIZE'] = get_task['SIZE']
                                self.local_transmit_task[meter_index]['DENS'] = get_task['DENS']
                                self.local_transmit_task[meter_index]['TIME'] = get_task['TIME']
                                self.local_transmit_task[meter_index]['SUBSTATION'] = int(get_task['SUBSTATION'])
                                self.local_transmit_task[meter_index]['REMAIN'] = self.local_transmit_task[meter_index]['SIZE']
                                self.local_transmit_task[meter_index]['DIV'] = get_task['DIV']
                                break
                            else:
                                #self.task_history[get_task['METER_ID']][get_task['TASK_ID']]['d_state'][get_task['DIV']] = -1
                                self.process_delay[get_task['TIME'], meter_index] = self.max_delay
                                self.unfinish_task[get_task['TIME'], meter_index] = 1


                # PROCESS
                if self.local_transmit_task[meter_index]['REMAIN'] > 0:

                    if self.local_transmit_task[meter_index]['REMAIN'] >= meter_feeder_cap[self.local_transmit_task[meter_index]['SUBSTATION']]:
                        self.meter_tran_energy[self.local_transmit_task[meter_index]['TIME'], meter_index] += meter_feeder_cap[self.local_transmit_task[meter_index]['SUBSTATION']] * self.meter_p_tran
                        self.meter_bit_transmitted[self.local_transmit_task[meter_index]['TIME'], meter_index] += self.local_transmit_task[meter_index]['REMAIN'] 
                    
                    else:
                        self.meter_tran_energy[self.local_transmit_task[meter_index]['TIME'], meter_index] += meter_feeder_cap[self.local_transmit_task[meter_index]['SUBSTATION']] * self.meter_p_tran
                        self.meter_bit_transmitted[self.local_transmit_task[meter_index]['TIME'], meter_index] += self.local_transmit_task[meter_index]['REMAIN'] 

                    self.local_transmit_task[meter_index]['REMAIN'] = \
                        self.local_transmit_task[meter_index]['REMAIN'] \
                        - meter_feeder_cap[self.local_transmit_task[meter_index]['SUBSTATION']]

                    #print(meter_feeder_cap)

                    # UPDATE substation QUEUE
                    if self.local_transmit_task[meter_index]['REMAIN'] <= 0:
                        tmp_dict = {'METER_ID': self.local_transmit_task[meter_index]['METER_ID'],
                                    'TASK_ID': self.local_transmit_task[meter_index]['TASK_ID'],
                                    'SIZE' : self.local_transmit_task[meter_index]['SIZE'],
                                    'DENS' : self.local_transmit_task[meter_index]['DENS'],
                                    'TIME' : self.local_transmit_task[meter_index]['TIME'],
                                    'SUBSTATION'  : self.local_transmit_task[meter_index]['SUBSTATION'],
                                    'DIV'  : self.local_transmit_task[meter_index]['DIV']}

        
                        self.substation_computation_queue[meter_index][self.local_transmit_task[meter_index]['SUBSTATION']].put(tmp_dict)
                        #print("_+_+____", self.local_transmit_task[meter_index]['SUBSTATION'])
                        self.task_count_substation = self.task_count_substation + 1

                        substation_index = self.local_transmit_task[meter_index]['SUBSTATION']
                        self.b_substation_comp[meter_index, substation_index] = self.b_substation_comp[meter_index, substation_index] + self.local_transmit_task[meter_index]['SIZE']
                        
                        self.process_delay_trans[self.local_transmit_task[meter_index]['TIME'], meter_index] = self.time_count - self.local_transmit_task[meter_index]['TIME'] + 1
                        self.local_transmit_task[meter_index]['REMAIN'] = np.nan


                    elif self.time_count - self.local_transmit_task[meter_index]['TIME'] + 1 == self.max_delay:
                        #self.task_history[self.local_transmit_task[meter_index]['METER_ID']][self.local_transmit_task[meter_index]['TASK_ID']]['d_state'][self.local_transmit_task[meter_index]['DIV']] = -1
                        self.local_transmit_task[meter_index]['REMAIN'] = np.nan
                        self.process_delay[self.local_transmit_task[meter_index]['TIME'], meter_index] = self.max_delay
                        self.unfinish_task[self.local_transmit_task[meter_index]['TIME'], meter_index] = 1
                        self.drop_trans_count = self.drop_trans_count + 1

                    # OTHER INFO
                if meter_arrive_task_size != 0:
                    tmp_tilde_t_meter_tran = np.max([self.t_meter_tran[meter_index] + 1, self.time_count])
                    self.t_meter_comp[meter_index] = np.min([tmp_tilde_t_meter_tran
                                                            + math.ceil(meter_arrive_task_size * (1 - meter_action_local[meter_index])
                                                            / meter_feeder_cap[meter_action_offload[meter_index]]) - 1,
                                                            self.time_count + self.max_delay - 1])
            
            # Calculate communication delay
        comm_delay = Config.FIXED_COMM_DELAY
    
        # Estimate processing delay based on current load
        proc_delay = Config.PROC_DELAY_FACTOR * self.current_load()
    
        total_delay = comm_delay + proc_delay
    
        # Check deadlines for each task
        for meter_idx in range(self.n_meter):
          for t_idx in range(self.n_time):
              task_size = self.arrive_task_size[t_idx, meter_idx]
              if task_size > 0:
                  deadline = self.task_deadlines[t_idx, meter_idx]
                  total_delay = comm_delay + proc_delay  # as before
                  if total_delay > deadline:
                      self.unfinish_task[t_idx, meter_idx] = 1
                      # Optionally count failed tasks
                      if not hasattr(self, "failed_tasks"):
                          self.failed_tasks = 0
                      self.failed_tasks += 1


        # COMPUTE CONGESTION (FOR NEXT TIME SLOT)
        self.substation_meter_m_observe = self.substation_meter_m
        self.substation_meter_m = np.zeros(self.n_substation)
        for substation_index in range(self.n_substation):
            for meter_index in range(self.n_meter):
                if (not self.substation_computation_queue[meter_index][substation_index].empty()) \
                        or self.substation_process_task[meter_index][substation_index]['REMAIN'] > 0:
                    self.substation_meter_m[substation_index] += 1


        # TIME UPDATE
        self.time_count = self.time_count + 1
        done = False
        if self.time_count >= self.n_time:
            done = True
            # set all the tasks' processing delay and unfinished indicator
            
            for time_index in range(self.n_time):
                for meter_index in range(self.n_meter):
                    if self.process_delay[time_index, meter_index] == 0 and self.arrive_task_size[time_index, meter_index] != 0:
                        self.process_delay[time_index, meter_index] = (self.time_count - 1) - time_index + 1
                        self.unfinish_task[time_index, meter_index] = 1


        # OBSERVATION
        Meters_OBS_ = np.zeros([self.n_meter, self.n_features])
        Meters_lstm_state_ = np.zeros([self.n_meter, self.n_lstm_state])
        if not done:
            for meter_index in range(self.n_meter):
                # observation is zero if there is no task arrival
                if self.arrive_task_size[self.time_count, meter_index] != 0:
                    # state [A, B^{comp}, B^{tran}, [B^{substation}]]
                    Meters_OBS_[meter_index, :] = np.hstack([
                        self.arrive_task_size[self.time_count, meter_index],         #Size of the current task
                        self.t_meter_comp[meter_index] - self.time_count + 1,        #Meters computation time 
                        self.t_meter_tran[meter_index] - self.time_count + 1,        #Meters transmission time
                        self.b_substation_comp[meter_index, :],                      #Substation computational load/capacities
                        self.meter_energy_state[meter_index]])                       #idle/normal/peak level for meters

                Meters_lstm_state_[meter_index, :] = np.hstack(self.substation_meter_m_observe)

        return Meters_OBS_, Meters_lstm_state_, done








