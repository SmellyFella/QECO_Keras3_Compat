import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class Random_Base:

    def __init__(self, n_actions, n_time):
      self.epsilon = 0
      self.n_actions = n_actions
      self.n_time = n_time
      self.reward_store = list()
      self.action_store = list()
      self.delay_store = list()
      self.energy_store = list()
      self.offload_store = list()

    def choose_action(self, observation):
      action = np.random.randint(0, self.n_actions)
      
      return action

    def do_store_action(self, episode, time, action):
      return
    
    def update_lstm(self, state):
      return
    
    def store_transition(self, history, a, b, c, d, e):
      return
    
    def do_store_reward(self, a, b, c):
      return

    def do_store_delay(self, episode, time, delay):
      while episode >= len(self.delay_store):
        self.delay_store.append(np.zeros([self.n_time]))
      self.delay_store[episode][time] = delay

    def do_store_energy(self, episode, time, energy, energy2, energy3, energy4):
        fog_energy = 0
        for i in range(len(energy3)):
          if energy3[i] != 0:
            fog_energy = energy3[i]


        idle_energy = 0
        for i in range(len(energy4)):
          if energy4[i] != 0:
              idle_energy = energy4[i]

        while episode >= len(self.energy_store):
          self.energy_store.append(np.zeros([self.n_time]))
        self.energy_store[episode][time] = energy + energy2 + fog_energy + idle_energy

    #Add a method for storing the list of successful task offloads
    def do_store_offload(self, episode, time, success):
      return

    #method for tracking capacity utilisation:
    def do_store_capacity_util(self, episode, time, capacity_util):
      return

    def learn(self):
      return
