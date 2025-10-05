class Config(object):
    
    # System setup
    N_METER             = 20                   # Number of Smart meters
    N_SUBSTATION           = 2                    # Number of Fog Servers
    METER_GEN_CAP_KW      = 5                 # Max reporting rate (kW) for meters
    METER_LINE_CAP_KW      = 20                   # Max communication bandwidth (kW)
    SUBSTATION_TRANS_CAP_KW    = 500                   # Substation Servers Handling Capacity (kWh)

    # Energy consumption settings
    METER_ENERGY_STATE  = [0.25, 0.50, 0.75]   # Meter Power profiles (idle, normal, peak) in kWh
    METER_PROC_LOSS   = 0.01                    # Energy cost of sampling in kWh
    METER_LINE_LOSS   = 0.03                  # Cost of energy to transmit reading to a substation
    METER_IDLE_ENERGY   = 0.001                  # Energy draw whilst idle
    SUBSTATION_COMP_ENERGY = 0.5                    # Energy to process readings or route power flows

    # Task Requirement
    TASK_URGENCY_FACTOR  = [1, 3, 5]      # Task Energy Density

    TASK_MIN_SIZE    = 2
    TASK_MAX_SIZE    = 25
    N_COMPONENT      = 1                    # Number of Task Partitions
    MAX_DELAY        = 10

    # Deadlines (in timesteps) for different task types
    CRITICAL_DEADLINE = 1      # must be processed within 2 timesteps
    NONCRITICAL_DEADLINE = 5   # non-critical tasks can wait longer
    
    # Latency introduced by communication/processing
    FIXED_COMM_DELAY = 0.15       # time to send reading to substation from meter
    PROC_DELAY_FACTOR = 0.1    # queueing or control latency at substations/fog servers


    # Simulation scenario
    N_EPISODE        = 500                # Number of Episodes
    N_TIME_SLOT      = 100                  # Number of Time Slots
    DURATION         = 0.1                  # Time Slot Duration
    TASK_ARRIVE_PROB = 0.2                  # Task Generation Probability
    N_TIME = N_TIME_SLOT + MAX_DELAY

    SMART_METER_PERIOD = 5   # tasks generated every 5 timesteps

    """
    # Algorithm settings
    LEARNING_RATE    = 0.01
    REWARD_DECAY     = 0.9
    E_GREEDY         = 0.99
    N_NETWORK_UPDATE = 200                  # Networks Parameter Replace
    MEMORY_SIZE      = 500                  # Replay Buffer Memory Size


    """ 
    #New
    LEARNING_RATE    = 0.001
    REWARD_DECAY     = 0.95
    E_GREEDY         = 0.9
    N_NETWORK_UPDATE = 200                  # Networks Parameter Replace
    MEMORY_SIZE      = 5000                  # Replay Buffer Memory Size
    
