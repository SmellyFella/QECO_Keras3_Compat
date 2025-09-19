class Config(object):
    
    # System setup
    N_METER             = 20                   # Number of Smart meters
    N_SUBSTATION           = 2                    # Number of Fog Servers
    METER_COMP_CAP      = 3                 # Max reporting rate (kW) for meters
    METER_TRAN_CAP      = 2                   # Max communication bandwidth (kW)
    SUBSTATION_COMP_CAP    = 100                   # Substation Servers Handling Capacity (kWh)

    # Energy consumption settings
    METER_ENERGY_STATE  = [0.25, 0.50, 0.75]   # Meter Power profiles (idle, normal, peak) in kWh
    METER_COMP_ENERGY   = 0.01                    # Energy cost of sampling in kWh
    METER_TRAN_ENERGY   = 0.005                  # Cost of energy to transmit reading to a substation
    METER_IDLE_ENERGY   = 0.001                  # Energy draw whilst idle
    SUBSTATION_COMP_ENERGY = 0.5                    # Energy to process readings or route power flows

    # Task Requirement
    TASK_COMP_DENS   = [1, 3, 5]      # Task Energy Density

    TASK_MIN_SIZE    = 1
    TASK_MAX_SIZE    = 7
    N_COMPONENT      = 1                    # Number of Task Partitions
    MAX_DELAY        = 10

    # Deadlines (in timesteps) for different task types
    CRITICAL_DEADLINE = 2      # must be processed within 2 timesteps
    NONCRITICAL_DEADLINE = 6   # non-critical tasks can wait longer
    
    # Latency introduced by communication/processing
    FIXED_COMM_DELAY = 1       # time to send reading to substation from meter
    PROC_DELAY_FACTOR = 0.5    # queueing or control latency at substations/fog servers


    # Simulation scenario
    N_EPISODE        = 20                # Number of Episodes
    N_TIME_SLOT      = 100                  # Number of Time Slots
    DURATION         = 0.1                  # Time Slot Duration
    TASK_ARRIVE_PROB = 0.1                  # Task Generation Probability
    N_TIME = N_TIME_SLOT + MAX_DELAY

    SMART_METER_PERIOD = 5   # tasks generated every 5 timesteps

    # Algorithm settings
    LEARNING_RATE    = 0.01
    REWARD_DECAY     = 0.9
    E_GREEDY         = 0.99
    N_NETWORK_UPDATE = 200                  # Networks Parameter Replace
    MEMORY_SIZE      = 500                  # Replay Buffer Memory Size

