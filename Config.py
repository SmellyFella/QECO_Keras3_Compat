class Config(object):
    
    # System setup
    N_METER             = 20                   # Number of Mobile Devices
    N_SUBSTATION           = 2                    # Number of Edge Servers
    METER_COMP_CAP      = 2.6                  # Mobile Device Computation Capacity
    METER_TRAN_CAP      = 14                   # Mobile Device Transmission Capacity
    SUBSTATION_COMP_CAP    = 42                   # Edge Servers Computation Capacity

    # Energy consumption settings
    METER_ENERGY_STATE  = [0.25, 0.50, 0.75]   # Ultra-power-saving mode, Power-saving mode, Performance mode
    METER_COMP_ENERGY   = 2                    # Computation Power of Mobile Device
    METER_TRAN_ENERGY   = 2.3                  # Transmission Power of Mobile Device
    METER_IDLE_ENERGY   = 0.1                  # Standby power of Mobile Device
    SUBSTATION_COMP_ENERGY = 5                    # Computation Power of Edge Server

    # Task Requirement
    TASK_COMP_DENS   = [0.197, 0.297, 0.397]      # Task Computation Density
    
    #TASK_COMP_DENS   = 0.297

    # Deadlines (in timesteps) for different task types
    CRITICAL_DEADLINE = 2      # must be processed within 2 timesteps
    NONCRITICAL_DEADLINE = 6   # non-critical tasks can wait longer
    
    # Latency introduced by communication/processing
    FIXED_COMM_DELAY = 1       # time to send reading to substation
    PROC_DELAY_FACTOR = 0.5    # scales with load (e.g., number of tasks in queue)

    TASK_MIN_SIZE    = 1
    TASK_MAX_SIZE    = 7
    N_COMPONENT      = 1                    # Number of Task Partitions
    MAX_DELAY        = 10


    # Simulation scenario
    N_EPISODE        = 500                # Number of Episodes
    N_TIME_SLOT      = 100                  # Number of Time Slots
    DURATION         = 0.1                  # Time Slot Duration
    TASK_ARRIVE_PROB = 0.3                  # Task Generation Probability
    N_TIME = N_TIME_SLOT + MAX_DELAY

    SMART_METER_PERIOD = 5   # tasks generated every 5 timesteps

    # Algorithm settings
    LEARNING_RATE    = 0.01
    REWARD_DECAY     = 0.9
    E_GREEDY         = 0.99
    N_NETWORK_UPDATE = 200                  # Networks Parameter Replace
    MEMORY_SIZE      = 500                  # Replay Buffer Memory Size

