from netpyne import specs, sim

netParams = specs.NetParams()
simConfig = specs.SimConfig()
cell = netParams.importCellParams('PT', 'Na12HMMModel_TF.py', 'Na12Model_TF' )

#Save Cell Params???

netParams.stimSourceParams['NetStim1'] = {'type': 'NetStim', 'rate': 10, 'noise': 0.5}
netParams.stimTargetParams['NetStim1->PT'] = {'source': 'NetStim1', 'conds': {'cellType': 'PT'},
                                              'weight': 0.01, 'delay': 5,'synMech': 'exc'}

simConfig.duration = 1000
simConfig.dt = 0.025
simConfig.recordTraces = {'V_soma':{'sec': 'soma', 'loc': 0.5, 'var': 'v'}}


simConfig.analysis['plotTraces'] = {'include': [0], 'saveFig': True}

sim.initialize(                     # create network object and set cfg and net params
    simConfig = simConfig,          # pass simulation config and network params as arguments
    netParams = netParams)
sim.net.addStims()                  # add stimulation
sim.setupRecording()                # setup variables to record for each cell (spikes, V traces, etc)
sim.runSim()                        # run parallel Neuron simulation
sim.gatherData()                    # gather spiking data and cell info from each node
sim.saveData()                      # save params, cell info and sim output to file (pickle,mat,txt,etc)
sim.analysis.plotData()