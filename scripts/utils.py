from brian2 import *

def make_monitors(group):
    return {
        'spikes': SpikeMonitor(group.neurons),
        'rate': PopulationRateMonitor(group.neurons),
        'I_AMPA': StateMonitor(group.neurons, 'I_AMPA', record=True),
        'I_GABA': StateMonitor(group.neurons, 'I_GABA', record=True)
    }