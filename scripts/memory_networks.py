from brian2 import *
from scripts.neuron_groups import *
from scripts.synapses import *
from scripts.utils import *

# Delay Line Network, meant to be at the top of the hierarchy
# Linearly connected Ad-Ex neurons
class DelayLine():
    def __init__(self, N, connection_parameters, name='Delay line'):
        self.connection_parameters = connection_parameters
        
        self.delay_line = AdExNeuronGroup(N=N)
        self.delay_line_inh = AdExNeuronGroup(N=N//4)
        self.syn_delay_line = ConfigurableSynapses(self.delay_line.neurons,
                                                   self.delay_line.neurons,
                                                   synapse_parameters=self.connection_parameters,
                                                   is_delay_line=True)
        self.syn_inh = Synapses(self.delay_line.neurons, self.delay_line_inh.neurons, model='wt : 1', on_pre='g_AMPA += wt*nS', method='euler')
        self.syn_inh.connect(condition='i // 4 == j')
        self.syn_inh.wt[:] = 2

        self.syn_inh_ex = Synapses(self.delay_line_inh.neurons, self.delay_line.neurons, model='wt : 1', on_pre='g_GABA += wt*nS', method='euler')
        self.syn_inh_ex.connect(condition='abs(i - j // 4) <= 2')
        self.syn_inh_ex.wt[:] = 4

        self.monitors = make_monitors(self.delay_line)