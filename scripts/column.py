from brian2 import *
from scripts.neuron_groups import *
from scripts.synapses import *
from scripts.memory_networks import *
from scripts.utils import *

# Represents a cortical column which is meant to be tuned to a particular frequency
# If the memory network is defined, it represents the top-most level of the hierarchy
# connection_parameters: a dictionary to define connectivity and weights between neuron groups, useful for simulation
class CorticalColumn():
    def __init__(self, N=40, connection_parameters=None, name='Column'):
        self.N = N
        self.connection_parameters = connection_parameters
        if self.connection_parameters is None:
            raise ValueError("Column parameters not defined!")

        # Neuron groups within the column
        self.error_neurons = AdExNeuronGroup(N=N, neuron_type='regular') # Pyramidal neurons
        self.prediction_neurons = AdExNeuronGroup(N=N, neuron_type='regular') # Pyramidal neurons
        self.interneurons = AdExNeuronGroup(N=N//2, neuron_type='fast_spiking') # Parvalbumin interneurons

        
        # Defining connections between neuron groups
        # Error --> Prediction
        self.syn_err_pred = ConfigurableSynapses(self.error_neurons.neurons,
                                                 self.prediction_neurons.neurons,
                                                 synapse_parameters=self.connection_parameters['err_pred'])

        
        # Prediction --> Inhibitory
        self.syn_pred_inh = ConfigurableSynapses(self.prediction_neurons.neurons,
                                                 self.interneurons.neurons,
                                                 synapse_parameters=self.connection_parameters['pred_inh'])

        
        # Inhibitory --> Error
        self.syn_inh_err = ConfigurableSynapses(self.interneurons.neurons,
                                                self.error_neurons.neurons,
                                                is_inh=True,
                                                synapse_parameters=self.connection_parameters['inh_err'])

        
        # Error --> Inhibitory
        self.syn_err_inh = ConfigurableSynapses(self.error_neurons.neurons,
                                                self.interneurons.neurons,
                                                synapse_parameters=self.connection_parameters['err_inh'])
        
        # Monitors for this column
        self.monitors = {
            'error': make_monitors(self.error_neurons),
            'prediction': make_monitors(self.prediction_neurons),
            'interneurons': make_monitors(self.interneurons)
        }

    
    def err_recurrent(self, connection_parameters=None, plasticity_parameters=None):
        self.recurrent_parameters = connection_parameters
        self.recurrent_plasticity_parameters = plasticity_parameters
        if self.recurrent_parameters is None:
            raise ValueError("Recurrent connection parameters not defined!")
        self.syn_err_err = ConfigurableSynapses(self.error_neurons.neurons,
                                                self.error_neurons.neurons,
                                                synapse_parameters=self.recurrent_parameters,
                                                namespace=self.recurrent_plasticity_parameters)
        if self.recurrent_parameters['model'] == 'STSD':
            self.syn_err_err.n = 1.0
            self.monitors['err_stsd_term'] = StateMonitor(self.syn_err_err, 'n', record=True)

    
    # Connect lateral column via inhibitory interneurons
    def connect_lateral_column(self, lateral_column, connection_parameters=None, plasticity_parameters=None):
        self.lateral_connection_parameters = connection_parameters
        self.lateral_plasticity_parameters = plasticity_parameters
        if self.lateral_connection_parameters is None:
            raise ValueError("Lateral connection parameters not defined!")
        self.syn_err_inh_lat = ConfigurableSynapses(self.error_neurons.neurons,
                                                    lateral_column.interneurons.neurons,
                                                    synapse_parameters=self.lateral_connection_parameters,
                                                    namespace=self.lateral_plasticity_parameters)

        
    # Connect higher level (not memory network)
    def connect_higher_level(self, higher_column, connection_parameters=None, plasticity_parameters=None):
        self.err_higher_connection_parameters = connection_parameters['err_higher']
        self.err_higher_plasticity_parameters = plasticity_parameters
        if self.err_higher_connection_parameters is None:
            raise ValueError("Higher level connection parameters not defined!")
        self.syn_err_higher = ConfigurableSynapses(self.error_neurons.neurons,
                                                   higher_column.error_neurons.neurons,
                                                   synapse_parameters=self.err_higher_connection_parameters,
                                                   namespace=self.err_higher_plasticity_parameters)
        if self.err_higher_connection_parameters['model'] == 'STSD':
            self.syn_err_higher.n = 1.0

        self.pred_lower_connection_parameters = connection_parameters['pred_lower']
        self.pred_lower_plasticity_parameters = plasticity_parameters
        self.syn_pred_lower = ConfigurableSynapses(higher_column.prediction_neurons.neurons,
                                                   self.prediction_neurons.neurons,
                                                   synapse_parameters=self.pred_lower_connection_parameters,
                                                   namespace=self.pred_lower_plasticity_parameters)
        if self.pred_lower_connection_parameters['model'] == 'STSD':
            self.syn_err_higher.n = 1.0


    # Connect thalamic input, this makes it the bottom-most level
    def connect_thalamic_neurons(self, thalamic_neurons, connection_parameters=None, plasticity_parameters=None):
        self.thalamic_connection_parameters = connection_parameters
        self.thalamic_plasticity_parameters = plasticity_parameters
        if self.thalamic_connection_parameters is None:
            raise ValueError("Thalamus connection parameters not defined!")
        if self.thalamic_plasticity_parameters is None:
            raise ValueError("Thalamic STSD parameters not defined!")
        self.syn_thal_err = ConfigurableSynapses(thalamic_neurons.neurons,
                                                 self.error_neurons.neurons,
                                                 synapse_parameters=self.thalamic_connection_parameters,
                                                 namespace=self.thalamic_plasticity_parameters)
        if self.thalamic_connection_parameters['model'] == 'STSD':
            self.syn_thal_err.n = 1.0
            self.monitors['thalamic_stsd_term'] = StateMonitor(self.syn_thal_err, 'n', record=True)

        self.monitors['thalamic'] = make_monitors(thalamic_neurons)

    
    # Connect Error to lower column (This is excitatory feedback as opposed to inhibitory prediction)
    def connect_error_to_lower(self, lower_column, connection_parameters=None, plasticity_parameters=None):
        self.err_lower_connection_parameters = connection_parameters
        self.err_lower_plasticity_parameters = plasticity_parameters
        if self.err_lower_connection_parameters is None:
            raise ValueError("Error feedback connection parameters not defined!")
        if self.err_lower_plasticity_parameters is None:
            raise ValueError("Feedback from error units STSD parameters not defined!")
        self.syn_err_lower = ConfigurableSynapses(self.error_neurons.neurons,
                                                  lower_column.error_neurons.neurons,
                                                  synapse_parameters=self.err_lower_connection_parameters,
                                                  namespace=self.err_lower_plasticity_parameters)
        if self.err_lower_connection_parameters['model'] == 'STSD':
            self.syn_err_lower.n = 1.0


    # Connect to memory network at the top of the columns hierarchy
    def connect_memory_self(self, memory_network, connection_parameters=None, plasticity_parameters=None, store_weights=True):
        self.pred_mem_connection_parameters = connection_parameters['pred_mem']
        self.pred_mem_plasticity_parameters = plasticity_parameters['pred_mem']
        self.mem_pred_connection_parameters = connection_parameters['mem_pred']
        self.mem_pred_plasticity_parameters = plasticity_parameters['mem_pred']
        if self.pred_mem_connection_parameters is None:
            raise ValueError("Self-Memory connection parameters not defined!")
        if self.mem_pred_plasticity_parameters is None:
            raise ValueError("Self-Memory STDP parameters not defined!")
        self.syn_pred_mem = ConfigurableSynapses(self.prediction_neurons.neurons,
                                                 memory_network.delay_line.neurons,
                                                 synapse_parameters=self.pred_mem_connection_parameters,
                                                 to_delay_line=True,
                                                 namespace=self.pred_mem_plasticity_parameters)

        self.syn_mem_pred = ConfigurableSynapses(memory_network.delay_line.neurons,
                                                 self.prediction_neurons.neurons,
                                                 synapse_parameters=self.mem_pred_connection_parameters,
                                                 namespace=self.mem_pred_plasticity_parameters)
        self.syn_mem_pred.n = 1.0
        
        if store_weights:
            self.self_weight_snapshots = []


    # Connect to memory network at the top of the lateral columns hierarchy
    def connect_memory_lateral(self, memory_network, connection_parameters=None, plasticity_parameters=None, store_weights=True):
        self.pred_mem_connection_parameters = connection_parameters['pred_mem']
        self.pred_mem_plasticity_parameters = plasticity_parameters['pred_mem']
        self.mem_pred_connection_parameters = connection_parameters['mem_pred']
        self.mem_pred_plasticity_parameters = plasticity_parameters['mem_pred']
        if self.pred_mem_connection_parameters is None:
            raise ValueError("Self-Memory connection parameters not defined!")
        if self.mem_pred_plasticity_parameters is None:
            raise ValueError("Self-Memory STDP parameters not defined!")
        self.syn_pred_mem_lat = ConfigurableSynapses(self.prediction_neurons.neurons,
                                                 memory_network.delay_line.neurons,
                                                 synapse_parameters=self.pred_mem_connection_parameters,
                                                 to_delay_line=True,
                                                 namespace=self.pred_mem_plasticity_parameters)

        self.syn_mem_pred_lat = ConfigurableSynapses(memory_network.delay_line.neurons,
                                                 self.prediction_neurons.neurons,
                                                 synapse_parameters=self.mem_pred_connection_parameters,
                                                 namespace=self.mem_pred_plasticity_parameters)
        self.syn_mem_pred_lat.n = 1.0

        if store_weights:
            self.lat_weight_snapshots = []

    def get_monitors(self):
        return self.monitors