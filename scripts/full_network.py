from brian2 import *
import numpy as np

# The complete network (thalamic input, column, memory network and defined inputs) that can be run by itself, used for simulation
class FullNetwork():
    def __init__(self,
                 thalamic_neurons,
                 columns_L1, columns_L2, columns_L3,
                 memory_networks,
                 impulse_times=np.arange(500, 10500, 500)*ms, deviant_times=np.arange(500, 10500, 500)*ms,
                 stimulus_time=50*ms, store_weights=True):
        self.thalamic_neurons = thalamic_neurons
        self.columns_L1 = columns_L1
        self.columns_L2 = columns_L2
        self.columns_L3 = columns_L3
        self.memory_networks = memory_networks
        self.neurons = []
        self.monitors = []
        self.impulse_times = impulse_times
        self.deviant_times = deviant_times

        for thalamic_neuron in self.thalamic_neurons:
            self.neurons.append(thalamic_neuron.neurons)

        # Collect all neurons and monitors for all columns
        for column in self.columns_L1:
            self.neurons.append(column.error_neurons.neurons)
            self.neurons.append(column.prediction_neurons.neurons)
            self.neurons.append(column.interneurons.neurons)
            self.neurons.append(column.syn_thal_err)
            self.neurons.append(column.syn_err_pred)
            self.neurons.append(column.syn_pred_inh)
            self.neurons.append(column.syn_inh_err)
            self.neurons.append(column.syn_err_inh)
            self.neurons.append(column.syn_err_higher)
            self.neurons.append(column.syn_pred_lower)
            self.neurons.append(column.syn_err_inh_lat)
            self.neurons.append(column.syn_err_err)

            self.monitors.extend(column.monitors['thalamic'].values())
            
            if column.recurrent_parameters['model'] == 'STSD':
                self.monitors.append(column.monitors['err_stsd_term'])
            if column.thalamic_connection_parameters['model'] == 'STSD':
                self.monitors.append(column.monitors['thalamic_stsd_term'])

            self.monitors.extend(column.monitors['error'].values())
            self.monitors.extend(column.monitors['prediction'].values())
            self.monitors.extend(column.monitors['interneurons'].values())

        for column in self.columns_L2:
            self.neurons.append(column.error_neurons.neurons)
            self.neurons.append(column.prediction_neurons.neurons)
            self.neurons.append(column.interneurons.neurons)
            self.neurons.append(column.syn_err_pred)
            self.neurons.append(column.syn_pred_inh)
            self.neurons.append(column.syn_inh_err)
            self.neurons.append(column.syn_err_inh)
            self.neurons.append(column.syn_err_higher)
            self.neurons.append(column.syn_pred_lower)
            self.neurons.append(column.syn_err_inh_lat)
            self.neurons.append(column.syn_err_err)
            self.neurons.append(column.syn_err_lower)
            
            if column.recurrent_parameters['model'] == 'STSD':
                self.monitors.append(column.monitors['err_stsd_term'])

            self.monitors.extend(column.monitors['error'].values())
            self.monitors.extend(column.monitors['prediction'].values())
            self.monitors.extend(column.monitors['interneurons'].values())

        for column in self.columns_L3:
            self.neurons.append(column.error_neurons.neurons)
            self.neurons.append(column.prediction_neurons.neurons)
            self.neurons.append(column.interneurons.neurons)
            self.neurons.append(column.syn_err_pred)
            self.neurons.append(column.syn_pred_inh)
            self.neurons.append(column.syn_inh_err)
            self.neurons.append(column.syn_err_inh)
            self.neurons.append(column.syn_pred_mem)
            self.neurons.append(column.syn_mem_pred)
            self.neurons.append(column.syn_pred_mem_lat)
            self.neurons.append(column.syn_mem_pred_lat)
            self.neurons.append(column.syn_err_inh_lat)
            self.neurons.append(column.syn_err_lower)
            self.neurons.append(column.syn_err_err)

            if column.recurrent_parameters['model'] == 'STSD':
                self.monitors.append(column.monitors['err_stsd_term'])

            self.monitors.extend(column.monitors['error'].values())
            self.monitors.extend(column.monitors['prediction'].values())
            self.monitors.extend(column.monitors['interneurons'].values())

        for memory_network in self.memory_networks:
            self.neurons.append(memory_network.delay_line.neurons)
            self.neurons.append(memory_network.delay_line_inh.neurons)
            self.neurons.append(memory_network.syn_delay_line)
            self.neurons.append(memory_network.syn_inh)
            self.neurons.append(memory_network.syn_inh_ex)
            self.monitors.extend(memory_network.monitors.values())

        # Create the network with all neurons and monitors
        self.network_op = NetworkOperation(self.inject_impulse, dt=stimulus_time)
        self.weight_snap_op = NetworkOperation(self.record_weights, dt=20*ms)
        self.net = Network(*self.neurons, *self.monitors, self.network_op, self.weight_snap_op)

    def record_weights(self):
        for col in self.columns_L3:
            if hasattr(col, 'self_weight_snapshots'):
                col.self_weight_snapshots.append(col.syn_mem_pred.wt[:] * 1)
            if hasattr(col, 'lat_weight_snapshots'):
                col.lat_weight_snapshots.append(col.syn_mem_pred_lat.wt[:] * 1)
            
    def inject_impulse(self):
        print(f'Test at: {defaultclock.t}')
        if defaultclock.t in self.impulse_times:
            print(f'Stimulus at: {defaultclock.t}')
            if defaultclock.t in self.deviant_times:
                print(f'Deviant!!')
                self.thalamic_neurons[1].neurons.I_ext[:] = np.random.normal(1.9, 0.3, self.thalamic_neurons[1].N)*nA
                self.thalamic_neurons[0].neurons.I_ext[:] = np.random.normal(0, 0.3, self.thalamic_neurons[0].N)*nA
            else:
                self.thalamic_neurons[0].neurons.I_ext[:] = np.random.normal(1.9, 0.3, self.thalamic_neurons[0].N)*nA
                self.thalamic_neurons[1].neurons.I_ext[:] = np.random.normal(0, 0.3, self.thalamic_neurons[1].N)*nA
        else:
            self.thalamic_neurons[0].neurons.I_ext[:] = np.random.normal(0, 0.3, self.thalamic_neurons[0].N)*nA
            self.thalamic_neurons[1].neurons.I_ext[:] = np.random.normal(0, 0.3, self.thalamic_neurons[1].N)*nA
    
    def run(self, duration):
        self.net.run(duration, report='text', report_period=5*second)
        
    def get_monitors(self):
        return self.monitors