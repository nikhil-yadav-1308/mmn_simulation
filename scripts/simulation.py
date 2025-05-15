from brian2 import *
import numpy as np
import h5py
from time import strftime
import os

from scripts.neuron_groups import *
from scripts.synapses import *
from scripts.utils import *
from scripts.memory_networks import *
from scripts.column import *
from scripts.full_network import *

# Simulation function
def simulate_network(simulation_time,
                     impulse_times, deviant_times,
                     model, parameters, stimulus_time=50*ms,
                     num_simulations=1, num_columns=2, N=40,
                     sim_file_title='Simulation', simulation_folder='/',
                     store_weights=True, smoothing_width=50*ms):
    
    simulation_data = {}
    simulation_file = f'{sim_file_title}_{strftime("%Y_%m_%d-%H_%M_%S")}.hdf5'
    if not os.path.exists('./simulation_data'):
        os.makedirs('./simulation_data')
    if not os.path.exists(f'./simulation_data/{simulation_folder}'):
        os.makedirs(f'./simulation_data/{simulation_folder}')

    # Load model parameters
    P = parameters

    for trial in range(num_simulations):
        start_scope()
        
        # Create multiple columns
        thalamic_neurons = [AdExNeuronGroup(N=N) for _ in range(num_columns)]
        columns_L1 = [CorticalColumn(N, P.column_connection_parameters, name=f"L1_C{column+1}") for column in range(num_columns)]
        columns_L2 = [CorticalColumn(N, P.column_connection_parameters, name=f"L2_C{column+1}") for column in range(num_columns)]
        columns_L3 = [CorticalColumn(N, P.column_connection_parameters, name=f"L3_C{column+1}") for column in range(num_columns)]
        memory_networks = [DelayLine(200, P.delay_line_connection_parameters, name=f"MN_{column+1}") for column in range(num_columns)]

        # Error recurrence
        for column in range(num_columns):
            columns_L1[column].err_recurrent(P.err_recurrent_connection_parameters, P.stsd_synapse_parameters)
            columns_L2[column].err_recurrent(P.err_recurrent_connection_parameters, P.stsd_synapse_parameters)
            columns_L3[column].err_recurrent(P.err_recurrent_connection_parameters, P.stsd_synapse_parameters)
        
        
        # Connect Thalamic neurons to L1 columns
        for column in range(num_columns):
            columns_L1[column].connect_thalamic_neurons(thalamic_neurons[column],
                                                        P.thalamic_connection_parameters,
                                                        P.stsd_synapse_parameters)
        
        # Connect Memory network to L2 columns
        for column in range(num_columns):
            columns_L3[column].connect_memory_self(memory_networks[column],
                                                   P.memory_connection_parameters,
                                                   P.stdp_parameters,
                                                   store_weights=store_weights)
            
        # Connect L1 and L2 columns
        for column in range(num_columns):
            columns_L1[column].connect_higher_level(columns_L2[column],
                                                    P.higher_connection_parameters,
                                                    P.stsd_synapse_parameters)

        # Connect L2 and L3 columns
        for column in range(num_columns):
            columns_L2[column].connect_higher_level(columns_L3[column],
                                                    P.higher_connection_parameters,
                                                    P.stsd_synapse_parameters)

        # Connect lateral columns
        for column_1 in range(num_columns):
            for column_2 in range(num_columns):
                if column_1 != column_2:
                    columns_L1[column_1].connect_lateral_column(columns_L1[column_2], P.lateral_connection_parameters)
                    columns_L2[column_1].connect_lateral_column(columns_L2[column_2], P.lateral_connection_parameters)
                    columns_L3[column_1].connect_lateral_column(columns_L3[column_2], P.lateral_connection_parameters)

        # Connect_error_to_lower
        for column in range(num_columns):
            columns_L3[column].connect_error_to_lower(columns_L2[column],
                                                      P.feedback_connection_parameters,
                                                      P.stsd_synapse_parameters)
            columns_L2[column].connect_error_to_lower(columns_L1[column],
                                                      P.feedback_connection_parameters,
                                                      P.stsd_synapse_parameters)

        # Connect the delay lines to prediction units of other columns
        columns_L3[0].connect_memory_lateral(memory_networks[1],
                                             P.memory_other_connection_parameters,
                                             P.stdp_parameters,
                                             store_weights=store_weights)
        columns_L3[1].connect_memory_lateral(memory_networks[0],
                                             P.memory_other_connection_parameters,
                                             P.stdp_parameters,
                                             store_weights=store_weights)
        
        # Create a network from these columns for simulations
        network = FullNetwork(thalamic_neurons,
                              columns_L1, columns_L2, columns_L3,
                              memory_networks,
                              impulse_times, deviant_times,
                              stimulus_time, store_weights)
        
        # Set external current input for all columns
        for thalamic_neuron in thalamic_neurons:
            thalamic_neuron.neurons.I_ext[:] = np.random.normal(0, 0.3, N) * nA
            
        for column in columns_L1:
            column.error_neurons.neurons.I_ext[:] = np.random.normal(0, 1, N)*nA
            column.prediction_neurons.neurons.I_ext[:] = np.random.normal(0, 1, N)*nA
            column.interneurons.neurons.I_ext[:] = np.random.normal(0, 1.5, N//2)*nA
        
        for column in columns_L2:
            column.error_neurons.neurons.I_ext[:] = np.random.normal(0, 1, N)*nA
            column.prediction_neurons.neurons.I_ext[:] = np.random.normal(0, 1, N)*nA
            column.interneurons.neurons.I_ext[:] = np.random.normal(0, 1.5, N//2)*nA

        for column in columns_L3:
            column.error_neurons.neurons.I_ext[:] = np.random.normal(0, 1, N)*nA
            column.prediction_neurons.neurons.I_ext[:] = np.random.normal(0, 1, N)*nA
            column.interneurons.neurons.I_ext[:] = np.random.normal(0, 1.5, N//2)*nA

        # This is when the network simulation actually runs
        network.run(simulation_time)

        # Storing all the data to HDF5 files
        with h5py.File(f'./simulation_data/{simulation_folder}/{simulation_file}', 'a') as f:
            trial_group = f.create_group(f"Trial_{trial+1}")
            sim_time = trial_group.create_dataset("sim_time", data=np.array(columns_L1[0].monitors['error']['I_AMPA'].t/ms), compression="gzip")
            
            L1_monitors = trial_group.create_group("L1_monitors")
            for col_index, column in enumerate(columns_L1):
                column_group = L1_monitors.create_group(f"column_{col_index+1}")
                for neuron_group in column.monitors.keys():
                    if neuron_group == 'thalamic_stsd_term':
                        column_group.create_dataset("thalamic_stsd_term", data=np.array(columns_L1[col_index].monitors['thalamic_stsd_term'].n), compression="gzip")
                        continue
                    if neuron_group == 'err_stsd_term':
                        column_group.create_dataset("err_stsd_term", data=np.array(columns_L1[col_index].monitors['err_stsd_term'].n), compression="gzip")
                        continue
                    group = column_group.create_group(neuron_group)
                    group.create_dataset("spike_indices", data=np.array(columns_L1[col_index].monitors[neuron_group]['spikes'].i), compression="gzip")
                    group.create_dataset("spike_times", data=np.array(columns_L1[col_index].monitors[neuron_group]['spikes'].t/ms), compression="gzip")
                    group.create_dataset("rate", data=np.array(columns_L1[col_index].monitors[neuron_group]['rate'].smooth_rate(window='flat', width=smoothing_width)/Hz), compression="gzip")
                    group.create_dataset("I_AMPA", data=np.array(columns_L1[col_index].monitors[neuron_group]['I_AMPA'].I_AMPA/pA), compression="gzip")
                    group.create_dataset("I_GABA", data=np.array(columns_L1[col_index].monitors[neuron_group]['I_GABA'].I_GABA/pA), compression="gzip")

            L2_monitors = trial_group.create_group("L2_monitors")
            for col_index, column in enumerate(columns_L2):
                column_group = L2_monitors.create_group(f"column_{col_index+1}")
                for neuron_group in column.monitors.keys():
                    if neuron_group == 'err_stsd_term':
                        column_group.create_dataset("err_stsd_term", data=np.array(columns_L2[col_index].monitors['err_stsd_term'].n), compression="gzip")
                        continue
                    group = column_group.create_group(neuron_group)
                    group.create_dataset("spike_indices", data=np.array(columns_L2[col_index].monitors[neuron_group]['spikes'].i), compression="gzip")
                    group.create_dataset("spike_times", data=np.array(columns_L2[col_index].monitors[neuron_group]['spikes'].t/ms), compression="gzip")
                    group.create_dataset("rate", data=np.array(columns_L2[col_index].monitors[neuron_group]['rate'].smooth_rate(window='flat', width=smoothing_width)/Hz), compression="gzip")
                    group.create_dataset("I_AMPA", data=np.array(columns_L2[col_index].monitors[neuron_group]['I_AMPA'].I_AMPA/pA), compression="gzip")
                    group.create_dataset("I_GABA", data=np.array(columns_L2[col_index].monitors[neuron_group]['I_GABA'].I_GABA/pA), compression="gzip")

            L3_monitors = trial_group.create_group("L3_monitors")
            for col_index, column in enumerate(columns_L3):
                column_group = L3_monitors.create_group(f"column_{col_index+1}")
                if model=='PC' or model=='CO':
                    column_group.create_dataset("self_weight_monitor", data=np.array(columns_L3[col_index].self_weight_snapshots), compression="gzip")
                    column_group.create_dataset("lat_weight_monitor", data=np.array(columns_L3[col_index].lat_weight_snapshots), compression="gzip")
                    column_group.create_dataset("self_weight_connections",
                                         data=np.vstack([columns_L3[col_index].syn_mem_pred.i[:], columns_L3[col_index].syn_mem_pred.j[:]]).T, compression="gzip")
                    column_group.create_dataset("lat_weight_connections",
                                         data=np.vstack([columns_L3[col_index].syn_mem_pred_lat.i[:], columns_L3[col_index].syn_mem_pred_lat.j[:]]).T, compression="gzip")
                for neuron_group in column.monitors.keys():
                    if neuron_group == 'err_stsd_term':
                        column_group.create_dataset("err_stsd_term", data=np.array(columns_L3[col_index].monitors['err_stsd_term'].n), compression="gzip")
                        continue
                    group = column_group.create_group(neuron_group)
                    group.create_dataset("spike_indices", data=np.array(columns_L3[col_index].monitors[neuron_group]['spikes'].i), compression="gzip")
                    group.create_dataset("spike_times", data=np.array(columns_L3[col_index].monitors[neuron_group]['spikes'].t/ms), compression="gzip")
                    group.create_dataset("rate",
                          data=np.array(columns_L3[col_index].monitors[neuron_group]['rate'].smooth_rate(window='flat', width=smoothing_width)/Hz), compression="gzip")
                    group.create_dataset("I_AMPA", data=np.array(columns_L3[col_index].monitors[neuron_group]['I_AMPA'].I_AMPA/pA), compression="gzip")
                    group.create_dataset("I_GABA", data=np.array(columns_L3[col_index].monitors[neuron_group]['I_GABA'].I_GABA/pA), compression="gzip")

            memory_monitors = trial_group.create_group("memory_monitors")
            for index, delay_line in enumerate(memory_networks):
                memory_group = memory_monitors.create_group(f"memory_network_{index+1}")
                spike_indices = memory_group.create_dataset("spike_indices", data=np.array(memory_networks[index].monitors['spikes'].i), compression="gzip")
                spike_times = memory_group.create_dataset("spike_times", data=np.array(memory_networks[index].monitors['spikes'].t/ms), compression="gzip")
                rate = memory_group.create_dataset("rate", data=np.array(memory_networks[index].monitors['rate'].smooth_rate(window='flat', width=smoothing_width)/Hz), compression="gzip")
                I_AMPA = memory_group.create_dataset("I_AMPA", data=np.array(memory_networks[index].monitors['I_AMPA'].I_AMPA/pA), compression="gzip")
                I_GABA = memory_group.create_dataset("I_GABA", data=np.array(memory_networks[index].monitors['I_GABA'].I_GABA/pA), compression="gzip")

            print(f'Simulation data stored in file: {simulation_file}!')
    return network