from brian2 import *

# Ad-Ex neurongroup class
# Includes spike and rate monitors
# N = number of neurons
# neuron_type: ('regular', 'fast_spiking', 'bursting')
# I_ext: Additional external current, in brian units
# recurrent_connections: True/False, within neurongroup recurrent connections
class AdExNeuronGroup():
    def __init__(self, N=40, I_ext=0*nA, neuron_type='regular'):
        self.N = N
        self.parameters = {
            'C': 281 * pF,
            'gL': 30 * nS,
            'taum': (281 * pF) / (30 * nS),
            'EL': -70.6 * mV,
            'VT': -50.4 * mV,
            'DeltaT': 2 * mV,
            'Vcut': -50.4 * mV + 5 * 2 * mV,
            'tau_AMPA': 5.0*ms,
            'tau_GABA': 10.0*ms,
            'E_AMPA': 0.0*mV,
            'E_GABA': -70.0*mV
        }

        if neuron_type == 'regular':
            self.parameters.update({
                'tauw': 144*ms,
                'a': 4*nS,
                'b': 0.0805*nA,
                'Vr': -70.6*mV
            })
        elif neuron_type == 'fast_spiking':
            self.parameters.update({
                'tauw': 144*ms,
                'a': 2*self.parameters['C']/(144*ms),
                'b': 0*nA,
                'Vr': -70.6*mV
            })
        
        # Adex Model Equations
        self.eqs = """
        dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + I_AMPA + I_GABA + I_ext - w)/C : volt (unless refractory)
        dw/dt = (a*(vm - EL) - w)/tauw : amp (unless refractory)
        I_AMPA = g_AMPA * (E_AMPA - vm) : amp
        I_GABA = g_GABA * (E_GABA - vm) : amp
        
        dg_AMPA/dt = - g_AMPA/tau_AMPA : siemens
        dg_GABA/dt = - g_GABA/tau_GABA : siemens
        I_ext: amp
        """
        self.neurons = NeuronGroup(self.N, model=self.eqs, threshold=f'vm>Vcut', reset=f"vm=Vr; w+=b", refractory=1*ms,
                                   method='euler', namespace=self.parameters)
        self.neurons.vm[:] = self.parameters['EL']
        self.neurons.I_ext[:] = I_ext