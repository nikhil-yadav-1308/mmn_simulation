from brian2 import *

class ConfigurableSynapses(Synapses):
    def __init__(self, source, target, is_inh=False, to_delay_line=False, is_delay_line=False, synapse_parameters=None, method='euler', namespace=None):
        self.synapse_parameters = synapse_parameters
        
        if is_inh:
            if self.synapse_parameters['model'].upper() == 'FIXED':
                model = '''
                wt : 1
                '''
                on_pre = '''
                g_GABA += wt*nS
                '''
                super().__init__(source, target, model=model, on_pre=on_pre, method=method, namespace=namespace)
        else:
            if self.synapse_parameters['model'].upper() == 'FIXED':
                model = '''
                wt : 1
                '''
                on_pre = '''
                g_AMPA += wt*nS
                '''
                super().__init__(source, target, model=model, on_pre=on_pre, method=method, namespace=namespace)
            elif self.synapse_parameters['model'].upper() == 'STSD':
                model = '''
                wt : 1
                dn/dt = (1 - n) / tau_rec : 1 (clock-driven)
                '''
                on_pre = '''
                n = clip(n*(1-u), 0, 1)
                g_AMPA += wt*n*nS
                '''
                super().__init__(source, target, model=model, on_pre=on_pre, method=method, namespace=namespace)
            elif self.synapse_parameters['model'].upper() == 'STDP':
                model = '''
                dapre/dt = -apre/taupre : 1 (event-driven)
                dapost/dt = -apost/taupost : 1 (event-driven)
                dn/dt = (1 - n) / tau_rec : 1 (clock-driven)
                wt : 1
                '''
                on_pre = '''
                n = clip(n*(1-u), 0, 1)
                g_AMPA += wt*n*nS
                apre += Apre
                wt = clip(wt + apost, 0, wmax)
                '''
                on_post = '''
                apost += Apost
                wt = clip(wt + apre, 0, wmax)
                '''
                super().__init__(source, target, model=model, on_pre=on_pre, on_post=on_post, method=method, namespace=namespace)
        if to_delay_line:
            self.connect(i=np.arange(source.N), j=0)
        elif is_delay_line:
            self.connect(j='i+1', skip_if_invalid=True)
            
        else:
            self.connect(p=self.synapse_parameters['conn'])

        if self.synapse_parameters['model'].upper() == 'STDP':
            self.wt = np.clip(np.random.normal(self.synapse_parameters['mean'],
                                                            self.synapse_parameters['sd'],
                                                            len(self.i)),
                                           0, self.synapse_parameters['wmax'])
        else:            
            self.wt = np.random.normal(self.synapse_parameters['mean'],
                                       self.synapse_parameters['sd'],
                                       len(self.i))
        
        self.delay = self.synapse_parameters['delay']