import numpy as np

from lava.magma.core.model.sub.model import AbstractSubProcessModel

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements, tag
from lava.proc.dense.process import Dense

from lca.process import LCA1Layer, LCA2Layer
from lca.lca_neuron.process import LCANeuron
from lca.accumulator.process import AccumulatorNeuron


@implements(proc=LCA2Layer, protocol=LoihiProtocol)
@tag('floating_pt')
class LCA2LayerModelFloat(AbstractSubProcessModel):
    def __init__(self, proc: LCA2Layer):
        threshold = proc.threshold.get()
        T = proc.tau.get()
        T_exp = proc.tau_exp.get()
        T_float = np.ldexp(T, T_exp)

        weights = proc.weights.get()
        weights_exp = proc.weights_exp.get()
        input_val = proc.input.get()
        spike_height = proc.spike_height.get()

        self.v1 = LCANeuron(shape=(weights.shape[0],), tau=T_float,
                            vth=threshold, two_layer=True)
        # weight_exp shifted 8 bits for the weights, 6 for the v1 output.
        self.weights_T = Dense(weights=-weights.T, num_message_bits=24,
                               weight_exp=weights_exp)

        self.res = AccumulatorNeuron(shape=(weights.shape[1],),
                                     spike_height=spike_height, bias=input_val)

        self.weights = Dense(weights=(weights * T), num_message_bits=24,
                             weight_exp=weights_exp + T_exp)

        self.weights.a_out.connect(self.v1.a_in)
        self.res.s_out.connect(self.weights.s_in)

        self.weights_T.a_out.connect(self.res.a_in)
        self.v1.s_out.connect(self.weights_T.s_in)

        # Expose output and voltage
        self.v1.s_out.connect(proc.out_ports.v1)
        self.res.s_out.connect(proc.out_ports.res)
        proc.vars.voltage.alias(self.v1.vars.v)
        proc.vars.input.alias(self.res.bias)


@implements(proc=LCA1Layer, protocol=LoihiProtocol)
@tag('floating_pt')
class LCA1LayerModelFloat(AbstractSubProcessModel):
    def __init__(self, proc: LCA1Layer):
        threshold = proc.threshold.get()
        T = proc.tau.get()
        T_exp = proc.tau_exp.get()
        T_float = np.ldexp(T, T_exp)

        weights = proc.weights.get()
        weights_exp = proc.weights_exp.get()
        bias = proc.bias.get()

        self.v1 = LCANeuron(shape=(weights.shape[0],), tau=T_float,
                            vth=threshold, bias=bias, two_layer=False)
        # weight_exp shifted 8 bits for the weights, 6 for the v1 output.
        self.weights = Dense(weights=weights, num_message_bits=24,
                             weight_exp=weights_exp)

        self.weights.a_out.connect(self.v1.a_in)
        self.v1.s_out.connect(self.weights.s_in)

        # Expose output and voltage
        self.v1.s_out.connect(proc.out_ports.v1)
        proc.vars.voltage.alias(self.v1.vars.v)
        proc.vars.bias.alias(self.v1.bias)


@implements(proc=LCA2Layer, protocol=LoihiProtocol)
@tag('fixed_pt')
class LCA2LayerModelFixed(AbstractSubProcessModel):
    def __init__(self, proc: LCA2Layer):
        threshold = proc.threshold.get()
        T = proc.tau.get()
        T_exp = proc.tau_exp.get()
        T_int = int(np.ldexp(T, T_exp + 24))

        weights = proc.weights.get()
        weights_exp = proc.weights_exp.get()
        input_val = proc.input.get()
        spike_height = proc.spike_height.get()

        assert np.issubdtype(input_val.dtype, np.integer),\
            "Input must be an integer"

        self.v1 = LCANeuron(shape=(weights.shape[0],), tau=T_int, vth=threshold,
                            two_layer=True)

        self.weights_T = Dense(weights=-weights.T, num_message_bits=24,
                               weight_exp=weights_exp)

        self.res = AccumulatorNeuron(shape=(weights.shape[1],),
                                     spike_height=spike_height, bias=input_val)

        self.weights = Dense(weights=(weights * T), num_message_bits=24,
                             weight_exp=weights_exp + T_exp)

        self.weights.a_out.connect(self.v1.a_in)
        self.res.s_out.connect(self.weights.s_in)

        self.weights_T.a_out.connect(self.res.a_in)
        self.v1.s_out.connect(self.weights_T.s_in)

        # Expose output and voltage
        self.v1.s_out.connect(proc.out_ports.v1)
        self.res.s_out.connect(proc.out_ports.res)
        proc.vars.voltage.alias(self.v1.vars.v)
        proc.vars.input.alias(self.res.bias)


@implements(proc=LCA1Layer, protocol=LoihiProtocol)
@tag('fixed_pt')
class LCA1LayerModelFixed(AbstractSubProcessModel):
    def __init__(self, proc: LCA1Layer):
        threshold = proc.threshold.get()
        T = proc.tau.get()
        T_exp = proc.tau_exp.get()
        T_int = int(np.ldexp(T, T_exp + 24))

        weights = proc.weights.get()
        weights_exp = proc.weights_exp.get()
        bias = proc.bias.get()

        assert np.issubdtype(bias.dtype, np.integer), "Bias must be an integer"

        self.v1 = LCANeuron(shape=(weights.shape[0],), tau=T_int, vth=threshold,
                            bias=bias, two_layer=False)
        # weight_exp shifted 8 bits for the weights, 6 for the v1 output.
        self.weights = Dense(weights=weights, num_message_bits=24,
                             weight_exp=weights_exp)

        self.weights.a_out.connect(self.v1.a_in)
        self.v1.s_out.connect(self.weights.s_in)

        # Expose output and voltage
        self.v1.s_out.connect(proc.out_ports.v1)
        proc.vars.voltage.alias(self.v1.vars.v)
        proc.vars.bias.alias(self.v1.bias)