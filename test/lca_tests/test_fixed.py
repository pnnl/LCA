#  Copyright (C) 2023 Battelle Memorial Institute
#  SPDX-License-Identifier: BSD-2-Clause
#  See: https://spdx.org/licenses/

import unittest

import numpy as np
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io import sink

from lca.process import LCA1Layer, LCA2Layer
from lca.lca_neuron.process import LCANeuron
from lca.accumulator.process import AccumulatorNeuron
from lca.models import LCA1LayerModelFixed
from lca.accumulator.models import PyAccumulatorFixed
from lca.lca_neuron.models import PyLCANeuronFixed


class TestLCAFixed(unittest.TestCase):
    def test_identity_matrix(self):
        weights = np.eye(5, dtype=np.int8) * 2**8
        weight_exp = -8
        input_val = np.array([6502, 29847, 14746, 8168, 12989])
        threshold = 1
        lca = LCA2Layer(weights=weights, weights_exp=weight_exp,
                        input_vec=input_val, threshold=threshold)
        
        v1_output = sink.RingBuffer(shape=(5,), buffer=1)
        res_output = sink.RingBuffer(shape=(5,), buffer=1)
        lca.v1.connect(v1_output.a_in)
        lca.res.connect(res_output.a_in)

        run_config = Loihi1SimCfg(select_tag='fixed_pt',
                                  select_sub_proc_model=True,
                                  exception_proc_model_map={
                                      LCANeuron: PyLCANeuronFixed,
                                      AccumulatorNeuron: PyAccumulatorFixed})
        
        v1_output.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        self.assertTrue(
            np.allclose(input_val, v1_output.data.get()[:, 0], atol=600,
                        rtol=1e-2),
            f"Expected: {input_val} Actual: {v1_output.data.get()[:, 0]}")

        v1_output.stop()

    def test_negative_residual(self):
        weights = np.eye(5) * 2**8
        weight_exp = -8
        input_val = np.array([26366, -18082, 5808, -10212, -25449])
        threshold = 1
        lca = LCA2Layer(weights=weights, weights_exp=weight_exp,
                        input_vec=input_val, threshold=threshold)

        v1_output = sink.RingBuffer(shape=(5,), buffer=1)
        res_output = sink.RingBuffer(shape=(5,), buffer=1)
        lca.v1.connect(v1_output.a_in)
        lca.res.connect(res_output.a_in)

        run_config = Loihi1SimCfg(select_tag='fixed_pt',
                                  select_sub_proc_model=True,
                                  exception_proc_model_map={
                                      LCANeuron: PyLCANeuronFixed,
                                      AccumulatorNeuron: PyAccumulatorFixed})
    
        v1_output.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        self.assertTrue(
            np.allclose(input_val, v1_output.data.get()[:, 0], atol=600,
                        rtol=1e-2),
            f"Expected: {input_val} Actual: {v1_output.data.get()[:, 0]}")
        
        v1_output.stop()
    
    def test_2_layer_competition(self):
        weights = np.array([[0, np.sqrt(1/2), np.sqrt(1/2)], 
                            [np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)]]) * 2**8
        weight_exp = -8
        input_val = np.array([0, 2**14, 2**14])
        expected = np.array([2**14 / np.sqrt(1/2), 0])

        threshold = 1
        lca = LCA2Layer(weights=weights, weights_exp=weight_exp,
                        input_vec=input_val, threshold=threshold)

        v1_output = sink.RingBuffer(shape=(2,), buffer=1)
        res_output = sink.RingBuffer(shape=(3,), buffer=1)
        lca.v1.connect(v1_output.a_in)
        lca.res.connect(res_output.a_in)

        run_config = Loihi1SimCfg(select_tag='fixed_pt',
                                  select_sub_proc_model=True,
                                  exception_proc_model_map={
                                      LCANeuron: PyLCANeuronFixed,
                                      AccumulatorNeuron: PyAccumulatorFixed})
 
        v1_output.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        self.assertTrue(
            np.allclose(expected, v1_output.data.get()[:, 0], atol=600,
                        rtol=1e-2),
            f"Expected: {expected} Actual: {v1_output.data.get()[:, 0]}")
        
        v1_output.stop()

    def test_2_layer_excitation(self):
        weights = np.array([[-np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)],
                            [1, 0, 0]]) * 2**8
        weight_exp = -8
        input_val = np.array([0, 2**14, 2**14])
        expected = np.array([2**14 / np.sqrt(1/3), 2**14])

        threshold = 1
        lca = LCA2Layer(weights=weights, weights_exp=weight_exp,
                        input_vec=input_val, threshold=threshold)

        v1_output = sink.RingBuffer(shape=(2,), buffer=1)
        res_output = sink.RingBuffer(shape=(3,), buffer=1)
        lca.v1.connect(v1_output.a_in)
        lca.res.connect(res_output.a_in)

        run_config = Loihi1SimCfg(select_tag='fixed_pt',
                                  select_sub_proc_model=True,
                                  exception_proc_model_map={
                                      LCANeuron: PyLCANeuronFixed,
                                      AccumulatorNeuron: PyAccumulatorFixed})
 
        v1_output.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        # TODO: this is a generous tolerance for the solution,
        # try to get a closer solution.
        self.assertTrue(
            np.allclose(expected, v1_output.data.get()[:, 0], atol=600,
                        rtol=1e-1),
            f"Expected: {expected} Actual: {v1_output.data.get()[:, 0]}")
        
        v1_output.stop()

    def test_boundary_check(self):
        weights = np.eye(2, dtype=np.int8)
        i16info = np.iinfo(np.int16)
        input_val = np.array([i16info.max + 1, 0])
        threshold = 1
        
        with self.assertRaises(AssertionError):
            lca = LCA2Layer(weights=weights, input_vec=input_val,
                            threshold=threshold)
            v1_output = sink.RingBuffer(shape=(5,), buffer=1)
            res_output = sink.RingBuffer(shape=(5,), buffer=1)
            lca.v1.connect(v1_output.a_in)
            lca.res.connect(res_output.a_in)

            run_config = Loihi1SimCfg(select_tag='fixed_pt',
                                      select_sub_proc_model=True,
                                      exception_proc_model_map={
                                          LCANeuron: PyLCANeuronFixed,
                                          AccumulatorNeuron:
                                              PyAccumulatorFixed})
        
            v1_output.run(condition=RunSteps(num_steps=1), run_cfg=run_config)

        input_val = np.array([0, i16info.min - 1])
        with self.assertRaises(AssertionError):
            LCA2Layer(weights=weights, input_vec=input_val, threshold=threshold)
            v1_output = sink.RingBuffer(shape=(5,), buffer=1)
            res_output = sink.RingBuffer(shape=(5,), buffer=1)
            lca.v1.connect(v1_output.a_in)
            lca.res.connect(res_output.a_in)

            run_config = Loihi1SimCfg(select_tag='fixed_pt',
                                      select_sub_proc_model=True,
                                      exception_proc_model_map={
                                          LCANeuron: PyLCANeuronFixed,
                                          AccumulatorNeuron:
                                              PyAccumulatorFixed})
        
            v1_output.run(condition=RunSteps(num_steps=1), run_cfg=run_config)

    def test_1_layer_competition(self):
        weights = np.array([[0, np.sqrt(1 / 2), np.sqrt(1 / 2)],
                            [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]])
        w = (np.einsum('bj,ij->bi', -weights, weights)
             + np.eye(weights.shape[0])) * 0.1 * 2 ** 11
        weights_exp = -11
        input_val = np.array([0, 2 ** 22, 2 ** 22])
        bias = (input_val @ weights.T * 0.1).astype(np.int32)

        expected = np.array([2 ** 22 / np.sqrt(1 / 2), 0])

        threshold = 1
        lca = LCA1Layer(weights=w, weights_exp=weights_exp, bias=bias,
                        threshold=threshold)

        run_config = Loihi1SimCfg(select_tag='fixed_pt',
                                  select_sub_proc_model=True,
                                  exception_proc_model_map={
                                      LCA1Layer: LCA1LayerModelFixed,
                                      LCANeuron: PyLCANeuronFixed})

        lca.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        actual = lca.voltage.get()

        lca.stop()

        # Note: lower precision of weights results in less inhibition.
        self.assertTrue(np.allclose(expected, actual, atol=20000, rtol=1e-3),
                        f"Expected: {expected} Actual: {actual}")

    def test_1_layer_excitation(self):
        weights = np.array([[-np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)],
                            [1, 0, 0]])
        w = (np.einsum('bj,ij->bi', -weights, weights)
             + np.eye(weights.shape[0])) * 0.1 * 2 ** 12
        weights_exp = -12
        input_val = np.array([0, 2 ** 22, 2 ** 22])
        bias = (input_val @ weights.T * 0.1).astype(np.int32)
        expected = np.array([2 ** 22 / np.sqrt(1 / 3), 2 ** 22])

        threshold = 1
        lca = LCA1Layer(weights=w, weights_exp=weights_exp, bias=bias,
                        threshold=threshold)

        run_config = Loihi1SimCfg(select_tag='fixed_pt',
                                  select_sub_proc_model=True,
                                  exception_proc_model_map={
                                      LCA1Layer: LCA1LayerModelFixed,
                                      LCANeuron: PyLCANeuronFixed})

        lca.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        actual = lca.voltage.get()

        lca.stop()

        # TODO: see if we can get a closer solution.
        self.assertTrue(np.allclose(expected, actual, atol=0, rtol=2e-1),
                        f"Expected: {expected} Actual: {actual}")


if __name__ == "__main__":
    unittest.main()
