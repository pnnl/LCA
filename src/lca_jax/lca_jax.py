#   Copyright (C) 2023 Battelle Memorial Institute
#   SPDX-License-Identifier: BSD-2-Clause
#   See: https://spdx.org/licenses/

import os
import time
import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import Partial


def soft_threshold(voltage, threshold):
    return lax.max(float(0), lax.abs(voltage) - threshold) * lax.sign(voltage)


class LCA_Jax:
    def __init__(self, dictionary, display_period, threshold, tau):
        """
        Parameters
        ----------
        dictionary - a flattened dictionary. (elements, features)
        display_period - how many iterations to run
        threshold - v1 threshold / sparsity penalty
        tau - time constant
        """
        self.dictionary = dictionary
        self.display_period = display_period
        self.threshold = threshold
        self.tau = tau
        self.weights = ((-dictionary @ dictionary.T)
                        + jnp.eye(dictionary.shape[0])) * tau

    def __lca_iteration(self, voltage, _, bias):
        output = soft_threshold(voltage, self.threshold)
        voltage += (-voltage * self.tau) + lax.dot(output, self.weights) + bias
        return voltage, None

    def get_bias(self, input_vector):
        return jnp.einsum("k, jk -> j", input_vector, self.dictionary) * tau

    def inference(self, bias):
        """
        Performs a single inference

        Parameters
        ----------
        bias - the input bias for the vector to sparse code.

        Returns
        -------
        The sparse code representing the input.
        """
        final_voltage, _ = lax.scan(Partial(self.__lca_iteration, bias=bias),
                                    jnp.zeros(bias.shape), None,
                                    length=self.display_period, unroll=8)
        output = soft_threshold(final_voltage, self.threshold)
        return output

    def get_reconstruction(self, sparse_code):
        return jnp.einsum("j, jk -> k", sparse_code, self.dictionary)


if __name__ == "__main__":
    print("Running benchmark")
    tau = 2**-7
    threshold = 2**0
    idx = 0
    display_period = 256
    repetitions = 10
    batch_size = 2**14

    folder = os.path.dirname(os.path.realpath(__file__))
    dictionary = jnp.load(os.path.join(folder, 'MNIST_weights.npy'))
    dictionary_flat = dictionary.reshape(dictionary.shape[0], -1)

    lca = LCA_Jax(dictionary=dictionary_flat,
                  display_period=display_period,
                  threshold=threshold,
                  tau=tau)

    dataset_val = jnp.load(os.path.join(folder, "MNIST_val.npy"))
    indexes = jnp.concatenate((jnp.array([idx]), jnp.arange(batch_size - 1)))

    batch_get_bias = jax.vmap(lca.get_bias)
    bias = batch_get_bias(dataset_val[indexes].reshape((indexes.size, -1)))

    batch_inference = jax.vmap(lca.inference)

    start = time.perf_counter_ns()
    # Repeatedly perform inference for benchmark
    sparse_code, _ = lax.scan(lambda x, y: (batch_inference(bias), None),
                              jnp.zeros(bias.shape), None, length=repetitions)
    sparse_code.block_until_ready()

    # Only get the first output to measure MSE to match Loihi 2 benchmark
    sparse_code = sparse_code[0]
    stop = time.perf_counter_ns()

    print("Reconstructions per second",
          repetitions * batch_size / ((stop - start) * 10**-9))

    reconstruction = lca.get_reconstruction(sparse_code)
    image = reconstruction
    target = dataset_val[idx]
    mse = ((image.reshape(-1) - target.reshape(-1)) ** 2).mean()
    sparsity = ((sparse_code == 0).sum() / sparse_code.size)

    print("MSE", mse)
    print("Sparsity", sparsity)
