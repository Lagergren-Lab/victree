import random
import sys
import os
from os import path
from matplotlib.pyplot import logging

import torch
import torch.nn.functional as f
import unittest
from utils import tree_utils


class treeUtilsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_one_slice_marginals_markov_chain_identity_transition_matrix_gives_init_state_throughout_chain(self):
        K = 8
        N = 100
        init_state = 3
        init_prob = torch.zeros((K,))
        init_prob[init_state] = 1
        trans_prob = torch.empty((N, K, K))
        no_trans = torch.diag(torch.ones((K)))
        for n in range(N):
            trans_prob[n] = no_trans
        state_probs = tree_utils.one_slice_marginals_markov_chain(init_prob, trans_prob)

        for n in range(N):
            self.assertAlmostEqual(state_probs[n, :].sum(), 1.0, msg=f"state probs don't sum to 1.0 at position {n}")
            for j in range(K):
                if j == init_state:
                    self.assertAlmostEqual(state_probs[n, init_state], 1.0, msg=f"state_probs != 1.0 at position {n}")
                else:
                    self.assertAlmostEqual(state_probs[n, j], 0.0, msg=f"state_probs != 0.0 at position {n}")

    def test_one_slice_marginals_markov_chain_uniform_init_and_uniform_transitions_gives_uniform_state_probs(self):
        K = 8
        N = 100
        init_prob = torch.ones((K,)) / K
        trans_prob = torch.empty((N, K, K))
        uniform_trans = torch.ones((K, K)) / K
        for n in range(N):
            trans_prob[n] = uniform_trans
        state_probs = tree_utils.one_slice_marginals_markov_chain(init_prob, trans_prob)

        for n in range(N):
            self.assertAlmostEqual(state_probs[n, :].sum(), 1.0, msg=f"state probs don't sum to 1.0 at position {n}")
            for j in range(K):
                self.assertAlmostEqual(state_probs[n, j], 1. / K, msg=f"state_probs not uniform at position {n}")

    def test_one_slice_marginals_markov_chain_coupled_states_should_have_equal_probability_for_equal_init_and_transition(self):
        K = 8
        N = 100
        state_1 = 0
        state_2 = 1

        init_states = (state_1, state_2)
        init_prob = torch.zeros((K,))
        init_prob[init_states[0]] = 0.5
        init_prob[init_states[1]] = 0.5
        trans_prob = torch.empty((N, K, K))
        no_trans = torch.diag(torch.ones((K)))
        for n in range(N):
            trans_prob[n] = no_trans
            trans_prob[n, 0, 0] = 0.5
            trans_prob[n, 0, 1] = 0.5
            trans_prob[n, 1, 1] = 0.5
            trans_prob[n, 1, 0] = 0.5

        state_probs = tree_utils.one_slice_marginals_markov_chain(init_prob, trans_prob)

        for n in range(N):
            self.assertAlmostEqual(state_probs[n, :].sum(), 1.0, msg=f"state probs don't sum to 1.0 at position {n}")
            for j in range(K):
                if j == state_1 or j == state_2:
                    self.assertAlmostEqual(state_probs[n, j], 0.5, msg=f"state_probs != 0.5 at position {n}")
                    self.assertAlmostEqual(state_probs[n, j], 0.5, msg=f"state_probs != 0.5 at position {n}")
                else:
                    self.assertAlmostEqual(state_probs[n, j], 0.0, msg=f"state_probs != 0.0 at position {n}")

    def test_one_slice_marginals_markov_chain_biased_towards_biased_state_for_time_homogeneous_random_transitions(self):
        K = 8
        N = 100
        init_state = 3
        biased_state = 5
        init_prob = torch.zeros((K,))
        init_prob[init_state] = 1
        trans_prob = torch.empty((N, K, K))
        random_trans = torch.rand((K,K))
        for n in range(N):
            trans_prob[n] = random_trans
            trans_prob[n, :, biased_state] = 1.
            #trans_prob[n] = trans_prob[n] / (torch.sum(trans_prob[n], 1).unsqueeze(-1))
            trans_prob[n] = f.normalize(trans_prob[n], dim=1, p=1)

        state_probs = tree_utils.one_slice_marginals_markov_chain(init_prob, trans_prob)

        for n in range(int(N/3), N):
            self.assertAlmostEqual(state_probs[n, :].sum().item(), 1.0, places=5, msg=f"state probs don't sum to 1.0 at position {n}")
            for j in range(K):
                if j == biased_state:
                    continue
                else:
                    self.assertGreater(state_probs[n, biased_state], state_probs[n, j],
                                       msg=f"biased state: {biased_state} (prob {state_probs[n, biased_state]}) \n"
                                           f" less probable than state: {j} (prob {state_probs[n, biased_state]}) at position {n}")

    def test_one_slice_marginals_markov_chain_biased_towards_biased_state_for_non_time_homogeneous_random_transitions(self):
        K = 8
        N = 100
        init_state = 3
        biased_state = 5
        init_prob = torch.zeros((K,))
        init_prob[init_state] = 1
        trans_prob = torch.empty((N, K, K))
        random_trans = torch.rand((N,K,K))
        for n in range(N):
            trans_prob[n] = random_trans[n]
            trans_prob[n, :, biased_state] = 1.
            trans_prob[n] = f.normalize(trans_prob[n], dim=1, p=1)

        state_probs = tree_utils.one_slice_marginals_markov_chain(init_prob, trans_prob)

        for n in range(int(N/3), N):
            self.assertAlmostEqual(state_probs[n, :].sum().item(), 1.0, places=5, msg=f"state probs don't sum to 1.0 at position {n}")
            for j in range(K):
                if j == biased_state:
                    continue
                else:
                    self.assertGreater(state_probs[n, biased_state], state_probs[n, j],
                                       msg=f"biased state: {biased_state} (prob {state_probs[n, biased_state]}) \n"
                                           f" less probable than state: {j} (prob {state_probs[n, biased_state]}) at position {n}")

    def tearDown(self) -> None:
        return super().tearDown()
