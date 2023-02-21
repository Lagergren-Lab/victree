from scipy.stats import dirichlet, multinomial, gamma, poisson
import numpy as np
import math

# FROM CopyMix
# TODO: DIC (see formula on wiki, this function implements that)
#   elbow method https://en.wikipedia.org/wiki/Deviance_information_criterion
#   to be preferred wrt log-likelihood (maybe)
def get_dic(self, clusters, cells):
    # calculate expected_hidden
    expected_hidden = np.zeros((self.K, self.J, self.M))
    sum_of_expected_hidden_two = np.zeros((self.K, self.J, self.J))
    for k in range(self.K):
        sum_of_expected_hidden_two[k] = clusters[k].sum_of_expectation_two()
        expected_hidden[k] = clusters[k].get_expectation()

    def handle_inf(x):
        import sys
        if np.isinf(x):
            return sys.maxsize
        else:
            return x

    # term_1 : -4 E[ log[p(Y | Z, C, Ψ)] ] w.r.t. final posterior values
    # term_2 : 2 log[p(Y|Z,C,Ψ)] where Z, C and Ψ are the modes (maximizing the posteriors)
    res = 0
    for n in range(self.N):
        for k in range(self.K):
            for j in range(self.J):
                res += np.sum(self.pi[n, k] * expected_hidden[k, j, :] *
                              calculate_expectation_of_D(j, self.epsilon_r[n], self.epsilon_s[n], cells[n]))
    term_1 = - res

    res = 0
    for n in range(self.N):
        for m in range(self.M):
            k = int(np.argmax(self.pi[n, :]))
            j = int(np.argmax(expected_hidden[k, :, m]))
            theta = self.epsilon_s[n] / self.epsilon_r[n]
            state = add_noise(j)
            D = handle_inf(math.log(poisson.pmf(cells[n, m], theta * state) + .0000001))
            res += D
    term_2 = 2 * res

    return term_1, 4 * term_1 + term_2

def calculate_expectation_of_D(j, epsilon_r, epsilon_s, cell):
    pass

def add_noise(j):
    if j == 0:
        j = 0.001
    return j

