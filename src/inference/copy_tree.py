from numpy import infty
from variational_distributions.variational_distribution import VariationalDistribution


class CopyTree():

    def __init__(self, p, q, copy_number_model: VariationalDistribution):
        self.copy_number_model = copy_number_model
        self.z_model: VariationalDistribution = VariationalDistribution()
        self.p = p
        self.q = q

        # counts the number of steps performed
        self.it_counter = 0
        self.elbo = -infty

        # TODO: set as parameters
        self.tol = 1e-10
        self.max_close_runs = 10

    def run(self, n_iter):

        # counts the number of irrelevant updates
        close_runs = 0

        self.init_variational_variables()

        for _ in range(n_iter):
            # do the updates
            self.step()

            new_elbo = self.compute_elbo()
            if abs(new_elbo - self.elbo) < self.tol:
                close_runs += 1
                if close_runs > self.max_close_runs:
                    break
            elif new_elbo < self.elbo:
                # elbo should only increase
                raise ValueError("Elbo is decreasing")
            elif new_elbo > 0:
                # elbo must be negative
                raise ValueError("Elbo is non-negative")
            else:
                close_runs = 0



    def compute_elbo(self) -> float:
        # TODO: elbo could also be a custom object, containing the main elbo parts separately
        #   so we can monitor all components of the elbo (variational and model part)

        # ...quite costly operation...
        return -1000.

    def step(self):
        self.update_T()
        self.update_C()
        self.update_z()
        self.update_mu()
        self.update_sigma()
        self.update_epsilon()
        self.update_gamma()

        self.it_counter += 1

    def init_variational_variables(self):
        # random initialization of variational parameters

        pass

    def update_T(self):
        pass

    def update_C(self):
        self.copy_number_model.update()

    def update_z(self):
        self.z_model.update()

    def update_mu(self):
        pass

    def update_sigma(self):
        pass

    def update_epsilon(self):
        pass

    def update_gamma(self):
        pass
