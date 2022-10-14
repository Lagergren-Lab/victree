from variational_distributions.variational_distribution import VariationalDistribution


class CopyTree():

    def __init__(self, copy_number_model: VariationalDistribution):
        self.copy_number_model = copy_number_model
        self.p = p
        self.q = q

    def run(self, n_iter):

        self.init_variational_variables()

        for i in range(n_iter):
            self.step()

    def step(self):
        self.update_T()
        self.update_C()
        self.update_z()
        self.update_mu()
        self.update_sigma()
        self.update_epsilon()
        self.update_gamma()

    def update_T(self):
        pass

    def update_C(self):
        self.copy_number_model.update()

    def update_z(self):
        self.z_model.update()


