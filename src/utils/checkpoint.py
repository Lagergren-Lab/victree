from variational_distributions.joint_dists import JointDist


class Checkpoint:

    def __init__(self, out_dir: str):
        self.checkpoint_dict = {}
        self.out_dir = out_dir

    def initialize(self, joint_q: JointDist):
        pass
    def reset(self):
        pass

    def append(self, dist_name: str, params: dict):
        """
        Adds params to list of params under the specified distribution label
        Parameters
        ----------
        dist_name: str, name of distribution obj.__name__
        params: dict, for each param, a key with its name and a numpy array
        """
        pass

    def write(self):
        pass
