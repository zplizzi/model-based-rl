from papers.muzero.network import Network


class ModelStorage(object):
    """A place to store and access the most recent network checkpoint.

    Needs to be thread-safe."""
    def __init__(self):
        self.network: Network = None
        self.i = 0

    def latest_network(self) -> Network:
        assert self.network is not None
        return self.network, self.i

    def save_network(self, step: int, network):
        self.i = step
        # This is actuall a state dict now
        self.network = network
