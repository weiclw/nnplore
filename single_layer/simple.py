from single_layer_nn import Runner
import torch


class SimpleRunner(Runner):
    def prepare_data(self):
        self.data = torch.randn(self.num_samples, self.input_size)
        self.target = (torch.sum(self.data, dim=1) > 0).float().unsqueeze(1)


SimpleRunner().run()
