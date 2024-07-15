from single_layer_nn import Runner
import torch


class SimpleRunner(Runner):
    def __init__(self):
        super(SimpleRunner, self).__init__()
        num_test_samples = 10
        self.test_data = torch.randn(num_test_samples, self.input_size)


SimpleRunner().run()
