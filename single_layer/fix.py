from single_layer_nn import Runner
import torch


# This demonstrates that by increasing the dimention of the state, from 2*1 to 2*2,
# the neuron can handle the case that otherwise will fail in previous test, "fail".
class FixRunner(Runner):
    def __init__(self):
        super(FixRunner, self).__init__()
        # Increase the internal state dimention from 2*1 to 2*2.
        # Pay attention to the output probabilities, it should be more than 0.9
        self.output_size = 2

    def prepare_data(self):
        assert self.input_size==2, "Input size must be 2"
        # Prepare training data
        self.data = torch.randn(self.num_samples, 2)
        results = [[float(x[0] > 0), float(x[1] > 0)] for x in self.data.tolist()]
        self.target = torch.tensor(results)
        # Prepare testing data
        num_test_samples = 10
        self.test_data = torch.randn(num_test_samples, 2)

    def validate(self):
        super(FixRunner, self).validate()
        print('The state of the neuron is ...')
        state_dict = self.model.state_dict()
        for name in state_dict:
            print(f' {name}: {state_dict[name]}')


FixRunner().run()
