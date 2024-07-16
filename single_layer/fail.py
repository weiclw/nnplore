from single_layer_nn import Runner
import torch


# The training should NOT yield a satifactory result, as a single neuron
# is Not enough to filter the input data.
# The condition (x[0] > 0 and x[1] > 0) calls for multiple neurons to
# have a good result.
class FailRunner(Runner):
    def prepare_data(self):
        assert self.input_size==2, "Input size must be 2"
        # Prepare training data
        self.data = torch.randn(self.num_samples, 2)
        results = [float(x[0] > 0 and x[1] > 0) for x in self.data.tolist()]
        self.target = torch.tensor(results).unsqueeze(1)
        # Prepare testing data
        num_test_samples = 10
        self.test_data = torch.randn(num_test_samples, 2)

    def validate(self):
        super(FailRunner, self).validate()
        print('The state of the neuron is ...')
        state_dict = self.model.state_dict()
        for name in state_dict:
            print(f' {name}: {state_dict[name]}')


FailRunner().run()
