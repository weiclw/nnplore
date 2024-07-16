from single_layer_nn import Runner
import torch


# In this test, we train the model to tell if the first dimension of the input
# is positive.
class ZeroRunner(Runner):
    def prepare_data(self):
        assert self.input_size==2, "Input size must be 2"
        # Prepare training data
        self.data = torch.randn(self.num_samples, 2)
        results = [float(x[0] > 0) for x in self.data.tolist()]
        self.target = torch.tensor(results).unsqueeze(1)
        # Prepare testing data
        num_test_samples = 10
        self.test_data = torch.randn(num_test_samples, 2)

    def validate(self):
        super(ZeroRunner, self).validate()
        print('The state of the neuron is ...')
        state_dict = self.model.state_dict()
        for name in state_dict:
            print(f' {name}: {state_dict[name]}')


ZeroRunner().run()
