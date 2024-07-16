from single_layer_nn import Runner
import torch


# Train the model to label points (x,y) whose x and y sum up positively,
# and use a few extra data points to validate the model.
class ValidatorRunner(Runner):
    def prepare_data(self):
        self.data = torch.randn(self.num_samples, self.input_size)
        self.target = (torch.sum(self.data, dim=1) > 0).float().unsqueeze(1)
        num_test_samples = 10
        self.test_data = torch.randn(num_test_samples, self.input_size)

    def validate(self):
        super(ValidatorRunner, self).validate()
        print('The state of the neuron is ...')
        state_dict = self.model.state_dict()
        for name in state_dict:
            print(f' {name}: {state_dict[name]}')


ValidatorRunner().run()
