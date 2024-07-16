from single_layer_nn import Runner
import torch


class ValidatorRunner(Runner):
    def __init__(self):
        super(ValidatorRunner, self).__init__()
        num_test_samples = 10
        self.test_data = torch.randn(num_test_samples, self.input_size)

    def validate(self):
        super(ValidatorRunner, self).validate()
        print('The state of the neuron is ...')
        state_dict = self.model.state_dict()
        for name in state_dict:
            print(f' {name}: {state_dict[name]}')


ValidatorRunner().run()
