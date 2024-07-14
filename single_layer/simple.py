from single_layer_nn import Runner


class SimpleRunner(Runner):
    def __init__(self):
        super(SimpleRunner, self).__init__()
        self.num_epochs = 2000
        self.num_samples = 200

    def validate(self):
        print('SimpleRunner Done')


SimpleRunner().run()
