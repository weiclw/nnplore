from single_layer_nn import Runner


class SimpleRunner(Runner):
    def __init__(self):
        super(SimpleRunner, self).__init__()

    def validate(self):
        print('Done with validating')


SimpleRunner().run()
