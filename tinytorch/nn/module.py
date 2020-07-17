class Module(object):

    def __init__(self):
        self.parameters = []

    def forward(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)
