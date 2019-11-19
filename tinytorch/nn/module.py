class Module(object):

    def __init__(self):
        pass

    def forward(self, *args):
        raise NotImplementedError

    def backward(self):
        pass
