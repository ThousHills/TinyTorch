class Module(object):

    def forward(self, *args):
        raise NotImplementedError

    def backward(self):
        pass
