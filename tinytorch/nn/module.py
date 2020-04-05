class Module(object):

    def forward(self, *args):
        raise NotImplementedError

    def backward(self):
        pass

    def __call__(self, *args):
        return self.forward(*args)
