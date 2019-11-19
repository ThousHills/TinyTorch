from autograd import creator


if __name__ == "__main__":
    x = creator.ones((2, 2))
    y = x + 2
    z = y * y * 3
    out = z.mean()
    out.backward()
    print(x.grad)
