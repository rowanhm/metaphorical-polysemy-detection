class InfiniteLoader:

    def __init__(self, loader):

        self.loader = loader
        self.iter = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        # Call next
        data = next(self.iter, None)

        if data is None:
            self.reset_iter()
            data = next(self.iter)

        return data

    def reset_iter(self):
        self.iter = iter(self.loader)
