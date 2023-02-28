class MyResultWriter:
    def __init__(self):
        self.log = []

    def write_row(self, epinfo):
        self.log.append(epinfo)

    def close(self):
        pass

    def clear(self):
        del self.log
        self.log = []
