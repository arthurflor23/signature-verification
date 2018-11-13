class Report():

    def __init__(self, console=True):
        self.console = console
        self.matched = []
        self.total = []
        self.percent = []
        self.log = []

    def information(self, log):
        if self.console: print(log)
        self.log.append(log)

    def finish(self, result):
        self.information("Accuracy: %s/%s (%f)\n" % (result[3], result[2], result[4]))
        if self.console: print("=======================")

        self.log += result[0]
        self.total.append(result[2])
        self.matched.append(result[3])
        self.percent.append(result[4])