class Report():

    def __init__(self, console=True):
        self.console = console
        self.matched = []
        self.total = []
        self.percent = []
        self.log = []

    def add(self, log):
        self.log += log

    def information(self, log):
        if self.console: print(log)
        self.log.append(log)

    def finish(self, total, matched, percent):
        if self.console: 
            print("\nAccuracy: %s/%s (%f)" % (matched, total, percent))
            print("\n=======================\n")
        self.total.append(total)
        self.matched.append(matched)
        self.percent.append(percent)