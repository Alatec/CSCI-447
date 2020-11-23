

class UntrainedUtilityError(Exception):

    def __init__ (self, utility=None):
        self.utility = utility
        if self.utility is None:
            super().__init__("A utility is trying to fit data without being trained")
        else:
            super().__init__(f"A {self.utility} utility is trying to fit data without being trained")


class SingleMultiError(Exception):

    def __init__ (self, multi=False):
        self.multi = multi
        if self.multi:
            super().__init__("This object was initialized as Multi-column and is calling Single-column functions")
        else:
            super().__init__("This object was initialized as Single-column and is calling Multi-column functions")

class TrainTestColumnMismatch(Exception):

    def __init__ (self):
        super().__init__("The number of columns in the train and test sets are inconsistent")
       
class UndefinedPerformaceMethod(Exception):

    def __init__ (self, multi=False):
        if multi:
            super().__init__("Method needs to be defined if there is more than one class")
        else:
            super().__init__("Method makes no since if there is only one class")