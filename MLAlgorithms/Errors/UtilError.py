

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