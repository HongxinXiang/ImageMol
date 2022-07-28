class NoIndexError(Exception):
    def __init__(self, message):
        self.message = message

class NoSuchNameError(Exception):
    def __init__(self, message):
        self.message = message