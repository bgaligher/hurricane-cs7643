"""
Original file from: DL Course A4 codebase

Edits:

References:

"""

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)