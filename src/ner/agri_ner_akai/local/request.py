import requests
import json


class ModelRequest():
    def __init__(self, text, type=None):
        self.text = text
        self.type = type

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)