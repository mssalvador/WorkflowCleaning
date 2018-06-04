"""
Created on 25 December 2017

@author: sidselsrensen

To be able to create json tables
"""

import json


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        return json.JSONEncoder.default(self, obj)
