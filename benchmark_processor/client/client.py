import requests
import json
from typing import List

class InferenceLLM:
    def __init__(self, route) -> None:
        self.route = route

    def __call__(self, prompts) -> List[str]:
        payload = {'prompts': prompts, 'max_tokens':800}
        headers = {'Content-type': 'application/json'}
        response = requests.post(url="http://127.0.0.1:5000/inference", data=json.dumps(payload), headers=headers)

        res = response.json()
        return res['outputs']