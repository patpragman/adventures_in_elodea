import os
import requests


def query_davinci(query: str) -> str:
    api_key = os.environ['API_KEY']
    url = "https://api.openai.com/v1/completions"

    data = {
        "model": "text-davinci-003",
        "prompt": query,
        "max_tokens": 2048,
        "temperature": 0
    }

    result = requests.post(url,
                           headers={'Content-Type': 'application/json',
                                    'Authorization': 'Bearer {}'.format(api_key)},
                           json=data)

    return result.json()['choices'][0]['text']


if __name__ == "__main__":
    # simple test here
    print(query_davinci("testing 1, 2, 3!  How do you read me?"))