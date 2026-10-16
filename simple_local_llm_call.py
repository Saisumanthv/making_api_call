import requests, json

url = "http://localhost:11434/api/generate"
payload = {"model": "llama3.2", "prompt": "Who is Elon Musk?"}

with requests.post(url, json=payload, stream=True) as r:
    for line in r.iter_lines():
        if line:
            chunk = json.loads(line)
            print(chunk.get("response", ""), end="", flush=True)
            if chunk.get("done"):
                print()
                break