import requests
import re
import sys

file_id = '1x47MyfkEoLZtVM9MyhEoCQZiOl1z_9RY'
dest = 'CSTBIR/data/CSTBIR_dataset.json'

session = requests.Session()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

url = f'https://drive.google.com/uc?export=download&id={file_id}'
r = session.get(url, headers=headers)
print("Initial status:", r.status_code)

# Parse inputs from the form
form_match = re.search(r'<form[^>]+id="download-form"[^>]*action="([^"]+)"[^>]*>(.*?)</form>', r.text, re.DOTALL)
if not form_match:
    form_match = re.search(r'<form[^>]*action="([^"]+)"[^>]*>(.*?)</form>', r.text, re.DOTALL)

action = form_match.group(1)
form_content = form_match.group(2)

params = {}
for m in re.finditer(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', form_content):
    params[m.group(1)] = m.group(2)

print("Form action:", action)
print("Form params:", params)

# Stream download from action URL with params
res = session.get(action, params=params, headers=headers, stream=True)
print("Download response status:", res.status_code, "Content-Type:", res.headers.get('Content-Type'))
print("Content-Length:", res.headers.get('Content-Length'))

CHUNK_SIZE = 1024 * 1024  # 1 MB chunks
total = 0
with open(dest, 'wb') as f:
    for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            f.write(chunk)
            total += len(chunk)
            if total % (20 * 1024 * 1024) == 0:
                print(f"Downloaded {total / (1024*1024):.1f} MB...")

print(f"Finished {dest}: {total / (1024*1024):.2f} MB")
