import os
import requests
import sys
import re

def download_file_from_google_drive(file_id, destination):
    session = requests.Session()
    URL = "https://drive.google.com/uc?export=download"

    # Step 1: Initial request
    response = session.get(URL, params={'id': file_id, 'confirm': 't'}, stream=True)
    
    # Check if we got the actual file or an HTML warning page
    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        html = response.text
        # Look for confirm token in form action or links
        match = re.search(r'confirm=([0-9A-Za-z_]+)', html)
        if match:
            confirm_token = match.group(1)
            response = session.get(URL, params={'id': file_id, 'confirm': confirm_token}, stream=True)
        else:
            # Check for form submit url
            form_match = re.search(r'action="([^"]+)"', html)
            if form_match:
                action_url = form_match.group(1)
                response = session.get(action_url, stream=True)

    CHUNK_SIZE = 65536
    total_bytes = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)
                if total_bytes % (20 * 1024 * 1024) < CHUNK_SIZE:
                    print(f"Downloaded {total_bytes / (1024*1024):.1f} MB...")
                    
    print(f"Finished {destination}: {total_bytes} bytes ({total_bytes / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    file_id = sys.argv[1]
    dest = sys.argv[2]
    print(f"Starting download of file_id {file_id} to {dest}...")
    download_file_from_google_drive(file_id, dest)
