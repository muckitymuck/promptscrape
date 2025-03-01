import requests
import sys
import json
from datetime import datetime

def get_jina_response(target_url):
    jina_url = f'https://r.jina.ai/{target_url}'
    headers = {
        'Authorization': 'Bearer jina_b0ea6d2a2aec43ff9ee2af7ab91ad5ba5AuI-cX7qMEqxatplJGTNANvpki8'
    }
    
    response = requests.get(jina_url, headers=headers)
    return response.text

def save_response(content):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'jina_response_{timestamp}.json'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    return filename

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python jina.py <target_url>")
        sys.exit(1)
        
    target_url = sys.argv[1]
    response_content = get_jina_response(target_url)
    saved_file = save_response(response_content)
    print(f"Response saved to: {saved_file}")
    print("\nResponse content:")
    print(response_content)
