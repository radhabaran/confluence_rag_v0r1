# app.py

import requests
from requests.auth import HTTPBasicAuth
import json
import os

# Replace these with your details
CONFLUENCE_URL = 'https://radhatrial.atlassian.net/wiki/rest/api'

confluence_api_key = os.environ['CONFLUENCE_KEY']           
API_TOKEN = confluence_api_key
if not API_TOKEN:
    raise ValueError("Missing environment variable: CONFLUENCE_KEY")

api_key = os.environ['OA_API']           
os.environ['OPENAI_API_KEY'] = api_key


USERNAME = 'radhabaran.mohanty@gmail.com'
PAGE_ID = '98319'  # The ID of the page you want to access

# Create the endpoint URL
url = f'{CONFLUENCE_URL}/content/{PAGE_ID}?expand=body.storage'

# Make the request
response = requests.get(url, auth=HTTPBasicAuth(USERNAME, API_TOKEN))

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    # Access the page content
    page_title = data['title']
    page_content = data['body']['storage']['value']
    
    print(f'Title: {page_title}')
    print('Content:', page_content)
else:
    print(f'Failed to retrieve page: {response.status_code} - {response.text}')