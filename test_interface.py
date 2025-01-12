# test_interface.py

import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import os
from bs4 import BeautifulSoup

# Replace these with your details
CONFLUENCE_URL = 'https://radhatrial.atlassian.net/wiki/rest/api'

# Retrieve API keys from environment variables
API_TOKEN = os.environ.get('CONFLUENCE_KEY')
if not API_TOKEN:
    raise ValueError("Missing environment variable: CONFLUENCE_KEY")

USERNAME = 'radhabaran.mohanty@gmail.com'
PAGE_ID = '98319'  # The ID of the page you want to access

# Create the endpoint URL
url = f'{CONFLUENCE_URL}/content/{PAGE_ID}?expand=body.storage'

# Make the request
try:
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, API_TOKEN))

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        page_content = data['body']['storage']['value']
        
        # Display the title
        st.title(data['title'])
        
        # Render the HTML content
        st.markdown(page_content, unsafe_allow_html=True)

        # Extract and display images
        soup = BeautifulSoup(page_content, 'html.parser')
        images = soup.find_all('ac:image')
        for img in images:
            attachment = img.find('ri:attachment')
            if attachment:
                filename = attachment['ri:filename']
                img_url = f'https://radhatrial.atlassian.net/wiki/download/attachments/{PAGE_ID}/{filename}'
                st.image(img_url, caption=filename)

    else:
        st.error(f'Failed to retrieve page: {response.status_code} - {response.text}')
        
except requests.exceptions.RequestException as e:
    st.error(f'Error occurred while fetching the page: {str(e)}')
except Exception as e:
    st.error(f'An unexpected error occurred: {str(e)}')