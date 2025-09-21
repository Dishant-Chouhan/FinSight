import logging
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import Any

load_dotenv()
APPS_SCRIPT_URL = os.getenv('APPS_SCRIPT_URL')

@st.cache_resource
def get_sheets_service():
    """Return a service object for Apps Script (just the URL for HTTP requests)"""
    if not APPS_SCRIPT_URL:
        raise ValueError("APPS_SCRIPT_URL not configured in .env")
    return {'url': APPS_SCRIPT_URL}

def create_new_spreadsheet(service, title):
    """Create a new spreadsheet via Apps Script"""
    try:
        response = requests.post(service['url'], json={'action': 'create_spreadsheet', 'title': title})
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            logging.error(f"Failed to create spreadsheet: {data['error']}")
            return None
        return data['spreadsheet_id']
    except Exception as e:
        logging.error(f"Failed to create spreadsheet: {str(e)}")
        return None