import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import sys
from dateutil import parser
import json
import requests
from logging import Logger
from typing import Any, Tuple
import bcrypt

# Import constants from config
try:
    from config.constants import TRANSACTION_TYPES, CATEGORIES
except ImportError:
    TRANSACTION_TYPES = ['Income', 'Expense', 'To Receive', 'To Pay', 'Pending Received', 'Pending Paid']
    CATEGORIES = {
        'Expense': {
            'Food': ['Groceries', 'Dining Out', 'Snacks'],
            'Transportation': ['Fuel', 'Public Transit', 'Maintenance'],
            'Housing': ['Rent', 'Utilities', 'Maintenance'],
            'Entertainment': ['Movies', 'Games', 'Events'],
            'Shopping': ['Clothes', 'Electronics', 'Home Items'],
            'Healthcare': ['Medical', 'Pharmacy', 'Insurance'],
            'Gift': ['Birthday', 'Wedding', 'Holiday', 'Other'],
            'Other': ['Miscellaneous', 'Unspecified']
        },
        'Income': {
            'Salary': ['Regular', 'Bonus', 'Overtime'],
            'Investment': ['Dividends', 'Interest', 'Capital Gains'],
            'Other': ['Gifts', 'Refunds', 'Miscellaneous', 'Pending Received']
        },
        'To Receive': {
            'Pending Income': ['Salary', 'Investment', 'Other']
        },
        'To Pay': {
            'Bills': ['Utilities', 'Rent', 'Other'],
            'Debt': ['Credit Card', 'Loan', 'Other']
        },
        'Pending Received': {
            'Pending Income': ['Salary', 'Investment', 'Other']
        },
        'Pending Paid': {
            'Bills': ['Utilities', 'Rent', 'Other'],
            'Debt': ['Credit Card', 'Loan', 'Other']
        }
    }

# Setup logging
def setup_logging(name: str) -> Logger:
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)

log: Logger = setup_logging("expense_tracker")

# Load environment variables
load_dotenv()
APPS_SCRIPT_URL = os.getenv('APPS_SCRIPT_URL')

# Streamlit configuration
st.set_page_config(layout='wide', page_title="💰 Smart Finance Tracker")

# Modern CSS for polished UI
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f5f7fa;
    }
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .stForm {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 40px auto;
        width: 100%;
        max-width: 800px;
        transition: transform 0.2s ease;
    }
    .stForm:hover {
        transform: translateY(-2px);
    }
    .form-title {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        color: #1a3c34;
        margin-bottom: 30px;
    }
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stDateInput input,
    .stSelectbox select {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 12px;
        font-size: 16px;
        transition: border-color 0.2s ease;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput input:focus,
    .stDateInput input:focus,
    .stSelectbox select:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }
    .stForm button, .stButton button {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        border-radius: 8px;
        padding: 14px 28px;
        border: none;
        font-weight: 600;
        font-size: 18px;
        transition: transform 0.2s ease, background 0.2s ease;
    }
    .stForm button:hover, .stButton button:hover {
        background: linear-gradient(90deg, #059669, #047857);
        transform: translateY(-1px);
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        max-height: 400px;
        overflow-y: auto;
        padding: 15px;
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .chat-box {
        border-radius: 8px;
        padding: 12px 16px;
        max-width: 80%;
        font-size: 15px;
        line-height: 1.5;
    }
    .chat-user {
        background-color: #e6fffa;
        align-self: flex-end;
        border: 1px solid #a7f3d0;
    }
    .chat-assistant {
        background-color: #f1f5f9;
        align-self: flex-start;
        border: 1px solid #e2e8f0;
    }
    .dashboard-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    @media (max-width: 768px) {
        .stForm {
            padding: 20px;
            max-width: 95%;
        }
        .main-container {
            padding: 10px;
        }
        .chat-container {
            max-height: 300px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to parse relative dates
def parse_relative_date(date_str: str) -> datetime.date:
    date_str = date_str.lower().strip()
    today = datetime.now().date()
    if date_str == 'today':
        return today
    elif date_str == 'yesterday':
        return today - timedelta(days=1)
    try:
        parsed_date = parser.parse(date_str).date()
        return parsed_date
    except parser.ParserError:
        log.warning(f"Could not parse date '{date_str}', defaulting to today")
        return today

# Google Sheets service
@st.cache_resource
def get_sheets_service():
    if not APPS_SCRIPT_URL:
        raise ValueError("APPS_SCRIPT_URL not configured in .env")
    return {'url': APPS_SCRIPT_URL}

# Gemini AI model
@st.cache_resource
def get_gemini_model() -> Any:
    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('gemini-1.5-flash')
        log.info("🤖 Gemini AI configured successfully")
        return model
    except Exception as e:
        log.error(f"❌ Failed to configure Gemini AI: {str(e)}")
        raise

try:
    model = get_gemini_model()
    service = get_sheets_service()
except Exception as e:
    log.error(f"❌ Failed to initialize services: {str(e)}")
    sys.exit(1)

# User Management
def initialize_users_sheet(service):
    try:
        response = requests.post(service['url'], json={'action': 'get_users'})
        response.raise_for_status()
        log.info("✨ Users sheet initialized")
    except Exception as e:
        log.error(f"❌ Failed to initialize Users sheet: {str(e)}")
        raise

def get_users_data(service) -> pd.DataFrame:
    try:
        response = requests.post(service['url'], json={'action': 'get_users'})
        response.raise_for_status()
        users = response.json()
        return pd.DataFrame(users, columns=['Username', 'Email', 'Password_Hash', 'Transaction_Sheet_ID'])
    except Exception as e:
        log.error(f"❌ Failed to fetch users data: {str(e)}")
        raise

def add_user_to_sheet(service, username, email, password_hash, transaction_sheet_id):
    try:
        response = requests.post(service['url'], json={
            'action': 'signup',
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'transaction_sheet_id': transaction_sheet_id
        })
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            log.error(f"Failed to add user: {data['error']}")
            st.error(f"Signup failed: {data['error']}")
            return False
        log.info(f"✅ Added user {username}")
        return True
    except Exception as e:
        log.error(f"❌ Failed to add user: {str(e)}")
        st.error(f"Signup failed: {str(e)}")
        return False

def create_new_spreadsheet(service, title: str) -> str:
    try:
        response = requests.post(service['url'], json={'action': 'create_spreadsheet', 'title': title})
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            log.error(f"Failed to create spreadsheet: {data['error']}")
            return None
        return data['spreadsheet_id']
    except Exception as e:
        log.error(f"❌ Failed to create spreadsheet: {str(e)}")
        return None

def signup_form(service):
    with st.form("signup_form"):
        st.markdown("<div class='form-title'>📝 Create a New Account</div>", unsafe_allow_html=True)
        username = st.text_input("👤 Username", placeholder="Enter your username")
        email = st.text_input("📧 Email", placeholder="Enter your email")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
        submit = st.form_submit_button("Sign Up")
        if submit:
            if not all([username, email, password]):
                st.error("All fields are required.")
                return
            users_df = get_users_data(service)
            if username in users_df['Username'].values:
                st.error("Username already exists.")
                return
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            new_sheet_title = f"Finance_Tracker_{username}"
            new_sheet_id = create_new_spreadsheet(service, new_sheet_title)
            if not new_sheet_id:
                st.error("Failed to create transaction sheet.")
                return
            response = requests.post(service['url'], json={'action': 'initialize_sheet', 'sheet_id': new_sheet_id})
            response.raise_for_status()
            success = add_user_to_sheet(service, username, email, hashed_password, new_sheet_id)
            if success:
                st.success("✅ Account created! Please log in.")
            else:
                st.error("Failed to create account. Please try again.")

def login_form(service):
    with st.form("login_form"):
        st.markdown("<div class='form-title'>🔐 Log In</div>", unsafe_allow_html=True)
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
        submit = st.form_submit_button("Log In")
        if submit:
            if not all([username, password]):
                st.error("Username and password are required.")
                return
            users_df = get_users_data(service)
            user_row = users_df[users_df['Username'] == username]
            if not user_row.empty:
                stored_hash = user_row['Password_Hash'].values[0]
                if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['sheet_id'] = user_row['Transaction_Sheet_ID'].values[0]
                    st.success("✅ Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            else:
                st.error("Username not found.")

# Transaction Management
@st.cache_data(ttl=300)
def get_transactions_data(service, sheet_id) -> pd.DataFrame:
    try:
        response = requests.post(service['url'], json={'action': 'get_transactions', 'sheet_id': sheet_id})
        response.raise_for_status()
        values = response.json()
        if len(values) <= 1:
            return pd.DataFrame(columns=['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description'])
        df = pd.DataFrame(values[1:], columns=['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except Exception as e:
        log.error(f"❌ Failed to fetch transactions data: {str(e)}")
        raise

@st.cache_data(ttl=300)
def get_pending_transactions(service, sheet_id) -> pd.DataFrame:
    try:
        response = requests.post(service['url'], json={'action': 'get_pending_transactions', 'sheet_id': sheet_id})
        response.raise_for_status()
        values = response.json()
        if len(values) <= 1:
            return pd.DataFrame(columns=['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status'])
        df = pd.DataFrame(values[1:], columns=['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Due Date'] = pd.to_datetime(df['Due Date'], errors='coerce')
        df = df.dropna(subset=['Amount', 'Type', 'Status'])
        return df
    except Exception as e:
        log.error(f"❌ Failed to fetch pending transactions: {str(e)}")
        raise

def add_transaction_to_sheet(service, sheet_id, date: str, amount: float, trans_type: str, category: str, subcategory: str, description: str) -> bool:
    try:
        response = requests.post(service['url'], json={
            'action': 'add_transaction',
            'sheet_id': sheet_id,
            'date': date,
            'amount': amount,
            'type': trans_type,
            'category': category,
            'subcategory': subcategory or '',
            'description': description
        })
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            log.error(f"Failed to add transaction: {data['error']}")
            return False
        log.info(f"✅ Added transaction: {trans_type} - {amount} ({category})")
        return True
    except Exception as e:
        log.error(f"❌ Failed to add transaction: {str(e)}")
        return False

def add_pending_transaction_to_sheet(service, sheet_id, date: str, amount: float, trans_type: str, category: str, description: str, due_date: str) -> bool:
    try:
        response = requests.post(service['url'], json={
            'action': 'add_pending_transaction',
            'sheet_id': sheet_id,
            'date': date,
            'amount': amount,
            'type': trans_type,
            'category': category,
            'description': description,
            'due_date': due_date
        })
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            log.error(f"Failed to add pending transaction: {data['error']}")
            return False
        log.info(f"✅ Added pending transaction: {trans_type} - {amount} ({category})")
        return True
    except Exception as e:
        log.error(f"❌ Failed to add pending transaction: {str(e)}")
        return False

def update_pending_transaction(service, sheet_id, row_index: int, status: str) -> bool:
    try:
        response = requests.post(service['url'], json={
            'action': 'update_pending_transaction',
            'sheet_id': sheet_id,
            'row_index': row_index,
            'status': status
        })
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            log.error(f"Failed to update pending transaction: {data['error']}")
            return False
        log.info(f"✅ Updated pending transaction at row {row_index} to {status}")
        return True
    except Exception as e:
        log.error(f"❌ Failed to update pending transaction: {str(e)}")
        return False

def process_pending_transaction(service, sheet_id, amount: float, trans_type: str, category: str, description: str) -> bool:
    df = get_pending_transactions(service, sheet_id)
    if df.empty:
        return False
    target_type = 'To Receive' if trans_type == 'Pending Received' else 'To Pay'
    matches = df[(df['Type'] == target_type) & (df['Amount'] == amount) & (df['Category'] == category)]
    if matches.empty:
        return False
    row_index = matches.index[0] + 2  # +2 for header and 1-based indexing
    new_type = 'Income' if trans_type == 'Pending Received' else 'Expense'
    success = add_transaction_to_sheet(service, sheet_id, datetime.now().strftime('%Y-%m-%d'), 
                                      amount, new_type, category, 'Pending', description)
    if success:
        update_pending_transaction(service, sheet_id, row_index, 'Completed')
    return success

def parse_transaction_input(model, input_text: str) -> dict:
    prompt = f"""
    Parse the following transaction input into a structured JSON format:
    '{input_text}'
    Return a dictionary with:
    - date: ISO format (YYYY-MM-DD) or relative date ('today', 'yesterday')
    - amount: float
    - type: one of {TRANSACTION_TYPES}
    - category: appropriate category (e.g., from {list(CATEGORIES.keys())})
    - subcategory: appropriate subcategory (optional, can be empty)
    - description: brief description
    - due_date: ISO format (YYYY-MM-DD) or relative date (for pending transactions)
    If the input is ambiguous, make reasonable assumptions or return empty strings for unclear fields.
    """
    try:
        response = model.generate_content(prompt)
        parsed_data = json.loads(response.text.strip('```json\n').strip('```'))
        return parsed_data
    except Exception as e:
        log.error(f"❌ Failed to parse input: {str(e)}")
        return {}

def validate_transaction(parsed_data: dict) -> Tuple[bool, dict]:
    try:
        if not parsed_data.get('amount') or float(parsed_data['amount']) <= 0:
            return False, {"error": "Invalid amount"}
        trans_type = parsed_data.get('type')
        if trans_type not in TRANSACTION_TYPES:
            return False, {"error": f"Invalid type. Must be one of {TRANSACTION_TYPES}"}
        category = parsed_data.get('category')
        if not category:
            return False, {"error": "Category is required"}
        if trans_type in ['To Pay', 'To Receive'] and not parsed_data.get('due_date'):
            return False, {"error": "Due date required for pending transactions"}
        if not parsed_data.get('description'):
            return False, {"error": "Description is required"}
        return True, parsed_data
    except Exception as e:
        log.error(f"❌ Validation error: {str(e)}")
        return False, {"error": str(e)}

# Date filtering for analytics
def get_date_filters(key="unique_global_filter"):
    initialize_filters()
    st.sidebar.subheader("📅 Date Filter")
    df = get_transactions_data(service, st.session_state['sheet_id'])
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        min_date = df['Date'].min()
        max_date = df['Date'].max()
    else:
        min_date = max_date = datetime.now().date()
    st.session_state.global_filter_type = st.sidebar.radio(
        "Select Time Period",
        ["All Time", "Year", "Month", "Custom Range"],
        key=key
    )
    if st.session_state.global_filter_type == "Year":
        st.session_state.global_selected_year = st.sidebar.selectbox(
            "Select Year",
            sorted(df['Date'].dt.year.unique(), reverse=True) if not df.empty else [datetime.now().year],
            key=f"{key}_year"
        )
        start_date = datetime(st.session_state.global_selected_year, 1, 1)
        end_date = datetime(st.session_state.global_selected_year, 12, 31)
    elif st.session_state.global_filter_type == "Month":
        st.session_state.global_selected_year = st.sidebar.selectbox(
            "Select Year",
            sorted(df['Date'].dt.year.unique(), reverse=True) if not df.empty else [datetime.now().year],
            key=f"{key}_month_year"
        )
        st.session_state.global_selected_month = st.sidebar.selectbox(
            "Select Month",
            range(1, 13),
            format_func=lambda x: datetime(2000, x, 1).strftime('%B'),
            key=f"{key}_month"
        )
        start_date = datetime(st.session_state.global_selected_year, st.session_state.global_selected_month, 1)
        end_date = (datetime(st.session_state.global_selected_year, st.session_state.global_selected_month + 1, 1) 
                   if st.session_state.global_selected_month < 12 
                   else datetime(st.session_state.global_selected_year + 1, 1, 1)) - timedelta(days=1)
    elif st.session_state.global_filter_type == "Custom Range":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.session_state.global_start_date = st.date_input(
                "Start Date", 
                min_date,
                key=f"{key}_start_date"
            )
        with col2:
            st.session_state.global_end_date = st.date_input(
                "End Date", 
                max_date,
                key=f"{key}_end_date"
            )
        start_date = datetime.combine(st.session_state.global_start_date, datetime.min.time())
        end_date = datetime.combine(st.session_state.global_end_date, datetime.max.time())
    else:
        start_date = min_date
        end_date = max_date
    return start_date, end_date

def initialize_filters():
    if 'global_filter_type' not in st.session_state:
        st.session_state.global_filter_type = "All Time"
    if 'global_selected_year' not in st.session_state:
        st.session_state.global_selected_year = datetime.now().year
    if 'global_selected_month' not in st.session_state:
        st.session_state.global_selected_month = datetime.now().month
    if 'global_start_date' not in st.session_state:
        st.session_state.global_start_date = datetime.now().date() - timedelta(days=30)
    if 'global_end_date' not in st.session_state:
        st.session_state.global_end_date = datetime.now().date()

def filter_dataframe(df, start_date, end_date):
    if df.empty:
        return df
    df['Date'] = pd.to_datetime(df['Date'])
    return df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Analytics functions
def show_overview_analytics(df, start_date, end_date):
    st.subheader("📈 Financial Overview")
    if df.empty:
        st.info("No transactions found for the selected period.")
        return
    df = filter_dataframe(df, start_date, end_date)
    total_income = df[df['Type'] == 'Income']['Amount'].sum()
    total_expense = df[df['Type'] == 'Expense']['Amount'].sum()
    net_savings = total_income - total_expense
    saving_rate = (net_savings / total_income * 100) if total_income > 0 else 0
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Income", f"Rs. {total_income:,.2f}", delta=None)
    with col2:
        st.metric("Total Expenses", f"Rs. {total_expense:,.2f}", delta=None)
    with col3:
        st.metric("Net Savings", f"Rs. {net_savings:,.2f}", 
                 delta=f"Rs. {net_savings:,.2f}",
                 delta_color="normal" if net_savings >= 0 else "inverse")
    with col4:
        st.metric("Saving Rate", f"{saving_rate:.1f}%", delta=None)
    st.subheader("Monthly Summary")
    monthly_summary = df.groupby([df['Date'].dt.strftime('%Y-%m'), 'Type'])['Amount'].sum().unstack(fill_value=0)
    monthly_summary['Net'] = monthly_summary.get('Income', 0) - monthly_summary.get('Expense', 0)
    fig_monthly = px.bar(monthly_summary, 
                        title='Monthly Income vs Expenses',
                        barmode='group',
                        labels={'value': 'Amount (Rs.)', 'index': 'Month'})
    st.plotly_chart(fig_monthly)
    st.dataframe(
        monthly_summary.style.format({
            'Income': 'Rs. {:,.2f}',
            'Expense': 'Rs. {:,.2f}',
            'Net': 'Rs. {:,.2f}'
        }),
        use_container_width=True,
        height=200
    )

def show_income_analytics(df, start_date, end_date):
    st.subheader("💰 Income Analytics")
    income_df = df[df['Type'] == 'Income'].copy()
    if income_df.empty:
        st.info("No income transactions found for the selected period.")
        return
    df = filter_dataframe(df, start_date, end_date)
    monthly_income = income_df.groupby(income_df['Date'].dt.strftime('%Y-%m'))['Amount'].sum()
    fig_monthly = px.bar(monthly_income, 
                        title='Monthly Income Trend',
                        labels={'value': 'Amount (Rs.)', 'index': 'Month'})
    st.plotly_chart(fig_monthly)
    col1, col2 = st.columns(2)
    with col1:
        fig_category = px.pie(income_df, 
                            values='Amount', 
                            names='Category',
                            title='Income by Category')
        st.plotly_chart(fig_category)
    with col2:
        st.subheader("Top Income Sources")
        top_sources = income_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        st.dataframe(
            top_sources.to_frame().style.format({
                'Amount': 'Rs. {:,.2f}'
            }),
            use_container_width=True,
            height=300
        )

def show_expense_analytics(df, start_date, end_date):
    st.subheader("💸 Expense Analytics")
    expense_df = df[df['Type'] == 'Expense'].copy()
    if expense_df.empty:
        st.info("No expense transactions found for the selected period.")
        return
    df = filter_dataframe(df, start_date, end_date)
    monthly_expense = expense_df.groupby(expense_df['Date'].dt.strftime('%Y-%m'))['Amount'].sum()
    fig_monthly = px.bar(monthly_expense, 
                        title='Monthly Expense Trend',
                        labels={'value': 'Amount (Rs.)', 'index': 'Month'})
    st.plotly_chart(fig_monthly)
    col1, col2 = st.columns(2)
    with col1:
        fig_category = px.pie(expense_df, 
                            values='Amount', 
                            names='Category',
                            title='Expenses by Category')
        st.plotly_chart(fig_category)
    with col2:
        st.subheader("Top Expense Categories")
        top_expenses = expense_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        st.dataframe(
            top_expenses.to_frame().style.format({
                'Amount': 'Rs. {:,.2f}'
            }),
            use_container_width=True,
            height=300
        )

def show_pending_transactions(service, sheet_id):
    st.subheader("📋 Pending Transactions")
    try:
        df = get_pending_transactions(service, sheet_id)
        if df.empty:
            st.info("No pending transactions found.")
            return
        to_receive = df[df['Type'] == 'To Receive'].copy()
        to_pay = df[df['Type'] == 'To Pay'].copy()
        col1, col2, col3 = st.columns(3)
        with col1:
            total_to_receive = to_receive['Amount'].sum()
            st.metric("To Receive", f"Rs. {total_to_receive:,.2f}")
        with col2:
            total_to_pay = to_pay['Amount'].sum()
            st.metric("To Pay", f"Rs. {total_to_pay:,.2f}")
        with col3:
            net_pending = total_to_receive - total_to_pay
            st.metric("Net Pending", 
                     f"Rs. {net_pending:,.2f}",
                     delta=f"Rs. {net_pending:,.2f}",
                     delta_color="normal" if net_pending >= 0 else "inverse")
        st.divider()
        tab1, tab2 = st.tabs(["💰 To Receive", "💸 To Pay"])
        with tab1:
            if to_receive.empty:
                st.info("No pending receipts.")
            else:
                st.write("### Pending Receipts")
                to_receive['Amount'] = to_receive['Amount'].apply(lambda x: f"Rs. {x:,.2f}")
                to_receive['Date'] = to_receive['Date'].dt.strftime('%Y-%m-%d')
                to_receive['Due Date'] = to_receive['Due Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(
                    to_receive[['Date', 'Amount', 'Category', 'Description', 'Due Date']],
                    use_container_width=True
                )
        with tab2:
            if to_pay.empty:
                st.info("No pending payments.")
            else:
                st.write("### Pending Payments")
                to_pay['Amount'] = to_pay['Amount'].apply(lambda x: f"Rs. {x:,.2f}")
                to_pay['Date'] = to_pay['Date'].dt.strftime('%Y-%m-%d')
                to_pay['Due Date'] = to_pay['Due Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(
                    to_pay[['Date', 'Amount', 'Category', 'Description', 'Due Date']],
                    use_container_width=True
                )
    except Exception as e:
        log.error(f"Error displaying pending transactions: {str(e)}")
        st.error("Failed to load pending transactions.")

def show_analytics(service):
    try:
        st.title("📊 Financial Analytics")
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                get_transactions_data.clear()
                get_pending_transactions.clear()
                st.rerun()
        start_date, end_date = get_date_filters(key="global_analytics_filter")
        df = get_transactions_data(service, st.session_state['sheet_id'])
        if not df.empty:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            filtered_df = filter_dataframe(df, start_date, end_date)
        else:
            filtered_df = df
        st.caption(f"Showing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Income Analytics", "Expense Analytics", "Pending Transactions"])
        with tab1:
            show_overview_analytics(filtered_df, start_date, end_date)
        with tab2:
            show_income_analytics(filtered_df, start_date, end_date)
        with tab3:
            show_expense_analytics(filtered_df, start_date, end_date)
        with tab4:
            show_pending_transactions(service, st.session_state['sheet_id'])
        log.info("📊 Analytics visualizations generated successfully")
    except Exception as e:
        log.error(f"❌ Failed to generate analytics: {str(e)}")
        st.error("Failed to generate analytics.")

def render_dashboard(service):
    st.markdown(f"""
        <div style="background: linear-gradient(90deg, #10b981, #059669); padding: 20px; border-radius: 12px; color: white; text-align: center; font-size: 22px; margin-bottom: 20px;">
            👋 Welcome, <b>{st.session_state['username']}</b>!  
            <br>Manage your finances smarter with AI-powered tracking.
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"📊 [View Google Sheet](https://docs.google.com/spreadsheets/d/{st.session_state['sheet_id']})")

    if st.button("🚪 Logout"):
        st.session_state['authenticated'] = False
        st.session_state.pop('username', None)
        st.session_state.pop('sheet_id', None)
        st.session_state.pop('show_quick_add', None)
        st.session_state.pop('quick_add_prefill', None)
        st.rerun()

    # Sidebar toggle for Quick Add
    st.sidebar.subheader("⚙️ Settings")
    if "show_quick_add" not in st.session_state:
        st.session_state.show_quick_add = False
    if "quick_add_prefill" not in st.session_state:
        st.session_state.quick_add_prefill = {}
    manual_toggle = st.sidebar.checkbox(
        "Show Quick Add Form",
        value=st.session_state.show_quick_add
    )
    st.session_state.show_quick_add = manual_toggle

    col1, col2 = st.columns([2, 1], gap="large")

    # Chatbot
    with col1:
        st.subheader("💬 Smart Finance Chatbot", anchor=False)
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            role_class = "chat-user" if message["role"] == "user" else "chat-assistant"
            st.markdown(f"<div class='chat-box {role_class}'>{message['content']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if prompt := st.chat_input("Enter transaction (e.g., 'Spent 50 on groceries yesterday')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            parsed_data = parse_transaction_input(model, prompt)
            is_valid, result = validate_transaction(parsed_data)
            if not is_valid:
                st.session_state.quick_add_prefill = {
                    "date": parsed_data.get("date", datetime.now().date()),
                    "amount": parsed_data.get("amount", 0.0),
                    "type": parsed_data.get("type", TRANSACTION_TYPES[0]),
                    "category": parsed_data.get("category", ""),
                    "subcategory": parsed_data.get("subcategory", ""),
                    "desc": parsed_data.get("description", ""),
                    "due_date": parsed_data.get("due_date", datetime.now().date() + timedelta(days=7))
                }
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ I couldn’t fully parse that ({result['error']}). I’ve opened Quick Add and pre-filled what I could."
                })
                st.session_state.show_quick_add = True
                st.rerun()

            with st.form(key="transaction_form"):
                st.markdown("<div class='form-title'>Confirm Transaction Details</div>", unsafe_allow_html=True)
                try:
                    date_value = parse_relative_date(parsed_data.get('date', 'today'))
                except Exception as e:
                    log.error(f"Failed to parse date: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": "Error: Invalid date format. Using today."})
                    date_value = datetime.now().date()
                date = st.date_input("Date", value=date_value)
                amount = st.number_input("Amount", value=float(parsed_data.get('amount', 0)), min_value=0.0)
                trans_type = st.selectbox("Type", TRANSACTION_TYPES, index=TRANSACTION_TYPES.index(parsed_data.get('type', 'Expense')))
                category = st.text_input("Category", value=parsed_data.get('category', ''))
                subcategory = st.text_input("Subcategory", value=parsed_data.get('subcategory', '')) if trans_type not in ['To Pay', 'To Receive', 'Pending Received', 'Pending Paid'] else None
                description = st.text_input("Description", value=parsed_data.get('description', ''))
                if trans_type in ['To Pay', 'To Receive']:
                    try:
                        due_date_value = parse_relative_date(parsed_data.get('due_date', 'today'))
                    except Exception as e:
                        log.error(f"Failed to parse due_date: {str(e)}")
                        st.session_state.messages.append({"role": "assistant", "content": "Error: Invalid due date format. Using today."})
                        due_date_value = datetime.now().date()
                    due_date = st.date_input("Due Date", value=due_date_value)
                else:
                    due_date = None
                submit = st.form_submit_button("Save Transaction")
                if submit:
                    if not category:
                        st.error("Category is required.")
                        st.rerun()
                    if not description:
                        st.error("Description is required.")
                        st.rerun()
                    if trans_type in ['Pending Received', 'Pending Paid']:
                        success = process_pending_transaction(service, st.session_state['sheet_id'], amount, trans_type, category, description)
                        if success:
                            st.session_state.messages.append({"role": "assistant", "content": f"Processed {trans_type}: Rs. {amount} ({category})"})
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": "No matching pending transaction found."})
                    elif trans_type in ['To Pay', 'To Receive']:
                        success = add_pending_transaction_to_sheet(service, st.session_state['sheet_id'], 
                                                                 date.strftime('%Y-%m-%d'), amount, trans_type, category, description, 
                                                                 due_date.strftime('%Y-%m-%d') if due_date else datetime.now().strftime('%Y-%m-%d'))
                        if success:
                            st.session_state.messages.append({"role": "assistant", "content": f"Added pending {trans_type}: Rs. {amount} ({category})"})
                    else:
                        success = add_transaction_to_sheet(service, st.session_state['sheet_id'], 
                                                         date.strftime('%Y-%m-%d'), amount, trans_type, category, subcategory, description)
                        if success:
                            st.session_state.messages.append({"role": "assistant", "content": f"Added {trans_type}: Rs. {amount} ({category}/{subcategory or 'None'})"})
                    if success:
                        get_transactions_data.clear()
                        get_pending_transactions.clear()
                        st.session_state.quick_add_prefill = {}  # Reset prefill
                        st.session_state.show_quick_add = False  # Auto-close Quick Add
                        st.rerun()
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Failed to save transaction."})
            st.rerun()

    # Quick Add Form
    if st.session_state.show_quick_add:
        with col2:
            st.subheader("➕ Quick Add Transaction", anchor=False)
            with st.form("quick_add_form"):
                pf = st.session_state.quick_add_prefill
                q_date = st.date_input("📅 Date", value=pf.get("date", datetime.now().date()))
                q_amount = st.number_input("💵 Amount", min_value=0.0, step=0.01, value=pf.get("amount", 0.0))
                q_type = st.selectbox("📂 Type", TRANSACTION_TYPES, index=TRANSACTION_TYPES.index(pf.get("type", TRANSACTION_TYPES[0])))
                q_category = st.text_input("🏷️ Category", value=pf.get("category", ""), placeholder="e.g., Food, Bills")
                q_subcategory = st.text_input("🔖 Subcategory", value=pf.get("subcategory", ""), placeholder="e.g., Groceries, Utilities") \
                                if q_type not in ['To Pay', 'To Receive', 'Pending Received', 'Pending Paid'] else None
                q_desc = st.text_input("📝 Description", value=pf.get("desc", ""), placeholder="e.g., Grocery shopping")
                q_due_date = st.date_input("📅 Due Date", value=pf.get("due_date", datetime.now().date() + timedelta(days=7))) \
                             if q_type in ['To Pay', 'To Receive'] else None
                q_submit = st.form_submit_button("✅ Add Transaction")
                if q_submit:
                    if not q_category:
                        st.error("Category is required.")
                        st.rerun()
                    if not q_desc:
                        st.error("Description is required.")
                        st.rerun()
                    if q_type in ['Pending Received', 'Pending Paid']:
                        success = process_pending_transaction(service, st.session_state['sheet_id'], q_amount, q_type, q_category, q_desc)
                        if success:
                            st.success(f"Processed {q_type}: Rs. {q_amount} ({q_category})")
                        else:
                            st.error("No matching pending transaction found.")
                    elif q_type in ['To Pay', 'To Receive']:
                        success = add_pending_transaction_to_sheet(service, st.session_state['sheet_id'], 
                                                                 q_date.strftime('%Y-%m-%d'), q_amount, q_type, q_category, q_desc, 
                                                                 q_due_date.strftime('%Y-%m-%d') if q_due_date else datetime.now().strftime('%Y-%m-%d'))
                        if success:
                            st.success(f"Pending {q_type} added: Rs. {q_amount} ({q_category})")
                    else:
                        success = add_transaction_to_sheet(service, st.session_state['sheet_id'], 
                                                         q_date.strftime('%Y-%m-%d'), q_amount, q_type, q_category, q_subcategory or '', q_desc)
                        if success:
                            st.success(f"{q_type} added: Rs. {q_amount} ({q_category}/{q_subcategory or 'None'})")
                    if success:
                        get_transactions_data.clear()
                        get_pending_transactions.clear()
                        st.session_state.quick_add_prefill = {}  # Reset prefill
                        st.session_state.show_quick_add = False  # Auto-close
                        st.rerun()
                    else:
                        st.error("Failed to save transaction.")

def main():
    if not APPS_SCRIPT_URL:
        st.error("Apps Script URL not configured.")
        return

    initialize_users_sheet(service)
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        st.title("💰 Smart Finance Tracker")
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            login_form(service)
        with col2:
            signup_form(service)
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Dashboard", "Analytics"])
    if page == "Dashboard":
        render_dashboard(service)
    else:
        show_analytics(service)

if __name__ == "__main__":
    main()