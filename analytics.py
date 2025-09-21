from logging import Logger
from typing import Any
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import sys
import requests
import bcrypt
from services.google_sheets import get_sheets_service, create_new_spreadsheet
from utils.logging_utils import setup_logging

log: Logger = setup_logging("expense_tracker_analytics")

st.set_page_config(layout='wide')
load_dotenv()
APPS_SCRIPT_URL = os.getenv('APPS_SCRIPT_URL')

try:
    service = get_sheets_service()
except Exception as e:
    log.error(f"❌ Failed to initialize services: {str(e)}")
    sys.exit(1)

def initialize_users_sheet(service):
    """Initialize Users sheet (handled by Apps Script)"""
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
            return False
        log.info(f"✅ Added user {username}")
        return True
    except Exception as e:
        log.error(f"❌ Failed to add user: {str(e)}")
        return False

def signup_form(service):
    with st.form("signup_form"):
        st.write("Create a new account")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign Up")
        if submit:
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
                st.success("Account created! Please log in.")

def login_form(service):
    with st.form("login_form"):
        st.write("Log in to your account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Log In")
        if submit:
            users_df = get_users_data(service)
            user_row = users_df[users_df['Username'] == username]
            if not user_row.empty:
                stored_hash = user_row['Password_Hash'].values[0]
                if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['sheet_id'] = user_row['Transaction_Sheet_ID'].values[0]
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            else:
                st.error("Username not found.")

@st.cache_data(ttl=300)
def get_transactions_data(_service, sheet_id) -> pd.DataFrame:
    try:
        log.debug(f"Fetching transactions data from sheet {sheet_id}")
        response = requests.post(_service['url'], json={'action': 'get_transactions', 'sheet_id': sheet_id})
        response.raise_for_status()
        values = response.json()
        if len(values) <= 1:
            log.warning("No transaction data found in sheet")
            return pd.DataFrame(columns=['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description'])
        log.info(f"📈 Retrieved {len(values)-1} transaction records")
        return pd.DataFrame(values[1:], columns=['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description'])
    except Exception as e:
        log.error(f"❌ Failed to fetch transactions data: {str(e)}")
        raise

@st.cache_data(ttl=300)
def get_pending_transactions(_service, sheet_id) -> pd.DataFrame:
    try:
        log.debug(f"Fetching pending transactions from sheet {sheet_id}")
        response = requests.post(_service['url'], json={'action': 'get_pending_transactions', 'sheet_id': sheet_id})
        response.raise_for_status()
        values = response.json()
        if len(values) <= 1:
            log.warning("No data found in Pending sheet")
            return pd.DataFrame(columns=['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status'])
        df = pd.DataFrame(values[1:], columns=['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Due Date'] = pd.to_datetime(df['Due Date'], errors='coerce')
        df = df.dropna(subset=['Amount', 'Type', 'Status'])
        log.info(f"📊 Retrieved {len(df)} pending transactions")
        return df
    except Exception as e:
        log.error(f"❌ Failed to fetch pending transactions: {str(e)}")
        raise

def initialize_filters():
    """Initialize filter values in session state"""
    if 'global_filter_type' not in st.session_state:
        st.session_state.global_filter_type = "All Time"
    if 'global_selected_year' not in st.session_state:
        st.session_state.global_selected_year = datetime.now().year
    if 'global_selected_month' not in st.session_state:
        st.session_state.global_selected_month = datetime.now().month
    if 'global_start_date' not in st.session_state:
        st.session_state.global_start_date = datetime.now() - timedelta(days=30)
    if 'global_end_date' not in st.session_state:
        st.session_state.global_end_date = datetime.now()

def get_date_filters(key="unique_global_filter"):
    """Common date filter UI component"""
    initialize_filters()
    st.sidebar.subheader("📅 Date Filter")
    df = get_transactions_data(service, st.session_state['sheet_id'])
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        min_date = df['Date'].min()
        max_date = df['Date'].max()
    else:
        min_date = max_date = datetime.now()
    st.session_state.global_filter_type = st.sidebar.radio(
        "Select Time Period",
        ["All Time", "Year", "Month", "Custom Range"],
        key=key
    )
    if st.session_state.global_filter_type == "Year":
        st.session_state.global_selected_year = st.sidebar.selectbox(
            "Select Year",
            sorted(df['Date'].dt.year.unique(), reverse=True) if not df.empty else [datetime.now().year],
            key="unique_global_year"
        )
        start_date = datetime(st.session_state.global_selected_year, 1, 1)
        end_date = datetime(st.session_state.global_selected_year, 12, 31)
    elif st.session_state.global_filter_type == "Month":
        st.session_state.global_selected_year = st.sidebar.selectbox(
            "Select Year",
            sorted(df['Date'].dt.year.unique(), reverse=True) if not df.empty else [datetime.now().year],
            key="unique_global_month_year"
        )
        st.session_state.global_selected_month = st.sidebar.selectbox(
            "Select Month",
            range(1, 13),
            format_func=lambda x: datetime(2000, x, 1).strftime('%B'),
            key="unique_global_month"
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
                key="unique_global_start_date"
            )
        with col2:
            st.session_state.global_end_date = st.date_input(
                "End Date", 
                max_date,
                key="unique_global_end_date"
            )
        start_date = datetime.combine(st.session_state.global_start_date, datetime.min.time())
        end_date = datetime.combine(st.session_state.global_end_date, datetime.max.time())
    else:
        start_date = min_date
        end_date = max_date
    return start_date, end_date

def filter_dataframe(df, start_date, end_date):
    """Filter dataframe based on date range"""
    if df.empty:
        return df
    df['Date'] = pd.to_datetime(df['Date'])
    return df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

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
    st.subheader("Recent Transactions")
    recent_df = df.sort_values('Date', ascending=False).head(5)
    st.dataframe(
        recent_df[['Date', 'Type', 'Category', 'Subcategory', 'Amount', 'Description']].style.format({
            'Amount': 'Rs. {:,.2f}',
            'Date': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
        }),
        hide_index=True
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Income Categories")
        income_by_category = df[df['Type'] == 'Income'].groupby('Category')['Amount'].sum().sort_values(ascending=False).head(5)
        fig_income = px.pie(values=income_by_category.values, 
                          names=income_by_category.index,
                          title='Top Income Sources')
        st.plotly_chart(fig_income)
    with col2:
        st.subheader("Top Expense Categories")
        expense_by_category = df[df['Type'] == 'Expense'].groupby('Category')['Amount'].sum().sort_values(ascending=False).head(5)
        fig_expense = px.pie(values=expense_by_category.values, 
                           names=expense_by_category.index,
                           title='Top Expense Categories')
        st.plotly_chart(fig_expense)
    st.subheader("💡 Spending Insights")
    col1, col2 = st.columns(2)
    with col1:
        df['Day_Type'] = df['Date'].dt.dayofweek.map(lambda x: 'Weekend' if x >= 5 else 'Weekday')
        daily_avg = df[df['Type'] == 'Expense'].groupby('Day_Type')['Amount'].agg(['sum', 'count'])
        daily_avg['avg'] = daily_avg['sum'] / daily_avg['count']
        st.caption("Weekday vs Weekend Spending")
        st.dataframe(
            daily_avg.style.format({
                'sum': 'Rs. {:,.2f}',
                'avg': 'Rs. {:,.2f}/day'
            })
        )
    with col2:
        df['Week_of_Month'] = df['Date'].dt.day.map(lambda x: (x-1)//7 + 1)
        weekly_spending = df[df['Type'] == 'Expense'].groupby('Week_of_Month')['Amount'].mean()
        fig_weekly = px.bar(weekly_spending,
                          title='Average Spending by Week of Month',
                          labels={'value': 'Amount (Rs.)', 'Week_of_Month': 'Week'})
        st.plotly_chart(fig_weekly)

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
    st.subheader("Income by Subcategory")
    subcategory_income = income_df.groupby('Subcategory')['Amount'].sum().sort_values(ascending=False)
    fig_subcategory = px.bar(subcategory_income,
                            title='Income by Subcategory',
                            labels={'value': 'Amount (Rs.)', 'index': 'Subcategory'})
    st.plotly_chart(fig_subcategory)
    st.subheader("💰 Income Stability Analysis")
    monthly_income = df[df['Type'] == 'Income'].groupby(df['Date'].dt.strftime('%Y-%m'))['Amount'].sum()
    income_stats = {
        'Average Monthly Income': monthly_income.mean(),
        'Income Volatility': monthly_income.std() / monthly_income.mean() if monthly_income.mean() > 0 else 0,
        'Highest Income Month': monthly_income.max(),
        'Lowest Income Month': monthly_income.min(),
        'Months with Income': len(monthly_income)
    }
    st.dataframe(
        pd.Series(income_stats).to_frame('Value').style.format({
            'Value': lambda x: f"Rs. {x:,.2f}" if isinstance(x, (int, float)) and x > 100 else f"{x:.2%}" if isinstance(x, float) else x
        })
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
    st.subheader("Expenses by Subcategory")
    subcategory_expenses = expense_df.groupby('Subcategory')['Amount'].sum().sort_values(ascending=False)
    fig_subcategory = px.bar(subcategory_expenses,
                            title='Expenses by Subcategory',
                            labels={'value': 'Amount (Rs.)', 'index': 'Subcategory'})
    st.plotly_chart(fig_subcategory)
    st.subheader("📊 Fixed vs Variable Expenses")
    monthly_category = df[df['Type'] == 'Expense'].groupby(['Category', df['Date'].dt.strftime('%Y-%m')])['Amount'].sum()
    category_consistency = monthly_category.groupby('Category').agg(['mean', 'std'])
    category_consistency['variation'] = (category_consistency['std'] / category_consistency['mean']).fillna(0)
    fixed_expenses = category_consistency[category_consistency['variation'] < 0.2]
    variable_expenses = category_consistency[category_consistency['variation'] >= 0.2]
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Fixed Expenses (Low Variation)")
        st.dataframe(
            fixed_expenses.style.format({
                'mean': 'Rs. {:,.2f}',
                'std': 'Rs. {:,.2f}',
                'variation': '{:.2%}'
            })
        )
    with col2:
        st.caption("Variable Expenses (High Variation)")
        st.dataframe(
            variable_expenses.style.format({
                'mean': 'Rs. {:,.2f}',
                'std': 'Rs. {:,.2f}',
                'variation': '{:.2%}'
            })
        )

def show_pending_transactions(_service, sheet_id):
    """Display pending transactions section"""
    st.subheader("📋 Pending Transactions")
    try:
        df = get_pending_transactions(_service, sheet_id)
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

def show_analytics():
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

def main():
    if not APPS_SCRIPT_URL:
        st.error("Apps Script URL not configured.")
        return
    initialize_users_sheet(service)
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if not st.session_state['authenticated']:
        tab1, tab2 = st.tabs(["Login", "Signup"])
        with tab1:
            login_form(service)
        with tab2:
            signup_form(service)
    else:
        st.write(f"Welcome, {st.session_state['username']}!")
        if st.button("Logout"):
            st.session_state['authenticated'] = False
            st.session_state.pop('username', None)
            st.session_state.pop('sheet_id', None)
            st.rerun()
        show_analytics()

if __name__ == "__main__":
    main()