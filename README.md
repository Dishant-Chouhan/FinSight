# 💡 FinSight

**Your Personal Financial Manager**

FinSight is a smart, comprehensive financial tracking and analytics application that helps users monitor their income, expenses, and pending transactions. It leverages **data analytics** and **NLP techniques** to provide actionable insights, trends, and automatic categorization of financial data — making personal finance management simpler and smarter.  

---

## ✨ Features

- **🔒 User Authentication:** Sign up and log in securely with hashed passwords.  
- **💰 Transaction Tracking:** Record and manage income, expenses, and pending transactions.  
- **📊 Data Analytics:**
  - Overview of total income, expenses, net savings, and saving rate.
  - Monthly trends and summaries.
  - Top income and expense categories with interactive charts.
- **⏳ Pending Transactions:** Track amounts to receive or pay.  
- **🤖 NLP Categorization:** Automatically categorize transactions using NLP for smarter insights.  
- **📈 Visual Analytics:** Interactive charts using Plotly for clear financial understanding.  
- **🗓️ Custom Date Filters:** Analyze data over different periods (All Time, Year, Month, Custom Range).  

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit (interactive dashboards)  
- **Backend:** Google Apps Script & Python  
- **Data Handling:** Pandas (data processing)  
- **Visualization:** Plotly Express (charts & graphs)  
- **Authentication:** Bcrypt (secure password hashing)  
- **NLP:** Categorization of transactions based on descriptions  
- **Environment:** dotenv (environment variables)  

---

## 🖼️ Screenshots

### Analytics Overview
![Analytics 1](https://github.com/Dishant-Chouhan/FinSight/blob/main/analytics1.jpg)  
![Analytics 2](https://github.com/Dishant-Chouhan/FinSight/blob/main/analytics2.jpg)  

### Main Dashboard
![Main](https://github.com/Dishant-Chouhan/FinSight/blob/main/main.jpg)  

### Login Page
![Login](https://github.com/Dishant-Chouhan/FinSight/blob/main/login%20page.jpg)  

---
## ⚙️ Environment Configuration

Create a `.env` file in the project root with the following keys:

```env
APPS_SCRIPT_URL=
GEMINI_API_KEY=
---




**## 📜 Google Apps Script **
function doPost(e) {
  // Check if postData exists and has contents
  if (!e || !e.postData || !e.postData.contents) {
    return ContentService.createTextOutput(
      JSON.stringify({error: 'Invalid or missing POST data'})
    ).setMimeType(ContentService.MimeType.JSON);
  }

  var sheet = SpreadsheetApp.openById(PropertiesService.getScriptProperties().getProperty('MASTER_SHEET_ID'));
  var usersSheet = sheet.getSheetByName('Users') || sheet.insertSheet('Users');
  if (!usersSheet.getRange(1, 1).getValue()) {
    usersSheet.getRange(1, 1, 1, 4).setValues([['Username', 'Email', 'Password_Hash', 'Transaction_Sheet_ID']]);
  }

  try {
    var data = JSON.parse(e.postData.contents);
    var action = data.action;

    if (!action) {
      return ContentService.createTextOutput(
        JSON.stringify({error: 'Missing action in request'})
      ).setMimeType(ContentService.MimeType.JSON);
    }

    switch (action) {
      case 'signup':
        return signup(data, sheet, usersSheet);
      case 'get_users':
        return getUsers(usersSheet);
      case 'create_spreadsheet':
        return createSpreadsheet(data);
      case 'initialize_sheet':
        return initializeSheet(data);
      case 'get_transactions':
        return getTransactions(data);
      case 'get_pending_transactions':
        return getPendingTransactions(data);
      case 'add_transaction':
        return addTransaction(data);
      case 'add_pending_transaction':
        return addPendingTransaction(data);
      case 'update_pending_transaction':
        return updatePendingTransaction(data);
      default:
        return ContentService.createTextOutput(
          JSON.stringify({error: 'Invalid action: ' + action})
        ).setMimeType(ContentService.MimeType.JSON);
    }
  } catch (error) {
    return ContentService.createTextOutput(
      JSON.stringify({error: 'Error processing request: ' + error.message})
    ).setMimeType(ContentService.MimeType.JSON);
  }
}

// Other functions (signup, getUsers, createSpreadsheet, initializeSheet, getTransactions, getPendingTransactions, addTransaction, addPendingTransaction, updatePendingTransaction) 
// follow the same pattern as above
