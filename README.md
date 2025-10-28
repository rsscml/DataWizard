### Agentic AI solution for chat-based data analytics & data query
Following two modes are provided:
1. File Upload Mode: 
   User can upload their own excel or csv files. The data will be displayed in OnlyOffice Spreadsheet View. User can then proceed to ask questions on the data in the chat window. Follow-up queries are supported within a session. To clear context & begin a new line of questioning, use the chat delete button. Outputs are displayed as floating or stacked output cards with visualization & Data pages. User can toggle between spreadsheet view and output view.
   This module uses Text-2-Python Code Generator & Executor agent with provisions for error correction & retries in case of failure.  
2. Database Mode:
   User can connect to one of the configured databases & ask questions on its contents. For now, only Unity Catalog Schemas are supported.
   This module uses two-step processing: 1) Text-2-SQL Agent for fetching the data (displayed in Spreadsheet View) followed by 2) Text-2-Python Agent (same as the one used in File Upload Mode) for further analytics on the retrieved data.
    
### Prerequisites:

#### 1. Local OnlyOffice Document Server 
1. The spreadsheet display functionality requires a separately deployed OnlyOffice Document Server. For the docker image for freely available OnlyOffice development server refer this link:
https://hub.docker.com/r/onlyoffice/documentserver
2. For a detailed walkthrough on setting up a local OnlyOffice Server and more info on OnlyOffice functionality, see the following:
https://helpcenter.onlyoffice.com/docs/installation/docs-community-install-docker.aspx

#### 2. Azure Databricks Unity Catalog with prepared Schema (for Database Mode)
#### 3. LLM API Key for either of the following providers (Claude, OpenAI, Gemini(untested)). For Azure OpenAI, set the relevant .env parameters

### Major .env Parameters:

#### OnlyOffice Configuration
1. ONLYOFFICE_SERVER_URL 
2. ONLYOFFICE_CALLBACK_URL
#### Standard Azure parameters
1. AZURE_TENANT_ID
2. AZURE_CLIENT_ID
3. AZURE_CLIENT_SECRET
4. SCOPE
5. SPN (Service principal Name)
6. SUBSCRIPTION_KEY
7. TOKEN_URL
#### Azure Databricks parameters (for database mode)
1. DATABRICKS_SERVER_HOSTNAME
2. DATABRICKS_HTTP_PATH
3. DATABRICKS_CATALOG (Unity Catalog Name)
4. DATABRICKS_SCHEMA (Schema Name in Unity Catalog)
#### OpenAI LLM Service Configuration
1. OPENAI_API_KEY
2. OPENAI_MODEL
#### Claude LLM Service Configuration
1. ANTHROPIC_API_KEY
2. CLAUDE_MODEL
#### AzureOpenAI LLM Service Configuration
1. AZURE_OPENAI_BASE_URL
2. MODEL_VERSION

