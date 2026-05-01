🛠️ API Troubleshooting Assistant

AI-powered assistant for analyzing API issues, finding relevant troubleshooting cases, and generating professional support responses.

🔗 Live App: https://apitroubleshootingassistant.streamlit.app
📦 Repository: https://github.com/jleskovets/API_troubleshooting_assistant

⸻

🚀 Overview

API Troubleshooting Assistant is a web-based AI tool designed to help system analysts and support engineers quickly diagnose API issues and respond to customer requests.

The system uses semantic search to find similar troubleshooting cases and generates structured support replies using LLMs.

This project simulates a real-world support knowledge base system.

⸻

🎯 Key Features

🔍 Semantic Search

Finds the most relevant troubleshooting case based on the customer message using vector similarity.

🤖 AI-generated Support Replies

Automatically generates:
	•	Issue summary
	•	Likely root cause
	•	Recommended next steps
	•	Professional customer email draft

📚 Knowledge Base Management

Supports:
	•	➕ Add case
	•	✏️ Edit case
	•	🗑 Delete case
	•	📋 View cases

Each case includes:
	•	API area
	•	Endpoint
	•	Error code
	•	Problem description
	•	Root cause
	•	Solution
	•	Tags
	•	Logs

🧠 Auto-suggest Tags

Generates tags using AI to improve search accuracy.

📊 Confidence Visualization

Displays semantic match confidence level.

🕘 Request History

Stores previously analyzed requests.

🏗️ Architecture

User Input
↓
Semantic Search (ChromaDB)
↓
Relevant Case
↓
LLM Analysis (OpenAI API)
↓
Structured Support Response

⸻

🧰 Tech Stack

Frontend
	•	Streamlit

Backend
	•	Python

AI
	•	OpenAI API
	•	Embeddings
	•	Semantic Search

Vector Database
	•	ChromaDB

Data Storage
	•	CSV-based Knowledge Base

Deployment
	•	Streamlit Cloud

📊 Future Improvements
	•	Multiple case suggestions (Top-3 matches)
	•	Log file upload (.json / .txt)
	•	Jira integration
	•	Case versioning
	•	API monitoring integration

👩‍💻 Author

Julia Leskovets

System Analyst | API | AI Tools

LinkedIn: www.linkedin.com/in/yulia-leskovets-bba364109