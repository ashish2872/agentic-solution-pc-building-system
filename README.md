# agentic-solution-pc-building-system
Agentic Solution that helps users build PC based on the user requirements. 

1.	Clone the repository:

git clone https://github.com/ashish2872/agentic-solution-pc-building-system
cd agentic-solution-pc-building-system

2.	Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

3.	Install dependencies:
pip install -r requirements.txt

4.	Configure environment variables:
cp .env.example .env
Edit .env and set the following:
OPENAI_API_KEY=sk-proj-...        # Your OpenAI API key (no quotes)
MODEL_DEPLOYMENT=gpt-4o-mini     # Or gpt-4o for higher quality
TEMPERATURE=0.0                  # 0.0 recommended for deterministic SQL
DATABASE_PATH=pc_components.db   # Path to SQLite database

5.	Run the application:
streamlit run app.py
