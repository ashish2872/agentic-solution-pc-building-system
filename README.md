# agentic-solution-pc-building-system
Agentic Solution that helps users build PC based on the user requirements. 

Architecture Diagram : 
graph TD
    %% Nodes
    Entry((Entry Point))
    GR[Requirements Agent]
    Sup[Supervisor]
    QA[Query Agent]
    CA[Critique Agent]
    RA[Response Agent]
    End((END))

    %% Connections
    Entry --> GR
    GR --> Sup
    Sup -- complete --> QA
    Sup -- incomplete --> RA
    
    QA --> Sup
    QA --> CA
    
    CA --> Sup
    
    Sup -- needs_requery --> QA
    Sup -- compatible --> RA
    
    RA --> End
    RA -- awaiting_user --> End

    %% Styling
    style Entry fill:#f9f,stroke:#333,stroke-width:2px
    style Sup fill:#ffcc99,stroke:#333,stroke-width:2px
    style End fill:#99ff99,stroke:#333,stroke-width:2px
    style GR fill:#d4e1f5,stroke:#333
    style QA fill:#d4e1f5,stroke:#333
    style CA fill:#d4e1f5,stroke:#333
    style RA fill:#d4e1f5,stroke:#333


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
