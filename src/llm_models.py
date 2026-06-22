from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv('.env')

def get_env_variables():
    env_variables = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'MODEL_DEPLOYMENT': os.getenv('MODEL_DEPLOYMENT'),
        'TEMPERATURE' : float(os.getenv('TEMPERATURE', 0.0)),
    }
    return env_variables

