# streamlit cloud 사용 시 필요
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import psycopg2

from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from study_bot_UI import CPT

# .env 파일 로드
load_dotenv()

st.set_page_config(
    page_title="질의응답챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


CPT()