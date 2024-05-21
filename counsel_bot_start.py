# streamlit cloud 사용 시 필요
import sqlite3
import sys
sys.modules['pysqlite3'] = sys.modules.pop('sqlite3')  # 로컬 디비

import os
import streamlit as st

from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from counsel_bot_UI import CPT

# .env 파일 로드
load_dotenv()

st.set_page_config(
    page_title="질의응답챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


CPT()
