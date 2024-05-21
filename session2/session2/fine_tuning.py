from dotenv import load_dotenv
import os
import json
import pandas as pd
import openai
from openai import OpenAI

# .env 파일 로드
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")

client = OpenAI()

client.files.create(
    file = open('output.jsonl', 'rb'),
    purpose = 'fine-tune'
)
# id : openai 쪽으로 파일이 업로드되는데 파일과 맵핑되는 파일 아이디가 발급된다. 기억.

client.fine_tuning.jobs.create(
    training_file = '파일명', #파일명 바꿔줘야 합니다.
    model = 'gpt-3.5-turbo'
)

finetuning_lst = client.fine_tuning.jobs.list(limit = 10)
for elem in finetuning_lst.data:
  if elem.training_file == '파일명':    #파일명 바꿔줘야 합니다.
    print(elem.status)