import os
import streamlit as st
from streamlit_pills import pills

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


#기본적인 chatbot ui를 위한 style 작성
def CPT():
    st.markdown(
    """
    <style>
    body {
        background-color: #D2E0FB; /* 연한 하늘색 배경 */
    }
    /* 전체 챗봇 창 가운데 정렬 */
    .full-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    /* 챗봇 창 스타일링 */
    .chat-container {
        width: 90%; /* 너비를 조정하여 대화창을 넓게 설정합니다 */
        padding: 20px;
        background-color: #F9F3CC;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
    }

    .chat-container2 {
        width: 90%; /* 너비를 조정하여 대화창을 넓게 설정합니다 */
        padding: 20px;
        background-color: #D7E5CA;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        float: right;
    }
    
    /* 사용자 메시지 스타일링 */
    .user-msg {
        background-color: #D7E5CA;
        color: #333;
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    
    /* 챗봇 메시지 스타일링 */
    .assistant-msg {
        background-color: #F9F3CC;
        color: black;
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")

    # PDF 파일 로드 및 텍스트 추출
    loader = PyPDFLoader('./files/믿음_학습지_교사용.pdf')
    documents = loader.load()

    # 텍스트를 적절한 크기로 나누기
    text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # 문장을 벡터로 변환한 뒤, vector_store에 저장
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(texts, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    # 사고력을 기르기 위한 챗봇 system_prompt 설정
    system_template_hint = """당신은 중등 특수교육 선생님입니다.
    사용자는 중학교 또는 고등학교의 특수교육 대상자이며, 정신 연령은 9~12세 입니다.
    당신은 사용자의 질문에 친절하게 답변해야 하며, 쉬운 단어를 사용해 답변해주세요.
    답변에 부정적인 단어를 포함해서는 안됩니다.
    
    ----------------
    {summaries}
    You MUST answer in Korean and in Markdown format:"""
    messages_hint = [
        SystemMessagePromptTemplate.from_template(system_template_hint),
        HumanMessagePromptTemplate.from_template("{question}")
    ]

    prompt_hint = ChatPromptTemplate.from_messages(messages_hint)
    chain_type_kwargs_hint = {"prompt": prompt_hint}
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    #사고력을 기르기 위한 챗봇 생성
    chain_hint = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs_hint
    )

    def generate_response_hint(input_text):
        result = chain_hint(input_text)
        return result['answer']


    st.caption('챗봇은 GPT-3.5-Turbo에 RAG 기술을 적용하여 제작되었습니다.')
    st.divider()
    st.header(f'믿음님, 반가워요:wave:')
    # st.markdown(":red[파이썬으로 00하는 방법이 궁금해.] 또는 :red[00하는 코드를 만들고 싶어.]와 같이 질문해 주세요!")
    
    # cpt봇이 말해주는 첫 문장 생성
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "믿음님을 위한 챗봇이에요! 무엇을 도와드릴까요?"}]

    
    # cpt봇이 응답한 직전 응답을 저장할 공간
    if "last_question" not in st.session_state:
        st.session_state["last_question"] = ""

    # cpt봇과 나눈 이전 대화 가져와서 채팅창에 표시(즉, 그 전에 있던 것)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-container2"><div class="user-msg">{msg["content"]}</div></div>', unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            if "```" in msg['content']:
                st.chat_message("assistant").write(msg['content'])
            else:
                st.markdown(f'<div class="chat-container"><div class="assistant-msg">{msg["content"]}</div></div>', unsafe_allow_html=True)

    # 사용자 질문과 그에 따른 응답 출력
    if prompt := st.chat_input("믿음님의 궁금증"):  # 만약 사용자가 입력한 내용이 있다면 
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-container2"><div class="user-msg">{prompt}</div></div>', unsafe_allow_html=True)
        with st.spinner('답변을 생성 중입니다💨'):
            msg = generate_response_hint(prompt)

        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.markdown(f'<div class="chat-container"><div class="assistant-msg">{msg}</div></div>', unsafe_allow_html=True)

        st.session_state["last_question"] = msg # 직전 질문 저장
        selected = pills("답변은 어떤가요?", ["만족해요", "스타일이 마음에 안 들어요", "이해가 안 돼요"], ["👍", "👎", "❓"], index=False)
