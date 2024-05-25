# ✅ 장애 아동을 위한 서비스 개발

## 🤔 프로그램 제작 목적
# - 유튜브는 다양한 연령대의 의견을 익명으로 확인할 수 있는 데이터 레이크
# - 통합 교육에 대해 반대하는 의견이 많은 것을 확인
# - 시각화해서 확인 및 근거 제공

## ♾️ 설계 방법
# - 유튜브 코멘트 다운로더를 불러와 원하는 영상의 유튜브 댓글 수집
# - 형태소 분석기를 활용한 명사 추출 및 불용어 제거
# - 스트림릿을 이용하여 워드클라우드 확인

# 코드 실행을 위해 필요한 것을 모두 불러왔습니다.
from youtube_comment_downloader import *  # 유튜브 댓글들을 불러올 목적
from itertools import islice
from youtube_comment_downloader import *
from transformers import pipeline
import matplotlib.pyplot as plt

plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

downloader = YoutubeCommentDownloader()

video_ids = ['DQU1M7eIJis', 'HfwDdLjaPC0', 'rlYdjc2_kJA&t=24s', 'YBEb7fUd7p4&t=23s']
url_list = ["https://www.youtube.com/watch?v=" + video_id for video_id in video_ids]

all_comments = []  # List to store all comments

for url in url_list:
    comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
    # limited_comments = list(islice(comments, 100))  # 최대 100개의 댓글만 추출
    #인기도에 따른 공감을 가진 50개의 댓글들을 추출했음
    all_comments.extend(comments)

# Extract the values corresponding to the 'text' key into a separate list
text_values = [comment['text'] for comment in all_comments]

classifier = pipeline("text-classification", model="matthewburke/korean_sentiment")
def classify_sentiment(text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    return label, score

# 텍스트별로 감성 분류
# for text in text_values:
#     label, score = classify_sentiment(text)
#     is_positive = label == 'LABEL_1'  # 수정: 감성 분류 LABEL_1을 긍정으로 판단
#     print(f"텍스트: {text}")
#     print(f"감성 분류: {label}")
#     print(f"감성 점수: {score}")
#     print(f"긍정 여부: {is_positive}")
#     print()

# True와 False의 개수 계산
# true_count = sum(1 for comment in text_values if classify_sentiment(comment)[0] == 'LABEL_1')
# false_count = len(text_values) - true_count

# 파이 차트 그리기
# 데이터
sizes = [60, 94]

labels = ['긍정적인 댓글', '부정적인 댓글']
colors = ['#66c2ff', '#ff9999']
wedgeprops={'width': 0.8, 'edgecolor': 'w', 'linewidth': 2}
explode = [0.05, 0.05]


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f'{val}개\n({pct:.2f}%)'
    return my_autopct



plt.pie(sizes,
        labels=labels, 
        autopct=make_autopct(sizes), 
        startangle=260, 
        counterclock=False, 
        explode=explode, 
        shadow=True, 
        colors=colors,
        wedgeprops=wedgeprops
        )


plt.show()

## https://www.youtube.com/watch?v=DQU1M7eIJis (발달장애 ‘특수반’ 따로? “우리는 함께 배워요” / KBS 2022.04.25.)
## https://www.youtube.com/watch?v=HfwDdLjaPC0 (발달 장애아의 일반학교 입학 도전기)
## https://www.youtube.com/watch?v=rlYdjc2_kJA&t=24s (자폐, 분리가 답일까?통합교육이 가야할 방향 [클립] | 9층시사국33회 (23.10.08))
## https://www.youtube.com/watch?v=YBEb7fUd7p4&t=23s ("똑같은 학생이니까"…함께 공부하며 장애·편견 극복 [통합교육] / EBS뉴스 2023. 08. 22)