```python
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
import os
from konlpy.tag import Okt
import MeCab 
from kss import split_sentences   
from pykospacing import spacing 
from tqdm import tqdm
```


```python
mecab = MeCab.Tagger()

def getNVM_lemma(text):
    tokenizer = MeCab.Tagger() # Mecab 토크나이저 
    parsed = tokenizer.parse(text) # parse 토큰화
    word_tag = [w for w in parsed.split('\n')] 
    pos = []
    tags = ['NNG','NNP','NR','NP','VV','VA','VX','VCP','VCN','MAG','IC'] # 일반명사, 고유명사, 동사, 형용사, 보조용언, 긍정지정사, 긍정부정사, 일반부사
    for word_ in word_tag[:-2] : # 맨뒤 둘은 EOS랑 빈문자열이라 무시
        word = word_.split('\t') 
        tag = word[1].split(',') # [0]이 단어고 [1]은 품사정보 있는 부분
        if (len(word[0]) < 2) : # 길이 1개인건 버리고 위에서부터 다시 실행
            continue
        if (tag[-1] != '*') : # 단어를 더 쪼개는 작업
            t = tag[-1].split('/')
            if (len(t[0]) > 1 and ('VV' in t[1] or 'VA' in t[1] or 'VX' in t[1])):
                pos.append(t[0]) # 위 셀에 예시 문장을 보면 "내린다", "하얘졌" 에서 '내리', '하얘지' 라는 어근을 추출하는코
            elif (len(t[0]) > 1 and ('NNG' in t[1])):
                pos.append(word[0])
        else :
            if (tag[0] in tags) : # 지정한 품사에 해당되면 해당 단어를 pos라는 리스트에 넣음
                pos.append(word[0]) 
    return pos # 조건을 만족하는 어근들로 구성된 리스트

okt = Okt() 

def preprocess_okt(text):
    #     text = spacing(text) # 띄어쓰기 보정 위에서 했으면 필요없습니다
    pos_words = okt.pos(text, stem=True)
    words = [word for word, tag in pos_words if tag in ['Noun', 'Adjective', 'Verb'] ]
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    stopped_words = [w for w in words if not w in stopwords]
    return stopped_words
```


```python

```


```python
# df = xx[['review', 'after', 'pos', 'cluster', 'comment', 'date','topic', 'weight', 'label',
#                   'review_len', 'review_senti',
#        'review_senti_mean', 'okt_pos']].join(zz[['com_okt_pos', 'com_num', 'com_len_mean', 'com_len_std',
#        'com_senti_dist', 'com_senti', 'com_senti_mean', 'com_senti_std']])
```


```python
# df.to_csv('엘지수업용집안일.csv', encoding='utf-8-sig')
```

1) 크롤링 단계
- 게시글 제목, 게시글 본문, 게시글 댓글을 크롤링 해옴

2) 전처리 단계
- 리뷰들을 읽어보며 집안일과 관련없는데 수집된 게시글 삭제
ex) 그건 저 집안일이고 너가 상환할 바가 아냐

- url, 이메일, 특수문자 아닌 문자열, 카페규정문구, 중복문구

- 맞춤법 교정
ex) 설겆이 > 설거지  

- 형태소 분석. okt를 활용. 동사원형복구 기능을 높이 사 사용함

3) word2vec & ward clutering 
- marketing science 논문 인용
- word2vec 에서 embbeding layer




```python
'밑에는 그냥 논문 플로우에 맞는 코드 내용만 복사한 코드, 실제 형식에 맞춘 코드는 아님'
```


```python
' 카페 규정, 규칙, url http 등 삭제'
# delete_word = ['http','----','===','☞','카페규정','카페규칙','▶','속풀이방','Loading','※ 질문','생활영어']
# for i in delete_word:
#     try:
#         df = df[~df['review'].str.contains(i)]
#     except:
#         pass

'중복행 삭제, 결측치 있는 행 제거 (제목과 본문이 없는 경우. 댓글은 없어도 ㄱㅊ)'
# drop_duplicates()

'전처리, 맞춤법   이의동의어 (아기 애기)'
# count = 0
# for i in range(len(data)):
#     data.review.iloc[i] = clean_str(data.review.iloc[i]) # 전처리
#     data.review.iloc[i] = spacing(data.review.iloc[i]) # 띄어쓰기 보정
#     data.review.iloc[i] = data.review.iloc[i].replace('설겆이','설거지')
    
#     count +=1
#     if count%100 == 0:
#         print(count,' 개 완료')

'형태소 분석'
# for i in range(len(data.review)):
#     df_pos['review'].iloc[i] = getNVM_lemma(df_pos['review'].iloc[i])
#     print(str(i)+'번: ',df_pos.review.iloc[i])

# empty_index = [index for index, sentence in zip(df_pos.index,df_pos) if len(sentence) < 1]

# print(empty_index) 

'word2vec  embedding_dim 과 window에 따른 실루엣 변화 측정 -> 여기선 계산효율의 증대, 인용논문에서의 활용성 입증 등을 근거로 사용 '
# from gensim.models import Word2Vec

# EMBEDDING_DIM = 30 # 임베딩 크기는 논문을 따름
# model = Word2Vec(sentences=df_pos.review, sg=1, size=EMBEDDING_DIM, window=7, min_count=1) #sg 0은 CBOW, 1은 SKIP-GRAM
# w2v_vocab = list(model.wv.vocab) # 임베딩 된 단어 리스트
# print('Vocabulary size : ',len(w2v_vocab)) 
# print('Vecotr shape :',model.wv.vectors.shape) 


'doc2vec 후 군집화 수행 ward  실루엣 좋은 군집 선택'
# def doc_vectors(padding):
#     document_embedding_list = []

#     # 각 문장은 리스트의 리스트 형태로 숫자인코딩 된 벡터
#     for line in padding:
#         doc2vec = np.zeros(EMBEDDING_DIM) # 0값의 벡터를 만들어줍니다
#         count = 0 # 평균을 내기위해 count해줍니다
#         for token in line: # 정수토큰 하나하나를 가져옵니다                    
#             if token in np.arange(1,len(embedding_matrix)):  # 제로패딩은 count안하게, Vocab_size까지만
#                 count += 1 
#                 # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
#                 if doc2vec is np.zeros(EMBEDDING_DIM): # 첫번째 단어때 필요한 문법
#                     doc2vec = embedding_matrix[token]
#                 else:
#                     doc2vec = doc2vec + embedding_matrix[token] # 단어의 w2v를 누적해서 더해줍니다
        
#         # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠줍니다 = 문장 벡터
#         doc2vec_average = doc2vec / (count+1) # 혹시나 있을 zero-divdend방지위해 +1
#         document_embedding_list.append(doc2vec_average)

#     # 각 문서에 대한 문서 벡터 리스트를 리턴
#     return document_embedding_list

'군집별 워드클라우드 시각화'




```


```python

```


```python

```


```python

```


```python
# 전처리가 완료된 데이터
df = pd.read_csv('페이퍼용집안일다시.csv', encoding='utf-8-sig')
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-5-565ad17709f2> in <module>
          1 # 전처리가 완료된 데이터
    ----> 2 df = pd.read_csv('페이퍼용집안일다시.csv', encoding='utf-8-sig')
    

    ~\anaconda3\envs\tf2.0\lib\site-packages\pandas\io\parsers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)
        686     )
        687 
    --> 688     return _read(filepath_or_buffer, kwds)
        689 
        690 
    

    ~\anaconda3\envs\tf2.0\lib\site-packages\pandas\io\parsers.py in _read(filepath_or_buffer, kwds)
        452 
        453     # Create the parser.
    --> 454     parser = TextFileReader(fp_or_buf, **kwds)
        455 
        456     if chunksize or iterator:
    

    ~\anaconda3\envs\tf2.0\lib\site-packages\pandas\io\parsers.py in __init__(self, f, engine, **kwds)
        946             self.options["has_index_names"] = kwds["has_index_names"]
        947 
    --> 948         self._make_engine(self.engine)
        949 
        950     def close(self):
    

    ~\anaconda3\envs\tf2.0\lib\site-packages\pandas\io\parsers.py in _make_engine(self, engine)
       1178     def _make_engine(self, engine="c"):
       1179         if engine == "c":
    -> 1180             self._engine = CParserWrapper(self.f, **self.options)
       1181         else:
       1182             if engine == "python":
    

    ~\anaconda3\envs\tf2.0\lib\site-packages\pandas\io\parsers.py in __init__(self, src, **kwds)
       1991         if kwds.get("compression") is None and encoding:
       1992             if isinstance(src, str):
    -> 1993                 src = open(src, "rb")
       1994                 self.handles.append(src)
       1995 
    

    FileNotFoundError: [Errno 2] No such file or directory: '페이퍼용집안일다시.csv'



```python
year=[]
month = []
for i in df['date'][:]:
    year.append(int(str(i)[:2]))
    month.append(int(str(i)[2:]))
df['year'] = year
df['month'] = month
```


```python
df[df['year']==16]
df[df['year']==17]
df[df['year']==18]
df[df['year']==19]
df[df['year']==20]
df[df['year']==21]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>review</th>
      <th>after</th>
      <th>pos</th>
      <th>cluster</th>
      <th>comment</th>
      <th>date</th>
      <th>review_len</th>
      <th>review_senti</th>
      <th>review_senti_mean</th>
      <th>...</th>
      <th>com_okt_pos</th>
      <th>com_num</th>
      <th>com_len_mean</th>
      <th>com_len_std</th>
      <th>com_senti_dist</th>
      <th>com_senti</th>
      <th>com_senti_mean</th>
      <th>com_senti_std</th>
      <th>year</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6534</th>
      <td>6534</td>
      <td>집안일 줄여주는 전자제품 잘 쓰시나요? 한국이었다면 힘든 독박 살림에 독박 육아에....</td>
      <td>집안일 줄여주는 전자제품 잘 쓰시나 요 한국이었다면 힘든 독박 살림에 독박 육아에 ...</td>
      <td>['집안일', '줄이', '전자', '제품', '한국', '힘들', '독박', '살...</td>
      <td>3</td>
      <td>[['제 픽은 지금도 기특한 물걸레청소기 브라바요. ㅋㅋㅋ'], ['오~ 물걸레 청...</td>
      <td>2101</td>
      <td>618</td>
      <td>-4</td>
      <td>-0.666667</td>
      <td>...</td>
      <td>[['제', '픽', '지금', '기특하다', '물걸레', '청소기', '브라', ...</td>
      <td>48</td>
      <td>83.520833</td>
      <td>55.938579</td>
      <td>[0, -6, -7, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0...</td>
      <td>-23</td>
      <td>-0.479167</td>
      <td>2.499913</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6535</th>
      <td>6535</td>
      <td>다들 집안일 하세요? 결혼하신분들 중...일하면서 집안일도 하시나요?가족들 밥 차려...</td>
      <td>다들 집안일 하세요 결혼하신 분들 중일 하면서 집안일도 하시나 요 가족들 밥 차려주...</td>
      <td>['집안일', '결혼', '집안일', '가족', '차리', '실습생', '언니', ...</td>
      <td>1</td>
      <td>[['그 언니는 이제 30살이고 결혼도 20대 초중반에 했어요 늙은 나이도 아닌데 ...</td>
      <td>2101</td>
      <td>243</td>
      <td>-10</td>
      <td>-3.333333</td>
      <td>...</td>
      <td>[['그', '언니', '이제', '살이', '결혼', '대다', '초', '중반'...</td>
      <td>16</td>
      <td>100.125000</td>
      <td>86.738022</td>
      <td>[0, 0, 0, 0, -12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...</td>
      <td>-12</td>
      <td>-0.750000</td>
      <td>2.904738</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6536</th>
      <td>6536</td>
      <td>제일 하기 싫은 집안일은? 오늘 아침엔 눈도 뜨기 힘들고,나가기도 싫더라고요~~굉장...</td>
      <td>제일 하기 싫은 집안일은 오늘 아침엔 눈도 뜨기 힘들고 나가기도 싫더라고요 굉장히 ...</td>
      <td>['제일', '집안일', '오늘', '아침', '힘들', '나가', '굉장히', '...</td>
      <td>0</td>
      <td>[['저는 설거지랑 음쓰버리기요...ㅜㅡ'], ['설거지..그것도 진짜 하기시러요ㅎ...</td>
      <td>2101</td>
      <td>429</td>
      <td>-9</td>
      <td>-2.250000</td>
      <td>...</td>
      <td>[['저', '설거지', '음', '쓰다', '버리다', '기요'], ['설거지',...</td>
      <td>22</td>
      <td>29.818182</td>
      <td>17.324802</td>
      <td>[0, 0, 0, 0, 0, 0, 3, 3, -6, -6, 4, 0, 0, 0, 0...</td>
      <td>1</td>
      <td>0.045455</td>
      <td>2.285871</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6537</th>
      <td>6537</td>
      <td>요리 하는게 집안일 중 힘든일 1순위? 저는 남편이랑 집안일을 공평하게 한다 생각하...</td>
      <td>요리 하는 게 집안일 중 힘든 일 1순위 저는 남편이랑 집안일을 공평하게 한 다 생...</td>
      <td>['요리', '집안일', '힘들', '순위', '남편', '집안일', '공평', '...</td>
      <td>1</td>
      <td>[['그렇게 죄~~~ 다 만들면 힘들긴 해요ㅠ 육아 제외하고 하기 싫은 집안일 고르...</td>
      <td>2101</td>
      <td>818</td>
      <td>-6</td>
      <td>-1.500000</td>
      <td>...</td>
      <td>[['죄', '만들다', '힘들다', '해', '육아', '제외', '싫다', '집...</td>
      <td>53</td>
      <td>79.622642</td>
      <td>62.123804</td>
      <td>[-8, 0, 0, 0, 1, 0, -6, -8, 0, -6, 0, 0, 4, 0,...</td>
      <td>-77</td>
      <td>-1.452830</td>
      <td>3.265405</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6538</th>
      <td>6538</td>
      <td>아이들 집안일 시키나요? 저희집 초등둘.올해 4학년 6학년 올라가는 남맨데제 눈에 ...</td>
      <td>아이들 집안일 시키나요 저희 집 초등 둘 올해 4학년 6학년 올라가는 남맨 데 제 ...</td>
      <td>['아이', '집안일', '시키', '저희', '초등', '학년', '학년', '올...</td>
      <td>1</td>
      <td>[['잘하고계시네요~ 저도 큰아이는 시켜요\n주말대청소할때는 2호도 자기방정리정돈까...</td>
      <td>2101</td>
      <td>262</td>
      <td>0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>[['저', '크다', '시키다', '주말', '청소', '때', '호도', '자기...</td>
      <td>55</td>
      <td>60.036364</td>
      <td>44.877597</td>
      <td>[3, 3, 3, -7, -6, 8, -7, -5, 0, 3, 0, 0, 0, 0,...</td>
      <td>1</td>
      <td>0.018182</td>
      <td>3.886999</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7372</th>
      <td>7372</td>
      <td>집안일은 끝도 없어요 ㅋㅋ ??이제 밥먹고 한숨돌리려니 건조기 빨래 다되었다네요 ㅎ...</td>
      <td>집안일은 끝도 없어요 이제 밥 먹고 한숨 돌리려니 건조기 빨래 다 되었다네요 커피 ...</td>
      <td>['집안일', '이제', '돌리', '건조기', '빨래', '커피', '내려놓', ...</td>
      <td>0</td>
      <td>[['집안일 하고 여유좀 부려보려고 하면 하원시간이죠 ㅠㅠ~집안일도 좋지만 아이 없...</td>
      <td>2103</td>
      <td>105</td>
      <td>3</td>
      <td>3.000000</td>
      <td>...</td>
      <td>[['집안일', '여유', '부리다', '보다', '하원시', '간이', '집안일'...</td>
      <td>10</td>
      <td>30.300000</td>
      <td>20.784850</td>
      <td>[0, 0, 0, -6, 0, 0, -6, 0, 0, -6]</td>
      <td>-18</td>
      <td>-1.800000</td>
      <td>2.749545</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7373</th>
      <td>7373</td>
      <td>벌써 배고파요 엄마~~~간식주세요 아침에 우유랑 콘푸라이트 말아주었는데배고프다네요....</td>
      <td>벌써 배고파요 엄마 간식주세요 아침에 우유랑 콘푸라이트 말아주었는데 배고프다네요 근...</td>
      <td>['벌써', '배고프', '엄마', '간식', '주세요', '아침', '우유', '...</td>
      <td>0</td>
      <td>[['사과껍질 좋다잖아요~건조기돌리면 금방 마르지 않나요'], ['글겠죠? 지금 세...</td>
      <td>2103</td>
      <td>123</td>
      <td>1</td>
      <td>0.500000</td>
      <td>...</td>
      <td>[['사과', '껍질', '좋다', '건조기', '돌리다', '금방', '마르지',...</td>
      <td>21</td>
      <td>45.190476</td>
      <td>21.944994</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0,...</td>
      <td>-9</td>
      <td>-0.428571</td>
      <td>1.399708</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7374</th>
      <td>7374</td>
      <td>소파수술 후 집안일 첫째있어서 소파술후 하루지나자마자 바로조금씩한거 같아요.보통 수...</td>
      <td>소파수술 후 집안일 첫째 있어서 소파술 후 하루 지나자마자 바로 조금씩 한 거 같아...</td>
      <td>['소파', '수술', '첫째', '소파술', '하루', '지나', '바로', '조...</td>
      <td>1</td>
      <td>[['저도 첫째 때매 마냥 쉴 순 없는데\n그래도 몸이 많이 힘들긴 하네요 ㅠㅠ 쉬...</td>
      <td>2103</td>
      <td>70</td>
      <td>0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>[['저', '첫째', '때매', '마냥', '쉬다', '순', '없다', '몸',...</td>
      <td>3</td>
      <td>54.333333</td>
      <td>25.037749</td>
      <td>[0, 0, 0]</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7375</th>
      <td>7375</td>
      <td>맘님들도 그런날 있죠??? 오늘 제가 종일 쳐묵쳐묵한거 고백할게요ㅋ아침은 원래 안먹...</td>
      <td>맘님들도 그런 날 있죠 오늘 제가 종일 쳐 묵쳐 묵한 거 고백할게요 아침은 원래 안...</td>
      <td>['오늘', '종일', '고백', '아침', '원래', '패스', '마시', '커피...</td>
      <td>0</td>
      <td>[['혹....셋..째... ㅡㅡ;\n가끔씩 좋은글 보고가네요~ 좋은밤!'], ['...</td>
      <td>2103</td>
      <td>685</td>
      <td>-2</td>
      <td>-0.285714</td>
      <td>...</td>
      <td>[['혹', '셋', '째다', '가끔', '좋다', '글', '보다', '좋다',...</td>
      <td>20</td>
      <td>53.550000</td>
      <td>50.708456</td>
      <td>[0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, -...</td>
      <td>-15</td>
      <td>-0.750000</td>
      <td>2.299456</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7376</th>
      <td>7376</td>
      <td>싸우고 정떨어져요.. 남편보다 상대적으로 제가 더 깔끔한편이라 제가 못참고 치우다보...</td>
      <td>싸우고 정 떨어져요 남편보다 상대적으로 제가 더 깔끔한 편이라 제가 못 참고 치우다...</td>
      <td>['싸우', '떨어지', '남편', '상대', '치우', '집안일', '프로', '...</td>
      <td>1</td>
      <td>[['안마 같은 거 해주지 마세요...ㅜㅜ 답답하시겠어요ㅠㅠ'], ['ㅜㅜ시간을 두...</td>
      <td>2103</td>
      <td>819</td>
      <td>-10</td>
      <td>-0.476190</td>
      <td>...</td>
      <td>[['안마', '같다', '거', '해주다', '말다', '답답하다'], ['시간'...</td>
      <td>32</td>
      <td>82.687500</td>
      <td>70.720593</td>
      <td>[-6, 0, 1, 0, -6, 0, 0, -6, 0, -6, -6, -5, 0, ...</td>
      <td>-66</td>
      <td>-2.062500</td>
      <td>3.131867</td>
      <td>21</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>843 rows × 21 columns</p>
</div>




```python
len(df[df['year']==16]),len(df[df['year']==17]),len(df[df['year']==18]),len(df[df['year']==19]),len(df[df['year']==20]),len(df[df['year']==21])
```




    (3290, 3244, 3228, 3136, 3325, 843)




```python
a1 = df[df['year']==18][df['month']!=1][df['month']!=2][df['month']!=3]
a2 = df[df['year']==19]
a3 = df[df['year']==20]
a4 = df[df['year']==21]
len(a1),len(a2),len(a3),len(a4)
```

    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\ipykernel_launcher.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      """Entry point for launching an IPython kernel.
    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\ipykernel_launcher.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      """Entry point for launching an IPython kernel.
    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\ipykernel_launcher.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      """Entry point for launching an IPython kernel.
    




    (2440, 3136, 3325, 843)




```python
2440+3136+3325+843
```




    9744




```python
df = pd.concat([pd.concat([pd.concat([a1,a2]),a3]),a4])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>review</th>
      <th>after</th>
      <th>pos</th>
      <th>cluster</th>
      <th>comment</th>
      <th>date</th>
      <th>review_len</th>
      <th>review_senti</th>
      <th>review_senti_mean</th>
      <th>...</th>
      <th>com_okt_pos</th>
      <th>com_num</th>
      <th>com_len_mean</th>
      <th>com_len_std</th>
      <th>com_senti_dist</th>
      <th>com_senti</th>
      <th>com_senti_mean</th>
      <th>com_senti_std</th>
      <th>year</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8165</th>
      <td>8165</td>
      <td>집안일 하실 때요 17개월 아기를 키우고 있는 초보맘이에요심각한 엄마껌딱지라서 하루...</td>
      <td>집안일 하실 때 요 17개월 아기를 키우고 있는 초보맘이에 요심각한 엄마 껌 딱지라...</td>
      <td>['집안일', '아기', '키우', '초보', '엄마', '딱지', '하루하루', ...</td>
      <td>1</td>
      <td>[['전그래서 반찬 다 사서먹고있어요 설거지만후다닥하고  뭐든하나는 포기해야되여ㅜㅜ...</td>
      <td>1804</td>
      <td>471</td>
      <td>-2</td>
      <td>-1.000000</td>
      <td>...</td>
      <td>[['래서', '반찬', '사서', '먹다', '설거지', '후', '닥', '뭐'...</td>
      <td>22</td>
      <td>128.818182</td>
      <td>75.703150</td>
      <td>[0, 0, 0, 0, 0, 3, 4, -6, 3, -6, 0, 0, 0, -12,...</td>
      <td>-44</td>
      <td>-2.000000</td>
      <td>4.067610</td>
      <td>18</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8166</th>
      <td>8166</td>
      <td>집안일 육아 11개월 아기 키우면서 집안일까지 하려니 힘드네요..집안일 하면 애기가...</td>
      <td>집안일 육아 11개월 아기 키우면서 집안일까지 하려니 힘드네요 집안일 하면 애기가 ...</td>
      <td>['육아', '아기', '키우', '집안일', '힘들', '집안일', '애기', '...</td>
      <td>1</td>
      <td>[['저두항상그래요전14개월인데요  항시가만히있질안아요   최대한 아침에  집안일은...</td>
      <td>1804</td>
      <td>180</td>
      <td>0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>[['항상', '그렇다', '요전', '개월', '항시', '있다', '최대한', ...</td>
      <td>3</td>
      <td>67.000000</td>
      <td>30.692019</td>
      <td>[3, -4, 0]</td>
      <td>-1</td>
      <td>-0.333333</td>
      <td>2.867442</td>
      <td>18</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8167</th>
      <td>8167</td>
      <td>임신초기 집안일 이제 5주차 된 새댁인데요,,임신초기에 집안일 어떻게 하셨나요?저는...</td>
      <td>임신 초기 집안일 이 제 5주차 된 새댁인데 요 임신 초기에 집안일 어떻게 하셨나요...</td>
      <td>['임신', '초기', '임신', '초기', '집안일', '어떻게', '작년', '...</td>
      <td>1</td>
      <td>[['남편이 주로 하고 정리같은건 제가 해요'], ['전 클로락스 같이 냄새 심하게...</td>
      <td>1804</td>
      <td>188</td>
      <td>0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>[['남편', '주로', '정리', '같다', '제', '해'], ['전', '클로...</td>
      <td>5</td>
      <td>80.400000</td>
      <td>71.166284</td>
      <td>[-5, 0, 0, -8, -6]</td>
      <td>-19</td>
      <td>-3.800000</td>
      <td>3.249615</td>
      <td>18</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8168</th>
      <td>8168</td>
      <td>집안일 중 가장 하기 싫은 것? 전 바로.. 빨래 개기랍니다며칠째 한숨만 나오네요 ...</td>
      <td>집안일 중 가장 하기 싫은 것 전 바로 빨래 개기랍니다 며칠째 한숨만 나오네요 누가...</td>
      <td>['집안일', '가장', '바로', '빨래', '개기', '며칠', '나오', '건...</td>
      <td>0</td>
      <td>[['아앗..! 저두요저두요... 필요한거만 건조대에서 골라입고^^;; 이모님 오셔...</td>
      <td>1804</td>
      <td>247</td>
      <td>-6</td>
      <td>-6.000000</td>
      <td>...</td>
      <td>[['앗', '두', '필요하다', '건조대', '고르다', '입다', '이모', ...</td>
      <td>30</td>
      <td>66.733333</td>
      <td>27.381178</td>
      <td>[0, -8, 3, 0, -6, -8, 4, 0, 0, 0, 0, 0, -4, 0,...</td>
      <td>-10</td>
      <td>-0.333333</td>
      <td>3.165789</td>
      <td>18</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8169</th>
      <td>8169</td>
      <td>끝도 없는 집안일. 식구들 다 자는데 저는 아직까지 못 자고 있네요.하루에 빨래 두...</td>
      <td>끝도 없는 집안일 식구들 다 자는 데 저는 아직까지 못 자고 있네요 하루에 빨래 두...</td>
      <td>['식구', '아직', '하루', '빨래', '소설', '육아', '지치', '이리...</td>
      <td>0</td>
      <td>[['진짜 집안일은 끝도 없는거 같아요.\n오늘 하면 내일 또 다시 반복 ㅠ\n전 ...</td>
      <td>1804</td>
      <td>156</td>
      <td>0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>[['진짜', '집안일', '끝', '없다', '같다', '오늘', '내일', '또...</td>
      <td>10</td>
      <td>43.300000</td>
      <td>10.826357</td>
      <td>[-6, 3, -8, -6, 0, 0, 0, -6, 0, 0]</td>
      <td>-23</td>
      <td>-2.300000</td>
      <td>3.579106</td>
      <td>18</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7372</th>
      <td>7372</td>
      <td>집안일은 끝도 없어요 ㅋㅋ ??이제 밥먹고 한숨돌리려니 건조기 빨래 다되었다네요 ㅎ...</td>
      <td>집안일은 끝도 없어요 이제 밥 먹고 한숨 돌리려니 건조기 빨래 다 되었다네요 커피 ...</td>
      <td>['집안일', '이제', '돌리', '건조기', '빨래', '커피', '내려놓', ...</td>
      <td>0</td>
      <td>[['집안일 하고 여유좀 부려보려고 하면 하원시간이죠 ㅠㅠ~집안일도 좋지만 아이 없...</td>
      <td>2103</td>
      <td>105</td>
      <td>3</td>
      <td>3.000000</td>
      <td>...</td>
      <td>[['집안일', '여유', '부리다', '보다', '하원시', '간이', '집안일'...</td>
      <td>10</td>
      <td>30.300000</td>
      <td>20.784850</td>
      <td>[0, 0, 0, -6, 0, 0, -6, 0, 0, -6]</td>
      <td>-18</td>
      <td>-1.800000</td>
      <td>2.749545</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7373</th>
      <td>7373</td>
      <td>벌써 배고파요 엄마~~~간식주세요 아침에 우유랑 콘푸라이트 말아주었는데배고프다네요....</td>
      <td>벌써 배고파요 엄마 간식주세요 아침에 우유랑 콘푸라이트 말아주었는데 배고프다네요 근...</td>
      <td>['벌써', '배고프', '엄마', '간식', '주세요', '아침', '우유', '...</td>
      <td>0</td>
      <td>[['사과껍질 좋다잖아요~건조기돌리면 금방 마르지 않나요'], ['글겠죠? 지금 세...</td>
      <td>2103</td>
      <td>123</td>
      <td>1</td>
      <td>0.500000</td>
      <td>...</td>
      <td>[['사과', '껍질', '좋다', '건조기', '돌리다', '금방', '마르지',...</td>
      <td>21</td>
      <td>45.190476</td>
      <td>21.944994</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0,...</td>
      <td>-9</td>
      <td>-0.428571</td>
      <td>1.399708</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7374</th>
      <td>7374</td>
      <td>소파수술 후 집안일 첫째있어서 소파술후 하루지나자마자 바로조금씩한거 같아요.보통 수...</td>
      <td>소파수술 후 집안일 첫째 있어서 소파술 후 하루 지나자마자 바로 조금씩 한 거 같아...</td>
      <td>['소파', '수술', '첫째', '소파술', '하루', '지나', '바로', '조...</td>
      <td>1</td>
      <td>[['저도 첫째 때매 마냥 쉴 순 없는데\n그래도 몸이 많이 힘들긴 하네요 ㅠㅠ 쉬...</td>
      <td>2103</td>
      <td>70</td>
      <td>0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>[['저', '첫째', '때매', '마냥', '쉬다', '순', '없다', '몸',...</td>
      <td>3</td>
      <td>54.333333</td>
      <td>25.037749</td>
      <td>[0, 0, 0]</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7375</th>
      <td>7375</td>
      <td>맘님들도 그런날 있죠??? 오늘 제가 종일 쳐묵쳐묵한거 고백할게요ㅋ아침은 원래 안먹...</td>
      <td>맘님들도 그런 날 있죠 오늘 제가 종일 쳐 묵쳐 묵한 거 고백할게요 아침은 원래 안...</td>
      <td>['오늘', '종일', '고백', '아침', '원래', '패스', '마시', '커피...</td>
      <td>0</td>
      <td>[['혹....셋..째... ㅡㅡ;\n가끔씩 좋은글 보고가네요~ 좋은밤!'], ['...</td>
      <td>2103</td>
      <td>685</td>
      <td>-2</td>
      <td>-0.285714</td>
      <td>...</td>
      <td>[['혹', '셋', '째다', '가끔', '좋다', '글', '보다', '좋다',...</td>
      <td>20</td>
      <td>53.550000</td>
      <td>50.708456</td>
      <td>[0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, -...</td>
      <td>-15</td>
      <td>-0.750000</td>
      <td>2.299456</td>
      <td>21</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7376</th>
      <td>7376</td>
      <td>싸우고 정떨어져요.. 남편보다 상대적으로 제가 더 깔끔한편이라 제가 못참고 치우다보...</td>
      <td>싸우고 정 떨어져요 남편보다 상대적으로 제가 더 깔끔한 편이라 제가 못 참고 치우다...</td>
      <td>['싸우', '떨어지', '남편', '상대', '치우', '집안일', '프로', '...</td>
      <td>1</td>
      <td>[['안마 같은 거 해주지 마세요...ㅜㅜ 답답하시겠어요ㅠㅠ'], ['ㅜㅜ시간을 두...</td>
      <td>2103</td>
      <td>819</td>
      <td>-10</td>
      <td>-0.476190</td>
      <td>...</td>
      <td>[['안마', '같다', '거', '해주다', '말다', '답답하다'], ['시간'...</td>
      <td>32</td>
      <td>82.687500</td>
      <td>70.720593</td>
      <td>[-6, 0, 1, 0, -6, 0, 0, -6, 0, -6, -6, -5, 0, ...</td>
      <td>-66</td>
      <td>-2.062500</td>
      <td>3.131867</td>
      <td>21</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>9744 rows × 21 columns</p>
</div>




```python
# df.to_csv('페이퍼용1804_2103.csv',encoding='utf-8-sig')
```


```python
# 전처리가 완료된 데이터
df = pd.read_csv('페이퍼용감성다시.csv', encoding='utf-8-sig')
```


```python
df.columns
```




    Index(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'review', 'after',
           'pos', 'cluster', 'comment', 'date', 'review_len', 'review_senti',
           'review_senti_mean', 'okt_pos', 'com_okt_pos', 'com_num',
           'com_len_mean', 'com_len_std', 'com_senti_dist', 'com_senti',
           'com_senti_mean', 'com_senti_std', 'year', 'month', 'Unnamed: 23',
           'Unnamed: 24', 'Unnamed: 25'],
          dtype='object')




```python
back_to_list = []
for i in df.okt_pos:
    back_to_list.append(i[2:-2].split("', '"))
df['okt2'] = back_to_list
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>Unnamed: 0.1.1</th>
      <th>review</th>
      <th>after</th>
      <th>pos</th>
      <th>cluster</th>
      <th>comment</th>
      <th>date</th>
      <th>review_len</th>
      <th>...</th>
      <th>com_senti_dist</th>
      <th>com_senti</th>
      <th>com_senti_mean</th>
      <th>com_senti_std</th>
      <th>year</th>
      <th>month</th>
      <th>Unnamed: 23</th>
      <th>Unnamed: 24</th>
      <th>Unnamed: 25</th>
      <th>okt2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>8165</td>
      <td>8165</td>
      <td>집안일 하실 때요 17개월 아기를 키우고 있는 초보맘이에요심각한 엄마껌딱지라서 하루...</td>
      <td>집안일 하실 때 요 17개월 아기를 키우고 있는 초보맘이에 요심각한 엄마 껌 딱지라...</td>
      <td>['집안일', '아기', '키우', '초보', '엄마', '딱지', '하루하루', ...</td>
      <td>1</td>
      <td>[['전그래서 반찬 다 사서먹고있어요 설거지만후다닥하고  뭐든하나는 포기해야되여ㅜㅜ...</td>
      <td>1804</td>
      <td>471</td>
      <td>...</td>
      <td>[-1.0, 1.25, 0, 0.75, 2.0, -0.7142857142857143...</td>
      <td>5.895538</td>
      <td>0.267979</td>
      <td>0.853791</td>
      <td>18</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[집안일, 때, 요, 개월, 아기, 키우다, 있다, 초보, 맘, 심각, 엄마, 껌,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>8166</td>
      <td>8166</td>
      <td>집안일 육아 11개월 아기 키우면서 집안일까지 하려니 힘드네요..집안일 하면 애기가...</td>
      <td>집안일 육아 11개월 아기 키우면서 집안일까지 하려니 힘드네요 집안일 하면 애기가 ...</td>
      <td>['육아', '아기', '키우', '집안일', '힘들', '집안일', '애기', '...</td>
      <td>1</td>
      <td>[['저두항상그래요전14개월인데요  항시가만히있질안아요   최대한 아침에  집안일은...</td>
      <td>1804</td>
      <td>180</td>
      <td>...</td>
      <td>[-2.0, -1.0, -0.5]</td>
      <td>-3.500000</td>
      <td>-1.166667</td>
      <td>0.623610</td>
      <td>18</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[집안일, 육아, 개월, 아기, 키우다, 집안일, 힘드다, 집안일, 애기, 놀다, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8167</td>
      <td>8167</td>
      <td>임신초기 집안일 이제 5주차 된 새댁인데요,,임신초기에 집안일 어떻게 하셨나요?저는...</td>
      <td>임신 초기 집안일 이 제 5주차 된 새댁인데 요 임신 초기에 집안일 어떻게 하셨나요...</td>
      <td>['임신', '초기', '임신', '초기', '집안일', '어떻게', '작년', '...</td>
      <td>1</td>
      <td>[['남편이 주로 하고 정리같은건 제가 해요'], ['전 클로락스 같이 냄새 심하게...</td>
      <td>1804</td>
      <td>188</td>
      <td>...</td>
      <td>[-2.0, -2.0, -1.0, 0.0, 2.0]</td>
      <td>-3.000000</td>
      <td>-0.600000</td>
      <td>1.496663</td>
      <td>18</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[임신, 초기, 집안일, 제, 주차, 되다, 새댁, 요, 임신, 초기, 집안일, 어...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8168</td>
      <td>8168</td>
      <td>집안일 중 가장 하기 싫은 것? 전 바로.. 빨래 개기랍니다며칠째 한숨만 나오네요 ...</td>
      <td>집안일 중 가장 하기 싫은 것 전 바로 빨래 개기랍니다 며칠째 한숨만 나오네요 누가...</td>
      <td>['집안일', '가장', '바로', '빨래', '개기', '며칠', '나오', '건...</td>
      <td>0</td>
      <td>[['아앗..! 저두요저두요... 필요한거만 건조대에서 골라입고^^;; 이모님 오셔...</td>
      <td>1804</td>
      <td>247</td>
      <td>...</td>
      <td>[2.0, -1.6666666666666667, 2.0, 0.333333333333...</td>
      <td>-5.000000</td>
      <td>-0.166667</td>
      <td>1.286864</td>
      <td>18</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[집안일, 중, 가장, 싫다, 것, 전, 바로, 빨래, 개다, 며칠, 한숨, 나오다...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>8169</td>
      <td>8169</td>
      <td>끝도 없는 집안일. 식구들 다 자는데 저는 아직까지 못 자고 있네요.하루에 빨래 두...</td>
      <td>끝도 없는 집안일 식구들 다 자는 데 저는 아직까지 못 자고 있네요 하루에 빨래 두...</td>
      <td>['식구', '아직', '하루', '빨래', '소설', '육아', '지치', '이리...</td>
      <td>0</td>
      <td>[['진짜 집안일은 끝도 없는거 같아요.\n오늘 하면 내일 또 다시 반복 ㅠ\n전 ...</td>
      <td>1804</td>
      <td>156</td>
      <td>...</td>
      <td>[-1.5, 2.0, -2.0, -1.6666666666666667, 0, -1.3...</td>
      <td>-8.500000</td>
      <td>-0.850000</td>
      <td>1.165356</td>
      <td>18</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[끝, 없다, 집안일, 식구, 자다, 데, 저, 못, 자고, 있다, 하루, 빨래, ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9739</th>
      <td>9739</td>
      <td>7372</td>
      <td>7372</td>
      <td>집안일은 끝도 없어요 ㅋㅋ ??이제 밥먹고 한숨돌리려니 건조기 빨래 다되었다네요 ㅎ...</td>
      <td>집안일은 끝도 없어요 이제 밥 먹고 한숨 돌리려니 건조기 빨래 다 되었다네요 커피 ...</td>
      <td>['집안일', '이제', '돌리', '건조기', '빨래', '커피', '내려놓', ...</td>
      <td>0</td>
      <td>[['집안일 하고 여유좀 부려보려고 하면 하원시간이죠 ㅠㅠ~집안일도 좋지만 아이 없...</td>
      <td>2103</td>
      <td>105</td>
      <td>...</td>
      <td>[0.75, 1.0, 0, 0, -1.0, 0, -1.3333333333333333...</td>
      <td>-4.083333</td>
      <td>-0.408333</td>
      <td>0.832041</td>
      <td>21</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[집안일, 끝, 없다, 이제, 밥, 먹다, 한숨, 돌리다, 건조기, 빨래, 되어다,...</td>
    </tr>
    <tr>
      <th>9740</th>
      <td>9740</td>
      <td>7373</td>
      <td>7373</td>
      <td>벌써 배고파요 엄마~~~간식주세요 아침에 우유랑 콘푸라이트 말아주었는데배고프다네요....</td>
      <td>벌써 배고파요 엄마 간식주세요 아침에 우유랑 콘푸라이트 말아주었는데 배고프다네요 근...</td>
      <td>['벌써', '배고프', '엄마', '간식', '주세요', '아침', '우유', '...</td>
      <td>0</td>
      <td>[['사과껍질 좋다잖아요~건조기돌리면 금방 마르지 않나요'], ['글겠죠? 지금 세...</td>
      <td>2103</td>
      <td>123</td>
      <td>...</td>
      <td>[2.0, 0, 1.0, 2.0, 0, 0, 0.6666666666666666, 0...</td>
      <td>16.833333</td>
      <td>0.801587</td>
      <td>1.147368</td>
      <td>21</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[벌써, 배고프다, 엄마, 간식, 줄다, 아침, 우유, 콘푸, 라이트, 말다, 배고...</td>
    </tr>
    <tr>
      <th>9741</th>
      <td>9741</td>
      <td>7374</td>
      <td>7374</td>
      <td>소파수술 후 집안일 첫째있어서 소파술후 하루지나자마자 바로조금씩한거 같아요.보통 수...</td>
      <td>소파수술 후 집안일 첫째 있어서 소파술 후 하루 지나자마자 바로 조금씩 한 거 같아...</td>
      <td>['소파', '수술', '첫째', '소파술', '하루', '지나', '바로', '조...</td>
      <td>1</td>
      <td>[['저도 첫째 때매 마냥 쉴 순 없는데\n그래도 몸이 많이 힘들긴 하네요 ㅠㅠ 쉬...</td>
      <td>2103</td>
      <td>70</td>
      <td>...</td>
      <td>[-1.5, 0, 0.5]</td>
      <td>-1.000000</td>
      <td>-0.333333</td>
      <td>0.849837</td>
      <td>21</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[소파수술, 후, 집안일, 첫째, 있다, 소파, 술, 후, 하루, 지나다, 바로, ...</td>
    </tr>
    <tr>
      <th>9742</th>
      <td>9742</td>
      <td>7375</td>
      <td>7375</td>
      <td>맘님들도 그런날 있죠??? 오늘 제가 종일 쳐묵쳐묵한거 고백할게요ㅋ아침은 원래 안먹...</td>
      <td>맘님들도 그런 날 있죠 오늘 제가 종일 쳐 묵쳐 묵한 거 고백할게요 아침은 원래 안...</td>
      <td>['오늘', '종일', '고백', '아침', '원래', '패스', '마시', '커피...</td>
      <td>0</td>
      <td>[['혹....셋..째... ㅡㅡ;\n가끔씩 좋은글 보고가네요~ 좋은밤!'], ['...</td>
      <td>2103</td>
      <td>685</td>
      <td>...</td>
      <td>[2.0, 0, 2.0, 1.5, 2.0, 0, 0, 0.0, 2.0, 2.0, 0...</td>
      <td>19.733333</td>
      <td>0.986667</td>
      <td>1.053850</td>
      <td>21</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[맘, 들다, 그렇다, 날, 있다, 오늘, 제, 종일, 치다, 묵, 치다, 묵, 거...</td>
    </tr>
    <tr>
      <th>9743</th>
      <td>9743</td>
      <td>7376</td>
      <td>7376</td>
      <td>싸우고 정떨어져요.. 남편보다 상대적으로 제가 더 깔끔한편이라 제가 못참고 치우다보...</td>
      <td>싸우고 정 떨어져요 남편보다 상대적으로 제가 더 깔끔한 편이라 제가 못 참고 치우다...</td>
      <td>['싸우', '떨어지', '남편', '상대', '치우', '집안일', '프로', '...</td>
      <td>1</td>
      <td>[['안마 같은 거 해주지 마세요...ㅜㅜ 답답하시겠어요ㅠㅠ'], ['ㅜㅜ시간을 두...</td>
      <td>2103</td>
      <td>819</td>
      <td>...</td>
      <td>[-1.0, 2.0, 0.0, -1.0, -1.0, -0.5, 0, 1.0, -2....</td>
      <td>-9.527473</td>
      <td>-0.297734</td>
      <td>1.138838</td>
      <td>21</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[싸우다, 정, 떨어지다, 남편, 상대, 제, 더, 깔끔하다, 편이, 제, 못, 참...</td>
    </tr>
  </tbody>
</table>
<p>9744 rows × 27 columns</p>
</div>




```python
from gensim.models import Word2Vec

EMBEDDING_DIM = 20 # 임베딩 크기는 논문을 따름
model = Word2Vec(sentences=df.okt2, sg=1, size=EMBEDDING_DIM, window=5, min_count=1) #sg 0은 CBOW, 1은 SKIP-GRAM
w2v_vocab = list(model.wv.vocab) # 임베딩 된 단어 리스트
print('Vocabulary size : ',len(w2v_vocab)) 
print('Vecotr shape :',model.wv.vectors.shape)
```

    Vocabulary size :  20354
    Vecotr shape : (20354, 20)
    


```python
w2v_vocab
```




    ['집안일',
     '때',
     '요',
     '개월',
     '아기',
     '키우다',
     '있다',
     '초보',
     '맘',
     '심각',
     '엄마',
     '껌',
     '딱지',
     '하루하루',
     '힘',
     '들다',
     '화장실',
     '마음대로',
     '못',
     '가다',
     '아이',
     '업다',
     '채',
     '밥',
     '먹다',
     '허리',
     '끊어지다',
     '것',
     '같다',
     '번',
     '기분',
     '좋다',
     '보이다',
     '과자',
     '치즈',
     '걸',
     '주다',
     '호비다',
     '뽀로로',
     '티비',
     '틀어주다',
     '눈치',
     '보다',
     '나',
     '편하다',
     '애기',
     '방치',
     '속',
     '상하',
     '미안하다',
     '죽다',
     '남편',
     '교대',
     '근무',
     '하루',
     '종일',
     '혼자',
     '날',
     '많다',
     '퇴근',
     '오다',
     '짠하다',
     '보이',
     '미안',
     '진짜',
     '어찌',
     '모르다',
     '끝',
     '없다',
     '볼',
     '수도',
     '어머님',
     '어리다',
     '두시',
     '어떻다',
     '살림',
     '요리',
     '설거지',
     '시간',
     '걸리다',
     '육아',
     '팁',
     '부탁드리다',
     '오늘',
     '고생',
     '모두',
     '편안하다',
     '행복하다',
     '밤',
     '되다',
     '힘드다',
     '놀다',
     '다리',
     '잡고',
     '소리',
     '지르다',
     '수',
     '건',
     '낮',
     '잠',
     '띠',
     '매다',
     '아프다',
     '잠깐',
     '내다',
     '달라',
     '매일',
     '시키다',
     '간단하다',
     '비법',
     '알다',
     '임신',
     '초기',
     '제',
     '주차',
     '새댁',
     '저',
     '작년',
     '유산',
     '경험',
     '조심하다',
     '온종일',
     '눕다',
     '지내다',
     '그렇다',
     '온통',
     '신랑',
     '처음',
     '재밌다',
     '거',
     '지나다',
     '조금',
     '다른',
     '미국',
     '궁금하다',
     '답글',
     '부탁',
     '드리다',
     '중',
     '가장',
     '싫다',
     '전',
     '바로',
     '빨래',
     '개다',
     '며칠',
     '한숨',
     '나오다',
     '누가',
     '큰일',
     '줄',
     '맞다',
     '건조',
     '기',
     '사서',
     '신세계',
     '지다',
     '석',
     '달',
     '젠',
     '더미',
     '옷',
     '고르다',
     '입다',
     '왜',
     '귀찮다',
     '거의',
     '해달라다',
     '개',
     '자리',
     '갖다',
     '놓다',
     '고민',
     '와중',
     '또',
     '착',
     '이렇다',
     '아니다',
     '식구',
     '자다',
     '데',
     '자고',
     '두',
     '청',
     '소설',
     '겆',
     '지치다',
     '고',
     '싶다',
     '안',
     '자기',
     '틀리다',
     '어서',
     '끝내다',
     '쭉',
     '펴다',
     '기계',
     '바닥',
     '청소',
     '로봇청소기',
     '자동',
     '말리다',
     '건조기',
     '삶다',
     '세탁기',
     '식기세척기',
     '분',
     '유물',
     '끓이다',
     '정수기',
     '음식물',
     '찌꺼기',
     '버리다',
     '분쇄기',
     '친정',
     '집',
     '쓰다',
     '더',
     '갈리다',
     '내려가다',
     '환경부',
     '인증',
     '제품',
     '합법',
     '갈아타다',
     '가격',
     '비싸다',
     '크게',
     '안일',
     '반',
     '확',
     '줄다',
     '자유',
     '부인',
     '휴일',
     '모처럼',
     '살',
     '딸램',
     '데리',
     '에버랜드',
     '가면',
     '토요일',
     '아침',
     '유치원',
     '학부모',
     '면담',
     '다녀오다',
     '애',
     '여름',
     '신발',
     '죄',
     '꺼내다',
     '빨',
     '어제',
     '크다',
     '딸',
     '딸기',
     '잼',
     '만들기',
     '놀이',
     '해주다',
     '난장판',
     '부엌',
     '치우다',
     '하하',
     '청소기',
     '돌리다',
     '뭐',
     '합',
     '꽈',
     '산더미',
     '인',
     '판',
     '소파',
     '위',
     '손짓',
     '개키',
     '라며',
     '햄',
     '볶다',
     '저녁',
     '보내다',
     '유',
     '공기',
     '리',
     '디',
     '수원',
     '즐겁다',
     '워킹맘',
     '현재',
     '외국',
     '장기',
     '출장',
     '이구',
     '도와주다',
     '어른',
     '서울',
     '계시',
     '일',
     '도움',
     '받다',
     '어렵다',
     '얼마',
     '교통사고',
     '치료받다',
     '중이',
     '생각',
     '노력',
     '서다',
     '돌보다',
     '주말',
     '이제',
     '짜증',
     '둘',
     '엉망',
     '인지',
     '조언',
     '구',
     '해보다',
     '해',
     '여유',
     '편',
     '완쾌',
     '주시',
     '도우미',
     '이모',
     '모시다',
     '혹시',
     '일주일',
     '분리수거',
     '등',
     '의뢰',
     '가능하다',
     '요일',
     '단',
     '덜다',
     '괜찮다',
     '절',
     '말',
     '길다',
     '읽다',
     '미리',
     '감사',
     '제일',
     '커피',
     '롤케이크',
     '돌아가다',
     '듣다',
     '갑자기',
     '좋아하다',
     '너',
     '널',
     '나서다',
     '가지런하다',
     '만족하다',
     '뿌듯하다',
     '않다',
     '이사',
     '대가',
     '넓어지다',
     '그릇',
     '자꾸',
     '발견',
     '구석구석',
     '박',
     '문지르다',
     '편도',
     '만',
     '해도',
     '표시',
     '안나',
     '늘',
     '적다',
     '그것',
     '옷장',
     '넣다',
     '들어서다',
     '정말',
     '건가',
     '게',
     '위로',
     '보고',
     '반성',
     '기능',
     '잡념',
     '전업',
     '돌',
     '지난',
     '순하다',
     '울',
     '돌아다니다',
     '중요하다',
     '비중',
     '뒷전',
     '일과',
     '이틀',
     '한번',
     '널다',
     '이유식',
     '치',
     '만들다',
     '닦다',
     '자주',
     '차리기',
     '목욕',
     '먹이',
     '기이',
     '렇게',
     '낮잠',
     '정도',
     '다음',
     '일어나다',
     '싹',
     '답',
     '답정',
     '노',
     '별로',
     '두다',
     '가요',
     '평일',
     '휴무',
     '늦잠',
     '일어나서',
     '동안',
     '댕댕',
     '노래',
     '낮술',
     '이다',
     '돋다',
     '몸',
     '낳다',
     '출산휴가',
     '복직',
     '섬',
     '왕복',
     '출퇴근',
     '그',
     '길',
     '장',
     '대부분',
     '보기',
     '식사',
     '얻다',
     '힘들다',
     '자마자',
     '외출',
     '친구',
     '만나다',
     '술',
     '마시다',
     '주',
     '웬만하다',
     '집안',
     '잔소리',
     '올해',
     '중순',
     '시작',
     '육아휴직',
     '먹이다',
     '장보기',
     '준비',
     '당연하다',
     '역시',
     '요즘',
     '억울하다',
     '폭발',
     '당신',
     '내',
     '돈',
     '벌다',
     '쉬다',
     '핀잔',
     '하자',
     '동참',
     '손',
     '전혀',
     '대다',
     '그동안',
     '네',
     '논리',
     '라면',
     '강요',
     '말다',
     '시대',
     '잘못',
     '태어나다',
     '우리',
     '아빠',
     '세대',
     '여자',
     '테',
     '순간',
     '남자',
     '실체',
     '정이',
     '뚝',
     '떨어지다',
     '남',
     '앞',
     '와이프',
     '사랑',
     '완벽하다',
     '해내다',
     '로',
     '스스로',
     '포장',
     '주변',
     '사람',
     '부부',
     '이상',
     '식',
     '말다툼',
     '들어오다',
     '가라',
     '여러',
     '방면',
     '바라다',
     '피곤하다',
     '대단하다',
     '요구',
     '걸다',
     '매우',
     '심각하다',
     '래미',
     '등원',
     '커텐',
     '티',
     '나다',
     '계시다',
     '일찍',
     '나가다',
     '늦다',
     '밀리',
     '타인',
     '필요하다',
     '절실',
     '느끼다',
     '주일',
     '무조건',
     '토',
     '유선',
     '무겁다',
     '선',
     '아들',
     '뽑다',
     '꺼지다',
     '스트레스',
     '무선',
     '질',
     '럿어요',
     '입성',
     '행복',
     '해지니',
     '덩달아',
     '쓸다',
     '매번',
     '기운',
     '빠지다',
     '방법',
     '바꾸다',
     '안방',
     '제대로',
     '이불',
     '화장',
     '선반',
     '보',
     '유리',
     '구석',
     '점점',
     '범위',
     '화장품',
     '정리',
     '앉다',
     '나르다',
     '마음',
     '먹음',
     '달리',
     '엉뚱하다',
     '짓',
     '마무리',
     '늘다',
     '울다',
     '넘치다',
     '노무',
     '공부',
     '프리',
     '바쁘다',
     '생활',
     '존경',
     '스럽다',
     '둘째',
     '중일',
     '첫째',
     '케어',
     '번의',
     '아픔',
     '기도',
     '불구',
     '안정',
     '사리다',
     '이번',
     '참고',
     '손길',
     '다행하다',
     '어린이집',
     '적응',
     '다니다',
     '유아식',
     '반찬',
     '등등',
     '눈',
     '작정',
     '하일',
     '생각나다',
     '틈',
     '조심성',
     '어디',
     '안다',
     '생기다',
     '기간',
     '소중하다',
     '댓글',
     '걸레질',
     '아궁',
     '중간',
     '퇴',
     '마저',
     '밀리다',
     '육',
     '퇴후',
     '마무',
     '시한',
     '기다리다',
     '노하우',
     '가용',
     '쉬',
     '반도',
     '미루다',
     '런가',
     '끝없다',
     '카페',
     '독박',
     '월욜',
     '출근',
     '잔뜩',
     '스쿨',
     '프랩',
     '보여주다',
     '몬테',
     '매',
     '눈알',
     '붙이다',
     '다가',
     '들락날락',
     '프',
     '랩',
     '빼온',
     '책',
     '어케',
     '따르다',
     '순딩',
     '고맙다',
     '점심',
     '맛점',
     '클리어',
     '아깝다',
     '스타',
     '뚜',
     '개운',
     '운',
     '먼저',
     '매트',
     '놀',
     '잇다',
     '감',
     '전기포트',
     '물',
     '보리차',
     '티백',
     '담그다',
     '후',
     '식히다',
     '병',
     '넣기',
     '완료',
     '온',
     '지금',
     '분유',
     '잔',
     '코코',
     '재우다',
     '옆',
     '잠시',
     '나가야',
     '산후조리',
     '음식',
     '거들다',
     '젤',
     '걱정',
     '시',
     '댁',
     '어머니',
     '상황',
     '선배',
     '남다',
     '부담',
     '남아',
     '희다',
     '꼭',
     '허둥대다',
     '출산',
     '언제',
     '재',
     '절개',
     '첫',
     '병원',
     '입원',
     '조리',
     '도저히',
     '퇴원',
     '대략',
     '돼다',
     '살살',
     '예약',
     '불편하다',
     '누구',
     '웬',
     '그닥',
     '편이',
     '성격',
     '이상하다',
     '언니',
     '비',
     '오니',
     '집콕',
     '미세먼지',
     '창문',
     '열다',
     '환기',
     '비설',
     '일요일',
     '요한',
     '무척',
     '짧다',
     '가족',
     '차리다',
     '지나가다',
     '쯤',
     '하혈',
     '생리',
     '휴직',
     '직장',
     '일이',
     '사실',
     '위험하다',
     '쑤시다',
     '선생님',
     '피',
     '곰',
     '냉장고',
     '글',
     '정신',
     '부모님',
     '모시',
     '베트남어',
     '영어',
     '콩글리쉬',
     '거기',
     '엄청나다',
     '덥다',
     '하니',
     '긴장',
     '더위',
     '쥐',
     '약',
     '최소한',
     '동선',
     '최대한',
     '만족',
     '미치다',
     '듯이',
     '다도',
     '중독',
     '하나',
     '어케들',
     '여',
     '아가',
     '자서',
     '칭얼대다',
     '젖',
     '물리',
     '자구',
     '프로',
     '문제',
     '낮다',
     '물리다',
     '빼다',
     '계속',
     '보채다',
     '타이',
     '모빌',
     '땐',
     '보통',
     '반드시',
     '가야',
     '붙다',
     '실증',
     '젖다',
     '안내',
     '거부',
     '법',
     '폰',
     '통화',
     '멍',
     '때리다',
     '침대',
     '감옥',
     '잠들다',
     '조용하다',
     '탈옥',
     '귀신',
     '알',
     '놔두다',
     '저건',
     '해탈',
     '컴퓨터',
     '뭐라다',
     '작성',
     '책상',
     '키',
     '보드',
     '두드리다',
     '지경',
     '하나요',
     '파',
     '썰다',
     '냉동',
     '실',
     '보관',
     '표고버섯',
     '쓰레기봉투',
     '미기',
     '주부',
     '안일하다',
     '사다',
     '울리다',
     '자지러지다',
     '가스렌지',
     '끝나다',
     '나머지',
     '내일',
     '다림질',
     '양복',
     '회사',
     '며',
     '구해',
     '쓰레기',
     '포함',
     '정리정돈',
     '페이',
     '격주',
     '금호동',
     '뮤',
     '사우나',
     '사이',
     '우선',
     '드렁',
     '타이거',
     '형님',
     '신곡',
     '주인',
     '원래',
     '밖',
     '아주',
     '끝장',
     '맑은',
     '하늘',
     '아래',
     '오픈',
     '실랑이',
     '벌이',
     '뒤',
     '돌아보다',
     '웃프',
     '고릴라',
     '청취',
     '창',
     '최파타',
     '라디오',
     '재미',
     '술술',
     '벌써',
     '방',
     '맞벌이',
     '가끔',
     '연락',
     '마디',
     '모하',
     '답답하다',
     '이르다',
     '말로',
     '넘어가다',
     '전화통화',
     '끊다',
     '더럽다',
     '국',
     '이기',
     '잠재우다',
     '가까이',
     '유미',
     '우미',
     '나름',
     '더하다',
     '나위',
     '일단',
     '짐',
     '서로',
     '진쩌',
     '탠대',
     '그게',
     '욕하겟',
     '몇',
     '배',
     '할애',
     '오전',
     '여유롭다',
     '오후',
     '딸아이',
     '유천',
     '쇼파',
     '엉덩이',
     '돌아서다',
     '거실',
     '마땅치',
     '자재',
     '볼일',
     '휙',
     '쳐다보다',
     '여기저기',
     '물떼',
     '평소',
     '정돈',
     '드럽다',
     '진정',
     '실컷',
     '체질',
     '월차',
     '심부름',
     '약속',
     '따다',
     '악어',
     '정쩡하',
     '잡히다',
     '쥬새',
     '불',
     '세트',
     '이건',
     '쥬근데',
     '새',
     '사시',
     '세탁',
     '얼집',
     '박스',
     '서서',
     '다리다',
     '몰다',
     '죙일',
     '차다',
     '스타일',
     '어지럽다',
     '패딩',
     '빨다',
     '까딱',
     '산',
     '죄송하다',
     '수고',
     '표',
     '진심',
     '외면',
     '겨우',
     '쌓이다',
     '인형',
     '쌓다',
     '일렬',
     '세우다',
     '항상',
     '취미',
     '붙이',
     '마치',
     '돌아오다',
     '멍하니',
     '경우',
     '우울하다',
     '날리다',
     '취향',
     '세',
     '저물다',
     '행주',
     '되어다',
     '깨끗하다',
     ...]




```python
print(model.wv.most_similar('집안일'))
print()
print(model.wv.most_similar('남편'))
print()
print(model.wv.similarity('귀엽다', '아기'))
```

    [('어째', 0.9035537242889404), ('일', 0.8981004953384399), ('정말', 0.8930031061172485), ('일거리', 0.8876669406890869), ('일과', 0.8872736096382141), ('지옥', 0.8706901669502258), ('주중', 0.8704385757446289), ('유독', 0.8692319393157959), ('처녀', 0.8674923181533813), ('모자라다', 0.865792989730835)]
    
    [('신랑', 0.9717484712600708), ('저', 0.95597243309021), ('제', 0.948318600654602), ('터치', 0.9376451969146729), ('거들다', 0.9345053434371948), ('하래', 0.9288206100463867), ('종종', 0.9283267259597778), ('요전', 0.9268803596496582), ('안해', 0.9262921214103699), ('일절', 0.9261221885681152)]
    
    0.67724484
    


```python
# save model in ASCII (word2vec) format
# 텍스트 파일로 단어들의 임베딩 벡터 저장
filename = 'imdb_embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)
```


```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.okt2) # train 데이터
word_index = tokenizer.word_index

len(word_index), word_index
```




    (20354,
     {'집안일': 1,
      '있다': 2,
      '요': 3,
      '안': 4,
      '제': 5,
      '되다': 6,
      '보다': 7,
      '거': 8,
      '남편': 9,
      '없다': 10,
      '것': 11,
      '집': 12,
      '저': 13,
      '먹다': 14,
      '자다': 15,
      '아이': 16,
      '일': 17,
      '빨래': 18,
      '같다': 19,
      '때': 20,
      '청소': 21,
      '오다': 22,
      '가다': 23,
      '들다': 24,
      '시간': 25,
      '애': 26,
      '오늘': 27,
      '싶다': 28,
      '설거지': 29,
      '그렇다': 30,
      '좋다': 31,
      '아기': 32,
      '게': 33,
      '않다': 34,
      '전': 35,
      '아니다': 36,
      '해주다': 37,
      '이다': 38,
      '정리': 39,
      '생각': 40,
      '밥': 41,
      '더': 42,
      '말': 43,
      '분': 44,
      '신랑': 45,
      '못': 46,
      '번': 47,
      '힘들다': 48,
      '엄마': 49,
      '정도': 50,
      '고': 51,
      '해': 52,
      '하루': 53,
      '수': 54,
      '주다': 55,
      '돌리다': 56,
      '내': 57,
      '많다': 58,
      '육아': 59,
      '또': 60,
      '시키다': 61,
      '후': 62,
      '싫다': 63,
      '해도': 64,
      '아침': 65,
      '끝': 66,
      '어떻다': 67,
      '쉬다': 68,
      '놀다': 69,
      '저녁': 70,
      '왜': 71,
      '지다': 72,
      '등': 73,
      '건': 74,
      '하나': 75,
      '받다': 76,
      '나': 77,
      '주말': 78,
      '혼자': 79,
      '나다': 80,
      '지금': 81,
      '날': 82,
      '정말': 83,
      '그': 84,
      '그냥': 85,
      '애기': 86,
      '보내다': 87,
      '시작': 88,
      '맘': 89,
      '퇴근': 90,
      '이렇다': 91,
      '주': 92,
      '진짜': 93,
      '중': 94,
      '데': 95,
      '버리다': 96,
      '나가다': 97,
      '뭐': 98,
      '자기': 99,
      '사람': 100,
      '구': 101,
      '알다': 102,
      '넘다': 103,
      '치우다': 104,
      '보고': 105,
      '살': 106,
      '두': 107,
      '모르다': 108,
      '집안': 109,
      '쓰다': 110,
      '놓다': 111,
      '둘째': 112,
      '힘드다': 113,
      '개다': 114,
      '이제': 115,
      '나오다': 116,
      '아프다': 117,
      '청소기': 118,
      '걸': 119,
      '두다': 120,
      '도와주다': 121,
      '손': 122,
      '돈': 123,
      '크다': 124,
      '몸': 125,
      '요즘': 126,
      '씻다': 127,
      '다니다': 128,
      '준비': 129,
      '눕다': 130,
      '계속': 131,
      '살다': 132,
      '매일': 133,
      '되어다': 134,
      '다른': 135,
      '결혼': 136,
      '기': 137,
      '만들다': 138,
      '저희': 139,
      '옷': 140,
      '다시': 141,
      '닦다': 142,
      '끄다': 143,
      '첫째': 144,
      '해보다': 145,
      '내다': 146,
      '아들': 147,
      '개월': 148,
      '서다': 149,
      '재우다': 150,
      '시': 151,
      '둘': 152,
      '키우다': 153,
      '아빠': 154,
      '쉬': 155,
      '반찬': 156,
      '만': 157,
      '도우미': 158,
      '달': 159,
      '출근': 160,
      '세탁기': 161,
      '챙기다': 162,
      '거의': 163,
      '바쁘다': 164,
      '화장실': 165,
      '요리': 166,
      '말다': 167,
      '마음': 168,
      '글': 169,
      '몇': 170,
      '동안': 171,
      '줄': 172,
      '조금': 173,
      '때문': 174,
      '여': 175,
      '어제': 176,
      '차다': 177,
      '밀리다': 178,
      '건지다': 179,
      '끝나다': 180,
      '밤': 181,
      '넣다': 182,
      '널다': 183,
      '방': 184,
      '종일': 185,
      '눈': 186,
      '우리': 187,
      '소리': 188,
      '편하다': 189,
      '임신': 190,
      '맞다': 191,
      '하니': 192,
      '회사': 193,
      '대다': 194,
      '쓰레기': 195,
      '언제': 196,
      '물': 197,
      '돼다': 198,
      '본인': 199,
      '늦다': 200,
      '스트레스': 201,
      '계시다': 202,
      '커피': 203,
      '얘기': 204,
      '드리다': 205,
      '개': 206,
      '보이다': 207,
      '듯': 208,
      '친정': 209,
      '일주일': 210,
      '남다': 211,
      '어린이집': 212,
      '기분': 213,
      '공부': 214,
      '들어오다': 215,
      '점심': 216,
      '자고': 217,
      '일어나다': 218,
      '맞벌이': 219,
      '잠': 220,
      '좋아하다': 221,
      '딸': 222,
      '피곤하다': 223,
      '케어': 224,
      '일찍': 225,
      '책': 226,
      '잔': 227,
      '나서다': 228,
      '티비': 229,
      '이르다': 230,
      '내일': 231,
      '용': 232,
      '쌓이다': 233,
      '바로': 234,
      '끝내다': 235,
      '줄다': 236,
      '건조기': 237,
      '귀찮다': 238,
      '이번': 239,
      '낳다': 240,
      '걸다': 241,
      '걱정': 242,
      '티': 243,
      '음식': 244,
      '주부': 245,
      '반': 246,
      '먹이': 247,
      '가끔': 248,
      '모두': 249,
      '살림': 250,
      '생활': 251,
      '써다': 252,
      '새벽': 253,
      '아가': 254,
      '모': 255,
      '가지': 256,
      '앉다': 257,
      '가요': 258,
      '짜증': 259,
      '친구': 260,
      '직장': 261,
      '궁금하다': 262,
      '오전': 263,
      '차': 264,
      '고민': 265,
      '들어가다': 266,
      '달라': 267,
      '싸우다': 268,
      '중이': 269,
      '운동': 270,
      '세': 271,
      '찾다': 272,
      '항상': 273,
      '지치다': 274,
      '차리다': 275,
      '낮잠': 276,
      '벌다': 277,
      '출산': 278,
      '늘': 279,
      '알': 280,
      '거실': 281,
      '갈다': 282,
      '도움': 283,
      '땐': 284,
      '꼭': 285,
      '니': 286,
      '이모': 287,
      '병원': 288,
      '깨다': 289,
      '앞': 290,
      '늘다': 291,
      '어디': 292,
      '바닥': 293,
      '읽다': 294,
      '얼마나': 295,
      '거리': 296,
      '남자': 297,
      '괜찮다': 298,
      '자꾸': 299,
      '곳': 300,
      '잠들다': 301,
      '워킹맘': 302,
      '목욕': 303,
      '미루다': 304,
      '문제': 305,
      '기다': 306,
      '제일': 307,
      '생기다': 308,
      '남': 309,
      '하원': 310,
      '부탁드리다': 311,
      '전업': 312,
      '사다': 313,
      '정': 314,
      '너': 315,
      '뒤': 316,
      '오후': 317,
      '적': 318,
      '분리수거': 319,
      '불': 320,
      '코로나': 321,
      '잇다': 322,
      '잠깐': 323,
      '울': 324,
      '사실': 325,
      '냉장고': 326,
      '원래': 327,
      '배': 328,
      '평일': 329,
      '장난감': 330,
      '모든': 331,
      '아주': 332,
      '뭘': 333,
      '올리다': 334,
      '걸리다': 335,
      '덥다': 336,
      '옆': 337,
      '신경': 338,
      '힘': 339,
      '상황': 340,
      '여자': 341,
      '대충': 342,
      '사': 343,
      '돌보다': 344,
      '퇴': 345,
      '쓸다': 346,
      '움직이다': 347,
      '끼': 348,
      '혹시': 349,
      '그때': 350,
      '처음': 351,
      '마무리': 352,
      '듣다': 353,
      '가능하다': 354,
      '맛': 355,
      '낮': 356,
      '울다': 357,
      '가족': 358,
      '뜨다': 359,
      '이유식': 360,
      '비': 361,
      '온': 362,
      '치다': 363,
      '주방': 364,
      '부부': 365,
      '허리': 366,
      '미션': 367,
      '미안하다': 368,
      '감사하다': 369,
      '먹이다': 370,
      '밉다': 371,
      '추천': 372,
      '느낌': 373,
      '데리': 374,
      '먼저': 375,
      '시댁': 376,
      '난': 377,
      '간단하다': 378,
      '필요하다': 379,
      '뭔가': 380,
      '하나요': 381,
      '먼지': 382,
      '보': 383,
      '벌써': 384,
      '꺼내다': 385,
      '어리다': 386,
      '편': 387,
      '식사': 388,
      '통': 389,
      '부모님': 390,
      '한번': 391,
      '방법': 392,
      '멀다': 393,
      '빠지다': 394,
      '평소': 395,
      '건가': 396,
      '외': 397,
      '머리': 398,
      '겠다': 399,
      '카페': 400,
      '따르다': 401,
      '성격': 402,
      '집다': 403,
      '속': 404,
      '산': 405,
      '간식': 406,
      '막': 407,
      '자주': 408,
      '나니': 409,
      '위해': 410,
      '게임': 411,
      '위': 412,
      '정신': 413,
      '간': 414,
      '얼마': 415,
      '상태': 416,
      '어찌': 417,
      '해주시': 418,
      '식': 419,
      '당연하다': 420,
      '근무': 421,
      '갑자기': 422,
      '애가': 423,
      '다녀오다': 424,
      '리': 425,
      '술': 426,
      '잠시': 427,
      '일단': 428,
      '화': 429,
      '만나다': 430,
      '조리': 431,
      '일어나서': 432,
      '겨우': 433,
      '이야기': 434,
      '수도': 435,
      '사진': 436,
      '다음': 437,
      '잔소리': 438,
      '편이': 439,
      '가사': 440,
      '고생': 441,
      '네': 442,
      '깨끗하다': 443,
      '기도': 444,
      '떨어지다': 445,
      '이상': 446,
      '수술': 447,
      '이불': 448,
      '어머니': 449,
      '맡다': 450,
      '보통': 451,
      '젖병': 452,
      '불다': 453,
      '갖다': 454,
      '즐겁다': 455,
      '심하다': 456,
      '고맙다': 457,
      '적다': 458,
      '열': 459,
      '날씨': 460,
      '아내': 461,
      '어렵다': 462,
      '빼다': 463,
      '물건': 464,
      '작다': 465,
      '장': 466,
      '사이': 467,
      '하라': 468,
      '수건': 469,
      '친': 470,
      '끓이다': 471,
      '따다': 472,
      '부분': 473,
      '일도': 474,
      '맛있다': 475,
      '분담': 476,
      '행복하다': 477,
      '사용': 478,
      '안녕하다': 479,
      '점점': 480,
      '그대로': 481,
      '싹': 482,
      '보기': 483,
      '독박': 484,
      '볼': 485,
      '제대로': 486,
      '부르다': 487,
      '외출': 488,
      '화가': 489,
      '그릇': 490,
      '느끼다': 491,
      '명': 492,
      '첫': 493,
      '누가': 494,
      '참고': 495,
      '안다': 496,
      '연락': 497,
      '답답하다': 498,
      '차려': 499,
      '중간': 500,
      '거들다': 501,
      '및': 502,
      '인': 503,
      '터': 504,
      '모습': 505,
      '놀이': 506,
      '당': 507,
      '그것': 508,
      '구해': 509,
      '돌': 510,
      '그게': 511,
      '입': 512,
      '여기': 513,
      '몫': 514,
      '눈치': 515,
      '젠': 516,
      '돌아가다': 517,
      '노력': 518,
      '침대': 519,
      '입다': 520,
      '등원': 521,
      '서로': 522,
      '체력': 523,
      '반복': 524,
      '깔끔하다': 525,
      '지나다': 526,
      '거나': 527,
      '나름': 528,
      '댓글': 529,
      '일이': 530,
      '학교': 531,
      '여유': 532,
      '역시': 533,
      '음식물': 534,
      '하자': 535,
      '함': 536,
      '수가': 537,
      '안나': 538,
      '라면': 539,
      '사랑': 540,
      '어지르다': 541,
      '맞추다': 542,
      '다가': 543,
      '더럽다': 544,
      '동생': 545,
      '아깝다': 546,
      '대신': 547,
      '금방': 548,
      '낼': 549,
      '유': 550,
      '전혀': 551,
      '세탁': 552,
      '가득': 553,
      '참다': 554,
      '바라다': 555,
      '그리다': 556,
      '가장': 557,
      '가면': 558,
      '미리': 559,
      '들이다': 560,
      '바': 561,
      '업체': 562,
      '초': 563,
      '지내다': 564,
      '미치다': 565,
      '케': 566,
      '무': 567,
      '지르다': 568,
      '빨': 569,
      '별로': 570,
      '밖': 571,
      '화이팅': 572,
      '내내': 573,
      '담다': 574,
      '나누다': 575,
      '완전': 576,
      '무슨': 577,
      '이해': 578,
      '문': 579,
      '이후': 580,
      '잡다': 581,
      '보이': 582,
      '노': 583,
      '산더미': 584,
      '단': 585,
      '기다리다': 586,
      '약': 587,
      '베란다': 588,
      '급': 589,
      '며칠': 590,
      '다르다': 591,
      '틀다': 592,
      '절': 593,
      '바꾸다': 594,
      '뭐라다': 595,
      '빨다': 596,
      '이틀': 597,
      '그동안': 598,
      '주변': 599,
      '폰': 600,
      '발': 601,
      '엉망': 602,
      '놈': 603,
      '이사': 604,
      '머': 605,
      '우선': 606,
      '경우': 607,
      '핸드폰': 608,
      '짜증나다': 609,
      '다리': 610,
      '점': 611,
      '부탁': 612,
      '스스로': 613,
      '보여주다': 614,
      '짐': 615,
      '준': 616,
      '죽다': 617,
      '유치원': 618,
      '얼집': 619,
      '난리': 620,
      '일상': 621,
      '질': 622,
      '걸레질': 623,
      '스타일': 624,
      '절대': 625,
      '추다': 626,
      '미니': 627,
      '자신': 628,
      '매번': 629,
      '감': 630,
      '우울하다': 631,
      '나이': 632,
      '주차': 633,
      '곧': 634,
      '관리': 635,
      '원하다': 636,
      '음': 637,
      '현재': 638,
      '복직': 639,
      '가정': 640,
      '실': 641,
      '돌아오다': 642,
      '월요일': 643,
      '그만두다': 644,
      '불편하다': 645,
      '원': 646,
      '해달라다': 647,
      '열다': 648,
      '샤워': 649,
      '사고': 650,
      '세상': 651,
      '걷다': 652,
      '셋': 653,
      '소개': 654,
      '푹': 655,
      '포기': 656,
      '아무': 657,
      '띠': 658,
      '테': 659,
      '나쁘다': 660,
      '아예': 661,
      '예전': 662,
      '짜다': 663,
      '기본': 664,
      '수업': 665,
      '자마자': 666,
      '선': 667,
      '학원': 668,
      '산후': 669,
      '물어보다': 670,
      '마지막': 671,
      '이유': 672,
      '덜': 673,
      '쪽': 674,
      '잘못': 675,
      '비우다': 676,
      '부지런하다': 677,
      '얼른': 678,
      '달다': 679,
      '나중': 680,
      '안일': 681,
      '순간': 682,
      '젤': 683,
      '덕분': 684,
      '타다': 685,
      '바람': 686,
      '싫어하다': 687,
      '솔직하다': 688,
      '터지다': 689,
      '이건': 690,
      '부터': 691,
      '오시': 692,
      '자리': 693,
      '여름': 694,
      '주시': 695,
      '와이프': 696,
      '완료': 697,
      '저리': 698,
      '정보': 699,
      '쇼파': 700,
      '손목': 701,
      '알아보다': 702,
      '로': 703,
      '지나가다': 704,
      '집중': 705,
      '육': 706,
      '배달': 707,
      '속상하다': 708,
      '요새': 709,
      '삶다': 710,
      '박': 711,
      '입덧': 712,
      '몰다': 713,
      '안해': 714,
      '년': 715,
      '해봤다': 716,
      '심': 717,
      '이용': 718,
      '크게': 719,
      '참여': 720,
      '찌': 721,
      '휴가': 722,
      '빠르다': 723,
      '예': 724,
      '설': 725,
      '마시다': 726,
      '붙다': 727,
      '힘내다': 728,
      '태어나다': 729,
      '기간': 730,
      '거기': 731,
      '가야': 732,
      '햇': 733,
      '찍다': 734,
      '해결': 735,
      '예쁘다': 736,
      '대해': 737,
      '싸다': 738,
      '요일': 739,
      '대요': 740,
      '급하다': 741,
      '상': 742,
      '새롭다': 743,
      '접다': 744,
      '대출': 745,
      '꼴': 746,
      '불만': 747,
      '예정': 748,
      '산책': 749,
      '언니': 750,
      '수유': 751,
      '욕': 752,
      '교육': 753,
      '조언': 754,
      '노래': 755,
      '월': 756,
      '관심': 757,
      '신': 758,
      '식구': 759,
      '키': 760,
      '삶': 761,
      '월급': 762,
      '시원하다': 763,
      '오니': 764,
      '마': 765,
      '칭찬': 766,
      '냄새': 767,
      '대화': 768,
      '별': 769,
      '싱크대': 770,
      '그거': 771,
      '김치': 772,
      '땀': 773,
      '구합': 774,
      '욕실': 775,
      '여러': 776,
      '국': 777,
      '핑계': 778,
      '기저귀': 779,
      '대한': 780,
      '지도': 781,
      '마트': 782,
      '펴다': 783,
      '좋아지다': 784,
      '활동': 785,
      '어머님': 786,
      '대부분': 787,
      '개인': 788,
      '전화': 789,
      '예민하다': 790,
      '어른': 791,
      '조용하다': 792,
      '직업': 793,
      '먹기': 794,
      '남아': 795,
      '입원': 796,
      '법': 797,
      '오': 798,
      '재료': 799,
      '만큼': 800,
      '느껴지다': 801,
      '양말': 802,
      '일요일': 803,
      '쯤': 804,
      '나머지': 805,
      '벌이': 806,
      '정돈': 807,
      '날다': 808,
      '버': 809,
      '마르다': 810,
      '숨': 811,
      '널': 812,
      '올해': 813,
      '육아휴직': 814,
      '대단하다': 815,
      '빵': 816,
      '돌다': 817,
      '시기': 818,
      '길': 819,
      '차라리': 820,
      '재활용': 821,
      '직접': 822,
      '부': 823,
      '즐기다': 824,
      '야근': 825,
      '들어서다': 826,
      '때리다': 827,
      '신생아': 828,
      '땜': 829,
      '질문': 830,
      '무겁다': 831,
      '버티다': 832,
      '알바': 833,
      '슬슬': 834,
      '우울증': 835,
      '명절': 836,
      '틈': 837,
      '게으르다': 838,
      '사오다': 839,
      '이구': 840,
      '부족하다': 841,
      '식기세척기': 842,
      '퇴원': 843,
      '감기': 844,
      '털다': 845,
      '잠도': 846,
      '식탁': 847,
      '깔다': 848,
      '모으다': 849,
      '눈물': 850,
      '얼굴': 851,
      '계시': 852,
      '안방': 853,
      '올라오다': 854,
      '이면': 855,
      '거지': 856,
      '겨울': 857,
      '남기다': 858,
      '재우': 859,
      '고프다': 860,
      '딱지': 861,
      '지난': 862,
      '약속': 863,
      '환기': 864,
      '주로': 865,
      '줍다': 866,
      '계획': 867,
      '와중': 868,
      '로봇청소기': 869,
      '짧다': 870,
      '선생님': 871,
      '새': 872,
      '가게': 873,
      '달래다': 874,
      '업다': 875,
      '토요일': 876,
      '무조건': 877,
      '취미': 878,
      '맡기다': 879,
      '가보다': 880,
      '다행': 881,
      '아파트': 882,
      '관련': 883,
      '진': 884,
      '통증': 885,
      '여행': 886,
      '오빠': 887,
      '후기': 888,
      '면': 889,
      '연휴': 890,
      '선물': 891,
      '만들기': 892,
      '분유': 893,
      '최대한': 894,
      '표': 895,
      '쌓다': 896,
      '물걸레': 897,
      '풀다': 898,
      '담당': 899,
      '전부': 900,
      '생활비': 901,
      '얼': 902,
      '치': 903,
      '놀': 904,
      '자라다': 905,
      '경험': 906,
      '부엌': 907,
      '붙이다': 908,
      '신나다': 909,
      '한숨': 910,
      '확': 911,
      '인지': 912,
      '누구': 913,
      '최근': 914,
      '구매': 915,
      '이제야': 916,
      '암': 917,
      '웃다': 918,
      '깨우다': 919,
      '현실': 920,
      '뿐': 921,
      '척': 922,
      '옮기다': 923,
      '소독': 924,
      '손가락': 925,
      '바뀌다': 926,
      '오기': 927,
      '찌다': 928,
      '신발': 929,
      '여기저기': 930,
      '건강': 931,
      '안고': 932,
      '켜다': 933,
      '혼': 934,
      '처리': 935,
      '오히려': 936,
      '뒤지다': 937,
      '확인': 938,
      '서운하다': 939,
      '나르다': 940,
      '병': 941,
      '막내': 942,
      '다닥': 943,
      '던지다': 944,
      '자랑': 945,
      '배고프다': 946,
      '사회': 947,
      '느리다': 948,
      '댁': 949,
      '업무': 950,
      '대로': 951,
      '마을': 952,
      '디': 953,
      '일과': 954,
      '토': 955,
      '냉동': 956,
      '마치': 957,
      '걸레': 958,
      '줄이다': 959,
      '과일': 960,
      '껌': 961,
      '생기': 962,
      '똑같다': 963,
      '배우다': 964,
      '잡고': 965,
      '뿌듯하다': 966,
      '프로': 967,
      '데리다': 968,
      '목': 969,
      '이혼': 970,
      '가기': 971,
      '채우다': 972,
      '살짝': 973,
      '무섭다': 974,
      '면서': 975,
      '주문': 976,
      '씻기다': 977,
      '말씀': 978,
      '관계': 979,
      '세척': 980,
      '밑': 981,
      '행동': 982,
      '중요하다': 983,
      '아시': 984,
      '돕다': 985,
      '각자': 986,
      '채': 987,
      '건조': 988,
      '답': 989,
      '예약': 990,
      '비슷하다': 991,
      '양': 992,
      '자르다': 993,
      '휴직': 994,
      '피': 995,
      '넘어가다': 996,
      '장난': 997,
      '용돈': 998,
      '번은': 999,
      '깨': 1000,
      ...})




```python
df_pad = tokenizer.texts_to_sequences(df.okt2) 
```


```python
print(df_pad[0])
print(df.okt2.iloc[0])
```

    [1, 20, 3, 148, 32, 153, 2, 1504, 89, 7009, 49, 961, 861, 1320, 339, 24, 165, 3615, 46, 23, 16, 875, 987, 41, 14, 1, 2, 366, 3350, 11, 19, 47, 213, 31, 207, 20, 1321, 1965, 19, 119, 55, 7963, 2931, 229, 1116, 515, 7, 1, 2, 77, 189, 86, 1046, 2, 11, 19, 404, 1449, 368, 617, 9, 1341, 421, 53, 185, 79, 16, 7, 82, 58, 9, 1, 55, 2, 90, 22, 1, 11, 7, 2036, 582, 2758, 93, 417, 108, 1, 66, 10, 16, 485, 435, 10, 786, 386, 16, 2272, 67, 1, 250, 1504, 166, 29, 25, 335, 3, 59, 250, 1047, 311, 27, 53, 441, 786, 249, 1671, 477, 181, 6]
    ['집안일', '때', '요', '개월', '아기', '키우다', '있다', '초보', '맘', '심각', '엄마', '껌', '딱지', '하루하루', '힘', '들다', '화장실', '마음대로', '못', '가다', '아이', '업다', '채', '밥', '먹다', '집안일', '있다', '허리', '끊어지다', '것', '같다', '번', '기분', '좋다', '보이다', '때', '과자', '치즈', '같다', '걸', '주다', '호비다', '뽀로로', '티비', '틀어주다', '눈치', '보다', '집안일', '있다', '나', '편하다', '애기', '방치', '있다', '것', '같다', '속', '상하', '미안하다', '죽다', '남편', '교대', '근무', '하루', '종일', '혼자', '아이', '보다', '날', '많다', '남편', '집안일', '주다', '있다', '퇴근', '오다', '집안일', '것', '보다', '짠하다', '보이', '미안', '진짜', '어찌', '모르다', '집안일', '끝', '없다', '아이', '볼', '수도', '없다', '어머님', '어리다', '아이', '두시', '어떻다', '집안일', '살림', '초보', '요리', '설거지', '시간', '걸리다', '요', '육아', '살림', '팁', '부탁드리다', '오늘', '하루', '고생', '어머님', '모두', '편안하다', '행복하다', '밤', '되다']
    


```python

```


```python
# 딥러닝 모델에 넣을 용도로 사전에 훈련시킨 워드임베딩 데이터를 불러옵니다

import os
embedding_dict = {}
f = open(os.path.join('', 'imdb_embedding_word2vec.txt'),  encoding = "utf-8")
for line in f: # 각 line은 단어, 임베딩백터값으로 구성된 하나의 문자열
    values = line.split() # [단어, 벡터값] 리스트 형성
    word = values[0] # 단어
    coefs = np.asarray(values[1:]) # 벡터
    embedding_dict[word] = coefs # key:단어 value:벡터
f.close()

embedding_dict
# 신경망에 사용할 embedding matrix 생성
embedding_matrix = np.zeros((len(embedding_dict), EMBEDDING_DIM))

# 여기서 word_index에선 OOV가 1입니다 
# word는 단어  i는 단어와 대응되는 정수토큰입니다 (숫자가 작을수록 빈도가 높습니다)
for word, i in word_index.items(): 
    if i >= len(embedding_dict): 
        continue      
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None: # get했는데 없으면 None 돌려줌
        embedding_matrix[i] = embedding_vector
print(np.shape(embedding_matrix))
print(embedding_matrix)


```

    (20355, 20)
    [[ 0.          0.          0.         ...  0.          0.
       0.        ]
     [-0.16928725 -0.38423637 -0.29523185 ... -0.12859762 -0.18557906
       1.0415484 ]
     [-0.27125937 -0.93488383  0.15244912 ... -0.00646071 -0.29867738
       0.27224615]
     ...
     [ 0.13503546 -0.05490058  0.06086274 ... -0.04782549 -0.08455915
       0.15241295]
     [ 0.10293397 -0.04439397  0.07568203 ...  0.02268357 -0.13813876
       0.19198692]
     [ 0.05035463 -0.03426788  0.04177142 ...  0.03854343 -0.09971406
       0.16610143]]
    


```python
def doc_vectors(padding):
    document_embedding_list = []

    # 각 문장은 리스트의 리스트 형태로 숫자인코딩 된 벡터
    for line in tqdm(padding):
        doc2vec = np.zeros(EMBEDDING_DIM) # 0값의 벡터를 만들어줍니다
        count = 0 # 평균을 내기위해 count해줍니다
        for token in line: # 정수토큰 하나하나를 가져옵니다                    
            if token in np.arange(1,len(embedding_matrix)):  # 제로패딩은 count안하게, Vocab_size까지만
                count += 1 
                # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
                if doc2vec is np.zeros(EMBEDDING_DIM): # 첫번째 단어때 필요한 문법
                    doc2vec = embedding_matrix[token]
                else:
                    doc2vec = doc2vec + embedding_matrix[token] # 단어의 w2v를 누적해서 더해줍니다
        
        # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠줍니다 = 문장 벡터
        doc2vec_average = doc2vec / (count+1) # 혹시나 있을 zero-divdend방지위해 +1
        document_embedding_list.append(doc2vec_average)

    # 각 문서에 대한 문서 벡터 리스트를 리턴
    return document_embedding_list
```


```python
len(embedding_dict)
```




    20355




```python
document_vectors = doc_vectors(df_pad)
```


```python
# 각 문장을 단어평균 임베딩 벡터로
len(document_vectors),document_vectors[:10]
```




    (9744,
     [array([-0.30841802, -0.64408356, -0.06304013,  0.16231523, -0.19571024,
             -0.46331818,  0.60492002, -0.37063423,  0.21129707, -0.33374631,
             -0.58401643,  1.22808931, -0.61525249,  0.01036064,  0.26744741,
             -0.07820513, -0.26848464, -0.16089978, -0.14429994,  0.65267741]),
      array([-0.35937723, -0.51816204, -0.01897925,  0.23495836, -0.17816657,
             -0.70176093,  0.57075847, -0.4339566 ,  0.37274778, -0.24908425,
             -0.50293126,  1.23486464, -0.56522872,  0.11346303,  0.32748221,
              0.03610096, -0.37535054, -0.20250559, -0.12517143,  0.65534017]),
      array([-0.30435738, -0.62370497, -0.03362386,  0.12712914, -0.15082603,
             -0.31815526,  0.45598101, -0.21283865,  0.26332451, -0.42601449,
             -0.72371632,  1.08885231, -0.67842232,  0.16799577,  0.47460111,
              0.1077486 , -0.21475624, -0.01316405, -0.03595221,  0.74349066]),
      array([-0.12391867, -0.63254468, -0.12171742,  0.22674671, -0.17630015,
             -0.51544739,  0.54455807, -0.31627534,  0.04955438, -0.57491122,
             -0.7718713 ,  1.02134304, -0.48233506, -0.10714164,  0.15512631,
              0.05542934, -0.44639839, -0.35040074,  0.08972221,  0.66994542]),
      array([-0.19270844, -0.6422484 , -0.04924866,  0.1171705 , -0.18609954,
             -0.60172939,  0.60353468, -0.33768489,  0.1134403 , -0.45542341,
             -0.50245186,  1.25791151, -0.34239712, -0.03048058,  0.24073335,
             -0.04405602, -0.35972974, -0.25658316, -0.03956194,  0.66104036]),
      array([ 0.06439594, -0.68440583, -0.24444291,  0.16179899, -0.38286098,
             -0.45941731,  0.41519458, -0.40026134,  0.08630832, -0.61287317,
             -0.62009877,  0.88440305, -0.35826429,  0.13358568,  0.0580112 ,
              0.19021729, -0.38376685, -0.31857429,  0.0763792 ,  0.60555095]),
      array([-0.19122546, -0.59824506, -0.14590565,  0.08504838, -0.38671714,
             -0.50034919,  0.47285955, -0.30775543,  0.0583377 , -0.37427921,
             -0.67445577,  1.14996425, -0.25003722, -0.15296308,  0.14477599,
             -0.00734518, -0.4064796 , -0.25420474, -0.05781135,  0.58273361]),
      array([-0.13353851, -0.73962078, -0.11882637,  0.08252359, -0.23982525,
             -0.34654653,  0.55181998, -0.29101037,  0.24373327, -0.30898257,
             -0.67896081,  1.17802882, -0.67938352,  0.14126455,  0.38462767,
              0.10257047, -0.16217096, -0.13874905, -0.00914745,  0.64347942]),
      array([-0.12983258, -0.59370447, -0.0426909 ,  0.21723088, -0.21591683,
             -0.49879073,  0.60312984, -0.37339648,  0.01821897, -0.54461183,
             -0.70586644,  0.99778637, -0.55677259, -0.04528822,  0.20167376,
             -0.04867792, -0.42205863, -0.21251468,  0.01139006,  0.6685407 ]),
      array([-0.30217264, -0.69413194, -0.2396161 ,  0.12851381, -0.2133929 ,
             -0.63494847,  0.57463769, -0.46975425,  0.2493232 , -0.36490133,
             -0.62586002,  1.17733538, -0.46926311, -0.05938631,  0.3198815 ,
              0.02522496, -0.30263724, -0.22838753, -0.03514657,  0.63682551])])




```python
# 시각화가 필요없다면 밑의 사이킷런 코드로 바로 내려가도 될 것 같습니다
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

```


```python
linked = linkage(document_vectors, 'ward')
plt.figure(figsize=(15, 9))
dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
```


```python
# 시각화가 필요없을 때 여기서 바로 시작합니다
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import silhouette_score

def visualize_silhouette_layer(data, num_cluster):
    clusters_range = range(2,int(num_cluster))
    results = []

    for i in tqdm(clusters_range):
        clusterer = AgglomerativeClustering(n_clusters=i,linkage='ward')
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        results.append([i, silhouette_avg])

    result = pd.DataFrame(results, columns=["n_clusters", "silhouette_score"])
    pivot_ac = pd.pivot_table(result, index="n_clusters", values="silhouette_score")

    return result, pivot_ac
```


```python
result, pivot_ac = visualize_silhouette_layer(document_vectors,100)
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_clusters</th>
      <th>silhouette_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.120931</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.129035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0.104097</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.105292</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.075588</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>93</th>
      <td>95</td>
      <td>0.016359</td>
    </tr>
    <tr>
      <th>94</th>
      <td>96</td>
      <td>0.016847</td>
    </tr>
    <tr>
      <th>95</th>
      <td>97</td>
      <td>0.016859</td>
    </tr>
    <tr>
      <th>96</th>
      <td>98</td>
      <td>0.017395</td>
    </tr>
    <tr>
      <th>97</th>
      <td>99</td>
      <td>0.017299</td>
    </tr>
  </tbody>
</table>
<p>98 rows × 2 columns</p>
</div>




```python

```


```python
result.to_csv('실루엣결과review.csv', encoding='utf-8-sig',mode='a')
```


```python
plt.plot(result.n_clusters, result.silhouette_score)
```




    [<matplotlib.lines.Line2D at 0x11d58db6048>]




    
![png](output_42_1.png)
    



```python
# 이건 하이퍼파라미터 도출용 (논문에 테이블로 들어갈거임)
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

벡터크기 = [10,30]
윈도우 = [3,5]
for EMBEDDING_DIM in 벡터크기:
    for WINDOW_SIZE in 윈도우:
        # 임베딩 크기는 논문을 따름
        model = Word2Vec(sentences=df.okt2, sg=1, size=EMBEDDING_DIM, window=WINDOW_SIZE, min_count=1) #sg 0은 CBOW, 1은 SKIP-GRAM
        w2v_vocab = list(model.wv.vocab) # 임베딩 된 단어 리스트

        # save model in ASCII (word2vec) format
        # 텍스트 파일로 단어들의 임베딩 벡터 저장
        filename = 'imdb_embedding_word2vec.txt'
        model.wv.save_word2vec_format(filename, binary=False)

        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df.okt2) # train 데이터
        word_index = tokenizer.word_index

        df_pad = tokenizer.texts_to_sequences(df.okt2) 

        
        embedding_dict = {}
        f = open(os.path.join('', 'imdb_embedding_word2vec.txt'),  encoding = "utf-8")
        for line in f: # 각 line은 단어, 임베딩백터값으로 구성된 하나의 문자열
            values = line.split() # [단어, 벡터값] 리스트 형성
            word = values[0] # 단어
            coefs = np.asarray(values[1:]) # 벡터
            embedding_dict[word] = coefs # key:단어 value:벡터
        f.close()

        embedding_dict
        # 신경망에 사용할 embedding matrix 생성
        embedding_matrix = np.zeros((len(embedding_dict), EMBEDDING_DIM))

        # 여기서 word_index에선 OOV가 1입니다 
        # word는 단어  i는 단어와 대응되는 정수토큰입니다 (숫자가 작을수록 빈도가 높습니다)
        for word, i in word_index.items(): 
            if i >= len(embedding_dict): 
                continue      
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None: # get했는데 없으면 None 돌려줌
                embedding_matrix[i] = embedding_vector

        document_vectors = doc_vectors(df_pad)

        result, pivot_ac = visualize_silhouette_layer(document_vectors,5)
        result

        pd.DataFrame(['벡터크기:{} , 윈도우:{}'.format(EMBEDDING_DIM,WINDOW_SIZE)]).to_csv('본문_하이퍼파라미터_TEST버젼.csv', encoding='utf-8-sig',mode='a',header=False)
        result.T.to_csv('본문_하이퍼파라미터_TEST버젼.csv', encoding='utf-8-sig',mode='a',header=False)
```

    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:22<00:00, 427.09it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:32<00:00, 295.58it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:23<00:00, 422.42it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:23<00:00, 409.87it/s]
    


```python
# 하이퍼파라미터 도출용
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# 벡터크기 = [10,30,100,300,500]
# 윈도우 = [3,5,10,15,20]
벡터크기 = [10,30,100,300,500]
윈도우 = [3,5,10,15,20]
for EMBEDDING_DIM in 벡터크기:
    for WINDOW_SIZE in 윈도우:
    
        # 임베딩 크기는 논문을 따름
        model = Word2Vec(sentences=df.okt2, sg=1, size=EMBEDDING_DIM, window=WINDOW_SIZE, min_count=1) #sg 0은 CBOW, 1은 SKIP-GRAM
        w2v_vocab = list(model.wv.vocab) # 임베딩 된 단어 리스트

        # save model in ASCII (word2vec) format
        # 텍스트 파일로 단어들의 임베딩 벡터 저장
        filename = 'imdb_embedding_word2vec.txt'
        model.wv.save_word2vec_format(filename, binary=False)

        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df.okt2) # train 데이터
        word_index = tokenizer.word_index

        df_pad = tokenizer.texts_to_sequences(df.okt2) 

        
        embedding_dict = {}
        f = open(os.path.join('', 'imdb_embedding_word2vec.txt'),  encoding = "utf-8")
        for line in f: # 각 line은 단어, 임베딩백터값으로 구성된 하나의 문자열
            values = line.split() # [단어, 벡터값] 리스트 형성
            word = values[0] # 단어
            coefs = np.asarray(values[1:]) # 벡터
            embedding_dict[word] = coefs # key:단어 value:벡터
        f.close()

        embedding_dict
        # 신경망에 사용할 embedding matrix 생성
        embedding_matrix = np.zeros((len(embedding_dict), EMBEDDING_DIM))

        # 여기서 word_index에선 OOV가 1입니다 
        # word는 단어  i는 단어와 대응되는 정수토큰입니다 (숫자가 작을수록 빈도가 높습니다)
        for word, i in word_index.items(): 
            if i >= len(embedding_dict): 
                continue      
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None: # get했는데 없으면 None 돌려줌
                embedding_matrix[i] = embedding_vector

        document_vectors = doc_vectors(df_pad)

        result, pivot_ac = visualize_silhouette_layer(document_vectors,31)
        result

        pd.DataFrame(['벡터크기:{} , 윈도우:{}'.format(EMBEDDING_DIM,WINDOW_SIZE)]).to_csv('본문_하이퍼파라미터.csv', encoding='utf-8-sig',mode='a')
        result.T.to_csv('본문_하이퍼파라미터.csv', encoding='utf-8-sig',mode='a',header=False)
```

    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:41<00:00, 232.67it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:34<00:00, 279.94it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:35<00:00, 276.53it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:45<00:00, 215.56it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:54<00:00, 179.91it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:22<00:00, 437.08it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:20<00:00, 471.10it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:21<00:00, 458.29it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:24<00:00, 402.61it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:49<00:00, 198.35it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:37<00:00, 258.72it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:36<00:00, 264.21it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:40<00:00, 238.42it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:41<00:00, 235.46it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:38<00:00, 253.87it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:38<00:00, 252.01it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:42<00:00, 227.86it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:34<00:00, 279.47it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:39<00:00, 244.99it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:37<00:00, 257.91it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:41<00:00, 237.40it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:23<00:00, 415.40it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:34<00:00, 281.33it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:51<00:00, 189.72it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:26<00:00, 362.82it/s]
    


```python
def visualize_silhouette(cluster_lists, X_features): 

    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math

    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)

    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)

    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):

        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
#         clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        clusterer = AgglomerativeClustering(n_clusters=n_cluster,linkage='ward')
        cluster_labels = clusterer.fit_predict(X_features)

        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)

        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")
```


```python
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

벡터크기 = [10]
윈도우 = [20]
for EMBEDDING_DIM in 벡터크기:
    for WINDOW_SIZE in 윈도우:
        model = Word2Vec(sentences=df.okt2, sg=1, size=EMBEDDING_DIM, window=WINDOW_SIZE, min_count=1) #sg 0은 CBOW, 1은 SKIP-GRAM
        w2v_vocab = list(model.wv.vocab) # 임베딩 된 단어 리스트

        # save model in ASCII (word2vec) format
        # 텍스트 파일로 단어들의 임베딩 벡터 저장
        filename = 'imdb_embedding_word2vec.txt'
        model.wv.save_word2vec_format(filename, binary=False)

        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df.okt2) # train 데이터
        word_index = tokenizer.word_index

        df_pad = tokenizer.texts_to_sequences(df.okt2) 

        
        embedding_dict = {}
        f = open(os.path.join('', 'imdb_embedding_word2vec.txt'),  encoding = "utf-8")
        for line in f: # 각 line은 단어, 임베딩백터값으로 구성된 하나의 문자열
            values = line.split() # [단어, 벡터값] 리스트 형성
            word = values[0] # 단어
            coefs = np.asarray(values[1:]) # 벡터
            embedding_dict[word] = coefs # key:단어 value:벡터
        f.close()

        embedding_dict
        # 신경망에 사용할 embedding matrix 생성
        embedding_matrix = np.zeros((len(embedding_dict), EMBEDDING_DIM))

        # 여기서 word_index에선 OOV가 1입니다 
        # word는 단어  i는 단어와 대응되는 정수토큰입니다 (숫자가 작을수록 빈도가 높습니다)
        for word, i in word_index.items(): 
            if i >= len(embedding_dict): 
                continue      
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None: # get했는데 없으면 None 돌려줌
                embedding_matrix[i] = embedding_vector

        document_vectors = doc_vectors(df_pad)
```

    100%|█████████████████████████████████████████████████████████████████████████████| 9744/9744 [00:19<00:00, 505.12it/s]
    


```python
# 위의 실루엣에 따른 적정 ccluster개수를 선정해 아래 n_clusters를 조정합니다
model = AgglomerativeClustering(n_clusters=3,linkage='ward')
cluster_info = model.fit_predict(document_vectors)
```


```python
vis = pd.DataFrame(pd.Series(cluster_info).value_counts().values).reset_index()
vis.columns = ['cluster','num_review']
vis
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster</th>
      <th>num_review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4740</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4318</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>686</td>
    </tr>
  </tbody>
</table>
</div>




```python
vis.num_review/np.sum(vis.num_review.values)
```




    0    0.486453
    1    0.443144
    2    0.070402
    Name: num_review, dtype: float64




```python
visualize_silhouette([ 2, 3, 4, 5, 6], document_vectors)
```


    
![png](output_50_0.png)
    



```python
df['cluster_review3'] = cluster_info
```


```python
from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter
from PIL import Image
```


```python
df.columns
```




    Index(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'review', 'after',
           'pos', 'cluster', 'comment', 'date', 'review_len', 'review_senti',
           'review_senti_mean', 'okt_pos', 'com_okt_pos', 'com_num',
           'com_len_mean', 'com_len_std', 'com_senti_dist', 'com_senti',
           'com_senti_mean', 'com_senti_std', 'year', 'month', 'Unnamed: 23',
           'Unnamed: 24', 'Unnamed: 25', 'okt2', 'cluster_review3'],
          dtype='object')




```python
word0 = []
for i in df['okt2'][df['cluster_review3']==0]:
    word0.extend(i)
word00 = Counter(word0)
```


```python
word00.pop('있다')  
word00.pop('요')  
word00.pop('거')
word00.pop('것')
word00.pop('니')
# word00.pop('저')
word00.pop('제')
```




    1117




```python
word00.pop('집안일')
```




    8020




```python
' 이거 보면서 어떤 stopwords 빼서 시각화할지 결정'
aa = pd.DataFrame.from_records(list(dict(word00).items()), columns=['c0_term','c0_count'])
aa.sort_values('c0_count', ascending=False).iloc[0:40]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c0_term</th>
      <th>c0_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>집안일</td>
      <td>7214</td>
    </tr>
    <tr>
      <th>58</th>
      <td>있다</td>
      <td>2718</td>
    </tr>
    <tr>
      <th>7</th>
      <td>빨래</td>
      <td>2534</td>
    </tr>
    <tr>
      <th>115</th>
      <td>요</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>176</th>
      <td>청소</td>
      <td>1923</td>
    </tr>
    <tr>
      <th>146</th>
      <td>먹다</td>
      <td>1895</td>
    </tr>
    <tr>
      <th>51</th>
      <td>없다</td>
      <td>1872</td>
    </tr>
    <tr>
      <th>71</th>
      <td>안</td>
      <td>1775</td>
    </tr>
    <tr>
      <th>48</th>
      <td>거</td>
      <td>1766</td>
    </tr>
    <tr>
      <th>256</th>
      <td>오늘</td>
      <td>1683</td>
    </tr>
    <tr>
      <th>13</th>
      <td>보다</td>
      <td>1587</td>
    </tr>
    <tr>
      <th>112</th>
      <td>되다</td>
      <td>1523</td>
    </tr>
    <tr>
      <th>162</th>
      <td>설거지</td>
      <td>1394</td>
    </tr>
    <tr>
      <th>4</th>
      <td>것</td>
      <td>1390</td>
    </tr>
    <tr>
      <th>296</th>
      <td>집</td>
      <td>1386</td>
    </tr>
    <tr>
      <th>270</th>
      <td>정리</td>
      <td>1363</td>
    </tr>
    <tr>
      <th>50</th>
      <td>끝</td>
      <td>1309</td>
    </tr>
    <tr>
      <th>118</th>
      <td>돌리다</td>
      <td>1275</td>
    </tr>
    <tr>
      <th>33</th>
      <td>제</td>
      <td>1263</td>
    </tr>
    <tr>
      <th>132</th>
      <td>들다</td>
      <td>1099</td>
    </tr>
    <tr>
      <th>179</th>
      <td>해도</td>
      <td>1088</td>
    </tr>
    <tr>
      <th>350</th>
      <td>같다</td>
      <td>1018</td>
    </tr>
    <tr>
      <th>572</th>
      <td>남편</td>
      <td>944</td>
    </tr>
    <tr>
      <th>288</th>
      <td>일</td>
      <td>939</td>
    </tr>
    <tr>
      <th>79</th>
      <td>좋다</td>
      <td>936</td>
    </tr>
    <tr>
      <th>53</th>
      <td>자다</td>
      <td>928</td>
    </tr>
    <tr>
      <th>90</th>
      <td>시간</td>
      <td>924</td>
    </tr>
    <tr>
      <th>129</th>
      <td>오다</td>
      <td>915</td>
    </tr>
    <tr>
      <th>393</th>
      <td>가다</td>
      <td>896</td>
    </tr>
    <tr>
      <th>55</th>
      <td>저</td>
      <td>842</td>
    </tr>
    <tr>
      <th>3</th>
      <td>싫다</td>
      <td>835</td>
    </tr>
    <tr>
      <th>139</th>
      <td>때</td>
      <td>816</td>
    </tr>
    <tr>
      <th>588</th>
      <td>아이</td>
      <td>801</td>
    </tr>
    <tr>
      <th>5</th>
      <td>전</td>
      <td>785</td>
    </tr>
    <tr>
      <th>70</th>
      <td>싶다</td>
      <td>761</td>
    </tr>
    <tr>
      <th>65</th>
      <td>밥</td>
      <td>743</td>
    </tr>
    <tr>
      <th>195</th>
      <td>게</td>
      <td>725</td>
    </tr>
    <tr>
      <th>91</th>
      <td>아침</td>
      <td>720</td>
    </tr>
    <tr>
      <th>97</th>
      <td>애</td>
      <td>713</td>
    </tr>
    <tr>
      <th>477</th>
      <td>시작</td>
      <td>694</td>
    </tr>
  </tbody>
</table>
</div>




```python
impath = '구름.jpg'
mask_im = np.array(Image.open(impath))
mask_color = ImageColorGenerator(mask_im)
```


```python
wc = WordCloud(background_color='white',
              width=1020, height=680,
              font_path='NanumGothic.ttf',
               max_font_size=120
              ,max_words=100, mask=mask_im,
               stopwords=set(['집안일'])
              ).generate_from_frequencies(word00)
fig=plt.figure(figsize=(10,10))
plt.imshow(wc.recolor(color_func=mask_color), interpolation='bilinear',cmap='YlOrBr')
plt.axis('off')
```




    (-0.5, 919.5, 919.5, -0.5)




    
![png](output_59_1.png)
    



```python
df
```




    Index(['Unnamed: 0', 'review', 'after', 'pos', 'cluster', 'comment', 'date',
           'review_len', 'review_senti', 'review_senti_mean', 'okt_pos',
           'com_okt_pos', 'com_num', 'com_len_mean', 'com_len_std',
           'com_senti_dist', 'com_senti', 'com_senti_mean', 'com_senti_std',
           'year', 'month', 'okt2', 'cluster_review6', 'cluster_review3'],
          dtype='object')




```python
df[['review_len','review_senti','review_senti_mean','com_num','com_senti','com_senti_mean']].corr(method='pearson')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review_len</th>
      <th>review_senti</th>
      <th>review_senti_mean</th>
      <th>com_num</th>
      <th>com_senti</th>
      <th>com_senti_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>review_len</th>
      <td>1.000000</td>
      <td>-0.361823</td>
      <td>0.009370</td>
      <td>0.198367</td>
      <td>-0.272109</td>
      <td>-0.184668</td>
    </tr>
    <tr>
      <th>review_senti</th>
      <td>-0.361823</td>
      <td>1.000000</td>
      <td>0.686628</td>
      <td>-0.095265</td>
      <td>0.193208</td>
      <td>0.157897</td>
    </tr>
    <tr>
      <th>review_senti_mean</th>
      <td>0.009370</td>
      <td>0.686628</td>
      <td>1.000000</td>
      <td>-0.013121</td>
      <td>0.046560</td>
      <td>0.056766</td>
    </tr>
    <tr>
      <th>com_num</th>
      <td>0.198367</td>
      <td>-0.095265</td>
      <td>-0.013121</td>
      <td>1.000000</td>
      <td>-0.604449</td>
      <td>-0.100036</td>
    </tr>
    <tr>
      <th>com_senti</th>
      <td>-0.272109</td>
      <td>0.193208</td>
      <td>0.046560</td>
      <td>-0.604449</td>
      <td>1.000000</td>
      <td>0.539959</td>
    </tr>
    <tr>
      <th>com_senti_mean</th>
      <td>-0.184668</td>
      <td>0.157897</td>
      <td>0.056766</td>
      <td>-0.100036</td>
      <td>0.539959</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python
df0 = df[df['cluster_review3']==0]
```


```python
df0.com_num
```




    3       30
    4       10
    6        3
    8       57
    10      20
            ..
    9731    21
    9732    32
    9739    10
    9740    21
    9742    20
    Name: com_num, Length: 4318, dtype: int64




```python
df0['com_senti_mean'].min()
```




    -2.0




```python
imp = []

for i in range(len(df0)):
    imp.append(10*((df0['com_num'].iloc[i]-df0['com_num'].min()) / (df0['com_num'].max() - df0['com_num'].min())))
df0['imp'] = imp

sat = []
for i in range(len(df0)):
    sat.append(10*((df0['com_senti_mean'].iloc[i]-df0['com_senti_mean'].min()) / (df0['com_senti_mean'].max() - df0['com_senti_mean'].min())))
df0['sat'] = sat


opt = []
for i in range(len(df0)):
    
    차 = df0['imp'].iloc[i] - df0['sat'].iloc[i]
    if 차 > 0:
        opt.append(df0['imp'].iloc[i]+차)
    else:
        opt.append(df0['imp'].iloc[i])
        
df0['opt'] = opt
```

    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\ipykernel_launcher.py:22: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    


```python
df0[['imp','sat','opt']].iloc[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imp</th>
      <th>sat</th>
      <th>opt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>2.929293</td>
      <td>4.583333</td>
      <td>2.929293</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.909091</td>
      <td>2.875000</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.707071</td>
      <td>5.640625</td>
      <td>0.707071</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.202020</td>
      <td>7.500000</td>
      <td>0.202020</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5.656566</td>
      <td>3.690789</td>
      <td>7.622342</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.919192</td>
      <td>5.875000</td>
      <td>1.919192</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.606061</td>
      <td>4.285714</td>
      <td>0.606061</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.303030</td>
      <td>8.750000</td>
      <td>0.303030</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.404040</td>
      <td>5.916667</td>
      <td>0.404040</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.808081</td>
      <td>4.444444</td>
      <td>0.808081</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
sns.distplot(df['com_num'])
```

    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    




    <AxesSubplot:xlabel='com_num', ylabel='Density'>




    
![png](output_69_2.png)
    



```python
plt.hist(df1['com_num'],bins=50)
```




    (array([676., 578., 516., 444., 327., 306., 204., 186., 144., 119., 105.,
             88.,  64.,  57.,  67.,  42.,  38.,  38.,  36.,  30.,  28.,  15.,
             22.,  19.,  18.,   7.,  20.,   7.,   9.,   9.,   9.,   5.,  12.,
              7.,   4.,   2.,   5.,   2.,   3.,   2.,   3.,   2.,   1.,   1.,
              0.,   1.,   0.,   0.,   1.,   8.]),
     array([  1.  ,   2.98,   4.96,   6.94,   8.92,  10.9 ,  12.88,  14.86,
             16.84,  18.82,  20.8 ,  22.78,  24.76,  26.74,  28.72,  30.7 ,
             32.68,  34.66,  36.64,  38.62,  40.6 ,  42.58,  44.56,  46.54,
             48.52,  50.5 ,  52.48,  54.46,  56.44,  58.42,  60.4 ,  62.38,
             64.36,  66.34,  68.32,  70.3 ,  72.28,  74.26,  76.24,  78.22,
             80.2 ,  82.18,  84.16,  86.14,  88.12,  90.1 ,  92.08,  94.06,
             96.04,  98.02, 100.  ]),
     <BarContainer object of 50 artists>)




    
![png](output_70_1.png)
    



```python
xx = 50
for i in range(len(df0[['review','comment']][df0['com_num']>xx])):
    print(df0['Unnamed: 0'][df0['com_num']>xx].iloc[i])
    print(df0['review'][df0['com_num']>xx].iloc[i])
    print()
    print(df0['comment'][df0['com_num']>xx].iloc[i])
    print()
```

    8
    제~~~일 하기 싫은 집안일..있나요? 커피랑 롤케이크 먹으며...세탁기 돌아가는 소리를 듣다가..갑자기 든 생각..전 빨래하는거 좋아해요~~너는거까지요널고 나서 가지런히 있는 빨래들을 보며혼자 만족하고 뿌듯해 해요설거지도 귀찮지 않아요~이사와서 조금 귀찮아지긴 했어요씽크대가 넓어지니 바로바로 설거지하지 않고 그릇을 자꾸 꺼내 쓰는 저를 발견했어요 ㅎㅎ구석구석 박박 문지르며 청소하는 편도 아니긴 하지만청소도 할 만 해요~~하긴 청소는..해도 표시안나니 늘 만족한 적은 없네요그런데...그것보다도빨래 개는건...ㅠ옷장에 넣는 거 까지..하기싫고 귀찮고..ㅎㅎ갑자기 이런 생각이 들어서맘님들은 정말 하기 싫은 집안일이 궁금해졌어요~나만 그런건가...같이 귀찮은거면 나만 그런게 아니라는 위로도 좀 받아보고 반성도 하고요 ㅋㅋ자동 기능이 되거나 누가 해줬으면 하는 집안일뭐가 있을까요?혼자 잡념합니다 ㅎㅎ
    
    [['으악~~저도 빨래 개는거 젤 하기 싫어요ㅠ빨래 개는것만 누가 해줬음 좋겠어요ㅋㅋ아이가 셋이라 빨래양도 어마어마해용..'], ['지난달 남편이 한동안 빨래를 개줬는데 어찌나 편한지..요즘 퇴근이 늦어 제가 하네요~\n세탁기는 마구마구 돌리고 싶은데 개는거 너무 싫어요'], ['아~~이불..ㅠ 이것도 개는거나 다를게 없죠'], ['저는 빨래돌리고 건조기돌리고 개는것까지괜찮은데 갖다넣기가,,,,,넘넘 시러요ㅠㅠㅋㅋ\n설거지랑 음식물쓰레기처리도요'], ['저희집 쓰레기는 남편 담당이라~~(이렇게 만들기까지 험난한 여정이 있었지요) ㅎㅎ\n내가 하면 정말 싫지요ㅜ\n개켜놓은 빨래 갖다넣는것도...못할짓이에요 ㅎㅎ'], ['저도ㅋㅋ 아예 신혼살림시작할때 딱 한번하고 도저히 못하겠어서 흐느끼는연기를 살짝더했더니,, 신랑이 화장실청소랑 설거지, 쓰레기처리해요ㅎㅎ 남편이 하는게 좋은듯요!  애들이 많아지니 빨래도 목욕도신랑몫이네요 저는 대신 맛있는 음식담당이요!!\n집안일은 못할짓이나 귀~~찮은일이 넘많은거같아여ㅠㅠ'], ['그래도 사람은 움직여야하니..힘들다..해도\n하고 있는..해야 되는..ㅋㅋㅋ\n뭐 그런거죠~~ㅎ'], ['맞아요ㅠㅠ 애들이 있으니 귀찮다 힘들다 노래함서 하고있네요ㅎㅎ  꽃다운님도 오늘하루 파이팅하세요♥'], ['맘님도..반사요^^\n좋은 하루 보내세요\n힘이 납니다~~~'], ['청소기돌리는거용..넘 귀찮..아!하나더..젖병닦기..ㅜㅜ왤케 하기싫은거죵?ㅜㅜ'], ['그나저나 맛난 롤케이크 혼자 먹는거임? 맛있는거 있음 좀 나눠먹져? 저희집 크림치즈프레즐 있는데 흥칫뿡'], ['아...젖병..ㅋㅋ 어떡하나~~~난 ㅋㅋㅋ\n\n크림치즈프레즐 좋아하는데..ㅠ 롤케이크 같이 먹어 줄테니 서우랑 놀러와요~~'], ['언니가 또 오시믄 안댈까여...? 저 딸린애가 하나더라 둘데리고 이동이 왤케 힘들까여 저 몸도 2명 몸이라ㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋ 내 몸도 둘이긴한데...붙어있는 미니미가 없으니..언니가 갈게요 ㅋㅋㅋㅋ'], ['씐나용ㅎㅎ아니믄 둘째더 이뻐 하시는 시어머니 찬스 이용하고 서진이랑 가겠어용ㅋㅋㅋ날짜만 잡아용 헤헷'], ['집안일.. 끝없는 도돌이표 ㅎ\n깨끗함 유지하는일은 참 힘든일인것같아요~ 주부님들 다같이 힘내요~^^'], ['전..마음을 내려놓고 보이는 곳만 깨끗하게..해요 ㅋㅋ\n집안일은 정말 힘든 일이에요\n우리 엄마들도 해낸 일이네요^^'], ['저는집안일은다하기시러요  ㅠㅠㅠㅠ 특히요리 ㅋㅋㅋㅋ ㅠㅠ'], ['ㅋㅋㅋㅋㅋ요즘 남편이 늦게 와서 요리를 안하니 제가 잊고 있었네요~~~~'], ['ㅋㅋ 저도요 집안일은 그냥 다시르네요'], ['개나 줘버리고 싶지만..내 일이니..해야지요머..\n힘들때 힘들다고 말하면서 하면 되고ㅠ'], ['화장실청소요ㅋㅋ제일시름'], ['전 깨끗하게 안하나봐요~\n화장실청소는 할만해요~\n남편한테 맡겼다가 마음에 안들어서 제가 하네요~~ㅎ 남편만 쓰는 안방 화장실만 남편이 해요~'], ['저는 빨래널고.접은거 넣는게 젤 싫으네요ㅎ왜그런지 ㅋㅇㅋㅋ'], ['이 일은 애증의 관계에요 ㅋㅋ'], ['저도 빨래정리요ㅜㅜ'], ['정말..싫죠~~\n남편한테 위임하고 싶어요'], ['화장실청소랑 쓰레기버리는게\n집안일 중에 젤 귀찮은것 같아용'], ['안먹고 안쌀수도 없고 ㅋㅋㅋㅋㅋ'], ['설겆이하는건 조아요ㅋ빨래돌리는것도 조아요ㅋ너는것도조아요ㅋ 개는것도조아요ㅋ근데 귀찮아질때가 있어요ㅋㅋㅋㅋ'], ['집안일 엄청 잘 할거 같은 느낌~~이네요 ㅋㅋ\n부탁하고 싶다~~~~~ㅎ'], ['저도 빨래는 좋은데...\n청소가 싫어요..\n그중에서도  걸레질 왜이리 하기 싫은징...\n빨래 개는건 각맞춰 잘 개고 좋아라합니다~~\nㅋ~~ 댓글 읽어보니 각양각색이네요~ㅋ'], ['청소는..워낙 광범위해서..해도 해도 안한 기분이에요~\n저도 각맞춰 개는것도 좋아하는데 넣는게..마음이 힘들어요 ㅎㅎ'], ['전 그냥 다싫어요~ㅎ특히 설겆이하는게 젤싫어요~해도해도 끝이없는 설겆이..'], ['해도 해도 분수처럼 계속 설거지거리가 나오죠~~ㅎㅎ\n저도 그냥 다 싫긴해요'], ['빨래개는게 안예뻐요ㅜㅜㅋ 제가 종이접기도 잘못해요\n걍 방치하면 남편이 접어넣더라구요ㅋ'], ['못하면 남편이 해 주는 군요 ㅎㅎ\n써먹어야겠어요~~'], ['저도 1순위 빨래 개는것이요!!!!!! 빨래 개는 기계도 있던데. 진심 사고 싶어요 ㅋㅋ'], ['그 기계..옷 마다 개줄 수 있나요?\n탐나는 제품이네요'], ['자세히는 모르지만 셔츠랑 바지같은건 개주는듯해요~~ ㅎㅎ 신기하더라고요'], ['빨래 갤 때 팔아픈데 ㅋㅋ 있으면 편하긴 하겠어요~~~~'], ['쌩뚱맞지만...  이런ㄱㅓ 혼자먹기없기'], ['ㅋㅋㅋㅋㅋㅋㅋ\n아이스 카라멜마끼아또 갖고 오면 줄게요~~~ㅎㅎ'], ['화장실청소요ㅠ 신랑보고 일주일에한번 해달라니깐 이리뺀돌 저리뺀돌 창틀청소는 해야지해야지 하면서 몇센치까지 먼지가쌓이나 두고보는중이지요 ㅋㅋ'], ['저도 두고보는거 많아요 ㅎㅎ\n모든걸 신경쓰기에는 제 능력이 부족해요~~\n몸은 튼튼해 보여도 체력은 안되네요 ㅎㅎ'], ['전 이불빨래요ㅜㅜ 커버분리하고, 부피커서 널기두, 다시 껴놓기두 넘넘 귀찮구 힘들어요ㅜㅋㅋ'], ['매일하는게 아니니 이불 빨래는 생각도 못했네요~\n저희 아들 기저귀떼느라 최근에 이불 빨래를 몇번 했어요 ㅎㅎ\n결국 방수요 구입했는데 구입하자마자 안싸네요 ㅎㅎ'], ['개는거 넣는거 두개가 젤 귀찮아요ㅠㅠ'], ['옷을 안 입을 수도 없고 ㅋㅋㅋ'], ['전 설거지요 ㅠㅠㅠㅠ\n밥하고 청소 하고 빨래하고 머 이런건 다 하겠는데 \n설거지 할때가 좀 귀찮네요 ㅋㅋ'], ['제 동생도 설거지를 그렇~~게 싫어하더라고요~\n동생집 가면 제가 해줘요'], ['저는 개는건좋은데 넣는게 싫어요 ㅠㅠㅠㅠ 빨래넣는거랑 설거지가 왜이렇게 귀찮은지...ㅋㅋ'], ['저도 넣는거 싫어요~\n옷장정리는...노답이죠 ㅎㅎ'], ['빨래....빨래 돌리는것도 너는것도 게는것도 다 싫어요.. 맨날 남편이 해줘요ㅠㅠ 그대신 설거지랑 싱크대 청소는 매일 같이 해요ㅎㅎㅎㅎ 전 주방일은 다 좋아요ㅎㅎㅎ'], ['저도 남편한테 맡겨야되나봐요~\n주방일 좋아했는데 요즘은 그것도 싫어지는 중이네요~\n만사가 귀찮아지고 있어요^^'], ['설것이요 ㅜㅜ 밥먹고 늘어져서 하기 싫어요'], ['아~~~~저도 그렇긴해요\n그러고보니 남편은 집에서 놀고 먹기만 하는거 같아요~~']]
    
    82
    전업주부면 집안일 혼자 다 해야해요? 그냥 궁금해서 다른분들은 어찌하시는지 여쭈어봅니다 ㅎ전 현재는 전업이구요 아이1명인데요 재작년부터1년반정도는 파트타임일 아이 어린이집 있는시간대에 했었어요그런데 남편이 집안일엔 신경을 1도 안씁니다 ㅡㅡ청소기 세탁기 전자렌지 밥솥 등등 가전 만져본적없구요 배고프면 컵라면 먹어요시켜도 안하고 아무튼 암것도 안해요..애처럼 밥먹고 어지르기만한다 생각하면됩니다 ㅎ 이젠 집이 점점 개x이 되가는데 혼자 하다보니 저도 점점 지치고 티도ㅠ안나니까 손을 놓게되요..그럼 육아는 하냐구요?오빠 정도에요..자기 체력되면 놀아주고..아님 핸폰이나 쥐어주는.. 목(애목욕 못시킴 옷못입힘 밥못먹임 잠못재움 병수발못함..심지너 애기때도 기저귀도 갈아본적 없음 )이건 집이 더러워도 제탓만은 아닌 남편도 좀 문제있는거 맞죠? 어제 냉장고에 유통기한 지난 음료 음식들로 구박들이니 심히 기분상했네요..휴우아니 거슬리면 니가 좀 버리던가!!
    
    [['처음부터 교육을 시켰어야했는데 ........ \n큰아들 지금부터라도 고쳐서 쓰셔요!\n몇시에 설거지좀 해줘. 몇시에 빨래좀돌려줘. 몇시에 쓰레기좀버려줘. 몇시에 화장실청소좀해줘. 그럼 잘합니다.'], ['남자들은 미션을 주면 한대서 해봤는데 다 무시해요 ㅠㅠ몹쓸아들이에요'], ['하.. 이해되서 댓글 남겨요ㅠ 전 그래서 가끔 참다가 폭발해요! 내가 그럼 집안일 전담할테니 육아를 반반 나누자고! 그래서 요즘 설거지라도 하고.. 냉장고에 있는거 지적하는거 완전 공감!! 그럴때 또 소리치죠 니가 좀 버리라고!!'], ['지적은 진짜 짜증나요 ㅋㅋ이해와 공감에 힘이됩니다'], ['남자마다성향이달라요ㅜ\n그럼니가나가서벌던가하는남편도있어요...\n성향이시댁따라가는경우도있어요~'], ['네~ 전 이럴때마다 시어머니욕도 함께나와요\n아들잘못키워주셔서..ㅜㅜ'], ['쓰레기 음쓰 분리수거 남편이하구요.  주말청소기는 저랑 아이 나갔을때 돌려요.  아이목욕은 제가아플때만 시켜주구요. 그런데 이거달라저거달라 귀찮아서 제가하는게 속편한ㅎㅎ'], ['저도 왠만함 내가하자 주의인데...이럴땐 폭발해요 ㅎ'], ['저도 그런편이라 보통은 참긴하는데 육아도 독박이다보니 ㅜㅜ 힘들때가 생기네요'], ['전 전업인데 신랑이 퇴근해서 청소기돌리고 ,걸레질하고,아이랑 같이 장난감정리하고, 아이 재우는거까지 해줘요.\n쓰레기봉투 버리고, 재활용두요.\n아이 38개월까지 기관안보내고 끼고있었고, 제가 항상 피곤에 쩔어있으니 퇴근하면 바로 알아서 하더라구요. 고맙기도 미안하기도하죠..'], ['복받으셨네요...ㅠㅠ 저도 애37갤에 첨 기관보냈고 그담엔 파트타임일 시작하게되서 일년반은 정신없었고..여전히 관둬도 제정신이아니네요 ㅋㅋ'], ['저도 전업이면, 집안일은 다 하는게 맞다고 생각하고 \n제가 다 합니다. 대신, 도와주지도 않으면서 간섭하거나 잔소리하면 일절 차단하구요. \n각자의 역할과 영역을 존중받으려면,\n힘들다 도와달라 불평하는것보다\n책임을 다하는게 중요하죠. \n대신, 아이가 기관에 가기 전 시기에는\n전업주부로서 하는 집안일은 예외입니다.'], ['2222 저도 의견에 동의해요. 전업이면 대부분 하는게 맞다고 생각해요.'], ['저도 공감해요...밖에서 남에돈 벌어오는게 쉽지 않잖아요...남편들 힘들어요...'], ['도와주지도않음서 잔소리를하니 제가 열받았어요 ㅋ'], ['직장맘인데 남편 1도 집안일 안합니다..심지어 주말에도 출근합니다. 독박육아에 독박가사...남편없을 때 아이랑 하고 싶은거 하고 삽니다..'], ['저도 그럴까봐 직장복귀를 못하겠어요 ㅡㅡㅋ 혼자 독박쓸게 보이니까..'], ['아이들 크면서 중.고딩되니 손도까닥안한다는..ㅜ\n넌 집에서 먹고노는데 이딴마인드라..\n딸이 분리수거도와주고\n모든집안일은 제가해요.\n윗분말씀하셧듯이 시켜도 안해요..잘하는 내가 하는데 이젠 돈벌로나갓음 하는 눈치더라구요..\n제가 항상깨끗하게 하니 집이 청소되있는게 당연하게생각해요..\n근데 인간개조 안되더라구요..ㅠ'], ['맞아요 ㅋ 사람은 안변해요 ㅠ'], ['저는 그냥 제가 하는 편이고 대신 필요한건 해달라고 하고요. 저도 사실 애기 유치원에 간 시간동안만 일을 하기때문에 시간이 없는게 사실이지만 남편이 종일 일하고 집에 와서 쉬는게 고작 한두시간이라 좀 쉬게 해주고 싶은 마음도 있고요. 근데 너무 안한다 싶은땐 말해요 ㅋㅋ 집안일이라는게 내가 안하면 누군가는 해야하는 일인데 자기가 안하면 고스란히 내가 해야하는 일이 된다고. 반대로 내가 안하면 자기가 해야하는 일이 되고. 그래서 나는 자기가 힘들거 같아서 내가 먼저 하는 거지 집안일이 취미여서 하는 게 아니라고. 그러면 미안하다면서 하더라구요 ㅋㅋㅋ 가끔 제가 하는걸 너무 당연히 보고 있으면 성질나서 버럭하긴해요 ㅠㅠ 그럼 딸에게 말해요. 너는 결혼을 하지마 라고 ㅋㅋㅋ 큰소리로 말하면 미안해하더라구요 ㅠㅠ'], ['양심있는 남편분이라 부럽네요ㅜㅜ\n일단 우리집아들은 양심이없습니다 ㅋㅋㅋ안미안해해요'], ['너무 힘들면 주 1-2회 도우미 도움 받으시는게 좋지 않을까요 남편분한테도 집안일은 같이 하는거라고 계속 주입시키시고요 전업이라도 집안일은 같이 하는게 맞다고 생각해요'], ['육아만 반씩해도 좋겠어요 ㅠㅠ'], ['사람쓰세요!!ㅎㅎㅎ 일주일에 한번만 써도 확실히 편해요! ㅎㅎ 1회 5만원인데 그돈 아까우면 당신이 집안일해^^ 당신이 안하는거 일주일에 한번 아줌마로 때우는거니까 잔소리하지마 해보세요!'], ['현재 개집인데 사람 불러도 될까요? 민망해서 ^^;;'], ['엉망인집 청소해주는 언니 있어요^^  저결혼전에 이삿짐 정리도 않았는데 부모님 가게 정리한다고 짐들어와서 발 디딜틈이 없었거든요 \n가격은 좀 있지만 전 완전 만족해요~~\n그 언니는 깨끗한 집은 청소 안하니 민망해할 필요도 없고 ㅋㅋ\n필요하심 쳇 주세요~~'], ['쪽지드릴게요 ~감사해요♡청소요정이 찾아왔음좋겠어요ㅋ'], ['저 집이 당체 정리가 안되서 도움좀 받고파요! 그분 정보좀 쪽지로 쥬세요옹~~~~~~ㅠㅜ'], ['쪽지 보냈어요~~^^ 제경험상 청소가 엄두가 안날땐 전문가의 도움을 받고 유지하는게 좋은거 같아요~~^^'], ['힝..전 쪽지 못빋았어여 ㅜㅜ  저도 좀 부탁드려요♡'], ['쪽지 보냈어요^^\n그 언니는 더러운집 청소를 즐겨하기에 본인이 보기에 깨끗한집은 그냥 도우미 부르라고 하셔요~~ 꼭 더러운 집만 합니다 참고하셔요 ㅋㅋ'], ['저도 정보 얻을수 있을까요? 저희집도 점점 개판이 되어가네요ㅠㅠㅠ'], ['쳇보내드렸어요^^'], ['전업이면 집안일 모두 주부가 하는게 맞다는분이 많으신데 육아도 집안일에 설마 포함시키고 계신건 아니겠죠?집안일은 그렇다쳐도 육아는 아니죠  근데 육아와 집안일의 경계가 모호할순 있어도 애는 같이  돌봐야  하는게 당연하다고 봐요 그건 애한테도 중요한 문제구요! 집안일 너무 심하게 안도와주는 남편이신듯 하니 아이와 주말에 시간보내라고 내보내시면 어떨지.'], ['육아만 없으면 사실 어려운일이 아닌데 아이가 있고 없고는 참 차이가 크네요..ㅜㅜ'], ['전 지금은 전업인데 평일엔 집안일 설겆이만 시켜요 어차피 늦게 오니까 해줄것이 없거든요. 주말은 전 일부러 제가 집안일하고 육아는 남편에게 맡기네요 육아가 더 어려워서요 그냥 애들 신랑더러 보라하고 어버이날 핑계로 친정에 1박2일 다녀오심 어떨까요? 을마나 힘든지 뼈저리게 느껴야 도와줄꺼같은데요'], ['친정엔 저랑 애랑 묶어서 보냅니다 ..혼자 애 안봐요 ㅜㅜ'], ['상황에 따라 사람 능력에 따라 다르게 해야죠 . 지금 전업이라도 파트 타임 하시니까 그만큼 바쁘시고 전업은 아니시네요. 남편이 당연히 도와야하네요. 저는 일을 안하기도 하고 남편하는게 맘에 안들어서 안시켜요  그나마 애는 최대한 봐줘야하는거 아닌가 해요'], ['주말에는 남편이 애랑 있어주고 그동안 전 집안일 밀린거 하면 참 좋을듯한데 ㅠㅠ힘드네요'], ['저도 윗분댓글처럼 아이어린이집가기전엔 신랑이 같이해야 하지만 아이어린이집보내는 전업이면 집안일은 와이프가 하는게 맞다고생각해요~^^; 대신 육아는 돕는게아니라 아이를위해 아빠가 같이놀아줘야하겠죠^^'], ['37갤까지 애끼고 독박육아했어요 흐규흐규'], ['육아와 집안일은 일단 다른것!\n전업이라 집안일을 다한다? 오케이! 그리고 가져다주는 월급에서 제 용돈 제해야 한다고 봅니다. 이용돈에서 즐기시던 사람을 쓰시던지 하는건 본인의 판단에 달렸다구 생각해요\n\n근데 육아까지는 전업주부의 일 아니라고 보구요.\n아이를 키우는데 아빠의역할도 크다고봅니다...\n더군다나 파트타임으로 일을 하시고 계신다면 반!으로 쪼개는건 사실상 불가능하지만, 그래도 어느정도 배분해야한다고 봐요...(이건 아이의 미래를 위해서도요. 보고배우는게 있잖아요~) 누군 잠자기 전까지 일하고 누군 본인 일만 하고 쉬나요? 저는 그꼴 못볼 것같아요...노예도아니고 - 지극히 저의생각입니다....'], ['네..육아라도 잘해주면 좋겠어요 ㅠㅠ'], ['부인이 참다참다 짜증내거나 화내면서 왜 안도와주냐 폭발해봤자 남편들은 알아듣지도 못하고 부인만 속상해 하더군요. 집안일을 반반씩하자 도와달라 이런건 남자들이 잘 못알아(?)먹습니다. 주말에 설거지나 분리수거 음식물버리는거 아님 욕실청소 딱 한개정도를 지정해주면 잘 하더라구요. 그렇게 몸을 움직이고 하다보면 다른것도 눈에 들어옵니다. 아기라고 생각하고 잘한다잘한다 달래줘야 된다는것 ㅠㅠ 그렇게라도 시켜야 되더라구요. 안그러면 제 몸이 너무 힘들어요'], ['현관앞에 쓰레기더미놔도 안보인다고 그냥가던데요?ㅋ\n지정해도 안해요 ;;;'], ['....죄송한데 모유수유빼곤 집안일 육아 다해줍니다.\n결혼전에도 시댁 집안일 도와준 신랑이라 시댁가서도 집안일 하고 시압쥐도 퇴근후 늘 저녁설거지는 하세요. 중요한건 시엄니가 그걸 가르치고 뿌듯해하셔서 저흰 시댁이던 어디던 늘 집에서 하던대로 지내요. 저희 남동생은 아들이 최고다! 하시는 친정엄마 덕분에 제가 귀에 딱지가 앉게 말해도 안들어처!먹어요. 누나가 애기안고 짐이고지고해도 뒷짐지고 가서 하..놀랬답니다..과장없이 다섯번 말해야 한번 실천하는놈입니다.\n옛날엔 애낳고 밥했다. 어디여자가..라는 사상이 박힌 10년째 싱글남입니다'], ['숙제를 하나씩 늘려주세요\n처음부터 안하면 버릇이 되어 버려서 안되요\n나누어서 하심이....'], ['전업주부이면 집안일 혼자 다 해야 한다고 생각하시는 분들이 많아요...\n\n그런데 저희 남편이 인사 담당하는데 결혼전부터 그런 얘기 많이 했어요.\n여사님들 경험담인데,\n\n남편이 퇴직후에 집에 있고 이제는 부인이 나가서 캐셔, 판매 같은 일을 하는거에요.\n돈이 필요하니까..\n\n젊어서부터 남편은 일해서 피곤하니까 아이랑 부인이 같이 자면서 남편 배려해주고,\n남편은 집안일, 육아 하나도 안하게 배려해줬더니\n나이 들어서 집안일 할줄을 모른대요.\n\n9시간씩 서서 일하고 집에 들어가서도 집에서 노는 남편 저녁을 차려주는게 너무 싫다고 하셨대요.\n저희 남편은 자기는 절대 안그럴거라며 집안일이고 육아고 다 같이 해요.\n\n참고로.. 저희 엄마 항암치료 하시는데 아빠가 할줄 아는게 하나도 없으셔서 항암치료의 고통속에서도 식사 다 엄마가 차려주세요. \n\n저는 누구나 상황이 언제 바뀔지 모르니까 누구든 꼭 다 할줄 알아야 한다고 생각해요.\n나이들어 상황 바뀌면 포지션도 바뀌겠지? 이게 안되는거 같아요.\n\n아!! 롤러코스터 습관이라는 노래에도 그러죠~\n습관이란게 무서운거거든~~'], ['집안일은 전담, 육아는 반반이죠. 나중에 아이한테 찬밥될거라고 은연중에 흘리듯 말 계속 하셔요~~'], ['평일은 남편의 출퇴근때문에 거의 독박육아 집안일도 전담하구요 주말엔 식사준비나 설겆이는 같이하는편이예요... 그리고 제가 애들하고 놀아주는걸 잘 못해서 남편이 육아할때 저는 집안일하구요.. 생각해보면 집안일은 제가 거의 하는것같고 육아는 거진 반반인것같아요'], ['다들 전업이면 당연히 집안일을 다 해야한다고 생각하시네요.. 전업이라고 남편이 집안일에 지분(?)이 없는건 아니라고 생각해요.. 엄마가 전업이라고 학생 때 엄마가 방 다 치워주시던가요? 그건 자기 일인거잖아요~ 전 전업이어도 집안일에도 일정부분을 자기 일로 남편에게 주어야한다고 생각해요.. 전 신혼 때부터 음식쓰레기는 무조건 신랑일로 정해주었어요. 쓰레기나 분리수거도 신랑이 하지만 사실 바쁘거나하면 제가 하기도 해요. 그런데 음식쓰레기는 신랑이 부탁하기전에는 절대 제가 안해줬어요. 자기몫이란 의미죠. 물론 사람 성향도 영향이 있었겠지만 그래서인지 지금은 집안일 무지 많이 해요~ 임신해서부터 아기 어린이집 갈때까지는 3년동안 청소기 한번도 안돌려봤어요. 처음부터 일을 나눠주어야 집안일을 하는거 같아요~'], ['아이하나일때 전업 잠깐했었는데 평일 집안일은 제가 다 했어요 주말에 대청소 , 화장실청소 , 분리수거는 신랑이 하고 육아는 반반 했어요 .. 당연히 육아는 같이하는거 아닌가요 .. 퇴근하고 오면 애들 씻겨주고 먹여주고 재워주고 다 같이했어요 .. 심지어 첫째 새벽에 수유할때 할거 없어도 같이 일어나 눈뜨고 앉아있었네요 ..'], ['애들이.아주 어릴땐.도와줬구요\n지금은 초딩인데 도와주지도 않지만 도와달라고도 안해요\n분리수거는.해주구요\n운전해주거나 도서관 책 반납 대출 애들 문제집이나 숙제 책읽는거 봐주고 짐나르고 집에 고장난거 고치고 컴퓨터 해결해주고.. 이런건 합니다\n설겆이.빨래 청소등은 오로지 제 담당이예요\n저 아파도 안해줘요 ㅠ 그냥 뭐 시켜먹고..부엌일은 1도 안해요\n일년에 두어번 라면 직접 끓여먹는.정도\n애들 미취학일땐 청소정도 가끔 도와줬네요\n애들은 잘 봐주긴 했어요 바빠서 몇시간 못봐줘서 그렇지..\n전 별 불만 없었고 지금도 없어요']]
    
    113
    젤 하기 싫은 집안일이 뭐일까요? 젤 하기 싫은 집안일이 신랑 와이셔츠 다리는 거랑  설거지에요ㅜㅜ진짜 설거지 거리는 매일 매일 쌓여있어요ㅜㅜ 음식 준비하면서 생기고 먹고 나면  또 생기니...밥 먹고 나면 설거지할 의욕 상실...매일 다음날에야 해요..임신 하니 설거지하면 자꾸 나온 배 부분이 젖어서 더 하기 싫어 지네요..
    
    [['저두요 설거지ㅜㅜ 왜 한번 밥먹고 나면 산처럼 쌓여있는지..'], ['맞아요..설거지 넘 지겹고 하기 싫어요ㅜㅜ 만드는것 까지만 제가 하고 설거진 신랑이 좀 해주면 얼마나 좋을까요ㅜㅜ'], ['빨래개기, 이보다 더귀찮은건 갠 빨래 서랍넣기요. 전이게 한번에 잘 안이어지네요^^'], ['맞아요 맞아!! 저도 빨래 개기 넘 싫어요 ㅎㅎ 빨아서 널기까지 잘 하는데...'], ['저도 이거 너무 귀찮아요... 맨날 건조기에서 꺼낸채로 쇼파위에 쏟아놔요ㅠㅠ'], ['저도 빨래개기 , 설거지 ㅠㅠㅠ 너무너므 시로요'], ['다 똑같은 맘인가봐요^^ 진짜 아줌마 쓰며 사는 맘들이 부러울 뿐이에요ㅜㅜ'], ['전 설거지랑 빨래개기~~ 근데 집안일은 다 싫은것 같아요ㅎㅎ 그래도 신랑이 많이 해주는 편이에요'], ['부럽네요..신랑이 해주신다니...전 진짜 한달에 한번 도와주면 대단한거에요--;;'], ['빨래널기 빨래개기요ㅜ'], ['전 빨래널기 잘해요 ㅋㅋ 개기가 싫어서 그렇죠~ 개서 옆에다 나두면 신랑이 궁시렁 거리며 가끔 넣어줘요 ㅋㅋ'], ['전 화장실 청소요 ㅠ 매트 다 들어서 솔로 바닥닦고  매트 닦고   ㅠㅠ'], ['생각해보니 하기 싫은.집안일 참 많네요'], ['화장실청소요~~'], ['전 화장실 청소만 잘해요 ㅋㅋ 왠지.집에서 화장실이 젤 더러운 곳일거 같아서요'], ['저는 청소가 젤 하기 싫어요ㅠㅠ 해도해도 티가 안나요ㅎ'], ['청소도..하기 싫은 일중 하나죠ㅜㅜ'], ['전다요\n회사일만 하믄 좋겠어여^^'], ['ㅋㅋㅋㅋㅋㅋ정답이네요!!'], ['빨래 정리하기요 ~~설거지는 세척기 들이고 해방됐어요 엉엉 ㅠ'], ['아 세척기!! 저도 사고 싶지만 집이 콩알만해서ㅜㅜ'], ['전 설거지랑 애기들 매트닦기요..ㅠㅠㅠㅠ 해도해도 티가 안나는거 같아요..ㅠ'], ['설거지 다 싫어하는군요^^'], ['전 요리요.\n밥 좀 안하고 살고싶어요\n먹성좋은 서방에\n입짧은 애들에\n전생에 군자금에 손댔나봐요ㅠㅠ'], ['ㅎㅎ전 하는것 까진 좋은데 치우는게ㅜㅜ'], ['걸레질요..넘 싫어서 무선걸레청소기 샀는데도\n여전히 할때마다 싫어요'], ['전 걸레질을 가끔해요'], ['전 욕실청소요.ㅎ'], ['욕실청소도 많이들 싫어하시는 군요^^;;'], ['저도 설거지요!  식기세척기 놨더니 세상편한데 그 마저도 하기싫어요ㅜㅜ'], ['식기세척기가 있어서 넣고 빼야하니 하기 귀찮긴 마찬가지일 듯요 ㅋㅋ'], ['다요! 임신하니 다 싫어요ㅠ'], ['저도 임신하니 더 하기 싫네요ㅜㅜ'], ['그중에 하나꼽으면 빨래개서 넣는거요 ㅎ 빨래산이 넘쌓여잇어요ㅜ'], ['빨래개서 넣기 ㅋ 거둬서 바닥에 던져놓고 겨우 개고 ㅋ 신랑이 집어넣어요  전 글케해놨어요 ㅋ 아님 애들총동원해요^^'], ['애들 총 동원 ㅋㅋ 저도 그래요^^'], ['전 욕실청소요 ㅠ'], ['설거지. 욕실청소. 빨래개기 3파전이군요 ㅎㅎ'], ['빨래개서 넣기ㅠㅠ\n요즘은 개켜서 식탁위에...쭉~~~\n각자 가져가 하고있네용~~@@;;'], ['설거지요.\n넘 하기 싷어요. 누가 해주는 것도 싫구요. ^^'], ['설거지요..,돌아서면 다시 시작되는 기분...'], ['화장실 청소... 젤 싫어요ㅜㅜ'], ['전 요리하기가 싫어요 맨날 비슷한 요리. 애들도 잘 안먹고  요리하고 나면 주방이 아수라장 ㅋㅋ'], ['저는 다 괜찮은데 쌀씻기가 제일 싫어요 물맞추는건 몇년이 지나도 스트레스받구요ㅎㅎ'], ['컵으로 쌀 떠서 컵으로 물재보세요..세상 편해요.ㅋ'], ['청소요! 특히 화장실 청소요!ㅋ'], ['전 설거지는 재밌던데요..ㅎㅎ빨래개서 정리하는게 젤루 귀찮아요'], ['저두 바깥일이 좋아요\n집안일은 다 고되요'], ['걸레질 화장실청소.....체력이거기까지는 안가요 ㅠㅠ'], ['빨래 넣기요ㅠㅠㅠㅠ'], ['정리정돈이요 그중에 일번은 옷 정리요 ~~;;'], ['냉장고 청소요.냉장고는 먹어서 비우는 일만 했음 좋겠어요ㅎㅎ'], ['다 하기싫어요....'], ['당연최고는 빨래개기..ㅋ'], ['일하고 들어오면 다 하기 싫어요\n매일 쌓이는 빨래.설거지 반찬해도  또 먹을게 없고 다 싫어요'], ['전 화장실청소요ㅠ'], ['음식만드는거요..ㅜ 저녁준비'], ['설거지.. 빨래개고 제자리에 넣는거요...'], ['전부다요...  흙\n적성에안맞나봐요ㅠㅜ']]
    
    185
    집안일 중에 뭐가 제일 하기 싫으세요? 저는 가사일중에 청소기 먼지통 비우는 일설거지 이불보 정리가 제일 하기 싫어요!아!! 빨래 너는것 정말 최고!!최고하기싫구요 ㅎㅎㅎ적고 나니 거의 대부분 인것 같네욧 ㅋㅋ
    
    [['창문청소..바닥닦이 두가지는 실시간 쌓이는 먼지들때문에  넘 힘들아요'], ['설겆이, 바닥청소요ㅋㅋ'], ['걸레질이요 ㅠㅠ'], ['전 설겆이.^^'], ['날씨좋고 미세먼지 없는 날 평일은 좋은데  주말 싫고요 세끼다 해먹기'], ['설거지요..'], ['바닥청소, 빨래개서 옷장에 넣기'], ['전 빨래 접어서 각각 서랍에 넣는거요\n'], ['다림질이요'], ['빨래개기 청소기 돌리기(그래서 로봇써요) 이불 바꾸기'], ['걸레질이랑 창틀청소.......ㅠㅠ'], ['요리요~ 전 청소는 그냥 하면 되는데 요리는 뭘할까부터 장보기, 요리하고 치우는거까지 한끼 해먹는데 설겆이는 왜 이리 많이 나오는지요'], ['다 하기 귀찮지만 ㅋ 그중하나라면 저두 설겆이^^'], ['화장실청소 씽크대배수구 청소요 ㅜ'], ['빨래개는거요ㅠ 둘째녀석때문에 가재수건이 어찌나많이나오는지ㅠㅠ'], ['청소요 ㅋㅋㅋ 설거지가 좋네요 청소는 싫어요 그래서.. 오늘도 집은 난장판...ㅋㅋㅋ'], ['화장실 청소요 ㅋㅋ'], ['설겆이 화장실청소 빨래너는거요..'], ['빨래개기요 ㅋㅋ 빨래 개주는 로봇 있다하니 어서 상용화되길 기대해봅니다'], ['저두 빨래 널고 걷어서 개켜서 옷장에 넣는거요~~'], ['바닥걸레질이요 밀기넘 귀찮아요 ㅜㅜ'], ['빨래개기요.. 시간이 많이 걸려서 싫어요 ㅋㅋ'], ['걸레질이요ㅋㅋㅋ'], ['다 하기싫어요 ㅠㅠ'], ['손걸레질이요'], ['젖병세척소독도 은근 귀차나요ㅋ'], ['좋은 게  없고  다  싫지만,  요리가  가장  힘들고...하기  싫고  그래요...'], ['빨래개서 옷장에넣는게 제일귀찮 ㅋㅋㅋ'], ['설거지ㅜㅜ'], ['빨래 개서 넣는거요 진짜 잘 안돼요 ㅜㅜ'], ['저는 밥하기....ㅋㅋㅋㅋㅋㅋ \n다행히 요리 좋아하는 남편님이 계십니다ㅋ'], ['하기좋은집안일이있나요 ㅠㅠㅠㅠ 다싫아용'], ['전 빨래 청소 설겆이 이런건 괜찮은대 요리가 너무 하기 싫어요~\n죽어라 만들었는대 맛없다고 안먹으면 맥빠져요~~ㅜㅜ'], ['마른빨래 정리요...매번  빨고 널기만 하고 정리는 안해서 마른빨래가 산을 이루고 있어요'], ['화장실 청소요 그리고 설거지는 왜 해도해도 안 없어질까요 ㅜㅜ'], ['바닥걸레질, 하수구 머리카락제거'], ['쓰레기비우러가는게 가장 귀찮아여..'], ['장보기도 집안일 이라면 전 장보기요.\n쇼핑이 최고의 스트레스입니다.ㅜㅜ'], ['빨래 널기요'], ['우열을 가릴수가 없어요\n이게 싫다할라고 하면 저것도 싫고,,,,,\n결론은 다 싫으네요 -_-;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;'], ['다요 ..... ㅠ'], ['요리요 ㅋㅋㅋ 맨날저만보면배고프다 지겨워요ㅡㅡ'], ['음식물 쓰레기 버리기랑 분리수거하러 나가기'], ['설거지요'], ['다림질'], ['집안일 다...ㅜㅜ'], ["빨래 접은 다음에 '제자리에 넣는거요' 요리도..화장실 청소도...그냥 다네여 ㅋㅋ"], ['222 격하게공감요'], ['저만인줄.. 왜 이게 이리 귀찮을까 제가 이상한 줄 알았는데.. 저도 공감백이요 ㅎ'], ['전부다요ㅠㅠ얼른 애들 크면 아줌마쓰고 직장다닐래요'], ['설거지요ㅜㅜ'], ['전...빨래 걷어서 개는거요ㅠ진짜싫어요'], ['모두 다요.. 우렁각시 있었음 해요..'], ['설거지 걸레질요'], ['화장실 청소ㆍ창문닦이'], ['와이셔츠 다림질 ㅠ'], ['밥하기 . 나머지는 모 할만해요.']]
    
    270
    집안일 집안일은 무슨....수민이 빨래 세탁기돌리고빨래돌아가는동안주방 청소하려고했는데 하답이없네요ㅋㅋㅋㅋㅋ주방후드는 작년에 이사오고 한번도안햇더닠ㅋㅋㅋㅋㅋ아ㅠㅠ뜨거운물에 불려놨으니 갔다와서 다시 씻어봐야겠어요...결국 집안일이라곤 수민이 빨래밖에 못널고이모찬쑤로 코코 갑니당이모 얼른 와요
    
    [['코코 언니더 가고깊다 ㅋㅋㅋㅋ'], ['ㅋㅋ저도 구경하러 가고싶네요 ㅋㅋ'], ['저도 구경가고싶어요 ㅋㅋ'], ['나도 코코.......'], ['코코 구경가나요?'], ['구경갑시다!!!ㅎㅎㅎ'], ['갑시다ㅋㅋㅋ'], ['상품권 사야겠네ㅋㅋㅋㅋ'], ['상품권사면 뭐주나여??'], ['코코에서 물건 살수있지'], ['오홍ㅋㄱㅋ출떵하나영'], ['나중에...... 정말 가고 싶을때 출동ㅋㅋ'], ['ㅋㅋ저는 2마트가 더 나은듯한..'], ['트레이더스??'], ['네넹ㅋㅋㅋㅋ'], ['트레이더스도 안가봄'], ['시민공원갓다 트레이더스 구갱갑시다'], ['6월에ㅋㅋㅋㅋ'], ['당근 6월에ㅋㅋㅋ'], ['6월에 시민공원에서 합체'], ['다모뎌라'], ['이리오너라 ㅋㅋㅋ'], ['모디라다모디라아'], ['12캔 모자라겠네'], ['이마트가서 한짝사긔'], ['모자라면 뛰어가긔'], ['가위바위보 지는사람 잼'], ['이기는 사람이 사오긔'], ['ㅋㅋㅋ왜이기냐며ㅋㅋ'], ['응 이겼으니 갔다오라며 ㅋㅋ'], ['ㅋㅋㅋㄲ웃기다붕'], ['아... 잠오긔'], ['자나요ㅋㅋ'], ['기절각 ㅋㅋㅋㅋㅋㅋ'], ['기절잼ㅋㅋㅋ'], ['욱이 소환에 바로 기절ㅋㅋㅋㅋ'], ['오늘도 기절각?'], ['글쎄ㅋㅋㅋ'], ['안기절인가요ㅋㅋㅋ'], ['먹는거에 따라서ㅋㅋㅋ'], ['기절?'], ['아직 안기절ㅋㅋ'], ['벌써한시다그램'], ['욱이 호츌 ㅋㅋ'], ['이이잉 등장인가요ㅋㅋㅋ'], ['어제 호출받고 기절각ㅋㅋ'], ['오늘도 호출후 기절인가여ㅋㅋ'], ['아직ㅋㅋㅋ'], ['12시되면 사라지겟군요ㅋㅋ'], ['수신 댓제가면 사라질라고 ㅋㅋ'], ['ㅋㄲ못사라지나요'], ['오늘 퇴근 못하나'], ['모리겟네여...'], ['어서 달려라'], ['댓제 도우미님 고생하셨어여 ㅋㅋ'], ['죄송해여ㅋㅋㅋ'], ['주부언니 댓글이엇네여\n도망갑시다'], ['그르네ㅋㅋ'], ['쏘리해영~ㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 방세 내놔여'], ['외상!'], ['ㅋㅋㅋㅋ 언제 갚나여'], ['6월에.....????ㅋㅋㅋ'], ['ㅋㅋㅋㅋ 기억할깨여'], ['수민이 샤랄라공주님 패션이네 ㅋㅋ'], ['공주옷입고 진상부리고옴ㅋㅋㅋ'], ['ㅋㅋㅋ착한공주님은 아닌걸로'], ['진상부리는 공주인척하는 김수민ㅋㅋ'], ['척수민 아니고 진상공주 수민으로 하자 ㅋㅋ'], ['진상공주 ㅋㅋ좋네용'], ['수영구에 사는 진상공주?  ㅋㅋㅋ'], ['그냥 진상녀ㅋㅋㅋㅋㅋㅋㅋ'], ['공주공주공주해줘 ㅋㅋ'], ['청소는 내일 하는걸로 ㅋㅋ수민이 샤랄라하네 이쁘다붕 ㅋㅋ'], ['ㅋㅋㅋㅋㄱ이쁘게입혀갔더니 진상진상ㅠㅠ\n청소는 내일하고\n빨래만 한다붕ㅋㅋㅋ'], ['오늘 비오는데붕... ㅋㅋㅋ난 내일 빨래해야겠다붕ㅋㅋ'], ['집오니 비내렸다붕ㅋㅋㅋ\n제습기가동중이다붕ㅋㅋㅋ'], ['이렇게 이쁘게하고 코코갑니까~'], ['ㅋㅋㅋㅋ잠옷입혔다가\n언니가 이건아닌거같다고 옷갈아입혔네여ㅋㅋㅋ'], ['ㅋㅋ 호피바지능요~'], ['호피바지입힐랫는데\n위에입을게없더라는용ㅠㅠ'], ['위에 반짝이 구매으드가야하나요 ㅋㅋ'], ['호피에는 반짝이가어울리나여?'], ['왠지 어울릴꺼같아서... ㅋㅋ 넘 눈부시려나요?'], ['찾아서 사야겟습니다ㅋㅋㅋ'], ['분홍이 반짝이 이쁠꺼같습니당 ㅋㅋ'], ['어디서 찾아봐야할까요ㅋㅋ'], ['작년 여름엔 본거같은데..올해도 팔겠죠?'], ['어디서봤어요?ㅋㅋ'], ['카스? 플마? ㅋㅋ 아디 핑크반짝이나시 이렁거'], ['그렇군요ㅋㅋ찾아봐야겟네요']]
    
    280
    가장 싫어하는 집안일은? 일요일날 집안일 하는게 습관이 되서오늘도 여느때와 다름없이청소,빨래등등을 끝내고 문득 여더분들은 어떤 집안일을가장 싫어하는지 궁금궁금^^1.청소2.빨래3.설거지4.요리5.분리수거6.욕실청소7.냉장고정리그밖에 기타등등같이 공유해봐요~^^집안일은 정말 해도해도 끝이없는듯^^
    
    [['저를요?'], ['6번이네요. \n글 보니 찔립니다요.'], ['그쵸~샤워하면서 같이하면 좋은데 피곤할때는 걍 사워만하구 쏙 빠져나온다는ㅋㅋ'], ['분리수거ㅜㅜ'], ['음식물쓰레기 가장 골치요;;'], ['아 그것두요ㅜㅜ에잇  그냥 다싫어요'], ['맞아요~생각해보면 다 싫어유ㅎㅎ'], ['6번7번 ㅋㅋㅋ'], ['비위도비워도 쌓여만가는 냉장고속 내용물ㅋㅋ'], ['다 싫은데요ㅠㅠ'], ['명답이에요^^'], ['3,4,6,7 \n\n물론 1,2,5 번도 좋아 하지 않음\n나 진짜 집안일 싫어하는구나 ㅋㅋㅋㅋ'], ['어지럽히기라도 보기에 하나 넣을걸...너무 싫은것만 제가 나열했네유ㅋㅋ'], ['미니멀하게 살려고 노력중..\n집안일 줄이려고 강제 미니멀 라이프 실현중 호'], ['채우는것보다 잘 버리는게 중요한것 같아요~살면서 깨우치네요^^'], ['창문틈 이나 문틈. 특히 아랫부분 청소...젤 짜증나구 힘들구 제거두 잘 안되구 성질 테스트 ㅋ'], ['오우~디테일하시네요~신문지에 물적셔서 꽃아놓으면 어느정도 제거되긴 하던데요...'], ['그런가요? 다음 청소땐 해봐야겠어요 ㅎㅎ 팁! ㄳ합니다~'], ['음식물 쓰레기요 ㅜㅜ'], ['미투요~^^냉동실에 보관해논 음식물쓰레기도 외출할때 꼭 까먹음...눈에 안보이니깐요ㅎㅎ'], ['네 그리고 썩은 냄새가 나서 토할거같아요ㅜㅜ'], ['요즘에는 날도 더워져서 관리를 잘해야죠^^'], ['맞아요 하루라도 안 버리면 벌레 날아다니더라구요'], ['맞아요~격하게 공감이요ㅠ'], ['공감해줘서 고마워요^^\n저녁밥 맛있게 드시고 \n즐거운 저녁보내세요 ♡♡♡'], ['네~하비님두 맛저하시구 평안한 저녁되셔요^^'], ['네 감사합니다 ♡♡♡'], ['빨래 널기여^^\n세탁기에  넣을땐 좋은데 펴서 너는 건 귀찮더라구요^^\n'], ['맞아요~빨래개기도 동급이에요~^^'], ['ㅋㅋ^^빨래개기도 힘들죠^^\n즐건 주일저녁보내세요~^^*'], ['네~스타니까님두 즐거운 저녁 보내셔요^^'], ['키워쥬스님두유~^^'], ['\n5번,.\n\n특히 음식물쓰레기요.'], ['아진짜 음식물쓰레기 젤 골치에요ㅎㅎ'], ['주방청소요...ㅜㅜ'], ['설거지 포함이죠?ㅎㅎ'], ['ㅋㅋㅋ당연하죠...\n개수대 청소가...으...'], ['개수대는 베이킹소다+식초 뿌린후 두시간정도후 물로흘려내리면 손 마니안가고 좋아요^^'], ['안쪽까지 할려니...ㅋㅋ\n안보이는곳..ㅜ\n첨 이사왔을때 예전에살던 사람이 어찌썻는지..\n드러죽는줄요...ㅋ'], ['ㅋㅋㅋㅋ드러죽는줄 아랐데ㅋㅋㅋ그 느낌 알것같아요ㅎㅎ'], ['와..진짜..토해써요ㅜㅜ'], ['으흐흐흐~진짜 그랬겠어요ㅠ'], ['청소하는데 이틀걸림요ㅋㅋ\n가스렌지도 더럽고...\n개수대도...\n아...ㅜㅜㅋ'], ['에고~몸살났겠어요. \n\n집안일은 운동효과보다는 말 그대로 노동인데...수고했어요~'], ['ㅋㅋㅋㅋ 이젠..손쉽게ㅜㅜㅋ\n\n키위님 맛저하세용~~'], ['네~텐시님두요^^'], ['욕실청소요.. ㅠ 요리도 그닥..'], ['욕실청소가 은근히 체력소모가 많아요~'], ['그래도 다행스럽게도 설겆이나 빨래는 좋아해서 다행이에요. ㅋㅋㅋㅋ'], ['어디서 들었는데 주방이 깨끗해야 근심이 없데요~생각해보면 그런것도 같아요^^'], ['집안일 잘 못하지만 그냥 안늘어놓고 모두 제자리가 젤로 맘이 편한건 사실이더라구요~^^'], ['맞아요~설거지 쌓아놓으면 잠이안와요ㅋㅋ오늘도 평안한 저녁되셔요^^'], ['설거지만빼고 다싫어요ㅠㅠ'], ['저두요~설거지는 하고나면 기분이 좋아지긴해요~^^다 끝내고 씽크대 물기 닦고 행주탈탈털어 널어놓을때 기분짱!'], ['저는 암것도 안해여~ \n불량 엄마 불량 딸 \n울엄니만 고생 하시네여'], ['저두 엄빠랑 같이 살때는 그랬어요~다 엄마몫! 저는 설거지만 도왔네유ㅠㅠ'], ['해도 해도 끝없는 청소인데 \n애완동물까지 키우는것보면 개인적생각이지만 대단하기도하고 어찌저렇게 같이사는지 이해가...'], ['애완동물! 말그대로 사랑의 힘 아닐까요?'], ['사랑 그힘이라면 가능할것같네요^^\n청소 죽을때까지 따라다니는 의무...\n갑부가 되자 !!'], ['엄마가 좋아 아빠가 좋아 ??이런 느낌이랄까요..'], ['전...아빠가 더 좋았고 지금도 아빠가 더 좋아요^^'], ['헉! 어머님 서운 하시겠네요 ㅎㅎ'], ['엄마는 카페회원이 아니니깐요~히힛'], ['다시러요..전 ㅎㅎㅜㅜ'], ['일하지 않는자 먹지도말라\n\n-울아빠 하시는말씀-\n\n그래서 기본적인건 해요~먹을려구^^']]
    
    303
    집안일...의 끝은 정녕 없는것일까요..? 간만에 쉬는날인데 점심약속이 있어서차를 쓰기위해, 아침에 신랑 출근시켜주고집에왔는데..  집안일이 제눈에 쏙쏙 들어오네요.신혼집와서 처음(부끄러운ㅜㅜ)으로 침대에까는이불,덮는이불 다른걸로 바꾸고,통돌이에 이불돌리고아침에 어메리칸스타일로 먹었던 설겆이들 하고,가스렌지 기름때는 왜이리ㅜㅜ신경쓰이는지..매직스펀지와 주방용티슈로 벅벅 문대고,분명 어제밤 청소기돌린거같은데우리냥이들이 밤새 한바탕 우다다로 인한먼지들...청소기돌리고,걸레질하구..양말들과 발걸레는 손으로 조물조물 손빨레해서널어버리고~~~화장대 쓰레기통도 비워주시고~있다가 54분후 세탁기에서 띠리링 멜로디가 나오면널어야겠죠...?날도 좋으니 건조대에 이불말려야겠어요~~사실..ㅜㅜ 어제 맥주 한잔마셧는데밤새 손발이 저려서ㅜㅜ 4시간밖에 못잤는데..집안일이 눈에 보이니 안할수도없구..ㅜㅜ손이 계속 저려서 미치겠네요이제..나이드는건지 ㅋㅋ 왕년엔 소주한병도거뜬햇는데.. 큰일이네요. .ㅜㅜㅠㅜ다들..쉬는날 어떻게 보내시나요~?
    
    [['맞아요 진짜 쉬는날에도 쉬는게 아니죠....누워있고싶어도 자꾸 눈에 보이니 신경쓰여서 그냥둘수도없고 ㅠㅠㅠㅠ이와중에 고양이 너무나 이뻐요 ㅠㅠㅠㅠ'], ['그니까요ㅜㅜ\n뭐헌디점심약속은잡아서ㅜㅜ\n잇다이불널구나가야해요 흑. \n이쁘게봐주셔서감사합니당^^'], ['살면서 어쩔수 없는거 같아요 저도 먹고나서 기름때 안닦이기 전에 바로바로 닦아야지 하면서 다 닦고 자고 일어나면 머리카락이니 뭐니 바닥에 있는꼴 보기 싫어서 맨날 청소기 돌리고 물걸레 닦고...힘들어영 ㅠㅠㅠㅠ 그나저나 냥이 넘 예뻐요 오드아이ㅠㅠㅠㅠ'], ['ㅜㅜ그니까오ㅡ..끝은없을듯요ㅜㅜ\nㅋㅋ오드아이..중성화하구통실통실해졋어용ㅋ'], ['어휴 정말 저도 결혼하기도 전에 걱정이에요 ㅠㅠ'], ['꺅 !!! 고양이 너무 이뻐요 ㅜㅜ 자주 사진 올려주시면 안될까염?'], ['ㅎㅎ네네 어렵진않죠~^^'], ['고양이 넘귀여워요~~ ㅋㅋ 집안일은 끝이 없나봅니다ㅜㅜ 할수록 더 보이는거같구요 ;;'], ['집사의삶이란ㅋㅋㅋ그래도 넘 예뿌자나용ㅎㅎ'], ['ㅋㅋ맞아요ㅜㅜ\n오늘도..맛동산치우러갑니당'], ['저희는 집안일을 70-80%는 다 예랑이가 해서........ㅜ.ㅜ\n친청 식구들한테 혼나요ㅜ 같이 좀 하라며ㅜ.ㅜㅋㅋㅋ'], ['크극 반전이죠?ㅋㅋ\n사진보구잠시나마힐링하세요♥'], ['집안일은 정말 끝이없죠 ㅋㅋ 청소기 돌리고 뒤돌아서면 우리집 개님털이..허허.. 그냥 부지런해지는 수밖에 없는거같아요 ㅋㅋ'], ['ㅋㅋ그나마 개님과 냥이님들덕분에..\n집사는오늘도열일합니다'], ['끝이없는거같아요 ㅜ ㅜ 저도 예랑이 현재 혼자살구있는집 가면 할일이 엄청 많아요ㅜㅜㅜ 청소도 안해놓냐고 잔소리 엄청 하구있어요ㅜ 에휴 ㅜㅜ'], ['ㅜㅜ남자는..잔소리를..안할수없는듯요ㅠㅎ'], ['집안일 ㅠㅠ 해도해도 먼지는 계속 보이고 너무 힘들거같아요 ㅎㅎㅎ'], ['맞아요ㅜ엄마가새삼..존경스럽습니다'], ['냥이가 너무너무 이쁜데요 꺅 >< 그나저나 집안일은 정말..ㄷㄷ 결혼하고나면 가사분담 확실히하려구요'], ['ㅋㅋ그러셔야죠\n저흰 신랑이 쉬는날은 하구,\n오늘은 제가삘받앗어요ㅎㅎ'], ['끼양 ㅠㅠ 너무 이뻐요! 저흰 강아지 있는디 이눔시키 뒷바라지도 은근 손 많이가요 ㅎㅎ 요즘 아파서 ㅠㅠ'], ['ㅜㅜ그쵸...\n아프면더케어신경쓰셔야겟어용'], ['고양이>_< 저두 고양이 키우는데.. 집에 고양이 있으니까 집안일이 훨~씬 더 늘더라구요 ㅠㅠㅠ'], ['ㅋㅋ맞아요ㅜㅜ\n어휴..ㅋㅋ화장실치우다 애들 오줌냄새에 눈이시려요ㅜㅡㄴ'], ['집안일 끝 없어요;...진짜 하다보면 하루다가있어요 잘시간 ㅠㅠ'], ['아침에일어나면 집안일 저녁에퇴근하고나면집안일 주말에도 집안일 신랑이랑 같이하는데도 집안일은 계속생겨요 ㅋ'], ['집안일이 그렇죠 뭐ㅠㅠ 오히려 평일에 회사와있는게 쉬는것도같아요 ㅋㅋ'], ['해도해도 끝이없는데, 해도해도 티가 안난다는 ㅎㅎ'], ['팩트입니다!!ㅎㅎㅎ 티가..'], ['집안일은 하루종일도 하려면 할 수 있을것같아요~ 고양이는 사랑입니다~♡'], ['정말 집안일의 끝은 없는것같아요..ㅠㅠ 주부의 경력도 인정해줬으면..'], ['ㅎㅎ그러게요ㅜㅜ\n오늘저녁반찬은뭘할까요..'], ['정말정말 끝없이 집안일할꺼리가 나오는거같아요..이거끝내면 이거신경쓰이고 ㅋㅋ'], ['ㅋㅋ눈을감고잇을수도없고..ㅋㅋㅋㅋㅋ무한릴레이에요'], ['똑같이 집안일하고 지내욬ㅋㅋ 끝이없습니당 ㅋㅋㅋ'], ['집안일 정말 끝이 없죠 ㅠㅠ 그와중 고양이님들 세상편하네요 ㅋㅋㅋ 엄청 예뻐요!!ㅋㅋ'], ['ㅋㅋ감사합니당^^\n뜨뜻한햇볕에누워서자는냥이들..새삼부럽네요ㅜㅜ'], ['어머 냥이들 너무이뻐요! 저도 집사예요! 냥이털이 진짜 집안일 늘리는데에 한몫하죠 ㅋㅋㅋ'], ['ㅋㅋ맞아요ㅜ\n오늘은장난감을..사망시켜놧더라구요...ㅋㅋ'], ['진짜 맞아요...저도 주말에 아침 8시반부터 일어나서 계속 빨래하고 청소하고 화장실청소하고 ㅋㅋ진짜 하루가 너무 짧더라구요'], ['ㅋㅋ엄마한테얘기햇더니\n원래그런거라고...ㅜㅜ.전이제시작인데.ㄱㅋ'], ['냥이 너무 이쁘네요. 맞아요. 집안일은 표도 안나는데 계속해서 나오는 일인 것 같아요.ㅠㅠ 쉬는 날 집안일도 하시고 신부님이 참 부지런하시네요. 전 게을러서ㅜㅜ..'], ['진짜 집안일은 티도 안나고 끝도 없더라구요 ㅜㅡㅜㅋㅋ 아직 결혼 전이긴하지만 휴일이나 주말 때 청소하다 하루가 다 간다능용'], ['맞아요ㅜㅜ진짜 눈에 왜이렇게 잘 보이는지...'], ['힝ㅜㅜ 저도 남편 눈치보여서 청소하고 막 그래요 쉬는 날ㅜㅜ 쉬는데 암 것도 안하면 눈치를 엄청 줘서....ㅋㅋㅋㅋ 그와중에 고양이들 너무 귀여워요! 오드아이네요!'], ['냥이들 너무 매력적이네요~집안일..저도 딱히 해본적이 없어 결혼하면 걱정입니다 ㅠㅠ'], ['냥이들 너무 귀여워요 ㅜㅜ 집안일은 해도해도 끝이 없죠.. 그맘 이해해요'], ['집안일은 해도 끝도없고 티도안나고 그렇죠ㅜㅜ 고양이 너무 귀여워여'], ['꺄 냥이 ㅠㅠ 부럽네요 집안일은 티가날라면 밖으로나온 모든걸 집어넣으면 엄청나게 깔끔..ㅋㅋㅋㅋㅋ진짜 어렵져'], ['집안일 힘든거 같아여 전 신랑이 집안일 거의 다 해주는데도 매일 치울게 생기고 빨래도 그렇고 ㅜㅜ 엄마가 얼마나 힘들었을지 이해 갑니다 ㅎㅎ'], ['맞아요....\n엄마한테 하소연하면...ㅋㅋ현실이라고...ㅋㅋㅜㅜㅜㅜ인정해야죠'], ['결혼을 하면 쉬는게 쉬는거 같지 않을 거 같아 저는 약간의 기대반 걱정반이에요ㅜ고양이 너무 귀엽네요!!!'], ['우왕 오드아이당 저희집 터앙이에요ㅋ.ㅋ'], ['오마나~~ 이뽀라^^'], ['꺄~!!!고양이 너무 이뻐요>_<\n진짜 ..집안일은 끝이없는거같아요....'], ['집안일은끝이없는것같아여ㅜㅜ그와중에\n냥이들너무귀엽네요♡'], ['저도 쉬는날 폭풍 집안일했답니다 ㅋㅋㅋㅋ'], ['집안일 해도해도 끝이없고 해도 티안나고 금방 다시 지저분해지고ㅠㅠㅠ 으'], ['집안일 해도 티도 안나고 넘 힘들죠'], ['그 와중에 냥이들 카..... 카와이ㅠㅠㅠㅠ 근데 집안일은 정말 끝이 없더라구요ㅠㅠ 저희는 그나마 남편이 좀 도와주는 편인데도 청소하고 빨래하고... 저희는 남편이 음식담당, 저는 설거지 담당이라 맨날 어질러 놓는 남편 때문에 가스렌지도 평일에는 저녁 해먹고 꼭 닦곸ㅋㅋ 주말에는 먹을때마다 닦고... 정말 해도해도 끝이 없어요!!!!!'], ['ㅋㅋ어쩌면..음식담당이나을수도잇을듯요ㅜ 어휴ㅋㅋ설거지가한가득이에요ㅜ특히신랑이음식하면..설거지뿐만아니라..ㅋㅋ가스렌지공감이요'], ['월요일 휴무였는데 저도 차 써야해서 예랑이 출근 시키고 혼자 빨래방 가서 한시간반... 집와서 세탁기 돌리고 널고를 세번을... 둘다 일을 해서 일주일에 한번 몰아서 하다보니 휴무인데 쉬는게 쉬는게 아녜요 늘 ㅠㅠ 빨래로 휴무를 날렸어요 더 피곤한 기분....'], ['ㅜㅜㅜㅜㅜ휴무란..분명쉬는건데ㅋㅋ어째더바빠버리죠ㅜㅜ....현실이라죠..흑ㅜ같이힘내욧'], ['저도오늘 간만에 쉬는데 할일이 태산이네요 ㅋㅋㅋㅋ 아무것도 하기싫어 죽겠는데 빨래는 쌓여있고 청소기도 돌려야할것같고ㅠㅠ 근데 오드아이냥이 너무 이뻐요 ㅋㅋㅋㅋㅋ'], ['냥이가 너무 이쁘네요 ㅎㅎ 집안일은 정말 하나부터 시작하면 끝도 없는거같아요...날잡아서 하는게 정말 답이예요ㅠ'], ['아침부터 부지런하시네용 저는 게을러서 걱정이에요 ㅠㅠ 신부님처럼만 잘 하고싶은걸용!!'], ['냥이 키우셔서 더 청소가 많을거 같아요 ㅜㅜㅎㅎ 아가들 넘 이쁘네요~~'], ['저는 이불빨래 엄두도 못내고 있어용 ㅜㅜ ㅋㅋ 에흉 ㅋㅋ'], ['정말이예요.. ㅠㅠ  저는 어제 세탁기 청소랑 화장실 청소 했어요... ㅠㅠ  정말 끝없는 집안일.. ㅋㅋㅋ'], ['아니!!! 이건 집안일 푸념을 가장한, 예쁜고양이 자랑 아니신가요?ㅋㅋㅋ 고양이 넘 이뻐요~~!!'], ['앗ㅋㅋ들켰네요ㅋㅋ감사합니당^^'], ['쉬는날에는 더더 눈이 띠더라구요ㅠㅠㅠ 근데 냥이들 미모 뿜뿜이네요....'], ['어머~ 오드아이네요~ 아이들 너무 예뻐요 ㅎㅎ\n전 아직 입주전인데... 코딱지만한 집에 청소가 매일 끝이 없어요 ㅋㅋㅋㅋㅋ'], ['저도ㅜ코딱지만한데..일이태산이네요ㅜㅜㅎㅎㅎ'], ['동물키우면 아기 키우는거랑 똑같다는....ㅎㅎㅎ 힘내세요 ㅠㅠ'], ['끼양 고양이 너무 귀여워요 집안일은 정말 끝이 없죠 ㅋㅋ\n치워도 티 안나고 안 치우면 티 확나고 ㅋㅋㅋㅋ 저도 어제는 남편이랑 등산갈까하다가 비온다고 해서 그냥 청소하고 쓰레기 버리고 분리수거하고 쇼파토퍼 털고 바닥 물걸레질하고 그랬어요 ㅎㅎ'], ['고양이 너무 귀여워요 ㅎㅎㅎ 집안일은 진짜 끝이없을거같아요ㅜ'], ['저는 청소하는거 너무싫어하는데 벌써부터 걱정 한가득이네여ㅠㅠ'], ['ㅠㅠ청소하는거 진짜 시러요 ㅠㅠ\n저희두 이주?삼주에 한번씩 맘먹고 대청소하는데\n하루왠종일걸려요..하 ㅋ']]
    
    305
    제일 하기싫은 집안일 맘님들은 진짜 짜증나는  젤 하기싫은집안일이 머에요?전 빨래개는거요ㅜㅜ진짜짱나요ㅋ오죽함 널려잇는거 입고싶어요그러고 다입음 또 빨고  널고 바로입고ㅋㅋㅋ진짜하기싫어용ㅋ
    
    [['저도 제일 싫은게 빨래 개키는 거네요 ㅋ'], ['왤케하기싫은지요ㅋ낼오전에 비온대서 개야되는데 미치겠어요ㅋ'], ['요리하는거요.'], ['저두 제가 장사할땐 쳐다도보기싫더니ㅋㅋ이제는 좀나아졌어요ㄱ'], ['저두 그래용. 글구 화장실 청소요.'], ['앗 맞아요ㅜ화장실물때청소 너무 싫어욧'], ['글구 개수대 청소도 싫어요. ㅋ'], ['맞아요ㅜ전 싱크대 음식찌꺼기 걸려주는 그거 있잖아요ㅋㅋ너무싫어요ㅋ'], ['전 설거지요 ㅋㅋ 그래서 설거지는 신랑 담당이에요 주말엔 설거지가 산을 이룬다는.... ㅍㅎㅎ'], ['설겆이는 싹다하고나믄 먼가 뿌듯이라도 한데ㅜㅜㅋㅋ그래도 신랑이 담당 해주시니 부러버용~'], ['전 청소좋아하고\n설거지도 가품 많이나서 재밌고\n빨래 널고 개고도 그닥 좋아하지도 않지만 싫어하지도 않아요 ㅋ\n근데 밥하는건 싫어요 ㅜㅜ'], ['ㅋㅋㅋ매일해야되는데ㅜㅜ 짜증나도 하긴해야죠?ㅜㅜ'], ['빨래 개고있음 한숨이 팍팍!어느순간 제 속옷,양말은 걍 둘둘말아 넣어둬요.\n화장실청소도 넘 싫어요.\n특히 변기닦는거ㅜㅜ'], ['헐 저도요 ㅜㅜ 하다가 제꺼는 막 그냥 넣어요 ㅜㅜ'], ['개는것보다 전 갖다넣는거ㅜㅜ왜케 귀찮을까요'], ['맞아요 저도 그거 포함해서 너무 싫어요 ㅋㅋ'], ['아 맞다 ㅜㅜ 그것도 싫으네요 ㅋㅋ'], ['저두 빨래 개는거요.. ㅠㅠ\n그래서 수건은 신랑있을때 개요.. ㅋ\n애들꺼는 던져놓았다가.. 청소할때 어쩔수 없이 개네요..'], ['그래서 저는 수건 두번만 접어서 넓게 그냥 쌓아놨어요 ㅋㅋ'], ['아하~~  \n공간이 작으니 접긴 한데..\n저두 그러고 싶네요'], ['전 빨래 갠거 갖다놓는거요ㅋㅋㅋㅋㅋㅋ\n빨래 개는건 드라마보면서 하면 시간 잘가는데 궁디떼는게 넘 어려워요ㅠ'], ['ㅋㅋㅋㅋㅋㅋ 여기저기 포진된 서랍에 넣는것도 너무 귀찮아요 ㅋ'], ['밥하는것과 빨래 개는거 완전 싫은데 그래도 꼽으라면 밥하는거요..ㅎㅎ\n누가 매일  밥만 좀 해주면 좋겠네요..ㅎㅎ'], ['ㅋㅋㅋ 매일 매일 해야 되는거고 먹어야 사니깐 진짜 하기 싫어도 해야 하니 ,,,, ㅋㅋ 우렁각시 잇었으면 좋겟어요 ㅋ'], ['설거지요ㅠㅠ 지금도 하러가야되는데 밥잘사주는예쁜누나 시청한다고ㅋㅋㅋㅋㅋ 뒷전이네용ㅎ'], ['요새 그드라마가 그렇게 잼나다면서요??ㅋㅋ'], ['네ㅋㅋ 정해인이 그렇게 멋쩌보일수가 없네용ㅎㅎ  대리 만족이랄까~~~;;;'], ['셔츠 다림질요ㅋ\n그래서 잘안해요ㅋㅋㅋㅋㅋ'], ['직장 당기는 남편을 두면 그것도 그렇네요 ㅋㅋ 그런점에서는다행이에요 ㅋㅋ 다림질 진짜 못하는데 ㅋㅋㅋ'], ['전 설거지요. . 식기세척기.살려구요. .ㅋ'], ['믿을만하게 잘 해주나요? ㅋㅋ 저도 사고 싶네요~ ㅋ'], ['바닥 닦는거요'], ['맞아요 글서 긴 밀대 샀는데 또 다딱고 나면 그 밑에 걸레 빨아야 되고 ㅋㅋ'], ['ㅎㅎ저도 빨래개는게싫어요ㅠ\n그래서 요즘머리써요\n둘째오기10분전에걷어서 미친듯이 속력내서 개켜요 그리고는 안넣어두면 애가 아까상황만들테니 미친듯이 넣게되는ㅋㅋㅋㅋ 닥쳐오면 하는걸 요렇게써먹습니다'], ['스스로 할수밖에 없는 다급함을 만드시는거네요 ㅋㅋ'], ['저는 요리하기  젤힘들어요ㅜㅜ   그리고 진짜 빨래 너는거보다 개는거 더더더 시르네요ㅜㅜ'], ['맞아요 저도 개는게 싫어요 ㅋㅋ'], ['전 빨래개키는것두 괜찮은데 갖다넣는거요... 그게 왤케 싫을까요? 그래서 개서 쇼파에 그냥 놔둘때가 많아요ㅋ'], ['그러다 애기가 막 헤쳐놓음 또 폭발해요 ㅋㅋㅋㅋㅋㅋㅋ'], ['빙고~~~ 괜히 초딩큰애한테 정리안하냐고 소리지르는요ㅋㅋㅋ'], ['설거지랑 빨래 갠거 옷장에 넣는거요ㅋㅋ\n지금도 쇼파에 개어둔 빨래 있어요ㅋ'], ['ㅋㅋㅋㅋㅋ 애기가 공격개시 하기전에 잘 치우셨나요??ㅋㅋ'], ['예전엔 설거지였는데 점점.. 빨래개는거에요.. 예쩐엔 너는것도 그랬느데 건조기들인후로..사람이 참 간사한게 이젠 개는것도 싫....어요 ㅋㅋㅋㅋ'], ['우왕 건조기~~ 부럽네요 ㅋㅋ 대신 말려주는 건조기 처럼 대신 개주는 기계는 없죠??ㅋㅋ'], ['글쵸\n\n개는건엄지여ㅜ고민하다 두어달전에건조기삿ㅆ어요 넘맘에들어요 저렴할때사서ㅋㅋㅋ 잘쓰구있어용'], ['전 빨래너는거요 ~'], ['그럼 건조기를 추천~~~ ㅋㅋ'], ['빨래 개는것보다 저도 넣는게 넘 귀찮아요 ㅜ'], ['그런게 해주는 로봇은 왜 없나요 ㅋㅋ'], ['저도 빨래하고 널고 걷어서 개야하고 진짜 태산요 예전에 안그랬는데 말이죠'], ['저두요 갈수록 하기 싫어져요 ㅋㅋ'], ['맞죠..갈수록 하기 싫어지는 뭘까요ㅜㅜ만사 귀찮음요ㅜㅜ'], ['걸레빨기.. 너무 싫어요 일회용물티슈 안쓰고싶은데.. 걸레 빨래가 싫네요'], ['맞아요 그래서 물티슈 쓰는 경우도 많지요~~ ㅋㅋ'], ['저두요 빨래개는거 그리고 요리요ㅋ아그냥 집안일 다요ㅋ'], ['증답~~~~~~~~~~~~~~~~ 그것이 정답입니다 ㅋㅋ'], ['저는 휴일날 애들한테 기빨리는거...공포예요'], ['내일까지 기빨리는데 ㅜ 잘 버티고 계시죠??'], ['반찬하는거랑 집안일은 아닌데 철마다 옷정리 하는거요'], ['맞아요 맞아 ㅜ 철마다 옷정리 그것도 진짜 ㅜㅜ 세상 귀찮죠 ㅋ'], ['저는 다~~~요ㅎ\n결혼안하고 걍 엄마집서 살걸그랬어요ㅜㅜ'], ['헐 진짜 고것이 정답이네요ㅋ']]
    
    357
    신랑님들 집안일 마무리까지 잘해주시나요? 신랑이 오늘 저녁 설거지도 해주고 젖병도 씻어주네요ㅎㅎ 저야 좋은데...설거지는 잘해놨는데 젖병도 잘 씻어놓고근데...  왜 젖병소독기에 안돌리고 바로 건조대에엎어 놓냐구요;;;; 설거지 실컷 잘 해놓고 한소리 하게 만드네요ㅋㅋ 한소리 해놨디 다음부터 안한다네요ㅋㅋ못삽니다 아주~마무리는 잘했다고 칭찬해줬는데 내가 안거들어도 알아서 해주면 안되나요신랑님들 집안일하면 알아서 마무리까지잘해주시나요?
    
    [['그래도 부럽네용ㅎㅎㅎㅎ\n저희 신랑은 마무리까진 바라지도 않으니 시작이라도 젭알좀 해주셨음 좋겠네여ㅎㅎㅎ ㅠㅅㅠ'], ['저희 신랑도 그닥 자주해주는게 아니라 가끔 해주는것도 참 감지덕지 해야 되겠지요?ㅎㅎ'], ['설거지해주고나면 마무리는 제가해요..ㅎㅎ싱크대물난리에 그릇 수저 지멋대로 엎어져잇네요ㅎ'], ['아오 어떤 상황일지 한번에 알겠네요 그러면 차라리 내가 하고싶고 시켜놨디 더 저지레해놓네요ㅋㅋ'], ['ㅋㅋ저희신랑 워낙 깔끔이라...저보다잘해요ㅋㄱ다만ㅋㅋ자주해주지않는거와ㅋㄱ하면서잔소리가득해요'], ['우와 신랑님이 깔끔하군요 그럼 한번할때 제대로 하겠네요 저희집이랑 반대인가요?ㅋㅋ'], ['너무 제대로해서요 눈치비요 해준다칼때마다...ㅋㅋ그냥 내가할게 하고말아요 ㅋㅋ그래도 전 맘님 남편님처럼 자주해주셧음좋겟네요 ㅋㅋ'], ['그냥 차라리 맘님이 하는게 맞겠네요 저희신랑도 그렇게 자주해주는게 아니에요ㅠㅠ'], ['네ㅋㅋㅋ제가하는게 편하지요ㅋㅋ여기는 닦니안닦니ㅋㅋ더럽니안더럽니 머리아파요'], ['아오 생각만 해도 스트레스 제대로 상승되겠어요 피곤해도 내가 하는게 낫겠네요ㅎㅎ'], ['저희 신랑인줄..\n저희 신랑은 그릇도 얼마나 잘 깨 먹는지..\n일이 더 늘어요..'], ['아 진짜요?ㅋㅋ 그릇 깨먹으면 일이 더 늘긴하겠네요 그렇다고 안 시키기엔 그렇구요'], ['저도마무리 제가 하네요 ㅋㅋ \n손이 가요 손이가~~ \n그냥 못넘기는 제성격탓도 잇겟지요? ㅋㅋ'], ['마무리까지 제대로 해주면 얼마나 좋을까요 한번은 눈 딱 감고 넘길만한데 잘안되지요ㅎㅎ'], ['저도 신랑이해도집안일 뭔가 조금씩 손이가요ㅋㅋ항상 10프로 부족한느낌이랄까?ㅋㅋ그래도 해주니 고맙다고해요ㅋㅋ'], ['다들 뭐든 만족스럽진 않을것 같아요 조금 부족해도 해주는거에 고마워해야 하나요ㅎㅎ'], ['남자들 늘 마무리가 좀 부족하지요\n그래도 잔소리 안해요\n그나마도 안할까싶어서'], ['맞아요 괜히 잔소리 하면 그마저도 안하니 뒷마무리는 신랑 몰래 하는게 낫겠지요ㅎㅎ'], ['저는 시작도 마무리도 제가 합니다ㅎㅎ\n그래야 두번 손이 안가요ㅠㅠ'], ['모든 신랑님이 그러하겠지만 그래도 한번씩은 시켜야해요 안그럼 계속 안하려하지요'], ['ㅋㅋㅋㅋ 저도 항상 맘 편하게 제가 마무리해요~ 그래도 자기딴에 해주겟다고 하는게 고마워서요 ㅎㅎ'], ['ㅋㅋ 해주겠다 하는 마음 너무 이쁘지 않나요?마무리는 맘님이 하시지만 그래도 해주니 고마운거지요'], ['울신랑 다넣어주는데 혹 제가 늦게뭐주고이러면 잔소리합니다ㅡㅡㅋ'], ['아 진짜요?신랑님이 오히려 잔소리를 하시는군요 그래도 가끔 집안일 해주는 신랑이 고맙지요~'], ['저희집 바깥양반은 오늘만해도 욕실청소에 거실청소에  화분에 물도주고하더라구용ㅎ'], ['이야 집안일 아주그냥 너무 잘 도와주시네요 넘넘 부럽습니다요ㅎㅎ'], ['알아서 척척이라 어떻게보면 저는 주방살림만 책임지면 된답니당ㅋ'], ['알아서 해주는 신랑님들 너무너무 부러워요ㅋㅋ 뭘 하더라도 입 한번은 떼야해서 피곤하지만 안해주는것보다 낫네요ㅎㅎ'], ['청소는 특히 알아서 하기에 입델게 없더라구요ㅎ 전행복한축이지용ㅋㅋ'], ['김치 맘님네는 완전 완벽한 신랑님 얻으신거에요 짱 부럽습니당ㅋㅋ'], ['저는 냉전 결과 일일이 시켜야 한다는걸 알게됐습니당ㅋ 마음비우고 일일이 다 시키려구요ㅋㅋㅋ'], ['아 진짜요?ㅋㅋ 일일이 얘기해주고 하는게 귀찮긴해도 그게 마음 편하지요ㅎㅎ'], ['하나부터 열까지 일일이 말해줘야 하는게 입아프긴 하지만 제일 안심되긴 합니다ㅋ'], ['어떤거는 일일이 얘기안해줘도 알아서 해주는데 결론은 일일이 하나하나 얘기해줘야하네요'], ['네 마자요ㅎ 하던 저희가 하는게 맞긴 한데 애보랴 집안일하랴 너무 힘드니 가끔 도와달라는건데ㅋ 성에 안차는게 문제지요ㅋ'], ['조금이라도 도와줘야 버티지요 아니면 힘들어요ㅠㅠ 성에 안차지만 그래도 해주면 영 낫지요ㅎㅎ'], ['우리집 랑이는 저보다 더 꼼꼼해서 오히려 저보고 늘 지적질이네요 집안일 다 해주고 지적질하니까 말도 못하겠구 ㅠ'], ['신랑님이 엄청 꼼꼼하시군요 너무 꼼꼼해도 피곤하긴 하지만 너무 안하는 사람보단 나을것 같아요'], ['하는거에 의의를두고 마무리는 제가해요ㅋㅋ 예를들면ㅠ 설거지하고나서 온사방천지에 물튄거 제가 닦는다던지 ㅜ 설거지건조대에 엉망으로 놔둔거 제가 바로 놔두던지 그런거요ㅎ'], ['아 역시 남자들은 그런것조차 마무리 안해놓고 다했다고 뿌듯해하지요 거의 비슷한가봅니다ㅋㅋ'], ['저희신랑은저보다더꼼꼼한성격이라....가끔제가민망해서화낼정도..ㅠㅠ\n넘잘해도완젼피곤하답니다..'], ['꼼꼼한게 좋을때도 있지만 남자가 너무 깔끔떨어도 피곤하더라구요 그래도 아에 안하는 신랑보단 낫구요ㅎㅎ'], ['마무리는 무슨요ㅋㅋ집안일이라도 하면 그날이 이상한날인걸요으익ㅋㅋ 그저부러울따름입니당'], ['ㅋㅋ 아 진짜요?저희신랑도 자주 해주는건 아닌데 요래라도 가끔 해주니 좋긴하네요'], ['ㅋㅋ저희신랑은 싱크대물기 다닦고 가스렌지까지 청소해줘여....ㅋㅋㅋㅋ 잘안해서글치.한번씩해주면 꼼꼼히하네요'], ['우와 제대로 해주시네요 자주 안해도 가끔씩 할때 제대로 해주면 너무 좋을것 같아요'], ['제스타일로 제자리에위치해놓지않지만 무조건잘했다고칭찬해요~ 그래야 자주해주더라구요ㅎ안볼때 제가다시 손대요~'], ['ㅋㅋ 폭풍칭찬 해줘야 자주 해주는건 맞지요 제자리 안갖다놔도 해주는거에 만족해야겠지요ㅋㅋ'], ['저희 신랑은 아.주.가.끔. 해주지만 마무리는 잘해주네용^^'], ['아주가끔ㅋㅋ 그래도 해줄때 마무리까지 잘해주니 다행이지요 아에 안해주는것보다 낫겠지요ㅎㅎ'], ['젖병소독기 물기 말리고 돌려야하는거 아닌가용?ㅋ 저희신랑은 애기꺼 빼고 설거지한적도있어요ㅋ'], ['아 진짜요?저는 물기있는채로 소독기 돌려요 소독기에서 완벽히는 아닌데 건조해서 소독되더라구요 해주는김에 같이 해주면 좋을텐데요'], ['전 물기있는 상태에서 넣지말라고  알고있어서용~ 저는 소독기있어도 잘안쓰고 열탕했지만요'], ['이때까지 물기안말리고 소독기에 넣었거든요 물기까지 빼고하려면 젖병 미리미리 씻어놔서 물기 빼야겠는데요 저도 열탕소독이 차라리 편하더라구요'], ['네네ㅋ 저도 물기다빼고 해야된대서 소독기안쓰고 열탕만했었어요~ 혼합하다가 번거로워서 결국 완모로 바꿨지만요'], ['그랬군요 열탕이 편해요 저도 젖병씻고 소독해보니 완모가 정말 편하구나 느꼈어요'], ['ㅎㅎ노노~전혀요~\n저도 꼭 잔소리하게 만들더라구요~\n설거지 다 하고 물기 안 닦고 그냥 두는거...맨날 잔소리해요ㅎ'], ['맘님네 신랑님도 그렇군요 남자들 한가지 밖에 생각 못하는건 어느집이나 비슷한것 같네요ㅎㅎ'], ['네~그러면서 잔소리하면 꿍시렁대요ㅎㅎ그럼 저는 잔소리 안 하게 끔하라고 더 잔소리 해요ㅎㅎㅎ'], ['ㅋㅋ 아 진짜요?저도 잔소리 한번 하고 말았네요 괜히 더 했다가는 안해줄것 같아서요ㅎㅎ'], ['마무리는 저도 기대 안합니당.. 그런데 이정도만 해줘도 전 고마울듯합니당ㅎㅎ'], ['ㅋㅋ 맘님네도 그런가요 맞죠 완벽한거 바라진 않지만 해주는거에 감사해야 되겠지요ㅎㅎ'], ['이정도도 안하기에.. 만약에 한번 하더라도 생색 오만상내고 잔소리하니 안하는게 전 더 편하기도..'], ['남자들은 다그런거같네요 저희남편도 설겆이하고는 싱크대 물흔건하게 놓고 욕실청도도 물바다로만 만들어놓고~~그래도 하는게에 의미를두고 오구오구 해줘요~~~ㅋ'], ['아오 댓글보는데 주먹이 불끈불끈합니다 저희신랑이랑 우찌그래 똑같은지요 특히 욕실 보다 못해요 해주는거에 감사해야겠지요ㅋㅋ'], ['네네ㅋ 하는거에 감사하고 오구오구 해야 더 늘죠~~ㅋㅋ\n잔소리하면 그마저도 안할까봐ㅋㅋ'], ['ㅋㅋ 안그래도 신랑한테 한소리했디 안하겠다면서 달래가 고맙다고 다음에 또 해돌라했네요ㅋㅋ']]
    
    396
    집안일 중 뭐가 제~~~일 귀찮으세요?? 전... 빨래 개키기가 제~~~일 귀찮아요!코 간질간질... 다시 정리하는곳도 귀찮고...아침 드라마 보며 빨래 개키기 시작하다 배도 안고픈데 아침에 신랑이랑 아들이 먹고 남은 김치볶음밥 먹고 다시 개키고 정리하고 초코파이 하나 먹고 이제 퍼질러 앉았습니다! ㅎㅎㅎ달달한 믹스 한잔 하고 이제 청소기 돌려야겠어요! 일회용 옷 사용을 규제해야하지만 옷도 일회용 나오면 좋겠어요! ㅋㅋㅋ
    
    [['전 설겆이요, 그릇까지 먹을 수 있음 좋겠어요ㅋㅋ이유식만 만들어놓고 손 놓고 있네요, 아우~~~'], ['그릇까지 먹는... ㅋㅋㅋ 빵 터졌네요! ㅎㅎ'], ['아~~전그냥 집안일요ㅋㅋ\n집안에서 하는일이 귀차나요.\n나가고픈데 추워서 쇼파에있는데.\n집에 김정은이 핵폭던지고간듯한 현장이라ㅋ'], ['ㅋㅋㅋ추운 덕분에 저도 집에 앉아서 빨래 개켰네요!ㅋㅋㅋㅋ'], ['전 아드님 장난감정리요..왜ㅜ아이방 청소는 해도해도 티가 안날가요 ㅋㅋ'], ['아이가 아직 어린가봐요~ 전 이제 작은 장난감 들이라 종류끼리 통에 담아서 차곡차곡 쌓았더니 숨겨져서...ㅎㅎㅎ'], ['아,, 오늘 댓글들 완전 빵터지는데요ㅎㅎㅎ\n저도 아침에 아이 소풍간다고 집안 개판 만들어놓고 왔는데ㅋ\n저녁에 치울생각하니 끔찍하네요ㅋㅋㅋ'], ['워킹맘이신가봐요~ 죙일 집에 있는 전업맘도 집안일 귀찮은데 워킹맘들은 진심 존경합니다!!'], ['네ㅜㅜ\n그래서 매일 주방이 폭탄 맞은듯해요ㅜㅜ\n누가 집좀 정리해줬음 좋겠어요ㅎ 가사도우미가 절실해요ㅎㅎ'], ['장난감 정리 등등 집안 정리요~~ 귀찮아요~~ㅠ'], ['해도해도 끝이 보이지 않고 무한반복이니...ㅎㅎ 충분히 이해하는 그 맘입니다!!!^^'], ['저도요 ㅜㅜ 빨래접는 기계있음좋겠어요 ㅜㅜㅜㅜ'], ['빨래접는 기계 있어요!!! ㅎㅎㅎ'], ['저도요!!!ㅋㅋ 얼마전에 건조기는 사서 너는 수고는 해결 했는데ㅋ 안그래도 맘님 처럼 빨래 개는 기계는 안나오냐고 얘기했는데ㅋㅋㅋㅋㅋ정말 귀차나여 빨래 개기ㅠ'], ['건조기 덕분에 저보다 조금은 덜 귀찮으시네요! 건조기 부럽습니다!!!'], ['청소 하기 싫어서 이틀 삼일에 한번 합니다. 그것도 걸레는 안하고요'], ['헉... ㅋㅋㅋ 전 쓸고 닦고는 겁내 열심히 해요!! 발바닥이 쓸데없이 예민해서...^^;;;'], ['빨래너는거요ㆍ개는것도시러요 그냥 집안일 자체가 시러요ㅋㅋㅋ'], ['ㅋㅋㅋ 모든 주부의 맘이 같네요~^^'], ['청소요 쓸고 닦고  넘싫네요  ㅎㅎㅎㅎ'], ['전 쓸고 닦고는 엄청 잘해요! 티가 안나니 문제죠! ㅎㅎㅎ'], ['먼지 닦는거요\n바닥은 운동겸 할만한데 가구들 책장 위에 먼지는 너무 귀찮아요ㅋㅋㅋ'], ['저도 동감합니다 ㅎ 먼지 닦아도 닦아도 계속 쌓이니까 ㅠㅠ'], ['빨래 개는건 앉아서 하니 그나마 괜찮은데 갖다 넣는거요ㅜㅜㅋㅋㅋ 왔다갔다 휴~~~'], ['저도 제자리 정리가 귀찮아요! ㅋㅋㅋ'], ['전 설겆이요 ㅜㅜ'], ['님에게도 먹는 그릇을 절실하군요! ㅎ'], ['저는 빨래 널기요ㅠㅠ 세상에서 제일 귀찮아요...'], ['전 너는건 잘하는데...^^ 널기도 나름의 규칙이 있어서 이뻐요~ㅎㅎㅎ'], ['전 전부다요..누가 대신 해줬으면 좋겠어요ㅎㅎㅎ'], ['맞아요! 전부다~~~ 대신 해주는 분이 있으면 좋지요!! 돈이 문제지만...'], ['저는 저녁밥이요ㅋㅋ퇴근후 아이들찾고집에가서 전쟁이라ㅠㅠ'], ['저녁밥은 문제 없어요! 저녁 메뉴 정하기가 너무 힘들어요! 워킹맘 존경합니다!!?'], ['저도빨래개는거요ㅠㅠ너는건좋은데..'], ['저도 널기만 하고 그 뒤는 남이 해줬으면 좋겠어요!^^'], ['설거지 너무 싫어해서 안해요 ㅋㅋㅋ ㅜㅜ'], ['남의 편에게 시키면 되지요!  ^^'], ['저는 걸래질후 걸래빨기요..ㅠ'], ['손빨래로 직접 하시나봐요? 전 물걸레 청소기라 걸레만 따로 모아 세탁기 돌리는데...^^;;;'], ['저 빨래개기요 ㅜㅜ몇일째 미루고 있어요'], ['ㅎㅎ 천장 건조대에 옷걸이로 널고 건조대 큰거 두개 작은거 하나 까지 꽉~~~채워 널어놨던 적도 있어요! 개키기 싫어서...ㅋㅋㅋ'], ['전 요리요..맛이없어요ㅜㅜ 망손..'], ['이건 가까운 반찬가게를 이용하시면...^^'], ['저는 화장실 청소요.......  \n특히 남편 쓰는 화장실이요......집에서 볼일 못보는 컴플렉스 가지고 있었으면 좋겠어요^^;; ㅋㅋ'], ['화장실 변긴 매일 씻기전에 습관적으로 닦아서...휘리릭 후딱 합니다!!! 바닥은 락스 쫙~~~'], ['화장실청소겠네요'], ['저도 기분 내키면 수세미로 밀지 그냥 락스로 쓰~~~윽!'], ['전 변기청소요~\nㅋㅋ 근데단어가~~~ㅋ\n빨래 개키기..\n빨래 갠다는 말인거죠????\n문장상알아는 듣는데 개킨다는\n말는 첨들어봐요~~~^^'], ['순간 사투린줄 알고 찾아봤어요! ㅋㅋㅋ'], ['방닦는거요 ㅜ ㅜ 진심 바닥이\n노란거같아요 ㅜ ㅜ플러스 화장실 청소'], ['아이만 없으면 바닥 청소도 목숨 걸고 하진 않았을꼬 같아요!!^^;;;'], ['전 청소요ㅋㅋ로봇청소기 샀는데 그거 걸레 갈아끼우기도 구찮아요ㅋ']]
    
    425
    집안일 중에서 젤 귀찮은것은 전 설거지랑 빨래개기요... 휴우.. 또 설거지가 산더미 빨래마른거가 산더미... 게으른 엄마는 늘 피곤하네요
    
    [['저랑같아요   전  설겆이도물론  바닥닦기와 이부정리넘싫으요'], ['그냥 집안일이 다 싫은건가봐요 전 ㅋㅋㅋ'], ['저도 설거지랑 빨래요!^^다 비슷한거 같아요~^^'], ['설거지 해야하는데'], ['저도요~^^'], ['전 빨래개놓고  서랍에갖다놓는게 귀찮아요ㅋㅋ'], ['맞어요 저도 바구니에 담아 2박3일 ㅋㅋㅋ'], ['ㅋㅋㅋ 저도 공감이요~ 다 개놓고 서랍에 넣기가 왜일케 귀찮은지 ㅋ'], ['빨래개서 정리하는게 ㅎㅎ'], ['저두 좀전에 3일된 갠 빨래 서랍에 넣었어요 ㅋㅋㅋ'], ['전 빨래너는거랑 개는걸 젤 좋아하는대신 옷장정리가 젤 싫어요!!!'], ['ㅋㅋ 저랑 똑같은 분이 계셨네요\n저도 빨래 서랍장에  두는걸 젤 귀찮아 하는데.ㅋ'], ['설겆이 빨래개기  쓰레기버리기요'], ['이것들만 누가 해줘도 참 좋겠어요'], ['ㅋㅋ맞아요.. 설거지 하루에몇번을하는지'], ['맨날 돌아서면 ㅠㅡㅠ 설거지'], ['찌찌뽕 ㅋㅋ저랑 같으다요~~\n건조기있으니 빨래 개켜주는 기계도 나왔으면 하는 바램이..ㅋㅋㅋㅋ\n쇼파는 빨래산이고 거기서 집어 입는게 일이에요^^v'], ['ㅋㅋㅋ ! 동감이요 전 건조기도 없어요 ㅠㅡㅠ'], ['전....소박하게 한 글자요~~\n다!\n'], ['소박한 한 글자..."다"에 웃고갑니당;;ㅎㅎㅎ'], ['빨래개면 누가 제자리에 가져다놨음 좋겠어요 ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['전 청소기랑 걸레질이 제일 귀찮네요~'], ['전 밥 안하고 싶어요 밥먹으면 설거지 생기고 설거지만 하면 왜케 다른 집안일이 눈에 들어오는지 ~~~결국 다 싫은가봐요 ㅋㅋㅋ'], ['저는 집정리가 젤 힘들어여.그릇정리,옷정리 그런거여.'], ['전 화장실가는것도 귀찮아요ㅎㅎㅎㅎ'], ['ㅋㅋ아 그럼 어떻게하죠 ㅋㅋ'], ['요기 1인추가네요~ㅋ'], ['많은분들이 동참해주시네요 ㅎㅎㅎ'], ['전 다 싫어요 ㅎㅎ\n청소 빨래 설거지 다요~ㅎ'], ['실은 저두 다 싫은데 ㅋ'], ['저도빨래개기요기계가나왔으면좋겠어요'], ['오.. 신박한 ㅋㅋ'], ['다 좋은데 밥만 안하고 살았음 좋겠어요~ㅜ ㅜ'], ['누가 해주는 밥 먹고 누가 설거지 해주는 삶은 드라마에만 ㅠㅡㅠ'], ['전 안귀찮은게 없네요....ㅠ'], ['살림이 체질이 아니에오 ㅋㅋㅋ'], ['설거지요ㅜ \n아휴 그냥 다 안 맞아요ㅎㅎㅎㅎ'], ['맞아요ㅠ적성에 살림 안 맞아요 ㅋㅋ'], ['전 밥하기...'], ['가끔 밥도 하기 싫어요 ㅋㅋ'], ['저도 설겆이 빨래개기 넣기요ㅠ 젤귀차나요ㅠ'], ['다들 비슷한줄 몰랐어요 ㅋㅋ'], ['다림질이 너무 힘들어요ㅜ\n운동화빨래도 요 ㅜ.ㅜ 그래서 \n업체 맡겨요 ㅋㅋㅋㅋ'], ['운동화빨래 하기 싫어 운동화 안신어요 ㅋㅋㅋ'], ['ㅋㅋㅋ 전 바구니애 담아 2박3일 ㅋㅋㅋ'], ['전 설거지가 젤로 좋아요 그래서 식기세척기는 평생안살거같은데 음식쓰레기 갖다버리기가 젤로 싫으네요 ㅜ ㅜ'], ['설거지 좋으신분 첨봐요'], ['음식하는거 빼고는 전부 싫어요ㅜㅜ ㅋㅋㅋ  \n음식도 가끔 너무 귀찮기도 하지만요 ㅡㅡ;;'], ['맞아요 ㅠㅡㅠ 배달음식 그리워요'], ['밥 챙기기요~~'], ['혼자살때가 그리워요 ㅋㅋ'], ['눈뜨는거부터귀찮 ㅜ ㅜ아어쩌나요'], ['저도ㅠ실은 그래요 ㅠㅡㅠ'], ['전 다림질이랑 화장실청소요ㅠㅠ'], ['앗 다림질은 안해요 ㅋㅋ 그냥 구겨진거 입어요 ㅋㅋㅋ'], ['저는청소요.\n제자리하면 또엉망.반복이 싫어요\n치우고쥐돌면 뽀얗게 먼지보이는것도싫고ㅋ\n\n설겆이.빨래는 하고나면 잠깐은 깨끗하니 좀 나아요ㅎ'], ['전 청소를 몰아서 하는데 다 하고 나면 뿌듯하더라구요 ㅎㅎ'], ['저도 저만그런줄 알았는데\n빨래개서 정리하기랑 설거지 집안일중에 젤 싫은데 ㅋ\n진짜 살림체질 아닌거 같아요~~~'], ['저는  청소가 제일 싫어요\n음식 하기는  재미있어요'], ['다싫어요ㅜ']]
    
    429
    남편 집안일 어느 정도 해주나요? 재활용 음식물 쓰레기 버리기 화장실 정리 가끔상닦기 먹은음식 정리어쩌다가 싱크대 인덕션 닦아주기 아주~~~가~~~아끔 빨래널어주기 ㅋㅋㅋ자진해서 이정돈 하는데 어떻게 분담해야 할지 아직  잘 모르겠어요11시에서 밤12시까지 일하고 오는데 뭐시키는것도 참...그렇고 해서 그래도 이거 저건 해달라 하는게 나을까요?낮엔 친정엄마가 아기 봐주시고 애도순하고 저도 하는게 별로 없는거 같긴 한데다들 남편들은 어떤지 궁금해요
    
    [['주말부부라...\n주말에 오면 설거지는 무조건 신랑이 다해요...\n대청소 해주구요...'], ['흠 ...설거지 하는걸 싫어하더라구요 \n'], ['울신랑도 청소를 두번하면 안되냐고...ㅋㅋ\n밥은 내가하니 설거지는 신랑몫...'], ['왜 싫어하는지 모르겠지만 설거지 안하면서 그릇 보고 덜닦임 잔소리해요'], ['그냥 지금 전체적인 말씀들으면 너무너무부럽네요'], ['잘도와 주고 있는건가요'], ['밥..부러워요!!!'], ['집안일 하는거 1도 없습니다 \n애낳고 나서는  쓰레기버리기는 합니다 ㅡㅡ'], ['저는 전업입니다ㆍ\n울 신랑은 저녁먹고 식탁정리\n휴일엔 한,두끼 식사준비\n눈 씻고 찾아도 저 두가지뿐이네요.^^;'], ['아 저도 전업인데 집에서 깨작깨작 일을하긴 하거든요 식사준비 부러워요..전 못먹을음식을 만들어 놓길래 손때라구..'], ['저는 신생아때 잘봐줘서 지금은 그냥 제가 다해요\n빨래는 건조기 샀고\n설거지는 식기세척기 렌탈로 쓰고있는지라\n딱히 도와달라고안하고있어요\n신생아땐 집에오면 저녁이나 새벽에우유먹여달라고하고  빨래널어주고했었어요\n밖에서 힘들게 일하는데 \n저도 일하면 모를까 \n집에있어서 전 제가 다해요\n대신 말은 해요\n나중에 도와달라할때 그때 해달라고요\n굳이 시킬필요는 없지 않을까요?\n가끔씩 하신다면'], ['쓰레기는 버려줍니다 ㅋㅋㅋ'], ['차려진, 다먹은 밥상 옮기기. 아주 가아끔 설거지. 청소는 아예 신랑몫. 빨래도 가끔 돌리구요.'], ['남편 퇴근이 빠른날은 설거지, 음식물.종량제쓰레기버리기, 목욕, 애들치카, 응가처리 해줘요~그런데 결정적인건 보통 퇴근이 늦다는요ㅠㅠ'], ['그래도 꽤 많이 해주시는것 같은데요?ㅎㅎ\n늦게 퇴근하시면 아무래도 뭐해달라고 얘기하기가 그렇지만 ㅠ 평일은 힘들더라도 주말에 집청소하는 것도 도와달라고 말씀해보세요~\n'], ['집안일은 애목욕 설겆이 신랑이하고 저는 청소 빨래 요두개요.. 그래서 신랑은 식기세척기를 노래부르고 저는 건조기를 노래부르구요 둘다질러버릴까 말까 한달째고민중이에요'], ['저도 일할때는 집안일 반반 했는데\n지금은 육휴중이라 \n집안일 제가 다하네요 \n음식물이랑 분리수거는 신랑이 하구요'], ['10시반 11시는 되야 퇴근하는지라..\n주말에 애들목욕시켜주고 애들과 놀아주는것만해도ᆢ크게 만족합니다~~ㅋ'], ['저도 만족하고 살아야겠어용'], ['힘들게 일하고 밤 11시쯤 오는데..\n그때와서도 집안일 시킨다면 서운할거같은데용... 주말에 도와주는거면 몰라두요 \n또 좀 일찍퇴근하심 모를까.. 퇴근이 늦어서..\n님이 배려해주심 남편분도 더 고마워할듯..\n지금 하시는것만으로도 괜찮아보여용'], ['저정도로 만족해야 할까봐욥ㅎㅎ'], ['상닦기.. 끝...ㅋㅋㅋㅋㅋㅋㅋ그래도 시키면 군말 없이 잘 해요^^ 정말 시키지 않아도 알아서 해주는 남자는 드문듯 해용ㅠㅠㅠ'], ['저희아빠는 척척척 하더라구요 워낙 자상하고 가정적이셔서 아빠보고 자라서 이게 맞나 했어요'], ['신랑이 평일은 9시쯤 퇴근하고\n한달에 일요일 2번 쉬고 \n토요이 공휴일도 출근합니다\n\n분리수거, 쓰레기는 버리는거 했었는데\n이젠 집안일 제가 다 합니다.\n\n아기가 아파 입원했을때\n퇴원전에 신랑이\n빨래돌리기 정리 설거지 집청소 등\n싹 다 해놓은적도 있엇어요. \n\n대신 \n평소에 제 식사관련해서 \n외식이나 고기먹고싶다하면 원없이 사주고\n식사 포장 원할때는 퇴근길이 사오고,\n집안일 관련해서 잔소리를 거의 안해서 \n저혼자 집안일+육아해도 아직은 괜찮네요\n'], ['외식은ㅎㅎ저도 먹고싶은건 잘사주네용ㅎㅎ'], ['저도전업인데, 쓰레기버리기 시키면하고, 젖병씻기시키면하고, 빨래널기시키면합니당. 자기일이라고는 잘생각안하더라구요. 집에오면 피곤하고, 제가 전업이라 거의 하니까요. 11시 12시 퇴근이라면 전 못시킬꺼같네요.\n아기순해 친정어머니가 아기봐줘.. 늦게들어오는데 집안일도 도와줘.. 정말부럽네용'], ['저정도 감지덕지 해야겠죠'], ['주말 대청소.  화장실 청소 등 하네요..   어쩌다가 설거지  빨래 하고요.. 주로 힘쓰는 대청소 위주로요. 쇼파 밑등 힘드는곳 청소는 남편 몫이요'], ['저는 전업맘이구요 \n화장실청소, 쓰레기배출(음식물포함) 분리수거는 항상 신랑이 하구요\n설거지,빨래, 아기목욕은 같이해요~근데 아기옷 빨래는 제가 꼭해요^^'], ['화장실청소랑,주말에 대청소 같이하고 나머지는 거의 제가하는거 같아요ㅋㅋ쓰레기 가끔 버려주고 늦게 오셔도 같이하셔야 집안일도 힘든거 아실거같아요'], ['저도 전업이라 주로 주말만 시켜요 평일은 아빠 씻을때 아이 같이 들어가서 씻고 나오는게 다네요'], ['분리수거해주고..\n밥먹고 뒷정리해주구여(설거지는안람)\n청소기돌리는거나 빨래너는건 가~~~끔 여유로울때 시키면!! 한번씩 해주네용 \n애기 우유주는거는 잘하구영\n저도 남편이 더 도와줘야한다고 생각은하는데 제가 휴직중이라 지금은 봐주고있어요 ㅋㅋㅋ'], ['이거 자랑글이에요 ㅋㅋㅋ\n저는신랑이 너무 일찍일어나야해서\n일할때는 잘 안시키고 \n쉴때\n얘들목욕하거나 쓰레기버리거나  빨래개기 그런것 시켜요\n'], ['저는 신랑이 많이 하는것같네요..자랑은 아니구요.시기적으로 결혼 3년뒤..연년생출산으로 몸이너무안좋아서 그때부터 전반적으로 많이 해주는것같아요~~'], ['맞벌이라서. 집안일 완전히 반반해요. 남편이 아침하면 제가 저녁하고, 제가 청소기 밀면 남편이 밀대 밀고...아이낳고 육아하게 되면 어쩔 수 없이 엄마쪽이 육아를 더하니깐 남편이 집안일 더 한다고 했어요.'], ['주말부부인데 주말에 애들과 책임지고 놀아주기. 산에도 데리고가고 놀이터에서 놀고. 전 그것만 해줘도 괜찮더라구요. 애들 데리고 노는 동안 전 쉬고 있구요. ㅎㅎ'], ['요리와...음식물쓰레기,재활용분리수거...100%신랑몫..\n가끔 아들 목욕시키기와 아들 밥먹이기?(제가 연주하면 시간이 늦어지므로ㅋ)\n\n설거지(이건 제 스트레스 해소용이라 무조껀 제가ㅋ)랑 청소기 돌리기? 정도 합니다ㅎ'], ['안해요....'], ['오늘부터라도 조금씩 시켜보자구용!!'], ['전업인데 쓰레기. 종이분리수거만해줘요ㅋㅋㅋ가끔큰애씻겨주고'], ['빨래널고 개고\n집대청소도해주구요\n주방청소다 다도아주고 설거지도자주해줘요\n음식물쓰레기랑 화장실쓰레기도 항상 신랑이해줘요'], ['음~~ 하숙해요" 남편님~ ㅋㅋ\n냉장고 문도 안열어봐요~'], ['ㅋㅋㅋㅋ 하숙ㅎㅎㅎ제가남편한테 하숙하고 싶네요'], ['그런건 다음생에 댓글다는걸로 할께요~ 맞벌이지만 손가락 하나 안움직입니다. 부모가 잘못 키운탓이겠죠! 그래서 다음생은 꼭 부모를 보고 결혼할라구요 ㅜㅜ'], ['아무것도 안해요ㅠㅠ'], ['조금씩 시키셔요!!!힘을내보자구요'], ['저희신랑은시체입니다 돈만벌어와요 ㅋ ㅋ ㅋ ㅋ ㅋ ㅋ'], ['ㅋㅋㅋㅋ돈벌어오는게 만족이면 뭐ㅎㅎ좋지않을까요'], ['애기없는집이라 전 제가 다해요 그래서 함씩 머 가꼬와죠함시롱 은근 시켜먹는중 ㅋㅋ'], ['평일에는 퇴근 후 애 봐주고 씻기고 먹이고 하는거 외에는 집안 일은 하나도 안시켜요\n낮에 제가 하면 되니깐 애만 딱 봐달라해요\n주말에는 딱 반반하고 애는 신랑이 주로 봐용'], ['숨만쉬네요\nㅠ'], ['에잇 숨도못쉬게 막아 버리세요 ㅂㄷㅂㄷ'], ['그냥 안하는게 절 도와주는겁니다~~!!!딱히 하라고도 안해요..'], ['요리, 밥하는거 외엔 안합니다. 설거지는 가끔 하고 그외는 아무것도 안합니다.']]
    
    440
    집안일 중 제일하기 싫은일은 뭔가요? 저는 빨래요ㅜ너는것도 개는것도 다 싫으네요신랑도 이거는 진짜 하기싫은일이라며신랑한테 해달라고도 못하는 유일한 저의 일입니다ㅠㅠ
    
    [['걸레질이요ㅋ 걸레빨기 넘나시름'], ['아ㅋ걸레질이 있었네요 저도 걸레 진짜 빨기싫더라구요 물티슈대체해서 잘쓰지요ㅋ'], ['맘님도 그렇군요 저도 세탁기넣고 돌리는거부터 마지막 개서 챙겨넣는거까지 다요ㅜ'], ['욕실청소요..빨래나 설거지는 좋은데..욕실청소 너무 하기 싫어요.ㅜ'], ['맞아요\n욕실안에 잇는물건을  다 어디로 옮긴후 해야해서((번거롭죠'], ['욕실은 샤워할때 저는 후딱해버려서 그닥 힘든지몰겠던데 빨래가 넘 싫으네요ㅜ'], ['설거지 쌓아두다가  하기싫고\n\n음식물버리기쌓아두다가😂😂\n빨래는건조기잇으니 그나마  조금편함\n욕실청소도타일청소\n'], ['건조기가 있군요 건조기있음 너는귀찮음은 덜고 좋지요ㅜ'], ['저도 빨래개는거요 ㅎㅎㅎㅎ \n'], ['역시 빨래개는거군요ㅜ\n저는빨래감이랑 연관된거 다싫으네요'], ['저는 음식물쓰레기 버리는거요..........................'], ['음식물이 여름돼가니 냄새나고 싫기는하지요ㅜ\n저흰 그나마 신랑출근할때버려주네요'], ['전 다하기 싫어요 너무 많고 끝도없이 무한 반복이지요'], ['하하ㅋ정답이 여기있었네요\n사실 저도 다하기싫어요\n무한반복ㅜ'], ['빨래 돌리는건 좋은데 널기나 개기는 싫어요'], ['저는 빨래 세탁기에 넣고 돌리는거부터 개서 옷장찾아서 넣는거까지 싫으네요ㅜ'], ['저는 설겆이요 진짜  저희가족한끼만먹어도 설겆이양이 진짜 장난아니거든요'], ['설거지는 놔두고 하기는진짜 싫던데 바로바로하면 또하겠더라구요\n맘님네는 오죽많겠어요ㅜ'], ['아이고 바로바로 하려고노력하는데 진짜 요리하면서도 바로 설겆이하는데 먹고난 설겆이는 더하기싫어지드라구요'], ['저도 요리설거지는 바로하는데 먹고난뒤에께 좀 하기싫기는하저라구요'], ['아네에 그렇지요 진짜 맞습니다 너무 힘들어요!배가부르니 더하기싫기도한거같습니다'], ['한가득쌓인거는 더하기싫지요?\n맘님은 오죽하겠습니까ㅜ'], ['저는 빨래하는게 제일 기분 좋던데요ㅎㅎ\n개는것도 티비 보며 슬슬ㅎㅎ 저는 설거지 제일 싫어요ㅎㅎ'], ['이야 맘님 빨래하는게 제일기분좋으신가요?ㅋ\n노하우가있나요ㅋ\n설거지는 그나마할만해요 저는ㅋ'], ['네~저는 빨래통 비어있을때가 제일 행복하더라구요ㅎ\n근데 싱크대에 설거지꺼리 보면 짜증나요ㅎㅎ'], ['맘님 오랜만이에요^^ 빨래개는거 우리엄마가 아주싫어하셨던.ㅎㅎ 그냥전아무생각업이하네요.ㅠ'], ['오랜만이네요맘님ㅎ\n아무생각없이 하는게 정답일듯요\n어차피 평생해야하는거니ㅠ'], ['저는 다림질이요ㅜㅜㅎㅎ 화장실청소도싫고..'], ['우와 다림질도하시는군요\n저는 다리미를 안만져본게 몇년째인지요ㅋ'], ['전 밥이요... 식사준비가 그렇게 하기 싫어요ㅋㅋㅋ'], ['맞아요 밥때되면 밥준비하는게 또 보통일이아니지요ㅜ반찬고민두요'], ['저는 설거지요'], ['저는 설거지는 또 하겠던데 빨래는 개고널고 하는게 왜그리 싫은지요ㅠ'], ['모든 집안일 시작하기요ㅋㅋㅋ 일단 시작만하면 그냥 하는데 시작하기가힘드네요ㅋㅋ'], ['맞아요 시작하는거부터가 싫으네요\n저도 막상시작하면 또하는데 말이죠ㅋ'], ['다 싫은데요 우째요ㅠ 일단 욕실청소 싫고.. 빨래개는것도 싫어용ㅋ'], ['빙고ㅡ저도 실은 다싫어요 청소도 설거지도 빨래도 싫으네요ㅜ'], ['맘님두요?? ㅋㅋ 저 일단 어제 빨래 세번해서 널었어용.. 오늘 잘마르길 바래봅니당ㅋㅋ 그럼 저녁에 개야지요ㅠ'], ['오늘은 잘마를겁니다ㅋ\n오늘해가 아주쨍쨍했잖아요\n저는 지금빨래해요ㅋ'], ['맞지요 빨래 아주 잘말랐더라구요.. 어제 저녁에 집에가자마자 빨래 두번해서 널었는데.. 오늘 날씨더워서 또 잘마를듯해요..'], ['네네 저는 어제저녁에했고 이따 한판 더돌릴 예정입니다ㅋ'], ['계절마다 옷장정리하는거요.ㅠㅠ.진짜시름~~~'], ['아이고 맞아요 계절마다 옷정리하는게 제일싫지요ㅜ저두요'], ['전 집안일 굳이 고르라면 화장실 청손디 바깥양반이  해주고있지요ㅋㅋ'], ['아이고 그라믄 뭐 힘든집안일이 없는셈이네요ㅎ 기침하셨나이까 김치맘님^^'], ['에헴ㅋㅋ쿨럭ㅋㅋ 주방일은 그리 귀찮은거 모르겠구용 아직 육아를안하니  에너지쓸데가 없어서 그런가봅니다ㅋ'], ['ㅋㅋ그런가요?\n진짜 육아하면 빨래도 두배넘지요ㅜ\n설거지도 모든게ㅜ곧경험해보시길요'], ['그러게요 제가생각해도 그렇게하면 저도 막 싫어지지않을까도 싶네용ㅋㅋ'], ['일단은 지금은 그런거아니니 즐겁게 룰루랄라 즐기며 하셔야지요ㅋㅋ'], ['저는 개는거까지는 괜찮은데\n서랍에 찾아 챙겨넣는거 좀 귀찮아요'], ['개는것도 서랍에 챙겨넣는것도 싫으네요\n서랍도 다 제각각 찾아넣는거ㅜ'], ['저는 빨래 개는거 너무 싫어요'], ['맘님도 빨래개는거 싫으시군요\n진짜 세탁기조작하는것도 싫네요ㅋ'], ['저두 너무 싫어요ㅋㅋㅋ 그냥 집안일은 다싫은듯요ㅜㅜ 지금 친정와서 집안일 아에안하니 천국이 따로 없네요ㅋㅋㅋ'], ['아고 그치요 암껏두 하고싶지않지요\n친정계시니 그나마 편하지요ㅋ'], ['저는 빨래 개는거랑 설겆이요  ㅎㅎ 근데 울 신랑도 그걸 시러라하네용 ㅋ'], ['뭔가 울집이랑 씽크로율 백퍼인듯요 \n울신랑도 그 두가지는 진짜싫어해요ㅡㅡ'], ['그쳐 조금은 달라야지  서로가 편할텐데 말이지요 세탁기 돌리고 너는건 엄청 잘해요'], ['아하 세탁기돌리고 너는건 좋아하시니 그나마 나은거네요\n울신랑은 빨래와관련된건 다싫어하지요ㅎ'], ['저는 빨래 산더미 쌓아놓고 골라입은 적 있어요ㅋ 무지 귀찮은 일 중 하나라죠ㅋ'], ['빨래를 안개고 그냥 더미로 쌓아놓고 골라입으셨다는거지요?ㅋㅋ\n빨래 어차피 펼건데 안개는것도 좋은방법이네요ㅋ'], ['네ㅋ 안개키고 산더미 만들어 놓은거 골라 입었어요ㅋㅋ 남편도 안개켜놓길래 이틀 방치한적 있습니당ㅋ'], ['하하ㅋ괜찮기는하겠어요ㅋ\n입을건데 말이죠\n대신 구깃구깃하긴했겠지요?'], ['구깃 구깃한건 덤 입니당ㅋ 널부러 놓는거 시른데ㅋ 아주 가끔 사용해먹는 방법이긴 해용ㅋㅋ'], ['보고 빨래걷다가 조만간입을 옷들은 저도 그방법 써봐야겠는데요?ㅋ'], ['저도 빨래가 싫으네요. 특히나 개는거요. 너는건 그나마 하겠는데 말이죠.'], ['맘님도 빨래가 싫지요?\n저는 빨래감 세탁기에 넣는거부터가 싫으네요ㅜ'], ['사실 오늘도 빨래를 해야하는데 날씨가 흐리기도하고 귀찮아서 패스네요.'], ['잘하셨어요\n저는 그나마 어제 다해서 괜찮아요\n내일 또 할게있긴하지만요ㅎ'], ['둘째 낳고 나서는 빨랫감이 더 늘어서 자주 돌리게 되는거 같아요.'], ['맞아요 거의 매일하지않나요?\n저희도 거의 이틀에한번은돌리네요\n어쩔땐 매일이구요'], ['저도 걸레질이요ㅋㅋ 걸레빠는거 특히 더 싫어요ㅋ 빨래 너는것도 싫었는데 건조기가 살려주네용'], ['걸레질ㅠ 맞아요 걸레질 ㅠ빠는것도 싫구요ㅜ\n저는 그래서 물티슈자주써요ㅋ\n'], ['그치요? 근데 물티슈는 닦고나면 먼지 잘생긴다고 신랑이 투덜투덜ㅋㅋ 그래도 전 물티슈쓰지요ㅋㅋ'], ['맞아요ㅋ물티슈가 마르고나면 먼지가 잘생겨요\n물걸레질이 제일 낫지요ㅋ'], ['네네ㅜ 그래서 요즘은 정전기포로 이용하구있어요ㅜ 괜찮긴하나 물걸레포도 사야겠어요ㅋ'], ['저도요 ㅠ 빨래 ㅡㅡ\n널고 개고 넣코 ㅠ\n제일 싫어요 ㅠ\n 집안일은  다 싫지만  그중에서  빨래용  ㅋ'], ['저랑 찌뽕이네요ㅋ\n진짜 세탁기넣는거부터해서 옷장에 넣는거까지 귀찮고싫으네요'], ['뽕찌용~ ㅎㅎ \n진짜 세탁하는게 일이예요~일~\n완전 싫어요~ 시러~'], ['옷분류해서 세탁기집어넣는것도 일이더라구요ㅜ손빨래하는것도 글코 싫으다요ㅜ']]
    
    547
    가장 미루게 되는 집안일은????ㅠ ㅠ 저는 설거지가 제일 하기 싫은가봐요 ㅠ ㅠ집안일 다해도 늘 설거지는 마지막에 남네요미루고 미루게 되요 ㅠ ㅠ ㅠ눈이 하는게 아니라, 손이 하는건데보고는 하기싫어가지구,,,,왜그럴까요 에휴ㅋㅋㅋ  막상하면 금방하는데 고무장갑끼기 까지가 먼길이네요맘님들제일 미뤄지는 집안일은 뭔가요???
    
    [['전 빨래개기요......너무 귀찮아요;;;;'], ['찌찌뽕~나도 빨래개는게 젤 귀찮아'], ['맞아요 빨래개기도 맘잡고 자리잡음 금방인데 참귀찮죠 ㅜ'], ['세상 귀찮다....얼굴 잊어먹긋엉~~'], ['글게ㅜㅜ\n주안이 언제 얼집가노? \n얼굴좀보자ㅜㅜ'], ['8월에~그때가 2학기라카더라고'], ['그전이라도 보자ㅜㅜ\n비가 안와야 보기도 좋을텐데ㅜㅜ'], ['저도 설거지랑 빨래가 미뤼지네요~^^'], ['다들 비슷하군요 ㅠ ㅠ'], ['저도 빨래개는거요~설거지는 미루면 하기싫어질거같아서 밥먹자마자 바로 해요 \n그게 더 마음이 편하고 좋더라구요 뭔가 개운한느낌이고ㅋㅋ'], ['바로 바로 하는게 최곤데 늘 산처럼 쌓여요 ㅋㅋㅋㅋ점점더 하기싫어짐의 반복이네요'], ['설거지까지 바로 해야 밥 먹은 느낌.... 설거지 남겨두면 응가하고 뒤 안닦고 나온거 같은 느낌적인 느낌'], ['반성합니다 ..배워야겠어요ㅎㅎㅎ'], ['맞다 완전 적절한 비유다ㅋㅋ'], ['저도 빨래개기요,.ㅋㅋ개기싫어서 거기에있는옷찾아입어요ㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋ 저도요 거기서 하나하나 빼쓰다보면  건조대 뼈다귀만 남죠..그러고 보니 전 다 하기싫나봐요 ㅋㅋ'], ['걍 집안일 자체가 다 구찮아요 ㅠㅠ'], ['완전 공감합니다 ㅠ ㅠ ㅠ끝도없어요'], ['집안일은 적성에도 체질에도 안맞아요\n땔치아고 싶음요'], ['아이구 완전 공감합니다!!!!'], ['빨래 개는거랑 서랍에 찾아 넣는거요ㅜㅜ\n지금도 안갠 빨래랑 개어서 쌓아놓은거랑 뒤섞여있어요..'], ['다들 비슷하게 살아가는군요 ㅜ \n혼자 하기싫어서 푸념하다가 왠지 의지가 되네요 ㅋㅋㅋ'], ['전 빨래 정리요 하긴 하는데 제일 하기 싫어요~'], ['맞아요 장마철이라 꿉꿉하니 \n제습기써도 엉망이네요 ㅠ'], ['전걸레질ㅠ일단 눈앞에 걸리는거 밟히는것만없음 되는성격이라ㅋㅋㅋ'], ['다들 그렇죠 ㅎㅎㅎ 전 비오니까 그래서  눅눅하니까 핑계로 미루다가 오늘 겨우 한번 닦았네요^^;'], ['빨래개는거요'], ['맞아요 ㅠ차라리 너는게 낫구요'], ['빨래개는건그나마하겠는데.. 빨래다갠거가져다놓는게젤싫어요ㅜㅜ'], ['맞아요 ㅠ저도 겨우 개서 쌓아놨네요,,,'], ['걸레빨기가 젤하기싫으네요~~ㅋ'], ['ㅎㅎ 공감합니다 ㅠ ㅠ'], ['저도 빨래요ㅠ \n하는것도 너는것도 개는것도 정리하는것도 다 싫어요ㅎ'], ['빨래가 압도적이네요 ㅎㅎ'], ['ㅋㅋ 저는 방 하나에 수북하네요 ㅠ ㅠ'], ['밑반찬 만들기가 젤 싫어요\n빨래 개는건 잼나던데\n찾아 넣기가 싫어요'], ['저희집은 애써 만들어도 버리는게 반이에요 ㅠ'], ['저는 빨래게기가 제일로 시르네요 ㅠ 몇일갑니다. ㅎㅎㅎ'], ['ㅎㅎ저도 그대로 두다가 다 걷어쓸때쯤 치우기도 해요 하핫'], ['ㅋㅋㅋ빨래요 ㅋㅋㅋ'], ['저는 청소기돌리는게 젤 시러요ㅠㅎㅎ'], ['빨래개기여ㅠㅠ'], ['다 개어진 빨래 서랍장 찾아 넣는거요. ㅋㅋ\n다들 비슷하네요'], ['다 귀찮아요'], ['창문틀닦기\n블라인드 닦기'], ['창문틀 블라인드....아직 한번도 못해봤네요 ㅠ ㅠ ㅠ'], ['저두 다 귀찮아요ㅋㅋ'], ['화장실 청소요ㅠㅠ'], ['맞네요 ㅜ ㅜ 생각해보면 끝도 없어요'], ['윗글들 보니까...ㅜㅜ 다~~네요..망구귀찮아요..'], ['저두 설거지 빨래개기요\n어차피 나의일이지만..   정말 하기 싫어용ㅜㅜ']]
    
    656
    열심히 집안일 했네요 @.@ 눈뜨고 아기 케어+청소+빨래..하아..쉼없이 일하고 밥생각 없어서짜장묜 시켰어요ㅜㅅㅜㅋㅋㅋ점심 먹곤 또 이유식도 해야하구..욕실청소도 해야하구..에어컨 필터도세척해야 하구..하아..-_-;;할게 왜이렇게 많죠.? ㅋㅋㅋ;;오늘만 이런게 아니라는거어~~거의 매일..주부는 쉬는 날도 없구..흑..
    
    [['ㅠㅠ 힘내셔요'], ['쫌 쉬고 있어여ㅠ 이유식을 시작으로 다시 스타뜨 해야 하지만요ㅠ'], ['고생이 많으시네요 ㅠㅠ 저도 이유식 해야되는데 초기라서 쉬워서 걍 이따가 하려구요 ㅋㅋ'], ['저두 큐브 만들어둔걸로ㅋ 육수 다 녹지도 않았는대ㅋ 걍 때려넣고 죽모드 돌렸네요ㅋ'], ['ㅋㅋ이미 만들어 넣은 재료 있어서 편하게 하셨네요 ㅋㅋ'], ['이래숴 미리미리 쟁여두면 좋은것 같아요^^'], ['그러게요 ㅋㅋ 정말 부지런 하셔요 ㅋㅋ'], ['할때 좀더 해두는건대요~ 다 나편하자고 ㅋ'], ['히히 그건 그렇죠'], ['주부는 쉬는 날이 없지요 ㅠ 힘내셔요'], ['그러니까요..이래서 가끔은 어디론가 훌쩍 떠나고 싶어요@.@'], ['에궁 주말이라도 잠깐 쉬묜좋을텐데요'], ['평일.주말 개념이 없지유ㅜㅅㅜ'], ['주부라 그렇긴하지요 늘똑같은..'], ['맞아영..이따금 다시 사회로 돌아가고 싶을때도 많아요..집에서 육아에 살림에..보통일 아니거든요.\n 뭐 일해도 육아+살림 손놓을순 없는거지만..밖에서 내일하고 싶은 욕심이 자꾸 불쑥불쑥 튀어나오네용ㅠ'], ['아가 키우고 시작하셔요 전그럴려구요 ㅎㅎ'], ['그게 또 주위에 보면 글케 쉽진 않은것 같더라구요..아무래도 경단녀라ㅠ'], ['에궁 ㅠ 나이도 있어서 그럴지도요'], ['글쵸..전문직 아닌이상..흐흐..'], ['저도 전문직이라 희망을 가져봐야겟네요 ㅎ'], ['저도 전문직이긴 한대..지금 복직하면 할수있을것 같은대 2~3년후에 할거 생각하면 좀 두렵긴해여ㅠ'], ['저도 나이땜네 좀 글킨하지만요 ㅠ'], ['저두 나이도 글쿠..일은 왜 쉬었다 하려면 부담스런면이 있잖아요 예전처럼 잘할수있을까 싶고..또 애기 하원시간에 마춰야 하기도 하구여..'], ['그건그러겟네요 일다시 적응하는게 쉽지않죠ㅜ 하지만 외벌이는 더힘들든요ㅠ'], ['신랑은 뭐 말은 집에서 내조 해주는게 좋다구 하는대..진심일까요.?ㅡㅡ'], ['진짜요? 그럼 진심이니 그케 말한느거겟죠'], ['글쎄요오..말뿐일수도.? ㅎㅎ'], ['전 말이라도 그케 안해줘요ㅎㅎㅎ'], ['오매..매정ㅠㅠ뭐 현실적인 걸수도 ㅠ'], ['외벌이가 힘들긴하쥬ㅜ'], ['글쳐 맞벌이도 힘든대..'], ['아가 크면 더 돈이 많이들거같긴해요'], ['아무래도..다른것보나 사교육비 무시못하져..남들 다 보내는대 안보낼수도 없구요..'], ['기본적으로 들어가는것도 많으니 같이벌어야될듯요'], ['기본적으로 들어가는건 제가 아끼고 하면 그럭저럭 유지되더라구여.'], ['저도 쓰고 싶어요 ㅠㅠ 제 인생도 한번뿐인걸요 ㅎㅎㅎ'], ['글킨하져..에휴..내인생 챙기기 왜케 걸리는게 많나요ㅠ'], ['에잇 걸리적ㅎㅎㅎ 틈틈히 저도 챙겨야죠 ㅎㅎ'], ['맘처럼 쉽지 않아서 ㅠㅠ'], ['그래도 챙기셔야 남들도 알고 챙기는 거여요~~~'], ['그라죠\n오늘만이 아니라지요\n토닥토닥 힘내요^^'], ['늘 오늘같은 내일인것 같아요 맘님도 홧팅^^'], ['그쵸\n조금만 크면 님도 또\n자유시간 생긴다지요\n저20개월 되서 얼집보내니 좀 편해졌어요'], ['얼집 보내는 그날..자유가 오는군요.? ㅋ'], ['네 ㅋ 한달 적응기간이 좀 힘들었는데 \n지나고 ㅋ ㅋ 잘적응 하면 또 편하더라고요'], ['글쿤요..근대 또 얼집 보낼생각하니..하아..뭔가 떨어뜨려 놓는것 같은 ㅠ'], ['마져요 첨엔 그런 기분이 들다가도 또 잘적응하고 아이들과 상호 작용하는거 보면 또 괜찮아 지는거 같아요~'], ['안그래도 얼집 보내는 선배맘님들 얘기들어보면 처음 떨어질때 힘들지 엄마도 여유시간 있어좋고 아가도 사회성도 기르고 엄마가 못채워준 교육.놀이 할수있어 좋다더라구영.'], ['오늘 계속 집안일 이시네요ㅜㅜ'], ['할게 평소보다 많긴 하네요~ 담주엔 생일도 있구 휴가도 가야하거든요ㅠ'], ['악~~저 오는주 평일 생일요\n비슷하시네요'], ['헛헛ㅋ 전 신랑이랑 생일같아요ㅋㅋ 현충일ㅋ'], ['저 그날인데~~\n전 음력이라 이번에 딱'], ['헉!! ㅋ 저흰 양력요ㅎㅎ 음력으로 하면 윤달이라ㅠ 늘 생일이 바뀐다능ㅎ'], ['넹 \n저희집은 다 음력이고\n신랑집은 다 양력'], ['저흰 신랑.저.글구신랑 누나까지 ..생일이 같아요^^;;'], ['오메!!\n근데 그럼 은근 안좋잖아요ㅜㅜ'], ['반반인듯요 ㅋㅋ 한번에 해서 좋긴한대 우째..내생일같지 않은ㅋㅋㅋ'], ['그러깐요\n자기생일인데 왠지~~그렇죠\n이날은 주방쉬어야죠'], ['거의 외식하긴 하는대..육아는 ㅠ'], ['말끔해졋겟어용!!화이팅!'], ['집안일은 해도해도 끝이 없네요@.@'], ['저는 오늘 빨래만 하면 되요 ㅎㅎ밥이랑..'], ['집안일 안하고 하루하룰 보낼순 없겠쥬.?ㅠ'], ['일주일에 한번에 몰아서 해볼까요 ㅎ'], ['것두 아기 없을때나 가능하쥬ㅠ'], ['ㅠㅠ집안일도 참 힘들어요ㅠㅠ'], ['맞아요..늘 하는대도 적응안되는ㅡㅡ'], ['ㅠㅠ아기나오면 더 힘들예정이에요 저 ㅠ'], ['에구..그래도 아가키움서 즐거운일이 더 많아요^^'], ['ㅎㅎ보기만 해도 즐겁겟죠 ㅎㅎ하루하루 커가는모습에 ㄹ'], ['신기하기도 하고 뿌듯하기두 하고^^'], ['ㅎㅎ신기할거같아요 ㅎㅎ젤리곰이엇던아이가 ㅎ'], ['그니까요..젤리곰이 뭐에요..그냥 점하나였는대 ㅎㅎ'], ['귀엽게 표현하려구 젤리곰 ㅠㅠ'], ['아녀 젤리곰 시절보다 그전에 점하나 였었자나요ㅎ'], ['넹 ㅎㅎ완전 보이지도안을만큼 작앗죠 ㄹ'], ['근대 자라는거보면 얼마나 신기한지여~']]
    
    689
    집안일 중에서~ 젤하기싫은건 빨래 개기 아닙니다ㅠㅠ갠빨래 제자리에 넣는게 세상 귀찮네요..
    
    [['수건 저건 우찌 말으셨는지 ㅋㅋ이쁘게도 말으셨네요 ^^ 저도 넣는게 싫네요'], ['수건 저렇게 말아야 장에 예쁘게들어가서 말이지용ㅎㅎ\n넣는거 무지 귀찮아요'], ['전 대충 삼등분으로 접어 넣거든요 ㅋㅋ귀찮아요 근데 저래 해두니 깔끔하네요ㅡ'], ['그러시군용ㅎ 삼등분 제일 많이들 그렇게하시지용ㅎ 그것두괜찮아요'], ['맞아오맞아ㅋㅋㅋ 개는건 할만한데 제자리두는게 진심 귀찮아여ㅜㅜ 저만그런게아녔군요ㅎㅎ'], ['그렇지요?ㅎ 개는건 티비보며 슬렁슬렁 개면 기분좋은데..넣는거 지지에요'], ['그런가요ㅎㅎ 건조기 있으시군요\n진짜 신세곈데 저희집엔 아직 없답니다ㅜㅜ'], ['수건진짜이쁘게마셨어요 저렇게마실려면 갤때힘들겠어요ㅜㅜ;;'], ['생각보다 힘들지않고 간단해용ㅎㅎ\n돌돌마는재미도있지요'], ['넘 깔끔하게 개셨어요~ 전 대충 하는데도 귀찮네요ㅠ'], ['개는건 재미지지요 티비도보면서 하면요\n근디 챙겨넣는건 대박 귀찮습니다'], ['폭풍공감이네요\n저도 빨래 널고 개는것까지는 괜찮은데정리가젤 싫어요ㅜㅜ'], ['하핫 맘님도 공감하시는군요\n진짜 정리해서 지자리넣는거 귀찮아요'], ['저는 빨래랑 관련된건 다싫어요ㅋ\n찾아서 넣는건 이방저방 여기저기 수납도 제각각'], ['맘님은 다싫으시군요ㅎㅎ\n워미 진짜 넣는거 이짝저짝 옷걸이도걸고 귀찮아요'], ['맞아요 이쪽저쪽 수납칸도 각기다르니ㅜ그래서 신랑이 빨래가 제일싫다고 하나봐요'], ['그러니까용 ㅎㅎ 앗 신랑님이 해주시는가보네요 \n아이콩 재밌네요\n참 좋으신분이십니다'], ['아뇨ㅋ신랑이 제일하기싫은건 손 안댑니다ㅋ빨래안해줘요 설거지도 싫어해요ㅋ'], ['아..그렇군요 역시 쉐프님은 요리만 좋아하시는군요 막 저지레함시롱ㅎㅎ'], ['저도 그래요\n개놓고는 신랑보고 갖다넣으라고 그래요ㅎ'], ['ㅋㅋ 그거좋은 방법인데요\n전 오늘 바깥양반  타이밍을 못맞춰버렸네요'], ['이야....진짜.정리잘하시는데요....맞아요.저도 빨래개어놓고는 그냥 하루둔적도 많아요..ㅎㅎㅎ'], ['그런가용ㅎㅎ 하루두는것도 괜찮네요\n그럼 바깥양반이  치울텐데요ㅎㅎ'], ['어머 치워주시면 감사하죠...ㅎㅎㅎ좋으시겠어용.ㅎㅎ'], ['네 치워줍니당ㅎㅎ 그나마다행이지요\n오늘은 타이밍 못맞춰서 정리했네요'], ['ㅎㅎ저도 그거 공감요~~요즘은 빨래 개 놓으면 해영이가 자꾸 흐트려 놔서 욱욱~하네요ㅎㅎ'], ['그렇지용ㅎ ㅎ 해영이넘귀욤ㅎㅎ\n저희김치도 빨래갤때 올라탄다지요'], ['저두요 이방 저방 이서랍 저서랍~~~  ㅠㅠ'], ['그니까용 이쪽저쪽  넣는데도 틀리구 왔다갔다 너무 귀찮은것같아요'], ['진짜공감이요 ㅋㅋ'], ['공감되시지용ㅎㅎ 개는건 괜찮은대 진짜 넣는게 완전 귀차니즘옵니다'], ['귀찮아요'], ['그렇지요..왜이리 귀찮은건지요ㅎㅎ\n빨래 개는건 재미진데말이지요'], ['저두요.ㅋ 수건 개신거 배우고 싶네요. 밖인데 노는것도 힘드네요.^^;'], ['수건 요고 의외로 쉽습니당 ㅎㅎ\n노는게 힘들지요 그래도 신나게 놀다오셔요'], ['요거 개는거 우째 글로는 배우기 힘들겠지요? 보통 솜씨가 아니신듯 합니다.'], ['넹 글로는 쪼메 힘든듯합니다 설명이 애매하지요 아..네이놈에 호텔수건개기 검색해보셔요'], ['아하ㅡ네이놈이 있었네요.요게 호텔수건개기인가봅니다.'], ['네넹 ㅎㅎ 한번 검색해보셔용 \n그럼 나올거에요 돌돌말기 수월하답니다'], ['빨래 이뿌게도 개셨네요..저도 넣는게 제일 싫어요.'], ['그렇지용? 저도 이것들 각자 자리에 정리해서 넣는게 너무 귀찮답니다'], ['아이고 진짜 야무지게 빨래를 개시는거같습니다특히 수건 너무 돌돌이뿌게도 개신거같습니다'], ['히힛 감사합니당 저희수납장에 요게 딱이거든요\n그래서 항시 요래 갭니다'], ['아이고 네에 진짜 알맞춤 수건개기네요~요래개면 시간도 더걸릴것같은 예감이듭니다'], ['네맞아용 시간은 조금더 걸리지만 수납해놓음 예쁘고 하나씩빼도 흐트러지지않아서 좋답니당'], ['아네에 장점이 더많을것같습니다 흐트러지지않는게 더좋은거같습니다'], ['그러게용ㅎ 하나씩 뽑아쓰면되니 좋아요  갤때번거롭지만 굿입니당'], ['ㅋ이뿌게도개셨네요~ 집안일중젤하기시른게저것이군요ㅋ전너무많아용ㅋ'], ['젤 귀찮은것 같아요ㅎㅎ 다른청소는 사실 바깥양반이 많이해서 제가 싫을게 없답니다'], ['ㅎㅎ완젼착한신랑 머든다해주니 이쁠만한데용~ 집안일말끔해지면 저는그리좋더라구용.ㅋ'], ['맞아요 제가 주방일하고있음 가만히 안앉아있어요청소하거나  가스렌지닦아줘요'], ['아이고 신랑잘구하셧는데용?? 어디서구하셧는지.ㅋㅋ잘해주시니 알아서 더 잘하시는듯요~ㅎ'], ['그르게요 근데 어디서나타난것보단 왜이리 늦게 나타났는지ㅜ 조금만더일찍오지ㅎㅎ'], ['호텔식 수건 개기인가요~ 다른것들도 이쁘게도 개셨네요~ ㅋ'], ['히힛 수건요래 개서 장에 딱넣으면 기분이 너무좋답니다ㅎㅎ'], ['전 시도해볼려다가 구차나서  패쓰 수건 담당은 신랑이라서ㅎ'], ['아공 그러셨군요ㅎ 수건은 또 신랑님이 정리해주시니 고맙네요\n집안일은담 좋아용'], ['맞벌이하니 같이 해야지요ㅎ 안 도와줬음 때리쳤지싶어요ㅎ\n잘 도와주니 다행이네요^^'], ['맞아요 잘도와주실 것 같아요 무지 자상할것같구요ㅎㅎ 굿입니당'], ['제자리에 넣는게 귀찮은가요ㅋㅋ\n저는 가져다 놓는 행동을 다섯번 이상 해야합니다ㅜ 미니가 다 흩어놔요ㅜ'], ['넹 귀찮아요ㅎ 왔다갔다 해야해서요\n미니 어쩔..웃픈데 귀엽네요ㅋㅋ'], ['빨래 개키면 맨날 허벅지나 발에 끼워놓고 갭니다ㅋ 안그럼 도로아미타불 되는건 시간문제에요ㅋ'], ['아하하ㅋㅋ 상상이갑니다 \n어떤자세인지 말이지요\n미니 너무귀여워요ㅋㅋ'], ['상상가시지요?ㅋ 엄마 도와준다고 어찌나 흩뜨려놓고 댕기는지 말이에요ㅋ'], ['그러게용 미니가 갠것이 맘에 안들었나봅니다 엄마 일 빡시게 시키네요\n'], ['넘나정갈하게 이쁘게잘 개셨어요 저도 빨래널고개는게 넘나귀차나요'], ['감사합니다ㅎ 너는것도 귀찮은거 맞아요\n허나 제자리 넣는건 더욱 귀찮지요'], ['수건 넘 잘 접으셨는데요 ㅎㅎ 전 귀찮아서 저거 알아도 그냥 접..ㅋ 그나저나 김치맘님 속옷은 숨어있나요? ㅎㅎ'], ['아 가슴가리개요?ㅎ 부끄러버서 숭갔다지요ㅋㅋ\n전 정리하는게  그리 귀찮아요'], ['아.. 그건 어찌 정리하시나 싶어 물어봤지요ㅋ 그거 정리하는거 제일 귀찮은데 최근 새로운 방법을 터득해서 그리 정리해야겠어요 ㅎㅎ'], ['그렇군요ㅎㅎ 전 반접어서 끈 돌돌말아 정리해용 이건 어찌 할방법이..좋은방법 있음 공유해용'], ['그게 제일 좋은 방법 같아요 ㅋ 저도 비슷하게 하고 있어요ㅎ'], ['그렇군요 ㅎㅎ 요래 해서 두면 찾기는 쉽더라구요 공간도 크게 안차지 하구용'], ['수건 저도 저렇게 말아서 개켜요 ㅎㅎ 처음엔 좀 번거로웠는데 이젠 이게 익숙해서 금방 후다닥 끝나요 ㅋㅋ'], ['수건찜뽕이네요ㅎㅎ 요렇게 개면 진열도 이쁘구 흐트러지지않아서 좋지용'], ['전 빨래너는거, 개는거, 넣는거 다 싫네요. 일단 건조기 있어서 너는거 없었음 좋겠고.. 다음에 개는거, 정리하는 로봇필요하네요ㅋㅋ 지금은 신랑시키지만요ㅋㅋ'], ['다싫으시군요ㅎㅎ 로봇까지 ㄷㄷ\n그래두 지금은 신랑이 해주시니 좋으네요'], ['네.. 제가 널은거 걷어서 침대위에 던져놓으념 주은이 육퇴하는 동안이나 언제 알아서 척척 개놓더라고요. 다만 제자리에는 제 몫 ㅠㅠ'], ['그렇군요ㅎㅎ 그래도 그만큼 도와주시는게 어디에요\n맘님 신랑님도 좋으신것 같아요'], ['그런가요~ 하긴 안좋진 않아요. 자기가 알아서 밥도 척척하고..이렇게 두면 말안해도 하고 다행이지요.'], ['완전 좋으신데요 ㅎㅎ밥도 해주시니 말이지요 \n아에 손놓고 안도와주시는 분들도 많이 계시다 하더라구요\n'], ['전 다 귀찮네요 속옷이 다 너무 이쁜거 아닙니까 색도 곱네요'], ['다귀찮..그러시군요ㅎㅎ\n색이고운가유ㅋㅋ\n아우  오늘 날좋은데 또 빨래한판해야겠어요']]
    
    840
    집안일 분담 오늘 낮에 영화보러 나가는 것 말고는 할일이 없어서늦잠자고 일어나서 아점으로 해물수제비 해먹었어요~남편은 밥투정 없이 해주면 잘 먹어서 좋아요~다른건 불만이 없는데집안일 분담이 생각보다 쉽지 않네요~제가 자영업자라 출근이 조금 늦기때문에아침에 빨래 청소는 제가 하고아침밥도 저만 먹기 때문에 밥이랑 설거지는 제가 다 해요~그래서 남편한테는 화장실 청소랑 최소 일주일에 한번 집안 걸레질만 해달라고 했는데제 성에 차지 않네요 ㅋㅋㅋㅋ전 샤워하고 바로 화징실 물기를 싹 닦아서 화장실을 말리면 좋겠는데남편은 몇시간 방치후 자기 전에 화장실 정리를 해요~안그래도 화장실에 창문도 없고 요즘 습도가 높아서 잘 마르지도 않는데본인은 집에와서 일하는 느낌이 싫다고 느긋하게 하고 싶다고 하네요~어떻게 하면 바로 할게 만들수 있을지 모르겠어요~별거 아닌거 같은데 이런게 성격차이인가봐요 ㅋㅋㅋㅋㅋ
    
    [['서로 다른 환경에서 살아왔으니 당연한것 같아요 ㅠㅠ 화장시 청소 말구.. 음.. 다른걸 맡길 수 있는 건 없을까요??'], ['울 예랑이는 다른건 엄청 느긋한데 밥먹고 치우는건 넘나 바로바로 여서ㅜ 솔직히 요리하고 조리도구 치우고 하다가 밥먹고 이제 한숨 돌리는데 바로 치우면 막 왠지 숨차는 느낌이에요ㅜㅜ집안일 스타일도 맞아야 되나봐요'], ['진짜 여태 따로 살아와서 삶의 방식을 맞춘다는게 쉽지않아보여요ㅠㅠ'], ['집안일 문제부터 여러가지가 자꾸 부딪히더라구요. 진짜 연애할 때와는 다른거 샅아요ㅜ 잘 맞춰가며 살아야겠죠'], ['이런부분이 적응하고 맞춰가야하는 부분인가봐요 남편분이랑 이야기를 한번 나눠보시는게 좋을거같아요'], ['정리 하면 다행인거 같아요 ㅡㅡㅋ 같이 안하는 타입이라.... 전 세탁기 식기세척기 로봇청소기 스타일러 등등 템빨로 ... 덜 싸워 보랴구요'], ['계속 다른 환경에서 살아왔으니까 다른건 인정해야죠ㅠㅠ 한쪽으로 맞추려고 하다보면 결국 다른 한쪽이 지치더라구요ㅠ 그래서 전 결혼 선배가 이 말을 해 주더라구용 서로 다름을 인정하라구ㅠ 그래도 하는게 어딘가요~'], ['지금까지 다르게 살아와서 한번에 고치기가 쉽지 않은 것 같아요 그래서 신혼 초에 많이 싸운다는 것 같고ㅠ 그냥 묵묵히 일단은 지켜보셔요ㅠ'], ['맞추는게 정말 쉽지 않더라고요ㅠㅠ 저희도 화장실 슬리퍼 두는 방향가지고도 다투게 되더라고요 언젠가는 맞춰지겠죠'], ['평생 맘에안차실듯.. ㅠㅠ 성격차이에요 ㅜ저도그래여.. ㅠ'], ['다른 환경과 다른 생활 습관속에서 살아와 둘이 하나가 되어가는 길이네요 ^^'], ['저는 씅에안차서 제가 하는스타일입니다 ㅠㅠ저만 고생하고 예랑이는 편하지요..ㅠ 어떻게할수잇는방법이 없는거같아요'], ['그래도 정리는 해주시니 다행인 것 같기두한데, 몇십년 다르게 살아왔으니 서로 맞춰가는데 시간 걸릴 것 같아요 ㅠㅠ'], ['살아온 삶의 방식이 달라서 그럴거애요ㅎㅎ 살면서 하나씩 맞춰가면될거애요'], ['그런 사소한것 때문에 많이 싸운다는거죠 ㅜㅜ 내가 하던지 포기해야해요..'], ['정말그런건..저는진작포기햇어요..그냥씻고나오면서 여기저기거품튀거나머리카락묻은것만 샤워기로쓸어내려달라고만부탁한답니다..ㅎ 이해와배려가아니라..ㅎ어느정도포기해야 그게 배려가되는것같아여ㅜㅎㅎ'], ['그러게요... 어쩌겠어요 그리 살아온것을.... 잘할수있는 집안일로 분담하는것은 어떨까요~~;;'], ['진짜 성에안차요ㅋㅋ\n근데 자꾸말을해주던가..\n아님 귀찮더라도 신부님이 하시는방법밖엔..ㅠㅠ'], ['각자살아온삶이달라서 어쩔수없는부분인거 같기도해요ㅠㅠ'], ['바로 하게 만들수는 없어용 강요니깐요 ㅠ어쩔수 없네요\n시간을 두고 정해놓고 기다리셔야할듯요~\n저희신랑두 느긋하게 청소하고그러는편이라 ㅋ 답답하면 제가 하고요 아님 더러운채로 납둬요^^\n둘중 내가 선택하는거죠 ㅎㅎ\n자꾸부딪혀봤자 좋을게 없어요\n\n반대로 신랑분이 반찬 투정이나 입맛 까다롭다고 생각하시면 끔찍하지 않을까요 ㅠㅠ \n좋은것만 보시구 조율해보세요^^'], ['그런거 두분이 잘 조율하셔야해요 ㅜㅜ 싸움꺼리가 되서ㅠㅠ'], ['너무 오래 다르게 살아와서 그런점은 오쩔수 없나봐요 ㅠㅠ'], ['살아온 방식이 달라서 쉽지 않은 것 같아여!그치만 초반에 서로 노력하지 않으면 더 힘들 것 같아요!'], ['다같은 이야기이지만 생활하던 습관이 다르다보니 서로 맞춰가기 위한 시간이 필요한 것 같아요~~ 포기해야한다고들 하지만 그래도 집안일은 같이 하는거지 돕는게 아니니 최대한 기분 상하지 않게 잘 이야기 하는 것도 하나의 방법인것 같아요 ㅎ'], ['ㄷㄷ경험해보기 전이지만 벌써부터 ㅋㅋ걱정되네요 잘 참고 예쁘게 말하면서 잘맞춰가야될텐데..ㅋ노오력'], ['각자 살아온 방식이 달라서 어쩔 수 없는 것 같아요. 그냥 맡긴 거면 성에 안 차도 냅두시는 게 나을 것 같아요.'], ['서로 스타일도 다르고 맞춰가는게 시간도 걸리고 많이싸운데요ㅜ 저도 신랑이 설거지도 해주고 청소도 해주는데 맘에 들지않아도 뭐라안해요 안 그럼 안해준다고 다들 그러더라구요ㅜ'], ['저도 좀 걱정이예요.  혼자 살땐 그냥 내버려두고 한꺼번에 정리하는 타입인데 ㅜㅜ'], ['안하시는것도아니고성격차이다보니 조금씩양보해야겠죠ㅜ'], ['화장실 물기 어떻게 말리시나요? 전 자연 건조 시켜서요 ㅎ'], ['맞아요 서로 다른환경에서살아왔으니 서로 맞춰야할 부분이 많더라구요'], ['아니면 다른걸 맡기는건 어떨까용 근데 남자들 하는거 어느정도는 그냥 넘어가야 한다더라구요ㅠㅠ 아니면 아예 안한다고... ㅠㅠ'], ['집안일은 결국 더러운거 못참는 사람이 한다던데..ㅠ 얘기잘하셔서 반반씩 분담하세요~ 결혼생활은 같이하는거니까요'], ['기왕 분담했으니 상대가 하는대로 이해해줘야할 것 같아요 ㅠㅠ 성향차이라 어쩔수 없는것 같아요 ㅠㅠ'], ['더 깔끔한 성격 가진 분이 어쩔수없이 조금 이해해야 하더라구요ㅠㅠ 저도 신랑이 화장실청소 해주는데 처음엔 구석구석 지저분한 곳이 많이 보였는데 먼저 나서서 열심히 청소하는 것만으로도 감사히 여기기로 했어요!! 성격 차이이다보니 조금씩 극복할 수 밖에 없는것 같아요~'], ['그쳐 원래 사소한것부터 부딪친다고 하더라구용ㅜ 그래도 서로 잘하는거 하는게 좋지요 머'], ['성격차이예요 ㅠㅠㅠㅠ 집안일때문에 몇번싸웠어요ㅠㅠㅠㅠㅠㅠㅠ 집안일 넘 힘드네요'], ['전 지금도 맘에 안드는데 아예 맡기고 신경끄거나 아님 내가 직접하거나 하는게 맘편한거 같아요ㅜㅜ'], ['진짜 이런부분 오떻게 할지 감이안오네여 ㅜㅜ'], ['이건 성격차에요.ㅠㅠ 답이 없는..ㅠㅠ'], ['아...저도 이 집안일문제.. 예랑이랑 진지하게 논의해봐야겠어요;;;;ㅜㅜ 둘이 살아온 환경이 다르니..아예 안하시는게 아니라 좀 쉬다 늦게 하는거니 신부님이 이해하셔욤ㅎㅎ'], ['저는 주말부부라서 일단 신랑이 엄청 열심히 하고 있는데... 급 신랑한테 고마워지는 글이네요 ㅜㅜ 같이 살다보면 싸우기 시작할까봐 겁나네요 ㅜ'], ['초반에는 이런걸로 조금 투닥투닥 하게되는것같아요~! 습관이 안되있어서 그런거니 점점 시간 지나면 변할거에요^^'], ['에휴 이거 힘들어요 저도 지난번에 싸웠었어요 ㅜㅜ 제가하는게 속편한 ㅜㅜ'], ['저와 같은 고민이시네요!! 저도 그래요, 바로바로 정리하고싶은데 남편은 모았다하더라구요'], ['저도 그래요~근데..다름을 인정해야 될 것 같아요.일단 하는 것에 의의를 두고, 아이한테 알려주듯이 해야 됩니다.ㅠㅠ'], ['그래도 하긴하니까요.. 내가 하는 방식대로 하라고 강요하는건 힘들거같아요 ㅠ'], ['ㅋㅋ 살림살이 분담이 진짜 제일 쉽지 않은 것 같아요~ 사람마다 이때쯤 청소해야겠다 하는 역치도 다 다르구... 그래도 젤 좋은 건 대화로 풀고... 서로 허용치를 넓히고... 그게 중요한가봐요~'], ['초반에는 이런걸로 싸우나봐요!! 잘 풀어나가세요!'], ['맞아요.. 같이 맞춰가는게 힘들죠... 남편분께서 일하고 집에와서도 일하는느낌이 드는게 싫다고 말씀하시는데 같은 직장인으로서 너무 공감가고 맞는 말이긴한데..... 집안일도.. 분명 일이긴 하죠ㅠㅠㅠㅠ'], ['맞춰가고 서로 이해하고 그게 중요한거같아요 버럭버럭하면..........ㅠㅠ힝'], ['맞아요 ㅠㅠ 초반에 서로 맞춰가는게 어려운것 같아요'], ['하나하나 가르치세요 ㅋㅋㅋ 저는 가르치는 중이에요 .. 생각보다 말 잘듣고 잘 따라주네요 ㅎㅎㅎ'], ['저도 하나하나 맞춰가는 중인데 어렵네요.ㅎ'], ['아... 저희는 업무 분담은 어느정도 되는데 자꾸 출근할때 불을 키고 가고 선풍기를 키고 가고 이런.. 문제들로 하아... ㅋㅋㅋㅋㅋ 화가 나게 되네요... ㅠㅠ'], ['성격차이죠 맞죠ㅠㅠㅠㅠ전 제가 귀차니즘이 심해서 신랑이 맨날 잔소리해요.,...'], ['안맞는거 한둘이 아니에요~ 저도 화장실 사용하고 물기 정리하고 물기 마르라고 문 열어놓는데 남편은 항상 물가득한 상태로 문을 닫아놓아요 ㅠ 문 열어보면 습기가 한가득 ㅠㅠ 옷 아무데나 벗어놓는것도 그렇고 아침에 이불정리 안하는것도~ 안 맞는거 찾기 시작하면 한둘이 아니에요~ 그래도 남편이 제가 부족한거 이해해주고 잘해주니 저도 그냥 참고 넘어가요^^;'], ['청소는 눈에 보이는 사람이 치운다고.. 치우는 사람만 고생이죠 ㅠㅠ 역시 어르고 달래면서 살아야 하나봐요 ㅜㅜㅋㅋ']]
    
    863
    28-집안일 시작 이제 집안일 시작해봅니당오늘 세탁기 한 세번을 돌아가야 될거 같네요ㅠ일단 1차로 돌려봅니다요
    
    [['저두맘편히집안일하고프네영'], ['맘님 오늘 퇴근하셔서 집안일 하셨나유?'], ['전오늘암것도안하고 이러고잇네요'], ['전 빨래가 너무 많아서 했었어요~'], ['습해서빨래도잘안마르고그러네요'], ['그래도 오늘은 해 쨍쨍하지 않나요'], ['저도 집안일 시작해야하는데.. 귀찮네여'], ['집안일 은근 귀찮아도 또 안할수가 없죠ㅜ'], ['그쵸 집안일은 안할 수가 없지요 ㅜㅜ'], ['맞아요~주말에도 쉴타임이 없내요ㅠ'], ['저도 몇번을 돌린지 모르겠어요 ㅠㅠ'], ['세탁기 매일 돌려도 어쩜 이리 많을까요'], ['그러게요ㅠㅠ계속돌려도 끝이없는 빨래ㅠㅠ'], ['앗 이거 보니 생각났어요 빨래 아까 다 됏는데ㅜ'], ['빨리 다 된거 널으셨겠지요?'], ['네 가서 후다닥 널고 나왔는데 딸이 부엌에서 울다가 놀다가;'], ['딸이 왜 부엌에서 놀다가 울어여??'], ['놀다가 저 와보라고 우는 거 같아요 ㅋㅋ'], ['집안일 모드를 시작 하시는군용'], ['퇴근했으니 이제 저에본업을 해야지요ㅍ'], ['저는 여새 본업이 바뀐거 같네요'], ['맘님 본업은 뭔가요 까페 댓글인가요?ㅠ'], ['카페 댓글이 많이 차지 하네요'], ['저도 그래요~집안일보다는 댓글이네요'], ['크으. 역시 엄마는 요일 상관없이 일을 합니다.ㅠ'], ['엄마에게 휴일이란 없습니다ㅠ'], ['저도 열심히 집안 살림하고 있습니다ㅋㅋ'], ['맘님도 집안일 많이 하셨나요'], ['저는 오늘 집안 살림 휴무 에요'], ['집안일은끝이없다지용으악'], ['맞아요 주말에도 쉴틈이 없네요'], ['전오늘도못쉴것으로예쌍합니당'], ['오늘도 바쁘게 움직이시나봐료'], ['오늘은더바쁠것같습니다용'], ['저도 낮에는 정신이 없긴하네요'], ['오늘도고생하셨습니다'], ['맘님도 고생하셨습니다'], ['오늘도열심히사부작해용ㅇ'], ['전 오늘 집안 청소 좀 하려구요'], ['전오늘도빨래전쟁입니다용'], ['오늘도 세탁기 열일시키는 건가요'], ['일차만하신건가요 저도 일차전만했어요'], ['저 이 차전으로 돌렸습니다!'], ['이차전까지 하신건가용 고생하셨어요'], ['제옷들이랑 수건을 빨았지요'], ['오늘은 빨래쉬는건가여 저도 일어나자마자 빨래를해야되여 ㅋ'], ['빨래는 다 해서 이제 없어요'], ['이제 제가하면되는건가여 ㅠ 어제 안하고 나갔더니 ㅋ'], ['맘님이 이어서 빨래 하시면 됩니덩'], ['바톤 터치해주세여 손내밀어주세여 ㅋ \n'], ['요이땅 이제 맘님이 세탁기 돌리면 됩니다ㅋㅋ'], ['다들 오늘 세탁기 돌리는 날이였나 봅니다'], ['쌓여있어서 안 돌릴 수가 없었어요'], ['ㅋㅋㅋ 저도 내일은 세탁기 돌리는 날입니다ㅠ'], ['내일 세탁기 열심히 돌아 가겠군요'], ['요즘 열심히 산책햇더니 ㅋㅋ 빨래 산더미'], ['맘님의 땀옷들 인건가요 ㅋㅋ현준이렁'], ['ㅋㅋ 현준이 오늘 설사 바지 만들어왓어여'], ['땀으로 설사 바지가 된건가요 ㅋㅋㅋ'], ['아니 기저귀 채웠는데 응가가 새버렷네여 ㅋㅋ'], ['아 그래서 설사바지 얘기하셨군요ㅋㅋ']]
    
    882
    집안일이 조금씩 편해지고 있어요 또 다른 건 없을지 조언구해요! 맞벌이하다 아이 생기면서 그만둔지라결혼한지는 오래됐지만 주부경력은 짧아요 워낙 집안일 하기 싫어하고 게을러빠진 탓에세상에서 설거지가 제일 싫었고맞벌이 그만 둔 이후로는 갑자기 밥타령해대는 남편이 낯설었죠여기서 밥타령은 반찬이 아니고 밥이 딱딱한 걸 못참는 남편,,, 갑자기 변하니 좀 당황스러웠어요이제 아이 좀 키워놓고 나니 제가 힘들었던 집안일이조금씩 편해지네요우선 식기세척기를 샀더니 설거지가 줄었구요 설거지가 없을 수는 없지만 정말 시간이 1/3으로 줄었어요그리고 로봇청소기를 사서 바닥을 맡겼구요애들이 크면서 매트를 없앴더니 가능해졌네요그리고 무선 물걸레 무선 청소기를 샀더니 좀 더 청소가 편해졌어요건조기는 사려다 포기했어요 친구가 옷을 몇벌 버리더니 절대 사지말라고 하고저도 건조기 놓을 자리가 마땅치않아서 ㅠㅠ그리고 에어프라이어기!! 아이들 생선구워주기에 정말 좋아요만두 떡 삼겹살 정말 굽기 편해졌어요그리고 압력밥솥!!!! 30분 걸리던 밥이 이제 15분이면 가능 ㅋㅋㅋㅋㅋ밥이 더 찰지고 맛있어요 그리고 엄청 빠르고요 ㅎㅎ또 뭐가 있을까요?? 조언 주세요~~아 혹시 빨래 때 잘 지워지는 법 좀 ㅠㅠ초콜릿 과일즙 물감..아주 스트레스에용 ㅠㅠ
    
    [['전 건조기가 살림최애템인데 ㅋㅋ 사기전엔 건조기 엄청 불신했었는데 지금은 없으면 안되요 ㅋㅋ 수건이랑 애기옷만 돌려도 좋아요!!'], ['저도 고민 참 많이 했는데 첫번째로는 놓을 자리가 없고 ㅠㅠ 두번째로는 제가 빨래를 바로바로 돌려서 적은 양을 돌리기때문에 세번째로는 친구가 옷 몇벌 버리고 절대 사지말라고 해서 ㅋㅋㅋㅋㅋ\n건조기는 좀 더 넓은 집으로 이사가면 구매하는 걸로...하려고요 ㅠㅠ 지금 집도 좁은 건 아닌데\n세탁실 크기가 재앙입니다 ㅠㅠ'], ['밥하기 싫을 때 반찬은 가끔(?) 사다 먹어요 ㅎㅎ'], ['정말 좋은 팁이네요 집앞 반찬가게를 한번 가보아야겠습니다 감사해요'], ['ㅠㅠ 이해합니다. 저는 그만두자마자 갑자기 아이를 키우게 됐답니다...그 곤혹스러움이라니..\n10년 넘게 회사원으로 일하다가 갑자기 주부라니... 당황스러웠지요 \n같이 힘내요'], ['스타일러 덕분에 세탁소가는 횟수가 줄어요.'], ['아 스타일러.. 남편이 계속 사자고 원하는 바로 그 스타일러\n압력솥값이 스타일러 뺨을 치는 바람에 그건 그럼 다음달로 미뤄봐야겠습니다 ㅠㅠ'], ['옷도 다려주니...다림질횟수가 줄어들지요...'], ['우와 옷도 다려주는 기능이 있군요. 정말 깔끔쟁이 남편에게 딱 어울리는 제품입니다.\n감사해요 ^^ 구매욕구 완전 업입니다'], ['건조기 완전 니트아니면 안줄어요 ㅎㅎ 전 면티랑 애기내복 다돌리는데 줄은거없어요 건조긴 최애템이예요. 정 불안한옷은 안넣으심되요.(전 양말, 골프웨어, 비치웨어 빼곤다돌려요) 수건 일반옷 건조랑 이불털기기능 진짜좋아요~^^'], ['건조기는 이상하게 동생들 다 쓰고 친구들도 다 쓰는데 별로 안 땡기네요..\n그 먼지의 양을 보니 정말 땡기긴 하는데... 친구들 보니까 110짜리 입힐꺼면 미리 130 사서 줄을 거 생각하라 하더라고요 건조기는 잠시 킵해두도록 하겠습니다. 아직 빨래는 그다지 스트레스는 안 받네요 신기하게 ㅋㅋㅋ'], ['아 그친구는 무려 2개나 있어요 \n그 네이버껀데 하나는 브라운 모양 귀여운 거랑 하나는 저 브라운 윗단계 모델인데 모양은 투박한데 이름은 샐리인 ㅋㅋㅋㅋ  아이들이 매일 대화를 시도해요\n노래틀어줘 동요들려줘... 얘도 미세먼지나 날씨 체크에 좋은 것 같아요\n'], ['흰빨래 누렇게 된거는 과탄산 소다 녹인물에 담궈놨다 세탁기에 담근물까지 같이 돌림 하얗게 되더라구요^^'], ['아 그렇군요 얼마나 담궈놔야 할까요? 혹시 흰빨래가 아닌 것들은 담궈놓으면 색이 바랠까요?\n흰옷보다 줄무늬 옷 혹은 색깔있는 옷이 많아서 ㅠㅠ'], ['그 세탁 라벨에 보시면 보통 염소계 표백 금지 이렇게 되있는건 락스 금지구요! 과탄산소다는 산소계라 괜찮더라구요~ 근데 염소, 산소계 둘다 안되는 옷들도 있어서 주의하셔야해요^^\n전 흰티 이런건 그냥 면속옷들이랑 아예 삶아버려서ㅋㅋ 정확한 시간은 잘 모르겠어요ㅠㅠ'], ['감사합니다. 애들이 어리다보니 아주 난리도 아니에요\n이젠 제 옷에다까지 묻히고 ㅠㅠ 빨래 즐거웠는데 이제 지워지지 않는 때 보면 스트레스 받아용 ㅠㅠ'], ['빨래때는 과탄산소다가 짱\n심한 얼룩은 몇시간 담궈 둬보세요\n뜨건물에 녹여서 담구시면되요'], ['그렇군요 과탄산소다를 열심히 ㅋㅋㅋㅋㅋㅋㅋ 애용해보아야겠어요 ㅎ'], ['과탄산은 색은 안빠지고 얼룩 때 등등만 없어져요..심한건 3시간정도 담궈요'], ['3시간 이나 담궈야 하는군요 감사합니다'], ['음식물 같은건 주방세제 발라놨다가 칫솔질 하라고 들은듯..'], ['주방세제 좋은 팁도 감사해요 ^^ 오늘부터 시작해보겠습니다'], ['ㅋㅋㅋㅋ 퐁퐁을 세탁기에 투하해도 좋겠네요 ㅋㅋㅋㅋ 한번 오늘 해봐야겠어요 ㅋㅋ 뽀독뽀독해지겠네요'], ['저도 지금 말씀하신 아이템 다 갖고 있는데 진짜 최고죠!! 더더 게을러지는중 이예요ㅋㅋ 전 다음 아이템이 물걸레로봇청소기예요~ 무선물걸레도 귀차나서요ㅎ 하나씩 늘려가는중이라 8월에 사려구 합니다ㅋㅋ 글구 건조기대신 제습기로 빨래 말리는데 방에 두고 돌리면 반나절도 안되서 다 마르더라구요~ 그리편한건 아니지만 그래도 금방 마르니 좋아요^^ 마지막으로 제가 젤 사고픈 아이템이 음식물분쇄기처리기요ㅋ 친구네 있어서 해보니 음식물 그대로 씽크대에 넣으니 대박!! 음식물때문에 냄새날일 없고 버리러 가지 아나도 되고 좋더라구요~ 금액이 비싸서 젤 마지막으로 미뤘습니당ㅋ 그거빼면 또 있을까요ㅋ'], ['우와 대단해요 다 가지고 계시다니 부럽습니다. 전 일단 1순위가 식기세척기가 제일 편해요\n제일 잘 산 아이템이고 제일 잘 쓰고 있네요\n댓글들 보다보니 정말 아이템이 참 많네요 일단 스타일러는 장바구니 저장저장 ㅋㅋㅋ\n제습기는 에어컨이 잘 해주고 있어서.. 흠 따로 사야 할까요??\n분쇄기는 환경오염때문에 안 좋다는 소리를 들어서 요샌 환경부 인증도 받고 좋아졌다는 소리는 들었는데...  좋은 팁 감사해용 ^^ 또 있을지 궁금하네요'], ['물걸레 8장 가지고 돌려가며 쓰는데 좀 더럽다 싶으면 제가 빨고 귀찮으면 세탁기에 돌려요 ㅋㅋㅋ\n일단 한번 애벌로 빨고 돌리는데 다른 빨래들이랑 섞이면 좀 그러니까 더러운 걸레들만 모아서 빨때 ㅋㅋㅋ'], ['전 구매한지 좀 오래됐는데 한경희꺼 써요. 요새 보니까 이효리네 민박에서 나왔었나봐요\n근데 무선은 밧데리에 한계가 있어요 전 한번 무상으로 교환받았어요 근데 택배보내기가 일이었답니다 ㅠㅠ'], ['스타일러보다는 건조기요~ 스타일러는 보통 겨울옷 냄새제거로 많이 쓰고요. 건조기는 맨날 써지네요~^^'], ['아 여러분께서 건조기 추천을 하시네요... 정말 땡기는데요 ㅋㅋㅋ\n건조기는 비싸기도 하고.... 놓을 데도 없고 하니 .... 다음 집 갈때 꼭 구매하도록 해보겠습니다\n조언 감사해요 ㅋ'], ['저는 윤스니님의 15분 압력밥솥이 너무 궁금해요~ 전기밥솥을 잘 못 샀는지 밥이 너무 맛이없고 변색이 빨라서요ㅠㅠ 밥맛이 좋다고 하시니 더 궁금해요^^'], ['휘슬러꺼 샀는데 백화점에서 구매했어요 \n무늬가 있으면 더 높은 등급이고 좋다고 해서 일부러 그걸로 샀는데... 뭐 크게 차이는 없는 것 같긴 한데.... \n가격이 좀 나가긴 하는데 밥이 넘넘 맛있게 되서 만족하는 중이에요 \n전기밥솥이랑 차이요? 정말 많이 나긴 합니다. 일단 압력솥으로 밥을 하고요\n전기밥솥에 보관을 하면 밥이 변색이 잘 안돼요 맛도 그대로구요\n이제 밥 하는 건 좀 익숙해졌기에 누룽지에 도전해보려고 해요\n요새는 압력솥이 넘넘 쉽고 편하게 나와서 ㅋㅋㅋ\n'], ['우와 꿀팁 너무 감사해요~ 당장 휘슬러부터 알아봐야겠어요^^'], ['구매하시려거든 다음에 저한테 챗으로 얘기해주세요 뭐 판매처를 알거나 그런 게 아니고\n저도 이거 사려고 정말 많이 알아봤거든요\n어떤 게 좋은지 알려드리고 싶어서요 ㅋㅋㅋ\n이게 한두푼이 아니고 제가 산 아이템중에 비싼 축에 속하니까 쉽게 구매결정할 물건이 아니라...\n나중에 구매할때 연락주세요~ 제가 아는 정보를 알려드릴께요 ㅋㅋ'], ['맞아요 맞아. 일반 압력솥은 안 써봐서 모르겠지만 정말 맛있어요!\n그리고 압력솥이 저 어릴때 친구네집 놀러갔다가 제 눈앞에서 뚜껑이 펑 터진 적이 있어서\n전 약간 트라우마??? 는 아니지만 압력솥은 어렵다. 무섭다 이런 생각이 있었는데\n이건 정말 넘넘 쉽고 빠르고 완전 좋아요 강력추천!!! 추 딸랑이는 그런 거 아니에요'], ['정말 감사해요~~~^^ 최고!'], ['걍 알려주시면 안 되나요 저도 궁금한뎅'], ['아 저도 모델명은 잘 몰라서 그리고 원하시는 기능이나 가격같은 거 다 다르자나요 ㅋㅋ 필요하심 챗주세용!!'], ['건조기 정말 좋아요~ 소량도 그냥 막 돌려요! 전기건조기는 꼭 세탁실에 안 넣어도 되니 추천합니당'], ['건조기 정말 좋군요~~ 전기건조기로 다음에 꼭 알아봐야겠네요 ㅋㅋ'], ['진짜 아끼는 옷만 따로 널고요~~!! 근데 생각보다 마니 안 줄더라구용'], ['다들 건조기 추천하시니 앞베란다에라도 놔야겠어요 오늘부터 남편에게 폭풍애교 들어가야겠네요 ㅋㅋ'], ['아이참 ㅋㅋㅋㅋ 건조기만이라도 킵해보려했는데..ㅋㅋ 일단 스타일러지르고 건조기도 겟해야겠어욬ㅋㅋㅋㄱㅅㄱㅅ'], ['근데 스타일러는 엄청 날씬한데 ㅜㅜ 휴 어디다 놓아야할지ㅡㅡ 결론은 세탁기와 멀어지는 수 밖에 ㅜㅜㅜㅜㅜ'], ['저 음식물쓰레기갈아주는 기계? 설치했는데 진짜 너무너무 편해요.. 매일 음쓰때문에 설거지 하기도 싫고 그랬는데.. 걱정이 하나도 안되요 ㅎㅎㅎ 하나 장만하세요'], ['전 진짜 게을러서 음식물쓰레기 버리러가는 게 힘들더라고요 ㅜㅜ 음식물쓰레기 분쇄기??것도 알아봐야겠어요 역시 돈쓰는 건 세상에서 젤 쉽고 좋네요 하하하하...넘 많이 필요하네요 그런데 ㅜㅜ 조언감사요'], ['압력밥솥 저희 친정은 휘슬러 우리집은 풍년꺼 쓰는데 풍년것도 괜찮아요^^ 가격차이는 많이 나지만 쌀이 좋으면 풍년것도 너무 맛나게 잘 되요~!!'], ['풍년것도 알아봤는데 제가 조리하기 힘들것같아서요 ㅜㅜ 조금만 어려워도 못하겠슈 ㅜㅜ 아마 밥맛은 비슷하겠죵 나중에 좀더 주부 스킬이 늘면 풍년도 도전 ㅋㅋ'], ['저두 압력밥솥-휘슬러 쓰고요 \n물걸레로봇청소기 - 브라바 랑 lg a9 무선청소기 같이쓰고 \n트윈워시 세탁기가 위아래로 열일해주고 건조기는 수건이랑 속옷용으로만 돌리고요 \n이만큼도 엄청 도움되는거지만\n\n가장 결정적인건 부지런한 남편을 두는것같아요 이건 아마 현생에는 힘들듯하지만요.. ㅎㅎ'], ['전 돈잘벌어오는 남편보다 부지런하고 집안일 잘도와주는 남편이 더 좋은데..저도 이번생은 포기 힝 ㅜㅜ'], ['요즘 식기세척기 알아보는 중인데\n어떤 제품 사셨는지 여쭤봐도 될까요?^^'], ['네 그럼요 ^^ 전 4인가족이라12인용 동양매직 샀어요 가격은 70만원선이었던듯.. 빌트인이라 설치비 5만원인가 있었구요 성능이나 이것저것 모듀 만족입니다 ^^'], ['감사합니다!^-^'], ['음식물쓰레기 처리기요~ 여름에 모아서 버리기도 힘들고 자주 버리자니 귀찮아서 설치했는데 완전 신세계예요~ㅋ']]
    
    885
    25.집안일 모터달기 집안일 모터달았어요ㅋ화장실 쓱싹하고바닥 걸레질 싹하고그릇정리하고왜냐고요 낼산부인과 가거든용왠지 내일일꺼같고 느낌이요ㅎㅎ딸이 아침에 동생언제 나오녜요~으앗
    
    [['분만 하시러 가시는거에요~?  저는 금요일에 검진 하러 가는데 신호가 그전에 왔으면 좋겠어요~'], ['분만하고싶습니다 이제 지쳤어여 ㅋ'], ['저랑 똑같은 마음이시네요~!! 저도 이제 낳고싶어요~자분 하시는거죠~?'], ['느낌이오나여!진짜낼인가여'], ['내일 내진한다고하네여 떨립니다 ㅋ'], ['게시글못봣는데 어찌됫나요'], ['어머 내일인가용 대박..내가 다떨려요'], ['맘님 떨고계시지마세여 ㅋ 떠는건 제가떨겠습니다 ㅋ'], ['ㅋㅋ그래요 그래도 같이 하면 덜떨릴듯'], ['이제 아가 만나는거예영? 조심조심 청소해요!'], ['네네 아가만날날이 얼마안남았어요 러브님 잘계시는거죠?'], ['옙 잘지내고있어영.. 떨리시겠어영'], ['낼 순산잘하세요 화이팅에여^^'], ['화이팅해봅니다 저너무 떨리네여 ㅋ'], ['어머~진짜요?내일 만나는 건가요?'], ['내일만나고싶은데 진통이 걸려줄까여 ㅋㅋ'], ['오늘 병원서는 모라했어요?'], ['디데이인가요~두근두근하시겠어요'], ['두근두근입니다 내일도 안오면 어찌되나여 ㅋ'], ['화이팅해볼게여 ㅋ 저너무 떨리네여 ㅋ'], ['와 ㅋ ㅋ말이쉬워요  저있다 병원갑니다ㅋ'], ['출산임박이신데 너무 빡신거아니십니까'], ['그러게여 어쩌다보니 빡세게 보내고있네여 ㅋ'], ['아가가 금방쑤~욱나오것네요'], ['어머나.세상에.. 내일 애기 만나는 거예요?!-!!떨리시겟어요ㅋㅋ'], ['아가만나러가는데 나왔으면 좋겠네여 힘들어여 ㅋ'], ['우와  언능 아가 만나고ㅜ싶겟어요그치용'], ['저도모터가동해야되용'], ['오늘도 모터 가동하고 계신건가여 아님 가신건가여 ㅋ'], ['오늘모터가제정신ㅇㅏ니네요'], ['에고 커피한잔드셔요ㅜ드실때된거같은데 말이죠'], ['커피한잔으로부족하지말입니다'], ['한잔더드셔요ㅋ  세잔은기본아닙니깡'], ['ㅋㅋ 마자영 동생 언제나오느거지여'], ['동생 내일 또 진료보러갑니다 ㅋ 이눔 나올생각을안하네여'], ['ㅋㅋㅋ 엄마 이거 하는 마지막에 나올듯여'], ['그랬으면 좋겠네여 마지막날 나오길바래봅니다 ㅋ'], ['ㅋㅋㅋ 그렇다면 제가 제일 축하해드리기로햇지여'], ['펑펑 울어주기로했잖아여 눈물 기다릴꼬예요 ㅋ\n'], ['ㅋㅋㅋㅋㅋㅋ아 내눈물 사진으로 찍어서 올릴거에여'], ['그럼 감사합니다용 ㅋ 저내일 안나오게 기도해주세여 ㅋㅋㅋ'], ['ㅋㅋㅋ 그게 무슨말인가옄 병원갓다가 입원하세여'], ['오잉 ㅋ 저입원하면 핸드폰으로 달려야됩니다여 ㅋㅋ\n'], ['그렇게 움직여도 안나오다니 빨리 나오기싫은가봅니다'], ['남들은 걸레질 한번하면 나온다는데 저는 두번해도 안나옵니다'], ['그러게요 말일까지 버틸껀가봅니다 ㅋ'], ['말일까지 버텨주면 고맙지요ㅋㅋ 힘들껀 뻔하지만여'], ['마지막주에 진통을 느끼면서 하는건 아니겠지요'], ['헉 그러면안되는데말이져 몇일안남았는데 아까워여 ㅠㅠ 아'], ['일단 내일 병원부터 잘 댕겨옵시다'], ['그래야겠어여 ㅋ 내일 오전퇴근하고 가야되는데말이져 ㅋ'], ['아~~이제 다되어가는군요 화이팅!']]
    
    940
    남편분들의 집안일 분담의 게시글을 읽고나서.... 우선 푹푹찌는 한여름에 불철주야 열심히 일하시는 우리 남편분들에게 진심어린 경의를 표합니다.새벽같이 일어나서 출근 시간에 쫓기어 따뜻한 아침 식사도 거르고 발디딜틈 없는 대중교통으로 땀이 한범벅 되시어 회사에 도착하는 우리 남편분들.... ㅠㅠ화이팅입니다!!!불금 시원한 맥주 한잔하시고 푹쉬십시오♡내일은 또 청소하는 주말이군요.. 젝일 ㅡㅡ;;
    
    [[''], ['엇 나는 이런 이모티콘 없는데.. ㅜㅜ'], ['새벽같이 일어나 아이챙겨서 데려다 주느라 따뜻한 아침도 거르고 정신없이 출근하는 아내들도 화이팅입니다 ㅎ'], ['어이쿠 물론입죠~~~\n\n아내분들이여 화이팅~~~~ ㅎ'], ['그런데 그 마누라가 꼼짝 못하는 대상이 있으니......  바로 아이들이네요.ㅋㅋㅋ\n그나저나 글씨체가 참 이쁘네요^^'], ['글쵸... 그래서 어머니는 위대하다고 하지요...\n\n어머님들도 화이팅~~~♡'], [''], ['우째 나만 이런 이모티콘이 없는지.. ㅡㅡ;;'], ['ㅋㅋㅋ \n살고 싶습니다...ㅋㅋ'], ['내일 대청소하자~~~!!!!!  ㅋ'], ['그래야겠어요^^'], ['앗..........'], ['아........\n'], ['울 베리베리♡'], [''], ['난 요즘 대들어. ㅎㅎ 흰눈동자 보이면 다 해결돼'], ['ㅎㅎ\n\n형 나도 드러누울까봐요 ㅋ'], ['울 신랑은 이 더위에 골프를ㅜㅜ\n짠하고 불쌍하다는ㅠㅠ'], ['아이구 우째요.. ㅠㅠ\n\n신랑분 오시면 아이스팩 해드리고 드시고 싶은거 다~  만들어드리셔요..  ㅎ'], ['소괴기 사놨죠ㅋㅋ'], ['역쉬 현모양처~~~♡'], ['저도 없어요~ 이모티콘! ㅡㅡ;'], ['어디서 구하는지 알아보고 기브엔 테이크 합시다  ㅡㅡㅋ'], ['그리 와닿지 않는 글이여..ㅋ'], ['일단 곰한테 코브라 어찌되었는지 물어봐바~~~  ㅋ'], ['코브라는 뭐여..ㅡㅡㅋ'], ['곰은 알어~  ㅎ'], ['그럼 난 악어할려..'], ['우리집은 사장님 선생님 마누라 다 있구만요 흠흠 ㅋㅋㅋ'], ['키야.. 맞네요 엄지척!!\n근데 일본은 또 언제 가셔요? ㅎ'], ['담주에 가요 ㅎ'], ['어쩜이리도 감동적일수가'], ['오~ 감동덕분에 하트가 23개네요.. ㅎ'], ['돈도벌고,  청소도 하는 누군가는 울집에~ㅋ'], ['대들었다간... 상상만해도... 지못미 울 이스트파이브~~♡'], ['잘해드려야 번개도 나올 수 있는법~~~ ㅋ'], ['ㅎㅎㅎㅎㅎ\n\n욱하던 분이 요즘 순해진 이유가~'], ['누나한테 대들었다간....  ㄷㄷㄷ\n\n^^♡'], ['빨리 깨닫는 사람이 난사람인겨'], ['시끄럿.. ^^♡'], ['안자냥 ㅡㅡ; 알람소리에 잠깸 ㅡㅡ'], ['ㅋㅋㅋ간단명료한 강의!!'], ['한마디로 "와이프님께 무조건 잘하자~"  ㅋㅋ'], ['ㅋㅋㅋ \n울신랑이 이걸 알아야하는데 ㅋㅋ'], ['카페 가입시켜요.. ㅋㅋ'], ['안본걸로 할텨~~'], ['형님은 모범상, 우수상 받으셔야죠..ㅎㅎ'], ['오늘도 더운 날\n가족위해 열일하는 엄마님들\n화이팅입니다'], ['넵~ 이제 닉네임 붙이셨네요.. 오늘도 화이팅입니다~^^'], ['불철주야야야야야~ 일하시는 흥부님 화이팅^^'], ['불철주야주주주 입니다..\n메이씨님도 날은 덥지만 화이팅입니다 ^^'], ['불철주야놀놀놀 같은데요~~ ㅋ'], ['설마요~~ㅋ'], ['완전 맞는 말이네요~ ㅋㅋㅋ 남편한테 보내야겠어요~'], ['우클릭금지 해제해 두었습니다..\n\n저장하기, 보내기...  ㄱㄱㅆ~~'], ['우리집 양반은 살기싫은가 봅니다 ㅎㅎ'], ['오늘처럼 더운날 베란다 청소...  시키십시오.. ㅎ'], ['소름돋네요. 지금 베란다 청소중입니다 ㅋㅋㅋ'], ['마누라한테 대드는 사람이\n제일 용감한 사람같어..\n우리집은 상상도 못하니 ㅡㅡ;;'], ['현명하신 형님 ^^\n삶의 처세술을 널리 알려주십시오.. \n\n아.. 어제 뵈었어야 했는데요 ㅜㅜ'], ['아 정말 너무 웃겨서 육성으로 깔깔거리며 웃었어요.. ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['남편 보여드리시고 내일 대청소하셔요.. ㅋ'], ['아침에 청소하고 점심때 걸레질 했는데 시꺼먼것이 역시 집에 사람이 많은 주말은 ㅠㅠ\n근디 쭉이네~~방학방학방학 ㅡㅡ;;;'], ['암만 싸미는 복받은겨~~~ ㅎ'], ['퍼갈게여 ^^'], ['노코맨트 하겠습니다. ^^; \n\n'], ['절반만...기억할게요. ㅎ']]
    
    1006
    집안일 스타트 어제 파업으로 인해 집이집이 전쟁터라죠 슬슬 세탁기 먼저돌리면서 청소 시작합니당
    
    [['저도세탁기돌리고있어요'], ['전 두번째로 또 돌리네요'], ['언니 저도 세탁기 돌리며 집안일 시작이네여 ㅋ 홧티하자구용ㅠ'], ['홧팅홧팅입니다'], ['땀이줄줄 이에여 언니ㅠ'], ['잠시 쉽니당ㅜㅜ 너무 더워요'], ['오늘도 하루가 저무네여ㅠ'], ['오늘은 시간이 잘가네요'], ['오늘도 시간은 잘 간다지여'], ['그런것같아요 벌써  8시가  되어가네요'], ['아홉시네여 아기재우고 저녁먹으니@.@'], ['시간이 참잘가죠'], ['네 하루 후딱이네요'], ['벌써 8시~~'], ['저녁먹고오니 아홉시네여ㅠ'], ['다 정리하고 하믄그리되더라구요'], ['하 오늘도 늦을듯한ㅡㅡ'], ['일찍 마무리가 안되죠ㅜㅜ'], ['그니까여 ㅋ 아직도 하고있네요 흐'], ['저는 계속할듯요'], ['저도 계속 하고있어염'], ['파이팅입니다!!^^'], ['화이팅입니다'], ['미먼때문에 세탁기 못돌리고있네요ㅠ'], ['그래도 돌리구있어요ㅋ'], ['전 실내에서 말리려구요'], ['화이팅입니다~~^^'], ['홧팅홧팅 입니당'], ['^^ 아침부터 바쁘시겠네요~~'], ['잠시쉬고있네요'], ['오늘도 너무덥네요.\n저도 집청소 해야하는데..모든게 귀찮네요ㅋ'], ['저도 반은 하고 기다리고 있네요'], ['끝이 없는 집안 일~ 홧팅!!^^'], ['끝이 있으면 좋겠습니다'], ['역시나하루안함티가많이나지요~ㅜ'], ['그니까요 너무 티나요'], ['홧팅하셔욧ㅎ저도돌리는중'], ['홧팅홧팅 해서 다했지요 대충요ㅋ'], ['크크 어제는 파업하실 만두 하셧쭁'], ['오늘두 그냥 대충해서  파업아님 파업이네요ㅡ'], ['크크 아기 있으면 집안일 쯤은 미뤄도 되죠~'], ['그랴두 되는데 안치우면 정말ㅜㅜ'], ['키키 전 그냥 포기했어요 시간 나면 한꺼번에 와아아악 치우고'], ['한꺼번에  치워두 되요^^'], ['고생하셨네요 저두 이틀 집안일에  댓글놀이를  못했네요'], ['집안일이 끝이 읍죠'], ['글킨하죠  근데 하기싫어요 누가해줬음 ㅎㅎ그게누구일까요?ㅋㅋ'], ['그니까요 없다죠ㅜㅜ'], ['신랑이있지요ㅋㅋ'], ['신랑ㅋㅋ 어쩌다 한번 도와주니 ㅋㅋ'], ['시켜야죠'], ['일찍가고늦게오니까요'], ['늣게할수있는일이 ?'], ['음..빨래정리정도요ㅋ'], ['그럼 그거라도'], ['그럴까요ㅋ'], ['네 ㅎㅎ'], ['알겠습니다 ^^']]
    
    1021
    집안일하기 귀찮아요ㅠㅠ 성가지도 해야하고 거실도 치워야하는데 귀찮고 하기싫어요ㅠㅠ 혼자 치워야되서 더 힘든거같아요치우긴 치워야하는데ㅠㅠ
    
    [['저두요 귀찮아요ㅜㅜ'], ['ㅠㅠ 아기 있어서 더 그런거같아요'], ['그니까요  더 부지런히 움직여야 하는데 그반대로 되네요'], ['이제 설거지 마치고 빨래 개네요ㅠㅠ 대충한다곤하는데 해야할일이 많아요'], ['집안일은 대충해도 일이  많다지요'], ['맞아요ㅠㅠㅠ 오늘은 집안일 쉬어볼까해요'], ['하루쯤 안해두되요'], ['그래요~ 바닥에 있는 부스러기만 치워야겠어요'], ['눈에 보이는것만 하자구요'], ['그래요~ 아기 데리고 나가고 싶은데 넘 더워보여요ㅠ'], ['여기는 선선해서 좋아요'], ['집에 있으니 아가도 답답해하는거같고 어지르네요@.@'], ['아기들은 나가는게 좋아하더라구요'], ['저도지금집이개판5분전인데대충만치웟어요ㅋㅋ또어지르니깐요^^'], ['ㅠㅠ 주방은 엄마인 제 구역이니 청소하러 갑니다'], ['저흰신랑도설거지하는데요ㅋ자주해줘용'], ['저희남편은 가끔요 아주가끔!'], ['전요즘매일해달라고햇더니해주네용ㅋ'], ['전 땀이 넘많아서 여름에 걸레질이 넘귀찮아요ㅜㅠ'], ['전 밀대가 있어서 걸레질은 괜찮네요'], ['저희도 있답니다 ㅋㅋ 굳밤'], ['저도치우기귀찮아용'], ['ㅠㅠ 저랑같네요 힘들어요'], ['ㅋㅋ 저도 맨날 귀찮아서 안해요'], ['전 안하면 금방 빨랫감 생기더라구요ㅠ'], ['ㅋㅋ 한꺼번에 하고 싶을때 하는거죵 ㅋㅋ'], ['ㅜㅅㅜ쉬엄쉬엄하세여..그러기쉽지않지만ㅜ'], ['빨리 헤치우고 우리 아가랑 놀고 싶은데 쉽지않아요ㅠ'], ['저두 요즘 귀찮아 잘안치게 되네요'], ['여름이라 그런가봐요~ㅠㅠ'], ['배두나오구 덥구 그래서 그런거 같아요'], ['손놓고싶지만 더러운꼴 ㅠㅠ 못보는성격'], ['저도요ㅠㅠ 너무 더러움 정신사나워요'], ['귀찮을 땐 그냥 안해버리기... 임신 후부터 조급한 제 성미 다듬느라 가끔 쓴 방법이네요 ㅋ'], ['저도 안하고 싶은데 더러운꼴 못봐서 대충이라도 치워놔요'], ['저도 하루종일 아가보면서 그냥 앉아잇다가 새벽반 오기전에 설거지에 정리햇어용'], ['전 대충 해놨어요 내일 일어나면 또 치울게 있을거지만요ㅠㅠ'], ['그러니까유 집안일은 매일매일 잇지요 전 내일 아들오기전에 책정리도 해야해영'], ['아들 주말에도 어디 가나요?'], ['아 지금 시댁에 격리되어잇어용 수족구땜시요!^^ 내일와용'], ['아하 그렇군요ㅋㅋ 아기 한명 더 추가되니 배로 힘들겠어요ㅠㅠ'], ['크큭 아직까지는 그래도 친정엄마가 와계셔서 봐주셔서 ^^ 담주부턴 본격적으로 저 혼자 해야함니당'], ['친정엄마가 와계시면 좀 수월하겠어요~^^'], ['그쳐*^^ 근데 담주부터는 안오실듯요.. 원래 6월까지만 조리해주신다고 햇엇거든용 힝'], ['매주 매일 오시다니 대단하셔요~ㅠ'], ['집이 바로 옆동네에요^^ 주말은 안오셧쥬 그래서 오늘은 혼자 두찌봣어용 신랑도 시댁가구~'], ['그러셨군요~ 첫째때보단 둘째가 신생아때 좀 여유로워요? 궁금해요ㅠ'], ['네 확실히요 크큭 한번 해봐서 아기를 다뤄봤으니요^^ 어제처럼 갑자기 열이 확오르는 경우는 예외지만요 힝'], ['어제 열올랐군요ㅠㅠ 지금은 괜찮나요~?'], ['네 39도 넘엇엇어요 응급실도 다녀왓네요 다행히 미온수로 닦아줫도니 금방내렷어뇨 지금은괨찮아요^^'], ['다행이네요~^^\n둘째 태어남 첫째 밤낮도 바뀌고 그런다던데 맘님은 어떠셨어요?'], ['아 그래서(?) 따로 자요 새벽에 괜히 아들램 깰까바 첫찌랑 아빠는 안방에서 두찌랑 저는 거실에서 자유 힝'], ['아하 그렇군요 그나마 덜 깨겠네요ㅎ'], ['둘째가 새벽에 깨서 울다가 첫째 깰까바영 킁 근데 첫째가 낮밤 바뀌면 답없겟는데요 하핫'], ['그러게요 상상만으로도 끔찍해요'], ['그니까유 저 오늘부터 큰애집가서 같이자는데 또 거실신세에요'], ['다 미뤄두고 싶을때가 있지요ㅜㅜ 저두 귀찮아용..'], ['다 미뤄두고 쉬고싶을때가 간절하지만 육아만은 쉬질못하네요 하하'], ['그쵸ㅠㅠ 어디 하루라도 맡기실때 없으셔용?ㅠㅠ'], ['친정은 엄마가 매일 일나가셔서 맡기기 어렵고 시댁은 눈치보여요ㅠ'], ['시댁은 왜 눈치보이세요ㅜㅜ'], ['애맡기고 밖에서 노는여자 취급할까 겁나요ㅎ\n한번 애맡기고 친구랑 술먹고 들어왔다 걸렸어요'], ['신랑시켜요 ㅎㅎ  좀 느려터져도 제가는하기싫고  느려도 시켜요']]
    
    1047
    아 이제 누웠네요... 혼자 살때는 집안일이 이리 많고 힘에 부치지 않았던거 같은데 왜 요즘은 힘들까요 ㅠㅜ애기가 눈에 안보이면 징징대서 재우고 해서 그런가...밀린 집안일 하고 미싱좀 하려면 힘드네요 ㅜ ㅜ언능 다시 취직해버리고 싶은데 그런다고 집안일이 사라지는 것도 아니고...비와서 빨래 쿱쿱한 냄새나니 건조기라도 들이고 싶은데 들일 공간도 여의치않고 비싸고..그와중에 남편은 이에 건조기값이 들어가요 ㅋㅋㅋ 오늘 신나게 댓글 놀이 하니 뭔가 현실도피 하는거 같아 신났네요 ㅋㅋ 오밤중에 괜히 센치해져서 징징댔습니다ㅋㅋ 굿밤되세요!
    
    [['건조기..좋긴좋아요ㅠ\n그나저나 혼자서 하루만이라도 있어보고싶네요. 모두힘내요♡'], ['그러게요 ㅋㅋ 남편이랑 비교되니 상대적 박탈감..ㅋㅋ 잠 줄이고 나만의 시간을 갖으면 좋긴한데 피곤하고 ㅋㅋㅋ 힘내야죠! 힘냅시다!'], ['뭔가 익숙한 우리네 일상이네요..ㅋㅠ'], ['다들 이런 상황인게 슬프네요 ㅜ ㅜ'], ['저도 건조기가 너무 갖고싶네요 ㅠ'], ['저도...ㅋㅋㅋ 세탁기위에 올리면 딱인데 미니워시를 못쓴다하니... 저희 돈 생각않고 지르는 스탈인데 못지르고있어요 ㅋㅋ'], ['매일이 똑같은 것 같아 정말 힘들어요 그래도 우리 힘내보아요'], ['넹 ㅜ ㅜ 이렇게 취미라도 하고 맘 맞는 사람들끼리 얘기하면서 버텨야겠어요'], ['맞아요~그나마 취미가 우리를 살게하는듯해요ㅎ'], ['맞아요 ㅋㅋㅋㅋ 취미가 마음에 기름칠해주는 느낌! ㅋㅋ 숨통이 틔어요 ㅋㅋ'], ['숨통이 틔인다는 말씀이 정말 공감이 가네요'], ['ㅋㅋㅋㅋ 숨구멍 장 뚫고 살아요 우리 ㅋㅋ 화이팅'], ['다 비슷한 일상이네요 저도 3살아들 7살딸 키우면서 집안일하고 미싱좀 할라치면 하원시간이라죠..밤늦게 봉틀이 돌리면 시끄럽고.. 혼자살땐 내것만 하면 됐는데 4명이 어지르고 저만 치우려니 너무 힘들어요 ㅠ\n요즘 애들한테 너무 소리지른것같아 미안하네요'], ['두명이나 있으시면 전쟁이겠네요 ㅠㅜ 저는 아직 한명인데 너무 엄살인가...ㅋㅋ 엄마도 사람이라 감정적일 수도 있는 거같아요 ㅜ ㅜ 힘내요 우리'], ['건조기 눈 딱 감고 사세요 정말 좋아요 ㅜㅜ 우리 어무님들 모두 화이팅이에요!!'], ['ㅋㅋㅋㅋㅋㅋ 둘데가 마땅찮아서 요리조리 궁리중이예왜 ㅋㅋㅋ 진짜 다들 화팅 ㅜ ㅜ'], ['다들 혼자만의 여유를 꿈꾸시네요~ 저도 마찬가지 이지만 ㅜㅜㅜㅜ \n곧 결혼 앞두고 양가 어른들 허락하에 먼저 같이 살기 시작했는데 ... 벌써 집안일이 힘에 부쳐요 ㅜㅜㅜ\n거기다 일도 하고 있어서 더 그런거 같은데 지금은 애가 없어서 이렇지만\n아이까지 있으신 분들은 정말 대단한거 같아요~~'], ['ㅜ ㅜ 같이 한다고 해도 뭔가 책임감이 더 들드라고요... 저는 집은 엉망이예요 ㅋㅋㅋ 결혼하고 회사고 친척들이고 남편밥은 잘 챙겨주니가 인사드라고요... 출퇴근도 내가 더 먼데 ㅜ ㅜ 저는 월급조금나오긴하지만 전업이라 워킹맘들이 진짜 대단한거같아요'], ['전 저부터가 맞벌이 생각하고 있었는데 ... 점점 자신감이 없어져요 ㅜㅜㅜ'], ['전 당연히 맞벌이 할줄  알았는데 임신하고 이사하고 아기 낳고 하니 자연스레 단절됐네요... 시댁친정도 멀고... 상상도 못했어요 ㅜ ㅍ'], ['ㅜㅜㅜㅜ 시댁이나 친정 둘중 하나라도 가까우면 좋았을 텐데요...'], ['ㅋㅋㅋ시댁말고 친정.. 시댁가도 독박육아예요 ㅋㅋ 애 내려놓지도 못하고 ㅋㅋㅋ'], ['시댁이란 그런존재군요....... ㅜㅜㅜㅜ'], ['ㅋㅋㅋㅋ 애를 유난히 못보셔서 그래요 ㅋㅋ 애기가 안가고 낯가리드라고요 ㅋㅋ 농사하느라 지저분해서 내려놓기도 그렇고...ㅋㅋ 남편은 농사도우러가고...ㅋㅋ허허근데 엄청 자주간다는게 함정..ㅋ'], ['저랑 친한 언니네도 그렇던데 ㅋㅋㅋ 그래도 그언니는 친정이 그런상황이라 좋아하면서 가더라구요~ ㅋㅋ'], ['친정가면 밥도 주고 애도 봐주고 ㅋㅋㅋ 좋은데 ㅋㅋ 애기도 이뻐하니까 잘가고 ㅋㅋ 시댁에서도 잘가면 좋겠어요 ㅋㅋ 낯가린다고 머라고하시고 하나하나 다 뭐라고 하셔서 스트레스 받아서ㅋㅋ'], ['맞아요 ㅜㅜㅜ 그럼 엄청 스트레스 받죠 .. 그만큼 남편이 도와줘도 괜찮을까 말까인데...'], ['집에선 잘하는데 시댁가면 아들노릇만 너무 잘해서 ㅋㅋㅋ 걍 죽는구나 하고 가요'], ['그래서 남편이 남의편이라 하는건가요..??? ㅋㅋㅋㅋ'], ['그런가봐요 ㅋㅋㅋㅋㅋ 집에서 잘한서 한번에 와장창 깍아먹어요 ㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋ 똑같은 실수를 늘 반복하고.. 그쵸??'], ['맞아요 ㅋㅋ 조금씩 고치기는 하는데 속도가 너무 더뎌서 답답하네요 ㅋㅋ'], ['그래도 고치려고 노력하는 남편 예뻐해주세요~ 제 남자친구도 우쭈주해주니까 아닌척 하는데 더 하려고 하는거 같더라구용~^^'], ['ㅋㅋㅋ 기대치를 발바닥으로 내려놔서 하나만 해도 이쁘다 해요 ㅋㅋ 우쭈쭈하면서 델고 살아야하는건가 ㅋㅋ 결혼하고랑 연애할때랑 달라진더같은건 느낌탓일까요 ㅜ ㅜ'], ['연애할 때 다르고 결혼하고 나서 다르다는 얘기 주변에서 많이 들었어요~ ㅋㅋㅋㅋㅋ 제 어머님은 남자는 늙어 죽을 때 까지 아이라고 잘 다독이라고 하셨네욬ㅋㅋㅋㅋ'], ['제 친구 할머님은 할아버님 임종순간에 철드셨다고 남자는 죽기직전에 철드는거라시드라고요 ㅋㅋㅋ 그땐 웃었는데 지금은 진짠가 싶어요 ㅋㅋㅋ'], ['그렇게 생각하면 조금은 편해지지 않을까요...?? ㅋㅋㅋ'], ['ㅋㅋㅋㅋ 저희는 서로 애 둘 키운다고 그래욬ㅋㅋ 저는 남편 큰아들이라고 하고 남편은 저 큰딸이라고 하고 ㅋㅋ전에 김미경씨 강읜가? 부부는 서로 키워주는 사람이라는데 그말이 딱인거같아요 ㅋㅋ'], ['건조기 강추해요~힘들땐 도움되는 기계들이 필요하더라구요^^'], ['맞아요 ㅜ ㅜ 육아도 살림도 템빨 ㅜ ㅜ 돈으로 행복을 사는거같아요'], ['그 마음 알것같아요 하루가 금방가고 나만의 시간 잠시라도 갖고싶은데 그조차 힘들죠 ㅠㅠ\n힘내세요 !!'], ['다들 그런데 저만 징징댔네요 ㅜ ㅜ 잘 읽던 책도 못읽고 운동도 못다니고... 언능 키워놓고 회사 복귀하고 싶어요 ㅋㅋㅋ 안받아주겠지만 ㅋㅋ'], ['아니예요 얘기하세요\n얘기를해야 그나마 기분이 풀려요 ㅎ'], ['맞아요 떠드니 좋네요 ㅋㅋ 비록 폰으로 떠드는거지만 ㅋㅋ친구들이 미혼이 많아서 징징대기가 미안하드라고요'], ['얘기가 잘때만 하려면 조급하고 스트레스받는것 같아요ㅜ조금커서 혼자놀때면 괜찮아지실꺼에요^^'], ['맞아요 ㅋㅋ 조급해서 맨날 망쳐요 ㅋㅋㅋ 혼자 노는데 엄마가 안보이면 징징대고 저도 미안해서...ㅋㅋ 어린이집 가는 날을 고대하고있어요 ㅋㅋ'], ['ㅋㅋ 근데 어린이집가면 밖에 놀러다니거나 외출하느라 미싱 또 많이 못한다는;;;전 어린이집가면 진짜 많이 만들꺼야하고선 그랬네요'], ['잉 또 그런가요 ㅋㅋㅋㅋ 많이 할수있을지 알았는데 ㅋㅋㅋ 지금 짬내서 열심히 해야겠어요 ㅋ'], ['주부는... 어쩔수 없는 같은 생활패턴에 익숙해지다가도 지치니.. 참.. 공감가는 글이네요~ 화이팅입니다~'], ['맞아요 어느날 갑자기 문뜩문뜩 쳐지드라고요 ㅜ ㅜ 같이 화이팅해요!'], ['아고...살림과 육아가 쉽지 않더라고요...우리 힘내요!!'], ['그쵸 ㅋㅋ 그냥 일하는게 편한데...ㅋㅋ 마침 어제 전 회사서 전화도 와서 더 싱숭생승했어요 ㅋㅋㅋ 힘내요 ㅠㅜ'], ['어맛!!능력자~~ 마음이..가는 어떤 결정이든 최선의 결정이길요~~'], ['후임이 그만뒀드라고요 ㅋㅋ이사해서 멀어서 못가는데 ㅋㅋ 일하고 싶어서...ㅋㅋㅋ'], ['아..고민 되셨겠네요..\n거리가 멀면 힘들죠..'], ['아예 못가는거라 고민도 안했지만..ㅋㅋㅋ 아깝더라고요 ㅋㅋ 지금 사는데서 재취직하면 반토막 이상 줄어서..ㅋㅋ'], ['아..재취직하면..그리 되는거..넘 속상한데요..ㅜㅜ'], ['시골이고 일자리도 없고...애기엄마 안좋아하니...ㅋㅋ어쩔수없지만 속상해요..ㅋㅋ'], ['어느 시골이 아녀도 비슷할거 같아요..힘내세요..'], ['넹.. 나중에 자격증이라도 하나 따서 도전해야겠어요 ㅜ'], ['자격증.  좋지요~~도전~~ 홧팅입니다^-^']]
    
    1294
    댁에 이런거? 하나쯤은 다 있으시죠? 결혼하고 제일 좋은건 집안일 도와주는것도 좋지만전 심부름 시킬때가 젤 좋아요~^^퇴근할때 뭐 사와라내려가서 아이스크림 사와라이거 차에 갖다놔라저거 친정 갖다줘라 등등...저희언니는 신랑은 도움안되고 애들 키워놓으니 심부름 잘해서 좋다하더라구요~ㅋㅋ다들 이런거? 하나는 두고 사시는거 맞죠?ㅋㅋ
    
    [['결혼 잘하셨네용 울집에도 하나 있긴한데 그집거가 더 좋은것 같네요 ㅋ'], ['그집거에서 빵터졌어요ㅋㅋ 근데 결혼잘했다는 말씀은 섣부른 판단이십니다ㅜㅜ'], ['ㅋㅋㅋㅋㅋ 저도 공감요!!'], ['닉넴이 댓글을 대변하는듯 하네요~^^;ㅋㅋ'], ['있으면 모하나요..한번더 제가 손을봐야하네요.'], ['전 전업이라 제가 대부분 하긴하는데.. 가끔씩 신랑이 해도 저보다 더 깔끔떨어서 다행히 두번손은 안가요~^^'], ["집이 아주 반짝반짝하네요ㅠ 전 제가 집에서 '이런 거' 같아요ㅋ 어제 밤 11시에 남편이 저보고 편의점 가서 초코하드 사오라고 하더라고요ㅠ 심부름? 제가 시키면 90프로 확률로 거절이라 시키지도 않아요 청소? 제가 밤새서 시부모님 생신상 차린 날이나 돼야 자발적으로 해요 쓰다보니 비참하네요ㅜ"], ['저도 제가 훨씬 더 많이하긴해요~^^ 신혼때 늦은밤 저한테 담배심부름 시킨적도 있고 소파에있는데 발로밀어서 물가져오라고도 하구요.. 하...ㅜㅜ'], ['없는 집도 있어요 부러워요 어흑ㅜㅜ'], ['그럼 저희언니처럼 아이를 키워서 하시는게 빠르실수도...^^;'], ['잘하는듯하다가 꼭 사고쳐놔서 혈압을 더 올려요..아..술만 정확히 잘 사와요ㅜㅜ'], ['아.. 저희집껀 술만 잘안사와요.. 늘 제 성에 안차게 사오더라구요~^^;'], ['인간은있지만 입력오류나서 제가 다해요\n차에 뭐 갖다놓기 가져오기 뭐사오기 택배받으러 경비실다녀오기 등등...\n갔다오라 명령입력해도 오류나서 누워만있네요 ㅋㅋㅋ\n기대안해요 ㅋㅋㅋ'], ['입력오류..ㅋㅋ 저희집것?도 본인이 신경안쓰이는건 잘안하려드는 문제가...  커텐떼고 블라인드 달으라했는데 이건 겨울에 하려는건지ㅡㅡㅋ'], ['뭐하시는건가요?ㅋ 베란다 정리하시나요?ㅋㅋ'], ['옥수수 한자루 까는중입니돠 ㅋㅋㅋ'], ['저희신랑 심부름시키면...잘해요..\n하지만 같이가재요 임산부인데도요 얄짤없이 같이가주길원해요ㅋㅋㅋㅋ 애긴가요...아직도 부끄럽나바요 \n이럴때만 껌딱지네요 휴'], ['데이트가 하고싶으신 모양입니다~^^'], ['아~아직  못  만들었는데..'], ['님도 남편분보다는 아이를 키워서 시키는걸 권해드려요~^^;'], ['저희집꺼는 저런기능 없는데. . 좋아보이네요. 가격대랑 정보 좀 요~'], ['근데 고장난데가 많아서 매일  징징거려요..ㅜㅜ 너무 저려미라 권해드리진 않습니다ㅋㅋ'], ['ㅋㅋ둘이나 있지만 만만찮게   잔소리들이 많지요 ㅋㅋ'], ['잔소리 안하는 기능은 이집에도 없습니다ㅋㅋ'], ['있는데 불량품이에요. 교환 환불안된다네요 ㅠㅠ'], ['저희집껀 진짜 불량이에요ㅜㅜ 친정에서 성한데 찾는게 더 빠르겠다고 할 정도네요~'], ['저희 아빠가 저런거(?)라서 남잔 다 저런거(?)인줄 알고 결혼했는데 대박 뒤통수 맞았어요'], ['전 기능 확인하고 데려왔습니다ㅋㅋ'], ['저도  있어요..알아서 못하니 서운해 하지말고 시키는건 다하겠다고 해서 열심히 시켜요..요즘 18년 되니 못들은척 할때도 있어요.'], ['그쯤되면 저희집것?도 고장나지 않을까 싶어요 18년이면 뽕 뽑으셨네요ㅋㅋ'], ['울집은 음쓰나 쓰레기버릴때만  사용하네요'], ['저도 있어요ㆍ 정확하게 지시해달라고 요구하고 음식을 못하니까 집밥 백선생보면서 스스로 성능 업그레이드 해요ㆍ 최근거라 인공지능이거든요ㆍ 스스로 상황판단해서 제가 저기압이면 안마도 해주고요ㅋㅋㅋ'], ['진심 부럽네요~~ 저희집엔 연식이 너무 오래되서 스티커붙여도 수거해가지 않는게 있는데ㅠㅠ'], ['인공지능 기능 넘나 탐나네요~ 저희는 제가 화나도 모른다는게 단점입니다ㅜㅜ'], ['저도 있는데 요샌 영리해져서\n안하고 저한테 시키려해요 ㅠ'], ['영리해진거면.. 스스로는 업그레이드 인건가요?ㅋㅋ'], ['음~ 저희집꺼는 오작동이 많아요 대답만 하고 실천은... 반품도 안되고 길들여야겠죠?'], ['일단 접수는 하는거 같으니 더 길들여보심이 좋겠어요~^^'], ['이런거^^저희집도 오늘 종일 일해주시고 미안해서 친구들이랑 술먹고오라고 인심써줬어요^^잽싸게 약속잡고 나가더라구요ㅋㅋㅋ'], ['저희집도 유독 열심히 하는날은 꼭.. 나 게임한판만 해도돼? 이말로 마무리 짓더라구요~ㅋㅋ'], ['우와 멋지네요 우리집은 시키는것만 하는데 그냥 제가해버려요ㅋㅋㅋㅋㅋㅋㅋ'], ['시켜서 하는것만도 넘 훌륭한거 아닌가요~ㅋㅋ 한다고할때 열심히 시킵시다요ㅋㅋ'], ['할말 디게 많은데.... 접고갑니다. 대신 아들교육에 힘쓰고 있어요. 음쓰도 얼마나 잘버려주나 몰라요....ㅎㅎ'], ['무슨말씀하실지 넘나 궁금하네요~ㅋㅋ'], ['저희집꺼도 하긴해요 근데 맘에 안들어서 문제네욪ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['우리 하긴하는거에 큰 의의를 두도록해요~^^'], ['저도 있는데 인공지능은 안되고 입력한것만 하네요'], ['저희집꺼?엔 인공지능 기능은 넘사벽입니다ㅋㅋ'], ['저희집꺼라 비슷해요ㅋㅋㅋㅋ 청각기능이 떨어져요ㅋㅋ'], ['저희집도 청각이 그닥 좋은편은 아닌것같아요..^^;ㅜㅜ'], ['제것도 음성인식형이라 하나하나 오다를 내려줘야 해요. . .게다가 음란마귀 씌인듯 해서 퇴마 as신청해야 할거 같아요.'], ['전 음란마귀 기능 좋아하는데...^^; ㅋㅋㅋㅋㅋ'], ['울집에도있기는하나 어찌나하면서 말이많은지~~~^~^\n귀가아프지만 귀막고 열심히 시키네요~~~^^'], ['다행인건지.. 저희집엔 음성기능이 잘돼있진 않아요~ㅋㅋ'], ['저희 집엔 없어요.혹시 판매처 알 수있을까요? 사진보니 울집에도 하나 들여놓고 싶은데  안방에서 코 드르릉하는 ㅅㄹㄴ땜에 그림의 떡이긴 하네요. 아. 부러워라.ㅜㅜ'], ['잠깐 저러고선 게임아니면 자는게 일이에요.. 그건 비슷한가 봅니다~^^'], ['전 주말에  요리거의안해요\n요리를잘하는서방덕분에요 ㅋㅋ'], ['첨엔 요리 잘하더니.. 요즘 안시켜버릇하니 기능이 점점 떨어지는듯해요ㅜㅜ'], ['맞아요 ㅋ저희집도 있는데 원래있던집(시댁)서 자꾸 다시 탐내는것같아요 ㅋ'], ['어디서든 탐내는 기능을 가지셨나보네요~^^ 궁금합니다ㅋㅋ'], ['부럽네요 < 저희집엔 이런거(?)없어유 ㅠㅠ ㅋㅋㅋㅋ'], ['깡유맘미님도 아이에게 기대해보심이..^^;ㅋㅋ'], ['ㅋㅋㅋㅋㅋ 글 너무 재밌게 쓰셨어요 ㅋㅋㅋ 그나저나 집이 넘 깔끔하네요~~ 저희집은 왜 맨날 치워도 더러울까요 ㅋㅋㅋ ㅠㅠ'], ['저희집도 엉망이에요~ 치워도치워도 티안나는게 내집인듯요ㅋㅋ 남의집은 다 깨끗해보이고ㅋㅋ'], ['없어요.없어~ㅜㅜ\n구하고싶은데..집에있는거 치우는방법을 몰라서..그냥 두고있어요;;;;;'], ['치우는 방법은 쭉 모르고 사시는게..ㅋㅋ'], ['헉 이런거라니ㅋㅋㅋ\n이런거가 신랑이죠..ㅎㅎ\n잘 도와주시니 멋진데요~^^\n제목만 보고 이런거라고 해서 전 뭐 기계인줄 알았어요ㅎㅎ'], ['ㅋㅋㅋ지금은 또 약국보냈습니다~ 어제부터 편두통이...ㅜㅜ'], ['그런건 어디서 사는건가요?저도 사고 싶네요'], ['저는 길에서 주워왔어요.. 저희는 길에서 처음 만난지라...ㅋㅋㅋ'], ['마트심부름? 그것도 \n명령조 넘 싫어하네요\n부탁조로 이야기해야 \n사다주는...\n집에선 허리 굽힐줄 모르는 \n우린껀 \n그나마 마트심부름으로 만족합니다!!\n'], ['마트심부름 훌륭해요~ 집에있을땐 현관문 나가기도 넘 귀찮잖아요ㅋㅋ 요즘같은 날씨엔 더더욱~'], ['ㅋㅋㅋ빵터졌어요 가위바위보 맨날 지는 저한텐 무용지물이네요..ㅜㅜ'], ['이런거?? 먼가한참을 봤어요ㅋㅋ 남편님이셨군요ㅎㅎㅎ'], ['아프다고 징징대는거 빼고는 아직 쓸만합니다~^^;']]
    
    1350
    집안일 분담 어찌하시나용? 안녕하세용~11월 결혼이지만 8월말부터 신혼집에 같이 있어용!!저 완전 생각지도 못했는데예랑이가 집안일을 다해줘요ㅋㅋㅋ수시로 돌돌이 테이프 돌려가며 청소하고설거지거리는 생기자마자 바로 하더라구요전 쫌 미뤄놓는 스타일인데-_-;;ㅋㅋㅋ"식탁만 좀 치워줘" 그러는데장난으로 "응 내일 치울께" 그러니까 본인이 다 치우네요ㅋ근데 짜증 내는게 아니라장난스럽게 볼 한번 꼬집고 혼자 다해용혼잣말로 청소기 밀어야겠다 그러고 있구요ㅋ대신 세탁기 돌리는거랑건조기 돌리는건 거의 제가해용건조기 다 돌아간건 같이 정리하구요~제가 스케줄 근무이고 예랑이가 재택근무라 절 많이 배려해주는거 같아용서프라이즈 이벤트나 비오는날 마중 나오거나 하는 로맨틱세포는 진짜 꽝인데-_-^성품이 참 착하네요 예랑이가ㅎㅎ
    
    [['예랑님 넘 좋으시네요 ㅎㅎ 저는 각자 잘하는걸로 하려구용'], ['저는 일안하는백수인데..\n요리+세탁기+건조기만...ㅋㅋㅋ\n세탁기랑건조기는 사실 기계라ㅋㅋㅋㅋㅋㅋ \n버튼만누르는건뎅 ㅎㅎ\n요리는제가워낙하는걸 좋아해서용\n신랑은\n화장실청소,집청소,음식물버리기,베란다청소담당이에요 ㅎㅎ거의다하는...느낌'], ['예랑님이 착하시네여 ㅎㅎ저희예랑도 제가안하면 자기가히는데 이제 같이살면잘 분담해서 해야겠죠?'], ['저희는 거의 반반 하는거 같아요 설거지랑 청소는 남편이 많이 해주고 저는 분리수거 화장실청소 세탁기 많이 해요'], ['진짜 좋으시겠어요 먼저 나서서 정리하구 청소도 해주시다닝 ㅜㅜ ㅎㅎㅎ'], ['대박이네용 ! 저는 신랑 시키기 싫어서 거의 제가 해요~ 서재 방만 신랑 담당이고요. 음쓰 일쓰 분리수거도 신랑. 근데 둘다 게을러서 청소, 빨래 일주일에 한 두번하고요. 집안 일이 쌓여도 서로 절대 잔소리 안해용ㅋㅋ'], ['저희는 미리 같이 살고잇는데 제가 아직 놀고잇어서 혼자 거의다해요ㅠㅠ 아빠가 하도 깔끔하셔서 그걸 닮앗는지 집에선 저도 잘 안치웠는데 신혼집와서는 먼지 있는꼴을 못보겟네용.. 설거지도 쌓이는거 싫구.. 예랑이는 반대로 청소 그만해도 된다구 장판 다닳겟다그래욬ㅋㅋㅋㅋ하아...'], ['진국인 신랑이시네요 ㅎㅎ 저희도 맞벌이라 결혼하면 반반씩 하기로 했는데 딱히 구분짓진않았어요 그냥 각자 손 닿는대로 하기로 ㅎㅎ 그러면 여자가 결국 더 많이 하게된다고는 하는데 평소 남친 성격을 봐선 오히려 더 많이 해줄거같아서 걱정 안하고 있네요 ㅎ'], ['구분 지어놓진 않았구 앞에 있는 사람이 해요 ㅋ 예랑님 멋지신데용'], ['우와 볼 한번 꼬집고 혼잣말하시는 그 모습 자체가 로맨틱인데용.....*_* 너무 부럽습니당....ㅋㅋㅋ'], ['저는 제가 밥하면 신랑이 설거지하고 ㅋㅋ 빨래 돌려놓고 같이 빨래 개고 정리하고 거의 같이하는거같아용'], ['부지런하시네용 예랑님이 ㅋㅋ 볼한번꼬집고 혼자열심히 하시는 예랑님 완전멋져용'], ['예랑이가 다할때도 있고 제가 할때도 있어요. 같이 할때도 있구요. 집안일은 돕는게 아니라 같이 하는거죠!! 옛날시대 남자가 아닌게 다행이예요. 예랑이분이 배려가 너무 좋으시네요.'], ['예랑님 너무 멋찌세요~전 벌써부터 집안일로 싸움나는거 걱정인데...제 예랑이도 그랬으면...'], ['볼 꼬집고 집안일 마저 하시는 게 이미 엄청난 로맨틱세포 아닌가요ㅠㅠ 설레고 갑니다...'], ['따로 분담이랄거 없이 저희 신랑은 다 같이해주고있어요!'], ['저는 예랑이가 바빠서 월~목은 제가 요리하구 금~일은 예랑이가 요리해요! 세탁, 청소는 같이하는거 같아용:)'], ['저도 요리랑 세탁은 제가 할거같은데 청소는 넘나 싫어해서 청소기 돌리고 물걸레 돌리는건 남편이 하기로 했어요!! 설거지도요!!'], ['예랑님이 배려 많이 해주시는 거 같아요. ㅎㅎㅎ 서로 잘 하는 걸 하면 좋죠~'], ['전 빨래 오빠는 요리로 분담했어요ㅎ'], ['저희도 저는 밥해주는 사람이고 예랑인 얻어먹는입장이라 설거지는 예랑이 ,화장실청소 예랑이,청소기도 예랑이..ㅋㅋㅋ저희도 예랑이가 프리랜서라좋으네용ㅎㅎ'], ['너무 좋은 분 만나셨네요!! 정말 좋으실 것 같아요!!'], ['저흰 둘다 수시로 청소해서.. ㅎㅎ \n요리 설거지는 거의 제가 하고, \n방바닥 청소기 수시로 밀고 다니는건 제가ㅋㅋ\n빨래 건조기 분리수거 .. 이불정리 이런건 오빠가 해요 ㅎㅎ'], ['와 예랑님 너무 자상하시고 좋으신것 같아요! 오래오래 행복하셔용!!'], ['아구 저는 제가 주로하고 시키는 타입인데 시키는거 별로 안좋아해서 제거 우선 다해요 ㅠ힘둘어용 ㅋㅋ'], ['저는거의제가하는거같은데 ㅜㅜ 그중에서좀힘든걸레질이나분리수거이런거는꼭 신랑이해주고 나머지는제가하면서 시켜요ㅎㅎ'], ['ㅎㅎ배려심깊은 예랑님이시네요!! 저희도 제가 더 느긋한 편이라 신랑이 많이 하는 거 같아요 ㅋㅋ'], ['시키는거는 다 해요 ㅎ 전 주로 정리만 하고 나머진 남친이 다..'], ['깔끔하신 분인가봐요 ㅋㅋ 부러워요 원래청소는 못견디는 사람이 먼저하게되잖아요ㅠ'], ['저희 신랑도 깔끔한편이라 저보다 빨리 많이 움직인답니다~^^'], ['저희는 벌써고민이에요ㅠㅠ 쓰레기분리수거랑 화장실청소는 남자친구꺼될듯해용 ㅋㅋ'], ['저흰 맞벌이고 딱 나눠 잇지않고 먼저 퇴근한사람이 자연스레 하는거같아요!! 남일이라고 미루지만 않으면 될듯해요!!'], ['좋으네용ㅎㅎ 행복하게 사세요'], ['우오ㅎㅎㅎ 저희는 각자 잘하는걸해요ㅎㅎ 같이하니까 금방금방 하더라구욥ㅎㅎ'], ['ㅎㅎㅎ 제목보고 질문인줄 알았는데!!! 자랑 한방 먹고 갑니당 ㅎㅎ 부러워요~~'], ['완전 부러워요~ 저는 반반 나눠야할 것 같아요ㅎ'], ['제가 요리하고\n남친이 설거지하고 뺄래널기 개기\n청소는 제가 셋팅하고 샤오미가 해줘요 ㅋㅋ'], ['예랑님이 엄청 좋으신데요? ㅎㅎ 저는 이제 시작인데.. 처음부터 싸울거같은 느낌이 ㅜㅜㅜ'], ['ㅋㅋㅋ그렇게 치워주는거 사소한것도 감동이죠! 😂전 청소는 제가 거의하고 설거지랑 빨래는 신랑이해줘용!ㅋㅋㅋ'], ['신혼초엔 제가 더 많이했어요 맞벌인데 제가 더 퇴근을 일찍하거든요 남편도 안하는건 아니었는데 남편이 아는거 보다 집안일이라는게 더 많잖아요ㅠㅠ 그래서 재가 하다 지쳐서 울고 싸우고나서는 둘이 거의 반반하는거 같아요 임신하니까 남편이 더 많이 해주고 요리만 아예 못해서 제가 해요'], ['성품이 정말 좋으시네요! 짜증낼법도 한데 ㅎㅎ'], ['구분안하고 그냥 더 여유있는사람이 하기로했어요^^'], ['집에있게 되는 사람이 아무래도 더 주도적으로 집안일을 하게되는거 같긴해요ㅎㅎ 저도 신랑이 집안일 굉장히 많이 하는편인데 제가 집에있다보니 사소한것들도 신랑 모르게 하는일이 엄청 많아요ㅎㅎ'], ['와~ 진짜 착하세용 ㅎㅎㅎ 저는 시켜야해여 ㅜㅜ'], ['우와 다해주시면 넘 좋으시겠어요 ㅠㅠㅎㅎ 저희는 청소기돌리면 한사람은 물걸레 돌리고 이런식으로 항상 나눠서합니당 ㅎㅎㅎ'], ['저희는 밥만 제가해요 ㅋㅋ 설거지도 할때두있구요'], ['저희도 .. 남친이 틈만 나면 청소기를 돌려요 ㅋㅋㅋㅋㅋㅋㅋ 바닥 닦는 건 저의 몫.. 설거지도 남친이 하고.... 저는 주로 세탁기 돌리고 요리하는 정도 합니다 ㅎㅎ'], ['저희는 설거지만 남친이 하고 나머지는 제가ㅜ 아! 화장실청도까지는 해줘요~ 근데 빨래는 못하겠데요 ㅋㅋ'], ['집안일은 정말 성향차이 인듯해요! ㅋ 좋은습관을 가진 신랑님이시네요!'], ['딱 정해놓진 않았는데 시간 많은사람이 더하게되는것같아요 ㅋㅋㅋ그래서 거의제가.....'], ['맞벌이 때는 반반씩 했는데 전업되고 나서는 거의 제가해요 ㅠㅠ'], ['ㅎㅎ 저희는 시간 되는 사람이 보이면 해요^^ 그래서 서로 더 많이 할 때도 있고 비슷하게도 하고 그래요'], ['저도 결혼전에는 몰랐는데 신랑이 집안일 의외로 많이 도와줘서 참 좋더라구요ㅎㅎ 의외의 매력에 푹빠져 살고있네용ㅎㅎ'], ['저는 제가 주로 많이 하고 신랑한테는 시키는 편이에요~ 본인이 자발적으로는 잘 안하니까 그냥 시켜요ㅎㅎ'], ['다들 의외의 모습들을 보셨군요 ㅋㅋㅋ 저희도 정해놓은건 아닌데.. 예랑이가 이렇게 깔끔한 스타일인줄 몰랐어요.. 분명 본가에서는 손도 까딱 안했다는데;; 제가 잔소리를 들을 줄이야...ㅋㅋㅋ'], ['신랑 밥 빨래 저 청소 설거짘ㅋㅋㅋㅋ'], ['저는 신랑 성격이 더 깔끔해서 신랑이 80%하고 있습니다ㅋㅋ'], ['저도 예랑이가 워낙 깔끔떠는 스타일이라 웬만한 청소는 다 해주고 있어요ㅠㅠㅋㅋㅋㅋ'], ['저는 반대로 제가 막 치우는성격이라.. 저만 고생하고있어요 ㅎㅎ'], ['예랑님 진짜 좋네요 전 제가 다 치워요.....예랑이가 밥먹은거 설거지랑 빨래 하는데 설거지 해놓은거 제대로 안닦여서 제가 다시 ㅠㅠ'], ['우와아~!!! 신랑님이 넘넘 온화하신 성격이신것 같아요 ♥♥신부님 참 좋으시겠어요 ♥♥ 집안일 함께하는 거라고 생각했는데 이렇게 상대방에게 닥달(?)하지 않고 해주시는 모습 참 멋있으신거같아요 ㅠㅠ'], ['저는 백수라서 주로 제가 하고 신랑은 화장실청소,음쓰버리기 정도? 매번 본인이 하겠다고 놔두라는데 제가 양에 안차서 두번 손대느니 제가 ㅜㅜ 그것도 바로바로 안하니 성질급한 제가ㅜㅜ 이럼 안될듯요 할때까지 기다려야 부려먹을수있을거같아여ㅜ'], ['우와ㅠㅠ 진짜 좋네요 예랑님 저희남편도 이것저것 다 잘해주긴하는데 약간 쌓아놓고 한번에하는경우가많아서ㅋㅋㅋㅋ 저희집이랑좀다르네용'], ['저도 집안일은 다 신랑이 해요~^^;; ㅋㅋㅋ\n제가 유일하게 하는건..요리인데..요리를 좋아해서 딱! 요리만 해요,ㅋㅋㅋ\n치우는거랑 청소랑 빨래랑 다 신랑이 알아서 척척 해줘서,ㅋㅋㅋㅋㅋㅋ\n그래서 친정에서 자꾸 저한테만 뭐라함요ㅠ,ㅠ'], ['아 자랑글이었네요~ㅎ 제목만 보면 불만글인거 같았는데 ㅎㅎ 넘 착한 남편을 만나셨네요~ 재택근무하시면 시간여유가 되시니 더 잘해주시나봐요~~'], ['저희는 제가 설거지 담당이고, 예랑이는 쓸고닦고 담당이에요'], ['화장실 2개와 주방은 신랑담당이고 나머지는 제 담당으로 정했어용'], ['우와ㅜ너무 좋으시겠어요..ㅜ 저도 집들어가기전에 분배하기로 하고 정했는데.. 같이 사니깐.. 오로지 거의 제 몫이에요.. 너무 힘들어요ㅜ'], ['예랑님께서 청소를 강요하시면 싸울문제인데 .. 완전 좋은분이세요 !! 제 예랑이도 그랬으면 좋겠네요 !!'], ['꺅 부러워요 ㅋㅋㅋㅋ 저는 지금 예랑님이 하는 행동을 제가 하고 있는데 ㅋㅋ 잘하다가 너무 안 도와주는 거 같으면 짜증내요 신랑한테 ㅎㅎ 그러니 예신님도 쪼끔씩만 도와주시면 엄청 잘하실거에요 편하게 사실 수 있으실거에요!ㅋㅋㅋ'], ['ㅋㅋ 결혼 잘 하셨네용^^ 저희두 요리만 제가 하구 나머지는 다 예랑이 하기로 했어요 \n알콩달콩 행복하게 잘 사시길요^^'], ['예랑님이 넘 착하시네요 ㅎㅎ 부러워요~^^ 저희는 가능한 반반 하려구요 ㅋㅋ'], ['마지막줄 넘 공감이예요 ㅋㅋㅋ 서프라이즈가 뭔지도 모르고, 마중 나오는 것도 부탁해야 나와주는... ㅎ..\n그래도 집안일은 잘 도와줘서 고맙지요 ^^'], ['집안일 좋아하는 신랑이라니!! 저는 제가 못참고 다 하는 스타일인데ㅋㅋ부러워요'], ['너무 잘도와주시네요~~~저희신랑도 처음엔 잘못하더니 이제는 말안해도 알아서 잘도와줘서 고맙게생각하고 있어요~^^'], ['결혼해서 살다보니 성품이 진짜 중요한거 같아요 ㅎㅎ\n맞벌이 하면 집안일로 많이 싸우고 신경전 벌이는데 신부님 부부처럼 서로 배려해주니 너무 좋아 보여요^^'], ['저희는 한명이 밥하면 다른 한명이 설거지하고 뒷정리해요~ 청소나 빨래는 예랑이가 더 많이 하는거 같아요 ㅋㅋ'], ['예랑님이 엄청 착하시네요! ㅎㅎ 보통 더 정리잘하는 사람이 스트레스 받긴 하죠 ㅎ 저는 예랑님 처럼 제가 더 먼저 치우는 편이에요'], ['진짜 자상하시네요~ 저는 제가 퇴근이 빨라서 거이 다해요ㅠㅠㅋㅋ'], ['엄청 자상하고 좋으신분이네요~~알콩달콩 좋아보여요~'], ['집안일 분담 현명하게 잘 하셔서 알콩달콩 행복하시길 바랄게요~~ㅎ'], ['저희도 각각 알아서해요!!누구하나가 잘 하면 결국 따라가는 것 같아요! 행복하세유'], ['저희도 맞벌이인데 저는 그래도 정시퇴근하고,\n남편은 개인사업자라 바쁜시기는 엄청 바쁜데ㅠㅠ\n평소엔 제가 요리하면 남편이설거지, 빨래는 번갈아 뭐이런식으로 했었는데 요샌 보이는대로 상대적으로 시간 여유있는 사람이해요 ㅎㅎ'], ['오 진짜 예랑님 착하시네여ㅎㅎ가정적인남자좋아여'], ['가정적인남자 진짜 좋아요!!'], ['저는 결혼하면서 일쉬고있는데 아침 간단하게 신랑이 차려주고 빨래돌리면같이널어주고 음쓰버려주고 제가 저녁차려주면 신랑이 싹설거지까지 다해줘요ㅋㅋ등등 너무 많지만 일단 빨래감이며 물건들이며 여기저기 널부러놓지 않고 그때그때 제자리에 두는게 최고인거같아요 키키 결혼만세~!'], ['역할분담을 잘해야겠어요 사소한걸로 싸우지 않게요 ㅋㅋ'], ['저는 예랑이가 해줬으면 좋겠는 부분들을 정확히 말해줘요~ 사실 제가 해야 성이 차는 성격이라 집안일은 그냥 제가 하는 게 속이 편할 때도 있어요ㅋㅋ'], ['와 비오는날 마중좀 안나오면 어떤가요. \n마중만 자기 차 타고 쉽게 나오는 대신 주말부부인데 집안일은 같이있을 때 하는거라고 하는 남편도 있네요 ㅋ'], ['저는남친보구다해달라고햇어요ㅜㅜ전지금도못하기때문에ㅜㅜ대신쓰레기버리는건제가한다구햇어요ㅜ'], ['진짜 좋으시겠어요.. 저는 제가 많이 하게 될 것 같아요ㅠ.ㅜ 가정적인남자 부러워여 으잌ㅋㅋㅋ'], ['저는 제가 훨씬 많이해요ㅠ 그래도 화장실 청소랑 쓰레기 버리는건 신랑 전담으로 시키구 있어요 ㅎㅎ'], ['센스가 넘치시네요 ㅎㅎㅎ 잘 도와주는거 같구'], ['볼한번 꼬집고 본인이 하신다는 것...넘나 스윗해여ㅠ!!ㅎㅎㅎ저희는 제가 백수라 평일엔 제가 더 하고 주말엔 제가 청소기 돌리면 예랑이가 물걸레 청소기 돌려주고 분리수거 하고 제가빨래널면 예랑이가 마른거 걷어주고 그래용ㅋㅋ'], ['전 일안하고있으니 거의 제가해요..주말엔 설거지조금 도와주고;'], ['좋네여 !!!!! 근데 제 남편 한 두달..? 그러다가 폭발해서 싸우게되었다는 ^^;;;;;;;; 저도 좀 몰아서 주말에 하는 스타일인데 ㅠㅠㅠㅠㅠㅠㅠㅠㅠ 요즘 많이 노력하고 있어요 ㅠㅠㅠㅠㅠ'], ['로맨틱세포 꽝 ㅋㅋㅋㅋ 그래도 엄청 자상하시네요'], ['와우 남편 잘 두셨네요! 제 남편도 설거지잘해줘서 이뻐용ㅎㅎㅎ'], ['이것저것 칼로 잰듯 분담하기 ㄴ그래서 서로가 해요 그냥 ㅎ'], ['신혼의 달달함 그 자체네요ㅋ']]
    
    1457
    집안일 중 젤 힘든게 뭐에요? 전....화장실 청소.. 넘 힘들고 젤 하기시러요ㅠㅜ욕조 청소.. 하고나면 온몸의 피가 거꾸로 올라온거 같구타일 바닥.. 사이사이 낀 물때 벗겨내다 허리쑤시고 손목아파서 넘넘 시러요ㅠㅜ그러다 요즘엔 건식으로 써보자. . 해서샤워커튼해서 욕조안에서 샤워하고바닥에 러그나 매트깔아서 쓰고 있는데바닥에 먼지는 생기니 매번 매트 들어내고바닥 솔로 밀고 매트에 낀 물때같은거 제거해야하고크게 힘이 덜드는거 같지가 않네요.세면대는 또 며칠만지나면비누자국,치약자국.. 휴우이건 뭐. . 매일 아침 씻고나서 세면대 닦는것도정말 큰일..사실 매일 하지도 못해요.화장실 어케 청소하세요?화장실 청소 잇템이나, 덜 더럽게 사용하는 팁?뭐 이런거 없으신가요?
    
    [['청소요 계속돌아다녀야해서요ㅠ'], ['청소는.. 다이슨 장만 이후 확실히 덜 힘들더라구요 저는요... 걸레질은.. 안하구요ㅋㅋㅋㅋ;;;'], ['아.. 요리. . 그러고보니.. 전..거의 포기요ㅎㅎ 걍 반찬사다먹어요;;;;; 이렇게보니 집안일 전 영역 낙제인듯 하네요;;;'], ['222제발안하면안되냐고는\n아닌데..시간투자대비 맛이 그닥.'], ['다림질이요. 친정엄마가 남편 처음 보자마자 나도 셔츠는 세탁소에 맡기네. 하셨죠 ㅋ'], ['저도 매일 새 셔츠 입고나가는 회사원 남편을 둬서.. 흠 근데 전 이거는 적응완료했어요^^ 결혼 초에는 도대체 내가 남의 셔츠를 왜 다려줘야하는지 도대체가 이해가 안가서 많이 싸웠었죠ㅎㅎ'], ['저도 청소 및 정돈이요.. 아까 다 치웠더니 저녁설거지하는 사이에 5살 14갤 딸램들 열심히 다 꺼내놨네요.. ㅠㅠ'], ['휴. . 전 소리질러서라도 스스로 치우게 합니다......;; 그것까지 하면 정말 열뻗쳐요ㅠ'], ['전 부엌청소요 ㅜㅜ \n정말 손도 대기 싫어요 ㅜㅜ'], ['싱크대 물때, 가스렌지.. . 이것도 아주 미치죠... 아 살림 넘 힘들다..'], ['저는 물걸레요\n오토싱이던 스팀청소기던\n그냥 그 물적신걸레로 거실닦는게 참말로 싫어요'], ['전. . 포기영역이에요ㅠ 다이슨으로 미는 것만으로 걍 만족.. 걸레질도 하시구 대단하세요. .!'], ['저도 제가 안해요^^;;\n남편이 유일하게 해주는 집안일이에요ㅠ  제가 포기하니 본인이 그냥하네요ㅋㅋㅋ'], ['저도 걸레질 잘 안해요 ㅋㅋㅋㅋ 자주 먼지를 닦는걸로 만족? ㅋㅋㅋ 그래서 청소는 어렵지 않은가봐요 ㅋㅋㅋ'], ['저는 요리요... 사실 전 요리를 제법해요.. 하는 속도도 정말 빠르구요. (1시간이면 여러가지 뚝딱뚝딱...)\n맛도 뭐 먹은 사람들은 대부분 맛있다고들 해주세요. 저희 애들은 엄마가 해준게 최고라는 극찬으로 저를 늘 기쁘게하곤 하는데\n그럼에도 불구하고 저는 요리가 너무 싫은 사람이에요 ㅠㅠ \n돈절약위해, 또 오랜 자취생활로 (20년)... 특히 뉴욕에서 살았던 10년때문에 (한국음식 늘 사먹기가 힘들고 야식도 없는 세상이니 ㅋ)\n해먹는게 자연스럽고 익숙해지긴 했으나\n참 하기가 싫으네요... 제 취향이 아니에요. (이리 말하면 놀라는 분들 많아요.. 늘 해먹으니깐;;ㅋㅋ)\n매일 매일 외식하고싶은 사람이에요 ㅠㅠ 매일 외식하고 싶어서 돈 많이 벌고 싶어요 ㅋㅋㅋ\n전 차라리 설거지가 낫고, 욕실청소가 나아요 ㅋㅋ 청소는 잘하진 않는데 싫진 않더라구요~\n스트레스 받으면 까스렌지 기름쩔은 냄비 마구 닦고있네요 ㅋㅋ'], ['ㅋㅋ외식하고 싶어 돈많이 벌고싶단말.. 진짜 완전 공감 이에요^^'], ['화장실1.\n렌지후드2.\n냉장고3.\n국끍이기4.\n\nㅠㅠㅠㅠ'], ['렌지후드... 아 눈감고 싶지요.'], ['설거지가 제일하기싫어요ㅜㅜ'], ['그래도 그나마 힘은 덜 드는 종목인거 같아요. 정신적으로 귀찮긴해도 육체적으로는 그나마 괜찮은..'], ['저듀 화장실청소요ㅜㅜ전동 청소솔 사고픈뎅 넘비싸여..'], ['저도 계속 검색중인데 전동솔이 있더라구요.. 리얼 후기 듣고싶은데.. 화장실청소 잇템 댓글은 없네요ㅠ ㅠ'], ['설거지요 !!!'], ['바닥걸레질이랑 화장실청소여'], ['요리후 뒷정리 ㅠ'], ['화장실청소랑 청소기 돌리는 거는 신랑이 해요 전 요리하는 거랑 치우는 거요 ㅠ\n그래도 맛있게 먹어주니 ㅎㅎ 그거 보는 기분으로 하네요..'], ['저희집 사람들도 맛있게는 먹어주는데.. 그래도 요리는 왜케 재미없는지. . 남이 해주는 거 먹고파요. 결혼전엔 그게 얼마나 행복한건지 몰랐드랬죠..'], ['요리가 젤 힘들어요.\n레시피 없인 안되니까 더 힘든것 같아요.'], ['매직블럭이랑 욕실세제 같이해서 쓰면 반짝반짝 빛나요'], ['매직블럭은 아직 안써봤는데.. 근데 그것도 허리 잔뜩 구부리고 해야대죠? 힘을 덜쓰니 손목은 덜 아플 라나..'], ['싱크대 개수대에 버려지는 음식물쓰레기 치우는거요. 설겆이도 싫소 청소 정리 꺄..좋아하는거 찾는게..집안일 중 그나마 좋아하는 건 없네요 ㅠ.ㅠ'], ['글게요. 저도 쓰고 보니 질문이 좀..ㅋㅋㅋ ㅋ'], ['화장실 청소는 샤워하면서 매일 하는게 습관이 되었어요. 그러니 많이 더럽지않아서 금방 끝나니 그게 더 나은거 같아요.  내몸에 샤워기물 맞으면서 가만히 안있고 바닥,변기,세면대 솔질해요. 그리고 비누칠끝나면 또 욕조도 쓱 문지르고. 그럼 끝 ㅎㅎ  샤워시간이 좀 길어지는 단점이 있지만 청소를 힘들게 안해도 되니 좋아여'], ['제가 추구하는거에욧! 그게 젤 덜 힘든 방법같아요.. 근데 그게 습관베기는게 쉽지가 않아요 ㅠ ㅜ 혹시 청소솔이나 도구는 뭐쓰세요?'], ['바닥은 마트에서 산 스틱솔로, 세면대랑 욕조는 아크릴수세미로 해요. 요 수세미가 아주 잘닦이거든요. 수전도 요걸로 문지르기만 하면 반짝반짝'], ['아하 저도 요거 한번 써봐야겠어요. 휴 습관드리기가 먼저지만요ㅎ ㅎ'], ['전 청소를하면 스트레스가 풀리는 느낌이고 요리를 하면 스트레스가 쌓이는 느낌이에요 ㅡㅡ;;;'], ['헉! 제가 쓴 글인지 알았어요!! 저도요... 요리할땐 근육도 긴장되는 느낌이예요ㅜㅜ'], ['저는 걸레빠는거요. 너무 싫어요. 물걸레기계로 닦아서 닦는건 얼마든지 하겠는데 그 걸레 빠는게 넘 싫어요 ㅠㅠ 세탁기엔 아이옷도 빨고 그러니까 세탁기 돌리긴 싫고 ㅠㅠ'], ['맞아요. 걸레질은 기계로 대신해도 걸레빠는것 내 몫.. 상상만해도 손가락이 쑤시는거 같네염..'], ['요리요 잘 못해요 ... 시간도 오래걸리고 \n근데 해 먹어야 해서 하는데 저만 먹을때 있어요 다들 맛없다고 \n다음은 다림질 ... 남편은 캐쥬얼이라서 다림질 할 필요가 없는데 제가 셔츠 입는 경우가 많아서 ... \n부모님이랑 살때는 다림질 다 아빠가 해주었는데ㅜㅠㅠㅠ'], ['저도 부엌 ㅡㅡ 특히 음식물쓰레기는 아직도 넘넘 싫어요.... 음식물쓰레기 미쿡처럼 분쇄해서 흘려버렸음 소원이 없겠....진않고 한결 살만해질것같아요'], ['다행히 화장실청소랑 운동화 빠는건 항상 남편이 해주고.. 전 요리.설겆이는 좋은데 다림질 싫어하고 못해요. 청소기 미는건 괜찮은데 밀대로 바닥닦는건 싫고.. 빨래 개는건 괜찮은데 개어놓은거 정리하는게 싫으네요. 근데 울 남편은 빨래돌리고 건조기돌리고 개는 것도 잘 하는데 빨래 정리는 절대 안하네요..'], ['정리정돈 못해요. 그 게 제일 어려워요...'], ['전 시간맞춰 끼니챙기는거요ㅜㅜ'], ['걸레빨기가1등이였는데 \n요즘 닥터브라운젖병씻기가 급 부상중이네요ㅠㅠ'], ['밥하기 ㅠ ㅠ 아 힘들어요\n삼식이들이 ㅠ'], ['첫째도 요리.\n둘째도 요리.\n셋째도 요리...입니다.\n\n청소는 로봇청소기\n빨래는 세탁기와 건조기\n설거지는 식기세척기\n\n대체가능한것들이 있는데 요리는 외식말고는 대체가 없..... ㅠㅠ\n아.. 음식해주는 이모님을 모시고 살아야 되나봐요.\n그러기엔 돈이 없... ㅠㅠ'], ['밥이요.  매일매일 큰숙제에요'], ['전 설거지가 젤 싫어요ㅜㅜ'], ['당근  음식이죠~~  음식하는  과정이  장보면서  메뉴선택 등 순간순간  판단 결정해야  하고  건강하고도 직결되는 문제라  심적 부담 백배~~  다른건 하기싫음 안해도 크게 문제 될거 없지만 밥은 평생해야 되니......아~~~~정말 하기싫고 귀찬고  부담되요...  살아생전 알약 나올   날 만 손꼽아 기달려요'], ['설거지가최악입니다커피마시고음료먹고매번싸이는데서서하면다리아프고손피부도늙는거같아서짜증납니다고무장갑끼면컵같은건깨끗이씻기는지걱정되서...흑흑']]
    
    1494
    25.집안일 세탁기돌리고잇어서다마른빨래 갭니다다애기꺼ㅋㅋㅋ애가방해하기시작하니후다닥 개야겟습니당
    
    [['애 키움 진짜 애옷이 한가득이예용\n시댁서 집으로 가는게 빨래3번 돌려야겠네요'], ['저도오자마자 어른빨래돌렷어여'], ['저흰 지금 어른 빨래 돌아가요 ㅎ'], ['저도 애기 빨래 왕창 개고 빨래 하고\n짐 탁기\n돌고 있네영'], ['전오늘도 두탕해야겟어여'], ['전 오전에 한탕을 돌렸다지영'], ['전지금 애기꺼 먼저돌립니당'], ['저도 애기꺼 돌렸는데 이제 널어야 하는데'], ['전널구잠시 쉬고잇어용'], ['저 이제 오늘러 빨래는 끝이네영'], ['전검은거한번더 돌리러구여'], ['저는 널 자리가 없어서 오늘은 패스'], ['저도지금 고민중이네여 졸려서여'], ['역시 애기께 많지요ㅜ 방해꾼은 저희집에도 있지요ㅋㅋㅋ'], ['ㅋㅋ가져가서 침묻히고그러나여'], ['물고 뜯고 집어던져요ㅋㅋㅋ 휙휙'], ['손수건이 많이 보입니다 삶으셨던거겠지요 ㅎㅎ'], ['넹 손수건삶고널고 시골가서여'], ['네 ㅋ 그때 그 손수건들인거 같더라그요'], ['저도 말려놓은 세탁물이 한가득인데...하..'], ['저는오늘도 한탕해야합니다'], ['역시애이빨래가제일로많구만용'], ['손수건이랑 같이잇어서그런듯'], ['손수건이제일로많이나오기는하지요'], ['난요새손수건 코닦는다고 많이쓴다'], ['ㅜㅜ다예아직도콧물 마니나용?!'], ['많이는아니고 조금씩나온다'], ['ㅡ그래도많이괜찮아진거네용?'], ['응ㅋ근데목소리가 너무아파보인다'], ['5)에고ㅠ빨리나아야되는데말이지요ㅠ'], ['그러게ㅜ약이잘안듣는가보다'], ['세븐 칠은 수건인 건가요'], ['애기수건이에용ㅋㅋ베페서받은거에여'], ['아하 베페서요 건지신 거군요'], ['난집아닐전부 금요일로미뤘다잉'], ['쉬는날이지그날?어고 쉬어야지'], ['완전 장보러도 가야하고 바쁘다잉 ㅠ'], ['이유식은 주문하고잇는거야?'], ['이번부터 엘빈즈꺼 시켜봤어'], ['그거먹여보고 괜찬으면 말좀해죵ㅋㅋ'], ['웅웅 ㅋ 유산균 엘빈즈꺼 먹이니 그거 먹여보려고'], ['아유산균도 파는데얌?몰랏넹'], ['응응 ㅋ 내가 가끔 드림에 올렸던거 엘빈즈꺼 ㅎ'], ['아그게그거엿구낭ㅋㅋ제대로안봣어'], ['집안일은 너무나도할게많아요'], ['하루쉬면 진짜늘어나는거같아여'], ['바짝 잘 마른거 같습니당'], ['이틀잇엇더니 잘말랏지여 ㅋㅋ'], ['아하 그래서 바짝 잘말랐나봅니다'], ['역시아가옷은한가득이지요'], ['맞아용 애기옷이 젤많은듯여'], ['내옷보다 애기 옷이 많은것은 ㅠ'], ['저도그래여 제옷은 사이즈가안맞아서'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋ와나저두여 살이 겁나 쪘네여 ㅠ'], ['큰일이에염..다시다살수도업고'], ['그러니까여 저는 바지가 그렇게도 안맞아여 ㅠ'], ['저도여..바지가맞아야 멀입는데말이죠'], ['고무줄 바지 입던지 ㅋㅋㅋ 치마나 입어여 저는'], ['전레깅스여ㅋㅋ이제추우니 치렝스입어야져'], ['집안일 후다닥하신건가용'], ['빨래널고왓습니당ㅋㅋ이제자유에여'], ['부럽습니다 자유라니용']]
    
    1517
    새벽부터 신나게 집안일하기 어제 밤11시 퇴근하고 집와서  닭개장 해장허구~~1시전에뻗어버렷어요ㅋㅋㅋ아침6시에 개운하게 깨고 새우를 다듬어요ㅋㅋㅋㅋㅋ식탁에 앉아서 조용히 까고있었는데  엄마나오더니 저보고 기겁하드라고요ㅋㅋㅋㅋ놀랏나바요새우튀김을 할거구요 나머지는 손질후 냉동실 넣었다가 라면이나 찌개에 넣든 할거에요 아주혼자 말없이 입다물고 조용히깝니다층층히 깔아서 보관하는데 랩을깔아주세요안그럼 지들끼리 붙어서 띠기힘들어요~새우손질 끝내고요 아침에 가족들과 라면을 먹었어요~ 제가속이 덜풀려서  라면이땡겼...ㅋㅋ 손질한 새우넣고 꽃게한마리 냉동실있어서 넣구요 문어새끼도 넣고 파랑 청양고추 고춧가루넣고 끓여요동생이 너무맛있다니까 엄마가 이재료넣고 맛없음 이상하다구ㅋㅋㅋ그냥칭찬해줌안대나유? ㅋㅋㅋㅋㅋ국물마시고 해장끝 역시구욷엄마는 아아~동생은 아이스라떼ㅋ 커피까지 타서 대령해줬어요( 곧 또 달려야되니 밑밥)커피타주고 짜장에넣을 고기사러 차타고 정육점찾아다녀옵니다..세상에 ..틴트라도바르고올걸 그지깽깽이마냥  나왔는데 젊은멋진남자가 고기를썰어주네요 고개를숙입니다.......ㅋㅋㅋㅋㅜㅜ담엔이쁘게 하고 가야겠다고 생각했어요^^짜장분말 개워주고요야채고기 볶아주고~ 아이가있어서 야채는 작게^^큰게 더 먹음직스러운뎅 ㅋㅋ짜장 완성하고 새우를튀길준비 기름부어~~ㅋㅋ깨끗부침가루로 이쁘게 분칠하고~계란물로 풍덩ㅋㅋㅋ빵가루를 덕지덕지 눌러서 기름에 쏙~~~  저는소금간 안했어요 짠거싫고 양념찍어먹을거라ㅋㅋ계란에만 간되어잇어요~~비린내제거 위해 소주좀 뿌려주고요ㅋ소리살벌~~~  요렇게 식탁에올려놓고 빨래널구 이제씻고일가야죠ㅋㅋㅋㅋㅋㅋㅋㅋ엄마가 새우튀김  맛보더니 고생했다며맛있다고 토닥ㅋㅋ 낼은 닭을튀겨줄까요? ㅋㅋㅋㅋㅋ
    
    [['진짜 저 재료넣고 안맛있으면 이상을 떠나 괴상한거지~ㅋ\n저 라면은 한뚝배기에 이만원은 하것구만~ㅎ'], ['ㅋㅋㅋ물양도 중요해용ㅋㅋ저라면에 소주각이죵? ㅋ 오빤 고기안조아하시니 라면이딱'], ['우와  ㅎ\n\n식당해라  ㅎ  오마카세 식당  ㅎ'], ['전  나중 술집하고싶어욧ㅋㅋ오마카세식당??저 그런거 잘 못해유~~~'], ['요리 잘하시네요^---^\n고기써시는분 얼마나 훈남인지 구경가보고싶네요ㅋㅋ'], ['ㅋㅋ미치게는아니구요ㅋㅋ그냥 훈훈한 젊은이!ㅋㅋㅋ남자구경을왜'], ['고기 어디서 샀니?? ㅋㅋ \n하... 해물라면도.. 새우튀김도.. 넘나 머꼬 시푸다..\n하지만 나레기의 현실은... 슬프다..ㅜㅜ'], ['양청리에서용ㅋ가티갈래용? ㅋㅋ아직도그래서 우째요ㅜㅜㅜㅜ몸 조아지믄 먹기루'], ['배가 쏘옥~ 드러가쪙.. ㅋㅋㅋ\n이틀정도는 굶어야 될거 같애..\n죽먹어도 화장실행이야~ ㅋㅋㅋㅋ'], ['2틀몸사리고 주말갑시돠언니~~ㅋ 나두뱃살쏙하고프당'], ['조신모드로 얌전히 있을게.. ㅋㅋㅋ\n뱃살 하루 먹으면 도로 나온다.. ㅋ'], ['어디나가지말구 꼭이욤ㅋㅋㅋ제가조은곳봐놧어요'], ['아오~~~!\n나 어제 근처 아파트 화요장터가서\n새우튀김 다섯개\n오징어튀김 두개 사다 먹었는데\n새우가 엄청 실해보여 사왔드만\n튀김옷을 잔뜩 입은 새끼손가락보다 가늘고\n성냥개비보다 굵은..ㅠ\n울 이쁜 로미 진짜 부지런하고 솜씨쟁이넹~'], ['헐ㅡㅡ새우가그리작아요??심각한데... ㅜㅜ 언니 음식잘하시니 해드시는게 날듯해요 기름도 안심되구요~~~ ㅜ 가족들잘먹으면 그게젤행복하죵'], ['옛날 명박이 선거하던 날에\n가래떡 튀기다가 화상 심하게 입어 \n병원 입원 15일하고\n6개월정도 햇빛 차단하느라 감금 당하고부터 \n튀김이 넘 무셩~ㅠ\n그전엔 치킨도 튀겨 먹었는데\n그 후로 짝지도 집에서 튀김 못해먹게 하궁~ㅠ\n무엇보다 짝지가 새우를  않좋아해서리\n어디가서 새우 나오면 다 내차지~ㅋ\n'], ['헐 그정도로 심하게ㅜㅜ그럼 무서울만해요.,  그냥사드셔요ㅜㅜ 저도 튀겨놓고 하나먹고왔어요ㅋㅋ 치킨좋져 집에서튀겨먹으니 개안턴데요 뭐니뭐니해도 똥집튀김ㅜㅜㅜ꺅'], ['아니에요오빠ㅋ~~ 감사해용'], ['셰프인신가요?ㅎ\n음식을 이쁘게 맛있게\n너무 잘 하시네요^^'], ['과찬이세용ㅋ그냥 술조아하는이~입니당^^  자취를일찍해서 해먹는거조아해요'], ['캬 금손이시네'], ['손에 어치기 금가루라도 뿌리까예? ㅋㅋㅋ금손되게롱~~ㅋ 감사해욤'], ['투자설명회 함 갑시다.~ 투자 1빠 ㅋ'], ['ㅋㅋㅋ 좋은아이템 구상들어갑쉬돠~~'], ['식당하시면 재료비에 망하시겟어요 얼마짜리 라면인지 ㅎ 맛나겟네요~~^^'], ['망하게 장사할까요ㅋㅋ 약았어요저ㅋㅋㅋㅋㅋ낚시하다 배에서먹는 라면이 그렇게꿀맛이람서용ㅜㅜ 근데배멀미할듯해서 그런날이오련지ㅋ'], ['금손 ^^ 요리사~\n'], ['요리라고 할것도없어용 기본적인거라서ㅋㅋㅜㅜ 감사해용'], ['진짜 부지런하다 와~\n\n사진이니 망정이지\n\n고생 엄청 했을텐데...\n\n뒷처리도 고됐겠다 \n\n수고 많았네 젊은이 ㅠㅠ ㅎㅎ'], ['언니 저진짜6시인나서ㅋㅋ출근전까지 집안일하다왔어요ㅋㅋ설거지며 ㅋ 남자들은알까요ㅋㅋㅋ여자들밥하는거 고생인거 ㅜㅜ감사해요 역시해보신분이 아신다능'], ['살림꾼이네 어쩜 이리 잘하누~~!!!\n그른데~~~정말 저 라면 먹고프다!!\nㅎㅎㅎ ^^;;;;;;;'], ['한번 오심해드릴수있는뎅ㅋㅋ라면하나있음 소주크~~'], ['한손에  라면들고 다른한 손에 소주들고 로미집 가야긋다~~^^'], ['오창으로 이사가고프닷ㅋㅋ'], ['언니이사오세요ㅜㅜ저 오창에서 너무외로워요ㅋㅋ동네친구하나없는ㅋㅋㅋㅜㅜ'], ['맘은 벌써갔다능ㅠ 집팔아서 가야것네ㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋ언니들 다 오창오셨음좋겠ㅋㅋ맨날모여서놀게욤'], ['칭찬감사합니다~~~^^  흉내는 잘내요ㅋㅋ'], ['다 먹으면 살찌겠죠? ㅜㅜ'], ['찌겠죠? ㅋㅋㅋ살걱정은 드실때노노노 전이미쪄있어서 ㅋㅋㅋㅋㅋㅋ 맛있게먹음0칼로리라는데 과연요'], ['완전 금손이네 \n넘요리 잘하는거 아녀 \n부럽 ㅋ 옆집에 살고프다'], ['저희집근처로 오십셔~~술파티ㅋㅋ제가술안주도 쪼매합니다'], ['청주로 오시는게  더 좋을듯해요ㅋㅋ\n안주 맛좀 보고싶네요ㅋㅋ'], ['율량동서온지 얼마안되서ㅋㅋ여기도청주시거등요? ㅋ오창읍일뿐ㅜㅜ'], ['통합되서 그렇군요^^\n제가 청주 사람이긴하지만 서울에서 살다 내려온지 얼마 안되서 잘 몰라요ㅠ'], ['와우 서울남자에요? ㅋㅈㅓ도 한남동서 어릴때 잠시살았던ㅋㅋㅋㅋ~~  청주는오래살아서 조끔알아욤'], ['서울남자x 서울에서 살다온 남자○\nㅋㅋㅋㅋㅋㅋ^----^\n'], ['저는서울여자ㅋ근데청주사는여자ㅋㅋㅋ맛저하세요\n알리님^----^*'], ['저녁 먹어써요ㅋㅋ\n헬로미님도 저녁 잘 챙겨드세요~^^'], ['못하는게 없네요 얼굴도 이쁘고 요리도 잘하구'], ['살을못빼는중....'], ['허걱 괜히 봤네 ;;  새우튀김 한개만 주셔요 ㅋㅋ크ㅡㅋㅋ'], ['드리고싶네요ㅋㅋ마음은 이미 드렸어요~~~'], ['헬로미님 존경스럽습니다 대박!!!'], ['헛 존경이라뇽ㅜㅜ부끄럽네요~~감사해용']]
    
    1550
    집안일은 왜 해도해도 표시도 안나는지.... 아침에 애들 챙겨 학교보내고 비오는날 날궂이한다고 청소 시작~~!!!욕실청소하다 미끄러질뻔하고ㅋ(천만다행 멀쩡...다들 욕실청소할때는 조심하시길)애들방 정리하고 갑자기 안방의 침대 위치가 눈에거슬려 밀어내고 청소기돌리고 닦고...땀한바가지흘리고 에휴~~근데 왜!!청소를 했는데 깨끗해보이지도 않고정리는 해도해도 끝이 안보이는지 ㅠ아침에는 비가 안와서 애들 우산도 안가져갔는데 우산도 갔다주러 학교가야겠네요^^
    
    [['저도 그래요ㅜ 치운다고 하는데 집은 항상 정리도 안되어 있는것같고 티도안나서 슬퍼요'], ['열심히했는데 뒤돌아보면 또 어질러졌네요'], ['그래서 저는 안합니다.ㅎㅎ그냥 대충 보이는곳만 청소하고 살아요.'], ['저도 그러는데...오늘같은 날씨엔  가만있음 잡생각이 나서 ㅎㅎㅎ 그래서 부지런 떨어봤네요^^'], ['미투요 집에12시와서 지금까정. 반찬만들고. 친정집잠깐갖다가 청소하고 티도안나요ㅠ'], ['이제 애들 간식준비해줘야겠네요'], ['ㅎㅎㅎ. 엄마는 바쁩니다\n간식맛나게 만들어주셔요^^'], ['공감이요ㅜ욕실청소해도며칠지나면또물때껴있고바닥은닦아도금방또뭐가묻어나고ㅋ설거지랑빨래는하고돌아서면또쌓여있구ㅜㅜ욕실청소조심하세요위험해요ㅜ'], ['욕실청소는 왜 나만해야하는지 ㅠ'], ['저는초반에신랑이좀해주더니여름이라지쳐서그런지요즘은그냥제차지구만요ㅜㅋ쉬엄쉬엄하셔용^^'], ['정말 안하면 티나는게 집안일이네요ㅠ'], ['집안일은 끝이 없어요 ㅠㅠㅠ \n뒤돌아서면 또 보이고~~ 제 머리카락 빠지는것도 화납니다 ㅜ'], ['아~~~머리카락!!!찍찍이로붙여두 어디선가 또나타나구요ㅠ'], ['집안일은 해도 끝이 없잖아요. 해도 티안나고 ㅡ.ㅠ'], ['집안일은 열심히해도 누가 알아주지도 않아요 ㅠ'], ['부지런하세요 ~ 전 오늘 날씨도 그렇고 대충 한 번 돌렸네요 ㅠㅠ'], ['부지런하진않아요 ㅎㅎ\n월요일만 부지런 떠네요 \n주말에 놀고 애들없을때 청소 ㅎㅎㅎ'], ['진짜 그래용....아무리 청소를 깨끗이 한다고하는데도 왜 그리 깨끗해 보이지 않는지....ㅡㅡ저만 그런줄 알았는데...아니군용...ㅎㅎ'], ['다들 같은 마음이라 공감하네요^^'], ['원래그렇잖아요. 비도 오고 축축 처지는데 살짝만 하시고 쉬세요'], ['네네~~~애들 간식 챙겨주고 커피한잔 하면서 쉬고있네요^^'], ['맞아요 집안일은 해도해도 끝이없고 티도안나고 어지르는건 금방이고ㅠ휴'], ['그니깐요~'], ['그쵸 집안일은 해놨는데도 그대로인것같은 느낌?ㅠㅠㅠ'], ['공감이요 ㅠㅠ~'], ['아ㅜ월요일 힘들고 싫으네요..\n주말동안 엉망된집... 첫째 등원시키고 하루종일 쓸고닦고 정리하고... 곧 하원시간 이네요...ㅜㅜ'], ['저두 이제야 애들 간식주고 커피마시네요~'], ['집안일은 원래 끝이업고 해도 태가 안나잔아요ㅠ 청소하고 뒤돌아보면 벌ㅋ서 딸아이가 어리러 났더라구욬ㅋ'], ['딸애들은 클수록 더 어지르는것같아요'], ['마쟈요 해도해도끝도없고 티도안나고ㅜ'], ['그쵸?저만 그런건 아니죠?\n다른 맘들은 다 너무 깨끗하고 부지런한것 같아서.....^^'], ['그러니깐요!!! 정말화가나는 부분ㅠㅠ 출근 퇴근시간도 없이 일하는데 티가안나요'], ['퇴근시간도 젤 늦어요 ㅠ'], ['하루죙일  청소했는데 신랑한테 청소했는데 깨끗하지?하고 물어보면 청소한거야?......이럴때....진짜🤜'], ['그니깐요~도와주지는 못할망정 수고했다는 말도 없고'], ['맞아요\n근데 이상한건\n하루건너뛰면 바로 티가 난다는거죠ㅜㅜ'], ['머리카락이며 먼지며~굴러다니니ㅠ'], ['표 안나는거~~~\n하지마셔요~~ㅋㅋㅋ\n놀아요~~~먼지도 뭉칠기회를 주셔요~^^'], ['그래야겠네요ㅎㅎ'], ['그러니 집에서 뭐했냐...라는 소릴 듣지요,,ㅠㅠ 주부들의 비애이지요,.,,, 가구위치를 바꾸던지 해야 티가 나고,,ㅎㅎ'], ['오늘은 신랑 등짝을 한대 때려보렵니다ㅎㅎ'], ['집안일은  안하면  확  표시나고  해도 해도  끝이없죠 ㅠㅠ'], ['그니깐요'], ['저도요.땀흘리고 청소하고 빨래해서널어뒀는데.그냥 정리정돈만 조금한정돈지.크게 티가안나네요ㅠ'], ['수고하셨어요~^^\n그래도 우리 서로 그맘을 다알잖아요~^^'], ['젤 힘든일이 집안일인것같아요\n해도 티도 안나고 ㅠ\n힘내세요'], ['티가안나는데 그래도 뿌듯합니다^^'], ['중요한 건 안하면 티가 확나요^^ 그래서 머라도 해야 좀 덜 더러운거 같은 느낌요 ㅎㅎ'], ['그니깐요ㅎㅎ \n공감하네요~'], ['해도해도 끝이없고 티도안나는게 집안일인거 같아요ㅜ'], ['그니깐요'], ['ㅠㅠ정말이요\n근데 해도 티는 안나는게...안하면 티나는게 참 이상하단 말이에요~'], ['그르게요 정말이상해요~'], ['맞아요~해도해도 티 안나는게 집안일이더라구요ㅠ\n그래서 하기 싫어요'], ['저두 하기싫은데 제가 안하면 또 누가하나 싶네요ㅠ'], ['맞아요ㅜㅜ 진짜 치우기는 열심히 하는데 티도 안나고ᆢ금방 또 어질러지고ㅜㅜ 애들이좀 크면 나아지려나요ㆍ'], ['커도 똑같아요 ㅠ\n근데 좀 도와주긴하더라구요 ㅎㅎ'], ['늘 집은 치워도 그런것같긴해요ㅋ뭐가 없어야 깨끗해보여요~'], ['저는 미니멀라이프하시는분들 대단한거같아요^^'], ['주방용품이 좀 많긴한데 전 그저 꾸미는걸못해선지 딱 잇을것만 잇네요ㅎ']]
    
    1703
    신랑이 하는 집안일 중 가장 고마운건 뭔가요? 어제 신랑이 욕실청소를 깨끗이 해뒀네요..미끄럼방지매트도 싹~했다면 더 좋았겠지만..일단은 칭찬이 우선이기에 말 못했습니다..ㅋㅋ분리수거나 음식물쓰레기버리는거..아침에 일어나 밥 준비하는것도 넘 좋아요..제가 아침잠이 많아서..ㅎㅎ맘님들은 신랑이 어떤 집안일 할때가 가장 좋으신가요~?
    
    [['ㅎㅎ눈은 티비에 고정되어있고, 손은 열일하고 계시고..달인이신건가요..암튼 시간을 효율적으로 활용하시는분이네요'], ['주말 오전은 신랑이 밥하고 애들봐줘요 ㅋㅋ늦잠자게해줘서 고맙지요 ~^^ 쓰레기는 매일 버려줘서..금방 분리수거하러갔습니다'], ['부지런하고 자상한 분같아요..저도 주말에 꿀맛같은 늦잠자고싶은데 애들이 깨우네요ㅎㅎ'], ['집안일을 안해서ㅠ 뭐라도 좀 해주셨으면 좋겠네요.. \n요즘에는 쓰레기 버리는거와 냥이 화장실청소는 위임해드렸어요 ㅡㅡ;'], ['맘님이 너무 척척 알아서하시는거 아닙니까? 잘해도 못하는척 연기가 필요합니다ㅎㅎ'], ['저도 집안일은 잘 못해요.. 근데 신랑이 더 못하네요 ㅠ 원래 그런걸 안했던 사람이라 더더욱... 이래서 가정환경이 정말 중요합니다 ㅠ'], ['그 가정환경..이제 새로운 가정을 꾸리셨으니 이 가정에 맞게 바꾸고 배우셔야하는거아닙니까ㅎㅎ'], ['그렇죠? ㅠㅠ 근데 그럴 생각을 안하십니다. 주변에 보면 결혼하면 안되는 사람인데 결혼한 남자들 많더라구요 ㅠ'], ['이상적인 남편은 정말 몇분 안되는것같아요.. 이래서  아직은 결혼하면 여자가 손해라는말이 나오는것같아요'], ['설겆이요 밥은 내가하니 정리는 니가 해라네요. ㅎ 가끔이지만요'], ['ㅎㅎ산더미같이 쌓여있는 그릇들 보면 한숨나올때 많거든요 그때 깔끔하게 클리어해주면 참 좋긴하더라구요'], ['저두요..제발 티비 틀어두고 같이 보지말고 놀이라던가 책읽기라던가 뭘 좀 해줬음좋겠어요'], ['돌아가며 요리하고 청소기돌려주고 장봐주는거요 그러고보니 아침 안먹어서 젤고맙네요.. 아침에 나가는줄도 모릅니다.. ㅋㅋ'], ['ㅎㅎ저도 한때는 그랬어요..밥만 취사예약해두면 국에 말아서 먹고 나가던때가 있었지요. 그것만해도 세상 편해요'], ['애들 씻겨주는거요~^^'], ['울 신랑은 혼자서 씻기는 일이 거의 없어요..거의 제가 씻겨주라고 얘기하는편이지요.첫째가 여아라서 그런가하고 넘어가지요'], ['애기씻기는거랑 음쓰버리구오는거요ㅎ'], ['아이 씻기는건 거의 제 몫이네요 저녁약속있어 외출할때도 애들 목욕시켜두고 나가라고합니다'], ['맘님 부럽습니다ㅋㅋ  저흰 맘님이 적은거중에 하나도 해당 안되요ㅜㅜ 아 그나마 쓰레기는 가끔 버려주는데 오늘은 제가 버렷네요ㅜㅜ 유일하게 하는 집안일은 쓰레기 갖다버리는거예요ㅋㅋㅋ 그것도 제가 다해줘야 버리기만 하는ㅜㅜ음식물은 못버린대요ㅋㅋㅋ참나...'], ['ㅋㅋ저도 결혼하기전에는 엄마가 음식물쓰레기버리라고하면 그렇게 싫었는데..남편분이 딱 그때의 제 모습인것같네요'], ['저는 설겆이요 ㅋㅋㅋ'], ['저도 산머미처럼 수북하게 쌓여있는거..스스로 고무장갑 끼고 설겆이하는거보면 참 흐뭇하더라구요'], ['남편 집안일 기여하는 수준은 정말 집집마다 천차만별인거같아요.. 욕실매트까지 청소해주는 남편.. 다음생에라도 꼭 만나고싶네요ㅎ'], ['다음 생애에 결혼하실껀가봅니다.ㅎㅎ다음에 다시 태어난다면 전 그냥 혼자 살아볼까해요'], ['설거지, 음식물쓰레기포함 쓰레기버리기, 화장실청소, 아이들목욕 등등 많지만 특히 아기 이유식만들기를 해줘서 너무너무 고마워요^^*'], ['아..맞아요 이유식 만드는거 도와주는거 엄청 좋더라구요..재료준비에 젓고 통에 넣고..뒷정리하고..진짜 큰 도움되잖아요..'], ['화장실 청소는 신랑이 무조건 해요 \n한번씩 냉장고 안에 정리도 신랑이 하고요'], ['신랑도 냉장고 정리하긴하는데, 한마디씩 하고 정리해서 글쓰다가 삭제했네요ㅋㅋ'], ['맞아요 그냥 설겆이하는게 아니라 알아서 해주는게 중요한것같아요'], ['저희신랑은 음식물버리기랑 재활용버리기 지담당여\n앗 빨래개기랑 요래는 무조건 지가합니다ㅎㅎ'], ['ㅎㅎ은근히 빨래 개는것도 귀찮더라구요 하다보면 순식간에하고 별일아닌데 해주면 좋더라구요'], ['그렇치요\n저는 이세상에서 빨래개는거를 젤 싫어하거든요ㅎㅎ 딴거는 내가해도되니\n빨래개기만잘해라해여ㅎㅎㅎ'], ['굳이 제일 싫어하는걸 할필요있나요..싫어하는거 도와주는것도 얼마나 행복한일인가요..'], ['그러이 싫어하는거 \n하기싫어 전 안해용ㅠ\n빨래갤거 한~~~거 쌓아노마 \n지가 노는날 개여ㅎㅎㅎㅎ'], ['아..빨대컵..대공감해요..저도 빨대컵 부속품 하나하나 빼서 씻고 빨대씻는거 귀찮을때 많거든요..아니 거의 귀찮아요..근데 그거 씻어주면 감사하지요~'], ['쓰레기버리기'], ['거의 둘째랑 같이 붙어다니다보니 유모차 몰고나가다보면 쓰레기 갖고나가는걸 잊는 경우가 많더라구요..그래서 거의 신랑담당이에요'], ['주말마다 애들아침준비해주는거요ㅎ 주말엔 전 늦잠자요ㅎㅎㅎ 한두번이 1년이다되가네요ㅋ 글고혼자 음식해서 먹는거요 제가 신랑밥도 안챙기니 편하더라구요 평일저녁은 제가해놓지만ㅜㅜ'], ['주말아침준비만큼 고마울때가 있을까요..전 아침잠이 많아서 아침밥해두면 그리 좋더라구요'], ['큰아이씻기고 양치시키고 약챙겨먹이고\n주말엔 모아뒀던 수건 푹푹 삶아 빨아주고\n욕실청소 싹 해주고\n음식물쓰레기 냉동실에넣어두면 언제버렸는지 밤마다 버리고와서 아침에보면 눈에안띄게해주는것.\n그리고 분리수거.. \n나는자게해주고 둘째아가 밤수유 새벽에 챙겨먹이는것.,\n음.. 그리고도 많은걸보니 많이도와주는것같네요^^;;;'], ['맘님 남편분 최고인것같네요..\n특히 수건 삶는건 보통 내공이 아니신듯해요..밤수유하시는거..진짜 최강이네요'], ['음식물쓰레기버리기, 밥차려주기 ㅋ \n남이 차려준밥이 제일 맛있어요'], ['ㅋㅋ네네 남이 차려준밥이 최고에요..특히 주말아침에 그렇게 좋을수가없더라구요'], ['잘때 분명 거실이고 방마다 아이가 다  전쟁통 처럼 해둔 집을 아침에 출근하기 전에 싹 ~ 치우고..일어나서 보면 정말 므흣합니다 ㅋ'], ['울 신랑이 정말 못하는거네요..장난감정리하는게 힘든지 그 난장판속에서 자고있어요..ㅎㅎ'], ['크크크 우리신랑도 난장판속에서 잘 잡니다요.. 대신 아침에 일어나서 치우고 출근 하더라구요 ㅋ'], ['조금전에도 널부러진 장난감 안치웠는데, 오늘은 왠일인지 상자에 넣어두고 잠드네요..해가 서쪽에서 뜨는건가요'], ['오호~ 오늘 바깥양반이 아주 착한일을 하셨군요..내일 아침은 더 맛난걸로 ㅎㅎ'], ['애들 세명 다 씻겨주고 양치시켜주는거요~ㅎㅎ\n 외에 소소하게 쓰레기버리기  걸레빨기 옷개기 이따금 도와주네용'], ['애들 씻겨주는것도 좋은데, 세명 모두 씻겨주시다니 대단하네요.전 씻기는건 제 몫이네요.'], ['입덧할때 음식하는거 곤욕이지요..역해서 할수도없구요..앞으로 더 많이 도와주셔야할텐데요..'], ['애들 케어 해주고 목욕 시켜주고 \n빨래 널고 개어주고 \n애들 등하원 시켜주고\n잠자리에 재워주고 \n저희 집도 항상 욕실은 락스로\n청소한다고 독하다고 신랑이 담당이라\n다 고맙지요^^\n'], ['맞아요..락스냄새 독하다고 청소하는데..특히 여름에 땀 범벅대서 하는거보면 고맙더라구요'], ['집안일은 둘째치고 애랑만 좀 놀아줘도 좋을듯해요.늦은퇴근으로 집은 딱.잠만자고나가는듯해요ㅜ'], ['퇴근이 늦으시다니 어쩔수없는거지요 대신 집에 계실때는 잘 도와주시겠지요..ㅎㅎ'], ['여기 글.읽어보니 이제껏 못해준다고 생각했던 신랑이 고맙네요 ㅜㅜㅜㅜㅜ 분리수거 설거지 청소 장난감정리 주말아기밥먹이기 . 쉬는날 아기케어 . 쉬는날 놀러다니기 .와이프 밥해주기 . 빨래개고 세탁기 건조기 돌리고 다 하는데말이죠 . 그럼에도 아기 씻기기 . 아기 손톱발톱깍아주기 .칫솔질해주기 . 밥먹여주기 화장실청소해주기는 억시로 안할려는 신랑이네요 . 뭘바라겠어요 저만하면 감지덕지헤야겠어요 ㅋㅋㅋ ㅜㅠㅠㅠ'], ['잘하는것만 있어도 그게 어딘가요..전혀 아무것도 하지않는것보다 하나라도 도와주면 좋지요..하지만 계속해서 그 숫자가 늘어나야겠지요ㅎㅎ'], ['빨래개어놓기랑 쓰레기 버려주는거요'], ['간단한건데도 그걸 그렇게 생색낼때가있어요ㅋㅋ둘째만 껌딱지가 아니였어도 그까짓꺼 광속처리하는데..'], ['쓰레기만 버려줘도 고맙더라구요ㅎ'], ['그것도 크지요..쌓인거 들고가다가 넘쳐서 떨어뜨리고..줍다가 다른거 또 떨어지고..ㅎㅎ 순간 욱~하게되더라구요'], ['부럽네요  집안일 거의 다 도와주나보네요 저는 주방쪽  나머지는 신랑 담당인데 출산하고 집에 있으니 제가 빨래도 하고 청소도  하네요 ㅜ'], ['맞아요..왠지 일하는 사람한테  이것저것시키는게 잘 안되더라구요..주부도 집에서 탱자탱자노는거아닌데 왠지 집에 있다는 이유로 해야할것같고ㅜㅜ'], ['자영업하는데 퇴근이 늦어요 그래도 아이랑 1시간 놀아주거나 목욕시켜주고 쓰레기 버려줄때 고맙더라구요'], ['늦게퇴근하시면서 피곤하실텐데 아이랑 놀아주시는거 대단한거에요..시간이 중요한게 아니라 얼마나 공감하며 알차게 놀아주는지가 중요하다더라구요'], ['빨래게는거랑 쓰레기 버려줄때요 ㅋㅋㅋㅋ\n눈은 티비에 잇고 손은 빨래에잇어 웃겨요 ㅋㅋㅋ'], ['그건 빨래개기 달인의 경지에 이른것같네요..ㅎㅎ남편이 티비만 보면 주부입장에서 눈에서 레이저발사되잖아요'], ['군말없이 하는게 어디에요..ㅎㅎ뭘 해야될지몰라서 그러실수도있어요..딱 정해주세요'], ['빨래돌린거꺼내서 옥상널고 저녁에걷어서 개어줄때요~~ㅋㅋ'], ['빨래 거내서 탈탈 털어서 너는게 참 귀찮긴하더라구요..그런데 그걸 옥상에 가져가서 널고 걷어오신다니 대단해요..'], ['저는 분리수거 해주는 거랑 음식물쓰레기 갖다 버려주는 거요..'], ['분리수거랑 음식물쓰레기만 버려줘도 고맙다생각했는데 사람욕심이란게..점점 만족을 못하고 더 해줬음싶더라구요'], ['네네.. 사실 제가 되게 하기 싫은 집안일 중에 하나거든요..\n그걸 해주니 고마워요~'], ['남편이 집안일 엄청 잘해주시네요. 저희 남편은 자기 밥 잘 챙겨먹는 것만도 감사한 사람이라..'], ['사실 집안일을 나몰라라하지는 않는데..정말 제가 해줬음하는건 못하는경우가 많아서 나름 불만도 있지요ㅇㅅㅎ'], ['저희 신랑은 집안일은 거의 안해요~ 제가 시키지 않는한.. 그냥 알아서 해주는 것만 고맙다고 ㅋㅋ'], ['ㅎㅎ그래도 시키면 하시잖아요..안하고 버티는 분들도 주위에 많이 있더라구요..진짜 속 터질듯요'], ['저희신랑은 시키면 하지만 바로바로 하는 편은 아니에요 사실.. 그래서 좀 답답한 건 있지요 ㅋㅋ']]
    
    1752
    집안일 중 젤 하기 싫은 거.. 다들 하나씩 있지 않나요?전.. 빨래개기가 그렇~~~~~~~~게 싫어요. ㅎㅎㅎ그래서 건조기에 어제부터 양말들이 있어요. ㅍㅎㅎㅎㅎ어떨 땐 큰 애한테 건조기에 양말 마른 거 있으니 꺼내서 신고 가.ㅋㅋㅋ 그런답니다.다들 뭐 싫어하세요??
    
    [['아.. 저도 뽀로로 매트.. 그 올록볼록 사이 때들.. 밤마다 물티슈로 닦아냈었던 기억이... ㅡㅡ;; 수고 많으십니다. 흑.'], ['전 걸레빨기요 ㅡ,.ㅡ'], ['ㅋㅋ 마자요. 그것도 있었네.. 흑.'], ['전애들 양말빨기여~~발바닥이새~~~까만 ㅡㅡ'], ['저도 그래서 손빨래 해보다가요.. 걍 싼 거 신기다가 버리자 싶더라고요. 힘들어요.'], ['빨래정리~요.개는거  까지는 하겠는데, 그거  제 자리에  넣는거가  왜케싫은지요..'], ['ㅋㅋㅋ 아 맞아요. 저도 그거. ㅋㅋㅋ 건조기서 꺼내서 개는 데 하루.. 개서 거실 테이블 위에 주~~~욱 구분해놓고는 각자의 방에 넣는데 이틀. ㅋㅋㅋㅋㅋㅋ'], ['저는 청소'], ['전.. 눈에 보이는데는 참 잘 치우거든요. 그렇지 않으면 견딜 수가 없는 스탈이라..'], ['음식물쓰레기요... 싱크대 하수구는 손도 못대겠어요...'], ['앗. 어제 저 삘 받아서 가스렌지 후드랑 싱크볼 하면서 하수구도 박박 씻어줬는데.. 하면 개운해요.'], ['전 설거지요 ㅓ'], ['전 설거지는 좋아라합니다.. 요리보다 더 자신있는 게 설거지.. (요알못이라.. ㅜㅡㅜ)'], ['전 청소기요ㅋㅋ'], ['잉?? 청소기는 걍 꽂고 돌리기만 하믄 되는 젤 쉬운 거 아녜요??'], ['설거지요..음식물버리기.'], ['음. 다들 참 많이 다르구나.. 느낍니다. ㅎㅎ 전 설거지 좋아해서..'], ['ㅋㅋㅋㅋㅋ 어쩌면... 정답!!!! ㅋㅋㅋㅋ'], ['저도 빨래요!!!!'], ['빨래 정리.. 진짜 젤 귀찮아요.. 수납이 넓직하지 않아서 긍가.. 제자리 넣는 게 젤 싫고 귀찮고..'], ['전 설거지요~ 힘들게하고나면 하나둘씩 또나와요 ㅠ'], ['전 하나 나오면 바로바로 해버리는 스탈이라.. 설거지 싫다 생각한 적이 없어서...'], ['빨래 너는거요ㅠㅠ 개기나 정리해서 넣기는 어렵지 않은데 빨래널기는 넘 싫어요~'], ['아.. 너는 거 싫어하시는 분도 계시군요!!!'], ['청소용 ㅠㅠ'], ['전 좋아해용.. ㅜㅡㅜ'], ['정리정돈요\n돌아서면 정리할게 천지 ㅠ'], ['우리나라는 사계절이라 더한  거 같아요. 내일은 꼭 완벽하게 옷장정리와 신발장정리를 끝내야지.. 맘 먹습니다. ㅜㅡㅜ'], ['전 다~~~~시러요~ㅠㅠ'], ['흑.. ㅜㅡㅜ'], ['화장실청소요~ 아뇨 그냥 다 싫어요 ㅜㅜ'], ['다 싫은 게 정답일지도요. ㅎ'], ['설거지가 세상에서 제일 싫어요 아니 청소도 싫고 쓰레기 버리는것도 아... 다 싫은거네요 흐엉헝'], ['으헝헝~~ 그래도 하나만 싫어해보아요~~~'], ['저는 세탁기에서 빨래 꺼내는거요ㅎ'], ['악!! 저도 그랬던 적이 있었어요. 허리가 아파서. ㅡ.ㅡ 근데 운동하니 좋아져서 요것도 괜찮아졌어요. ㄹㄹ'], ['저도 빨래개서 서랍장에 정리하기..정리정돈 힘들어요'], ['맞아요. 저도 빨래 갠 거 이틀 동안 테이블 위에 정리?? 해놓은 적 있어요. ㅎㅎㅎ'], ['저도 건조기에서 빨래꺼내 소파위에 널부러놓고 며칠 지내적 있어요 ㅠㅡㅜ'], ['저는 요리요^^;;;; 요알못이라 ㅠㅠㅠ'], ['저도 못해서 반찬가게 이용해요. ㅎㅎㅎ'], ['밥하기...'], ['밥솥이 해주는디요. 흑.'], ['걸레빨기...물걸레청소기 있음 모하나요..걸레는 빨아야하는데...넘 싫어요..'], ['아.. 저도 손으로 걸레질 해서.. ㅜㅡㅜ 닦다가 다시 빨러 욕실 들어가기를 몇번이나 하는지.. 그래도 운동이다... 생각하고 합니다..'], ['전 청소...반찬만들기요'], ['반찬은.. 걍 사요.. ㅜㅡㅜ'], ['전......,\n다~~~~요....\n반백살 넘으니\n다 귀차너요....ㅠㅠ'], ['우왕~~~ 큰  언니시구나~~~ 다 해주는 로봇 안 나오려나요..'], ['다싫어요..ㅜㅜ\n그냥 전부..ㅋㅋㅋ'], ['ㅋㅋㅋㅋ 마음의 소리. ㅋㅋㅋㅋ ㅜㅡㅜ'], ['저는 청소는 하고나면 뿌듯한데 요리는 하기전부터 머리아포요ㅜㅜ'], ['저두 요리는 살짝 그런 거 있어요. 특히 뭔 날이어서 하는 요리는 더더욱.. 하기 전부터 시뮬레이션 해 싸코 난리난리.. ㅜㅡㅜ'], ['저도요!! 빨래 개는거 세상 귀찮아서 건조기에서 꺼내서 바닥에만 며칠 던져둘 때도 많아요 ㅎㅎ'], ['전 요리랑 음식물쓰레기 버리기요.... \n요리는 정말 아예 어떻게 해야될지 감도 안오고 할 생각하면 벌써 어렵고 지겹고 그래요 ㅠㅠ 그래서 다 사먹고 싱크대는 그냥 컵 쟁반 이런거 최소의 설거지만 하고 싶은데 남편이 자꾸 밥을 해주려고 해요ㅠㅠㅠ 설거지나 청소는 하고나면 뿌듯해서 좋아요 ㅎ 그래서 평일은 사먹고 주말 한끼는 남편이 해주고 제가 설거지하는데 이렇게 많이 사먹어도 될까 걱정이에요ㅠㅠ'], ['빨래개기 생각하며 클릭했어요 음식물 버리기요'], ['저도 빨래개는게 젤 귀찮아요ㅜㅜㅋㅋ'], ['전 설겆이가 젤루 싫어요.'], ['창문닦기요...틀도 닦아야 하고 엄청 귀찮아요.'], ['전 요리요~혼자라면 대충 먹겠는데 아기 끼니 챙겨줘야해서  항상 고민되요ㅋㅋ'], ['설겆이요!'], ['저도 빨래정리랑 설거지요ㅎㅎ넘 귀찮아요ㅋㅋ'], ['걸레빨기 늠 싫어요ㅡ'], ['저는 설거지요~^^'], ['전 밥이요']]
    
    1781
    집안일 했어욤~^^ 집에 오자마자 엄마가 싸주신 것들 정리했어요~김치도 김치통에 옮겨담고,다진마늘도 소분해서 냉동실 보관하고,그리고 나서 보니 지난주에 밭에서 뽑아온 파가....점점 시들어가고 있네요ㅠㅠ엄마가 새벽부터 뽑아주신건데ㅠㅠ그래서 파 손질 들어갑니당~^^다행히 끝부분만 노랗게 시들었네요~잘 다듬어서 깨끗하게 씻어서 말리는 중입니다~그래서 집안이 파냄새가 진동을 해요 ㅋㅋㅋ청소도 하고 빨래도 하고~ 커피한잔 타서 컴터 앞에 앉았네요~자... 이제 시작을 해볼까요?^^
    
    [['아침부터  집안일많아하셨네요~~^^    커피한잔드시고  쉬세요'], ['쉴수가 없어요~ 밀린 숙제해야되요 ㅋㅋㅋㅋㅋ'], ['으악ㅋㅋㅋㅋ 파 보관은 어떻게 하실꺼에요??'], ['바짝 마르면 키친타올 깔아서 냉장 보관하려구요~ 반은 잘라서 냉동보관하구욤^^'], ['우와 살림꾼!!! 한수배워가요~~'], ['헉!! 살림꾼 아닌데ㅠㅠ 으앗 ㅋㅋㅋㅋㅋ'], ['아침부터 어마어마하네요;;ㅎ'], ['친정갔다오면 할일이 태산이죠ㅠㅠ 뭔가가 자꾸 생겨나요 ㅋㅋㅋㅋ'], ['와우~~아침부터 분주하게 움직이셨네요\n뿌듯하겠어요\n저도 시골에서 시엄마가 파 뽑아서 보내주시면 반은 냉장 반은 냉동고로가서 두고두고 잘먹고있어요ㅎㅅ'], ['한동안 파 걱정없이 지낼수 있을듯 해요 ㅋㅋㅋㅋㅋ'], ['저도 커피 한잔하고싶네요ㅎ'], ['늦게 마시면 못자니 일찍 마셔버렸어요 ㅋㅋㅋㅋ'], ['아침부터 완전 바쁘세용ㅎㅎ'], ['친정갔다온 날은 늘 바쁘네요 ㅋㅋㅋ'], ['마자요 저두요ㅠㅠ'], ['ㅋㅋㅋㅋ 저만 그런게 아니었군요 ㅋㅋㅋ'], ['짐정리하고 쌓여있는 집안일하고...ㅋㅋㅋ'], ['ㅋㅋㅋ 그러다보면 한두시간 훌쩍 지나가고 그리고 보면 금방 저녁이 되고 ㅋㅋㅋ'], ['마자요... 전 중간중간 아기케어하고.....하하하핫 하루가 순삭입니다'], ['진짜 저도 하루가 순삭일때가 오겠네요ㅠㅠ'], ['아직 멀었어요! 지금 즐기세용!ㅋㅋㅋ'], ['ㅋㅋㅋ 열심히 즐기겠습니당 ㅋㅋㅋ'], ['ㅋㅋㅋ맛있는것도 많이 먹으러 다니세요~~'], ['먹는거는 아무래도 힘들고 그냥 즐기기만 하겠습니당 ㅋㅋㅋ'], ['임당중이신가요?ㅠㅠ'], ['넹~ 인슐린 맞고 있는데, 아마 다음 진료때가면 용량이 올라갈것 같기는 합니다~'], ['ㅠㅠ힘드시겠어용 ㅠㅠ'], ['솔직히 힘든건 먹는거 참는게 힘이 드네요ㅠㅠ 먹고 싶은게 너무 많아서요ㅠㅠ'], ['그렇죠 뱃속에 아가도 있는데 먹고싶은거 못먹는게 힘들죠ㅠㅠㅠ'], ['그러니까요 ㅠㅠ 근데 그 안먹는게 오히려 아가한테 좋은거라~ \n그 이유로 참고 있습니당 ㅋㅋㅋㅋ'], ['빨리 아가 낳고 드시고싶은거 드셔야겠어용~'], ['출산만 하면 먹고 싶은거 다 먹을거예욤ㅋㅋㅋ'], ['다 드세여! 지금부터 뭐먹을지 생각은 안해놓으셨나용~'], ['음.. 일단 칼국수 먹을거구요~ 아스크림이랑 빵 종류별로 다 사먹고, 떡볶이도 먹을거구요~ ㅋㅋ\n아.. 쓰고보니 죄다 밀가루네요 ㅋㅋㅋ'], ['ㅋㅋㅋㅋ저도 임신중에 밀가루 자제 엄청했어요 모유가 진득해져서 막힌다고 많이먹지말라했거든요ㅠㅠ'], ['아,... 밀가루 많이 먹음 그래요? 그럼 밀가루 줄여야겠네요ㅠㅠ'], ['모유수유할거면 밀가루 줄이라고 그러시더라구요ㅠㅠ 수유중에도 줄여야하는데 그게 잘 안되요ㅠㅠ'], ['이런...전 출산하고 맘껏 먹을 생각이었는데, 밀가루는 출산후에도 안되는군요ㅠㅠ'], ['밀가루가 참 안좋은 음식인가봐요 ㅎㅎㅎ'], ['왜 안좋을까요? 맛있는건 다 몸에 안좋은거 같아요ㅠㅠ'], ['그러니깐요ㅠㅠ밀가루가 이리 몸에 안좋은거 임신하고 알았어요'], ['저두욤 ㅠㅠ 그치만 아가를 위해선 먹지 말아야지요~'], ['네ㅜㅜ 저도 많이 자제중이예요 밀가루를 너무 좋아해서ㅠㅠ'], ['저도 오늘 병원가서 쌤한테 혼나서 진짜 이제 식단 조절 해야되요 ㅠㅠ'], ['크~~ 파 다듬을 때도 눈 따가운데~ ㅎㅎ 고생하셨어요.'], ['오늘은 다듬을때보다 씻어놓은거 말릴때가 더 눈이 따갑네요 ㅋㅋㅋ'], ['파 양이넘많은데요ㅋㅋ'], ['근데 막상 다듬어놓으니 얼마 안되더라구요 ㅋㅋㅋㅋ'], ['우앗 오징어넣고 호박넣고 파전해먹어도 맛나겠어요 *.*'], ['악!! 맞아요 ㅋㅋㅋ 딱 파전해먹기 좋은 크기네요 ㅋㅋㅋㅋ'], ['신랑분오시면 함께 파전 해서 드세용 >~<'], ['ㅋㅋㅋ 신랑오면 해먹을게 너무 많아서 ㅋㅋ 차례가 파전까지 갈까 모르겠네요 ㅋㅋㅋ'], ['ㅋㅋ가장 먼저 드실건 뭐에용???'], ['음... 장봐서 고등어조림을 먼저 할라했는데, 장을 못봤으니 김치찌개로 일단 시작하네요 ㅋㅋㅋ'], ['우아 다 맛있는거네용 ㅜㅜㅜㅜ \n내일 친정가는데 저녁에 고등어조림해달라고 해야겠어요 !!'], ['오오~ 저녁에 고등어조림 드셨어요? 전 오늘 마트갔는데 고등어가 떨어졌대서;;; 그래서 그냥 순두부로 메뉴 변경이요~ 제육볶음이랑~ ㅋㅋㅋㅋ'], ['전 고등어조림대신 굴비구워주셨어여 ㅋㅋㅋ\n순두부는 찌개해서 드셨어요??ㅎㅎㅎ'], ['아니요~ ㅋㅋ 오늘아침에 해줄라 했는데 늦잠자서 ㅋㅋ\n대신 신랑이 토스트 해줬어욤 ㅋㅋㅋ'], ['우앗 오늘 아침엔 깨비맘님 요리 안해도 되서 여유있으셨겠어용 *.*'], ['넹 ㅋㅋㅋㅋ 근데 내일은 일찍 일어나야되요~ 6시에 신랑 나간다고 하네요~'], ['깨비맘님께서 아침 차려주시는거에용??'], ['아침은 사무실 갔다와서 먹겠대요~ 잠깐 나가서 얘기해야하는거라서~\n사무실 가있는동안 준비해놓으면 될듯요 ㅋㅋㅋㅋ'], ['우앗 그럼 조금더 여유있으시겠어요 ㅎㅎㅎ 6시 맞춰서 차리면 힘들었을텐데 ㅜㅜㅜ'], ['결국 ㅋㅋㅋ 둘다 6시반에 일어나서 그냥 안갔어요~ 사람들이 6시반에 사무실에서 외부로 나가는데\n신랑 안오니까 전화왔더라구요~ 그냥 나간다고 ㅋㅋㅋ 그래서 그냥 다시 잤어욤 ㅋㅋㅋ'], ['우앗 ㅋㅋㅋ 깨비맘님한테는 좋네요 *.* 밥 준비도 더 늦게해도 되고 꺄아 ㅎㅎㅎㅎㅎ\n같이 자니 더 일어나기 힘든거 같아요 ㅜㅜ'], ['근데다 제가 요즘 너무 늦게자서 많이 피곤했나봐요 ㅋㅋㅋ'], ['우린 홀몸이 아니라 피로도 더 잘 누적되는거 같아요 ㅜㅜㅜㅜㅜ\n깨비맘님이 확실히 신랑분 오시기 전보다 늦게 주무시는거 같아요 히히'], ['ㅋㅋㅋㅋ 오히려 신랑없을때 일찍잤던것 같네요 ㅋㅋ \n그때는 내 맘대로 해도 되니까욤 ㅋㅋㅋ'], ['그땐 진짜 맘님 일찍 자고 일찍 일어나셨었는데 ㅠㅠㅠ'], ['ㅋㅋ 그리고 그때는 지금처럼 댓글에 열정적이지 않았지욤 ㅋㅋㅋ'], ['ㅋㅋ 지금이 전 더 좋아욬 ㅋㅋㅋ 서로 알아가고 *.*\n그나저나 위에있는 파로.. 파전 해드셨어용????ㅎㅎㅎㅎㅎ오랜만에 파보니 생각나네요 ㅋㅋ'], ['아니요~ 그냥 음식할때 넣았어요 ㅋㅋ'], ['ㅎㅎ 파전은 사먹는걸로 *.* ㅎㅎㅎ 파전먹고싶은데 집에 오징어도 없고 다른 재료도 읍네요 ㅜㅜ'], ['ㅋㅋ파전은 집에서 하는거 아니예욤 ㅋㅋㅋ 사먹는거지욬ㅋㅋ'], ['그죠..? ㅋㅋㅋ 괜히 어설픈 제가 시도했다가 이도저도 아닌.. 파를 희생시킨 셈이 되겠죠? ㅋㅋ'], ['맞아요~ 맘님이나 저나 괜히 나서서 한다고 햇다가는 죄엇는 파만 피봅니다 ㅋㅋ'], ['ㅋㅋ 깨비맘님은 요리 고수일거같은데 ㅜㅜㅜ'], ['절대요ㅠㅠ 제가 하던것만 할 줄알아요~ 가끔 새로운거에 도전은 하는데 5:5라 ㅋㅋㅋ'], ['아침밥상 올라올떄마다 다 너무 맛있어 보여요......'], ['그런가욤? ㅋㅋㅋㅋ 진짜 밥상한번 차려드리고 싶으네요 ㅋㅋㅋ'], ['오늘 깨비맘님 아침밥상 보고도 침을 줄줄 흘렸어여.. 전 점심 뭐먹어야하나 벌써 고민이에요 ㅋㅋㅋ'], ['ㅋㅋㅋㅋ 얼른 저희 만나야겠는데욤? ㅋㅋㅋ 뭔가를 먹기 위해서라도요 ㅋㅋㅋㅋ'], ['ㅋㅋㅋ흐흐ㅋㅋㅋㅋ 전 진짜 깨비맘님 밥상 보고 메뉴 정해요 ㅋㅋㅋㅋㅋㅋㅋ'], ['오오~ 진짜요? ㅋㅋ 그럼 앞으로 더 신경써서 해야겠네욤 ㅋㅋ'], ['노노 지금처럼만 하셔도 대용 ㅋㅋ\n게란말이 보고, 저도 게란말이 하고, \n장조림 보고, 저도 장조림 하고 히히히히히'], ['ㅋㅋㅋㅋ 진짜욤? ㅋㅋㅋㅋ 저도 매일 메뉴때문에 고민이네요ㅠㅠ\n이럴때 그냥 신랑 출장갔을때가 좋았던거 같아요 ㅋㅋㅋ\n'], ['메뉴 고민이 젤루 싫어요 ㅜㅜ\n차라리 신랑이 딱 뭐 해줘 이런게 있으면 좋은데, 즤 신랑한테 뭐 먹고싶냐고 물어보면\n아무거나래요.. 저 하고픈거 하래요..ㅡ3ㅡ 이 대답이 젤루 싫어요 ㅋㅋ'], ['어쩜 저희집이랑 그리 똑같은지.. 맨날 아무거나.. 그래서 아무거나는 없다고 하면 하고싶은거하래요 참나 ㅠㅠ'], ['아으 아무거나가 제일 싫어요 진짜 ㅋㅋㅋ\n오늘 국 뭐해줄까 ~ 했을때 딱 먹고픈거 말하면 어련히 맛나게 해줄텐데... 쳇\n아이 키우다 보면 메뉴고민이 더 많아지겠죠? ㅜㅜ'], ['그때는 아빠메뉴에 아기 이유식 메뉴까지 더 힘들어지겠지요 ㅠㅠ\n그나저나 저 큰일이여유 ㅠㅠ 신랑 집으로 온다네요ㅠㅠ\n오늘 외박이라고 해서 저녁 걱정도 안하고 하루종일 컴터 할라고 했는데...\n저녁 준비도 해야하고 컴터도 오래 못하겠어욤 ㅠㅠ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋ가끔 ..신랑 출장이 그립죠... ??? 흐흐 이런생각하면 안되는디 ....;;;'], ['이렇게 오래 같이 있을때는요 ㅋㅋ 가끔 그리워요~ 안가나 하고 ㅋㅋㅋㅋ'], ['ㅋㅋ또 맘님은 할게 많은데..못하게 할때는.. 없을때가 그립다.... 싶기도 할거같아요 ㅋㅋㅋ'], ['ㅋㅋㅋ 맞아요~ 근데 신랑이 댓글은 달게 해요~ 오히려 티비보면 보지말고 댓글달라고~\n근데 자러 들어가면 그때는 같이 가야되요~'], ['오늘은 아직 신랑분 티비보셔용????ㅎㅎㅎㅎ'], ['어제 혼자 방에가서 폰보는것 같더니 가보니까 자고 있더라구요 ㅋㅋㅋㅋ'], ['ㅎㅎㅎ 요기 카페 신랑분들은 미남이 많은가봐요 다들 잠이 많으셔요 ㅋㅋㅋㅋ'], ['아하하 그말은 진짜 뻥인거 같아요 ㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋ 미남은 아닌데..잠만 많은걸까요..?ㅋㅋㅋㅋ'], ['아마도요? ㅋㅋㅋㅋ 잠만보들 ㅋㅋㅋ']]
    
    1796
    집안일 끝~^^♡♡ 식탁에 앉아서 아이스아메 마셔요저녁은 또 뭐하나요?앉아서 고민하네요저녁 뭐드시나요?
    
    [['시원하겟네요^^'], ['네 집안일 다 해놓고 시원한 아이스아메 마시니 좋네요 \n여유롭게 앉아서 먹네요ㅋ'], ['저도 한잔 마셔야겟어요'], ['네 맘님 시원하 아이스로 한잔하세요^^\n집안일 다 해놓고 먹으니 좋네요ㅋ'], ['아이고 부지런하십니다 저도 슬슬 움직이야겠네요 ㅎㅎ'], ['어제 많이 드셔서 계속 누워 계셨나보네요\n맘님댁은 저녁 뭐하실려나요?'], ['글쎄요 그냥 대충대충 만들어서 먹어야지요 ㅋㅋ뭘 해야될까요'], ['전 어제남은 볶음밥 애들줘야겠어요\n올 애들 학습봐주고 좀 놀다가 줘야겠어요ㅋ'], ['저도 오늘 애들 공부 좀 시킬라고 합니다 넘 놀린것 같아요'], ['ㅋㅋ 요즘 초딩학생들 학습 어렵던데요\n저 내년에 걱정이네요\n애들 스스로하나요?'], ['저도 집안일 끝내놓고, 세은이 낮잠 자길래 옆에 같이 누워있어요~'], ['저도 잠이 솔솔 오네요\n자기는 시간이아까워서 카페놀이하면서\n친구랑 통화중이네요ㅋ'], ['저두 잠자는 시간 아까워서 웬만하면 안자려고하는데,  몸이 말을 안들어서 한숨자고 일어났더니 쫌 괜찮아졌네요~'], ['맘님은 좀쉬어야죠\n감기걸리고 몸도 안좋으시잖아요ㅜ'], ['네~ 여유있을때마다 쉬어주려구요~\n체력충전을 위해서요^^;;'], ['체력보충 하셔야죵\n저도 다해놓고 쉴틈  있을때  쉬고있어요ㅋ'], ['저도 커피마셨는데 사진보니 시원한 아메 또 땡기네요^^한시간있음 하원시간이네요ㅜ어찌이리 빨리가는지요ㅡㄱ'], ['커피먹어도 사진으로보면 또 먹고싶지요\n저도 그래요\n시간 진짜 빨리가네요\n조금 쉬다가 씻고 데리러가야지요ㅋ'], ['하원시간되니 잠이 막 쏟아지네요ㅜㅜ마치고 또 집으로 안오고 놀이터 끌려가지싶어요ㅜ'], ['저두요 하원시간되면 그렇게 졸립네요ㅡ\n하원후는 전쟁이잖아요ㅜㅜ\n곧 전쟁 시작입니다'], ['저두 놀이터 한시간 놀다와서 밥먹이고 씻기고 육퇴만 기다리고 있네여ㅜ'], ['한시간이나 힘들어겠어요\n전 오늘 빨리가자고 했어요\n딸이 수업이 있어서요ㅋ'], ['끼니걱정좀 안하고살고싶네요ㅋㅋㅋ저는믹스한잔마셨어요'], ['저두요ㅜㅜ\n뭘하지 고민하고있네요\n애들은 어제남은 볶음밥 주까싶은데\n신랑은 뭐해주나요?ㅜㅜ'], ['저는 신랑이 오늘 안들어와요ㅋ그래서 애들이랑 간단히먹었어요'], ['야근인가요?\n아님 야갸인가요,\n저희신랑은 회식이라고 하더라구요\n그래서 앗싸 를 외쳤답니다ㅋ'], ['당직이였어요ㅋ저도신랑없는날 완전좋아요 입꼬리가 올라가요 ㅋㅋ'], ['ㅋㅋ 저두요ㅋㅋ\n회식한다고 좋아했는데\n12시 1시넘어도 2시가 되어도 안오네요\n화가 엄청나게 났죵ㅋ'], ['보기만해도 시원하니 아메한잔하고프네요 저녁에는 무얼먹어야할지요'], ['그러니꺄요 숙제가 생겼네요\n고민하다 애들은 어제 남은 볶음밥\n신랑은 뭐해줄까요?'], ['아이고 아아로 충전중이신가봅니다 ㅎ 저도 늘 고민이네요 ㅠ 오늘은 김밥이나말아볼까합니다ㅋㅋ'], ['오~~김밥^^\n전 애들은 어제남은 볶음밥주고\n신랑  저녁은 뭐할지 고민이네요ㅜ'], ['먹고 싶네요 ㅠ'], ['앗 한잔 드릴까요?\n커피 오늘 내려서 향이 좋고 맛있답니다^^'], ['커피 컵이 너무 이쁘네요,. 저도 저녁을 뭘로 해야 하나 고민에 들어갔습니다.'], ['요거 잔 맥주잔도 하고 아이스커피도 마시고 하네요\n여기에 마시면 맛도 더 있는거 같아요ㅋ\n저두요ㅜ애들꺼는 결정했는데요\n신랑꺼는 고민이네요ㅜ'], ['피곤해도 집안일 모두 끝내고나서 션하이 마시면 기분 짱 좋지요ㅎㅎ 그나저나 저도 저녁 뭘해먹을지..'], ['맞아요\n오전부터 바쁘게 움직이니 여유가 조금 생기더라구요.30분정도ㅜㅜ'], ['할일 모두 다 끝내고 쉬면 좋지요.. 저도 그래야하는데 전 그런날도 있고 아닌날도 있고 그래요ㅋ'], ['저도 그래요\n오전에 농띵부리면 오후에는 마음이 급해지죵ㅜㅡ바쁘게 움직입니다'], ['그지요.. 그래서 일찍 서둘러서 하고나서 쉬면 좋은데.. 그걸알면서도 매번 부지런하지 못하다는 함정이ㅋ'], ['전 제성격이라서 부지런히 움직입니다ㅋㅋ\n그래서 살뺀이후 살이 안찌네요\n체력이완전 바닥이예요ㅜ'], ['언제쯤 밥걱정 안하고 살 수 있을까요? ㅋ \n아가씨 시절 엄마가 뚝딱 내놓던 엄마밥이 그립네요 ㅎ'], ['아 저도 그래요\n이제는 제가 뚝딱 만들어야지요\n매일 걱정입니다~^^'], ['저도 저녁 메뉴 생각중인데 아무것도 생각이 안나요. 뭐하실거에요? 따라쟁이할랍니다 ㅋㅋ'], ['전 오늘 신랑  회식한다고 하네요\n앗싸 캤지요\n신랑꺼는 안해두 되고\n애들 어제한 볶음밥 줄려구요ㅋㅋ\n편하지요ㅋ'], ['저는 신랑 회식하는 날은 더 싫은데.. 그냥 밥만 먹고 온다고 해야 아싸지요 ㅋㅋ'], ['그렇긴해요\n늦게오고 술잔뜩먹고 왔지요\n그래서 토요일하루 주방요리사했지요ㅋ'], ['그래도 늦게 오고는 하루 주방 요리사하고 괜찮네요. 오늘은 요리사 안했어요?ㅎㅎ'], ['일요일은 머했지?\n기억이ㅜ\n늦게와서 저녁 애들 계란볶음밥 신랑하고 전 애들 씻겼지요ㅋ'], ['아이스커피 시원해보이네용~^^ 감기로 당분간 커피 끊었더니 더 간절해지네용~^^ 오늘 저녁은 어제 동생이 가져다준 비싼수제햄부터 조리해야겠어용~~^^'], ['아궁 ㅜ 감기로 커피 못 드시는구나ㅜ\n비싼수제햄 저도 먹고싶네요ㅋㅋ\n인증샷 올립니까?올리면 댓글 남겨주세요\n구경갑니다'], ['동생이 수제햄먹으라고 가져다준것 오늘 두개 구워서 저녁먹었어요^^\n제입맛엔 맛나네용♡'], ['오~~인증샷 좋습니다ㅋㅋ\n앗 저 그햄 먹어본거 같아요\n동생이 회사에서 받았다고 줫거든욪ㅋ'], ['맛나죠~~!!저두 밖에나갔다가 방금 집안일다햇네요~~ㅋㅋㅋ이제쉬어야겠어요'], ['전 잠시 여유부리구요\n그것도 30분밖에 없네요ㅜ\n곧 전쟁 시작입니다'], ['축하드려요~~♡ 시간아 천천히~~♡'], ['주문걸어주세요~^\n잠시뿐이더라구요\n딸 먼저하원하고 두아들 기다리고있어요'], ['집안일 완료하고 아이스아메 한잔 드시는군요 얼음동동 시원해보여요'], ['네 시원하게 한잔 마시고 씻고 애들 데리러갔죠\n근데 또 믹스아이스 태워서갓답니다ㅋ']]
    
    1999
    집안일 시작~ 미세먼지 최악인데 집안꼴도 최악이에요아직 아침안먹어서 배는고프지만 집안일부터해야겠어요
    
    [['청소하려면 힘이있어야 하실텐데..!힘내세요!!'], ['넹 힘내서하고잇어요 ㅎ'], ['환기도 못 시키고... 공기청정기만 열심히 돌아갑니다 ㅠ\n저두 청소해야겠어요'], ['저도요 요즘 공청이 열일하고있어요'], ['요즘 미세먼지 너무 안 좋네여ㅜㅜ 환기도 못 시키고.. 배고프실텐데 아침겸 점심 든든하게 드세요~'], ['네 이제청소마무리하고 먹을까해요 맘님도 맛점하세요~'], ['먹구하시지ㅠㅠ저는어제집청소했네용ㅋ'], ['몇일째제대로 못치워서 넘 심난해서요 ㅎㅎ 청소기 후딱돌렸어요'], ['빈속에 힘드시겟어요 ㅜ ㅜ'], ['배고픈데 집안꼴이넘어수선해서요ㅜ ㅎ일단정리하고청소기만돌렸는데도 엄청후련해요'], ['고생하셧어요'], ['네 이제 애기깰때까지 좀 쉬어야겠어요 ~'], ['네네 쫌쉬세ᄋᆞ'], ['네~~'], ['힘내세요! 깨끗해진 모습 보면 또 뿌듯하죠ㅎㅎ'], ['넹 청소기만 돌렸는데도 후련하네요 ㅎㅎ'], ['ㅠㅠ 식사하고 천천히 하셔요'], ['일단청소기만돌렷어요 ㅎ 애기깨서요 ㅠ 나머지는 밥먹고 해야겠어요'], ['아 ㅋㅋ 맛점 하세요 ㅋ'], ['넹 맛나게먹고 집안일 마무리 햇어요 ㅎㅎ'], ['오 ㅋㅋ 빠르시네요 ㅋㅋ 저는 먹고 설젖이 쟁여 놨어요 ㅋㅋ'], ['저도 설거지 쟁여놨다가 애기가 잘자줘서 후딱했어요 ㅎ'], ['환기도 못시키고 나가지도 못하고 저도 청소나 해야겟어요'], ['그니까요 환기못시켜서 공청틀엇어요'], ['저희집도 해야하는데....\n미세먼지가 좀 좋아짐 좋겠어요. 문 활짝 열어놓고 해야 청소할 맛도 나는건데 말이죠 ㅎㅎ\n무튼 청소기도 돌리셨겠다 본격적으로 청소하시고 맛점하세요~'], ['맞아요 대신공청틀어놧어요 \n청소기만 우선 돌렸고 나머진 밥머고 해야겠어요 ㅎ 맛점하세요~'], ['장군이 쉬고있어요? 힘내요 ㅠㅠ'], ['네 모빌보면서 놀고있길내 청소시작했어요 \n정리하고 청소기돌렸을뿐인데 후련하네요 나머진 밥먹고 상황봐서하려구요'], ['와.. 대단해요... 화이팅'], ['ㅎㅎ 집안일 다했어요 ㅎㅎ 속이후련하네요'], ['오오 멋져요.. 무리하시는거 같은뎅 ㅠㅠ'], ['몸은힘든데 너져분한거보는게더힘들어서요 안그래도미세먼지가극성인데 \n남편은 한다하고 안해요 요즘 청소는 좀 헤이해졌어요'], ['오 우리 하숙생은 진짜로 잠만자고 씻고 밥만먹고 나가요.. 요즘 청소가 먼지 잊어버렸나바요'], ['에고 저희집양반이랑똑같구만요 굴러다니는 먼지들이안보이나봐요'], ['근데.. 아침 풍경은 다들 비슷하겠죠?\n오늘은 갑자기 화가나더라그요.......'], ['왜요 먼일있으셧나요?'], ['특별히 먼일은 없는데.. 인나서 씻고 밥먹고 바쁘게 나가는 사람이 왜케 하숙생같던지 ㅎㅎㅎ'], ['ㅎㅎㅎ 그냥별다른거없이밥만먹고 바로나가셧나요'], ['늘 그렇됴 머 \n인나서 씻고 밥먹고 나가기 ㅎ\n걍 얘기 좀 하고 인사하는 정도죠 머 ㅎ'], ['일을 넘 많이하시는거같아요 ..'], ['힘든사람이에요\n직장도 반포까지 가야해서 1시간 넘게 출퇴근하구....ㅎ\n머 승진파겠죠 ㅎㅎ'], ['헐 출퇴근부터가일이네요 ㅜ'], ['그죠 힘들어요오옹 ㅠㅠ'], ['에고 고생하시네요'], ['저도아직집안최악인데ㅜㅜ애가아침부터징징ㅜ'], ['에고 아침부터 엄마 힘들게하는군요 ㅜ 저도 하는도중 찡찡대서 청소기만 돌렷어요'], ['전이불빨래중이요ㅜ건조가오래되네요'], ['에고 이불빨래 힘든데 고생하셨어요 ㅜ'], ['아직도건조기돌아가요 이불3번째나눠하는데이불모드는건조기3시간씩돌아가네요ㅋㅋ'], ['헉 엄청오래걸니네요 빨래다끝나면 하루가 다지나가겠어요'], ['저도 집안꼴이 최악이네요ㅡㅠ'], ['그런거보면 넘 답답하고 심난해요 ㅜ 그래서 맘잡고 치웠어요'], ['출산하기전에  치워놓고가고싶은데\n왜케 기운이 없을까요?ㅠ'], ['일단 몸이 무거워서 움직이기도 힘든게 젤큰이유 아닐까요 날씨도 흐리고 ㅜ 막달은그냥가만히만있어도 힘들어요'], ['그런거같아요\n누워있는것도 힘들어요ㅠ'], ['맞아요 걷기도힘들고 눕기도힘들고 자세바꿔도힘들고 다힘들죠'], ['출산은 무섭고ㅠㅠ'], ['아 정말두렵죠 그래서더긴장되구요 ㅜ'], ['상상이상이래요ㅠ'], ['저도 그런줄알았는데 막상 겪어보니까 그정도는 아니었어요 넘걱정마세요~'], ['막연한 공포감이 엄청나네요ㅠ'], ['이제얼마안남아서더그런걸수도있어요 ㅜ'], ['10월까진 괜찮았는데\n11월 들어와선 장난 아니었네요'], ['점점 예정일가까워올수록 그런거같애요'], ['저희집도 최악이라쥬ㅠ해도해도 끝이없어요ㅜ'], ['그죠 식구들이많아서 일거리도 더많겄어요 ㅜ 맨날하는데도 왜케끝이안나는지모르겄어요'], ['그러니까요 ㅠㅠ 징해요 ~ 집안일하다가 끝나요맨날 ;'], ['맞아요 그건모든집이다똑같은거같애요 ㅜ'], ['그러니까요 ㅠㅠ넷째낳으면 이제 카페 들어올수있을지나 모르겠어요ㅠㅠ'], ['그러게요 ㅜ 저는 애기낳고 한동안 못왔어요 몸도힘들고 애기도 계속봐야되고 도무지 시간이안되더라구요 ㅠ 가끔들어와서 눈팅만햇는데 얼마나아숩던지요 ㅜ'], ['그러니까요.. 한동안 그럴꺼같아서 .. 급 섭섭해지려해요ㅠㅠ'], ['에구 아직시간이있자나요ㅜㅜ  지금을즐기세요~~'], ['저도 이렇게 앉아서 노트북하니까 ,. 조마조마해요매일 ㅠㅠ'], ['에고 카페들어올날이 얼마남지 않았네\n이런생각드시나요?'], ['그런것도 있고 , 조산기로 입원하면 어쩌지.....싶어요 ㅠㅠ\n저 셋다 조산기로 입원했다가 ,34주35주에 낳았거든요ㅠ'], ['에고 정말요? 셋다그러셨으면 더걱정되시겠어요 ㅜ 더더욱 무리하시면안되겠네요ㅜ'], ['그래서 지금 늘 불안하긴해요ㅋㅋ'], ['에고 지금몇주되셨어요?'], ['월욜되면 34주예요ㅋ'], ['얼마안남으셨네요 그래서더걱정하시는군요 \n3주무사히 넘겨야 안심되시겠어요 ㅠ'], ['그정도되면 수술할꺼같아요ㅋ\n37주에할예정이라ㅋ'], ['유도하시는거에요?'], ['수술이유 ㅠ'], ['아 수술하신다고 쓰셨는데 제가 잘못봣네요 ㅠ'], ['ㅋㅋ괜찮습니다 ㅋㅋ\n저도 착각할때 많아요 ㅠ'], ['이해해주셔서 감사해요~^^'], ['세번째 수술인데도, 참 두근거리네요ㅜ'], ['경험해봐서 더 두근거리는거아닐까요 ㅜ'], ['첫애들때 수술한게 약간은 생생해요ㅎㅎ\n배가르는 느낌을 그대로 느껴서ㅠ'], ['헉 마취해도 느껴져요?'], ['네네 , 처음에는 느껴졌다가 , 나중에 안느껴지더라구요 ㅎㅎ\n근데 그게 너무 생생하게....ㅠㅠㅠㅠㅠ'], ['아 느끼고싶지않은 느낌이네요 ㅜ 마취가 들된상태에서 시작하셧나 ㅜ 어찌그걸느끼죠 ㅜ'], ['그랫나봐요ㅋㅋ진짜 배가르고ㅋㅋ안쪽 뜯는느낌이 낫다가 ..\n아아아..하고 잠들엇는데 애기낳앗더라구요ㅋ'], ['으으윽 마취했어도 어느정도 고통은 느끼신거네요 ㅜ'], ['네네 , 근데 셋째때는 먼저 재워달래서 ㅋㅋㅋ느낌 1도안났어요 ㅎ'], ['아하 첫째때 경험이 있으시니까 셋째때는 미리 재워달라고 하셨군요 ㅎㅎ\n그 느낌 또 느끼는건 싫죠 ㅠㅠ'], ['네네ㅋ수술실이 춥기도추워서..\n긴장하고잇는데, 추우니까 덜덜덜거렸어요ㅎ'], ['맞아요 춥더라구요 \n온도좀 높혀달라니까 신생아기들 한테 맞는 온도 설정해놓은거라고 안된다해서 \n저도 덜덜떨었었네요 ㅠㅎㅎㅎ'], ['흐어 .. 그게 너무 싫은거같아요 ㅋㅋㅋ\n무섭기도한데 ,춥기도하고ㅠㅠ어후'], ['그죠 가뜩이나 긴장되는데 춥기까지하니까 좀 무서웠어요 ㅎ']]
    
    2052
    이제 집안일.. 무슨 빨래가..해도해도ㅋ끝이없쥬?ㅠㅠ따님셋-하루에한벌씩(학교입고간거)밖에나갔다오면 하루에 두벌씩도가능ㅠ 속옷도하루에하나씩남편님-작업복 속옷수건은 5명이 하루에 쓰는게 7장ㅋㅋㅋ거의 그정도하나요?애들옷빨래, 우리옷빨래,수건빨래전부나눠서 하는데ㅠ귀차나요ㅠ그냥같이하고픔ㅠㅠㅠ
    
    [['으아.... 지겨워라....\n저도 하루에 세번 할 때도 있어요... 겨울되니까 더 많아보여요 ㅎㅎㅎ\n맘님 진짜 힘들겠어요'], ['ㅋㅋㅋ징해요 징짜 ... 건조기가 있어서 다행이지 ..\n없었으면 맨날 징징거렸을꺼같아요ㅠㅠ'], ['ㅎㅎ 저도 수건은 하루에 세네개정도씩 나오다보니 건조기돌려요 ㅎ'], ['제가 요새 씻는것도 힘드니까 , 덜씻었더니,7장씩나와요 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ\n예전에는 하루에 두번씩씻었는데 ..... 배가 너무 나오니,.씻는게 왜케힘드까요ㅠㅠ'], ['아하하 다행이네요 맘님이라도 덜써서 ㅎㅎㅎㅎㅎ'], ['ㅋㅋㅋㅋㅋ예전엔 하루라도 안씻으면 난리낫엇는데ㅡ.ㅡ'], ['나도나도 ㅎㅎㅎㅎ\n지금은 추워서 피하고.... 괜찮은거같아서 피하고.... 하하하'], ['ㅋㅋㅋㅋㅋㅋ아근데, 씻으면 다리는 못씻으니까 그것도 짜증나고 ㅠㅠ\n앉아서 씻자니 .. 그냥 돌이라 찝찝하고 ㅋㅋㅋㅋㅋ\n언니는 어케 씻어요 ? 별아빠님이 씻겨주시나요 ?+_+'], ['난 욕실에 목욕탕 의자 사다놨지 ㅎㅎㅎㅎ 그러게 별아빠는 왜 발한번 씻겨준적이 없지.....하하하 아 전에 있었구나.... 우씨...ㅎ'], ['아하 , 욕실의자도 사놓으면 됐을텐데 ..멍청했군요 ㅋㅋ안씻어진다고 ㅋㅋㅋㅋ혼자그러다 \n닿는곳만 씻고 ..근데 요새 씻고나옴 너무 추와요 ㅠ'], ['어차피 발도 좀 담그거 하려면.. 그게 필요할거 같아서..'], ['글긴할꺼같아요 ㅠ.ㅠ\n이제 밥시간됐어요~!! ㅜㅜ'], ['저녁준비 ㅠㅠ 난 요즘 저녁준비 안해서 좋아 ㅎㅎㅎㅎ.'], ['흑흑 부러운말중에 젤 부럽네요ㅜ'], ['아침준비만 열심히 ㅎㅎ'], ['아침이 더싫을꺼같아요ㅜㅜ애들 밥줄때 어우ㅜ'], ['그지.. 지금 11시에 나가니 다행인데... ㅎㅎ\n쏙 인나서 씻고 아침먹고 나가는 하숙생같아 ㅎ'], ['저희집도 하숙생한분계시죠~ㅋㅋㅋ\n저녁밥만 먹습니다 ..간혹 본인이 해서 먹기도하죠~ㅋ'], ['아하하 너무 웃기넹\n그래도 잘해줘야지 그만큼 힘들텐데 ㅎ'], ['그쵸 ㅋㅋ서로 잘하긴해야하는데 .. 그게 참어렵네요~ㅠㅠㅠ'], ['누구든 어려운거 같아.. 나도 그래..'], ['그러니까요 .. 초반에는 안구랬는데 ㅠㅠ히융~'], ['초반에만...ㅎㅎ\n초반엔 다 글치 머 ㅎㅎ'], ['그러니까요 ㅋㅋㅋ아 ,. 초기때로 돌아가고싶다요~ㅋㅋ'], ['ㅎㅎㅎ 늘 초기때같은면 좋은디 뭐든 ㅎㅎ'], ['그러게말입니다 ㅠ'], ['그게 안돼 사람이라... 다들 그렇게 살거야.. 너무 힘들어하지느..'], ['그러게요 ... 안싸우는 부부들은 어떤삶을 살까요?ㅋㅋ부럽네요 ㅋㅋㅋ'], ['안싸우는 부부가 어딨어\n그냥 내색을 안할 뿐이지...\n우리도 맨날 싸워 ㅎㅎㅎ'], ['그러까요?아 노력을한다고 하는데도 ..잘안되니 ㅜㅜ'], ['다 본인들만의 사정이 있는거니까.....\n세상 좋은 사람들도 쌓아놓고 사는 사람도 있고.. 그걸 꺼내서 싸우고 사는 사람도 있고......'], ['그러니까요 .. 사람은 본인만이 아는거라지유~'], ['더 힘든 사람들도 많을거야.. 우린 양호하다고 생각하고 사는거지'], ['그러니까요..아근데 나는양호하다고 생각하지않은데ㅠㅠ\n남편월급이 작아서 ...하루하루가 비틀비틀 ㅠㅠ'], ['에효... 어뜨케... 맞춰서 살아야지.... \n산교갔는데 보험하는 사람이 자기 연봉 6억이라고.... 내가 속으로 저런건 왜 나한테는 안왔지...ㅎㅎㅎㅎ'], ['ㅋㅋㅋㅋㅋ연봉6억인데..어쩌라는거지? 라고 저는 생각했을텐데 .ㅎㅎ\n저 보험회사 2년했지만 ... 6억이라고 뜨는사람은없었는데 ..1억5천인가는 있엇궄ㅋ'], ['아하하 수령액이 6억이라던데.. 뻥쟁이가 ㅎㅎ'], ['저희 한번씩 타지역가서 교육듣고올때면\n그런얘기해주거든요ㅋ어찌어찌해야 돈을 버는지등등ㅋ'], ['아~ 근데 보험해서 6억벌면.. 나도 하고싶다 ㅎㅎㅎㅎㅎㅎㅎㅎㅎ'], ['글믄 진짜 나도 계속햇쥬ㅋㅋ\n내친구는 빚진경우도잇어요ㅜ'], ['으아... 영업은 하는 사람이나 하지... 못해'], ['ㅋㅋㅋ나도 돈은 벌긴했는데 .. 그만큼 힘들기도햇고ㅠㅠ'], ['아 대단해.. 난 죽어도 뫃할거 ㅎㅎ'], ['글지글지 ..그거 영업적인거라... 말도 잘해야하고.. 구걸아닌 구걸도해야함 ㅠㅠ'], ['으아.. 더럽고 치사해 ㅎㅎㅎ'], ['진짜 .. 어마무시하지유.. 진짜 더럽고 치사한직업이긴한듯;~!!!'], ['그래 영업이 얼마나 힘든건데...'], ['맞아맞아 ..진짜 힘두렁ㅋㅋ'], ['영업하는 사람들 진짜 대단해'], ['나도 2년동안하면서 진짜 드럽고 치사하고 ㅡㅡ'], ['그만하길.잘했어..\n진짜 우리나라에서 서비스직은 점점... 특히 영업직은... 아니 누구 아랫사람도 아닌데..'], ['그래도 그일을 하면서 애들을 잘키웠쥬~!!'], ['참 진짜 대단하다... 진짜 세아이 키운거 존경받을만 한거..'], ['별거시기라서 돈도안주더라고ㅡㅡ\n지새끼테 나가는돈마져..'], ['애초에 그런사람이었네....'], ['그런거같음 ~ 진짜 드릅고치사해서 영업하면서 돈벌어서 애들 키우고ㅠㅠ\n막내원비라도 내달라니까 ㅋㅋ내가 정해서 보낸곳이니 내가 해결하랰ㅋㅋㅋㅋㅋㅋㅋ\n심지어 애들 폰요금이라도 내달라니까 .. 그것도 내가 한거니 내가내랰ㅋㅋㅋ\n원비야 20만원이지만 .. 폰요금은 3만원인데 ㅡㅡ'], ['으아.. 아빠자격이 없는 사람이었네\n진짜 원에 있음 그런 전화 가끔 와.. 아빠한테 보내지 말라..엄마한테 보내지말라.. 이런거.. 경찰도오고 막.... 가정불화로 막 그런거 .. 진짜 몰래 데려가고 나쁜 사람들도 마나... 그냥 그런 사람이라 관심 없는게 다행일수도있럴'], ['그거 생각하면야 .. 진짜 다행중다행인데 ..\n14살에 데리고간다는말 들으니까 .. 아오 빡쳐서 ㅡㅡ'], ['참나.. 접근금지 신청해... 무슨 자격으로 데려가'], ['내년에나할까해~ \n양육비도 더 높게신청할꺼구!'], ['그래 할 스 있음 해야지\n그러다 진짜 나쁜맘 먹으면 무서우니까..'], ['아,애들 성이랑 친권바꾸고싶은데ㅜ\n그건 절대 동의안해줄듯하고ㅜㅜ아오짜증ㅠ'], ['그게 이혼했는데도 동의가 필요해? 그럼 새 가정을 꾸리면.. 전 남편 동의를 얻어야해?'], ['응응~!! 그래도 동의가 필요하는부분이야 .. 우리나라 법은 완젼 거지가틈 ;;'], ['와 이미 정리를 했는데.. 양육권도 준거자나.. 친자라 그런가... 이상하다 그건\n그럼 새가정이 불편하자나'], ['14살에 데리고 간다고 ㅈㄹ중임... 그래서 내년에는 친권이랑 포기하게끔 소송하던할려구~!!'], ['웃겨 머해줬다고..\n진짜 접근금지해야겠다\n아니 그동안 잘하고 친권에 대해 목맸음 말을 안해\n머 다키워놓으면 데려간대?'], ['ㅋㅋㅋ그래서 몇달전에 다 포기하랬더니 .. 데리고갈꺼래ㅡㅡ\n이번에 막내 치과비용좀 같이하자했더니 읽씹..ㅋㅋㅋㅋㅋㅋㅋ하하하\n안해줄꺼알고 보낸거긴했지만 .. 참 ..'], ['ㅎㅎㅎ 그런거 다 증거로 모아놔.. 개뿔 신경도 안쓰고 지돈 아끼면서 애들을 왜 데려거'], ['전 아직 콩콩이 태어나기전이라 아직은 많지않네욤\n\n아기 옷들 손수건 수건 양말등등 빨게ㅠ ㅠ\n\n오늘 하루도 끝까지 마무리 잘하세욤 ㅎ ㅎ ㅎ'], ['저희는 초딩님들 빨래인데도 많네요 ;\n이제 넷째태어나면 더많겠지요 ?ㅠㅠㅠ'], ['저도 옷빨래 애기빨래 수건빨래나눠서해요ㅋㅋ요즘은 하나더붙어서 애기이불빨래요ㅋㅋ'], ['저도 꼬봉이태어나면 ,. 더 많아지겠지요 ? ㅠㅠ 으허ㅜㅜ'], ['ㅋㅋㅋㅋ하루에다섯번하실거같네요'], ['최고많이햇던게 세탁기6번돌렷던적이잇엇죠ㅋㅋㅋ'], ['곧최고10번이되실수도ㅋㅋㄱㅋ 빨래만돌리다하루가시겠어요'], ['아앜ㅋㅋㅋㅋㅋ 생각만해도 너무 끔찍한대요 ?ㅋㅋ'], ['으아~빨래하다 하루다가겠어요ㅠ'], ['이제 집안일 다하고 쉬네요ㅠㅠ'], ['빨래하다가 하루 끝나겠네요ㅠ'], ['그러니까요 ;;빨래개는것만 2시간하는듯해요 ;ㅋㅋㅋ'], ['고생이 많으세요ㅠ'], ['ㅋㅋ괜찮아요 ㅋㅋ건조기 덕을 많이봐서~ㅋㅋ\n'], ['전 건조기가 없어요'], ['나중에 기회되시면 꼭 사셔요 ㅠㅠ'], ['빨래하다가 하루다가겠군요'], ['그러니까요ㅠㅠ 건조기가있어서 얼마나 다행인지 몰라요ㅠㅠ'], ['빨래를 하루에 세번이상 돌리는집이 대부분일꺼예요'], ['그쵸? 아이고 ,.뻐치네요 진짜 ㅜ'], ['저희은 하루에 두번에서 세번돌려용'], ['ㅋㅋ저희는 최대 6번까지 돌려봣어요 ㅠ'], ['식구가많은집은많이 돌리더라고요'], ['네넹ㅋㅋ그런거같아요ㅜ'], ['사람이 많으니 배네요^^'], ['그러니까요 ;ㅋㅋ 힘드네요 ㅠㅠ'], ['저흰 남편이 수건을 두번써서 별로 안나와요\n전 하루 한개 ㅎㅎㅎ \n빨래는 두 그룹으로 나눠서 빨구여~\n아이 태어나면 횟수가 어마어마해지겠쥬?ㅜㅠ'], ['저도 지금 걱정이랍니다 ㅜㅜ 우리 초딩님들 그리 깔끔하지도않은데 ;\n옷은 왜케 잘갈아입는건디 ... ㅜㅜㅜ'], ['ㅋㅋㅋ나 초딩때는 그냥 입던거 입었던 것 같은데\n은근히 깔끔쟁이들인가봐요~\n저도 미리 겁먹어서 건조기라도 사놓으려구요ㅎㅎ'], ['건조기 진짜 사세요~ 전 건조기 득을참많이바요~ㅋㅋㅋㅋㅋ'], ['네, 안그래도 가격 좀 안떨어지나하고 체크하고 있어요.\n이달안에 사려구요~\n특가로 좀 가격 빠졌음 좋겠어요ㅎㅎ']]
    
    2362
    오늘은...아줌마 되는날??ㅎㅎ^^ 오늘은 월차라서 집안일 했습니다~올해는 월차도 다못쓰고 몇일 남았네요~하...쓰레기 버리고,설거지,빨래,집안 쓸고닦고 정리하고~음식물 쓰레기도 버리고~~집안일이 끝이없네요~~엮시 힘드네요~그냥 일하는게 더 쉬운듯...대한민국 주부님들~정말 대단하십니다~!!이제 한숨돌리고 밥먹네요~^^;그냥 귀찮아서 컵라면에 공기밥~ㅎㅎ나가기도 귀찮고~비도내리고~영화보면서 뒹굴거려야겠습니다~^^날추운데 건강조심하시고 점심맛있게드세요~^^오늘도 이쁜하루 되시길 바람니다~^^
    
    [['멋저요'], ['바람의 아들님~잘지내시죠?오랜만이신듯 합니다~^^추운날씨에 건강조심하시고 점심 맛있게드세요~^^항상 감사드립니다~^^'], ['예\n기냥 숨만 쉬고 있어요\n서천 눈코만치 눈이 왔내요'], ['여기도 눈이 약간내리더니 지금 빗방울 떨어지네요~뵙고싶습니다 멋지신 바람의 아들님~^^'], ['왕뚜껑 인가요?'], ['네 그냥 귀찮아서요^^;;점심 맛있게드시고 즐거운 하루되세요 사오리님~^^'], ['와 대박 멋집니다 전안하거든요'], ['별말씀을요 박산초님~^^그냥 심심해서요..^^날씨가 장난아닙니다 건강조심하시고 점심 맛있게드세요~^^'], ['전심심해도안합니다 ㅎㅎㅎ그래서 더대단하단겁니다 추운데 감기안걸리시게 쉬엉쉬엄 하루쉬세요'], ['대단하십니다~^^멋지시구요~박산초님도 감기조심하시고 파이팅넘치는 하루되세요~감사합니다~^^'], ['저도 도와주긴해야하는데.\n'], ['말나온김에 오늘저녘에 도와주심이...^^;;힘드네요..ㅎㅎ점심 맛있게드시고 행복한 하루되세요 골드king님~^^'], ['멋짐폭발!!! 맛있게 드시고 꿀휴식 취하십시용~~~'], ['네 럭셜구찌님~^^간만에 낮에 집에있으니 이상하네요~ㅎㅎ그리고 럭셜구찌님이 진짜 멋짐폭발~이시죠~^^점심 맛있게드시고 오늘도 기분좋은 하루되세요~^^항상 감사드립니다'], ['저도 간혹 휴가내고 집안일하다보면, 주부들이 대단하다고 생각됩니다.ㅎㅎㅎ 즐건 연차 보내세요 키류님~~'], ['엮시~승빠님도 잘 도와주시는군요~^^집안일 힘드네요 주부님들 정말 대단하세요~항상 감사드립니다 승빠님~점심 맛있게드시고 오늘도 행복한 하루되세요~^^'], ['우리집장관님은 안합니다..\n그래서 제가합니다..\n본인이 청소할테니 애보라고하길래..제가한다고 했습니다 ㅋㅋㅋ'], ['캬~멋지십니다~고도리님~!!^^근데..애기보시는것보다 청소하는게~~ㅎㅎ^^점심 맛있게드시고 편안한 하루되세요~^^말씀 감사합니다 고도리님~^^'], ['어허~~~좋은 아빠에 좋은 남편까지 이러시면 곤란합니다 ㅋㅋ'], ['아 그렇게 되는건가요?ㅎㅎ^^감사합니다 영진아범님~^^영진아범님도 분명 멋지신분 같습니다~^^점심 맛있게드시고 행복한 하루되세요~항상 감사드립니다~^^'], ['저두\n한번씩 할만 하드라구요...ㅎ'], ['엮시 멋지십니다~!!^^저는 너무 티도안나고 그래서 힘들더라구요^^날씨가 상당히 차갑습니다 감기조심하세요~점심 맛있게드시고 오늘도 기분좋은 하루되세요 율하스텝님~^^'], ['키류 아주머니~~~가까운곳에 계시면 커피 한 잔 하는건데~~~ 아쉽네요~ㅋㅋ'], ['그러게요~써니님~근처시면 바닐라 라떼라두 한잔 하는데요~^^언젠가  기회가있겠죠~^^점심 맛있게드시고 이쁜하루 되세요~^^감기조심은 필수요~^^'], ['정말 집안일이 젤 힘든거 같습니다~\n영화재밌게 보세요 키류님~~^^'], ['네 태우기님 티도안나고 힘드네요^^;항상 감사드립니다 점심 맛있게드시고 파이팅넘치는 기분좋은 하루되세요~^^'], ['라면갖고 됩니까??밥을드셔야죠'], ['네 쩔지주현님 공기밥도 한그릇 했습니다^^;점심 맛있게드시고 오늘도 기분좋은 하루되세요 항상 감사드립니다 쩔지주현님~^^감기조심하세요~'], ['맞습니다...집안일,육아... 제일 힘듭니다... ㅠㅠ'], ['네 멋진제네글님~일본은 잘 다녀오셨나요~집안일은 티도안나고 육아는 더 힘들고...고생많으십니다 점심 맛있게드시고 오늘도 파이팅넘치는 하루되세요 멋진 제네글님~^^'], ['고생하셨습니다.키류님.라면에.공기밥.최고죠.즐오후 보내세요'], ['감사합니다 이쁜남님~^^티도안나고 힘만드네요ㅎㅎ~^^;;점심은 드셨나요?어제보다 더 추운듯합니다~항상 건강조심하시고 행복하고 파이팅넘치는 오후되세요~^^'], [''], ['항상 감사드립니다 조박사스텝님~^^점심식사는 하셨습니까~!눈도내리고 영하의날씨에 항상 건강조심하시고 오늘도 행복하고 기분좋은 하루되세요~^^'], ['평일날 쉬고싶다 ㅜㅜ\n'], ['좋기는한데 심심하네요 얼짱 문늬중님~^^;이쪽으로 오셔서 저랑 같이쉬시죠~ㅎㅎ^^;;점심은 드셨죠?날씨가 춥습니다 건강조심하시고 오늘도 기분좋은 하루되세요~^^'], ['라면엔 역쉬 식은밥이죠\n역쉬 식도락을 아시는분답게 맛나게 드시네요\n집안일은 돌아서면 생긴다고 하니 돌아서지 마세요'], ['역시~흰둥이 G님이 잘아시네요~^^라면에는 식은밥이죠~^^말씀대로 돌아서면 또 생길듯하여 그냥 누워있습니다~ㅎㅎ^^눈까지내리는 날씨에 감기조심하시고 즐겁고 신나는 하루되세요 항상 감사드립니다 흰둥이 G님~^^'], ['독고남 일인으로서 많지 않은 집안일이지만 할때마다 힘든거 같습니다 ㅠ 고생 많으셨네요!\n'], ['네 홍홍님~집안일이 티도안나고 끝도없고 힘드네요 홍홍님 항상 고생많으십니다 대단하시구요~^^눈내린날씨에 건강조심하시고 행복하고 기분좋은 하루되세요~항상 감사드립니다 홍홍님~^^'], ['열심히일한당신 떠나세요~  ㅋㅋ'], ['지금 뒹굴거리며 누워있는데 호야님뵈러 떠나야것습니다~ㅎㅎ^^날씨가 많이춥네요~항상 건강조심하시고 오늘도 행복하고 편안한 기분좋은 하루되세요 호야님~^^감사합니다~'], ['왕컵, 왕뚜껑 저도 좋아합니다. ^^\n저도 매주 일요일은 주부되는 날입니다. ㅡㅡ\n와이프가 일을해서요. 청소, 빨래, 애들목욕은\n힘들지 않는데 밥먹이는게 젤루 힘듭니다.ㅜㅜ\n애들반찬은 할때마다 고민입니다. ㅋㅋ'], ['매주 일요일마다 주부~대단하십니다 가람님~^^엮시 멋짐폭발이세요~^^그렇죠 아이들 식사,반찬이 힘들죠 지금은 커가지고 괜찮은데 아들 꼬마때 혼자있을때 애먹었던 기억이나네요^^;;항상 좋은말씀 감사드립니다 가람님~눈내린날씨에 건강조심하시고 행복하고 즐거운 기분좋은 하루되세요~^^'], ['키류님~ 고생 많으셨습니다. 힘드시죠? ㅎㅎ\n저는 매주 분리수거, 주 1회 청소(청소기로만), 월 1회(스팀청소) 이 외 애들 밥이나 제가 먹을 간식이나 음식은 직접 만들거나 챙겨 먹고 있습니다.설것이 포함. 물론 장 보는 것도 제가 주로 하구요. 절대 가정적이거나 와이프를 생각해서가 아니라 그냥 타고난 제 팔자인것 같습니다.ㅋㅋ 니가 안하면 내가 하고말지뭐.... 이런타입이요.ㅎ'], ['와~훌륭하십니다~깡통제네님~!!결코 쉬운일이 아니신데요~사모님의 사랑을 독차지하실듯 합니다~^^깡통제네님께 비하면 저는 병아리...^^;인성이 참 좋으셔서 그러신거죠~엮시!^^한수배움니다 깡통제네님~^^오늘하루도 고생많으셨습니다 저녘 맛있게드시고 행복하고 포근한 기분좋은밤 되세요~^^'], ['대단하십니다~~^^ 해보면 당근히 힘든건 아시는거지만~~ 그것을 진심으로 알아주시는걸 더 고마워 하실겁니다~~^^\n진심이 느껴지니깐요~~♡\n아내분도 카페 가입해서 이런 키류님 마음 보시면 흐뭇해 하실 것 같네요^^ 편안한 밤 되세요~~♡'], ['이미 가입했을지도 모름니다 미니천사님~^^산타페신형이면서 다시 G70을 눈독드리고 있어가지고...저의꿈은 자꾸 멀어져가네요....ㅎㅎ^^;;미니천사님의 넓으신 상품과 인격을보면 사모님과 자녀분에게 최고이실듯 합니다~^^주옥같은 좋은말씀 늘 감사드립니다 기온이 장난이아닙니다 항상 건강조심하시고 즐겁고 편안한 기분좋은밤 되세요~^^'], ['공감합니다~~\n컵라면은 역쉬 왕뚜껑 이죠'], ['네 뉴비님~컵라면은 왕뚜껑~^^공감 감사합니다 뉴비님~^^올해들어 가장 추운듯합니다~건강조심하시고 행복하고 즐거운 불금되세요~^^'], ['전 평소에........ㅠㅠ'], ['헉!대조영함장님~평소에 하시는군요~엮시 대단하시고 멋지십니다~!!^^사모님과 자녀분들에게 최고의 아빠신듯합니다~^^날씨가 장난아닙니다 건강조심하시고 행복하고 기분좋은 불금되세요 항상 감사드립니다 대조영함장님~^^'], ['왕두껑 좋습니다 ㅎㅎ바쁜하루보내셨네요'], ['멋진시아님 추운날씨에 고생하셨습니다 어제보다 더 춥네요~꼭 뵙고싶습니다~올라가면 꼭 연락드릴게요 낚시하시느라 피곤하셨을텐데 푹쉬시고 행복하고 즐거운 불금밤되세요~^^']]
    
    2388
    남편이랑 보이지 않는 전쟁중입니다. 전업주부하다가 남편의 강권으로 취업했습니다.취업한지 이제 1주일인데요. 여전히 집안일은 제 일인 것으로 인식하더군요.아니겠지라고 생각하다가 1주일동안 집안일엔 신경도 안쓰는 거 보고(아, 한가지 했네요. 빨래 개기만 한거요. 가져다놓지 않고요.)지금 저도 아무것도 안하는 중입니다.취업하고 맞는 첫 주말이라 남편이랑 얘기해서같이 집안 정리 해야겠다 생각했어요.화장실청소,설거지,음식물쓰레기버리기, 싱크대 및 가스레인지 청소, 빨래, 와이셔츠 다리기이래해야한다고 말해줬더니 나중에 하자 하고 자버리대요? 저도 같이 잤습니다. 토욜 지나고 일욜 돼서눈 뜨자마자 밥먹자 이러길래빵도 있고 떡도 있고 밥도 있으니 알아서 챙겨먹으라했더니쟁반짜장 시켰다고 합니다.ㅋㅋㅋ 옳다구나 저도 옆에서 몇젓가락 뺏어먹고그 후에 집안일 좀 같이 하자 했더니좀 쉰다네요ㅋㅋㅋ 그 후로 계속 잡니다.저도 잤습니다.이게 현재까지의 일입니다.여전히 집안은 엉망인데 계속 버텨야겠죠?
    
    [['버티세요 이겨내시길!!!!!!!'], ['ㅋㅋㅋㅋ 양말 와이셔츠 없음 누가 손해인지 몰겠네요'], ['버티세요 무조건이요 ㅎ 타협도하지않으려하시는거같으니..'], ['그러게요. 자기도 저 일한다는 거 알아서 인지 그냥 얼굴에 짜증만 올라와있네요ㅋㅋ'], ['맞벌이이면 당연히 집안일 나눠해야줘~ 남편이랑 상의해보세요 어떤거할래? 구체적으로 정하자~ \n이런식으로 말해보세요~'], ['두 번 정도 말했는데 나중에 하자고 그래서요. 그리고 계속 자버려서요ㅋㅋ'], ['본인이 일하라고 해놓고 이제와서 이런행동은 진짜 짜증나는데요;; 그냥 님도 아무것도 하지말고 똑같이 하시는게 답이네요~'], ['그래도 자기보다 못버니 해줄줄 알았나봐요ㅎㅎ 이겨봐야죠'], ['근데 그렇게해서 집안꼴이 엉망이 돼도 남자들은 끝까지 신경 안쓰더라구요. 못참는 여자의 몫이 되고ㅠ'], ['저도 그럴 것 같아서.. 그래서 걱정되어서요. 말은 상황봐가면서 하면 되지라고 하는데 왜 그 상황은 자기한테 유리한쪽으로 해석되나 몰겠네요.'], ['맘님이 편하신대로 하셔요~\n남편분은 맘님이 생각하고 계시는것처럼 깊은생각은 못하시지않을까요ㅜ\n대화를 해보시는게 어떨까싶어요^^'], ['알고 있긴 한 것 같아요. 오늘 말투나 얼굴 표정에 짜증이 올라왔더라고요.'], ['와 막상막하~꼭 승리하세요'], ['저야 맘만 불편하지 집안일 안하니까 넘 편하네요^^'], ['버티세요~~ 같이 일하는데 집안일은 왜 여자만 해야하는거죠ㅡㅡㅋㅋ 남편분도 알면서 일부러 더 안하고 버티는거같아여 누가 이기나 해보자 하고요'], ['네 자기도 아니까 저한테 화도 못내고 그냥 자기 혼자 짜증나서 툴툴대고 있는 것 같네요.'], ['잘하고 계십니다. ㅎㅎㅎ\n언제까지 버티나보자~ㅇ 심정으로 사세요 ㅋ'], ['네 그럴려고요~ 당장 양말 와이셔츠 없으니 자기가 짜증나겠지요'], ['남자들 모르는척 하는것뿐 다 알고 잇어요.다만 내  일이  아니다..하고 신경끄고 잇는거에요.자의도 아니고 남편에게 떠밀려서 일하시는거라면 끝까지 잘 버텨서 합의를 이끌어내시길..^^'], ['그러니까요. 자기일 아니고 배려해준다고 생각하는 것 같아요. 제가 맞벌이만하면 상황에 맞춰 나도 하겠지라더니ㅋㅋㅋ 웃기네요.'], ['무너지시면 안됩니다 버티세요!!!!!@'], ['감사합니당!!!'], ['나중에는 별향님 빨래와 자녀가 있으시면 자녀꺼까지만 하세요~남편분도 지금 기 싸움 하시는것 같은데 남자들은 바보 같은것 같애요ㅎ'], ['남편이 바보같은게ㅋㅋ 저는 애를 거부하고 남편은 애 낳자는 입장인데도 이러네요ㅋㅋ'], ['아...더 불리한 입장인데도 그러네요?ㅎ왜그럴까...그럼 별향님껏만ㅋ빨래를..ㅋ'], ['ㅋㅋㅋㅋ 제가 갑질한다네요'], ['ㅋㅋㅋㅋㅋ무슨 논리로 갑질하신다는거죠?ㅋㅋ집에서 일하는게 갑인건가요?ㅋ저도 후기가 너무 궁금해지네요! 정말 화이팅이에용!'], ['아뇨 너가 잘해봐 내가 애 낳아줄게 이렇게. 애 낳는 능력있는게 갑질이래요ㅋㅋㅋ 꼭 후기 올릴게용'], ['ㅎㅎㅎㅎㅎㅎ지금 남편분 하시능거 보면 애 낳으면 집안일 손도 안대실거 뻔하고..애기 낳으면 육아스트레스 받는거 이해 1도 못하실분 같은데욯ㅎㅎ'], ['인내하여 승리하셔요!!^^'], ['혼자 얼큰한 라면 한사발 했습니당 고맙습니당'], ['하 진짜 나쁘네요....응원합니다'], ['ㅋㅋㅋㅋ 저흰 싸울때도 우아하게 싸워서 더 꿀잼이네용'], ['이긴 후기 듣고싶습니다 ㅎㅎㅎ'], ['ㅋㅋㅋ 후기 올리겠습니다!!'], ['못 견디는 사람이 하게되있죠 ㅠㅠㅠ 근데 보통 애들생각 위생생각해서 여자가 못 참으니 문제죠 ㅠㅠㅠㅠ'], ['다행히 애는 없네용ㅋㅋㅋ 전 원래 좀 더러운 편이라 끝까지 가볼라고요'], ['응원합니다!! ㅋㅋㅋㅋ'], ['다행히 낼 출근이라 좋네요'], ['응원합니다ㅠㅠ!!!!'], ['ㅋㅋㅋ 살짝 속은 쓰리네요 달달한게 땡기는거 보니'], ['오오옷!!!  격하게 별향님 응원합니다.맞벌이라면 집안일도 함께해야죠.!!! 암요 암!!! 홧팅!!!'], ['응원 감사합니다^^'], ['화이팅입니다! 꼭 이기셔야해요!'], ['넹~~ 서로 아무렇지 않게 대화하구 있어용ㅎㅎ'], ['버티셔야해요...\n더러워서 내가한다\n하시는 순간..모든 집안일은 님의 몫입니다..\n화이팅'], ['어휴 그렇겐 안되죠ㅎㅎ'], ['남자들 진짜 얄밉네요.  왜  자기무덤을 자기가 팔까요? 응원합니다!! 이번에 지면 일에 집안일까지 하셔야하니 꼭 이기세요!!'], ['네 제가 대학원까지 다니고 있어서 집안일 독박이면 진짜 안될 것 같아요'], ['아까 댓글달았는데 나중일이 진짜 궁금해서 댓글 다시남겨요 ㅋㅋㅋ 꼭 이기셔서 후기도 남겨주세용!!!'], ['ㅋㅋㅋ 중간 중계 해드리면 전 라면 먹었고 (같이 먹자 깨웠는데 안일어나대요) 저 라면 다먹고 밥 안먹냐 했더니 배 안고프대요 한참 있다가 일어나더니 담배피러 밖에 나가나했더니 밥 먹고 온다 톡왔네용ㅋㅋ 전 티비보고 놀고 있습니당.'], ['아이없을때 정해야지 아기낳으면 그냥 지는거에요..ㅠㅜ 끝까지 버티시고 이기세요!!  응원요^^  전 시댁문제로 결혼초에 결정못내고 져줬더니 시댁인간들 저를 빙신으로 아는건지..결혼 14년차..이제야 시댁인간들 안보고 기싸움중..저도 이겨볼랍니다~~  누가 손해인지 모르는 사람들.. 정말 바보같아요..'], ['피임 열심히 하고 있고 아이문젠 늘 최종결정권자는 여자라고 말하고 있습니다.  배려해주면 그게 당연한거라고 생각하나봐요. 맘님도 꼭 이기셔요!!!!'], ['화이팅! 이겨내셔서 후기 들려주세요~'], ['저 완전 응원합니다~~~'], ['견디세요. 전 싱크대 그릇에 곰팡이 필 때까지 견뎠습니다ㅋ'], ['더럽거나 귀찮거나 불편함을 못참는사람이 하게되는게 집안일이에요.\n전 꼭 참아내셨으면 좋겠네요.\n\n이게 맘 카페라서 그런게 아니고.\n맞벌이는 원하면서 월급의 차이가 있다라던지 일의 강도 차이가 있다면서 부인에게 과한 요구를 하는 남자들이 잇어서.. 인식이 아직은 시대를 못따라와 속상해서요ㅠㅠ'], ['견디셔요~ 저는 더러운 꼴 제가 못봐서 제 무덤 팠습니다. 남편은 더러운거 진짜 잘 참더라구요ㅠ 결혼전부터 더럽게 해놓고 살던데 그때는 그게 크게 안보였어요ㅠ 그러더니 이젠 완전 제 일이 되었어요. 더 웃긴건 시댁에서 여자가 하는게 당연하다 말하심요ㅠ.ㅠ'], ['꼭 버티세요!!!']]
    
    2390
    82; 일단 클리어했는데- 원래의 계획은 이시간에 클리어와 퇴근을 동시에 하려 했으나,예상치 못하게 클리어만 먼저하네요ㅠㅠ아직 많이 남아있는데 말이죠ㅠㅠ뭐.. 잠시 집안일하고 오면 뭐 다시 쌓여있으니그럼 퇴근과 동시에 새벽반 입성하면 되겠어욤 ㅋㅋㅋ저 일단 집안일 & 내일 친정갈 짐도 좀 싸고저녁도 먹고 오도록 하겠습니당 ㅋㅋㅋ조금있다가 다시 만나요^^
    
    [['네 조금있다 만나용'], ['네넹 ㅋㅋ 근데 아직도 카페에서 방황하고 있는 저랍니당 ㅋㅋㅋ'], ['아직 1000개 달성 못하셨군요'], ['네~ 아직이요 ㅋㅋ 이제 얼마 안남았어욤 ㅋㅋㅋ'], ['끝까지 화이팅입니다'], ['이히히~ 화이팅 해야지욤 ㅋㅋㅋ'], ['네~~1등을 위해ㅋ'], ['ㅋㅋ 감사합니당^^'], ['오늘도 화이팅입니다'], ['맘님도 오늘 화이팅 하시옵소서 ㅋㅋㅋ'], ['ㅎㅎ네~~좀만 놀다가야죠'], ['ㅋㅋ 맘님 왠지 조금이 아닐것 같은데요 ㅋㅋ'], ['저 지금 엄청 밀렸어오ㅡㅡ'], ['ㅋㅋ 맘님도 오늘 엄청 달리셔서 그래요 하루종일 ㅋㅋㅋ'], ['하 정말 끝도 없습니다'], ['그래도 하다보면 언젠가는 클리어한답니당 ㅋㅋㅋ'], ['하 지쳐요ㅋㅋ'], ['ㅋㅋ 지칠만 하시죠 ㅋㅋ 얼마나 많이 있으시겠어요ㅠㅠ'], ['아 끝이 없어요ㅋ'], ['설마.. .끝을 보실라고 하신건가요? 탈퇴하시기 전까지는 끝이 없을걸요 ㅋㅋㅋ'], ['그런가요 탈퇴는 노노~~~'], ['ㅋㅋ 말이 그렇죠 ㅋㅋ 탈퇴는 안합니당 ㅋㅋㅋ'], ['ㅋㅋ그렀죠 이리 좋은곳을'], ['ㅋㅋ 오히려 이런곳을 늦게 안걸 후회하고 있다지요 ㅋㅋㅋ'], ['저두요 전 11월에 산교 후기쓰면서 알았죠ㅜㅜ'], ['진짜요? 그럼 저보다 더 늦게 아신거네요~\n전 산교 가입할라고 알게된건데... 8월에 ...'], ['아 네 그래서 저번달도 11월 중순부터해서 넘 아쉬웠어요'], ['이번달은 순위권이 목표신건가요?'], ['네 매일 들어오니 10위 안에는 들겠죠??'], ['일단은 매일 댓제는 못해도 500개정도씩은 해야될것 같아요~\n이번달이 은근 달리시는분들이 계셔서요~'], ['그렇죠 열심히해야겠군요'], ['이게 쉽지가 않더라구요~ 왠만큼 해서는 순위권 들기가 힘들어요ㅠㅠ'], ['앗 정말요\n깨비맘님은 순위 들어보셨어요??'], ['저 5등은 해본적 있어요 ㅋㅋ'], ['오~~~5등요 그때도 지금처럼 치열했어요??'], ['그때는 지금보다 더 햇던거 같아요 ㅋㅋㅋㅋ'], ['근데 댓제 1000개 아니예요??  저 1000개도 안했는데 걸려서요ㅜㅜ'], ['다른카페랑도 댓글연동이라 그래요~'], ['앗  진짜 다른카페 후기도 작성해야하고 바쁜데 곤란하군요ㅜㅜ'], ['그럼 일단은 여기서 할수 있는 최대한을 하시는게 좋아요 ㅋㅋㅋ'], ['그래야겠네요ㅎㅎ'], ['꼭 순위권에 드시길 바랄게요 ㅋㅋㅋㅋ'], ['네  꼭 그러고 싶습니다ㅎ'], ['맘님 화이팅~^^'], ['넹~~화이팅입니다'], ['퇴근을 못하셨군요... 오늘 제가  뜸했지요~~'], ['언니가 오다가 안오다가 해서 그래욤 ㅋㅋㅋ'], ['아토리가 짬을 안줘요^^;;;'], ['아... 아토리 미워요ㅠㅠ 아토리가 샘을 내나봐요 ㅋㅋㅋ'], ['폰 보고 있음 얼굴을 막 들이대요~ 근데 그게 엄청 귀여워요^^'], ['아웅 ㅋㅋ 상상이가요 ㅋㅋ 귀요미 아토리 ㅋㅋ 아우 이뻐 ㅋㅋㅋㅋ'], ['그 귀염둥이 새벽 4시에 깨서 혼자 엄마하면서 놀고 있더라구요ㅋㅋ'], ['아이고 ㅋㅋ 울지도 않고요? ㅋㅋ 착하네 ㅋㅋㅋㅋ'], ['착하다니요~~~ 텐트 밖으로 혼자 기어나가서 놀고 있는데 깜놀했어요'], ['아... 그 난방텐트요? 그거 집에 하셨어요?\n그럼 놀랄수밖에요ㅠㅠ'], ['네~ 따수미 매일 치고 자요~ 소리는 들리는데 애는 없고 완전 깜짝 놀랬어요^^'], ['ㅋㅋㅋ 진짜 식겁하셧을듯 ㅋㅋ 혼자 뭐하고 놀고있었대요?ㅋㅋㅋ'], ['구석에 떨어진 장난감 주워서 기어온던대요?'], ['아.. 저때 일어나서 그런거죠? 지금 일어나서 그런게 아니고? 아고 놀래라;;;'], ['네~ 새벽에요~ 얼떨결에 텐트 밖으로 나가서 놀았는데 들어오진 못하고 나중에 엄마 부른듯해요^^'], ['혼자 나가서 놀기도 해요? 일어나서요? 우와~ 깜깜할텐데 안무섭나?'], ['어둠에 대한 무서움이 아직 없대요~ 그뒤론 못나가게 텐트 바깥쪽을 막아놨어요^^'], ['아... 애들은 그렇대요? 어둠을 안무서워한다니 ㅋㅋ'], ['네~ 그래서 수면교육할때 불 다 꺼도 괜찮아요~'], ['안그래도 그게 궁금했어요~ 언니는 수면교육 언제 시작하셨어요?\n전 8주쯤 시작할까 하는데 그래도 괜찮을까해서요ㅠㅠ'], ['지금도 업어서 재우니까 수면교육 안시킨거 같은데 잠잘 시간엔 이불 펴놓고 무조건 불다끄고 티비 끄고 눕히던지 어부바해요~ 늘 듣는 자장가 틀어주거나 노래 불러주고요~ 그럼 이제 자야 하구나 인지해서 자기도 준비하는듯요.. 시기는 통잠잘때부터요~ 아토린 4개월부터 밤수 안하고 통잠잤거든요. 그때부터 시작했어요^^'], ['와~ 통잠을 일찍 했네요~ 진짜 통잠만 자줘도 좋을텐데요ㅠㅠ 전 따로 할생각인데 시기는 일단 지켜보고 해야겠네요~'], ['네~ 아가 상황보구요~^^ 안그럼 역효과날듯요~'], ['그니까요~ 일단은 지켜보라고 하더라구요~ 애기성향이 어떤지를'], ['맘님은 지혜롭게 잘 하실거예요~^^'], ['그럴까요? 잘 할 수 있을까요? 벌써 걱정이네요ㅠㅠ'], ['닥치면 하게 되고 힘들면 카페에 도움청하면 되고 걱정마셔요~'], ['그럴게요~ 많이 물어볼테니 도와주세욤~^^'], ['네네~ 잘하실거면서^^'], ['못해요ㅠㅠ 전 완전 초보맘이자나요ㅠㅠ'], ['친정엄마도 계시구 맘님은 야무져서 잘 할거예묘^^'], ['제발 그랬으면 좋겠어요~^^'], ['ㅎㅎ 퇴근까지갈길이먼거는아니쥬? 저 지금 소식함 채워드리고 있어요 ㅎㅎ'], ['멀지는 않아요 ㅌㅋㅋ 금방 끝날것 같아요 ㅋㅋㅋㅋ'], ['내일 친정 가시는군요~~준비할게 많겠어요~'], ['뭔가 많으네요ㅠㅠ 몇달 집을 비워야해서 집정리도 다 해야되고\n짐도 싸야되고 정신이 없네요ㅠㅠ'], ['냉장고도 비워놓고 가야되고 비워두니신경 쓰이겠어요'], ['아무래도요 ㅠㅠ 그렇다고 중간중간 와볼수도 없고 ㅠㅠ\n애를 냅두고 몇시간을 어찌 나와있어요ㅠㅠ'], ['신랑도없고 그냥 몇달은 비워야죠ㅜㅜ 전기코드뽑고 안전하게만하고 가심 별문제 없을거예요~'], ['전기코드 다 뽑고 가스밸브 잠그고 문단속 잘하고 보일러 온도 낮춰두고 쓰레기 다 버리고~\n아마 중간중간에 신랑이 집에 잠깐은 왔다가 갈거예요~'], ['몇달살고 집에오면 온기도 썰렁하겠어요. 겨울다지나서 오면 다행이죠ㅅ'], ['그래서 저도 시간이 되면 중간에 한번은 와보려구요~ 아무래도 애기 델꼬 오기전에 집도 청소해야되고 ㅠㅠ'], ['아기오기전에 청소도 신경쓰이죠 비어두면 먼지가 제일ㅜㅡ'], ['그니까요ㅠㅠ 먼지가 대박일듯해요ㅠㅠ 그래서 오기직전이나 그전에 와서 청소한번 하고 갈라는데\n애기 몇시간 냅두고 와도 괜찮겠지요?'], ['ㅋㅋ오늘저녁메뉴는뭔가요~?숟가락들고렛츠궈궈!!!ㅋㅋ'], ['오늘은 그냥 볶음밥입니당 ㅋㅋㅋ 매콤한 ㅋㅋㅋ'], ['깨비언니의솜씨를아니그냥먹어도막맛잇을듯요ㅠ'], ['ㅋㅋㅋㅋ 저건 제 솜씨가 아니라 어제 먹고 남은 해물찜의 솜씨입니당 ㅋㅋㅋ'], ['새벽반을 해야 순위에 드나봐여 ㅋㅋ 그러고보니 새벽반을 한번도 한적이 없네요ㅜㅜ'], ['새벽반을 해야 그나마 댓글수량 채울수가 있으니까욤^^'], ['조금이따만나요'], ['네넹 ㅋㅋ 새벽반 하실거죰?ㅋㅋㅋ'], ['새벽반힘드네요ㅜㅠ'], ['그쳐? ㅠㅠ 오늘 많이 힘들어보이시는데 얼른 가서 쉬세요ㅠㅠ'], ['안가고달리네요ㅋㅋ']]
    
    2426
    주말부부의 신혼밥상 이에요 ㅋㅋ 서울과 광주를 오가는 주말부부에요.제가 와따가따 거려서 체력적으로 힘들지만매일매일 혼자 밥챙겨먹고, 집안일 다 해놓는 신랑이 고마워서주말에라도 맛있는거, 좋아하는거 해주려고 노력하는편이에요 ㅎㅎ사진보니까 크림파스타가 반...이네요?ㅋㅋ전 토마토파 라서 크림 만들어두고 신라면 끓여먹은 적도 있네요 ㅋ매일매일 메뉴 고민하시는 새댁님들 오늘도 화이팅하시고 감기조심하세요!
    
    [['음식솜씨 좋으시네요~ 저렇게 잘 만드시는 분들 보면 그저 부러울 뿐입니당ㅠㅠ'], ['맛난거 너무 잘해드세요!! 비주얼도 최고에용^~^'], ['어머 플레이팅이 넘 예뻐요!!'], ['우와 음식솜씨가 너무 부럽습니다ㅎㅎ'], ['매번 다른메뉴에 정말 맛있어보여요'], ['우앗 다 맛잇어보여요 ! 신혼신혼해요 ㅎㅎㅎ'], ['플레이팅 너무 예쁘고 메뉴도 너무 맛있어보여요~ 밥상에서 신혼느낌이 폴폴 나네요!!ㅎㅎ'], ['우앙 식기류 너무 깔끔하고 좋네요~ 음식도 다 맛있어보여요^^'], ['힘이 날 것 같아요 ^^\n또 신혼의 힘이 밥상에 차린 밥심이죠'], ['ㅎㅎ,, 밥보다는 술상이 넘 좋네요,,,핳,,, 다 맛잇어보여요 플레이팅 잘하시네욥!'], ['우와~~ 요리솜씨가 넘 좋으세요!! 부러워요~~ 다 맛있어보여요^^'], ['와우 굿굿이네요 ㅋㅋㅋ 울신랑이 급 불쌍해지네요ㅠㅠㅋㅋㅋ'], ['우왕 요리솜씨 짱이세요ㅠㅜㅜ'], ['우와 진짜 밥상 너무 이쁘고 플레이팅도 대박인데요?? 전 그렇게 해준적이 없네요 ㅠㅠ'], ['너무 맛있어 보이는데요? ㅎㅎ 솜씨가 좋으신 것 같아요~ㅎㅎ'], ['플레이팅 넘 이뻐요ㅎㅎ 예랑이가 넘 좋아할것같아요ㅎㅎ'], ['비빔면에튀김보고 없던입맛 다시살아나네요ㅋㅋ'], ['음식솜씨도 대단하시고 플레이팅이 너무 이뻐요ㅎㅎ'], ['요리사이신거같아요 먹고싶어요 ㅠㅠ'], ['음식뿐만아니라 플레이팅까지 ㅎ 넘 예쁘네요 ㅎ'], ['음식솜씨 뿐만 아니라 플레이팅 센스도 있으신 듯 해요!!! :) 너무너무 예뻐요!'], ['음식솜씨가 너무 좋으시네요 ㅎㅎ 맛있겠어요!!'], ['대박... 반찬까지 넘나 먹음직 스러워요!! 저희집은 왜 저런 비쥬얼이 안나오는지 으흑'], ['플레이팅 느낌있게 잘하시네요!! 식기류도 다 예쁘고ㅎㅎㅎ밥 먹을 맛이  나겠는데요!?'], ['음식솜씨도 넘 좋으시고 플레이팅 솜씨까지 최고세요! 진짜 먹음직스러워보이네요.'], ['음식 정말 맛있어보여요 ㅎㅎ\n너무 부러워요 ㅎㅎ'], ['입이 떡벌어지네요 ㅎㅎㅎ 저희 신랑이 매우 부러워 하겠네요'], ['음식도 맛있어보이지만 역시 플레이팅이 한몫!\n그릇들도 참 예쁘구여 ㅎㅎ'], ['제 로망이에요☺️☺️\n저도 요리 잘해서 결혼해서 매일 맛있는거 해주고 싶어요:)'], ['깔끔하고 예쁜 밥상이네요 행복가득해보여요'], ['음식 솜씨도 좋으시고 플레팅 솜씨도 완전 최고에요!! 저도 결혼하면 꼭 부지런한 부인이 되고싶네여ㅎㅎㅎ'], ['전부 맛있어보여요~~ 깔끔하고 넘 예뻐서 눈호강되네요~'], ['행복해보이시네요ㅎㅎ 저렇게 음식 먹으면 사랑받는 느낌일것같아요ㅎㅎ'], ['우와 너무 깔끔하고 예뻐보여요 ^^\n예랑에게 이런거 안 보여주고 있어요 ㅋㅋㅋㅋㅋ'], ['주방소품들 식기 다 예뻐요ㅠㅠ 재주가 부러워용!!'], ['플레이팅을 잘하시네요 ㅎㅎ 다 맛있어보여요!'], ['남편분 뿌듯하시겠어요~!!저는 요알못이라,,,,,ㅋㅋㅋㅋ걱정이네요ㅠㅠ 다른분들 하신거 보면 너무 부러워요~!'], ['음식솜씨 좋으시네요  맛이없을수가없겠어요ㅎ'], ['어머나ㅠㅠ 너무 귀여워요 아기자기해용!!'], ['우와 솜씨가 좋으세요!!완전 맛스러워보여용 ㅎㅎㅎ'], ['주말부부인데도 저렇게 여러가지 차려주시다니 좋으시겠어용! ㅎㅎ'], ['아기자기 예쁘고 맛나보여용ㅎㅎ'], ['플레이팅 넘나 이쁘게하시네요~^^'], ['술 눕혀서 플레이팅하신거 넘 귀여워용~~ㅎㅎ'], ['플레이팅 잘해놓으셔서 그런지 너무 예뻐요! 주말부부라 힘드실텐데 서로 대단하세요!>.<'], ['우완 플레이팅이 넘 이쁘네요~~\n음식 솜씨도 좋으신듯^^'], ['오아ㅜㅜㅠ플레이팅 진짜 잘하셨네용 테이블매트는 어디꺼에유???'], ['이마트에서 두장씩 들어있고 양면인거.. 몇천원 안주고 샀어요 ㅎㅎ'], ['와음식플레이팅짅짜예쁘게잘하신는거같아요 저는..ㅠ_ㅠ플레이팅도 요리도...영..'], ['어머 그릇이 진짜 너무 깔끔하고 이쁘네요~!'], ['같이 뭐 만들어먹는 재미가 넘넘 좋은거같아요><ㅋㅋ살은 찌겟죵...ㅋㅋㅋ ㅠ.ㅠ'], ['플레이팅도 잘하시고 음식도 넘 맛있어보여요^^'], ['음식솜씨가 대단하시네요!! 너무 맛있어보여요'], ['엄청 이쁘게 차리고 드시네요ㅋㅋ 전 똥손이라 뭐 신혼스럽지 않게 가득가득 담아서 먹어요ㅋㅋㅋㅋ ㅠㅠ'], ['다 맛있어보여용ㅎㅎ 플레이팅도 아기자기 넘 귀엽네용 :)'], ['캬~ 엄청 다양하게 하셨네요~!!! 칭찬해요~~ 오모오모! 솜씨가 굳입니다 ㅎ'], ['플레이팅이 완전 취향저격이에요!ㅎㅎㅎ 예쁜 그릇에 정갈하게 담긴 음식이 넘 맛나보이네용~ㅎㅎ'], ['어머머 요리책 내셔도 되겠어요 그릇도 예쁘네요 ^^ 행복한 밥상이네요'], ['음식 너무 잘 만드시네요~ 플레이팅 너무 이뻐요!!^^'], ['와 깔끔하니 진짜 잘하시네용 ㅎㅎ 넘 예뻐요'], ['넘 맛나게 해서 드시네요~~대단해요'], ['우오왕 완전 솜씨가 좋으시네용부러워용!'], ['우와 다양하게 이것저것 다 잘 만드시네요.ㅎㅎ'], ['우와~~너무 맛있어보여요! 금손이시네용^^'], ['맛있는거 많이 해드시네요~저는 반성 좀 해야겠어요ㅋㅋㅋ'], ['세팅이 돋보이는 신혼뿜뿜 밥상이네요'], ['식기류도 예쁘고 음식들도 너무 맛있어보이네요~~!!'], ['오와 ㅎㅎ 어떻게 저렇게 요리를 잘하시는거에용!! 비법좀 알려주세요 ㅎㅎ 저는 너무 비루하네용 ㅎㅎㅎ'], ['뭔가 귀여운 한상차림이네요 >_< 맛있겠어요ㅎㅎ'], ['진짜 정갈하고 예쁘게 꾸며드시네요 대단하세용'], ['요리도 잘하시는데 사진도 잘 찍으시네요~저도 저렇게 해먹을  수 있겠죠? ㅋㅋ'], ['오와 정말 맛잇어보여요!!! 솜씨가 좋으셔용 ㅎㄹ'], ['아정말 ㅋㅋㅋ 완전 다 맛나보이잖아효'], ['우앙 ㅋㅋ넘 이쁘게 드시는거 아녜요?? 전 진짜 요리꽝인데 큰일이에요 ㅜㅜ 예랑이가 저희 엄마 음식 솜씨만 보고 닮았을까 하고 기대하구있어요 ㅜㅜㅜ'], ['어머진짜금손이시네요 ㅠㅜ'], ['이야 진짜 금손이에요ㅋㅋㅋㅋ엄청 잘하시는데요?'], ['중간중간 소주와 맥주가 출현하네요 ㅋㅋㅋㅋㅋ 알콩달콩 신혼느낌 뿜뿜 >_< 주말부부인데도 정말 예쁘게 차리고 드셔요 굿굿!!! ^^'], ['ㅎㅎㅎ어머 음식이 참 예쁘기도하네요!ㅎㅎㅎㅎ'], ['우와 !진짜 잘해드시는거같아요 ! 자랑할만 해요  ㅎㅎ 저도 이렇게 해줘야하는데,,하'], ['깔끔하게 잘 차리셨네용, ㅋㅋ 저는 플래이팅에 재주가 없는건지 손이 큰건지 ㅋㅋ왕창 담아서 보기가 ㅠㅠ'], ['갈수록 술상이 보이는 ㅎㅎ 술상 차려먹는게 신혼잼 아니겠어요?ㅋㅋ'], ['계란후라이 틀 사용하시나봐요 동글동글 예쁘네요 ㅎㅎ\n혹시 전복은 뭘로 구우셨어용~? 살 떼서 칼집 내고 구운담에 다시 담으신건가요~? ^^\n이번 주말 메뉴로... 몇 개 참고하고 갑니다~ 맛난 신혼 생활 하세요~~ ^-^'], ['저도 저기 껴주시면안되나요!!ㅎㅎ 완전~ 부럽네요'], ['ㅎㅎ 맛있어 보여요~ 깔끔하게 잘해드시네요!'], ['우와 잘해드시고 그릇도 이뻐요~'], ['엄훠 ㅎㅎㅎ 요리에 일가견이 있으신 분이군여'], ['너무 귀여운 밥상이네요 ㅋㅋ 전 귀찮아서 그릇에 안담거든요ㅠ'], ['잘해드시네요~~ 맛있어보여요 ㅎㅎ  계란후라이가 넘 귀여워요'], ['그릇도 담음새도 넘나 이쁘네용!! 저도 주말부분데 ㅠㅜㅠ 왔다갔다 넘 힘들지만 그래도 보러갈땐 아직 설레는거같아요 ㅎㅎ'], ['오아 대박이시네요 ㅎㅎㅎㅎㅎㅎ 참, 저렇게 차려주는게 쉽지 않으실텐데- 넘 대단하세요 ㅎㅎ 저도 많이 노력해야겠네요 ㅎㅎ'], ['딱 신혼상 같이 아기자기하고 예쁘네요!'], ['플레이팅이 아기자기하게 이쁘게하고 드시네요 ㅎㅎ 전복구이 맛있겠네요 ㅎㅎ'], ['플레이팅 잘하셨네요`~^^ 넘 기여워요~첫 사진에 계란 후라이!!넘 귀여워요~'], ['이쁘게 플레이팅하시네용ㅋㅋ 저는 그런 재주가 없어서 ㅜㅜ'], ['어머 음식이 플레이팅덕분에 더 이뻐요 맛깔스럽네요ㅕ~ ~'], ['저는 술상에 눈이 더 가네요ㅎㅎ 츄릅ㅜㅜ'], ['우아~~정말 이쁘게하고 먹네요~전 먹기바빠서 플레이팅안하고 그냥 막먹어요~부러워용'], ['맛있게 플레이팅 잘하셨네요! 전 음식도 눈에 가지만.. 하나같이 다 이쁜 그릇들에 눈이 가네용! >< ㅎㅎ'], ['음싣 플레이팅 진짜 넘 이쁘네요~ 다 너무 맛있게 생겼어요ㅋㅋㅋ'], ['첫번째사진 반찬이 저희집이랑 90프로 씽크로율 ㅋㅋㅋㅋ 너무이쁘게 드시네용 ㅋㅋ']]
    
    2490
    ◈집안일 클리어 집안일 클리어했습니다~🤗🤗  청소기밀고, 물걸레질, 장난감 소독욕실청소, 빨래개기빨래 2번 돌리기중 마지막 한판 20분 남았구요^^;;;  이제 우리모녀,아점 먹을준비 해야겠어요♥  다들 점심 뭐드셨나요?맛난거 드셨나요?   카페댓글도 클리어하고 싶네요...😭😭
    
    [['우잇 대단하신데요 시간 정말 잘 갔을 것 같아여ㅜ많은 걸 하셨네요 대단하십니다'], ['집안일하는데 2시간 가까이 걸리니,\n집안일하고나면 하루가 빨리 지나가는듯해요'], ['오전은 정말 그거하다보면 시간이 금방 지나가죠 그럼 또 금방 아이 올 시간이고 ㅜ'], ['오전에는 집안일때문에\n하루가 다가고, 오후에는 세은이가 낮잠을\n3시간씩 자니깐 하루가 다 가는것 같아요'], ['우와 낮잠을 정말 길게 잣네요 ㅋ 저도 오늘 왠일로 막둥이가ㅜ잘 자서 저도 낮잠을 ㅋ'], ['세은이가 잠이\n많아도 얼마나 많은지 모릅니다ㅎㅎ\n낮잠은  무조건 3시간씩 꼭 자더라구요'], ['아이코 집안일 완벽하게 클리어 하셨네요 너무 개운하겠어요 슬리퍼보니 개운함이 느껴집니당 맛점드세요^^'], ['막상 시작전에는 귀차니즘 발동인데,\n움직이다보면 시간이 정신없이 지나가네요^^\n오늘도 집안일 끝내고나니 개운합니다요~ㅎㅎ'], ['아침일찍부터 움직이셨나봐요 부지런하십니다^^'], ['오전 11시 넘어서부터 시작했어요.\n카페놀이 안즐길때는 11시쯤이면 \n끝냈을 시간인데, 자꾸 게을러지는듯 하네요.'], ['오우 저도 맘먹으면 아침부터 움직이긴하는데 미세먼지에 오늘 날씨도춥고..만사귀찮아서 이불에서 이불콕이네용 ㅠㅠㅠ'], ['저도 이불속에서 나가기가 싫었는데,\n어차피 해야할일이기에 후딱 끝내놓고\n쉬자싶어서 귀찮아도 움직였었네요'], ['이아!! 대단하셔요!! 근데..청소 다하고 ㅋㅋ빨래 몇분 남았을때 요래 ㅋ카페놀이하면 너무 뿌듯하지 않나요?!?! 그럴때 있더라구여!'], ['네네~맞아요^^\n집안일 다 끝내놓고, 세탁기 끝나가길 기다리는\n그시간 너~~~무 좋지요^^\n뭔가 다 해냈다는 기분이랄까? ㅎㅎㅎ'], ['그죠!! 저만 그런거 아니였어용!! ㅋ왠지 세탁기가 ㅋ그 남은 시간을 기다리는게 어느순간 행복해지더라구요!ㅋ'], ['모든 맘님들이 같은 기분아닐까요?\n세탁기가 다되면 빨래만 널면\n오늘할일 끝이라는 생각에 괜히 \n기분이 들떠지는것 같더라구요'], ['오!!! 맞아요~ 그 마음 ㅋ진짜 느낌이 확 오네요!! 빨래까지 탁탁 털고 나면..진짜 오늘 끝이다!! 하지요..그러고...애 얼집 하원데리러 ㅠㅠ흑..ㅋ'], ['맞아요~!!! \n예전에 김혜수가 광고하던  "빨래끝~"이\n생뚱맞게 생각이 나네요ㅋㅋㅋ'], ['아 이거보니 화장실이 생각납니다 저도 청소하고 올께요'], ['맘님 화장실청소하러 가신겁니까?\n제가 곧 검사하러 들리겠습니다ㅋㅋㅋ'], ['네 완성샷은 없습니다 ㅋㅋ맘님 덕분에 화장실청소 했네요'], ['맘님네 화장실청소 끝낸거\n검사하고 왔었었네요~\n화장실청소하고나면 기분이 완전 개운하지요^^'], ['네 개운한데 안방화장실 청소는 못했었네요 피곤해여ㅡㅡ'], ['욕실청소 한군데만해도\n피곤한데, 두군데까지는 무리이기는 하지요'], ['아이고야 큰일 하셨네요..저도 오늘 환기함서 청소기 두번 돌리고.. 아이옷, 우리옷 세탁기도 두번.. ㅠㅠ'], ['늘~하는일인데도 왤케 귀찮은지 모르겠네요.\n저는 오늘 침대매트랑 이불빨래 다 끝냈네요.'], ['엄마야.. 오늘 대청소 하는 날입니까요?? 몸살 나신건 아니지요?? 오늘 고기 드시러 가셔야겠는데요 ㅋㅋ'], ['주말에는 신랑이 있어서\n청소 못하게해서 쉬지만은,\n평일에는 매일 대청소같은 청소를 대충하네요ㅎ\n'], ['매일매일 청소하는데 대청소같은 청소를 하십니까요?? 맘님도 스스로 힘들게 하는 스타일 같네요.. ㅠㅠ'], ['그렇지요???\n성격이 그렇다보니 안하면 몰라도\n뭘하기 시작하면 완벽하게해야\n직성이 풀리거든요'], ['오늘 집안일 너무 무리하신거아닌가요 장난감소독까지 대박 최고입니다'], ['늘~매일 하는일입니다^^;;;\n그래서 신랑이 주말에 청소못하게 하네요.\n가끔씩 소꼽놀이 장난감은 입에 넣기때문에\n매일 소독안해줄수가 없더라구요.'], ['우아 진짜 많이하셨는데요 ㅎㅎ 대단하십니다 ㅎㅎ 저는 아직 한게없네요 ㅎㅎ 얼른챙겨드셔용'], ['요즘 게을러져서 늦게하는거네요.\n예전에 카페놀이 즐기기전에는 10시전에\n집안일 다 끝내고, 하루종일 여유 즐겼었거든요'], ['그러시군요 ㅜ 저도 부지런히 움직여야하는데 그렇게 못하는거같아요 ㅜ 집안일이 늘 쌓여있어서 주말되면 엄청바빠요 ㅠ'], ['저는 주말에는 집안일 쉬어요.\n사실, 신랑이 못하게하네요ㅠㅠ\n도와달라고 하는것도 아닌데...'], ['맘님이 평일에 너무 열심히하셔서그런게 아닐까요 ㅜ 주말이라도 쉬시라구요 ㅠ 전 진짜 주말이 저 바쁘더라구요 ㅜㅜ'], ['아마두 그런가봐요.\n집안일도 쉬는날 있어야한다면서\n주말만큼은 집안일 하루 쉬어도 된다면서...'], ['우와.. 이렇게 청소를 끝내시니 제 속이 다 시원합니다..\n저도 욕실 청소를 주말에 해야겠어요.'], ['저희집은 욕실청소 매일 안하니깐\n물때가 잘 끼이더라구요.\n그래서 매일 안할수가 없네요ㅠㅠ'], ['맞아요.. 그리고 매일 씻고, 저같은 경우는 매일 머리를 감으니..'], ['그러니깐요.\n욕실은 가족모두다가 자주쓰이는 공간이니깐\n청소를 매일 안할수가 없더라구요'], ['맞아요.. 특히나 저희 애들은 신발을 안 신고 맨발로 욕실을 드나들거든요.'], ['아고~물기가 있음\n위험하겠어요.  욕실바닥이 미끄러우니깐\n항상 조심해야겠어요'], ['저는 집안일 클리어~~ 이러고 말 좀 할 수 있었으면 좋겠는데.. 화장실 청소는 진짜 언제할지요ㅠㅠ'], ['저희집은 화장실청소를 매일 안하면\n물때가 끼여서 안하고 싶어도 안할수가 없네요'], ['저희도 물때 있지요.그래도 매일은 힘들어요ㅠㅠ 일주일에 한번도 엄청 자주 하는거라지요ㅠㅠ'], ['눈에 안보이면 몰라도\n눈에 보이면 욕실갈때마다 신경 거슬려서\n몸이 힘들어도 청소해버리는게 맘편하네요'], ['맘님이 부지런쟁이라서 그런가 봅니다. 저는 천성이 부지런이랑은 좀 멀어요ㅠㅠ'], ['썩 부지런한편은 아니예요~\n정리하고, 청소, 빨래 요런것들만\n나름 부지런한편이예요.\n미루면 찝찝하고 그렇더라구요'], ['으지 집안일 마니하셨네요\n세은이는 집안일할때도 얌전하게놀구잇죠?'], ['제가 집안일할동안에는 세은이는\n자기 노는거에 충실합니다.\n한번씩 물티슈 뽑아서 엄마 돕는다면서\n바닥 닦아주고 그러네요ㅋㅋㅋ'], ['오 증말여?\n완전 다큰애같네용\n서아도 바닥은 잘 닼아죠여ㅎㅎㅎ'], ['나이는 분명 아직 아기아기한데\n하는 행동이나, 덩치보면 어린이같아요ㅋㅋ']]
    
    2562
    모닝 집안일~~ㅋ 식구들 모두 코~~하는동안조용히 사부작 커피 한잔 마시고~~♡세탁기돌려 빨래 널고빨래 개고초록이들 물한번 싹주고~~~♡이제 모닝와이드 뉴스 보러~~ㅋ오늘 할일 반은 끝냈네요ㅋㅋㅋㅋ모두 굿데이요♡
    
    [['새나라의 어른ㅋ'], ['ㅋㅋㅋ좋은하루 보냅시다♡'], ['하하님 너무 부지런 하심 아이들 때문에 빨래감 많이 나올때죠^^'], ['내복이 아침 저녁으로 나오니ㅜㅜ\nㅋㅋ 그냥 웃으며 합니다 hahahaha'], ['부지런 끝판왕.. 하루에 몇시간 주무세요?'], ['평균4ㅡ5시간자는듯요ㅋㅋㅋ'], ['대단하당.. 역시 부지런한 사람은 잠을 덜 자네요ㅋ 잠탱이는 반성합니당'], ['미인이시라서 그런거에요~~푹주무세용ㅋ^^'], ['미인은 아니예욬ㅋㅋㅋ좋은 하루 되세요!'], ['하하님의 부지런함은 못따라가겟네여ㅠ 좋은하루 되세여^^♡'], ['ㅋㅋ그래서 늘 하루가 길어요ㅋㅋ시간은 참 잘도갑니다 ㅋ좋은하루요'], ['나는 9시만되면 졸림ㅡ애들보다 더 빨리자~~~ㅎㅎ몸이 기억해 자동적으로 지금 일어는 나있는데...언능 밥 앉히고 쫌만 누워야겠음~수고햐^^'], ['ㅋㅋ나도  피곤할땐 일찍 자^^그럼 3시에 눈떠지고ㅋㅋ 화이팅♡'], ['빨래가 가지런하니 이쁘네요~~^^ㅎㅎ\n저희집은 초록이 출입금지입니다..\n왜 올때초록인데...조금지나면 노랑이가 될까요...ㅡㅡ'], ['초록이들도 나름 성격이 다르니께 쪼금씩의 공부는 필요한거같아요^^\n저의 힐링이들 초록이 ㅋㅋ'], [''], [''], ['대박이에요~저희식구 다 기절할 시간에 오늘할일 반을 끝내시다니...ㅋ저라면..일찍 일어나도 폰만 만지작 거리고잇을텐데~!!!^^;'], ['ㅋㅋ심심하니까..사부작^^\n어차피 할일 뒤로 미루도 해야하니까ㅋㅋㅋ'], ['ㅋㅋㅋ생각해보니 초록이들 물주는 것만 해도 시간 엄청 쓰실거같애요'], ['온 집안 구석구석에 있어서...ㅋ 찌끔 걸리긴하죠^^  틴란드시아는 스프레이도 해줘야하고ㅋ 팔떨어질듯해요ㅋ'], ['언니 도대체 몇시에 기상이신가요ㅠㅠ'], ['오늘 눈뜬거는4시쯤?ㅎㅎㅎ'], ['부지런바지런...\n그래서 살이안찌나바요'], ['ㅋ팔다리는 바빠서 팔목 발목만  살이 들쪘어요..배는 뚱뚱ㅋㅋ 좋은하루되세용'], ['4-5시간 자고도 정신이 돌아왔음 좋겠네여\n하루종일 정신 혼미할듯 ㅋ'], ['ㅋㅋ이것도 습관되니 아무렇지 않네요^^즐하루되세요'], ['부지런하시내요 \n~아침부터빨래만산더미니 ㅜ'], ['언능 세탁기로 고고고ㅋㅋㅋ방학이니 빨래가 맨날 리필되네요ㅎ'], ['전이제서 눈떴는데 하하님은 금손에 부지런하심까지있으시네요.'], ['푹자고 일어나면 좋지요~~^^개운한하루되세요'], ['부지런의 끝판왕이시군요 ㅎ'], ['ㅋㅋㅋ사부작사부작^^;;;즐하루되세여♡'], ['네네 HaHa님도 즐거운하루♡'], ['부지런하시네용~~~전...생각만하고..몸이 움직이질않네여ㅋㅋ'], ['저랑 반대.,ㅋ 쉬고 싶은데 몸이 먼저 움직이고있네요^^  화이팅요♡'], ['진짜...어쩜  이리  부지런할수  있죠..전  자꾸  깔아지기만  하는데ㅡㅡ..사진에  보여  질문인데요.틴란드시아는 어디서 구매하셨을까요?'], ['저는 많이 사기에는 가격이 부담되드라구요ㅋ 화원은 가격이 쫌 쎄서^^\n저는 위메땡ㅋㅋ'], ['ㅎㅎ감사합니다..거기서  사도  실한가봐요~'], ['포장도 잘되서 오고..관리하기 나름인거같아요^^'], ['소중한  정보  감사해요~^^'], ['역시 부지런끝판왕!! 전 아직도 뜨끈한 장판에 등지지고있는데..'], ['저는 할일 끝!ㅋㅋㅋ\n커피타임~♡\n'], ['아~~수목원..애들 견학한번보내도 될까요?'], ['ㅋㅋㅋ보내세요~~공주셋이 놀면되겠네요^^'], ['사진에 틴란드시아 색변한건 뭔가여?마른거에여?\n변해서 죽었다 생각하고 다 버렸는데 아닌건가여...?'], ['색변한것들..아주 예전에 처음 샀던 녀석들인데...제가 한동안 몸 안좋아서 스프레이를 못해줘서 메롱해졌는데..찌끔씩 초록한 애들이 삐져나오고있어서 그냥 지켜보고있어요ㅋㅋ'], ['분갈이 힘드실거 같아요\n저의 요즘 고민이라서요ㅜㅜ'], ['봄되면 날잡아서 베란다에서 해야죠ㅋㅋㅋ'], ['우와..어쩜저리잘관리를..부럽네용ㅎㅎ'], ['사랑의 힘으로~~~~~♡ㅋㅋ'], ['부지런한 HaHa언니♡\n전 오늘도 방전ㅠㅋㅋ'], ['어머 어쩜 ㅠㅠ 부지런함이 짱이십니다']]
    
    2584
    남편 집안일 글을보고서.. 결혼10년차인데저희 남편은 이제껏 화장실청소 한번 한적이없어요ㅜㅜ분리수거나 음쓰만 버리는정도고요.전 전업인데 너무 안시키는건가요?
    
    [['말도안되요..ㅠ 주중은 그렇다치고 주말은요 당연히 해야죠..'], ['제가 이상한가봐요ㅜㅜ\n화장실청소하란말은 안했지만\n하려하지도 않거든요..'], ['저는 반대로 12년차인데\n화장실청소 한번도해본적없어요 저도전업이요;;'], ['부럽네요ㅜㅜ\n제가 너무 안시키나봐요ㅜㅜ'], ['쓰레기는 주말에같이버려도..\n전 청소는 신랑이해요 \n화장실청소나,창틀청소,아이운동화빨기등등 손목힘든일.. 이런건 전부신랑이해요\n지금부터라도 조금씩 시키셔요~'], ['저도 안시켜요 대신 설거지랑 쓰레기만 시켜요\n어차피 딴거해도 다시\n제가 할테니ㅜㅜ 저 두개만 완벽하게 하라고 말햇더니 곧잘해용'], ['저도 반대로 화장실청소는 칫솔버리기전에 조금 문대고?버리는 정도밖에..\n남편담당이에요 쓰레기 분리수거 음식물 쓰레기 싱크대청소 가스렌지 전자렌지 청소 세탁기청소 등등 다 지저분한건 다 담당시켜요 ㅎㅎㅎ'], ['대박!최고예요!\n전 제가다해요ㅜㅜ\n\n'], ['애랑이라도 놀아줬음 하는데....ㅠㅠ'], ['아..대신 애랑은 잘놀아주네요.평일은 넘나바뻐서 주말만요~\n그외 전업인 제가 다하는걸로 아네요..\n'], ['가끔 씻고 욕조청소는 하고 나와요\n따로 화장실청소를 시킨적은 없어요\n전업이니..\n집안일은 기본적으로 저의 일이라고 생각하고 있구요\n분리수거와..힘쓰는일..기술을 요하는일등은 남편이해요\n가끔..거실청소정도요\n전업이긴한데..사실..\n저도 집안일을 잘하는편이 아니라서..\n반찬은 사먹고..\n식기세척기,로봇청소기,건조기가 있어서 집안일이 많이 수월해졌거든요..\n굳이 화장실 청소까지 일부러 시키고 싶진않아요'], ['저도 일부러 시키는편은 아닌데\n밑에글보니 남편도 집안일을 돕는구나..약간 마음의동요가 일더라고요~\n\n외식자주하고\n식기세척기.브라바.건조기등 있어서 그런지\n 진짜 손까딱안해요 ㅋ ㅋ\n\n'], ['시부모님.오신다고 돕는 남편\n넘 멋진데요?저는 그런날이오면 진짜 바빠지는데 오히려 남편은 왜청소하냐고~그냥 두라하니 더 속터져요ㅜㅜ\n\n네..각자상황에 맞게..^^\n\n'], ['저도10년타차.. 우울증온것같아요. 주말에 밥안해요. 늘어지게 늦잠도자고 낮잠도자요. 화장실청소 음식물쓰레기는.제가버리구요. 큰봉투쓰레기만.버리게해요 분리수거도 제가하구요. 전 밥하는게 스트레스라 밥만시켜요.'], ['저도 큰애가 초등학생이 되니\n살짝~늘어지더라고요~\n주말엔 낮잠자고 ,늦잠자서 어쩔땐 아침도 밖에서 사먹고오라고해요(남편과 애들)^^\n\n\n'], ['전 맞벌이에 주말부부인데도 잘안시켜요 시키면 제대로 안되서 하나하나 지적질하면 내가 다시안한다 이딴소리하거나 또 그냥 넘어가면 생색이ㅜㅜ 도를 넘거든요 에휴 걍 좀 드럽게 살고 가끔 제가 해버려요'], ['세상에..분리수거 음식쓰레기 전혀안도와주는데ㅠㅠ..\n근데 시키고싶지도않네요 밖에서돈버느라 힘드니까ㅠ'], ['이런걸 남들하고 비교하면 끝도 없는것 같아요ㅜ 그냥 정말 각자 만족하고 살면 그만이고 또 남편이 잘한다 한들 만족 못하는 사람도 있고 그냥 내 마음에 평화와 안정을 주는 남편이 좋은 남편인걸로 ㅎ'], ['분리수거도 일년에 두세번 하나....... 애 어릴때만 쓰레기 버리는거 전담으로 했고 그 후로는 아예 안시켜요. 어차피 제 손이 가야해서 괜히 시키고 고작 그거 했다고 비유맞춰주느니 제가 다 하고 내 비유 맞춰라 하는게 낫더라구요 전....'], ['깔끔떠는 신랑이 아니어서\n화장실 청소맡겼다간..\n또 제가 손봐야할껄요?ㅜ\n야무진 신랑들 최고~~\n쩝..장점을 봐야죠. 장점을.\n어디보자~~~~~'], ['저 12년차 전업..음쓰도..안시켜요...제가 걍 낮에 왔다갔다할때버리고 .,분리수거도 걍 제가왔다갔다하며 버려요.\n설거지? 걍 제가하던가 제가하기귀찮으면 미뤘다 담날도해요ㅋㄱㅋ. 걍  굳이 시키지않고 내맘대로ㅋ \n남편이 스스로도와주는거면 몰라도 ㅎ\n그래서인지 일절 서로에게 잔소리할일없어요. 청소안해도 뭐라안하고. . 밥하기싫음 외식하고 ㅋ 평일에도 외식자주하네요. \n대신 제가 못할땐 힘들땐 도와달라해요. 그럴땐 무조건해주죠.   화장실청소도, 설거지도, 집청소도  도와달라고하면 무조건(1년에 한두번도안되니ㅋ) .\n걍 각자상황에 맞게살면되죠.  전 저의일, 신랑은 회사일.. .\n각자일은 스스로...만약 맞벌이였다면? 조금은 바꼈겠지만요.\n저는 잔소리안하고 잔소리안듣고 사는게 젤좋네요ㅎ.'], ['그냥 신랑이 일하는 동안 제가 집에서 일하는거라 생각해서요 \n대신 아이랑 잘놀아주고 아이랑 둘이 나가서 시간도 보내고 음식물쓰레기나 힘쓰는거 도와주고 아이목욕해주고 외식때 아이케어 해주고 전 그정도로 만족해요'], ['네!마음이 이쁘세요.\n그런데 주부는 퇴근이없는 직업인거같아\n어쩔땐 욱~하더라고요^^전^^'], ['주말에 육아도 집안일도 안한다면 문제겠지만 아이랑 잘 놀아주신다니 육아는 남편전담 집안일은 아내전담 하면 되지않을까요? 남들이 시킨다고 우리남편한테도 똑같이 시켜야겠다는 생각이 보통 싸움의 발단이 되는것 같더라구요ㅎㅎ 전 신랑이 애를 잘봐줘서 주말엔 신랑이 육아전담, 제가 집안일 전담해요. 대신 전 육아 휴무이니 점심직전까지 늦잠자고 신랑은 애랑 아침부터 전쟁입니다ㅎㅎㅎ'], ['전 전업 결혼 17년차 화장실 청소는 남자가 해야줘~\n'], ['그런가요?^^이제껏 암말없다가\n급 시키면 당황스럽지않을까 걱정 입니다ㅜ\n알아서 해주시는 남편들 넘나 부럽습니다!'], ['분리수거는 남편담당이에요 . 전업이고 애랑 놀아주라는데 자꾸 시키지않은 집안일하고 생색내서 못하게 하고있어요. 무조건 애랑 놀고 자기 잠자리정리.밥먹은 그릇정리.이건기본인데.. ㅠ 습관이 안되있어서  이것만은 꼭 시켜요 ㅎ'], ['저희 신랑은 안들어와서 시킬수가 없어요~~ㅍㅎㅎ ㅜㅜ\n일반쓰레기나 분리수거. 음쓰는 제가 다 하죠..청소도..\n그나마 애들이 크니 종량제봉투나 재활용 봉투는 내려갈때 하나씩 들고가라 시키고~\n정말 일년에 한두번 제가 몸이 안좋으면 그때 설겆이? 하는정도~'], ['애가 어리고 남편이 육아에 많이 도와준다면 안해도 돼죠 일하는것도 힘들어요 ㅠㅠ'], ['울남편은.음쓰도안버려요.\n심지어.저는.같이일합니다.'], ['저도 전업인데 신랑이 온갖 집안일 다 하려해요. 근데 전 제발 안했음해요. 진짜 엉망진창ㅜㅜ 다 망가뜨리고... 하지말래도 계속 하는 심뽀는 뭔지...'], ['몰스엔 좋은 랑님들 많네요.\n맞벌이때도 전업인 지금도\n분리수거.음쓰.청소 제 담당이죠 맞벌이땐 주말에 한번씩 청소기돌리고 걸레질 해 준듯요. 전업되고나선 넌 집에서 쉬니깐 노니깐 니가해 이런 뉘앙스예요. 에효 전 걍 그러려니 해요.\n돈버는 유세 하는구나 하고요.'], ['네..몰스엔 좋은 남편분들많은거 같아요~~^^'], ['저희 남편이란 똑같아요 ㅠㅠㅠㅠㅠㅠㅠㅠ'], ['울집 남자는 재활용품이랑 쓰레기 버리기, 자기 옷 빨래하고 다림질 하기 정도만 해요.. 음쓰도 같이 버리면 좋겠구만 비위가 약해서 도저히 못 버리겠다네요.. 신혼 초에 밀어 붙였어야 하는 건데ㅠㅠ'], ['자기옷빨래랑 다림질하기 최고예요!\n전 제남편이 하고있는 상상도안되네요!ㅜㅜ맞아요!신혼초에밀어붙여야했어요...커튼도 안달아주고 기술적인일?도 안해줘서 전 만삭때 커튼달다 뒤로넘어져서 구급차타고 응급실도갔어요ㅜㅜ'], ['화장실 바닥에 머리카락도 주을줄 몰라요..ㅋ 결혼 12년차 들어갑니다~ 내 아들은 그리 안가르친다 하고 있습니다.ㅋ'], ['아..저역시 아들 둘인데\n잘 가르쳐야겠어요!생각치도 못했다가..댓글 감사합니다^^'], ['ㄴㅔ!제 자식이라도 집안일 돕게끔 가르쳐야겠네요!편안한밤 되세요^^'], ['우와 글보고 남편들 많이 도와주시네요 저희신랑은 쓰레기 음쓰 버리기 애들씻기기 이것만 시키는데 생색 백만번하는데ㅋㅋㅋ저희집은 재활용 일줄에한번 집앞에 그냥 다모아서 나두면 들고가는 시스템이라 일이 없어요'], ['저도 해봐야 다시 해야하고 일단 퇴근하고 오면 좀 안 되어보여서(저 퇴직전에 퇴근하면 노는거 말곤 아무것도 하기 싫었던게 생각나서) 재활용이나 한번 버리러 보낼까..안 시켜요. 되도록 저녁전에 후딱 해치우고 같이 놀아요. 애기 어릴땐 애기보라그러구요.. 살림도 너무 열심히 하지 마시고 요령껏 대충 해치우고 사세요. 집안일 누가 하냐 신경 곤두세우는거 너무 피곤해요. 그냥 아침 커피 만들기와 주말 식사 한번은 꼭 시켜요.빈둥대고 있으면 청소기 한 번 밀래? 하면 군소리없이 미는 정도로만.'], ['여지껏 불만은없었어요^^\n집안일 누가하나 신경 곤두세운적도없고요~^^\n그냥 당연히 제가 했는데 몰스카페 신랑분들이 많이하시길래 ...이제야 좀 화장실청소정도는 부탁해볼까~?했는데 왠지 그래도안될거같네요..이미 많은시간에 길들여진거같아요..ㅜㅜ댓글감사해요^^'], ['몇몇 댓글들이 남편 자랑하기 같아요ㅋㅋㅋㅋ\n내일 미세먼지 최악이라는데 저는 혼자 열심히 분리수거 하는 날이네요 \n진짜 너무안하니까 재수없어요'], ['댓글중 모닝커피받는거랑 남편요리먹어보기는 넘.부러워요~\n전 이번생엔 틀린거같아요ㅜㅜ\n왠지 그런거 받는것도 제가어색할거같아요 ㅋㅋ\n어쩌다 애둘 목욕시키는것도 엄청 힘들다하는데(주1회정도?)매일 하는 부인들을 좀 생각해줬음좋겠어요ㅜㅜ\n전업이라 당연히 해야하는건맞지만\n퇴근이없는 직업같아서 힘들때가 더 많거든요~ㅜㅜ\n'], ['솔직히 그런 댓글 넌씨눈 같아요 \n저도 이번생에는 틀렸어요\n저도 전업인데 돈벌어오는 유세 너무떨더라고요 ㅜㅜ'], ['전 음식물 쓰레기 분리수거 버리러 나가서 다른집 남자들이 쓰레기 버리러 오면 그게 그렇게 부럽더라구요..'], ['^^남편이 하는말이 밖에 쓰레기는 다들 남자들이 버리러 나온다고 하던데..\n님 남편분도 한번 느끼셔야하겠습니다!\n슬쩍~한번 같이 버리러가자고 해보심이~'], ['결혼한지 14년정도 되는 것 같은데 욕실 화장실 그런거 해본적 한번도 없네요. 맞벌이 할때도 아무것도 안했어요. 최근 몇년전부터 두어달에 한번 청소기로 대충 청소하고 분리수거나 음식물쓰레기 한번씩 버리는 것 말고는 아무것도 안해요. 밥쳐드신 그릇도 고대로 놔두시네요. 그래도 잘하는거 하나 있어요. 먹을거 사러가는 심부름은아주 잘합니다....지가 먹어야 하니 ㅠㅠ\n말하고 보니 난 왜 이렇게 살았나 싶고억울하고 그렇네여 ㅠㅠ'], ['집마다 사람성격도 상황도 각자 다르고 분명 집안일 말고 다른 분야로 잘하시는것 있으실텐데...  한가지로만 판단하고 맘상하는거 그렇잖아요..\n댓글보고 맘상하지 마시고 내가 상관없고 가정이 평안하면 된거 아닐까요..\n비교하지 마세요ㅠㅠ'], ['화장실 하라하니 물만 뿌려대서 그냥 제가 해요.ㅠㅠ\n시키고  스트레스 받느니.'], ['빨래 분리수거 쓰레기 음쓰 청소기 남편이고 전 요리랑 화장실청소만 해요'], ['화장실 청소 한다는 남편 부럽기는 하지만, 제가 집안일 중에 제일 하기 싫어하는 설거지 해주는 걸로 위안 삼아요.'], ['저희집신랑도화장실청소는~한번시켜봤네요.그후로는지금까지제가해요~^^집안일은음쓰.재활용.쓰레기봉투버리는거는무조건해요~저는아무것도해달라고안할테니.쓰레기버리는거랑~애들이랑놀아주는것만하라고했어요~지금은설거지까지알아서하네요.저는신랑이랑같이일을했어서~힘든거알다보니못시키겠더라구요ㅜㅠ전업이니까집에서내가하는게맞다고생각하기도했구요~'], ['저 휴직때 신랑이 미리선전포고하더라고요. 집안일에서 손떼겠다고. 정말 1도안했어요.대신 애들이랑놀아주는걸 주문했죠~~\n복직한지금. 누구보다 열씨미 집안일합니다;;;; 돈벌어오는게 좋은가봐요ㅜ'], ['저희집도 그 정도만 하는거 같아요. 대신 아이랑 잘 놀아주고, 제가 피곤하다그러면 아이랑 나갔다가 오고.. 아이한테 잘하니까 불만은 없어요~'], ['저희신랑도 화장실청소한적 없어요.결혼 7년차구요.청소기돌려주고 설겆이정도?\n제가 딱히 시키지도않고요,안해도 안이상해요~집안일은 내 일이라고 생각해서 제가 다하는편이요.워킹맘되면 달라질꺼 같긴해요!\n제가 집안일안도와주는 남편테 딱히 섭섭하지도 않은 이유가 시키는건 거절않고 다 해줘서 불만없어요~'], ['..... 음식물 쓰레기도 안버려요.. 분리수거 겨우 해주는데 것도 안할때 많아요.... 저는 어쩌죠....'], ['전 요즘 손목이 아파서 자꾸 시켜요. 화장실은 솔로 닦아야 하니 시키고 아이 운동화도 손목아파서 오빠가 해줘~ 덜 깨끗해도 내가 안하고 남편이 해준게 어디야 하며 못본척해요'], ['맞벌이인데 남편 분리수거 외에는 안해요. 도우미 1주일 4시간쓰고 나머지는 제가 해요. 이건 정답은 없는거 같아요. 맞벌이면 반반이 맞는데. 성격에 따라 다른거 같애요. 젊은 부부중에는 남자가 더 많이 하기도 하드라구요. 부지런함이나 깔끔함 기싸움 이런 거 영향받는 듯요. 전 더 시켜보려했는데 남편이 게으른 편에 기가 세서. 걍 뒀어요. 에휴ㅡㅡ'], ['저랑 같으세요~ 근데 남하고  피교하다보면 끝이 없고 더 우울해질 뿐.....이죠;;']]
    
    2651
    몰스님들은 어떤 집안일이 제일 하기 싫으세요??? 전 설거지요ㅋㅋㅋㅋㅋ 저희 집 오면 다들 애 키우는 집 맞냐고 할정도로 깨끗하고 저 스스로도 청소하는걸 엄청 좋아해요ㅋㅋ 스트레스받으면 청소하는 스타일이거든요. 그런데 저는 설거지는 도저히 못하겠어요ㅜㅜ너무너무 더러워요. 기름이랑 고춧가루 묻은 그릇들을 손에 만진다고 생각하면 너무 싫어요. 밥은 제일 잘먹는다는게 함정ㅜㅜㅋㅋㅋ 저 진짜 이상하죠??그래서 설거지는 하루종일 안하고 남편이 퇴근하고 오면 해줘요ㅋㅋㅋ 근데 또 웃긴게 저 음식물쓰레기 버리는건 아무렇지도 않아요ㅋㅋㅋ 화장실청소도 한번도 남편 시킨적 없고 저 혼자서도 잘해요. 설거지할래 화장실청소할래 하면 전 화장실 청소합니다 ㅋㅋㅋ여튼... 전 설거지가 이상하리만치 너무 싫은데 몰스님들은 뭘 제일 싫어하시나욤?
    
    [['저는 빨래 개고 그 갠 빨래를 가져다 놓는거요 정말 싫어요ㅜㅜ'], ['전 빨래 개면서 희열을 느끼는 사람이에요ㅋㅋㅋㅋㅋ'], ['하루종일 거실에서 있어요'], ['정말요? 진정 빨래 개는게 희열이 느껴지시나요?\n전 애가 셋이라서 한번 개는데 40분정도 걸려요ㅜㅜ'], ['저도요 저도요\n개는 것도 힘든데 ...'], ['제가 손이 빠르기도 하고 각잡아서 넣을때의 희열이 있어요....ㅋㅋㅋㅋㅋㅋㅋ'], ['저두 빨래개는건 그렇다치고 가져다놓기 진짜시러요. 집이 31평인데 한 50평되서 수납공간많아지면 좀 나아지려나요ㅜㅜ'], ['오오 저랑 똑같네요 ㅎㅎ\n전 청소 설거지 요리 다 잘하는데 빨래널고 개고 그게 세상에서 젤 시러여ㅜㅜ\n그래서 빨래는 신랑이 해요\n애들이 옷찾을려면 아빠찾아여ㅋㅋ'], ['저두요 건조기 사서 좋긴좋은데 빨래무덤이 자주생겨요ㅠㅠ 빨래 개주는 기계 누가 안만드나요'], ['저도 건조기 써서 빨래 무덤이 생긴답니다 ㅋㅋ 제가 매일 신랑에게 하는 말이에요\n빨래 개주는 기계 좀 생기면 좋겠다고ㅎㅎ'], ['저두요ㅜㅡㅜ'], ['ㅋㅋㅋㅋ 건조기 생기니 이제 개고정리하는게 싫어지죠 ㅋㅋㅋㅋㅋ 아'], ['50평되면 둥선이 길어져서 더 귀찮아져요. ㅎ'], ['저는 아이들 밥먹고오는 식판이요ㅠㅠㅠㅠ'], ['그것도 냄새 진짜 역겹죠....ㅜㅜ 으'], ['저두요ㅋ너무 귀찮아요'], ['빨래 개서 서랍장에 넣는거요 ㅋ'], ['저 빨래 개는거는 진짜 잘해용ㅋㅋㅋ'], ['빨래 개기요 ㅠ'], ['빨래 개는걸 많은 분들이 안좋아하시나봐용. 전 개면서 희열을 느끼는 사람이에요ㅜㅜㅋㅋ'], ['설거지는 맞는데 구찮아서요 \n전 음쓰도 싫어요 ㅋ\n화장실은 뭐 건식 써야하니 하는거구요'], ['음쓰도 좋진 않지만 차라리 음쓰가 설거지보다 나아요 저는ㅜㅜㅋㅋㅋ'], ['전 진심 정리요..아 정리하기시러요.'], ['정리하려고 맘먹기가 쉽지 않아요ㅋㅋㅋ'], ['요리....맨날 뭐먹을까 고민.. 장보는것도 그렇구요  차라리 설거지가 속편해요'], ['전 설거지가 싫어서 요리를 못하는 여자에요ㅋㅋㅋ 집에서 고기 한번도 안꿔먹어봤을 정도에요ㅋㅋ 기름 닦기 싫어서'], ['저도 설겆이,화장실청소가 싫어요ㅜ\n사실 집안일중 좋은건 없어요;;;'], ['요리요 못해서요'], ['애들 재우고 힘빠진 상태에서 집안정리요 ㅜㅜ'], ['애들 재우면 그냥 퇴근하셔야죠 ㅋㅋ 전 안해요 그냥 ㅋㅋㅋ'], ['설거지요~다른 청소는 부지런히 하고 빨래는 좋아하는데 설거지는 너무 귀찮아요..'], ['싱크대 하수구 청소요;;'], ['밥이요. 그래서 남편시켜요ㅋㅋ'], ['저도 정리요ㅜㅜ 너무 못하고 하기 싫어요ㅜ'], ['걸레질이요'], ['전 설거지, 빨래개기 다 좋은데 청소요~~화장실도 거실도 다 하기싫네요'], ['빨래개고 정리하기~~'], ['전 화장실 하수구 머리카락 빼는거요 아 생각만해도 넘 싫어요'], ['정리요... 진짜 ...,하'], ['전 청소여 ㅜ 빨래 설거지는 그래두 괜찮은데 왤케 청소는 해도 한 거 같지두 않고 좀 있다 또 해야 할 거 같고 채력 소모도 많고 청소 한 번 하자면 할 곳은 왤케 또 많은 지...ㅠ'], ['전 청소기 돌리고 정리하는거요 ㅠㅠㅠ\n번외로 여행가방 짐풀기도 ㅜㅜㅜ'], ['전 정리랑 청소요 ㅎㅎ 좋진 않지만 그래도 요리,설거지,빨래는 훨씬 낫네요 ㅋ'], ['밥 차리기요ㅠㅠ'], ['밥 하는 일요.남이 차려주는 밥만 먹고 싶어요'], ['청소요ㅠ쓸고닦고'], ['걸래빠는거요ㅜㅠ'], ['전 화장실 청소요ㅜㅜㅜ'], ['전 화장실 청소요...진짜 너무 싫어요.'], ['설거지가 전 젤편해요 아이장난감정리하는거랑 걸레질이젤힘든거같아요. 그리고음식물쓰레기랑 분리수거는 평생 신랑담당..전 쓰레기버리는게 그렇게싫으네요..쥐가나올것만같고..'], ['저는 바닥에 널부러진 애들 장난감 남편이 대충벗어놓은 옷 정리하는게 젤싫고 짜증나요'], ['빨래개는것까진 괜찮은데 갖다놓기 분리해서 여기저기 넣는거 넘귀찮..ㅜ'], ['빨래 개서 서랍에 넣는거 \n이것땜에 집안일이 너무 오래걸려요--'], ['빨래너는게싫어서 건조기.물걸레질귀찮아서 로봇청소기. 설거지는 식세기가..\n화장실청소가 전 싫은데 대신 해줄수있는 기계가 없네요ㅜㅜ'], ['빨래 서랍에 넣는게 제일 귀찮아요ㅜ'], ['다림질요...'], ['설거지 걸레빨기 다 싫어요 어깨아픔ㅠㅠㅠ'], ['옷 개서 넣는거요. 진짜 젤 싫어요 ㅠㅠ 구석에 맨날 쌓아놔요 ㅋㅋㅋㅋㅋㅋ'], ['화장실청소요.ㅠ'], ['빨래여'], ['전 걸레빨기요'], ['어떻게 하나만 싫을 수 있나요???? \n전 ㄷ ㅏ !!!'], ['전 정리정돈이요 ㅜㅜ 설거지는 씻어서 건조해서 넣으면 되서 금방해요~~~근데 수납장도 부족하고 옷개기도 힘들고 ㅜㅜ'], ['하수구청소요...ㅜㅜ'], ['빨래정리하기 ,,  베개 이불 커버 씌우는거요 ㅎㅎ'], ['전부 다요. ㅠㅠ'], ['요리요'], ['빨래개고 서랍장에 정리하기요~~~. 근데 저랑 같은 분들 많으시네요~^^'], ['한개만 골라야 하나요? 너무 어려운 질문 ㅜㅜㅜㅜㅜ'], ['음식물쓰레기 버리는 거요~ 3년차 주부인데 여태 한번도 안 버려봤어요 ㅠㅠ 비위 상해서 도저히...ㅠㅠ'], ['빨래갠거 정리하는거요~\n빨래개는것도 하겠는데..\n정리하는거 진짜 귀찮귀찮ㅜ'], ['뭐가 싫다기 보다는 집안일 전체가 자잘한 잡일이 계속 있고~안하면 나중에 하기 힘들다는 총체적 난국이 된다는게 싫어요~ 내가 계속 움직여야 유지가 되니까요~ㅜㅜ'], ['빨래정리요. 세상 귀찮아요.'], ['싱크대 배수망 빼서 그 밑에 관 닦는거랑 화장실 하수구 밑에 거름망 빼서 청소하는거 최악이요'], ['빨래 각자 가져가는거 5살후반부터 교육시켰어요.ㅋㅋㅋㅋ\n지금 셋째 젖병이랑 빨대컵이요ㅠㅠ\n시즌상품으로..세탁실에 빨래 종류별로 분류해놓기..집안 구조상 세탁실이 너무 추워서 다들 세탁실 문앞에 던져 놓으면 제가 색깔별로 종류별로 나누는게 일이에요ㅠㅠ 아 생각만 해도 욕나와요 진짜 겨울엔 세탁실 저만 들어가요'], ['다 싫지만 그중에서 설거지가 젤 싫어요\n음식물쓰레기 뒷처리까지 포함이요 ㅠㅠ'], ['저도 빨래정리요 ㅋ 건조기쓰는데 빨래쌓아두면 거기서 찾아 입어요 ㅠ\n저 시엄니께서 너무 깔끔하게 사는거 아니냐 지적받은 며눌예요 ㅋ'], ['빨래 개기요... 건조기 사기전엔 빨래널기...ㅎㅎㅎ'], ['빨래의 전과정 제일 피곤해요.\n시간도 오래걸리는데 자주 해야하니까요.\n\n그중 다 갠 빨래 각 서랍에 넣고 걸고 이게 제일 귀찮아요.'], ['설거지요'], ['전 걸레빠는거가 너무 싫어요.\n걸레 전용 미니세탁기를 뒀어요. \n미니세탁기 버린후엔\n물티슈나 1회용포 물걸레포만 써요.\n꼭 걸레를 써야하는곳이면 메리야스나 수건 잘라서 쓰고 바로 버려요.'], ['저도요 ㅋㅋㅋ 걸레빠는거 베란다에서 쪼그려하는거 게을러서 못해요 ㅋㅋ 샤오미 걸레는 작아서 세면대에서 씻어요 ㅋㅋ'], ['갠빨래  각자 서랍장에 넣기..이게 젤루싫어요.\n아이가 셋이니 방방다니면서 넣는거 힘듬'], ['씽크대 하수구 청소랑 걸레질이요~~너무싫어요~'], ['밥하는거는 진짜 애들 있어서 하네요'], ['전 그냥 다요ㅠㅠ'], ['저는 철마다 옷장정리 하는거요\n우리나라는 왜 사계절인지'], ['전 설거지가 젤좋아요 청소가젤시러요 극혐..'], ['저도 설거지요..그담에 밥하는거ㅎㅎ 한끼 먹고나면 그릇이 뭐이리 많은지 ㅠㅠ 큰애가 5살인데 엄마는 뭘 제일 잘하냐고 했더니 설거지래요. 완전 충격ㅠㅠ 제가 매일 설거지만 하나봐요..'], ['빨래개는거요\n진짜 빨래개서 제자리 넣어놓는게 젤 귀찮아요;;;;'], ['저도 빨래 널고 개고 넣고 하는게 젤 싫던데 다림질도 싫고... 그러고 보니 옷이랑 관계 되는 거네요ㅎㅎ 옷은 좋나하는데 참.......ㅋㅋㅋ'], ['악 저랑 똑같아요 ㅋㅋㅋㅋ 설거지 진짜.. 손계속씻고싶어져요ㅜㅜ'], ['설겆이-식세기\n청소-다이슨, 브라바\n빨래-건조기\n\n빨래개키기... 제일 힘들어요. \n\n식세기 사세요.'], ['전 반대로 설겆이가 젤 좋아요\n씻다보면 홀가분한기분들어요\n대신 걸레질 음식물쓰레기가 젤 싫어요'], ['설거지가제일싫었는데 식기세척기산이후로는 괜찮아요 저는 정리가제일힘들어요ㅠ'], ['밥하기가 젤루 시러용 ㅋㅋㅋ'], ['음식물 쓰레기 버리는 거 진심 싫어요. ㅜ'], ['고무장갑 끼고 해서 전 설거지 조아해요~ 설거지다하고 부엌 물기까지 싹 말라있은거보면 세상뿌듯해요ㅋ 대신 음쓰랑 화장실청소는 못하겠어요 특히 변기.. 귀찮은건 빨래갠거 정리하기네요ㅋ'], ['반찬, 음식이요'], ['저도 설거지ㅠ그래서 남편시켜요ㅡㅡ'], ['저두요 그래서 식기세척기 없으면 안돼요']]
    
    2715
    당직하고와서 집안일과 요리해주는 신랑>< 신랑이 일주일 내내 당직이었는데 딱 하루 쉬어서 집에 왔거든요. 근데 제가 일을 하고 있는 사이 본인이 일주일동안 못한 집안일(못박기, 분리수거하기, 화장실청소, 설거지, 관리비 확인 등) 을 다 하고 저녁밥까지 차려놨더라구요 ㅠㅠㅠㅠㅠ 감동쓰....신랑 이뻐죽겠어요!!!!!!!!!!!!!!!사진은 신혼집은 아니고 신혼여행 풀빌라에서도 요리해줬던 신랑사진이에여 ㅋㅋ
    
    [['좋으시겟어요 ㅎ 당직이 많다니ㅜㅜ고생많으시겟네오 신랑분 맘이 넘 예브네요'], ['어머~~힘드실텐데ㅠㅠ 식사까지 해주시다니 감동이네여'], ['그와중에 주방인테리어가 보이는데 너무이뻐영'], ['신랑의 따뜻한 배려가 무한 감동이네요 ㅜㅜ'], ['서로 당연시 하지 않는 마음들이 넘 좋아보여요♥♥'], ['순간 신혼집 주방인줄...ㅎㅎㅎ 서로 배려와 사랑이 팟팟 느껴지네요'], ['ㅎㅎ좋으시겠어요 ㅎㅎ 신혼여행때라니 ㅎㅎㅎ'], ['우왕 진짜 사랑이네요+.+ 당직까지 하시고는 집안일을 ㅎ'], ['자상한 남편입니다 ㅎㅎ 저도 남편이 해주는 요리 먹고싶어욬ㅋㅋ'], ['ㅎㅎ넘스윗하네요. 제신랑은언제쯤 저런걸..'], ['평소 요리를 좋아하시나요? 아무리 좋아해도 본인이 힘들면 못하는건데.. 행복하시겠어요ㅎㅎ'], ['우와 배려랑 사랑이 가득 ㅠㅠ부럽습니당'], ['우와아 다정하신 신랑님이시네용 부럽습니다!><'], ['꺅!! 신부님이 너무 좋으신가봐요 ㅋㅋㅋ 피곤하실텐데 ㅜㅜ 요리를 ㅜㅜ 감동이에요!!'], ['당직까지 하고 오셔서 ! 대단하시네요 사랑꾼이신거같아요 ㅎ'], ['자상한 신랑님 ㅠㅜ 식탁의 하트까지 완벽하시네요'], ['남편분 감동이네요ㅜ 사랑이 느껴져용'], ['세상에 진짜 너무 스윗하네요~!! 깨소금 파파팍'], ['최고의 신랑님 이십니다.. 이것은 칭찬받아 마땅할 일입니다.. 그리고 넘 부럽네요ㅋㅋ 신랑님 칭찬합니다ㅋㅋ'], ['저라도 넘넘 감동이였겠어요!!ㅎㅎ'], ['어머머 두분다 마음이 너무 예쁘세요ㅜㅜ'], ['대박ㅠㅠ사랑이 넘치시는 커플이네요! 부러워요~~행복한 결혼준비되세용^^'], ['대단하시네요 ㅎㅎ 신랑분이 신부님을 향한 사랑이 넘치세용!!ㅋㅋ'], ['흑 요리하는 신랑 ㅜ.ㅜ 멋져요 ㅎㅎ'], ['우왕 가사를 부담없이 서로 배려하며 하시네용 축하드립니다'], ['우와 ㅠㅠ넘 멋진 신랑분이시네요^^'], ['저희 예랑이도 여행가거나하면 요리해줘요 ㅎㅎ 다정하네요 ㅎㅎ'], ['넘 좋우시겠어여~~ 부럽습니당~~'], ['너무 자상한 신랑분이시네용ㅎㅎ'], ['결혼잘하셨네여! 저런 남편있음 믿음직스러울것같아요'], ['너무 착하신 분이네요 결혼 진짜 잘 하셨어요 ㅎㅎ'], ['신부님을 위해 요리부터 청소까지 넘 부럽네용'], ['신혼집인줄 알고 벽돌벽이어서 깜짝 놀랐어요 ㅋㅋㅋㅋ'], ['꺄오~ 멋지세요 +_+ 이거해 저거해 시키면 여자도 짜증나잖아요 ㅠㅠ 같이 일하는데....저희 신랑은 하도 이것저것 신경쓰는게 많아서 주방은 터치하지말라고 경고했네요! 나머지 청소는 참 잘해요 ㅎㅎㅎ'], ['와 이런 남편분이 어딨나요 진짜 대단하세요!! 보기좋네요'], ['와 당직이라 힘드셨을 텐데 멋진 신랑님이네요!'], ['남편분 멋지세요! 대단하시네용 행복하세요~~'], ['대단하시네요!! 당직 하고 와서 힘들텐데 집안일까지 도와주니 얼마나 고마워요ㅠㅜ'], ['신랑님이 마음이너무 예쁘세요! 배려심도 깊고 부럽네요^^ 신부님이 행복하시겠어요~'], ['진짜 사랑받을줄 아는 신랑님이시네요~너무 보기 좋으세요~^^'], ['우왕 신랑님이 애정이 넘치시네요'], ['우와! 일도 힘드실텐데 남편의 애정뿜뿜'], ['일도 하시고 집에와서 요리까지 하려면 진짜 힘드실텐데..ㅠㅠ 신부님 사랑하는 마음이 엄청 크신가봐용!!ㅎㅎ 부러워용'], ['피곤하실텐데 요리까지 해주시다니 감동이네요ㅠㅠ'], ['정말 자상한 남편이 짱인거같아용!ㅋㅋㅋ'], ['신혼집이 너무 예쁘신대요!!'], ['멋진 신랑님~~칭찬해요\n두분 배려가 너무 이쁘네요'], ['부러워요 ㅠㅠ 사랑 뿜뿜할거같아요 !'], ['역시 결혼은 자상한남자가 최고인거같아요!!ㅎㅎㅎ'], ['요리해주는 신랑은 사랑이죵 헤헤 저희 신랑도 저한테 항상 요리해주려고해서 너무 행복해용!'], ['부러우네요^^제 신랑도 당직이 많은 업종이예요'], ['어머 너무 감동이네요~피곤할텐데 와서 집안일까지 해주시다니 ㅠㅠ~~'], ['와 멋짐폭발하네요 ㅎㅎ 행복하세요'], ['피곤할텐데 그렇게 해주면 엄청 기특할 것 같아요!'], ['오모나 자상하셔~ 너무 좋으네요! :)'], ['피곤하실텐데 정말 다정다감한 신랑이시네요 ㅎㅎㅎ'], ['이러면 넘나 듬직할 거 같아요~'], ['남편분 넘 멋지네요~~ 피곤할텐데 챙겨주면 넘 좋을꺼같아요! >< 행복한결혼생활되세요~~'], ['어머 너뮤다정다감하시네요!!!'], ['우와 일하고 오셨으면 피곤하실텐데 밀린 집안일까지 다 해주시다니... 정말 다정하세요~~ 행복이 별견가요 ㅋㅋ이럴때 가장 행복할듯요^^'], ['힘들어도 도와주시다니 자상하시네요~부럽습니다'], ['넘 착하네요 ㅎㅎ 상주셔야겠어요~~ㅎ 저도 신랑이 알아서 분리수거하고 쓰레기버리고 해놓으면 잘했다고 폭풍칭찬해줘요^^'], ['아내의 사랑을 듬뿍 받으실만 하시네요^^'], ['자상한 신랑분이시네요 멋져요!! 당직하고 힘들텐데두 보기좋아보여용><'], ['당직까지 하고 오셨는데 대단하세용!!'], ['신랑분 피곤하실텐데 대단하시네요 ㅎㅎ 너무 자상하고 부러워요 ㅎㅎ'], ['이야아 너무나 사랑꾼이시네요! 부럽습니당 알콩달콩 신혼 행복하세용!!! 리스펙입니닷 ㅎㅎ'], ['너무 자상하시네요 요리잘하는 남편이 사랑받는거같아요 ㅋㅋㅋ'], ['와 신랑님이 진짜 사랑받으실만하네요.. 당직까지 하고와서 집안일까지...^^;; 부러워용'], ['와 남편분 넘 스윗하신데요~ 부러워용'], ['남편분이 너무 다정다감하세요 ㅜㅜ  대단하세요'], ['최고의 남편분이시네요.. 대단하십니다!!후후'], ['요리잘하는 남편이라니 너무 부럽네요 ㅎㅎ 저희 신랑은 요리 진짜 못해요 ㅋㅋㅋㅋㅋ'], ['우아 대박대박 ㅠㅠ 집안일&요리하는 남편.. 너무 좋네요...♡'], ['와.... 백점짜리 남편이네요!!!!!\n집안일 뿐아니라 요리까지\n저희는 ㅋㅋㅋㅋㅋㅋ 요리는 제가하기로 했는데 두개 다해주는 남편 너무 부럽네요!!!'], ['남편님 넘 무럽고 멋지십니다'], ['ㅠㅠ신랑분도 피곤하싱텐데ㅜㅜ진짜 마인드도 부럽습니당!!'], ['오 참멋지네요 저는 반대인데요 제가 당직서고 집가서 신랑 반찬 만들어준다는 ㅋ 그래도 신랑이 밥알아서 잘 먹고다니더라고요 ㅋㅋㅋ'], ['우왕 ㅋㅋㅋ 엄청 로맨틱하시네용 ㅋㅋ 저도 빨리 신혼 즐기고 싶네요'], ['우와 주방 인테리어가 돋보여요 ㅎㅎ 신랑님두 짱이시네요'], ['신랑님 짱이시네요~ 당직하고 오셨으면 많이 피곤하셨을텐데 !!'], ['진짜 ㅎㅎ완전감동이었겠어요 부럽네요ㅜㅜ']]
    
    2732
    하기 싫은 집안일 뭐에요?(부제: 빨래 개는 기계는 없나요?) 해도 티 안나고,안하면 엄청나게 티나는게 집안일이죠..저두 연휴때 가족들 다 있으니 대충 지냈고오늘은 다 나가고 나니 청소가 눈에 보이네요.집안일 다 싫지만 그중에 좀 괜찮은거 있고정말 하기 싫은거 있지 않나요?전 그나마 잼있는건 설거지인데빨래는 느~~~~무 싫어요ㅠㅠ빨래는 세탁기가 한다지만 너는거, 개는거는 사람이 해야하잖아요.그래서 너는거라도 줄이려고 건조기를 샀죠.너는거 안하니 신세계~ no 먼지 신세계~!!하지만 개는게 남았네요ㅠ지금도 1분 남은 건조기를 보며 에휴=3 했네요ㅋ개고 서랍마다 넣는것도 일..ㅋㅋㅋ말하고 나면 기분이라도 시원하니하기 싫은 집안일 뭔지 공유해봐여.우리~ㅋㅋㅋ(건조기 다돌았네여...ㅋㅋ)
    
    [['빨래 개기 넘 귀찮죠~\n오늘 주부심들 가사 노동인 날이네요~\n잠시 티타임 가지시고 힘내세요~\n'], ['네네~ 빨래 개고 커피 한잔 하려구용~^^ 여행짱님두 홧팅입니다~^^'], ['집안일이 정말 해도 해도 끝이 없는거죠...ㅠㅠㅋ 고생 많으세요...ㅜㅠ'], ['남편분들도 밖에서 고생많으시져..^^;;'], ['옛날에 페이스북에 동영상에나왔어요 빨래개우는기계있더라고요ㅎ 외국에요ㅎ 저도사고싶었다는..ㅜㅜ'], ['댓글보구 찾아봤어요ㅋㅋㅋ\n세탁기처럼 일반화 되는 날이 오길~~~^^'], ['빨래개는판 이라도 사시면 좀 더 수월하시려나용 ㅜㅋㅋ'], ['요것두 찾아봤네요^^ 하지만 시간이 더 걸릴듯한 느낌이~~ㅎㅎ'], ['전청소ㅜㅜㅜㅜㅜㅜ'], ['청소기 미는것도 귀찮긴 하죠 ㅠ'], ['집안일 힘들고 다 싫어요ㅠㅠㅎㅎ'], ['정답입니다~~!!! ㅋㅋㅋㅋ'], ['정말 빨래시러요.\n연휴지나고 세번돌렸네요.\n널곳이 없어요'], ['널곳이 없었던적 많아요~ 그나마 건조기 있으니 그런 고민은 줄었네요^^;;'], ['맞아여 맞아여~ ㅋㅋㅋㅋㅋㅋ'], ['저두 빨래개우는게 넘 싫어요ㅜ'], ['그쵸그쵸? 제가 비정상이 아니었군요ㅋㅋㅋㅋ'], ['빨래개는거까진 진짜 참고 하겠는데 서랍에 넣는게 전 너무 귀찮더라구요ㅎㅎ\n개인적으로 정말 찾아가서 입었으면 좋겠어요ㅋㅋ'], ['진심 공감하는 바입니다ㅋㅋㅋ'], ['전 다싫어요ㅎㅎㅎ'], ['암요 암요~ 다 싫죠 ㅋㅋㅋㅋ'], ['빨래개기 기계ㅎㅎ   일이 끝이없죠ㅜㅜ 차한잔 드시고, 힘내세요'], ['고맙습니다~^^하면 하는데 투정 한번 부려봤드랬죠ㅋㅋ커피한잔으로 기분전환해야겠네요ㅎㅎ'], ['저는 설거지요ㅠㅠ'], ['아~ 저는 설거지는 그나마 낫던데 산더미 설거지는 부담스럽기도 해요^^;;'], ['전 바닥청소가 젤 싫더라구요'], ['두번째로 싫어하는 일입니다ㅋㅋ\n그것도 걸레질^^;;'], ['저는 설거지 진짜 싫네용ㅠㅠ'], ['지저분한 그릇 냄비들 보면 비호감이긴 하죠^^;;'], ['첨엔 건조기만해도 신세계였는데 빨래가 쌓이는 개는기계 생겼음좋겠다 했었어요ㅋㅋㅋ\n이모님 아주머니 있는분들이 부럽네요\n'], ['그쵸그쵸? 저두~ ㅋㅋ\n바라는게 끝이 없네요ㅋㅋㅋ'], ['저는 고양이를 키우다보니 이불털기를 \n하루 두번 해요\n걸레질 하는것도 보통일 아니고\n하루종일 청소만 하는 듯 ㅜㅜ\n암튼 집안일은 중노동 입니다'], ['아..그렇겠네요.. 청소기 돌리고 걸레질하고 정말 보통일 아니죠ㅠ'], ['건조기에 말려서 그냥 꺼내입는걸로 패턴 바꿔나갑시다ㅎㅎ'], ['집안일 다싫지만, 걸레질은 더더더 싫네요ㅠ'], ['그죠그죠~\n저두 청소기는 매일 돌려도 걸레질을 매일하진 못하겠더라구요^^;;;'], ['ㅋㅋ저도 그생각 했는데 내년인가 빨래 개어주는 기계 출시된대요ㅋㅋㅋㅋ기계 작동하는거 보니 신기하긴 한데ㅋㅋ집에 넣을 곳이 없을 것 같아요ㅋㅋ'], ['똑똑한 기계들인데 부피가 커가지구..ㅋㅋㅋ우리가 편해지는만큼 자리를 내줘야 할려나요...^^;;;'], ['전 걸레질 싫고 다 닦은 후 걸레 빠는건 더더더 싫어요.ㅠㅠ\n그래서 가끔 일회용부직포 써요'], ['그래서 저두 일회용 부직포 쓰거나\n걸레도 모아서 세탁기 돌려요ㅎㅎㅎ'], ['빨래가 젤 싫어요 세탁기돌리기도 널기도 개는일도 다 ㅜㅜ'], ['저랑 똑같네여~~ㅠ 안 입지 않는한 끝나지 않는 일이죠ㅋㅋㅋ'], ['건조기있어서 널지않아도되고 넘나편하지만 개는건 정말싫다는ㅠ 설거지도싫어유ㅜㅜ'], ['결론은 다싫은거ㅋㅋㅋ\n빨래 겨우 개고 서랍에 넣기 싫을땐 아이들 시킵니다~ 각자 자기꺼 넣고 나면 오백원! 오백원에 하더라구요ㅋㅋ'], ['전 바닥 걸레질이요~~~ㅠㅠ느므 힘들어요'], ['걸레질도 힘들죠ㅠ 그래서 손걸레질 절대 안해요^^;;\n깨끗하긴 하지만 하고 나면 진 빠져서..^^;;'], ['저는창틀먼지나 선반구석먼지닦기 제일하기싫어요 그래서 자주안해요 ㅋ'], ['맞아요~~ 선반에 물건들 하나하나 치우고 닦으려면..ㅠ저두 가끔씩..^^;;'], ['설거지, 손빨래 싫어요 ㅠ \n저는 음식 하는건 좋은데 설거지가 그렇게 싫어요 ㅎㅎㅎ'], ['그러신분들 많이 봤어요ㅎ 음식,설거지 고르라면 설거지네요^^;;'], ['집안일하니 시간이 훌쩍 가버렸네여ㅠㅠㅠ'], ['해도해도 끝이 없고 매일 해야 현상유지, 안하면 엉망진창에 은근 죄책감도 들고..가족들이 있기에 힘들지만 참고 하는거죠~^^;;'], ['빨래 하나 개고 딴짓하고 또 하나개고 딴짓..한시간 걸릴때도 많아요 ㅠ'], ['공감공감~저두 비슷해요ㅋㅋㅋㅋㅋ아니면 다 개고 안넣고 한~참 딴짓하다 서랍에 넣죠ㅋㅋ']]
    
    2884
    (2월18일 월요일 오늘의출석은요?)우리아이들 심부름 집안일 도와주나요? 저희 둥이들은본인이 알아서 스스로하는게거의 없어요  ㅠㅠ자기책상치우는것도 무슨 큰일한것 처럼  ㅋㅋ대신 제가 시켜요..빨래 널자   정리하자설거지 오늘당번은할머니랑 샤워할사람 ㅋㅋ제가 지독하게 많이시키는건가요?어떤분들은 어차피 난중에 많이해야할 일인대우리애는 안시키고 싶다고...하시는 분들도 계시더라구요?어떠신가요?.울 어뭉님들은?
    
    [['아직 어리지만 자기 물건은 정리하기 시작했어요'], ['안시키니 안하드라구요. 그냥 청소해줘 책상정리해줘 시키는편이에요 ~'], ['어지르지 않는 편이지만 가지고 놀았던건 정리하도록 습관 들였더니 정리는 알아서 잘 해요.\n아직 집안일은 안시키네요.'], ['스스로 먹는것 챙겨먹기\n라면도 잘 끓여먹어요\n심부름도 잘하구요\n정리정돈은 지맘 내키면 하네요\n뭔 바람인가하고 기특해할때도 있는데 잘은 안하고 아주 가끔요^^'], ['저희집은 어릴때부터 엄마 도와주는게 잘 훈련되어 있었지요..저 혼자서 애들 다섯을 케어하는 자체가 힘들기에 고사리 손들을 빌릴 수 밖에 없었거든요ㅠㅠ지금은 학교 수업시간에 가사분담 교육도 받아서 종종 나서서 도와주는데..요즘 첫째가 그동안 쭈~욱 해왔던게 많이 지쳤는지??모르쇠..할때가 있더라구요...실과시간에 과일깎기 수행평가도 있고,바느질 하기,등등 수행평가가 그렇다보니 전혀 못해도 해볼 수 밖에 없더라구요...그러다보니 자연스레 혼자서도 밥 정도는 차려먹고...설거지,실내화빨기,빨래널고 개기 정도는 당연히 하더라구요..역시 공교육의 힘이 대단해요!!!'], ['정리는 가끔 하는데 다른건 얘기해야 해요^^;;'], ['이제 초3되는 딸램.. 자기방 정리외 책상정리밖에 못 시키네요.. 4학년쯤 되면 간단한 설겆이랑 청소기 돌리기정도는 시켜도 될꺼 같긴 한데..'], ['스스로 할 수 있는 것을 조금씩 늘려가며 시키는 편이에요..손까딱 안하는 신랑이랑 살다보니..넘 힘들어서..ㅜㅜ'], ['첫째는 취미가 코딩이라 폰만 가지고 있고 학습은 알아서 하고 모든 하고 나서 제자리에 두는데 작은 아이는 뭘 만들고 꾸미고 색칠하고 종이가 한가득 ㅜ\n그리고 그대로 ㅜ  \n집안 일은 거의 혼자해요 ㅜ\n흑...\n'], ['본인방 침대랑 책상정리는 학교 입학하면서 했고요..그 외 집안일은 부탁해요 ~해  말고 ~ 도와줄수 있어? 요로코롬요 그럼 냅따 넘어와요 ㅋㅋ'], ['큰딸보다는 아들이.. 잘도와줘여..수건정리하는거...재활용 버릴때..'], ['우리 딸들은 책상정리,가방정리,벗은옷 빨래통에 넣기,식사한 자기그릇 설거지통에 넣기..이정도요^^;;\n가끔 부탁하면 신발정리,빨래개기 정도 해줘요^^'], ['기본적인 습관은 만들어 놔야지가 맞아요. 나중에 지 마누라한테 구박 안 맞으려면 시켜야된다가 맞지요ㅋ'], ['분리수거. 수저정리, 자기옷 넣기등은 해요'], ['작은아이는 잘 도와주는데, 큰 아이는..... 그래도 가끔 힘쓰는일은 잘 도와줘요~ㅋ'], ['어릴때부터 자기물건은 자기가 정리하게 습관들였더니 잘해요~'], ['어릴때부터 습관이 중요한것 같아요 저희아이도 늘 미루지만 조금씩 해보게하고 있어요'], ['시켜야지 스스로 하는 버릇이 생기더라구요\n자주 시킬려구요~'], ['7세지만 자기 장난감은 자기가 스스로 치울 수 있도록 유도하는 편입니다.'], ['자기가 먹은 그릇정리,벗은 옷  빨래통넣기,공부후 책상정리 같은 자기주변정리만 시키고있는데 그 외는 “내가 왜?””왜나만?”이런 마인드네요'], ['저희집은  시켜도  잘 안해서  부글부글하지만   계속   정리하자  같이하자  하며 참여시켜요.훈련계속해야 할듯요.'], ['정리보다 쌓아놓는 습관이 길러진듯 새여'], ['집안일은 시킬때만 하네요.\n하지만 자기가 어지르는건 스스로 치우기 하네요'], ['스스로 정리하기 간단한 마트심부름 \n바닥청소하기'], ['다른건 기대안하고 최소한 가지고 놀았던 장난감은 꼭 스스로 치우도록 독려중입니다 ~ ^^'], ['자기 물건은 스스로 정리하게 하구요~ 심부름가는건 좋아해서 자주 시켜요 ㅋㅋㅋ'], ['속이 터져서 시키다가 제가 다 정리하는 것 같아요 ㅠㅠ'], ['하나씩  어떻게  정리하고 오라고  시킵니다\n당근을 주고요~'], ['자기가가지고논거나어지른거는정리하라고합니다나머진아직못하네요ㅜ'], ['자기 할일은 스스로~\n설겆이는 아이들이 하고나면 뒷처리가 더 많은거같아서 제가 하는데 신랑은 아이들과 같이하거나 시키네요 ㅎㅎ;;'], ['저희 애들도 시켜야 하죠...스스로 한적은 본인 게임하고싶을때 잘보일라고 하네요..ㅎㅎ'], ['전 많이 시키는 편이예요. 독서하고 있을 때 이럴 땐 안 시켜야 하는데 .. 혼자는 너무 힘들어요'], ['자기스스로는 절대 안해요 ㅎㅎ그래도 이것저것 자잘하게 시키는 심부름은 아직까지는 잘해줘요.두살어린 동생과 경쟁하면서 서로하려다 싸움납니다..ㅎㅎ'], ['시킵니다\n우선 본인 책상. 침대 위 정리부터 기본이구요. 바닥은 청소기 돌리지만 밀대로 닦는 거 시킵니다 큰 아이는 가끔 설겆이도 시킵니다 빨래 개는 거는 무조건 아이들 몫입니다\n단. 시켜야지 합니다 ^^;;'], ['제법 혼자하는것도 많이생기고\n빨래개는거 부터 집안일을 잘 도와줍니다.\n기특해요'], ['심부름은 물떠오는 정도.. 대신 주말마다 장난감 정리함 엎어서 다시 분류, 정리 시켜요~'], ['전 조금씩 치우라고해요 \n책상하나치우는것도 엄청대단한일한거만냥~~\n크면더않할것같아서 지금부터 조금씩 시키고있어요  심부름도시키구요 ^^'], ['자기방 정리정돈만 하죠..자기방만 치워요..^^::'], ['적당히 시켜야할꺼같아요. 울집은 애들이 옷골라입으면 옷장이 엉망이되서 제가 맨날 골라주는데 ..요즘 숙제입니다'], ['천천히 시키고 있은데 외동에 딸 이다 보니 ㅜㅜ 애교에 자꾸 넘어가네요'], ['성격에 따라 다른것 같아요..꽂히면 하는 스탈이네요..ㅎ\n제가 청소하고있으면.. 옆에서 자기도 한다고 책상정리하곤 합니다. 잘한다고 칭찬해주고 밑밥(?)깔아주면..ㅎㅎ 좀 합니다. 아직어려서인지 유도를 하면.. 하는편인데... 보니 습관 들여 놓는게 좋을것 같네요^^'], ['아들녀석은 자신이 먹는 과자봉지도 안치워요... (@@!!) 근데, 딸은, 작은손으로, 어쩔때는 설겆이도 도와주려고 하고, 빨래 널을때, 속옷이랑 양말은,,, 뭐, 척척~ 갖다 쌓기는 하지만 ㅋㅋ 그래도 많이 도와줍니다'], ['저는큰애7살부터 쉬운것부터 하게했어요\n빨래개기.세탁기에 다된빨래꺼내기,\n냉장고에 반찬꺼내 밥차려먹고씽크대 그릇정리해놓기\n남자들이라 지금말잘들어줄때 해놓지않으면 크면서 안할까봐\n제가 힘든건 다도와주라고하는편입니다^^'], ['어려서부터 하던거는 잘하구요(밥먹고 그릇치우기, 책상정리등)\n방정리며 거실에서 간식먹고는 그대로 널려 있네요 ㅠ 말해야 치우는 ㅋ'], ['시켜놓고 치울때까지 기다리는거 넘 힘들어서 자꾸 치워주게 된다능....ㅠㅠ\n이제 조금씩 습관을 잡아 가야 겠어요'], ['자기가먹은밥그릇은 정리하게해요.가방정리,놀잇감정리도요.계속이건 너희가해야하는것이라고, 말해주고..계속얘기해요.😊 \n(물론스스로하는날은가뭄의콩이납니다요🤣)'], ['시키는데 잘 안하네요~~~'], ['스스로 정리하기랑 벗은 옷 빨래통 넣기를 시키긴 했는데 생각날땐 하고 안할때는 쌓여 있네여^^;;'], ['출석]\n 심부름이 뭔가요??? \n하라고 하면 대답만 하고 안 움직이는걸요...\n답답해서 제가 다 하고 화내교..........'], ['저도 계속 시켜보곤 있는데....영 안되네요 ㅠㅠ 습관 들이기가 가장 힘든거 같아요 ㅠㅠ'], ['자기 물건 정리 하는것도 잘 안 된데요..'], ['초등 올라감 집안일 조금씩 시키는게 교육상 좋다는군요.근데 엄마 성엔 안차죠~'], ['전 직장맘이라 주말에 같이 하려고 하는편입니다. 물론 시켜야 하지요. \n신랑, 아이와 빨래개기.설거지등 시켜요.'], ['정리정돈은 잘해용~'], ['전 4살때부터 자기밥그릇 싱크대 갖다놓기,장난감정리 시켰고 저희딸은 초6이라 집안일 도와주며 용돈벌고 있어요 주로 설거지나 동생장난감정리 해줘요 그냥 용돈 받을땐 돈의 소중함을 몰랐는데 스스로 용돈벌기 하고 난후부터는 돈의 소중함을 알고 아껴쓰고 있어요'], ['댓글보고 도움 받아가요.\n습관잡히게 조금씩이라도\n시켜야겠어요.'], ['수저놓기해요'], ['청소기 돌리게 바닥 치워달라하면 싹 치우긴 하더라구요. 막 어딘가에 쑤셔넣어놔서 문제긴 하지만요ㅋㅋ'], ['심부름시키면되려저한테하라고하는두따님들ㅋㅋ대신놀았던자기물건들정리만해주는것도감사하게생각합니당'], ['요즘들어 큰애 초딩5 아이가 많이 도와줍니다 어릴때야 칭찬해주니 도와줬는데 좀 컸다고 자꾸 도와주네요^^'], ['ㅎㅎㅎ자꾸 도와준다는 부러운 댓글이 있네요 ㅎㅎㅎㅎ 부럽네요 진심 ㅎㅎㅎ'], ['정리는 가끔 마음내키면..\n집안일은 도와달라고 하지 않으면 안해요..'], ['정리를 한다고하는데.. 아직은 많이 서툴러욤~~'], ['저도 가끔 시키려고해요.'], ['댓글을 보니, 오늘 부터 전 뭐라도 시켜야 겠네요.\n그냥 아기라고 생각들어 오냐오냐 했나 싶네요.\n빨래 개기 부터 시켜 볼까 봐요.'], ['저도 이제 슬슬 시켜봐야 겠어요.'], ['여전히 잔소리해야 정리하는 아이들이네요ㅠ언제 스스로 좀 할까요?ㅠ'], ['아직은 엄마 껌딱지라 고사리같은 손으로 빨래도 함께 개주고 신발정리도 해줘요'], ['보상이 있어야하네요 ㅋ'], ['시키면 잘해요.. 안시키면 안하지요...'], ['공부도 중요하지만 나중에 어른이 되어 누구의 도움없이 스스로 잘 챙기면서 사는게 진정한 어른이고, 진정한 독립이라 생각해요. 그래서 조금씩 집안일은 늘려서 시켜볼 생각이에요.'], ['자기 물건은 스스로 정리 하라고 이야기하고 있습니다.'], ['엄청 어지르네요. .\n시키면 심부름은 잘하는데 치우기를 잘 안하려고해요..'], ['큰아이는 자기방 정리는 조금씩 하고 있어요..\n근데 다른일들을 자꾸 도와 주려고 하는데... 솔직히... 가만 있는게 도와주는거라... --;;;;'], ['제가 하는 일을 몇번 같이했어요..\n설겆이 쌀씻기 청소기..그랬더니 언젠가부터\n아이가 먼져 같이하자 말하네요ㅎㅎ\n요즘엔 같이 아이방 정리중이에요..\n언젠가 혼자할때 도움되길 바라네요..ㅎㅎ'], ['정리만 잘해요..'], ['네 잘하는편이에요~'], ['아직은 어려서 제가하는게 더편해요.ㅋㅋ'], ['집안일은 같이하도록 시키고있어요'], ['저희도 다둥이네라큰애한테미안하죠 오남매님대단하세요'], ['시키면 하는수준? ㅎ'], ['첫째는 아들이라 어려서부터 시켰어요..ㅋㅋ\n 분리수거한거 내다버리기..수건개서 욕실에 정리하기...빨래정리하면 각방에 갖다놓기..수저놓기..반찬통정리..밥먹고 설겆이통에 넣기..물컵정도는 마시고 씻어놓기..방정리등등.. 가끔 청소기도 밀어주고 제법 많이 도와줘요..\n둘째는  저정도는 아니지만 오빠보구선 자기 방정리는 아주 깨끗이  잘해요..\n'], ['저도안시키는편인데.그러니엉망이네요'], ['어리지만 둘도 없는 효녀라 잘 도와줘요\n엄마를 도와주는 일을 즐거워하는거 같아요'], ['빨래를 개줘서 너무 좋아요ㅎ'], ['7세,8세, 방정리와 책상정리 가끔이요 ㅎㅎ'], ['11세. 집안청소 밥차리기 다되요. 9세, 청소와 정리가 되구요, 8세, 물티슈로 방닦기를 해요;;;'], ['10세.작년부터스스로샤워정도는하네요.아직놀이감정리는동생보다떨어집니다ㅎㅎ'], ['정리 잘 하는 것만으로도 만족하네요 ㅎㅎ'], ['아니요~ 암것도 안해요 ㅠ'], ['자기방 치우기, 빨래돌려서 개기, 저녁시간에 숟가락 놓기.. 집안일 시켜야 해요..']]
    
    2966
    투썸 아메 드림[노는게제일좋아님께 완료] 저번에 영화예매 해드리고 받은거예요스벅은 제가 마실 예정이구..요고 한잔..ㅎ그냥 날도 꾸리하고 맘도 꾸리하고..집안일 할께 천지인데 암것도 하기싫고요로코롬 카페 구경만 하구 있네요ㅋㅋㅋㅋ번호 안 꼬이게 셀프 달아주셔요~ 날짠 3/19일까지예요!!! 추첨은...아마 늦은 저녁이 될지도 몰라요ㅋㅋ
    
    [['2'], ['3번요^^'], ['4번이요!!'], ['5'], ['6\n번 ^^'], ['7'], ['저도줄서봅니당~~~8번'], ['9번.저도요,줄서봐오^^'], ['저도 줄서봐요^^'], ['줄 서봐요^^ 12번~'], ['조리원에있는뎅 줄서보아용 바로 앞에 투썸있어서요'], ['14번^^'], ['15번 맞나요~? 줄서요^^'], ['16번 줄서요~^^'], ['17번요~!'], ['줄이 꼬였네요 18번'], ['18번'], ['20번이용^^'], ['21번'], ['22번이요'], ['23번'], ['24번 줄서요'], ['25번이용'], ['26번\n\n새해 복 많이 받으세요^^'], ['27번요~'], ['29번유'], ['30 줄서봐요^^'], ['오~~ 31번 줄 서 봅니다^^'], ['32번줄서바여'], ['33번 줄서보아요'], ['34번 저도 줄서봐요~'], ['35번 줄이요~'], ['36번용'], ['37번 줄서봅니다^^'], ['38'], ['39번 / 맘님 즐거운 설연휴 보내셨나요^^ 꿀꿀한 마음 털어내시고 행복한 시간으로 가득하길 바라봅니다^^'], ['39'], ['40\n저두 설명절 증후군이요ㅜㅜ'], ['41번이요^^'], ['42번'], ['43번이요^^'], ['44번이여 ^^'], ['45번 새해 좋은일만 가득하시길요😊'], ['46번요'], ['47번\n병원에 아픈 둘찌때문에 창문통해서 햇빛보고잇네요....'], ['48번 줄서봅니다~~'], ['49번요ㅎ'], ['50번 ~'], ['51번 드림응원해요^^'], ['52번 이요♡'], ['53번이요'], ['54번♡'], ['55번'], ['56번 \n줄서보아요~~'], ['57~^^'], ['58번 줄서요~'], ['채팅주세요~'], ['꺅 감사합니다^^'], ['59번 줄서욧'], ['60번 줄서보아요.'], ['61번'], ['62번 줄서보아용'], ['63번이에요~~~^^'], ['64번 줄섭니다'], ['65번요^^'], ['66번이요^^'], ['67번 드림 멋져요!'], ['68번줄서요.\n드림도응원드려요~~'], ['69번요~\n이렇게 줄서면 되나요?'], ['70번이요^^\n투썸 넘나 좋아해요'], ['71번줄서봐용'], ['72번 퇴근하고 아이병원가는데 줄서봐용~'], ['73번 줄서볼께요~^^'], ['74번 줄서봐요^^'], ['75번 줄서요'], ['76번줄서보아요'], ['77번 줄서보아용^^'], ['78번 줄서봅니당'], ['79번 줄서봅니다><'], ['80번 줄서봅니다~^^'], ['82 줄서요'], ['83번요^^'], ['84번 줄서봅니당^^'], ['85번^^'], ['86번^^'], ['87'], ['88번요~'], [''], ['89'], ['90  줄서봅니다'], ['91번 줄설게요^^\n명절 피곤한 몸 따뜻한거 마시고 힘내고싶어요'], ['92번 투썸 좋아요^^'], ['94번~ 줄서봐요^^"'], ['95번. \n멋진 드림이네요!!♡'], ['96번 집앞에 투썸있어요~조심히 줄서봅니다'], ['97번^^❤️'], ['98번 줄서봅니당♡'], ['99번투썸 줄섭니다'], ['100번 줄서봅니다^^']]
    
    2973
    집안일 끝냈으니~~또 놀러가도 되것쥬~~?!^^ㅋ 한동안 제 소식 뜸했죠~?!^^은근..바빴네요~~아이들 방과후 수업도시작하고..학부모 참여수업도..있고....언니들과 만나서 밥먹으며 수다도 떨어야하고ㅋ놀러갔다와서...맘잡고 집안일에만 신경쓰려했지만~~그런데~~그런데~~하늘에서 저에게 바다로 나가라는 계시가ㅋㅋㅋ오늘~~내일이 물이 가장~~많이 빠지는날~~고로~해루질,조개캐기 아주~~좋은 날이라는거죠ㅎㅎ맘같아선..낮에도 가고싶지만ㅜㅜ운전도..못하고.. 신랑도..출근을하니....오늘 퇴근후 8시에 출발~~~~~12시면 물이 들어오기때문 그전에 눈부릅뜨고낙지찾으러댕겨야해유~~~놀러가기위해서 집안일을 했죠~냉장고가 텅~텅 비웠거든요ㅎㅎ하루는 만두만들고 시금치데쳐서 말리고하루는 청량고추청,금귤(낑깡)청,끝물이라맛없는 딸기로 딸기 콩포드 우유에 넣어먹을라고만들고 오늘은 반찬만들었어요~~날씨좋으며 토요일도~일요일도 가려고요ㅎㅎ아이들은 위험하기에 신랑이랑 저랑만출동~~~~만통할수있도록 ~~응원해주세용~~^^♡왜케 저녁이 ..안되지..언능가고싶은데ㅋㅋ
    
    [['서해가서 워킹해루질하려고요~~수중해루질..하고싶은데 거리가 멀어서ㅜㅜ'], ['만통 꼭 성공 하셔요^^~\n기운 팍팍 드려요'], ['기운~~팍팍 받아서~~낙지~~무지하게 잡고~~살려서 오겠슴돠~~기대하주세요~'], ['조개캐러 지도 가고싶으네요~'], ['저도여ㅜㅜ\n차만있었음~~낼~~아침에도 가는건데.. 오토바이에 네비달고 가믄 오토바이 엔진터지겠지여?ㅋㅋ'], ['진정한 금손 이세요~~'], ['감사합니다~^^♡'], ['오이무침 레시피좀...부탁드려요^^'], ['전 새콤달콤한거 좋아해서~~\n식초,설탕,소금에 절여놓고 고추장조금 고추가루 매실액,다진마늘,다진파넣었어요~~^^'], ['냉장고에 피클하고 난 오이가... 이따 무쳐볼게요^^'], ['비쥬얼은 다르지만\n해먹었어요.\n설탕을 덜 넣은듯 해요.ㅎ'], ['캬 보기만 해도 눈이 호강해요^^ 여행사진도 궁금궁금'], ['기대해주세용~~~많이 잡아올께용~~^^'], ['우와~대단하시네요^^\n어찌이리 부지런하신지ㅎㅎ\n맘님글 볼때마다 제가 부끄럽네요~\n진정 금손이시네요^^'], ['놀기위해서 움직입니다ㅎㅎㅎ'], ['셋맘님집에가고싶어요..'], ['오세용~~ ㅎㅎㅎ\n월요일은 친한언니랑 친구랑 집에서 김밥말기로 했어요~'], ['ㅋㅋㅋㅋ마음은 이미갔답니다ㅋㅋ'], ['우와 금손이시네요 반찬가게 하셔도 되겠어요~~'], ['장사는 제 성격과 맞질않은것같아용~~^제가 소심하거든요ㅋ'], [''], [''], ['와!! 대단하다는 말밖에 안 나오네요\n최고예요~'], ['칭찬감사합니다 ~~'], ['오이무침 레시피 궁금해요^^'], ['전 새콤달콤한거 좋아해서~~\n식초,설탕,소금에 절여놓고 고추장조금 고추가루 매실액,다진마늘,다진파넣었어요~~^^'], ['소금에만 절여서 물기짜고 넣으니 물이 생기던데 셋맘님 레시피대로 해봐야겠어요!!~~주말 잘보내세용^^'], ['만두..ㅜ\n반찬도 다 제가 좋아하는...\n아...진심으로 금손이 부럽습니다:)'], ['저도 첨에는 못했어요~하다보니 빨라지고 감으로~~막하는거예요~^^'], ['차량만 되면 모셔가도 되나요 ~^^'], ['저도~데리고 서해에 버려주세요~~~~~~~~~~~~~~'], ['국화도로 가시는거예요?\n거기서 캐도 되나요? 거기서 조개캤다가 아주머니들에게 엄청 혼나고 다 주고왔던 기억이...ㅜㅜ\n해루질 부럽습니다~~ 만통하소서~^^'], ['국화도는 물때만 보는거예요~~요즘..못캐게 하는곳이 많아요ㅜㅜ'], ['맛있어보이네요 솜씨가좋으시네요 잘잡고 오세요~^^'], ['많이~잡아오겠습니다~^^♡'], ['소식없...으면....못잡은걸로...ㅎㅎㅎ'], ['저도 그래서 서해로 해루질하러왔는데\n어제 낮엔 낙지한마리잡고\n어젯밤엔 완전 허탕치고ㅠㅠ\n낮엔 굴이랑 조개랑 작은게 잡아왔네용^^\n권선셋맘님도 가득가득 잡아오세용^^'], ['오늘은 많이 잡으실꼬예요~~~저도 못잡을까봐걱정이....신랑은 못잡는다고 바지락캘생각은 하지말라는데.. 몰래 가슴장화속에 호미 챙겨갈까봐여ㅋㅋ'], ['전~개띠여ㅎㅎ\n그래서 빨빨되고 돌아댕기나?ㅎㅎ\n제가 바다에서 가슴장화신고 전동킥보드타고 다니니깐 어떤분이 신세대 어부인줄알았데요ㅎㅎㅎ\n전 하도 돌아댕기고 움직여서그런지 감기도 안오네요~~전...안움직이면 병나요...좀쑤셔서ㅋㅋ'], ['넘 맛있어보여요^^ 콩나물 무침 레시피 알 수 있을까요? 얘들 먹을 수 있게 매번 똑같은 방법으로 해서요.'], ['콩나물에 물조금넣고 뚜껑닫고 끓이시다 콩나물이 숨죽으면 식용류 한바퀴~~고추가루,간장,멸치진액이나 참치액넣고 끓이고 마지막에 다진마늘,다진파,참기름,참깨 끝~~^^'], ['와 감사합니다. 캡쳐할께요^^'], ['새우볶음 레시피 부탁드려요'], ['식용류,버터,고추장,올리고당 넣고 끓이시다 새우  견과류넣고 볶으시면 되용~~^^'], ['오~~~~~~~~저~좀 짱인가여?ㅎㅎ\n제가 오늘 낙지많이 잡아서~더~놀래게해드릴께용~~'], ['쫓아가고싶어유'], ['막내~업고 해루질댕겨볼텨?ㅋㅋ\n난 애기없고 해루질하는 멋진엄마봤다는~~완전 멋있다는ㅋㅋ'], ['100마리 잡아오세요♡♡♡'], ['넹~~후기 올릴께용~~~~~^^♡'], ['시금치는 말려서 어찌쓰시나요?'], ['건나물처럼 볶으면 되용~~~^^'], ['응원합니다^~^ \n다른분들께 알려주시는 레시피 저도 캡쳐좀 해가겠습니다. 감사합니다~~♡'], ['넹~~맛있게 만드세용~^^♡'], ['갑자기 님 반찬보니 반성하게 되네요'], ['아고~지송합니다~~^^;;'], ['세상 부럽습니다..ㅋㅋ'], ['부러워하지마셔용~~~~~~\n사서 고생하는거예요ㅋ'], ['진짜금손이세여 우왕'], ['칭찬감사합니다 ~^^'], ['저 정도 반찬을 하루에 만든다고요??? 일주일이 아니고?'], ['반찬은 2시간도 안걸려요~주부의 내공이죠ㅋㅋ'], ['배고프네용~~^^;;;  아~~ 금손 맨날맨날 부러워용~!!\n'], ['맨날~~혼자 맛난거 묵죠?밉게시리...ㅋ'], ['우와~~~~~최고최고!'], ['감솨~~합니다~^^'], ['ㅋ멀또저리 많이요?ㅋ조거할램 15일걸릴듯ㅋ그래서 안하지만ㅋㅋ'], ['이거말고도 엄청 했다는ㅎㅎ\n언니~~해루질가서 많이 잡아올테니 초장들고 기둥기세요ㅎㅎㅎ'], ['와 최고최고 저걸 어케 다~~~~\n부러버욤 ㅜㅜ\n'], ['그리어렵지 않아요~~~^^\n방법만알믄~'], ['금손 셋맘님!\n진짜 솜씨부럽습니다!\n저도 해루질  해보고싶어요.\n맛조개잡으러는 종종가보는데\n몇~~년전 같지 않고 성적이 안좋더라구요ㅠㅡㅠ'], ['사람많은곳은...사람들이 씨를 말려버려죠...ㅜㅜ\n유명?한곳은 안가야해유~~조개가 읎어유~~~'], ['냉장고 정리 하신거 보고 깜놀한 1인 입니다ㅋㅋ 낙지 조개 많이 잡아오세요~~^^'], ['냉장고에 소스랑 음료수만 있네요ㅎㅎㅎ'], ['오이무침 아래있는건 이름이뭐예요? 예전부터긍\n궁금했는데ㅜㅜㅜ아직도모른다는...ㅜㅜ'], ['무장아찌에서 무를 설탕에 절여서 간장에 담가놓은거 썰어서 무치기만해요~^^'], ['반찬너무 잘하시는거아니예요?  수업받고 싶네요  ㅎ ㅎ'], ['전문가가 아니라서...수업이...ㅎㅎ\n야매라ㅎㅎㅎ'], ['진짜금손이시네요'], ['칭찬감사합니다 ~~^^'], ['요거 다운받으시면 되용~~'], ['진정한 금손이신듯 합니다~~'], ['만두는 쪄서 지퍼백에 넣으시는거예요~?^^'], ['냉장고 선반위에 깔아놓은 저\n녹색?은 뭔지 알수있을까요? ㅎㅎ'], ['진정 멋지십니다!!!'], ['엄지 척~~!!!'], ['볼때마다 정말 대단하시네요!\n만통 하시길 ><\n이번엔 어디로 다녀오시나요?\n매번 다른곳으로 가시는거예요?\n'], ['오이무침 보고 임산부인 저.. 군침 돌았어요ㅜㅜ♡ 바로 해먹어야겠어요~']]
    
    3036
    오늘도 집안일😊😊 오늘도 집안일 해야하네요!주말엔 다 파업하고 싶으네요 빨래부터 개야겠어요!
    
    [['빨래개키기가 젤 하기싫어서 널때까지 기다리고있습니다 ㅋ'], ['맞아요! 저도 빨래개기 널기 너무 귀차나요! 방금 빨래 걷고 빨래널고 이제 개야지요 주말엔 쉬고 싶네요'], ['세탁기 한시간반있음 울리지싶어요 널어라고 ㅋㅋ그때까지 쉴랍니다'], ['저는 빨래 다개고 다시 세탁기 돌아가고 있어요ㅋㅋ오늘은 하루종일 돌아갈거 같으네요ㅋㅋ날잡아서 다 돌려버려야지요'], ['저희집도 오늘은 빨래 세번 돌릴꺼랍니다 ㅋ열일해야지요'], ['오늘 세탁기 열일하는날인가요ㅋㅋ저희집은 벌써 두번째네요ㅋ 아 건조기 너무 사고싶으네요ㅠㅠ'], ['저도 아침부터 빨래돌리고 널고 잠깐 쉬고 있어요 주말에도 주부는 쉴틈이 없네요 화이팅합시닷'], ['맞아요! 주말이라는 개념이 없어요! 맨날 일일일ㅋㅋㅋ넘나 싫으네요 차라리 돈이라도 받으면 모를까요'], ['주말에도 어김없이 집안일하시는군요 은근 주말집안일 할께 더많아지더라구요'], ['네네 어김없이 빨래 돌리고 빨래널고 빨래개고 설거지하고 음식하고ㅠ 아 진짜 피곤합니다'], ['그렇겠어요 저도 신랑있는날은 괜히 더할일도많아지고해서 귀찮더라구요'], ['맞아요! 그래도 오늘 쌓여잇던 박스들 다버리고 치우고 했네요~ 싹 비우고 나니 편하고 좋으네요'], ['저도요 그동안 쌓인박스정리도하고 오늘 재활용쓰레기도 정리하고나니 개운하더라구요'], ['저도 얼른 움직여야하는데 지금 미루고미루고있어요 다 쭈굴해지겠네요ㅠㅠ'], ['저도 널고 걷어와서 개야하는데 놔두고 카페놀이만 하고 있어요😱 어서 빨래 개야겠어요ㅋㅋ'], ['저두요\n어제 못한 집안일해야되지요\n우선 커피한잔하구요ㅋ'], ['그렇죠 엄마들은 주말에도 집안일이네요ㅋ 오늘은 좀 쉬고 싶으네요ㅠ 넘나 귀찮습니다'], ['저도 쉬고싶지만요\n제가 안하면 집안일이 난리가나니\n또 부지런히 움직이지요ㅋ'], ['저도 그래요 근데 오늘 박스 다 치워 버렸더니 완전 좋으네요ㅋㅋㅋ최고입니다 박스 맨날 쌓여잇어서 진짜 비좁고 그랫거든요'], ['박스요?어떤박스일까요?\n전 오늘 몇일 재활용 못버렸더니 한가득이더라구요ㅜ'], ['택배 박스죠ㅋㅋ저희는 집에다가 모아서 버리다가 보니 다 쌓여가지고 장난아니였네요'], ['전 초4되는 아들램한테 나만의 빨래개는 법 갈켜줬더니 손끝야무진 아들래미 저보다 더 깔끔하게 개네요 물론 지 바쁠땐 미뤄둘때도 있지만 3번중 1번은 개주니 그것만도 한결 편해요 ㅋ'], ['아직 딸래미가 어려서 개놓은거 펴지만 않으면 고마울거 같아요ㅠ좀더 크면 그렇게 시켜보려고요'], ['다들 주말에도 쉴틈 없이 바쁘네요ㅋㅋ\n저도 포함해서요 ㅠㅠ'], ['그러게요! 엄마들은 다 같은거 같아요! 주말에도 쉴틈 없이 집안일 해야죠 좀 누워서 쉬고 싶으네요'], ['저희도 2주가량 집을 비웠더니 먼지가 뽀얗게 앉았겠어요. 오후에는 집부터 정리해두고 애 델고 가야겠어요.'], ['아하 2주동안 집을 비우셨군요ㅋ 그럼 정리좀 해야지요ㅠ 저는 빨래부터 개야하는데 손이 안가네요'], ['애가 아파서 시댁으로 2주 와 있었어요. 너무 징징대니 혼자는 감당이 안되드라고요.'], ['아 그렇군요 전 아파도 시댁가까워도  어머님도 몸이 안좋아서 갈수가 없네요 그래도 애 이제 괜찮지요?'], ['빨래개는거가 저두 ㅠ  오늘 그나마 남편이 장실 청소해줘서 널널하네요'], ['부럽네요 저희신랑은 한밤중이요 죽엇나 확인 하러 가봐야겟어요~ 솔직히 깨우고 싶은 마음도 안생기네요'], ['으히 그시간에 한밤중이라니  전 아이낳고는 절대 혼자 자게 안합니다 ..'], ['맨날 피곤하다 피곤하다 하니 자라고 안깨웠더니 안일어나기에 죽엇나 확인하러 갓는데 딸래미가 깨우더라고요'], ['아히고 맘님은 착하시네여  전 절대 자게 안내버려둡니다  같이해야죠 육아는'], ['도움1도 안되서 걍 없는 사람 취급하는거지요ㅋㅋ귀찮아서 포기한거에요 저 그다지 착한사람 아닙니다ㅋ'], ['주말은 빨래부지런히 돌아가는날인거같아요ㅡ빨래개는거 넘나 싫어요ㅋ'], ['맞아요! 계속 카페 하느라 안개고 있다가 그냥 빨리 개고 치웠네요ㅋ 이제 정리하러 가야지요ㅋ 아 귀찮아요'], ['오오ㅡ부지런하셔라ㅋ전 입을옷은 얼마없는거같은데 빨래개고 정리하러 서랍열면 한숨나와요ㅋㅋ서랍장이 더 필요하담서ㅋ'], ['저도요! 다 버리고 싶은데 버리고 나면 찾게 될거 같아 버리지도 못하네요ㅠ 아이꺼도 제대로 정리해야하는데요ㅠ'], ['ㅋㅋ헌옷 엄청모아놨어요ㅋㅋ그래도 서랍장은 부족해요..더 버려버려야겠어요ㅋ1ㅡ2년안입은거는 과감하게 버려봅시당ㅋㅋ'], ['그래야 하는데 저는 귀차니즘이 심해지고 있어요 정리가 안되요 안되요ㅠ 언제쯤 버릴려나요ㅋㅋ'], ['저는 결혼식와서 구경중이네요 ㅋㅋ 배고픈데 배고픔보다 울컥울컥하네요ㅠㅠ'], ['아 결혼식 가신다고 했었죠ㅋㅋ저 기억력3초인가바요ㅋㅋ저도 결혼식 보면서 제가 막 울컥울컥ㅜㅜ아줌마가 되가나봅니다'], ['그러니까요ㅜ 결혼식 제 결혼식도아니면서 왜그리 울컥하는지...ㅜㅜㅜㅋㅋㅋ'], ['세탁기 돌려만 놓고 아직 못널었어요..ㅋㅋㅋ\n헹굼 탈수 한번 더 해서 널어야되는데 ㅠㅠ 애 재워놓고 하렵니다..'], ['아 넘나 귀찮죠ㅠ 그기분 알아요 저도 널어야 하는데 놔뒀다가 다시 돌리고 널고 그랫네요ㅋ 좀 쉬시고 이따하세요'], ['결국 나갔다와서 저녁에야 널고 말았습니다ㅋㅋㅋ 그래도 오늘 널어서 다행이죠..ㅋㅋ어른옷은 다음날 널기도 해요'], ['아하 전 어른옷도 다시 돌려서 널고 그래요ㅋ 저는 그냥 건조모드로 돌려놓고 나갓다왓어요!!'], ['주말에 우리도 쉬고싶어요. 평일엔 퇴근하고 싶어요~\n집안일의 끝은..없어요. 흑흑'], ['그르게요 전 빨래 다되면 널고 이따 설거지나 좀 해야겟으요ㅋㅋ귀찮아서 조금만 쉬려고요ㅋ'], ['저도 청소랑 빨래랑 주방정리 다 해놓고 누웠어요. 날 찾지마라는 말과 함께요. ㅎㅎ'], ['좋으네요! 저는 저때 시댁갓어요 아버님도 오라하셧엇고 애도 안자고 해서 그냥 나갓는데 넘나 힘들엇네요'], ['하악~ 갔는데 넘나 힘드셨으면...ㅠㅠ\n할아버지보며 빵긋 웃으며 잘 놀고 돌아와서 빨리 잠들어야 엄마가 쬐끔 편한데 말이지요.'], ['밥먹으러 나갓는데 징징 짜증짜증 어우 진짜 집이였으면 궁디 팡팡했지 싶어요~'], ['빨래 넣고 개는거 싫지용 그래도 안할수는없다는.... 화이팅입니다'], ['맞아요! 널고 개는거 진짜 너무 싫습니다ㅋ 원래는 이렇게 까지 싫어하진 않앗는데 요즘은 귀찮아요'], ['저는 빨래는 급한 것만 일단 했어요. 이제는 널 곳이 없어서 못합니다. 건조기 사고프네요ㅠㅠ'], ['저도요! 건조기가 절실합니다 드럼도 건조기능이 되긴한데 줄어들더라고요ㅠ 건조기가 필요합니다'], ['우리는 드럼 아니라서.. 드럼 건조기는 전기도 많이들어서 별로라고 하더라고요.'], ['2박3일 주말내내 청소만 하다 시간 다 보낸것같아요ㅜㅜ 내일은 청소 좀 쉬고싶어요'], ['저는 청소해야하는데 넘나 귀찮아서 다손놓고 누워있어요~ 애는 낮잠도 안자고 짜증납니다ㅜ.ㅜ'], ['분명 청소를 하긴했는데 왜이렇게 티가 안나는지 의문입니다ㅎㅎ'], ['해도해도 티안나는게 집안일이지요 저도 그래요 그러니 더 보람이 없는거 같아요ㅋㅋ'], ['티안난다고 안하면 그  티는 얼마나  잘 들어나는지 ㅎㅎ\n숨쉬듯이 그냥 하는것 같아요'], ['맞습니다ㅋ 저는 해도 해도 그자리라 요즘 좀 지치기 시작했어요 마음딱잡고 청소하면 깨끗한데 금방 어지럽히니깐요ㅠ']]
    
    3372
    집안일은 정말 끝이없네요ㅜㅜ 큰애 얼집보내고 한번도 안쉬었는데도 할일이 태산이에요ㅠ
    
    [['...더 중요한건....열심히했는데ㅠㅠ...티가안나요.....'], ['맞아요... 공감 100'], ['마자요ㅠㅠ집안일 진짜 하루종일해도 시간 금방가고 끝도 없어요'], ['벌써 2시반이에요ㅜㅜ'], ['해도해도 끝이없죠 진짜ㅠ\n열심히 해도 알아주는 사람도 없네요  ㅠ'], ['저희들끼리 알아줍시당ㅜㅜ'], ['진짜끝이없어여ㅜㅜ 저두 청소하고반찬하고 지금까지 걍 앉아만잇네여.. 빨래 게우고 밥먹고치우고 밥해놓고.. 할일이태산인데 너무태평하네여지금 ㅠ'], ['잠깐은 쉬어봅시당 ㅎㅎ 커피타임~'], ['맞아요. 이일했으니 좀쉬자싶으면 해야할일이 떠오르고 눈에보이고..'], ['그러게요 자꾸 눈에보여요ㅜㅜ'], ['아예 외출해버려야하는데..근데 외출해도 밀린집안일이 자꾸 떠오르는..헤어나올수없는 주부의늪...이긍..'], ['저두요... 오늘 큰언니네가 온대서 계속 청소하는데도 끝이없네요ㅠㅠ\n아니 저녁에나 출발할줄 알앗는데 \n벌써출발햇데요ㅠㅠㅠㅠㅠ'], ['아이쿵 마음이 급해지시겠네요~ 힘내세요!'], ['맞아요 집안일은해도해도 끝이 없고 티도안나더라구요ㅜ 그래서 더 슬퍼요'], ['네ㅜㅜ 티가안납니다 정말열심히했는데'], ['그쵸 저두 아이 보내놓고 뭔가 부지런히 움직였는데 티는 나지 않고 시간은 가고;;;'], ['벌써 두시반이에요ㅠㅠ'], ['맞아요....치워도 치워도 그대로 인 것 같고..ㅜ매일 해야하고...ㅠ'], ['네ㅠㅠ 시간은 또 훅 가버리고ㅠ'], ['저두오전내 청소하고 둘째 낮잠자니 저녁해야해요 저두옆에서진심자고픈데..할일이 넘많네요'], ['맞아요ㅠㅠ 낮잠좀 맘편히 자고싶네요'], ['그러게요 진짜 집안일은 해도해도 끝이없어요ㅜㅜ 해도 티도 안나구요'], ['티안나도 우리끼리는 알아줍니당 ㅋㅋ'], ['집안일은 정말 티안나고 힘들기만한 일이죠.\n보상도없고요ㅜㅜ\n'], ['보상ㅜㅠ 은 없죠 흑흑'], ['맞아요 집안일은 정말 해도 해도 끝이 안보여요  너무 힘드네요'], ['저두요ㅜㅜ 허리가아파브네용'], ['티도안나는집안일ㅜ\n맨날똑같이하는데좀만\n물건올라와있어도\n지저분해요ㅜ'], ['눈에보이니 계속하게되죠 ㅜㅜ'], ['해도 해도 끝이없는게 집안일이예요.. 해도 티도안나고 할일은태산이고ㅜㅜ'], ['맞아요ㅜㅜ끝이없어요ㅜㅜ'], ['마저여ㅠㅠ전아직어린아가라ㅜ아침에일어나서아기뒤치닥거리하다보면금새잘시간이예요ㅜ'], ['낮시간은 왜이리 후딱가는지ㅜㅜ'], ['맞아용... 공감합니다.. 집안일 하고해도 .. 끝이안나는1인....'], ['저도 그 플러스 1인입니다'], ['잠이 올락말락 청소기만 밀어놓고 설거지거리쌓여있는데 에라모르겠다 그냥 이불덮고 누웠어요ㅡㅡ'], ['저녁밥하기전에 하심 되겠어요 ㅋㅋ'], ['저도요 어제 폭풍 방정리부터 시작해서 침대매트 바꾸고 주말에 신랑이랑 하면되지만 내맘같이 안되니 그냥 혼자 후다닥ㅠㅜ어제 하고나니 오늘은 버덕만 닦고띵가띵가입니다^^'], ['저도 차라리 혼자하는게 맘편하답니당ㅎㅎ'], ['전 집안일은 포기한지 오래\n어찌해야하는지도 모르겠어요.\n누가좀 와서 해주면 좋겠어요'], ['저도 포기하고싶네요ㅜㅜ'], ['맞아요맞아ㅜ 저두이시간까지 쓸고닦고했는데 변한게없어요ㅜㅜ'], ['하다보면 애기 얼집에서 올시간이네요ㅠ'], ['맞아요맞아 저두그래요 하루가청소하다 끝난다니깐요ㅜ'], ['그런깐요.일은 해도해도 끝이없어요.티도안내고~^^;;;'], ['티안나는 집안일 ㅋㅋㅋ 아 정말 ㅜㅜ'], ['집안일 대충하거나 열심히 하거나\n별 차이 없더라구요\n전 급한 순서대로 하고\n힘들면 건너뛰기 잘합니다^^'], ['건너뛰기^-^ 좋은방법이네요'], ['진짜 해도끝이없어요 빨래돌리고 청소기돌리고 닦고 티가안나요'], ['우리가 서로 알아줍시당 티안나는 집안일 ㅋㅋ'], ['티가안나니...내색좀하고싶은데 ㅠㅠ\n우리끼리 알아주니 덜 서럽네요 ㅋㅋㅋ'], ['우리끼리라도 의지해야죠~^^화이팅'], ['집안일은 안보인게 답인거같아요..보이는건 다 해야할일 같아요ㅠ'], ['눈에 너무 잘보여서 탈이네요ㅜㅜ'], ['맞아요.. \n저는 시작하지도 않았는데 왜 힘들죠??ㅎㅎ \n시작하기가 두려워요 ㅋ'], ['상상만으로도 피곤해지는 집안일ㅠ'], ['저는 워킹맘이라 대충하고 삽니다. 주말에 한번에 몰아서~ 눈에 보여도 모른척 해요 저는....... 못참는 사람이 하는거죠 ㅎㅎ'], ['하하.. 워킹맘은 어쩔수없죵ㅠㅠ'], ['제일허무한일이 집안일이에요 열심히 쓸고 닦고 치우면 그당일날만 흔적있고 좀지나면 했는지도모르게 되어버리니 머라고 이런걸 하나 또 안하고 있거나 안한상태 계속 유지하면 어찌나 안한 태는 잘도 나는지ㅜ 열심히하나 대충하나 누가 알아주는 사람 없는데말이죠'], ['도로묵이어도 안할순없죠ㅜㅜ계속해야죠 힘내봅시당'], ['집안일은 해도 해도 끝이 없는것 같아요ㅜㅜ 끝이 보이면 좋겠어요'], ['끝은 보이지않을꺼같아요 영원히ㅜㅋㅋ'], ['집안일은 미루고있을때만 티가나요 내가 꼼지락대야 돌아가니 안할수도 엄꼬요'], ['ㅎㅎ 맞는말씀이네요'], ['일이 끝도없어요..\n집안일 우습게 보면  안됩니다~\n'], ['누가 우습게보나요ㅠㅠ 집안일은 넘 힘드네요'], ['주말에는 집안일 하기가 더더더 싫고 힘드네요ㅜ 자꾸잔소리하게되고 마녀가 되나봐요ㅜ']]
    
    3415
    진짜 혼자서 소꿉놀이 하듯이 사는데, 왜죠!? 저 진짜 집에 냄비 하나 없거든요!?정말 아무것도 안하고 넷플릭스 보고 그냥 뒹굴거리고, 운동하러 나가고 회사가고 증맬로 재미 없게 사는데, 구런데도 왜 집안일은 끝도 없을까요?🤦🏻‍♀️빨래도 세탁소 런드리서비스 이용하면 직접 하는건 몇번 안되구요,대부분 간단한 식사 우유 시리얼 등그렇게 먹으니깐 그릇 한두개정도 설거지적으니까 진짜 별게 없는데요 ㅠㅠ옷도 항상 제자리 그때 그때 넣어두고치우는거 시러해서 어지르질 않아요. 그런데도 사부작 사부작 집안일 하다보면 오늘 뭐했다고 하루가 저물어갑니다  그래서 곰곰히 생각해보니,주업무는 마켓컬리 박스 버리기 이구요,로켓프레쉬 보냉팩 버리기 입니다 ! 헤헤헤헤 (집밥의 고수님들 존경합니다👍🏻)낮에 빈둥빈둥+사부작 사부작 거리다가 새로운 러닝 코스를 뚫으러 나갔는데용 우와 전쟁기념관 폐장후라 그런지,사람 1도 없고 조명도 적당히 좋고!우리집도 보일정도로 가깝구,달리기 좋겠더라구요! 그래서 오 이거다여기 자주 나와야겠다 했는데 갑자기 비가와여... 겨우겨우 꾸역꾸역 이번주의 미션‘매일 달리기’를 수행하고 대충 시마이하고 들어왔지요 헤헤 ’-‘커피 한잔사고 신나게 집에 돌아왔는데내가 얼마나 우리집을 좋아하는지알았어요 얼마나 집에 빨리 들어가고싶었으면 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ신발들이 들어온 흔적은있고 나간 흔적은 없네요?얼마나 후다닥 들어갔으면짝맞춰 벗어놓을새도 없이 난장판 웁쓰🤦🏻‍♀️ 정리하고 뿌듯하다 사진찍었는데,엄마 보여주니 신발장에 넣어두면안되겠녜요. 아 그렇구나!뭐 그렇다구요..암튼 집안일의 고수님들, 특히 집밥의 고수님들 존경하옵니다!
    
    [['치우는거 싫어해서 어지르지 않는거 자체만으로도 굉장히 부지런하신거ㅋㅋㅋ 전 집에들어오면 일단은 입구에서부터 막얹어놔요 짐이든 옷이든ㅋㅋㅋ 결국엔 산처럼 쌓이는 ㅠ'], ['아 집이 좁아서 그걸 얹어놓으면 제가 사부작 거릴 공간이 없어져요 ㅋㅋㅋㅋ그래서 그런건데 쫌 부지런한척 해봤어요우(이런 예리하신분!)'], ['현관이 예전에 지냈던 합숙소 떠오르게 하네요 ㅎ'], ['죄송해요.....혼자사는집이예여......🥺'], ['ㅋㅋ 드림님댁 정도면 집을 엄청 좋아할만하죠'], ['씨울님이다!!!!오예!!! 저는 옛날집때부터 아무것도 없어도 집순이여써요 ㅋㅋㅋ엄마아빠랑 살때는 방순이였는데, 그걸 엄마아빠가 서운해하셔서 다시 나온거예여 ㅋㅋㅋ'], ['씨울님 저도 만나주세여🙏🏻(갑분고백)'], ['네?! ㅋㅋㅋㅋ 저야 감사하지만🙈 천사 금요일님의 과하신 말씀에 넘 큰 기대를 가지시면 안됩미다 \n부담돼서 이제 모임도 못나가겠어요 ㅋㅋ'], ['전쟁기념관은 밤에 너무 사람없지 않아요?? 무서울텐데요...'], ['음 그냥 음악듣고 뛰는데 저 본관 앞에 광장(?)은 그래도 불이 켜져있어서, 이제 날씨 좀 더 풀리면 뛰는 사람들이 나타나길 바래봅니당'], ['혼자사는데 신발이 저렇게 많다니... 전 정말 소박하네요.'], ['그래서 엄마가 시러하세요 지네냐고....😔그래서 자주 신는 일부만 들고온건데....'], ['진짜 부자네요. 저정도면... 전 운동화 2. 구두 3, 슬리퍼 2개. 제가 쇼핑을 안하긴하네요ㅋ'], ['오 저도 쇼핑 안좋아해요! 뭔가 사러 목적을 가지고 돌아다니고 구경하는거를 별로 안좋아해서요!'], ['제가 오늘 딱! 이생각했는데... ㅋㅋ 제 일기를 대신 써주신듯!!! (너무 격하게 공감했어요)'], ['가지런히님 ㅠㅠ 저 진짜 지금 울고싶은게 아까 분명 한바탕 박스 버리고 왔는데 싱크대위에 요거트통 ㅠㅠㅠㅠㅠㅠ 휴'], ['저는 퇴근후/주말에 집안일 하는게 넘넘 재밌던데요ㅎㅎㅎ'], ['주부100단!!! 버니님 집밥도 엄청나시던데여!!!!!전 안될인간인가바요........이번주 미션은 완료했으니 다음주는 저도 집밥해먹기 미션 해볼까바요'], ['운동화 5개도 많다고 생각했는데 ..넘사벽이시네 ...\n\n근데 ..맨발로 운동화신으시나요.......?'], ['원래 운동할땐 쿠셔닝때문에 양말신고 신는데, 오늘은 런드리 보냈더니 양말이가 없었....근데 평상시엔 맨발에 신어요!'], ['헐 ㅋㅋㅋㅋ 맨발로 운동화신는사람 첨봐요 ㄷㄷㄷ'], ['오 진짜요? 전 겨울에 추울때 빼곤 맨발에 신는데..... 발에 땀이 없는편이라 가능한일이긴해요!'], ['운동화 부자시네요 ㅎㅎ'], ['뾰족힐도 신고, 플랫슈즈도 신고, 더비슈즈도 신고, 다양하게 신어융 근데 게으름뱅이라 근래 운동화 자주 신게되네융 ㅋㅋ'], ['저도 러닝훈련 들어가려 하는데...오늘도 걍 지나치네요..비가 오다니 모르고 있었는데..진짜오네요..ㅋ'], ['비온다구여 진짜라구여! 거짓말 아니예요 마음같아서는 10K뛸라그랬는데 아깝다!!!!! ㅋㅋㅋㅋ(라고 뻥을 쳐봅니다!)'], ['맥스97 98 베이퍼 빤쓰\n아기들 다 이쁘네요+_+'], ['우잉 월플라워님 운동화 좋아하시는군요! 전 나이키는 맥스만요! 97 98 베이퍼 95 순으로 좋아합니도 ㅋㅋ 그런데 최애는 척테일러 ㅋㅋㅋㅋ세상에서 제일 좋아요! (뻥치시네!)'], ['다섯명 놀러오면 다같이 스탠딩으로 파티해야됩니도 뚜쉬뚜쉬! 앉을자리가 없거든요 헤헤 집안일 정말 네버엔딩🤦🏻\u200d♀️'], ['전쟁기념관 야간뷰가 좋네요~\n지네 운동화 잘보고 갑니다 ㅋㅋ'], ['전쟁기념관 좋죠? 미사일이랑 탱크 옆을 뛰고왔써요 지네운동화와 함께💕'], ['벤시몽!! 저도 사고싶었는데 제가 신으니까 왠 다리짤린애가.... ㅋㅋㅋㅋㅋ 운동화 이쁜게 넘 많아요 🤩'], ['벤시몽 제가 참 사랑하는데여, 유치원 실내화냐고 놀려도 꿋꿋하게 신습니당! 근데 하도 많이 신으니까 항상 빵꾸가 나여.......끈없는 하얀 고무신 벤시몽은 빵꾸날때까지 신고 버리기를 반복해도 1년에 두켤레씩 꼭 사는 아템이예여!'], ['저렇게 신발 많이 둘수있는 현관이 있으심에 부럽슴다^^'], ['이집의 지분을 따져보면 저 신발장까지만 제꺼정도 됩니다 ! 😔 부러움을 거두시옵소서🙏🏻'], ['치우기 싫어서 안어지르는 사람 = 저구여 ㅋㅋㅋ 심지어 신발도 벗으면 바로 신발장행.. 엄마가 이렇게 살면 아무도 너랑 살기 싫어할거라고 하곤 했는데....'], ['인장님 그렇게 살아도 같이 살수있쪄요! 어뜨케 8월말로 알아보구여? ㅋㅋㅋㅋㅋㅋㅋ(한없이 질척거릴꾸😘)'], ['이미 고수이신듯 ~!'], ['무슨 고수를 말씀하시는거죠?제가 잘못들은거겠됴? ㅋㅋㅋ'], ['정리하는게 너무 귀찮다는걸 알아도 자꾸 어지르게 되더라구요! ㅠㅜ 꿈님은 실천하시니 고수이신걸로~~'], ['아이쿠 감샤해여 그정도도 고수라고 칭해주시니 왠지 뿌듯합니다 헤헤😍'], ['런닝 좋아하시나봐요? 저도 한때 열심히 달렸는데 일에 치이니 그것도 귀찮다고 멀리하게 되더라구요 ㅠㅜ 나이키 어플키고 달리는 재미가 쏠쏠하죠'], ['네 달리기 좋아하는데 좋아만해요...(짝사랑같은존재,,,좋아하지만 애틋한관계...잘 못만나요....그이를)'], ['신발 다 혼자 신으시는거예요?'], ['네...정리가 안되어있는것이 포커스인 사진이었는데,, 지네가 더 충격이셨나봐요....실은..신발장안에 더 있는데......🙊'], ['ㅋㅋ저도  98 파슬 있는데 눈에 딱 보이네요🤭'], ['오 가북이님도 파슬 신으시는군요! 저는 건담도 가지고 있는데 막상 신는건 파슬 ! 파슬이 더 많이 신게되는거 같아요!'], ['색이 무난해서 여기저기 신기 편해요 ㅋㅋ98은 발이 커보여서 97이 더 좋더라구요 98신면 오리발 신은 느끼,,,ㅁ이랄까요 ㅋㅋ'], ['97에 비하면 쉐잎이 옆으로 좀 더 통통해서 발이 통통해보이죠 ㅋㅋ그 느낌으로 신긴하는데, 저도 97의 그 얄상한 느낌때문에 맥스들 중 가장 좋아해요! 근데 같은 사이즈라도 97이 더 큰 느낌! 여성 5반 신는데 베이퍼는 그나마 발에 착 감기는데 97 98은 좀 발이 놀아요'], ['현관 신발 너무 공감되네요 ㅎㅎ 저희집은 신발장이 작아서 저도 저렇게 신발이 현관에 주루룩 있어요👟 ㅋㅋㅋㅋ'], ['전 이제 다 넣어놨답니다 현관 면적 체감 백평 캬캬캬캬(이정도면 뭐 거짓말의 생활화 찡긋)'], ['첫번째 현관 사진 보고 빵터졌어요ㅎㅎ'], ['워 저도 놀랬어용! 저정도인줄은 ㅋㅋㅋㅋ그래도 다 치웠져요!'], ['너무이뻐용'], ['무엇이 말이옵니까? 😊'], ['사진요 ㅎㅎㅎ']]
    
    3491
    해장엔 라면~ 둥이 오늘 얼집에서 꽂구경간데요~  출근시켜놓고 집안일 하기전 해장부터 해봅니다ㅋ 오늘도 좋은하루되세요~^^
    
    [['옆에 갓김치 ㅋㅋ 악진짜 사랑해유 갓김치 ㅋㅋㅋ 한통뿌니 안남앗는대 엉엉'], ['갓김치가 딱 제가좋아하는만큼 익어서 아주 맛있어요ㅋ저도 얼마 안남앗는데 친정 냉장고 털러가야되여ㅋ'], ['히히 저도어머니 찬스 ㅎㅎ 아껴먹어야겟어요 ㅎㅎ 조만간 또 갓김치 해서 주시겟지요?ㅋㅋㅋ'], ['살포시~ 얘기하세요ㅋ 아..김치가 너무 맛있어서 그것만 먹다보니 다 먹엇다고ㅋ 그럼 또 해주실꺼에요ㅋ'], ['와우ㅎ저 방금 침이..막..삼켰네요ㅋㅋ갓김치 올려서ㅎ호로록~~ ㅎㅎ맛나게 드시고 오늘도 홧팅! 입니다ㅎ'], ['김치가 침이 솟져?ㅋ 아주 많이 익은건아닌데(제 기준ㅋ) 색깔이 많이 익어보여요ㅋ'], ['라면에 김치. 꿀조합이네요 ㅋㅋ 맛있어보여요 :)'], ['허하던 속이 든든하게 채워졋어요ㅋ 이제 일해야되는데 자고싶어졋네요ㅜ'], ['맛있겠어요 갓김치에 오메나 ~~'], ['살찌는건 다 맛있자나용ㅋ 밥 말려다가 참앗어요ㅋ'], ['갓김치에 라면...헝헝헝\n익은갓김치 정말이지 너무 맛있게보여요ㅠ\n침샘 제대로 자극받았습니다...'], ['한입 드리고싶네요ㅋ 아침식사하셧죠? 오늘도 날씨가 좋아요~ 좋은하루되세요^^'], ['라면도 맛있겟지만 갓김치가 !!!! ㅎㅎ 맛있게 드세요^^'], ['갓김치가 끝내주게 맛나게 익엇어요ㅋ'], ['해장은 라면이죠ㅋㅋ계란 2개넣어서 드시면 속이 훨씬 더 편할꺼예요ㅋㅋ'], ['하나만 넣엇는데 두개 담엔 넣어볼께요ㅋ 라면엔 계란~ㅋ'], ['라면  얼큰하니  해장되시겠네요 \n맛있겠네요~~맛나게  드세요'], ['해장됏어요ㅋ 근데 잠이와서 큰일이네요ㅜ청소부터 해놓고 쉬어야 하루가 편한데ㅋ'], ['해장으로 라면 좋지요 ㅎㅎ\n김치에다 먹음 맛있는데~배고프네요'], ['사실 집에서 혼자 해장할만한게 라면밖에 없어요ㅋ 식사하세요~~^^'], ['라면으로 해장하시네요 익은 갓김치랑 먹으면 더 맛있겠네요'], ['라면엔 무조건 김치지요ㅋ 잘익은 김치하나면 한냄비 뚝딱~'], ['갓김치도 너무 잘 익었네요\nㅎ 라면에다가 먹기 딱이겠어요`~ㅎㅎ'], ['딱 제가 좋아하는 스타일데로 익엇어요ㅋ 라면에 먹을때가 젤 맛있어요^^'], ['맛난 갓김치에 얼큰한 라면 맛있지요~~속이 확풀어지면 편해지죠~~\n침고이는 갓김치 ~~~ㅎㅎ'], ['지금 너무 편해져버렷어요ㅋ 설거지도안하고 싱크대에 갖다놓고 그냥 드러누웟어요ㅋ'], ['갓김치가 넘나 맛나게 익었구만요ㅠ해장이 필요없는데 라면 땡깁니당ㅎ'], ['자 이제 가스불을 켜고 냄비에 물을 붓고 끓입니다ㅋ 보글보글~ 맛나게 끓으면 호로록~ㅋ 어여 드세요^^'], ['그럴까봐요ㅠ아들도 아침은 바나나에 구운계란으로 줬는데..고민입니당ㅎ'], ['저희 둥이는 귤이랑 포도 먹여서 보냇어요ㅋ 오늘 제철 백운대 벚꽃보러간다더라구요..부럽게 ㅋ'], ['전 어제 술도 안마셨는데 왜케 침이 꼴깍 넘어가는걸까요 ㅠㅋ먹고파요'], ['배고픈거아니에여?ㅠ 언넝 드세요~ 간단히 라면도 좋자나요~'], ['전 라면먹으면 소화를 잘 못시켜서 한달에 한번 먹을까말까하고있어요 ㅠ배는고픈데 막둥이가 쭈쭈를 안놔주네요 ㅠ'], ['아.그러시구나ㅠ 우리 막둥이가 아직 배고픈가봐요~ 좀더 주세요^^'], ['돌지난 언니라서 이걸론 배가 안찰텐데도 구지구지 물고 계시네요 ㅠ안주믄 짜증내고 ㅠ하 ㅠ'], ['헉!벌써 돌이 지낫어요? 오메메~진짜 빠르다요~~ 이제 쭈쭈 내놓으라고하세요ㅋ'], ['라면에 갓김치최고네요.갓김치도 좋아라해서 눈이 먼저 가잇네요ㅋ'], ['파김치는 호불호가 갈리는데 갓김치는 많이들 좋아하시더라구용~ 너무 잘익엇어요ㅋ'], ['갓김치는 익을수록 더 맛잇잖아요~좋아하는거라 더 침이 고여요'], ['와아 저도 국에 갓김치 먹었네요 우리 아들은 놀고 있네요'], ['오늘은 바람도 덜하고 먼지도 없으니 아가데리고 잠깐 외출 괜찮을거같아요~ 즐기세요^^'], ['아침 라면이좋지요 ㅋ 김치까지먹으면좋것어요 ㅋ 점심은 라면먹어야겠어요'], ['라면 누가 만들엇는지 참 좋은 음식인거같아요ㅋ'], ['점심 라면에김치먹으니 너무맛 나드라구요 역시\n라면은짱\n이에요'], ['라면 안먹어본지 오래네요ㅠ모유수유중이라ㅠ이제 곧 도전해봐야겠어용'], ['수유중엔 가려야할게 너무많져ㅠ 쫌만 더 참앗다가 단유하시면 실컷드세요ㅜ'], ['캬 갓김치는 라면에 최고인것같아요! ㅎㅎ'], ['파김치 갓김치는 뭐 말이 필요없져ㅋ 맛있어요~'], ['라면 먹은지 오래 돼 더 맛있어보여요\n오늘 아점은 라면을 먹어줘야 겠어요'], ['라면은 가끔 먹어야 더 맛있어요ㅋ 넘 자주먹으니 그냥 별 느낌이 없더라구요ㅋ'], ['맞아요\n저도 육휴 시작하면서 거의 매일 혼자 아점을 점심으로 때웠더니 한동안 질려서 지금 안먹고 있꺼든요\n그동안 안먹었으니 이제 먹어보려구요'], ['자도 오늘 아침에 라면 먹엇는데 해장은 아니지만 어제저녁부터 참던거러 아침에 후루룩'], ['아고 참다가 드셧으니 얼마나 맛있엇을까용~ㅋ'], ['너무 먹고 싶어서 밥까지말고 먹고 나서 후회햇어요 ..ㅎㅎ 또 저녁은 굶네요 ㅋㅋ'], ['저는 6시쯤 김밥한줄먹엇는데 지금 배고파요ㅠ 먹을게없어서 다행이다생각중이네요ㅋ'], ['옆에 갓김치가 왜이리 맛나보이는거지여?? 라면에 먹으니 더 맛날거 같아용!!'], ['갓김치가 라면이랑 잘어울려요~ 어떤 김치라도 라면이랑은 짝꿍 아니겟어요~?ㅋ'], ['갓김치보니 침이 고이네요\n너무 맛있어보여요 라면이랑 환상궁합일듯요.\n맛있게 드세요'], ['맞아요ㅋ 둘이 궁합이 잘맞는거같아요ㅋ 갓김치가 딱 맛있게 익엇네요^^'], ['라면에 계란 하나 탁 ㅎㅎ 저도 어제 한잔 했는데 괜히 라면 먹고 싶어지네요 ㅠㅠ 참아야합니다'], ['왜 참아요~ 해장 하셔야져ㅋ 언넝 물부터 끓이세요~^^'], ['갓김치가 잘익었네요 라면에 한입이면 꿀맛일듯 밥은 안마셨나요 ㅜㅜ'], ['라면이 뿔어서 배가부르더라구요ㅋ 첨엔 밥도 말아먹으려고햇거든요ㅋ 참앗습니다~'], ['라면에 갓김치 맛잇어여 ㅎㅎ 어제 술드셧나바영 저희애들도 어제 벚꽃보고왓다하더라구여 ㅎㅎ 어린이집에서도 바깥놀이를 슬슬 시작하네여'], ['네~오늘 벚꽃보러가구 담주엔 딸기체험 마지막주는 백운산간데요..걱정걱정~ㅜ'], ['해장에는 라면입니다~김치랑먹음 뭘먹어도 맛있습니다.ㅎㅎ'], ['라면도 라면 그대로 맛있고 김치도 김치 그대로 맛있는 음식인거같아요~ㅋ'], ['맞아요~두가지 조합은 진짜 너무 좋아요~\n맛있는 김치좀 데리러 가야겠어요~'], ['갓김치 안그래도 좋아하는데 뜨끈한 라면에 갓김치 짱입니다요^^'], ['밥을 못말아먹어서 쫌 아쉽네요ㅋ 아깐 배가불러서..ㅋ'], ['앗 ㅋㅋ 진짜 해장엔 라면만 한게 없어요 ㅋㅋ 출근해서 맨날 옆에 분식집가서 먹던 생각 나네요 ㅋㅋ'], ['집에서 혼자 해장할만한게 사실 라면밖에 없자나요ㅋ 그만한것도없궁ㅋ 이제 낮잠좀 자볼까싶어요ㅋ'], ['숙취엔 라면만한것도 없습니다 갓김치 인가요? 와 맛있겠어요'], ['집에서 할수있는 최대한의 숙취해소 음식 아니겟습니까ㅋ 갓김치 맞아요~']]
    
    3506
    남편 집안일 얼마나하나요 직장다니다 육휴중인  백일맘인데 독박육아에 집안일도 안하고 그나마 치워놓면 어지르지나 말든가...복직해도 저럴지 꼴도 보기 싫으네요..다른건 다 내가할테니 욕실청소랑 쓰레기(분리수거도아니고 일반쓰레기봉투)버리는것만 해달라했더니 더러운것만 시킨다나?? 그럼 딴걸 다하라고했더니 그냥 무시하네요...애기를 안아주길하나...분유를한번 타서 먹여주길하나...내가 집에있으니 참자..하다가도 꼭지가 도네요...오빠  형부들은 결혼후 3년까지는 다들 애보는거며 집안 힘든건 다 하는걸 보아와서 더 짜증나네요... 이사람도 결혼 전엔 자기가 집안일도 같이  많이 할꺼고 애기낳기만하면 지가 다 키운다하고 하더니 이렇게  힘들게 하다니...현재 기본급 나와서 생활비도 똑같이 부담중인데  그것마저 안나오면 어깨에 힘을 얼마너 더 넣으려고...           
    
    [['남편도 초기엔 좀 그러다가 싸우면서 얘기하면서 조금씩 나아지더라구요.. 한동안 공백기가 있으면 다시 초기화되긴하지만...'], ['어려워요..'], ['저도요 그것땜에 초기에 엄청 싸웠어요\n한번은 진지하게.. 내가 당신이랑 살이야하는 이유가 하나라도 있나 그냥 따로 살자고 했더니 좋아지더군요'], ['그렇게 말해봐야겠어요'], ['룰을 정해 두시는게 좋을 것 같아요. 남편 담당 영역을 정해두시고, 안 지킬 시에 벌금이라던지, 육아 몇시간이라던지.. 아님 남편 빨래랑 밥 해주지 마세요.'], ['그렇게 룰을 만들자고했더니 쌩~~\n밥 안해주면 한대 때릴꺼같아요'], ['전 끝까지 안했어요. 남편 할때까지.. 그랬더니 지가 답답해서 하더라구요.  너도 안하면 나도 안해라는 마인드로 정말 하지 말아야해요. 남자들은 진짜 똑같이 안당해보면 안고쳐지더라구요'], ['이런말씀드림 더 화날거 같은데 ㅠㅠ 첫애땐 제가 거ㅣ 다했는데 둘째부턴 조금씩 남편이 하더니 셋째 임신중 일과 살림은 거의 다하네요 저는 컨디션 조을때만 ... 애 버는것도 힘든거라며 남편이 이해 마니 해줘요'], ['어찌 바뀌신건지..'], ['첫 애때는 남편이 도와줄 시간이 없었어요 거의 새벽에 출근해서 12시 넘어 출근하는 날이 많았거든요\n둘째때는 이직했는데 제가 입덧기 심해서 남편이 많이 이해해주고 아이 씻기는 건 남편이 했어요 그런데 하나 하나 남편이 도와주다보니 많이 맡기게 된거 같아요\n지금 셋째 임신중인뎉남편이 도와주지 않았음 셋째 생각은 하지도 않았을거 같아요'], ['육아도 집안일도 시키세요. 안그럼 진짜 하나도 안해요. 저희집은 집안일은 재활용버리기, 육아는 집에 있을때는 수유, 목욕, 기저귀 갈기등 적당히 하는데 그것도 시켜야 해요.'], ['시키면 가끔 하는척만하고 일하느라 힘들다고 징징'], ['맞아요. 쉬는날 청소기 돌려달라하고 저는 바닥 물걸레 돌렸는데, 바닥 하나도 안치우고 보이는데만 청소기를 돌리고..아우 머리야 진짜 팰수도 없고ㅠ'], ['맞아요..힘들면 하지말래요...'], ['아기어릴땐 쓰레기버리는것정도 해줬구요. 외벌이라 어린이집 보내고나서부턴 도와주는거 전혀없구요. 가끔 아이 목욕씻겨주는정도... 내년부터 맞벌이 계획인데 트러블 많이 생길것같아요. 집안일이라면 끔찍하게 생각하는 사람이라... \n다행인건 돈쓰는거가지고 터치 일절안하고 돈도 외벌이로 먹고살만큼 벌어서 그냥 그러려니해요..'], ['게다가  똑같이 생활비 부담하는데도 젖병하나 기초화장품 하나 사는것까지 터치하는사람..이를  어쩐대요..'], ['미친거같아요 화장품사는걸 왜 터치해요? 본인은 아무것도 안사고 안바르고 하나요? 우리남편도 돈쓰는거 잔소리 좀있지만 화장품까지 간섭은 안하는데 정말 남자가 쪼잔하기가 도가 지나치네요 와 화나요!'], ['기초화장품은 당연한건데 그거가지고 간섭을 하면 우째요.ㅡㅡ 정말 너무하네요.'], ['자기꺼보다 비싸다고요..이런 쪼잔함 못보셨죠?\n집에 눌러 앉는다고하면 이혼하자고하려나.. ㅋ'], ['sbs스페셜 아이낳아?말아?(다큐제목이예요^^ㅋㅋ) 이거 남편이랑봤는데 그뒤로 좀더 잘도와주고있어요^^'], ['찾아서 같이 봐야겠네요'], ['안하네요 진짜.ㅡㅡ 더럽게'], ['그 두가지만 시키지면 안돼요 아기도 보라고하고 애 목욕도 시키라고 하시고 가끔 설것이나 밥차리는것도 시키셔요 남편은 퇴근이 있지만 아내는 24시간 노동이잖아요 육아는 같이 하는거예요 더군다나 기본급 생활비에 보태고 계신데요 더 떵떵거리셔도 돼요 애낳고 힘든 몸도 생각하셔야죠'], ['222 도와주는거에 고맙다 생각하지말고 당연하다 생각하세요.. 솔직히 여자만 집안일해야되는건 편견이죠ㅡㅡ 조선시대인가; 같이 사는집이잖아요'], ['씨도 안먹히는 인간하고 살고있네요\n본인이 세상에서 젤 힘든일 하는사람 같아요\n제가 직장생활 평생 할꺼같냐며..\n애기 잘때 논다고 생각하는인간'], ['그럼 뭐 본인은 직장생활 평생하겠어요? 요즘 정년퇴임 나이가 몇인데 늙어서 이혼당하기 싫으면 똑바로해야죠 임신출산육아가 얼마나 힘든데 이럴때 같이 안하면 그게 부부인가요? 평생동반자라고 생각하고 결혼했는데 힘들때 나몰라라하는게 남편이예요? 집안일, 육아는 도와주는게 아니고 같이 하는거예요 뭐 누구는 직장 안다녀봤나요? 어디서 애낳아본 엄마한테 본인이 제일 힘들다소릴해요  전 전업이지만 임신하고서 입덧도 심하고 체력도 딸려서 남편 많이시켜요 그래도 서러워서 울때 많아요 님 너무 혼자 다 끌어앉지 마시고 같이 하세요'], ['쌍둥이 31개월째 독박인데..\n아빠는 1도 안해요? ㅠㅠ\n독박의 정석입니다...'], ['첫애때는 철없이 친구만 만나러 다니고\n그러더니 지금은 그래도\n설거지,빨래, 쓰레기버리기,\n애기목욕, 애기재우기,등등 하네요\n그래도 술은 아주 오지게 쳐마셔요ㅜ으휴\n알콜중독인가 싶기도하고....'], ['그래도 할건 다하며 노니 덜 밉겠어요\n놀기만해요'], ['청소, 빨래 빼고는 거의다해요 신랑이 교대근무라 시간 맞으면 애들도 학원서 찾아오고 상황에따라 병원도 다녀오고 분리수거, 주말에 저 늦잠자라고 애들 밥챙겨먹이는것까지 청소빨래빼고는 거의하네요 쓰레기 버리는것도 당연히 신랑이 해요'], ['그게 이상적인 생활인데..'], ['전업이라 90%제가해요. 남편은 분리수거하러 같이 가는거랑 고양이화장실담당정도? \n요즘 입덧때문에 음식물쓰레기부탁하고있는데 내일해줄게 내일해줄게 자꾸미루네요ㅡㅡ 바쁜척해요ㅠ냄새나는데ㅠ'], ['청소기,물걸레청소기는 잡아본적도 없답니다..^^;;; 자기가 만지면 고장날거같다나 뭐라나;;'], ['전 똑같이 생활비 부담하고 있는데도 그래요..\n육휴전 임신때도'], ['전 전업인대도 청소 애기꺼손빨래 세탁기돌리기 아침 저녁밥해서 밥상차려주고 설거지마무리까지 싹다 남편이 해줘요\n전 애기만 봐요.. \n이렇게 안도와주면 와이프가 힘들다는걸 느껴야지 해주실거같아요 \n'], ['전 육휴라 신나서 노는줄알아요'], ['첨부터 신랑이 음쓰랑 분리수거 담당이엇고ㅎ 맞벌이할땐 집안일같이하고 지금 둘째임신중인데 컨디션 안좋은날엔 첫째목욕, 저녁설거지까지해줘요~ 간식도 주고...입덧할땐 배달반찬으로 저녁준비도 다하고ㅜ 알아서 하진않는데 시키면 다해주는편이에요ㅎ 몸안좋거나 힘들때 시키면 짜증내기도 하구요ㅋㅋ'], ['결혼후 청소.분리수거.설거지반반\n첫째출산후 퇴근후애보기.젖병소독반반.청소.분리수거.빨래널고개기.설거지반반\n둘째 젖병소독.애보기.청소.분리수거.빨래.\n육아집안일  공동입니다..애들 좀더크기전까진 둘다 전투육아모드입니다... ;;'], ['저는 남편이 더 많이해요! 제가 더 퇴근을 훨씬늦게하거든요! 차라리언능 복귀를 하세요 !!'], ['저도 그래야하나 싶지만 제일이 배가 될까 두렵네요..직장에 육아에..'], ['애 밥이야 굶기지 않을거고...집에돌아와 씻기고 재우고 하는거 하고 집안일은 걍 손놔버리세요 본인것만 하고.. 바쁘고 힘들다고ㅠㅠ 둘다 바쁘고 힘드니 이삼일에 하루라도 집안일 사람 불러서 하자고 하구요. 그럼진짜 도우미를 부르던가 돈아까움 본인이 조금이라도 하던가 하겠죠 ㅠ'], ['자기엄마랑 같이 살자고 할꺼같아요..\n임신때  막달까지 입덧도 심하고 너무힘들어서 퇴근후 쓰러지기 일쑤였는데 바닥에 먼지가 뒹굴어 다닐 지경이었어요'], ['그럼 강력하게 도우미를 쓰자고...하심이...엄마랑 같이살자하면....우리엄마랑도 같이살자해여...다같이살자고 ㅡㅡ 하 저런남편 진짜싫어 ㅠㅠ 저같음 진작갖다버렸음 ㅠㅠ요즘에도 저런 똥배짱부리는 남자가 있다니...제친구는  남편이 저래서 애는 없지만 1년만에 이혼했거든요'], ['저도 진지하게 생각중이에요'], ['결혼초반엔 지가 쳐먹은 식기도 치울줄 모르는 ㅂㅅ으로 키워놨더라구요 시엄니가.. 다시 가져가라고 던지고 싶은심정입니다'], ['저희집에있는애랑 똑같네요'], ['여기두요..ㅡㅡ\n결혼3년차에 개수대 갖다놓더군요..휴우ㅡㅡ\n결혼5년차에 분리수거도합니다ㅡㅡ\n사람만들어놓기참어려버요;;\n당연한게당연한게아니니\n매번싸우고..'], ['외벌이때도 음식물 화장실청소 쓰레기버리기 분리수거 걸레빨기.... 사실더러운것은 결혼하자마자 지금까지(13년차..) 다 신랑 몫이에요... 세탁기 돌리기 널기 뻘래개기 까지 요새 추가됏다고 좀 궁시렁거리는데.. 성격인거같아요..'], ['부럽네요'], ['결혼하고 맞벌이할때 정했던거 그대로 하는중이네요.(신랑 술 엄청 조아하고 술마심막말하고 그래요. 그것땜에 애기어릴때 이혼하고 싶을정도 였고.. 얼마전애 싸우고 집나간 신랑이라는... 다 만족하긴 어렵네요.) 먹는것 좋아해서 맛있는 음식 해주면.. 거의 다 본인이 하네요'], ['아직 백일밖에 안 되서 그래요...몇년 차 아빠들 다 잘해보이는데 그게 연차가 쌓여서 그렇지 처음엔 마찰 많습니다. 저도 뭐 시키면 남편이 짜증 또는 생색내고 하더니 자기도 익숙해지고 적응하는지 점점 할 수 있는 일이 늘더라고요. 돌 지나니깐 3박 4일 독박육아 가능한 아빠가 되었어요. 하나씩 시켜 나가보세요. 화나지만 꾹 참고 우쭈쭈 칭찬 꼭 해주시고요!'], ['맞벌이라서 반반해요.\n쓰레기 버리기, 분리수거는 남편이 요리는 제가 하고요. \n세탁, 설거지, 청소는 나눠하거나 집에 같이 있을 때 함께 하네요.'], ['저 육아휴직했을때는 신랑이 퇴근해서 오는것만 기다렸답니다. 일시킬 사람왔다 눈반짝!! 모드였어요 ㅋ 전 종일 애보느라 집안일 하나도못했다 시전 ㅠ (사실이었지만요) 퇴근후 신랑이 쌓인설거지 애목욕 쓰레기처리 등 맡아주고 애기저귀좀갈아줘~ 분유좀타와줘~ 애좀안아줘~등.. 자꾸시켜야해요. 첨부터 시키다보면 익숙해져요~'], ['제가 바보같은가봐요..한번 말해서 안하면 짜증내면서 제가 해요..맨날 힘들다고 피곤하다 징징대는소리 듣기 싫어서'], ['토욜아침.. 아침먹고(이건제가했어욤) 저는 다시 침대에 누워있으라하고..음쓰버리고 분리수거하고..고양이 다섯마리 화장실청소하고 설겆이하고 있는 즈희신랑이 넘 고맙네요..저러고도 생색안네요..신랑분께 욕좀해주세요'], ['본인이 세상에서 젤 힘든일을 하시는분이라 자기 애쓰는거 생각안한다고 되려 저를 원망해요\n욕 바가지로 하고싶어요\n새벽까지 티비보다 늦게자고 온몸이 쑤신다며  자고있네요'], ['.'], ['평일엔 거의 저혼자 독박육아구요~~신랑 주말도요새는 계속일해요ㅠㅠ바빠서...하지만 애기랑놀아주고씻기고재우고 집안일도 요리는 신랑담당이에요'], ['부러워요..쉬는날도 저혼자 ...항상 힘들고 피곤하다하니..'], ['라면끓여 쳐먹고도 스프흘려놓고 봉지 고대로 놓길래 시누이한테 얘기했더니 지네엄마가 아들하나라고 그리 키웠다길래 반쯤 포기했는데 5년차... 애셋인데도 안고쳐져요... 저도 복직 얼릉해야하는데 임신중이라... 게으른것들이 또 성욕은 많아서ㅜㅡ \n저는 제아들들 절대 이리안키우려고 지금도 할수있는건 다 시켜요 집안일'], ['라면봉지...ㅋ어쩜 그리 똑같은지..저도 아들이 아빠랑 똑같을까 겁나요'], ['독박육아에 집안일해서,,,,그래서  전 절대 바깥일 안할려구요.  돕는것도 할 놈이 해준다고,,   안하는 놈은 안하기때문에,,  집안일,  육아에 돈까지 벌어주면 저만 개고생아닌가요.  그냥  집구석청소만하고  쉴래요ㅋㅋㅋ어차피,  전  쓰는돈도 없으니깐요'], ['화장실청소.씽크대청소는 신랑담당~연애때부터 더러운곳은 자기가 청소해주겟다하여 결혼생활쭉 해오고잇어요~'], ['자꾸시키세요.계속시키시구요.\n안들어주면 원하는걸 해주고 한개씩 얻어오세요.\n그리고 당연히하는것으로 인식을 자꾸시키셔야해요.\n고마운게아니예요.\n같이하는거고 당연한거예요'], ['집안일 육아 5프로도 안해요 너무 싫어요']]
    
    3516
    🐷 집안일 하시나요? 등원했으니 집안일 스타트 하려구요ㅎ 어제 하루 집안일 안했다고 엉망이네요ㅜㅜ집안일 하려니 미세먼지가 참 안도와주네요그래도 해야겠지요ㅎ집안일 시작하시나요?
    
    [['미세먼지 나빠도 청소는 해야겠지요 저도 청소기 돌리고 창문닫고 물걸레질 했네요 저희도 공청 풀로 돌아갑니다'], ['저도 시작해야하는데 빨래만 돌리고 커피 한잔 마신다고 그냥 궁딩이 붙였네용.ㅠ.ㅠ'], ['저는 주말내 빨래 돌려서 집안일만 후딱 끝냈네요 집안일 언제하나 했는데 하고나니 속은 시원하네요ㅎㅎ'], ['ㅎㅎ시작해야지ㅡ..하고 생각만하고잇네요ㅠㅠ'], ['맘 먹었을때 후딱하고 치워야지요 저도 하기 귀찮았는데 하고나니 개운합니다ㅎㅎ'], ['역시 부지런하신 분!!! 칭찬합니다!!'], ['그나마 아이가 등원해서 집안일하지 안그럼 또 귀찮아서 안할듯합니다ㅎ'], ['어제 차니 없을때 혼자 사부작 대청소 쌱 했지요~그리고 출근해서 코피 한잔 합니당^^'], ['어제 하셨으면 오늘 하루는 쉬어도 되지요 역시 출근하시고 커피한잔은 필수로 마셔줘야지요ㅎ'], ['넵 집에가면 난리부르스 어지럽혀져 있겠지만..회사 있는 지금 잊고싶네여 ㅋㅋ'], ['ㅎㅎ 그건 나중에 집에가시면 생각하세요 회사에서는 회사일만 신경쓰시구요'], ['저는 왜 집안일을 쉬지않고하는데도 계속치우고있는거죠?ㅎㅎㅎ성격이 하나라도 제자리에없음 불안해서 미치겠어요!  열심히 청소정리하고 잠시 쉬네요ㅋ'], ['맘님 저도 물건은 제자리에 항상 있어야해요 그게 성격인것 같아요 어쩔수 없지요 저도 청소하고 커피한잔 하면서 쉬고있어요ㅎㅎ'], ['일단 세탁기는 가동시작했어요 ㅎㅎ 주말만 되면 낮잠도못자고하니 수면부족이네요ㅠ'], ['세탁기부터 열일 하는군요 아이 등원했을때 얼른 주무세요 저는 막상 잘라하니 잠도 안오네요'], ['한숨자고 점심때 일어나서 느즈막히 밥먹고~ 이제 주방가동 좀 할까하고있어요 ㅎㅎ 하기싫으네요ㅠ'], ['아 정말요? ㅎ 저도 어제는 집안일이 어찌나 하기싫은지 근데 하고나면 또 개운한데 말이지요'], ['맞아요 하고나면 뿌듯한데 말이에요~ 어제 느즈막히 했다가 초저녁부터 온집안 굽굽해서 에어컨 가동했네요ㅠㅠ'], ['잘하셨어요 굽굽하면 에어컨 틀면 한방에 날아가지요 저도 아이 하원하고 에어컨 잠시 틀었네요'], ['전 그래두 창열고 거실만 매트들어서 후딱햇어용  커피한잔후 설거지하고 빨래개려구용'], ['저도 청소기 돌릴때만 창문 열어놓고 후딱 닫았어요 공청이 열일 하네요 저도 집안일 끝내고 커피한잔 합니다ㅎ'], ['그춍 공청은 계속 일을하고 ㅋㅋ 전 나갓다 더버서 잠시 에어컨 가동햇어용\n  샌드위치와 컵커피마셔요'], ['나갔다오시면 더울것같아요 날이 어제오늘 덥기는 하네요 그래도아직은 에어컨 안키고 있어요ㅎ'], ['저두 오늘도 미용실 다녀오긴햇는데  그냥 샤워하고 쉬구잇어용'], ['오늘 미용실 다녀오신건가요? 새글을 안봐서 몰랐네요 샤워하고 시원하겠는데요ㅎ'], ['이제 건조기 넣고 엉망인 집 보며 커피 마시며 마음 다잡고 있네요^^'], ['ㅎㅎ 집안 엉망되어 있으면 치우고는 싶은데 막상 귀찮은날이 있지요 얼른 치워놓고 푹 쉬세요'], ['환기 살짝하면서 청소기 돌렸어요. 걸레질만 오후에 하면 될것같아요.\n청소 빨래는 다 해두고 나왔어요'], ['저도 청소기 돌리면서 창문 열고 물걸레질 할때는 닫았어요 그랬더니 공청이 막 돌아가네요 반이상은 하고 나오셔서 개운하겠는데요ㅎ'], ['네, 볼일 다 보고 이제 집에 들어왔어요.\n2차전으로 다시 물걸레질하고.. \n분명히 빨래 바구니에는 또 쌓이겠지요? ㅎㅎ'], ['아 갔다와서 하기 제일 서글픈데 그래도 후딱 물걸레질 깨끗하게 하셨겠지요?ㅎ'], ['저녁까지 다 먹고는 다시 집 전체 바닥만 걸레질 했어요. 세탁기도 또 돌아갔네요. ㅎㅎ\n'], ['세탁기는 늘 열일 하는것같네요ㅎ 내일 또 집안일 해야하는데 늘 꽃가루가 신경쓰이네요'], ['이제 아침 먹었어요\n청소 시작해야지요\n바쁜 월요일이에요'], ['간단히 아침드시고 집안일 하시는군요 저는 집안일부터 하고 밥 먹으려고 일단 다하고 커피부터 마시고 있어요ㅎ'], ['환기는 해야할꺼같아서 청소하면서 창문은열어놨네요ㅠ.ㅠ 귀찮아도 후딱해놔야 또 저녁엔 느긋하게 쉬겠죠'], ['맞죠 저도 주말내도록 환기를 못시켜서 청소기 돌릴동안만 창문 열었어요 이제 청소다하고 쉬네요'], ['저는 어제 대청소 해서 오늘은 빨래만 하고 청소는 휴대용청소기로 대충 하게요 ㅋㅋ'], ['그럼 좀 푸근하겠는데요ㅎ 저는 주말내도록 집안일을 못해서 하고나니 개운합니다ㅎㅎ'], ['환기시키고 청소후 바닥  물티슈로 폭풍닦았어요 하얀물티슈가 노랗게 변해가네요'], ['환기를 오래 시킨건가요? 저는 청소기 돌릴때만 열어놨는데 미세먼지랑 꽃가루 때문에 오래 열어두지도 못하겠어요'], ['아이들 학교랑 어린이집 데려다주고 온다고 그 사이에 환기시켰지요'], ['그 방법도 괜찮겠네요 환기도 꽃가루가 날리니 오래는 못 열어두겠더라구요'], ['들어온만큼 냉큼 딱아내고 거실 쇼파에 앉아 커피한잔 딱~~하는거지요'], ['청소후 커피한잔 너무 좋지요 오늘도 날 밝으면 후딱 청소해놓고 쉬어야지요'], ['저도 해야되는데 환기를.. 후.. 이제 꽃가루는 갠찬아졋나 모르겟네요 미세보다 꽃가루가 더 무섭네요 ㅎ'], ['맞죠 저도 청소기 돌릴때만 문 열어두고 꽃가루 때문에 맘놓고 창문을 못 열어두겠더라구요'], ['세은이 문센데이인데 일찍기상해서\n한숨재우고 가려고 재웠어요.\n문센갔다와서 집안일 하려구요'], ['세은이 오늘 생각보다 일찍 일어났지만 그래도 다시 푹 자는거지요ㅎ 문센 조심히 다녀오세요~'], ['오전에 이른낮잠을\n안자고 문센가면 잠이 와서\n보챌것 같아서 일부러 재웠네요'], ['맞아요 안자면 보채더라구요 문센하는데 보채면 난감하지요 잘하셨어요'], ['저도 청소기 한번 돌리고 닦을 타임에 성민이 깨서 안고있네요 ㅠ'], ['아 다하고 성민이 깼으면 딱 좋았을텐데 신랑님한테 좀 닦아 달라고하세요 베란다 청소도 안해주시면서요ㅜㅜ'], ['신랑 방에서 나오지도 않고있어요ㅡㅡ 결국 제가 닦았지요~ 오늘은 진짜 드라이브 하거나 친정가고싶네요ㅠ'], ['아 신랑님 좀 도와주시지 제가 다 안타깝네요ㅜㅜ 맘님 너무 스트레스 받지마세요'], ['시작이죠ㅜㅜ 빨래개기부터 시작하고있습니당~ 청소기돌리고 설거지도 하공 애기도 재우공 해야죠'], ['그렇군요 저는 청소기부터 돌리고 환기시키고 빨래는 오후늦게 개던지 해야겠어요'], ['창문열고 청소기 돌렷네요~ 다른것도 남아잇지만 궁뎅이가 무거워요'], ['저도 청소기 돌릴때만 잠시 창문 열어뒀어요ㅎ 오늘 월요일이라 더 그런가봐요ㅎ'], ['ㅋㅋㅋ월요일이라 더 그렇다고 저도생각하고 빨래를 넣어야 겟어요..'], ['저도 어제 집안일 겨우했네요 이따가는 빨래도 개켜야하는데 만사 귀찮네요']]
    
    3542
    집안일 어디까지 하나요? 21주차 접어든지 3일째인데요6갤쯤 되신분들 집안일 어느정도까지 하나요?저는 병원서는 하지말라는데 집에와서 집을 보면 지저분한데 매번 남편보고 치우라 말하는것도 그렇고 그렇다고 알아서 하지는 않고 ㅠㅠ한다면 방청소. 주방청소. 화장실청도 등등 다양한데 어디까지 하나요 어제는 변기가 지저분한거 같아 남편시켰더니 대충하기에 구석구석 해야한다니까 아까한거 못봤냐며 눈이 안보이냐고;;;  짜증을 내더라구요... 아휴 시키고싶지도 않은데 세제 냄새며 자세며 힘들어 시키는데 이러니 더더 답답하기만해요다른집들도 그런가요?             
    
    [['네 남자가 하는데 오죽한가요\n눈을 한쪽 감으세요 그래야 님 몸건강 정신건강에 조아요^^\n저는 양쪽눈다 감고 살아요 ㅋㅋ'], ['아... 많이 감았다 생각하고 참는데도 더 모른척 해야 되려나봐요ㅜㅜ'], ['임신중 입덧이 심하거나 조산기 있어서 병원에서 누워있으라고 하는경우 아닌이상 \n일상생활 그대로 하는게 제일 좋대요.\n단, 화장실 청소는 락스도 써야하고 하니 직접 못하죠 ㅠㅠ\n\n오히려 너무 안움직이고 애낳으니 근력 다 빠진상태에서 애키우니깐 각종 골병이 오는거에요.ㅠㅠ\n'], ['조기 수축와서 집에서 나가지도 못하고 있는데요\n일상생활은 최대한 하려고 하고 운동도 무리되지 않게 하려고 노력은 하는데 집안일과 운동은 다르다보니 ㅠㅇㅠ\n\n그냥 제가 하는게 나으려나요ㅠㅠ'], ['그냥..남편시키시고 눈에 보이는거정도만..남편들 믿어봤자에요ㅜㅜ'], ['아이고 ㅜㅜ 지금도 이러니 애들 태어나면 어쩔지 걱정이에요;;;'], ['집안일은. 원래처럼 하는데.. 대신 양이 적은것만 해요. 예를들어 설거지도 그릇 몇개안되면 내가하고 냄비랑 각종 그릇들 쌓이면 신랑이. 청소기도 살살 돌리는건 내가하고 대청소 수준으면 신랑이. ㅎ'], ['대청소할땐 그냥 알아서 잘해주세요?\n저도 살살하는건 해보는데 손대다보면 일이 점점 커지고 해놓음 깨끗하다고 손을 안대요ㅠㅠ'], ['꼭 찝어서 어디를 청소 하라고 얘기해줘요~ㅋㅋ 예를들어 여보~~ 쇼파밑이 넘 더러워~~ 우리아가 먼지 다마시겠다ㅠㅠ 이러면서~ ㅋㅋ\n근데 한번말해선 안되요. 몇번 말하면 신랑집에두고 외출 다녀오면 싹 해놓더라구요ㅎㅎ \n그게 맘에 안들어도 궁디 팡판 해주면서 수고했다고 하고 담엔 좀더 깔끔하게 퐈이팅 부탁한다고 해요 ㅋㅋㅋ 내맘처럼은 안되요ㅠㅠㅎ'], ['수축땜에 집콕이라 외출은 안되고 시킨건 언제하나 싶은게...ㅠㅠ\n다들 그러는가봐요... 아휴 걱정에 걱정이 쌓이네요;;\n애기들도 아들들인데...'], ['짐8개월인데 제가 다해요..장실포함'], ['괜찮으세요?\n배 아프거나 하진 않으신지...\n그냥 그러려니 하고 하게 되나요? ㅠ'], ['딱히 아프거나 그런건 없고요..신랑이 늦게오는 날도 많아서 제가 거의 하는편이죠'], ['임신이 유세는 아니지만 그래도 힘드실텐데 무리하는건 아닌가 싶기도하네요;;;'], ['저는 그때그때 바로해야하고 남편은 좀 쉬었다하는 스타일이라서 성격급한 제가 설거지.청소기.식탁치우기, 빨래널고 개기 등 했었는데요, 요즘은 몸 힘드니 내려놓게되더라구요ㅋ눈에보이는 설거지 세탁기 돌리기 정도 해두면 나머지는 남편이해요.저는24주예요'], ['집안일 보이면 시키지 않아도 알아서 하세요?\n즤집 남잔 보고도 말 안하면 계속 냅둬요ㅠㅠ\n옷을 다 개고 정리해서 넣는데까지 일주일 걸렸어요...\n이런걸로 스트레스받음 안되는데 보이니 자꾸 신경 쓰여요ㅠㅠ'], ['남자는 정확하게 지령(?ㅋㅋ)을 내려줘야한대요~ 저희 남편은 꽤나 능동적으로 하는편이지만 여자들이하는거랑은 또 다르잖아요? 그래서 처음에는 좀 명확하게 요구할필요가 있어요ㅋㅋ세탁기 끝나면 널어줘~/ 빨래 개서 자리에 넣어줘~~까지 말해야하고 꼭 칭찬해줘야해요! 고맙다 수고했어 등등^^; ㅋㅋ'], ['즤집 남자는 그리 얘기하면 기분 나쁘데요;;;\n그래서 두리뭉실하게 바닥청소좀 해줄수 있냐 이렇게는 하는데 ... 무리인가봐요ㅠㅠ'], ['전 그냥 다 했어요\n지금 9개월인데 청소, 빨래 널고 정리하고 식사준비 등 아직은 다 해요~\n배가 큰편이 아니고 아직은 움직일만 하더라구요\n\n집에 코카한마리 키우는데 8개월까진 산책도 직접시키고 다녀와서 목욕도 직접 시켰고\n반찬도 정말 오래 걸리는거 아니면 웬만한건 했구요\n\n대신 화장실청소때 락스냄새, 바닥청소때 쪼그려 앉는등 \n이런건 피했고\n음식만들때도 너무 오래 서있거나 그런건 피했어요~\n집안일 하고나면 충분히 휴식 가져주고요~\n\n물론 신랑 집에있을땐 대부분 신랑이 하구요 ㅎㅎ\n'], ['9갤이면 넘 힘들지 않으세요?\n배도 부쩍 나오고 하니 서있는것도 힘들때가 있더라구요\n그래도 이러니 저러니해도 직접해야 되는가봐요ㅠㅠ\n집에 남편있을때 시키면 싫어해서 몇번 말하고는 그냥 냅두는데 냅두면 안하더라구요  \n에휴... 제가 너무 큰 기대를 하는가봐요ㅠㅠ'], ['산모마다 상태가 다르니 ㅎㅎ\n제가 집안일 하고 그런건 몸에 부담 안되니깐 \n할만해서 한거예요~\n힘들면 당연히 쉬셔야죠!\n그래도 너무 안움직이는 것 보단 기본적인건 하는게\n아기 주수에 맞게 자리잡는데도 좋아요~'], ['그쵸... 너무 기대하고 있어 그런것도 있나봐요;;\n셤관으로 몇년 고생하다 힘들게 얻은 애들이라 조심스러운데 남편은 그냥 그런거 같은... \n제가 좀더 기대는 놓고 슬슬 해봐야겠어요'], ['아아 ㅠ 그럼 더 조심 하셔야죠~\n신랑분 한테도 힘들게 생긴 아기라는걸\n항상 상기시켜 주세요~\n애기 낳은 뒤 부터 육아 시작이 아니라\n뱃속 태아부터 육아는 시작된거니깐요~'], ['감사합니다 ㅠㅠ\n임신만하면 변할거라 잘할거라는 막연함만 있었네요ㅜㅜ\n거슬려도 앞으로를 위해 계속 얘기해야겠어요'], ['저 출산일주일 남겨놓고도 제가 다해요~~~'], ['헉... 그리 무리해도 되나요;;'], ['무리해서까진 안하고 살살하는거죠~~^^'], ['어찌해야 무리안가게 할지 고민이네요;;;'], ['미소어플깔아서 청소 신청해보세요! 엄청싸고 신세계 열립니다..ㅋ'], ['아... 도우미요? \n즤집 남자가 돈에 민감해요;;;\n통장관리도 다 하는지라 쓸수가 없어요ㅜㅜ\n본인 생각에는 온집안일 다 하는줄 알아요\n'], ['저는 설겆이랑 세탁기 돌리고 눈에 보이는 간단한것만해요~ 신랑분 시킬때 마음에 안들어도 잔소리하지마세요~ 시켜도 안하는 신랑 많아요~'], ['그쵸... 시킬때 암말 안해야하는데... 안봐야되는데 보면 말이 나오더라구요ㅠㅠ\n몇번 말해야 한번 하는데... 에휴 답답한 사람이 알아서 해야하는 가봐여;;;;'], ['저는 제가 무던한 편이라... 그냥 남편하는대로 하게 둬요~\n청소든 뭐든 님기준에맞게 시킬순없어요~ 돈을주고 사람을 써도요ㅋㅋ\n\n님기준에 70프로만 되도 만족하세요~\n직접해야 100프로죠ㅋㅋ\n\n글구 잔소리하면 앞으로 더더안해요~\n그냥 좋게좋게 한두가지정도만 엇?저기 아직 묻어있다. 저기좀 더해줌안되? 요러케 시켜야 다음에도 기억하고 더하죠~~~'], ['기분 상하게 얘기한게 아닌데도 그러더라구요;;\n방청소 일주일에 많이 하면 두번 적게하면 한번 시켜요\n장실청소는 2주에 한번쯤 하는듯 하구요...\n\n아무리 이래저래 시켜봐도 하기 싫은거하니 늘 리셋이라 속만 터지고... 너무 깨끗한거 바라는건 아닌데 바닥이 먼지랑 머리카락에 ㅠ-ㅠ\n수축만 아님 제가하는데 그저 답답하기만 해요'], ['저 그래서...  로봇청소기 샀어요!!!\n매일 예약시간에 한번 도는데 일주일만 돌려도 먼지통이 가득해요!\n\n평소에 먼지도 거의 안굴러다녀요~~~\n로청이랑 건조기 산건 신의한수ㅋㅋ\n\n그러케 최대한 자동화하구요~~~\n청소할땐 최대한 남편보는 앞에서 해용. 빨래 개는것두ㅋㅋㅋ\n그냥 조용히하면서 허리두들기고 끙끙앓고있으면 슬며시 옆에서 같이하더라고요ㅋㅋ'], ['로봇 청소기 +_+ 진심 사고싶습니다...ㅠㅠ\n애기 용품살거 이것저것 외벌이로 충당하려니 제옷 하나 사는것도 쉽지 않아 로봇청소기는 그냥 꿈 같아요\n진짜 그거 있음 이런고민 좀 덜어지겠어요'], ['저 치후360이라고 저렴한?중국꺼 샀는데 1년째 잘쓰고있어요~\n지금 37마넌하네용~ 부담되시려나요?ㅜ\n아님 단후이라고 좀더저렴한 가성비좋은 국산도있어요! 아마 20만대일거예요~\n한번 보시고, 가능하시면 구매하세요ㅜ\n바닥 먼지 머리카락 스트레스만 없어져두ㅜ 크더라고용'], ['저라도 같이 벌면 덜할텐데 이럴땐 참;;;\n로봇 청소기 한번 의논해봐야겠어요\n바로 하자 할거같지 않지만 청소가 진짜... ㅠㅠ'], ['전 10갤이고 예정일 다 되었는데 끝까지 제가 했어요~  신랑이 회사갔다와서 좀 쉬는데 집에 있을때 집안일 시키기 좀 그렇고 저도 움직이는게 좋고해서 다했네요'], ['평일엔 힘들까봐 안시켜요;;\n주말에 쉴때 시키는데 그것도 귀찮아해서 안되겠다 싶을때 한번씩 얘기하는데 ... 에휴 수축만 없음 해버릴텐데 답답하네요ㅠㅠ'], ['저도 초기때 빼곤 4개월부턴 직장다님서 쉬는날은 집안일은 많이 하고 있어용 설거지는 신랑이 많이 해주는데(밥통에 밥풀 그대로 굳어있고 ㅡ,.ㅡ;;;)   밥하면서 중간중간 많이 쌓이면 미리 제가 해버리구요 ㅋㅋ우엉잎 한박스 괜히 주문해서 5시간 넘게 씻고 찌고 정리하고 등등 혼자 사서고생은 다해요 ㅡ,.ㅡ'], ['일하시면서 집안일;;;\n괜찮으셨나요?\n전 몇일 일한다고 계단 오르내리기 좀 했더니 조기 수축오고 해서 안정 취하라던데...아이고 집안일 안되는건 저만 그러는게 아닌가봐요ㅠㅠ'], ['하지말라는거 안하는게 좋은건 알지만 ㅠㅠ\n방바닥 닦아보면 물티슈 색이....;;  밖에도 못나가지만 미세먼지 안좋다그래서 청소는 못해도 2일에 한번은 해주면 좋은데 일주일에 한두번이라 답답해요;;;\n수축없었을땐 제가 매일했는데 지금은 하지도 못하는데 시키려니 그렇고 시켜도 시큰둥해서 시키기도 싫어져요'], ['둥이라 무게 조절은 계속 하고있어요\n혹시몰라 임당이나 임중 올까봐서요;;;\n저도 수축 오기전까진 시키지않고 제가 다 했는데 수축 오고나서부턴 할래도 할수가 없어 시키는데 시키니 잘 안하고 매번 그러려니 내키지도않고 답답해서요ㅠㅠ'], ['넵 최대한 조심은 하려는데 어느정도까지는 그냥 해야하나 하고 있어요ㅠ'], ['맞벌이시면 더 힘드시겠어요ㅠㅠ\n남편도 말은 힘든데 하지말라는데 안하면 아무도 안하니..청소기하고 밀대까지하면 좀 낫겠어요\n즤집은 둘중 하나 겨우 하는데... 외벌이로 저는 집에 있으니 쉬엄쉬엄이라도 해야하는가봐요ㅠ\n'], ['제가 남편한테 기대하는게 너무 많은가봐요;;;\n애기들도 하나도 아니고 둘이 한꺼번에 나올거라 이래저래 저혼자 분주하고;;;\n부탁도 매일 하면 짜증내고 피곤하다고 그래서 주말에만 해도 피곤하다고...ㅠㅠ\n집에 있음 자거나 게임하거나 등등 하는데도 이러니 속만 새카맣게되요'], ['수축 없을땐 주말에 출근 안할때만 신랑 시키고 나머지는 제가 다 했는데 수축 생기고는 무리안하려고 냅뒀더니 이렇게 되네요;;;\n놓는다고 많이 놓은건데도 아직 더 내려놔야 하는가봐요\nㅠㅠ'], ['다른건 제가 다 하고 화장실청소는 남편한테 부탁해요 자세가 불편해서 그건 못하겠더라구요'], ['하루 하나씩 청소하나요?\n한번 시작하면 다 하게되고 그거 하고나면 힘들고 뭉치고;;\n뭘 어찌해야할지... 안할수는 없고 하자니 힘들고 ㅠㅠ'], ['무선청소기는 매일 돌리고 물걸레질은 2,3일마다 하구요 빨래는 2,3일마다 돌리구요 음식물쓰레기는 남편이 버려주고 설거지 제가 하구요 이정도 인거같은데.. 저는 그정도로 안뭉쳐요 개인차가 있어서 그럴거 같아요 님 뭉치면 남편 시켜야 해요 님이 다 하려 하지 마시구요 ㅠㅠ 힘내세요'], ['감사해요ㅠㅠ\n워낙 임신전부터 일은 많이 했었는데 임신하고도 조심안해서 그랬는지 수축 온 뒤론 조심한다해도 뭉치더라구요;;\n천천히 해보고 그래도 안되면 다시 얘기해야겠어요'], ['저는 8갤인데 그냥 다해요ㅋㅋ 제가 다하고있으면 남편이 오히려 말리면서 자기가해요~ 그런 분위기 형성을 위해... 그렇지만 제스스로 그렇게 무리가 가지도 않더라구요ㅋㅋ'], ['저희남편은 말리긴해도 말뿐이에요ㅠㅠ\n다해놓고 나와서 하지말라니까 왜 무리하냐고;;;\n임신하고는 계속 이래요.. 부탁하거나 시키는건 듣기 싫고 제가하면 왜그랬냐 그러고 알아서는 안하는 ;;;\n시어머니나 친정엄마한텐 본인이 다 하고있다 그러고..\n그나마도 안할까싶어 그래그래 하고 넘어가는데 속은 터져요;;;;'], ['저희도 결혼초에 집안일로 엄청싸웠거든요~ 게다가 임신후에도 맞벌이인데 제가 더 많이하는것같아서 짜증나서 기준을 정했어요ㅋㅋ 남자들은 언제가 청소할때인지 잘 모르더라구요! 예를들어 화장실 세면대에 물때가 보이면 화장실청소하기 이렇게 정하세여 두분이 상의해서'], ['이게 정해도 안하더라구요;;\n어느집이나 남편들이 잘해주는 집은 드문건가봐요\n집안일로 이럴거라곤 결혼전엔 한번도 생각안해봤는데 역시 현실은 다르네요\n6년째 이러니 앞으로가 걱정이에요;'], ['무선청소기만가끔돌렸어요ㅎ\n남편도저도집안일놨어요ㅋㅋ\n뱃속에아이가첫째라\n위생에그닥신경안쓰고살았답니다ㅜㅜ'], ['무선청소기도 쓰려니 배에 힘들어가던데 괜찮으세요?\n저도 첫아이고 둥인데요 \n위생은...  놓을래야 놓을수가 없어요ㅠㅠ'], ['저는임신전에진짜 한깔끔했는데\n중기에 조산기로 눕눕하다보니\n위생포기하게되더라구요ㅠㅠㅋ\n무선청소기 저는 빼고 다시 꽂을때만 힘들었어요\n아 그리고 걸레질은 아주아주 가끔했지만 무선걸레청소기에\n일회용청소포 끼워서 대충 돌리는 정도로만 했어요ㅜ'], ['포기하면 빠르고 편하긴한데 이게 그리 안되네요;;;\n일회용 물걸레청소라도 해둬야겠어요ㅠㅠ\n임신도 셤관이라 어려웠는데 임신 유지도 쉽지않네요 흑'], ['여기 계신분들 다 부처시네요...ㅋㅋ\n제가 성격이 모난건지 예민한건지\n전 결혼 하자마자부터 남편이 집안일 알아서 안하는것땜에 스트레스받아하고 많이 싸웠었는데\n개버릇 남못준다고  임신하고도 딱히 알아서 먼저 하는게없어서 너무짜증나요ㅋㅋㅋㅋ하... 생각만으로 한숨이....\n아니 집안일을 왜 여자가 알아서 다해야하죠? ㅠㅜ 남자들은 손이없나요 발이없나요.. 여태 맞벌이하면서 지냈는데.. 돈도 똑같이벌고 ㅋㅋ\n그럼 최소한 임신햇을때만이라도  신경 안쓰이게 해주면 안되는지\n왜 그게 어려운건지 참 이해가 안가네요\n보통 집안일 똑같이 분담하다가도\n남편이 아프거나 늦게들어오거나 하면 여자들은 알아서 더 하잖아요 ㅋㅋ 진짜웃김 생각할수록\n저는 지금 그런걸러 하도 싸우고 잔소리해대서 많이 내려놓긴 했지만\n그래도 일일히 다 시키면서 살아여 ㅋㅋㅋㅋㅋ 시키는것도 짜증나긴한데 몸힘든거보단 나으니까..'], ['저도 첨엔 님처럼 생각하고 파이터가 됐었어요;;\n남자는 뭐라고 일끝나고오면 해주는밥먹고 또 쉬나하구요\n근데 변하지않고 계속 싸우니 정은 정대로 떨어지는거 같고 그래서 그때부턴 말을 아끼고 그러다보니 싸움도 줄었답니다... 친구들도 저보고 부처라고 해요;;;  \n결혼 6년차에 4년동안 애기 안생겨서 병원다니다 셤관하고 이제야 성공해서 둥이 품고 있는데 달라지려나 하는 기대를 품으면 안되는데 그걸또 기대하고 있더라구요;;;;\n그래서 더 힘든건가싶기도해요\n나름 시키거나 부탁은하는데 귀찮아하거나 모른척하면 두번은 안시키고싶더라구요ㅠㅠ'], ['즈이 남편은 다행히 시키면 해줘요\n해준다는 말도 웃기지만 ㅡㅡㅋ\n그래도 말로 시키고 몸 안움직이는게 나으니까 시키고있네요 ㅠㅠ\n근데 시켯을때 장난으로라도 싫어하는티내고 한숨쉬면 정떨어져요 ㅋㅋㅋㅋㅋㅋㅋ휴...\n전 성격이 더러워서 그런꼴 절대못보겟어여;;  성격성향이 변하지않는다지만 세뇌당할때까지 시켜봅니다 ㅋㅋㅋㅋㅋ'], ['뒤에서 꽁하는거 보단 나아요\n다소 과격? 하게 보일지는 모르지만 뒤끝도없고 할텐데요\n나쁘지 않아요\n그저 얼마나 더 해야 세뇌가 되려는지^^;;\n여자들은 세뇌 안시켜도 알아서 하는데 남자랑 다르긴 참 많이 다르네요.. 이래서 아들램들 어찌 키울지 ㄷㄷㄷㄷ'], ['고위험 아니면 그때는 안정기인데 다 하죠. 무거운 거 드는 일 아니면'], ['안타깝게도 고위험 산모에요;;\n안정기라면 맘놓고 할텐데 병원 갈때마다 겁만 잔뜩 먹고 오게 되니 더 걱정과 답답함에 글 올려봤어요ㅠㅠ'], ['걍 눈감고 대강하시고 몸좋아지면 좀 하세요. 저 아는친구 걸레질하다가 애기나왔는데 조산이라 애기 수술하고 병원에 오래있고. 친구는 몸조리도 못하고 맘고생 몸고생 하고 산후풍도 왔어요. 애기가 중환자실에 있으니 몸조리 했겠어요...\n근데 또 다른 친구는 조산기 있어서 입원해서 주사맞고 버텼는데 애기가 40주 넘어서도 안나와서 유도했어요. \n병원에서 조심하라고 하는 이유가 있을테니 지금은 대충 하시고 몸 좋아지면 다시 살살 하세요. \n그리고 남편분ㅋㅋㅋ 혼내주세요. 눈이 안보이냐니... 죽겠다고 엄살좀 부리세요'], ['즤집 5층 계단이라 아예 밖엘못나가니 남편도 안나가고 하느라 일하랴 힘들겠다싶어 되도록 말 못되게 해도 참는데 한번씩 저런말에 상처가 되긴해요ㅠㅠ\n진짜 대충 있다 괜찮아지면 제가 해야할까봐요 에휴\n집안일 알아서 다 해놓음 이젠 또 괜찮은줄알고 시댁에 잘못 얘기할까 걱정이기도 하지만 그게 맘 편하겠죠?!\n(임신 초기때까진 시누가 필요하다하면 큰아들램 봐주고 가게 심부름 하고 했었어요)'], ['점점 배가 불러오니 숨도 차고 집안일이 힘들어요 빨래 널때는 빨래통에서 꺼내는거는\n남편에게 시키시고 요새 화장실 청소하는거 일회용으로 나오는게 있더라고요 그걸로 청소하고 물에 흘어보내면 되니까 그걸로 하고 있어요~ 집청소는 나누어서 하고 있고요 절대로 무리하시면 안되용 저 무리했다가 하혈기가있어서 병원댕겨왔어요 ㅜㅜ'], ['하혈...ㅠㅠ 무섭네요 \n둥이 이벤트는 계속 얘길해도 우리 애들은 안그럴거다 이러고만 있어서 ... 에휴\n일회용 청소라고 검색해봐야겠어요..\n화장실은 진짜 안될거 같아 냅뒀는데 집안에 하수구냄새가 나기 시작하는듯해요 @_@;;;;'], ['님편분 가상 출산경험(?)시켜드려야겠어요\n너무 모르신다 쌍둥이는 더더 조심해야하는데..\n전문서적을 사서 읽어보면 남편이 많이 도와줘야한다는 내용이 있는데 ..'], ['인터넷.. 하다못해 어플이라도 하루 한번 들어가서 보면 좋겠어요;;;\n본인은 임신전보다 많이 하고 있어서 이정도 해주는 남편없다고 매일 얘기해요ㅠㅠ\n싸우느니 그래그래하고 있답니다 흑'], ['저두 지금 21주인데 초반 입덧할땐 아예 손안대고 신랑이 해줬는데 요새는 화장실이며 집안 물걸레질이며 오늘은 냉장고청소 같이 하고 누워있네요 ㅋㅋㅋ 그래도 신랑이 많~~~~이 도와줍니당 ㅋㅋ'], ['같이 한다는 말이 참 좋네요ㅠㅠ\n너무 무리하지마시고 쉬엄쉬엄하세요\n단태아 21주면 안정기긴해도 혹시모르잖아요^^;;'], ['38주인데 별로 힘든거 못느껴서 평소랑 똑같이 다해요ㅋㅋㅋ 남편 쉬는날은 남편이 알아서 싹 다하구요ㅋㅋ 남편이 꼼꼼한 편은 아니라 뒷처리는 조금씩 해줘야하는데 쉬는날빼고 하루11시간 12시간 일하고 집오면 11시인 사람이라 쉬는날 청소 빨래 화장실청소 이불털고 정리 설거지 다 해주는거 만으로도 고맙게 생각하고잇어요ㅋㅋㅋ'], ['하루 8시간 딱 일하고 오고 평일엔 음식물 없으면 쓰레기도 안버리고 그나마 주말에 쉴때 (격주 토욜근무에요) 집안일 부탁하는데 이것도 쉽지 않네요ㅜㅜ\n38주시면 이제 막달이라 배도 많이 부르실텐데 대단하세요\n저는 벌써 배가 많이 나와서 누웠다 일어나면 배가 찌릿찌릿 할정도로 반동없이는 혼자 못일어나는데;;;\n이리 만삭 가까인 분들도 다 하는데 제가 첫아이들이라 너무 엄살인건가 싶기도하고 그러네요ㅠㅠ']]
    
    3547
    🐷 집안일 끝낸후 커피한잔~ 집안일 끝내놓고 씻고밥먹기전에 커피 한잔부터 해용~청소하는데 어찌나 덥던지요ㅜㅜ쉬면서 아점 뭐먹을지 생각해봐야겠네요아침 드셨나요?
    
    [['오늘도 날이 덥네요 집안일 대충~하고 누워잇어요 ㅋㅋㅋ 커튼사이로 햇빛들어오는게 커튼을 못걷네여'], ['맞죠 저도 청소하다가 땀이 삐질나서 씻고 여유롭게 커피마시니 너무 좋더라구요ㅎ'], ['집안일 끝내고 시원하니 커피한잔하시나 봅니다 저도 시원한 아메 한잔했지용ㅋ'], ['네네 집안일 끝내고 커피한잔 하는 시간이 제일 좋은것 같아요 맘님도 아메 드셨군요'], ['맞아요 할일 다 개운하게 끝내놓고 커피한잔 너무 좋지요 시원하니 맛나게 드셨지용'], ['그러니요 중간에 쉬는건 더 그렇더라구요 그럼 다시 청소를 해야하니 귀찮구요ㅜㅜ'], ['맞아요 할때 얼른 확실히 끝내놓고 쉬는게 좋은거 같아요 청소후 커피한잔 시원하니 좋지요'], ['그러니깐요 내일도 집안일 해야하는데 아이있을때는 집안일하기도 힘들어요ㅜㅜ'], ['저도 갑자기 커피가 땡기네요. 커피는 언제나 사랑입니다.'], ['맞죠 저도 커피를 너무 좋아해서 집안일 해놓고 시원하게 먹으니 좋더라구요'], ['집안일하시고 개운하게 한잔드셔요~~\n 전 아침볶음밥 먹엇는데 적어서 지금 미리 먹으려구용'], ['집안일하고 커피 마시는 시간이 너무 좋은것 같아요 점심은 든든하게 드셨으려나요?'], ['그춍  할거 딱 하고 여유로이 마시면 넘 좋더라구용  점심에도 남은 볶음밥 먹엇답니당'], ['맞아요ㅎㅎ 주말에는 그런여유도 못느끼니 평일에라도 느끼도록 해야지요ㅋ'], ['넹 맘님 ㅋㅋ 얼집보낸지 우리 오래안되엇는데도 ㅋㅋ 월욜이 기다려지네요'], ['그러니깐요 주말내도록 같이 붙어있으니 힘들어요ㅜㅜ 그래서 늘 월요일이 기다려지네요ㅋ'], ['쪼매만 움직여도 몸에서 열이 후끈후끈이네요 새벽부터 김밥먹고 커피도 한잔 했네요'], ['그러니깐요 김밥 말으셔서 오늘하루 끼니 걱정은 없으시겠는데요ㅎㅎ'], ['아침도 점심도 김밥먹었는데 저녁은 또 다른거 해줘야할것 같아요'], ['두끼는 김밥 드셨으니 저녁에는 다른거 해주시면 아이들 또 잘 먹겠어요 뭐 드셨을까요?'], ['만둣국 끓여줬어요 금요일은 유독 바쁜날이라 만사가 귀찮더라구요'], ['그러셨군요 뭐라도 해주시면 되지요 저는 어제 미역국 끓여놔서 한동안 미역국으로 밥 먹으려구요ㅋ'], ['저도방금 집안일 끝냈는데 오늘 어찌나더운지 여름이 성큼 왔음을 느끼네요'], ['맞죠 땀이 땀이 장난이 아니더라구요 날이 갑자기 또 더워지니 적응 안되구요'], ['지금도 이렇게 더운데.. 벌써 여름이 무섭습니다ㅠㅠ'], ['그러게요ㅜㅜ 주말되면 또 더울거고 벌써부터 더워서는 한여름에 참 걱정이네요'], ['이제 일어났네요..집안일 벌써끝낸 맘님부러워요ㅠㅠ전 늦게일어나니, 하루가 너무빠듯하네요'], ['늦게 일어나셨네요 제일 부러워요 아무래도 늦게 일어나면 하루가 금방이더라구요'], ['부지런하시네요..아침부터집안일하시구전가계부쓰고커피한잔마시구누워있네요..'], ['아이 얼집 가고나서 해야지 안그럼 더 힘들더라구요 가계부 저도 써야하는데 달력에다가 막 적어두고있네요ㅎ'], ['아침일 후다닥하고 먹는 아이스커피가 정말 최고죠ㅎ_ㅎ\n'], ['맞죠 집안일 다 해놓고 커피마시는 시간이 왜이렇게 좋은지 너무 좋아요ㅎㅎ'], ['맘님도 얼음동동 시원하게 마시네요..요거 드시고 식사도 어여 챙겨드세요~'], ['네네ㅎ 첫끼는 라면으로 먹었네요 밥맛이 왜케 없는지요 라면이라도 든든하네용 점심 맛나게 드셨나요?'], ['첫끼로 모닝라면도 좋지요 ㅎㅎ 집에와서 아이는 누룽지끓여서 한그릇 먹이고 저는 아직도 못먹었네요..같이 낮잠을 ㅎㅎㅎ'], ['효은이 누룽지 잘 먹는가보네요 푹 주무셨겠지요? 병원에서 너무 고생하셨어요'], ['네네 누룽지 완전 좋아해요.. 저희 국그릇으로 한그릇 가볍게 비웁니다요 ㅎㅎㅎ'], ['누룽지는 거의 죽 수준이라 그정도는 먹어야 먹은것같지요 잘먹으면 이쁘지요ㅎ'], ['부지런하세요 ㅎㅎ전 이제 일어나서 밥먹고있는데🤣😅'], ['강제기상해서 이렇게 부지런하지 저도 늦게까지 자면 안 부지런해요 그래도 청소해놓으니 푸근하네요'], ['전 아침에 열무랑 된찌 비비빅해서 머것답니다~ 집안일 하시구 시원하이 한잔드셔요 오늘 징짜 덥지요 ㅠ.ㅜ'], ['맘님 드시는거 봤지요ㅎ 집안일하고 느긋하게 커피 마시는 시간이 정말 좋은것같아요ㅎㅎ'], ['집안일..오늘 안하고있어요ㅎ청소기는 밀었으나 걸레질 안했답니다.냉장고속 재료들도 얼른 얼른 정리해야되는데 말이지요'], ['그렇군요 저는 청소기만 돌리기 그래서 걸레질도 같이 했네요 집안일 해놔야지 그나마 편하더라구요'], ['편하기는 미리 해야 편한거 맞지요~~아니면 나중에  하원시간되어갈때 급 바빠지니요'], ['맞아요 하원하고오면 정신없긴해요 아이랑 놀아줘야하고 저녁준비 해야하구요 조용할때 집안일 해놓는게 맞지요'], ['얼음 동동 띄워서 시원해보이네요~ 저는 오늘 왜이리 몸이 천근만근인지ㅠㅠ 아침에 설거지만 겨우 해놓고 널부러져있네요'], ['맞죠 천근만근일때는 그냥 쉬는게 맞아요 저도 그런날 있는데 집안일 다 제쳐두고 쉽니다ㅎ'], ['오늘은 청소하면서 땀을 흘릴 날씨인 거 같아요..\n청소하느라 고생하셨어요.'], ['맞죠 청소하는데 덥기는 했는데 다하고 나니 개운하더라구요 아이스커피 마시는시간도 너무 좋구요'], ['오늘은 청소도 살살 해야 할 거 같아요..\n날씨가 너무 더워서 조금만 움직여도 땀 나겠더라구요.'], ['그래야지요 주말에는 더 더워서 쪼매만 움직여도 땀이 한가득일것같아요'], ['맞아요.. 저는 진짜 여름엔 큰일입니당..\n더위를 엄청 타거든요..ㅠ'], ['아 진짜요?저두요ㅜㅜ 더위도 추위도 많이타서 여름이 다가오는게 두렵네요'], ['와 청소하고 진짜 아아는 꿀맛이죠잉^^ 고생하셨어영~좀 쉬시다가 점심드세영^^'], ['맞죵 아아 한번씩 태워먹기 너무 좋더라구요 얼음 가득넣어서 먹으니 더 맛있었구요'], ['요새 아아 자주 드시네용?ㅋ 완전 제 스타일이라서 침 꿀꺽입니다 저도 한잔 마시는중이긴합니다만ㅋㅋ'], ['하루 한잔은 꼭 먹는것같아요 블랙 커피도 맛있더라구요ㅎ 역시 맘님 스타일일줄 알았어요ㅎㅎ'], ['ㅋㅋ 아침에 얼려둔거 한잔하고 지금은 컵커피 한잔하는 중입니다 당이 필요하더라구요'], ['그렇죠? 하루에 한번은 당충전을 해줘야해요 안그럼 힘들거든요 저는 뒤늦게 라떼한잔 또 마셨네요ㅎㅎ'], ['라떼한잔 더 드셨군요ㅋ 오후에도 역시 카페인이 필요합니다~ 밤엔 션한 아이스커피가 땡기네요'], ['맞아요 자기전에 커피를 마셨는데도 피곤하더라구요 결혼전에는 커피도 늦게 못 마셨거든요'], ['이제는 조금만움직여도덥지요 ㅎ청소해놓고 마시는커피한잔이 최고더라구요'], ['맞아요 집안일 해놓고 씻고 시원하게 아아 한잔하니 아주그냥 푸근하더라구요ㅎㅎ'], ['그죠 ㅎ오늘도 역시 집안일하면서 커피마시고하니 왜그렇게좋을수가없나몰라요'], ['저도 그래요 집안일 실컷하고 씻고나와서 시원한 커피 한잔하는 시간이 왜이리 좋은지요'], ['그죠 애보내고 이런시간 너무좋아요 내일은 저도 청소하고 커피한잔해야겠어요'], ['집안일 하고 느긋하게 커피 마시는 평일이 너무 좋은것 같아요 주말에는 꿈도 못 꾸네요'], ['아아는 늘 땡기지요 오늘 같은날 시원하게 태워서 마시면 정말 좋지요'], ['저도 꼬멩이 낮잠재우고 매트걷어서 거실 청소하고 저녁준비미리해놓고 컵커피한잔했네요 요시간이 젤 좋네요ㅎ'], ['오호 저녁준비까지 미리 해놓으셔서 푸근하시겠어요 커피 마시는 여유로움이 정말 좋지요ㅎㅎ'], ['아..저는\n아직 집안일 못끝내고 쇼파에서 이러고 있네요..부지런하십니다~'], ['그렇군요 저도 딱히 부지런하진 않는데 아이가 일어나니 청소 할수밖에 없네요ㅎ'], ['집안일 끝나면 아아로 시원하게 한잔해야지요\n 전 점심 뭐 먹을지 고민하네요ㅋ'], ['네네 시원하게 한잔하니 더 좋더라구요 저는 첫끼로 육개장 컵라면으로 때웠어요ㅎㅎ'], ['육개장 컵라면 드셨군요\n전 있는반찬 꺼내서 먹었지요\n부추김치가 젤 맛있더라구요ㅋ'], ['네네 부추김치도 맛나지요 라면이랑 먹어도 맛있지요 부추김치에 흰쌀밥 먹고싶네요ㅋ'], ['흰쌀밥에 먹으면 진짜 맛있지요^^\n라면에도 맛있구요\n친정표가 짱이지요^^'], ['네네 맞아요 친정표는 사랑이지요ㅎ 저는 반찬류는 사다먹어서 친정표는 못 느껴보네요ㅜㅜ'], ['아침부터 부지런히 움직이셨네요. 저도 오늘은 점심 설거지만 하면 끝이네요^^'], ['요래 또 움직여야지 오후가 편하지요 집안일 해놓고 쉬는게 제일 푸근합니다ㅎㅎ'], ['맞아요. 오후에 나가기전에 하려면 아무래도 더 귀찮아지더라고요. 저는 점심 먹고 바로 하고 있네요 요즘 ㅋㅋ'], ['맞죠 그래서 나가기전에 집안일 다 해놓고 나갔다오는게 푸근하더라구요 나갔다오면 정말 귀찮아요'], ['그러니까요. 근데 전 요즘은 하원하고 들어와서 대충 청소하고 그래요. 요즘 낮에 자꾸 놀아요 ㅋㅋ'], ['그렇군요 저는 그나마 등원때 청소를 해야해서 후딱하고 노는게 차라리 편하더라구요ㅋ'], ['시원하겠네요'], ['네네 아아라서 시원합니다 집안일하고 먹으니 더 개운한거 있지요'], ['집안일 끝내고나서 마시는 커피한잔은 더더 맛나고 좋지요. 션하이 쭉쭉 들이키셨을듯해요~'], ['맞죠 집안일하고 씻고나오니 어찌나 개운하던지요 커피 한잔이 꿀맛이었어요ㅎㅎ'], ['암요 그 기분 알지요 할일다하고 샤워하고나서 마시는 커피는 꿀맛이지요'], ['그렇죠 아침에 날 밝으면 또 집안을 후딱 해놓고 커피한잔 할까싶네요ㅎㅎ'], ['오늘도 집안일 언능 끝내놓고 커피한잔 드시는 걸까요?? 전 오늘 일 많아서 이제 카페 들어와보네요~'], ['네네 집안일 끝내놓고 오늘도 아아 한잔했어요 너무 시원하더라구요ㅎㅎ']]
    
    3628
    집안일 중...뭐가 제일 하기싫으세요?? 전 빨래개는거요.....시어머님께 막내 출산선물로 건조기를 받고신세계를 맛봤죠.....하지만 이젠 빨래 개는게 너무 귀찮아요^^;;건조기는 다 돌았는데...문을 열기가 싫어요...하하^^;;그다음엔 쓰레기 버리기ㅠ아...다른집안일은 다 했는데....제일 하기싫은 이 두가지만 남겨두고누워서...이러고 있네요ㅋㅋ
    
    [['밥하기랑 쓰레기버리기요ㅎㅎ'], ['저는 청소는 다 .. ㅋㅋㅋ걸레빠는게 제일 시러욧!!ㅋ'], ['앗! 동지다 ㅋㅋㅋ'], ['저두.저두여  다시름요  ㅋ'], ['전 설거지요ㅋㅋ'], ['청소랑 설겆이요~~ㅋ'], ['청소랑 빨래 너는 거요'], ['저는 청소기돌리고 걸레질하는거요 지금 등돌리고앉아 카페 구경하고 있지만 ㅜㅜ 외면할수 없는 현실이여~~~~~;;'], ['화장실청소요ㅋㅋㅋ'], ['설거지요ㅋㅋ 누가 돈받고 설거지만 안해주나요...'], ['설거지하는동안 세탁기돌렸는데 설거지 끝나가니 세탁다됐다고 소리나서 빨래널어야하는 타이밍!  그때 진짜 귀찮아요ㅋㅋㅋㅋㅋㅋ'], ['설겆이요ㅋㅋㅋ'], ['설거지...넘나싫어요ㅡㅠㅠ'], ['설겆이요...그래서 신랑이 식기세척기 사줬어욬ㅋㅋ'], ['사실 집안일은 다 하기싫죠...'], ['빨래개는거랑... 냉장고청소요ㅠ'], ['밥준비하기요'], ['밥하기 설거지요 매일 외식하고싶어요 ㅎ'], ['저도 빨래개는 거 왜케 싫죠!ㅠㅠ\n특히 애들 빨래는 자잘하게 넘 많아서...ㅋㅋㅋ\n지금도 빨래개다가 딴짓하네요^^;;'], ['가스렌지 닦는거랑...  젤 귀찮아요'], ['음식이요ㅠㅠ'], ['설거지 진짜 미치게 하기 싫어요ㅠㅠ'], ['다 싫어욧!!!!!'], ['빨래개기,설거지,....젖병씻기..'], ['다요..ㅋㅋㅋ그래도 꼽자면 청소ㅠㅠ'], ['저는 음식물쓰레기 버리는거요 ㅠㅠ'], ['저도 빨래 널고 개는거 디게 싫어해서 건조기 사용한지 3년째에요 ㅎㅎ 그래도 개는거 싫어서 개는거 신랑담당 맡기고요~ 그렇게되니 설겆이 구찮아서 요샌 가끔 빌트인 식기세척기 돌리는 중이에요.'], ['쓰레기 버리기인데.. 그건 신랑담당이라서 그나마ㅋㅋㅋㅋㅋㅋㅋ'], ['전 설거지요. 빨래개는건 좋아해요~!! ㅋㅋㅋ'], ['화장실 청소여 청소는 신랑 몫 ㅎㅎ 훨씬 잘해여'], ['물품정리 보기좋게하는정돈이랑 철마다하는 옷정리요'], ['청소요.....ㅜㅜ 진짜 귀찮아요.....'], ['저도 빨래개고  갖다놓는거요'], ['화장실청소요 ㅎㅎ 죽어도 안해용 ㅎㅎㅎ'], ['설거지요'], ['설거지.\n 빨래개는거.\n청소기돌리는거\n화장실청소\n등등..\n\n그러보니 집안일이 다 싫네요ㅋ\n'], ['저는 다림질이요ㅠ'], ['밥하기, 화장실청소랑 음식물쓰레기 버리는거요 ㅋㅋㅋ'], ['전 설거지가 젤 시러요ㅠㅠㅠㅠ'], ['걸레빨고 걸레질하는게 젤 싫어요ㅋㅋ'], ['음쓰요!!!'], ['아기 키울때 그렇게 ~~젖병 설거지가 싫더라구요~~ 글서 신랑이 해줬어요~ 지금 물병 (물병솔) 같은거 설거지가 젤루 시러요 ㅋ'], ['설거지랑 빨래요 ㅎㅎ'], ['저두 빨래 개기 시러햇는데....티비 예능 잼난거 틀어놓구 개먄 금방 끝나용^^ 함 해 보세용~~'], ['반찬이요. 요리고자라...  해도 맛도없고 시간은 엄청걸리고.....'], ['손빨래랑 화장실청소여'], ['화장실청소. 걸레질이요~~'], ['전 밥하기요'], ['빨래를 개서 옷장에 넣어 정리하는게 미치도록 싫어요ㅜ'], ['다요..'], ['설거지 넘 싫어요ㅜ'], ['설거지..\n\n\nㅠㅜㅜㅜ'], ['전 빨래게는거까지는 좋아요! 근데 겐거 서랍에 정리하는게 넘 귀찮아요 ㅋㅋㅋㅋㅠㅠ'], ['2222'], ['건조된 옷이 산이 되고 있어요 ㅠㅠ'], ['뭔들 좋겠어요ㅠㅠ 다 싫어요ㅠㅠ\n그중 화장실청소 가스렌지 닦기가 최고인거 같아요ㅋㅋ'], ['밥하기요 ㅋㅋ 빨래개는건 좋아요 ㅋㅋ 근데 넣는건 싫어요 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['222공감여 ㅋㅋㅋㅋㅋ빨래게는건 티비보면서 무의식적으러하는데 갖다놓는건 진짜 귀찮..'], ['설거지요ㅠㅜㅜㅜㅜㅜ'], ['저는 설거지랑 쓰레기버리는 거....누가 해주면 좋겠어요ㅎㅎㅎ쓰레기는 치워도 치워도 맨날 많구 설거지는 씻어도 씻어도 또 잔뜩이네요'], ['ㅋㅋㅋ다싫은데 그중에 제일은 설겆이랑  빨래개기ㅜㅜ'], ['음식물쓰레기 버리는거요ㅜ'], ['설거지.....요리하는거요ㅜㅜㅜㅜㅜㅜㅜ젠병이네요'], ['요리,청소,걸레질,설거지,빨래널기,빨래개기,빨래갠거 정리하기,화장실청소,애들 목욕시키기 집안일 죄다 싫으네요ㅜㅜㅜㅜ'], ['설거지요😭😭'], ['저도 빨래개는거요ㅠㅠㅠ 그냥 건조기 꺼내서 위에 다 쌓아놓네요'], ['ㅇㅏ 저는 설거지가 너무 싫어요 근데 또 결벽증같은게 있어서 신랑 시키지도 못하고(나보다 깔끔하게 못하는거 같아서 ㅠ)\n스트레스 받으면서 혼자 거의 다 하네요 ㅎ'], ['유선청소기라 청소기 돌리는거..ㅋㅋ 무선 사고싶네요ㅠㅠㅠ'], ['전 설거지랑 화장실청소요..'], ['청소랑 빨래는괜찮아요\n설거지가 극혐이에요ㅡㅡ'], ['먼지닦기  창문 창틀'], ['음식물버리기? 술먹고 담날 버리려면 ....  으악...'], ['저두요 빨래개는거랑 음쓰 일반쓰 버리는거요.'], ['설거지랑 빨래개기요ㅋㅋㅋ건조기만으로도 행복했던게 엊그제같은데 사람이 참 간사하죠 이젠 빨래개는 기계가 나오면좋겠어요..ㅋㅋ저희같은사람들을 위해서 삼성 엘지가 열일하고있을거에요'], ['화장실청소,걸레빨기,신발빨기요 ㅠㅠ'], ['밥차리기와 설겆이요ㅜ넘넘싫고귀찮아요']]
    
    3663
    투잡해요 ^^ ㅅㅂㄴ이 맞벌이임에도 불구하고 항상 소파한몸일체를 실천해주신것도 모자라며칠 전에 돈가지고 엄청 치사하게 굴어서지난주부터 집안일할때마다 돈을 받기로 했어요 으하하하하ㅏ하하핳 시녀살이 하려고 늦은 나이에 결혼한것도 아닌데 이럴꺼면 돈이라도 벌어야겠다 싶어서 ^^ 시어머니 원망하면 뭐하겠어요~ ㅎㅎㅎ가사도우미 시급이 4시간에 5만원이니 전문가는 아니니까 1시간에 만원으로해서 30분에 5000원!!! 집안일할때 타이머 켜놓고 일해요 ㅎㅎㅎ집안일이 이렇게 행복한거였군요~그리고 밥 차려주는것도 건당 5000원간단하게 채리는건 3000원 !!!하도 신랑이 자기가 빨래 더 많이 했다고 우겨서 집안일 할때마다 달력에 적어두기로 했는데 억지로 시키지 않아도본인이 한 일을 잘 적어두더라구요 참내.....그리고 집안일 한개하면 건당 3000원씩 마이너스 해주기로 했더니 설거지하고 있거나 빨래 널려고하면계속 본인이 한다고 하네요 ㅋㅋㅋㅋㅋㅋㅋ 아참 주전자로 물 끓이는것도 사연이 긴데 것도 마트 옥수수수염차 1리터에 1000원이니건당 500원!!! ^^ 그건 매일 끓여서 한달내내 물만 끓여도 15000원 버네요~~~
    
    [['옥수수차만 끓여도에 만오천원에 빵 터짐 ㅍㅎㅎㅎㅎ\n'], ['1년이면 자그만치 18만원 ^^ 돈벌려고 작은 주전자 산건 아닌데... 어쩌다보니 그렇게 됐네요 ㅎㅎㅎㅍ'], ['222222'], ['돈벌생각에 물도 엄청 마시는중 ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋ 집안일이 즐거울텐데 ㅋㅋㅋㅋ\n서방님 지갑은 슬픈...ㅋㅋ'], ['근데 신랑이 하는일은 굉장히 소소한건데 건당 3000원은 넘 과한거 같아요 1500원으로 줄여야할듯!!'], ['재밌게 사시네요\n웃다 갑니다~~~'], ['잔소리하거나 시어머니한테 일러바치는것도 하루이틀이지 전혀 개선이 안되서 ㅠㅠ 근데 우선 정산할때 돈을 줄지 안 줄지 의문스럽내여 ㅎㅎㅎ'], ['아고 왜이리 웃기시나요 ㅎ'], ['ㅎㅎㅎ 이렇게 버는방법도 있군요^^\n난 매일아침 4시에 도시락 9인분씩 싸는데.\n금부치라도 해달라고 해야겠어요^^'], ['저는 둘다 맞벌이인데 집안일을 저만하는게 넘 억울해서 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 에어컨도 작년에 제 돈으로 샀는데 괜히 샀다고하길래... 앞으로 에어컨 켜면 니방들어가있고, 잘때도 니방에서 에어컨 없이 자라고 해놨어여 ^^ 소양강님도 금부치나 9인분 도시락 건당으로 계산 고고 ㅋㅋㅋㅋㅋ케'], ['그래야 겠어요^^'], ['뜨악ㅎㅎㅎ\n넘 재미나게 사시는거 같아요~\n두분 귀여워요^^'], ['신랑입이 방정이라 그래요 ㅎㅎㅎ \n제가 맨날 잼있게 해줘서 신랑은 싱글벙글이고, 저는 짜증나서 개죽상하고 있어요 ㅋㅋㅋㅋㅋㅋ'], ['아이코ㅎㅎ'], ['알콩달콩 재미지다^^'], ['ㅎㅎㅎ 애도 없는데 이런맛이라도?? ㅎㅎㅎ'], ['재미있네요~ 집안일하고 건당 돈을 내통장으로 옮겨야겠군요. 근디, 모든 돈과 통장이 내꺼로 되어있어서 저는 그돈이 그돈이네요~^^\n저는 그냥 단돈 천원도 마누라한테 받아가는 애기같은 신랑보면서 그냥 웃어요~'], ['신랑돈을 제 돈으로 만들 수 있는 비법 전수 부탁드립니다 ㅋㅋㅋㅋㅋㅋ'], ['저는 첨 결혼할때부터 신랑통장으로 급여가 들어오면 제이름 통장으로 싹 옮겨버렸어요. 그러고 제이름으로 된 통장이 여러개라서 그중 하나를 신랑이 사용하는 통장으로 활용~ 제이름으로 된 체크카드를 줬어요. 사용내역 문자가 저한테 오지요~ㅋㅋ 기름값과 술약속이 있으면 미리 제가 그통장으로 돈을 넣어줘요~\n현금은 그때그때 필요한 음료수값 타가요~\n저희 신랑은 첨부터 본인 이름으로 된 신용카드가 지갑에 있을 경우 술이 취했을때,\n아~ 내가 낼께~\n큰소리치는 걸 안할 자신이 없다고 하더라고요. 총각때는 그게 용서가 되지만, 가정이 생겼으니까 그럴 수는 없다고 얘기하면서 첨부터 본인 신용카드를 저한테 주더라고요~'], ['아~ 저랑 반대네요 ㅠㅠ 제가 신랑이랑 결혼하게 된 제일 큰 이유가 신랑이 물욕없는 삶을 살아서이거든요... 저는 있는돈도~ 없는 돈도 다 긁어 쓰자주의라 ㅠㅠㅠㅠ 저렇게 반대로 안해주는게 오히려 다행이네요 ^^;;;;'], ['ㅍㅎㅎㅎ 이집  시트콤이여..\n역쉬나~~'], ['요즘 그나마 덜 시트콤같아진거같아요 ^^ 주말내내 붙어있어서 짜증도 날만한데 이번주말은 웃긴 일들이 많아서 잼나드라구요 ㅎㅎㅎㅎ'], ['ㅎㅎ 궁금하나 참겠어요..'], ['하나하나 상세히 썰을 풀겠나이다 ㅎㅎㅎ'], ['ㅋㅋㅋ잼나네요ㅎㅎ 몇달이나갈지 궁금합니다ㅎ'], ['우선 이번달은 내 꼭!!!! 받아내리 ㅋㅋㅋㅋㅋㅋ'], ['우와~~  저도 따라할까봐요  ㅋ'], ['아라윤님은 저 따라하면 애들 케어비까지 들어가서 떼부자 되시겠는디유??'], ['취미로 패턴사고 원단사고 그런걸로 나름 퉁치고 있다죠?'], ['암요암요~ 그걸로 퉁치면 충분하죠 ^^'], ['ㅋㅋㅋ 귀엽게 사시네요~'], ['ㅋㅋㅋㅋㅋㅋㅋ 과연 그럴까요 ㅠㅠㅠㅠ 제가 아는 어떤분은 남편 없는 삶은 생각하기도 싫을정도로 완전 의지하고 산다던데 ㅋㅋㅋㅋㅋㅋㅋ 비교하면 안되겠죠?? ㅠㅠ'], ['그분이.... 이상한거..... 아닌가.... 요? ㅋㅋㅋ'], ['캬캬캬캬 빵 터졌으여 ^^ 그쳐?? 그분의 신랑이 이상한거???'], ['당연한듯 치사한듯 \n알다가도 모르겠소 ㅋㅋㅋㅋㅋ'], ['아하?? 몬말인지 이제 이해함 ^^ 오죽하면 제가 이러겠습니꽈 ㅋㅋㅋㅋㅋ'], ['서방이 잘못했네 ㅋㅋㅋㅋ \n가사노동 도 노동인데 대가를 받는게 당연하다는 뜻인데 ㅋㅋㅋ\n참 내가 이거 했으니 돈달라 대놓고 말하기는 쫌 치사해보이고 ㅋㅋ'], ['아하 이제 완벽이해 ^^'], ['즤집엔 씨알도 안 먹힐 소리인뎁~\n그저 알콩달콩 부럽습니다~\n때부자되소서~~^^'], ['ㅅㅂㄴ은 알콩달콩이라 생각안하실듯해요 ㅋㅋㅋㅋㅋㅋㅋㅋㅋ 때부자안되도 되니까 분담한 가사일을 좀 열심히 해주었으면 하는 바람이 ㅋㅋㅋㅋㅋ'], ['재밌게사시는거같아 부럽네요~'], ['헤헤헤 아닙니다~ 제가 잼난 부분만 적어서 그래요 ^^'], ['부러워요 ㅎㅎ'], ['에이~ 뭐가 부러워요ㅠㅠㅠㅠㅠ 요새는 남자도 집안일 기본인데 ㅠㅠㅠ 저희 남동생만하더라도 거의 살림을 산다수준으로 집안일 잘하더라구요 ㅠㅠㅠㅠ 친정엄마가 남동생을 어릴때부터 교육을 잘 시켜서 그런거 같아요'], ['역시나ㅋㅋㅋ\n조용하길래 뭐라고 할줄알아써 ㅋ\n구원미님은 역시..대형이야ㅋㅋㅋ'], ['열심히 글 적고 있는데 ^^ 많이 조용한가유?? ㅋㅋㅋ 저는 전생에 개였던거 같은데 이왕이면 소형으로 해주시면 안될까유?? 소형견?? ㅋ캬캬캬캬캬캬캬캬'], ['혹시 모르니 중간정산 권해드립니다'], ['안그래도 보름마다 정산해야되겠다고 말은 던져놨어요 ^^ 근데 콧구멍으로 듣는 이 싸한 기분 ㅠㅠㅠㅠ'], ['ㅎ 재미있게 사시네요'], ['감사합니다 ^^ 사실 글을 잼나게 적어서 그렇지 엄청 왈왈왈거리면서 많이 싸워요 ^^'], ['우리집에도 시행이 시급합니다\n우리꿘미언니 이름이 지혜언니고 \n형부이름이 현자님???']]
    
    3729
    청소 어떻게 하시나요.. 맞벌이 가정이고 주부 15년차에요.. 지금와서 이런말 하는게 참 이상하네요맞벌이에 딸아이 하나에요아이 아빠는 매우 가정적이고 자상해서 아이의 식단 하나까지 잘 챙기고요 정시 퇴근해서 본인이 할수 있는 집안일을 먼저 해주기고 하구요.. 집안일 나누는걸로 힘들게한적 없어요평일에 로봇 청소기가 아이 집에 올 시간 맞춰 하루 4번씩 돌아가고 얼마전까지 공기청정기도 있었는데 어떤 이벤트 이후로 치웠어요매일 저녁 공중에 분무기로 물 분사해 아너* 청소기로 바닥 청소 하구요일주일마다 뜨거운 물로 이불 세탁하고 건조기로 돌려 말리고요 가족의 옷은 드라이크리닝 제품 외에 당일 입은것 바로바로 세탁+ 건조기 돌려요환기 좋아해서 자주 환기하구요집안에 물건 쌓아두는거 안좋아해서 가구나 잡스러운 묵은 물건도 잘 없어요아이도 건강식 자연식을 좋아하고 치킨 피자 햄버거 등은 거의 손도 안대요작년 1년동안 수영도 잘 다녀서 수영 한타임동안 1000m 넘게 수영 잘 했고 한해동안 감기도 거의 안걸렸어요..아이는 태어나서 90일에 모세 기관지염으로 1주 입원했었고 이후로도 폐렴으로 자주 입원했어요그러다 6세에.천식 아토피 판정받았어요그래서 그때부터 건조기 들였고 9세에 천식 치료 종료되어 이후 문제없이 잘 컸으나 항상 천식의 위험이 있어 청소나 빨래는 하던대로 했어요그리고 지금 13살... 갑자기 4년만에..3월 초부터 아토피가 심해지더니 3월 15일.. 천식 재발.. 이후로 대학병원에서 1주 치료받고 퇴원했다 3일후 재입원.. 이런식으로 오늘까지 입원해있네요집에가서 하루 이틀 후부터 다시 아프기 시작하니 집에 문제가 있나 싶어 애아빠랑 집을 샅샅히 소독하다시피 청소하고 다녀요..그래도 또 1주일을 못넘기고 응급실.. 입원..반복이에요스팀 청소기로 천장 벽 걸레받이 위.. 구석구석 청소해요..병원에서는 한번에 200만원 넘는 신약 치료도 두번이나 했는데 아이 상태는 나빠져만 가요..결국.. 오늘 삼성 서울로 전원가요...직장맘이라 제딴엔 한다고 한 청소가 덜됐던걸까요.. 하루 종일 청소를 어떻게 해야하나 검색하고 공부하고..스사모 카페를 처음 오게 된것도 아이 천식 때문이었는데지금.. 제가 무얼 놓쳤을까 계속 고심중이에요..깔끔하신 전업맘님들..청소는 이렇게 하는거다.. 최소한 이런 청소는 이정도 주기로 해야하는거다.. 팁좀 주세요..진심으로 간절합니다..
    
    [['아토피는 오히려 너무 깔끔해서 생긴병입니다.\n흔한 병균조차에도 드러난적이없으니 공격받으면 무방비인거죠..너무 자책마시고 면역력에 좋은 음식 드시고 병원에서 하라는데로 달하시고..청소쪽은 평범히 지내보셔요'], ['맞벌이여서.. 다른 아이들보다 깔끔 떨며 키우진 못했어요..\n면역력에 좋으라고 파프리카 신선한 제철 과일 홍삼 유산균 비타민 등등 철마다 챙겼는데..\n이번엔 뭔가 단단히 문제가 생겨버렸어요.. \n엄마로서 너무 미안해요..\n돈벌자고 애를 못봤나 싶고요'], ['요즘 초미세먼지의 영향도 분명히 있을거 같아요\n집에선 아무리 관리해줘도 아이들은 학교에 머무르는 시간이 넘 길잖아요\n마스크 잘쓰고 다니게 하시고\n집안에 혹시 공청기 쓰신다면 필터관리 정말 잘해주셔요\n환기는 자주 시키신다니 된거같고 필터 권장주기보다 열어보고 더럽다싶음 자주 교체하는게 좋아요\n프리필터 먼지도 잘제거하구요...\n 저희아이들 아토피 없앤 비결은 천연세제 살돈으로 수도요금낸다 생각하고 헹굼을 미친듯이 하는거였어요....온가족 빨래 전 할때마다 15회 헹굼 합니다 유연제 안쓰시는게 더 좋구요~~ \n'], ['답글 너무 감사드려요.. 공기청정기는.. 잘 틀고 살았어요 그런데 어느날 밤 .. 낮동안 꺼져있던 공기청정기를 틀자마자 아이가 천식 발작이 시작됐어요.. 2월에 필터 새로 갈았고 저 천식발작은 3월 일이에요.. 그러면서 넣어뒀어요.. 공기중에 물 뿌려 먼지 제거하자 싶어서요.. 헹굼은.. 저도 5회는 했었는데 조금 더 신경 써보겠습니다 감사합니다'], ['청소는 완벽하네요\n공청기는 항상 켜 두는 게 좋다고 해요\n껐다 켜면 필터에 습기가 생겨  안좋다고 들었어요'], ['저도 조카가 아토피 땜에 넘 고생해서   맘이 참 힘들었네요. 이방법 저 방법 다 쓰다보니 어떤방법이 맞았는지 모르게 나았는데 사실 아토피는 뿌리까지 뽑긴 힘들다고 동생이 그러더라구요. 집에 있을땐 괜찮아도 밖에 나가면 더 심해지는것 같더라구요.  면역 키우게 하고 동생은 땀 흘릴 운동 많이 시켰어요.'], ['땀흘릴 운동.. 올 겨울에 살이 좀 찌며 운동을 거의 못시키긴 했어요\n이미 천식 발작 이후에는 운동을 시킬 컨디션이 안되더라구요.. 걷는것도 힘들정도라서요..\n급성기가 조금만 지나면 운동 시켜볼께요 감사합니다'], ['저도 땀흘리는 운동과 유산균 정량보다 훨씬 많이 권합니다'], ['정승맘님^^ 아토피관련글 찾아보다가 쪽지를 보내신다는 몇몇 글을 봐서요~ 혹시..저도  부탁드려도 될까요?'], ['네넹저는 듀오락atp 제품 하루에 6봉에서 8봉까지도 먹였습니다.'], ['청소를 안하면 더 심해지는게 아니라면 청소는 이미 최대치로 하시고 계신거 같아요. 원인이 무엇이길래 더 심해질까요ㅜㅠ\n조리도구와 모두 스텐이죠? 이것도 중요하더라구요. 식재료도 늘 신선하게 자주 장보는것도 중요한거 같아요.'], ['네에 아이 천식 아토피 진단받을때부터 이미 모든 조리도구는 스텐이에요.. 식재료도 한번에 많이 보관하거나 냉동보관 오래되는거 싫어 바로바로 사는 편이라 양문형 냉장고가 반 이상 차는 적이 없어요.. 음료는 우유가 다이고 탄산은 아이가 싫어해서 안먹여요..\n앞이 막막하니 또 뭐가 있을가 찾게 되네요'], ['저도 어려서 아토피가 굉장히 심했었거든요\n아이의 괴로움도 엄마의 괴로움도 잘 이해가 되네요 ㅜㅜ\n얼른 치료되길 바랍니다..'], ['공감해주셔서 감사합니다.. 찾다보면 답이 보일꺼란 생각에 글 올려봤어요\n분명 현명하신 분들이 계실테니까요..'], ['제가 잘 모르지만...더이상 청소는 하실수도업고 하실것도 없는거 같아요.  수영장물이...좋지는 않을거 같은디 자극이 돼서 그런걸까요? 세제사용은 어떻게 하시는지 .. 전 향수사용 안하고 세제 극제한. 화장품도 아기로션 하나만 발라요.. 인공향이 알러지 유발시키는 느낌이라..'], ['수영은.. 어릴때 천식 치료를 받았지만 폐기능이 다른 아이들보다 낮다고 병원 교수님이 폐기능 위해 시키라고 했고 다행히 집 주변에 새로 오픈해서 필터를 국제규격 8배 장착한 곳이 있어 보냈었어요.. 최근 천식 발작 이전 까지 괜찮았는데.. 그래도 믿을수 없은 물이었겠죠...?'], ['수영은 상관없을수도 있어요..박태환이 수영을 시작한 계기가 천식때문이었대요..의사선생님이 제대로 추천해주신듯해요..'], ['아토피 비염 천식은 한형제예요\n나름 관리도 잘하신듯한데 일단 병원 예약하셨으니 알러시 검사 먼저 받아보세요\n'], ['네에.. 4월에 알러지 검사 받았었는데 9살때 받았을때랑 그닥 달라지진 않았었지만..\n지금 진료 대기중이네요.. 최선을 다해봐야겠죠..'], ['집은 문제없을 것 같아요 24시간 집에만 있을수도 없잖아요 햇볕 많이 보게 해주시고 면역력을 키우는 수밖에는 없는 것 같아요 우리 아이도 만성 두드러기이다가 요즘 좀 좋아지고 있는데 정말 기다림이 답인가봐요 쉽지 않죠 ㅠ'], ['햇빛을 못본건 맞아요.. 한겨울 집에만 있었거든요..\n비타민 D를 먹이긴 했지만 가장 좋은건 뛰어 노는거겠죠?'], ['청소 그렇게 열심히 안하고 일주일에 1-2번 하는데 아이들 안아프고 잘커요. 건조해서 그런것도 같고요. 타고난게 폐가 약한것 같은데 미세먼지 심해지면 더 심해질텐데 안타깝네요.'], ['정말 우리나라를 떠나기도 쉽지 않은데.. 미세먼지.. 너무 원망스러워요..'], ['울 둘째도 아토피 심하게 왔는데 지금은 많이 좋아져 간절기때마다 바르는 약도 건너 뛰고 있어요.그래도 비염은 달고 사네요. 전 장에 좋다는 음식들 찾아서 먹이려고 노력중이고요 청소는 전업이지만 님의 반도 못하는거 같은데 이미 최대치로 하시고 계시는거 같으신데요.^^'], ['위로 감사해요.. 장에 좋은 음식.. 그러고보니 요즘 유산균에 신경 못썼네요.'], ['청소는 차고 넘칠만큼 잘하고 계십니다.\n오히려 너무 하는건 아닐까싶을만큼이랄까요.\n저희랑 상황 비슷하세요.\n저도 15년차 맞벌이 주부고요. 외아들 천식 있어 깔끔떨며 키워요.\n직업이 간호사라 나름 관리도 잘했는데 사춘기 오며 체질의 변화가 있는지 새로운 알러지가 오더라구요.\n저희같은 경우는 구강알러지라고 특정 과일 먹으면 목이 부으며 호흡곤란 오는거라 뭐 먹을때 느낌 이상하면 스스로 그만 먹는 식으로 조절이 가능해요.\n글쓴 님의 경우도 체질변화와 관련 있어보이구요.\n병원 옮기신다니 일단 가보시고 알러지검사도 새로 하시고 집 청소나 이런건 손볼거 없어보이세요.\n힘내셔요. 지금 무지 잘하고 계신거예요.'], ['아.. 어쩜.. 저도 간호사에요..\n저희 딸도 사춘기인데.. 사춘시 전에는 이유 있는 알러지 반응이었는데 지금은 이유 없는 반응에 스테로이드 기관지확장제에 반응이 현저히 떨어져서 전원 오게됐어요..\n급성기가 계속이니 너무 걱정되네요.. 체질변화.. 알아봐야겠어요'], ['그러면서 또 적응되더라구요.\n급성기만 잘 넘어가면 좋을텐데요.'], ['네에 정말 급성기만 넘기면 그다음은 운동을 시키던 체질 변화를 시도하던 할텐데.. 퇴원하고 길어야 1주일을 못버티고 다시 입원하니 뭘 해볼수가 없어요.. 링겔 달고 있는 애 운동 시키기도 힘들고 스테로이드를 들이 부으니 살은 찌고... ㅜㅜ 먹겠다는 애 말리기도 한두번이지.. 하... 집에서 관리해서 살수 있는 정도만 되라.. 싶네요'], ['지나친 깔끔\n수영장물\n세탁시 맹물이 아니라면 세제 등등~\n원인은 잘 몰라도\n넘 깔끔하면 저항력이 없어져서 아토피가 더 자주 심해진다고,,\n저역시 뚜렷한 이유도 모르고 약간의 아토피?\n가려움증에 괴로울 때 많아요~\n식품첨가물 쪽도 한번 체크 해보세요.'], ['아이가 자연식 외에 과자나 인스턴트는 스스로 싫어해서 안먹는데 딱 하나.. 초콜렛만 먹거든요.. 그것도 끊고 있어요..'], ['아이 천식 옆에서 지켜보기 함들죠ㅡ 저희 아이도 어릴 때 소아 알러지 천식이 있어 일년에 열댓번은 병원으로 뛰곤 했어요. 5살부터 서서히 괜찮아지더니 공기 맑고 건조한 해외에 살기 시작하면서 4년 넘게 천식 한 번도 안왔네요. 일단 알러지 천식인지 알아보는개 중요하겠고, 천식은 호흡기 약한 아이들에게 나타나는 질환이니 미세먼지 영향 있을거같아요. 환 자주 시키시는데 공기청정기 안쓰시니 집안 공기 상태가 어떠려나요. 종합병원 가신다하니 꼭 원인 찾으시고 치료도 잘 하시길 바래요'], ['정말.. 해외로 나가고 싶어요.. 오염된 공기가 너무 원망스럽내요.'], ['작년에 아토피에 그렇게 안좋다고 해도 슬라임 갖고 놀고.. 그러긴 했눈데 그때는 당장 아프지 않아 잔소리정도만 했었어요.. \n다시 생활 개선위해 잡아야겠어요'], ['수영장 물이 안 좋아요 소독을 하기 때문에 아토피나 비염에 안 좋더라구요\n\n병원 치료중이시니 저는 초유와 프로폴리스 추천드립니다.\n\n우리 아이들 어렸을때 아토피와 비염으로 고생할때 많이 먹였어요 효과도 있었구요'], ['아이 어릴깨 초유는 먹였었는데.. 다시 시작해볼께요 감사합니다'], ['비염이 오면 천식이 같이 오는거 같아요 친정아버님 비염으로 인한 천식으로 10년째 대학병원 다니면서 약 드시고 계세요 평상시 비염 증상이 있었다면 알러지도 확인해 보는것도 좋을꺼 같아요..'], ['네에.. 알러지성 비염은 항상 있었어요... 환절기라 그렇겠거니.. 하기엔 약에 반응이 없어서 걱정이에요..'], ['사실 말로는 집안 청결이니 청소라지만 더 더러운 오염된 지역에 사는 사람들은 씻지 않아도 건강하잖아여. 제가 생각하기로는 개인 면역 같아요. 저보다 청소며 빨래며 대충 사는 지인의 아이들은 너무나 건강해여. 아이 아토피를 유발한건 집에 있는 요인이 아닐거란 생각이 드네여. 그리고 요즘 아이들이 밖에서 노는 시간이 줄어서 예전보다 면역이 약하단 생각도 들구요(미세먼지 때문에 놀수도 없겠지만...) 엄마로선 최선을 다하신걸로 생각드네여.'], ['위로 감사합니다 \n뭔가 원인이라도 알고 싶네요'], ['저는 연수기랑 보습(로고나로션+로고나오일),섬유유연제 중단,세제 최소량으로...지금은 아토피 없어요.외관상으로 막 심했던 건 아니었고 살 접히는 부분과 가려움 요정도였구요.도움되시길^^;'], ['네네 감사합니다 \n섬유유연제는 저도 안쓰는데 세제는 줄여볼께요'], ['혹시 화분 새로 들인거 있으신가요? 집안에 환경의 변화가 뭐가 있나 곰곰히 생각해보세요'], ['식물은 키우는.재주가 없어서 못키워요...ㅋ\n제일 처음 시작이 아이가 저녁에 엄마 아빠와 이야기 하다 저희 보는 눈앞에서 온몸에 발진이 생기며 였어서.. 더욱 당황스럽네요'], ['플라스틱을 줄여보세요 특히 식기...\n비스페놀 파라벤 프탈레이트 나오는 제품 줄여보기요\n벽지가 실크벽지일경우에도 아토피가 심해질수 있다했어요 벽지도 확인해보세요\n'], ['2년전 집 도배 장판 바꿀때 일부러 합지로 했어요.. 플라스틱은... 안써요... 그래도 혹시 모르니 찾아볼께요  감사합니다'], ['공청기 필터도 첨엔냄새도많이나고 포름수치도오르더라구요\n저도 아토피에 비염아이들키우고 저도 예민한 사람인데 공청기 틀땐 필터교체전 한달정도 밖에 빼놓고 냄새빼고 사용해요\n\n무튼 우유는 먹이지마시고  소화잘되는 음식으로 주시고청소는지금도 부족하지않으신거같아요\n오히려 너무깨끗해그런게 아닌가싶기도하네요\n\n아토피에 비타민D도 중요해요\n해많이보게해주시고 변은잘보나요?\n몸에서 해결하지못해 피부로나오는데 요즘 공기도 워낙좋지않으니ㅠ\n저는 얼굴이가렵거든요\n\n에휴 아이키우기 참힘들어요\n아이가 나아졌던 경우도있으니 다시 좋아질거예요\n지금 몸과 마음이 힘들고 죄스러운 마음 갖고 계시다면 가장 힘들 아이를 한번더 안아주시며 나아질거라고 자꾸얘기해주세요\n\n힘내세요!'], ['감사합니다\n아이에게 제 감정을 전달하지.않고 엄마가 담대해야하는데 제 약해진 모습을 자꾸 보여줘 미안할 따름이에요..'], ['그동안도 잘 해오셨고 지금도 충분히 넘치게 하고계신걸요.. 너무 자책하지마세요~ 천식아토피알러지비염 원인은 자가면역때문인데...아마도 아이 나이를 보니 호르몬변화때문에 면역체계가 무너진건 아닌가싶어요..제가 해결책은 알려드리지못하지만 토닥토닥 해드리고싶네요.. 이 고비 잘 넘기고 빨리 좋아지길 바래봅니다ㅜㅜ'], ['지독한 사춘기를 겪는다고 하면 이 시기가 지나갈꺼니까.. 버틸수 있는데 어제 밤 입원해서 지금까지는 병원 의사들이 돌아가며 계속 문진만 하고 고개를 갸우뚱 하네요.. 허허.. \n위로 감사합니다'], ['지금 하시는 것도 지나치게 잘 하시는 것 같아요..아이가 사춘기라 하는거 보니 한창 성장기인가본데..아이가 성장하느라 면역력이 떨어져서 그럴수도 있어요...멀쩡한 아이도 그 시기 되니 삐쩍 마르고 아무 이유없이 입술 부르트고 계속 늘어지고 하더라구요...\n아이 영양상태에 신경써주시고..운동만 좀 꾸준히 해주고 병원에서 하라는 대로 하는 수밖에 없지 않을까요?\n그정도 청소하시고 주변환경 유지하시느라 힘드셨을텐데 너무 자책하지 않으셨음 좋겠어요..'], ['네에 퇴원만 할수 있게 되길 기도해요.. 집에서 뭐든 해주려 준비중인데 집을 오지 못하네요'], ['제 생각에도 청결 문제는 아닌것 같습니다. 먹는것도 더 잘 먹이신 듯 하고요...\n확실한 근거는 없으나... \n산에 자주 데려 가시는 방법은 어떠세요? 첫째도 아토피가 있었는데 우연히 산을 끼고있는 동네로 오면서 한두달 만에 없어졌어요.\n어느날 보니 제가 아예 신경을 안쓰고 있더라고요.\n무언가를 놓치거나 부족한 점은 없는것 같습니다. 자책 하시 마시길 바래요. ㅠㅠ'], ['네에 그래서 아이 아빠와 이제.주말마다 산에 가자고 했어요 근력도 키우고 담대함도 키우고~ 계획은 가득 가지고 있네요..'], ['청결문제는 아닐것 같다는 생각이 조심스레 듭니다.. 혹시 사시는 지역이 어디신지요..? 환경과 관련있을지도 모르겠어요. 저희아이(6세) 도 봄에 비염,알러지성 피부트러블, 귀, 기관지등등 심하게 왔었거든요. 근데 제가 그것이 한참심할때 따뜻한나라로 가족여행을 갔었어요. 도착하자마자 덥고 습한 공기와 햇빛때문인지.. 모든발진이 싹 가라앉아서 잊고있다가, 한국돌아온 다음날 아침부터 아이 눈이 붓기 시작하더라고요. 저희신랑은 천식도 왔고요. \n\n공기나 환경에 밀접한 관계가 있지 싶습니다. \n\n나중에 여유 되실때 외국에 햇빛좋고 공기좋은곳에 한달쯤 추천드립니다. 아토피 심한 아이데리고 치료목적으로 한두달살기 가던데, 정말 말끔히 나아오더라고요. 신기하게도요. 저도 이번에 제대로 느꼈습니다.'], ['지역은 인천인데.. 공단 지역은 아니고 한가로운 주택가에요.. 요즘 같아서는 정말 해외로 나가고 싶어요..\n근데 이렇게 당장 아플때는 한국을 나가기도 겁나네요'], ['딸이 3살때 아토피가 갑자기 생겼어요. 의사샘이 태어날때 아토피가 있던 아이들은 크면서 없어지는데...없던 아토피가 생긴 아이는 안 없어진다고 해서...아토피공부 열심히 했어요. ebs아토피에 관한 다큐 다 찾아봤었지요. \n 결론은 몸안에 좋은 세균의 비율이 높아야한다는 거였어요. 그래서 하루에 한포 먹는 유산균을 아침2포, 저녁 2포 먹였어요. 3개월 정도후엔 아토피는 사라졌구요.\n청소는 오히려 너무 깔금해서 문제가 되는것 같습니다.\n'], ['저도 아이가 한창 관리받을때는 유산균을 들이부었었는데 ㅎ 요즘 게을러졌나봐요\n다시 유산균 먹여볼께요\n감사합니다'], ['우리아이는 음식이 가장 영향이 큰 것 같아요. 단순 인스턴트나 첨가물 등아 아니라 그런거엔 오히려 반응이 미미하고 단백질에서 염증반응이 폭발해요.\n밀. 달걀. 육류 등에서요.. 지금보니 1년전 글이네요. 호전됐길 바랍니다..'], ['ㅎㅎ 감사합니다\n그 시기 이후로 약 1년동안 잘 지내고 있습니다\n물론 주의는 기울이지만요..\n참고로 8월에 다른 지역으로 이사왔는데 새아파트임에도 불구하고 주변이 청정지역이라 그런지 아이 컨디션이 아주 좋아졌어요\n저희가 이사 나오고 얼마 후 그쪽 인근 지역에 환경평가에서 거주 불가 이주 명령 떨어졌더라구요\n저희 살던 지역과 좀 떨어져있긴 했지만 워낙 예민한 아이라 영향을 받았었나봐요\n\n그래도 지나치지 않고 답글 달아주셔서 감사합니다 \n코로나로 어지러운 요즘 ..\n건강 관리 잘 하시고 웃는 모습으로 마스크 벗고 꽃놀이 할 날을 기다려봅시다^^']]
    
    3881
    ●● 집안일 하나씩 하기~~ 월요일은 만사 귀찮고 피곤하지요ㅜ금요일부터 사온 콩나물 있어서다듬었어요ㅋㅋ아들 좋아하는 무침해놓고~~또 뭘해놓을지~~ 끝도없을듯요^^;;아침 챙겨드세요~~♡♡
    
    [['월욜부터많이바쁘시네용ㅜㅜ'], ['청소기도 밀어야되고ㅜ\n월요일이라 너무 일찍오거든요ㅋ'], ['나도 콩나물한봉다리있는데 콩나물밥해주까해용ㅋㅋ'], ['콩나물밥 좋아해요~^^ \n반만 무치고 반은 뭘할지~~~^^'], ['보통 반은무치고 반은 국 끼리죠ㅋㅋ'], ['콩나물 국 맛나게 못끓이겠더라구요ㅡ\n차뇽이 무침만 좋아해요ㅋㅋ'], ['신김치넣고 국간장으로간해봐요~~'], ['악 그거 좋아해요~~먹고싶네요\n맑은 콩나물 그거생각했는디ㅋㅋ'], ['저는 애들데려다주고 책정리좀하고 아침 애들이먹고남긴딸기 케익처리했어요ㅠㅠ반찬 뭐해야될지ㅠㅠ'], ['맘님 요리도 잘하시자나요ㅋ\n저는 진짜 잘안해요ㅜ\n거기다 우석이한테 빠져서 나몰라라~~~~^^'], ['저도 요즘 귀차니즘으로ㅠㅠ'], ['잘하는겁니당~ 전  원래도 음식하는거 싫어해서ㅜㅜ'], ['저흰 그대로 하는데 하나하나 다 다듬으시나봐요'], ['이번에 산건 꼬리 깨끗하고 정리할거 없네요ㅋ\n콩부분 안좋은거 다 떼버리~~^^'], ['아항 그래요 저도 어제 콩나물 사뒀는데 ㅋ 급 콩나물밥 냉각이나네요'], ['양념장 만들어 비벼먹으면 진짜 맛있죠~~^^'], ['아침부터 배가 어찌나 고픈지 상상만으로도 침고인답니다'], ['그죠 저도 아직 아침전입니다~~\n뭘먹어야할지 고민중이예요^^'], ['월욜아침부터 요리하실생각하시는군요 콩나물무침 저녁메뉴반찬이네요ㅎㅎ맛나겠어요'], ['아들 좋아하는거라 똑 떨어지면 반찬거리가 없네요ㅋㅋ\n애 아픈건 좀 어때요???'], ['아침에 해열제먹이고 열떨어지는거보고 저는 출근했답니다ㅋ집에서 어떤지는 전몰라요 나머진 오늘 집에있는 신랑의몫.... ㅋ'], ['학교안가고 집에 있어야겠네요~~\n그래도 오늘 신랑분 계셔서 다행이예요'], ['저두 만사가 귀찮은거 있죠~^^'], ['월요일은 모두 힘들지요~~~~^^\n아들도 겨우 깨서 갔어요'], ['우리도 둘째가 엄마 나 열나는거 아냐?어린이집 안갈래 이러는거 있죠. ㅎ'], ['앗 귀엽네요~~~^^\n진짜 열나는건 아니겠죠???^^'], ['둘째가 쌍커플끼면 열나는거라 어제 꼇길래 니 열나는거 아이가 하면서 이마 만져보고 열나면 안된데이 이랫거든요.  약간 날듯말듯 해서~^^\n그거듣고 아침에 가기 싫으니 이불속에서 안나오고 그말하는거 있죠. ㅎ'], ['ㅋㅋ역시 둘째들은 눈치도 빠르고 재미나요~~~^^\n'], ['저두 일단 미역국부터 끓입니다..시원할때 해놓을려구요'], ['오늘 시원하니 좋네요 울집은 국은 없습니다ㅋㅋ'], ['국물이하나있어야지 애들이 아침에 밥먹고가기편하더라구요.신랑도 글쿠용'], ['아침에는 국물이 잘 넘어가지요\n미역국 맛있게 끓이는 팁 있을까요??^^'], ['팁이랄것까지있나요뭐..오늘은 황태대가리가있어서 그거랑 표고버섯넣고 한시간푹 끓일려구요'], ['와우~~ 국물맛이 끝내줄거 같아요~~^^'], ['저두 콩나물 살려고 했는데 아갸 이유식으로ㅎ \n월요일은 해야할일이 더 많은 요일인거 가타요 ㅠ'], ['일가는것도 더 힘드시죠ㅜㅜ\n월요일은 누구한테도 힘든하루네요^^'], ['금토일이 제일 조흐네요 ㅎ\n할거는 많은데 안하고 놀고 있기 ㅎㅎ'], ['주말에는 또 밀린 집안일 해야되지않나요??\n워킹맘들 정말 대단해요~~'], ['주말에는 아기 없으니 더 신나게 놀기 ㅎㅎㅎㅎㅎㅎ\n집안일은 잘 안해서 더럽으요 ㅠㅠ'], ['헙 주말에도 애기봐주나봐요~~~\n워킹맘 짱!!!^^'], ['차뇽이가 콩나물 무침을 좋아하는군요👍'], ['네 좋아해요~~ 요즘은 질렸다고 덜 먹을때도 있지만 \n새로 만든날은 엄청 먹어요'], ['질리면 콩나물 쏭쏭 잘라 넣어서 볶음밥으로 전 주네요 ㅋㅋ'], ['볶음밥에도 넣어주는군요~~\n안넣어봤어요^^'], ['야채넣고 볶다가 넣어주기도 하고 ㅋㅋ김볶먹을때도 같이넣어 볶아버리네요'], ['볶음밥에 넣는다 생각안해봤는데~~~^^\n다음에 있음 넣어볼께용ㅋ'], ['전 4일 놀면서 해야지 한던 일들 하나도 안하고 쉬기만해서 월요일 마음이 더 무거워요~'], ['푹 쉬셨으면 그걸로 된거죠~~\n새롭게 또 시작해봐용^^'], ['8월 휴가까지 노는날이 없어서 우울합니다ㅡㅠ'], ['휴가전까지 일해야되는군요ㅜㅜ\n화이팅입니다!!'], ['아침부터 부지런하시네요~전 일단 음식물 쓰레기 버리고 오고 애들 먹고 간거 설거지 하고 놉니다. 탱자탱자.ㅋㅋㅋ'], ['밥도 아직안먹었는데 시간은 참잘가네요ㅜ'], ['저도 할거 다해놓고 좀전에 밥먹고 다시 탱자탱자 하고 있어요.ㅋㅋ'], ['전아직도 안먹었어요ㅜㅜ\n울아들 곧와요ㅡ무섭다ㅋㅋ'], ['에고...제때 드셔야죠~굶지 마세요~진짜 밥심이 최고예요~^^'], ['ㅋㅋ1시넘어 차려먹었어요\n너무맛있어서 두공기 먹은듯요^^'], ['전 아이얼집 보내고 잠시 누버있네요 청소랑 해야는데... 피곤함만 몰려오는건 뭔지... 콩나물은 뭘해 먹어도 맛나지욤^^'], ['오늘은 학교에서 넘 일찍오는 날이라ㅜㅜ\n곧올것만 같네요ㅋㅋ'], ['우리집은 저녁메뉴 정해져서 두부사러가야겠네요\n아들이 두부덮밥 주문해서요'], ['이렇게 주문해주면 좋을거같아요\n두부 좋아하나봐요~~^^'], ['콩나물찜이요..백종원아저씨가 가르쳐준거있는데 전 가끔그거해먹어요ㅋㅋ'], ['콩나물찜요?? 다른거 넣는거 있어요??\n반 남긴걸로 찜해먹을 양은 아니네요^^'], ['월요일.. 진짜 헬요일이네요..\n어제 하루 종일 집에서 뒹굴뒹굴 했는데도.. 피곤한건 뭘까요?\n콩나물국도 션하게 맛나겠지요..ㅋㅋ'], ['그냥 주말지나면 피곤한거같아요~~\n콩나물 냉국 맛나겠어요^^'], ['맞아요.. ㅋㅋㅋ \n'], ['저도콩나물있는데ㅠ\n얼른먹어야하는데ㅠ 귀찮네요ㅋ'], ['시들까봐 아침에 생각나서 급 무쳤어요~~^^'], ['월욜은... 진짜 더 귀차나지지요ㅠㅡㅠ;;  저두 할일 태산인데 말이지요;;'], ['아직 밥해결을 못했어요ㅠ배고프네요'], ['뜨앗.. ㄴㅓ무 많은 일들을 하신거예요ㅠ?'], ['월요일이라 할일들이 많더라구요\n1시쯤 먹었어요~~^^'], ['에고.. 늦게 드셨네요~   저는 월욜은 왜이리 만사 귀찮은지ㅠ;;'], ['몸이 힘들죠ㅜ그러곤 4시넘어 2시간 자다깼어요ㅋㅋㅋ'], ['저도 아침부터 저녁메뉴고민했어요 뭐먹일지 모르겠어요ㅜ'], ['배부르니깐 저녁생각이 없네요ㅋ\n전 아들 간식인 떡볶이 하는중입니다ㅋㅋ'], ['전 콩나물 반찬 잘 못하겠더라구요 은근 어렵다는ㅜㅜ\n저희친정엄마는 콩나물볶음 잘해주시는데^^'], ['볶음은 뭘까용~~~^^\n콩나물로 무침만 하고 다른건 잘안해용ㅋㅋ'], ['콩나물볶음 저 엄청좋아해용ㅎㅎ\n맛나니까 기회되면 꼭 해보셔용~~~♡'], ['볶음은 뭘까 궁금하네요^^\n검색해 봐야겠어요~~^^'], ['콩나물 다듬기 정말 귀찮던데 잘다듬으셨네요~~~~'], ['이번껀 대체적으로 깨끗하더라구요\n콩껍질이랑 그런것만 정리했어요~~^^']]
    
    3901
    ♡ 집안일하네요 신랑은 장례식장 갑자기 가서..전 오전부터 집콕이 되었어요 🤣🤣🤣세탁기 일주고 설거지하고..신랑은 언제 올라는지..신랑오믄 뭐 소금 굳이 안뿌려도되죠?제가 간게 아니니까요?아 설거지 뜨신물로 했더니 덥습니다😭😂
    
    [['그런것도 있어서? 아..딴데 들릿다 오라 ..이거군요 그쵸?'], ['ㅋㅋ금방 통화했어요 아이스크림으로 사오라고요 크하하 쌩유입니다'], ['집에 올때 편의점이나 아이스크림가게 같이 어디 들렸다가 오면 될꺼예요 저도 오늘 집콕입니다'], ['네네 통화했어요 아이스크림만 사오라고요 신랑이 웃네요'], ['저도 눈뜨자마자 세탁기님한테 일거리줬네요 두번째 돌리고있는데 한번더 줘야하네요ㅋㅋ 장례식장 갔다가 소금안뿌리는거면 여러군데 들렸다오면 된다고 하더라구요! 뭐 다 미신이긴하지만'], ['ㅋㅋ네네 시댁에 들리고 마트 다녀올것 같습니다 ㅋㅋ미신이지만 ㅋ'], ['저도 미신이지만 저희 신랑 저번달이였나 갔다오는길에 여러군대 들렸다오라했어요ㅋㅋ 괜히 튼튼이한테 지장온다고'], ['네네 저도 그냥 ㅋㅋ미신이지만 전해줬습니다 잘하셨어요 맘님^^'], ['집에오실때 어디 몇군대 들렷다가 오시라고하셔요 맘님 홀몸아니시니깐요\n저도 주니가졋을때 이리저리 2~3군대 들어갓다가오라캣어요 ㅎ\n미신이긴한데 집입구에서 소금도 투척해주시구요~ ㅎ'], ['ㅋㅋㅋ집에 오기전에 여러군데 들릴듯요 시어머니랑 있거든요'], ['설겆이는 뜨신물에해야 기름도잘지고 깨끗하지요ㅎㅎ저는소금 뿌렸네요ㅜ'], ['뜨신물이라기보단 ㅋㅋ 뜨거웠어요 ㅋㅋㅋ온도를 높게 설정해둬서 맨손으론 못할정도로 ㅋ'], ['아ㅋ온도를설정도할수있나요ㅎㅎ그럴땐고무장갑을껴야지요ㅎㅎ기름기는쫙빠지고좋지요'], ['네 집전체 물온도 젤 뜨겁게 나오도록하면 진짜 뜨거워요ㅡㅡ'], ['신랑이 갔다와도 뿌려야지요. 저는 마트도 갔다오라하고 그래요. 맘님은 아이까지 있으니 더 신경쓰는게 낫겠쥬~'], ['안뿌리고 시댁이랑 마트만 다녀올듯요 미신 안믿는디 ㅋㅋ이번엔 뭐든 다하네요'], ['미신 안믿으면 사실 안해도 상관은 없다더라고요.  그래도 좋은게 좋은거지요^^'], ['여기저기 다니오믄서 아이스크림 사왔더라구요 ㅋ'], ['그렇군요. 신랑분도 그런거 신경 그래도 쓰시나봐요. 막둥이 때문이겠지요^^'], ['신랑은 시키면 시키는데로 해줘요 본인 아이스크림도 챙기왔더라구요 옥동자로ㅡㅡ'], ['집에 오기전에 세곳정도 들렸다오라는 이야기도 있더라구요ㅋㅋ근데 그런 미신들 다 따라하려면 그것도 보통일이 아니더라구요'], ['미신 안믿어요 근디 장례식장에 다녀온거는 좀 찜찜하더라구요'], ['저도 믿진않는데 괜시리 찝찝하고 그렇지요ㅎ 맘님께서는 아가들도 있으니 이왕이면 좋은쪽으로ㅎㅎ'], ['네 신랑 여기저기 다녀오고 옥동자 사오셨네요\n그느메 옥동자ㅡㅡ'], ['저두 막 믿진 않지만 3군데 정도 들리면 좋다더라구용 \n  전 이제 남편한테 가용'], ['세군데는 들리지싶어요 남편님이 어디 계신건가요'], ['넹 그렇게만 하심 충분할고에여\n  오늘 일욜 당직이라서 ㅜ 흑'], ['아이고 오늘 일하시는군요 피곤하시겠어요 화이팅 응원해주셔요'], ['일이라기보단 회사지키기 ㅋ 먼가 근데 집에잇는것보다 좋아하는듯해요 으힝'], ['시원하고 혼자있어서 그런걸까요 남자들 은근 혼자만의 시간이 함씩 필요한듯요 ㅎ'], ['딴데 들렸다 까까나 하나 사오라하세요ㅡㅋㅋ문앞에 소금두고 혼자 뿌리라 하셔도 되구요ㅡ전 차에도 뿌려요ㅡ몸에 소금도 뿌리지만요ㅡㅎㅎ임신했을땐 안보냈고요ㅡㅋㅋ'], ['신랑이 시어머님이랑 있어서 ㅋㅋ미신 믿으시는분이라 잘 하고 오지싶어요'], ['ㅎㅎㅎㅎㅎㅎ'], ['결혼하고 박깨트리고 집에 들가는데 ㅋㅋ아직도 기억이 납니다 ㅋㅋ박살 냇거든요'], ['아하ㅡㅎㅎㅎㅎ그거요ㅎㅎㅎ'], ['저도설거지하고빨래하고 쇼파누웠어요 나른한오후네요ㅠㅠ'], ['나른한 오후입니다 전 이제 화장실청소 해야겠어요🤣🤣'], ['나들이 가셨군요 ㅋㅋ네 시댁도가고 마트도 가고 잘 들리고 올꺼에여'], ['잘하셨습니다 ㅋㅋㅋㅋ그냥 뿌리면 맘편하고 좋지요 신랑님 뭐라 안하던가요?'], ['아 신랑님 귀여우신데요 아주 팍팍 뿌리 주시지 그러셨어요 ㅋ'], ['윽 저는 아직 설거지가 산더미에영 ㅠㅠ'], ['글쿤요 전 장실 청소하려고 물 뿌리고 좀 기다리고 있네요'], ['오우 ㅋㅋ 저도 집안일 하고 짐챙기다가 잠시 쉬는중이네요😂'], ['그러시군요 저는 이제 화장실 문질문질하고 오겠습니다😊'], ['저는 귀찮아서 화장실은 ... ㅋㅋㅋ 담주에 와서 하든가 해야겟어요 ㅋㅋㅋ'], ['화장실 청소하고 우아 에너지 방전됐어요ㅜㅜ피곤합니다'], ['아이궁 청소까지 다하고 떠나신거군요 엄청 피곤하시겟어요 ㅋㅋㅋ'], ['네네 그래도 깨끗해진 화장실보니 뿌듯 하더라구요'], ['다 미신이죠 소금 안뿌리셔도 되요ㅋㅋ사람들 많은곳 다녀오시니 \n빨리 씻으시면 됩니다ㅋㅋㅋ'], ['네 소금은 전 안뿌릿는데 신랑이 여기저기 다녀왔어요'], ['찝찝하시면 집에오기전에 사람많은 마트나 편의점 다녀 오라고 이야기하셔요 설거지는 뜨뜻하이 한번씩해주어야 뽀듯한거같아요'], ['이미 다녀왔습니다 여기저기 들렸어요 아이스크림도 사오고요'], ['그럼됏지요^^안들으면 개안치만 듣고서는 찝찝해서는 다하게되더러구요'], ['네 벼리님 오늘 계속 카페놀이 하시나봅니다 실시간이네요'], ['오전에 일찍일어나서인지 오늘은 집안일도 2/3다해가네요 ㅎ 요런 좋은점도 잇네요'], ['그러셨군요 저도 오늘 집안일 하고 ㅋㅋ놀고 ㅋㅋ했네요 🤣🤣 안피곤하시나요']]
    
    3980
    부부 가사문제ㅡ 뭐가 정답일지.. 회사 직원의 얘기이며ㅡ 듣고선 음.. 음.. 생각만하다 점심시간 끝났네요.ㅡㅡㅡㅡㅡㅡㅡ남편과 결혼 3년차.꽤나 행복합니다.우린 맞벌이고남편은 저보다 집안일을 많이합니다.빨래돌리고 마른빨래 접고, 청소, 쓰레기버리기, 설거지..그냥 온갖거 다합니다.근데 딱 하나음식은 안합니다.아주 안하는건 아니고 가끔 어쩌다 가끔 시키는대로 혹은 도전해보고픈 메뉴를 하긴하지만음식은 '당신의 몫!' 이란 의식을 갖고있는것같습니다.제가 남편보다 퇴근이 늦는데남편은 그동안 청소 빨래 등은 하지만저녁준비는 하지 않습니다.ㅡ배고파서 집에가면 바로 밥먹을거야. 저녁을 부탁해ㅡ라고 말해놓는 날엔 내가 끓여둔 국과 해놓은밥에 계란후라이, 김, 김치 이정도 그냥 차려놓는 정도?로 행해집니다.그렇다고 제가 다른 집안일을 전혀하지 않는것은 아닙니다.저도 가끔은 밀대를 들고 왔다갔다하고 화분에 물도 주고 빨래도 돌립니다.횟수로 따져보면 2대8  혹은 1대9정도의 비율정도로 보여지긴 합니다만..여튼 나도 못하는 집안일을 하는만큼 남편도ㅡ나는 요리에 소질이 없으니까ㅡ란 말만 하지말고노력을 해줬으면 하는 생각입니다.퇴근하고 가서 저녁메뉴고민에 불이나케 저녁준비하다보면 짜증이 밀려옵니다.차라리 청소 두번할꺼 한번만하고 그시간에 요리를 하나했음 좋겠는데..ㅡ역시 자긴 음식솜씨가 좋아ㅡ이런 듣기싫은 칭찬보다 맛이어떨진 모르겠지만 열심히 만들어봤어ㅡ하는 노력을 원합니다.어제도 퇴근후 또 저녁준빌하는데 짜증이 올라와서 결국 싸움까지 이어졌네요.물론 남편은 어제도 퇴근후 거대한양의 빨래를 접고있긴 했습니다.참고로 남편은 청소를 좋아합니다. 탈탈 털어서 깔끔하게 널어진 빨래며 다 마른 빨래를 각잡아 접는것도 좋아합니다.제가 할라치면 ㅡ내 취미생활을 뺏아가지 말아줘ㅡ라고 합니다.본인이 하는거야 좋아서 하니까요.근데 저도 요리에 소질도없고 좋아하지도 않으니 문제네요.ㅡㅡㅡㅡㅡㅡㅡ이런 얘길하는데여러분같음 뭐라 얘길해주시겠습니까?저야 이친구 입장에서 편도 들어주고싶고좋게좋게~ 방법도 말해주고싶은데ㅎㅎ그냥 허허~~ 하고 웃음만 나오네요ㅋ
    
    [['음식하면서 짜증이 난다면 힘들어서일듯한데 저라면 음식정도는 사먹거나 반찬주문해서 힘들이지않고 해결할것 같아요~ 그나저나 그외 집안일 잘하는 남편 부럽네요!!!'], ['반찬 사먹는것은 생각못했네요ㅎ\n고걸 추천해야겠네용~~^^ 감사합니다.'], ['글만 봐선 부럽네요^^;; 저희는 제 직장이 여의도, 남편직장은 판교인데 제가 모든 집안일 다하고 등하원 육아 전부 다하고있어요. 하지만 저는 아무말 못(안)해요(제 직장이 있는 곳에 살고 있기 때문이죠. 출퇴근 3시간 정도의 시간에 살림한다는 개념?ㅡ주말엔 분담합니다) 그것도 남편은 힘들다 하네요ㅎㅎ'], ['참고로 제가 남편보다 몸무게 많이 나가요ㅋㅋㅋ남편이 집안일하는거 애처로워서 걍 제가해요 다시태어나면 결혼안할거에요ㅋ'], ['저도 부럽더라구요ㅎ\n초니초니님.. 날도 더운데 바깥일 집안일하시느라 애쓰시네요.. 힘내세용♥'], ['저는 거기에 추가로 제가 직장이 한시간 거리예요. 남편은 여의도.'], ['ㅋㅋㅋ 다음 생엔 미혼! 적극 응원합니다~~'], ['너무 각박한거 같아요...서로 뭐를 하든지요...남편 부럽네요..\n여자분이 너무 이기적인거 같아요'], ['좀 그런거같죠? 글타고 ㅡ너가 너무 이기적이야ㅡ라고 말해주긴 좀 글쿠.. 그냥 잘 타협해서 다투지말고 지내~ 이정도로만 얘기해줘얄거같아요^^;'], ['남편분이 나머지 다하면 전 기분좋게 음식은 할 것 같아요..전 요리보다 빨래 널기 개기 다림질 쓰레기 분리수거가 훨 힘들어요..시간이 없다면 밥은 미리 하셔서 냉동실에 넣어두시고 먹을때마다 전자렌지에 2분30초 돌리면 햇반처럼 맛나구요..반조리식품(이마트몰, 마켓컬리, 등등) 많이 팔구요..온라인 반찬가게에 주문하시면 매일 배달된데요..나머지를 다 남편이 해주시니 이 정도는 하셔야 할듯해요..솔직히 맞벌인데..남편분한테 너무 많은걸 바라는듯해요..'], ['당신이 집안일하는건 당신이 좋아서 하는거고, 요리는 둘다 못하니 같이 반반하잔 생각같아요^^;;  이해가 되기도하고 안되기도하고.. 뭐 이해할 필요가 있는건 아니지만ㅎ 그르네요ㅋ'], ['헉..-.-; 집안일을 좋아한다고 한명이 다 하고..싫어하는건 반반하자 하는건..진짜 이기적인 생각이신듯............해결책은 한달정도 집안일을 반반 분담하시면 느껴지는게 있지 않을까 싶습니다..'], ['ㅎㅎ 그러라고 말해주긴 좀  그럴거같구요.\n얘기들어보니 그런 마음인거같드라구요~'], ['맞벌인데 남편은 가끔 쓰레기버리는거만 하네요 넘나 부러워요 ㅎㅎ'], ['저도 솔직히 부러워요ㅎ'], ['요리라고 거창하게 생각하면 못합니다. \n\n그냥 쉽게쉽게 생각하세요.\n\n깐 메추리알 사다가 간장하고 물엿 붓고 끓이면 메추리알 장조림\n어묵 사다가 짜투리 야채 대강 썰어 넣어 볶고, 굴소스로 간하면 어묵볶음\n쉰김치에 돼지고기 넣고 물붓고 끓이면 김치찌개\n\n참 쉽죠?~~~'], ['ㅎㅎ 요리를 글로 배우는건 쉬운일 같아요ㅎ\n그나저나 어묵볶음에 굴소스를 넣는군요. 득템한 기분입니당~'], ['남편이 다하고 할줄모르니 안하는건데 그거가지고 부부싸움은 좀 아닌듯요.\n반찬이 있을땐 차리기도 해주는데\n정말 여자분이 이기적이네요.\n아무것도 안하고 싶다는거네요.\n저런남편 쉽지않은데 복에겨웠네요.'], ['복에겨웠다고 얘기할께요~ㅎ 주위 친구들보면 남편들이 집안일을 더 많이하는 분위기인데.. 또 그렇지 않은분도 많이 계신거같아요~'], ['허허 웃음이 나오실 수 밖에 없으셨겠네요 ^^;;'], ['둘다 못하고 싫어하는 일인데 그냥 도우미 쓰면 안되나요..억지로 하는건 서로 화만 키우는듯...ㅠㅠ'], ['저도 그걸 추천해봤는데\n남편이 도우미 쓰는걸 싫어한다네요~'], ['남편문제네요..본인 하고싶고 좋아하는 일만 하고자하고 타인의 배려가 없네요....'], ['억지로 하다가 싸우느니 정말 안하고 안싸우는게 현명한 방법이더라구요.'], ['저랑 비슷한 상황이예요~ 저도 남편이 요리는 절대 안해요 못한다고...다른 집안일은 참 잘하는데...저도 요리 잘 못하는데....그래서 저희는 거의 외식해요..그래서 외식비가 어마어마하죠...'], ['이친구네도 외식 자주하는거 같드라구요~ 그냥 지금처럼 외식을 자주하라 그래야겠네용~'], ['친구에게남편자랑하신거같은데요?!ㅋㅋ\n저라면 요리에집중할거같아요~\n대신 설거지는 남편~\n하고싶은것만 하고살수는없으니까요~하나씩양보하며살아야죠~'], ['글게요ㅡ\n저도 자랑같더라구요ㅋㅋㅋ\n근데 또 이걸로 다퉜다그러니^^;;'], ['하고 싶은 건 남편이 선점했고, 지금은 둘 다 싫은 일만 남았네요.. ㅎ 저는 청소기 돌리는 걸 엄청 싫어하는데, 남편이 다른 건 내가 다 할게 너는 청소기만 돌려. 이러면 좀 진짜 빡칠거 같긴해요. ㅋㅋ 그냥 일주일/한달씩 돌아가면서 하면 안 될까요?'], ['글타고 이친구가 다른 집안일을 좋아하는거 아니구요ㅎㅎ 일주일씩 하는거 추천해봐야겠네요~~'], ['남편이 집안일을 많이 해줘도 불만이 있다니 여자분이 넘 이기적인것 같아요..........\n그런데  남편이 배려가 없다는 댓글도 있으니 제가 나이가 적은 편이 아니라 그런지 놀랍네요ㅜ\n'], ['이게 참 쉬운문제는 아닌것같아요~'], ['바람직하고 멋진 남편과 사시는 것 같은데.. 빨개 개는 걸 좋아하는 멋진 남편 삽니다. 그냥.. 말없이 댓글을 보여주시는 건 어떨까요?ㅋ'], ['ㅎㅎ 오일리님 댓글 보여줄께요~^^'], ['여기달린  댓글을  친구분께  보여주세요'], ['ㅎㅎ 그래도 될까요?'], ['자랑 아니시죠?  진짜 부럽내요. 저는 제가 더 잘벌고 했지만 거의 모든걸 제가 다했내요. 시집 다시 가고싶당 ~~'], ['차마 자랑하냐? 물어보진 못했네요ㅎ'], ['^^여튼 부러워요'], ['근데 아직 애도없고 신혼이라 그런거 아닐까요?ㅎ 이렇게 위안삼아봅니당ㅎ'], ['남편에게 엄마를 바라고 있으신 것 같아요. 그런데 나중에 애낳고 이유식 시작하면 요리는 엄마 몫으로 정착되지 않나요?'], ['아직 애가없으니 애생김 어떻게될지 저도 궁금하네요~'], ['저희 아이 태어나기전 루틴이랑 같네요. 남편이 집안일 다하고 저는 요리만 했었고 저희는 나름 서로 만족했어요. 아이 태어나고 일이 서로 너무 많아지자 좀 티격태격했었는데 결론은 아이와 함께 보내는 시간 제외한 일반 가사일은 대부분 외주 돌리면서 ㅎㅎㅎ 반찬배달 세탁소 청소이모님.... 아름답게 정리되었습니다. 서로 둘 다 하기 싫은일이 있다면 돈을 써서 몸과 마음을 아끼시길 추천드려요. 관계에 좋습니다.'], ['현명하십니다~~^^'], ['남편이 너무잘해주니  점점 더요구  하는 느낌이드네요 우선 부럽기도 하지만 ᆢ\n그래도 너무 이기적인 느낌이들어  남편이 짠한생각이 듭니다'], ['저도 많이 외식하고 그렇지만 매번 그럴 수는 없으니 남편분의 솜씨가 부족해도 김칫국이나 계란, 김만이라도 꺼내서 혹은 포장반찬이라도 곁들여 차려도 무방할테니 아내도 차려놓은 밥상 받을 수 있어야 하지 않을까요?\n\n요즘 우리의 모습은 외양은 상투시절 옛날 사람들의 모습 같은 것은 없고 아침국보다는 모닝커피, 브런치 카페를 찾는 세상으로 변해가고 있으나, \n현실은 아들 자식은 키워서 결혼하면 차려진 밥상에서 먹고 딸자식은 키워서 결혼시키면 일하고 늦게 돌아와도, 배가 고파도 차려진 밥상을 거의 못받고 산다. \n\n이것만 놓고 본다면 정신적인 세계는 습이 되어 참으로 쉽게 바뀌지가 않은 부분이구나를 생각하게 합니다.\n\n'], ['22.......남녀 바뀐 글이었다면....늦게 퇴근하는 남편 밥 안차려주냐는 댓글 없었을까요 ㅎㅎㅎㅎㅎ'], ['배부른소리 하시는 것 같아요 ㅎㅎ 그런 것 조차 손하나 까딱 안하고 맞벌이 하는 사람들이 얼마나 많은데요 ㅠㅠ \n불평하기 시작하면 끝이 없는 것 같아요. 좋다좋다 생각해야 정신건강에 이로운 것 같아요. 무조건 희생하라는 말은 아니지만 남편이 저정도 해주는데 요리 못해도 충분히 해줄 수 있는 것 같아요. 그것도 싫으면 그냥 사먹어야죠 어쩌겠어요.'], ['전 음식만자신있어서 음식만하고 나머진다해줌좋겠는데 ㅜ ㅋㅋㅋ 뭐든딱맞게 만나기 참 어려운거같아요'], ['정답이 있겠습니까만, 각자 선호하는 집안일하는게 서로 좋은 것 같습니다. 전 남자인데 평일 주말 요리 담당이구요 와이프는 청소 빨래 담당입니다. 전 요리해도 와이프 피곤한거 같으면 설거지도 하고 와이프도 똑같습니다. 누가 더 많이하냐 따지는 것보다 그냥 내가 더 하자하는게 맘이 편합니다. 서로 잔소리할때도 있지만 습관들이면 적응되고 맞춰지더라구요'], ['진짜 부럽네요...남편이 빨래에 청소만 해줘도 저는 땡큐일듯인데...'], ['가끔은 아무 말 안하는 것도 방법입니다.ㅋㅋㅋ\n나랑 사는 것도 아니니 그냥 그러려니 하고 잊으시는게 좋지 않을까요?'], ['남자인데 저런 남편 원합니다'], ['(맞벌이든 아니든) 남자가 집안일 덜 하는게 디폴트값이니 저것도 안하는 남편을 둔 입장에서 여자가 이기적이라하는거같아요. 학급 청소시간에 친구사이라 생각하면 답 나오지않나요?? 할 일 10 중에서 한 친구가 나는 교실쓸고닦고 책상의자 배열하고 쓰레기통 비울게(약 6-7) 너는 화장실청소 딱 하나만해!!!(3-4밖에 안되잖아?) 라고 정해놓은 상황인데요 ㅎㅎㅎㅎ 내가 교실쓰는걸 선점자보다 더 선호하지 않는다고해서 화장실청소를 하고싶은건 아니거든요...ㅎㅎㅎ;;; 비슷한예시로 조장님~ 제가 개요짜고 피피티만들고 보고서쓰고 다른건 다 할게 발표만 해주세요^^!! 조장이 발표 싫다하면 이기적인건가요...? ㅎㅎ'], ['저는 여자분입장도 좀 이해되는데요..\n사실 요리를 좋아하면 괜찮은데 싫어하자나요.\n다른 집안일은 사실 조금 미루거나 늦어도 크게 상관없지만 밥은 때가 있어서 마음이 급해요. 그리고 계속 서서해야하기 때문에 힘들어요. 퇴근하자마자 집에와서 급하게 밥부터 해야하면 너무 힘들 것 같아요.\n게다가 먼저 퇴근한 남편이 기다리기까지하고 있으면...ㄷㄷ\n저는 전업주부라 의무감으로 하지만 진짜 밥하는거 너무 싫거든요.'], ['삶의 기준은 높게 잡아야 한다고 생각해요. 그정도면 배부른소리다 라는 말은, 어느정도 가사분업을 나누고 있는 당사자분 입장과는 전혀 관련 없다고 생각해요. 가사노동 안해주는 다른집 남자들을 보면서 자기위안하며 내가 욕심이 많은거다, 내가 틀린거다...라고 생각하는 거 별로에요...그리고 청소 보다는 요리가 훨씬 노동강도가 세다고 생각합니다. 뜨거운 불 앞에서 연기 마시고, 조리하고, 냉장고 왔다갔다 꺼내고 다시 넣고 칼질하고 기다리고... 요리 하는 내내 조금이라도 딴짓하면 음식 망하고....  평소에 정리정돈 잘 된 집이면 청소 하는거 많이 어렵지 않잖아요? 청소는 몇시간 텀 두고 쉬었다가 해도 요리처럼 뭔가 망할 걱정은 없잖아요. 뭔가를 깨끗하게 만든다는 행동 역시 사람에게 만족감 주기도 하구요. 그런데 요리라는 과정 자체는 어지르게 되기만 하니까요..'], ['이건 집안일을 나누는 걸 떠나서 상대방에 대한 배려 문제인 거 같아요. 배우자가 굳이 하기 싫고 자신없다는데 기대하는 아내 분이나, 아내가 원한다는데 원천봉쇄 안 하시는 남편 분이나..서로 한발씩 양보하면 될 일인데 아직 신혼에 가까워서 이런 조율이 계속되고 있는 거 같네요(저도 한 5년차까진 이런 사소한 걸로 자주 싸웠더랬죠ㅎ). 가사분담 문제가 아니다, 서로가 원하는 부분들을 저금씩 맞춰가야는 것 같다..좋게 말씀해보심 어떨까요?'], ['문과생에게 매일 수2 풀라는 것과 같습니다. 좋아하는 것 해야죠. 음식 사다 드시는 것을 추천합니다.\n공평하려고 결혼한 것이 아니기에 결혼생활엔  모범 정답이 없죠.'], ['반찬을 한 일주일치는 구매하면서 해보는 것도 좋을 것 같아요'], ['정답은 없지만, 현명하게 잘 터놓고 얘기해보시면 어떨까요? 아직 초반이라 그런 것 같아요~'], ['이기심이 강하다고 느껴지네요\n일단 남편을 존중하는 마음이 느껴지지 않구요 만일 남편이 외벌이였다면 아내분이 어땠을까 상상해봤는데\n일하고 들어온 남편분을 그냥 편하게 두지 않을 분 같네요'], ['ㅠㅠㅠ 어렵네요. 그치만 남편분 역할도 큰 역할을 하고 있기에 머라고 얘기해야할지 ㅠㅠㅠㅠ'], ['여러 의견들 감사합니다. 화해했다고 하니.. 더 얘기하진 않으려구요^^ 담에 또 이문제를 얘기한다면 주신 답변들 전하도록 할께요~'], ['잘 살고 계신 것 같아요. 조금 조율만하시면 될 것 같아요. 배고픈데 밥 없음 짜증 나죠. 자기가 좋아하는 집안일만 해서 짜증나는 포인트도 이해합니다. 앞 댓글 처럼 반찬 사두기. 주말에 해놓기. 등 현명하게 해결하실 수 있을 것 같아요. 좋은 남편 분이신 것 같으니 이해해주세요. ^^'], ['너무완벽한남편을 바라는것 아닐까요? 여자가 이기적으로 느껴집니다 그리고 어린여자같아요 철없는..'], ['남편분이 엄청 대단하시네요. 부럽습니다.']]
    
    4044
    묘하게 집안일 시키는 미라 7세 3세 키우는 워킹맘입니다.고단한 주말 잠들기전 미라들어왔다애둘 운동화 손세탁하고 잠듭니다.묘하게 집안일 시키는 미라!다들 굿밤^^♡
    
    [['맞아요~~~~~청소하는거 귀찮아서? 미라 시작했는데 은근히 사람 엄청 부지런하게 만들더라구요~~~~ㅎㅎㅎ\n지금 저도 내일을 위해 건조된 빨래 정리하고 잠자리에 누웠어요~~~ㅎㅎㅎ\n더더 화이팅 해요~~~우리~~~~^^'], ['ㅋㅋ맞아요\n괜히 자극받고 힐링되고 그런 미라에요'], ['저두 그래요 ㅎ 설겆이  미루려다  저절로  손이 가서 다  치웠네요ㅎ  부지런하게 만들어주는거 같이  미라카페  더  애정이 가네요'], ['ㅋㅋ\n미라하고 설거지도 빨래도 자주?하는 제가 신기하기도 하고\n\n수박1통이 냉장고에 그대로 들어가기도 하고 \n\n집안일이 쉽게 자주하게 되는 매력 가득한 곳이에요ㅎㅎ'], ['격한 공감이에요.... 저 미라하고나서...\n저 집안일안하려고 미라시작했는데\n한달째 대청소애요 ㅠ'], ['저 둘째임신때 휴직하면서 미라절정기를 달렸는데\n\n그때 살이 안찌더라구요ㅋㅋ\n\n지금은 주말마다 100개씩 버리고 있어요ㅋㅋ\n\n근데 한 이틀 휴가내고 다 뒤집고 싶어요'], ['헉.... 저 휴가냈어요\n오늘 어느정도했고 내일.내일모레\n정말 각잡고 할려구요.\n진짜... 이상태로 여행다녀와도 여행다녀온기분이 안들것같아서... 냈답니다..ㅠㅠ 와 저랑같은생각이시네요'], ['ㅋㅋㅋ 근데도 신기한게  예전처럼 집안일이 짜증안나네요 ㅎㅎ'], ['약간 성취감? 들죠ㅋㅋ'], ['더 웃긴건  예전에는 신랑이 집안일좀 그렇게 해줬음 했는데 이제는 그런맘이 안드네요 ㅎㅎ 내가하는게 힘들지않으니 원하지도않게되니 이것이 미라클이죠 ㅎㅎ'], ['설거지가 너무 귀찮았는데 식사 후 바로바로 일어나게 만드는 미라에요 ㅎㅎ'], ['그래서 다들 미라클 미라클 하나봐요^^'], ['동의합니다!'], ['맞아요^^'], ['제목을 묘하게 집안일 시키지 마라?로 읽고 뭐지 하고 들옴요ㅋ'], ['마라ㅋㅋㅋ\n그것도 맞아요ㅎㅎ'], ['우와 표현 죽입니다 ㅋㅋㅋㅋㅋㅋ'], ['다들 동의하시죠?ㅋ'], ['저도 주말이면 누워서 아무것도 안하고 쉬어야지 다짐했다가;;;; 꼭 미라보다 못쉬고 청소하게되용 ㅋㅋㅋㅋㅋㅋ'], ['이상하게 까페들어오면 \n집안일 하고 있어요ㅋ'], ['인정합니다 미라가 저 일시키요!!ㅎㅎ'], ['근데 기분 나쁘지 않다는거ㅋㅋ'], ['맞아요^^~~ ㅎㅎ'], ['ㅋㅋ공감해요~~~'], ['다들 같은 맘이죠?'], ['아 맞아요 ㅋㅋ 정리하면서 자꾸 눈에 거슬려서 청소하고 그래요'], ['사진보다 내집보면 속터져서 움직이게 되요ㅋ'], ['저도 자투리 시간에 안방 화장실 청소하고 나왔어요^^ 곰팡이 살짝 핀 부분에 약 발라놓구요~ 외출하고 돌아오면 뽀송뽀송 깨끗한 화장실이 기다리겠죠? ㅎ 성취감 들어요~~'], ['성취감ㅋㅋ\n이거 중독성 있어요'], ['격하게 공감되요\n저도 여기에 들어와서 정독하고 집안일 계속하는데 잼나요'], ['한참 비울때는 식욕도 떨어지고 진짜 잼났어요ㅋ'], ['묘하게 더 청소하게만드는\n일하게만드는  정말공감해요^^\n\n월차쓰고 집 정리하고픈맘이 막 솟구쳐요^^'], ['저도 휴가쓰고 싶은맘 가득한데\n애들 아플때마다 쓰다보니 휴가가 부족해요ㅜㅜ'], ['묘하게 충돌질 하는 미라. ㅋㅋ 정리의 동력을 얻게됩니다.'], ['충돌질ㅋㅋ맞아요'], ['아침에 미라 꼭 보는데 몸을 움직이게 하는 원동력이죠 ㅎㅎㅎ 좋아요~~'], ['맞아요!\n돈도 안들고 깨끗해지고\n가진것에 감사하게 되는 ㅎㅎ'], ['완전 공감합니다!!!! ㅋㅋㅋ 더 부지런쟁이가 된 것 같아요 ㅋ'], ['ㅋㅋ맞아요\n더?! 부지런쟁이 ㅎㅎ'], ['ㅎㅎ 동감합니다 주말동안 3년간 모았던(못버리고 쟁여놓은?) 큰애 학습지며 전집 시트지들 정리했어요 이전같으면 한번 더 보겠지~ 동생이라도 보여야지~ 이러면서 미뤘을텐데 갑자기 의지가 생기면서 전집 시트지 요약 부분만 뜯어서 책에 붙여주고 다 버렸어요 ㅎㅎ'], ['ㅋㅋ미라의 마법에 걸리셨군요'], ['깨끗한 집 사진들이 자극줘서 게으름피우다 정신번쩍들어서 청소하게되죠'], ['맞아요\n누가 비교하지 않는데\n혼자 자극받죠'], ['미라는 부지런해야한거같아요 ㅋ'], ['부지런하게 만드는 것 같아요ㅎㅎ'], ['본의아니게 현모양처의 길로 가고있어요 ㅎ'], ['저도저도\n현모양처 한번씩 해요ㅋㅋ\n\n댓글에 힘입어 오늘은 천사엄마가 되어보려구요!ㅎㅎ'], ['묘하게 일하고 있는 나~~~ 아무것도 안하는데 왜 깨끗하지 생각했는데 이젠 사부작사부작 움직이고 있더라고요'], ['맞아요ㅋㅋ\n사부작 사부작ㅎㅎ\n물티슈쓰고 바로 안버리고 뭐든 닦는 모습이 가끔 신기해요'], ['이거보다 화장실 청소하러 간 1인입니다~'], ['전 샤워할때마다 변기랑 세면대 닦고 있더라구요-.,-;;;'], ['묘하게~\n가만있다가 글보면 벌떡일어나게 만드는 미라~\n설겆이얼룽,화장실얼룽,현관얼룽 이건뭐죠?\n자석예요 자석!!!'], ['내사랑님도 저와 같은 증상이군요\n미라보다 소파에서 \n벌떡 일어나 씽크대 서랍정리하고\n벌떡일어나 상부장 정리하고\nㅋㅋㅋㅋㅋㅋ\n'], ['저는 이글쓴 이후\n자기암시처럼 다시 버리기 축제가 시작되었어요ㅋㅋ'], ['숙제아닌 축제가 시작되셨군요 축제!!축하합니당'], ['ㅋㅋ감사합니다\n월.사무실\n화.집\n수.차\n목.집\n금.사무실\n\n비움 스케쥴입니다ㅋㅋ'], ['집안일 시키며\n사람 평온하게 만드는\n미라입니다ㅎ'], ['맞아요^^;;\n여백의 미...선조의 지혜를 배웁니다ㅋ'], ['ㅋㅋ너무 공감이 돼서 글남겨요~~\n 오늘하루도 계속 움직이고나니 학교마치고온 아들이\n엄마 오늘 또 짐 옮기셨나봐요~? 하네요 ㅎ\n몸은 바빠져도 기분은 좋아지네용'], ['다들 공감하시니\n저도 분발합니다'], ['저도 완전 동감이요 ^^  집에서 쉬질 못해요'], ['어제는 사무실 책상 뒤지고 파쇄 100장 한것 같아요ㅋ'], ['저또한 화장실 청소하러 갑니다~~^^'], ['저는 미뤄둔 병원진료갑니다ㅋ'], ['진짜게을르고 집정신없어서 시작한미라..매일청소하고  월요일되면 대청소하게되구...계속버리고,유치원서 계속가져오고,\n게을르고싶어요.대신밥은..안해집니다 ㅎㅎ밥하면 지저분해져서요.'], ['저는 애들 잘때 반찬해요ㅋㅋ\n기름튀고 다칠까봐ㅜㅜ'], ['동감입니다!!'], ['ㅋㅋ\n다들 까페보다 집안일하시죠?\n그런거죠?ㅎㅎ'], ['방금 벽 한면 닦았네요 ㅎㅎ'], ['맞아요 진짜 동감이요 ㅎㅎㅎ']]
    
    4083
    집안일이란... 뭘까요... 빡침   어제 화장실을 들어갔는데 갑자기 물때가 너무 거슬리는거예요 ㅋㅋㅋㅋㅋ 락스를 뿌리고 나왔는데 게임하고 있는 남편놈을 발견했습니다가만히 생각해보니 화장실 락스 청소 왜 맨날 나만 하지 싶은거예요 ㅋㅋㅋㅋㅋ 속으로 화가 났지만 티는 안내고 베란다에서 분리수거 쓰레기를 내놓으려고 사부작 댔어여  그랫더니 왜 자꾸 뭘 하냐면서 제가 집안일하니까 자기가 편히 쉴수 없다고 궁시렁 대며 설거지를 하더라구옄ㅋㅋㅋㅋㅋㅋ  지금 상황 마음에 안듦 얼굴에 써붙여놓곸ㅋㅋㅋ  어젯밤 쓰레기 버리러 오면서부터 서로 한마디도 안함.. ㅋㅋㅋㅋㅋㅋㅋㅋ 저희는 언쟁하는 스타일이 아니라서 이게 서로 기분 상한거고 어쩌면 싸운거랑 동급이거든요 ㅜㅜ  둘다 맞벌이에 주말에는 주말이라고 안하고 평일엔 평일이라고 안하고 ㅋㅋㅋㅋㅋ 도대체 언제 해요 ?ㅋㅋㅋㅋ  그렇다고 남편이 집안일을 안하는 건 아닌데 미뤄뒀다 하는 스타일이라 제가 그 새를 못참는 것 같아요 ㅠㅠ  아아...정말 집안일이란 뭘까요... ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ
    
    [['그 마음 저도 알죠ㅠㅠ자꾸 거슬리고..ㅠㅠ'], ['집안일이 그런거 같아요 미루고 미루다 하는 타입은 나중에 내가 할랬는데- 이러고 눈에 보이면 바로 치워하는 성미는 그거 다 내가 치우잖아! 하고 짜증나고.... 서로 조율을 잘 해보심이 어떨지요???'], ['저도 그런게 너무 걱정되는데.. 하 저는 메리지블루가 아니라 지금도 안하는데 결혼한다고 할까 답답하네요^^;'], ['열받을것같아요 한사람만 하는느낌이면'], ['집안일은..... 더 부지런하고 더 깔끔한 사람이 지는 게임인것 같더라고요 ㅠㅠㅠㅠ 둔한자가 승자'], ['저도 한 게으름 한 둔감함 하거든요? ㅋㅋㅋㅋㅋㅋㅋ 남편이 더 심한거같아요.... ㅜㅜㅜ 진듯'], ['말을 해야해요!! 저도 열받아서 한바탕 말했는데 일찍퇴근하는날은 하더라고요'], ['저는 신랑이 알아서 하는 편이긴 한데..뭔가 100% 딱 마음에 안들게 해요~ 하지만 그렇다고 뭐라 하면 한 성의가 있는데 기분 나쁘니까~ 말없이 제가 뒷처리를 마무리하죠~ 진짜 집안일은 해도 끝이 없어요.......ㅠㅠ'], ['저희 남편도 나름 자기는 한다고 설거지도 해놓구 박스도 치우구 하는데 왜 화장실 청소는 안할까욤... ^^...'], ['같이살게되면 정말 사소한 부분들도 많이 부딪히는것 같아요ㅜㅜㅜㅜ'], ['집안일 누구라고 할거없이 해야하는건데, 갑자기 훅 치밀때가 있어요 ㅠㅠ\n'], ['진짜 결혼하면 그런것 하나하나가 걱정이에요.. 안맞을까봐'], ['그래서 사실상 집안일은 서로 정해놓고 하는게 좋은 것 같아요'], ['헐 신부님 어쩜 저랑 똑같으세요ㅠㅠ\n집안일은 정말 더러운거못보는 사람이 하는건갸봐요ㅡㅡ 저도 넘 빡쳐서 막 하고잇음, 저희 남편도 쉬고싶다.. 편하게잇을수없다 하면서 같이해요... 집안 더러운거보고 어떻게 쉬고 어떻게 편하게있죠;;;;;ㅜㅜㅜㅜㅜㅜ'], ['저 결혼전에 엄마랑 살 때 개판쳐놓고 잇으면 엄마가 화내셨는데 이제 그 마음 이해가 돼욬ㅋㅋ \n왜냐면 저걸 치우는건 나일테니까 ...ㅠㅠㅠ'], ['보통 주변에 나이 좀 있는 맞벌이 부부 분들도 일주일에 하루 시간을 정해두고 하시더라구요ㅎㅎ 집 안에 룰을 만들어놓는 느낌? 제가 들은 거로는 다음 일주일을 위해서 일요일은 무.조.건 대청소를 하신대요. 요일을 딱 정해놓고 싹 다 냉장고, 화장실, 빨래, 바닥청소, 물건정리 다 해결해놓고 일주일 편하게 지내는 거죠 ㅋㅋ'], ['계속 그러시면 속에 쌓일거예요ㅠㅠ 평생햐야하는 일인데...정해두시고 서로 하시는 게 좋아요ㅠㅠ'], ['정해두고 하지 않는 이유가... 이거 당신 담당이잖아 라면서 책임전가하고싶지 않아서거든요 ㅜㅜ'], ['맞벌이부부들의 흔한 공통사이기도 하죠 ㅠ_ㅠ 저희는 그래서 청소 날짜 정해놓고 각자 맡은거 해결해요ㅎㅎ'], ['ㅠㅠ결혼해서는 서로 잘 말하며 조율해야지싶어요'], ['저도 비슷해요 ㅜㅜ 에휴 청소담당이 남편이라 언제 하나 두고보자 하면서 저도 안하고 지켜봤거든요 그랬더니 물때에 먼지에 하아..... 결국 제가 했네요 ㅜㅜ  화나서 막 뭐라고 했더니 결국 말싸움으로 번지고 서로 감정상하고 어떡해야할까요?'], ['그래서 저는 같이 해요. 얘기해요 남편한테'], ['후한 빡침 흔한 일상  맘을비우고 청소는 내일이라생각하고 하는게 맘편할듯'], ['아 집안일 생각만해도..같이 배여하면서 해야할듯요..'], ['제 말이 그말이요. 그래서 그노매 게임 하고 있는 꼴이 아주 보기 싫어 죽겠어요.\n어제는 하다하다. 난 내가 가정부 같다고. 스스로 일을 찾아서 하면 안되냐고 했는데 .................... \n그냥 신랑 눈에 안보이는거겠죠.. 시키기는 싫은데 알아서 하면 얼마나 좋을까요 ㅠㅠ'], ['저도 비슷한 성향인데 첨엔 집안일땜에 많이 빡쳤어요 저도ㅋㅋㅋ 요즘은 명확히 나는 이거 할테니까 자기는 저것 좀 해줘~ 좋게 말하니까 바로바로 일어나서 하더라규요'], ['집안일이란 해도해도 끝도없고 혼자하자니 왜 나혼자하지? 같이하자니 그래 힘들텐데 조금 배려하자 내적 갈등이 어마어마한 것이예요 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ\n그래서 저희는 미리 이야기해요, 아까 집보니까 이런거이런거는 조금 치워야 할 것 같은데 오늘할까? 내일할까?? 이렇게요\n피곤하다고 하면 그날은 같이 안하고 그냥 둬요'], ['청소할때 같이 하자고 말을 해봐요!ㅎㅎ'], ['못참는 사람이 결국하게되어있대요ㅠ\n저는 좀 놔두는 타입인거같은데..\n오빠가 안그런것같아요ㅠ'], ['같은 스타일이 아니라면 하는 쪽을 더 따라야하는게 아닐까요 ㅠ\n해서 나쁠껀 없으니까요 ㅠㅠ'], ['하.. 이런 생각하면 벌써부터 스트레스 받으려 해요ㅜㅜ'], ['헛 저랑 너무 비슷하신데요... ㅜㅡㅜ 제가 못 참아서 해요 ㅜㅡㅜ'], ['더 부지런떠는사람이 하게되는거죠뭐ㅠㅠ결국아쉬운쪽이하는거같아요'], ['아 ... 잠깐 남동생이랑 둘이 살때도 집안일때문에 박터지게 쌰웟는데 그때 동생이 아쉬운 사람이 하면 되는거 아니냐고 하던게 떠오르네여 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['헐 진짜 ㅜㅜ후 집안일'], ['규칙을 만드시는건 어떨까요? 맞벌이인데 어느 정도 나눠서 하셔야 될거같아요!!'], ['맞아요 공감가네요~ 해도해도 끝이없는집안일~ㅠㅠ'], ['남자분들 집안일 시키려면 몇시까지 이것좀 해줘 이렇게 부탁해 보세요 데드라인을 정하고 시켜야 움직여요! 이거 다큐멘터리에서도 나왔던 건데요 항상 신부님이 다하지 마세요.. 저녁 9시반까지 화장실 바닥이랑 변기청소 쓰레기 버리는거 해줘! 이렇게 말해요!!'], ['맞벌이 하시는데 혼자 하면 좀 화가 나긴 하죠.... 저희는 쓰니님댁하고 반대 성향이라 제가 몰아서 하는 편인데 지금 현재는 외벌이라... 불평없이 하고 있다죵ㅠㅠ'], ['저는 남편이 사부작댄다고 못 쉰다고 투덜대는 스타일은 아닌데 정말 느려요.... 그리고 매일 안해요... 그리고 해도 제 맘에 안들어요... ㅋㅋㅋㅋㅋㅋ 어찌해야 할까용ㅋㅋㅋㅋㅋ'], ['제가 먼가 하고 있으면 잘 도와주는 편이긴한데 그래도 본인 하기싫을때는 저도 청소하길 바라지 않더라구요..본인도 해야하니까..'], ['저희 남편도 그거예요 ㅋㅋㅋㅋㅋ 안하는 사람은 아닌데 쉬고싶을때 제가 청소하나까 그게 싫은거 ㅠㅠㅠ'], ['집안일 진짜 힘들죠ㅠㅠ 저희부부도 집안일 하느라 시간 다 쓰더라구요ㅠㅠ 저희는 둘이 같이하는편인데 \n제가 일 다니고부터는 퇴근이 저보다 빠른 신랑이 좀더 마니하는것같긴해요ㅋㅋㅋㅋ'], ['작은 섭섭함이 쌓이는 것 같아요 집안일은 엄청 많고ㅠㅠ'], ['그리고 맡겨서 깨끗하게하면 몰라 성에안차면 제가 마저해요'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 오 맞아요 \n설거지 해놓은거 쓰려다가 안쓰고 다시 설거지통에 넣음.....'], ['이런것 때문에 초반에 많이 싸우게 되는거 같더라구요 ㅠㅠ 서로 대화 많이 하면서 맞춰나가야 하지 않을까요'], ['같이하자고 분담해보셔요ㅎㅎ'], ['규칙을 만드시는게 좋을 것 같아요 저도 저만한다고 생각하면 화가나서 신랑한테 짜증내게 되더라구요ㅠㅠ'], ['남편이 화장실 청소 해주는데.. 물때 그대로 맞은거 보여요... ㅠㅠ 고마운데... 너무 고마운데.. 하아.. ㅋㅋ'], ['ㅋㅋㅋㅋ...... 남편들 눈엔 물때 안보이나봐요... \n저 제가 손잡고 들어가서 보여줬는데도 남편이 깨끗한데? 이랫어여 ㅋㅋㅋㅋ\n아 혈압 오를듯ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['저희는 제가 사부작대면 짝꿍이 뭐야? 뭐해?라고 오는편이에요..저러면 진짜 짜잉날거같아욬ㅋㅋㅋ'], ['거의 남자들은 비슷한거 같아욤 자기가 한다고 해놓고 세월아 네월아라 저도 제가 해버려욤ㅋㅋㅋ'], ['저도 눈에보이면 제가 먼저 하는편이라서 나중에는 그냥 시켰어요~ 내가 이거할테니까 여보는 이거 해줘 하고.. 무조건 직접하지 마시고 시키는 버릇?을 들여보는것도 좋을거 같아요!'], ['자꾸만 같이 하셔야 돼요 그러다가 온전히 본인 몫이 돼요ㅠㅠㅠㅠㅠ'], ['아... 맞벌이시면 나눠서 하셔야 하는데... 신랑분이 정 하기 싫다 하시면 일주일에 한두번정도 가사도우미 부르세요... 저는 백수인데도 신랑이 힘들면 가사도우미 부르라고 하는데.. 신랑분 너무 집안일에대한 중요성을 모르시는 것 같아요..'], ['ㅜㅜ 너무 스트레스 받지마세여~ 날짜를 정해서 하는게 어때요'], ['아 ㅠㅠ 분담을 하셔야 될 거같아요. 전 신랑이 오히려 더 깔끔해서 ㅠㅠ 아예 나눴어요. 나눴지만 각자 야근 또는 회식이 있으면 대신할때도 있구요 !!!! 조용히 혼자 하시면 나중에 뭐라하면 왜 이제와서 뭐라하냐고 할거에요. 첨에 잘 잡고 가야함...'], ['남자는 교육시키기 나름이라던데 넘 어려워요 ㅜㅜㅜ'], ['그러면 서로 정해놓고하세요~'], ['끝이없는 집안일 증말 휴 한숨나와요'], ['집안일 빡치죠 ㅜㅜ 저희뉸 화장실은 신랑이 하기로 햇는데 물때가 껴도 안하고 잇으니 참 거슬리는데 참습니닷 ㅋㅋㅋㅋ 제가 화내면 자기딴엔 그래도 설거지도 허고 빨래개달라그러면 개주고 한다면서 받아치네요 ;; 눈에 안보이는것도 알아서 해주면 좋겟는데 그건 무리겠죠 ㅜㅜ'], ['업무 처럼 서로 나눠서 해요'], ['ㅠㅠ 너무 힘들면 하지 마세요 ㅠ 스트레스받아여ㅠ 그리고 남편분에게도 역할을 정확히 분담해주는게 필요할거 같아여~ 혼자 끙끙앓지 마세여 ㅠ'], ['저희도 초반에 집안일로 엄청 싸우다가.. 물걸레로봇청소기 사고 식기세척기 사고 하면서 템빨 받아서 그나마 좀 나아졌어요..'], ['ㅋㅋㅋ 서로 조율하고 하기전에 같이하자고 해요 ㅋㅋ'], ['집안일 되게힘들죠ㅜ 저 결혼14일차고 신행다녀온뒤로 쉬는날없이 집안일이안끝나요ㅜㅜ'], ['진짜 진심 남자들은 한번에 하면되지 아님 안해도 되는거 굳이 한다고 생각해여ㅜㅜ 너무 힘든거 같아영'], ['게임하고 있는거 보면 화날거 같아요 시키는건 어떠세요?!'], ['에효.. 저도 그렇습니다. 업무분담하시고 남편이 할일은 절대 손대지 말고 냅두세요. 그러면 눈치보다가 결국에는 하더라구요. 심리전이 필요합니다.ㅠ'], ['ㅋㅋㅋㅋ 저도 그냥 방치하는 스타일인데 여름되니까 이걸 못참겠어요 ㅋㅋㅋㅋㅋㅋㅋ \n초파리도 생기고 ㅠㅜ 설거지통에 초파리 익사 현장 보고 얘기했더니 해맑게 닦아쓰면 되지 라고 하는뎈ㅋㅋㅋㅋㅋ 휴....'], ['ㅠㅠ 흐잉 저희 신랑은 화장실청소랑, 쓰레기통비우기랑, 음쓰 버리기 직접해요 ㅠㅠㅠ 요새는 다 남자들이 많이 하던데 ㅠㅠ'], ['분담 확실히하세요 더러운거 참기 힘들어서 내가 자꾸하면 다 내일됩니다'], ['맞아요! 맞벌이는 같이하면서 집안일은 제가 더 많이 하게되서 진심 빡칠때 있어요!'], ['에거 저도 분담 확실히 정하는거 추천해요 ㅠㅠ'], ['저희모습 보는거 같아요 ㅋㅋ'], ['청소는 같이 한번에 하는게 나은거같아요! 집안일 진짜 힘들어여ㅠㅠ'], ['어우 저랑똑같네요! 저도 눈에보이면해야되고 신랑은 주말에몰아서하려고해서 결국제가 계속움직이게되요ㅜㅜ'], ['집안일은 해도 티도 안나고........너무 지치는 거 같아요ㅠㅠ'], ['집안일 해도해도 끝이없는거같아요ㅠㅠ'], ['저두 속터져서 제가 하고말아요ㅠㅠ'], ['저도 그래요 ㅠㅠ특히 글쓴이님처럼 화장실이요..결국 제가 답답하고 지저분해서 청소하게되요 ㅠㅠ지치게되네요.'], ['저도 화장실 예민한데 남편은 화장실에 제일 둔감한거같아요 ㅋㅋㅋㅋㅋ 으어 ㅠㅠ'], ['ㅎㅎㅎ 욕실청소랑 쓰레기 버리는건 남편담당인데.... 욕실 바닥닦는건 잘모르는거같아여;;;; 물때껴도 잘못느끼는듯ㅋㅋ 그냥 제가하고.. 나중에 얘기해요 어떤상태가 됏을때는.. 청소해야한다고'], ['ㅋㅋㅋ 저도 제가 막 부산시럽게 움직이면~  그래도 신랑이 뭐하냐고 물어봐주고 다른 집안일 하고 해서~\n저는 그래도 괜찮은 것 같아요 ㅋㅋㅋ  그래도 집안일은 정말 힘들죠 ㅠㅠ'], ['먼저 움직이는 사람이 지는거죠 ㅠ'], ['할때 같이하면 조은데 그맘 충분히 이해되요'], ['저희는 반대;; 전 쇼파에 누우면 안움직이는데 신랑은 청소기 돌리고 물걸레돌리고 세탁기돌리고 출근할때 분리수거하고.. 가끔 미안하네요 ㅠㅠㅠ 그래서 화장실물때청소는 한달에 한번정도 제가해요;;'], ['같이 청소하자고 해요 ㅎㅎ 화장실이 근대 제일 문제'], ['네 저도 들었습니다만...  집안일(특히 정리정돈 및 청소)에 한해서는...\n비교적 더 둔하고, 개의치 않는 타입... 그리고 구태여 비교하면 좀더 지저분한 것을 잘 견디는 이가 승리한다고 들었습니다...\nㅠㅠㅠㅠㅠ 티안나게, 어떤건 티 엄청나게도 힘든 집안일 ㅜ_ㅜ 하'], ['그냥 못참겠더라도 참으세요~ㅠㅠ.. 신랑분이랑 신부님이랑 스타일이 다른거니까.. 그냥 이따 하겠지.. 하고 넘겨요!!'], ['ㅠㅠㅠ 저 원래 그런 타입인데... 어제는 정말 제안의 진짜 제가 참을수 없었나봐요 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 난 이 더러운 집에서 살고싶지 않다고 외쳤어요 ㅠㅠㅠㅠㅠㅠ'], ['저희도 둘다 맞벌이라 걱정이에요 ㅠㅠ 락스청소....진짜 남자들은 몰라요 ㅠㅠ'], ['요즘같을때... 서로 같이 도우면서 하셔야죠'], ['부지런한 분들이 먼저, 더 많이 하게된다하더라구요;;;'], ['맞벌이면 같이해야하는데 왠지 여자가 더 하느 느낌적인 느낌ㅠ'], ['할일을 정확히 정해서 하세요'], ['제가 그래요!! 약간 사서 고생하는 스타일? 남편 시켜도 되는데 깨끗하게 맘에 쏙 들게 안하니 다 제가 하게되는... 언제쯤 이런 성격을 내려놓을지ㅠㅠ 흑'], ['ㅠㅠ다그런거같아요 예비 신랑집 가끔 가는데 치우질 않으니.... 눈에 거슬리고..'], ['참는 사람이 승리라고 하던데..뭔가 구역을 좀 정해서 하시는게 좋을 것 같아요! 괜히 감정 상하면 안되니까요~']]
    
    4112
    다들 최애 집안일은 뭔가요? ㅎㅎ 전 화장실 청소요 ㅋㅋㅋ하고나면 그렇게 기분 좋을 수가 없네요 ㅋㅋㅋㅋ집안일 중 제일 먼저 하는게 장실이에요 ㅋㅋㅋ이제 청소 하러 가야겟어요😃
    
    [['설겆이 및 부엌청소요ㅎㅎ 깔끔해지면 기분 좋아요ㅎㅎ'], ['전 설거지가 제일 싫어요 ㅋㅋ'], ['화장실청소가 젤싫은데ㅠㅜ 집안일은 다싫다는게 함정이네요ㅋㅋㅋ'], ['정답이에요 다 싫죠 사실 ㅋ'], ['전.. 다시러요 ㅠㅠㅠ ㅋㅋㅋㅋㅋㅋ'], ['그것이 명답입니다 ㅋㅋ'], ['이불정리하고 청소기돌리는거요\n먼지 싹 걷힌느낌 좋아요'], ['ㅋㅋㅋ아 청소기도 짱 귀찮아요ㅜㅋ'], ['헉 화장실청소는 진짜 하기싫은 청소인데.. 저는 주방청소요^^ 가스레인지 주변 기름때 닦아내면 속이 다 시원해요~'], ['보통 장실은 싫어하시더라고요 ㅎ'], ['전 거실방청소후 공청돌리는거요ㅎㅎ\n공기가 달라요ㅋㅋ'], ['ㅎㅎㅎ 맞아요 청소한 후에 뭔가 쾌적 ㅎ'], ['최애라신깐...ㅎㅎㅎ 강아지 소변 치우기요?\n요즘 우리집 할매가 거실매트에 쉬아를 잘하시네요ㅎ'], ['또잉 ㅋㅋ 배변패드 갈기 입니까 ㅎ'], ['빨래가 좋아요 ㅎㅎ'], ['세탁기가 다하지요 ㅎ'], ['저도 좋은게없어요 집안일은 ㅠㅠㅋㅋㅋㅋ'], ['그건 그래요 ㅋ 널부러지고 싶죠'], ['ㅋㅋㅋ화장대 정리.. 전 개판입니덩'], ['저는 빨래가 세상좋네요~~\n그반대는 요똥이에요 ㅠㅠ'], ['요똥이 모에요?'], ['요리...못하는 요리똥손이요 ㅠㅠ'], ['아.. 저도 뭐 ㅋ 요즘 시판 음식 잘나와요 ㅋㅋㅋㅋㅋㅋ'], ['냉장고ㅜ 노하우 좀요.. ㅋㅋㅋ'], ['저는 그게 왜이리 귀찮을까요 ㅜ ㅜ'], ['씽크대 닦는거요.ㅋ 그렇게 후련해요..'], ['ㅋㅋㅋㅋㅋ 설거지 후 뜨신 물 들이 부움 깨끗해지는 기분나죠'], ['저는 청소 못해요 ㅠ\n냄새에 예민해서~'], ['와 그럼 신랑분이*.*?'], ['네 제가 원하는건 그것뿐이지요 ㅎㅎㅎ'], ['좋은데요? 그래도 신랑분이 이해해주고 해주시니 ㅎ'], ['최애 집안일이란건 있을 수가 없습니다... 하하핫'], ['그건 그렇... 지만 그래도 하나.. ㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋ설거지 젤 싫어요 흐규'], ['최애 집안일이란것도 있나요??ㅋㅋㅋㅋ 전 그런거 없어요ㅠㅜ 청소의 ㅡ도 싫습니다ㅋ'], ['ㅋㅋㅋ다들 청소는 싫다로 대동 단결 중이네요 ㅋ'], ['기분 좋아지는게 정말 없어요ㅋㅋ 하고싶어 하는게 아니라서 그런것같아요ㅋㅋㅋㅋㅋ'], ['그건 그래요 ㅋ 청소하고 싶어 하는 사람이 몇이나 되겟어요ㅜㅋ'], ['저도요ㅋ화장실바닥줄눈깨끗한거보면 세상기분좋아용'], ['맞아요 ㅋㅋ 청소 후 바닥 맨발로 디딜 때 그 깨끗함이 느껴지죠 ㅋ'], ['요리요ㅎ 이것도 집안일에 포함되려나요'], ['요리금손 부럽습니다ㅜㅎ'], ['요리를 잘하는게 아니고 청소,정리를 잘못해요ㅎ 일 벌리는것만 잘해요'], ['저도 정리 진짜 ㅜ ㅜ 정리 수납 한번 받고픕니다 ㅋㅋ'], ['저도요ㅎㅎ'], ['전 다시러요~ㅋㅋ'], ['이게 제일 명답이긴 합니다 ㅋㅋㅋ'], ['댓글을 차근차근 읽으며 난 무엇일까 아무리 생각해도 최애는 없음요..또르르ㅡㅜ  그래도 수납같은 곳 여기저기 정리정돈 이런거는 좋아해요ㅋㅋ'], ['ㅌㅋㅋㅋ그 정리정돈 노하우 좀 나눠주세요'], ['전 빨래널기요\n나머진다싫어요ㅜㅜ'], ['빨래널기요??? 그거 정말 귀찮던데ㅜ ㅜ 부지런 하십니다^^'], ['화장실 세제 추천좀 해주시겠습니까^^\n저도 열심히 해보고싶네요^^;;;'], ['저는 홈스타 초록색 써요 ㅋㅋ'], ['그나마고른다면 빨래가젤... ㅋㅋ'], ['화장실청소젤시른데 대단하셔요 ㅋㅋ'], ['ㅋㅋㅋㅋ이상하게 장실 청소가 젤 좋아요'], ['설거지가 제일싫어요 먹는건좋은데 ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋ 먹는건 언제나 옳아요'], ['전 다 귀차나요 ㅠ.. 이거 집안일 암것도 안하고 싶은 뇨자라네요'], ['그게 제일 좋죠 ㅜ ㅜ 청소는 귀찮아요'], ['청소가 제일 고비인듯해요.. \n대충대충 하다가 한번뒤집으면 하루다지나가고 ㅠ 혼자 일하면서 하려니 너무  귀차는.. ㅠ'], ['워킹맘이면 시간적 여유가 더 없죠 ㅜ 암요 이해해요'], ['시간여유ㅠ 에휴 ~~'], ['전 가스렌지 청소요ㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋ아스토니쉬 좋더라고요'], ['저도 월요일에 바짝하고 나머진 설렁설렁해요 ㅎ'], ['택배뜯기,\n사온반찬 통에담기요\nㅋ'], ['그건 저도 잘할 수 있을 것 같아요 ㅋㅋㅋㅋㅋㅋ 택배는 사랑임당'], ['싱크대 부엌 설겆이후요~~'], ['전 부엌일이 젤 싫더라고요 ㅋㅋ ㅜㅜ']]
    
    4174
    집안일 힘드네요.ㅜ.ㅜ 늦잠자고 일어나 냉면 만들어 먹고 강아지 산책후 목욕시키기, 이불빨래,대청소 휴~~힘들다.3일전 청소를 너무 무리해서 아직도 허리가 아픈데. . . ㅠ. ㅠ어쨋든 시간은 갑니다.ㅋㅋ두브로브니크 반예비치에서 썬텐도 하고 여행을 즐기는 두모녀를 위해 이 한몸 기꺼이 바치겠습니다.ps. 여긴 중국인관광객이 없답니다. ㅋㅋ
    
    [['우와~~~부라덜 혼자ㅜㅜ\n청소힘들어요ㅎㅎ\n사진 너무이뻐요^^'], ['허리 부러진거 같아요.ㅠ.ㅠ\n지금 열무비빔냉면 만들어서 먹고 있어요.  아까 면을 너무 많이 삶아서 남은거 처리중.ㅜ.ㅜ'], ['설마 청소한다고 허리가 ㅜㅜ ㅋㅋ비빔냉면\n저 서울가서 삼겹살에 김치찌개 먹을거예요 ㅋㄱ'], ['며칠째 허리가 안펴져요.ㅠ.ㅠ'], ['ㅋㅋㅋ집안일이 그리 힘들어요 ㅎㅎ'], ['살려줘유~~'], ['저도 살려주세요~~~~~비행기좀 태워줘요'], ['낼 타믄 되는데 뭘 걱정이유.ㅋ'], ['지금가고싶다구요~~~~~~~'], ['워째유?  내사랑 보라카이를 이리 시러하다니.ㅜ.ㅜ'], ['아니 보라카이를 왜 좋다하는지 이해가안돼요'], ['내맘이쥬.  좋은데 어째유.ㅋㅋ'], ['ㅋㅋㅋ알았어유 ㅋㅋㅋ'], ['집안일이 힘들어요ㅠ'], ['저는 청소를 너무 완벽하게 해야돼서 고생을 합니다.ㅜ.ㅜ'], ['일부러 여러번 안하시려구요??\n적당히 자주 하시는게 이롭습니다ㅎㅎ'], ['저는 원래 하루종일 청소하는 강박증 환자입니다.ㅜ.ㅜ'], ['집안일은 계속 하는거라 강약조절을 잘하셔야 해요 그래서 티도 안나고 힘만 든다고 하자나요'], ['저는 누가 보든 안하든 항상 합니다.  날 위해서. 그래야 기분이 좋아요.'], ['골병 들어유~ 슬슬 하세요'], ['저랑 살면 피곤하겠쥬?ㅋ'], ['저라면 살림을 놓겠습니다ㅋㅋㅋ'], ['그래서 저는 입 다물고 참고   삽니다. 머리카락 줍고 다녀요'], ['맘에 안드시죠?ㅎㅎ\n그래도 참으셔야 합니다'], ['그래서 그냥 더러운 돼지우리속에 돼지로 살아요.ㅋ 나만 깨끗하믄 뮈해유'], ['돼지우리ㅋㅋ 그래도 청소 매일하는건 힘들어요ㅠ'], ['군대에선 하루종일 청소해요'], ['하... 군대에 평생 사는거 아니자나요ㅋㅋ\n글고 청년 시절과 같나요~'], ['남들은 비오는 날 차를 타고 가죠?  난 꺼꾸로 차가 드러워지는게 싫어서 대중교통을 탑니다'], ['엇 울남편도 차 비맞히는거 안 좋아해요ㅎㅎ'], ['아 드러~~전 비 안맞게 주자장에 고이 모셔놓는다구요'], ['네 즤신랑이 그런다니까요'], ['하여간 드러운건 싫어요'], ['차만 그런게 아니시자나요\n그럼 주변 사람들이 힘든데ㅠ'], ['사진만  보내주는군요~~^^'], ['사진도 보내주고 톡도 하고 그러쥬.  근데 그시간이 꼭 새벽이유.ㅜ.ㅜ'], ['ㅋㅋ 아부지에 대한 배려가 읍네요~~~^^'], ['시차가 안맞으니까 당연하쥬.ㅋ'], ['옴마 애기궁둥이 이쁘네요 ㅋㅋ시선강탈'], ['풉~ 옆에 더 큰 방뎅이도 있는데. . .'], ['ㅋㅋ아우 귀여운것만봣어요 ㅋㅋ'], ['잘보면 울딸도 있어유'], ['앗그럼다시보러..ㅋㅋ'], ['앗 혹시 형광노랑 비키니요?'], ['동양인 한명밖에 없잖아유.ㅋㅋ'], ['ㅋㅋ까만비키니분도 뒷모습은..음 ㅋㅋㅋ'], ['아무래도 첨에 찾은게 맞을듯.ㅋ'], ['히히 정답이네유 ㅎㅎ'], [''], ['옴마~ 아름다운곳이네요 바다색이 예술 ㅡ 집안일은 힘든데 끝이 없다는거 ㅎㅎㅎ'], ['바다 예쁘죠^^ 저기서만 있구 딴덴 안가구 싶어요.ㅋ\n강아지 저녁주고 저도 놀러갑니다'], ['일하셨으니 놀아야죠 ㅋㅋ'], ['너무 멀어서 그런가 중국사람 안보인데요.ㅋ \n친구 만나러 구리 동태탕 갑니다'], ['안돼유.  조신하게.ㅋㅋ'], ['친구랑 약속장소가 구리.ㅋ'], ['지금 구리 가고 있어유.ㅋ'], ['삐돌이.ㅋ'], ['나 어려울텐데 ㅋㅋ'], ['지금 인났슈.  아 물만 먹고 해장국 먹구싶당'], ['씻고 바지락칼국수 먹을라고 하는데 목욕탕에 내려갔다 오기가 귀찮구. . 다시 누워서 강아지랑 멍때리기 해요'], ['청소까지👏👏👏\n쉬엄쉬엄 하세요;;티도 안나는 집안일인데'], ['완전 티나게 바닥이 반들반들하게 해놨어요^^  강아지도 깨끗하게.  이불도 깨끗하게'], ['어직 오실람 멀었잖아요ㅋㅋ'], ['저혼자 살아도 깨끗하게 살아야쥬^^'], ['ㅋㅋㅋ그렇긴 한데 집안일 넘 힘들잖아요;;'], ['너무  아름다운곳이네요~~~~^^'], ['지금이 정오라 오후엔 버스타고 산에 다녀온대요. 일몰본다구. 세상은 넓고 아름다운 곳이 많은가 봅니다.  이도시 성벽이 천년이 넘었다고 하니 대단하죠^^'], ['우와 너무 예뻐요 >♡< 콩심님 힘내세유'], ['자다 인나서 배고파유.ㅡㅡ'], ['와~\n여기 넘 좋아요~^^'], ['산이랑 나중에 가세요^^\n근데 산이가 배신때리고 여친이랑 갈수도. . .ㅋㅋ'], ['에휴~\n그럴수두요~ ㅎㅎㅎ\n제가 그 여친 여행비까지 대면~ ㅎㅎㅎㅎㅎㅎ'], ['와 너무 너무 이쁘네요..!혼자서도 깔끔히 ㅎㅎ 일등 신랑감이셔요'], ['제가 그전부터 결혼정보업체에서 등급이 아주 높았습니다.ㅋㅋㅋ'], ['앜ㅋㅋㅋ 확인할수없는 정보는 허위정보로 간주하겠습니다 ㅎ'], ['움마 왜 안믿어유?ㅋ\n'], ['아참참 !!!\n김밥에서 높은 점수를 받으셨을 거라 생각들어 인정하도록하겠습니다~~~~'], ['그냥 다 이쁘네요~~^^'], ['애기 엉덩이가 특히 이쁘죠?ㅋ'], ['네 눈에 들어 왔어요~~^^'], ['비키니가 울딸입니다.ㅋ'], ['따뜸 자랑하는거죠ㅎㅎ\n비키니 잘 어울리네요~~~'], ['ㅎㅎ 난 팔불출인가봐요']]
    
    4178
    집안일 중 제일 하기싫은것은??? 전 설거지요~~~~~다른건 다하겠는데 물에담궈둔 설거지는 왜케하기싫은지...오늘도 미루고미루다 지금다했네요~~하원하면 도시락 물통에 빨대까지 하느라 진빠져요~~~~애솔맘들은 뭐가제일 싫은가요~~~??^^
    
    [['맞아요..,건조기사기전 너무귀찮았죠...지금은 또 개는게 귀찮네요 ㅎㅎㅎ'], ['전 걸레질이요ㅜㅜ\n너무귀찮은 ㅋㅋㅋ'], ['물걸레청소기 추천합니다~~^^'], ['요리요. 남이해주는 밥이 제일 맛나요ㅋ'], ['어머 답글보다보니 하기싫은게 많았어요~~~요리 저도 꽝손이여서 ㅠㅠ'], ['빨래 개켜서 옷장에 넣는거\n방닦은 걸레 빠는거요 ~'], ['꺅 걸레빠는거 한몫하죠~~~~'], ['음식물 쓰레기 버리는거요...'], ['여름이니 더 자주버려야해서 귀찮죠ㅠ'], ['전 바닥닦는거랑 빨래개서 그 자리까지 넣는거요 ㅠㅠㅠ'], ['빨래갤거 저~~~기 두고 애솔중입니다ㅎㅎ'], ['저는 빨래 개고 넣는거요 ㅋㅋㅋㅋㅋㅋ 옷 정리는 엊그제 한거같은데 돌아서면 또 엉망이고 진짜 누가 해줬으면 좋겠어요 ㅠㅠ'], ['그니깐요....빨래개는거 특히 애들옷은 작은이 하나하나 일이죵~~~'], ['설거지랑 빨래마른거 정리하는거요...너무 싫어요.....'], ['그쵸,,,그두개세트를 하나하고 마른건 일단 두고보고 있읍니다.,ㅎㅎ'], ['그냥 다요...ㅠㅠ\n해도해도 끝이없고 티가안나는 집안일..ㅠㅠ'], ['오....빙고네요,,,티가안나요ㅠㅠ'], ['전 설거지중에 수저 젓가락 닦기. 빨래 개는거까지는 좋은데 갠빨래 서랍장세 넣기 요게 제일 싫어요 \n지구력이 부족한가봐요 마지막 마무리가 안돼요ㅠ'], ['맞아요 저도 뒷힘이 부족해서 한꺼번에 못하고 쉬다가 나눠해요ㅎㅎ'], ['청소요~ 쓸고 닦는거 넘 싫어요. 나름 정리는 할만한데 청소도구 손대기가 싫다능~ ㅎㅎ'], ['ㅎㅎ저희집은 아담해서 그나마 할만해요 무선물걸레청소기 추천해드립니다!!^^'], ['저도 설거지가 싫어요. 5인식구인데\n건조기샀어도 진즉샀어야했는데 한번 어그러지고 다시 알아보려니ㅠ 잘안되고 매번 밀리네요'], ['그쵸...설거지통보믄 마음이 안비워져요....힝~'], ['저도 설거지요ㅠ 싱크대 뒷정리도 해야하고 건조대도 정리해야하고ㅠ 빈그릇넣어야하고..아..ㅠㅠ시러용'], ['그쵸...설거지이후에뒷처리 못하니 지져분해도 바로 소파로와서 뻗어요~~~~'], ['걸레질하는게 제일 깨끗하긴해요~~~'], ['전 다 싫긴한데 특히 빨래요!! 왜 이리 귀찮은지... 누가 좀 해줌 좋겠어요~'], ['빨래 개는거랑 걸레 빠는거요~~ 건조기 돌리고 하루 있다 꺼낼때 많네요 ㅠㅠ'], ['ㅎㅎㅎ저도요~~~!! 전 이틀까지 그냥 문만열어뒀어요~~~'], ['전빨래 너는거랑 걸레빨기요ㅎ'], ['빨래 개서 서랍찾아 넣는거까지가 너~무 귀찮아요ㅜ 베란다 나가기도 싫어요'], ['빨래개기요 ㅋ 지금해야하는데 널부러져있어요 ㅡㅡ'], ['화장실청소요 허리아파요ㅜ'], ['저도빨개널기가젤시러요ㅋㅋ풉'], ['빨래개기요\n넘나싫은거ㅜㅜ'], ['빨래요ㅜㅜ\n널고개기 진짜 싫어요 흐어엉'], ['전 요리요.. 덥고 냄새나고 설거지 생겨서 싫은데, 젤 싫은건 힘들게 했는데도 맛이 없어서요..엉엉ㅠㅠ'], ['찌찌뽕~~~~맛없음에 길들이고 있답니다~~^^'], ['저는 화장실청소요...안하면 티 팍팍나고 그나마 하면 정상으로 보여요..화장실청소한 담에는 쓰기 아깝다는...ㅋㅋㅋ'], ['설거지랑 분리수거 하는거요.진짜진짜 귀찮지만 빨리 안할수 없는거라 짜증내며 합니다.~ ㅎ'], ['저는 빨래요~ 널고 걷고를 반복 ㅠㅠ싫어요ㅠㅠ'], ['ㅎㅎㅎ맞아요~~건조기사서 신세계였는데 또 이렇게 싫어지는게늘더라구용'], ['설거지요.. 쌓아뒀다가 남편 퇴근하면 부탁할때 많아요..;;'], ['저도 가끔 그걸 노린답니다~~~너무힘든날은 걍쌓아둬요~~~^^;;;'], ['전부다요..날 더워지니 파업하고 싶어요ㅠㅜ'], ['ㅎㅎㅎㅎ여름엔 방학이 필요합니다~~주부방학 ~~~'], ['걸레빨기 다림질(요건 스팀다리미 덕에 조금.편해졌어어요)반찬 만들기T T'], ['다림질,,,덥죠...전 다리미가 아예없다가 저번주에 샀어용ㅠ'], ['다요~ 몽땅 다요 ㅠㅠ'], ['집안일!\n다하기시러요!'], ['청소기 돌리기요  요즘같이 더운날. 무거운 청소기 돌리면.  뜨거운 바람때문에. 온몸이 후끈후끈 ㅠ.   화장실 청소도 싫고.   생각해보니 다 싫네요 ㅋ'], ['저는 요리요...여름에는 더워서 봄가을엔 날씨가 좋아서 겨울엔 실내공기를 위해..\n하기 싫어욬ㅋㅋㅋㅋ'], ['손걸레질이요 팔 아파 죽겠어요ㅜ'], ['빨래 개는거요..ㅋㅋㅋ'], ['음... 오늘 저녁 뭐먹지?? 고민하는거요 ㅋㅋㅋㅋ'], ['전 걸레질이요~\n진짜 걸레질 한번하려면  큰맘먹고 하루날 잡아서 하고있네요 ㅠㅠ'], ['청소..그 중에서도 화장실 청소요 ㅠㅠ'], ['걸레빨기 싫어서 걸레질 하기 시러요ㅡㅡ'], ['전 실내화 빠는게 그렇게 귀찮고 싫어요ㅠ하면 금방하는데 하기전에 왜이렇게 귀찮고 싫은지 모르겠어요;;ㅠ'], ['걸레질요...ㅠㅜ방바닥..'], ['화장실청소요.\n\n머리카락 정리하는게 너무 힘들어요'], ['걸레빨고 걸레질하기요~~~~~ 아..진심 귀찮아요~ 요즘은 더더욱 다용도실 들어가서 걸레빨기 넘나 싫어요~'], ['개어 놓은 빨래 서랍에 정리하는거요 ㅠㅠ'], ['저랑 비슷한 분들 많으셔서 반갑네요. 저는 빨래 널기 & 개기가 그렇게 귀찮아서 아들들 출동 시키곤 합니다^^'], ['밥'], ['전 청소요 화장실 집청소 해도해도 티가안나요'], ['저도 요리랑 걸레질이요\n이 두개만 누가 해줬음 좋겠네요'], ['식판.물병씻기랑 옷 개는거요ㅠ쥬'], ['식판...누가좀 해줬으면 좋겠어요ㅎㅎㅎ'], ['음식물쓰레기 버리기는 영~~~냄새도 나고 엘리베이터타면 옆사람 눈치도 보이고~~ㅜㅜ'], ['그걸 어떻게 하나만 고르죠?ㅜㅜ\n제일 하기 시른거.. 집안일 한개요...'], ['애기 젖병닦는게 젤 시러요 ㅜㅠ'], ['싱크개수대 음식물처리하는거요~~~깔끔하게 털기 어렵고 만지기도 싫어요 엉엉ㅠ'], ['밥 하는거요 .정말하기싫어요 ㅠ'], ['빨래개서 서랍에 넣는거요 ... 진짜 너무싫어요 ㅠㅠ'], ['전 걸레질이랑 다림질이요 ^^'], ['손가락 까딱도 하기 싫어요...\n넘 더워서 최소한의 움직임으로 살고프네요...\n어제 저녁엔 남의편이 애덜 하고 작당해서 고기 구워먹자고....에어컨 켜고 고기 구워 먹었네요...집안일 안하고 살고파요~~~~~'], ['맘들 다들 저랑 비슷한 생각들이시네요^^\n그중에서 설것이랑 빨래 너는 것 제일 싫어요  ㅎㅎㅎ\n너무나 단순 노동이에요  시간도 많이 걸리고요']]
    
    4187
    집안일 끄읕  ㅡㅡㅋ 아침에 다시물내고 빡빡장 비스무리하게 짭짤하게 찌지고이리 찌지나놔  서울가기전꺄지 영감묵고  서울도 가꼬가고 ㅎㅎ 제사두부도 묵고싶어가ㅡㅋ제사두부형식 꿉고 ㅎㅎ김밥한줄 냉장고 처박아논것도 꿊어는데사진 못찍엇슴ㅠ1층들어오는 현관 밀데로 청소다하고 거실방ㅡㅡ밴소까지 청소끋수건은 어제 삶앗응께이제 오후에 미장원 예약기다리는것만 남앗네요어제  못본 김밥남녀나보고 라디오나 듣고시간 때아야겟네요오늘 하루도 화이팅하세요
    
    [['부지런하셔요 ㅎㅎㅎ'], ['아침잠이 읍어가ㅡㅋ\n고맙습니다'], ['아침부터 일을많이했네요^^\n좋은하루되세요^^'], ['고맙습니다 \n잠이 읍어가 일찌감치 일어납니데에'], ['부지런하십니다 상쾌한아침이겠어요 빡빡장?에 가지 감자 양파 고추넣으신거에요?^^'], ['가지가 고지혈 저지혈  당뇨좋타고 하는데\n문제는 짭게 묵으먼 안되는데 ㅎㅎ'], ['와...부지런하시네요\n이른아침부터 고생하셨어요ㅎ\n근데 빡빡장이뭔가요^^;;;;ㅎㅎ'], ['된장인데\n쩨작해게  찌지는거요\n그걸빡빡장이라고 합니다\n근데 오늘은 국물잇꾸로햇어요\n제가고지혈이 쫌심햇가 넘짭꾸로 하믄ㅇ그래서ㅠ'], ['존경스럽습니다!!!'], ['아이고마ㅠ\n부끄럽꾸로 ㅎㅎ\n아침짬이 읍어가 그럿습니데이'], ['누가 아프길래??\n병원에서 밥묵으면 ㅠ\n더럽게 맛읍긴읍죠ㅠ'], ['아따마ㅠㅠ\n두배로 고생이겟네예ㅠㅠ\n얼릉나수세요ㅠ\n알라들ㅠ\n요즘 알라들 아픙기 유행이라서ㅠ\n큰일이긴큰일이에요ㅠㅠ\n엄마도 몸조리 잘하고\n빡빡 장쫌갇다주까요?\n어느병원에 잇능교??'], ['내 5시 쯤 미장원 예약해서 나가는데\n그때갓다줄께요\n양덕동 집아닝가요??\n몬순이 엄마가 그리말한거 가튼데ㅡㅋ'], ['말 아닌데 ㅠ\n중리서 \n진해머리하러 가니깐 갈때 후딱 던져주고가믄되요ㅋㅋ'], ['몬순이 엄마는 우리집와서 머리도깜고가유 ㅋㅋ\n머 그가꼬 그래쌋능교 ㅎㅎ\n일단 그럼담에 보는걸로 패스 ㅎㅎ'], ['후딱 알라들이나 나수고 빨리 퇴원시키세요'], ['전 영감 밥차리드리고 \n휴식중ㅡㅋ\n\n짜바서 2그릇이죠 ㅠㅠ'], [''], ['껄쭉하게 해야  하는데 \n고지혈때메 국물잇꾸로 햇네예 ㅎㅎ'], ['빡빡장 맛있겠어요'], ['짜바에\n근데 뒷맛은 딸딸한맛도 나네요 ㅎㅎ'], ['아침에 일 많이 했네요'], ['그러게요 ㅎㅎ\n잠이 읍따보이 가끔 이럽니다 ㅎㅎ\n'], ['체력이 좋으신가보네요'], ['그런건 아니고요 ㅎㅎ\n어릴때부터 아침잠이 읍어요 \n이상하게도\n근데 문제는 초저녁에나는  병든  병아리죠 ㅡㅋㅋ'], ['난 덩치는 산적인데 50  넘으니 힘이 딸리고 메가리가 없어요'], ['저도 산적입니다ㅡ\n근데 알라때부터 잠이 읍엇어요\n근데 내동생은  알라때부터 잠이 많앗죠 ㅠ\n저랑늘반대ㅡㅋㅋ\n50 넘엇으면 무슨띠인가요??\n저보다 나이가 많으신분 거의 못본거 같은디 \n언니인거 갇네오\n양.닭띠?'], ['아침부터 진짜로  부지런합니다~~^^!!!'], ['잠이 읍어가 그럽습니다 ㅎㅎ\n초저녁잠이 많아가 문제지만 ㅋㅋ'], ['양띠면  얼마나 좋깃슈~말띠여~'], ['알라들 다키앗겟네요ㅡㅋ\n말띠믄 ㅎㅎ\n언니는 어디사능교??'], ['미용실은 어디로 다녀요?'], ['진해 자은동 다닙니다\n삼계서 내머리 마끼믄\n빨깐머리앤  만들어삡니다ㅠ\n전에 잘하는 미용사쌤이 어디가고 난뒤 아무리 돈 많이 두고 잘한다고 해도 안되요\n빨간대가리만들어 사서ㅠ\n진해까지갑니다\n이제 1년된는데\n진해갓는지\n갈색머리 되어갑니다ㅡㅋ'], ['양덕'], ['저도 태어난곳은  서울대학교병원 \n자란곳은  부산 \n학교는 울산서 \n결혼은 부산서 생활하다가 \n남편 직장땜에 17년전에 창원왓어요ㅡㅋ\n첨에 왓을때\n백화점 코딱지 만해서  울고 ㅡㅋㅋ\n\n근데 지금은 살아보니 마산만큼좋은동네 읍네요\n부산집 다팔고 \n중리에 터 내랏삣네예 \n이제 부산에 짚안식구들잇어도 \n마산이 더좋아요\n놀러가기 좋고  ㅎㅎ\n차안막히고\n사람좋고 ㅎㅎ\n물가사고\n언니도 정붇혀보세오\n그라고\n학군약하고 ㅎㅎ'], ['타지방 살다가 이사온지  몇달 되었네요'], ['ㅋㅋㅋ'], [''], ['아침부터 부지런히 맹그셨네요ㅎ 저도 두부 저리 꾸버난거조아하는디ㅎ 먹고잡네용ㅋ'], ['일찍일어나가  ㅡㅡ\n지금은 라디오듣고잇는데\n휴직중입니다'], ['제사두부는 맛읍는데\n무다이 저런두부묵고싶을때가잇어요\n그럴때  꾸버서 묵어예'], ['벌써부터  고지혈이믄 우야노ㅠ\n큰일이네ㅠ\n나도 일찍 약묵엇지만서도ㅠㅜ\n난 묵은거  바로 설거지ㅠ\n안하믄 이상혀ㅠㅠ'], ['할마씨가 머꼬ㅠ\n잠이 읍다 아침잠이  ㅋㅋ'], ['할일 업으가 라디오 듣는다해사꼬 윽시 바빴는데예??  요리에  청소에 이 믑니꽈?~~~~~~~~'], ['머시기 이리 오래걸리노\n대도 꾸무덕데네ㅎㅎ'], ['대도  꾸무덕데네 ㅋ ㅋ ㅋ ㅋㅋ ㅋㅋ ㅋㅋㅋ  ㅋ  ㅋ ㄲ'], ['빡빡장,   미장원, ㅋ ㅋ ㅋ ㅋ ㅋ ㅋ ㅋㅋ ㅋ ㅋ ㅋ ㅋ ㅋㄱㄱ'], ['김밥남녀 아이고  검법남녀 아인교?'], ['내가 김밥 남녀라캣나ㅡㅋ\n기냥패스햐ㅡㅡ\n꼭 찝어내내 \n나 무바라 손가락 굵어가 글도 잘안치진다아이가ㅠㅠ'], ['나 무바라 ㅋㄱㄱ ㅋ ㅋㅋ  즈짜게 글 적은거 보이소'], ['안빈다ㅠ\n저까지 만다꼬 올리 볼끼고ㅠ\n눈깔아프구로 ㅠ'], ['빡빡장 호박잎싸묵고 밥비벼묵고싶ㅇ네요 아 배고파라~~~~']]
    
    4257
    🍨집안일 끝내놓고 나가요.. 샌드위치메이커겁나 앗뜨거 앗뜨거해서열좀 식으라고뚜껑열어 놓고 변신하러 슝떠나봅니다
    
    [['변신?어딜 가시나요?머리하러가요?아님 놀러?ㅋ비소식있으니 우산챙기셔요'], ['할머니에서 아줌마로 변신해볼까싶으네요 새치염색이요ㅜㅜ'], ['아 새치가 많은가봐요ㅎ저도 요즘 부쩍 군데군데 흰머리가 많이 보여서 뽑긴한데ㅡㅡ'], ['이건 뽑다가는 아마 대머리될지 싶어요 흰머리카락 절대 뽑지말래요'], ['전 그게 안되네요 눈에 보이니 뽑아야 기분이 좋다고나 할까ㅡㅡ완전 미침요'], ['머리숱 걱정없으시면 속시원하게 뽑으면 되죠 내 기분이 좋아진다는데 망설일필요있나요 뭐'], ['변신까지 오늘 어디가시나용?\n주방 넘 깨끗하게 정리해놓으셧네용 ^ㅡ^'], ['새치가 거의 머리를 다 덮어버려 할머니에서 아줌마로 변신하러 나왔어요 예약하고 가는거라 아침부터 바빴네요'], ['뿌염하시러 가셧군요 저도 요새 쪼매씩 비네요 ㅠ.ㅜ 뽑느데 한계가잇네용'], ['뿌염하면 얼룩덜룩할까봐 아예 전체염색하고 왔는데 흰머리카락 안보이니 속은 후련하네요'], ['저도 한번은 해야될꺼같은데 ㅠ.ㅜ 무슨색으로 염색하셧어용?'], ['색깔 그런건 모르겠고 가장 기본적인 갈색으로 했어요 미용실가면 거의 사장님 의견에  따르게 되네요'], ['핫뜨거~~ 라이언 열일하고 쉬는군요 \n이쁘게 머리하고 오세여'], ['10개월만에 가는데도  늘 가던곳이라 그런지  알아봐주시네요 늘 하던 새치염치하러왔네요'], ['네~^^ 이쁘게 하시고요 머리도 다듬으시고요 변신하십시요😆'], ['돈 안들고 예뻐지긴 쉽지않지요  그래도 할머니에서 겨우 아줌마로 변신완료했네요'], ['변신하니 아이들이 알아봐주지요? 오늘도 어디 가시는가요?'], ['그나마 딸래미는 알아봐주더라구요  딱히 선약도 볼일도 없고 뭐 그렇네요'], ['변신하시고 어디 가시나요\n낭군님과 점심 데이트 가시나요\n조심히  잘 다녀오셔요'], ['새치염색 하러 미용실왔어요 염색후 예쁜 우리 똥강아지들 데리러가야지요'], ['아하 예쁘게 하러가셨군요  지금은  예쁜 아가들 만나셨겠네요'], ['3시40분쯤에 데리러갔네요 역시나 애들은 엄마 머리한거 전혀 모르더라구요'], ['깔끔히 정리하시고 어디가시나 햇는데 댓글보니 머리하러 가시나용\n  이쁘게 하고 오셔요~~'], ['어제 급 미용실 예약해서 아침부터 부랴부랴 움직였네요   맘님 만나기 이틀전이네요ㅋㅋ'], ['히힛 이쁘게 염색하셧나용 \n  저는 수유끝나고 젤 먼저 염색한고같아요\n  그쵸 이틀 ㅋ 이틀뒤를 위해 오늘은 차분히 청소하고 쉬고잇오요~'], ['아직 약 바르고 대기중입니다 새치 커버가 잘 나와야할테데말입니다 첫인상이 중요하지요ㅋㅋ'], ['잘나오실고에용  그래두 미용실 가는건 기분전환에 좋더라구용\n  ㅋㅋ  저는 그럼 낼부터 ㅋㅋ 아주 잠시 소식을하고 뵐게용~'], ['머리카락끝 상한거 다  잘라네니 염색보다 컷트가 더 기분전환이 되었어요'], ['어디 예쁘게 하러가시는걸까용?\n오늘도 샌드위치 메이커 이용하셨군요 탐납니다ㅎㅎ'], ['예쁘게 해야할텐데 제가 고작 할수있는건 새치 염색 뿐이네요'], ['아항 염색하러가셨군요 염색만해도 분위기 확달라지더라구용'], ['저는 새치염색이라 그런지 딱히 달라진건 모르겠는데 상한거 다 정리하니 머리가 가뿐하긴하네요'], ['상콤하게 변신하셨네용~ 새치정리만 좀 하고 머리카락만 다듬어도 새로워보이더라구요'], ['파마를 해줘야 좀 사람들도 알아보고 그럴텐데 미용실가면 돈이 너무 깨지네요ㅜㅜ'], ['변신이라~~ 미용실 가시는거에요?\n이리 정리 다 끝내고 외출 넘 좋습니다^^'], ['미용실 예약을 해둔 상태라 부랴부랴 정리할것만 대충하고 미용실 도착했지요'], ['저두 주말에 식구들 미용실로 총 출동할려구요~ 남자들은 미용실 넘 자주가요~ 그렇다고 기를수도 없고ㅋㅋ'], ['저도 8월되면 또 아들 컷트하러가야하네요 일년치 계산해보면  여자들 한번에 미용실에 쓰는돈보단 저렴하긴하네요'], ['염색하러 가는거?? 암튼 이쁘게 변신하길 라이언은 열일하고 쉬는구만 ㅋㅋ'], ['할머니에서 아줌마로 변신할려구요 도저히 내가 눈뜨고 못보겠는거있죠ㅜㅜ 라이언땜시 토스트기가 파업들어갔네요'], ['ㅎㅎㅎ 오늘 젊은이로 변신했구만요 ㅎㅎ 라이언있으니 토스트기는 찬밥이죠'], ['근데 더 늙어보이는건 기분 탓이겠죠ㅜㅜ머리숱을 얼마나 쳤는지 가볍긴한데 너무 달라붙어 지지네요'], ['ㅎ며칠 지나면 또 적응되지 않을까 싶구요 ㅎㅎ 머리숱 많이 치면 엄청 가볍긴하지요 내머리는 왜 펌이 다풀린거 같을까 ㅜㅜ유'], ['나랑같이 셋팅 다시 말러갑시다ㅎㅎ나도 파마가 넘 하고픕니다'], ['앗뜨거 라이언 좀 쉬고있으라요...\n밖에 흐리긴한데 많이 꿉꿉하지는 않아요'], ['생각했던건 보단 덜 꿉꿉합니다 아침에 빗방울이 쏟아지더니 소나기였나봐요'], ['아침에 비 왔었나봐요..\n오늘 걸으니 땀도 많이 나지않고 날이 괜찮더라구요.'], ['진짜 가랑비처럼 찔끔 왔었어요 어제 저녁에 그렇게 쏟아질려고 그랬었나봐요'], ['어제 저녁에 우왕~~ 저 그 퍼붓는 비 한가운데 서 있었어요...\n우산을 쓰고있는데도 우산 찢어질까 겁나더라구요.\n천둥은 또 얼마나 치는지요..'], ['저녁에 무슨일로 나가신거에요?저는 애들 하교하고 오면 왠만해서 나갈일이 없거든요'], ['멋내기는 둘째치고 새치가 너무 자라서 새치염색하러 왔어요 지금 딱 할인행사중이라네요ㅋㅋ'], ['이젠 그냥 흰머리카락만 나오네요ㅜㅜ뽑으면 안된다고 해서 계속 방치했더니 완전 할머니가 따로없었네요'], ['뽑지말고 차라리 뿌리 살려둔채 가위로 싹뚝 잘라야한다네요  뽑다가 탈모올지도 모른다네요'], ['미용실가신거에요?ㅎ 댓글로\n맘님 일정을 스토킹 해봅니다 ㅎ'], ['네네 타카페 협렵업체  미용실 파격 할인행사하길래 겸사겸사 새치염색하러 왔네요'], ['샌드위치 메이커  엄청  뜨겁군요ㅋ \n변신하러 가시나본데 이쁘게 염색하시고 오셔요^^'], ['자칫하면 화상입을 정도로 열기가 대단하더라구요 멋내기 염색은 못하고 오로지 새치 커버 염색하러왔네요'], ['진짜 그정도로 뜨거운가보군요ㅜㅜ\n그럼 애들 있을땐 조심해야겠네요ㅠ\n저는 머리카락도 적은데다 새치가 있어도 염색도 못해요 없는머리 다 빠질까봐요 멋내기가 아니라도 새치 커버 염색만해도 다르지요~~^^'], ['저도 머리숱이 없는편인데 새로 자라나는 머리카락까지 흰머리라 헤어쿠션만으로는 감당이 안되더라구요'], ['미용실가시는건가요~~ 저 어제 가서 머리정리하고 파마 다시했더니 속이 시원합니다^^~ 깔끔하게정리하고오셔요 ㅎㅎ'], ['파마하면 진짜 다른사람으로 변신되는거죠  어떤 파마하신거에요? 저는 셋팅말곤 해본적이 없네요'], ['전 어정쩡하게 긴머리였는데 아예 단발로 짜르고 보니펌? 해봤어요 ㅎㅎㅎ 몽실몽실하니 마음에들어요! 기분전환으로 제격입니다 ㅎㅎ'], ['목덜미가 시원시원하시겠어요 단발 후회는 없으신거죠?'], ['전혀요! 왜 이제했나 싶을정도로.. 마음에 들어서 다행이에요^^~'], ['저도 2년전에 숏커트했었는데 미용실 나온순간은 완전 대만족이었네요']]
    
    4390
    집안일 중 제일 하기 싫은것은?? 기냥 궁금해서요1.빨래 개기2.빨래 널기3.설거지4.청소(화장실청소 포함)5.음식물 쓰레기 버리기6.분리수거7.닦기(각종 선반위 바닥 등등)전 신랑이 5.6분은 전담이라2번 빨래 널기가 이상하게 하기 싫어요설겆이중 아이 빨대물통 씻기도 정말 ㅠ
    
    [['저도 빨대 물통 씻느게 그렇게 귀찮네요.'], ['수세미로 불통 외부 씻고\n빨대솔질 하고\n물통안쪽용 솔로 씻고 하니\n총 도구가 3개 ㅠ'], ['전 설거지가 젤 시러요'], ['그죠^^'], ['전 음쓰요 ㅋ'], ['오호~~'], ['설거지욧!!!!한자리에 계속 서 있는게 너무 싫어요 흑'], ['청소 설겆인 나름 깨긋해지는 맛이라도 호'], ['ㅋㅋㅋ'], ['하하하 정답인듯요ㅋㅋ'], ['빨래 서랍에 넣는거요...'], ['오호 맞아요 제자리에 갖다 넣는거 너무 귀찮음'], ['저도요\n'], ['저두 에브리띵이요. 똥손이라 해도 표시도 안나고, 그나마 저보다 잘하는 남표니가 다 하고 있어요ㅜ'], ['청소 진짜 청소너뮤시러요'], ['다시러용.... .ㅋㅋㅋㅋㅋ하핫.......'], ['전..창틀 닦기ㅎㅎ 그리고..빨래 서랍 넣기.지금도 개놓고만 있네용ㅎ'], ['ㅎㅎㅎ 이건 힘들어요 맞아요'], ['저도 설거지가 싫으네요 ㅎㅎㅎ'], ['건조기가 있으니 이제 빨래 개서 넣는게 젤 싫어요 ㅎㅎ'], ['빨대물통 공감 ㅎ 하기 싫어서 미루고 아침에 씻을 때가 많네요ㅜ'], ['저도 하기 싫어서 요즈음은 맨먼저 해버려요 ㅋㅋㅋ 나중에 미루면 안할까바 ㅎ'], ['설거지싫어 세척기샀더니 세척기에 그릇 넣는것도 귀찮고 \n빨래 너는게 싫어서 건조기샀더니 개는게 귀찮고......\n\n집안일 다 싫어욥 ㅋㅋㅋㅋㅋ'], ['ㅎㅎㅎㅎㅎㅎ아놔 ㅎㅎㅎ'], ['난다림질ㅋㅋ'], ['그런가 안해요 ㅎㅎㅎ'], ['저도 3번요 ㅋ'], ['마른빨래 정리 한거 서랍 지 자리에 넣는거요 ㅎ 이리갓다저리갓다 ㅜㅜㅋ'], ['전 다림질이요 ㅎㅎㅎㅎ'], ['설거지요!!!'], ['순위에없는것중엔요리가젤싫네요~~늘뭐먹을지고민이라그건만대신해주면감사요~~'], ['전 설겆이보단 요리가 좋은듯요'], ['전 4.5.6은 남편이 하고 그외는 제가 해요\n저도 애기 빨대컵과 젖병 씻는게 제일 귀찮네요'], ['4 7번이요    솔직히말하면 점부하기싫어요ㅋ'], ['전 이상하게 빨래 개는건 그렇다치고 정리하기 싫어요 ㅋ'], ['빨대컵ㅋㅋㅋㅋ증말공감용ㅜㅜ'], ['ㅋㅋㅋㅋㅋㅋ'], ['다들 공감이 많이 되어 웃고 넘어갑니다^^'], ['전 1번 빨래개기ㅋ'], ['청소요ㅠㅠ'], ['저도요...빨래널기...빨대씻기..ㅎㅎㅎ주부들의 마음은 다 똑같나봐요'], ['진짜 이렇게 많이 글 달릴줄 ㅋㅋㅋ'], ['전 화장실 청소 ㅠ\n화장실 청소하고나면 한달 청소 다한거 같아요\n좀전에도 땀을 땀을 얼마나 흘리며 화장실 청소했던지\n언능 여름 갔음 좋긋어요'], ['전 은근 화장실 청소 재밋어요 깨긋해지는게 확실히 눈에 보여서 ㅎㅎ'], ['화장실청소요~남의편아 니똥물은 니가좀치아라ㅡㅡ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋ 전 제가 답답해서'], ['1234567 다 하기 싫으네요 ㅋㅋㅋ'], ['화장실 하수구 청소...아닐가요??ㅎㅎ'], ['저도 다 싫어요ㅠ'], ['빨래 개는거요~빨래는 세탁기가 해주지만 개는건 진짜 귀차늠요 ㅎ오히려 설거지는 전 좋더라구요 뭔가 개운해지는 느낌요'], ['저는 개는것고 정리 느낌이라 갠찬고 설겆이도 같은 맥락이라 갠탄은데\n빨래 너는게 왜이리 귀탄은지 신랑해줄때까지 개겨요 ㅋㅋ'], ['모두모두 다 싫습니다!!'], ['애들방 치우기요ㅜㅜ'], ['빨래개는거요...귀찮아요 너무너무너무너무ㅠㅠ'], ['치우는거라 다 괜찮은데 ..\n전 정말 요리하기가 싫어요ㅜ'], ['빨래 개는거 넘 넘 싫어요 ㅠㅠ'], ['다 싫지만 ㅠㅠ 그중 젤 싫은건 땀 한대야 흘리면사 치우고 나면 다시 개운한 몸으로 다시 다른일 하는 내가 젤 싫거 짜증나용 ㅠㅠ'], ['다 싫지만 그중에 5번 넘싫어서 매번 남편찬스 써요 ㅠㅠ ㅋㅋㅋㅋ'], ['전부다요ㅋㅋㅋ집안일 체질에 안맞는다고 입에 달고사는 1인입니다요...ㅋㅋㅋㅋㅋ'], ['빨래개기랑 요리요 ㅋㅋㅋ'], ['빨대 씻는거요.. 그게 왜 이리 싫죠ㅜ.ㅜ'], ['빨래개기요ㅜㅜ'], ['전 화장실 청소-- 닦아도 티가 정말 조금나서 하면 기운빠져요ㅠ'], ['전 7번 각종 먼지닦기ㅜㅜ'], ['먼지닦기도 힘들어요 맞아요 ㅎㅎ'], ['건조기를 추천드립니다 ㅋㅋㅋㅋ'], ['ㅎㅎㅎ 돈 아까워요 ㅎㅎㅎㅎ'], ['왜이제 삿나...후회막심하면서 돈값해서 훌훌 털어버리고 신세계에 접할껍니다 ㅋㅋㅋ'], ['ㅎㅎㅎㅎ 아직은 머 버틸만해서 \n바짝 모으는게 더 재밋어서 ㅎ'], ['신랑은 사자하고 제가 됏다고 ㅋㅋㅋ'], ['ㅠㅠ 다른걸로 모으시고 이건 구매각을 추천드립니당 ㅋ 빨래널기도 필요없구 장마철도 필요없구 ㅋㅋ'], ['ㅎㅎㅎㅎ 보고요 전부 다 내일이 되면 그때'], ['다 싫지만 빨래 개는게 좀 더 싫으네요..ㅋ']]
    
    4488
    💫집안일의 끝은 어딜까요 ! 주방 가동하고 뒤돌아보니ㅜㅜ 어제 아이가 쏟아두었던 블럭이 보이내요ㅜㅜ .... 흑흑 언제쯤 끝날까요
    
    [['끝이 잇을까요? ㅠㅠ 아이들이 어느정도커야...ㅠ 전 셋째 생각하믄 5년이상은..ㅠ 이제 53일이거든요..'], ['아 ... 앞으로 5년이군요ㅜㅠ 저는 아이가ㅜ정리하는 습관이 잘 안되서 미치겠내요'], ['저희도 마찬가지예요~ 7살인데.. 넘 해줘서 그런가봐요ㅠㅠ 1'], ['그런가봐요ㅠ저도 속이 터져서 그냥 제가 정리하는 습관이라소 그런가봐요 이제 아이랑 정리하는\n연습을 같이 해야할 시기인것 같네유'], ['저희집도 귀신 나올것 같아서  치워야되는데  피곤에  절어서  엄두가  나질 않네요\n여행  후유증 크네요'], ['맞아요 여행 휴유증으로 정신없이 쉬다가 급하게 .... 지금 청소하고 주방 가동중이내요ㅜㅜ'], ['저도  지금  오전  허무하게  보내고  수건 세탁기  돌리고   에어콘  바람  밑에서    멍하니  앉아있어요\n시간 너무  잘 가네요'], ['맞아요 저도 오전에 한거는 별로 없는데 벌써 2:30분이라니 시간이 정말 빠릅니다 이러다 저녁까지 휙 가겠어요'], ['그쵸  저도 수건  돌린거  널고  쉴려고  앉으니  이  시간이네요  저녁에도  일  생겨서  나가야되는데   벌써  급  피곤하네요'], ['저도 지금 급 피곤해졌내요ㅠㅠ 왜 집안일은 쉼 없이 해도 끝이 없을까유 ?'], ['집안일이라는게   과연  끝이 있기는  있나요?\n해도해도 끝이  없네요'], ['끝이 없을듯 합니다 ^^ 놀러갈때가 제일 좋은것 같아요 신경 안써도 되니요ㅜㅜ'], ['아이콩 ㅎㅎ 아이가 격하게 놀았군요\n혹여나 밟으면 뜨악! 얼른 치우셔야겠어용'], ['네 어제 블럭을 다 쏟아서 놀았내요ㅜㅜ 아침에 일어나서 멘붕이 왔답니다ㅜㅜ 할일은 천지인데'], ['집안일은 끝도없지용? 거기다가 육아까지 엄마는 참 대단한거 같습니다'], ['네 오늘은 빨래만 몇판 하는지 모르겠내요ㅜㅜ 집에도 에어컨 오늘 풀 가동 중입니다'], ['아고 힘드셨겠어요\n빨래 분리해서 빨아야하니 늘 일회로는 안끝나지요'], ['맞아요 빨래는 분리하다보니ㅜ기본 4판이내요ㅜ아이꺼도 있다보니 한번씩은 멘붕이 옵니다'], ['아이랑 하루종일 있으면 집안일은 끝이 안보이지요.. 저정도면 착합니다요 ㅎㅎ'], ['그런가요ㅜㅜ 아이랑 함께 있다보니 진짜 청소기 땡길시간도ㅠ없어서 더 정신 없내오'], ['그렇지요.. 그래서 저는 오전에 아이 자고있을때 시간이 나니 그때 후다닥 합니다'], ['전 오늘 청소를 결국 못했습니다 😭\n청소기 세척한게 다 마른것 같으면서도 혹시나 해서 하루 더 말리고 낼 청소 할려고요'], ['아하.. 청소기.. 세척하셨군요.. 좀 찜찜하면 하루 더 말리시는게 맞지요'], ['네 ㅠㅠ 청소기 세척하였는데 진짜 속이 시원하더라고요 뭐가 그리더럽던지 ㅠㅠ 진짜 청소기에게 미안했내요'], ['끝이 있을까요?ㅠㅠㅠㅠ막내가어느정도커야..ㅠㅠㅠ가능하지않을까요?ㅠ'], ['그럴까유ㅠㅠ 우린 첫찌하나인데ㅜ장난감을 다 쏟고 다니내유'], ['ㅋㅋㅋ어느정도 놓으면편해요ㅎ사진찍어놓음 돼지우리같은게ㅠㅠ단점이쥬ㅠㅠ'], ['맞습니다 ^^ 항상 집 사진 찍으면 뭔가 ^^;; 어수선하고 난장판입니다ㅜㅜ 아이없을때ㅜ청소하는게 제일 편하내요'], ['ㅠㅠ있을때청소하믄 뭘합니까ㅋㅋㅋㅋ뒤에서어질르고있는데ㅋㅋㅋ'], ['정말 공감합니다 청소하고나면 또 구석에서 블럭 쏟고 또 정리하면 또 자동차 쏟고 무한반복입니다'], ['ㅋㅋㅋ그래서 집안일을 안하고이써요ㅋㅋ어디부터 손을대야할지 참난감하네요'], ['저도 그렇습니다 아이가 있다보니 정리를 해도 다시 원점이라 이밤에는 포기 하였내요 낼 아이 어린이집 보내고 대청소 해야겠어요'], ['ㅋ저는 오늘이불빨래만하고 청소기돌리고 장난감있는 베란다는 안했어요'], ['저는 ... 오늘 청소기ㅡ세척하여서 청소를 못하내유 😅낼 열심히 해야겠지유 청소기 세척하다가 시껍했내유'], ['저는 집에있는 청소기가 시원치않아서 차이슨으로주문해놨어요'], ['차이슨 주변에 쓰는 사람 보니 다들 만족하더라고요! 흡입력도 장난 아니라고 하십니다~~!!'], ['주말엔 포기 하시구여~~애들 있을때도 포기 하시구여\n애들 말귀 알아들으며 같이 하시구여..\n더크면 정리시키는 수 밖에 없어요\n그래도 다른게 지져분 하지만요 ㅋ'], ['그렇내요 ... 정말 주말에는 정리하는거를 포기해야되겠더라고요 아이가 있으미 정리도 안되고 청소도 안되내유'], ['되돌이표인고 같아용\n  그래두 장난감은 아이가 직접 넣고 정리하니 한결 낫더라구요'], ['저도 이제 아이에게 정리하는 법을 알려주어야겠어요 아직 정리하는 습관이 안되어서 그런지 더 힘드내요'], ['아앙 ㅜㅜ 그쵸  우린 직접 통에 넣고 하는걸 어느순간 해주더라구요\n 오늘두 수고하셧어요'], ['그래야겠어요 지금 소변가리는 연습도 하고 있는데 같이 해보아야겠어요ㅜㅜ 이놈의 자동차들이 너무 많네요'], ['아아 지금 기저귀 거의 다 떼가시나봐요 전 아직 시도도 안하고 잇어요'], ['아직 계속 하는중이에요ㅜㅜ 아직은 팬티에 싸는 경우가 더 많내요 점점 좋아지겠지요 ?'], ['이래서 집안일은 해도해도 표도 안나고 끝도 없나봐요~그저 웃지요~'], ['그렇지유 청소 다해놔도 티 안나고 엄마는 힘드내요 ^^;; 특히 주방쪽은 더 티안나서 힘드내요'], ['집안일은 끝이 없는거 같아요~\n치우믄 따라 다님서 그자리 다시 고대로 만들어 두네요'], ['맞아요 저희아들이 딱 그렇내요 엄마를 따라다니면서 장난감을 쏟아두어서ㅠ멘붕이 온답니다'], ['맘님도 그러나 보네요ㅠㅋㅋ\n저희 애둘이가 그러네요ㅋ\n진짜 막 치우다가도 딱 치우게 싫어지는거 있지요 ㅜㅜ'], ['맞아요 열정적으로 치울려고 하는 엄마에게 찬물을 끼 얹는 아들이라서 힘드내요 아이 없을때 청소하는게ㅜ정답인가봐요'], ['맞죠~어쩔땐 너무 울고 싶을때도 있다니까요 ㅋ\n애기 없을때 후다닥 치우는게 답인거 같아요~'], ['맞아요ㅜㅜ아기 없을때 정리랑 다 닦는게 정답인듯 합니다 아이 있을때는 절대 못합니다'], ['해도해도 끝이없고 표도안나는게 집안일이죠 ㅋ\n그나마 첫째가 좀크니 알아서 정리하더라구요~^^'], ['이제 저희아들도 정리하는 습관을 알려주어야겠어요😭그러면 좀 나아지겠지요 ? 함께 정리하면 덜 힘들것 같내유']]
    
    4592
    ♡집안일 중에?? 맘님들은 집안일 중에 뭐가 제일 하기 싫으세요??저는 빨래 너는거랑 설거지요청소는 청소기가빨래는 세탁기가근데 설거지는 오롯이 제가 다 해야해서넘넘 귀찮아요하루 설거지 5번 넘게 하네요ㅠ오늘은 귀찮아서 죄다 몰아서 해볼려구요ㅎ될려나 모르겠어요좀 찝찝해서ㅠ
    
    [['저는 다 싫어요~ ㅋㅋ 근데 저도 설거지가 젤 싫어요..ㅋㅋ \n흠.. 청소도 싫습니다.. (그럼 뭘하지? ) ㅋ'], ['저도 사실 다 싫어요\n결혼 했을때는 뭐든 의욕적으로 했는데 이게 어느정도 지나니 집안일 참 하기 싫으네요ㅠ'], ['빨래와 설거지요ㅋㅋ\n근데 집안일은 다시러요😆😆'], ['다들 비슷하시구나\n저는 설거지가 그렇게 싫으네요\n우리 애들 간식 먹을때 그릇이 6개씩 나온답니다\n뭐든 새 그릇에ㅠ'], ['빨래접어서 정리해서 제자리 넣는걸 가장 싫어합니다 ㅎ'], ['저도 그거 싫어하는데 그래도 그건 덜하더라구요\n설거지 하두 많이 해서 요즘 진짜 하기 싫어요\n손도 다 트구요ㅠ'], ['다른건 하겠는데...설거지랑 화장실 청소요ㅎㅎ 화장실청소는.. 증말하기싫은ㅎㅎ'], ['화장실 청소는 보이는 곳만 살살 해서 그런지 그닥이네요\n오늘은 제가 씻고 하긴 했어요\n설거지는 끝도 없어서요 해도 해도 자꾸 나오니 싫으네요ㅠ'], ['이제 주방가동하면 계속 나오겠죠. 안해먹기도 그렇고.. 그래도 해서 먹는게 낫긴하겠죠ㅠ'], ['오늘은 그래서 최대한 그릇을 좀 적게 꺼내보도록 할려구요\n될려나 모르겠어요 근데 벌써 설거지 3번 했지요ㅠ'], ['주방가동만하면 설거지가 넘쳐나는거같아요ㅎㅎ 안할 수 도 없고ㅎㅎ맛있게 먹긴하는데 뒷정리가 왜  이렇게하기싫은지 모르겠어요ㅎ'], ['맞지요\n그릇 총 출동하는거 같아요\n냄비 후라이팬 접시 집게 등등요\n정리 싫으네요'], ['냉장고정리 ㅋㅋㅋㅋㅋㅋ'], ['맘님 부지런하신갑다~저는 냉장고 정리는 진짜 마음 먹어야 해서\n그냥 눈에 보이는 것만 대충 한답니다'], ['요리는 전혀 생각 못했네요\n저 요리 꽝이에요\n하두 못해서 할줄 아는거 몇가지만 늘 하다가 이젠 것도 못하겠어요ㅠ'], ['설거지 정말 싫어요 ~ ㅎㅎ 저도 몰아서 한꺼번에 하네요 오늘 저녁은 뭘먹어야 할까요??'], ['저는 늘 그때 그때 나오면 하는데 설거지거리가 넘 많아요\n애들 오면 과자도 다 따로 줘야하고 과일은 또 따로 \n그러니 그릇이 산더미 됩니다ㅠ'], ['저는 침대매트 트는게 제일 힘드네요.\n비가오나 눈이오나 꼭 한번 창밖으로\n탈탈 털고서 매트리스 살짝 들어서\n다시 까는데 그게 제일 디네요'], ['그거 힘들지요\n저는 날 맑은 날만 대충 털고 침대 청소기 한번 하는데 마음 생기면 하고 아님 그냥요\n침대 저만 쓰는지라ㅠ'], ['저도 설거지요 ㅜㅜ 다음이사갈때는 식기세척기 고민해봐여겠어요'], ['식기 세척기 친구 집에 있던데 그것도 그닥인가 봐요\n손으로 할때가 더 많다고 하네요\n오래 걸린데요ㅠ'], ['저랑 같으시네요\n저도 결혼해서는 엄청 부지런 떨었는데 이제는 갈수록 집안일이 하기 싫어요ㅠ'], ['전 빨래 장실청소요\n설거지는 많으면 하기싫은데\n하고 나면 개운하니~~'], ['저는 반대로 화장실 청소는 하면 표가 나서 좋은데 설거지거리는 넘 많이 나와서요\n해도 해도 자꾸 나와서 싫어요ㅠ'], ['그렇치요\n장실은 표가나지요\n설거지는 해도 끝이없지요ㅜ'], ['그러니요\n오늘 몰아서 할려고 했는데 도저히 안되서 했어요\n벌써 간식그릇까지 3번 했네요ㅠ'], ['저는 설거지 많이 쌓이면 정말 하기 싫어지고 엄두가 안나서 요리하는 중에도 예열하거나 익히거나 시간날때 간단히 씻을수 있는건 그때그때 하나라도 씻어버리네요..\n그러니 손에 물마를 일이 없어용..ㅠ\n귀찮고 많은 집안일중에 윗분들 말하는 요리며 화장실 청소며 다 귀찮은데 저는  집에 건조기가 없다보니 빨래널고 걷어서 정리하는게 젤 귀찮은거같아요~'], ['저도 일 하면서 하나씩 씻는 스타일인데 그러니 하두 설거지를 많이 해서 장갑을 껴도 글코 손이 자꾸 트고 갈라지고 피부과까지 다녔는데 물에 안 넣어야 한다는데 그게 되나요ㅠ\n지금도 다 갈라지고 피나고ㅠ'], ['빨래널기 개기보다 정리해서 넣는거요 ㅋㅋㅋ 그거랑 냉장고 정리... 그거랑 설거지 요리 ..생각해보니 좋아하는건 읍네요'], ['정리해서 넣는거 싫지요\n남편은 세상에나 저 없을때 빨래는 해서 널고 그냥 그대로 옷 걷어 입고 나가고 그랬더라구요ㅠ'], ['앗 ...저도 가끔 그러는데요?.그래도 빨래는 해서 너셨네요.. 그정도믄 참 잘했서요입니다'], ['설거지 넘 하기 싫어서 오늘 좀 몰아서 할려고 했더니 친정 엄마가 기습방문\n그래서 도저히 안되서 했지요'], ['저는 왠만하면 안나오게 할라합니다..느므 시러요 그릇도 마니 끄내는거 싫어해여'], ['저는 설거지거리는 싫은데 앞접시를 억시 많이 줍니다\n뭐든 따로 먹으라고요ㅜ\n그러니 산더미ㅠ'], ['저도 식기세척기 사고픈데 제성격에 식기세척기돌릴때까지 쌓아논걸못볼꺼구.. 여벌세척하면서 다닦을거같구해서 아예소형을사서그때그때돌릴까합니다ㅎㅎ'], ['맞아요\n친구집에 식기 세척기 있는데 깨끗하고 살균되니 좋은데 그거 시간이 엄청 걸리더라구요'], ['전 빨래 널고 개는게 제일 싫어요 ㅠㅠ 그리고 화장실청소!!!ㅋㅋㅋ 사실 집안일.. 다 싫긴 싫어요..ㅋㅋㅋㅋㅋㅋ'], ['저도 집안일은 다 싫으네요\n월급 주는 것도 아니고 알아주는 것도 아니구요\n해도 표도 안나고 에휴ㅠ'], ['그냥 다 싫은데 가장 귀찮은건 마른빨래 접어 제자리 두는거네요🤣'], ['접어 정리하는게 힘드시구나\n저는 그건 괜찮네요~빨래 너는게 좀 귀찮고 오래 걸려서 힘들구\n설거지는 자꾸 나오니 싫어요ㅠ'], ['저도 설겆이 제일 싫어합니다 그다음 빨래 널고 개는거요ㅋㅋ'], ['그러시구나\n저도 설거지는 넘 싫어요 해도 해도 자꾸만 나오네요\n미뤘다 하는거 힘들어서 하나 나옴 하고 그러니 더 힘들어요ㅠ'], ['설겆이는 정말 싫어요 솜씨는 없지만 밥하고 음식은 하겠는데 하고 남 설겆이 꺼리 정말 많이 나와서 더 음식이 하기싫은듯요ㅠ'], ['저는 할줄 아는건 없는데 그릇이 정말 많이 나오는 편이에요\n뭐든 새 접시에 담는 스타일이라 한가득 나옵니다ㅠ'], ['제가그래요ㅜ요리한번하면 초간단요리해도 그릇 이빠이나와요ㅜ'], ['그러니 뭐든 하기가 싫어집니다\n그릇도 넘 나오는데 결과물은 꼴랑 한두가지라서요'], ['저는 빨래 너는 거랑 빨래 개는 거요..ㅠㅠ 너무 하기 싫어요..ㅠㅠ'], ['저도 빨래 너는 거 정말 싫어요\n세탁기에서 꺼내는 것도 일이구요\n예전엔 남편이 잘 도와주더니 요즘은 내가 너는 거 마음에 안 들제? 그카미 은근히 빠져 나가요ㅠ'], ['요즘은 신랑이 대구에 있어서 거의 다 해줘요..\n신랑이 이제 가고 나면 저는 우짜나 싶어요.'], ['그러시구나\n남편분 넘 다정하고 좋으시네요\n저희도 좀 떨어져 있어야 하나요ㅠ 그래도 안 도와줄거 같아요ㅠ'], ['11월 말이면 이제 강원도로 가야 하는데 그때는 신랑의 손길이 너무 그립지 싶습니다.ㅠ'], ['앗~그러신가요\n그럼 그전까진 재미나게 알콩달콩 지내셔야겠어요'], ['설거지가 좀 귀찮긴해도 밥하는것도 그나마 나은편이라 하고 나면   제일 깨운하네요'], ['저는 매번 그릇을 좀 많이 쓰는지 한가득이라ㅠ\n애들 오면 간식 먹을때 그릇 한꺼번에 6개씩 나오네요'], ['그릇에 예쁘게 담아주시나봐요  저는 그냥 그릇하나에 같이 먹일때가 더 많은편이라'], ['그럼 싸움 납니다ㅠ 따로 줘도 막 서로 더 먹을려고ㅠ\n간식은 싸우고 난리도 아닙니다'], ['유난히 양이 적은것만   나눠주고 봉지과자 같은건 그냥 그릇하나로 같이 먹네요']]
    
    4609
    24.집안일 집청소하고티비보네용집안일 해도해도 끝이옴네요ㅜ
    
    [['저는 그냥 손떼고 친정왔어용ㅎㅎ'], ['친정가까우신가봐여 ㅋ 저도 지겨워서 나갔어영 ㅎ'], ['살림이 여유로우면 일주일에 두번정도는 도우미분께 맡겨보고싶어요ㅜㅜ'], ['저도여 도우미있으면 편한듯해여'], ['신랑이 좀더 벌어와야할거같아요😆'], ['집안일은 판도라의 상자같지요 ~'], ['그니깐여 애기있으면 더해여 ㅎ'], ['전 이제 빨래 널어야지요. 맘님 지금은 장터 구경가셨지요? ㅎㅎ'], ['네 집안일 하다가 지겨워서 나갔어영 ㅎ'], ['집안일은해도해도끝이없어요ㅋ'], ['그니깐여 그래도 산처럼 쌓이지여'], ['맞아요 진짜 끝이 옶는 집안일입니다 ㅠㅠ'], ['집안일하다가 나가여 지겨워서 ㅎ'], ['지긋지긋 ㅋㅋ 전 잘 하지도 않는데 보기만해도 아휴 ㅋㅋㅋ'], ['이사가면 해야지여 지금은 안해도되잖아영 ㅎ'], ['그럴까요 어차피 다시 해야하는고 ㅋㅋ'], ['저는이제 소환가야되네여 ㅠ'], [': 결국 다시 소환인건가요 ㅠㅠ'], ['네 소환갔다가 이제야오네여 ㅎ'], ['5) 새벽반 하시면서 고생하시네요 ㅠㅠ'], ['그럼여 ㅋ 오늘은 그나마 잘자긴하네여 ㅎ'], ['나도어제는 세탁기를 좀 돌려줬고'], ['돌려도 오늘또 나오지않나여 ㅎ'], ['그렇지 근데 어제 두탕해서 오늘 안하지'], ['그쿤여 저희는오늘도 빨래가많네여'], ['세탁기는 내일도 못돌릴꺼같고'], ['그죠 집안일은 왜이리 끝도 없나요 ㅜ'], ['그니깐여 집안일 지겨워여 ㅎ'], ['저도요 제가 나가서 돈벌어 오고 싶다지요'], ['저도여육아보단나을듯하고여'], ['그죠 집에서 애 봐봐야 어떤게 힘들지 알테니 제가 일하고 싶어진다요'], ['맞아영 육아가쉬운지아는거같아영'], ['그러니까요 예전에 윤이 어릴떄 그런걸로 싸웠다죠 애보는게 유세냐며  아주 죽이고 싶었어요'], ['남자들은 잘모르는거같아여 그러면서 하나씩 포기하지여 --'], ['몰라도 너무 모르는거 같아 참 답답하다지여'], ['그렇군여 ㅋ 항상시켜야되니힘들어여 ㅎ'], ['저도어제 간만에 청소햇어여'], ['오호 어제는 집안일하셨나봐용'], ['청소기열심히 돌렷다지여'], ['바쁘셨겠어여 그것도 힘든데말이져'], ['오랫만이라 상쾌햇다지여'], ['상쾌 ㅋ 오늘은 집안미세도 맑음이겠어영'], ['그렇겟죠 ?ㅋㅋ오늘은 환기좀시켜야죠'], ['환기시켜려고열엇더니 밖에 태우는냄시가'], ['악 아침부터 멀태우고잇던가여'], ['동네에서 머태우나봐여 ㅠ 공장에서'], ['청소하기 너무 시르다지요'], ['청소는 진짜힘든데 정리는안되네여 ㅎ'], ['저두 집정리가 안된답니다요.ㅜㅜ'], ['옷정리하다가 내팽개쳤네용'], ['저는 시작두 안했다지요'], ['맘님도 집 넓어서 청소하사기 ㅅ 힘드실거같구요'], ['안넓어여 청소하면 후닥하지여'], ['아넓어보이는디 좁나여 ㅋ'], ['집안일은죽을때까지해야한다지'], ['죽을떄까지여 너무 슬픈데여'], ['안보여야 안하고 살지.. 슬프다']]
    
    4667
    오늘은 밀린 집안일 하는날이에요 윗 지방은 벌써 쌀쌀함이 느껴지는듯해서밤에 입힐 어르신 내복도 싹 꺼내보고이불도 싹 돌려 볼려구 해요 ㅋㅋ이불은 대충 다돌아 갔는데나머지들은 집만 엉망만들어 놓고 또 내일로 미뤄질 이 불길한 느낌은 머지요그래두 세부에서 가지고온 유연제 향기 대빵 좋아 신나하며 돌리고 있어요^^막간에 뒷베란다 나무들 보면서 코피코 커피한잔 마시며 잠시 휴식을 가져보아요^^
    
    [['세부의 향기 집안에 가득하시겠습니다.\n저도 집 좀 치워야 하는데 엄두가 안나네요.ㅋ'], ['저는 어설프게 움직여서 ㅋㅋㅋㅋ 집이 더 엉망이 되었어요 온집안이 세부향으로 가득해서 마음이 든든하네용'], ['전 아껴서 한번씩만 쓰고 있어요.ㅋ 섬유유연제도 한가지만 쓰니 향이 둔해져서 중간에 한번씩만 써요. 근데 세부꺼가 강력하긴 해요. 여름빨래 꿉꿉한 냄새가 안나는걸 보면요~ㅋ'], ['글치요 ㅋㅋㅋ 저는 용감하게 1리터 질러왔지요 ㅋㅋㅋ 1바이1겟 상품도 담으려다 수화물 오바될까 참았는디 아쉬워 죽겠어용'], ['다시가실때까지 아껴쓰셔요.ㅋㅋ젠틀베이비만 사왔는데 다른것도 소포장으로 사와봐야 겠어요~'], ['젠틀베이비 향이 너무 좋아 ㅋㅋ 전엔 그것만 쓸어 담아 왔었는데 이젠 좀 질려서 다른거 가지고 왔는데 이것두 좋아욤'], ['고건 뭐쥬? 남자화장품 냄시같은것도 있던데유ㅠ'], ['핑크색 꽃 그려진거유 ㅋㅋ'], ['델 아니고 다우니쥬?'], ['아뇨 ㅋㅋ 완전 새로운 아이요 ㅋ'], ['오~~또 다른게 있나봐요~~'], ['그냥 저렴하니 별루면 버리자 하고 사왔는데 향도 좋아용'], ['색다른 것에도 도전해봐야겠네요. 수하물이 문젠데 옷은 단벌로 가야겠어요.ㅋ'], ['옷은 ㅋㅋㅋ 워쉬앤드라이로 보내세용'], ['대충 티셔츠만 몆개 챙기쥬 뭐.ㅋㅋ 맡기고 찾으러 갈시간도 없을 것 같아요.ㅋㅋㅋ'], ['파크몰 구경 갈꺼 아닙니꽈 ㅋㅋ 비싸두 4시간만에 되는거 맡겨놓고 밥먹구 마사지 받고 찾으십쇼'], ['파크몰에 마사지 받을 곳이 마땅치 않아 고민입니다. 마사지는 원정을 가야할듯해서요.ㅋ'], ['찾아보면 있을텐디요ㅠㅠ'], ['한군데  타임스퀘어쪽 로컬 환상적인 가격 하나 봤는데 남표니가 그른데 싫다고ㅠ'], ['남편님한테 빨래 셔틀 시키십시오'], ['고결한 그분이 또 혼자 그런거 맡기러 다니시고 하시는 분이 아니십니다ㅠㅠ'], ['지니님도 다음번엔 버리고 갑시다'], ['진짜 이번에 가서 토달면 버리고 가야겠어요.ㅋ'], ['절대 변하지 않트라구요 ㅋㅋ 버리고 갑시다'], ['영감탱이!!! 비비고한보따리 사다두고 떠날까봐요~ㅋㅋㅋㅋ'], ['친절한 지니님 저는 암꺼두 없이 떠날랍니다'], ['하이난 방송하네유\nhttp://www.tmon.co.kr/deal/2465072526'], ['봤는데 ㅋㅌ 배보다 필수옵션비가 더 나가요 ㅋㅋㅋ'], ['노쇼핑이라더니 필수옵션이 많나봐요? 이런된장!!!'], ['날씨 좋네요.. \n울 동네는 흐리네용..'], ['곧 햇님이 방긋 하실거예용'], ['쌀쌀해요.. 급 가을~'], ['글츄 ㅡㅡ 오늘 급하게 어르신 긴팔 내복 다 꺼냈어요'], ['감기 계절이 다가옵니다.'], ['이불빨래가 싫어요 저는 ㅋㅋㅋ'], ['저는 좋아해욤 ㅋㅋ 차렵이라 그런지 그냥 휘리릭 돌려 건조까지 마쳐놓음 뽀쏭뽀쏭 너무 좋아서 매주 돌리는듯해요 ㅋㅋㅋ'], ['진짜요>?저는 귀찮아서 한달에 한번???ㅋㅋ건조기 돌리면 또 시간이 배라서 쫌 지겨워요 ㅋㅋㅋㅋ'], ['빨리 마리고 빨리 세탁 할려구 ㅋㅋ 마이크로화이버였나? 암튼 좀 가볍고 폭닥한걸루 샀어요 ㅋㅋ'], ['세탁기가 한두세개 됐음 좋겠어요 이게 까는거랑 덮는거랑 같이 막 돌리고 싶은데 안빨릴까봐 몇개씩 나눠서 빠니깐 이것만 해도 시간이 꽤 걸리더라구요 ㅋㅋ'], ['글츄 ㅋㅋ 세탁기는 크고 봐야합니다 ㅋㅋㅋ'], ['드럼에서 통돌이로 바꿔야 하나봐요 ㅜㅜ 드럼 믿을수가 없어요 ㅋㅋ'], ['전 통돌이 샀다간 세탁기안으로 추락할까봐서 겁나서 안되용'], ['애들한테도 위험하긴 한데 저도 빨래끄낼때 까치발 들고 해야되서 ㅋㅋ'], ['전 이불 꺼내다 이불과 같이 통속에 쳐박힐거 같아여 ㅋㅋ 단신의 슬픔이지요'], ['저랑 같으시네용 푸하하하ㅏ 저도 항상 팔이 아팠어요 통돌이 썼을떄 ㅋㅋ드럼은 안성맞춤입니당 ㅋㅋ'], ['저는 남편이 너 세탁기에 쳐박힌다 드럼사 그래서 샀어요 ㅋㅋㅋ'], ['앗 푸핫 저 엣날에 그 목욕탕 가면 앉는 의자 있죠?? 고런데 밟고 올라가서 꺼내고 이랬어요 ㅋㅋㅋ'], ['그냥 드럼 씁시다 우리 ㅋㅋㅋ'], ['계란과자님 덧에 또 드럼의 소중함을 알게되었네요 ^^'], ['글치유 ㅋㅋ'], ['저도 이불... 바꿔야할거같은데...\n귀찮아서 그냥 두고잇어요....ㅎㅎ'], ['얼릉  바꿔요 추워지면 더 귀찮아져용'], ['하늘시 예쁘네요~^^'], ['파란 하늘보면서 커피한잔 하믄 좋아요^^'], ['']]
    
    4801
    집안일 남편이 어떤거 도와주시나요? 집안일은 대부분 하기 싫지만 그중에서도 손에 꼽으라면빨래 널기 개기 설겆이 거든요그나마 빨래널기는 건조기가 대신하고설겆이는 식기세척기가 해주니 괜찮은데빨래개기는 넘나 싫어서 건조기에 방치해놓았다가 개곤했는데얼마전부터 남편이 본인빨래를 찾다가 그 꼴을 보기 싫었는지개주기 시작하더라구요남편은 이것도 귀찮음 왜 사냐고 하는데 넘 심한가요?^^;;;;
    
    [['설거지랑 빨래널고 개기 쓰레기버려주기 이정도만 도와주네요'], ['애 아빠는.. 주말에 청소하기 하나만 해요..\n\n'], ['남편이 설거지, 분리수거, 음식물쓰레기, 장난감정리 해줘요~ 청소기 돌리는건 가끔ㅎㅎ \n전 화장실청소가 제일 싫어요ㅜㅜ'], ['맞벌이고.. 화장실청소, 설거지, 분리수거, 청소, 빨래(돌리고, 너는거, 본인빨래개는거) 정도요.. 매일 해주는건 아니고.. 시간날때 알아서 해요'], ['화장실청소 장보기 가끔 아주가끔 분리수거해요\n전 전업이라 이거도와주고  남편이\n억울하다 그래요'], ['분리수거,음식물쓰레기 버리기, 시키면 설거지,주말에 청소기돌리기요~~'], ['음ㆍㆍ분리수거, 청소기, 빨래 털고,널고, 정리하기, 화장실청소정도요\n시키는거 대부분하는 편이에요~'], ['맞벌인데\n빨래돌리고 건조기 후 개는거\n분리수거 종량제 버리기 \n설거지만 하는거 같아요 ㅋㅋㅋ전업이라면 부탁 안할거 같아요~ \n'], ['대부분 해요. 요리만 빼구요. 근데 안했으면 좋겠어요. 잔소리가ㅠ 입을 때려버리고 싶어요. 하지를 말든가 누가 하라고 시킨것도 아닌데 누가 들으면 저는.... 최근에도 난린쳤는데도 쉽게 바뀌지는 않나봐요. 한번만 더 그러면 엎어버릴려구요. 전 깔끔쟁이는 아니여도 누가 저희집 불쑥와도 될정도로는 해놓거든요.'], ["ㅋㅋㅋ'입을 때려버리고 싶어요'에서 지금 빵 터졌어요 저도 그렇거든요!!ㅎㅎㅎ 카페 글도 공감이지만 댓글도 저의 공감이 무한입니다ㅎㅎㅎ"], ['집안일의 모든걸 다 도와줍니다~~^^'], ['남편분들이 모두 가정적인분들이네요 우리 왕자님은 분리수거 다 해놓은거 딸랑 갖다버리는것만 간간히 하십니다\nㅡㅗㅡ짜증나서 맞벌이 하고싶어요 하지만 능력없는ㅜㅜ'], ['저희는 그것도 안해요 ㅎㅎ\n밥먹고 잠만자고 씻고 나가는  하숙생이에요'], ['전 맞벌이인데도....다 제가 해요....여기 계신분들은 전생에 나라를 구하셨나...전 일본 쪽바리였나봅니다..'], ['우리 왕자님ㅋㅋㅋ 저도 오늘 남편에게 우리 왕자님~이라고 불러봐야겠어요(반어법의 말투 연습 좀 해서요ㅋㅋㅋ)'], ['전 맞벌이고 휴직한적은 없어요 애낳을때 한달정도가 다고요. 일은 계속했고요. 전 진짜 살림최하인간이라서요.\n애낳고 일년간은 우리집 세탁기 전원키는것도 몰랐어요. 신랑이 거의 해줬고요. 지금도 화장실청소 방청소 청소기 돌리기 이런거 남편이 다해여. 설거지는반반하고 그대신 식사담당은 제몫이에요 사먹던 하던 제가 해야해요. 분리수거는 시간되면 같이하고 되는사람이 하는편이고요. 빨래도 그래요 건조기에서 꺼내서 하면 같이할수있음같이개고 같이정리하고 한명없으면 그냥 바구니차면 본사람이해요. 지금 남편한달출장중인데. 다이슨청소기 먼지통 다차있는데 그거 한번도 분해해본적이없어서 한줄을 제가 모르더라고요 신랑한테 물어봤더니 빨간거 누르라는데 왜안되는지. 얼릉 신랑왔으면좋겠어요ㅠ'], ['가끔 빨래 같이 널어주고 가끔 쓰레기 버려주고 가끔 애 씻겨주고요 나머지는 거의 제가 다해요ㅠㅠ'], ['분리수거, 음식물쓰레기, 설거지, 주말에 집대청소 해줘요. 저희남편은 설거지를 엄청 좋아해요 ㅋㅋㅋㅋㅋㅋ'], ['저보더 잘해요~'], ['전업인데.. 솔직히 신랑이 해줄게 없어요ㅋㅋ 아무리 빈둥거려도 신랑 퇴근전에 일 다 끝나서 ㅋㅋ 가끔 주말에 저 늦잠자면 아침에 애기 밥 챙겨주기? 가끔 음식물쓰레기버리기? 그리고 그외 잡일 ㅋㅋㅋ 그리고 가끔 요리?? 정도 하는거 같아요~'], ['맞벌이인데 남편 평일에 늦어서 주말에만 화장실청소, 음식물쓰레기, 분리수거 해요. 그 외엔 기분좋을때나 제 눈치 볼때만 설겆이, 빨래 널기, 개기 해주면서 해줬다 생색네요.'], ['맞벌이인데 주말에 집청소기돌리고 걸레질 정도 하는거 같아요...설거지도 가끔하고...빨래널기나 개기는 시키면 하긴해요...'], ['재활용...쓰레기.\n제가 씻고 말려서 종류별로  나뒀던거 들고나가 버려줘요.  박스등 종이류는 나오자마자 버리기에 제가합니다.\n그외...없어요.'], ['와~ 다들 많이 도와주시네요.\n캡쳐해서 카톡에 보내야겠어요.\n우리남편은 낫띵....  아무것도..... ㅠㅇㅠ\n출,퇴근늦어 집에선 자고 씻고 가요~ \n다행인건 집에서 밥도 잘 안 먹어요ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['전업인데 남편은 아무것도 안해요^^ 재활용쓰레기 모아서 묶어놓은거 갖다 버리는거 어쩌다가 손에 꼽을정도로 몇번? 먹고 자고 씻고 출근'], ['설겆이,빨래돌리고 개기, 청소기돌리고 걸레질, 분리수거, 음식쓰레기 종량제 버리기해주네요'], ['우리 신랑도 집안일 1도 안해요^^\n바쁜 아빠라 조금이라도 짬이나면 아이를 위해 시간쓰기로 약속했거든요 집안일은 도움받을수도 있고 사람을 써도 되는데 아빠역할은 오로지 신랑만 가능하니 그렇게 하기로 했는데.....진짜 1도 안하네요~~'], ['빨래정리.. 저도 하기싫어요 ㅋㅋ\n다행히 즤남편은 집에 있는한 무조건 일해요\n밥준비.설거지.분리수거.화장실청소.세탁기돌리기 건조기넣기 빨래정리 등등 뭐 ㅋㅋㅋ \n'], ['외벌이인데 평일엔 주로 저녁까지 회사에서 잘 먹고 오고 (제가 남편밥 차릴일이 없어요^^.) 집에서는 식사를 안하니 도와줄 집안일도 없고...저도 힘들게 일하고 온 남편 일 시키고 싶지 않아요~^^ 편히 있으면서 아들램 책읽어주고 목욕시켜주고 잠들 때까지 같이 놀아주기 해주네요. 주말에는 집에 있을 때 쓰레기 분리수거, 설거지, 빨래 널고 개기, 화장실 청소 등 잘 해주는 편이에요. ㅎㅎ\n돈도 잘 벌어오고 매일 힘들게 전쟁터에서 전쟁하는 남편 저도 직장생활 오래 해 봤기에 얼마나 힘들지 이해하고 집에선 되도록 편히 쉬게 해주고 싶어요. 맞벌이면 당연히 뭐든 같이 해야겠지만요~^^'], ['제가 시간여유가 많으니 다 하는 편이예요.\n분리수거도 두세번씩 혼자 왔다 갔다 해요.\n대신 돈 많이 벌어오니 그러려니 해요.\n아빠가 안 하니 아이도 당연히 안 하네요.\n나중을 위해서 아들한테는 조금씩 가르쳐야겠어요.'], ['저희 신랑은 출근전 밀대로청소, 퇴근후 밀대로청소, 빨래 각잡고 널고~각잡고 개고. 분리수거등등 요리빼고 다해여~ 너무 깔끔떠는 사람이라 엄마들 초대하믄 부담스럽다고 해요ㅡㅡ \n잔소리 없이 본인 스스로하니 엄청 편하지만\n전 쫌 편하게 살고싶네요~'], ['도와주는 개념 없어요. 부부가 사는 공간이니 같이 해요.'], ['저는 건조기가 없어서 건조대에 널어놓고, 개기 싫어서 그냥 그때그때 입을때 건조대에서 걷어입어요.ㅎㅎ 다음 빨래하기전까지는 건조대에 걸린건 다 입게 되더라구요. 남편이 도와주는건....화장실 청소, 무거운거 들거나 전기,기계 고치는 것뿐이예요. 두가지는 진짜 못하겠어서..ㅠㅠ 외벌이에 처자식 먹여살리느라 고생하는데 집안일까지 시키고 싶지 않더라구요. 따지고 보면 일의 강도나 종류에 차이가 있겠지만......전업이 이것도 안하면.....정말 살아야할 이유가 없겠죠.'], ['화장실청소,분리수거,음식물버리기\n주말설겆이,청소기,다림질등등~주말에몰아서하네요~ㅎ'], ['맞벌이인데..\n정기적으론\n1. 음식물 쓰레기랑 종량제 쓰레기 버리기(1회/1~3주)\n2. 세탁기에 빨래넣고 돌린후 널기\n만 합니다. \n3. 아. 올 하반기에는 식기세척기에 그릇넣기도 3번정도 한 거 같네요.\n\n곤죽이 되도록 밟아버리고 싶을때가 종종 있어요. 허허허허..'], ['라면끓여주기.\n저녁설겆이\n빨래개기.\n쓰레기 버려주기 해줘요..'], ['빨래널기 쓰레기버려주기 가끔 설거지해주는거요ㅎㅎ'], ['제가 젤 하기 싫은 화장실 청소, 음쓰 버리기 두개 하네요\n다른거 시킬래도 씅에 안차서리...'], ['청소는 커녕 어지럽히지만 않았으면 좋겠네요..\n집안일 도와주시는 남편있으신분 넘 부러워요'], ['집안일은 안해요. 대신 매일 아이랑 목욕하고 숙제 봐주고 가끔 아이방 정리 아이랑 같이 해요. 집안일은 되도록 일절 안시키고 아이 관련된것만 주로 시켜요. 아!!거실 욕실청소는 신랑 담당이네요~전 안방욕실만 사용해서..ㅎㅎ'], ['친정아빠는 다른건 다 안해도 화장실청소는 꼭 하시길래 모든 남자가 그런 줄 알았는데...ㅋㅋ아니더라고요. 빨래 개기(넣는건 죽어도 안함), 가끔 청소기돌리기 하고, 제가 밥하면 설거지는 본인이 해요. 반대로 남편이 밥하면 제가 설거지..아이 목욕은 잘해줘요. 맞벌이에요'], ['설거지, 음식물 쓰레기 버리기는 가끔 해주고, 분리수거는 전담으로 맡아서 버려줘요. 그 외 나머지는 다 제가해요. 가끔 화장실 청소 해줄때도 있긴해요~~^^'], ['지금 제가 쉬고있어서 집안일은 제가해요 도와주면 고마운거구요~그대신 육아는 같이해요'], ['전업이라 제가 하는데 건조기도  식기세척기도 없습니다.. 손목아프고 어깨아프고 짜증나고..자기먹은 그릇이나 설거지하면 좋겠네요..그래서 먹은거 설거지안하면 할때까지 옆으로 밀어놔요.. 놔둔 쓰레기도 안치워요..전 제가 치울건 깨끗이 치우고 정리안하면 못참는 성격이지만 뒤치닥거리,파출부 되라고 부모님이 낳아주신건 아니니까요..'], ['식기세척기 강추에요!'], ['분리수거, 쓰레기버리기 하고 있어요~ 남편 퇴근이 늦어 혼자 다 해버릴까하다가... 남편도 집안일에 참여하면 좋을 것 같아서요^^ 시간되면 부탁하는거 다 해줘요~ㅎ'], ['전업인데 시키고 싶어도 늦게오니 ㅋㅋ 제가 거의 다해요 남편이 쓰는 화장실 빼고요 아이 어렸을땐 집안일 도와주시는분 오셨는데 이젠 혼자서 다 커버할수있어서요'], ['맞벌이라 청소.빨래.집안정리는 남편이 하고 요리와 설거지는 같이 해요.남편이 깔끔한 성격이라 저보다 집안일을 잘 해요'], ['맞벌이인데요.\n분리수거 쓰레기버리기 화장실청소 남편이 해요. 8년 결혼생활동 한번도 안해봤어요. 그외 설거지 청소 같이하는편이고 빨래하기 개기, 요리는 제가 다해요. 우리는 자기 영역을 나누는 편이예요.\n남편 바빠져서 일주일 한번 가사도우미 오세요.'], ['전업이예요\n남편 평일 퇴근하면 음식물 쓰레기 및 분리수거. 빨래 건조기에서 꺼내놓은거 개줘요\n주말엔 아침에 누룽지 끓이기.삼시세끼 설거지랑 아이 목욕.운전.가습기 물 채워 끼우기\n빨래 건조기에 넣고 개기요\n'], ['전 전업맘이라.... 뭐 시킬게 없네요..\n주말에 청소같이하는게 다에요 ㅋㅋ\n쓰레기 들고 출근하구요~'], ['전맞벌이긴한데...\n남편이 집안일 더 많이해요\n아이 먹을거 요리랑 화장실청소만 빼고 거의 신랑이 하는거 같아요\n대신 시어미보다 심한 잔소리는 덤이에요 ㅋㅋㅋ'], ['맞벌이라 저녁 설거지는 무조건 신랑 몫이구요. 빨래 하기,음식물버리기 분리수거, 가끔 대청소하기, 종종 평일주말요리하기, 이정도 하는거 같아요.']]
    
    4829
    9. 집안일 저 이제 진짜 밀린 집안일하고ㅋ시댁갑니당ㅋㅋ그동안 많은 알림 부탁드려요^^크크댓글놀이 넘넘 재밌어요ㅎㅎ
    
    [['밀린 집안일.. 저는 언제하나요 ㅎ'], ['후다닥 해결하고 갔어요ㅠㅠ \n사실 다 못하고 나간건 안비밀입니다'], ['흐흐 원래 그런거지요 다 하고 다니기는 힘듭니다'], ['그래서 집에와서 신랑 시켰어욬ㅋㅋㅋ\n불쌍한 신랑ㅋㅋㅋ'], ['앗 ㅋㅋ 신랑님이 그래도 해주신거군요 ㅎ'], ['퇴근하고 와서 제가 못한 부분은 늘 해줘요. \n근데도 제 눈엔 부족하구요ㅠ \n그래서 다투는데.. 뒤돌아서면 미안해요ㅠㅠ'], ['그죠 그래도 그렇게 해주시니 좋은거라지요 ㅎ 울 신랑은 안했었던 ㅋㅋ'], ['재윤맘께서 부지런하셔서 도와줄 부분이 없었을 거 같아요! 깔끔하시고 잠도 없으시구용^^'], ['아하하 아닙니다 ㅎ 지금이야 이러는데 그전에 아들 어릴때는 아들한테만 메달려서 아무것도 못했지요 ㅜ'], ['애들 케어도 깔끔하게 잘 하셨을 것 같아요!! 안봐도 훤 합니당😘'], ['으하하 그렇게 봐주시니 감사합니다 ㅎ'], ['ㅎㅎ 네 제가 다 보냈지요 ㅎㅎ 저는 이제 세탁기 돌리고 집안일 합니다. ㅎ'], ['네^^ 스위티님 자주 만나요 우리♥️\n전 다 못하고 나갔어요ㅋㅋ'], ['오늘은 뭐하고 계신가요? ^^ 저는 나갔다가 좀전에 왔지요 ㅎㅎ'], ['저는 오전내 아가와 씨름하고 두시부터 다섯시까지 바느질 선생님 오셨다 가셨어용ㅎ'], ['앗. 바느질 배우세요? 집으로 선생님이 오시나보네요? 오머낫.. ^^'], ['나라사업 중 하난데 신청해서 됐어용ㅎㅎ\n저렴하게 이용합니당ㅋㅋ \n아가 용품 만들어용ㅎㅎ'], ['아하.. 그런것도 있군요. 좋은데요 ^^ ㅎㅎ 아가용품 만들기.. 잼있을거 같아요'], ['넘넘 재밌어요!!무엇보다 선생님이 진짜 진짜 좋으세용'], ['^^ 집으로 와주기까지 하니 너무편하고 좋을거 같아요'], ['네^^ 오셔서 애가 넘 울면 애도 봐주세용ㅎㅎ'], ['젖병 두개뿐인데 씻으시는건가요'], ['저는 두개 모아지면 씻어용^^\n자기전엔 하나도 씻고 잡니당ㅋㅋ'], ['그러면 세제를 너무 많이쓰는거같아서 저는 네개 모이면씻어요 ㅠ'], ['아.. 저는 두개 이상 쌓이면 마음이 불안하드라구요 ㅜㅜ 어쩐지 세제가 금방 없어진다 했어요;;'], ['젖병을 저는 그래서 열개를씁니다 ㅋㅋㅋㅋㅋㅋㅋㅋ 하루종일 모아도 여섯개라 담날도 든든하구요'], ['와우!! 많이 쓰시네용^^\n저는 8개에용ㅎㅎ 근데도 그러네요ㅠㅠㅎㅎ'], ['여덟개면 넉넉하지않아요? 하루종일 아침부터 저녁까지 3시간텀으로줘도 밤에자니까 6개밖에 안쓰던데..'], ['맞아용 ㅎㅎ넉넉해요!! \n하루 수유 4번 정도 하네용!! \n근데도 쌓이면 뭔가 해야될거 같고 그래용 ㅠㅠㅎ'], ['이제 모아서 해봅시다 ㅋ 세제비 아껴보아요'], ['진짜 그래야겠어요ㅠㅠ 저 좀 시집살이 당하는 기분이었는데 말이쥬'], ['그쳐? 댓글놀이 재미있쪄? 조심히 잘 다녀오세요'], ['네^^ 재밌네용ㅎㅎ 오늘 새벽반 달리고 시픈데 가능할지 모르겠어요'], ['하실 수 있으면 함께해요~ 저번에 함께였을 때 재미있었는데!ㅋ'], ['같이 하고픔데 맨날 자요ㅜㅜ 저도ㅠ모르게ㅜㅜ 양치 일단 하고 올게용'], ['잠이 오니까 자게 되지영! 그래서 저는 먼가 먹으면서 해여!ㅋ'], ['저는 양치하고 왔어용^^ \n하리보 젤리 사왔는데 먹고 시프네요ㅠ\n양치 했는데 말이쥬'], ['저는 계속 먹으면서 하는데... (새벽반 하려면 먹으면서 하라고 그러더라구용)'], ['저 요며칠 그냥 잠들어서 양치도 못했어요ㅜㅠ 간식은 늘 먹었는데 말이쥬'], ['5. 저도 그런 적 있어요. 지쳐 쓰러져서 그냥 자버렸다는!ㅋㅋㅋㅋ 아침에 하면 개아나여.'], ['진짜 아침에 눈뜨면 입안이 너무 텁텁 찝찝 하더라구용 ㅠㅠㅎㅎ 저도 아침에 햇어용ㅋㅋ'], ['우와 시댁은 어쩐일로 가시나용 ㅎㅎ'], ['점심 얻어먹으러 갑니당ㅋㅋ진짜 밥만 먹고 나왔어요'], ['우앙 좋으네영 시엄니 좋으신가용'], ['시어머니 안 계세용 ㅠㅠ \n도움 주시는 이모님이 계세용;;'], ['어머 시어머닌 일찍 돌아가신건가여 ㅠㅠ'], ['결혼하고 두달만에 돌아가셨어요ㅠ \n암 말기셨어요ㅠㅠ'], ['헉.. 그러셨군요 정말 슬프셨을거같아요 ㅠㅠ'], ['마음이 많이 힘들었어요ㅠㅠ물론 신랑이 더 힘들었지만요!ㅠ 생각하기도 싫으네요ㅠㅠ'], ['5 그러게요 힘들어하는 남편분이 ㅠㅠ 에휴 생각맙시다ㅠ'], ['네ㅠㅠ 다 지나간일 덮어둬야죠'], ['저도 집안일해야하는데말이져'], ['저는 다 못 하고 나갔어요ㅠㅠㅋ'], ['저도여ㅋ설겆이도안햇네여'], ['저는 밥을 안 먹어서 설거지는 없고\n청소 빨래 젖병이 있네요ㅠ'], ['아고 젖병도 은근귀차나여'], ['진짜요!! 왜케 귀찮은지 모르겠어요ㅠㅠ'], ['젖병솔따로해야하자나여'], ['네ㅠㅠ 젖병솔도 따로 해서 뽀드득 소리나게 닦으야해요ㅠ'], ['애거라서 더신경쓰이고여'], ['네 맞아요ㅜㅠ 성인이면 이정돈 아닐듯요'], ['시댁 가셧군요 저는 이제 저녁준비를 ㅠㅠ'], ['저는 저녁 치킨으로 때우고 들어왔답니당\n'], ['치킨 좋네요~~저는 대충 먹엇어요'], ['전 커피숍 가려고 나갔는데ㅜㅜ 치킨집을 만나는 바람에 ㅜㅜ'], ['시집에서 점심만 먹고 들어오셨나요'], ['네^^ 점심 먹고 나와서 알라딘 갔다가 트더 갔다가 치킨 먹고 돌아왔어용ㅎㅎ'], ['어제 좀 바쁘셨네요 ㅎㅎ 점심만 먹고 나오신 거군요 ㅎ'], ['네^^ 시댁이 워낙 가까우니 종종 자주 가용ㅋㅋㅋ'], ['아고 걸어서 가까우신가보네요'], ['걸어서 5분 거리입니당 ㅠㅠㅋㅋ 너무 가까워요'], ['헐...걸어서 오분이라니 헐...ㅜㅜ'], ['아주 그냥 엎어지면 코 닿고도 남아요ㅠㅠ']]
    
    4838
    집안일 중에 제일 하기 싫은게 뭔가요? 집안일 중에 제일 하기 싫은게 뭔가요?저는 설거지요.가족 수도 안 많은데때때로 나오는 설거지 양은얼마나 또 많은지요.또 그릇만 씻으면 되는게 아니고냄비에 후라이팬에... 어후.다 씻고 나면 말려서 수납까지 싹 해야 하고.저는 빨래도돌리고 널고 정리해야해서손이 많이 가서 싫지만ㅋㅋ설거지는 정말 지옥이네요. 흑흑**
    
    [['방닦기요ㅟ방닦는게 제일 귀찮아요😭😭'], ['아... 방닦기.\n저는 그냥 물걸레청소기로 \n최대한 힘껏 쓱싹 밀어버리고 말아요.\n세아들맘님은 어떻게 하세요?'], ['저도 방닦기요ㅋ'], ['성현님도 방닦기가 싫으시군요.\n방청소를 어떤 방법으로 하시는지요? ^^;;'], ['걸레빨기ㅎㅎㅎ 제일시러요ㅎㅎㅎ'], ['아... 걸레빨기. 복병이죠~^^;'], ['진짜 어찌나 시른지ㅎㅎㅎ'], ['설거지는 안 싫으세요?\n의외로 설거지 싫어하시는 분은\n별로 없으시네요~^^;\n저 혼자 엄청 게으른 듯요ㅠㅋㅋ'], ['저는 설거지는 밥먹고 바로 하면 안싫은데 미뤘다 하면 시러요ㅎㅎ'], ['저희는 한끼에 나오는 설거지양도 많은데\n미뤘다하면 진짜 주방전쟁날 듯요(@@)'], ['저도 설거지요   이사가면 식기세척기 꼭 사기로했어요 ㅋ'], ['식기세척기♡\n정말 두번 손 안대도 될 만큼\n깨끗하게 세척이 싹 되어 나올까요? (=귀가 솔깃ㅋㅋ)'], ['빨래개키는거요.너무 귀찮아요ㅋㅋ해서 널기는 하겠는데 뒷정리가 싫으네요ㅋㅋ'], ['저도 뒷정리 너무 귀찮아요.\n빨래 개키고 \n각자 자리에 배달까지 시켜야하고ㅜ 퓨우~'], ['빨래개고 정리하는거요ㅠ'], ['빨래 개키고 정리하는거 \n진짜 은근 귀찮죠?\n돌리는것부터 시작해서\n말리고 개키고 정리해서 넣고.\n이것도 정말 은근한 노동력이 동반되는 듯요ㅠㅠ'], ['빨래개키기요ㅠ넘싫어요ㅠ'], ['맘님들 집안일 중에\n빨래 개키는거 싫어하시는 분이 많으시네요^^'], ['전 빨래널고 개키는거요ㅜㅜ'], ['그렇죠. 빨래 양이 많을땐 \n아주 스트레스 받아요(@@)\n널때도 없는데 말예요ㅜ'], ['저는 차라리 설거지나 빨래 걷고 개는것도 좋은데..욕실청소가 제일 넘나 진짜 하기 싫습니다.ㅋㅋ'], ['아, 진짜 끝도 없는 집안일ㅠ\n욕실청소도 제때 안하면\n표시가 너무 많이 나죠?ㅋㅋ'], ['저요 저 설거지 제일 싫어해서 남편한테 토스했습니다 ㅎ 저녁먹고 설거지 쌓아두면 신랑이 퇴근해서 해줘요 ㅋㅋ 대신 주말 독박육아 설거지외 가사 등등 다 제 몫이라 부러우실건 없습니다~'], ['막 부러울래다가...ㅎㅎ\n주말 독박육아 설거지외 가사 등등이\n욱율은맘님 몫...이라니... 퓨우~'], ['방닦는거랑 빨래개키기는 괜찮은데 정리해서 넣는거 넘 시러욧'], ['그쵸? 정리해서 넣으려면\n도대체 엉덩이를 몇번을 떼야 하는건지...\n설거지만큼이나 빨래도 귀찮긴 마찬가지인듯요~^^;'], ['저도 빨래널고 개키는거요ㅋㅋ근데댓글보니 그런맘님들많네요ㅋㅋㅋ'], ["맘님들께서 하시기 싫어하시는 일 중에\n'빨래'가 압도적이신거 같네요.\n저는 빨래 돌려놓고 꺼내서 \n너는 것 부터도 싫더라구요**\n(=어쩐지 기가 빨려서요ㅠ)\n\n그 이후에 개켜서 정리하는 건\n말할것도 없구요.\n그래도 역시 저의 집안일 복병은\n설거지...라는요ㅠㅋㅋㅋ"], ['빨래개서 정리하기ㅋㅋㅋㅋ하....\n너는것까진 괜찮은데 그뒤작업이 너무 너무싫어요ㅋㅋㅋㅋㅋㅋㅋㅋ😭😭😭'], ["맘님들이 하시기 귀찮은 일들중\n'빨래'가 1등인듯 하네요~^^;"], ['그 하기싫은 빨래..또 개야하는데 쳐다만보고있어요ㅋㅋㅋ'], ['전 빨래개는거랑 닦는거오..\n물걸레청소기있는데 초창기모델이라 선이있어서ㅠ싫으네요'], ['선 있는 물걸레 청소기 쓰시나봐요ㅠ\n닦는것도 은근히 손이 많이 가지요?\n체력저하의 주된 원인ㅜ'], ['3년전인가?샀었어서ㅠ선이있어요ㅠ그래도 집이크지않으니 그럭저럭ㅎㅎㅎ됩니다 근데귀찮아요'], ['맞아요. 귀차니즘이\n늘 문제인 것 같아요**'], ['저는 밥하기요ㅋㅋㅋㅋㅋ\n기본이 안되있는 주부입니다ㅜ'], ['차뇽이님은 분식 좋아하시는것 같으시던데요?\n각자 좋아하는거 먹으면\n그게 행복이죠^^'], ['치우고 설거지하는건 자신있는데 밥하기 넘 시러요ㅋㅋ'], ['아하~ 밥하기를 싫어하시는군요!\n그래서 저는 밥하기는 아주 기본적인 것만 해요(**)'], ['저는 싹다 싫어요ㅎㅎ 그나마 설거지가 가장 나은것 같아요ㅎ'], ['진짜요? 와아~ 설거지ㅠ\n설거지가 가장 나은것 같으시다구요?\n에휴, 저는 설거지할때마다\n스트레스 받네요(@@)'], ['전 모든집안일에 스트레스요ㅎㅎ\n빨래가 제일 싫어요ㅎㅎ'], ['분리수거요~~'], ['분리수거 싫어하시는군요.\n저는 또 분리수거는 괜찮더라구요~^^;'], ['빨래개고 넣기요ㅜㅜ'], ['역시 빨래 싫어하시는 분들이\n강세네요~^^;;'], ['빨래개는거 까진 괜찮은데 각 방에 정리하는거요~~'], ['몇번을 왔다갔다 하며\n움직여야하니-;;;\n그쵸?\n아~ 그러고보니 저는 설거지도 싫어하고\n빨래도 싫어하네요ㅋㅋ'], ['빨래돌리고 널고 개고 하는건 괜찮은데\n설거지랑 애들 어질러논거 정리하는게 막상막하로 제일 싫으네유'], ['흐억** 애들 \n어질러놓은거 정리하는건\n정말 힘든것 같아요.\n그렇다고 안할 수도 없고 말예요!\n전 어질러놓은 거 정리할땐\n아예 마음을 비우고\n멍 때리면서 해요.\n안 그러면 홧병이...ㅋㅋ'], ['전 옷정리정돈이요계절변화있을때 더힘드니 그런것같아요ㅋㅋ'], ['맞아요. 간절기때 옷 입히기가\n정말 난감하죠. 다가올 계절을\n미리 준비하자니 이르고, \n그렇다고 하지 않고 있자니\n뭔가 뒤쳐지는 것 같고 말예요ㅎㅎ'], ['빨래요\n빨래 정리해서 다시 서랍에 넣는거..그게 그렇게 싫으네요'], ['그게 진짜 귀찮죠?\n저도 빨래 개켜서 서랍에 넣을때까지\n10만년은 걸리는 듯 해요ㅠ'], ['저두 설거지요~ 설거지에 젖병 씻기, 이유식 그릇 씻기, 식판 씻기 등등 포함 이요~ 넘 하기 시러요 ㅠㅠ'], ['젖병에 식판까지...\n설거지거리가 다양하시네요ㅠ\n아이코오...'], ['저는 음식하는거요..해도 해도 안느는게 음식하는거예요..간도 잘 못맞추겠고..스트레스예요.'], ["딸기친구님은 특이하게 \n음식 간을 못 맞추시겠다고\n하시네요~^^;\n저도 요리는 잼병이어서\n진짜 기본만 해요.\n'사람이 먹을 수 있을 정도면 된다' 요런 마인드로요ㅎㅎ"], ['저도젤싫은게 음식하는거요ㅜ 재료들 손질하는것도 시간너무걸리고 반찬몇가지 국한가지 매일하면 하루 한두시간은 기본이에요 설거지나 방청소 정리 이런건 30분안에 다되는데 밥하고 반찬하고 국끓이고 뒷정리다하면 2시간은정말기본이라.. 특히 여름에 불앞에서잇는건 진짜ㅋㅋ..'], ['저도 방닦는거요. 물걸레 밀대로 하다가 아너스로도 하다가...결국 손 걸레질 합니다.ㅎ']]
    
    4846
    신랑이 하는 집안일 맘에 드세요? 둘째임신하고 신랑이 가끔 설거지랑 쓰레기버리는거 도와주는데요.집안일은 도와주는게 아니라 같이하는거라 하는데요저희신랑은 돈안버는 제가 하는게 당연하다 생각하는사람이라 도와주는거라 할게요..ㅠ첫째키울때도 쉬는날 설거지 한번씩 하는거빼곤 제가 다 했는데 그나마 둘째임신하고 입덧하니까 설거지랑 쓰레기전담해서 해줘서 고마워하고있어요ㅎ두달가량 그리하고 요즘 입덧이 나아져서 어제는 주말이고 설거지 쌓아두기가 그래서 오랜만에 설거지를 했는데..수채구멍은 물론이고 하수구 속이 새까맣게 ㅋㅋㅋ고무장갑끼고 수세미질했더니 수세미는 까만 덩어리들이 덕지덕지ㅋㅋ청소다하고 수세미랑 고무장갑 버렸네요ㅠ가끔 설거지한 그릇에 기름때가 남아있을때도 있는데잔소리하면 안해줄거같아서 입다물고있어요ㅎ손이 야물딱진 남편님들도 계신거같던데우리신랑은 꼭 손이 한번 더 가야되용ㅋ그래도 도와주는게 어디냐며 고마워해야겠쥬?손이 야무진 남편님 두신분 부럽습니다ㅎㅎ
    
    [['저희신랑도 그래여\n본인은 하고나서 뿌듯해하는데.. 제눈엔 할거투성이ㅋ 그래도 암말안하고 잘한다잘한다 해줍니다 \n글고 나중에 제가 다시 정리하죠ㅋ'], ['그쵸ㅋㅋㅋ잔소리를 극도로 싫어해서 한마디도 못해요ㅋ\n다시는 안한다할까봐ㅋㅋㅋ'], ['부러운게 아닌거 같아용ㅠㅠ 남편이 자취를 오래해서\n집안일 잘하긴 하는데 엄청깔끔해요...\n문제는 잔소리를 엄청해요!! 그거때문에 많이싸워요\n잔소리좀 안하고 해주면 좋을건데~!'], ['ㅎㅎ그래도 부럽습니다ㅎㅎ\n저는 잔소리들을테니 깔끔하게 해주면 제손안가서 좋을거같아요ㅎㅎㅎ'], ['저희도 똑같아요 ㅋㅋㅋ \n좋은데 다시 부탁하는편이에요 \n요샌 그래도 보고 좀 따라하려고 해줘서 \n나아지고있어요 그래도 제손이 다시가야해요 ㅋㅋㅋㅋ 세세한부분이 달라서요'], ['다시 부탁하면 짜증내는 타입이라 그냥 제손한번 더가도 놔두고있어요ㅠ\n아예 안해주는거보단 나아서ㅋㅋㅋ\n여자와 남자의 디테일이 다른거같아요ㅎ'], ['맘에 안들죠~맘에 안들어도 잘한다 하면서 계속 시켜용 그래야 계속 도와줄려고 하니깐요'], ['그쵸ㅎ잘한다잘한다해주며 계속 시켜먹는게ㅋㅋㅋ\n잔소리하면 니가해라 안한다 하더라구요ㅠㅠ'], ['저도 이래저래 시켜보다가 맘에 안들어서 하지마라하다보니 하는게 이제 별로없어요🤦\u200d♀️🤦\u200d♀️잔소리하면서 에너지쓰기 싫어요 쓰레기버리기 무거운거 드는거 위주로만 시켜요'], ['초기입덧때는 진짜 물냄새만 맡아도 울렁거려서 식기 기름때보고도 그냥 썼네요ㅜㅜ맘에 안드는데 제가 못하겠으니ㅋㅋㅋ진짜 설거지하나 제대로 하기 어렵나봐요 즈희남편은ㅠ'], ['이집 진상도 그래요 한번보고 병걸릴거같아서 다시하면서 주방에 손대면 죽인다고 들어오지말라햇어요 ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋ카리스마ㅋㅋㅋㅋ\n저는 병걸려도 좀 부려먹다 버리든가해야지ㅋㅋㅋ\n고마 어제는 기겁했네요ㅠㅠ'], ['그냥 서로 잘할수있는거 하자! 하며 서로 스트레스안받고 삽니더 ㅋㅋㅋ 그게 최선이지.. 살림까지 잘하는 넘의남편보며 부러워하기에는 내가 너무 비참해서 ㅋㅋㅋ 안합니더 ㅋㅋ'], ['ㅎㅎ네..꼼꼼하고청소를저보다더잘하네요ㅜㅜㅜ'], ['정말정말 부럽습니다👍👍'], ['빨래널어라니..걸쳐논수준.개라그럼..뭉개논수준.. 그냥손한번더가서 서로스트레스안받게 제가다합니다.. 뭐.. 변기막아놓으면..제똥은잘뚫어줘요..지도싸야되서..'], ['ㅋㅋㅋㅋ공감가요 개는건 저도못해서 괜찮은데 널어둔거보면 털지도않고 막 겹쳐서ㅋㅋㅋ\n변기는 막지나않음 좋겠어요ㅋㅋㅋ'], ['전.. 제맘에 들게하게 시켜요 ㅋㅋㅋㅋ 애교 좀 섞어가면서 칭찬하면서 ㅋㅋ 근데 이렇게해주면 더 좋을거같애! 이런식으로 얘기하면 나중엔 그렇게 하더라구요 ㅎㅎ'], ['우리신랑은 애교 안통해요ㅋㅋㅋ\n뭐든 다시시키면 개짜증ㅜㅜ\n자상한 남편분 두셨네요^^'], ['자상하진...시키면 짜증도 내는데.. 꾹 참고 시키는거죠 ㅠ'], ['전 집안일 안시켜요..어차피 제손이가야하기에 투덜거리면서 제가다해요..그대신 잔소리를 어마어마하게 쏘아요ㅋㅋ그래야 속이시원해서ㅋㅋㅋ'], ['ㅋㅋㅋ부럽습니당ㅋㅋㅋ\n저는 잔소리하면 배로 돌아와서 꾹참고 지켜봅니다ㅠㅠ'], ['저희는 저희신랑이더깨끗해요ㅋㅋ\n해줄때마다 고마워~~라고해줘요'], ['넘 멋진 신랑 두셨네요~~!!\n저도 잘 못하는데 저희신랑 두배세배 심해요ㅠㅠ'], ['다하고 나면 항상고맙다 이쁘다 최고다 궁디팡팡해줘요\n그리고 하나씩 여보 요건 담에 이렇게 해주면안될까요??  요렇게 기분 안나쁘게 얘기하면  담에 신경쓰더라고요\n맘에는 백프로안들어도  나아지고있어서 항상 칭찬해주고 고맙다고해요\n진짜 몰라서 못하는거니깐요ㅎㅎㅎ'], ['진짜 몰라서 못하는거겠죠?ㅠ\n기름때가 남은게 눈에보이는데ㅠ\n거름망 음식물 비우면서 시커먼 곰팡이도 보이는데 모르는척하는거 아닐까 의뭉스럽습니다ㅋㅋㅋ'], ['보통 몰라요\n설겆이는 딱 그릇만 씻는거라 생각하는 사람들이에요ㅋㅋㅋ\n주변에 물기도 같이 닦아야한다는걸 3년차에 깨달은거같더라고요ㅋㅋㅋ\n다행인건 제가 더 게을러서 참을만해요ㅋㅋㅋ\n근데 요샌 진짜 살림잘하는남자들이 많더라고요ㅎㅎ'], ['저도 제가 더 게을러서ㅋㅋㅋ참습니다..\n살림잘하는 남자가 존재한다는게 신기합니다ㅎㅎ'], ['울팀 남자분들은 애기들 목욕부터 간식 저녁담당하더라고요\n차리는 정도가 아니라 간식 만들어 먹이고 와이프 퇴근하면 밥상차려주고ㅋㅋ\n또 다른분은 와이프 도시락 까지 싸주는ㅎㅎ\n잘하는 사람도 존재하긴 하는듯해요ㅋㅋㅋ'], ['진짜 조상이 덕을 쌓아야 그런분 만날수있을까요ㅋㅋㅋ\n어느날 붕어빵틀을 사더니 핫케익을 구워줬는데 1년에 한번 얻어먹어요ㅋㅋ\n올해는 말만하고 아직안해줬는데 왜샀냐고 타박한번더 해야겠어요ㅋㅋㅋ'], ['타박하지말고 해주면 넘 좋을거 같다고 먹고싶다고 하면 기분좋게 해줄거같아요\n단순하잖아요ㅋㅋㅋㅋ'], ['붕어팬 내다버린다 협박했는데ㅋㅋㅋ\n쉬는날 먹고싶다고 꼬셔봐야겠네요ㅎㅎ'], ['남편이 저보다 더 잘하는거 같아요~ 제가 집안일을 잘 못해서...ㅎㅎ'], ['저는 저도못하는데 더한놈을 만나부렀어요ㅠㅠ'], ['저도 신랑이 더 손끝야무져요~근데 속도가ㅋㅋ 답답할때도 있어요ㅋㅋ'], ['느려도 좋으니 야무지게 끝내주면 소원이 없겠습니다ㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋ설거지 기본 1시간입니다ㅋㅋㅋㅋㅋ 그래도 도와주는 편이라 감사히여기고 삽니당ㅋㅋ'], ['한시간이고 두시간이고 완벽하게 해주면 고마워서 눈물날거같슴니다ㅠㅠ부러워용~'], ['설겆이 한시간 걸려요 \n고마 제가 합니다 \n방 걸레질 함해주믄 입 댓발 나와요 \n고마 제가 합니다\n쓰레기버리기 청소기 돌리기하믄 뻗어서 못일어납니다\n고마 제가합니다\n간혹 해주는데 \n고마 제가하는기 속이편해요\n'], ['ㅋㅋㅋㅋ마자요 고마 제가하는게 속편한데 넘 힘드네요 집안일이ㅜㅜ\n남편분 청소기돌리고 뻗는다니 넘 웃겨용😂🤭'], ['신랑 야무지게 한다고 하는데~하수구는 늘 청소안하더라구요~그래서 한번은 물어봤어요 왜 안하냐고 그니까 엄마가 그건안시켜서 안해도 되는건줄알았다 하더라구요ㅋㅋ셤니엄청 깔끔하신분인데 설거지만 시키고 그런건 안시켰나봐요~그래도 집안일 해주는게 어디냐며ㅠㅠ'], ['마자요 해주는게 어디냐며 되뇌이며..하수구청소를 했네요ㅎㅎㅎ'], ['남자들 거의 그렇지 싶어요 ㅠ\n전 그냥 하지말라고 해요^^제가 하는게 속 편하고...'], ['저도 제가하는게 속편하긴한데 넘 게을러서 큰일입니당ㅜㅜ'], ['나름 깔끔하게는 하는데 제가 하는 방식으로 정리하지는 않아서 다하고나면 정리해서 물기 빼요...'], ['깔끔하게만 해주면 정리는 안해도되는데ㅜㅜ\n1도 안깔끔합니다ㅠ'], ['저도 임신초긴데 저희신랑은 1도 안해요ㅡ평일엔 저도 상관안하는데 주말에 제가 밥차리고 설거지하고 분리수거하고 청소 다할동안 누워있는거 보고 진짜 그냥 나갔으면 싶다는...'], ['저도 그꼴보기싫어서 더럽게해도 하는게어디냐며 참네요ㅠㅠ\n임신초긴데 힘드시겠어요ㅠㅠ'], ['저희신랑은 너무 꼼꼼해서 설걷이 하루종일해요 ㅋㅋㅋㅋ\n물은 또 졸졸졸 틀어노코 세월아 네월아 네요 ㅋㅋ'], ['세월아 네월아 해도 꼼꼼하게 서있는걸 보고싶어요ㅋㅋ설거지를 5분만에 끝내는데 제대로 씻은건지ㅠㅠ'], ['진짜로 성향 다르면 너무 피곤해요 ㅜ ㅜ\n저희남편도 정말 안치우고....\n손만 댔다하면 온천지 난리가 ㅜ ㅜ\n설거지 하고 남 주위 물바다되고 ㅜ\n너~~~~무 승질나서\n집안일 1도 안시킨지 8년 넘었어요 ㅋㅋ\n일주일한번 분리수거. 이거 딱하나 해달라합니다~\n꼼꼼한 남편들 부러워요 ㅜ ㅜ ㅜ ㅜ'], ['마자유ㅜㅜ저흰 게으른성향은 똑같은데 저는 한번손대면 꼼꼼하게 끝내야해서 피곤하고 신랑은 대충 빨리 끝내는 성격이라ㅜㅜ\n싱크대밑에 물바다 공감되네요ㅠ\n원래 분리수거도 제몫이었는데 해주는게 어디냐며ㅜㅜ'], ['저보다 잘해요😭 정리를 못하는데 정리를 각맞춰 너무 잘해요ㅋㅋ'], ['짱부럽습니다ㅜㅜ\n저도 살림에 소질이 없어서😭'], ['잘안해주지만  한번해주면  잘해요~특히  설거지! 행주까지 빨아서 척 널어놓고ㅎㅎ'], ['한번할때 제대로해줌 얼마나 이쁠까요~ㅠㅠ\n행주는 상닦고 고대로 싱크대위에 방치...'], ['엄청 열받게 해서 부부싸움하면  해줘요~~이걸 조타고 해야하는건지 ㅋㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋ우리신랑은 싸우고나면 이상한음식을 해주던데 설거지나 제대로하지ㅡㅠ'], ['머든 하려고  한다는 노력에  만족합시다ㅋ'], ['그럼유~~ㅎㅎ안하는거보다 낫죵~~^^']]
    
    4849
    집안일 하나 클리어~ 이제야 집안일 하나 클리어네여~아직 다하기누멀었지만여~다행이 셋찌가 자서 이것도 금방했네여~빨래는 이불만 있으니~ 어여 돌리고 널어야 겠어요~
    
    [['빨래 하나만 비워내도 큰 일 치룬 것 같아요.ㅎ'], ['이불빨래라 안할 수 가 없더라고요 ㅠㅠ'], ['전 이불은 모아서 그냥 빨래방에 가요.ㅎ'], ['정말 이불빨래는 빨래방이 편하긴해여~ 주말에 빨래방 가야할까요!? ㅠㅠ'], ['만원씩 돈 내는 거 아까워서 건조기까지 샀는데...\n용량이 작으니까. 확실히 빨래방이 편하더라구요.'], ['건조기 사셨구만요~ 만원씩 내면 아깝죠~ 근데 빨래방은 크다보니 편하더라고요~'], ['그렇더라구요.ㅎㅎ 그래서 지난 번에 첫째가 이불에 지도 그려놨을 때도 그냥 빨래방 갔어요.ㅎ'], ['집에서 하는것보다는 정말 빨래방이 편하긴 하더라고요~ 저도 빨래방 잘 이용하는 편이라~ㅎㅎ\n집에 기계가 있어도 말이져~'], ['하나 클리어 ㅎㅎ \n전 아직 시작도 못했어유ㅠ'], ['빨래랑 청소기만 완료네요~ ㅠㅠ'], ['집안일이 진짜 시간 잘갑니당ㅎㅎ'], ['그니깐요 ㅠㅠ 시간이 어떻게 빨리빨리 가는지ㅠㅠ'], ['저도집안일이 산더미여요ㅜ\n이불돌려놓은게다여요'], ['저도 이불만 ㅠㅠ 큰애가 실수 해서요 ㅠㅠ'], ['저도 집안일 해야하는데 일단 설거지랑 청소기만 돌려놓고 앉아있음다. ㅎㅎ'], ['ㅋㅋㅋ 전 빨래랑 청소기만 완료했어요~ㅎㅎ'], ['전 빨래도 내일로. 미뤘지요 ㅎㅎ'], ['잘하셨어요~~ 피곤하시면 내일로~~ㅋㅋ'], ['ㅋㅋ그치요 오늘안한다고 큰일안나니...ㅠㅋ'], ['넵넵 ~ 쉬엄쉬엄 내가 하고 싶을 때 하면되요~ㅎㅎ'], ['ㅎㅎㅎ 언제가는 해야하는데 또 미루고 있네요. 이따가 낮에는 as기사님도 오신다해서.. 김치냉장고 수리좀 하고.. 빨래도 하고 집안일 시작해봐야지요 ㅎㅎ'], ['그니깐요~ 저도 여름옷 정리해야하는데 안하고 있어요 ㅠㅠ'], ['5 앗 그래요 얼른 하셔야 겠는데요. 애들 패딩이나 완전 겨울옷은 안꺼냈는데 이제 슬슬 꺼낼까봐요. 세탁해놓고 하려구요'], ['그러게요~ 정리는해야하는데 왜이렇게 옷들이 많은지 말이져 ㅠㅠ'], ['이제 강제취침 들어가야지요. 불다끄고요 ㅎㅎㅎ'], ['강제취침 해야죠~ 빨리 자주면 좋을텐데 말이져~'], ['저도 아침에 애들 등원하나 시키고 빨래 돌리고 등원보내고 널었네요'], ['저는 등원시키기전에 빨래 돌려 놓고 나가요~~ㅋㅋ'], ['등원을 직접 데려다주시나요'], ['아니요 1층으로 차가 와서 1층까지만요~ㅎㅎ'], ['아하 1층까지 ㅎㅎ 저도 집 밑으로만 내려가서'], ['?넵넵~ 차가 집앞까지 오니깐요~ 1층에 있는 어린이집 다니나요?'], ['저는 빌라라 근처 아파트단지내어린이집요 ㅎ'], ['아~ㅎㅎㅎ 그럼 거리가  있구만요~ 매일 등하원기키시는건가요?'], ['집에서 멀지는 않은데 횡단보도를 건너야 해서 아침마다 바쁠 거 같아서 차량이요'], ['차량이 편하긴 하더라고요~~'], ['늘 많은 집안일...ㅠㅠ 끝이없지요'], ['그쳐 ㅠㅠ 정말 끝이 안보이네여 ㅠㅠ'], ['빨래 개고 하면 기분 너무너무 좋지용?'], ['넵넵~ 뭔가 일이 빨리 끝나는것 같아요~ㅎㅎ'], ['그러니까영. 그 빨래개는 걸 저는 이제서야 해서 말이져T'], ['ㅠㅠㅠ 저도 내일 개야할빨래들이 산더미네요 ㅠㅠ'], ['저는 이미 다 빨래는 개서.. 내일 또 갤 것이 저기 보이지만요~ 말라가네여'], ['저도 말라가는 빨래가 저기 보이네요~ㅎ'], ['집에 가면 저도 해야죠. 할일 투성이거든요ㅡ'], ['외출하셨구만여~ 저는 집안일 오늘은 하나도 못했어여 ㅠㅠ'], ['5. 오늘 하나도 못했어영? 저는 일단 빨래는 다 해두었답니다.'], ['넵넵 ㅠㅠ 셋째가 안도와줘서요 ㅠㅠ 카페놀이도 못하고요 ㅠㅠ'], ['할 일이 넘쳐나는 집안일이지요'], ['그러게요 할일이 넘쳐나네요..'], ['그렇지용 저도 태산이에요 할일이.ㅠ'], ['근데 몸이 안따라져서 문제인것 같아요~'], ['엄마들은 정말 일이 산더미라 끝날기미가 안보여요 ㅠㅠ'], ['맞아용 카페까지 하니 원.ㅋㅋㅋㅋ'], ['맞아요~ 카페 일을 하니 더 못하는것도 있는것 같아요~'], ['마자용 그래서 이번달에만 집중해야하나싶어여;ㅋㅋ'], ['그러게요~ 저도 고민이 되더라고요 ㅠㅠㅠ'], ['아기가 어릴 수록 더 빡세요'], ['그러게요 ㅠㅠㅠ 갈수록 빡세지는것 같아요ㅠㅠ'], ['빨래 깔끔하게 잘 개네요!! 우리집으로 올래용?ㅋㅋ'], ['ㅋㅋㅋㅋ 저희집 빨래도 정말 감당하기 힘들답니다 ㅠㅠ'], ['앜ㅋㅋㅋ 제가 가서 도와줄게영!! 초대해줘영'], ['ㅋㅋㅋ 쭌이맘님도 힘들잖아요~ㅎㅎ\n언제든지 놀라오세요^^'], ['담주엔 정말 만나요!! 일정 한번만 보고 날짜 잡아서 만납시당'], ['넵냅~ 일정체크하시고 연락주세요~ㅎㅎ'], ['담주는 수목 괜찮을 거 같아용!! 제가 갈게영ㅋㅋ'], ['ㅋㅋㅋ 언니가 놀러어세오 ~ 막이러고~~ㅋㅋㅋ'], ['그럼요 ㅋㅋ 달려갈게영ㅋㅋ 놀러나갑시당ㅋㅋㅋ'], ['달려 오세요~ㅎㅎㅎ 기달리고 있겠습니다~ㅎㅎㅎ'], ['네^^ 좋아영ㅋㅋㅋ 곧 만납시다용']]
    
    4922
    꼬리에 꼬리를 무는 집안일 청소기 돌리고 나니세탁기가 끝나 있고건조기 돌려놓고 나갔다 오니설거지가 쌓여 있고설거지를 하고 나니건조기가 끝나 있고건조기 빨래개고 나면또 무엇?
    
    
    
    
    동영상
    
    
    Brown Eyed Girls 'Abracadabra' (Performance Version)
    2010 한국대중음악상 Korean Music Awards (Best Dance/Electronic Album of the Year : Sound-G) 2010 한국대중음..
    youtu.be
    
    [['간식준비? ㅋㅋㅋ\n\n안하면 표나고 해도 표안나는게 집안일이죠~'], ['큰애는 간식먹을 나이 지났고\n다행히 꼬맹이는 친구집에 놀러갔어여ㅎㅎ'], ['집안일도 도와주십니까? 이상적인 남편상 입니돠~~'], ['맞벌이인데 니일내일 하긴 그렇잖아여ㅎㅎ;;'], ['ㅋㅋㅋㅋㅋㅋㅋㅋ\n살림꾼쥬도님이시네요ㅎ\n저희집 남자는 살림은 1도못하셔서..\n이제 자고일어나셔서 식사하시능..오매..반품해버리까..ㅡㅡ'], ['아래와 같은 경우 반품/교환이 불가능합니다 \n\n1. 반품요청기간이 지난 경우\n2. 구매자의 책임있는 사유로 상품 등이 \n멸실 또는 훼손된 경우\n... ... ... 등의 사유\n\n결정적으로\n5. 시간의 경과에 의하여 재판매가 곤란할 정도로 \n상품 등의 가치가 현저히 감소한 경우...'], ['시댁이 존재하는 경우..반품가능하다 들었습니다만..'], [''], ['집안일 도와주시는군요. 멋지다요~ 집안일이 원래 그렇다죠. 집안 어지르는 꼬맹이 있음 더한 챗바퀴돌기 청소!!'], ['나중에 큰애가 결혼해서 맞벌이 하는데...\n독박쓰면 기분이 나쁠 것 같아여...\n\n돕는다기 보다는 같이 해야져ㅎ'], ['음마! 쥬도님 멋찌다\n집안일은 돕는게 아니고 같이하는거래요...라고 할랬는데 이미 답을 알고계신..\n근데 우리집엔 같이도 아니고 도움도 안주는 ㅋㅋ 에이야~띠~ 삐삐삐!!!'], ['일단 사귄다고하믄 면접부터 보세유~~~'], ['어? 정답인데? 우리 행사에서 이렇게 하믄 건조기 받았!!!!'], ['빨래개고 이불털기 고고??'], ['여름이불은 다 끝냈슈'], ['화장실 청소~'], ['앗!!~~ㅋㅋㅋㅋㅋ~~👍'], [''], ['그렇다는군요!  ㅎㅎㅎ'], ['유니님 비결 좀요!! \n쉬지 않고 돌릴 수 있는 비결요 ^^;;'], ['안 까먹었네?'], ['음... 일단 자격증 셤 하나를 신청하시고 휴일날 집에서 나오면 됩니다?? ㅎㅎㅎㅎㅎㅎ'], ['전 그것보다는 유니님의 미모가 아닐까 하고 생각하고 있었어요^^'], ['헐.... 모닝문님 뭐가 필요하신가요???!!! 뭐 드시고 싶으신거라도~~~ ♡.♡'], ['쥬도님 너무 자상한데여??ㅋㅋ'], ['딱 밥때됩니당 ㅋㅋㅋ'], ['저녁은 그래서 나가서 먹으려구여ㅋㅋ'], ['베란다 청소..맛동산 채취...'], ['즤집은 확장형이라 다행히 베란다가 읎슈...\n맛동산은 이따가...\n냄새도 맛동산이면 좋으련만...'], ['커피한잔 하시구.. 한숨 주무시구 저녁하셔야겠네요..\n멋진남편!!\n\n(착한사람만 보이는 댓글)'], ['인쟈 운동 끝났으니 꼬기 먹으러 갑니당'], ['집에선 쇼파에 누워서 \n여보~~\n물~~\n여보~~ \n리모컨~~\n이래야져~\n어디 건방지게 여자일을 넘봐여?? ㅋㅋ\n'], ['에?'], ['그게 여자일인가요?? \n그럼 전 여자 안할래요.'], ['인제 저녁해야죠 ㅋㅋ 주부쥬도님'], ['쥬부인가여ㅎㅎ;;\n저녁은 나가서 먹으려구여~\n힘듦... 먹구와서 장실청소예약'], ['재활용 버리는 날이더라구요 우린 오늘ㅋㅋㅋ'], ['그래서 ㅅㅂㄴ 찬스 쓰셨나여?ㅎ'], ['그분일입니다!(단호)'], ['어므나 암것도 하기 싫으시다더니 이렇게 집안일을 착착하고 계셨네용👍👍👍'], ['뭐그냥ㅎㅎ...\n인제 다하고 자려고 누웠슴다~'], ['아이들 숙제 봐 주기. ^^♡ 준비물 챙기기~ 끝이없네요. ㅜㅠ'], ['그건 교육이 다 되어 있어서ㅎㅎ;;\n큰애는 고2라서 손이 안가고\n둘째는 초4인데 교육이 잘 되어 있슴다~'], ['외식하고 들어오심서 장봐오기! \n하면 끝날듯요^^'], ['장은 주말에 보는 거지여~ ㅎㅎ'], ['멋진 남푠이시네요~~~ \n다 같이 해야쥬~~~ 같은 가족인데 니일내일이 어딨겠습니다^^ 리스펙~~'], ['맞벌이 하는데... 독박은 좀 아닌듯 하여ㅎ\n시간 남는 사람이 걍 하는거져'], ['부지런하시고 자상하신걸로 ㅎㅎ \n\n'], ['뭐 별로 그렇지도 않아여ㅋ\n게으릅니당~'], ['커피누나는 참...\n그래도 감당이 되니까 하시는거져?'], ['집안일이...\n그렇더라구요 ㅠㅜ\n해도 해도 딱히 테도 안나는데\n쌓이기는 어쩜 그리 쌓이는지 ㅎ😂😂😂😂😂'], ['맞아여ㅋㅋ\n무섭게 쌓입니다~\n밥 뭐 할지 생각하는 것도\n큰 스트레스 더라구여'], ['쥬도 홧팅~'], ['아자아자~\n흑조누나도 화이팅~♡']]
    
    4940
    18 집안일 시작 오랜만에 하는기분입니당^^음악 틀어놓고 일단 1시간 움직여봐유♡할게많습니다
    
    [['화이팅 입니다. 천천히 쉬엄쉬엄 집안일 하셔요~'], ['도저히 못하겠어서 그만 하기로 했어여 ㅋㅋ'], ['ㅎㅎㅎ힘들지요ㅎ\n저도 애들방청소는 담주신랑있을때로..ㅋㅋ미뤄봅니다요ㅎ'], ['전 월욜에 걸레질 위주 한번 해볼까봐염ㅋ'], ['전.. 장난감정리.. 아주 작정하고 담주중에 다 끝내야지요 ㅎㅎ'], ['장난감이 엄청 많이 있는가봐용ㅠ'], ['자잘하니 엄청 많이 있지요 ㅎㅎㅎ'], ['저도 드림받고 사고하니ㅠㅋㅋ\n아직 못치우는게 슬프네여'], ['5  잘정리해서 꾸며줄라구요ㅎ'], ['방을 꾸며줄 계획 이시구만용'], ['네 이사를 가려다가 좀더 살기로 해서 그럼 방을 제대로 꾸며주자 해서.. 도배부터 싹 할 생각이지요 ㅎㅎ 지금은 너무 유아틱해서... 이제 학교 입학하니... 공부방으로요 ㅎ'], ['음악 틀어두고 하면 집안일 할 때 쫌 신이나죠'], ['유투브 연결해서 하니까 근데 자꾸 끊기고용 ㅠㅠ'], ['이런.. 왜 자꾸 끊길까여... 흐름.. 이어져야 하는데 ㅠ'], ['긍게영ㅋㅋ월욜엔 라디오를 들어볼까염ㅋ'], ['저는 지금도 라디오를 듣고 이써영 ㅋ'], ['그런거 좋은거 같아여 ㅋㅋ 핸드폰으로 트시는 건가여'], ['라디오 가지고 와서 켜기 귀찮을 때 종종 폰 잘 이용해요 ㅋㅋ'], ['라디오가 있긴 하시군용ㅎㅎ'], ["그럼요! 집에 떡하니 라디오가!ㅋㅋㅋ'친정 부모님께서.... 뚜정이 5월 5일 어린이날 선물로 사다주셨어용;;; 음악 많이 들으라고.. 뚜정이.. 5월 24일 탄생했는데.. 미리 사심..ㅋㅋ'"], ['5 라디오를 어린이날 선물로..ㅋㅋㅋ'], ['저도 빨래하고 청소기 돌리고 햇네요 여기까지만 ㅠㅠㅋ'], ['전 이것저것 했더니 티도 안나네여ㅠ'], ['이것저것 많이 하셧을텐데 어찌 ㅠㅠ'], ['워낙 집에 물건이 많아서 그런것 갇기도ㅠ'], ['저희도 집에 뭔 짐이 이리도 많은지 원 ㅠㅠㅠ'], ['전 자꾸 사기만하고 안버려서ㅋㅋ'], [': 저도 그래요 ㅋㅋ 신랑이 좀 버리고 사라고 ㅋㅋ'], ['ㅋㅋ 비움을 먼저 해야하는데 채움을 하니 멀 버려야 할지 모르겠고용 ㅋㅋ'], ['5) 그러게요~버리려니 아깝기도 하고 ㅋㅋ'], ['5 언젠가 쓸것같은 그런느낌이지용ㅎ'], ['저도 블루투스 스피커로 노래 잘 들어영 ㅋㅋㅋㅋ'], ['ㅎㅎ담주엔 라디오를 좀 들어 볼까봐염ㅎ'], ['ㅋㅋ 라디오 재밋나용 ㅋㅋ 웃기나용'], ['웃기는 라디오도 있지용ㅎㅎ사람사는 예기들음 좋을때가 있더라그영ㅋ'], ['왠지 훈훈한 이야기도 많을거 가타용 ㅋㅋ'], ['아침엔 특히 그런거 같아용 ㅎㅎ\n오후엔 직장인들 잠깨야 해서 신나는걸루'], ['아아 그렇군용 아침에 한번 들어보아야겟네용ㅋ'], ['저도 월욜아침에 어플받아섬ㅎㅎ'], ['5 오오 그러시군용 오늘이 토요일이네용 ㅎㅎ'], ['넹ㅋㅋ내일이지남 열무는 등원하고ㅎㅎ'], ['음악 틀어놓고 집안일 하면 신나죠~'], ['특히 1900년대~2000년대초반이 신나유'], ['아마 우리(?)시대 노래라서 그런 거 아닐까요?ㅎㅎㅎㅎ'], ['그런가봐영ㅋㅋ요즘껀 다 모르겠어여ㅋ'], ['요즘 아이돌 누가누군줄도 모르겠고~ 어제 티비에서 슈퍼주니어 보니 반갑더라구요 ㅋㅋㅋ'], ['꺅ㅋㅋ저도 그거봤어용ㅎ\n어젠 백지영이랑 김태우 손호영도 나왔어유'], ['음악들으면서 하는 건 그래도 좋은거 같아요'], ['마쟈영ㅋㅋ항상같은음악 이지만ㅋ'], ['흐흐 그래도 음악들으면 좋은거지요 전 들어본지가 언제인지여'], ['곧..티비틀고 할것 같습니당ㅎㅎ'], ['흐흐 그럴려나요 여유를 즐기고 싶다지여'], ['지금 즐기세용 ㅎㅎ 들어가시면서 커피도 한잔'], ['그러고 싶었으나 빕스를 가야 했길래 불가했던 ㅋㅋ'], ['미리예약하신 빕스ㅋ델러오셨으니 좋지용'], ['그지요 ㅋ 집에서 좀 거리가 있어서 안오면 살짝 난감할뻔요ㅎ'], ['5 대기하고 계셨을것 같습니당 ㅎㅎ']]
    
    4966
    집안일 안하는데 잔소리도 안하는 남편 어떠세요? 저희 남편 ㅋㅋ 집에서 손가락 하나도 까닥 안하는데 (아기 목욕시키는거만 도와줘요 저 혼자 시키다가 애기 다칠뻔 했어서) 집 너무 더러워도 잔소리도 절대 안하고 맨날 반찬 사먹고 배달시켜먹어도 싫은 소리 안해요 ㅋㅋ 애기랑 안놀아주는건 좀 그렇지만 개인적으로 저는 누가 제 살림 건드는게 너무 싫어서 저희 남편이 좋은데 저희 부부 성격이 잘 맞는거죠? 몰스님들은 저런 남편 어떠세요?
    
    [['저희신랑도그래요 ㅎㅎ 전 좋더라구요~~~ 마음도 편하고..'], ['좋아요 저도 울 신랑도 손가락 까딱 안하는대신 잔소리 안해요 세상에서 제일 싫은게 잔소리라 ㅎㅎ'], ['저두 맘이 편해서 좋던데요ㅎㅎ 제가 쉬엄쉬엄 해요 시간될때~ 잔소리 안하니 여유있어 좋아요~'], ['몰아서 청소하는 스타일인데 뭐라 안 해요, 본인이 힘들면 먼저 청소해주기도 하구요. \n물론 잔소리를 없습니다.  그게 궁합인거 같아요. \n'], ['좋은거예요~'], ['어차피 안해줄거면 잔소리라도 안해야 좋은 남편인거죠~ㅋㅋ 좋은데용...'], ['엄청 도와주는데 잔소리도 해요. 살살구슬리면 잔소리 안할거같기도한데 그런 여우짓은 다시태어나야하는 관계로.ㅋㅋㅋㅋ 저도 그냥 안도와주고 잔소리안했음 더 나을거같아요.ㅋㅋㅋ 근데 또 하는게 저보다 많은사람이라.....ㅋㅋㅋ'], ['저는 잔소리 안하는데 \n도와달라고 시키게되더라고요ㅠㅠㅠ\n댓보고 반성하고가요;;'], ['도와주면서 잔소리도 안하는 이가 있기는 하겠죠?? 근데 확률이 적을테니..저희집도 손 하나 까딱 안하는데 터치도 없어요. 댓글들 보니 긍정적으로 생각해야 겠네요.'], ['네 좋아요. 저희집도 잔소리 1도 안해요. 저도 남편에게 잔소리안해요. 평화롭습니다^^ \n제살림 건드는거 넘 안좋아해서 관심꺼주는게 더 좋아요. 그래도 쓰레기버리기 거실화장실청소는 꼬박꼬박해줘요  그외 집안일은 안시키고 자발적으로도 안합니다 ㅎㅎ'], ['잘 맞는것 같은데요\n우리집도 잔소리 안하고 집안일 안해줘도 마음 편해서 좋더라구요'], ['설겆이도 빨래도 청소도 밥도 제가 안하면 그냥 본인이해요~ 뭐라안해요~ 신랑이 시작하면 같이하고 제가 음식하거나 집안일 시작해도 둘이 같이해요~ 좋죠'], ['저희 신랑도 그래요. 그런점은 정말 맘에 들어요. ㅎㅎㅎ'], ['우리집도요.ㅋ 군데좀 도와줫음 좋겟어요 ㅜㅜ'], ['저희남편도 잔소리안해요. 집안일은 화장실청소 가끔분리수거정도..  더시키고싶어도 워낙 바빠서 못시켜요. 돈만 잘벌어오라하네요. 대신 쉬는날 육아는 잘합니다.'], ['저희남편이요..전 좋아요'], ['열심히도와주시는분 잔소리도 많으시더라구요ㅜ'], ['그러고보니 저희 신랑도 집이 정신없고 지저분해도 잔소리를 안하네요ㅋㅋㅋ 고마워해야겠어요^^'], ['제가 하면 되니까 안하고 잔소리 안하는 사람이 좋아요\n어차피 신랑이 해도 제가 한번 더 해야 되는 성격이라 그냥 어지르지만 말아주라......😑'], ['저희남편이 그래요ㅋㅋㅋ 집이 지저분하든 반찬이 없든 뭐가됐든 일절말안해요.대신 자기도 암것도 안해요. 제가 얘기해야 쓰레기버리기 박스버리기 정도?시키면 하지만 굳이 하진않아요..그런데 제가 어찌해도 말없으니 차라리 편해요'], ['두분이서잘맞으니좋은거요~~^^'], ['집안일안하면서 잔소리하는 남편보다는 이쁠거같네여;;;'], ['잘맞는거죠~~'], ['울남편도 집안일이든 뭐든 잔소리 1도없어요~ 근데 전 좀 도와줬으면 좋겠어요ㅋ 도움받는건 사실 포기했고...  그래도 애들 공부는 자알 봐주니 이걸로 만족합니다^^'], ['저희집도 그래요,,집안일 하나도 안하고, 잔소리도 안해요...잔소리 안하고 집안일은 하는 남자였으면 하는 바람이있어요,,ㅋㅋ'], ['도와주면서 잔소리 열심히 하는 저희 아주버님보다 안도와주고 잔소리없는  저희 남편이 차라리 나은거같아요. 둘다 진짜 싫다 ㅋㅋ'], ['ㅋㅋㅋㅋㅋ 공감요ㅋ'], ['급반성하고 갑니다요 ㅎㅎ ㅎ전 맨날 배달음식에ㅡ인스턴트 막 주고 집도 잘 안치우는데 그에 비해 잔소리 안하고 다 먹고 그냥 살면서 애도 다 씻기는거 전담하는 남의 편인데 전 제 성에 안차서 맨날 시키네요 ㅋㅋㅋㅋ두분이 서로 만족하면 잘 맞는거죠'], ['여기 한 명 더 있어요 ㅎㅎㅎ 전 잘 맞는지는 모르겠지만요 ㅎㅎㅎ\n제가 잔소리 할때마다 저테 잔소리 안하는데 전 왜 잔소리하냐 싸웁니다... \n울 새언니 왈 잔소리하고 청소잘해주는 울 오빠가 더 나은것 같아~~ \n'], ['죄송해요. 잔소리도 안하고\n집안일( 청소, 빨래, 건조기돌려 개놓고,설거지, 애들 목욕까지 )다하는 남편있어요^^; 참 쓰레기도 버려요'], ['훠우 전생에 나라를 구하신....'], ['저희 신랑이네요. 본인도 드럽고 집도 드럽고 아침밥 안해주고 저녁은 배달이라도 그저 잘해줘요. 근데 본인은 손하나 까딱 안해요.'], ['집안일 안하는데 잔소리 하는 남편과 살아요ㅜ'], ['헐.. 소름...우리 신랑이 자신을 항상 저렇게 어필하져..  내가 하려고 하면 잘 할수 있지만 그럼 눈에 보여 잔소리 할거라구.. 선택하라더군요.\n그래서 잔소리말고 내비두라했어요ㅋㅋㅋ'], ['신랑 첨엔 잔소리안하더니 나이먹더니 말이 늘었어요~~~~'], ['신혼때 시댁 집들이하는데 어머님이 보시고..이제 집정리만 하면되겠네~하시는데.\n울부부 동시에 다한거에요~했더니..\n어머님왈..그래 둘이 똑같으니 됐다~~ 하시던게 기억나네요 ㅋㅋ'], ['저희남편도 안해요 스스로해요  ㅋㅋㅋ요리맛이없어도 잔소리안해요 \n포기한듯요 ㅠㅠㅠ 근데 티비보고 있음 제가 시켜요 ㅋㅋㅋㅋㅋ'], ['저희랑 비슷해요. 둘다 게으릅니다. 울엄마가 맨날 그래요. 너 참 다행이라고'], ['찰떡궁합'], ['아 근데 다들 넘 착하신거 아닙니까. 저희남편 저런st 인데 저는 가끔 화나거든요. 좀스스로나서서 해주면 좋겠고 같이사는집이고 같이낳은 아기들인데!!'], ['잔소리 안하면ㅎㅎ\n안도와줘도 부담없을것같은요ㅎㅎ\n도와주면서 잔소리가 백만배싫어요'], ['두분이 잘 맞으면 최고죠~'], ['최고네요'], ['그나마 괜찮은거죠 ㅋㅋㅋ\n근데 육아는 같이 하는걸로 하세요^^\n집안일은 남의일처럼 보면서 잔소리하면 진짜 때리고싶어요 ㅋㅋㅋㅋ'], ['저도 전업인데 잔소리 일체 없고.. 전 하는 거 못하게 해서 주방은 거의 안건드리는데, 퇴근하면 애들 장난감 치우고, 책은 쌓아놔요. 이곳저곳 막 꽂아놔서.. 그것도 하지말라 해놔서ㅎㅎ'], ['아, 쓰레기는 남편담당이네요~'], ['저희 신랑도 그러세요\n안도와  주면서 잔소리 하면 싸다구 날려야죠'], ['회사에서 와이프 욕은 하시던데요 ㅜㅜ\n집도안치우고 더럽다고 ㅜㅜ\n깨끗한여자랑 살고싶다고.\n이번생은 틀렸다며.\n근데 본인도 손끝하나 안거드나봐요.쳇'], ['저희 남편은 여초회사라 그런 소리하면 인간쓰레기 취급당해서 못해요 ㅋㅋ 다행~ 휴'], ['저희랑 비슷하시네요ㅋㅋ'], ['집안살림 하면서 시시콜콜 잔소리하면 미칠듯요ㅋㅋㅋㅋ\n저희 남편은 딱히 말 없어요~ 넘 맘편해요ㅋㅋㅋ'], ['저게으르고 지저분해요ㅠ\n다행히 남편잔소리안해요 ㅋ \n남편이 정리정돈 잘하고 청소도 저보다 잘해서 자주는.아니지만 애들재우러 들어가면 혼자 청소나 설거지해놔요~\n처으엔 그게 화가나더라구요ㅠ\n진짜완벽하게 노터치이길바랬던거 같아요 ㅋ 조용히.스스로 하는게 무언의 압박처럼 느꼈나봐요 ㅋ 근데 10년넘게 살다보니 그냥.고마워하고 넘기고살아요 ㅋ'], ['ㅋ저희신랑은 왜집을 치우냐 화장실청소도 왜자주하냐 왜 이불을 일주일에한번빠냐 드럽게 살고 힘들다 아프다 하지마라  이래요 ㅎ 게으르고 귀차니즘 신랑이지만 잔소리안해요 뭐 치우는건 제가 하고 싶어 하는거지만 가끔은 힘들어 누워있는거 보면 화딱지가ㅠ 납니다요'], ['집안일 안하고 잔소리도 안하고 음식물,쓰레기,분리수거,화장실청소정도만 하는데 이게 편하네요. ㅋㅋ'], ['저희 남편도 그렇긴 한데, 전 가끔은 도와줬음 싶더라구요.\n전 청소 귀찮아서 청소 도와주면 좋겠는데\n요리 해줘요. 가끔 ㅋㅋ\n잔소리 안하니 감사하고 살아야겠죠? ㅜ'], ['우리집계시네요... 그래도 한번씩 시키면하니 괜찮아요...    자기주도학습은 못해도 시키면해요... 시키면... 애야뭐야..'], ['전 그걸로 많이 싸웠어요\n애들 케어뿐아니라 분리수거 하나도 안해줘서 많이 힘들었어요 \n지금은 다른분이 청소해줘서 싸울일이 하나도 없네요 ㅎㅎ'], ['집안일 안하면서 잔소리 하는 남편보다는 훨 낫죠~ㅋㅋ'], ['저희신랑이네요\n잔소리안해요~자기 시킬까봐~\n딱 시키는것만해요~'], ['본인이 대부분 하는데 저보고 잘 못한다고 구박하는 남편은 어떤가요? (맞벌이에요) 대부분은 고맙기도 하고 몸은 편한데 가끔씩 구박당할 때 짱나네요'], ['항상  새벽에 일어나고\n일어나서  정리정돈  청소기 살짝돌리고 ㅠ\n새벽에 출근하는  저데려다주고\nㅡㅡㅡㅡ\n휴일에도   일찍일어나  \n분리수거   일어나라  눈치주기 \n샤워할때마다  화장실청소  \n밥먹고 바로 설거지 안하면 \n본인이하는데   \n잔소리 가   미칠지경\n부럽네요  너무']]
    
    5305
    부부간에 할 도리의 기준이 뭐죠 어제 남편이 그러더라고요할 도리만 하고 살자고아내가 할 도리남편이 할 도리무엇인지 모르겠어서요남편은 돈만 벌어오면 끝인가요?아내는 밥 제때 차려주고 빨래 청소 잘하고 집안 깔끔한 상태로 만들어 놓는거?그럼 여자만 너무 힘든거 아닌가 해서요.원래도 집안일 1도 안도와주는 사람이  저렇게 말하니 자기가 대단히 집안일에 관심을 가진 사람인줄 아는거 같아서요 아니면 저보고 일 똑바로 하라고 하는 소리인건지...제남편은 밥을 차려놔야 먹는 사람이거든요자기가 알아서 만들어 먹는거 라면하나냉장고에서 찾아먹는 것도 제가 미안해 해야할 일이거든요청소하는것도 마음에 안든다고 하던데다들 청소기 걸레질 매일 하시나요?제가 너무 더럽게 사는건지 게으른건지남들은 어떻게 하고 사는지 몰라서 알수가 없네요
    
    [['저도 신혼때 비슷한 말 하길래 집안일은 내가 할테니 육아는 정확히 반반 나누자고 했어요.'], ['저희남편은 그러면 반찬 먹을거 3가지씩 상에 새로운거 올리고\n청소는 매일 해야한다고 하대요\n그게 집안일 하는 사람들은 알테지만 얼마나 어이없는 말인지요\n자기는 그런엄마밑에 컸으니 저보고 못할게 아니고 억지가 아니라 하던데...\n벽이랑 말하는 느낌이라서요'], ['그럼 엄마랑 결혼하지 왜 나랑 결혼했냐 그게 좋으면 애 데리고 엄마랑 살아라 하세요'], ['월 천만원 생활비 에 세가지반찬 하자고 해야죠'], ['저는 자기전 3M 물걸레질 대충 슥하고 말아요 ㅋ 그나마 \n그 하루도 남편 없슴 안하구요'], ['처음부터 끝까지 같은 마음이네요.\n저희집도 똑같아요. 화가 많이 나요.\n'], ['왜이렇게 아내한테 바라는게 큰지 모르겠어요\n남자들은 돈만 벌어오면 자기할일 끝이죠\n그러면 주말에 쉬고.\n아내는 그럼 언제 쉬나여? 매일 짬짬히? 그게 쉬는거라고 할수 있는건지 참'], ['상대방 기준에서 할 도리가 뭔지 알아야지 싶네요\n할 도리가 뭔지 좀 적어 달라해야겠어요\n그나저나 서글프네요\n'], ['아내가 집안일을 똑바로 하는 기준\n매일 새로운 반찬 세가지 상에 올리는것\n아침 차려주는 것\n빨래 제때에 하는것\n청소 매일 청소기 걸레질 하는것\n자기엄마는 그렇게 했으니 저도 못할일이 아니라고 하네요\n참고로 제가 아이들 학교 학원 픽업 모두 하는데 시간이 매일 달라요\n학원앞에서 기다리는게 일인데 저걸 언제 다 하라는건지 참나'], ['그에 상응하는 남편의 도리는 월 800만원이라 하세요.\n어머님이 과도한 도리를 하셨네요.'], ['반찬 새로운거 하나둘씩사서주고  로봇 청소기 매일 돌려줌 되겠네요  그렇게 나옴  딱 그렇게만 대접해드려야죠뭐'], ['학원앞에서 기다리는거 하지말고 스케줄을 짜세요. 전 아이 하나지만 오전7시 애차에 델고 한시간거리 직장앞  어린집가고 5시 픽업해 집에와서 집안일 다했어요. 초등가서도 아침에가서 6시에 애가 집에딱와요. 차량으로. 픽업되는학원요. 물론 매일 청소.반찬은 말이안되지만 학원앖기다리는거만 안해도 진짜 시간 남잔아요'], ['저는 반찬 사고 국만 끓이고요\n청소는 부직포만하고 잠자는방만 청소기,걸레질하고 이불빨아요.\n\n집은너도같이사는데\n같이하던가\n같이안하자고했어요.\n\n니빨래 내가 넣고 갠다고\n운동화도 내가 빨아주다가 세탁방에 맡기고찾아오고\n\n요정도가 기본도리인듯 제기준 \n\n플러스 아이들케어'], ['저희 시어머니가 아들에게 늘 하는 말이 있어요..  아내에게 잘해라 여자에게 잘해야한다..지금이야 네가 돈벌어오고 그러지만 이제 나중엔 늙어서 언제까지 돈벌줄아느냐 남자는 돈줄끊기면 끝이다;;; 그러니 젊어서 아내에게 잘해야 너 늙어서 대접받는다. ..   전 이게 명언이라고 생각해요.    늙어서 대접받고싶으면 젊을때 잘해라.    남편분께 말씀하세요^^'], ['이게 남편분이 진짜 뭘모르시네요\n돈을 벌 수있는게 누구덕인지...\n\n가족의 가장은 남편이어도\n중심은 아내거든요.\n\n아내가 있어야 가족이란게 돌아가는건데...\n\n아놔\n한마디하고싶네요\n남편분께\n\n정말 돈의 가치로만따지면 원 800-1000정도의 일을 하고잇는건데'], ['돈버는 유세인가요? 그런식으로 할꺼면 죽을때까지 돈 벌라고 돈 안버는 순간 죽도 밥도 없다고 할꺼 같아요'], ['ㅋㅋㅋㅋㅋ돈안버는순간 굶어주는거군요 ㅎㅎ\n통쾌'], ['ㅎㅎㅎ이렇게 말하면 죽을때까지 돈벌테니까 지금 하는거나 잘하라고 말할 사람이에요\n나는 잘할 자신있고 잘하고 있으니 너나 잘해라 이런 논리를 가진 사람입니다...'], ['그에 상응하는 남편의 도리는 음~ 한 월천만원 정도인것 같으니 그선에서. 서로 도리만 하면 되려나요?\n도리는 내가 현실에 맞춰 할수 있는 선인거지 왜 누군가가 너의 도리는 이만큼이라고 규정하나요? 누가 그 권리를 줬나요?\n그런 논리며 남펴이 벌어와야 하는 돈도 내가 정할수 있어야 하는거 아닌가요? 참나~ 빈정상하게 하네요'], ['오오오 돈버는걸 정할수있다라\n멋진생각 ㅋㅋㅋ'], ['그렇게 말할 생각은 또 못했네요 분하고 억울해서\n자기는 내기준을 세워놓고선 내가 자기기준을 세워 말하면 버럭하는 스타일. 내로남불인 사람이라'], ['정말 맞는 말씀이네요~~ 내 할 도리를 무슨 권리로 일방적으로 정하는건지..'], ['할도리 이야기 하는거 읽자마자 헉...하고 흥분했어요. \n저라면...장기간 한다면 큰 싸움 될 수 있지만 보름에서 한달정도는 아이들 육아는 엄마인 내가 전담할테니 육아말고 집안일을 반으로 나누는건....어떨까요.\n그렇게 집안일을 쉽게 생각하는 신랑이라면 집안일 반쯤 나눠하는것은 일도 아니실테니 월,수,금 저녁준비는 아내가한다면 화, 목, 토 저녁은 신랑이 장보고 저녁상 차리심 될듯요.\n일요일은 외식.\n퇴근 시간이 늦다는 핑개대면 전날 퇴근하고 자기전에 준비해두시라고 꺼내어 차려먹는건 내가 하겠다하고 만약 내가 저녁 차린날 반찬가지고 트집잡으면...  다음날 똑같이 트집잡고 그게 얼마나 당하는 입장에서는 스트레스인지 느끼게 해줄듯요.\n장기간하면 큰 싸움 될 수 있으니 한달정도...\n아니면 최소 보름정도만 가사일 반반 부담한다면 얼마나 집안일이 열심히 해도 티 안나는 일인지 느끼실듯요.\n저녁 차리는것뿐 아니라 화장실 청소도 당연히 번갈아서 해봐야 최소한의 고마움이라도 느낄듯하네요.\n하지만... 이 글을 쓰다보니 결국 가정의 평화가 가장 중요한일이라 선뜻 추천드리기는 그렇네요. 힘내세요.\n'], ['이게 쉽지않을거예요. 저런생각하시는 남자랑은 이렇게 정하는것도 안먹혀요 ㅠ\n\n그냥 아내파업하시고 남편분이 개쌩돈을 써봐야 조금 느낄겁니다 ㅠ'], ['이럴줄 알았으면 결혼 안했을거 같아요..\n정말 과거로 돌아간다면 울면서 저 잡던 남편 시원하게 차주고 싶네요...\n아침부터 크게 한바탕 하고 댓글 남겨요 ㅠㅠ'], ['맞벌이인데도 그런 소리를 들을때가 있었습니다. 각자 잘 하는 일을 하자며...그래서 그뒤에 그에 대한 모든 일에 손을 뗐습니다. 원래도 집안일은 안하는 인간인지라..자기 빨래는 자기방에 모아두고 밥은 애들과 제가 먹을 반찬만 딱하고 바로 치워버리고 나가고 들어오는지도 모르고 살았었어요. 여행도 아이들만 데리고 다니고...그러더니 지몸이 아팠나봐요. 이러고 살다가 자기가 죽어도 아무도 모르겠다고 하얀깃발 흔들며 나오길래 이제는 빨래해서 쇼핑백에 때려담아 그인간 작업실에 넣어드리고 밥은 해놓는 정도이고 그인간은 청소와 설거지 정도는 합니다. 그래도 그말이 여전히 뼈에 남아 너 늙고 병들면 난 안돌본다고 했습니다. 인간들이 태생이 이기적인건지...모자란건지..'], ['오 독하게 아주 잘하셨네요. 통쾌해요 ㅠㅠ'], ['저는 나가서 돈도 못벌게 해요\n애들 잘 키우는게 돈 버는거라며..\n저도 하숙생 취급하면 고쳐질까 시도해봐야겠네요'], ['애들 잘키우는게 돈버는거라며 일도 못하게 만들어놓고 또 도리는 도리대로 다하라구요?? 진짜 너무 이기적이고... 남의 남편이라 욕도 못하겠고 너무 화가 나네요.'], ['저 정도 집안일이라면 세후 월 1000은 집에 갖다 줘야 할 소리인거 같네요. 그러고 은퇴하는 순간 밥도 끊겨야겠죠? 만약 지금 남편이 그 정도 생활비로 가져다 주신다면 도우미도 쓰시고 반찬도 좀 사다가 섞어서 내고 하세요. 느낌이 난 이렇게나 많이 벌어다주는데 그에 비해 넌 하는 일이 너무 없다 이런 뉘앙스 아닌가도 싶어서요.'], ['아내를 사무실 청소하는 사람쯤으로 생각하는걸까요? 저희집에도 그런사람있어서 너무 가슴이 답답하네요'], ['그럼 너의 도리는 머냐고... 되물어 보신거죠??? (아오 열받아서 그럽니다)'], ['못했어요ㅠㅠ 너무 분해서 반박만 하다가 졌어요\n송파맘에서 위로받고 있어요'], ['반찬 매일 사세요\n전 반찬 사먹는게 처음에 양심에 걸리더라구요\n제가 할일을 안하는거 같고.. 근데 저도 시간이 많아지고 남편도 냉장고에 늘 반찬이 가득하니 만족해 하는거 같아요(제가 만든 반찬인줄 알아요) 전 가끔 국이나 찜같은 메인요리만 해요'], ['돈으로 해결하면 되는 좋은 방법을 두고 이렇게 끙끙대고 있었네요\n돈벌어온다고 유세했으니 돈으로 되갚아주면 되겠어요'], ['싸우는것도 힘들잖아요 그냥 편하게 사세요...\nㅠ 내가 고생해봤자 남편은 몰라요'], ['남편은 돈 버는게 의무라면 얼마나 버느냐에 따라 와이프 집안일 완성도 달라지는거죠.  많이 주면 뭐 아줌마도 쓰고. 로봇 청소기 반찬가게 다 이용하는거구요 ㅡㅡ;'], ['한달에 몇천씩 벌어오시나봐요.. 밑에분말대로 돈버는거. 집안일 서로 나눈다면 액수에따라 집안일의 퀄리티가 달라지지않을까요. 신랑분 어이없는소리하고있으시네요;;;'], ['한달에 천만원씩 꼬박꼬박 가져다주면 황송해서 싸움도 안하겠네요...\n없는 살림 빠듯하게 사는것도 힘든데 저러니 답답해요'], ['뭐 수억 벌어다주나요? 매일 새로운 반찬이라니..  돈 많이 벌어다주면 사서 놓는거 못할까요..기분 나쁜 말이네요.요즘 세상에 집안일은 아내한테만 하라니요;;;'], ['그리고 진짜 죄송한데 꼭 님이 살길 만들어놓으셔요... 일못하게한다해도 그래도요...'], ['이혼을 몇번을 말해도 안해주네요\n저없으면 불편할거 자기도 알거든요\n전 행복할 자신있는데.'], ['이혼이 살길은 아니것같아요. 본인이 잘하시는거 찾아서 준비하길바랍니다.\n이제 아이들이 초딩이상되는것같으니ㅠㅜ'], ['준비는 이미 하고 있어요..\n지금 취업해도 써준다고 하는 곳이 있어요 월급이 많지 않을 뿐이죠\n그런데 취직도 못하게 하고\n육아 살림 혼자 다 하려니 힘드네요\n애들도 어느정도 커서 자기들이 씻고 밥먹고 할수있게 준비도 했고요\n위기감 느껴서 저러는지 예전처럼 자기만 바라보길 원하는거 같아요 애처럼'], ['월 천 가져다주면 매일 새로운반찬 5가지도 할수있다하세요. 반찬가게에서 매일새로운거 사서 주면 될것같고요. 청소도우미 세시간씩 쓰시구요. 대신 천이하로 내려가면 반찬가지수도 한가지로 내려간다구 하시구 청소도 일주일에 한번만 한다고 하세요. 그게 도리라구. \n남편이 너무 요즘 물정 모르시고 요구만 하시네요. 집에서 놀기만 하는 아내들이 어디있나요. 가정주부가 요즘은 극한직업인데...남편분 반성하시라고 여기댓글들 보여주세요.'], ['네 이것댓글들 보여주세요'], ['어머님이 그러게 하셔서 그걸 원하시면  본인도 친정아빠가 해주셨던것처럼 해주라하세요 \n엄마처럼 해주길 원한다면 본인도  아빠처럼 대해줘야 맞는거죠 \n정말 유치 뽕짝이시네요 ㅠ'], ['글엄 반찬 3첩 으로 해주고 청소 하고 벌어오는 돈 그냥 써요'], ['월 몇천 벌어오시나봐요 ~ 벌어오시는만큼만 해주세요'], ['흐흐 매끼니 삼첩 머가 어렵나요..하면서 따박따박 요구하세요..매일 반찬가게 가서 겹치지 않게 3가지 사서 한달 해보세요..나머진 얼렸다가 일주일 후에 내놓으세요..지난번 먹은 반찬인데 그럼 당신 월급으로 겹쳐서 내놓을 수 밖에  없다 더 벌어오심 한달 내내 안겹치게 해줄 수 있다 하셔요..전쟁 선포하면 남자들100퍼져요..저 잘 할 수 있는데  까지껏 머든 죽기아니면 살기로 댐비면 됩니다..말도 극존칭 써가면서~~~~근데 이젠 늙어서 이런거두 구찮아요..일단은 남편의 도리 써달라고 하셔요..나도 쓴다고 거기에 조목조목 많이 써서 넣으세요..'], ['청소 매일하죠.\n로봇청소기가...물걸레질까지요..ㅡㅡ\n말이라도 좀 이쁘게하든가..거시기하네요.'], ['주변에 천만원이상씩 갖다주는 남편들은 오히려 와이프 집안일하는거 입도 안대고 자기와이프 늙을까봐 고생할까봐 도우미쓰라 반찬은 사다먹어라 오늘 힘들었지 외식하자 이러던데요. 오히려 아껴쓰고 궁상떨면 되게 싫어해요 내가 그정도도 못벌어주냐며.. 그리고 잘사는 동네일수록 아빠들이 육아참여도도 높아요. 남편분 왜그러시는지 모르겠지만 늙어서 가정에서 소외당하기 싫으면 지금부터라도 애들한테 아내 존중하는 모습 보여주고 집안일 적극 참여하세요. 그렇게 군림하다가 나이들어 아이들한테 외면당하고 그제서야 가장의 외로움이니 atm이었니 해봤자 늦어요. 한달에 몇천 갖다주는것도 아니면서 atm 소리하는것도 어이없지만 암튼 집에서 그런 포지션 스스로 자처한게 누구입니까. 애들도 은연중에 알아요 아빠가 무례하고 이기적이라는거.. 그러다가 사춘기지나고 성인되면 아빠랑 서먹해지는거구요'], ['맞아요 완전 동감해요'], ['완전그래요.\n\n바깥에서 대접 못받고 궁상떠며 사니까\n집에서 큰소리치는거죠.'], ['솔직히 전 남편 밥은 차려줘요.. 늦게오든 우리가 다 먹고 치운후에 퇴근했든. 도리라고 까진 생각 안해봤지만 힘들게 돈 벌어오는데 이정돈 해줘야지 라는 생각이에요. 사실 전 집안일보다 직장생활이 더 힘들다고 생각하거든요.. 제가 직장생활하는동안 집밥이 그렇게 좋았었어요. 어쩔땐 평일에 한끼먹기도 힘든게 집밥이라 어쩔땐 밤 10시에 와도 엄마한테 밥달라고 하고ㅜ 뭐 그랬던터라 전 밥은 차려주는게 전업주부인 저의 역할이라 생각해요. 그렇다고 5첩반상 해주는거 아니구요ㅋ 반찬도 쉬운거만 몇개 하고 사기도 하고 어쩔땐 김치찌개 하나 이렇지만요. 밥해주는사람한테 입맛 맞추라고 했어요 제가. 엄마밥 그리우면 그리로 가라고. \n청소같은건 할때도 있고 안할때도 있는데 잔소리하면 ‘그럼 니가해’ ‘보이는 사람이 해’라고 해버려요. 청소하는게 맘에 안든다 하면 니가하라고 하세요. 난 이게 최선이라고.'], ['글쓴이분도 밥을 안차려주는건 아니었는데\n먼저 대놓고 저렇게 도리 따지며 얘기하니 기분상하신것같아요.\n\n전 집안일이 100배힘들던데 ㅠㅠ\n\n끝도없고\n휴식없고\n보상없고\n나 없고...'], ['공유님께서 제 마음을 너무 잘 이해해주셔서 다운된 기분이 좀 나아졌어요\n릴렉스하세요 스트레스받지 마시고ㅜㅜ 저때문에 속상해하지마시고..\n\n제가 화가 나는건요\n평소엔 그래요 집안일하고 애보는거 힘들지? 그래놓고 싸울때면 니가 똑바로하는게 뭐냐는 식으로 얘기해요\n저는요 아무리 화나서 싸워도 니가 돈 잘벌어오냐 애를 봐주냐 안하거든요\n그래서 이런말들으면 평소에 한말이 진심이 아니구나 하는 생각이 들어서 배신감이 느껴져요\n저희남편 새벽에 들어와도 밥차려줘요 새걸로 찌개 끓여줘요..\n아침먹으면 소화안된다고 총각때부터 안먹는 사람이 싸우면 꼭 아침 안차려준다해요 너무 웃기지않나요..\n컴퓨터하면서 먹은 간식이며 커피마신 컵도 안치우는 사람이 저보고 청소를 안한대요 자기가 먹은거 치우는 거까지도 제역할이라는 소리죠.. 전 솔직히 지금 하는것보다 더 잘할 자신도 없고 그러고 싶지도 않아요'], ['아 진짜. 저도 할말많은데 많아서 못 적어요 ㅠㅠ 그냥 이렇게 털고 또 제자리로 돌아가겠죠. 내남편에게 기분나쁜걸 제가 너무 감정이입했죠? ㅋㅋ\n\n이번주도 우리힘내요♡'], ['나가서 돈벌어도 되는데\n대신 애들이랑 자기한테 피해안가게 하래요\n나만큼 벌어와야 자기도 가사분담 한대요\n물론 싸울때 하는 소리니 백프로 진심은 아니겠죠\n근데 이렇게 막말을 싸우면서 하고나면 그 상처는 아무렇지않아지나요\n전 아니거든요\n근데 남편은 싸우고나서 서로 잘못해서 싸운건데 꼭 사과하는 과정이 필요하녜요\n그만싸우자해도 멈추질않고...\n정나미 떨어져서 같이 사는게 힘드네요'], ['헉... 말이 너무 심하신거 같은데요ㅜㅠ 그리고 님은 지금 굉장히 잘하고 계시는거 같구요.. 요즘 저만큼 하는 여자 없다는거 아셔야할텐데ㅜㅠ\n당당하게 나가세요!!! 니가 말하는 도리, 난 남들이상 하고 있다! \n너는 내가 생각하고 있는 남편으로서의 도리를 하고 있지 않다!\n'], ['남에 남편인데 솔직히 말뽄새가 아주 못됐네요\n도리를 왜 자기가 정해준답니까~?\n그럼 남편도리도 세세히 와이프가 정해도 되는거네요\n\n남편이 원하시는 와이프의도리가 그렇다면\n그에 상응하는 남편의 도리는 \n월 생활비 천만원에 시댁에서 물려받는 집,상가는 있어야된다고 정해주세요~!!!!!!\n글읽다 하도 어이없어 댓글 남겨요\n할말 못하고 사시지 마시고 조근조근 할말 하시고 사세요~!홧팅입니다'], ['ㅠㅠㅠㅠㅠㅠ넘 화나고 서운하시겠어요'], ['정말이지 어이가 없네요. \n부부간의 도리는 서로 이해하고 보살피고 아끼는 거죠. 반찬 투정 따위나 하고 일방적으로 이것저것 요구하면서 상처주는 게 아니라요. \n\n드라마에서 동백이가 500잔으로 까불이 때려잡고 했던 대사가 생각나요. \n\n"까불지마는 뭘 까불지마야. 사람 많으면 아무것도 못하는 주제에! 까불이? 까고 자빠졌네"\u200b\n\n이 상황에 딱 맞는 대사. \n"부부간 도리는 무슨 도리야. 자기 아내 마음 하나 헤아리지 못하는 주제에! 도리? 까고 자빠졌네!"\n\n남의 남편에게 할 말은 아니지만 \n소녀님 남편이랑 말 섞기도 싫으시겠어요. 대꾸할 가치가 없네요.'], ['맞아요 말 안섞고 있어요 쳐다도 보기싫네요\n그러면서 밥은 차려줬어요 이런 제가 싫습니다ㅠㅠ'], ['아 이런상황에선 먹던말던 안차리시는게. 맞아요. 아직 한번도 꿈틀하신적없으실것같아요...\n\n이러니 그분이 땅 깊은줄모르고 까불고있네요'], ['아이고 늙어서 진짜 구박박으려고~@@ 시간이  천천히 가는줄 아시는군요'], ['그렇게 따질거면 너 혼자 살 때 벌던 돈에서 나, 아이 하나씩 가족 추가될 때마다 인당 200만원씩 월급 더벌어오라고 했어요. 어디 혼자 살아도 그만큼 벌어오고 그렇게 살 껄 생색을 낸대요. 가장의 도리를 다하려면 지가 추가시킨 만큼 돈으로 떼우라고 하세요.'], ['돈벌어 갖다주면 가장으로써  90프로\n이상 먹고 들어간다 생각하시는 남편분인가봐여\n남편만 하겠다는건지 아빠만 하겠다는건지 하숙생만 하겠다는건지... 원하시는게\n정확하게 뭐랍니까.\n 하숙생처럼 생활하기 원하시면서 남편 대접과 아빠대접도 해달라하면 안되는거지요..\n'], ['저는 애들밥 챙기느라  제밥도 못먹으니까 차려달라말도 안하고 본인이 알아서차려먹고 어떨땐 차려주기도 하는데..정말 요즘남편들은 많이도와주고 그러던데 어느시대에서 오셨답니까...ㅜ'], ['남편분이 아이들 케어는 해주세요?ㅜ 힘드시겠어요. 남편분 기준과 글쓴님 기준이 다른거고.. 서로 못하겠다면 타협해야죠. 무조건 한사람 의견대로하면 그건 독재죠.ㅜ\n그리고 시아버님이 바깥 일만하고 집에서 손하나 까딱 안하셨다면 남편분은 그대로 보고 자라서 몸에 벤거예요.ㅜ 그건 진짜 답없구요..남편이 스스로 깨닫지 않는 이상..ㅜ\n아들이 있다면 꼭 시키세요. 남편 하듯 보고 자라게 돼 있어요.\n진짜 가정에서 보고 자라는게 무섭더라구요.\n저희도 그래서 노력해요. 애들을 위해서']]
    
    5366
    🐷 집안일 해야되는데😷 추버서 서글프네요ㅜㅜ오후에 하려니 귀찮고 얼른해놓고쉬는게 맞는데 추우니 자꾸 움츠려드네요그래도 후딱해놓고 쉬야긋어요집안일 시작하시나요?
    
    [['어제보다 오늘이 더춥네요. 날씨가 엄청 추워요. 패딩이 필수네요.'], ['아침기온이 영하 7도니 엄청 춥지요 그나마 오후에는 풀린다고 하네요 롱패딩 필수입니다'], ['저는 아직도 이불속이네오ㅜ 집안일이고뭐고 배고파서 일단 밥부터 먹어야할듯요ㅠ'], ['배고프니 일단 배부터 채워야지요 저도 배고파서 밥먹고 집안일했네요'], ['드시고드시고집안일 하셧군용 저는 오늘 집안일하나도안햇네요 막막합니다 ㅋㅋㅋ'], ['주말에 하시면 되지요 저는 주말에 성이가 있어서 혼자 있을때 해야 되겠더라구요'], ['그건그렇지용 저는주말에 또 김장도해야하고 공부도해야하고 이번주는글럿네요ㅠ'], ['아고 바쁘시네요 집안일은 다음주에 천천히하시고 급한것부터 하는게 맞지요'], ['오늘은 밖에일이잇어 잠시외출햇더니 너무너무춥네요ㅠㅠ무릎이시려요ㅋㅋㅋ'], ['그렇죠 저도 등원한다고 잠시 나갔는데 추워도 너무 춥더라구요 감기조심하세요'], ['맞아요ㅠㅠ애기 똘똘싸맷는데도 엄마추워 하더라구요ㅠ.ㅠ따시게입힐옷도사야되네요'], ['춥기는 춥지요 마스크는 필수로 해야겠더라구요 코에 바람 들어가면 감기걸리시 쉽지요'], ['그쵸ㅠㅠ이제 완전무장할때가 됐네요 언제까지추울런지ㅠㅠ'], ['주말내도록 춥다고 하더라구요 다음주에는 어떻게 될지 모르겠구요 날씨를 매번 챙겨봐야합니다'], ['너무추워서 온수매트키고 누워잇어요ㅠㅠㅠㅠ못일어날만큼 추운날씨😥'], ['온수매트 완전 따시하겠는데요 오늘은 추워서 꼼짝도 하기 싫은날입니다'], ['저도 더 귀찮아지기전에 후다닥 해치웠어요ㅋㅋㅋ'], ['집안일 일찍 끝내셨네요 저는 좀전에 집안일 해놓고 쉬는 타임입니다'], ['전 청소기돌리고ㅋ세탁기돌려놓고 또 쇼파에 앉았어요ㅜㅠ 밀대질좀 해야되는데..쉬고싶어요 ㅋㅋ'], ['세탁기까지 돌려놓고 부지런하십니다 저도 생각나서 이불빨래 돌렸어요 저는 집안일 다하고 쉬고있어요ㅎㅎ'], ['집안일은 한번에 다하고ㅎ쉬는게 좋은데ㅠ 왜자꾸 하다말다 하는가몰라요..ㅎ'], ['저는 하다말다하면 더 하기 싫더라구요ㅜㅜ 그래서 한번 할때 다 해버려야 직성이 풀리더라구요'], ['맞아여ㅋ한번에해버려야되는데ㅋ몸이 자꾸 쉬고싶어해요ㅠ 이래서 살만찌나봅니당ㅋ'], ['무조건 한번에 해야해요 안그럼 너무 귀찮지요 청소한번에 한다고 살 빠지는건 아니더라구요ㅜㅜ'], ['오늘은 날씨가 추워서 집안일 하지 맙시다..ㅋ 하루 정도 건너뛰어도 돼요..ㅋ'], ['안하려고 했는데 이틀이나 집안일을 안해서 오늘은 꼭 해야겠더라구요 해놓고 쉬고있네요'], ['저는 평일에는 매일 안 해요..ㅠ 일 때문에 못하기도 하지만..ㅋㅋㅋ\n주말에 합니당.'], ['일하시니 그렇지요 주말에는 쉬시니 해야겠지요 일하면서 집안일까지 매일은 못하지요'], ['네네..ㅋㅋ 근데 매일 못 하니 저절로 게을러지기는 하는 거 같아요..ㅠㅠ'], ['그렇지요 해버릇하면 괜찮은데 가끔 하면 정말 게을러지더라구요'], ['저는 청소기대충돌리고 빨래돌리고있어요~조금만 쉬다가 점심준비하려구요~~'], ['저도 생각난김에 이불빨래 돌려놓고 집안일은 다 끝내고 쉬고있어요'], ['저두요ㅎㅎㅎ방금 빨래개고 둘째랑 놀이방에서 놀면서 쉬네요~^^'], ['그렇군요 하기전에는 집안일 우짜노캤디 하고나니 엄청 개운한거 있지요 쉬세요^^'], ['전 어제부터 감기약 먹고 얼매나 잠이 쏟아지는지 몰라요 어제도 낮잠자고 일어났는데 오늘도 그럴듯요ㅎㅎ'], ['아고 감기 걸리셨군요 그럴때는 푹 쉬는게 맞아요 얼른 쾌차하세요'], ['귀찮지만 미뤄둠 더 하기 싫으니까 후딱하고 쉬시길 추천드립니다.ㅋㅋㅋ'], ['안그래도 미루면 귀찮을것같아서 아침먹고 집안일 끝냈어요 이제 이불빨래만 널면 되네요ㅎㅎ'], ['이불빠셨군요..우와..속이 시원하겠어요..ㅎㅎㅎ\n저도 지난주에 이불 돌렸는데 상쾌하더라구요~'], ['이번주에만 이불빨래 두번하네요 이불 빨래해놓으니 너무 개운합니다 햇빛이 있으니 마르긴 하네요'], ['추우니까 만사귀찬드라구요ㅜㅜ 빨래며 청소며 끝냇습니다...후후'], ['맞죠 추우니까 더 귀찮긴합니다 저도 집안일은 끝내놓고 세탁기에 이불만 돌아가면 널면 되요'], ['저는 그냥 집에 쌓아두고 출근했네요 ㅋㅋ\n오늘 퇴근해서 가서 마무으리 해야겠어요 ㅋㅋ'], ['주말에 하셔도되지요 저는 아이 있을때는 집안일하기 그래서 오늘 싹 했네요 이불빨래만 널면 되구요'], ['ㅋㅋ 수고하셧네용!!ㅎㅎ 겨울엔 빨래가 금방금방 안말라서 그게 전 너무싫더라구요 ㅠㅠ\n여름엔 빨리 말라서 좋은ㄷ ㅔㅠㅠ'], ['그러니깐요 그래도 방에 말리고하면 마르긴 하더라구요 대신 자주는 이불빨래 못하지요'], ['네네 저도 ㅋㅋ 밖에 널어놨다가 ㅋ ㅋ저녁엔 거실에 들여놓고 자고 그래요 ㅋㅋ 보일러트는김에 빨래나 말리자 이러면서 ㅋ ㅋ'], ['그렇지요 그러면서 가습기 역할도 하구요 오늘 돌린 이불은 얇아서 그런가 금방 마르더라구요'], ['추우니 만사 귀찮지요 귀찮지만 집안일 끝내고 쉬고있네요 끝내놓길 잘한것같아요'], ['아이잘때 자거나 그래야하는데 그시간이 아까워서 놀다자면 잠이나잘걸 하면서 그러더라구요'], ['그러니깐요 아이 잘때는 내 시간을 만들고싶지 집안일하는데 허비하고 싶지 않더라구요'], ['집안일 다 하셨나요 오늘은 춥다니 재활용은 내일 버릴려구요 \n할일 다 하셨으면 자유시간 즐기세요'], ['집안일 다 끝내고 쉬었어요 아니면 귀찮을것 같더라구요 추워서 재활용 버리기도 귀찮긴합니다']]
    
    5425
    집안일 중에 뭐가 제일 싫으세요? 전 설거지요ㅜㅜ오늘 출근해서 너무피곤해서눈이 감기는데설거지 하기싫어서 못자고 있어요.전생에 수랏간 무수리여서 평생 설거지만했는지...어차피 내가 할일이야. 하자. 라고 마음먹고다시 뒤돌아서 누워버린ㅠㅠ
    
    [['다림질이요.'], ['우리 안그래도 집안일 힘든데\n다림질은 세탁소에 줘버립시다.\n전 진작에 소질없음을 발견하고\n세탁소 사장님과 친하게 지내고있습니다.\n'], ['빨래개서정리하기\n화장실청소ㅠ 전 설거지가 백번낫다는요..'], ['빨래.음식물 쓰레기 버리기'], ['빨래 널고 개고 정리하는거요ㅠ'], ['음식물.빨래갠것 넣기. 이불빨래말리기 \n옷꼬매기 냉장고정리 등등등등 다싫으네요 ㅎㅎ'], ['설거지요\n전 이상하게\n설거지 하다보면 옛날생각이 너무 많이 생각나  싫어요 ㅋㅋㅋㅋㅋ'], ['옛날생각이 왜나요?'], ['저도 설거지요ㅋㅋ 기분나빴던 일들이 자꾸 생각나서 계속 한숨쉬다가 설거지 끝날때쯤엔 씩씩거려요... ㅋㅋ'], ['음식물쓰레기처리 빨래정리요....\n식세기들이세요 삶의질이 달라져요'], ['식세기있어요. 근데 한번씻어서 넣는게 더싫어요. 상전모시고 사는기분이라ㅋ'], ['아ㅋㅋㅋ 전 그것만해도 너무 좋던데ㅠ 안쓰시는 분들도 계시더라구요'], ['작은거라 도마냄비 안들어가서\n설거지 두번하는기분들어요.'], ['전 빨래 설거지 둘다 다요 ㅋㅋㅋㅋㅋㅋㄱㅋ'], ['요리요ㅜㅠㅜ'], ['밥이요..ㅠ'], ['설거지요,저두~ㅋㅋ'], ['아직 인덕션이 아닌지라 ㅠ 가스 렌지 청소요'], ['저도 설거지 느무느무 싫어요~~~ 고무장갑만 끼면 답답하기도해요....'], ['빨래개서 옷장마다 갓다넣기...너무너무너무너무너무 싫어요 ㅜㅜ 티도 안나고 개운하지도 않고'], ['빨래 개서 옷장에 정리해 넣는거요 ㅜㅜ'], ['전 청소오!!가스렌지도 장실도...ㅠ\n설겆이를10시간 하긋어요'], ['요리 설거지요ㅡ.ㅡ'], ['설거지,빨래개키기서 넣기ㅠ'], ['빨래개서 제자리에 넣기 \n젤 싫어요'], ['요리요'], ['집청소요......\n빨래 설거지는 하면 티는 나지만 집청소는 티도안나요!!!근데 안하면 안한티는 엄청나요!!!0'], ['빨래개고 옷장에 정리정돈 하는게 너무 귀찮어요.....설겆이는 뭔가 깨긋해지는거같아서 기분이라 좋은데 \nㅠㅠ'], ['욕실청소요'], ['빨래랑 청소요ㅜ해도 다음날이면...ㅜㅜㅜ'], ['다요... 끝이 없고 돈도 못받고 ... 집안일은 그냥 희생이예요'], ['설거지를 하면서..\n엄마는 왜 난테 시집가라고 안달복달이였지;;\n이생각까지ㅋ'], ['빨래개기,,,,끝은 있는걸까요ㅜㅜㅜㅜ'], ['제가 빨래개기 안하는법 알려드릴까요?'], ['오잉? 그런게 있을까요? ㅎㅎ'], ['빨래개서 제자리 넣기요 ㅠㅠ'], ['다요........엄마밑에 살던때 그 고마움을 몰랐네요\n그래서 전 아기 크면 다 시키려구요ㅋㅋ'], ['저도 설거지요.\n왤케하기싫죠?전엔  손걸레질이 젤 싫었는데,나이드니 설거지가 정말 싫어요.'], ['다 안하고싶어요 ...'], ['저도 집안일 전부요ㅋ 요리 사랑하는데 하고싶을때만 해주고싶어요ㅎㅎ'], ['모두 제자리요 그게 제일 힘들어요\n빨래 제자리\n물건 제자기\n마른 그릇 제자리\n결국 다시 다 나올것들 제자리 넣는게 의미없는거같으면서도 가장 많이 하는 일이에요'], ['자려고 딱 누웠는데,\n빨래돌린거 기억날때요ㅜㅜ\n'], ['음쓰요.'], ['설거지 싫어서 식기세척기 샀어요ㅠㅠ\n건조기보다 더 좋아요\n저녁마다 30분씩 하던 설거지 타임이 사라져서 넘 좋더라구요.'], ['식세기 애벌세척하기 싫어서 \n잠깐만 누워야지 하고 침대 드러누워\n이제 이불속 깊숙히들어와있네요.\n전.. 식세기보다 건조기가 더좋아요.\n나의사랑♡건조기\n'], ['가사의 모든 것이 싫어요. 어릴 적부터 쭈우우욱 싫었는데 왜 결혼을 했는지 이해불가;;;;;'], ['전 결혼하면\n진짜 완벽히 잘할줄알았는데.\n지금은 저랑 살아주는 남편이 고마워요.'], ['격공합니다ㅋㅋ\n저도 저랑 살아주는 신랑에게 고맙더라구요'], ['설거지요'], ['전 요리하는게 제일 귀찮아요..... 남편만 없으면 그냥 맨날 혼자 대충 김치꺼내두고 김싸서 먹어버리고 그런다는 ㅠㅠ'], ['요리는 좋지만 설거지 싫구요.\n청소, 세탁, 정리, 계절별 대청소, 그냥 내가 혼자해야하는 집안일 모두가 싫어요.\n그나마 음쓰, 재활용, 일반쓰레기 \n는 남편이 버려줘요.'], ['빨래개서 정리하는게 제일싫어요.'], ['요리요,  제일 창작적인? 일이잖아요ㅜㅜ'], ['소파밑에 들어간 장난감 꺼내서 정리하는거요..ㅜ'], ['장난감정리 싫어요ㅠ 어차피 몇시간 후면 또 난장판 되는데 ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ 퍼즐 ㅠ 블럭 ㅠ  지옥이에요ㅜ'], ['다 싫지만ㅠㅠㅋㅋ 그래도 가장 싫은 건 화장실 청소, 음식물쓰레기 처리요~\n신랑친구들 온다고 며칠전에 화장실 2개 빡빡 청소하다 허리 나가는 줄 ㅠ ㅠ 아직도 허리가 계속아프네요ㅠ\n음.쓰도 개수통에 가득 차면 모으는 통에 넣고, 밖에 버리러 나가는 게 왤케 세상 싫은 건지요ㅠㅠ'], ['다 싫어요. 결혼하면 좋아질줄알았는데ㅜㅜ청결보다 귀찮음이 이기네요.'], ['빨래 개고 정리하기요'], ['식기세척기 사세요.편해요.\n전 음식물 쓰레기버리기랑 분리수거 싫고 설거지도 싫어요..ㅜ'], ['빨래 개기요~\n개서 넣기까지가 왜 이렇게 하기 싫은지 ㅜ'], ['다~~~요 \n설거지는 그나마 바로바로 하니 조금밖에 없는데 ... 만삭에 쭈구리고 화장실 청소하고  개털흘리고 다니는 우리 개님...2시간마다 청소기밀고 닦고 ...'], ['정리정돈이요.']]
    
    5582
    저희 남편이 집안일 대단히 많이 하는건가요?? 분리수거랑 쓰레기 버리는 건 남편이 다 전담해서 하고 있습니다~ 집안 청소는 거의 제 전담입니다. 진짜 제가 너무 힘들어서 한달에 한번? 화장실 청소 좀 해달라고 하면 해주는 정도?? 아기 케어도 거의 제 전담..목욕준비랑 뒷정리는 남편이 하지만 목욕은 신생아때부터 제가 시킵니다~ 빨래 세탁기랑 건조기 돌리는건 남편이 할때가 많고~개고 정리하는건 제가 합니다~ 설거지나 자잘한 집안일(가습기 세척, 젖병삶기 등)은 그때그때 여유 있는 사람이 하는 편입니다~ 어제 자기가 집안일 진짜 많이 하는것처럼 말하면서 자기 친구들은 하나도 안한다고 저보고 모 대단히 도와주는 것처럼 말하는데~곰곰히 생각해보면 쓰레기버리는 거랑 자잘한 집안일 뿐인 거 같은데~저희 남편 집안일 많이 하는 건가요??;;;         
    
    [['친구분들은 뭐 팔다리가없나봐요?\n집안일은 말그대로 지가 사는집청손데 많이하고말고가 어딨데요 당연히 하는거지 웃기네요 ;; \n대체남자들이 집안일하는게 뭐가대단한건지 ;;;'], ['그러니까요~도와준다는거 자체가 전 말이 안된다고 생각하는 편인데;;밖에서 일하고 돈 벌어오는 거 힘든거 알아요~이해 많이 해주려고 하는데~진짜 저렇게 말하면 짜증나더라구여ㅠ'], ['어차피 혼자살아도 본인이 다해야하는거고 가족과살면 같이하면되는거지 웃기네요 누군태어날때부터 집안일하나요'], ['저희 남편은 쓰레기, 설거지, 빨래 담당인데 저는 더 열심히 하고 힘내라고 \n우쭈쭈해주고 있어요 ㅎㅎㅎㅎ 저는 조련 중... 대단한일 하는거라고 옳지옳지 해주세요 ㅋㅋ 일 더 많이 하라고 ㅋㅋㅋ'], ['222222고마워.최고야.해주시면\n남자들이더잘하지않을까요?ㅎㅎ'], ['3333우쮸쮸~~~칭찬ㅎ'], ['육아는 제가 집안일할때 아기 봐주죠ㅠ그것도 옆에서 자꾸 핸드폰 하고 해서 제가 잔소리하면 싫어하고;;'], ['많이 하는 건 아니고 적당히 잘 하는듯 보여요 저희 남편도 그정도 하는데 밖에선 엄청 많이 한다고 말하고 다녀요 다른 남자 분들은 아예 안하는지…주변 분들도 그걸 많이 하는 거라고 하더라고요 그러려니 해요'], ['어 저희 남편도 밖에서는 자기가 애 다 키우는것처럼 말해요;;ㅎㅎ'], ['저희남편도이래요. 진짜어이없어요 누가보면 나공주대접받는지알겠네,ㅠㅠ'], ['쓰레기만 버려요,,\n음식쓰레기도 가지고 가기 싫다고 \n처리기 설치했네요,,\n자기가 먹은 라면냄비는 좀 씻어두면 좋겠네요 ㅠㅠ'], ['그냥 잘한다고맙다 인정해주심이..어때요?ㅋㅋ어이가없지만 그런칭찬듣고싶어서 그렇게생색내듯말하더라구요ㅋㅇㅋ'], ['ㅠㅠ칭찬해주고 싶다가도 제가 집안일하고 아기 케어하는 거 아무 일도 아닌것처럼 말할때가 많아서 해주기 싫어요ㅠㅠㅎㅎ'], ['마자요ㅋㅋ핵공감 ㅋㅋ 지가하는것만 대단한것처럼 내가하는건당연하고...그러면서자기는 인정받고싶어하죠ㅠㅠ\n저도진짜꼴베기가싫어서  맨날싸웠어요ㅋㅋㅠㅠ힘내세요'], ['맞벌이면 너무 안하시는것 같고 외벌이면 저는 적당히 해주시는 것 같네요.. 저는 맞벌이하는데 제가 집안일을 남편분 만큼하는데도 힘드네요ㅜㅜ 회사 일 하고나면 너무 지쳐서요ㅜ'], ['제 주위 어르신들이  남의편한테 잘한다잘한다 하라고  그래야 부려먹는다고... 그냥 잘한다잘한다 하세요 ㅎㅎㅎ'], ['저희남편이랑 비슷하시네요. 쓰레기 전담이고 화장실청소 해줘요. 주말에 시간날때 청소기 돌리고 빨래 개고요. 평일엔 육아+집안일 온전히 저 혼자해요. 남편은 가만있는데 주변에서 최고남편이래요.ㅋㅋㅋ 그러려니 하고 살아요 ㅋ'], ['저희남편보다는 많이 하시는거같네요,  저희남편은 아무것도 안해요. 모든일이 저혼자..ㅜㅜ'], ['ㅠㅠ다 하시다니 힘드시겟어요ㅠㅠ'], ['집안일은 해도 끝이 없는데 같이 하는 거죠 생색낼 정도는 아닌 거 같지만 칭찬해줘야 더 하려고 하더라구요ㅎㅎ저희 남편은 설거지, 수건 개기,아기 목욕 준비, 목욕시키기(같이해요), 가끔 청소기 돌려주고 바닥 닦아줘요 대신 제가 집안일 하는 동안 아기 정말 잘 놀아주고요'], ['잘 놀아준다니 그거 정말 좋네요!!저희 남편은 옆에서 핸드폰하고 스피커로 자기 노래 듣고;;자기 쉬는 타임인줄 알아요~집안일 제가 하고 끝나면 제가 아기랑 놀고 하니 휴식이 없네요ㅠㅠ'], ['지금은 가끔 폰 만지면서 애기 볼 때도 있는데 그 전엔 폰게임만 내도록 해서 잔소리 좀 했더니 많이 줄었어요ㅎㅎ주말에 남편한테 아기 케어 다 시키고 평일에 내가 어땠을지 느껴봐라고 하고 있어요'], ['격공.... 지 휴식타임인줄 알죠...'], ['해준다는 생각자체가 트렬먹은거 같네요 참나'], ['어금니 꽉 물고 즈른다즈른다라고 해주세요..^^; 가정의 평화를 위해... 어렵네요. 하ㅜㅜ'], ['저희남편도 스스로 다른남자들보다는 잘도와주는편이라고 얘기하지만 전 그럴때 그럼그것도안하냐고 얘기해요.\n지금이야 둘째출산앞두고 휴직해서 제가조금더 할순있지만 맞벌이일때는 어림도없죠'], ['윗분들 말씀처럼... 잘한다잘한다 해주세요.. 원래 요즘사람 다 이정도 하지 하고 싸우면 더 하기싫어하는데 우쭈쭈 해주면 하고나서 칭찬기다리는 강아지같아요 ㅋㅋㅋ'], ['남편 근무가.정신적으로 육체적으로 힘든편이라 저희남편도 저정도 도와주는데 많이 같이한다 생각해요 저는..'], ['신랑외벌이 육아는 함께하고\n집안일은 설겆이해주고 쓰레기 음식물쓰레기 버려주고 빨래정리 청소는 같이해요 밥준비할때는 재료정리하는거 도와주고 중간중간부탁하는거 저절하는거 없어요..\n육아에 많이 참여 못한다고 집안일이라도 도와준다고 하더라구요 생색내지 않아요 당연히 해야할일이라구..'], ['우리집남자는 손하나 까딱 안해요.제가 전업주부라서 안하는거래요ㅡㅡ'], ['저는 지금 30주 임신중인데 모든 집안일 신랑이 다하고 저샤워하는거까지 시켜주고있어요.. 저는 저 먹는거만 챙겨먹고있는데... 이정도해야 다 하는게 아닐까싶네요'], ['맞벌이시면 많이 도와주는건 아닌거같고 외벌이시면 많이 도와주시는거지요~'], ['저희남편 신혼때부터 지금까지\n평일저녁설거지 전담\n퇴근후 육아전담\n목욕전담 \n음쓰, 분리수거 전담\n주말엔 대청소,화장실청소 혼자다해요\n저는 빨래와육아정도하구요\n항상 풍족한 칭찬과 여보가 해주니까 너무깨끗하다\n나보다 여보가 꼼꼼하네... 애기가 여보랑 목욕하니까\n너무좋은가봐^^ 여보고마워 등등...\n특급칭찬을 해줘요\n칭찬이 길들이긴 최고의 방법이죠ㅋ\n'], ['일단은 비교당하면 너무 싫을듯요\n\n딱히정해진건없는데\n시키면해줘요\n\n쓰레기 무거운거 버려달라하고\n분리수거해놓은거 버리고오라하고\n음식물도 쌓이면 버리라그래요\n평소엔 제가버리고\n\n빨래는 같이하고\n화장실과 설거지는 잘안해요\n손에꼽힘\n\n\n제가가장하기싫은게 화장실이랑 설거지인데\n그래서 불만은 있음\n\n아직 애기가 없는데도 이정도에요\n\n신랑은 더러워도 잘사는편이에요\n\n전 깔끔쟁이는 아니지만 눈에보이면 하는편이죠'], ['보통같아요  안하는 남자들널렸어요\n그리고 일을 분담하는게 씻기는거면 준비부터끝까지하고   빨래도널었으먼 개고 이래야지 저렇게하면 더힘들텐데ᆢ'], ['울신랑은 집안청소도맡아서해요 청소기돌리고 걸레질까지 화장실도남편담당입니다..쓰레기분리수거음쓰도남편담당..임신중인데 요즘엔설겆이도남편이해요...저는밥하고 빨래돌리면 널고개는것만해요 이런남자한테많이한다해야하지않을까싶어요~~'], ['맞벌이는 아니신거죠? 남편분 잘 하는 것 같아요..목욕도 사실 준비..뒷정리가 더 귀찮은거고..^^'], ['당연할걸 생색내시네요;;; 요기답글좀 보시고 아니라는걸 아셨음 좋겠네요ㅜ\n맞벌이 아직아이 없는 저희는 신랑이 요리 제가 설겆이\n분리수거는 저 신랑은 음쓰\n빨래.널기.개기는 함께해거나 신랑이 하거나\n청소기는 핑퐁핑퐁\n화장실 청소 신랑전담\n화분물주고 정리 제전담이예요\n님이 아이돌보는데 저정도는 당연한거아닌가요? 같이 사는 집이라 같이해야되는걸 많이한다고 생색내는게 이해가 안되네요;;;\n'], ['적게 하는거는 분명 아닌데.. 본인 입으로 친구들 안하는데 나만 한다 이야기하는건.....  음... 좀 그래요..^^;'], ['저희신랑은 요리 빼고 다하는데요 청소 빨래 쓰레기버리기 애기 목욕시키기 전 요리 밖에 안해요 생색 엄청 내요 근데 맞는 말이니 우쭈쭈해줘요 ㅋㅋ'], ['ㅋㅋㅋ뭘 대단히 많이 하는 정도는 아니시고ㅋㅋㅋㅋ그냥 안하는 건 아닌? 근데 뭐 이걸 이렇게 따지기도 애매하네요ㅋㅋㅋ 근데 즤집도 애기랑 놀아주기, 번갈아가면서 식사준비&설거지 정도인데 엄청 한다고 얘기해유 그냥 한귀로 듣고 한귀로 흘리심이 좋아요ㅋㅋ저희 남편은 밖에서는 완전 사랑꾼에 집안일 완벽히 하는 사람이예요ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['저흰 그냥 딱. 애목욕만 시켜줘요 근데 제가보기엔 많이 하는거같진 않는데 거기에 비하면 저흰 거의 뭐 독박이네유..ㅜ 진짜 육아와 집안일은 딱딱 분담해서 했음 좋겠어용'], ['전업이면 잘해주시는거지옹. 맞벌이면 당연분담해야하고..ㅠ\n저는 집안일은 제가하고 남편 퇴근하면 내가 집안일하는동안 아기랑 놀아주라구해요.\n집안일 안 도와줘도되니까 아기좀 보고있어달라고하세요. 육아는 함께해야지요'], ['전업주부이신가요?? 그럼 남편분이 많이도와주시는거죠~~'], ['그 정도면 어느정도 해주시네요ㅎㅎ 제 남편은 아침7시 출근 저녁8시30분 퇴근이고 휴무 조차도 간혹가다 있어서 어쩔땐 완전 독박인데요, 그래도 휴무날엔 빨래도 같이하고 설거지도 같이하고 밥상도 같이 차리고 청소도 같이해요! 무조건이요 ㅎㅎ 항상 옆에 붙어서 제가 청소기 돌리면 걸레질 하고 빨래 널면 반대편에서 같이널고 설거지하면 그릇씻어서 놔두기 해요 ㅋㅋ 처음엔 안했는데 한번 하면 칭찬 해주니 자기도 신이나서 하는거 같더라구요 ㅎㅎ 한번쯤은 칭찬 가득히 해줘보세요 ㅎㅎ 안하던것도 스스로 하려고 해요 ㅎㅎ'], ['남자들은\n그게잘못됐어요\n하면 하는거지.뭘 지가 대단한것해주는것처럼..미친..\n저희신랑도 음.쓰 몇번 버려주더니\n자기가 다해줬다고 그래요..\n그뒤로안시켜요\n'], ['그걸 대단하다고 생각하는 사람이나...집안일 많이하네 라고 얘기하는 사람이나 이상한거같아요 당연히 같이하는거죵!!'], ['그리고 따져보면 많이하는거 같지도 않아요'], ['결혼 10년차.. 진짜 가~~~끔 쓰레기 버리러가주고 진짜 가~~~~끔 청소기들고 대강해주고 진짜 가~~끔\n세탁기 세제넣고 버튼눌러주는...\n요즘 제가 우울증 초고도로와서 집안살림 손놓고나선 남편이 집안일도와는주지만 그전까진 손까락 까딱안하고 그거 가~~~끔한일이 주변에다간 집안살림 겁늬잘해주는 1등신랑이라고 떠벌리고다녔어요ㅋㅋ\n'], ['잘하는거같아요 그렇지만 그런거가지고 우쭐하거나 특급칭찬은아닌거같고요 ㅋㅋ 저ㅡ희집인간은 1도안해요 하면 죽는줄알아요 화장실청소 지가한다더니 결혼 4년차 10번도안햇을거고 지방이나 어지르고 음쓰가끔 지눈에띄거나 나한테미안해할때해주고 쓰레기는 문앞에 직접 가져다드려야 버립니다'], ['전업일때나 맞벌이 할때나 밥하는 거 빼고는 다 남편이 하는 편이에요.\n음식엔 소질이 없어서 식사준비는 거의 제 전담, 나머지는 거의 남편이해요.\n육아는 남편이 더 잘 하는 편이고 저 없어도 엄마 없는 표 안날정도로 잘 케어하는 편인데.\n이제 아이가 10살이라 혼자 잘해서. 공부만 봐주면 되요.\n아이도 집안일은 당연히 아빠가 하는 건줄 알아서 아빠한테 방 치워달라, 옷빨아달라 하는 게\n자연스러워요. 그래도 생색내거나 하지 않아요. 같이 해야한다고 생각하고.\n체력이 좋은 남자가 더 해야한다고 생각하는 거 같아요.\n'], ['아예 1도안해요ㅠ여긴ㅠ'], ['전업이신가요?\n그럼 많이 도와주시는거같은데요.\n우리신랑이 저정도만해주면 업고다니겄네요'], ['저희 집 양반보다는 많이 하시는건 틀림 없지만..그건 제가 안 시키기도 해서입니다..그러니 이건 부부사이에서 정해야 하는거 같아요.'], ['요새 친구남편들이나 저희 남편보면 집안일 거이 다 해요...ㅋ 퇴근 후 또는 쉬는 날 아기 보는것도 물론이구요.... 젊은 사람들은 많이들 그렇게 사는 듯해요\n저는 거이 집에서 음식만 하는 것 같아요.. 그리고 남편이 설거지 빨래 청소 쓰레기 등등 다 하구요.. 아기 목욕시키고 밥 먹이고 이런것도 다 해요...., \n우쭈쭈해주면서도 요새 다들 한다더라!! 라고 해요\n근데 한가지 단점은 있어요, 싸울때 할 말 없음 지가 집안일까지 다 한다는 이야기가 꼭 나와요....ㅋㅋㅋㅋ'], ['분리수거 정리는 제가, 버리는건 남편\n낮설거지는 제가, 밤설거지는 남편\n남편 쉬는날 빨래돌리고 게는거 남편, 쉬는날 아기목욕은 남편, 평소에는 제가,\n일반적인 청소, 화장실청소는 제가 하구.. \n아기빨래는 제가해요\n할때마다 잘한다 고맙다해주면 뿌듯해하면서 알아서 잘해요..ㅋㅋ\n'], ['저 육휴중이고 저희 신랑도 저 정도하는데.. 전 잘하고 있다고 얘기해줬어요. 애기 낳기 전에는 진짜 집안일 1도 안하는 사람이었거든요ㅠ'], ['결혼생활 10년!! 한번도 쓰레기 버려본적이 없네요.재활용 .음식물 싹다신랑이!!  그러곤 눈치껏 설거지.청소.빨래 다 해줘요~~  늦게퇴근해서 할시간이 많지는 안치만 여유 이쓰면 뭐든  해주려하는ㅋㅋ'], ['적당히 일하는 남편이면 적절히 도와주는것같은데용?'], ['진짜 아예 손도 안대는 남자들 많더라구요 저희남편도 음쓰와 쓰레기 버리고 아기목욕한거 뒷정리해주고 아주가끔이지만 시키면 설거지와 젖병도 닦아주는데 남들앞에서 칭찬 많이해줘요. 보면 또 남편만큼 잘하는 남편이 없더라구여;; 그 남편들 들을라고 일부러 더 해요']]
    
    5634
    제일 하기 싫은 집안일 있으시죠..? 전...애 낳기전에는 설거지하다가 우웩...ㅎㅎㅎ설거지 제일 싫어했는데 애 키우다보니 그나마 좀 나아졌거든요~근데...건조기에서 나온 빨래 개는것까진 그럭저럭-그 후 이방 저방 정리해서 넣는게 너무 귀찮아요..ㅎㅎㅎ그냥 움직이는게 귀찮은 것 같기도 하고요😂신랑은 청소를 제일 싫어하더라고요~전 또 청소는 기분이 좋아지고ㅎㅎㅎ다들 제일 싫어하는 집안일할때 어떠세요ㅠㅠ
    
    [['저는 설거지를 엄청 좋아하고....재활용버리는게 제일싫어요ㅋㅋ'], ['설거지 좋아하시는 분들 존경합니다👍👍 전 아직도 음식물 치우는게ㅠㅠㅠㅠㅠㅠ'], ['전 빨래개는거ㅜㅜ 너무싫어요'], ['전 그나마 앉아서 개는건 괜찮은데.. 정리가ㅠㅠ 가끔 개놓은거 쌓아놓고 입어요😂😂😂'], ['저도 빨래 개는거는 괜찮은데 가져다 놓는게 넘나 싫으네요ㅠㅠ'], ['그쵸그쵸그쵸!!!!!!! 빨래들고 여기서랍 저기서랍 넣는게 왜케 귀찮을까요ㅠㅠ'], ['맞아요!!!!!!!! 티비보면서 사부작사부작 개는건 괜찮은데 그 후가 최악이에요😂😂'], ['저도 설거지랑 음쓰 버리러 가는게 젤루 싫어요'], ['진짜 음쓰는..... 신랑 있으면 설거지는 안하게 돼요^^;;;;; 애 키우면서 좋아졌어도 여전히 힘든 일 중에 하나네요ㅠㅠ'], ['화장실청소요ㅎㅎ'], ['아~!!!! 전 그나마 청소는 좋아하는 편인데 신랑 보니 청소기 돌리는것도 끔찍해하더라고요ㅎㅎㅎ 화장실은 오죽하겠어요😅'], ['저는 걸레질이요.'], ['저도 걸레질은 너무 귀찮고 힘들어요ㅠㅠ 친구가 로폿물걸레 청소기 샀는데 보니까 신세계더라고요~ 침대 속까지 싹싹!!!!'], ['하 ....다싫어요ㅠㅠㅠ'], ['푸하🤣🤣🤣🤣 맞아요 다 싫긴해요ㅋㅋㅋㅋㅋㅋ'], ['저도 빨래 서랍에 정리하러 이방저방 다니는거요\n서랍도 모자라서 구겨넣어야하구ㅜㅜ'], ['저도 그거요!!!!!!!!!! 이방 저방 다니면서 넣고 정리하고ㅠㅠㅠㅠ 하아......'], ['걸레질이요'], ['안하면 티나고... 해도 그뿐인 걸레질... 애들 발자국 손자국 보면 해야하는데 제일 몸 쓰는 집안일인것 같아요ㅠㅠ'], ['건조기에서 꺼내는것부터가 싫어요ㅋㅋㅋ 개는것도싫고 넣으러다니는것도 넘 싫어요ㅋㅋ 설거지가 제일 나아요 ㅋㅋ 개운한 느낌...'], ['아하~ 설거지는 좀 좋아하시네요~ 건조기 다 돌아갔다는 소리에 스트레스 받으시겠어요😂'], ['전 설거지요. 너무 시러요대충할 수도 없고 빡빡하다보니 힘들고 오래 걸리고요'], ['흑흑ㅠㅠ 해도해도 하루에 몇번씩 해야하는게 설거지인것 같아요ㅠㅠ 뒤돌아서면 또 나와있고.. 애들 방학이라그런지 계속 주방에 서있는 것 같아요ㅠㅠ'], ['맞아요 ㅋ 빨래 갖다놓는게 귀찮아요'], ['네.........진심...... 그때만큼은 제가 무척 게으른 것 같아요^^;;;;;;'], ['걸레질이요~~ 제일 하기 힘들어요ㅠㅠ'], ['노동이죠 노동ㅠㅠㅠㅠ 팔도 아프고ㅠㅠ 물걸레로봇청소기 산 친구 보니 신세계가 따로 없더라고요..특히 침대 밑👍👍'], ['설거지, 마른 빨래 정리 등 다수.....넓은집에서 아줌마 도움 받아가며 살면 좋겠습니다.'], ['모두의 꿈이겠죠~~?? 저도 꿈꿔 볼랍니다~~'], ['저는 주방일이 제일 ... 청소는 그나마 낫네요 !'], ['저도 주방에 서있는거 별로 안좋아해요ㅠㅠ 청소기들고 이리저리 왔다갔다하는건 개운한데 말이죠ㅎㅎㅎ'], ['하~설겆이후 그릇정리. 걸레질후 손빨래 진짜하기싫어요...ㅈ'], ['악!!! 걸레질 후 손빨래!!!!!! 대박이죠ㅠㅠ 전 밀대로 밀고 청소포 버려요ㅎㅎㅎㅎ'], ['집안일에 요리도 포함이면 요리요 ㅠㅠ 청소 설거지 빨래 다 괜찮은데 밥 하는거 넘 귀찮아요 ㅠ'], ['남이 해주는 요리가 젤루 맛나죠.... 애들 챙겨야해서 어쩔 수 없이 하는것중 하나가 요리인듯요ㅠㅠㅠㅠ'], ['하나같이 다~~~시러요~~~암것도 안하고 누가 해줬음좋겠어요~~'], ['맞아요... 그게 정답이네요ㅎㅎ 그 와중에 뭐가 좋고 뭐가 싫고ㅋㅋㅋ 의미없죠😂'], ['전 화장실청소랑 쓸고닦고 방청소 너무싫어요'], ['저희 신랑이랑 똑같으시네요ㅋㅋㅋㅋ 청소기 돌려달라고 부탁하면 인상부터 팍!!!! 청소빼곤 다 괜찮다고 차라리 다른거 시키라고 해요ㅎㅎㅎ'], ['냉장고정리랑 집안일은아닌데 아이둘 목욕이요 목욕때.거의 짜증 만랩찍네요ㅠ'], ['악!!! 제가 잠시 기억을.... 저도 애둘다 어렸을땐 목욕이 최악이였어요ㅠㅠ 한놈 울고 한놈 욕실바닥 기어댕기고...🤤🤤🤤'], ['저도 설거지랑 음식물쓰레기 진짜 싫었는데 이젠 좀 적응됐구요. 화장실청소는 여전히 싫어요ㅠㅠ 화장실청소만 하고나면 매번 거의 기절해요ㅜ \n빨래,청소,정리정돈은 좋아해요'], ['화장실 청소 엄청 열심히 하시나봐요^^;;; 전 화장실은 씻는김에 설렁설렁ㅎㅎㅎ 신랑이 청소 진짜 싫어하는데 그 중 화장실 청소가 최고래요ㅋㅋㅋ'], ['저는 쓰레기정리요 ㅜㅜ'], ['분리수거부터 갖다 버리는게 보통일은 아니죠ㅠㅠㅠㅠ 생각해보니 저도 싫어하는 것중 하나네요ㅠㅠ'], ['음쓰랑 유리 닦는거요. 유리 닦는건 못하겠어요. 그래서 유리란 유리는 다 더러워요 ㅋㅋㅋ'], ['유리닦기도...하시나요...😂😂😂 차마 그것까진....\n저의 몫이 아닌듯해요ㅋㅋㅋㅋ'], ['재활용 버리는거, 빨래 너는거, 걸레질, 화장실 청소요\n이중에 유독 제일 싫은걸 고르자면\n재활용 버리기ㅋㅋ'], ['엄청 엄청 귀찮죠ㅋㅋㅋㅋㅋ 며칠 쌓이면 플라스틱 대박이고... 연말에는 무슨 박스박스가 그리 나오던지요...퓨우ㅎㅎㅎㅎ'], ['저도 설거지는 차라리 괜찮아요 청소도요 근데 빨래 개는거 너무 시러요  극혐이애요 ㅋㅋㅋㅋ'], ['아 음쓰는 진작에 갈아버리는거 설치했어요 진짜 신랑도 시러하고 저도 시러흐고 평화가왔어요 ㅋㅋ'], ['와우!!!!!!!! 신세계 경험하고 계시겠네요~!!! 개는거 싫어하시는구나...전 넣는거요ㅋㅋㅋㅋㅋ'], ['아기낳고 냉장고정리 1년동안 못했어요 어떡하죠'], ['아.......냉장고 정리....... 아기 어리면 뭐가 어디에 있는지도 잘 모를때죠ㅠㅠㅠㅠ 이유식 거리도 사다놓고 버리고 또 사고ㅎㅎㅎ'], ['그냥 가만히 이불속으로~~~'], ['ㅋㅋㅋ맞아요~정말 집안일은.....퓨우ㅠㅠㅠㅠ 식세기도 사고싶고 로봇청소기도 사고싶고 그러네요ㅋㅋㅋ'], ['집안일은 다 싫어요.ㅋㅋ'], ['그러게요ㅋㅋㅋ 뭐가 더 좋고 싫고 의미없죠😂😂😂'], ['저는 정리하는게 싫어요 ㅋㅋㅋ 남편이 꿍시렁거리면서 해요...'], ['저 아는 언니는 자기 손 닿는 곳에 뭐든 두고싶어하거든요ㅋㅋㅋㅋ 신랑이 결국 포기했더라고요🤣🤣🤣'], ['다 괜찮은데 청소기 돌리는거랑 물걸레질이요 ..  ㅠㅠ 특히 걸레질좀 누가 매일 해줬으면 좋겠어요 ㅋㅋ'], ['걸레질 매일 하시는구나...와우👍👍👍 그러고보니 저도 걸레질 싫어하네요..... 가끔 해요ㅠㅠ'], ['설거지요 ㅎㅎㅎ 너무 싫어요'], ['저 결혼하고 설거지하면서 얼마나 웩웩 거렸는지ㅠㅠㅠㅠ 애 낳고 별별거 다 보다보니 비위가 좀 강해지더라고요ㅎㅎㅎ 그래도 여전히 힘든것 중 하나네요😝😝'], ['ㅋㅋㅋㅋㅋ거의 다네요ㅋㅋㅋㅋㅋ 사실 저도 그래요......😝😝😝'], ['철 지날 때마다 옷정리요~~\n빨아서 넣어 놓고 계절 옷 다시 꺼내 놓고...아...식구별로 다 해놔야 하니..넘 귀.찮.아.요'], ['악!!!! 잊고 있었어요ㅠㅠ 저도 제옷에 애둘 옷에...철마다 작아진 옷 처분하고 넣고 꺼내고.... 어떨땐 며칠씩 걸려요ㅠㅠ'], ['저는 정리랑 청소요. 애들이 아직 어려서 어쩌다(?) 치워도 금세 난장판되니 당최 어떻게 치워야될지 엄두가 안 나요T^T 잘 버리는 스타일도 아니라서 장난감이랑 잡동사니도 많고;; 일단 정리가 안 되니 발디딜 틈이 없어서 청소도 어렵네요;;;; 정리랑 청소는 놓은지 오래 됐어요 ㅡㅡ; 정리랑 청소가 젤 싫어용 ㅠ.ㅠ'], ['버리는것부터 시작인 것 같아요ㅎㅎㅎ 저도 버리려고 마음먹으면 거의 일주일동안 정리하게 되니까 마음먹기도 쉽지 않아요ㅠㅠ 애들 어릴땐 거의 눈 감고 살죠😂😂'], ['빨래 개키는게 젤 싫긴한데..\n이상하게...바닥 걸레질을 안하게 되요 ㅡㅜ 무선걸레도 있는데...']]
    
    5682
    집에 오면 집안일하기가 시러요ㅜㅜ 이직한지 얼마 안 되어서가뜩이나 넘 힘든데아직 결혼 전인데 동거 중이거든요ㅎㅎ둘이 사는데도 뭔 일이 일케 많은지ㅜ빨래강아지 똥오줌치우기설거지청소밥하기다하기 싫어요ㅋㅋ오빠가 집에 늦게 오니자연스레 제가 해야 하네요전 깔끔한 편두 아니라ㅜㅋㅋㅋㅋ집 난장판 해놓구 살아요결혼하면 좀 치워야할거 같은디ㅜ
    
    [['그래서 요즘 가사도우미 업체가 잘 된다고 하더라고요ㅠㅠ 집안일 너무 힘들어요...'], ['전 오빠가 늦게오는데도 오빠가 거의 다 해요.. ^_ㅠ'], ['진짜 집안일 너무 시러여 ㅠㅜ 공감공감'], ['신랑이 손 깁스를 하고있어서ㅜㅜ\n퇴근하고 혼자 설거지 재활용 음쓰버리기 세탁기 2번 돌리고 건조기 돌리고 빨래 정리하고 오늘저녁 내일 먹을것까지 밥 해놓으니 10시더라구요ㅋㅋㅋㅋ10시에 처음 앉았어요ㅜㅜ'], ['일하고 집안일 하는거 힘들죠 ㅠㅠ 집에서 노는것두 아니구 ㅠㅠ'], ['퇴근하고 일부러늦게들어가야하는거 아니죠ㅋㅋ 아무것도하기싫을듯요...ㅎㅎ'], ['저희도 맞벌이고 서로 힘들고 피곤한거알아서 기본적으로 치워야하는것만 치우고 거의 1주~2주에 한번씩 청소업체 불러요 가격도 그렇게 비싸지 않구요 그게 스트레스안받고 편해요ㅠ'], ['맞아요ㅠㅠ 서로 같이일하면 진짜 더 힘든것같아요ㅠㅠ 저희도 한번에 몰아서 대청소 하게돼요ㅠㅠ'], ['저도 겨우 주말에 신랑이랑 크게 싹 청소하고, 평일에는 겨우 청소기 돌리기도 어렵네요ㅜ'], ['흠 저의 미래를 보는 것 같아요..'], ['그래서 안해요 ㅜ 점점 기계에 의존하게 되구요 ㅋㅋ'], ['ㅜㅜ저도 제가 좀 더 집에 있는 시간이 많다보니 제가 하게돼요.. 저는 머리카락 한올도 용납 못해서.. 더 힘들어요....'], ['저두 이직해서 제가 늦게오니 집안일 하는게 더 귀찮네요ㅠ'], ['진짜 일찍오는 사람에 어쩔수없이 주섬주섬 하게되더라구요 ㅜㅜ'], ['하루종일 회사에서 녹초돼서 집와서 집안일 힘들죠ㅜㅜ'], ['기계의 도윰이 필요해요ㅠㅠ 식기세척기,건조기,로봇청소기 필수라고 생각해요~'], ['진짜 공감이용 ㅜㅜ 결혼전에는 부모님이 다 해주셨었는데 ㅜㅜ'], ['저도 처음에는 막 열의 넘쳐서 했는데 점점 귀찮아지더라고요ㅠㅠ'], ['결혼전인 지금도 이렇게 방 청소하기 싫은데 나중되면 더 하기 싫겠져...ㅠㅋㅋㅋ'], ['저희도 그래요 ㅋㅋㅋ 집 들어와서 눕기 시작하면 일어날 수가 없어요 ㅋㅋㅋ 부지런하게 살기 힘듭니다 ㅋㅋㅋ'], ['저도 처음에는 좀하다가 이제는 내버려두게되요'], ['저도 그래요 퇴근하면 기진맥진 입니다 손가락 하나 까딱 할 힘이 없어요'], ['아무래도 일하고 오면 힘들고 귀찮져 언제는 저녁도 먹기 싫어요'], ['진짜 너무 귀찮아요 ㅠㅠ 씻는 것도 귀찮음 ㅋㅋㅋㅋㅋㅋㅋㅋ 힘들면 그냥 몰아서 하고 그러고 있어요 ㅠ'], ['진짜 퇴근하고 와서도 제 2의 일 해야해욬ㅋㅋㅠㅠ 결혼하고 다 좋은데 집안일이 싫어요..'], ['다 그래여 다들 힘들어여ㅠㅠ'], ['현재 제모습같네요ㅋㅋ진짜 다하기싫어요ㅠ 다해놓으면 다음날왜 원상복귀되는건지ㅋㅋ아 진짜 집안일 못해먹겠어영ㅋㅋ'], ['맞아요 퇴근하고 와서도 요리하고 그러는게 힘들더라구요'], ['저흰 둘다 안해요 참다참다 겨우 청소하고 ㅎㅎ 큰일입니당 ㅠ'], ['저희도 맞벌이에 신부 건강이 요즘 좋지않아서 매일 집안일 하기가 쉽지는 않네요 ㅠ 설거지는 못하더아도 1일 1청소는 하고 자요'], ['진짜 엄마랑 살때가 행복한 때였어요 ㅠㅠㅠ'], ['ㅜㅜ 맞아요 집안일이 제일 힘들고 귀찮죠'], ['아이구 저는 출근 시간도 빠르고 회사도 멀어서 집 와서 밥만 해먹어도 금방 잘시간인데...걱정이네요 ㅠㅠ 식기세척기랑 샤오미 로봇청소기 필수네요'], ['피곤하니깐 아무래도 하기쉽지않아여 ㅠ'], ['퇴근 후 집에오면 정말 하기 힘들죠 ㅠㅠㅠ 잘 배분해서 해야할거 같아요'], ['에고 저도 걱정입니다 ㅠㅠ 퇴근하고 집안일까지 .. 멀티는 너무 힘들어요 그래서 초반에 남편분이랑 역할분담이 중요한 것 같아요'], ['저두요ㅠㅠㅠㅠㅠ 가능하면 그냥 주말에 몰아서 해요'], ['집안일 누구나 하기 싫은것 같아요ㅜㅜ부모님 이랑 살때가 편하죠ㅜㅜ'], ['평일엔 식사를 안해서 설거지 거의없고요 더러운 대로 살아요 청소 2주에한번하고 그것도 하기시러 빌빌대네요'], ['맞아요.. 집안일이 왜케 많은건지...일하구와서 하려니 더 힘드네여 ㅜㅜ'], ['맞아요 ㅠㅠ 엄마랑 살다가 살림을 다 제가 하려니 힘들더라구요 늘 막막'], ['정말 은근 할일이 많더라구요 ㅠㅠ 넘 귀차나용'], ['공감해요 ㅜㅠ 일하고 퇴근 후 집 오면 암것도하기싫어요\n저녁도 밖에서 먹고 집 오면 맥주 정도만 마셔요ㅜㅜ'], ['결혼전엔 엄마가 다 해주시니 이렇게 할 일이 많은 줄 몰랐어요ㅜㅜㅋㅋ'], ['저두요 ㅠㅠ 엄마랑 살 때가 가장 편했어요~'], ['ㅜㅜ상상만 해도 너무 바쁠거 같아요ㅠㅠ곧 현실인데..'], ['일주일에 한 두번 도우미 손을 빌리는것도 괜찮은거 같아요~'], ['맞아요ㅠㅠ일도 힘들게하고 집에오면 녹초가되서 주말에 몰아하게되는거같아요ㅠㅠ'], ['퇴근하고 와서 집안일 하기가 쉽지않죠 ㅜㅜ 너무 지치게 하지마시구, 최소한만 하시다가 주말에 같이 하세요~혼자 다하려면 힘들어요 ㅜㅜ'], ['미뤄뒀다가  일주일에 한두번씩 같이해요  이직하고는 당연  집안일 손에 안잡히죠 ㅠ'], ['결혼하기전에 룰 정해서하세요 저도 동거할떄 규칙없이 하다보니살림가지고 엄청싸웟어요 니가하네 내가하네 마네 이러면서요']]
    
    5711
    10)집안일 막둥이재울혀다가 제가잠들어 버렸네요~빨래 널고 나니 막둥이가깨네여 ㅠㅠ우유먹이고 청소기좀 돌려야 겠어여^^
    
    [['오늘하루도 화이팅입니다'], ['하루가ㅜ벌써 즐거운 밤 되세여~'], ['하루하루가 너무 잘간다지요'], ['그쵸 하루가 정말 빨리 지나가지요 ㅠㅠ'], ['조만간 설이겠다지요 ㅜ'], ['ㅠㅠㅠ 그러게요 ㅠㅠ 올해설은 너무 빨라서 ㅠㅠ'], ['오늘 하루도 화이팅 입니다. ㅠ.ㅠ 저도 집안일 이제 끝이요'], ['집안일 을 하면 반나절은 그냥 훅지나가는것 걑아여 ㅠㅠ'], ['맞아요 집정리하고  시계보면 오전시간은 후딱가있죠'], ['그니깐여 ㅠㅠ 늘하는거 없이 쉬지도 못하고 그냥 숙숙 지나가는것 같아요~'], ['저도 집안일 이제 거의 다끝냈어요'], ['레나님 정말 고생하셨어여~'], ['오늘은 신랑 찬스가 있는 주말이라 편하구만요^^'], ['ㅋㅋㅋ 신랑찬스가 가능한 주말이 그립긴해여~ㅋㅋ'], ['앗 신랑분이 어디 출장가셨나요?'], ['아침에 출근 했었디여 ㅠㅠ 독박 육아였어녀 ㅠㅠ'], ['전가득인데 언제해보려나여'], ['저도 다하지 못했다죠 ㅠㅠ 조금만 ㅠㅠ'], ['내일은부지런히해야될것같아여'], ['하실수 있으셔여~!!!내일 천천히 화이팅입니다 ~'], ['해도해도 끝이없다지요ㅜ아휴'], ['맞아요 ㅠㅠ 티도 안나고 말이져 ㅠㅠ'], ['맞아요 ㅠㅠ 진짜 서러워여 어쩔떈 ㅋㅋ'], ['그니깐여 신랑이 집치웠냐 이말항때말이쟈 ㅠㅠ'], ['집안일은 끝이 없구요- 고생하셨어요'], ['정말 끝이 없어여 ㅠㅠ 집안일이 정말 ㅠㅠ'], ['그건 이해해 줄거예여 신랑분이요'], ['이해는 해주는데 정리가 늘안되니 ㅠㅠ'], ['애 셋이면 정리까진 매일 힘들거 같아요'], ['넵넵 진짜 다같다 버리고 싶어여 ㅠㅠ'], ['저도 그래요— 장난감 없음 그래도 잘 놀겠지요 ?'], ['첨부터 없었으면 아마도 그랬뎄죠??ㅋ'], ['그럼 하나하나 정리해야겠네요'], ['조굼식 정라하고 있는데 주의에서 또주세여ㅠㅠ'], ['어른빨래는 주말에만 가능해용 ㅠ'], ['저는 안하면 싸여있어서 안되요 ㅠㅠ'], ['슈슈빨래만으로도 힘들어용 ㅠㅠ'], ['저는 둥이들 빨래더 있다보니 감당이 ㅠㅠ'], ['헉 진짜 맘님 진짜 대단해용 근데 지금 이시간까지도'], ['막둥이 밥먹을시간이 다가오고 있어서여~ㅋㅋ'], ['아 막둥이가 또 맘마를 기다리고 있군용'], ['넵넵~ 맘마먹을시간이라 지금 일어났네요 ㅠ'], ['5 그럼 막둥이 맘마주시고 맘님도 굿밤되세용'], ['넵넵~ 오늘도 정말 고생하셨어요&^'], ['집안일하다보믄 시간다가용'], ['맞아요 시간이 쑥쑥 지나가는것 같아여 ㅠㅠ'], ['자야되는데 댓정리하고가야죠'], ['저도 그래서 아직 못또나고 있다죠 ㅠㅠ 늦게 들어온것도 있고요 ㅠㅠ'], ['많이 하시고 가신건가용'], ['두시간 바짝 달리신했는데 모르겠어녀~ㅠㅠ'], ['두시간이믄 이백개는 하지않앗을까용'], ['ㅋㅋㅋ 이백개~ ?? 두시간에 과연 그정도 했을까요??ㅋ'], ['한번봐보세요 저도궁금하네요ㅋ'], ['ㅋㅋㅋ 지금부터 체크해봐야 겠구만요~ㅋㅋㅋ']]
    
    5733
    밀린 집안일하는중 까페가 궁금한데  며칠비운집정리하느라바쁩니다요맛난점심들 드셨죠   오늘도 많이웃는날요
    
    [['언제나 예쁘십니당^^'], ['찌찌뽕입니다\n\n저두 그말할려고했는데~~~^^'], ['어머나!  감사하다고해야죠 ㅎㅎ'], ['감사합니다'], ['집 비웠다가면 할일이 많죠ㅠ 화이팅이요^^'], ['맞아요'], ['저 모래성 지난번엔 못보았는데~\n담엔 꼭 보고싶네요'], ['12월엔 저도 못봤어요'], ['헐 겨울엔 없나보군요ㅜㅜ'], ['지금도 올라오는거없죠?'], ['네 못본것도 같아요'], ['단속중맞나봐요'], ['단속도 하는군요'], ['그러니 못만들겠죠'], ['그쿤요 근데 보니 넘 바가지도 있긴하더라구요'], ['그쵸 예전엔 안그랬는데'], ['오늘 카페 뒤숭숭해요ㅜ'], ['그러게요'], ['에효ㅜ 나쁜사람 하나땜에'], ['어찌 ㅠㅠ'], ['진짜나빠요ㅜㅜ'], ['여러사람  힘들게'], ['아름답습니다.^^사진이♡'], ['보라의 상징~^^'], ['12월엔 없더라구요'], ['그래요~\n크샌앞에 많은데~'], ['그땐 단속기간이었나봐요'], ['아~ ㅎㅎ\n워낙 바가지가 심해니~ ㅜㅜ'], ['전 모레성만  몰래 찍은적도 ㅋㅋ'], ['전 항상 몰래요~ ㅎㅎ'], ['그쵸 제가안나오면어때요'], ['그럼요~^^'], ['집안일은 그닥 티도 안나는데\n며칠 소홀하면 엉망이 되지요 ㅠㅠ'], ['왜그럴까요'], ['까페가 뒤숭숭하네요'], ['왜뒤숭숭해용'], ['그쵸'], ['뭔 예약사이트가 폐점했다나봐요'], ['오늘두 삶의터전에서 전쟁하다보니 까페를 이제야 이런일이있었군요'], ['저도 이사이트잘몰라요'], ['이사진 진짜  이쁘게 나왔네요'], ['감사합니다'], ['저도 해야하는데요'], ['하하하 뭐떨어져요'], ['누구든 해야해서요'], ['을이시군요'], ['을입니다 ㅋㄱ'], ['전 제가을입니다'], ['바꿔달라고 하세요'], ['안바뀔겁니다'], ['꾸준히ㅓ말씀하시고 칭찬하면 바뀔 수 있어요'], ['아가씨줄 알겠어요'], ['어머나!  복받으실겁니다'], ['진심입니당'], ['감사합니다'], ['아 너무 예쁘시네요^^'], ['너무 감사드립니다']]
    
    5798
    요리한 지 35일차 주부입니다. 3월 결혼식을 앞두고 12월에 이사해서 처음으로 밥을 해봤어요.집안일은 다른 건 다 해도 요리만큼은 친정에서 안했었는데 그래도 어깨너머로 배워서 그런지 금방 익숙해지더라구요.이게 처음 한 밥!!가스가 늦게 들어와서 부루스타 사다가 밥을 했어요.가스 들어오자마자 신랑이 제일 좋아하는 반찬들을 했어요.진미채를 제일 좋아하는데, 냉장고에 들어가도 딱딱해지지 않는 진미채를 만들기 위해 엄청난 서칭을!!요렇게 아침밥도 해주고...출장갔다 온 날은 고기랑 차돌박이된장찌개!!귀찮을 땐 동그랑땡이랑 옛날소세지만...신랑이 느끼한 걸 좋아해서 아침부터 느끼한 걸 만들어줬어요ㅋㅋㅋㅋㅋㅋ고추장불고기만 해주다가 처음 간장불고기 도전!!시부모님 오시는 날엔 소갈비찜에 도전했어요.고기가 다루기 제일 쉬운 이유는 제가 육식파라서 그런가봐요.이사하는 타이밍이 이직 중간 쉬는 텀과 겹쳐서 한 달동안 이것저것 해봤어요. 요리 한달차라 종류가 많지 않은 건 함정ㅠㅜ요즘은 일이 바쁘기도 하고, 신랑이 야식도 안 먹는데 살이 4키로나 쪘다고 저녁에 콩미숫가루만 먹어서 밥을 차릴 시간이 줄었어요ㅎㅎ근데 집안일 계속 해보니까 막상 엄마나 시어머니 만날 땐 절대 음식하게 하기 싫어지네요ㅡㅡ 집안일하시는 거 싫다고 무조건 외식으로 유도를...!!
    
    [['우왕 솜씨가 좋으시네요 ~ 전 똥손이라 결혼하면 걱정이네요 ㅠㅠ'], ['정성이 느껴져요. 대단하시네요ㅎㅎ'], ['맛있겠당 ㅠㅠㅠ 정성이 느껴져요 ~~!!!'], ['그래도 종류 많은걸요~ 저는 메뉴선정도 어려울것같아요 ㅠㅠ  음식메뉴 사다리타기 있었으면 좋겠어요 ㅠ'], ['한달차 치고는 이것저것 많이 하셨네여 그래도 솜씨가 좋으시네요'], ['정성이 엄청 느껴지는 집밥이네요'], ['정성이 느껴져요. 너무 맛있어보여요.. 금손이시네요.'], ['에이~ 요리한디 35일이 아닌 3년5개월 이상 되신거 같아요! 너무 잘하시네요ㅠㅠ 저는 자취10년인데 주방에 들어가본적도 없어요ㅠㅠㅠㅠㅠ😭'], ['요리가 정성이 느껴져서 넘 좋아요'], ['한달인데 종류도 다양하게 맛있게 하시는거같아요'], ['음식들이 하나같이 다 맛있어보여요!! 음식솜씨가 좋으신가봐요 ㅋㅋ'], ['정성이 대단하시네요!! 저걸다어떻게 만드는지.. 솜씨가 부러워요'], ['푸짐하고 맛있게 잘 해드시네요 ~^^'], ['정성이 가득가득한 요리네요 ㅎㅎ 솜씨도 좋으시네요!!!'], ['와 정말 맛있어보여요ㅠㅠ 저는 진짜 요똥인데... 너무 부러워요'], ['우와 정성가득한 음식들만 만들어주셨네요 맛있어 보여요'], ['우와.. 너무맛잇어보이는데요ㅠㅠ? 침고여용'], ['와 정말 요리솜씨 좋으시네요 ... ㅠㅠ 배고푸네요'], ['저는세달째인데 저보다 잘하시네요 부러워요'], ['오 ㅎㅎ완전맛잌오보여용^^정성가득이네용'], ['음식이 다 맛있어보여요~~~배고프네요ㅎㅎ'], ['우와 음식이 전부 너무 맛있어보이네요~'], ['우와 유부위에 뭐 올리신건가요? 참치랑 치즌가...맛있을거같아요'], ['와 금손이시네여 음식빛깔이 다 먹음직 스러워여ㅠㅠ 전 요똥인데..ㅋㅋㅋㅋ'], ['우와 음식이 짱이네요 ㅎㅎ 어떻게 이렇게 하시나요 ㅠㅠㅠ'], ['와 진짜 솜씨 너무 좋으시네용'], ['우와 엄청 맛잇어보여요! 사진보니까 급배고파지네용 ㅠㅠ'], ['얼마 안되셨는데 솜씨 좋으신거같아요~'], ['음식에 재주가 있으신가봐요^^ 특히 갈비찜이랑 고깃국에 무..건져먹고시포용ㅋㅋㅋ'], ['다양하게 요리하셨네요~ 전 해도 안늘어요 ㅎㅎ'], ['우와 진짜 사랑이 아니면 하기힘든~~ 정성이 마구 느껴져요!'], ['와우 한달 솜씨 치고는 일취월장이시네요.\n군침이 도는 사진 감사드려요'], ['솜씨가 좋으신데요..전 3년차주부인데도 아직 할줄아는음식이 없어요ㅠ아무리 블로그 찾아봐서해도 망쳐서 신 랑이 음식안해도된다네요 ㅋ'], ['헉 전 엄두도 못냐겠네요 ㅜㅜㅜ 밥도 차려주고 요리도 하고 해야하는데 조심히 반성하고 갑니당 ㅜㅜ'], ['우와 35일차인데 베테랑 같은 솜씨이신거 같아요~!!'], ['메뉴 진짜 다양하네요~\n신랑님도 시부모님도 많이 사랑해주실것 같아요^^'], ['정성스런 밥상이네요’! 솜씨가 있으세요!'], ['처음하시는것 치고는 너무 잘하시는데요!!'], ['정성이 장난아니네요~~ ㅎㅎ 금방 이것저것하시는거보면 실력이좋으신거같아요'], ['여러가지 다양하네요! 금손이십니다 ㅎㅎ'], ['헐 메뉴랑 너무 잘하시는 빛깔인데요?'], ['우와 식단이 정말 다양해요! 요리는 하다보면 다 느는거같아요~ 이제 수준급같아보이는데요?^^'], ['35일이아니라 35년차 아니세요? 넘 요리솜씨가 좋으신데요?'], ['솜씨가 진짜좋으시네요너무 맛있어보여요'], ['요리 하기 귀찮을때 많잖아요 ㅎㅎ 그래도 요리하는재미가 있을 것 같아여'], ['짧아도 완전 비주얼 굳이네여 ㅎ'], ['요리한지 35일차인데 실력이 아주 굿굿 ~ 소질이 있으시네요~'], ['한두번 해보신 솜씨가 아닌데요? 진짜 요리 다양하게 잘 하시네요~'], ['몇년차 솜씨인데요? 대박이에요'], ['우와 너무 맛있어보여요 ~ 솜씨가 대단하세요 !'], ['우와 너무 맛있는거 잘 만드시네요~~ 전 결혼한지 3개월됐는데... 하신거 반의반도 못해요..'], ['35일차이신데 하신 요리가 굉장히 많으시네요~! 저는 반도 못따라가는거같아여'], ['저는 채식파라서 고기맹ㅜ 야채를 더 다루기쉽드라구요 ㅋㅋ'], ['집안일 힘들죠ㅜㅜ 실력이 좋으시네요 맛있겠어요 ㅎㅎ'], ['35일차 실력이 아니신데용??ㅠ 요리 잘하셔요!!'], ['잘하시느데요 ~?ㅋㅋㅋㅋ 35일차 아닌거같아요 ㅋㅋㅋ 능력자시네'], ['손이 정말 크시네요 손크신거보니 요리 엄청 잘 하실것 같아요 ^^'], ['소박한듯 푸짐한듯\n맛있는 음식들이 가득가득 하네요^^'], ['소질이 있으신가봐요!! 전 저렇게 이쁘게는 안되더라구요ㅠㅠ 솜씨좋으세요~'], ['크~~ 사진만봐도 침넘어가요! 다양하게 잘하시는것같아용:)'], ['35년차 잘못 적으신거 아니죠?? 우와~~ 전... 왜.. 늘지 않을까요 요리가?ㅠㅠㅠ'], ['우앙~ 정성스럽게 느껴지네용~~ 맛있어 보여요~~~ 집 밥 넘흐 좋앗~~>_<'], ['너무 잘하시네요!!!'], ['정갈하게 음식솜씨가 좋으셔요'], ['35일차가 아닌거같아요! 35일차에 이정도시면 나중에는 어마어마한 실력을 가지시겠군요 ㅎㅎ 저도 요리연습좀 해야겠네요'], ['금방 잘하시네요ㅠ!! 저도 얼른 배워야겠어요'], ['와 엄청 다양하게 하고계시네요 ~ 자극받고 갑니당 ! 남은 결혼준비 힘내세요 ^^'], ['요리 안해보신 거 마자유,,,,? 저희 신랑도 진미채 젤 좋아하는데 전 친정엄마한테 받아다먹어요 진미채는ㅋㅋㅋㅋㅋ 자신이 없어여ㅠㅠ'], ['요리를 잘하시는 것 같아요!! 다 맛있어보이네요 아침메뉴도!! 영양생각하시는 것 같구!! ㅎㅎ 행복하세여~~'], ['우와 요리 센스가 만점이시네요ㅎㅎ 신랑분 건강해지시겠어요~'], ['저랑 결혼날짜가 비슷하신 거 같은데 저는 실력이 1도 안늘었어요 부러워요~'], ['아 엄청 배고파지는 사진이네요 ㅠㅠㅎㅎ'], ['요리만 봤는데... 꽁냥꽁냥이 느껴지는 건 저뿐인가요?ㅠㅠㅋㅋㅋ'], ['맛있을것 같아요 !! 아침까지 저렇게 차려주시는거면 대단하세요 ~'], ['넘 맛있어보이는게 35일차같지 않은 고수의 향이 느껴지는걸요~~ㅎㅎ'], ['매 끼니때마다 정성이 정말 느껴지네용!!'], ['집밥 느낌 제대로에용 :)'], ['우아 요리솜씨가 장난아니에용ㅎㅎ'], ['요리 전혀 안하셨다더니 솜씨가 좋으시네요! 불고기 양념도 직접 하셨나봐요~ 맛있어 보여요~'], ['하나같이 너무 정성스러운 요리네요^^진미채 마요네즈 넣으면 냉장고 들어가도 촉촉하고 고소하고 맛있어요ㅋㅋ'], ['요리 한달차 맞으셔요?ㅎㅎ 엄청 수준급이신데요?ㅎㅎ 엄청 정성스럽고 맛나보여요 ^^'], ['솜씨가 좋으시네요~\n점점 어려운 음식들도 척척 잘 하시는거 같아요! ㅎㅎ'], ['한달밖에 안되셨는데 솜씨가 남다르시네요~'], ['와 솜씨가 정말 좋으시네요'], ['오오 갈비찜까지!! 다 진짜 맛있어보여용'], ['와! 솜씨가 있으시네요! 갈비찜까지~대단하시네요!'], ['유부초밥 위에 치즈 올리는거 첨봤는데 오오 완전 맛있을것같아요~><']]
    
    5801
    전업주부 일맘님들 남편 뒤치닥꺼리 어디까지 이해하세요..? 일본와서 아기키우며 전업주부 입니다.직업이 전업주부인거니까 집안일 다 해야하는건 불만이 없고요,남편도 요구하면 잘 도와주는 편 이긴 한데,남편이 쓰레기를 쓰레기통에 직접 넣지 않고아무데나 버리는 습관이있어요..사용한 휴지나, 과자등 비닐포장을 뜯으면 생기는 귀퉁이의 작은 삼각형 부분 있잖아요. 전에 한두번 말했는데 한번은 안그러겠다고 하고 그다지 안고쳐지고두번째는 왜 자잘한 것 가지고 그러냐고 하더라고요.저도 사실 거실 등에 제 짐(쓰레기는 아니고 소품이나 등등)을 늘어놓기도 하기 때문에 , 남편도 불만이 있을수있겠지,그러니까 내가 그냥 군말없이 다 치우자 싶으면서도아니 남이 쓰고버린 자그만 쓰레기까지 내가 다 치워야 하나 하고한번씩 화가 치미네요. 어제밤도 홧병이 날뻔...전업주부니까 이해해야 하는걸까요..?저는 한국에서 집안일 도우미 오실때도 그런 제 개인쓰레기는눈에 안보이게 직접 다 치우고 일 시켰던 성격이라..ㅠㅠ
    
    [['저도 어느정도 해주다 한 번씩 폭발해요~신랑은 그 때마다 알았다고 말은하는데 잘 안고쳐지더라구요..'], ['폭발 동지 반갑습니다... 아 폭발할때마다 막 늙네요 ㅠㅠ'], ['집 군데군데 작은 쓰레기통 놔뒀어여 정말 큰것보다 작은것들이 오히려 더 화나는거 같아여…'], ['맞아요 작은것들은 해줘도 티안나고 할려니 열받고 힘들고 ㅠㅠ 더 화나요'], ['음...?\n내가 만든 쓰레기는 내가 쓰레기통은\n내가 쓴 물건은 내가 제자리에 \n가 맞는 거 아니에요?!\n전업주부든 맞벌이든....'], ['저도 그렇게 생각했는데 안그런 사람과 살다보니 그게 아닌가....? 하는 세뇌가 된것 같아요;;'], ['남편만 그러면 다행이게요.애들도 똑같이 닮아 셋이 여기저기 버리며 온 집을 초토화시켜요. 그림자처럼 따라다니면서 치웁니다.잔소리 하는 게 피곤해서요.그러니 습관이 안 고쳐져요. 정색하시고 초장에 습관 잡아주세요'], ['으악 애들까지 그러면 힘드시겠어요 ㅠ 초장이 이미 지나버려서 고민스럽네요;;'], ['전업주부 면이해해야되고..\n이런걸 떠나서 성향이신거같어요ㅜ\n저도 가끔 성질날 때가 있는데🥵\n새끼 손톱만한 동그란 투명한..여드름패치를 늘쓰는남편.\n샤워하면서 그 패치를 욕실 벽에 붙여놔요🤯처음에는 투명해서\n안보이다가 점점 시간이 지나면 \n그게 누렇게 뜨면서 ...\n여기저기 보이는데 정말 폭발해요💭'], ['ㅋㅋㅋㅋㅋㅋㅋ그 패치 뿔은거 상상했어요 ㅋㅋㅋㅋ열받으실텐데 죄송해요 ㅠㅠ'], ['저도 좀...처음엔 그렇게 생각했는데 저희 신랑은 아침 7시반에 나가서 저녁11시는되야 집에들어와서;;; 주말에 아이들 봐주는걸로 만 만족해요..불만을 가지자면 싸우게 되더라구요..그래서 그냥 그러려니 하고 신랑이 쓰레기 버리는거나 주말에 요리를 해주는데 그걸로 만족해요...스트레스 받는다고 생각하면 더 화나니까 ...그냥 다 제일이라고 생각해요^^;;'], ['남편분이 그렇게 바쁘시면 춈 이해는 해드려야겠네요 ㅠ'], ['집안일 1도 않하는 남편!!그냥 포기요!! 🤣밥 먹을때 부르면 제때나 왔으면 좋겠어요!! ㅠㅠ'], ['222222\n공감백배입니다'], ['3333333ㅠ'], ['44444ㅎㅎ'], ['밥 굶기셔요 ㅠㅠ'], ['저도 1보다 2번 더 추천드려요 어차피 매번 잔소리도 잘 안먹혓던거고, 날잡아서 아주 심각하고 진지하게 나 너무 힘들고 스트레스 받는다면서 잘 얘기해보세여ㅠ'], ['제말이 바로그말입니다 종취급....ㅠ 하 한국말로 하면 딱 어감전달이 되는데 외쿡인이라 그참....'], ['깔꼼히 다치우고 나서 만족스러워하고 있을때.. 고런 자잘한게 눈에 들어오면 좀 승질나죵 ㅠㅠ \n\n저희 신랑은.. 저두 다른건 그냥 나도 그럴때 있으니까~😁하고 신경 안쓰는데요..\n회사에서 빨리빨리 움직이며 작업도 해야하는 일인지라 거기서 급하게 버리지 못한건지 띠었다 버린 테이프, 클립, 고무밴드, 스탬플러심 이런거가 포켓 여기저기서 나올때가 많아요 거기서 좀 버리지 왜 갖고 집까지오나; 그리 바쁜가 싶다가도..세탁기에 빨래할때 깜빡하고 체크 안하면 아오~ 난리나유;;; \n버릇이 됐는지; 집에서도 그 자잘한 쓰레기를 인마이포켓 해부러서 사탕 봉지, 껌 종이.. 뭐 이런게 나오고 하니까 빨래할때 다른옷들도 상하고 매번 그런거나오니 힘들다! 큰소리 내니까 ㅋㅋㅋ 깨갱한 후로 좀 조심해주고 줄어들더라구요 에휴~~~🤨🤪'], ['아니 그걸 왜갖고오시나요 ;;; 허리끈에 봉다리 하나 매달아서 거기 버리라고 하셔요!!^^'], ['ㅋㅋㅋㅋㅋ 그러니까요 ㅋㅋㅋ 우선 일터는 깨끗이 하고 버리지 않는다는거;;;;;; 자기도 모르게 손이 알아서 인마이포켓;; 막말로 돈도 아니고 ㅋㅋ 쑤레기를;; 아오 혈압오루죠 하하하하; 요즘에는 매일 말해서 그런지 많이 포켓이 깨끗해져서 와요 하하핳'], ['헉... 전업주부면 집안일을 전담해야한다는 생각을 가지고 계시다니 너무 존경해요 ㅠㅠ 저도 전업이지만 애 키우는 것도 하기에 가끔 한번씩 집안일 시키고있는데...  책상위에 과자 부스러기 안치우는건 눈감아주고있는데... 본인이 먹은 간식접시 안치우거나 우유마시고 컵 그대로 식탁위에 놓아두면 폭발해요...'], ['컵은 그냥 저는 내가 주부니까...남편이 회사가느라 바쁘니까...하고 그냥 치우는데 그놈으 쓰레기...써레기써레기가 문제네요 ㅠㅠ'], ['전 맞벌인데도 제가 버려요 백번을 말해도 안 하네요ㅠㅠ'], ['으악 그럼 전 대 폭발 해요...! ㅠㅠ 힘드시겠어요'], ['전업주부면 왜요? 월급받는것도 아닌데 ㅋ 육아 정확히 반반 하는거면 모를까...근데 말해도 안고쳐져요 옷 좀 뒤집어 벗지 말라고 십년을 얘기해도 안 고쳐지는데 정말 병인가 싶고 그 엄마까지 짜증나요'], ['저도 그점이 불만인데 어떨땐 그냥 뒤집힌채 개어버립니다 ㅋㅋ'], ['코..딱지요?ㅋㅋㅋㅋ 밥에 넣어주시면 안...될까요???'], ['아놔 저도 어제밤에 폭발해서 오늘 아침엔 머리까지 아푸더라구요ㅠㅠ \n저도 잔소리 하시 싫은 타입이라 제가 하고 마는데 오래 길들여진 버릇이라 쉽게 못고쳐지는거 같아요ㅠ \n저희 남편은 쓰레기도 쓰레기지만 손씻을때 수건 쓰고나면 수건 걸이에 안걸어놓고 세탁기 위에 올려놔요 \n별거아닌거 같지만 왜 늘어뜨리냐고여ㅠㅠ \n저희 남편은 정말 출근 빠르고 퇴근 늦어서 정말 손하나 까닥해주는거 없는데 자기가 먹은거 좀 치우고 어지럽히지만 않았음 좋겠네요'], ['으윽 저도 머리가 아팠어요~! 저도 잔소리하기는 싫어서 제가 하다보면 폭발을 하게 되네요...'], ['예리한 지적이십니다. 회사에서 휴지를 그리 늘어놓고 퇴근할까요??'], ['전업주부니까 해주는범위인게 아니라 남편분이 그런습성이라 치워줘야하는거아닌가요?\n전 전업주부아닌데 남편이 저러고 있어서 제가 치우고 있습니다'], ['저는 전업아닌데 그러면 뒤집어 엎어버릴겁니다 ㅠㅠ 대단하셔요'], ['봉지찢어서 버리는놈을 엎어버리고 이혼하고 다른남자랑 결혼하면 그놈은 양말뒤집어서 구석에 처박이놓더라구요...결국 이놈이 그놈이고 그놈이 이놈이더라구요'], ['흐미 손발톱은 춈... 아부지 나뿌셨다 ㅠ 어머님이 대단하시네요...!!'], ['그대로 두어 보심이...저도 남편옷있는방 허물들은 터치를 안해요 더이상 ㅋㅋㅋ'], ['전업주부=가정부 착각하는것 같아서 가끔 저도 짜증나요.남편방 알아서 청소하라고 냅두고 쓰레기 아무데나 버리면 즉각 얘기하는편이예요..토달면 입주 가정부에 베이비시터 고용한거면 월급부터 검색하라고 따질려고 했는데 아직까진 토는 안달더라구요(*´-`)'], ['그니까요 가정부... 아니 전 가정부(?) 도우미아주머니 와도 그렇게는 안한단말이죠 ㅠ 즉각 말씀하시기도 피곤하실텐데 흑'], ['쓰레기통을 목걸이처럼 목에다가 걸어주세요. ㅋ\n진지하게 고민하시는데 기분 나쁘셨으면 죄송해요.\n\n잔소리 밖에 없죠. 아니면 더이상 뒤치닥 거리 못하겠다고 파업 하세요.\n\n아기거만 하고 다른거 다 직접하라고선전포고 ㅋ'], ['전업과는 상관없는거 같아요. 테이블이나 식탁근처에, 100엔숍에서 파는 작은 휴지통 놔보세요.'], ['저희집 냥반도 지나간 자리에 항상 쓰레기가ㅜㅠ 게다가 뭐 꺼내서 쓰면 제자리에도 안놔요… 가끔 울화통 터지지만 그냥 제가 치워요ㅋㅋ 하나하나 잔소리하면 서로 빈정상하고 남편은 남편대로 쉬는날 거의없이 밖에서 일한다고 고생하니깐… 그리고 다른 일들은 같이 잘해주니깡… 진짜 참다참다 이건 아니다 싶으면 아주 화를 눌러삼키고 미소를 띄며 큰일 아닌것 처럼 애교를 섞어가며-_- 말합니다 ㅋ'], ['저희 아빠는 본인이 드신걸 하나도 쓰레기통에 안 넣어서 엄마가 40년 넘게 평생 사는 중인데도 뭐라고 해요ㅋㅋㅋ 약 봉지 찢어서 먹고 그 약 봉지 그대로 놔요. 근데 저희 남편이 약 봉지를 쓰레기통에 안 넣는 거에요. 완전 뭐라고 했죠ㅋㅋ 근데 요즘도 가끔 안 버리고 그대로 놓더라구요... ㅠㅠ 계속 뭐라고 할 생각이에요.'], ['똑같아요 귤 껍질이며 과자 봉다리며 ㅋㅋㅋ 드신거 그대로 놔요ㅋㅋㅋ 엄마는 너네 아빤 이 평생 본인이 먹은 쓰레기를 쓰레기통에 넣지 않는다고 ㅋㅋㅋㅋㅋ 진짜 엄마 존경해요ㅜㅜ 전 제 남편이 그러면 쓰레기통에 갖다버리라고 그때마다 뭐라고 하는데 ㅋㅋㅋ'], ['과자든 음식이든 잘흘리는 남푠이라 신혼초반부터 저는 흘리지마라 ~ 이거좀 (남푠이만든쓰레기)버려라 ~ 수시로 이야기했더니 살짝 주눅든거같아 가끔 미안할때도 있어요 ㅠ.ㅠ 그래도 너무 안치워요 .. 정리잘하고 잘치우는사람들은 그냥 타고난거구 남자들은 드문거같아요 ㅋㅋ\n'], ['집에 오면 아무것도 만지지 말고 들어왔던 그대로 나가줬으면 좋겠다 싶을 때 많아요😑 먹고 자고 응가싸는 것조차도 야무지게 하질 못하네요😑 가끔 자기를 존중해 달라고 하는데 너부터 사람답게 좀 굴어보라고 하고 싶어요. 인종과 국경을 초월하는 건 사랑이 아니라 수컷들의 등신력 아닐까요 ㅋㅋㅋㅋㅋㅋ'], ['읽다가 대공감요 ㅋㅋ 어찌 저런 모지리가 회사를 들어가 다니고 있을꼬... 싶을 때가...;;;'], ['양말 뒤집어놓는거랑 쓰레기 그냥 식탁위에 올려놓거나 싱크대에 넣는거 ㅡㅡ.... 진짜 너무 너무 싫은데요 그냥 참고.. 참다참다 싸움 커질때 있음 그때 다 폭발해서 말해요 ㅋㅋㅋㅋ'], ['생활 습관은 전업주부인 것과는 상관없지 않나요!\n저는 그런 부분은 아이가 그대로 보고 따라하니 지나간 자리는 깨끗이 하자고 이야기하니 바로 안 그러더라고요 워킹맘이라고 피곤하다고 아무데나 버리진 않죠^^;; 그냥 습관인듯요'], ['쓰레기뿐이겠어요\n옷꺼내고 반쯤 열어둔 서랍하며 \n클로젯은 늘 열어제쳐두고 \n세탁옷은 세탁통에 반쯤 걸쳐져 있고 \n불은 방마다 다 켜고 다니고 \n자고 일어난 이부자리 정리는 버럭안하면 안해도 되는줄 알고 \n의자에 걸쳐둔 외투며 \n현관에 널부러진 가방 등등..\n다 열거하자면 댓글등록 글자수 오버 될거 같은데요 ㅎㅎㅎ\n미친년처럼 버럭도 해보고 타일러도 보고 했는데 10년넘게 변함없는 한결같은 신랑입니다 \n\n근데 언젠가 인터넷 댓글에서 \n“그래도 저 덜떨어진 사람이 회사나가서 매달 따박따박 돈 벌어다주는거보면 기특하다”고..ㅋㅋㅋㅋ\n신랑 뒷치닥거리 하면서 속에서 천불 올라올때마다 저 한마디 댓글을 떠올려요 ㅎㅎㅎ'], ['제 남편이랑 똑같아서 뻥 터졌다는 ㅋㅋㅋㅋㅋ 옷 꺼내면 안에 개놓은 옷을 다 헤집어놔요. 양말도 위에서 꺼냄 되는데 위에껄 꺼내도 헤집어놓구 ㅋㅋㅋㅋ 등등 진짜 많다는ㅋㅋㅋ 이불도 안 개고 나가요 아주 호텔 체크 아웃 하고 나간 사람 같다는. ㅡㅡ불도 안 끄고 나가고 붙박이장 문도 다 열어놓고 어후'], ['이건 남편님 습관으로 전업주부랑은 전혀 상관없능거 같아요 \n아무렇게나 놔두다가 아기가 주서먹으면…이건 아기를 위해서라도 고치셔야할꺼같아요'], ['전업주부지 시녀는 아니잖아요.. 아기들도 말배우기전에 ゴミはゴミ箱ポイポイ배우는데 쓰레기를 아무데나 버리다니ㅜㅜ'], ['저는 매일 욱하고 한번씩 폭발해요. 문화적 차이인지 개인 성향 문제인지 지치네요 ㅠ'], ['저희신랑도 못고치는 버릇이 있었는데 뭐 거의 포기상태였었는데요. ㅜㅜ 강아지를 키우기시작하니 이것저것 먹어대서 고치드라구요. 그리고 요즘은 아이가 어느정도커서 다 너보고 배운다고 아빠가 그래서야 나중에 아이훈육할수있겠냐고 말하면 조금은 노력하려고 하드라구요.'], ['전 밥먹고 고대로 일어나서 쇼파거서 핸드폰 쳐보는 것 땜에 지금도 한판하고 너무 열받아서 글 올리려했어요ㅠ.. 휴 지치네요\n임신중이라 그런가 첫애 케어에 남편까지 뒤치닥거리해야한다는게 자꾸 욱해요 ㅠ'], ['이거 습관이에요. 저희 남편도 그런데 저희 시누도 그러더라고요. 어머님이 다 해주니까 자각을 못하더라고요;;'], ['전 식구들이 테이블에 쓰레기 버리면, 테이블이 쓰레기통이야?? 하고 말해줘요.거실 바닥에 떨어져 있으면 우리집이 쓰레기통이야?? 합니다..근데 잘 안고쳐져요. 걍 입꾹하고 제가 치우는게 빠르기도 하고 ㅠㅠ'], ['전업주부를 떠나서 쓰레기는 쓰레기통에 분리해서 버리는게 기본아닌가요. ㅎ 내가 전에는 그 기본이 안되었던 사람이었는데 깔끔쟁이 남편과 십년살다보니 바뀌었어요. 가끔 내자신이 놀랍다는... 애가 배운다 생각하니 습관이 바뀌더라구요.'], ['전업주부를 떠나서 그건 아닌 것 같아요ㅠㅡㅠ\n전 막 따지는 성격이라..\n그런식으로 쓰레기 놓으면 전 남편 퇴근후 저도 집안일 퇴근할 듯요..\n부인이 남편 회사일할수있게 도와주면 남편도 부인이 집안일 더 수월하게 할수있도록 돕던지 적어도 방해는 하면 안될 것 같아요ㅠ 전 집안에선 제가 독제자라 애들이든 남편이든 빨래는 빨래통에 그릇은 설거지통에.. 분리수거는 분리수거함에 안들어가면 폭발이에요..ㅋㅋㅋ 참고로 애들은 6살 4살 2살이지만 엄마의 괴팍함에 두려워 다들 잘 정리합니다.. 남편께 괴팍함을 보여줄 때인 것 같네요..ㅎ']]
    
    5913
    바이러스 덕에 못나가고 집안일하면서 지내고 있어요... 저번주까지만 해도 저희 지역은 확진자가 없어서 안심이었는데이번주부터 매일 늘어나고 있어요그후로 아예 외출을 못해서 집안일이며 귀찮아서 미뤘던 일들 하고 있네요봄이오니 기분 맞춰서 포근한 차렵이불로 교체하구요~침구 바뀌면 우리 고양이들이 제일 좋아해요~  이사하면서 정리 없이 마구잡이로 쑤셔놨던 물건들다시 꺼내서 차곡차곡 넣어 봤어요~전에 물건 넣을 공간이 없었는데 공간이 생겼어요!!!! 시간이 남으니 수건도 막~ 호텔식으로 접어보고요~ 거실은... 전 사진이 없어서 티가 안나지만 고양이들 캣타워 다시 재배치하고 쇼파위에 패드 하나 깔아서 분위기 변신을... ㅎㅎ고양이 반려중이라 고양이들 살림이 많아요  저희집 싱크대가 무광도어인데 냥이들이 발자국을너무 남겨놔서 에탄올로 지우고 있어요~잘 지워지네요~사실 손잡이가 있어서 사람손자국은 거의 없어요 ㅎㅎ 그리고 매일 뉴스를 보면서 이런 인테리어 소품도 만들어요무지개 마크라메입니다~단순반복이라 뉴스보면서 하기 좋아요~모두들 걱정 많으실텐데 건강 조심하세요~
    
    [['다들 똑같군요~미뤄두었던 집안일 하는거요 ㅎ냥이 넘 귀여워요'], ['그러게요.. 집에만 있으니.. 뭔가 하나씩 눈에 들어오네요~\n냥이는 사랑이죠~~~'], ['하나씩 눈에 들어오는거 왜케 공감 되죠? ㅎ화장실 줄눈까지. .  큰일입니다'], ['아 화장실 줄눈 200프로 공감해요'], ['저 수건 접는 거 저도 해보고 싶어요'], ['저도 이번에 처음 해봤는데 해보니 괜찮더라구요\n어렵지 않으니 해보세요~~ ^^'], ['씽크색이 넘 예뻐요^^'], ['역시 싱크대는 맨날 칭찬 받아요~ \n우리집에서 제일 예뻐요~~~'], ['앗 귀요미들!'], ['귀요미들 보는 재미에 집에 있어도 심심하지 않아요'], ['저희집도 한마리ㅎㅎ'], ['옴마나 백설기 같네요 쿠션 이뻐요이 포즈는??? 안에 사람든건 아니죠?????'], ['자주있는 흘러내리냥 포즈에요ㅋ 제가보기엔 불편해보이는데.. 제딸은 괜찮나봐요ㅋㅋㅋ'], ['싱크대 색상 넘 이쁘네요~'], ['싱크대 색상은 늘 칭찬 받아요.. 감사합니다.. \n저도 막 뿌듯하네요'], ['저도 고냥이 짐정리 해야긋네요'], ['옴마나~~ 집사님~~~ 반갑습니다~~~\n오래된 스크래쳐도 버리고 애들 용품들 구조도 바꿔줬더니\n냥이들이 더 신나해요~~~ ㅎㅎ'], ['저희도 이사온지 7개월이 넘었는데..직장 핑계로   밀린 정리  해야겠어요.울꾸까들 짐정리도 같이요~~'], ['하얀 고양이 너무 로망이에요푸릇푸릇 화분도 있고 부럽습니다 저희집 냥이떼는다 엎어버리고 뽑고 흙을 파서 다 치웠거든요'], ['싱크대도 넘나 예쁘고 프로필 사진 냥님 솜뭉치 발에 심쿵! 하고 갑니다'], ['저희집은 딱 두개가 이뻐요싱크대랑 고양이요 ㅎㅎ솜뭉치 귀엽지요'], ['냐옹이 넘나 귀여워요!! 코로나 조심하세요^^'], ['감사합니다 앙리앙리님도 조심하세요'], ['이참에 푹쉬는것도 좋을듯해요'], ['네 그래야겠어요 봄이 올때까지 기다려보아요'], ['야옹이 넘 귀엽네요~'], ['너무 귀엽죠^^ 맨날봐도 안 질려요'], ['와~집도 이뿌고 금손이시네요~!^^'], ['집칭찬도 금손 칭찬도 좋으네요 감사합니다'], ['사실 고양이 자랑이 하고 싶었어요 ㅎㅎ 감사합니다'], ['냥이들 너무 귀엽네요'], ['네 다 매력이 있어요같이 있어서 다행입니다  고양이 이쁘게 봐주셔서 감사합니다'], ['고양이 너무 귀여워요 ^^'], ['하는짓도 얼마나 귀여운지 그나마 얘들때문에 웃어요'], ['저도 집콕ㅠ 집이 따듯하고 아늑해 보여요'], ['네 추우면 고양이 안고 있으면 따땃해요냥이 체온이 사람보다 높거든요 감사합니다'], ['금손이세요 😘냥이들은 너무 사랑 스럽군요'], ['만드는걸 좋아해서요감사합니다 뭘 찍어도 고양이가 함께하면사진 퀄리티가 올라가요 ㅎ 감사합니다'], ['주방색 너무 이쁘네요^^ 이쁘고 깔끔하게 꾸미고 사시네용~~부럽슴다'], ['사진으로 봐서 그렇지 그리 깔끔하지 않아요 지문닦은것도 두달만이에요 ㅎㅎ그래도 칭찬해주시니 감사합니다'], ['고양이 호텔 같습니다. ㅎㅎ'], ['놀러오시는분들도 그래요 ㅎㅎ'], ['싱크대 손잡이 넘 예뻐요. 어디서 구입할 수 있을까요?'], ['오영민제작소 검색하시면 됩니다 인터넷으로 샀어요'], ['반려묘와 함께 생활하실때는 전선 케이블 보호 및 정리를 위해 몰드 설치하시면 깔끔합니다. ^^;\nhttps://smartstore.naver.com/ductmold'], ['푸힛~ 전선 씹는 고양이가 있는거 어찌 아셨을까요~~~ ^^ 감사합니다'], ['저희집도 냥이두마린데 두마리다 똑같이생겼네요 ㅎ'], ['ㅋㅋㅋ 저희집 고등어랑 러블이는 엄청 투닥거리는데 두녀석 사이가 좋으네요~\n다정하게 자는 모습 귀여워요~'], ['정말 생산적으로 하루하루 보내시네요~~~~  전 밥 하는 것만도 충분히 힘든데 ㅜㅜ'], ['조금씩하고 있어요저도 하루 세끼 해먹는게 일이네요누가 해주는밥 먹는게 젤로 좋아요 ㅜ'], ['싱크대 너무 예쁘네요, 무슨 색으로 하신 건가요? 저 지금 싱크대 색 고민 중이라서요ㅠㅠ'], ['사제 업체에서 pet 무광도어로 골랐고 샘플에서는 피스타치오라고 표기되어 있었어요딥그린 느낌이 강하게 나요'], ['감사합니다. 보조주방 딥그린 색으로 할까 고민하고 있었는데 이 색으로 할까봐요^^'], ['이쁘게 봐주셔서 감사해요 뭘하든 이쁠꺼에요'], ['고양이 너무 귀여워요!!'], ['쟤들은 귀여울라고 태어난 생명체에요 ㅎㅎ']]
    
    5949
    맘님 남편의 집안일 점수는?/하는 것은? 저녁 먹고 다시 폰행 이네요.. 맘님들은 맛저 하셨나요? 오늘 남편이 요리 해줘서 또 간신히 저녁 해결 했네요 ㅎㅎ 사실 저희 집 남편은 빨래.쓰레기 치우기 .요리 등 하는데. 맘님들 남편은 어떴나요??ㅋㅋㅋ 급 궁금하나요 저는 별 5개의 3개네요. 맘님들 정보 공유해요~❤
    
    [['별0개요!!!!!'], ['집안일 아예 안하나요??'], ['네. 냉장고 정리 아아아주 조금에다 하수구 머리카락 빼준다고 생색내요... 차라리 분리수거랑 쓰레기를 버리지.'], ['ㅜㅜㅜㅜㅜㅜ'], ['집안일을 안하나요?'], ['먹고 싶은것이 뭘까요? ㅎㅎㅎ ㅜㅜㅜㅜㅜㅜㅜㅜ'], ['ㅜㅜㅜ 이거라도 하는것이 다행인걸까요?'], ['ㅎㅎㅎㅎ 감사해요~~'], ['별그림 마저 다 지워야해요ㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋ 집안일을 안하나요?'], ['회사일하는것만으로 만족합니다ㅋㅋㅋㅋㅋㅋ'], ['ㅎㅎㅎ 그렇다고 봐야 하나요?'], ['어쩌다 청소기? 어-쩌다 설거지? 어쩌다 요리? ㅋㅋㅋㅋㅋ 가뭄에 콩나듯.. 한번씩 도와주고 그래요'], ['ㅋㅋㅋㅋ 그게 어디라고 해야 하나요?? 그래서 점수는???'], ['분리수거 빨래세탁기넣기 건조기돌리기 주말설거지 끝이네요 ㅋ 2점요 ㅋ'], ['그래도 2점.. 어떻게 봐야죠?'], ['음... 별5개중에서라하니.. 별100개중 5개정도요ㅋㅋㅋㅋㅋ\n주말에 밥준비하면 주방알짱거리기만하구요. (애들은 둘이 싸우고있고ㅜㅜ신랑은 주방알짱ㅋ).. 그런데. 담배피러가면서 쓰레기 버려줘서 5개요ㅋ'], ['100개 중 5개라.. 어떻게 봐야 하죠?'], ['100점만점에 5점으로 보심되여 ㅋㅋㅋ'], ['ㅜㅜㅜㅜ'], ['아. 근데 집에 안붙어있어서 가산점? ㅋㅋㅋㅋㅋㅋㅋ'], ['ㅜㅜㅜ 도와주지도 못할망정.. ㅜ'], ['...... 할말이 없네요ㅜㅜ'], ['저두 같아용 ㅠ 자긴 제대로 안하면서 엄청 잔소리해요. 애를 제대로 봐주면서 저한테 청소할 시간을 주는거도 아니면서 ㅠ'], ['ㅜㅜㅜㅜ 각자가 맡은 자리에서 최선을 다해야죠!'], ['맞벌이면 당연 도와줘야하고 외벌인데도 도와주면 얼마나 도와줫냐보다 그냥 별5개일듯여 ㅋㅋㅋ 실상은 외벌이지만 도와준거 맘에 안들어도 다음에 시키려고 그냥 넘어가옄ㅋㅋ'], ['ㅋㅋ 도와주는게 어디라고 봐야되나요?'], ['ㅎㅎ 밖에서 돈버는일을하고 저는 집안일을 맡았으니 어쩔수없죠~ 대신 서로 바깥일 집안일에대해선 노터치에옄 맘님 남편분은 최곤데여 ㅎㅎ'], ['ㅎㅎ 감사해요~~ 노터치를 합의해 정말 다행이네요 ㅎ'], ['청소기먼지빼기 쇼파밑 청소 빨래개기 요정도만도 감지덕지해요~ 화딱지날때도있지만용^^'], ['ㅋㅋ 도와주는게 어디라고 봐야되나요?'], ['별1개? 한달에 설거지한번 분리수거한번 청소기한번? ....왜 그러는걸까요 울신랑은 ㅡㅡ'], ['ㅜㅜㅜㅜ'], ['별 5개요~\n집안 휴지통비우기 분리수거 설거지 어항청소 먹고싶은거 포장해오기 ㅋ\n저도 맛있는 요리해주고 뒷바라지 합니다~^^'], ['도와주는 것이 어디에요?'], ['저도 별3개?\n시키면? 청소기 밀기 ㅋ, 요리, 장보기, 설거지는 해주다 식세기 사주고는 일절 안함, 쓰레기 분리수거, 음쓰 버려줘요\n아... 별4개 줘야하나요 ㅎㅎ'], ['식세기 사주고 안하는건 생색 아니에요?ㅎㅎ'], ['좀 좋은놈으로 사주지\n젤 싼놈으로 사줘서 맨날 끝나는 시간 맞춰서 문열어요. 제가 수동으로 ㅋ\n자동 문열림 광고보고 바가지 긁었네요~~'], ['ㅎㅎㅎ 식세기를 사도 불편하네요~'], ['전별다섯개요~퇴근후 육아전담이니깐ㅎ전 그걸로 대만족입니닷 🤣'], ['만족 하시면 되죠. 이제 바랄게 뭐 있습니까 ㅎㅎ'], ['ㅎㅎ욕심이라면 가정부를 세워주는거🤣🤣'], ['별점은 1정도?\n각자 맡은일 열심히 하는걸루~'], ['ㅎㅎㅎ 각자 맡은 곳에서 최선을 다하는 것이 제일루 좋죠 ㅎ'], ['양쪽 화장실 청소(락스성분 청소세제때문에), 분리수거, 냉장고 정리(본인이 좋아해서..), 건조기 먼지 제거, 무선청소기 헤드 필터 먼지망청소, 공기청정기 먼지정리 겉필터 씻기, 본인쉬는 날 가끔 요리.설거지 정도요? 별은 음 몇개줘야할지~'], ['많이 하는데요. 별 3개 줘도 될듯요 ㅎㅎ'], ['근데 잔소리가 심해요.. 하 냉장고 냉동고 한번 뒤집고 선반 뒤집는 날엔 귀에서 피나요😂😂 미니멈라이프(신랑)과 맥시멈(저)이 만나니 불편한쪽이 하더라구요'], ['그건 어쩔수 없는건가요?'], ['별점 0개요~\n손하나 까닥안합니다'], ['ㅜㅜㅜㅜㅜㅜ 요즘에도 있나요??'], ['네 어쩌다 쓰레기버려주는 정도네요~'], ['ㅜㅜㅜ 어쩌다요?'], ['집에 매직캔 쓰레기통 비울줄도 몰라서 제가하고~얼마전 드라이기 걸려고 본드녹여서 붙여서 거는거 모른다고하고ㅡㅡ설거지 쌓여있어도 지밥먹고 그위에 그대로 쌓아놓고~~\n말을 하면 에효 저만 답답하네요'], ['답답하시겠어요 ㅜㅜㅜ'], ['ㅍㅎㅎ...저녁밥 먹고 퇴근해줘서  별 2개..\n집안일 안해줘서 별1개..(뒷손이 더감요) \n'], ['ㅋ 이걸 좋다고 해야하나요? 말아야 하나요?'], ['사는 저는 좋은데 애들한테는 그다지 좋은편은 아니네요..\n늦게 들어오니 아빠보는날이 거의 주말밖에 없으니까요..'], ['그건 맞네요. 장단점이 있네요~'], ['많이 하네요~~ 좋겠어요 ㅎ'], ['가전제품 달인이네요 ㅎ'], ['늘늦게퇴근하는통에못도와줘서미안하다고식세기랑건조기들여줫어요~ 그래두제가하는일에잔소리1도없고 애들이랑잘놀아주고 여유있을때는말하지않아도 건조기안에세탁물잇으면꺼내서정리해주고 쓰레기분리수거랑음쓰버리기.청소기돌리고물걸레까지. 애들이랑놀아주기.등등 다른신랑님들해주는건알아서해줘요~'], ['좋으 시겠어요 ㅎ'], ['전 집안일 남편은 애들 케어를 택했어요!! 설거지할때 애들씻기고 음식할때 애들캉 놀아주고 청소할때 애들데리고 나가주고..  저는 이걸 택했네용'], ['구래도 집안일 혼자 하면 힘들지 않나요??'], ['힘들어요ㅜㅜ 근데 애들이랑 잘놀아주고 엄마가 못해주는 부분 몸으로 논다든가 블럭만들기라든가 로봇조립하는모습 보면 내가못해주는부분 해주니 위로가 되더라구요'], ['장단점이 있는거 같아요~'], ['주말동안 저 만삭에 생일찬스라 그런지 \n점심저녁 첫째랑 같이 먹을 요리하고 첫째랑 놀아주고 첫째목욕에 욕실 바닥청소 했네요. 쓰레기 재활용은 원래 신랑몫이구요. 근데 주말부부라 ㅠ 주말동안엔 4점이요~ 설거지, 청소 다 해줘도 제가 하는만큼 성에 차진않아요^^;;'], ['장단점이 있겠죠~?'], ['저녁 먹고 설거지는 항상 신랑이 하고, 주말에는  애들 아침은 신랑이 차려주고, 청소기밀어주고 해요 ㅎㅎ'], ['좋으시겠네요 ~~^^'], ['별다섯개요ㅋㅋ\n하는건 화장실두개 청소 재활용 가끔 버려주는것..\n하지만 젤중한건 제가 겔름 부려도 집안일 안한다고 잔소리는 안해요\n그래서 별다섯개요ㅋㅋ'], ['좋으시겠어요~'], ['깔끔한성격에 해야되는일 미루는법 없고  꼼꼼하고 계획적인 신랑♡ 살림은 별다섯개요  .상위1프로의 깔끔남이지 싶어요'], ['도대체 어떤것을 하길래요?'], ['스스로 척척 알아서 다해요.전실 닦기.세탁기 먼지망 세척하기.에어컨 필터 세척하기 등등 디테일 한 부분까지도요.'], ['옹.. 좋으시겠네요. 상위 인정 합니다😁'], ['빨래 쓰레기 요리 하는데 3점이라니\n그럼 전 1.5점이여'], ['ㅎㅎ 저는 점수를 잘 모르겠어서요ㅜㅜㅜ 평균일거 같은 수 넣었어요'], ['댓글 읽는데 저희남편은 지극히 평범하네요ㅎ 평일은 늦게들어오니 제가 다하고 주말에 청소. 저녁 설거지정도 해주네요ㅋㅋ'], ['ㅎㅎ 평범이 최고죠!!'], ['저희 신랑은 4개..매일은 아니여도 빨래,청소,설거지,음쓰,일반쓰,화장실청소,요리,혼자장보기,재활용,아이들 목욕후 내보내면 닦이고 로션바르고 옷입히고(집에있을때 전담)등등 모든걸 다 도와주고있어요..쉬는날은 먼저 일어나 아이셋 아침까지~~ 4개 줄만하죠? ㅋ'], ['4개 줄만 하죠! 그런데 가끔이 아깝죠 ㅎㅎ']]
    
    6122
    집안일 중에서 가장 하기 싫은게 뭐에요? 저는... 설거지요...저녁 다 먹은지 한참 지났는데 설거지 하기 싫어서과일먹고..과자먹고..육포먹고..계속 먹고만 있어요...놔두면 냄새나고 더 하기 싫어지는거 아는데도 계속 미루고 있어요..식기세척기 있으면 바로바로 할려나요?ㅜㅠ맘님들은 가장 하기 싫은 집안일이 뭐에요~~
    
    [['저도 설거지요..  젤 시러용'], ['그죠~ 설거지가 젤 싫어요~😖'], ['설겆이요 식기세척기 사고 싶네요ㅋ'], ['저도 설거지 싫지만 식기세척기는 기본 1시간 40분 걸립니다.. 애벌로 한번씩 다 손대야 하고ㅠㅠ\n그리고 밥,국 그릇 등등 외에 냄비, 후라이팬, 도마 등등 큰거 나오면 어차피 설거지 해야 해서 요즘은 걍 건조대로 쓰고 있어요ㅋ\n진짜 괜찮은 식기세척기 나오면 저도 바꾸고 싶네요ㅠㅠ'], ['식기세척기 있다고 해결될것이 아니군요..ㅜㅠ'], ['그죠 .. 시간이 생각보다 길더라구요\n애벌하면 설거지걍하지뭐 이래될거같고 ㅋㅋㅋ'], ['그니깐요 베란다가 없어서 건조기는 잘쓴데 식구가 적어서 식기세척기는 망설여지는데 단점도 있네요ㅋ그냥 귀찮아도 설거지 해야겠네요ㅋㅋ'], ['맞아요 애벌 하느라 시간 보내고 그럼 간이 설거지 돌려야지 하고 나면 또 세제 덜 씻긴것 같아서 다시 꺼내면서 헹구고ㅋ\n그러다 보면 이게 뭐하는 짓인가.. 싶더라구요ㅋㅋ'], ['설거지요 ㅋㅋㅋㅋ\n식세기살까하면서도 빌트인?은 싫고 고민하고있어요🤣\n이게 왜필요한가를 ㅋㅋㅋㅋ\n정리도 누가좀해줌 좋겠네요🤣'], ['맞아요~ 제자리 찾아서 정리하는 것도 귀찮아요😅'], ['설거지가 싫어지네요 ㅎ 매번 신랑이 더 많이 해서 안사고 버티고는 있는데'], ['와우~ 좋은 신랑님 두셨네요^^'], ['빨래는 열심히 하는데...빨래개기가 젤 귀찮아요.'], ['전 티비보며 빨래는 개는데 서랍마다 넣는게..😁'], ['2222 저도 개는건 괜찮은데 자리찾아 서랍에 넣는게 귀찮아서 ㅠ'], ['장난감 정리요ㅠ'], ['아..장난감 정리..에너지 소모가 엄청나죠..\n그나마 아이가 좀 크니 자기방은 정리해서 나아졌어요^^'], ['다 갖다버리고 싶어요ㅠㅠ'], ['전 걸레질요 ㅋㅋ'], ['전 출산하고 무릎이 너무 아파서 걸레 다 버리고 밀대랑 물티슈만 사용해요~ 강추!!'], ['빨래개서 제자리 갖다놓는거요ㅎ\n궁디가 왤케 무거븐지...'], ['저두여..... 빨래 개는것도 싫은데 서랍에 넣는거까지는 더 귀찮고 하기싫어요 ㅋㅋㅋㅋ'], ['저두 걸레질 이 젤 싫어요.'], ['저도 설거지요ㅋㅋ 청소보다 설거지가 더 싫어요ㅠㅠ'], ['저는 빨래널고 개는거요'], ['전 청소요 특히 물걸레질...물걸레 청소기 있는데 유선이라 잘 안써지네요 ㅜ'], ['전 설겆이도 싫지만 ...걸레질요.. 청소기만 돌린적 많네요 ㅋㅋ'], ['전 밥하는거요ㅡㅡ 누가 매일 밥차려줬으면 좋겠어요ㅜㅜ'], ['22222ㅠㅠ 급식먹고싶어요....메뉴고민 지긋지긋해요ㅋㅋ'], ['전 빨래용..널고 개는게 진심 귀찮아요 ㅎㅎㅎ 정리정돈도 귀찮고..그냥 다 귀찮네영ㅡ.ㅡ ㅋㅋ'], ['밥하는게 제일 싫어요ㅜㅜ'], ['걸레질이요ㅜㅜ 정말 하기싫어요ㅋ'], ['식세기 저는 정말 잘쓰고있어요~^^\n청소기도 a9청소기물걸레키트 요번에 샀더니 신세계라 좋아요ㅋ\n난 밥하는게 젤싫어요ㅜㅠ'], ['전 분리수거랑 음쓰버리는거요ㅠ\n분리수거는 신랑이 해주는데 음쓰는 꺼리더라구요ㅋㅋ'], ['저는 걸레빠는거요'], ['식세기랑 물걸레청소기있어요...그래도 청소하기 애들어리니 청소하기힘들고 밥차리는게 젤 싫네요'], ['빨래 개는거요 ㅠㅠ'], ['빨래개기요..  빨래개주는 기계가 나옴 좋겠어요 너무 귀찮아요..'], ['저는. 청소요- 바닥청소는 하겟는데... 티비다이나 쇼파 이런곳닦는게 젤 시러요 ㅜㅜㅜㅜ'], ['빨래개기요'], ['걸레질하고 빨래개켜서 정리하는거요..'], ['전 설거지랑...빨래개기요,,.'], ['저도 설거지요  ㅠ'], ['저는 빨래개기진짜ㅋ세상 귀찮터라구요   쌓여있음 더더 개기싫더라구요ㅠ'], ['닦기요ㅠㅠ 티비다이, 바닥, 장난감들먼지 등등 먼지닦는게 젤귀찮아용ㅋㅋ'], ['저도 설거지요 ㅜ'], ['전 청소 ... ㅠㅠ 책장이나 뭐 이런데 먼지 쌓이는건 보이는데 닦기가 싫으네요..... 귀차니즘ㅋ'], ['전..화장실 청소가 제일 힘들고, 싫더라고요ㅠㅠ 남편이 도와줘도 힘들어요'], ['전 설거지랑 빨래개서 넣는거요 ㅎ'], ['빨래 서랍에 넣기요'], ['저도 설거지요ㅋ'], ['가장하기싫은거\n전부다 가장 하기 싫네요 ㅠㅠㅠㅠ'], ['전 빨래요 ㅜㅜ 건조기 못돌리는거 너는것도 귀찮고 개는건 더더더더더 ㅜㅜ 개서 넣는것도 귀찮아요 😭'], ['저두 설거지요ㅋㅋㅋㅋㅋㅋㅋㅋㅋ 왜이렇게 싫죠? 막상 하면 깨끗이 기를 쓰고 닦으면서.... 식세기 너무 사고싶어요ㅋㅋㅋㅋ'], ['식재료다듬기요ㅋㅋㅋ 요리자체는 안힘든데 재밌기도한데 그전에 식재료 다듬고 손질하는게 오래걸리고 쓰레기도 나오고 요리다해먹기도전에 개수대가 더러워져서 싫어요 뿌리채소같은건 흙털고 씻기랴 두껍한 무우나 딱딱한 당근같은거는 써는것만으로도 손목아프고... 저는 설거지보다 더 싫네요ㅎ'], ['전 방 닦는거요 ㅋㅋㅋㅋㅋ'], ['저는 청소..ㅜㅜ 바닥에 있는 물건&장난감 정리부터 청소기 돌리고 바닥 닦는거까지 다 싫어요 흐엉ㅜㅜ 설거지나 빨래널기&개기 다 괜찮은데.. 치우기를 비롯한 청소는 넘 싫어요ㅜㅜ'], ['설거지는 식세기가 해주고~(진짜 식기세척기 없었음 어찌 살았을지ㅠ)바닥 걸레질은 1회용(밀대로 쓱 밀고 버리면 끝) 빨래는 세탁기\n밥은 시켜먹기가능\n근데 스스로 다 해야 하는건 빨래개는거랑 화장실청소ㅡㅜ\n이 둘은 누가 좀 해줬으면 좋겠어요ㅎㅎ']]
    
    6196
    끝없는 집안일 페넬로페의 베짜기 : 쉴 새 없이 하는 데도 끝나지 않는 일을 가리킬 때 쓰인다.아이가 올해 초등 입학이라 3월부터 육아휴직했어요.. 아이랑 둘이 집콕중인데 거실이나 주방에 뭐 나와있는걸 싫어해서 매일 정리하느라 바쁘네요.. 슬슬 지치기도 하지만 아직은 포기가 안되는데..이제 아이를 위해 거실에 책장을 놓고 이것저것 좀 붙여놓을까봐요..
    
    [['넘나 깨끗'], ['저희집보다가...안구정화되었어요'], ['이야~모델하우스같아요ㅎ대박쓰'], ['저도 눈이 시원... 지금 책쌓기 놀이해서 집구석이 아주그냥 ㅋㅋ 둘째좀크면 저도 이리살고픕니다ㅜㅜ'], ['이런집에서 죽기전에는 살아볼수 있을까요?엄지척! 입니다.'], ['와~~~~이런 집이 있었네요~~^^\n넘 깨끗해요~^^'], ['우리집 거실로 나가기 싫어지네요..넘 깨끗해요~'], ['부러워요ㅠㅠ'], ['우와 청소할게없네요.깔끔합니다'], ['헉 집보니까 이사가고 싶네요 꼭대기층인가요? 앞이 탁트여 너무 좋네요'], ['탑층은 아닌데 20층 이상이고 앞이 트여 있어서.. 요즘 집에만 있음에도 살만해요..'], ['진짜 거실에만 있어도 기분 좋으시겠어요\n내년 19층짜리 아파트 19층으로 가는데 맨 앞동이라 지금 너무 기대되네요\n요즘 인테리어 사진만 보고 사네요...'], ['오~ 좋으시겠어요~ 이쁘게 인테리어하세요^^'], ['어우 깨끗!!!\n얼마나 쓸고 닦고 치우셨을지 알아서 존경스럽기 까지 합니당...'], ['제가 원하는건데요.넘 멋지세요.\n그릇들은 다들 어디있는건가요?\n그릇건조기가 서랍안에 있는건가요?~^^'], ['그릇은 식세기 돌린후에 씽크대에  정리해요'], ['아~^^ 댓글 감사드립니다'], ['아..반성하게되는 사진입니다ㅋㅋ\n저..퇴근후..집청소 할께요^^;'], ['이런게 가능한가요? 에이 잡지사진이죠?🤣'], ['청소하기도 편하겠어요  이미 너무 깨끗해서^^'], ['매트 없는게 젤부럽네요'], ['우와 대단하십니다. 바로바로안치우면 저리안될텐데.. 저도 나와있는거싫어서 다 넣는데 밑에열면 와르르 ㅋㅋㅋ ㅠㅠ비법좀전수해주세요'], ['주말에 씽크대 정리도 다하느라 몸 부서질뻔 ㅠㅠ 매일매일 치워요 ㅠㅠ'], ['넘나 깨끗하네요. 저희집이랑 비교되네요.ㅠㅠ'], ['모델하우스같아요~ 넘 깔끔하고 이뻐요'], ['모델하우스인가요,?\n저희집은 바닥에 빈공간을 찾아 다녀야하는 수준ㅜ'], ['이렇게 깔끔히 사시니 집안일이 끝이 없죠.. 좀 내려놓으면 편해요ㅎㅎ 부러워서 하는 말이예요~~'], ['곧 내려놓을듯요 ㅋㅋㅋㅋㅋ'], ['부럽네요 ㅋㅋ 깔끔한집 ㅋㅋㅋ 저희집은 치우면 5분이내 원상복귀 ㅠㅠ'], ['씽크대 아래 발매트 정보궁금해요~~'], ['한샘에서 샀어요~'], ['와우! 즤집은 거실이 책장이 많아서 아무리 저리해도 ㅠㅠ'], ['거실에 책장 있음 그렇죠... 저도 이제 그 세계로 가려합니다~^^'], ['급 청소해야겠다는 의지가 ㅋ하지만 현실은 쇼파에 붙어있네요'], ['전망이 굿 넘나 좋네요'], ['와..  모델 하우스같아요^^'], ['제 눈이 깨끗해지는 기분이에요~~👍🏻'], ['ㅜㅜ 모델하우스 아니네요 부엌이 제일부럽네요'], ['진짜 주방이 제일 부럽네요~~~~\n모델하우스 저리가라네요ㅋㅋ'], ['전 청소고자라..이렇게 깔끔하게 사시는분들 보면 넘. 부럽고 제 자신이 부끄러워져요..언제나 저렇게 깔끔하게 살아볼까 싶어서요(이번생은 포기에요ㅠㅠ)'], ['와우~~~진정 부럽습니다~^^'], ['나름 저 깔끔쟁이인데 애둘 집에있으니 말끔히 치워도 10분만에 도루묵이라ㅜ 안구정화하고 갑니다~ ㅎㅎ'], ['저도 뭐 나와있는거 싫어서 맨날 치우는데 치우고 나면 애들 남편 여기저기 꺼내고 쌓아놓고ㅠ'], ['와 부럽네요'], ['집안일은 해도해도 끝이 없고 누가 알아주지도 않고..그와중에 정말 깨끗한 집이네요'], ['아이를 위해서???? 그러지 마세요~ !!! \n아이 정서에 이렇게 단조로운것 아주 많이 좋을꺼에요..ㅎㅎㅎ \n너무 너무 부럽습니다.'], ['사람 사는 집 맞죠? 존경합니다!!!'], ['집이 어찌 이렇게 깨끗하죠'], ['와~ 누가 우리집 좀 이리 치워줬음 좋겠네요. 부러버요.'], ['조리도구들도 다 안에 넣으시는거예요? 쓸때마다 빼시는건지.. 진심 부럽네요 ㅋㅋ'], ['조리도구랑 도마는 나와있어요~'], ['와~~~~!!!!!!!!!!!!!\n세간살이 어떤식으로 정리했는지 넘궁금해서 찬장문들 열어보고 싶은건 저뿐인가요^^???\n진짜 한수배우고싶어요'], ['ㅋㅋㅋ 몇개 열어볼까요?'], [''], [''], [''], [''], ['역쉬 대단하시네요 정리잘하시는것도 재능인거같아요'], ['혹시 실례가 안된다면 냄비랑 후라이팬 뚜껑같은건 서랍에 어찌 정리하시나요~? 정리의 신이네요ㅠ'], ['저희 집이 수납이 많은 편이라 정리가 쉬운것도 있어요~ 후라이팬은 이렇게 후라이팬 정리대요~'], ['자주 쓰는 냄비는 인덕션 옆에 수납하고 큰 냄비들은 따로 수납해요~'], ['ㅎ 사진 감사합니다 저도 수납장은 많은데 시작이 잘못된거같아요 낼은 저길 싹 엎어야겠어요 ㅎ'], ['아..넘 예뻐요ㅜ 저희집 하루에 최소 3번 청소기 돌리는데요....아무도 안 믿을거에요ㅜ 이를 어째ㅜ'], ['심플하면서도이쁘네요그와중에  토스트기계가눈에  들어오네요'], ['진짜부럽네요'], ['헉 이게 가능한 현실집인가요 저희집이랑 너무 비교되요 ㅎㅎ'], ['네???? 우리집도 이번에 초딩 입학하는 백수있는데.....왜이리 다른가요?\n이건....엄마 잘못이었군요ㅠㅠㅠㅠ'], ['우와.........................부엌에 저렇게 뭐 안올려둘수 있나요.....부러워요'], ['집이 너무예뻐요 ~~'], ['남편분 행복하실듯요.'], ['와우~~! 너무 깔끔하고 좋네요!!'], ['옛날에 딘딘님? 그분집처럼 깨끗하네요 전모델하우스인줄요\n대단하세요 엄지척'], ['반성합니다 ㅠㅠ'], ['제가 꿈꾸는 주방이네요ㅜㅜ'], ['ㅋ 아이키우시는집 맞으시죠? 대박짱이세요'], ['딸 하나라 그나마 좀 괜찮고 아이한테도 매일 쓰고 제 자리에 놓으라고 해요.. 잘 안되긴 하지만요 ㅎ'], ['우린 언제 저럴 수 있을까..보기만해도 좋네여 ㅎㅎ'], ['집만 봐도 너무 깨끗해서 기분이 좋네요...'], ['부럽쓰요'], ['몇평이에여?'], ['40 요~'], ['헐 모델하우인줄요 대단대단'], ['모델하우스인가요? 뷰도 좋네요♡'], ['휴롬옆에 있는건 뭔가요?집이 넘 깔끔하네요. 댓글에 수납사진도 감탄하고 갑니다^^'], ['토스터기요~ 감사합니다^^'], ['힐링 하우스 같아요. 평화롭네요'], ['저런집이면 집콕해도 힘들지않겠어요~넘 부럽네요~♡♡'], ['와~~\n실화인가요?\n너무 깨끗해서 어지르기 미안할듯요...\n정리 비법을 좀 알려주세요~\n진심 부럽다요~~'], ['깨끗하니 참 좋네요'], ['구조좋고. 깔끔하고. 최고에요~~'], ['우와  저도 저렇게 살고싶네요'], ['우와집좋다아'], ['안구정화 하고갑니다'], ['난 이렇게 할수있을까?ㅋㅋ\n리스펙입니다 ~~'], ['놀라울 따름입니다~!!'], ['최고네요.님 사진보고 제주방 보니 한숨만 나옵니다...'], ['진짜진짜궁금한데 이렇게 깔끔하려면 아이가 아끼는 코딱지만한 장난감이나 원에서 가져온 교재교구는 싹다버려야 하는거겠죠?'], ['그래서 제일 정리 어려운게 아이방이예요~ 유치원에서 종이접은거, 재활용으로 만든거 이런것도 못버리게 해서요.. 아이방 수납장에 넣어놨다가 버려도 될만한건 조금씩 버려요 ㅋㅋ'], ['아래층 올라올까봐 초1,초3 딸둘맘은 아직도 매트를 깔고살아요..\n이런 거실 언제쯤...'], ['어맛!!모델하우스라고 말해주세요~~~']]
    
    6360
    제일 하기 싫은 집안일은 어떤거에요? 오늘은 집안일 몰아서 하는 날이에요.매트리스커버, 이불커버 세탁하고, 청소하고, 냉장고 정리하고, 반찬도 만들고 등등 오랜만에 몰아서 하고 있어요.근데 다 싫은데, 그 중에 개운 빨래를 옷장으로 넣는게 제일 싫으네요 ㅎㅎㅎ 다른 분들의 제일 싫은 집안일은 어떤건지 궁금해요!(니들이 알아서 옷장에 들어가줄래?)
    
    [['아 저도 ㅋㅋㅋㅋ\n개는 것까진 어찌어찌하는데 양말 속옷 흰티 바지 츄리닝 나눠서 넣는게 후~~~~'], ['그쳐그쳐...옷장에 나눠서 넣는게 참 귀찮아요 후~~~~'], ['빨래정리 받고 설거지에 한표 더~~'], ['전 설거지는 재밌어요 ㅎㅎ 싱크대가 조금만 높으면 더 좋을듯 ㅎㅎㅎ'], ['설거지 플러스 음쓰요!!'], ['아아아 음쓰는 세대주에게 맡깁시다'], ['빨래개기요ㅋ\n진짜 하기싫은데 건조기 산후론 강제개기해야 하네요ㅋㅋ'], ['쇼파에 쌓아둘때가 한두번이 아닙니다만 쿨럭쿨럭'], ['빨래개켜서 넣기가 싫어요ㅋ'], ['이심전심이네요 ㅎㅎㅎㅎ'], ['하나만 찾을수가 없는데요ㅜ'], ['아 그쳐 ㅠㅜ 만삭이라고, 산후조리한다고 몇개월동안 집안일을 안해서 참 좋았더랬는데 ㅠㅜ'], ['저도요 ㅋㅋㅋㅋ 거실바닥에 그대로 두고 며칠 재운적이 수도 없네요 ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋ 저는 양심껏 쇼파위에 살포시 그대로 ㅋㅋㅋㅋㅋㅋㅋ'], ['저도 빨래개서 서랍에 넣는 게 젤 싫어요~~ ㅋㅋ 빨리 건조기에 이어 알아서 개서넣기까지하는 기계가 나왔으면 좋겠어요 ㅎㅎ'], ['저도 그 얘기했어요 ㅋㅋ 남편이 사람쓰는게 빠를거래요@@'], ['저도 다된 짤래 개사 옷장 넣기요'], ['전 서랍장 위에 쌓아둡니다 ㅋㅋㅋ 입었던 옷이랑 막 섞이고 ;;;'], ['저도 그거요 ㅋㅋㅋ 개는건 좋은데... 넣기가...'], ['이상하게 그게 잘 안되요;;;;'], ['전 빨래널기까지는 괜찮은데 개기가.....ㅠㅜ'], ['티비보면서 개워봅니다 ㅎㅎ'], ['하나만고르기 힘들지만 개킨빨래 못넣고 거기서 골라입은적 많습니다 ㅋㅋ\n서랍공간이 넉넉하면 넣는일이 좀 수월하긴해요\n요즘 애들 빨래개키기 알바시켜요\n이젠 설거지가 젤싫으네요'], ['우리딸 얼른 키워서 알바시켜야겠네요 ㅎㅎㅎㅎ 개운상태로 쇼파위 또는 서랍장 위에 그대로 있는 날이 허다해요'], ['저도 옷접는거요\n아무리 예쁘게.접어도 잘안되어요'], ['예쁘게 접으려면 시간도 많이 투자해야하고, 특히 티비보면서 예쁘게 접기가 어렵죠 ㅎㄹ'], ['저두 ㅋㅋㅋ빨래개서 넣는게 젤싫어요'], ['남편이라도 알아서 넣어주면 좋을텐데, 저보다 더 싫어해요 @@'], ['이거 저만그런거 아니었네요ㅋ 저도 개서 넣는거 귀찮은데 사실 개는것도 귀찮아요...ㅋ 그래서 일부러 남편 퇴근한 뒤에 해요!ㅋ 같이 할 수밖에 없을 시간이요ㅋㅋ'], ['오오오! 전략적이네요 ㅎㅎㅎ'], ['전 요리가 싫어요 ㅠ'], ['요즘 반조리도 잘 나오잖아요! 저는 제가 먹고싶은것만 해먹어요 ㅎㅎ'], ['빨래개는건 하겠는데 넣는게 싫어요\n정리해서 넣어야하니ㅜㅜ'], ['그쵸..찌찌뽕! 그 마무리가 제일 싫어요'], ['전 속옷류 빼고는 다 걸어요.\n세탁후 옷걸이에 걸어 건조된 후 그대로 옷장이나 행거에 걸어요.\n그럼 여러가지가 편하더라구여'], ['좋은 방법이네요'], ['저도 옷장에 옷 넣는거요ㅋㅋㅋㅋㅋ 개는거까지 어떻게든 하겠는데... 그거 넣는게 왜케 싫은지ㅋㅋ'], ['그러니까요! 그게 뭐라고 쇼파에 쌓아두는지 ㅋㅋㅋㅋ'], ['전 반찬하고 난 설거지요... 냄비, 후라이팬 닦는거 넘나 귀찮.....'], ['전 찌든얼룩이 지워지면 묘한 성취감이 들더라구요 ㅎㅎㅎ 아 근데 귀찮기는 해요 ㅎ'], ['전...청소요~ 정리하고 청소하는게 젤 싫어서 요즘 신랑이 주로해요~'], ['대신해주는 사람이 있어서 얼마나 다행입니까 :)'], ['설거지받고 화장실 청소여~~'], ['화장실청소 하고 나면 금방 다시 물얼룩이 생겨서 좌절 ㅠ'], ['저도요! 전 진짜 설거지도 좋아하고 빨래 널고 개는 것도 좋아합니다.. 근데 넣는 게 너무너무느어어어무 귀찮아요 흑흑'], ['저랑 완전 찌찌뽕!'], ['화장실 청소랑 빨래 개고 넣는거 귀찮아요 ㅋ ㅋ ㅋ'], ['갑자기 엄마가 보고싶어요!!'], ['다림질이 제일 하기 싫어요'], ['다림질 그게 뭔가요? 다림질 안한지 진짜 오래됐어요 ㅋㅋㅋ'], ['악 저두요 개는것부터 싫은데 구지 해야되면 개는것까지만해요ㅋㅋㄱ'], ['옷들이 알아서 옷장으로 차곡차곡 들어가면 얼마나 좋을까요? ㅎㅎㅎ'], ['빨래 개기를 제일 싫어해요\n그래서 전 중딩아들 초딩딸 오라고 해서\n각자 옷 개어서 갖고가라구 시켜요'], ['다컸네요!!!'], ['전 화장실 청소요^^'], ['화장실 청소는 하고나도 금방 얼룩이 생겨서 참 하기 싫죠@@'], ['맞아여 ㅋㅋㅋㅋ 제일하기싫으네여 ㅋㅋ'], ['요리 그자체요 ㅋㅋㅋㅋ 꼭 먹어야 하나 의문입니다 ㅋㅋㅋ'], ['오늘 미역국을 끓였는데, 세대주님이 치킨을 시키더라구요 ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['저도 빨래개기랑 제자리 넣는게 시르네요'], ['빨래가 제일 싫어요. 손이너무많이가요.. 넣고 널고 개고.ㅋㅋ 힘들어요']]
    
    6363
    (수다)아오씨..ㅠ 😱 맘님들은 가장 하기시른 집안일이 뭔가요? 저녁에 감자랑 양파 때려넣고 카레해서아이랑 남편 먹이고 오늘은 남편한테 아들램 맡겼어요 재워달라고요...사실은... 재워달라고 쓰고 안방에 강제로 가두다 라고 읽는다...😑지금 혼자 집안일 하는중요...ㅋ이틀동안 집안일을 못햇어요..이래저래...🤔 응? 나 왜 바빳니?? ;;;저는요...빨래 정리하는게 그케 싫어요...ㅠ왜 때문일까요ㅋㅋㅋㅋㄲㅋ건조기 쓰기 전부터 말이죠...세탁기를 돌리기는 하는데널기도 싫고 걷기도 싫고게기는 더 싫고 겐거 넣기는 더더더 싫고그래서 건조대에 널린채로 걷어입고..그러다 건조기 들이니 건조기에서 빼기가 싫고ㅜ건조기 먼지통 털기가 싫고건조기에서 한참만에 꺼내노면이게 참 사람이 입는건가 싶고...꺼내서 던져놓고 꺼내서 던져놓고한 이틀 쌓이다보면 뭐 이게 집인지 쓰레기장인지분간도 안가고...ㅠ맘님들은 어떤게 젤 하기 싫어요?
    
    [['전 설거지가 제일 싫어요ㅜ'], ['전 둘다 싫어요...ㅜ.ㅜ 왜이럴까요'], ['저는 설거지는 좋아요 할때 뽀독뽀독 소리 날때 희열을 느껴요😌 나 변탠가봐~~~~'], ['ㅋㅋㅋㄱ👍 그게 정상입니다'], ['사람다 똑같은가봐요~~~ㅋㅋㅋㅋ저도 건조기없을땐 빨래 널기가 그렇게 하기 싫었는데 진짜 광땡님이랑 똑같은 심정입니다~~~ 제 속마음 들킨줄 알고 놀랬어요ㅋㅋㅋㅋㅋ'], ['맞죠????????  완전 미치겠어요ㅋㅋㅋ 건조기 다돌아가고 멜로디 나오면 짜증이 훅ㅋㅋㅋㅋ'], ['전 다 싫어요 ^^;;;;;;;'], ['🤣🤣🤣🤣🤣🤣👏👏👏👏👏'], ['저두설거지요  식세기사고싶어요'], ['전 설거지는 뜨건물로 뽀독뽀독 씩는거 좋아해서ㅋㅋㅋ설거지 세제 이것 저것 사는거도 완전 조아욬ㅋㅋㅋ 그래서 식세기도 사기시름요ㅋ'], ['빨래갠거 제자리에 가져다두는거요ㅜ'], ['와 대박 시러요그거 그래서 한숨만 쉬고 있어요 지금도ㅠ'], ['글읽다가 빵터졋습니다~~~제 이야기인줄ㅎㅎ빨래 개비는거랑 ..갠거 넣기는 더 싫어요ㅋㅋㅋㅋㅋㅋ'], ['미치겠네요 어떨땐 잘 넣다가 마지막엔 한서랍에 몽땅 때려 쑤셔넣기🤣🤣🤣🤣🤣'], ['한서랍이라도 때려 넣음 다행이네요ㅋ\n거실에 개빈채로 몇날며칠잇다가 속옷도 \n겉옷도 그냥 거기서 입기ㅋㅋㅋㅋㅋㅋ'], ['제글인줄알았네요 저도 옷무덤이 거실 안방 곳곳에 있네요 ㅎㅎ 어여하시구 편안한밤되세요'], ['ㅋㅋㅋㅋㅋ게는건 다햇어요 이제 쑤셔넣기만 .....하...;;  맘님도 굿밤용🤗'], ['우왁ㅜ저너무 공감되용!!!전 빨래너무 ㅜ싫으네요 빨래는 세제만넣음 세탁기가다해주니깐ㅜㅜ그렇다쳐도 근데 건조기넣고 꺼내고 개키는게 너어~~무싫어요ㅜㅜ'], ['진짜맞죠 특히 전 먼지통....ㅠ 눈물나게 시러요'], ['지금 건조기안에 빨래 다됐는데 이러고 누워있네요ㅡㅡ아마도 쭈글쭈글한채 내일꺼낼것같은 느낌적인느낌이네요ㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋ먼지통도 그때그때 비워야 한다는게 안습...썹쎈타 가서 여분을 몇개사다 돌려쓰까 생각중이에요'], ['집안일 다 싫어요~~~~ㅠㅠㅠㅠ'], ['인정....ㅋ'], ['완전 공감되네요ㅋㅋㅋ 지금 저희집에도 저리 앃여있네요'], ['전 이제 다 했어요~~~~~😁'], ['저두요 아~까 끝난 건조기 인제 꺼내고 개비는중에 뜨끔하네요 개비서 나왔음 좋겠다고 노래를 부르네용 ㅎㅎ'], ['🙏헐 진짜 그럼 대박이네요 ㅋㅋ'], ['저도 빨래 개는거 너무 싫어요 ㅠㅠㅠㅠㅠ 개고 정리하능거 정말 최악이에요 😭😭😭'], ['그져...미춰버립니다ㅠㅜ😱'], ['설! 거! 지! ㅠㅠ 제일 싫어요 손이 느려서 ㅠㅠ'], ['....전 설거지는 제가밥을 배 째지게 먹지 않은 이상 뜨건물로 뽀드득 씻어 엎어노면 기분 좋드라구요ㅋ'], ['집안일이 시러요~~^^ㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋ정답입니당🤗'], ['저도 공감해요~~~\n건조기 쓰니까 이제 개비는게 귀찮아용 ㅋㅋㅋ'], ['원래 서면 앉고싶고 앉으면 눕고싶죸ㅋㅋㅋ인간이란ㅋㅋㅋ'], ['설거지....헬이죠ㅜㅜㅜ'], ['글쿤요 설거지랑 빨래정리가 젤 하기시른가바여...'], ['전 음식하는거랑 설거지가 젤 싫어요~'], ['아 음식도요?  보통 일을 벌이는건 안 싫어 하잖아요..뒷처리들을 싫어하긴 하는데ㅎ 음식 만드는건 재미잇지 않나요?  아냐아냐 것두 재료손질하고...으;;  한번두번이지 맞아요 맞아ㅋㅋㅋㅋ  저도 싫어하네요 생각해보니ㅋㅋㅋㄱ 저도 싫어서 맨날 남편 시키면섴ㅋㅋ 재밋지 않냐니 미쳣네미쳨ㅋㅋㅋㅋㄱ'], ['전 정리가 제일 싫어요ㅜㅜ'], ['음...정리....이건...저도 굉장히 싫어하는거 중에 하난데요..이게 미루기 시작함 나중엔 정말 내가 감당할 수 없는게 되어있더라고요..그래서 저는 되도록이면 미루지않고 되도록이면 쌓지않고 되도록이면 버려버립니다ㅋㅋㅋ'], ['그래서 하긴하는데ㅜㅜ뭔가 어수선하니ㅜㅜ좀 배워야될까 고민중입니다ㅎ'], ['그 중리수납 전문가들요 강습도 하시더라고요..저도 한번 배워볼까 해요 나중에 아이 얼집가면요'], ['저도 배워봐야겠어요ㅜㅜ정리잘하고 인테리어 잘하는거보면너무너무 부러워요~~'], ['저는집안일너무좋아해요~ 365일 밀고닦고 수시로 정리하고 반찬도절대안사먹고 매일3~4가지 요리하고 치우고 주4회는 술마셔서 밤마다 안주 만들고 아무리 취해도 습관적으로 다치우고자요. 빨래도 매일돌리고 집안일하느라 항상바빠요ㅋㅋ 신랑한테는 1도안시켜요 제가하는게 편하고 루틴?이 집혀있으니 그냥척척해지더라구요 집안일하는거넘재밌어요~ 지금 결혼 십년쯤됐는데 입주첫날과 오늘이 똑같답니다..ㅋㅋ 요새는 세달째 가정보육중이라 두배로 바빠요ㅠㅡ'], ['저도 님 과 같네요 ㅋ'], ['여보😍'], ['바닥 걸레질요.. 제가 직접미는것도아닌데도 왜이리 하기시를까여'], ['걍...문명에 도움을 받습니다ㅋ'], ['전 빨래개는거 제일 싫어요....ㅠㅠ'], ['저도 이게 젤시러요 밖ㅇ니ㅣ 비오네요 비오기전에 전 해치웠어요  올레🖖'], ['건조기돌려도 결국은 개는건 내손으로 해야되니...거기다 서랍찾아넣는건 더 귀찮;;;이 지긋지긋한 빨래ㅜ'], ['습관을 들이라캣제!!!\n그리고 바구니 사다가 착착 개비 담아놓으라캣제!!\n두달만 억지로 해봐 습관된다\n어차피 평생 아니 1-20년은 해야 될 일이다 ㅡㅡㅋ'], ['착착담아노면 저노무시키가 잘도 놔두겟다ㅋㅋㅋ그리고 좀만 더 기다리면  빨래 게서 나오는 기계나올지도 모른다 좀만 더 기다리보께🤣🤣'], ['꾸덕찐~~~득한 쪼꼬케이쿠가 먹고싶다\n아아랑'], ['온나'], ['사도🤣🤣🤣🤣'], ['심한욕심한욕\n햄 입원중이시다 새꺄'], ['믄데 내가 자학개그같은거 하지마랫제!!!!'], ['그냥 집안일 할게요. 애좀 누가 봐줘요. 애보는게 제일 싫...'], ['아하..........이것도 전..얼마전까지 우울증 올 정도로 힘들엇는데 지금은 걍 숙명으로 받아들였👉👈 ㅋㅋㅋ사실 그래도 힘들져...아이가 어리신가봐요ㅠ 힘내세요 오르막길 내리막길이 있는거 같아용ㅎ 곧 내리막길이 나올거에요🤗'], ['전청소가젤싫어요 ㅠ해도해도끝이없고티도안나고뒤돌아서면그대로인이유가 뭐죠?'], ['집안일은 몰아서 한번에 육퇴후에 합니다 안그럼 저도 스트레스받고 아이를 혼내게 되더라고요ㅋ'], ['비염알러지 있는애들땜에 건조기 샀는데.. 빨래가 바닥에 산처럼 쌓여 더이상 못봐줄정도 되야 정리해요.. 온 먼지 다 쌓아놓고 말이죠ㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ저도 아기도 먼지 알러지 있어요  그래서 이불도 세사꺼쓰고 청소기도 바꾸고 ㅋㅋ 건조기돜ㅋㅋㅋ 근데 저도그래욬ㅋㅋ쌓인빨래 개면서 에취~하지만 그래도 확실히 안쓸때보단 덜해요..그래서 계속 쌓는건가....🤔'], ['건조기쓰니 빨래가 끊임없이 계속 나오는 단점도 ㅋㅋㅋ 그래도 애들 좀  커서 지들옷 정리하는건 각자 시키니 살만하네요 ㅎㅎ'], ['그니깐옄ㅋㅋ건조기없을땐 빨래돌려 널고나면 먼가 1차 할일 클리어 되고 텀이 생기는데 건조기 들이고나니 틈이없엌ㅋㅋㅋㅋㅋㅋㅋ'], ['저는 설거지.빨래널기&개기&넣기.음쓰.재활용버리기.걸레질하기. 폴더매트 청소가 제일 싫어요    \n적고보니 집안일전부가 싫어요']]
    
    6394
    왜 집안일은 끝이 없을까요 ㅋㅋㅋ (6주 기다린 쇼파 사진 투척) ㅎㅎ 기분좋은 금요일입니당이상하게 하루종일 집안일을 하는데 왜 끝이 없는 것 같죠 😅어제 새 냄비 사와서 세척하고 끓이고 연마제제거하고 ㅋㅋㅋㅋ 아침에 간단히 먹은거 설거지하고 :) 손님용 수저 젓가락 다 세척하고 ㅋㅋㅋㅋ미루던 서재방 정리하다가 청소기 돌리고6주 기다린 쇼파가 오늘 드디어 왔어요!!! 예쁘지만 또 닦아야하고 ㅋㅋㅋㅋㅋ1주일에 한번하는 화장실 청소도 하고 씻고 다시 머리카락 정리하고..이제 물걸레랑 남은 정리해야하네용 😀😃😂😅 어제 빨래 2번하고 쓰레기 다 버렸는데도 ㅋㅋㅋㅋ 왜 끝이없죠 ? 😂 진짜 집안일 하는분들은 다 대단하신거 같아용 6주 기다려 받은 쇼파!!! 사진 투척해용저희집 침대와 가격이 같은ㅋㅋ 막상 받으니 정말 너무 예뻐요 ㅠㅠㅠㅠ 그래서 다 좋네요 💛💛 신혼집 컨셉은 제 맘대로 핑쿠입니다 😆 정리하면서 찍은 집안 용품과 사진 몇개 더 투척하고 가요 ㅎㅎㅎ모두 좋은 주말 보내세용
    
    [['맞아요 집안일 해도 티 잘 안나고 안하면 바로나지만요..ㅋㅋ'], ['ㅋㅋㅋㅋ진짜 공감이요 하면 티 안나는데 안하면 티나요ㅠㅠㅋㅋㅋㅋ'], ['인테리어 깔끔하게 잘하셨네용ㅎㅎ 집안일은 끝이없죵,,'], ['진짜 오늘 끝이 없이 했는데 아직 쓰레기까지 또 버려야하네용 ^^..ㅋㅋㅋ'], ['원래 집안일은 해도해도 끝이 없다고 하잖아요~~ㅎㅎ'], ['진짜 요즘 너무너무 공감해요~~ㅎㅎ'], ['원래 집안일은 해도해도 끝도없다고 하드라구요 ㅜㅜ이해가되요 ㅜㅜ'], ['인테리어 깔끔해서 좋으네요 ㅎㅎ집안일은 평생이죠 ㅠㅠ'], ['ㅎㅎ 깔끔하게 해놨는데 먼지가 계속 보이는 현실이네욬ㅋㅋㅋ'], ['진짜 집안일은 끝이 없어여ㅠㅎ더구다나 전 청소한 빛이 안나여ㅠㅋ열심히한건데ㅎ쇼파 예뻐여>_<'], ['ㅠㅠ저두요 열심히 했는데 여기 바닥이랑 벽지가 좀 오래되서 .. 흑 아쉬워용 감사합니당 >_<'], ['오 뭔가 포근한 느낌드네요^^'], ['ㅎㅎ포근한 집에서 집순이하려구요'], ['저희집이랑 밥통도 같고 쇼파도 비슷해요❤️'], ['밥통 정말 너무 예뻐요~~~취향이 비슷하신가봐요><'], ['집 너무 깔끔하고 넓은것같아요~ 저도 느껴요 집안일은 끝도없다는걸 ㅠㅠ'], ['ㅋㅋㅋ어젠 하루종일 일하고 오늘은 쉬는데 진짜 끝이 없네요ㅠㅠ 25평인데 둘이 살기 너무 좋은거 같아요'], ['진짜 돌아서면 또 생기고 또 생기더라구요.. 난 분명히 1시간 전에 청소기를 돌렸는데?? 하고 제 눈을 의심하고 ㅋㅋ 방금 다 설거지 한 거 같은데 조금 지나면 컵에 뭐에 또 한가득 ㅠㅠ ㅋㅋㅋ 그나저나 6주나 기다려서 받으면 더 좋고 기쁘셨겠어요!! 주말내내 소파에서 뒹굴뒹굴 기다린만큼 마음껏 누리세요! >_< (소파에서 보내는 일상이 최고 행복한 1인 .... ㅎㅎ)'], ['진짜요 !! 특히 머리카락이랑 먼지 ㅋㅋㅋㅋㅋ 저는 심지어 식기세척기 돌리는데도 무슨 설거지가 이렇게 많은지ㅜㅜㅋㅋㅋㅋ 네 진짜 쇼파에서 넷플릭스 미친듯이 보려구요 >_<'], ['맞아요 집안일 너무나 힘들어요..누구한테 칭찬도 제대로 못받는거 같아요..'], ['공감이요ㅠㅠ 칭찬이 아니라 너무 당연한거라 생각하니 하핳 쉽지않죠 그래도 힘내요~~!! ㅎㅎ'], ['집안일이 진짜 끝이없나봐요ㅠㅠ 고생하셨어요ㅠㅠ\n집 분위기가 깔끔해욤!!'], ['ㅋㅋㅋ네.. 하고 2차전으로 물걸레질이랑 이불도 한번 털고 먼지들 다 닦고 혼자 열심히 하구 있네용ㅎㅎ'], ['저도 집안일 하려면 정말 큰 맘 먹고 시작해야해요ㅠㅠ 해도해도 해야할 일들이 계속 눈에 보이더라구요ㅠㅠ'], ['ㅋㅋㅋ진짜 매일 큰 맘 먹는것 같아요 ㅠㅠ 근데 저만 눈에 보이나봐욬ㅋㅋ어질러져 있으면 거슬려요ㅠㅠ'], ['진짜 집안일 끝이 없다는 말 너무너무 공감이예요!ㅠㅠㅋㅋㅋㅋ 톤 잘 맞추신듯요 예뻐요~'], ['ㅋㅋㅋㅋ앞으로 일상이겠죠? 감사합니당ㅎㅎ'], ['ㅎㅎ인정합니다. 해도해도 끝이 없어요. 티도 안나고~'], ['티도 안나고~ 근데 해야는 하고 ㅎㅎㅎ 참 어렵네용'], ['집안일은 하면 할수록 할게 늘어요ㅠㅠㅠㅠ'], ['이거 하면 저게 있고 이거 하면 저게 있고..ㅋㅋㅋㅋ무한반복이에요'], ['헉 집 너무부러워요 햇살 엄청 잘드네요'], ['ㅎㅎ햇살이 잘 들어서 넘 다행이에요...요즘 집에 있는 시간이 좋더라구요'], ['그릇부터 사신 가전제품 모두 다 예뻐요'], ['감사합니다 ㅎㅎㅎ'], ['집안일은 정말 해도해도 끝이 없고 티도 안나요ㅠㅠ'], ['티도 안나지만...열심히 해야하는 현실 ㅠㅠ 힘내요 저희 !!'], ['사신 소형가전들 너무 귀욤귀욤하네요 ㅎㅎ 소파 진짜 편해보여요'], ['ㅎ_ㅎ 진짜 편해요 3번이나 가서 고민하다가 샀는데 너무 맘에 들어용'], ['오호 너무 깔끔하고 예쁘네요~ 진짜 청소하다 보면 하루가 금방 갈 것 같아용~ㅎㅎ'], ['오늘도 그렇게 하루를 보냈습니다 ㅎㅎㅎ 남편 출근시켜주고 하루종일 집안일 했는데 남편 델러갈 시간이네요'], ['인테리어가 깔끔하니 예쁘네요 ㅋㅋㅋ'], ['저도그렇게생각해요 끝도없고 티도 안나는 집안일'], ['진짜 주부도 직업이라는게 정말 와닿아요 ㅠㅠ일이 끝이없어요'], ['진짜 직업 맞는것같아요... 끊임없이 할일ㅋㅋㅋㅋ'], ['저두요.. 살때는 좋은데 배송오고 설치되는거 까지는 좋은데 닦고 관리하고 유지하는게 왤케 귀찮은지 모르겠어요 ㅎㅎ'], ['진짜 공감이요 !! 보는건 다 좋은데 관리 유지가 젤 힘들어욬ㅋㅋㅋ'], ['집안일은 진짜 해도해도 끝이 없어요 ㅋㅋ 그래서 하루 할당량 정해놓고 더이상 손대지 않고 있어요 ㅋㅋㅋㅋ'], ['젤 좋은 방법인거 같아요ㅠㅠ안그러면 정말 끝없이 하는거같아요'], ['집이 깔끔하네요!! 음식두 잘해서 드시는거같애요 이쁘게 잘사시는거 같아용ㅋㅋ'], ['ㅎㅎㅎ 아직 첨이라 예쁘게 잘 하고 싶은 맘이라 그런거 같아요 제발 오래가길ㅋㅋㅋ'], ['그쵸! 집안일 정말 끝이 없어요!! 적당히 안보고 살려구요..ㅋㅋㅋ 쇼파 따뜻한 느낌들고 예쁘네요!'], ['ㅋㅋㅋ 적당히 안보고 살아요 저희... 감사합니당'], ['집안일은 정말 해도해도 끝이없어요..ㅋㅋ'], ['오오 쇼파가 아주 고급스러웁니다 너무 예뻐요'], ['3번이나 가서 보고 산 보람이 있어요ㅠㅠ'], ['집안일은 뒤돌아서면 생기는것같아요ㅎㅎ 저도 쇼파기다리는중인데 쇼파 집분위기와 넘잘어울리네요ㅎㅎㅎ6주기다린보람있으시겠어요ㅎㅎ'], ['네 진짜 기다리면서 힘들었는데 오니깐 너무너무 좋아요 ㅎㅎㅎ 이제 뒤 안돌아 봐야겠어요'], ['쇼파 정말 예뻐요~~ 집이 예쁘고 깔끔해서 집안일 할 맛 나시겠어요~'], ['감사합니다 ㅎㅎ오늘은 기쁘게 집안일 했어요'], ['소파 기다리실만했네요ㅎㅎ 집안일 넘 많아요ㅠ'], ['ㅎㅎ네 기다린만큼 예쁘네용ㅎ 집안일 화이팅이에요👏🏻'], ['진짜 집에 있으면 집안일하느라 시간 다 가는 것 같아용ㅋㅋㅋㅋ'], ['오늘 하루는 집안일로 시작해 끝났네요 ㅋㅋㅋ'], ['집안일 해도해도 안끝나는거 공감해요 ㅋㅋ'], ['ㅋㅋㅋㅋㅋ그냥 안봐야 끝나는거 같아요'], ['와 상차림 진짜 왜 일케 이뻐요'], ['그릇이 다했어요 👍🏻'], ['청소하고 돌아서면 또 청소거리가 생겨나는 것 같아요ㅋㅋ'], ['ㅋㅋㅋ맞아요 진짜 오늘 하루종일 집안일만 했네용'], ['와,,, 쇼파 너무 예뻐요 살작 저층인가요??? 창문에 나무가 보이니까 푸릇푸릇하니 너무 에쁘네요 !'], ['네 어쩔 수 없이 4층으로 했는데 나무가 보여서 예쁜거 같아요 :)'], ['오히려 나무가 보여서 더 예쁜거같아요 ^^'], ['와 집 너무예뻐요 소품이 하나하나 다 예쁘네요'], ['>_< 감사합니다'], ['집 소품도 넘나이쁘네요 ㅎㅎㅎ'], ['ㅎㅎ감사합니당'], ['쇼파는 원래 오래걸리나봐요ㅠㅠ 저는 두달기다리래요 흑 ㅠㅠㅋㅋㅋ'], ['헉 두달이나.... ㅜㅜ 진짜 넘 오래걸려요'], ['쇼파 너무 예뻐요 ~^^\n어디 브랜드일까요?'], ['에싸라고 자코모 쇼파 패브릭 라인이에요 ㅎㅎ'], ['냄비 궁금해요~~좋은가요?? 넘 이뻐요 ㅋㅋ'], ['네 냄비 좋아요! 찜기용으로 가볍게 쓰려고 샀는데 엄청 잘써요 ㅋㅋ'], ['ㅎㅎ 집에 그릇, 주방용품들이 너무이뻐요']]
    
    6433
    제일 하기 싫은 집안일은? 음...탁탁 털어 빨래 너는것까진 기분 좋은데다 마른 빨래....개서 치우는게 제일 하기 싫어요ㅠ거실에 마른빨래가 쌓여있습니다ㅡㅠ 제일 하기 싫은 집안일, 어떤거세요?
    
    [['설거지요....'], ['아침에 일찍일어나 아침밥 차리는거요.ㅋㅋ 잠이 많은지라..'], ['요즘은 다~~ 하기 싫어요~~ㅜㅜ'], ['222'], ['333333'], ['요리, 설거지요:-)'], ['청소요..'], ['밥하는 게 제일 힘들어요^^;'], ['전 싹다~~😂'], ['미챠🤣🤣🤣🤣🤣🤣🤣\n깊이 공감합니다^^;;;'], ['걸레로 닦는거요ㅋㅋㅋ'], ['밥..'], ['설겆이요.'], ['밥! 그리고 먹고 난 후 설거지요ㅠㅠㅠ'], ['밥차리는거 ,빨래개는거요...하기싫어요'], ['우열을 가리기 힘들게 박빙이지만그래도 1번은 쌓여있는 설거지요 ㅜㅜ'], ['설거지가 젤 싫었는데 세척기 들이고 나서는 빨래 개는거랑 애들 늘어놓은거 치우기요 ㅋㅋ'], ['전 빨래 개켜 제자리에 넣는 건 좋은데 빨래 탁탁 털어넣을 때 습랭감이 싫어요. 항상 남편이 해요. 본인이 하는 거 좋아해요.'], ['욕실청소요... ㅜㅜ'], ['청소하는거 좋아하는데 설거지는 왜이리 하기싫을까요^^;'], ['빨래널고개는거요.ㅋ'], ['저도요ㅜ'], ['밥하기가젤루싫어요ㅜㅜ'], ['화장실청소요..'], ['음식물버리는거요'], ['갠 빨래 정리해서 넣는게 젤 귀찮아요..ㅠㅠ'], ['다하기 싫지만..반찬 만들고 하는게 힘들어요..ㅠ'], ['전 설거지요 ㅠ'], ['전 설거지옥이요ㅠㅠ.. 식세기 들여야겠어요 코로나로 설거지만 하루에 몇번을 하는지 부엌떼기네요'], ['음쓰 버리기'], ['설거지랑 빨래너는거요ㅜㅜㅜ'], ['화장실청소요 ㅠ'], ['청소기돌리는거요 걸레질이요 엉엉 ㅜㅜ 먼지는왜매일닦아도다시쌓일까요'], ['하고싶은 집안일이 있냐고 먼저 물어봐주실래요?? ㅜ 하 호텔에서 살면서 룸서비스로 매일청소해줬음좋겠어요'], ['저두 빨래개는거.어떤날은 빨래더미에서 찾아입기도 해요.ㅋㅋ'], ['저는 다림질요ㅎ'], ['저도 빨래개는거요~~~~'], ['집안일이야 다 하기싫죠\n요즘은 제 한몸 씻는것도 너무 귀찮아요'], ['밥상치우기랑 빨래개기ㅎ'], ['걸레질이요..'], ['화장실청소, 운동화세탁요ㅠ'], ['저두 설거지하는거요 ㅠㅠ'], ['걸레청소요ㅜㅠ 다른건 100번도 다할수있는데 물걸레청소는 빨고 가구들어서 틈틈히 닦고 다시 빨고 널고 으어 너무싫어용ㅜㅜ'], ['걸레질하고 빨래정리요~~읔~~'], ['운동화 빨기요  ㅜ.ㅜ'], ['화장실 청소가 젤로 싫어요'], ['창틀 청소요 ㅠ'], ['청소기 돌리는거요;;;'], ['저도 화장실청소요ㅠㅠ'], ['모든거다하기싫어요ㅎㅎ'], ['방청소가 넘나 싫어요..ㅜㅜ'], ['정리요. 정리를 못하겠다는 ㅋㅋㅋ'], ['요리요'], ['완전 동감요!!\n빨래 개는것도 싫은데요 \n젤 귀찮은건.. 갠 빨래 제자리에 넣는거요... \n큰애 서랍장 .작은애 서랍장도 달라서 무릎 연신 폈다 접었다 너무 싫어요 ㅠ ㅋㅋㅋㅋ'], ['욕실청소요..특히 욕조 닦기,바닥닦기ㅠ'], ['저녁밥 차리는거랑 설거지요'], ['아침이 와서 눈을 떠야하는 일이.... 하루시작이 싫어요 푹 쉬고싶어요ㅋㅋ'], ['빨래정리요'], ['빨래개는거 제일싫어요'], ['빨래개기요'], ['전 제습기 물비우고 가습기 청소하고 물채우는거요 😂'], ['전 개는거 까진 괜찮은데 제자리에 넣는걸 못해..개켜진 빨래,안 개켜진 빨래가거실에..방에 쌓여있어요..ㅎㅎ설겆이도 싫고.. ㅎㅎ다들 그러시는구나..저만 그런게 아님에 조금은 안도하네요'], ['물걸레질. 운동화빨기. 손빨래. 창틀닦기. 현관.베란다바닥청소. 먼지닦기. 욕실..\n요즘은 밥차리기도 싫고..\n다 싫네요.. 사표내고 싶다!'], ['전 밥하는거요\n반찬걱정 너무 짜증나용 매일 머 해먹을까 고민해야하구요'], ['설거지옥~~~!\n저녁에 식사후 깨끗이 정리해놓고 자고 일어나면 아침에 씽크대가 넘치도륵 쌓여있어요..\n밤새 무슨일이 자꾸 자꾸 일어나네요 ㅋㅋㅋ ㅠ.ㅠ'], ['전 대청소여 그래서 애낳고 부터는 남푠이 한답니당'], ['빨래 개서 정리하는게 너무 하기 싫어요ㅠㅠ'], ['빨래개기'], ['집안일은다싫어요 ㅎㅎ'], ['전  청소만 아니면 괜찮을것 같아요ㅜ'], ['빨래개기 빨래개서 정리하기 \n전 빨래를 건조기에서 꺼내입어요.ㅋㅋㅋㅋㅋ'], ['설거지..화장실청소요..ㅡㅡ'], ['설거지요~~ 다 먹고 치우고 티비보고 놀고 있는 식구들보면서 나혼자 설거지 하면 단전에서부터 화가 치밀어 오르는듯 해요 ㅋㅋ']]
    
    6437
    남편이 집안일 어디까지 도와주나요? 저희는 일하느라 힘들다고가끔 가끔 아주 가끔 걸레닦는거랑 쓰레기버리는거에요설거지가 제일힘들고 손이많이가서인지요즘 주부습진걸리고 설거지하기가 너무 싫으네요ㅡ
    
    [['분리수거빨래널기음쓰버리기아이들목욕둘째책읽어주기주말에 청소기밀기설거지하는동안 아이들이랑 몸으로 놀아주기'], ['저희남편도 요정도 해주는데~ 저는 맞벌이가아니라 주부니까 같이하는거라기보다 도와준다는말이 맞는것같아요. 아직 아이가 어려 설거지가 자꾸 쌓여서 식기세척기들였는데 정말 추천해요 설거지가지고 스트레스안받아도되고 스팀으로 소독까지 되요!'], ['지금은 주말부부라서 쉬는날에 설거지랑 청소기랑 쓰레기랑분리수거 애들목욕이랑다 해줘요ㅜㅜ 같이살때도  설거지랑 청소기는 자주도와줬어여ㅎㅎ 너무고마워요ㅜㅜ'], ['설거지는 말하면해주구 제표정보면서 하기싫어한다싶음해줘요ㅋㅋ주말엔 청소기걸레화장실베란다청소하구! 쓰레기는 출근하면서 한번씩버려요ㅎㅎ'], ['저희남편은 쓰레기담당(분리수거,음쓰,화장실쓰레기통비우기)이랑 화장실청소구요. 평일엔 집안일 도와줄 시간이 거의 없지만 제가 몸 안좋거나 힘들어하면 설거지+국 끓여놓는정도 해요.\n평일 2~3번 출근전 아침밥 해놓고 가기도 하고 주말은 거의 남편이 식사담당(3끼 중 2끼) 빨래(세탁기 돌리고 널고 정리까지 포함) 한달에 4번정도 해주는 것 같네요~'], ['분리수거쓰레기 빨래개기 매일밤설거지! 집정리할때도잇고 애들목욕이랑설거지이두개만해주면정말좋울텐데ㅜㅜ'], ['1도 안도와줘요ㅡㅡ^'], ['저희두요 ㅠㅠ'], ['분리수거&쓰레기버리기 랑 빨래널기, 주말청소 도와주기요~ 다른건 별말없는데 주말은 쉬고싶다고 해요ㅋ'], ['저희는 도와주는 개념은 아니고 그냥 분리수거,음식물쓰레기,쓰레기 쌓이면 그때그때 먼저 본 사람이 버리구요. 신랑이 요리하는걸 좋아하는편이라 퇴근후나 주말에 주방은 보통 신랑담당, 청소는 제가 담당하는편이예요~ 각자 적성에 더 맞는걸로요 ㅋㅋㅋ 신랑이 정리는 젬병이라 청소나 빨래개기는 거의 제가 하는거같네요'], ['손하나 까딱안해용ㅋ'], ['바쁜신랑은 평일엔 전 독박육아고 , 주말엔 신랑이 삼시세끼 요리 다해요 , 전 아이들 케어하고요 , 솔직히 살림은 장비빨이라고 모든 기계가 거의 해주니 , 신랑의 집안일 몫은 요리밖엔 없네요 ;;'], ['다들 자상하시네여\n우리집 남의편은 가뭄에 콩나듯하는데ㅠ.ㅠ\n남의편한테 잔소리하러 갑니다'], ['분리수거 빨래개주기 설거지할동안 애봐주기요'], ['집안일이란걸 ... 하나요?????'], ['밖에 내논 쓰레기버리기....아이들 장난검정리.....샤워후 변기청소 욕실정리.....그리고 없어요ㅋㅋㅋㅋㅋ'], ['쓰레기 버려줘 청소해줘 해달라해야지 해주던데요 ㅎㅎ'], ['제목만봐도 딥빡이 스치네요ㅋㅋㅋㅋ'], ['설거지 싫으시면 식기세척기 고려해보세요~ 정말 편해집니다. 요즘 문 자동으로 열리는것아닌거는 많이 비싸지도 않아요. 저도 설거지 정말 싫어하는데 남편이 해주지도 않으면 괜히 감정 상해야하잖아요'], ['분리수거\n아이목욕\n강아지 산책...'], ['분리수거, 음식물쓰레기버리기, 애들목욕, 애들재우기, 책읽어주기, 설겆이, 청소기밀기, 화장실청소, 강아지목욕, 먹이주기 등등... 애들애기때는 맘마먹여주고 기저귀갈아주고 씻기고 재웠어요 ㅎ'], ['분리수거만 해요 ㅜㅜ'], ['제가 늦게 퇴근하면 아이들 씻기고 재우기\n그외엔 시키는 것만ㅡㅡ;;'], ['화장실청소빼고 다 해줘요\n신랑이 교대근무해서 자주는 아니지만\n시간날때마다 해줘요\n맞벌이예요'], ['휴무 땐 요리, 설거지, 빨래돌리고 개기, 화장실청소, 분리수거포함 음쓰, 장난감닦기, 애들보고 목욕시키고 밥먹이기, 기저귀 등등 본인휴무에는 제가 쉬는날이라고 대부분 다 해요. 안하는걸 말하면 둘째 이유식 만들기~ 첫째땐 잘하더니 둘째는 안해요'], ['분리수거,음식물쓰레기,아들육아,강아지산책및목욕,청소기돌리기,설거지\n시키면 다 해용ㅋㅋ'], ['대부분 잘 하는데 제 성에는 안차네요.. 뭐라하면 앞으로 안할까봐 입 다물고 있어요~'], ['쓰레기 빨래 집청소 해요ㅋㅋㅋㅋ 요리랑설거지빼곤 다하는듯해욤'], ['아....슬퍼요. 가사는 “같이”하는건데왜 우리는 “도와준다”라고 말할까요....한국은 아직 많이 변해야 합니다 ㅠㅠ남자들은 뭐 하나 가사 하면“도와주었다”며 으스대죠. “함께”하는 가사.....빨리 정착되면 좋겠습니다'], ['222'], ['코로나 터지기전 쓰레기버리기  \n잠들기전까지 애들이랑 놀기 \n주말엔 남편이 애들이랑 자기\n자기밥은 자기가 알아서 먹어라 였구요\n코로나 터진후는   제가 애들땜에 청소를 못해서\n청소기돌리고 물걸레돌리고 하고 있어요'], ['저희는 제가 교대라 신랑이 8개월 아가 거의 독박이에요~ 집안일도 거의..저는 이유식만 가끔 ㅎㅎ'], ['아플때ㅋㅋㅋㅋㅋㅋ해줍니다ㅡㅡ그게1년에 두번해줄까말까....'], ['음식하는것만 쓰레기버리는외에 전부다요 ~~'], ['저희는 모든 집안일 같이해요\n둘이 똑같이 일하고 똑같이 자식키우는데 왜 도와준다 라는 말을 하는지 전 이해가 안돼요. 내가 하는건 당연한거고 남편이 하는건 도와주는건지.. 결혼 전부터 얘기했어요.   모든건 같이하자고~~\n'], ['요리빼곤 다해용~~'], ['쓰레기도 비닐다묶어서 딱 들고가게해놔야 버립니다ㅋ 뭔가 하긴하는데 손이 엄청감..'], ['2222공감가요'], ['시키는거 빼곤 안해요'], ['설거지, 분리수거, 음식쓰레기, 빨래돌리고 널고..는 남편이 해왔었어요. 제가 설거지는 한달에 한번정도?할까말까 할정도로 남편이 해왔는데 , 집에서 음식을 많이 해먹는 편이라 이번에 식기세척기 들였고.. 빨래 널고 정리해넣는건 이제 첫째아들이 맡아서 하기로했어요. 대신 저는 청소, 음식이요~~ 음식은 코로나때문에 하루4-5번은 하는듯해요ㅠㅠㅎㅎ 남편퇴근하면 안주 만들어 주는것까지..'], ['요리랑 화장실청소, 바닥청소빼고 다 신랑이 전부 해주고있어요~!'], ['평일엔 워낙 늦게마쳐 거의다 제가하고 쉬는날은 밥,설거지 빼곤 다해줘요. 같이 일할땐 반반씩 했구요.'], ['집안일은 원래 도와주는게 아니고 같이 하는거라 생각됩니다맞벌이인지라 우린 같이 하는편이라서 설거지 분리수거 청소는 대부분신랑이 하구요음식하는거만 제가 주로하고 시간나는 사람이 집안일하고있어요 서로 그렇게 한지 오래되서 맞벌이가능한것 같아요 '], ['음 맞벌이라그런지 다해요~~ 저도다하구요. 눈에보이는거는 서로 알아서 하는.....'], ['분리수거,,,,,,,,,,,,,빼곤 읍네요ㅠㅜ'], ['맞벌이고 남편이 주로 요리, 설거지, 쓰레기버리기해요 저는 청소기 돌릭ㅋㅋ'], ['분리수거. 청소기돌리기. 설거지. 아이목욕시키기. 빨래개기. 주말에 점심 요리하기. 퇴근 후 아이랑 놀아주기  이렇게 해주고있네요. ^^맞벌이예요.'], ['아이들과놀아주고 ,목욕, 설거지, 빨래개기 ,아주 가끔 해서 테도안나는데요.. 매일이렇게 해주시나요?'], ['왜 도와준다고 말씀하시는지..ㅜㅜ \n\n외벌이면 육아는 엄마가 살림은 같이\n\n 또는 퇴근하고 아빠가육아한다면 그때 엄마가 살림! \n이렇게 같이하는거라고 생각하고\n\n맞벌이면 육아살림 같이 라고 개념을 심어줘야해요\n\n퇴근하고 오면 \n제가 애기목욕시키면 신랑은 강아지패드갈아주고 챙겨주고\n한사람이 밥차리면 다른한사람은 설거지하기 \n빨래는 손남는사람이\n쓰레기는 신랑이 대체적으로 버리고요\n\n주말에는 대청소하면서 분담해서 하고요~~ \n능동적으로는 잘안해서 제가 말을 해줘요.. ㅋ'], ['분리수거 쓰레기버리기\n청소기 \n집에 공구사용해야하는 일\n주말라면 끓이기?ㅡㅡ\n평소 퇴근이 늦어요ㅜ'], ['쓰레기 버리기,  욕실청소 가끔    딱두가지네요'], ['집안일 육아는 도와주는게 아니라 \n같이 하는거라고 하더라구요\n도와 달라 하지말고 같이 하자 같이 해 라고 하세요\n여자가 뭘해도 솔직히 남자들보다\n하는일이 더 많은데 ㅡㅡ\n왜 같이 안하고 항상 여자들이 더 손해보고 고생하는지 원 ㅠㅠ\n저희신랑도 시켜야하지만\n시키는거라도 하는게 어디냐\n생각하고 삽니당~^^'], ['도와준다기보다... 알아서 팔걷어부치고 다 해요. 그냥 싹 다해요. 물이라도 한잔 마시려고 기웃 하면 왜? 그냥 앉아있어 내가 다 해줄께 이런 식이에용... 부모복 없는년 남편복은 터졌나보다 감사하고 살아요...\n식기세척기 고려해보세요... 힘드시면 방법을 찾아봐야쥬...ㅠㅠ']]
    
    6453
    집안일 중에 젤 하기 싫은게 머예요? 아~~~몇달째 진짜 살림만 하려니 우울증 오네요 ㅠ다 하기 싫지만 그중에 젤~  싫은건 빨래예요..하는건 세탁기지만 털고 널고 말린 후엔 접어서 서랍에 넣는것 까지 다 싫어요 ㄷㅏ~~~~~건조기는 놓을 자리도 없거니와 수건만 좋다해서 아직 안들여 놨는데 ..그거 있어도 개서 정리하는건 마찬가지겠죠? 아 진짜 집에만 있는데도 왜이케 많이 나오는지 ㅠ
    
    [['빨래 개어놓은것 옷장,서랍장에 넣는게 젤 싫어요ㅜㅜㅜㅜ'], ['222222\n차라리 설거지가 훨 나아요ㅜ'], ['333 하면 금방인데요 싫터라구요 ㅠ'], ['444444444 엇 저도요! ㅋㅋ'], ['헉 555555 저같은 사람 또 잇네요 신기방기 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ \n그래서 정리해놓은거 방치하고 그거 애들이 발로차고다니고ㅠㅠ 흑흑'], ['ㅋㅋㅋㅋㅋㅋ 그래서 또 개고..'], ['설거지 정말 하기 싫어요.ㅜ.ㅜ 아침은 대충 먹이는데  점심  저녁 금방 돌아오네요'], ['전 정리요~~\n음식하고 치우고 빨래는 다하겠는디..\n아무것도 없는집을 원하는데 현실은 난장판이네요ㅜ'], ['전 설거지요ㅜ손 습진이 너무심해요'], ['저도요  점점심해져요'], ['음쓰 처리요'], ['건조기에서 빨래 꺼내와서 개는거요...'], ['빨래 서랍으로 탁탁 넣기가 젤로 싫어요.'], ['설거지랑 분리수거.......'], ['설거지도 할만하고 빨래너는건 향기나서 괜찮고 개는건 티비보며 하니 괜찮은데 선반먼지 닦는거랑 화장실 청소요 ㅠㅠ'], ['전..설거지가 젤 싫더라구요..ㅋ'], ['전.. 빨래 돌리는거 빼고 다 싫어요ㅠ.ㅠ'], ['전...요리요..  ㅠㅠ'], ['빨래개기요... 정말 싫어요ㅋ'], ['전 요리요 ㅡㅡ 넘 싫어요 ㅡㅜ 요리똥손이라..'], ['전 바닥청소요.. 매트까지 다 들고 털고 하려니 힘드네요ㅠㅠ'], ['전 빨래 털고 널기요ㅠ'], ['전..신랑 저녁해주는거요...ㅠㅠ'], ['전 요리요..요똥이라 힘드네요ㅜㅜ'], ['전 청소요~ㅠ'], ['전 청소여~~ 청소기 미는거지만...밀고 닦고..넘나귀찮은것..ㅠㅠ'], ['전 다싫어요'], ['저도요!!! 빨래도 요리도 걸레질도 욕실청소도 모두모두 싫어요'], ['전부다 ... 라고 하면 너무 불량주부인가요 ㅠㅠ \n특히세탁에서는 빨래개서 정리하기 \n요리에서는 채소같은 원자료 다듬기 \n청소에서는 걸레질 \n육아에서는 용변뒷처리 못하는 애가 둘이나 있는통에 하루에 몇번씩 들려오는 ‘엄마 다 쌌어요~’소리요'], ['빨래 정리요. ㅠ 사람이 많다보니 서랍에 일일이 정리하는게 힘드네요 ㅠ'], ['정리하는게넘나어렵고힘든데 대충해도스트레스가 없어요. 근데 닦고쓰는게 너무힘들고 더러운게보이니 스트레스에여'], ['밥을 삼식 안챙겨먹을 순 없지만 설거지, 빨래가 제일 싫으네요.. 매일 반복적인 집안 일인데도 매일 매일 저혼자 다 하려니까 너무 귀찮아도 어쩔수 없는 시댁 살림..'], ['전 화장실청소요ㅜㅜ'], ['전 요리요~ \nㅜㅜ'], ['걸레 빠는거요 ㅋㅋㅋㅋ 손빨래류..'], ['전 그나마 빨래가 가장 좋아요.. 설거지 쓰레기 처리 화장실 청소 더러운거 손대기 싫어요ㅜㅜ.'], ['냉장고청소요. 자주 하는 것도 아니지만..그래도 제일 싫어요!'], ['다..요ㅜㅜ\n다하기싫어요ㅋㅋㅋㅋㅋ'], ['설겆이요ㅋ 그래도 신랑이랑 같이해서 그나마 좀 나아요ㅋ'], ['전 다림질이요\n생각처럼 잘 안되고 툭하면 데여요\n생각해보니 변기청소도 넘 싫네요\n'], ['설거지..지금도 엄청나게 있다지요 ㅠㅜ 아~ 저녁해야 하는디'], ['화장실 청소요... 최근에 식기세척기와 건조기 구매해서 설거지와 빨래는 수월해졌는데... 화장실 청소는 도무지 ㅠㅠㅠㅠ 더러운게 뻔히 보이는데도 너무 하기 싫으네요 ㅠㅠ'], ['화장실 청소요 ㅎ'], ['빨래 개기요...'], ['좋은게 하나도 없는데 공동 순위로 설거지,  빨래개기와 정리하기, 쓸기, 닦기 그중에서도 걸레빨기여!  그냥 다 싫은가봐요 ㅎㅎㅎㅎ'], ['빨래개기가 제일 싫어요'], ['설거지가 제일 ㅜㅜㅜ귀찮네여'], ['걸레질이랑 빨래 개는거요ㄱㄴ'], ['다........ㅠㅠ\n밥 하는거. 걸레질. 빨래 개는거.....창틀 청소 쓰레기.....버리기...악..ㅎ'], ['화장실청소랑 빨래개기용ㅠㅠ'], ['저는 화장실 청소용. 글고 밥하는거요;;ㅡㅜ'], ['빨래너는거랑 개는거요~~~ 안개고 쌓아두면 알아서들 건져입는게 일상입ㄴㅣ다;;;;;;'], ['전..부..다..요..ㅜㅜ'], ['청소가 싫어요!\n청소기 돌리고...대걸레로 닦아애해서 ㅜㅜ 싫어요! \n\n신랑 꼬셔서....스팀청소기 사자고 했어요!하지만...아직 안사주네요 ㅜ'], ['다요~~다다다다다~~~~~~~~\n집안일이 싫어요ㅜㅜ'], ['전다한빨래정리ㅠ 아이들세놈에 덩치큰신랑놈옷까지 정리하다보면 왜이리 많은지ㅠㅠ\n얼마전에 건조기가 옷걸이에 걸어진채로 건조되는게개발되야한다고혼자생각하고 궁시렁궁시렁ㅋㅋ'], ['음식하기요.. 머 해먹어야할지  고민고민..'], ['걸레질이랑 화장실청소요 ㅠ 진이 다빠져요 ㅠ'], ['설겆이..걸레질..빨래 개기..남편 먹을꺼 요리하기..내 새끼는 어떻게 하겠는데 남편 니 입맛은 왜 그러니 싶네요ㅋㅋ'], ['전 세탁은 좋아요~~\n음식 하는거 치우는게 젤 시러요ㅠㅠ'], ['음식물쓰레기버리기요... ㅠㅠ'], ['화장실청소 너어무 시러요.....ㅜㅜㅜㅜ팔아파여ㅠㅠㅠ'], ['요리요 힘만들고 보람도 없어요ㅜㅜ'], ['설거지요ㅠㅜ결혼 전엔 생각도 못했던 집안일을 내가 하고 있다니,,,이런 생각이 간혹 들어요ㅠ'], ['요리죠 먹어치우면 없어진다는 게 ㅎ']]
    
    6479
    집안일 젤 싫은거 뭐세요? 저는화장실 청소요ㅜ 유독 민감해요ㅜ 습한걱도 싫고 뽀송뽀송하고 상쾌해야하는데늘 관리가 힘들어요ㅠ 그다음 밥 하는거요 ㅋㅋㅋㅋㅋㅋㅋㅋ치우는것도 싫은데 누가 끼니 걱정만 덜어주면 좋겠어요 코로나로 두끼이상 하려니 ㅠ 해먹을거 한정적이고 매일 시키는 거 (계란,고기,생선) 똑같아요ㅜㅜㅜ다들 뭐 제일 싫으세요 ㅋㅋㅋ솔직히 다 싫은데 저는 화장실청소랑 밥하기요ㅠ
    
    [['설거지요..ㅜ'], ['빨래 갠거 넣는거요ㅠ 소파를 빨래가 점령했어요ㅠ'], ['222222갖다놓기 겁나싫어요ㅜㅜ'], ['333 진짜 싫어요'], ['저도 이거요.\n개는 것까지는 잘 하는데 왜케 넣기가 싫은지 모르겠어요 ㅠㅠ'], ['저도 쇼파가 빨래더미로 점령당했어요 왜이렇게 개기가 싫죠...'], ['저만 그런게 아녔네요 ㅎ'], ['저도요~~~!!!!!'], ['전개는거부터 넣는거까지  젤 싫어요 ㅡㅡ'], ['저도 이거ㅜㅜ개서 넣는거 너무 귀찮아요ㅜㅜ건조기에서 꺼내야하는데 외면중..이네요'], ['고를려니 괴롭습니다..다 싫어요 25년을 했는데도 하나도 익숙해지질.않아요 전생에 아씨였나?  아무것도 하기 싫어요 ㅎㅎ'], ['333333 저도 다 하기싫어요~~~~ㅠ.ㅜ'], ['44444\n세탁기 돌리는것까지만 괜찮아요\n나머지는 너무 힘들고 다 하기 싫어요'], ['4444444 모두 다 박빙에요 ㅜㅜ시키는건 잘 하는편인데...풉🙊'], ['뭐니뭐니해도 설거지요.'], ['화장실청소와 걸레빨기... ㅜㅜ 근데 전 집안일 자체가 다 힘든것 같아요..'], ['저도 설거지... 전 집안일 체질에안맞는걸로ㅋㅋㅋ 제가어지르고다녀요🤣🤣'], ['걸레질이요 ㅠㅠ 확실히 손으로 하면 깨끗해서 ㅠ'], ['무한반복은 마찬가지지만,\n애들 장난감 정리랑, 놀이 후 뒷치닥거리가 제일 싫어요.\n치우면서 기가 막히고 화나서..'], ['저두요..장난감..정리하다 욱해서 다 던져버리고싶어요.ㅜ'], ['빨래 개기와 넣기요.~ ㅜㅜ'], ['전 밥이요ㅜㅠ'], ['설거지요..'], ['빨래랑 청소는 진짜...  노답 지금도 생각만해도 한숨뿐이네요ㅜㅜ'], ['밥이요. 다른건 안해도 되니까... 참을성만 기르면 돼요. 나는 안보인다. 나는 괜찮다.'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ빵터졌어요ㅋㅋㅋㅋㅋㅋㅋㅋㅋ 나는 안보인다 나는 괜찮다ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['남편 뒤치닥거리요. 다 괜찮아요. 하라면 화장실청소도 매일할수있어요. 근데 남편 뒤치닥거리는 진짜 싫어욧!'], ['2222 책상위에 음료캔먹고 그대로 둔거 치우기 싫어요'], ['걸레빨기랑\n\n빨래한 옷 집어넣기요\n\n이상하게 귀찮고 하기싫어요 😑'], ['집안일이 다 싫어요..ㅎㅎㅎ'], ['저도 꼽을수없을만큼 다싫어요ㅋㅋ 밥하는것도 치우는것도 설거지도 빨래도 청소도ㅋㅋ 좋은게 1도없네 우짜죠?ㅜ'], ['저는 바닥 물걸레질이요.. 그담은 걸레 빠는거 ㅠㅠㅠ 브라바 이모님이 잘 해주시지만;; 그래도 가끔 물걸레질 해야해서... 이게 너무 싫어요ㅠㅠ'], ['김치써는거요'], ['걸레질 ㅜㅜ그냥 일회요 물걸레 써도 하기싫어요이사온집 바닥은 무광인데 발바닥이 찍히네요그러려니해요닦아도 보이고 안닦아도 보이니 ㅜㅜ'], ['모 하나 맘에 드는 거 없이 다 싫으네요'], ['밥먹고난 다음 치우는거요~ㅠㅠ\n배부르니 앉아 있고 싶고 누워 있도 싶고~ 한번 일어나면 눈에 보이는거 다 해야하는 성격이라...ㅠㅠ 정말 계속 앉아있고 싶어요'], ['전 요리 및 식사준비요'], ['아 다 제얘긴줄 ..진짜 쓰레기 버리러 나가는것도 귀찮고 집안먼지도 미치겟어요 ㅋㅋㅋㅋ ㅠㅠ'], ['이불빨래...남들은 얼마나 자주하고 사는지 모르겠지만저는 미루고 또 미루네요ㅜㅜ지금 막 돌리고 있습니다...'], ['걸레빨기요. 넘나 시름 ㅠㅠ'], ['설거지요ㅜㅜ설거지통 다치우고그릇넣고 하수구막히거나 기름기잇는거잇으면ㅜㅜㅜㅜ'], ['바닥 걸레질이요 ㅠㅠ'], ['저도 화장실청소인것 같아요..;;\n\n그리고 저도 화장실 습한거 싫어해서..\n화장실문을 항상 활짝 열어놔요~\n집안 습도조덜도 되고 화장실도 마르라고요..\n근데 이게 사바사더라구요~~'], ['부엌일이요 \n부엌에만 들어가면 허당이되요ㅜ \n차라리 청소가나아요!'], ['삼시세끼 밥만 반찬만 누가 해주묜 나머지는 잘 할거 같아요'], ['결혼하고 20년 요리인생인데 시간도 실력도 늘지않아서 정말 하기싫어요 ㅜㅜ\n남들은 뚝딱뚝딱 맛나게 한상 잘도 차려놓던데 저는 식사때만 되면 너무너무 스트레스 받아요...엉~엉~'], ['저는 청소가 제일 싫어요 ㅜ ㅜ'], ['다요~~ㅠㅠ'], ['저는 쌀씻기가 제일 싫어요.. 전기밥솥에 쌀 넣고 취사 누르는 것도 극혐이구요 ㅠㅠ'], ['저도 청소요... ㅠㅠ 청소기 닦고 바닥닦고... 힘들기도하고 시간 다가요 ㅠㅠ'], ['전  요리하는거요.....\n요이는  너무 어려워요 ㅠㅠ'], ['다 싫어요~ 다다다다 싫어요~ ㅋㅋㅋㅋ 설거지도 싫어서(계속 서있고 반복업무) 핸펀으로 동영상 보면서 설거지 하고, 화장실은 청소해도 티도 안나고 한 번 청소하면 1시간 ㅠㅠ..'], ['운동화 빨기요\n그거하고 나믄 쓰러져요'], ['설거지 유독 싫으네요.....싫으면 생길때 바로바로 하래서 해보니이건 뭐 종일 설거지만 하다 끝나는 기분...전 차라리 화장실 청소가 나아요 ㅠㅠ전세라 아직 식세기 못들엿는데.....전 설거지 누가해주면 대장금될 자신잇어요 ㅋㅋㅋㅋㅋ'], ['음식물쓰레기 버리는거요 ㅠㅠ'], ['반찬만들기, 화장실청소, 계절 옷정리 등등요.'], ['저는 이불빨래요ㅠㅠ그리고청소기ㅠ']]
    
    6642
    남편한테 집안일 시키나요 ? 전 쓰레기 버리는거 빼고는 안시켜요 차라리 제가 설거지하고 집안일 할 동안 애기랑 놀으라고 해요  몇번 시켜봤는데 오히려 안하는거보다 못하더라구요 두번 손이 안가게끔 해주면 좋은데 그게 아니더라구요마음에도 안들구요 ... 저같이 남편 집안일 안시키시는분 있나요 ?
    
    [['그래두 시켜야지 안그럼 내가 평생 주구장창 하게됩니다.'], ['그건 그렇더라구요.답답해서 하나둘씩 하다보니 세상에 제가 다하고 있네요'], ['저희는 남편이 너무 바빠서.. 피곤에 쩔어서 밤에 와서 시킬래야 시킬 수 없네요.. ㅠㅠ 그리고 가끔 주말에 시키면 잔소리 쩔어요... 뭐가 드럽다느니 분리수거가 안됐다드니.. 그냥 속편하게 제가 하네요..ㅠㅠ'], ['222'], ['저도 그냥 제가 집안일 하는동안 아이랑 놀으라해요 집에오면 무조건 육아참여! 못다한 집안일이 남았으면 아이들 재우고 해달라하고요 ㅋ'], ['처음에는 마음에 안들어도 자꾸 해보면  잘하더라구요. 도와달라해요.남자들은  말안하면 님힘든지 몰라요. 집안일 별거 아니라 생각해요.'], ['222.남편 생전 집안일 못하는 사람인줄 알았는데\n휴직하며 집안일 시작하더니\n이제는 저보다 더 깔끔하게 해요.\n해버릇하니 잘하더라구요...\n뭐든 시켜야 하나봐요\n\n시키고나서 음쓰처리기, 식세기 구입했답니다...ㅋㅋㅋ\n본인이 해보니 너무 귀찮고 힘들었던거죠 ㅋㅋ'], ['그쵸. 다들 해봐야 알아요. ㅋㅋ'], ['처음부터 잘하는사람있나요. 요령 알려줘가면서 시키면 곧잘하더라구요'], ['분리수거 하고 애들하고 주말마다 놀아주고 청소기 건조기도 돌리고 가끔 방도 닦고 화장실청소도 해요\n'], ['마음의 문제라고 생각해요. 얼마나 잘 하냐를 기준으로 삼기 보다 저사람이 내 고생을 덜어주려고 노력하는구나.. 고맙다 이런 따뜻한 느낌을 받을때 행복한거 같아요. 꼼꼼한건 덜하겠지만요.. 대신 남편한테도 제가 뭐 무거운것도 잘 못 들고 하니 시원찮으려니 생각하며^^ 그리고 아이들 교육적으로도 아빠가 가정일을 도우면 자연스레 배울것 같아요.. 저희는 시아버님께서 가정적이셔서 그런지 남편이 가정일을 당연히 하는건줄 알고 있더라구요~'], ['가끔 물걸레청소기만 돌려달라해요 나머진 제가 다하는편이에요 두번하느니 ㅎ 제가 혼자잇을때 하는게 맘편해서요 서로 잘하는거 하기로햇어요 남편은 돈버는거? 저는 집안일 ㅋㅋ'], ['안시키는데..남편이 설겆이,청소..알아서 다 해요. 문제는..하면서 잔소리..ㅠ  먹은 그릇은 물을부어놔라...그런요...'], ['부러워요^^'], ['대신..요즘 돈을 못벌어와요ㅋ 휴직수당받는중요..ㅋ 그래두 부러우시나용?ㅋㅋ 에효..코로나가 길어지니 남편이 심심해죽는지...집안일 찾아해요ㅎ'], ['남편이 해서 결과가 만족스런 일만 시켜요쓰레기 버리기, 필터갈기, 운동화빨기, 세차하기, 빨래개기이런거요 ㅋ시키지 안으면 절대 알아서 하지는 안아요'], ['주말 식사담당해요^^ 뒷정리는 제 몫입니다용'], ['저보다 잘하는거 몇가진해요 빨래개기 재활용버리기 저없을때 먹은거 설거지정도요'], ['반이상 신랑이 해요^^\n처음에는 맘에 안들어도 꾹참고 보지말고 계속 칭찬하면서 시키세요\n그러다보면 살림이 늘어요\n'], ['저희 남편은 원래 하나도 안했는데 애가 둘이다 보니 도대체 안되겠더라구요. 이제는 청소기 돌리는 거와 저녁 설겆이는 고정이예요. 비교하면 안되지만 저희 제부는 집청소, 설겆이, 쓰레기 버리기 다 하고 주말 되면 혼자서 애 보면서 동생 필라테스 보내주는거 보고 ... 멘붕 오더라구요. 그래서 저도 슬금슬금 시키고 있어요.'], ['평소에 분리수거,화장실 청소는 남편 담당이구요 빨래 게는거 자주 도와주고 주말아침엔 집안 청소 해줘요~'], ['남편이 하는 딱 두가지분리수거 가끔하고 요리요주말에 거의 세끼 준비를 남편이해요전 거들거나 안하거나요평일도 가끔 퇴근후 저녁준비요참고로 애셋이요ㅎ'], ['전 이미 글러먹었나봐요 맘에 안들어서 안기켰더니 이제는 당연한줄 아네요 ... 이번생은 틀렸어요'], ['안시키다 어쩌다 시키면 그래요맘에안들어도 계속 시키고 나중에 좀 손 보세요'], ['돈벌어오니 집안일은내담당이다생각하는 중입니나ㅠ일을워낙힘들어 하기도하고 ᆢ애들이랑 잘 놀기도하고 데리고 산책도잘가고요설거지는진짜하루종~~일 하니 속터졌고 지금은 식세기있지만사용법도 모르네요ㅠㅠ'], ['저희집 신랑은 알아서해요ㅎㅎ'], ['했으면 좋겠는데 안하네요~ 자랄때부터 안해와서(누군했나..) 허허...왕자님처럼 자랐더라구여 ㅎㅎ(허허 나는 공주였는데)그냥 하기싫어하니 아이랑 놀으라고 합니다. 그래서 3대 가전을 들였어요. 그리고 집도 대충청소합니다 ㅎㅎㅎㅎㅎ다행히... 더럽다고 터치안해서 그냥 그러고 살려구여 ㅋㅋㅋㅋㅋㅋㅋ말년에 힘없을때 부려먹어야지 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['애놀아주는것도, 집안일도 둘다 해야죠ㅋㅋㅋ'], ['저도 거의 안시켜요~ 못하더라고요 ㅡㅡ\n손이 더가요~'], ['화장실청소 쓰레기 분리수거 주말 밥차리기 설거지 이렇게 시켜요~해야 힘든거 알더라구요~'], ['분리수거, 설거지요..그거라도 해야 그나마 내맘속의 억울함이 조금은 누그러지는것 같아서요^^ '], ['요리빼고는 남편이 다해요 결혼초반엔 아무것도 안시키고 저혼자 싹 다 했었어요 골병을 얻었죠.. 그이후에 제가 몇갤동안 몸이 많이 안좋았을때가 있었는데 그계기로 남편이 다하기 시작했어요 이젠 저보다 잘해요 나중엔 지혼자 다 할줄 안다고 나보고 나가라고는 안하겠죠?.. 😂'], ['전 시켜요 이것좀 저것좀 이러면서;;;'], ['전혀안시켜요 ^^밖에서 돈벌어오니까 다 제가 ㅋㅋ'], ['저도 두번은 꼭 거치는데 그래도 시켜요.. 님만 너무 힘들고 스트레스잖아요 그리고 모르면 집안일 우습게 보니... 전 집안일은 제일이라고 했던 즤남편이 이젠 설거지며 방닦기 빨래개기 가끔 밥해주고 이젠 장실이 담당이 되었답니다.. 그리곤 집안일이 참 힘들다고해요.. 돈벌어오는거 고생하는거 잘 아는데 그래도 다는 아니여도 같이 하기도해야된다 생각해요 그래야 아니깐요!'], ['저두 맘에 안들어도 시켜요,...안그렁  평생 안해요ㅋㅋㅋㄱ'], ['맞벌인데 집안일도 애도 못돌보는 남편은 뭐 시켜야할까요 애맨날 크게 울려요 아빠가 이세상에 제일 싫데요 깐족깐족 건들고 괴롭히고 협박하고 발로 밀면서 귀찮아 저리가 이러네요 아빠 책한권만 읽어줘가 소원이에요ㅡㅜ 다행이 사회성은 좋아서 돈만 벌어옵니다'], ['맞벌이인지라..시킵니다...청소랑...설거지요...저도 처음에는 맘에 안들었는데...맘에 안들어도 잘한다..우쭈쭈 하면서조금씩 가르쳤더니..이제는 아주 잘해요...결과물이 맘에 안든다고 거기서 포기하지 마세요...여자들은 뭐 태어날때부터 살림 배워서 태어난거 아니잖아요....계속 하다보니 실력이 느는거지...남자도 똑같아요..실력을 키울수 있는 기회를 마구마구 주세요!!!!^^'], ['알아서 잘 해요. 저도 시키고 뭐고 하고 싶진 않더라구요. 그걸 시켜야 하나요.. 같이 사는데 알아서 하는거지-라고 저희 남편이 말하더라구요. 요즘 남자들은 다 한다고 고맙다는 인사도 사양해요. 가사 분담 스트레스 있음 넘 힘들거 같아요 ㅠ'], ['전 주부인데 아침식사. 빨래 널기. 분리수거는 신랑이 해요. 종종 반찬도 만들고 청소도 해주는데 그렇게 도와주니 편해요.'], ['전업이라 아예안시켜요\n돈버는거 외엔  아무것도 안해요'], ["신생아다 생각하시고 수저 놓는거, 정리하는거, 씽크대에 그릇 놓는거, 헹구는거, 빨래통에 빨래 넣는거, 양말 벗는거, 티셔츠 벗는거, 세제 넣는 법, 밥솥에서 밥 뜰때 어떻게 떠야 하는지, 반찬통에서 접시로 반찬 덜 때 어디 부분을 어떻게 어느정도로 뜨는지 등등 다 알려줘야해요. 내가 자라면서 집에서 배웠듯이 저사람도 기본 생활습관은 돼있겠지 라는 기대를 가지고 대했더니 돌아오는건 뒷통수 후려치기 더라구요. 지나고보니 처음부터 신생아다 생각하고 잘 가르치는 태도로 임했으면 갈등이 훨씬 덜했겠다 싶은 생각이 들어요. \n\n저는 결혼하고 5년 정도는 저사람은 신생아구나를 깨닳았고, 이후에는 그 신생아가 나가서 사람들이랑도 안싸우고 잘지내고, 돈도 벌어오고, 공부도 잘하니까 대견하다 생각하고 집에서 맞춰갔어요. 지금은 10년 넘었는데 제법 잘해요. 가르친 보람이 있더라구요. 지금부터라도 '저 사람은 무의 상태다'라고 생각하고 마음을 내려놓으시고 하나하나 친절하게 가르치세요. 그럼 언젠가는 돼요. ^^"], ['저는 시킵니다~일도힘들지만 집안일도 힘들다는걸 신랑도알아야된다고생각해서요'], ['ㅠㅠ 벌써부터 스트레스가 .. 재활용버릴때 같이 갖다버리고싶네요 ㅠㅠ'], ['집청소, 쓰레기버리기만 시켰더니 일주일에 한번 하면서 얼마나 생색을 내던지요. 요즘은 제가 몸이 안좋아 누워만 지내야해서 다른 집안일도 다 시키는 중인데......정말 차마 눈 뜨고 볼수 없이 어설프지만 꾹 참고 있어요. 직접 해봐야 집안일이라는게 시간 많이 소요되고 손 많이 가도 티안난다는걸 알듯요'], ['완벽함을 바라지 말것, 칭찬해줄것, 꼭 경험 해보도록온전히 맡겨(부탁)볼것! 입니다..뭐든 해봐야 귀찮은 일, 힘든 일, 희생과 배려가 필요한일이 집안 일이고! 본인을 시키지 않는, 온전히 감당하는 아내에게 고마워합니다.저도 애 낳고 많이 가르쳤네요.. 요리 빼곤 함께 합니다.'], ['남편이 더 잘해요.음식도 남편이 더 잘 하는 것 같아요.전 걍 잔소리 하던 말던, 지금처럼 살래요.대부분의 남편들이 지금 제 맘 아닐까 이 글을 쓰면서 생각해봅니다.'], ['알아서 하네요~~~'], ['제가 따로 시키지는 않고가끔 해주면 고맙더라구요~^^'], ['계속 가르쳐주고있어요\n세탁기돌리고 빨래빼서 건조기에넣고\n분리수거하러갈때\n음식물쓰레기같이 가져가게하고\n청소기돌리고\n먼지털고\n밥먹고나면 설겉이통에 담그고 식탁닦는일 등등\n화장실청소 기타여러가지\n점점 추가시키고있어요\n못한다고했던일 대부분인데\n스스로했던일할때 잘한다해주고\n정말 천천히 추가시켰더니\n거부감없이 점점 하데요\n다만 빨래게고\n아이씻기고 재우고\n음식 주방일은 제가 온전히하고\n다른일은 대부분 같이요~\n마음에안들고 두번일같아도\n제가힘들땐 그거라도 함께하는게\n큰힘이 되는거같아요'], ['저희집은 집청소 화장실청소 설거지 전부 신랑이 저보다 더 잘해서 신랑시킵니다! ㅋㅋㅋㅋㅋㅋ'], ['알아서 하는 분들 넘 부럽네요 ㅠ'], ['남편이 집에 잇음 많이 도와주긴 하는데 마무리는 제가 다 해야해요 ㅎ 설거지를 다해줘도 물이 흥건하고 청소기를 돌려줘도 모서리나 선반 이런데는 제가 작은툴로 다시 해야하고 화장실 청소도 마무리는 제가 하구요. 그래도 그렇게라도 도와주면 넘 편하더라구요. ^^ 노력하는 모습을 보면 기분도 좋구요~'], ['저는 설거지 시켰더니 식기세척기를 알아서 들이고, 빨래 널으라고 시켰더니 건조기를 구입하더라고요... 음쓰도 시켰더니 처리기까지 달고..... 지금은 다 기계의 힘을 빌려서 하고 있습니다. 뭐 나쁘지 않네요^^;;;;'], ['저는 최대한 기계사서쓰고 나머지는 하는거알려줘요.음식이야 사먹는다치지만 청소랑쓰레기랑빨래는해야죠.저죽음 우째사나싶어서시킵니다.'], ['저는 집안일은 제가 하고 애들하고 놀아주라고 해요. 제가 애들한테서 벗어나고 싶은맘과 애들이 아빠랑 함께 있는 시간을 갖게끔하는 의도인데ㅋ 안시키는 여자같네요ㅋ 그래도 분리수거랑 화장실청소는 알아서 잘하네요 ㅋㅋ'], ['너무 못하지만 꾸준히 시키고 있어요\n시킨지 낼모래면 십년가까이 되는데\n점점 더 못하네요......\n저랑 두뇌싸움 하는듯 해요 ㅋㅋㅋㅋㅋ\n시킨일도 요새는 잘 안하기도 하구요 ㅋㅋㅋ\n아오 !!'], ['시키진 않지만 워낙 가만히 못있는 사람이라서 청소, 빨래, 설거지, 분리수거 스스로하긴해요. 전 좀 느긋한 편이구여. 그치만 화장실청소와 같이 디테일이 중요한 곳은 도저히 못맡기겠어요.. 그래서 화장실 청소만은 제가하고 설거지 마음에 안드는건 아무말 안하고 넣기전에 빼서 한번 더 닦아요.. ㅋㅋ 기름진거 먹는날은 제가 하구요. 잔소리하면 싫어해서 움직일때마다 칭찬해주고 빠릿하다. 엉덩이 두들려주니 좋다고 잘 하는 첫째 아들입니다..'], ['전업이면 집안일보단 육아를 같이 하는게 더 맞지않나 싶어요. 돈을 벌어오고 집안일을 하는걸로 나눈셈치고요. 맞벌이라면 이모님 따로 부르지 않은이상 체력의 한계 때문에라도 나누게 되죠.'], ['빨래널구 개기,분리수거,물고기 관리 수조 청소 정도요..워낙 바빠서 더 시킬래도 시킬수가 없어요ㅠ'], ['주말에 청소기 돌리기도 시킵니다~안 시키니까~ 집안이 엉망이어도 할 줄도 ~하려는 맘도 없어 보이고 뺀질거려서~ ^^'], ['자주하다보면 늘어요~^'], ['둘째4살인데 둘째낳고 자연스럽게 주말마다 밥차려주고 설거지까지 해줘요ㅋ 분리수거.화장실청소.운동화도 빨아주고 청소기도 돌리구요~~ㅋ'], ['시키면 싫어해서 그냥 두는 편인데ㅡ알아서 하지를 않네요..그래서 가끔 폭발해요. 답답한 내가 하지 뭐ㅡ하다가 몸아프고 에너지 소진되는 날은 열받아서 시키네요...빨래는 못한다하니 설거지,분리수거 정도만..'], ['제가 딱 글쓴님이랑 똑같아요~저는 쓰레기도 제가 버려요~쓰레기를 보물단지처럼 안고 내려가서 속터져서요ㅠ'], ['분리수거, 쓰레기버리기랑 가끔 주말에 화장실청소정도요.  평일은 아이들이랑 놀아주거나 씻기는정도만해요.'], ['평일에는 피곤해하고 늦게와서 씻고 자기 바쁘고\n주말에 스스로 부부욕실청소, 창틀청소, 식사준비랑 설거지해요 \n오히려 분리수거나 쓰레기는 제가 쌓인게 보기싫어서 평일에 치워요 ㅋㅋ'], ['맞벌이라 주말에 전 요리, 신랑은 설거지 청소 빨래 화장실청소 어항청소 다 합니다. 자기가 도와주니까 편하다 우쭈쭈 필수네요.ㅎ 그래도 저 장보고 요리하는게 시간상 더 오래걸리고 본인은 할일 딱 하고 게임방으로..그 꼴 보기 싫어서 요리할때 저 준비하는동안 주걱으로 저으라고 시키는거까지 도와달라고하니 군말없이 하더라구요..시켜야 압니다..  요리나 집안일이 얼마나 끝없고 힘든건지~'], ['맘에 안들어도 꼭 시켜요..화장실 청소,주말에 청소기 돌리기는 항상 시켜요'], ['전 남편이 스무살무터 자취해서 요리 빼고 어지간한 집안일은 제법 하는 편인데, 애들이 어리다보니 집에 오면 애보라고 하고 제가 집안일해요(전 육휴중) 두번 손이 가거나 가르치기 귀찮다기보단 전 하루종일 애보는 게 지치니까 집안일 핑계로 남편 있는동안 최대한 애 안보려고 ㅠㅠㅎ;; 또 아빠도 애들하고 유대를 쌓아야 하니까요. 그래도 남편이 일한다고 고생한다고 알아주니 서로 윈윈이랄까.. ㅎㅎ'], ['제가  팔골절이후 할사람이 없으니 시작했는데 처음엔 맘에 안들고 답답했는데 이젠 아주 만족해요.ㅎㅎ\n완치후에도 할꺼냐고 하니 자기체질이합니다.\n시켜보고 일단 믿어보세요.'], ['저도 얘보라해요 제가하는게 속편해요 ㅎ'], ['청소기돌리기 빨래널기 개키기 쓰레기버리기 단순한건 시키고요 애가 어릴땐  진짜 많이 도와줬는데 요즘은 제가 편해져서 가끔 시켜요']]
    
    6658
    남편분들 집안일 어느정도 하시나요? ㅎ 밥먹고 반찬통 냉장고에 넣어주고 식탁행주로 닦아줘요 몇달에 한번? 현관신발 놓는곳 물티슈로 청소하고 분리수거, 음쓰 나올때마다 버리는거, 가끔 빨래정리, 아주 가끔 빨래 널기, 이불정리, 청소기 대충 가끔해주네요그외 걸레나 화장실은 절대 안하구요 ㅠ 이 정도면 집안일 많이 하는 편인가요? ㅎ다른 분들은 어떠신가요??
    
    [['제가 힘들어서 밀림 해요.\n가사도우미 모드 아주 가끔요.\n주말마다 ㅋㅋ ㅜㅜ'], ['저도 밀림 하긴 한답니다 ㅎ 가사도우미모드 좋네요^^'], ['전혀 안 합니다워킹맘에 아들 둘주말엔 본인 쉬어야 한대서아들둘 데리고일박이일 대한민국 안 다닌곳이 없이벌써 고등이네요일요일 밤 9시즈음기진맥진어린 아들  둘  데리고운전하랴 애들 챙기랴평일 내내 늦은 퇴근에피곤한데 주말도 반납..그 몸으로 집에 들어와씻기고 재워 놓음꽃게탕 끓여내라는신랑입니다~본인손으로 라면도 안 끓여 먹고식탁 셋팅도 안 해요~이 점 빼고는다 백점이라 데리고 살고 있습니다의리네요 ㅎㅎ'], ['저희신랑 전혀 안해요 이번생은 망했어요ㅠ'], ['그죠 ㅜ 시대가 아직 ㅠ'], ['평소에 많이 해줘요. 정리도 설거지도. 할줄 아는거에 한해서 요리도요.\n음쓰 ,분리수거는 온전히 남편 몫이네요.. 대신 장실은 제가 해요ㅇ'], ['장실 청소 너무 힘들어요 ㅋ'], ['분리수거,일주일 한번 주말에 청소(화장실도) 고정은 이것만 해요 나머지 도와달라고 말하면 도와주지만 거의 주중은 퇴근이 늦어 제 몫이죠^^'], ['하긴 일하고 오면 피곤하죠 ㅎ'], ['일주일에 한번 주말청소는 꼭 남편이 해요. 화장실은 제가 하고 제가 식사 만들어서 차리면 설거지는 해주죠 ㅋㅋ 그치만 결국 육아와 살림은 거의 제몫....ㅠㅠ 맞벌이에 항상 힘드네요 ㅠㅜ 방금까지 내일 먹을 반찬 만들었어요 ㅋㅋㅋ'], ['반찬 ㅜ 해도해도 끝이 없네요'], ['때론 마음 놓는게 편한 것 같아요'], ['빨래 너는건 건조기가. 설거지는 식세기가 하다보니 개는것과 정리 정도합니다'], ['요즘 정말 잘나오죠 ㅋ 저도 갖고 싶네요'], ['이정도면 훌륭한거 아닌가요?저희도 이 정도인데 전 엄청 훌륭한지 알고 살았네요 ㅋㅋㅋ다른 집 남자들보고 시댁가서 얘기했더니 거짓말이라고 절 비웃던데요 ;;;'], ['그죠 ㅋㅋ 저도 엄청 도와주는줄 ㅜㅜ'], ['정해진건 없어요.식사준비랑 욕실청소는 온전히 제몫이지만 나머진 그냥 아무나해요.'], ['욕실청소는 해줬으면 좋겠어요 ㅋ'], ['쓰레기는 버리라면 버려주고\n말안하면 안버림..ㅠㅠ\n주말에 집대청소,싱크대,화장실,창틀 기타등등\n그것외엔 집안일 일체 안하는데 그래도 전 뭐라안해요ㅋ\n애들 태어나서 \n젖병소독이며 밤중수유(분유먹일때)며 목욕까지 항상 시켜줘서요~\n작은애 육퇴도 신랑이해요ㅋㅋ;\n육아를 담당해줘서 집안일은 안해도..ㅋㅋ'], ['육아 담당도 크죠~^^'], ['담배피러 나가며 음식물쓰레기 버리고\n주말 아침 간단히 차려주고(계란볶음밥,토스트)\n가끔 요리해고(튀김류 잘해서 튀김은 항상 신랑이하네요.)\n라면 끓일일 있음 신랑이 끓여요. \n신랑 씻고있을때 걸레 나오면 빨아줘요. \n운동화 세탁할일 있음 직접 해줘요.\n뭐든 부탁하면 잘해줘요.\n\n딱히 하는거 없다 생각했는데\n적다보니 몇개 있긴 하네요ㅋㅋ\n'], ['멋져요~'], ['저희 남편은 오자마자 빨래,청소,주방부터 봐요.ㅋ 그것도 많이 좋진않아요.잔소리좀 있거든요ㅠ'], ['헉 잔소리 너무해요 ㅜ 차라리 혼자 할래요'], ['전혀 안해요. 가끔 쓰레기 버리라고 시키면 할까 ㅡㅡ 자기 밥 먹은 그릇 조차 정리해본일이 없네요.'], ['힝 ㅜ 밥그릇은 물에 담궈 주시지 ㅜ'], ['아무것도 안하지만, 시키지도 않아요.\n그냥 그 부분은 기대도 안했고,\n안해준다해서 속상하거나, 서운하거나, 화나지도 않아요.'], ['전 화장실은 해줬으면 좋겠어요 ㅋ'], ['전혀 안해요~~ㅠㅠ'], ['퇴근하고 오면 피곤하긴 하죠 ㅠ'], ['남편이 거의 재택이라 아침하기,(저는저녁담당) 쓰레기 버리기, 빨래 개어놓기는 남편 담당이요.  화장실은 각자 써서 각자 청소하구요.'], ['각자청소 좋네요 ㅎ'], ['우와 너무 너무 부럽네요~'], ['분리수거, 설거지, 음쓰버리기, 애들이 어질러놓은 장난감 정리, 청소기 돌리기, 빨래 개기가 매일 하는 코스요. 교대근무라 낮에 집에 있을 때가 많아서 많이 해요. 저보다 깔끔한 죄로 ㅜㅜ 쓰고보니 미안하네요. 근데 요리를 못 해서 식사준비는 항상 저요. 애들 공부시키는 것도 저요.'], ['딱 좋은데요? ^^'], ['똑같은 교대근무였는데....1도 안하는데.....심지어 저는 연년생 워킹맘....이예요......\n이래도 세상에서 자기가 제일 좋은 남편 아빠인줄 알고 사네요......휴........'], ['시키는 것만 아주 잘해요(설거지 요리 쓰레기버리기 청소 정도) 그치만 알아서 해주면 좋으련만....'], ['그죠 ㅎ 부지런한 남편들 부럽네요'], ['음... 나갈 때 쓰레기 버리고, 제가 정리를 잘 못하는지라 아주 가끔 집안 전체 정리정돈해주는데... 결정적으로 요리를 잘해줘요. 본인이 쉴 때 삼시세끼 중 한끼는 꼭 해줍니다. 기 ㅁ 수미씨나 배 ㄱ 종원씨 요리 벤치마킹해서요... 그런데, 이것도 사실 제가 요리를 잘못해서 입니다. 웃프네요..ㅠㅠㅜ'], ['요리.. 전 남편이 담배를 펴서 그냥 제가 해요 ㅋ'], ['스레기봉투,.분리수거..버리고 강아지 소변패드 갈아주는...딱..그것만..ㅠ 즤 남의편은 머리카락이 뭉치로 굴러다녀도..절대.,안치워요..대신 잔소리도 안해요;;;;;'], ['ㅋ 저도 청소 잔소리는 안해서 그냥 넘어가요'], ['음식빼고 웬만한건 거의 다해요~~ \n본인이 더러운거 못참아서 하는 스타일이에요 덕분에 몸은 편해요 맘이 불편해서 그렇지'], ['그래도 부럽네요 깔끔한 성격 ~'], ['거실화장실청소, 주말설거지 전부다, 평일저녁설거지 가끔, 커다란 택배박스 나올때마다 출근할때 가지고나가서 버림 요정도네요~남편이 결혼내내 주말에 청소기랑 스팀걸레 밀었는데 작년에 애브리봇이랑 로봇청소기 구매이후 안하더라구요ㅋㅋㅋ  저는 전업주부에요'], ['저도 화장실청소만큼은 좀 해줬으면 ㅜㅜ'], ['일반쓰레기만 버려요. 담배피러 갈 이유가 있어야 하니까요. 그렇게 쓰레기를 찾네요'], ['맞아요 백번공감 ㅋ'], ['다 똑같은데 신발장 물티슈로 닦아주는 건 안하네요. 해달라고 해야겠어요. ^^;;'], ['맞아요 ㅎ 먼지투성이인데 ㅜ 그정도야 ㅋ'], ['저희도 1도 안해요 자기는 돈벌지 않냐고 하는데 저 돈벌때도 안했어요'], ['조금이라도 도와주시면 좋을텐데 ㅜ'], ['주말 같이 대청소하고 평소엔 설겆이해요 밥 물양 잘맞춰 밥도 가끔 해줘요~저 가끔 주말출근하면 아이들 밥해주고 분리수거는 일주일에 한번 꼭 해주구요 무엇보다 아이들과 잘놀아주니 그게 제일 좋더라구여ㅋ'], ['멋져요~'], ['반반하는 것 같아요 육아시간 쓰고 있어서 저는 아침 남편은 저녁에 아이 보면서 집안일 해요\n한명은 아이 보고 한명은 밥 하고 먹고 나서는 교대요\n규칙은 아닌데 같이 살려니 자연스럽게 그렇게 되더라구요'], ['해달라는대로 다 하는데 꼭 말해야만 하네요. 왜 스스로 할수없는지 의문이에요.'], ['22222 \n딱 공감해요. 시키는건 군말안하고 다 하는데, 좀 스스로 하면 좋겠어요.\n매번 말하고 부탁하는것도 귀찮고 지침요ㅠㅠ'], ['남편이 6시에 퇴근해서 집에오는 순간부터 저는 집안일 퇴근, 남편은 출근이요. 밥준비부터 치우고 설거지 ,후식, 아이들씻기고 방에 들여보내고 쓰레기버리고 뒷 정리까지 전부다 해줘요...'], ['저희는 제가 전업인데 저녁 설겆이와 주 2회 걸레질 하고 평상시 정리 도와주고 화장실은 이야기 하면 해줘요.'], ['전업인데 집안일은 남편이 스스로 알아서 더 해요 제가 안하니 하는것도 있겠죠ㅋㅋㅋ\n밥도 잘못해서 고민하다 지난번에 요리못해서 어떡해하니 밥차릴라고 결혼했냐구 해서 감동한적 있네요ㅋㅋ'], ['시간날때는 꽤했는데\n\n요즘은 분리수거,음쓰,애 씻기기\n가~끔 빨래돌리고 개고\n그외 뒤치닥거리(고치기,보수하기)정도네요\n(최근 책 1~2권 읽어주기 시작)\n\n\n제가 생색 엄청 냅니다~~\n전 일하지만 재택근무라 \n비교적 시간이 많다보니 하거든요\n\n신랑도 시간나면 아이 등하원 ,유치원가방챙기기,아침이나 저녁 간단,간식 ,가끔 요리,빨래,세차 다하는타입이에요\n(설거지빼고)\n\n부부의 시기마다\n포지션 변경중이에요ㅎㅎ\n\n'], ['외벌이지만요 집안일 완전 잘해요 부탁안해도 손도 많이 빠르고 자취를 오래해서 알아서 눈에 보이는건 다해요(살림고수)\n퇴근후 샤워후 욕실청소하고, 개수대에 음식물쓰레기 보이면 갖다버리고요 평일 아침에 기분좋게 일찍일어나면(전날 본인이 좋아하는 요리를 제가 해줬을때) 밥해놓거나 아님 샌드위치 만들어두고 출근해요\n주말에는 대청소해주고, 애들하고만 나가서 2시간정도 외출했다가 들어오고요 마트에서 장봐오고요 주말에 두끼정도 요리해주고, 제가 손 느리다고 설거지도 주말에는 해주고요(비싼 그릇 다 이 나갔어요 ㅜㅜ)\n평일에 애들 자기전에 양치질도 시켜주고... 잠들기전 책한권씩 읽어주고요 여자아이들이라서 이젠 목욕 못시킨다고 아쉬워해요\n작년에는 큰아이 초1때 아침등교시키고 출근했어요\n잔소리는 살림의 참여도가 높아서 조금있어요'], ['담배 피러 가는김에종량제.음식물 딱 두가지요이것도 제가 정리해서 내놓으면요외벌이요'], ['외벌이... 것도 10년차 주말부부인데 아무것도 안해요\n본인은 타지에서 고생하고 집엔 쉬러온다 하는 인간이라\n진심이지 애들 케어만도 힘든데 이인간 집에 오는 금요일은 확마 인상부터 써져요\n어쩌다가 이런걸 만나 결혼을 했나 도끼질합니다'], ['저희 신랑은 주로 더러운거 담당이요 ㅎㅎ 음쓰 버리기, 분리수거, 화장실이랑 배란다 청소, 어항 관리요. 밥차릴 때 상 놔주고 치우는건 항상 같이 하구요'], ['빨래개기\n쓰레기와 분리수거 전부\n요리할때 볶고 다듬고 쌀씻고\n무거운거 들고 나르기 집안 잡일?\n이게 답니다.\n저희집도 식세기와 건조기와 로봇청소기와 로봇걸레가 있어요.그거외에 안합니다ㅜ\n대신 말하지 않아도 자기일인줄 알고 찾아서 딱 저만큼은 알아서해요.'], ['밥은 주로 제가하고 나머지 몽땅해요ㅜㅋ 저질체력이라 언제가부터 그리되어버렸어요. 화장실청소 빨래 집정리 계절옷정리 음쓰 분리수거...쓰다보니 뜨금하네요.맞벌이긴한데 제가 체력이 늘 딸리는 쪽이라 ㅜ'], ['음쓰, 분리수거, 화장실 청소는 무조건 신랑, 일주일 3번 정도 집 청소해 주고 설거지, 빨래널기, 개기 신랑이 자주 해주네요. \n주말 아침, 점심도 신랑이 애들 챙겨줘요.\n그래서인지 아이들이 탕 종류는 저보다 신랑이 한게 더 맛있다 하네요^^;;\n저 전업주부인데도 신랑이 집안일 함께 해줘서 너무 고마워요~'], ['1도안해요 애들이랑 열심히 놀아주기는해요ㅋㅋㅋ'], ['분리수거 음쓰 쓰레기버리기는 100퍼 신랑담당이요 ㅋㅋ아침스스로차려먹기 퇴근함 빨래정리 도와주구 주말엔 돌아가며 식사준비설거지 청소해줘요 전업인데 제가애보느라저질체력이라 신랑이 같이많이해요 맞벌이하면서 분담했던게 얼추남은것같아요^^; 집안일은같이하는게맞죠 대신 육아관련한것은거의제가전담이예요'], ['청소전담, 분리수거, 둘째 씻기기 치카, 본인 옷정리, 스타일러 전담...저 일찍 쉬거나 피곤해하면 설거지하고 배달시키면 뒷정리 다 하구요 사실 이정도 해도 제일이 산더미라는거ㅠㅠ 나 일할때 너 빈둥대는거 제일 싫다고 우리집은 너 없어도 문제없다했더니 그뒤로는 하네요'], ['분리수거 ,쓰레기버리기, 빨래통에 옷넣기'], ['전 9:1 남편:저'], ['전업인데 퇴근이 늦어서 평일엔 거의 못 도와주고 주말에 화장실 2개 청소는 꼭 해주고 간단식(라면, 토스트, 인스턴트 등)해주거나 가끔 청소해줘요 건조기, 식세기 있어서 나머지는 제가 하고 대신 애들 잘 케어해요(목욕담당)'], ['식사하고나면 둘 중 한 사람이 테이블정리 한 사람은 그릇 애벌해서 식세기에 넣구요~ 분리수거나 음식물 쓰레기는 주2회정도 버려줘요~ 그래서 그 때 분리수거 모으는 편이고 음쓰는 제가 먼저 버릴 때도 있구요~ \n매일 로봇청소기 비우고 걸레 씻어 끼우는 거 제가하고 남편이 큰 청소기로 돌리고 밀대 밀어주는 거 주1회 주말에 해요~\n대신 7시에 식사하고 이후 10시에 아이 잠들기 전까지 씻기고 놀아주고 재우는 등 아이케어 전적으로 다 맡아줘서 고맙게 생각해요'], ['서로 상부상조'], ['적어주신거 남편이 매일 다해요; \n남편이 요리만 못해서 요리는 제가해요\n최근에 식세기를 사서 남편이 더 좋아합니다 ㅋㅋ'], ['분리수거\n화장실청소\n운동화빨기\n쓰레기정리\n음식쓰레기버리기\n이불털어주기\n빨래널기\n이 정도요~\n설거지 해달라고 하면 해주고요'], ['저흰 맞벌이구요.\n집안일 육아의 비중을 따지자면\n남편90  저 10이에요.\n저는 저질체력이라 남편이 밥하고 빨래하고 청소하고 애보고 거의 다해요.'], ['빨래 분리세탁만  제가 하고 다른건 남편이  거의 다 해요 \n가구 옮기고 싶다. 책장 옮기고 싶다.. 그러면 나가서 놀다오라하고 혼자 다해놔요\n주말오전은 아이들 밥 챙겨주고 오후엔 저 쉬라하고 놀이터든 키카든 데리고 나가구요 \n전업인데 참 고맙네요'], ['저는 저녁식사담당\n신랑은 그시간에 빨래개우고 서랍에넣어주 는거해요\n화장실청소도 100번중90번신랑이해요. 분리수거는 신랑이버려주고 냄새예민한남자라 음쓰는 100번중 90번 제가버려요. 아이셋이고 결혼9년차.. 전업8년에 워킹맘1년하고 또다시 육아휴직중인데 워킹맘때처럼 여전히 빨래개우기랑 넣어주기 도와주는 남편이 고맙네요.. 대신 제가 육아는거의오롯이혼자..ㅋㅋㅋ 뚜벅이지만 애들셋데리고 여기저기 잘 다녀요. 평소에는 자전거뒤에 유아안장에막내태우고 아들딸 첫째둘째는 킥보드타고 같이 여기저기 다니기도하고 ㅎㅎ 목욕도 100번중 95번은 제가시켜요 아이들.. 아이들이 엄마껌딱지.. 운동화세탁도 항상 제가해요..'], ['아예 아무것도 안해요..\n쓰레기 현관앞에 내놓으면 출근할때 그것만 들고나가서 버리는것만..ㅠㅠ\n\n로봇청소기.건조기 사주고는 아예 안도와줘요ㅠㅠ\n일이 힘드니 그러려니 해요..\n돈잘벌어오니 그냥 그래..힘들지 하고 아무말안해요ㅎㅎ'], ['분리수거박스랑 안방화장실요'], ['맞벌이땐 같이 했어요.. 지금은 외벌이인데 거의 와이프가 해요. 같이 도와주고 싶어도 이젠 체력이 안되네요 ㅠ 대신 식세기 건조기 스타일러 등등... 살림에 도움될만한거 사달라고 하면 사줘요... 글 적다 보니 미안해지네요 ㅠ'], ['분리수거 중 박스만 버려줘요.\n로봇청소기, 식세기 등등 기계들였다고 내가 노는줄아나봐요.\n하지만 집안을 내맘대로 난리를 쳐도 가만히 있어요.'], ['음식쓰레기 매일 버리기.\n청소기 매일 돌리기\n물걸레 매일 밀기\n분리수거 담당\n\n\n딱 여기까지예요.\n정리 할 줄 모르고 요리할줄 몰라요;\n'], ['침대정리 식사후 자기자리정리 쓰레기 분리수거 쉬는날 청소기정도 하네요ㅎㅎ'], ['제 기준으론 많이 하시네요\n\n전 그냥 도와주면 고맙고 집안일은 내 몫이다 생각해요'], ['일할때는 안하고 백수되니 잘하네요'], ['그냥 딱 정해놓지는 않고 신랑퇴근전까지는 제 몫 퇴근후나 주말엔 신랑이해요 그러니 보통 요리는 제가하고 그 외는 신랑이요.. 주말엔 반반..'], ['맞벌이라 화장실청소 각종가전청소는 남편 전담이고 나머지는 거진 반씩 나눠서 해요 저는  빨래는 거의 제가  하고요'], ['요리빼고는 거의 다 해요.물론 저도 같이 하구요. 할수있는사람이 하고 있어요. 거의 남편이 전담으로 하는건 분리수거,일쓰, 음쓰버리기,화장실청소, 아이들데리고 놀이터가기,아이들목욕.\n 같이하는건 청소기밀기,설거지,빨래 정도 되네요ㅎㅎ'], ['저흰 맞벌이인데 저질체력에 둘째가지고있고 몸이 안좋아서 남편 90% 저 10% 대신 깨끗하게 치우고 살진않고 몰아서 청소하고 합니다. 요새 건조기랑 식세기샀는데 신세계예요. 남편 완전 만족중 ㅎㅎ'], ['음쓰버리는거 화장실청소 걸레질 결혼 10년인데 한번도 해본적 없어용~ 설겆이담당,막내 목욕도 신랑 담당 ㅋㅋ케'], ['매일저녁설거지 아침에 장난감 정리,분리수거버리기. 음식물버리기. 애기 유치원등원. 고기굽기.간단요리 해요~'], ['저도 망'], ['저는 그냥 제가 시켜요 남자들은 시켜야되나봐요 ㅠ 쓰님도 그냥 부리세요 어쩔수없어요']]
    
    6659
    집안일 중에 가장 하기 싫은 일은? 전 빨래개기에요~진짜진짜 왤케 빨래개기가 싫은지 모르겠어요.죄다 뒤집어 벗은 양말에 티에..어젯밤에 빨래개다가 신랑한테 궁시렁대며빨래개는 기계가 나옴 천만원이래도 사겠다~했더니 그냥 자기가 갤테니 두라네요ㅋㅋㅋ저처럼 진짜 하기싫은 집안일 하나씩은 있으시죵?^^
    
    [['설거지요!! 땀나고 허리아프고 어깨아프고..ㅋ'], ['전 설거지는 개운한 맛에 그냥저냥 합니다....라고 말하지만...식세기 사고파요ㅋㅋ'], ['그래서 저흰모셨죠?.식세기...ㅋ매일 신세계경험입니다..강추드려요^^'], ['부럽슴돠..아직 즤집엔 아날로그를 선호하는 신랑이 있어서요ㅠㅠ'], ['게고 제자리에 넣는것도 너무 귀찮아요 저는 ㅎㅎㅎㅎ'], ['맞아요! 갖다가 넣어두기까지 한스텝이니까요~ㅋ'], ['요리요ㅠ 저는 먹는것도 넘나 귀찮아요..'], ['전 요리는 괜찮지만 치우기가 싫어욧!ㅋㅋㅋ\n먹는건 진짜 좋아요^^'], ['음쓰치우기요. ㅠㅠ'], ['아..요것도 싫은 리스트 중에 하나이긴 하죵...그래도 저에겐 빨래개기를 이길 순 없네요ㅋ'], ['저는 화장실청소요..ㅋㅋ\n화장실청소해주는  지니가 있었으면좋켔어요\n빤딱빡딱 깨끗하게..ㅋㅋ\n'], ['전 청소랑 설거지는 빤딱빤딱한 결과물을 보면서 뿌듯해 하는지라 해줄만(?)하답니다~~~^^'], ['ㅋㅋ저도 빨래개기요! 빨래개고 갔다놓는게너무 귀찮어욬ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['찌찌뽕입니다~~~\n전 널기까지는 해줄만한데 왤케 개는게 힘든지 몰겠다니까요~아..진짜 싫어요ㅠㅠ'], ['그쵸?ㅎㅎ 저도 너는건 좋은데 개는건 싫어 남편한테 넘겼어요~~'], ['저두 개는건 남편한테 넘겼는데 이누무 남편이 티비 봄서 갠다더니 맨날 티비보다 그냥 자요ㅠㅠ\n아침에 혈압이 협압이~~~~~'], ['옷정리요. . 잘못해요\n철바뀌면 싹정리하고버리고 ㅜㅜ\n옷도별로없는데도 하기시르네용'], ['옷정리 것두 보통 일이 이니쥬..\n그거 하다봄 온 집안이 먼지가 먼지가~~~\n그러다보면 결국 집대청소까지 하게 되더라구요ㅠㅠ\n그러니 조심조심 최소한만 건들이며 정리한답니다ㅋㅋ'], ['전 청소여^^'], ['나는 지금 격한 운동중이다~~~라고 자기암시를 한다쥬^^;;;\n이것마저 안움직이면 전 진짜 몸을 써서 땀 흘릴 일이 없을 거 같아서ㅋㅋㅋ'], ['음쓰버리기요ㅜㅡ'], ['저두 한동안 남편 마이 시켰는데요..\n이쁜 음쓰통을 사게 되면서 그거 관리한다고 요즘 부지런히 내다버리네요.\n버리고 들어오면서 잘샀네~잘샀어~그럼서 흐뭇해한다쥬~^^;;;\n근데 요게 얼마나 가려나 싶기도 합니다ㅋ'], ['전  설거지요.. 저녁에  국수 해먹었는데 냄비두개.  후라이팬. 소쿠리.컵..등등 간단한 한끼지만   너무 많이 나와요..'], ['전 음식하면서 설거지 진짜 많이 해요. 안그럼 나중에 너무 많아질까봐 무서워서..\n그래서 음식하는 시간이 많이 걸리는거 같은데 가끔은 이러나저러나 어차피 다 내가 할텐데 나 지금 뭐하는거뉘?하는 생각이 들 때가 있어요..또르르...'], ['전 그냥 집안일하고 안맞는거같아요 ㅎㅎ'], ['사실 능력만 된다면 일하면서 내 돈으로! 당당하게 도우미분 도움받고 싶은게 찐 마음이긴 해요...'], ['걸레질이요 ㅠㅠㅠㅠ 젤싫어요 ㅠㅠㅠ'], ['예전 한동안 무식하게 무릎꿇고 걸레질 주구장창 한 적이 있었더랬죠.\n근데 결국 무릎에 무리가ㅠㅠ\n그래서 그당시 큰 맘 먹고 물걸레 청소기 들였는데 좋습디다ㅋㅋㅋ\n그 후 걸레질은 할만허나 그 걸레를 빠는게 일이 됐다쥬ㅠㅠ'], ['전 빨래개기 좋아해요;; 그냥 티비보면서 무념무상 멍때리기 좋아요. 전 근데 빨래 너는게 싫어여 ㅋ 건조기 해도 널게 있긴 하더라구요'], ['빨래개고 서랍에 넣는거 재밌어요 ㅎㅎ'], ['아..울신랑하고 같은 얘기하시네요.\n근데 차이가 있다면! 울남편은 갠다고 해놓고 티비만 보다 잠들어버린다는거~~~\n아오! 빨래 베고 티비 켜놓구 잠든거 아침에 볼 때마다 승질이~~~승질이~~~'], ['오옷..진쫘요???\n전 1도 공감 할 수 없어요ㅠㅠ'], ['설거지요 ~~'], ['설거지도 많으네요~ 하긴 배부르면 만사귀찮긴 해요^^'], ['밥차리는거요...^^'], ['ㅋㅋ 맞아요! 밥은 차리는거 말고 먹는거만 했음 좋겠어요~~~'], ['빨래 개는건 괜찮은데 각각 서랍이나 옷장에 넣는게 진짜싫어요ㅜㅜ'], ['갖다 넣는 것도 엄청 일이라니까요. 방마다 다 돌아다녀야 하고.. 또 넣다봄 다시 정리해야 하는 상황도 생기구..이해함돠ㅠㅠ'], ['설거지요 ㅠㅠ 진짜 하기 싫은데 하루에 3번이나 해야해요'], ['설거지..맞아유.. \n밥도 세 번 차리고 설거지두 세 번..\n보통 일은 아니에요ㅠㅠ\n그러니 빨래개기만은 안하고 싶어요~~~~~'], ['정답입니다! 다다 힘들어요~엉엉~\n비 오거나 흐리믄 몸이 귀신같이 알아서 축축 늘어져 더 힘들구요.\n진짜 암것두 하기 시르네요ㅠ'], ['전 청소요ㅠㅠ'], ['전 청소는 운동 대신이다~ 생각하면서 합니다~\n평소 운동을 1도 안하거든요ㅋㅋ\n이마저도 안하면 진짜 굴러만 댕기지 싶어서 핑계삼아 움직입니다^^;;;'], ['걸레빨기, 빨래갠거 서랍에 넣기ㅋ'], ['ㅋㅋㅋ\n위에 제가 다 하기 싫다고 한거에요~~~\n걸레질은 할만하나 걸레빨기가 싫고 \n빨래널기는 할만한데 개고 넣고 하긴 싫고~~~ㅋㅋ\n사실 내가 마이 게으른건가..싶을 때가 있어요^^;;;'], ['화장실 청소요..ㅜㅜ\n진짜 해도해도 티안나는 집안일중 갑 아닐까요...ㅜㅜ'], ['결혼후 신랑이랑 첨 싸운게 화장실 청소에요~ 즤집 남편은 화장실 청소는 물만 뿌리는거라고 생각하는 사람이드라구요.\n그 후 보란듯이 빤딱빤딱하게 해놓고 청소란 이런것이다~를 보여준다는게 결국 이제껏 제가 한다쥬ㅋㅋㅋ\n드럽고 힘들긴해도 하고남 나름 뿌듯~해서 할만합니다^^'], ['빨래개기 장난감 정리정돈하기요.. 치우면어지르고 치우면어지르고 반복이네요ㅜ'], ['맞아요. 아이가 어릴 땐 장난감 정리도 엄청나죠ㅠ치워도 치워도 끝이 없고..가지런히 정리도 하루이틀이고..결국 나중엔 큰 장난감함을 구해다가 다 때려넣었어요~눈에 안보이니 청소한거 같아 좋기도 하구ㅋㅋ애기들 어릴 땐 그냥 좀 내려놓고 사는게 낫더라구요~😄'], ['다른건 다 신나서 하겠는데 \n진짜 정리정돈은 눈물 쥘쥘 흘리면서 겨우겨우해요.. 너무 싫어요.. 제발 정리정돈 좀 누가 해줬으면 좋겠어요ㅠㅠ'], ['제가 또 정리정돈 한 때는 차~암 잘했지말입니다... 그러나!!!왜 나이가 들수록 인내심이 바닥인건지 막막 조금만 뭘하다가도 답답하고 빨리 끝내야하는데 안되고 하니 승질이..ㅋㅋ이젠 걍 다 때려 넣는걸로 정리는 대충 마무리 합니다ㅋㅋㅋ엄마도 집안일을 모든걸 다 잘할 수는 없으니까요~^^'], ['계절옷 정리해야는뎅ㅠ\n느므 하기시러 하나씩 꺼내입고 빨고하다보니 다섞임요ㅋ\n 서랍장이 난리가났어욤^^;; 흐흐흐'], ['저두 그러다그러다 서랍 안닫기고 결국 강제로 닫다가 뒤로 옷이 막 넘어가고....ㅋㅋㅋㅋ\n그렇게 버티고버티다가 날잡고 정리했어요🤭'], ['음식....음식하는게 세상에서 제일 가치 없고 쓸데없는 일 같아요정말 너~~~~~~~~무 싫어요'], ['전 바닥 물걸래질이요'], ['설겆이에 한표~식구가 많다보니 양도 어마무시하네요']]
    
    6673
    집안일 오늘도 어김없이 집안일~~ 청하맘마주고 이불털고 이불털다 이슬이가 묻힌 흔적이 보여서 ... 이불빨래~ 돌리고 바닥 청소기 돌리고 바닥 이슬이 털 안묻을때까지 닦고 걸레빨아서 마당에 널고~ 이불빨래 끝남 이제 옷돌리고 그담 수건 돌려야겟으요~~~ 아침이 늦으니 점심도 늦은 청하 ㅋㅋ 청하 점심은 리조또?? 조동이 애기가 좋아한다길래 저도 해볼려고요~ 도전언
    
    [['리조또좋죠 바쁘셨네요 점심맛있게드세요~^^'], ['네~♡♡ 감사합니다'], ['ㅎㅎ리조또 잘먹겠네요 반찬없어도 되구ㅋ'], ['전 맛나는디 잘 안먹네요 ㅋㅋㅋ'], ['집안일도 한가득이지요~'], ['네 ㅜㅜ 끝이 없네요 돌아서면 또 생기고요'], ['퇴근하고 전 다시 집안일 시작이예요ㅠ'], ['맘님이 더 고생이 많네요 ㅠㅠㅠ'], ['근데 일하는게 더 나은것 같아요'], ['네 ㅠㅜ 저도 그리 생각해요 ㅜㅠ 육아보다 일하는게 난것같아여'], ['육아가 더 힘들어요~;;'], ['네 ㅜㅜ 청하보고.집안일에 차라리 일하는게 낫겟다 싶더라구요 ㅜㅜ'], ['워킹맘들이 다 그렇게 이야기 하더라구요~ 육아보다 일하는게 더 낫다고~'], ['ㅠㅠ 워킹맘 아니지만 일하는게 낫다는거에 동의해용'], ['저는 둘다 해봤는데 저도 일하느게 나은것같아요~ㅎㅎ'], ['저도 얼릉 일하고프네요 ㅜㅜ'], ['둘째 낳고 일하시려구요?'], ['그러고픈데 둘째도 어는정도 키워놓고햐여겟죠 ㅜㅜ'], ['저도 둘째 3살때부터 다시 일했어요~'], ['일자리가 잇을지 모르겟어요 ㅋㅋ 전 시간때가 그래서 ㅜㅜ 애를 늦게까지 맡길수도옶고 ㅜㅜ'], ['저도 일자리가 있을까 걱정했는데 있긴있더라구요~^^너무 걱정마세요~있을꺼예요~^-'], ['하던일은 없을수도요 ㅠㅠ'], ['저도 결혼전에 하던일이 아니라 다른일해요~ㅎㅎ'], ['저도 그로갯죠 ㅋㅋ'], ['하던일 찾기는 어렵더라구요'], ['그래도 뭔가 할수있다면 좋을것같나요 ㅋㅋ'], ['저도  요즘 감사하며 다니고 있어요~^^'], ['일할수 있다는거에 정말 감사할거같아요 ㅠㅠ'], ['저도 7월까지라 8월부터는 다시 일자리 알아봐야해요~;;'], ['이직하세요?'], ['지금하는 일은 계약직(?),알바(?) 처럼 짧게 하는거예요'], ['컴퓨터 업무죠??? ㅠ 전 컴터 못하는데용 ㅜ좋은데 또 찾으실거예요~'], ['네~ 컴퓨터는 기본적으로 해야되요~정말 이만한 자리 또 있을지 걱정이네요'], ['배웟다 안하니 또 까먹엇어요 ㅠㅠ'], ['저도 안하면 까먹게되요~전 컴퓨터 자격등도 다 까먹었어요~ㅎㅎ'], ['ㅋㅋ.전 따지도 않앗어요'], ['기본만 할줄알면 되요~ 쓰는것만 써요'], ['기본 ㅜㅜ 타자치기요 ㅜㅜㅜ'], ['결혼전에 하던일이 이여서 할수있으실꺼예요~벌써부터 걱정하지마세요~^^'], ['ㅋㅋㅋ 결혼전에 컴터 쪽으로 안해서요 ㅜㅜ 전 컴터 다루는건 다 못할듯요'], ['어딘가에는 청하맘님이 원하는 일자리가 있을꺼예요~^^'], ['아... 정녕 임산부가 한일이 맞나요ㅠㅠ'], ['매일하는일이예요 ㅋㅋ'], ['ㅠㅠ에효.. 쉬셔요'], ['비오는날이 그나마 이불안터니 좀 쉬어요 ㅋ'], ['에고 힘드시겠어요 청하는자나요?'], ['자면 좋겟아여 ㅜㅜ 트램펄린 사줄랴고 보는데 가격대가.천차만별이네요'], ['저희 샀는데 애들용은 아니고 어른운동용 같은..?50인치안되는데 롯마에 5마넌안되길래 샀어요'], ['오 ~ 저는 거실에 둬야하는데 이슬이도 잇고해서 남편이 펼쳐놈 좁아보인다하니... 36인치로 주문햇어요 ㅋㅋ 잘땐 치우는걸류 하고요'], ['저것도 접히는건데....;;쉽지않아요 조립은쉬운데 힘이 많이들더라구요ㅠㅠ'], ['접는게 많이 힘든가요??? 그럼 안되는데요ㅠㅠㅠ'], ['오늘도 빡세게 보냈군요'], ['네 ㅜㅜ 이불빨래만 추가됫네요'], ['고생했어요 ㅠ'], ['ㅜㅜ 할일이 항생 추가되는듯요 ㅋㅋ'], ['리조또 성공하셨나요?'], ['네 ㅋㅋ 성공햇어요'], ['히히 사진보니 맛나보였어용'], ['청하가 첨에 뱉어서 난감하고 놀랫지만요 ㅋㅋ'], ['처음먹는거라 그럴걸요?'], ['ㅋㅋㅋ 그럴까요 ㅋㅋ 치즈 잘안먹긴하거든요'], ['진짜 신기한게 치즈 잘먹던아이가 어느순간 안먹더라고요'], ['수시로 바뀌더라구요 ㅠ'], ['ㅋㅋㅋㅋ알다가도 모를 입맛'], ['ㅋㅋㅋ 잘먹다 안먹다 ㅋㅋㅋ 오르락내리락요 ㅋㅋ'], ['늘 잘먹는 아이들이 부럽긴하더라고요'], ['네 ㅜㅜ 잘먹어야 ㅎㅑ줄맛도 나고요'], ['ㅋㅋㅋ그건 그래용']]
    
    6732
    집안일 싹 해놓고 티비 보는 지금 이시간이 행복❣️날은 흐리지만 환기 싹하구 잇네요이불 건조기에 돌리는대오늘 꿀잠 잘수잇겟어요다들 뭐하세요??
    
    [['아들이랑 놀아요 신랑은 출근가고ㅠㅠ유난히 찡찡거리네요 😭'], ['아이고ㅠㅠ 육아 힘 ㅜ ㅠ!!! 화이팅입니다 주말 순삭이네요ㅠㅠ'], ['집좋고 넓네요 부럽 엄청깔끔하심 저도 헬육아로 지치고 집이 지저분하네요 저리깔끔하게 해놓고 있을수가 없네요 애들이 저지러서요'], ['성격이 ㅋㅋㅋ 지저분한걸 못봐서요 ㅋㅋㅋ... 걍 치우기 합니다'], ['전..첫째 밥먹여요~~ 피곤하네요.....쉬고파요ㅠㅠ'], ['언니는 어째 주말이 더 바빠요ㅠㅠ 엉엉 커피 수혈이라도ㅠㅠ'], ['커피 한잔 진즉했어요ㅋㅋㅋㅋ또 마셔야할거같아요ㅠ..둘째깨면...전쟁일꺼예요ㅋㅋㅋㅋ'], ['ㅜ  ... ㅠ 엉엉 둘찌야 좀더 푹 자줘ㅠ ㅜ.... 언니 화이팅이에유ㅠㅠ'], ['일어날때되어가요ㅋㅋㅋㅋ둘찌자는동안..집안일하고첫째 놀아주고ㅋㅋㅋㅋ'], ['악 ㅜ ㅠ.. 이제 곧 일어나는가요 엉엉 언니 밥은 드신거래요..?...'], ['아침엔 시리얼 후딱말아먹었어요ㅋㅋㅋㅋ'], ['엉엉 그건 밥이 아닌대ㅠㅠㅠ 엉엉 언넝 뭐라도 드세유ㅠㅠ'], ['첫째 밥먹이고있어서요ㅋㅋㅋㅋ둘째깨면 둘째도 먹여야하고......ㅋㅋㅋㅋㅋㅋ'], ['ㅜ ㅠ 엉엉 언니는 진짜 주말이 더 바쁜거 같아요 힘내보아요ㅠㅠㅠ'], ['넹ㅋㅋㅋㅋ얼른..평일이 오면좋겠어요ㅋㅋㅋㅋ출근하게ㅋㅋㅋㅋ'], ['출근이 더 좋은 언니... ㅋㅋㅋ 전 주말이 넘 빨리 지나가버려서 속상'], ['저도 애 한명일때까진..주말을 기다리고.ㅈ.주말이 빨리가서 아쉬웠는데....지금은...너무 힘들..ㅠ'], ['언넝 내일이 오길요 언니ㅠㅠ!!\n이제 반찬좀 하러떠나봅니다 ㅋㅋ 아'], ['넘편은 공  때리러~전 세탁기좀 닦고 뒷베란다 나가는 창 닦으려구요.그리고 비빔국수  해먹을려구요.지원금 쓰러도 나가야겠어요.'], ['비빔국수 넘 좋네요 ㅋㅋㅋㅋ \n저는 아점은 먹엇고 저녁은 뭘먹을지 ㅋㅋㅋ'], ['맞아요 집정리 다하고 차한잔 마시면서 우너하는것 하는 그 시간이 가장 행복한것 같아요 여유를 맘껏 즐기세요'], ['ㅋㅋㅋ 맞아용 히히 이제 커피 한잔 내려서 마셔야겟어용 히히'], ['여유스러운 시간이 길어야 좋을텐데 . 지금도 여유를 즐기시고 계시나요?마냔부럽습니다. 전 출근했어요 ㅠㅠ'], ['네네 ㅎ 아직도 여유 즐기구 잇지요 ㅎ 커피 한잔 내려서 먹으니 꿀❣️\n출근 하셧구만요ㅠㅠ'], ['시간 시간~~좋네요~~^^'], ['네네 ㅎㅎ 이시간이 제일 좋네용 ㅎㅎ 섬유유연제 냄새 폴폴'], ['정매력님 집이 왤케너무 깔끔하신가요\n살림 배우고 싶습니다!!'], ['보이는 곳만 깔끔해유.... ㅋㅋㅋㅋ 속고계십니다요 히히'], ['저도 이른시간에 청소 후다닥해놓고 근무중이네요~~'], ['저도욘 ㅎ 주말인대 역시 시간 순삭입니다 ㅎㅎㅎ 늠 여유로워요'], ['아점 먹고 빨래 돌리고 있어요ㅋㅋ청소 시작 해야지요!'], ['ㅋㅋㅋ 저도 마지막 빨래 돌아갑니단 ㅋㅋㅋ 이불만 돌아가면 끝입니닷'], ['저두 빨래 돌린거 건조기돌리공 ㅎㅎㅎ신랑 일어나면 점심먹을려구용😋그전에 꽃봄이 맘마먼저 ㅋㅋㅋ'], ['오마 이게 누구래여ㅠㅠㅠ 갱이님 잘 지내세우??? 꽃봄이 잘 자라쥬?'], ['엄청 많이 컸어요 ㅋㅋㅋㅋ많이 소통하고 싶은데 아직 전처럼 되지가 않아요 ㅠㅠ😭😭'], ['아이고ㅠㅠ 아직 케어하느라 정신 없으시죠ㅠㅠㅠ 육아 화이팅입니다'], ['온능 돌아오겠숩니당 ㅎㅎㅎㅎ😍'], ['아침먹고 청소하고 지금은 또 점심먹을 준비해야하는데엄청 하기싫으네요ㅠㅠ'], ['저희는 아점으로 어제 사온 닭갈비 댑혀서 묵기만 햇어요 ㅎㅎ 저녁은 남은거에 밥 볶으려구용 ㅎㄹ'], ['남은양념에 밥 볶는거 짱이지요~^^'], ['그럼 또 한끼 해결 되니 늠늠 조아용 ㅋㅋㅋ 꺄하 넘 든든'], ['집 느므 깔끔하그 깨끗하네요\n저희집은 .. 한숨이 푸욱 ㅋㅋㅋ'], ['저희집은 애기가 없어서 깔끔하지 싶어요 ㅋㅋㅋㅋㅋㅋ 성격이 이모양이라 ㅋㅋㅋㅋㅋ 지저분하면 화나요 ㅋㅋㅋ'], ['휴 첨엔 그래서 애 따라다니며 치우고 그랫는데\n세살정도부터는  ...해탈\n포기햇어요 ㅋㅋ'], ['ㅋㅋㅋㅋㅋ 악 해탈이라니 ㅋㅋㅋㅋㅋ 애기 생기기전에는 깔끄미 하게 지내 보려구요'], ['서로 스트레스 받더라구요\n정말 애 어릴때는 아예 한손에는 \n돌돌이를 들고다녔다니깐요 ㅋㅋㅋㅋ'], ['ㅋㅋㅋ 악 청소할때 젤루 조은 돌돌이 ㅋㅋㅋㅋ 스트레스 받지말아여ㅠㅠㅠㅠㅠ'], ['청소하고 쇼파에 앉아서 쉴때 젤 뿌듯하고 행복하죠^^'], ['마자요 ㅎㅎㅎ 뽀득한 기분요 ㅎㅎ\n환기하고 잇는대 한번 더 닦아야 겟어요..'], ['저도 쇼파에 앉아있는데 저는 밥만 해먹었네요 ㅎㅎ 저희집에 청소요정이 찾아왔으면 좋겠네요 ㅎㅎ'], ['ㅋㅋㅋ 제가찾아갈까요 제가 청소를 할테니 슈슈님이 밥을 해주세요'], ['우리매력이 발꼬락ㅋㅋ귀엽네ㅋㅋㅋ일요일도 부지런하고만'], ['발구락 뾱ㅋㅋㅋㅋ 언니 일주일에 한번 대청소지뭐ㅠㅠ...... 엉엉'], ['저도 이불이랑 베개커버 돌리는중이요ㅋㅋㅋㅋ 청소는 아직 안끝났는데 조금하고 폰하고 조금하고 앉아있고 이래요 ㅠㅠㅠㅠ'], ['ㅋㅋㅋ 저는 청소 빨래 다햇어요 \n건조기에서 이불만 돌아가면 끝!!!'], ['저도 청소하고 혼자 티비시청중임다.'], ['저도요 ㅋㅋㅋ 부세 어제꺼 못봐서 보려고 대기중이에요 꺄하'], ['신랑 야채샐러드 10통 만들고 방금 앉았네요~근데 곧 점심시간 이라는....;;;\n애둘 없어서 자유는 오늘까지^^'], ['오아 저도 야채샐러드 누가 만들어 주면 좋겟어요ㅠㅠ 완성샷 하나 올려주세용 ㅎㅎ'], ['별거없어요^^빨.노 파프리카에 오렌지.가지포도.아몬드.방토.블루베리 들어가요^^야채샐러드가 어니라 과일통같네요..ㅋㅋ'], ['크으 부지런 하십니닷 !!!!\n이렇게 해놓음 편하겟어요👍'], ['도시락을 싸가서^^그래도 반찬 안만드니 세상 편하네요..ㅋㅋ'], ['오 저게 식사인 거에요?? 엉엉 저거먹고 하루를 버티시는건가요...'], ['ㅋㅋㅋ닭가슴 2팩에 고구마 2개 단뱍질쉐이크  추가욤~'], ['ㅜ ㅠ 그렇게 챙겨주는 것도 쉽지 않을텐대 200점님 대단하세유ㅠㅠㅠ'], ['전 반찬하는것보다 저게 더 좋아요^^\n반찬 뭐할까.. 걱정안해도되서욤...ㅋㅋ'], ['짝짝 대단하십니다 ㅋㅋㅋ 우렁각시 잇음 좋겟어요 ㅋㅋㅋ'], ['에이~매력님이 더 잘하시면서^^\n너무 겸손해도 아니되옵니다~~'], ['집정말깨끗하네요\n부지런하고야무진성격이보여요'], ['헤헤 감사합니닷 ㅎ 어질러 잇는 걸 못봐서 깔끔해.?. 보이는거 같아요'], ['집안일 다~해놓고 티비보시는 시간.행복한시간이쥬.저는 한시간전쯤일어나서.이제아침먹엇으니..어질러진 집 치워야겟어요ㅜ'], ['저는 그럴지 알구 아점 일찍 먹고 청소 햇지요 ㅋㅋㅋ 꺄하'], ['나..갑자기 정매력님이랑 체인지하고싶다요ㅜㅜ부럽부럽.그래도 커피는마시고 일할꺼예요ㅋ'], ['ㅋㅋㅋ 헤헤 부세 마지막 회보려고 아침부터 사부작 햇어요 히히'], ['ㅋㅋㅈㅓ는어제 친구랑 맥주한잔하면서 보고 울엇어요ㅜㅜ'], ['저는 어제 못본 그것이알고싶다 보구잇어요ㅎㅎ오늘 바람도 시원하니 문열고 환기 시키니 너무 좋네용'], ['ㅋㅋㅋㅋ 저도요... 닫고 이제 공청 켜봅니당 저는 부세 기다리고 잇어요'], ['쉬는날인데도 일찍일어나셨네요 이시간\n집안일까지 끝내시고 \n늦잠 안자유? 습관되서그른가ㅎ'], ['낮잠 안자요ㅠㅠㅠㅠ 이상하게 낮잠이 안와요... 저도 제가 시러유ㅠㅠ'], ['잠도 안자면 줄고 잘수록늘긴한디 한번씩 피로도풀겸 자줘야하는디요'], ['이상하게 눈뜨면 ㅋㅋㅋ 다시 잠이 안들더라구요..... 하하'], ['저는 순천만습지 다녀오고 아랫장 다녀왔지요~'], ['크으 가을에도 이쁘지만 초록하니 이쁘죠 ㅋㅋㅋ 친정근처라 괜히 반갑습니다 ㅋㅋㅋ'], ['하루종일 청소하고 정리하고.. 역시 주부들은 다 그런가바여.. .ㅠ'], ['응응 주부의 삶이란 ㅜ ㅠ 그래도 그 뽀독한 기분 너무 조아 ㅋㅋㅋ'], ['맞아여ㅋㅋㅋ근데 저는 그기분이 얼마가지 못해요..다시 어질러질테니..']]
    
    6770
    맘님들은 집안일 중에서 뭐가 젤 좋고 뭐가 젤 싫으세요???? 전 빨래널기랑 음식만들기가 젤 싫고그나마....설거지랑 청소가 낫네요....
    
    [['낮에 집 지키는게 제일좋고\n그외엔 다 싫어요ㅠㅠ 흐잉'], ['100프로 공감ㅋ'], ['아깔깔ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['젤싫은건 빨래 청소 밥하기 정리하기\n젤좋은건 없어요ㅠㅠ'], ['다 싫어요..라고 하면 안될까요? 😂\n역시 반찬은 남이 만든 반찬이 맛있쥬...전 그래도 그나마 빨래가 찌이끔 괜찮아요'], ['설거지랑 밥차리기가 젤싫어요.  청소는 괜찮아요'], ['설거지 싫어요 빨래걷어서 개고 정리하는거 싫어요ㅠ'], ['핫 저랑 반대에요 ㅎㅎ 전 설거지가 세상 싫어요 ㅋㅋㅋ 만들기는 좋고요'], ["좋은거-화장실청소(하고나면 티갇젤많이나는듯해서 ㅋ)\n싫은거/'-빨래개기.....그냥싫어요 ㅋㅋㅋ 넘나귀찮아요 ... ㅠㅠ"], ['요즘은 설겆이요~~'], ['좋은것ㅡ 제자리에 정리하기(강박수준의 원위치)\n싫은것ㅡ개수대 음쓰정리, 화장실 물때와 곰팡이 제거\n해도 티 안나고 안하면 확 티나는 집안일ㅜㅜ주부의 삶은 어디서 의미를 찾아야되는걸까요ㅎㅎㅎ'], ['전 걸레질이요! 매일하면서도 매일욕해요ㅎㅎ'], ['음 오토비스같은거로 안하시고 손으로 하시느거세요??ㅜ ㅜ\n그럼 손목 나가요 ㅜ ㅜ \n물걸레청소기 하나장만하셔유 ㅜ ㅜ'], ['오토비스도 있는데 그 전용걸레 빨때마다 속목이 나갈것같아서 기냥 밀대랑 가구들은 손걸레질하구요 3일에 한번정도만 오토비스로 눌러닦는듯해요ㅎㅎ'], ['그러네요 거기까지 생각못했어요 ㅋㅋㅋㅋ\n그거 빨때는 욕나오죠 저도 그거 빨 때  욕하는데 ㅜ ㅜ ㅋㅋㅋ'], ['그쵸? 역시 저만욕하는게 아니었군요ㅋㅋㅋ 정말 그걸레 징하죠 잉ㅋㅋ'], ['빨래개기'], ['빨래가 제일좋고 설거지랑 밥차리기 싫어요ㅠㅠ'], ['설거지요!!!  밥하는거라요 진짜싫어요!!!'], ['전 다 하겠는데 이상하게 걸레빠는거 느무느무 싫어요ㅜㅠ 차라리 밥이 편해요'], ['걸레빨기요 너무 귀찮아요'], ['신랑 퇴근할때가 젤...싫....'], ['음음  알아요 알아 이심점심이요'], ['저는 ㅋㅋ 창틀닦는거요 ㅋㅋㅋㅋ 매주하면 금방 끝나는데 제일 마지막에 해서 제일 귀찮아요..'], ['ㅎ 화장실 청소싫고. 걸레질 싫고..빨래 널기 개기 다 싫고..  \n그나마.음식하기이나 설거지, 분리수거는 좋아해요..'], ['그냥 다싫어요.. 누워서 잠만자고싶네요ㅋㅋㅋㅋㅋㅋ'], ['전 설거지가 젤싫어요. . 하면서도 짜증나서  몇개닦아놓고  나눠서 설거지할때도많아요 ㅋ ㅋ'], ['화장실청소 제일 싫어요'], ['설거지싫어요..ㅜㅜㅜㅜㅜㅜㅜ\n흑흑다시러요...밥하기도싫고...ㅋㅋㅋㅋ'], ['좋은거 청소기돌리기\n싫은거 요리하고 설거지하기요 ㅋㅋ\n'], ['빨래 세탁기 돌리는건 좋은데\n다된 빨래 널고 개는게 너무 싫어요ㅋ'], ['설거지가 제일좋고 빨래개키는건좋은데 제자리가져다넣는게 젤 싫어욯ㅎㅎ'], ['ㅋㅋ저도요 빨래 서랍에 갖다두는게 젤 귀찮~~^^'], ['식구들 많은데 일일히 개서 제자리 찾아놓는거 진짜 넘나 귀찮지않나요??\nㅜ ㅜ ㅜ ㅜ  ㅜ ㅜ ㅜ'], ['빨래랑 널고 개기 제일 싫어요 ㅠ'], ['밥하기 젤싫어요 ...특히 요즘같은땐  누가 밥좀차려줬음 좋겠어요ㅠ\n빨래랑 정리하는건 좋아요 ㅋ'], ['전 빨래개키기가 젤 싫어요\n좋은건 그나마 청소기돌리기요'], ['요리가 미치도록 싫어요ㅜㅜ 청소는 행복합니다~'], ['정답!🙋\u200d♀️🙋\u200d♀️🙋\u200d♀️욕실청소와 빨래 접기 입니다😭으헝'], ['아! 가장 싫은것! 접은 빨래 각각 온 방에 자리에 가져다놓기!!!!!!😬😬😬😬😬'], ['빨래개기 청소하기 설겆이하기\n그냥 다 요..\n요리는 유일하게 좋아요\n근데 요리는 그냥 제 인생에서 유일하게 잘 하는거라..요리하면 스트레스 풀려요..육아하느랴 요리를 못 하니까 스트레스네요\n악!!'], ['다싫어요. 그냥 내가 하고플때 음식만드는것만 좋고 다싫어요 ㅋㅋㅋ'], ['음..전에 살던 동네에서 아줌마들한테 난 집안일이 안 맞는거 같아 라고 했더니 맞아서 하는 사람이 어딨어?해야 하니깐 그냥 하는거지라고ㅋㅋㅋ전 다들 잘하고 잘맞고 좋아서 하는거라고 생각했는데 아니더라구요ㅠㅠ전 너무 적성에 안 맞아서 도우미분 도움 좀 받았더니 요샌 그나마 좀 나아요ㅋㅋㅋ집안일도 공부해야 하나봐요;;정리정돈 수업 몇개월 들으면 잘할수 있을것 같은데;;ㅋㅋ전 못해서 다 싫어요;;;ㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋ저도 집안일이 너무너무시러요 안맞아요\n ㅜ ㅜ ㅜ ㅜ ㅜㅋㅋㅋ'], ['빨래하기. 청소하기는 좋은데\n음쓰버리기. 음식하기는 싫어요.'], ['반찬만들기 싫고,설겆이 빨래 젤 좋아요'], ['이번 코로나로 인해 돌밥돌밥하면서 느꼈어요.\n설거지는 식기세척기에게 맞겨야되겠다고🤦\n2년뒤 새집으로 이사갈땐 꼭 식세기 장만하려구요 ㅜㅜㅜㅜㅜ'], ['청소빨래요 그나마 밥하고 설거지는 괘안아요'], ['좋은거는 없어요그나마 할만한건 설거지랑 요리싫은거는 청소>빨래개기>빨래널기최악은 싱크대하수구랑 화장실청소요분기마다하는 옷정리는 그냥 안하구싶구욧젤좋은건 아이러니하게도 깨끗한 집 식탁에 앉아 아이스아메리카노 마시며 폰보기요ㅋㅋ'], ['전 빨래개는거요,, ㅠㅠ'], ['좋은건 없고 ㅋㅋ 설거지가 제일 싫어요~ 식기세척기 사자고 신랑 꼬시는 중이예요 ㅎ'], ['책정리 빼고는 다 싫어요'], ['젤좋은건빨래,청소,설거지\n젤싫은건빨래걷고,요리하는등\n그외나머지집안일은더싫어요ㅋ'], ['음식만들기 좋아하는데 나만 먹는게 싫어요.'], ['빨래 너는게 귀찮아 건조기를 샀는데 건조된 빨래 개키지를 않아 쌓여있어요ㅋ']]
    
    6811
    집안일 중에 뭐가 젤 하기 싫으세요?ㅋㅋ 전 빨래에요!하는 것도 싫고 개는 것도 싫어요ㅋㅋ세탁실에 분리수거함이 있어서일까요..들어가기 정말 싫어요ㅋㅋ
    
    [['설거지요..거지같애요진짜'], ['진짜 왜 다하고 돌아서면 또 쌓여있을까요ㅋㅋ'], ['설거지요 진짜 시룸ㅠ 짱시룸ㅠ'], ['저두 설거지요~ ㅋㅋ'], ['청소요..'], ['저도 두번째가 청소 ㅠㅜ'], ['전.. 집안일 다 괜찮은데 요리가 하기싫어요😭ㅋㅋㅋ'], ['저두 요리하기 싫었는데 요새 재미붙이고 나니 조금 낫네요ㅎㅎ'], ['저도설거지가 제일싫어요 ㅠㅠ'], ['밥 설거지 빨래 다 하기 싫어요~^^;;'], ['전모두다 하기싫어여ㅜㅜ'], ['화장실청소가 하기 젤 싫어요'], ['화장실청소는 안하고 살아서 잊고 있었네요..ㅋㅋ 한달에한번하는듯해요ㅋ'], ['다 싫어요ㅠㅠ처녀때로 돌아가고 싶어요'], ['설거지가 왜케 하기싫죠 ㅠㅠ'], ['설거지요! 아..진짜너무싫어요ㅜㅜㅜㅜ'], ['요리 청소요 특히 걸레질이요 무릎나갈거 같아요 ㅠ'], ['걸레질 머리카락 엉퀸거 너무 싫어요ㅠㅜ'], ['다 싫은데 설거지가 젤 싫어요 ㅠ'], ['다~아'], ['청소요.. 특히 걸레질하는게 왜 그리 귀찮은지요;;;'], ['걸레 빨기 싫어서 버린게 몇개나 되네요ㅋㅋ 세면대 막힐까봐 쪼그려앉는거 힘들어요'], ['설거지랑 빨래개기요\n진짜 돌아서면 쌓이는ㅜㅜ'], ['설겆이요....ㅜ ㅜ 특히 반찬뚜껑 씻는거요'], ['저도 설거지요 ㅜㅜ'], ['반찬이요ㅜ'], ['전 메인은 하되 잔잔바리 반찬들은 아직 사먹어요ㅋㅋ'], ['저두 반찬 사먹었는데 주인분외 보조들도 마스크없이 일하고 계셔서. .'], ['헉 코로나를 떠나서 위생상 하셔야하는데..'], ['전 청소가 넘 싫어요....'], ['설거지랑 빨래개기요ㅠㅠ'], ['다하기싫지만 전 청소요ㅜ 청소를 너무 못해요'], ['빨래 개는거요... 그나마 빨래는 세탁기가 건조는 건조기가 해주지만 빨래 개는건 왜 제가 꼭해야할까요ㅋㅋㅋㄱㅋㅋㅋㅋㅋ 누가 빨리 기계좀 만들어줬음좋겠어요'], ['저도요ㅋㅋㅋ빨래개는게 젤 시러요ㅠ 저도 맨날 기계소리하는데 ㅋㅋ기계나오면 줄서서라도 살려구요ㅋㅋㅋㅋ'], ['저두요ㅋㅋㅋㄱㄱ 제가1번으로살겁니다!!!'], ['설거지요....정말싫어요~~~ㅠㅠ'], ['걸레빨기..,ㅜㅜ이것때믄에 물티슈로 방닦아서 엄청써요'], ['저두 3M꺼 써봤는데 화학냄새가ㅜㅠ혹시 추천해주실만한거 있을까요?'], ['빨래개기랑 설거지요..해도해도 싫어유..그나마 설거지는 용돈줘가며 가끔씩 큰애들 시키는데 빨래개는건..진심 귀찮'], ['설거지요 ㅠㅠ 밥 먹고 나도 쉬어야지~~ 하면서 쉬다가 싱크대 쳐다보면 정말 한숨 나와요😫'], ['전걸래빠는거요......'], ['전 청소요...'], ['빨래 개고 난뒤 제자리 갖다놓기요 젤싫음ㅋㅋㅋㅋ'], ['다 싫어요 ㅋㅋㅋ'], ['다요...ㅠ'], ['전 빨래 개키는거요 ㅠㅠ\n'], ['빨래개고 갖다놓은거요ㅎㅎㅎ'], ['그냥 다 싫어요.............'], ['설거지요....'], ['전 옷정리 하는거요..ㅡㅡ'], ['젖병이랑 빨대컵 씻는거 젤 귀찮아요 부속품 많고 입구좁고 빨대는 또 빨대브러쉬로 씩어줘야되고요'], ['설거지 젤 싫어요. 전 빨래는 세탁기가 해준거 널고 개는건 좋아해요 ㅋㅋ'], ['설거지랑 화장실청소요~~'], ['장난감정리하기요ㅜㅜ'], ['설거지랑 빨래개는거용 ㅜㅠㅜㅜㅜ'], ['요리요']]
    
    6925
    내가 가장 좋아하는 집안일은? 아까 나갔다와서 집안일이 밀렸는데 낼 아침 눈뜨면 싹 다 치워야 할것같아요그러다 생각이 난 건데 우리 포수방 맘님들은 어떤 집안일을 가장 좋아하는지 리서치해볼까요^^참고로 저는 설겆이가 가장 좋아요거품 퐁퐁 나는것도 좋고 시원한 물에 손 담그고 있는것도 좋고 깨끗해진 싱크대 반짝거리는거 보면 속이 뻥 뚫리는것같아요~^^
    
    [['전 다 싫어요ㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋ\n다 싫다하시는데 어쩐지 닉네임이 우아하게 느껴집니다ㅋㅋㅋㅋㅋ'], ['전..음..모든 엄마들의 마음이랑 똑같습니다ㅜㅜ'], ['ㅋㅋㅋㅋㅋㅋ 전부 조선시대 마님으로 태어날 팔자였네요ㅋㅋㅋ'], ['악..모두 저와 같은 맘일줄이야..'], ['현재 투표1위ㅋㅋ 이마이 다 싫다 하실줄이야ㅋㅋㅋㅋ'], ['다싫다 누지름용 ㅋㄱㅋㅋㅋㅋ'], ['에라이~~~집안일이고 나발이고ㅋㅋㅋㅋㅋㅋㅋ'], ['다실어용ㅋㅋ\n요듬은더더욱이용ㅋㅋ'], ['옛쏘! 자유시간 24시간 쿠폰 들고 산이며 바다며 떠나시오!!!!!!\n\n하고프네요~~😭😭😭😭'], ['ㅋㅋㅋ\n그림의 떡...ㅜㅜ힝힝'], ['사실은 나도 갖고 싶은 헝헝헝'], ['전 요즘 ㅎㅎ로봇하나만있얺음좋긋네융ㅎ'], ['다좋다 꾹~~~눌럿다요ㅋㅋㅋ\n피할수없으면 즐겨라ㅋㅋㅋ🤣🤣🤣'], ['다 좋다 누르신 최초이자 최후일것같은 쩡이맘님 모시겠습니다~'], ['정녕 나뿐일까요???움뫄~~~😂🤣😂🤣'], ['쩡이님 이제부터 존경합니다!! 피할 수없음 정신 놓아뿌까 하는지라ㅎ \n나는 다 싫다인데 ..\n혹시 육아도 들어가있으면 좋다에 투표하나요?^^'], ['그래도 때론 정신줄 혼미할때도 있음뫼다~~ㅋㅋ\n그래도 어차피 내가해야할 몫이고 내가아님 누가하리  심정으로 그런데말입니다...육아는 정말 힘듭니다 이또한 내몫이다 생각하려고 노력 또 노력중이지요~~😭😭'], ['고민도 안하고 다싫다~~~에 눌렀어요 ㅋㅋㅋㅋㅋㅋㅋㅋ 전 게을러서요 ㅠㅠㅠㅠ'], ['우리가 결혼을 하고 엄마가 되면 이 모든것들이 +a로 양이 배가 되고 오롯이 내몫이라는걸 왜 학교에서는 가르쳐주지 않았을까요ㅠㅠ'], ['ㅠㅠ 맞아요 좀 더 결혼과 육아에 대한 현실적 교육이 필요하다고 봅니다.. 근데 그러면 아무도 결혼 안하지 않을까요?! ㅋㅋㅋ'], ['아니 그렇게 깊은 뜻이? (서경석버전)\n학교 가정시간에 사랑하는 남자를 고르면 골라 결혼하는 동시에 집안일폭탄은 내 몫이다 했음 진짜 아무도 안했을것같아요 그냥 연애나 했지 싶어요'], ['다좋다 누구실지 궁금해요 저 존경할려구요^^'], ['쩌~ 위에 피할수없으면 즐겨라 명언을 남기신 쩡이맘님께 박수를ㅋㅋㅋ🙌🙌🙌🙌'], ['난 울딸램 힙시트에 앉히고 청소기돌리고 놀아가며 걸레질하고 그럽니다ㅜㅜ🤣🤣🤣'], ['진정 즐길줄 아는 쩡이님 win👍👍\n(우리집 초청하고프다요ㅎㅎ)'], ['읍사무소 다라이받으러 갈깝쇼~~이번에도~ㅋㅋㅋㄱㄲㅋㅋㅋㅋㅋㅋㅋ😄😄😄😄😄'], ['다라이에 바가지 얹혀 드리겠습니다!!^^ 존경존경!! \n애본다고 집안일 못한다ㅎ 육아는 애들은 방목이다하고 풀어놓고 ㅍㅎㅎ'], ['에잇~저보다 다들 잘 하시잖아요~~ㅋㅋ'], ['이번에 가심 먼저꺼까지 두개 달라카이소ㅋㅋㅋㅋㅋㅋ'], ['없어요. ㅜㅜ\n다 싫어요.ㅋㅋㅋ'], ['다 싫은데 다 해야하는 현실이 더 서글프네요😭😭'], ['다 싫긴한데 그래도 고르라면 설겆이 좋네요~~ ㅋ'], ['그죠 써니님~\n저도 개중엔 설겆이 ^^\n전 설겆이는 보람을 느껴서 좋아요ㅎㅎㅎ'], ['걍 집에서 숨만 쉬고 싶다고 저 아랫마을에 사는 분이 전해 달래요~~ ㅋㅋ'], ['격하게 아무것도 안하고 싶은 맘 윗마을까지 찌릿찌릿 전해오는것 같습니다'], ['이제 눈 감고 숨만 쉴려구요~~~\n굿밤 되셔요^^'], ['전 다싫다에 한표했어요 아 애들 신발빠는거 하나는 보람느껴요 ㅋㅋㄱㅋ'], ['헛 신발!!!! \n저희집 많이 있는디 원하심 가져다 드림당 😅😅😅'], ['헛...잠시만요.......ㅋㅋㅋㅋ다시 생각해볼게요ㅋㅋㅋ'], ['압도적인데요ㅎㅎ저는 요즘 설거지가 넘 싫으요먹은건 별로 없는디 그릇은 와그리 많은동~~뜨신물에 한번 찬물로 한번더 이중으로 씻으니 서 있는동안 허리 아파요 안그러고 싶은데 습관이 되나서 대충이 안되네요ㅎㅎ'], ['저는 첨부터 끝까지 다 찬물로 해요ㅎㅎㅎ\n찬물로만 한번해도 요래 안죽고 살아있어요ㅋㅋㅋ\n습관 바꾸기 한번 도전해보셔용'], ['압도적인 결과였네요ㅋㅋ저의마음이 포수방 맘님들 마음이군요ㅋㅋ'], ['이래 압도적은 예상치 못한 결과예요\n저 아래 처박힌듯한 청소 빨래 쫌 측은해보여요ㅋ'], ['다싫음~~~요\n건조기사고 빨래널기해방인줄 알았더니 \n식기건조기.음식물건조기 다 하고싶은데.....저만그런가싶네요^^;;'], ['그래서 인간을 간사한 동물이라 하잖아요ㅋㅋㅋㅋ\n식세기 사도 식세기 돌리기도  귀찮지 싶어요ㅋㅋㅋㅋ'], ['저도 다 싫어요 .... ㅋㅋㅋㅋㅋ 하기싫어용 ㅠㅠ'], ['집안일들은 열표도 안 나오네요ㅋㅋㅋㅋ\n다같이 한마음 한뜻이네요ㅋ'], ['저도 없어요 업어 다 싫어 그냥 아 그냥 아무것도 하기가 싫어요 ㅠ'], ['정말 홍남매님 상황 아무것도 안하고 싶은거 격공이예요ㅠ'], ['헉..다싫어요 그나마 그중에 하나 굳이 고르라고 한다면 빨래돌리기요ㅋㅋㅋㅋ 그냥 넣고 끝ㅋㅋㅋ'], ['전 빨래돌리기까진 괜찮은데 널고 개고 넣고ㅠㅠ'], ['맞아요 늠나시러요 특히 개고 정리하는것도 일이에요😭'], ['다 싫다 인데 ㅋㅋ 그중 고르라면 빨래 ?? 세탁기에 집어넣음 알아서 빨아지고 빨아진거 건조기에 넣으면 알아서 말라서 나오고 ㅎ 쓰고나니 게으름이 티나네요 ㅋㅋㅋ'], ['건조기가 있어서 부럽슴당^^\n게으르다 정도로 얘기할꺼면 개고 넣는것도 귀찮다 해야해요ㅋㅋㅋㅋ'], ['아 ㅋㅋ빵터짐요 저도 다싫다 선택햇는데 다싫다 압도적입니다 다 한마음 아니겟습니까 ㅋㅋ'], ['ㅋㅋㅋㅋㅋ\n이마이 다 한마음 한뜻인지 몰랐어요ㅋㅋ']]
    
    6927
    집안일..뭐가.제일 싫으세요?? 저는..설거지빨래분리해서 세탁하기..꺼내서 접고 서랍에넣기..청소기돌리기..이불접기..먼지닦기..전부요........어제도 쉬었는데 오늘 증말 아무것도 하기싫으네요   ㅠㅠ일도힘들고 집안일도힘들고..애들 밥겨우주고 온라인켜주고 쇼파에 계속누워있어요집이 난리났는데 언제움직일수있을까요?😭
    
    [['빨래요'], ['빨래개기랑 남편 셔츠다리기요ㅜ'], ['김치냉장고청소요ㅋ'], ['상추씻기요'], ['존경합니다 ㅠㅠ'], ['다요ㅡ'], ['모두... 얼른 가정에도 스마트시대가 왔으면 좋겠어요'], ['빨개 개는거요 ㅋㅋㅋ'], ['화장실 청소요'], ['222'], ['333333'], ['4444'], ['55555'], ['걸레 빠는거요 전 이게 정말 싫어서 덩달아 먼지 청소도 잘 안하게 되네요'], ['빨래널기여'], ['요리한 다음에 치우는 거요,, 저 초보라 여기 저기 재료, 수저, 작은 그릇들이 조리대 전체에 펼쳐져 있어요ㅋㅋ큐ㅜㅜ'], ['저는 빨래개는거요..ㅋㅋ ㅠㅜ 집안일 참 열심히 한거 같아도 뭔가 그대로인거 같고 참 힘드네욬ㅋ ㅜㅜ'], ['청소~'], ['다 싫지만 ㅋㅋ 그중 전 빨래 정리가 젤 싫어요^^과정도 많고 시간도 젤 많이 걸리고ㅜ.ㅜ'], ['반찬하는거요.  솜씨도 없으니 더 힘들어요.'], ['밥차리고 설겆이요 ㅋㅋㅋ'], ['설거지요.ㅋㅋ'], ['전 빨래 개놓은거 서랍에 정리해서 넣는거랑 화장실청소ㅡㅡ.너무 싫어요'], ['다림질이요ㅜ'], ['저도 다 싫은데 젤 싫은거 고르라고하면..ㅋㅋㅋ 빨래 접는거 젤 싫어요...접어다 가져다 놓는거.ㅋㅋㅋㅋ 이상하게 싫어요...'], ['설거지와 화장실청소요~~ 해도해도 티도 안나고 넘 힘들어요 ㅠ'], ['실파까기요'], ['왜 다 싫죠ㅠㅠㅠㅠㅠㅠ친구가 저에게 처음 건조기 샀을때.. "이제 너도 건조기에서 옷 꺼내입을껄?" 이라고 하더라고요ㅠㅠ설마...했는데 얼마후 그러고 있는 저를 발견했죠ㅠㅠ'], ['욕실청소여'], ['다싫은데 빨래개키는거 정리하는거 넘짜증이여 ㅡㅡ'], ['음식물쓰레기처리 너무 싫어요..ㅠㅠ갈아버리는거 사고 싶은데 쓰는분들 만족도가 그닥 높지 않은 경우도 있어서 고민만 하다 여름이 됐어요ㅠㅠ'], ['빨래개고 제자리에 놓는것ㅡㅡ'], ['빨래개는거랑 정리요~;;;넘나 귀찮아요;;;'], ['다림질;;;'], ['방닦기요ㅠ'], ['요리하기, 욕실청소요'], ['설거지랑 빨래 개는거요ㅠㅠ 젤 하기 싫어요'], ['화장실 청소요~~'], ['화장실청소. 쓰레기버리기 넘나 싫어요. ㅜㅜ'], ['냉장고 화장실 청소요 ㅜㅜㅜㅜㅜㅜ '], ['전 아기장난감 정리요 ㅠ\n정리하면 꺼내고 정리하면 꺼내고 끝이 없어요 ㅠㅠ'], ['아이가 어질러놓은 책 물건 장난감정리요 ㅠㅠ 초등2인데도 잘 못치우네요'], ['설거지.. 싫어서 밥도잘안먹게되요 ㅋㅋ'], ['다 싫지만 밥하는거요....'], ['요리와 설거지요ㅎㅎ \n아니 요리만요, 누가   매끼 요리만 해줘도 ...'], ['짐정리요하나씩 사재기하면서 쌓아두다보니 방하나가 꽉찼써요ㅠㅜ정리해야하는데 하기싫어 죽겠써요정리할 방법은 이사 밖에 없는건가 싶어요'], ['설거지요'], ['음식물 쓰레기 정리요.욕실청소랑'], ['설거지....옥...... ㅠㅠ'], ['양말널기요.ㅜㅜ'], ['ㅋ ㅋ 가끔씩 베란다 바닥에 늘어놨다가 마르면 바닥에앉아서 정리에요 ㅋ ㅋ'], ['다.싫긴하지만 빨래개는거요. \n늘긴까지는 괜찮은데. .\n아~넘.빨리 마르는것같아요~ ㅋ ㅋ\n'], ['바닥 닦기도 추가요~ ㅋ ㅋ'], ['전부다요..ㅠ..아..시르다요 진짜....;;;\n매일 먹어야하는것도 싫고\n매일 새롭게 쌓이는 먼지도 싫고\n매일 마르는빨래.ㅋㅋㅋ도 싫고..\n굴레예요 굴레.'], ['저는 청소가 제일 싫어요ㅠㅠ'], ['전 화장실 청소요'], ['빨래가 많네요~~\n그래서 건조기가 인기있나보군요~\n전 청소가 제~~~~일 싫어요~--;;'], ['설거지요... 진짜 너무 싫어요. 하루 몇번을 해도 금방 잔뜩 나오고 ㅠ ㅜ 다른청소는 하면 기분이 좋아지는데 설거지는 해도 해도 나오니 다 해놔도 홀가분한 기븐이 안들어요'], ['청소가 젤로 싫어요'], ['설거지요 ㅜㅜ 젤싫어요'], ['저는 끼니챙기는거요~저는 식욕도 그닥 없고 미식가도 아니라 배 채우기만 하면 되는데 애들이랑 남편은 구색 갖춰 차려줘야하니 매일 스트레스예요ㅜㅠ'], ['저도 음식이요ㅠ 다른 건 다하겠는데 뭐먹을까 뭐만들어야하나 냉장고에 뭐상한건 없나 뭘더사야하나 제일 싫어요'], ['설거지요ㅜㅜ아진짜 바로바로 닦은기억이없어요ㅋㅋㅋㅋㄱㅋ'], ['다지만. ..그중 빨래개기요'], ['청소요 ㅋㅋㅋ'], ['저두 전부요 ㅠㅠ  왜케 싫을까요?'], ['밥하기가 젤 싫어요']]
    
    6970
    집안일 하기 좋은 날 약속없는 주말은집안일 하기 좋은 주말이렇게 볕이 좋으면이불빨래가 하고 싶어져요일어나자마자이불과 베게커버 세탁기행평소 출근길에로봇청소기만 열일 시키고 나가요꽤 청소를 잘하는 친구지만구석구석은 못하기에청소기들고 코너부터 창틀까지밀대처럼 서서 슉슉 미는 물걸레로전체적으로 닦고나니배가 고픕니당컨닝을 좀 할까하다 실패하고선간단히? 김밥말쟈오늘은 두줄만!호다닥 마트로 가요김밥재료와 주말 먹거리들좋아하는 노래 틀어두고흥얼흥얼 따라부르며돌돌돌 김밥을 말아요김밥 먹었으니 후식인 메론도 썰면서 끄트머리 두개는 입속으로메론 먹다가새싹메론 사진을 찍어봅니다정말 쑥쑥 자라요메론이 넝쿨과 식물이라울타리?가 필요하다네요울타리 어디가서 사야하나...앞으로 남은 집안 일욕실청소수건 삶기세탁소에 다녀오기집안 일하기 딱 좋은 날입니다지금은뱅갈고무나무 수형다듬기중묭실 온것처럼 예쁘게 해줄께요 고갱님♡그래도 시간이 남으면드라이브 다녀올까
    
    [['맛있는 김밥 꼬다리'], ['찌뽕김밥 꼬다리 넘 좋아요재료들이 더 많이 들어있어서 ㅋㅋ'], ['치타님~ 즐건 주말입니당데이트 가셨나요?^^'], ['아하~ 담주..시간도 많은데 두분 데이또 하는거 몰래 구경갈까요? ㅋㅋ'], ['음.. 그렇다면 교대운전 가능한 파티원 두분 더 섭외해야겠어요 ㅋㅋㅋㅋ'], ['ㅋㅋㅋ\n두자리 남았어요\n선착순 아니고 면접과 도로주행 셤이 있습니다\n마니마니 신청해주세용. ㅋㅋ'], ['꼭꼭꼭 눌러서 말았더니 모양이 예쁘게 잡혔나봐요 :)냥이집사님도 즐주말 중이시지요?'], ['정우빠님 부산에 계시지 않으셨어요?'], ['뭔가 신기한 나무네요\n산으로 바다로 부러워요\n가족들과 즐거운 시간 보내세요~~'], ['선물..  설레는 단어네요\n고민해보겠습니다 ^^'], ['\n굿 애프터눈^^\n\n오와오와..\n\n청소에 요리에 과일도\n잘깍으시고 거기에..\n식물키우기까지!!\n\n조강지처가 될 상인데요?!\n\n남자사람 한명만 곁으로 오면 ㅎ'], ['청소는 청소기가 ㅋㅋ\n과일은 술안주로 많이 먹어봐서요 ㅎㅎ\n\n아직 집안일이 남아서\n끝내면 늘어져서 쉬려구요\n\n즐주말입니다!'], ['\n그래요~\n열심히 마무리하시고\n푹쉬면서 주말을 즐기셔요'], ['저는 이제 반죽 숙성시키고 있어요~\n\n배고파요~~ ㅠㅠ\n'], ['설마 아직 공복이예요??\n쉬성되는 반죽에서 깊은 내공이 보여요\n쫀득한 수제비\n호박이랑 감자 많이 넣어주세요!'], ['저거 감자전분 들어간거라서 감자빼고 호박만 넣을거에요 ^^\n'], ['아~~~\n그럼 호박 마니마니!'], ['그리고 저 내공 없어요 ㅋㅋㅋㅋ\n\n레시피 보고 첨 만들어요 ㅋㅋㅋ\n'], ['비닐에 꽁꽁 싸둔게 깊은 내공이 있어보였는데 ㅋㅋㅋ\n맛나게 되길 응원합니다!'], ['맛나게 먹었습니다~^^\n'], ['벌써 다 드셨어요?\n유월님 글보러 출동~'], ['김밥 진짜 맛나보여요 ㅋㅋㅋ'], ['시장이 반찬이라\n배고플때 먹었더니 맛나더라고용 ㅋㅋ\n주말인데 좋은 계획 있으세요?'], ['일어난지 2시간됐는데 또 잠오네여ㅋㅋㅋ'], ['그땐 과감하게 또 잡니당~'], ['오.... 메론새싹 엄청 컸네요☺️\n도로시님 김밥은 유난히 예뻐요!\n집안일 후다닥 끝내시고 오후 시간엔 푹 쉬세요🙂'], ['새싹들은 볼때마다 놀라요!\n너무 이쁘고 잘자라서 ㅎㅎ\n파파야향기님 오디 많이 따셨어요?\n엄청 잼나겠어요!'], ['전 체험하는 사람들 안내해주는데 안내해주면서 앉아서 오디 동생이 따주는거 먹는데 손톱 어케요 ㅠㅠ\n완전 지지 됐어요 흣\n\n새참 먹고 있어요 ㅋㅋㅋㅋ'], ['앜ㅋㅋㅋ누가 물으면 프랜치 네일이라고해요 ㅎㅎ아까 빵도 좀 사올껄 그랬나..'], ['ㅋㅋㅋㅋ 프렌치네일 \n이왕이면 딥프렌치^^'], ['힝... 손톱 어쩔 ㅠㅠ\n엄지랑 검지만 써서 먹었더니 왼손 오른손 엄지 검지 난리예요🙊'], ['좀 많이 지지니 딥프렌치로 ㅋㅋㅋ'], ['ㅋㅋ 크리미빵 추억돈내ㅋㅋㅋ'], ['완전 순삭했어요 ㅋㅋㅋ'], ['오랜만에 먹었더니 살살 녹더라고요ㅋㅋ'], ['오디가 유독 그런거 같아요입도 까매지기 때문에데이트할땐 그거 먹고 웃으면 안돼요 ㅋㅋ'], ['아까 동생이 제 입술보면서 넘 푸르스름해서 창백하다며.. 슬며시 립글로스를 내밀었어요 ㅎㅎㅎ'], ['아무리 잘 말아도 도로시님 김밥은 따라갈수가 없내요~\n김밥여신 도로시님^^'], ['김밥여신 ㅋㅋㅋㅋ친구분과 즐건시간 중이세요?'], ['아뇨 저녁에나 올거같아요\n미용실  뿌염가능한지 저나해보려고요^^'], ['뿌염하고 계세요?전 집안일 목표달성하고집근처 등산중이예요머리는 복잡하고어제 고기 먹어서 기운은 넘치고 ㅠ'], ['아니 예약끝났대요^^'], ['엄청 부지런 하십니다 ~~~\n오늘 열일 했으니 내일은 좀 더 푹 ~~ 쉬세요 ~~'], ['집안 일 끝내고나면숙제 끝낸것처럼 후련해요 ㅎㅎ블루엠제이님도 편안한 주말중이시죠? ^^'], ['김밥 최고요ㅎ'], ['김밥은 언제 먹어도 맛나용♡'], ['김밥도 예쁘게 마시는 도로시님!!\n주말엔 대청소죠✋ 저도 청소좋아하는데.. 집에 비글두마리가 너무 어질러서🤪🤪🤪포기각 ...'], ['저는 청소 그닥 안좋아하는데주말엔 꼭 해야한다는 강박관념이 있나봐요 ㅋㅋ귀요미들이 어질른다니 용서되고 막 ♡♡♡'], ['김밥도~~^^참 맛나 보입니다.^^메론도~어쩜 이리...잘 키우셨는지...주렁 주렁 열릴것 같아요.^^전엔~식물 깍아주는것 엄두도 못 냈는데~지금 참 잘 해줍니다. 신기하게~~머리카락 자라듯~참 잘 자라더군요.^^  더 예뻐지고 숱도 많아 지는~~^^'], ['가인님 식물사랑은 제가 잘알죠♡♡\n벤자민도 건강하게 잘키우셨어요'], ['집에서 에어컨 틀고 뒹굴기 딱 좋은 날이예요. 이러다 한숨 자고 일어나려고요,ㅎㅎ'], ['딩굴딩굴 하기 좋은 날,\n저는 몸을 좀 써야겠어서 혹사시키고 있답니다~~'], ['저도 격하게 몸 뚱이를 써야 하는데 말이지요...ㅎㅎ'], ['김밥 맛나게드셨어요? ㅎㅎ'], ['야채 많이 든 김밥을 좋아해서 취향대로 당근이랑 부추 듬뿍 넣었어요 ㅎㅎ\n가달님도 짝꿍님과 즐거운 주말 보내고 계세요?'], ['아 부추인가요? 시금치인줄 ㅎㅎ'], ['저는 부추 데쳐서 넣는 김밥이 더 맛있더라구요 히히'], ['오~좋네요 손도 덜갈꺼같고~ ㅎㅎ'], ['아이코 김밥 예쁘게도 마셨네용😋저도 실은 얼마전 김밥 만들기 한번 도전해 봤는데요ㅋㅋㅋㅋ 진짜 지옥에서 온 김밥인줄...😔오늘도 부지런하신 로시님께 한 수 배웁니당😌'], ['저도 첨엔 그랬어요 ㅋㅋㅋ\n터지고 처참한 광경 하하\n사실 그닥 부지런하지 않아요\n약속없으니 하는거 ^^\n우루사님도 주말 잘보내고 계시죠?'], ['김밥이 알차네요\n못하는게 없으시네요~\n드라이브 무사히 다녀오셨을까요 :)'], ['드라이브 대신 등산을 택했답니다\n집근처 너무 힘들지 않는 산에 다녀왔어요\n초록이들 보며 땀 흘리고 왔더니 힐링 그 자체 :)\n\n오늘 즐겁게 보내셨어요?'], ['등산이 더 좋은 선택인거 같아요\n잘 다녀오셨네요\n땀도 한 줄기 흘리셨다면 운동도 충분히 되신거 같아요\n초록초록이 주는 힐링은 언제나 좋죠 :)\n\n저는 소소하지만 은은한 행복감이 있는 하루였어요\n커피와 함께 드라이브도 다녀오고\n밤산책도 다녀오고 😄'], ['도로시님 굿모닝요.ㅎㅎ\n\n저 너ㅡ무 늦게 왔죠?ㅎㅎ\n\n어제 아콩이 끌고 다니느라 정신이 없었네요.ㅎㅎ\n'], ['아콩이와 항상 안전운전 하시길 바래요컬러도 디자인도 너무 이뻤어요♡'], ['\nㅎㅎㅎㅎ\n\n안전운전할게용♡\n']]
    
    7112
    이도저도 소득없이 바쁜하루~ 아침에 공홈에서 리밋 주문하고 카폐보니 광교점 오픈행사 하길래 민탱님 덧글보며 고민좀 하다가 후다닥 갔는데 줄이줄이ㅋㅋ12시 도착했는데 257번이래요ㅋㅋㅋ이 표 가지고 일주일안에 오면 오픈 사은품까지 준다기에 담주 다른 리밋 소식도 있길래 한번에 겟해야겠다는 큰 그림을 그렸는데...망했어요ㅋㅋ출고했대요ㅋㅋ이왕 망했으니 점심은 묵어야쥬~올만에 큰아들과 둘이 데뚜~밀크티 안좋아했는데 지난번 비올레따님이 보내주신 밀크티 맛보고 뿅가서 밀크티도 겟해서 왔네요ㅎ오늘은 밀크티랑 아까 사온 빵이랑해서 저녁먹어야겠어요ㅎ집안일 다 팽개치고 나갔다왔는데...설겆이 산더미에 이 몸은 귀차니즘...좀 쉬었다 스피드하게 집안일 해야겠어요ㅋ
    
    [['ㅎㅎ\n원래 일이 그리되서 집에오면 만사가 다 귀찮아지죠~^^\n맛있는 빵 드시고 업업하세요~^^'], ['진짜 암것도 하기 시르네요ㅋㅋ'], ['민탱님 가시고 제가 도착했군요~\n제가 50분에 도착했는데ㅋㅋ'], ['아~우째요~ㅠㅠ\n\n담주에 또 갈라했것만 이미 캡슐이 출고 완료뜨고~\n넘나 슬퍼요~'], ['나폴레옹빵 너무 맛잇는데 광교 갤러리아에 생겼나봐용!'], ['지하에 입점했더라구요~\n더 집어 올려다가 담주에 또 갈거니 워워하고 왔는데 큰그림 실패로ㅋㅋ\n걍 있는것만 먹어야겠어요ㅋ'], ['저도 11시쯤 갔다가 11시45분쯤 나왔는데, 번호표 있음 오픈사은품 준다니..  왕복택시비랑 시간이 아깝네요..ㅠ 번호표도 어떻게 하는건지 미리 얘기를 해주던가 대응이 좀 짜증나요..  ㅠ'], ['전 저오기 전부터 나눠준지 알았는데 아니군요~\n에혀~\n오셨다 그냥 헛걸음 하셔서 어떻해요~?\n\n저도 주차비 안낼려고 2만원 쓰고 왔어요~\n어차피 살거였지만 그래도 좀 속상하긴 하드라고요~'], ['제가 막 무리를 이탈할때 번호표를 준비하고 있다는 얘기가 들리긴했는데요ㅋ 그게 갖고있다 오늘 와야되는 번호표인지(식당대기번호처럼..) 느긋하게 일주일안에 오면 되는 번호표인지는 얘기가 없었어요. 그래서 번호표 들고 계속 갤러리아 있을 상황이 아니라서 나온건데 나오면서 보니 부티크 안에서 다들 넘나 느긋하심... 직원한분이라도 그냥바로 계산할사람 응대해주셨음 좋았을뻔했어요..얼음트레이 다시사려다 겸사겸사 간건데교훈으로삼고 초록창검색해서 손잡이 내열유리 큰머그랑 얼음트레이 살거예요!토닥토닥해주시니 더울컥해지네요ㅠ'], ['아~\n이거 완전 무슨상황인지 알것 같아요~\n\n저도 고게 뭐라고 이라고 있는지...\n이람서 왔는데...\n속상하셨겠어요~\n\n혹시 담주에 가실생각 있으심 번호표 우편으로 보내드릴까요~?\n\n전 이미 공홈에서 출고 완료가 되서 소용이 없어져서ㅠㅠ\n\n담주에 가서 사먹으려고 달고나 스콘도 안사왔는데...\n슬프네용ㅠㅠ'], ['ㅋㅋ고생해서 받아오신건데 이제갓건조된 나부랭이가 달라고 떼쓸만큼 염치가없진 못해용~~달고나스콘 사서 택배로 보내드릴까싶어 봤더니 요즘날씨에 택배가서 맛있을 빵도 아니고요ㅎㅎ어서어서 즐거운 다음주일정 잡으세요!'], ['전 아마도 안가지 않을까 싶어요...\n이미 출고가 됐는데 가면 지름신만 더 오시게 되니 그냥 공홈에서 보내주는 캡슐이 집에서 마구마구 뚫을거예요ㅋㅋ'], ['빵이 맛있어보여요~~~^^ 차가운 아아랑 잘 맞을것 같아요 ㅎㅎㅎㅎ저도 리밋 구매했어요 조금만 ㅎㅎㅎㅎㅎ'], ['저도 조금만 사야지 했는데 그노매 사은품이 먼지~\n어떻게 다 해결할지 벌써 걱정이예요ㅋㅋ'], ['전 눈 꽉 감았어요 ㅎㅎㅎㅎ여름내내 시원허게 드시면 ㅎㅎㅎ'], ['더 이상 캡슐이 고만사자하고  눈 딱 감고 결제 했는데 담주에 또다른 리밋이ㅋㅋ\n이건 뭐 날리도 이런 날리가 아니네요ㅋㅋ'], ['담주에 또 달려야 하나요 ㅎㅎㅎㅎㅎ'], ['45분?50분 거리라...\n민탱님도 만날겸 바람쐬고 올까? 막또 그러고 있는데ㅠㅠ\n에혀~\n모르겠네욤ㅋㅋ'], ['겸사겸사~~~좋은 님도 만나고 바람도 쐬고 커피도 겟하고 ~🤣🤣🤣'], ['앗~!!제가 한번 움직이면 1석3조군요ㅎㅎ'], [''], ['핸펀 AS를 가야겠네요...\n바꾼 핸펀이 GPS가 안잡혀서 네비 기능을 못하네요~\n지난번 라초비님 만나러갈때도 길 헤맸는데 오늘도 동승자 없었음 고속도로에서 울뻔했어요~'], ['허걱!!!! 폰부터 손보고 담주에 고고고고~~~'], ['낼 AS센터 가봐야겠떠요ㅠㅠ'], ['어머...저 정도인가요?\n전 시도도 하지 못할 광경😟\n\n고생하셨어요~\n맛난빵 드시고 좀 쉬셔용'], ['줄이 장난 아니였는데 오다보니 안쪽으로 숨은 줄이 또 있더라구요~\n포기 하길 잘했다 했어요~\n\n빵은...\n메몽님 나눔에서 탈락하고 급하게 검색해서 간김에 사왔네요ㅎㅎ'], ['헐......줄서서 기둘렸다 번호표도 받아야 하는 상황인가요?? 저걸 사용하려면 담주에 또 가야하는거구요???  ㅠㅠ 토욜오전에 잠깐 다녀올까 했는데......흠......고민되네요.....그냥 공홈에서 주문하는게 이득일런지......에효......'], ['오늘 첫날이라 더 몰려서 그럴지도 몰라요~\n주말에 쉬엄쉬엄 한번 다녀오시는것도~^^'], ['전 10시50분에 ㅠ 줄보고 너무 놀라서 다시 돌아왔어요 ㅠ 줄을 설 엄두가 안나더라고요~ ㅠ 날씨까지 흐릿흐릿 기운빠지는 날이네요~'], ['오늘 그냥 오신분들 많으시군요~\n저도 줄보고 순간 헉~  했어요~\n근데 오늘 리밋도 그렇고 오픈 첫날이라 더 그런것 같기도하고 그런거 같아요~'], ['어머어머~부띡 소식은 슬푸지만 ㅜㅜ\n저 밀크티와 통큰 빵들은 👍🏻👍🏻칭구님 통 크셔서 맘에 든다요~❣️ 저 빵들....음식들....\n울 큰 아드님과 데이트도 잘 하시고 보기 좋아요~~😘'], ['내가 담주에 가믄 꼭 사먹어야지 하고 봐둔 스콘이 있는데 담주에 귀차나서 갈란지 몰것네요~\n그거 보면서 젤루친구님 생각했는데.....\n뿌띡 나빠요ㅠㅠ'], ['헉! 저도 스콘 보면 친구님 생각나는데😱😍\n이제 더워지니...마카롱도 글쿠....그나마 스콘이 보내기 나은거 같은데 좋아하신다니 넘 다행이쥬~맛난 스콘집 보이면 보내드릴께요~❤️'], ['스콘친구님~♡\n걍 광교 찍어서 스콘 사들고 목동 넘어가야될까유?ㅋㅋㅋ\n택배보다 빠를듯ㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋ장거리 가능하십니꽈?? 그럼 저는 당장 먹을 수 있는 마카롱으로 준비하겠음돠 ㅋㅋㅋㅋ'], ['우리 넘 웃겨요ㅋ ㅋ\n혹시 요거 묵어봤어요~?\n난 첨봐서 그대랑 먹어보고싶어 담주에 다시 갈까 했는디ㅋㅋ'], ['미챠미챠! 이거 달고나스콘이죠!?! 나 먹고픈데 울 동네에서 팔았던 곳 물어봤더니 이제 안만드신대유 ㅜㅜ'], ['오늘 광교가니 떡하니 있는디 요기도 줄이 살포시 있어서ㅠㅠ\n글서 담주를 기약했는데 공홈 전화연결 안되더니 막 출고 시키고~\n의욕상실이예요~ㅠㅠ'], ['이거 줄 설거 같아요... ㅜㅜ\n그래도 하루빨리 맛본다 생각하시고...스콘 위해 고고해보셔요...그리고 맛 알려주셔요.....😰😱😩'], ['글게요~\n이래저래 오늘 넘 아쉬운 하루네유ㅠㅠ'], ['내생각은.....엣햄'], ['그대는 영탁이랑 던킨, 피클보면 생각나요ㅋㅋㅋ'], ['맞다~옥포수박이랑~'], ['흥칫뿡~~~'], ['왜유~?왜유~?\n나 모 잘못한거여유~?'], ['ㅋㅋ'], ['내사랑 삐치지 마요~\n택배는 담주에 붙일께요~♡\n받고 맘 풀어유~^^'], ['ㅠㅠ부띠크 댕겨올 수 있는 지영님 부러워요❤️ 사은품 받으시면 구경시켜주세요!☺️'], ['공홈 전화연결이 계속 안되더니 이미  출고가 되었네요ㅠㅠ\n괜히 헛걸음 한거같아 속상해요ㅠㅠ'], ['쟈쥬니님~공홈 결제 잘 하셨쥬??\n전 공홈 주문하고 부띡 간건데....리밋들 다 있어서 ㅋㅋ 왜 공홈 주문했지?순간 생각이 들었어요🤣\n리밋 왠지....계속 있을것만 같은 😅'], ['네쏘 상담도...일처리도.. 이번에 진짜 별루였어요ㅠㅠ 홈페이지 담당직원 마이 혼나겠쥬?'], ['과연 혼낼까유~?\n어차피 반품 되는것도 아닌데...\n그냥 그렇게 흘러갈듯해요~\n\n모든지 소비자 책임이쥬~ㅠㅠ'], ['그르니까요.. 밤에 올라간건 적은 수량으로 연습용(?)으로 올렸던건가봐요 실수로 고객들한테 노출이 된거 같구 ㅠㅠ 슬리브팩 오후에 다시 품절 풀리더라구요 ㅋㅋㅋㅋ'], ['이놈들 고객들 맛보기 시켰네유ㅠㅠ'], ['팔아야 하는데 하루만에 품절이구 끝이면 얘네도 넘넘 손해일거에요😅 매장은 왠만하면 있는거 같아요~'], ['취소된다더니 출고부터 하는 멋진 전략...ㅋㅋㅋㅋ고단수인가봐요 🤣'], ['ㅋㅋㅋ 어차피 제글이여서 다 보여용ㅋㅋ'], ['그쵸.. 어제 품절떠서 괜히 쫄았....😭 뷰레시피보다 젤루님 사신 버춰컵이 이쁘던데 괜히 맘만 급해서 많이산거 같아요ㅠㅠ'], ['난테 말씀하시는거 아닌데 나도 막 댓 다는 이상한 짓ㅋㅋ\n\n저도 젤루님 컵보고 젤루님집 가야하나 막 고민했떠요ㅋㅋ'], ['단체톡 같아서 좋아유ㅋㅋㅋㅋㅋㅋㅋㅋ😘 구경가고싶네요😍'], ['저도 젤루님집 구경가고 싶퍼요ㅋㅋ'], ['저도 사실 이번에 주는 뷰레시피보다 버춰라인이 더 맘에 들어요~그래서 오늘 실물보고 산거에유~♡'], ['오디갔다 급 나타나셨대유?'], ['흑흑...사진에 보이는 컵이 다여서 부끄러버요😂뭐가 더 엄씀요 ㅋㅋ'], ['나두 컵이 없어요ㅋ\n근데 네쏘 컵은 오리지널보다 버춰컵이 더 이쁜거 같아용ㅋㅋ'], ['그쵸?저도 버춰라인이 더 예뻐요~♡'], ['글서 지난번에 캡슐 두줄만 사고 버춰 컵이랑 샘플만 받아오려고 했는데 일잘하는 언니 꼬임에 넘어가서 버춰 컵 포기하고 캡슐이 한가득 들고왔쥬ㅠㅠ'], ['아 그랬구나여~ㅠㅠ 우리 담엔 캡슐은 고만 사고 차라리 컵 받아유~😅❣️'], ['그래야긋어요~\n진짜 이제 캡슐은 고만~\n언제 다 마실지 걱정이네유ㅠㅠ'], ['이렇게 인기 폭발인데 뷰띡 좀 더 늘려줬으면 좋겠어요...'], ['저도 뿌띡 한번 갈라하믄 큰맘 먹어야해서 저희동네도 하나 들어왔음 하는 바램이  있네요~ㅠㅠ'], ['그래도 맛있는 빵 득템이셔요~~츄르릅^^'], ['저녁은 핑크씨앗님이 먼저 보내주신 신상 뚫어서 요빵 신나게 먹어치울거예요~\n으흐흐흐~♡'], ['허걱**신상 리밋에다 나♡레옹 빵 \n최고 궁합이네여'], ['멋찌게 마시고 먹어줄꼬예요ㅋㅋ'], ['맛나게 즐기세여♡'], ['감솨합니다~'], ['아 오늘 광교점 사람 박터지던데오픈이벤트는 끝난거에요~?'], ['번호표 받으사람은 일주일안에가면 오픈이벤 챙겨준다는데 이미 전 공홈에서 출고가ㅠㅠ'], ['번호표는 끝났겠죠?ㅠㅠ 이제 퇴근해서 가는중인뎅'], ['가시면 소식좀 전해주세요~^'], ['오~빵순이 빵보고 츄릅 합니다~'], ['거기있는빵 다집어오고싶은거 참고 소소하게 담아왔네요ㅎ'], ['그레이스님 오늘 광교 다녀오셨어요?!저기서 그럼 안될거같아 보니 그레이스님 글이 똬악🥰🥰전 광교는 나~~~중에 가려고 다른 부띡 갔어요😆😆밀크티의 세계에 입문하셨군요😉😉저도 아직 입문 못한 세계.....'], ['책임지세요~'], ['아웅...책임지라니 전 이제 그레이스님꺼?'], ['저희집에서 젤 가까운곳이 분당이라서 비올레따님 일하시는곳 어딘지 알았음 아닌척 커피마시고 왔을텐데  못가니 광교로 떠났어요ㅎㅎ'], ['내꼬~좋아요💖💕💘❤'], ['오시면 맛난 코피 한잔 타드렸을텐데 아숩네여🤣🤣🤣광교갤러리아 괜찮나여?사람많아서 좀 한산해지면 가볼까도 고민중이에여전 오늘 퇴근하고 수내부띡으로 달려갔어여🏃\u200d♀️판교현백이 좀 더 가까운데 현백포인트가 없어서 롯데로....🤣🤣'], ['전 이제 그레이스님꺼쾅쾅!'], ['커피숍 어딘지 알려주떼요~\n근처가면 들리고싶어요~🙏'], ['사내카페라 출입이.....😭😭😭😭요기말고 다른곳은 알려드릴수가 있어요~근데 월~수 저녁때만 있는지라🤣🤣🤣'], ['밤나들이 좋아합니다ㅋㅋ\n어느순간 꽃히면 달려갑니다ㅋㅋ\n삘로사는 뇨자라~✌'], ['부담스러우시면 안가르쳐 주셔도 되요~^^'], ['노노,전혀 노부담입니당😆😆😆😆쪽지드릴게여❣❣'], ['감솨합니다~ㅎㅎ\n언제 갈지는 아무도 모릅니다ㅎㅎ'], ['쪽지드렸어요😆😆😆저 역시 삘받아 비행기탔던 여자입니당✈✈삘에 살고 삘에 죽는 즉흥파~~'], ['오~ 저도 급 삘받음 막 다닙니다ㅋㅋ'], ['역시 그레이스님과 운명이군요😘😘😘😘삘이 마구마구 통해요ㅋㅋ']]
    
    7130
    사람은 참.. 욕심이 끝도 없는것 같아요... 절 두고 하는말이에요...ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ귀차니즘이 너~~~무 심해서 집안일이 싫은데안할수도없고...빨래 너는게 너무 귀차나~~~ 누가 좀 널어줬음 좋겠어~~하다건조기들이고 잠시 신세계 맛봤더니개키고 정리하는것마저 귀찮고ㅋㅋㅋㅋㅋㅋ그거 해주는 기계 어디없나 ...서서 설거지하니 허리도아프고 별 음식 하지도않는데 설거지거리가 너~~무나와서식세기들이고 설거지옥에서 해방되었더니이젠 설거지다된 그릇 닦고 정리하는게 넘 귀차나요ㅋㅋㅋㅋㅋ참.. 욕심이 끝도 없습니다ㅠㅠ집안일 해주는 로봇은 언제 개발되려나용
    
    [['ㅋㅋ 윤민은 괜찮아~ \n한참 피곤하고 그럴때임요~^^'], ['늘 그랬지만ㅋㅋㅋㅋㅋ 지금은 그래도 더 이해받을수 있을때쥬~??ㅋㅋㅋㅋㅋㅋ 즐겨야겠어용ㅋㅋㅋㅋ'], ['암요~ 암요~ 맘껏 즐겨용~\n담에 맛난거 먹으러가요~^^'], ['맛난거는 언제나 환영이쥬~~~~😍😍'], ['가정부가...필요하군요 ㅋㅋㅋㅋ'], ['역시... 찰떡같이 알아들으시는 ...ㅋㅋㅋㅋㅋ🤭🤭 암요 가정부가 피료합니다ㅋㅋㅋㅋㅋㅋ'], ['핵공감입니다\n건조기 들이고 신나서 춤춘게 엊그제 같은데 \n먼지거름망 청소도 귀찮고 말씀대로 개고 정리하는거 너무 귀찮아요ㅋㅋㅋ'], ['악 맞아여ㅠㅠ 개고 정리하는것도 일이지만 매일 필터 먼지빼고 씻고 말리고ㅠㅠ 어째 일이 더 늘어난거같고 막ㅋㅋㅋㅋ 그래도 비오는날 부담없이 뽀송하게 말릴수있음에 감사합니당ㅋㅋㅋㅋ'], ['저 시간당 5만인데 어찌 감당 되시겠습니까? 싸모님? ㅋ'], ['그냥.. 제 허리 포기할께요....ㅋㅋㅋㅋㅋㅋ 넘 고급인력 아니십니꺼!!!!!!!'], ['와ㅡ아 쎄다!!!🤭🤭🤭\n한번 써보고 싶었던 ㅉㅉㅃ분인데...'], ['저 초성은 뭔가여!'], ['아ㅡ세대주 글에 ~~ 힌트있어요~🤣🤣🤣'], ['앗!'], ['일하러가서 잔소리 하능거 아닙니꼬??ㅋㅋ'], [''], ['ㅋㅋㅋ 싸모님에게는 친절히~~~ ㅋ'], ['ㅉㅉㅃ 보고 빵터졌어요ㅋㅋㅋㅋㅋㅋㅋ 찌찌 뽕인걸 알리지마라니ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['돈주고 잔소리 사는 격이죠???ㅋㅋㅋㅋㅋㅋㅋ'], ['근데 옹니 잔소리는 귀여움요~♡\n집은 엄청 깨끗해질듯~ ㅋㅋ'], ['잔소리로 저를 일시키실것같아요!!!! 그래서 깨끗해지는...? 집안일강사.. 라서 비싼걸까요?ㅋㅋㅋㅋㅋㅋ'], ['돈 받아서 돼갈 먹여드릴게요 ㅋㅋㅋ'], ['뭐죠ㅋㅋㅋㅋㅋ 그럼 또 손해보는건 아닌거같고ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['그쥬~~저 같은 가정부 어디에도 없쥬?ㅋ'], ['저듀.....ㅎ 살포시 공감해봅니다ㅎㅎㅎ 건조기사니 담엔 빨래개는 로봇나왔움 생각해봤어요ㅎㅎㅎㅎ'], ['역시...ㅋㅋㅋㅋ 빨래 개는 기계있던데 가정용으로 쓰긴 참 거시기 하더라구요ㅋㅋㅋㅋㅋㅋ 그냥 제손 쓰기로....'], ['아ㅡ신세계 맛보고 싶네요~~\n식세기!~ 진짜 넣어보고싶은데~~\n건조기!~ 도 없구나~~🤔 \n[새로 시집가고 싶네요~🤣🤣🤣]'], ['댓글 달고 보니 이콘이 똑같네요~'], ['앗!!~ 🤣🤣🤣\n[같이 시집 고고~~🤣🤣🤣]'], ['ㅋㅋㅋㅋㅋㅋ 이사가 아니고 시집을 다시ㅋㅋㅋㅋ'], ['이번생은....... 틀렸습니다~😭'], ['다음생에는 집안일하는 로봇이 있는 세상이길..... 아 그전에 그런거 살 돈부터.....'], ['저 빨래 개는게 제일 싫어요ㅋㅋㅋㅋㅋ\n저희집엔 10살짜리 빨래 개주는 인력이 있어서 다행입니다'], ['저도 6살짜리 새 인부를 고용해야할까요ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 찬찬히 가르쳐 봐야겠어요ㅋㅋㅋㅋ신랑수업!!'], ['수건 개기부터 가르치면 잘 해요ㅋㅋㅋ\n7살때부터 시작 했습니다!'], ['전 좀더 조기교육을 시켜야겠습니다!!! 낼부터 수건 개기 교육 시~~작ㅋㅋㅋㅋㅋㅋ 두번 일하는거 아닌가 몰러요ㅋㅋㅋㅋ'], ['이콘처럼 코 파주는 기계는요?ㅋㅋㅋ'], ['코는 그래도 제손으로 파야 시원해서ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['ㅋㅋㄱㅋㅋㅋㅋ 요 이콘 코파는것도 웃기지만 표정도 넘 웃겨요🤣🤣🤣'], ['저 잘 따라할수 이써용ㅋㅋㅋㅋㅋㅋ'], ['제가 요즘 그래요..ㅠ 권태기 인지 만사 다 귀찮고..ㅠ 하기 싫고..ㅠ\n셤니는 빨래는 세탁기가 해주고 청소는 청소기가 해주는데.. 그 기계를 사용하고 정리하는것도 다 내 몫인데.. 왜 그건 생각을 안해주시는걸까요..?? ㅠ'], ['맞아여ㅠㅠ 밥은 밥솥이하는데 밥솥이 쌀도 씻어주냐구요!!!!ㅋㅋㅋㅋㅋ'], ['욕심이 끝이없죠...전 자동목욕기 있음 좋겧어요. 정리 다 끝내고 애들도 자러가고 나면 에너지라곤 없어져서 정말 식기세척기 세탁기안에 들어가고 싶어지네요...누워있음 머리도 감겨지고 샤워도 되는 기계..'], ['어머.. 그거 저도 탐나네요ㅠㅠ 누가 좀 씻겨줬으면ㅠㅠ'], ['부경맘 할인해드릴께요ㅋ\n시간당 마넌? 어때요?ㅋㅋㅋ'], ['그럼 딱!! 한시간만 할께요 한시간안에 모~~~든 집안일을 다 해주세용ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['와... 당했다ㅋㅋㅋㅋㅋ'], ['건조기 부럽사와요❤❤'], ['건조기는 사랑이지유ㅠㅠ'], ['아..ㅠ 진짜..ㅠ 작동하는것도 내가 다 하는데.. 누구하나 도와주지도 않으면서.. 너무 쉽게들 생각하니 서운하고 섭섭하네요..ㅠ'], ['본인들도 다 해보셨으면서 왜들그러실까요!!!!ㅠㅠ'], ['반성합니다..밥도안하고 집안일도 잘안하는데 만사가 귀찮습니다...자는거빼고 귀차나여😭😭😭'], ['요즘은 숨쉬는것도 귀찮구나.. 느끼는참입니다ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['마쟈요!ㅋㅋㅋ']]
    
    7334
    청소보다 더싫은 집안일  .... 다 자는디 저도 자고잡은디 이거 언제다  정리할까요?  ㅜㅜ진짜 싫으네요  아들있을때 할것을  7살고사리손으로  수건이라도  처리해 주는디  ㅎㅎ
    
    [['하 제가 제일 하기 싫어하는 집안일이예요..\n빨래개기...빨래감도 많아서\n개는 시간이 오면 진짜 공포의 시간 ㅠㅠ\n힘내셔요 맘님🙏'], ['어찌어찌 다하고 자긴했는데 정리는 아직입니다 저녁에 할래요  ㅋㅋ'], ['완전 공감이여..빨래개우는거 저도 싫어요ㅠ 설거지는 싫어하는편은 아닌데..빨래개우기는 너무 귀찮아요ㅠ'], ['엄마들이 다 같은가봐요 정리잘하고 청소잘하는분들보면 부러워요'], ['저도 은근 빨래개는게 제일 귀찮고 하기 싫더라고요~ 왜 그런지 이유는 모르겠지만요ㅋ'], ['매일하는 일이고 해도 또있고 반복되니 더 귀찬은것 같아요'], ['저는 설거지요ㅠ빨래는 신랑이 많이 접어주는편이라..ㅠ 지금 설거지 쌓여있는데 하기싫으네요ㅠ'], ['쌓인 설겆이 보면 짜증나요  하다보면 금방한디 시작이  힘드네요'], ['전 설거지두 빨래개기도 싫은건 왜 일까요?~;;아니 걍 집안일 다 싫은건 왜일지ㅠ'], ['저두요 다 싫어요 귀찬고  한살이 먹어가면서 더 귀찬아저요'], ['빨래 정리 하는게 은근히 귀찮은 것 같아요 ㅎㅎ 세탁기.건조기 까지는 괜찮은데... 마지막이 귀칞죠'], ['다 귀찬아요 ㅋㅋ 빨아서 옥상에 널었다다가 걷어와야하고 개서 제자리찾고  다 귀찬아요 ㅋ'], ['혼자하면 하기 싫을때 있어요 생각 안하고 있는데 수건이라도 게워서 주면 고맙더라구요'], ['맞아요 엄마 도와주고 잘래하면서  개서  제자리가저다주면 고맙다라구요'], ['빨래개는건 할만한데..지자리에 가져다놓기가  싫어요애들꺼 가져가라해도 내꺼.막내꺼신랑꺼는 가져다넣어야하는데..귀찮..'], ['그래서 다 개서  바구니에 넣어서 방지시켯어요 ㅋㅋ  저건이제 저녁에나 정리할듯합니다 ㅋ'], ['애들이 크면 살림하는것도 수월해져요..쫌더 기다려보세요  그리고...화이팅요'], ['그럴까요?  근디  둘째가 이제 3살 언제 크나요? ㅋㅋ  시간이  약이겠네요'], ['다 자는 시간에도 ..  같이 쉬는게 아니라 남은일 한다는게 참..ㅠ'], ['그쵸  근데  또  누가 부르지않고 신경쓸거 없을때  하는게 더 편할때도 있더라구요'], ['빨래개는것  진짜 일이에요ㅠ 개어서 또 다 서랍장에 넣고! 집안일은 끝이 없네요ㅠ'], ['맞아요  귀찬아서 개기만하고  서랍에 넣진 못하고 놀다잣어요 ㅋ'], ['저두 빨래 마른거 바구니에 넣어서 한쪽에 밀어놨네요ㅋㅋ 넘 하기시러용ㅠㅠ'], ['옥상에서 걷어온거라  미뤄두면 다시 구겨질까비  늦게라도 처리했네요'], ['맞아요저도 집안일이 젤로 싫어요집안일이 젤 못해요ㅜㅜ'], ['정리정돈만 않해도 살겠어요  그시간이 너무 지루하고 힘드네요'], ['요즘 빨래 개키는게 왜이리 싫은지요특히 수건...\xa0\xa0•᷄⌓•᷅\xa0 어마무시해요 진짜'], ['수건 하루 6장씩  꼬박꼬박 나옵니다 젖은거라 않빨고 미룰수도 없지요'], ['전 다싫어요ㅜㅜ 빨래 가끔한쪽에두면알아서 찾아서 입고..ㅋ 건조기에 며칠있기도해요ㅜㅜ'], ['건조대에서  필요한거만 찾아서 입기도하는더ㅣ이건 옥상에서  걷어온거러 ㅎ'], ['빨래개워주는 기계는 안나오나몰라요ㅠㅠㅠㅠㅠ'], ['있긴있다고 들었어요 근데 시간이 엄청 걸려서 별루 쓸대없다고요'], ['저도 한보따리 방금 신랑이랑 수다떨면서 개웠네요 ㅠㅠ'], ['신랑은 진즉 잣답니다  않잘때 쌓아두면 개주기는 하는데  ㅎㅎ'], ['아 진짜 집안일중에 빨래 개는게 제일 귀찮아요ㅜㅜㅜ 빨래 개주는 로봇이 있음 좋겠네용ㅋㅋㅋ'], ['청소로봇에 밥해주는 로봇에 빨래개주는로봇까지있음 정말 편하겠어요'], ['집안일은 해도해도 끝이 안보이는거 같아요...ㅎㅎ 어자피 해야하니 힘내셔요ㅠㅠ'], ['맞아요 어자피해야할일이라 전 더 짜증나요 그래도 어째요 다하고 잣네요'], ['그리고 집안일은 열심히 한다고 했는데 티가 안나서 더 화가 나요ㅜㅜ'], ['빨래가 제일 하기 좋던데ㅜㅜ 설거지가 왜케 싫을까용... 저랑 바꾸실래욬ㅋㅋㅋ'], ['ㅋㅋ 전 둘다 싫어요 정리정돈은 저랑 정말 않맞는것 같아요'], ['저는 저리 쌓으면 눈이 먼저 스트레스받아 빨랫대에서 빼면서 갭니다'], ['저도 그거 해봣는데  서서개는게  더 힘들어서 앉아서해요'], ['정말 하루 빨래.설겆이 하고나면 또 그시간 돌아오고 늘 같은일반복이죠~~'], ['반복되는 일상  지루하고 힘들어요  오늘도  전 빨래를 열심히 해봅니다 ㅜㅜ'], ['ㅜㅜ그럴것같아요.여름에는 특히나 수건 벗어놓은 옷들 이불들 장난아니죠ㅜㅜ파이팅입니다~~'], ['맞아요ㅜㅜ 진짜ㅜㅜ 집안일ㅜㅜ 요즘제가 그렇네요ㅜㅜ'], ['빨래개는거  세상귀찬일이예요  저거라도 신랑시킬까봐요 ㅋ'], ['건조기 있으니 이게 게워주는 기계 나옴 좋겠다 싶어요 ㅎㅎㅎ 후아'], ['저희집엔 건조기도 없어요 ㅜㅜ  건조기부터 장만해야겠어요'], ['으앗 ㅠㅠㅠ 닉넴 보심 자녀분들도 많고 그러신거같은대ㅠㅠㅠ 신랑분한테 언넝 사자고 하셔요ㅠㅠ고생 많으십니다ㅠㅠ!!! 진짜 부지런하시네요'], ['닉넴에 아들하나는  서방입니다 ㅋㅋ집에 건조기 놀자리가 없어요'], ['진짜 집안일은 해도해도 끝이없고 또 해놔도 티도 안나요ㅜㅜ']]
    
    7350
    첫째 학교 숙제 집안일 학교 숙제도 많은데 엄마 일 도와주기가있네요 그냥 써 하니 그거 거짓말이야첫째 참 고지식해서 지금 장난감 정리빨래 정리해서 자기옷 넣어 놓고 숙제해하니 아니 아직 남았어요화장실 청소를 한다고 하길래 아니야했더니 가방에서 뭔가를 보여주면서 선생님이 화장실 청소 도와주라고 나눠주셨다고 하시네요.청소 도구도 주시면서 숙제 하라고 하네요.
    
    [['아~ 요즘 학교는 이렇게 친절하군요.'], ['그러니깐요 집이 깨끗해졌어요'], ['점점 모든 곳이 친절해지나봐요'], ['옛날하고 너무 다른 수업 방식이네요'], ['그렇게 바뀌는 게 좋은 것 같아요'], ['오늘 제가 행복하네요'], ['선생님이 가정에 행복까지 ㅎㅎ'], ['요즘 선생님 참 바쁘시네요 다 챙겨야 하시니요'], ['그러네요. 그 선생님이 좋으신 분인듯요'], ['오. 그런숙제 좋은데요 ㅋ'], ['최고예요 오늘 행복하네요'], ['그 행복 하루종일 누리시길..^^'], ['장은 보셨어요?'], ['오늘은 간편장.. 낼 또 볼거예요. 삼계탕 하러..ㅋ'], ['저는 오늘 삼계탕 합니다'], ['오늘  신랑 저녁약속 있어서 내일 하기로 했어요. 삼계탕 굿~~'], ['오호 신랑 오늘, 낼 없어요 ㅠㅠ'], ['이틀이나요? 그런데 삼계탕 하시나요? ㅎ'], ['신랑은 밖에서 먹을거 같고 저희 아이들만 오늘 먹일려구요'], ['그렇군요 미리 닭 사놓으셨나봐요'], ['태권도에서 보내주셨어요'], ['잉? 태권도에서 그런것도 보내주나요?'], ['제자들 더위 먹지 말라구요 미션으로 끓여서 맛있게 먹기요'], ['오 인증도 해야하나요? ㅋ'], ['미션 인증하면 낼 아이들 수박 파티 하신다고'], ['오. 수박파티를 위하여 인증샷 날려~~'], ['방금 인증샷 날렸습니다'], ['오.. 잘했어요. 수박파티는 따놓았군요'], ['오늘 저녁까지 70명은 넘겠죠 ㅋㅋㅋ\n안넘어도 하실분이지만요'], ['오 디게 사람 많나보네요'], ['도장 아이들 좀 많아요 지금 코로나로 덜 받고 계시지만요 관이 2개거든요'], ['오 관장님 돈 잘 버시겠당'], ['버는 만큼 이렇게 많이 쓰세요 ㅋㅋ'], ['그래야지 더 벌지요 잘하고 계시는거'], ['아이디어가 좋은거죠'], ['네 사람 끄는 방법을 잘 아시네'], ['그래서 인기가 많나봐요'], ['그래서 돈도 잘벌고 ㅎ'], ['저희도요!! 둘째 설거지해야한다고 얼마나 귀따갑게 얘기하는지..'], ['오늘 집안 청소 아이가 다 해주네요'], ['아~~  고맙긴한데..  또 다시 해야ㅠ'], ['아이가 더 깨끗히 해요 ㅋㅋㅋ'], ['전 너무 안시킨걸까요 ㅋㅋㅋㅋㅋㅋ'], ['시켜보세요 잘해요'], ['한녀석 시키면 나머지 한녀석이 난리라;;\n엄둘 못냈거든요'], ['덕분에 엄마는 한 번 더 웃으셨겠어요^^'], ['네 너무 좋아서요'], ['아이들이 그러면 사랑스럽죠ㅎ'], ['소감도 잘 써줬어요'], ['담임이 중요한 것 같아요~'], ['첫째가 선생님 복은 많은거 같아요 \n둘째도 지금 첫째 1학년때 담임이신데 두분다 너무 좋아요'], ['저희 담임 선생님은 말투부터가 무서워요ㅠ'], ['저희 선생님이 좋아 천사 첫째 담임은 외모도 인형 같아요 말투도 진짜 이쁘세요'], ['얼굴 예쁜 사람이 어찌 성품까지^^'], ['그래서 눈에 쏙 들어 오네요'], ['그런 사람은 전생에 나라를 구했겠죠?'], ['전생에 나라를 두번정도 구했나봐요 ㅋㅋ'], ['나는 전생에 나라를 팔아 먹었나ㅠ'], ['와~  선생님 센스있으신 선물이네요'], ['그러니깐요 첫째반만 이렇게 하시는거 같아요'], ['그런 선생님이 계시더라구요'], ['저는 너무 좋네요'], ['그러게요~~ 이번 선생님이 센스쟁이~']]
    
    7500
    남편 집안일 어디까지 하나요? 다른 신랑들 집안일 청소도 포함이요청소기나 빨래 등등 포함해서 다들 집안 일 청소설거지 등등 어디까지 도와주시너요? 너무 궁굼해요
    
    [['저희남편은ᆢ 1도0도 안해요ㅜ'], ['ㅜㅜ 너무 힘들꺼같아요 으앙'], ['육아 집안일 암것도안해요ㅎ'], ['좀 도와주시지요 ㅠㅠ'], ['왜결혼했대요? 참~ 그러지마시지...'], ['글게요 말해뭐하나싶어요ㅜㅎ'], ['왜결혼한지모르겟어요ㅜㅜ지이미지관리 차원에 결혼이필요햇던건지 싶을정도네요 ㅎㅎㅎ'], ['청소 빨래 설거지 쓰레기버리기... 깔끔하게 마음에 들게 하진 않지만 그래도 많이 해요. 음식하기 빼고는 다 합니다.'], ['청소나 설거지는 며칠에 한번하나요? 빨래도 하나요?? 박수를 보냅니다 ㅋㅋ'], ['ㅋㅋㅋ청소기 돌리는 건 일주일에 서너번(깔끔하지 않아요 걍 대충ㅋ) 빨래는 쌓여있는  거 못보는 조급함으로 적당히 있다 싶음 무조건 돌리구요. 설거지는 같이 식사할 땐 무조건 후처리 맡아 하다가 최근에 식세기 들여서 도움 받고 있죠.^^;; 그 외 쓰레기 분리수거 담당, 음식물쓰레기 담당입니다.'], ['빨래개고 분리수거 시간나면 해주고 설거지나 식기세척기 돌려주고~ 꽤 많이 하네요~'], ['애기 장난감 정리. (1주일에 한두번 할랑가?) \n분리수거(한달에 한번 하나?) \n화장실 청소( 한달에 두어번 하라고 하라고 해야하고) \n빨래개기( 한달에 두번 정도) \n\n애랑 겁니 잘 놀아주는 육아 외엔 다 제가 해요 \n\n저는 전업이라 내 일이다.. 하고 \n육아는 공통이다 생각하고 남편이랑 같이 해요'], ['ㅋㅋㅋ애기랑 겁나 잘 놀아주니 좋은 아빠네요!! 좋은 마인드에요 두분'], ['저녁설거지만 겨우 합니다;'], ['그래도 설거지는 매일 해주나요?'], ['분리수거랑 청소기, 화장실청소요 ㅎㅎ'], ['저도 화장실청소해주는데 너무 안한거 같이해서 짜증나요 ㅠ'], ['간간히 설거지와 분리수거 화장실2개중 한개청소 빨래돌리고 시간되면 널기까지...그외 시간되면 아이도 씻기고 마늘까기 찧어서넣기등 잡다한거...원래 안했었는데 10년간의 교육과 잔소리의힘! 이제좀 편해졌어요...'], ['그럼 잔소리 지속해야겠어요 ㅋㅋ'], ['아마 지치실거에요~그러다 쌈도하실거고...내가 왜이러고살아야하나싶고...하지만...굴하지 마시고...끝까지 니가이기나 내가이기나...해보세요~전 남편이 시댁에서도 설거지해요...저 시어머니 설득 2년걸렸어요...님께서도 함 시도해보실만하면 함 시도해보세요~시댁가는거 세상편해요~'], ['가끔가다 주말에 청소기나 돌리지ㅜ 상전이에요 아주~'], ['아기가 있는데 본인이 상전!????ㅋㅋㅋㅋ'], ['그러니까요ㅠ'], ['재활용쓰레기만 버려요. 가끔 손님 초대할때 청소 도와주고요. 신혼일때는 재활용쓰레기 버리는거랑 청소는 했는데 애둘낳고 더 바쁜데 전보다 덜 하네요ㅜ'], ['출근할때 음쓰버리기. 재활용. 화장실청소. 설거지. 각종 조립. 청소 (선풍기날개. 에어컨필터. 가스렌지후드 청소등 ) \n...\n마니 하네요? ㅎ ;;'], ['1도 안해요ㅜㅜ 포기여'], ['애기는 없지만 개두마리 키우는데 개독박육아에 집안일 99% 제가 하지만 대신 전 집밥을 일절 안해여..배고픈 자가 알아서 차려먹습니다. 아니면 둘다 퇴근후 배달시켜먹거나! 신랑 챙겨가서 먹는 점심밥도 신랑이 알아서 합니다.'], ['화장실청소 음쓰 일반쓰레기버리기 재활용쓰레기버리기 저녁설거지 신랑이 전담하구요 아이 저녁먹이기(일찍 퇴근시) 아이랑 저자고나면 아이 장난감 책 정리해주기 등등 해쥐요'], ['오늘은 퇴근후 청소기 해주고\n저녁 차려줬어요\n거의 매일저녁은 자기 안주삼아 이것저것 만들어줘요~~\n술을 자주먹어서 저도 같이\n5kg ~7kg 쪄서 화나지만 \n몸은 편하니 살거같아요 ㅎ'], ['아이꺼 우리꺼 설거지 도맡아주고요, 빨래 주로 돌리고널고 해주고요, 일반쓰레기 내놓기, 음식물쓰레기 버리고오기, 재활용 해주고요... 주된 택배정리도요. 쓰며보니 다 남편이 하네요^^; 아이목욕도 남편이ㅋ'], ['쓰레기랑 음식물쓰레기는 남편담당이고, 빨래개기, 설거지나 요리는 시간되는사람이 번갈아가며해요. 대청소는 일이주일에 한번 같이 하는정도..\n맞벌이에 대학원논문에 육아까지해야하니 체력이 딸려요'], ['시키는것만해요~'], ['출근할때 쓰레기 내놓기인데 가끔 요청,주말 아침식사 준비및 애둘먹이기와 설거지 ,주말에 집대청소 ..일찍퇴근했는데 부인이 아이들과 외출시엔 청소정도..평일엔 늦퇴라...기대도안해요'], ['세탁기돌리기.건조기돌리기.분리수거.쓰레기배출.화장실청소 맡아서 해요'], ['분리수거 쓰레기버리기 청소기돌리기 는 전담으로 해주고 \n가끔 밥도하고 설거지도하고 빨래도 해요~ \n써보니까 많이 하네.... 고마워 여보....'], ['정말 고마운 여보님이네요 !!'], ['평일 퇴근이 늦어 도와줄순없고 \n간혹 일찍 오는날은 아이랑 놀아줘요.\n주말에 회사안나가면 항상하는건\n아침육아 \n거실,안방 화장실청소\n매트들고 청소기 밀어주기 및 청소기 먼지털기\n자기가 먹은 컵은 항상 바로 설거지해놔요\n나머지 그때그때 부탁하면 해줘요\n(현관에 쓰레기놔두면 버리기 등)\n둘째가 어릴땐 평일퇴근이 늦어도\n오늘의할일이라고 제가 바닥닦아주기나 분리수거하기 보내놓으면 해놨었고 지금은 첫애도 유치원가고 둘째도 커서 평일은 독박해요\n(대신 반찬배달시켜먹어요)'], ['분리수거,화장실청소,쓰레기버리기,퇴근하며 장봐오기, 요리 저보다 잘해서 주말이나 일찍 퇴근하는날 요리해요 저는 빨래,청소,설거지담당.아이 목욕,교육,식사는 제담당이고 놀아주는건 남편. 저녁먹고 자기전 1시간씩 놀아줘요(아들)아빠가 젤좋은 친구래요ㅋ'], ['저녁 설겆이나 저녁밥도 가끔 해주고 잠자기전 간단 청소 방닦기,분리수거,화장실청소(뜨믄뜨믄),자기전 아이 일주일에 2회정도 책읽어주기,퇴근 후 아이랑 놀아주기(매일은 아니에요),아이목욕 \n이것도 신혼초 잔소리로 얻어낸 결과입니당ㅋㅋ \n남자는 가르쳐야되여 말안하면 모르더라구요ㅎ'], ['먼저 나서지는 않아도 시키면 잘 도와줘요'], ['*자발적으로 하는 일: 주1회 분리수거 쓰레기 배출, 에어컨 청소, 화분 물주기\n*시키면 해주는것: 음식물 쓰레기 버리기, 청소기 밀기, 아이 목욕(이제는 아이가 혼자해요)'], ['설겆이, 빨래, 청소,화분물주기,쓰레기버리기,.. 딸이랑 제 속옷은 삶아서 직접 손빨래까지해줘요..돈벌어다주고 살림해주고 흔하지도 않지만 좀 유별난 남자죠^^;;'], ['제 남편이 저보다 더 잘하고 깔끔해요...\n오히려 제가 해놓으면 남편이 한 번 더해요 ㅠ믿어주고 시켜주면 좋겠어요'], ['거의 다 해요..맞벌인데 제가 더 늦을때가 많아서ㅠㅠ 아이 씻기고 등원준비까지'], ['평일에 빨래개기. 주말엔 화장실청소. 설거지.분리수거.음쓰.일쓰버리기.정리정돈이요. 정리가 취미인가봐요 ㅋ'], ['평일에는 퇴근후 아이들 씻기고 놀아주고 쓰레기 버리는정도요~\n주말에는 제가 아이들이랑 마트나 놀이터 외출하는동안\n남편이 화장실 포함 대청소해요. 주말에는 설겆이,빨래정리도 하구요.\n주말 만큼은 저도 푹쉬는 편이에요~\n아이들 키우는게 제일 힘든일이라고 많이 도와주는편이에요.'], ['이제 제가 힘들어서 설겆이는 식기 세척기, 바닥은 로봇 청소기가 해서 한달에 한번 꼴로 분리수거, 쓰레기 버리라고 문앞에!!  꼭!문앞둬야 버려줘요.(13년차라서 답답한 제가 하는편이죠)'], ['저희 남편은 안해요. 대신 시키면 다 해요. 음쓰버리기, 재활용, 물걸레질 등?? 제가 안 시켜서 그렇치... 그냥 제가 후다닥 해버려서'], ['주말 요리 남편이 두끼이상은 해줘요~ 쓰레기버려주고 아들씻겨주는건 남편담당이요~'], ['청소기돌리기 걸레질 설거지 쓰레기버리기 빨래개기 등등해요'], ['분리수거 버리기 주말에 두끼정도는 남편이 만들어요 \n아이 머리말려주기정도? 아이랑은 자주는 아니지만 놀아주는 편인것 같아요'], ['분리수거,음식물이랑 일반쓰레기 버리기, 아이들 목욕, 주말엔 청소, 아이들 실내화 빨기, 설거지는 식기세척기 주로 돌리기 때문에 양이 적을땐 남편이 손으로 하기도 해요. 그 외에 빨래 널거나 개기도 해달라고 하면 해줘요~도와 달라고 하면 다 해주는 편이에요'], ['셋키우는데. . 늘 12시나 새벽한시. .  진짜 1도 아무것도 할수가 없죠. .  주말도 마찬가지구요. .  ㅠ저같은분 있으실까요?중간중간 불려나가 일도 합니다. . .'], ['자영업하시나봐요ㅜㅜ저도가끔나가서일해요ㅜ자영업부인으로살기진짜힘들죵ㅜㅜ애셋대단하세요진짜'], ['청소기돌리기, 빨래, 쓰레기분리수거, 음쓰버리기 가끔 변기청소, 설거지도 종종?'], ['빨래, 청소 설거지, 음쓰, 화장실 청소, 쓰레기 재활용버리기 주말 요리 등등 많이 해줘요. 그리고 부탁하면 다 해주구요. 임신전 맞벌이때도 잘 해줬지만 임신 후에는 거의 다해요. 육아도 남편 친구들이 와이프 자유부인 해주고 애들 데리고 카페에서 커피 마시고 놀러다니는 라떼파파 모임 있는데, 이제 아기 생기면 거기 낀다고 엄청 기대중이에요.'], ['일주일에한번재활용버리기는꼭해줘요~ 일주일에한번이라 어마어마해서 남편이 저일어나기전에 싹버리고와서 일가네요~~~ 그외엔?? 흠...밥먹고설거지정도구요ㅋㅋ자영업해서 저 머리하러가거나 일있음 애들가게다두고 갔다오고해서 육아도하는거죠?ㅋㅋ대신쉬는날이한달에 세번밖에없어서ㅜㅜ 이정도로도와주는것만해도 고맙게생각하려구요ㅜ 정리도한번씩싹해줘요 정리정돈신이라ㅋㅋㅋ근데하고나서 저보고 잔소리대박이네요^^ 조용히절부르고 혼내요.........'], ['주말에 요리,분리수거,쓰레기비우기,아주가끔 화장실청소요ㅎ'], ['남편이 집안일을 도와준다, 해준다라는 표현 속에 집안일은 여자의 일이라는 인식이 들어있어서 남자들이 더 안 하는 것 같아요 집안일이라는 게 사람이 살아가기 위해 해야 하는 일들이라 아기 아니고서야 가족 모두 자신이 할 수 있는 일은 해야 맞는 건데 말이죠^^'], ['젊어서는 시키는 것만 하지만 나이 먹으면 여성 호르몬 많이 생겨서  모든  알아서 잘 해요 좀만 참고 기다리세요.'], ['설거지, 쓰레기분리수거,빨래개기,청소기 돌리기 또는 걸레질\n요정도해줘요'], ['결혼초에 교통정리 했어요. 자기가 돈을 벌어올테니 그돈으로 집안일 도와주시는분을 쓰라고~~~  육아는 같이 하는데 집안일은 아예 안해요~ 근데 제성격상 누구를 시키는걸 잘못해서 아주머니 안쓰고 혼자 다하니까 분리수거 정도는 해주는구만요.ㅋㅋ 대신 벌어오는돈이... 이것저것 합치면 연 2억은 넘는것같아요. 더많이 벌면서 도와주시는분들도 많겠지만.ㅎㅎ  불만없습니다.'], ['빨래개서  넣기   설겆이  애목욕시키기  외엔  암것도  안해요'], ['남편: 음식물 쓰레기 버리기, 분리수거, 빨래 개기, 스팀 청소, 주말 요리나: 주중 요리, 설거지, 빨래 하기, 청소기 돌리기기본으론 이렇게 나눠놓고 사정이 있을 때 상대가 대신 하니까 좋더라고요. 서로 잘하고 좋아는 일이라 미루거나 기분 나쁘지 않게 집안일 해요.'], ['평일은 아이목욕 재우면 집정리\n분리수거,음식물,빨래개기\n주말은 음식만들기\n\n제일중요한 아이가 계속놀아달라는 스타일이라 계속놀아줘요ㅋㅋㅋ\n이게 제일 고맙네요'], ['제가 하는 일 중에 화장실 청소 제외하고는 일반가사는 7:3정도로 하고 있어요~ 음쓰는 싫다고 했더니 출퇴근전후로 버려줘요.. 음쓰, 분리수거는 남편 전담이예요.. 주중에 평일2일은 저녁에 제 자유시간 저녁식사 애들이랑 알아서 하고 챙겨재우는 것까지 해요.. 전 처음부터 돈 많이 벌어오는 것보다 노동력을 원한다고 이야기했어요~'], ['분리수거, 음쓰, 기저귀 쓰봉정리, 아기 샤워 시키기, 저녁 설겆이, 빨래 개기 및 널기, 아기 양치 시키기 및 장난감 정리, 저녁 설겆이, 주말 청소 및 요리. ㅋ 어쩜 저보다 더 많이 할지두 ㅋㅋ (전 전업주부예요.)\nㅠㅠ 남편아 고마워~'], ['주말 밥,설거지요,재활용버리기'], ['같이밥먹은거 식기세척기 돌려주기(식세기 들어가는 만큼만..남은그릇은 제가설거지)\n주1회 재활용\n쓰레기갖다버리기. 딱 이거하고요. 대신 초3,1  아들둘 목욕은 매일 시켜요.'], ['설거지.빨래.음쓰.화장실..은 신랑이밥은제가. .하네요'], ['잉여남 1이요'], ['우와.. 다들 꽤 많이들 도와주시는군요...'], ['아이들 어릴땐\n쓰레기.분리수거.화장실청소 .설겆이.청도 다 해줬는데,\n애들 크고나서 지금은 분리수거만 해줘요.\n남편이 늦게까지 일하니 그점은 불만없고요.\n대신 마늘까기.생강까기.밥차리기.주말에 밥이나 간식만들기 이런건 종종 알아서 해줘요~'], ['정말 안하던 사람인데 어떤 계기로 인해 그래도 좀 바뀌었어요 그래봤지지만...음식쓰레기는 그나마 시키면 싫은티 내던거 안내며 하는거랑(음식물처리기 설치해서 횟수가 줄어서 짜증을 못내는것 같음)애들 목욕,설거지 간간히,가끔 바닥청소,분리수거,빨래개기 정도네요 근데 다 시켜야 한다는거요 꾸준히 하지를 않아요 그나마 짜증안내는게 변한거에요'], ['맞벌인데 무조건 집안일 같이 해요한명이 요리하면 한명은 설거지빨래, 청소 같이~뭐든 똑같이 같이 해요~'], ['외벌이인데 분리수거, 빨래정리, 음쓰버리기 고정적으로 하고 퇴근시간 맞음 애기 목욕시키고 재우고 쉬는 주말에는 제가 일어나기 전에 오전 내내 애기 밥먹이고 케어해줘요..'], ['남편이 거의 다해요 집청소 빨래 음식쓰레기 음식쓰레기 버리기 분리수거. 그런데 깨끗하게 하진 않아요\n 설거지통 청소 화장실청소 밥이랑 설거진 가끔 제가합니다.'], ['거의 90프로 이상의 집안일을 남편이 하네요. 올 해 들어 설거지는 100프로 남편, 음쓰 , 재활용, 화장실청소는 결혼하고 어쩌다보니 제가 한 번도 못 했네요. 세탁기도 거의 남편이 돌리고 빨래 개는 것까지 남편..ㅠㅠ 제 속옷이랑 옷은 제가 정리해요. 그렇다고 제가 남편 일할 때 매번 쉬는 건 아니고 체력이 허락하는대로 애 돌보고 물건 정리 틈틈이 하려고 노력해요. 진짜 힘이 없어 못하요 ㅠㅠ 전 힘만 나면 집정리하는 사람이고 집 더러운거 싫어하는 사람인데 제가 체력이 안 좋다보니 남편이 좀 고생하네요 ㅠㅠ'], ['저희남편은다해요.맞벌이~'], ['음식물,분리수거.가끔걸레질,화장실청소 맡아서 해주고 다른건 시키면 겨우해요. ㅋㅋ'], ['저희집도 까딱 안해요.. 돈 벌어다주니 자기는 할일 다 했다는 사고방식. 아버님이 그러했다네요. 어머님은 평생을 살림만 하시고.. 지금은 아무것도 하기 싫으시다고 명절도 울집서 하자고 하네요;;;'], ['전 외벌이\n남편 : 냉장고 및 서랍 각종 정리(꼼꼼함), 휴지나 세제같은거 떨어지면 넣기, \n주말만 빨래널고 개기, \n분리수거, 큰아이 공부담당, \n대청소할때 함께하기, 가끔 설거지 일케해요.\n주말엔 전날 휴대폰하다 쓰러져 퍼자는 저를 대신해.애들 밥주고 놀아주네요.. 😪😪'], ['즤집은 시키면 시키는거 다해요.. 근데 시키는것만 딱해요예를들어 설거지해라 하면 설거지만 딱해요 대부분 설거지하면 옆에 싱크대도 닦고 가스레인지도 닦고 하잖아요정말~설.거.지만하고빨래 개 하면 빨래만 개요.대부분 빨래개면 옷장에 갖다두고 수건도 욕실에 두고 하잖아요.그냥 딱 예쁘게 빨래만 개요.'], ['주말에 요리, 분리수거, 청소기돌리기, 빨래돌리고 널고 개고..가끔 식기세척기..어항청소..심지어 요린  훨씬 잘하신다는.남편이 여자로 태어나면 아무리  박색이라도 놓치지않을거에요.'], ['신혼 초부터 하나씩 가르치는 느낌? 지금은 세탁기 돌리고 분리수거 음식물버리기 빨래 개기 이정도 하고 계셔요~']]
    
    7530
    제일 싫은 집안일은? 저는 설.거.지에요........도돌이표같은 느낌..?전 워킹맘인데 직업 특성상 주말 중에 하루는 출근하는 편이에요.근데 이번주 금토일을 쉬었어요.금요일에 쓱배송으로 이것저것 시켜서 금토일 중한끼빼고 집밥을 먹었어요. (왕감동.....)오징어 세마리 사서 오징어무국, 오징어볶음, 오징어간장볶음 (손질할때 온갖 인상 다 찌푸리고..ㅠㅠ)계란말이, 오뎅볶음, 앞다리 사서 제육볶음, 만둣국, 짜장면, 소세지볶음, 유부초밥, 황태국, 참치미역국.....3일동안 한거에요.냉장고에 반찬 만들어놓고 먹는 타입이 아니라 그때 그때해요 ㅠㅠ (냉장고에 들어가면 손 안대는 신랑^^)저는 정말 대단하다고 생각해욬ㅋㅋㅋㅋㅋㅋㅋ배달을 주로 시켜먹는 제가... 대견해요 ㅠㅠ근데 설거지가.. 해도해도 계속 나오고 ㅠㅠ저는 설거지하는 시간이 너무 아까워요 헝 ㅠㅠㅠㅠㅠ아이랑 조금이라도 더 놀아줄 수 있는데..식기세척기는 손안대도 알아서 해주나요?맘님들은 제일 싫은 집안일이 뭔가여..저 설거지 극혐이에요마무리하고 대자로뻗었어요
    
    [['저도 설거지가 제일 거지같고 하기싫어요... 그리고 빨래 다된거 개는것도 너무싫어요ㅠ 귀찮아요...'], ['하...저도 거지같아여...... 빨래든 뭐든 다 괜찮은데 설거지 ㅠㅠㅠㅠㅠ 악몽 ㅠㅠ'], ['식세기 넘 사고싶어요ㅜㅜ'], ['저도요ㅠㅠ 진짜 신세계인가여 ㅠ'], ['화장실청소요 방충망창틀도 싫고 힘들어요ㅜ'], ['ㅠㅠㅠㅠㅠㅠㅠ 아 창틀도 있었네요'], ['전 빨래 개는거 까진 괜찮은데 제자리 찾아 넣는게 제일!!!!싫어요..ㅜ'], ['2222222제발 누가좀 넣어줬으면ㅋㅋㄱㅋ'], ['33333 저두 이거요!! 이건 기계도 없고 넘 싫어요 ㅋㅋㅋ'], ['444444 저도 이거요ㅋㅋㅋ 개는거야  티비보면서 사부작 사부작 하겠는데 이리저리 자리찾아 넣는게 넘 귀찮아서 어쩔땐 거실에 개어논 옷들이 묵혀있을때도 있다는ㅋㅋ'], ['5555 남편있을때만 개요.. 개켜놓으면 넣어줘요ㅋㅋㅋㅋㅋ'], ['666666 오오오오오!! 이렇게 동지들이 많다니 ㅋㅋㅋㅋ 저두 제자리 넣는거 싫어해서 남편보고 넣으라고 거실에 늘어놔여 ㅋㅋㅋ'], ['와우 격하게 공감요! 개놓고 걍 자면 남편이 넣어놓을때도 있고 아닐때도 있고ㅋ'], ['악.. 저만 그런줄알았어요 !!!! ㅋㅋㅋㅋㅋㅋ'], ['2222'], ['앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 댓글 너무웃겨욬ㅋㅋ'], ['격공이요ㅠㅠ옷 빨고 건조기 돌리는 것 까지는 괜찮은데 개놓기 귀찮아서 맨날 건조기 안에 두고 꺼내입어요ㅋㅋㅋㅋㅋ'], ['저도요! 귀찮아도 잔소리듣기싫어서 개놓고 안볼때 그냥 농속에 쳐박아놔요! 나만 찾을수 있어서 이것도 언젠가 정리를 해야할듯. . .ㅜㅜ'], ['이거 적으러 들어왔어요ㅠㅠㅠ'], ['ㅋㅋ 저같은 분들이 많으시네요\n저도 제자리 찾아 넣는게 넘나 싫어서 그대로 둔적도 있어요 ㅋ'], ['와 대박 저도요 ㅠㅠㅠ ㅋㅋㅋㅋㅋㅋㅋ'], ['저 음식하는거요!!!\n너무 시러요 ㅠ\n화장실청소도 설겆이도 다다 제가 할 수 있어요.\n음식하는거보다 뒷정리가 전 더 낫다는 ㅋㅋㅋㅋㅋ\n식사요정이랑 같이 살았으면 좋겠어요 ㅎㅎ\n아이 낳기 전에는 그래도 요오리 제법 했던 것 같은데 이젠 정말 시르네요 ㅠㅠ'], ['저두요 ㅠㅠ 완전 음식하는거 지짜 너무싫어요 ㅠㅠ'], ['맞아요.... 음식하려면 맘먹고 하는 스타일..ㅋㅋㅋ'], ['전 다시름요ㅜㅜ'], ['그러네요.. 좋은게 없네요ㅠ'], ['집안일 3글자가 극혐...'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ \n끝도 없고 답도 없는 ㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 앜ㅋㅋㅋㅋ'], ['화장실청소,설거지요'], ['언니 댓글 보고 설거지라고 고쳤어요 ㅋㅋㅋㅋㅋㅋㅋ 받침 틀리는거 극혐인데 내가.... ㅠ_ㅠ'], ['응? 왜? 나 틀려써??'], ['아뇨 제가 설겆이라고 썼어욬ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['어릴땐 나도 설겆이라고 배움 ㅋ ㅋㅋ 아 나이 티나써..\n모르는 개 산책 이런거 아님 아무도 몰라.. 티안나 ㅋㅋㅋ'], ['앜ㅋㅋㅋㅋㅋ 갑분 나이 현타 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], [''], [''], ['모든 집안일이요..'], ['그쵸.. 전 그중에 설거지가 젤 시러여ㅠㅠ'], ['전 안싫은데 하루에 1번만 해요.\n화장실청소가 싫음'], ['하루에 한번 몰아하기.. 제가 잘해요 ㅋㅋㅋㅋㅋㅋㅋ\n근데 이번엔 한끼먹고 하고 한끼먹고 했더니 정신 나가겠어요 ㅋㅋ'], ['요리요;;;'], ['요리도 싫죠.. 그쵸ㅠㅠ 맞아요'], ['전 바닥청소요.... 거실 전체에 폴더매트라 다 들고 틈까지 닦아야해서...ㅜㅜ'], ['아.... 폴더매트 사이 끼는거 ㅠㅠㅠㅠ  저 그래서 아이 어릴때 바꿨어요ㅠㅠ'], ['설겆이ㅋㅋㅋ저는 설겆이가 싫어영'], ['역시 ㅋㅋㅋㅋ 저랑 같으세여>_<'], ['ㅋㅋㅋㅋ악 ㅋㅋㅋㅋ진짜 너무 싫어여ㅋㅋㅋㅋ'], ['빨래 개서 넣기요ㅠㅠㅠ 아 진짜 소오오오름돋게싫어요ㅠ'], ['앜ㅋㅋㅋㅋㅋㅋㅋ 저는 설거지가 소름돋던데.... 댓글보다보니 빨래 개서 넣기 싫어하는분들이 많네요 ㅋㅋㅋㅋㅋ'], ['저도 설겆이가 제일 시러요 ㅠㅠ  참다참다 요즘 식기세척기 검색중이예요 ㅠ'], ['진짜 식세기 사야할판이에요 ..'], ['저도 집안일은 다~~~전부다~~~싫어요ㅜㅜ십년을 넘게 해와도 늘지도 않아서 더 하기 싫은가봐요ㅋ'], ['맞아요 ㅠㅠㅠㅠㅠ 저도 다 싫은데 그중에 고르라면 설거지가 제일 싫더라고요'], ['저는 설겆이가 제일 좋...핸드폰 거치해서 드라마 보면서 하거든요... 저는 화장실이 젤 싫어요'], ['옴마 배우신분ㅋㅋ 배워감당ㅋㅋ'], ['자동차에 붙이는건... 자꾸 떨어져서ㅋㅋㅋ'], ['오 ㅋㅋㅋㅋㅋㅋㅋ 저는 노래틀고해요 ㅋㅋㅋ'], ['저는 닦는거요 먼지도 방바닥도 설거지도 닦는건 다 귀차네요0ㅜㅜ'], ['아... 전 그래서 놔버려요. 청소기만 돌리기도 하고.. 바닥 닦는건 귀차나서 몇일에 한번 ㅠㅠ'], ['저도 집안일 진짜 못하고 정리도 못하는데ㅠ 그중에 설거지가 진짜 제일 싫어요ㅠㅠ 지금도 싱크대에 설거지..,  아 하기싫어요ㅠ'], ['그쵸ㅠㅠ 제일 극혐 ㅠ 너무싫음 ㅠㅠㅠ'], ['저도 설겆이요..진짜진짜 싫어요. 청소랑 빨래는 어떻게라도(로봇청소기)하겠는데 설겆이는  식세기있는데도 진짜 하기싫어서 미뤄둡니다.ㅎㅎ'], ['진ㅁ자싫어욬ㅋㅋㅋㄴㅋ 식세기 있으면 편한가요?'], ['정작 식세기는 저보다 신랑이 더 자주써요. ㅎ 암튼 전 설겆이 NO!NO! ㅎㅎ'], ['집안일 너무 좋아하는데 ..빨래 넣는 건 정말 싫어요'], ['빨래넣기 ㅋㅋㅋㅋㅋㅋㅋㅋ 싫어하시는분 진짜 많으시네요'], ['그 묵은 물때낀거 청소하는거요\n그릇놓는 철제나 가스렌지 닦기등ㅜㅜ\n전 아직도 가스라 청소하기 귀찮네요..\n오히려 설겆이 화장실청소는 즐거운마음으로해요;'], ['아..... 저도 즐거운 마음을 갖도록 노력해봐야겠어요 ㅠㅠ'], ['음식하는거요..ㅋㅋ 다른것들은 그남아 할만한데..정말 음식하는거 싫어요.'], ['음식하는겈ㅋㅋㅋㅋㅋㄴ 맞아요 세상귀찮..'], ['집안일 자체가 너무싫어요 ㅋㅋㅋㅋ 차라리 야근하고 주말에 두탕뛰라면 그걸하겠어요 ㅋㅋㅋ 직장일은 해도 집안일은 너무싫어요.. 전업주부는 저랑 정말 안맞아요.'], ['맞아요 제가 딱 그거에요!!!! 선택할 수 있다면 일을 하겠어욬ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['설거지랑 손빨래요\n행주빨기 걸래빨기 어려워요'], ['손빨래요? ㅠㅠ 전 손빨래 해본적이.....\n행주도 일회용쓰고 걸레는 안쓰고 물티슈만 써요.. 그러고보니 저 진짜 게으르네여;;'], ['밥하는거 집안일이면 밥하는거요~'], ['아...... 그 큰 일이 있었죠.. ㅠㅠ'], ['설거지랑 음식하는거만 괜찮고 모든게 싫어요~설거지도 양많으면 싫어요 ㅠ청소가 제일 싫기는 해요 열심히 해도 티가 안나니ㅠ'], ['ㅋㅋㅋㅋㅋㅋ앜ㅋㅋㅋㅋ 설거지랑 음식만 괜찮으시다니 대단해요!!'], ['음쓰 처리요... 유일하게 못하는 집안일이에요..'], ['헉 진짜요? 전 그게 젤 쉽던데ㅠㅠ'], ['식세기는 설거지 짜증지수를 아느정도 카바가 되요.\n개수대가 그릇으로 넘치게 쌓여도 아 난 식세기가 있지? 하면서 원래 미루던겅 20프로정도 미루고\n갓 세척 하고 나온 따끈하고 코팅된 그릇들도참 예쁘고 이뻐요 ㅎ'], ['우와....... 식세기 질러야하나봐요 ㅠㅠ'], ['전 머리카락 치우기요...\n저나 딸 아이나 길어서 ㅜㅜ'], ['아... 전 청소기 슉슉 돌리면 끝나니까 차라리 그게 낫더라고요ㅠㅠ'], ['저는 빨래 개는게 제일 하기 싫어요.'], ['아.... 저는 그건 좀 낫더라고요ㅠㅠ 설겆이는 온몸이 다 힘들어요\n다리도 어깨도 손가락도 팔도 다힘듬 ㅠㅠ'], ['저는 화장실청소랑 바닥 물걸레청소요ㅠㅠ 청소기는 슥슥 쉬운데 물걸레랑 스팀하려면 진짜 맘먹고해야하는것같아요ㅠㅠ 설거지도 싫었는데 요새는 넷플릭스로 워킹데드 보면서 설거지해서 시즌1부터 지금 7까지 왔네요 어느새 설거지 즐기고있어욬ㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋ우왘ㅋㅋ 저도 노래부르면서 하렵니다ㅠㅠ'], ['남편ㅅㄲ 밥 차려주는거'], ['흐억 ㅠㅠ'], ['저는 빨래 개키고 정리하는 거요 ㅠㅠㅠㅠ 하아 정말 싫어요 ㅠㅠ'], ['ㅇ ㅏㅠㅠㅠㅠ 저는 설거지랑 빨래 선택하라하면 빨래요ㅠㅠ'], ['저는 설거지랑 화장실 청소요 ㅜㅜㅜ'], ['저랑 같으세여 ㅠㅠ'], ['저도 설거지.. 제일 효자는 세탁기... 근데 식세기 사도 냄비며, 설거지 품목들이 너무 다양해서 넣는것도 일이고 또 꺼내는 것도 일일것 같아. 고민중이네요.'], ['아.. 하긴 넣고 꺼내기가 ㅠㅠ 저도 고민해봐야겠어요'], ['전 빨래개는것까진좋은데 이걸 각각 가져다정리해놓는게 너무싫어요ㅠ'], ['ㅠㅠ 전 후다닥 갖다놓으니 괜찮던데ㅠ 설겆이는 너무 오래걸려요 ㅠㅠ'], ['남편시키 밥차려주는거요,.  흥칫뿡..ㅠㅠ']]
    
    7533
    집안일 중에 뭐가 제일 싫으세요? 아침먹고 커피한잔 때리고.. 가장하기 싫은 집안일 화장실청소를 해치웠네요. 올여름 유난히 습해서 그런지 화장실에 곰팡이도 더 잘생기는거 같네요. 바닥에 물때도 너무 보기싫고.. 화장실청소 한번 했다고 아주 온몸이 땀이라 샤워까지 하고 나니 왠지 엄청 큰 숙제 끝낸거 같아 뿌듯합니다. 사람에 따라 특히나 싫어하는 집안일이 있을텐데 전 화장실청소가 너무 귀찮고 싫으네요. 안하면 제일 티나는곳도 여기고.. 여러분은 어떠신가요? ​
    
    [['빨래 개는거랑 설거지가 막상막하인데요새는 애들 빨래 때문에 개는거요ㅠ'], ['저는 화장실청소가 제일 좋아요 ㅎ\n싫은건 빨래접어 넣는거요 ㅎ\n\n요즘은 그냥 걷어입으라해요 ㅎㅎ'], ['전 빨래랑 청소기요 ㅋㅋㅋ 그래서 빨래랑 청소 담당은 남편,,ㅋㅋㅋㅋ화장실은 그냥 씻으면서 쓱싹쓱싹 하면 기분 좋아서 제가 해요 ㅎㅎ 설거지도,,, 그러고보니 설거지랑 화장실청소는 때를 문질러서 벗겨내고 뽀드득거리게 한다는 공통점이???'], ['음쓰버리기요ㅠㅠ'], ['밥하고반찬만들기요\n삼시세끼 정말 스트레스네요 ㅠ'], ['다 싫어요ㅠ'], ['22222 저도요ㅠ 끝도 없는 반복ㅠ 티도 안나고ㅠ'], ['33333333극공합니다'], ['ㅋㅋㅋ저도 다싫어요!!!'], ['55555 격공하고 갑니다!!!!'], ['아 생각해보니 맞네요..ㅋㅋ'], ['ㅋㅋㅋ저두요~~~~~!!!!!'], ['생각하고 분류하고 선별하다고 집어치웠어요.\n저두 다  싫어요 \n8 8 8 8 8 8 8 8  8 8 8'], ['화장실청소 설겆이 보다 더 못하겠는게 정리에요ㅠㅠ'], ['전 화장실 호스사이에 낀 때 벗기는 청소, 주방 배수구청소요..ㅜㅜ'], ['저도 샤워기줄 곰팡이에 되게 민감했었어요..근데 다이소 가면 이런 일자 호스 있는데 이걸로 갈고나서 스트레스에서 완전해방요ㅋ'], ['이거 있음 때가 안껴요.!!?'], ['이게 그냥 일자호스라~ 그냥  가끔 쭉 닦음되요!'], ['좋은정보감사해요♡♡♡♡'], ['설거지요'], ['빨래 관련 모든거요ㅠㅠㅠ'], ['요리 빼고 다 싫어요 😱'], ['음쓰처리, 배수구 청소요. 우웩...ㅠㅠ'], ['화장실청소랑 다림질이요'], ['밥하는게 제일 싫어요..'], ['저도 화장실청소요... ㅠ'], ['빨래 걷어서 정리해넣는거요ㅜㅠ'], ['옷장종리 ㅠㅠㅠ 빨래 개는거요ㅠㅠ'], ['다시러요 ㅋ'], ['진짜 하나도 안빼놓고 다 싫어요 ㅋㅋㅋㅋ'], ['빨래 개는거랑 갠거 서랍에 넣는거요ㅠ 빨래 개는 기계는 언제 나오나요'], ['끼니마다 요리하는거요! 애 낳기 전엔 베이킹도 하고 취미로도 요리했는데 엄마가 되서 하는 요리는 마감시간 쫓기듯 하는거라 재미는 하나도 없고 스트레스만 엄청 받아요!!! ㅠㅠ'], ['다 싫지만... 밥은 안 할 수 없기에 일단 밥하는 게 제일 싫네요. ㅎㅎ'], ['화장실 구석구석 닦는거랑 싱크대 수채구멍닦는거랑 청소기 로봇청소기 먼지통 씻고 먼지닦아내는거랑 건조기먼지통비우고 스타일러물채우고버리고 먼지통비우고 이런 잡스러운것들이요ㅋㅋㅋ'], ['2222\n이런 소소하지만 남편은 절대 못해주는..가르쳐도 못할 항목들! 제가 꼭 해야해서 더 싫은듯요 ㅋㅋㅋ'], ['맛아요 백퍼공감 ㅋㅋㅋㅜ'], ['전 다 싫지만설거지랑 음식물 쓰레기..ㅠㅠ음식물쓰레기는 아무리해도..가끔 헛구역질 나요😅'], ['저도 밥하기끼니때마다 밥하느랴 진 빠져요아....영화속에서 나오는 우주인들이 먹는 알약캡술같은거 있음 좋겠어요...'], ['음식해먹이는거요 😭'], ['22222222'], ['설겆이가 너무너무 싫어요\n그래서 건조기는 안샀는데 세척기는 샀어요ㅠ'], ['설거지랑 빨래요 ㅠㅠ'], ['전  집안정리요ㅜㅜ 안쓰는물건 처분하고 정리해야되는데 정말 손놓고있어요ㅜㅜ'], ['요리만 좋아요..나머진 다 싫어요.ㅋㅋㅋ'], ['설거지....가만 서 있는게 싫어요 ㅎㅎ'], ['화장실청소요 ㅋㅋㅋㅋ'], ['다다다 ~~~  다 싫어요.ㅜㅡㅜ\n결혼이란걸 하면 안되는거였어요.'], ['화장실 청소가 제일 좋다는 분이 많으셔서 깜짝 놀랐습니다!ㄷ ㄷ  저는 청소는 뭐든지 다 싫고 비교적 힘이 안들어가는 정리정돈이나 빨래가 그나마 나아요 ㅠ'], ['222222  딱 제 마음이에요!!'], ['전 음식만들기요.. 요리똥손이라..!!ㅠㅠ'], ['다 싫지만...제일 싫은 순서1. 음쓰->음쓰기 샀어요^^2. 화장실청소3. 설겆이4. 물걸레 청소5. 청소기6. 빨래 널기7. 빨래 접기8. 그 외 집안일...다 싫어요~~~^^'], ['정말 다 싫지만 그중에서도 시댁서 설거지요ㅠㅠ'], ['빨래요.'], ['전 물걸레질이요ㅜㅜ 물걸레 청소기 따로 샀는데도 싫어요. 걸레 빠는게 넘 싫어서 그런가 싶기도 하네요. 하고 나면 너무 좋은데 내가 하는건 너무 싫어요'], ['그닥 싫은게 없었는데..\n요즘은 다 귀찮음이요ㅡㅡㅡㅠ\n'], ['화장실청소요ㅠㅠ'], ['걸레질이요..ㅋ']]
    
    7572
    설문조사요(남편들 집안일+월급) 한국사림들은 각자 관리하거나 남편이 많이 하더라구요그럼 설문조사 들어갑니다1.밥 주/월 몇회 한다2.집청소 주/월 몇회 한다3.각자 돈관리&월급 준다&내가 관리&남편관리저희는 둘 같이 일하고밥은 제가 주로하고청소는 남편이 주로 하고돈관리는 제가 했는데이분달부터 갑자기 반반 하게 되었네요궁금한 밤이네요~
    
    [['우리집은 1번 2번 해다없네요. 3번 급여는 내가관리 남편용돈받아쓰기요~~'], ['주부신가요?일하시나요?'], ['일다녀요. 맞벌이 입니다~~'], ['그럼 맞벌이 하시면서 밥도 청소도 애들 케어도 전부 님이 하시나요'], ['앗~애기는 아직이구요. 같이 일하면서 밥하고 청소는 제가요. ㅜㅜ.'], ['아~~네^^ ㅎㅎ즐거운 밤 되세요'], ['밥  ×  (하핫,민망하지만 나도 주1회  할까 말까 입니다 ㅎㅎ  )\n\n집안일×(쓰레기  치우기○)\n돈  (제가  다 관리^^)\n\n저희집은 요래요 ^^'], ['위위님은 일하시니까요~^^'], ['주말부부에요  연애때부터 월급은 다 주는데  밥상에 초차이 4.5가지 올려야 집안일 해요  😂'], ['그래도 하긴 하네요~돈관리 다들 와이프가 하시네요~초채 올리고 시키면 하니까 얼마나 좋아요 ㅎㄹ'], ['밥 주말만  청소 주말만  우리신랑은 먹어주는거   돈관리 내가'], ['밥하구 청소 주말만 언니 하구?'], ['우린 시엄니 계시니까  밥하고 청소는 주말만 내가 하지  평일엔 시엄니하시고'], ['아~~그치에 부럽슴다'], ['맞벌이 엿다가 육아휴직중이요 밥은 실랑이 청소는 제가 돈관리는 각자요'], ['처음부터 각자 하셨나요?'], ['주부고요\n남편 퇴근후 애샤워 놀아주기 담당\n쉬는날 아침밥은 남편이해요 \n돈은 내가관리해요'], ['와 양호 남편이네요 아직까지 남편관리 하는분 못봤네요'], ['저희는 가사 분담해요.   남편이 요리하는거 좋아해서 요리만 하구요. 요리를 제외한  모든 가사는 제가 해요. 가끔 도와준다고 나서도 남편이 했던데는 꼭 제 손이 한번 더 가야돼서 아예 안 시켜요. 둘이 맞벌이라서 돈관리는 각자 하는데  10원이라도 돈 쓸일 있음 서로 상의해요. 7년간 같이 살면서 신뢰를 깬적이 없어서 믿고 맡겨요.'], ['좋네요~저희도 아예 각자 해야겠어요 이참에 근데 각자하면 반반 뭐든자 분담하나요?'], ['공과금 관리비 보험 생활비 등등 모든 지출은 남편카드로 해요.  제 월급은  저축하구요.'], ['그렇군요.남남이 사는게 참 어렵네요'], ['저희는 신랑이 알아서 챙겨먹어요. 저는 애셋반찬만 신경쓰면  끝이네요.돈은각자알아서 관리해요.임대료.관리비.은행대출 신랑이다 내구잇구요.머리아파서 돈관리안해요.ㅎㅎ'], ['알아 챙겨먹는게 좋네요~ 각자 관리하고 공과금 전부 남편앞인가요? 반반이 아니고?'], ['큰거는신랑이다납부해요.저는 핸드폰요금. .인터넷요금 만내구요.ㅎㅎ'], ['와~완전 좋네요.ㅎㅎ'], ['저는 주부인데 밥은 안하구요 아니 못하구요 ㅋㅋ 솜씨가 없어요 대신 청소는 백프로 제가 해요 돈관리는 같이해요 얼마.들어오구 얼마 나가구 같이 투명하게'], ['그럼 그 돈을 누구 통장에 넣어서 같아 관리하시나요?'], ['신랑 월급통장은 제가 가지구있구 신용카드도 신랑명의지만 내역서가 제 핸드폰으로 들어와요 적금통장은 신랑명의지만 제가 관리하고있구요'], ['아~~그러면 돈관리는 님이 하시고 투명하게 하시는거네요'], ['전업주부 주말빼고 평일하루아침밥한끼만해요 저녁은주로 배달많이시킵니다 주방 아기자는방빼고 나머지공간은 신랑청소담당 돈관리는제가합니다'], ['와 좋으시네요 부러워요.돈관리 대부분이 와이프가 하시는군요'], ['1 밥 안한다.\n2 청소 주 1회??그냥 청소기만     돌린다 쓰레기버리는건 남편전담\n3 좀 남기고 준다 울집 돈관리가 제가\n\n이렇습니다.\n맞벌이입니다.\n'], ['밥은 그럼 남편 전담인가요? 그 좀이 월급에 몇프로정도 되나요?'], ['10프로정도 남기는거같아요...실은 \n저도 요번에 남편월급2번째 받아봅니다. ㅠㅠㅠ\n밥은 제가다 하죠...\n\n'], ['아 진짜요 ㅜㅜ 원래는 각자였나요?'], ['네...줫다가 도로 다 가저갓거등요...근데 이젠 안주기로 햇습니다.'], ['부부가 사는게 참 어려운 일인것 같아요'], ['네...서로서로 맞춰가면서 살아가야하는같아요...\n물론 상대방도 그리 생각하겟지만요..\n'], ['그런것 같아요.^^남보다 못할때두 있지만 그래도 가족이죠'], ['맞벌이부부이구요!남편청소 다해요!(화장실청소는 내가.빨래는 서로서로 어느정도차면 알아서~식사준비는내가~설겆이는 신랑이~쓰레기는 집에있는시간이많으니 제가~버려요!돈은 신랑이 관리해요!저는 신랑50만주고나머지는 내가알아서써요하지만 신랑카드 들고다니면서필요할때써요! ㅎ.ㅎ돈관리안하니 좋아요!'], ['각자 하자고 해서 그러자고 했는데 궁금해지더라구요~모범 신랑이시네요~'], ['맞벌이인데  밥은 매일 내가 하고  집안일 내가 더 많이 하고  남편은  쓰레기버리고  분리수거하고  아주가끔  빨래널고  아주가끔 설거지해주는정도요   아주가끔  애들  치카해주고  씻기고  그러고보니  애도 거의  내가  혼자 다 키웠네요   \n월급은  남편이랑  나랑  2배 차이나고   돈관리는  제가해요    용돈  쬐꼼만 줍니다    \n그분이 힘들게 일하는지라  이정도로 만족합니다\n'], ['저희랑 비슷하네요 ~부지런하시네요~돈관리 각자하면서 돈도 벌면서 맞춰 간다는건 참 어려운갓 같아요'], ['저희는 외벌입니다\n 전업주부라 밥은 제 담당\n청소. 분리수거ㆍ 음식물쓰레기  \n다 제가 해요~ 한달에 한두번 남편이  청소 도와줘요\n돈은. 공과금 직원들 월급주고 나머지. 제. 주는데ㅜㅜ 몇달째\n못받구ㅜㅜ. 코로라땜시 마이너스통장 되가네요\n돈은 반반씩 관리하자구 하네요\n\n'], ['반반 하는 집도 몇분 보이시네요~반반하는것도 나쁘지 않은것 같아요~코로나가 얼른 지나가서 정상으로 돌아오길 바래요'], ['집 청소 밥 분리수거 애목욕 다 합니다맞벌이구요 심지어 남편 도시락도 쌉니다돈은 적금으로 들어가는건 같이 입금하고 지출이랑 각자관리 대신에 다 알려주기입니다'], ['님이 다 하신다구요? 남편은 손 까딱 안하나요?대단하신것 같아요.'], ['ㅋㅋ네 뭐 따지면 머리 아프니 그냥 다해요'], ['와~~멋지십니다~^^'], ['1번 2번은 시간되는 사람이 하기 내가 백수되고부터는 내가 많이해요\n3번은 재작년까지 내가하다가 이젠 반반해요~'], ['허브님네도 각자 하시는군요 각자하면 좋은점 뭐가 있어요'], ['좋은점은 남편이 마누라 눈치안보면서 돈쓸수 있어서 기 살았다? ㅋㅋㅋ\n그동안 내가 너무틀어쥔 같아서 좀 반성햇어요 \n조만간 또 뺏어와야죠 ㅎㅎㅎㅎ'], ['저도 틀어쥐고 있었지만 아껴쓰고 해서 적금도 들고 했는데 갑자기 각자 하게 되었네요.'], ['맞벌이입니다 밥은 주5회 신랑이\n거의하고 주말만제가 해요 \n청소는 제가 더 많이 하는편이고 \n쓰레기 빨래등등 가끔씩 신랑해줘요\n결혼초에 돈관리 맞벌이 아닐때\n제가 했는데 돈이 안모여지는거에요\n그래서 신랑하라고 넘겨준지 십년\n각자관리해요 모든공과금 등등은\n신랑이 많이 버니까 냅니다'], ['좋은 남편이네요~각자 하면 더 많이 모아지나요?'], ['제가 버는건 돈이 안모여요 사정상\n친정부모님병원비로 지출이 많아요\n그래도 신랑알아서 적금도 들고 하더라구요'], ['좋은 남편이네요~좋은 점은 그냥 각자 편하게 쓰는거밖에 없는것 같아요.'], ['맞아요 돈쓰는거에  각자터치를\n안합니다 근데 돈관리는 한사람이(여자가) 해야 모인다는 \n주변분들 이야기가 많더라구요 \n전제가 헤퍼서 돈을 못모아요 \n팔자에 돈이 안모이더라구요 ㅎ'], ['저도 아글타글 혼자 애 둘 대리구 육아할때부터 모았는데 갑자기 각자라니 허무하기도 하고 시원섭섭하네요'], ['저흰 남편 월급 통장 제가 관리하고 남편 용돈 한달에 40만원씩 줘요.제 월급은 얼마 안되서 남편한테 대충 한번 예기해줬어요.\n밥은 일요일빼고 제가 하고 남편은 가끔씩 일찍 오는 날 반찬하고 설거지,쓰레기 버리기는 매일 남편이 하고 세탁기랑 건조기는 제가 돌릴때 많고 옷 개는건 같이 할때가 많아요.\n저희는 돈 쓰는거 각자 터치 안해요.\n오로지 집이나 차 정도의 큰 거 살때만 상의 그외엔 알아서 해요.\n말수가 없는 남편이지만 애들 커가면서 점점 농담도 잘 하고 말이 많아졌네요.\n전 제맘대로 하고 살아와서 만족한데 딱 한가지 넘 효자라 시엄니 말씀이라면 다 해주고 저 몰래 해주고 돈 줄때가 있어서 그게 좀 서운할때가 있네요.\n뭐든 완벽할 수 없으니 그냥 행복하다 생각하고 살아요~~ㅎㅎㅎ'], ['와우~ 만족하시면 되는거죠~너무 잘사시는것 같아요^^서운한거야 다 있죠~다투어도 바로 푸는게 부부잖아요~'], ['맞벌이.\n밥은 거의 제가 해요.\n집안일은 남편이 주말에만 도와줘요. \n평일엔 설겆이 정도.\n돈관리는 제가 하고 남편은 용돈 받아써요. \n결혼초 남편이 관리햇는데 마이너스가 되서 제가 넘겨받은지 6년됐어요..\n신랑통장. 카드. 카드문자 .적금통장 등등 모두 제폰으로 와요. \n\n투명하게 관리해요. 남편도 인증서가 있으니 들어가보면 다 보이니가 숨길수도없어요 ㅋ'], ['저희도 십여년 제가 관리했었어요 근데 갑자기 반반 하게 되어 궁금하더라구요 좋은 점 뭐가 있는지 해서요'], ['전업주부면 이해갑니다~남자들이 힘들게 돈 벌어오니까요~성격이 꼼꼼하시면 또 해야 편하더라구요'], ['전업주부. 독박육아 애가 9개월이예요.남편 주말만 집에오고,평일엔 다 제가해요.\n이유식도 집에서 해먹이구요. ㅠㅠ\n대신 남편 주말이나 휴식하는날에 집에오면  제가 힘들다고 다 남편이 해줘요. 쉬는날 음식은 어쩌다 제가 할때있고 남편이 하는데 맛없어서 대부분 배달음식이요.\n돈은 일체 다 제가 관리해요. 적금도 제통장으로, 남편 통장 카드 다 제가 가지고있어요.입출금 문자도 저한테로 오구요.남편 용돈 없음.대신 신용카드 줘요.\n'], ['완전 착한 남편이네요~힘드시겠어요 독박이라 저도 일년 혼자 애둘 독박 하면서 주말에 남편이 오면 그래도 힘들어한다고 상다리 부러지게 차려놨었는데...추억이네요~^^'], ['저희 집은~1. 밥, 아침은 신랑이하고 저녁은 제가해요~     주말에는 신랑이 다 해요~2. 청소, 평일은 제가하고 주말에는 신랑이해요~    (쓰레기 치우기는 다 신랑이함)3. 돈관리 제가해요.전 지금 집에서 애만 보는 중이예요~'], ['저도 몇년째 맞벌이 하다보니 요즘 자꾸 회사랑 권태기 오면서 쉬고싶네요.집에서 주부 하고싶은 맘이 굴뚝 같아요'], ['그러시면 단분간이라도 집에서 시간 좀 보내는것도 나쁘지 않다고 생각해요~돈벌어서 다 행복하게 살기 위해서인데 회사 다니면서 기쁘지 않으면 나중에 병생겨요~남편분이랑 상의하셔서 잘 결정하세요~'], ['네~^^ 감사합니다~편안한 밤 되세요'], ['한국인남편,맞벌이~밥은 거의 제가합니다~ 남편이 한건 맛없어서요 ㅋㅋ월급은 남편이 용돈만 빼고 다 줍니다~^^청소는 시간되는사람이 해요~ㅋㅋ제가 더 많이 하긴하죠~'], ['그렇군요~한국인 부부들은 주위에 보니까 남편이 하는 분들이 많더라구요~~용돈은 몇프로 받나요'], ['15만원이 남편용돈입니다~술 담배 일절 안해서 돈이 안 필요해요~ㅋㅋㅋ\n대신 주유비,통신비,다달이 고정비용은 다 내줍니다^^'], ['저희도 그랬었는데 ㅎㅎ 각자 하자네요.주위에서 돈관리 다 각자한다고 하면서 ㅎㅎ'], ['경제권은 여자가 맡아야죠~ㅋㅋㅋㅋ'], ['ㅎㅎㅎ 싸우기 귀찮아서 그러라 했어요'], ['정해진게 없이 흐름에따라 살아요. \n전업주부지만 남편이 프로젝트 적고 안바쁠때는 애도 매일 씻기고 놀아주고 설거지도 해주고 주말에는 매트까지 들어 대청소까지 해줘요. \n프로젝트도 많고 바쁠땐 집에서 밥먹고 아무것도 하지말라고 제가 다 알아서 하는편이예요. \n용돈은 서로 없구요. 신용카드 가족카드 두장 만들어서 각자 갖고다니면서 사용하고 현금이 필요하면 제가 찾아다 주는편이예요. \n돈관리도 그때그때 상의해서 하는편이구요~'], ['협의가 잘 이루어지는 집안이시군요.흐름에 따라 사는게 좋은것 같아요~서로서로 배려하면서요'], ['저희는 외벌이\n코씨 오기전에 남편 자주 중국출장가고 퇴근해도 저녁 8시반에 집 오니 \n독박육아에 밥, 집안일 전부 다 제가해요 밥은 점심 빼고 하루 2끼 차려주고 돈관리는 제가 관리하는거 같은데 다 투명해요 통장 하나로 카드 두개 개설해서 쓰니 출금문자는 나한테로 오고 남편은 많이 쓰는편 아니라 각자 알아서 쓰고 큰돈 나갈때만 서로 알려줘요\n남편이 하는 일은 딱 쓰레기 버리기 분리수거도 할줄 모름 제가 다 해요 요즘은 설거지 전담해요'], ['와우 전업주부신데 설겆이 전담 쓰레기 버리기면 잘하는것 같아요~돈관리는 안나님이 하시는데 투명하군요~갑자기 각자 쓰자고 하면 각자 쓰시겠어요?'], ['와~ 설겆이와 쓰리기 버리는거만 해도 잘한다구요? ㅎㅎㅎ 다른 집 남편들 보면 너무 부지런해서 부럽던데 ㅜㅜㅜ 저는 각자 돈을 안쓸거 같아요 제가 돈을 많이 벌면 몰라도'], ['제 주위에 분들은 암것두 안하는 남편두 많더라구요.우리 해피맘 남편들이 잘하시는것 같아요.주위에 안하는 영향 받아 남편도 하던것이 안하고 돈관리도 각자 하자로 되었어요'], ['저는 주위에 아주 가정적이고 집안 일 요리도 잘 하는 남편들 많이 봐서 그런지 저는 왜 이리 남자복 없나 싶었어요 요번에 시어머님까지 저보고 남편 교육 시켜라고 해서 시댁에서 오자마자 오랫동안 안하던 설겆이 전담하기로 했어요 차근차근 분리수거 화장실청소도 시킬려구요 ㅎㅎㅎㅎ근데 돈 관리 각자하면 돈도 따로 모아요? 왠지 돈까지 각자하면 멀어지는거 같아서 너무 서운해요'], ['맞아요 주위에 하는 사람들 많으면 또 따라가게 되더라구요.저도 이번이 처음이라 모르겠어요.그냥 알아서 쓰고 터치하지 말자에요.각자 쓰는것도 나쁘진 않을것 같아요.'], ['전-가정주부 남편-한국사람 \n\n밥-(제가해요.남편은 주말아침에 한번)\n청소-남편(분리수거.음식물.화장실포함)\n돈-제가 관리해요'], ['대부분이 반반 하면서 돈관리는 와이프가 하네요~~'], ['밥 ( 주로 내하지만 가끔 남편은 반찬만 할대 잇어요 ) 집안일 ( 설거지  빨래널기 가끔 해주고 쓰레기는 남편몫돈 ( 집에서 놀지만 내가 관리 ㅎㅎ )'], ['부지런하시네요~대부분 울집 남자랑 비슷하거나 더 잘하는데 울집은 그렇게 생색내네요'], ['밥은 항상 제가 하고 \n\n청소 (평일은 남편,주말은 제가 대청소)\n\n돈은 제가 다 관리해요...'], ['오~일하시면서 대단하시네요~~'], ['밥은 상황에 맞게 서로 하고\n집음 제가 거둬요\n돈은 제가 관리 하고 적금 들어놓고\n남편 카드엔 달마다 백만원씩 넣어둬요'], ['와 통이 크십니다 달마다 용돈이 그럼 백만이네요~~'], ['용돈에 그기 카드에거 보험이랑 빠져요 그래거 더 넣어줘요'], ['아~~그러시군요^^'], ['맞벌이일때도 밥 일년에두번?쓰레기  일년에 두번정도?돈은 터치않하는 사람  쓰던말던 지는 용돈만 딱.'], ['맞벌이하면서 안도와주면 힘듭데다.용돈은 몇프로 줌까'], ['우린딱담배살것만15만 현금쓰구  자기취미나  그런데쓸때면 카드로 일체쓰니 달 30-40은웃고쓰는같어\n힘드나머나 말도아니지'], ['남자들은 참...같이 살림하고 가족을 꾸려나가는건데...상전처럼 살구...그렇다구 남들처럼 월 오백씩 주므 떠받들구 살텐데 개뿔두 없으메 큰소리나 치구...에휴']]
    
    7596
    제일 하기싫은 집안일이 뭔가요? 저는 다림질이요..너무 하기 싫네요.남편이 말라서 티 입으면 없어보여서 남방을 많이 입어요.여름이라 더 자주 갈아입으니...너무 힘드네요.그렇다고 제가 자주 다려준건 아니예요.지금 일주일도 넘게 모아놓은거 다리려고 준비하는데 너무 하기 싫어서 또 이렇게 하소연하고 있네요.다림질을 하지 않았어도 쌓인 빨래를 보고 있으면 다림질 한것처럼 피곤해요.다른분들은 다림질 어때요? 쉽게? 즐겁게? 하는 노하우가 있을까요?
    
    [['화장실청소요'], ['222222 화장실중 변기 닦는거요..'], ['화장실청소....썩 즐기지는 않지만 샤워할때 찬물에서 따뜻한물로 넘어가는 동안 그 물 아까워서 변기에 뿌리고 솔로 문지르고 해요. 자주하니 그렇게만 해도 괜찮더라구요'], ['저도 그렇게 청소해요. ㅎㅎㅎ'], ['세탁소에서 다림질만 해주기도해요 근처 세탁소에 문의해보세요ㅋ 저는 음쓰버리기...'], ['세탁소...오늘은 그냥 다 들고가버릴까요? 자꾸 게을러지고싶네요...\n음쓰는 외출할때 습관처럼 들고나가요..'], ['다림질 저도 세상 귀찮은 것중 하나예요\n전 그래서 잔체크무늬 셔츠 주로 사요\n그건 잘 안다려도 티가 많이 안나서요ㅎ'], ['그런가요? 잔체크  ㅎㅎㅎ 요새 하필 유행원단이 린넨이라..다리기도 더 힘들어요'], ['그래서 전 린넨 안사요ㅋ\n관리 귀찮은건 아무리 예뻐도 패쓰'], ['욕심이 앞설땐 그 노동까지도 감수가 되는데..막상 빨고나면 후회해요ㅜㅜ'], ['식사준비는 할만한데 뒤설거지가  하기싫어요'], ['사람마다 다 다르긴 하나봐요. 저는 음식준비가 스트레스요..생각해서 음식해야하니까요ㅜㅜ'], ['다 싫어요 ㅎㅎ'], ['우문현답이네요^^'], ['마쟈요.다림질진짜시러요ㅠ'], ['꿀팁은 진정 없는거지요? ㅜㅜ'], ['빨래다게어놓고 옷장에 갔다넣는거요....ㅋㅋㅋ왜그럴까요정말~~;;;'], ['이것도 싫어요 ㅎㅎㅎㅎ'], ['2222222222저두요 ㅋㅋㅋ'], ['저도 다 싫어요 누가 해주면 좋겠어요ㅎㅎ'], ['다 싫은게 정답이네요^^'], ['스팀다리미로 샤샥~'], ['부지런하시네요 ㅎㅎ 다리미는..코드 꽂는것부터가 싫어요 ㅎㅎ'], ['그것도 너무 싫어요 ㅜㅜ\n그래서 저는 욕조를 없앴어요'], ['헉 다림질 집에서 하시다니ㅋㅋ저는 대충 옷걸이에 걸어두는데.....\n전 청소기돌리고 바닥닦는거요ㅠ'], ['그러게요. 집에서 하지 말아야할것을 제가 굳이 하려나봐요. 청소기는 그나마 요새 로봇이 돌려주고, 바닥은 안닦아요 ㅎㅎ'], ['빨래개기요 ㅋㅋㅋㅋ'], ['너무 싫을땐 온식구 다 불러요.\n각자 옷 개서 가져가라구요^^'], ['설거지 젤싫어요 요리는 하겠는데 설거지 너무싫고 ㅜㅜ 빨래 개는건 좋은데 제자리 찾아넣기 싫어요 ㅋㅋ'], ['ㅎㅎㅎ 저는 차라리 설거지가 더 좋아요^^ 근데 빨래 갖다넣는것도 무지 귀찮아서~~ 걸레처럼 거실을 몇날며칠을 뒹굴어야 제자리에 들어가요^^'], ['청소 끝내고 걸레빨기요.\n정말정말 귀찮고 싫어요.ㅜㅜ'], ['꺄아~ 이거 정말 싫어요 ㅎㅎㅎ \n걸레 물에 담가놓으면 한달은 가나봐요 ㅎㅎ 그래서 빨기 더러워서 버려요 ㅠㅠ'], ['전 걸레질이요ㅠㅠ'], ['걸레질..그래서 안하고 살아요 ㅜㅜ'], ['설거지ㅠㅠ 너무 싫어요\n화장실청소가 좋음ㅋㅋㅋㅋ'], ['어째요..ㅜㅜ 설거지는 과장 조금보태면 하루종일 해야하잖아요...ㅜㅜ'], ['구래서 신랑담당이에요 ㅋㅋㅋㅋ'], ['우와~~~~ 좋으시겠어요. 저희 남편 담당은 놀고먹기예요 ㅎㅎ'], ['제가 차리면 치우기 담당입니다 ㅋ 주거니 받거니 해야죠!!!'], ['걸레질이요 ..그보다 싫은건 닦고 나서 걸레빨기..'], ['걸레질을 안해서 바닥이 끈적끈적해요. 청소기만 겨우 돌리고 살아요^^;;'], ['밥하고 차리고 치우는거 너무 싫어요~ㅜㅜ 죽겠으요ㅜㅜ\n다림질은 재미난 드라마 틀어두고 보면서 하면 그나마 낫더라구요~'], ['저랑 같으시네요~~ 그나마 빨래개기, 다림질은 티비라도 틀어두고 하려고 해요. 근데 그래도 다림질은...하기전부터 보기만해도 힘들고..땀이 미리나요 ㅎㅎ'], ['화장실이요 ㅋ'], ['샤워할때 그때그때 해보세요. 그럼 조금 덜 힘들어지는듯 해요. 제가 대충해서 그럴수도 있지만요..ㅎㅎ'], ['흐흐 저는 물곰팡이 진짜 ㅠ 아무리 깨끗이 해두 물곰팡이는 어쩔수 .매번 없애고있어요 ㅋ'], ['아주 깔끔하셔서 그렇군요. 저는 대충이라 ㅜㅜ 청소는 이렇게 해야하는데..또 반성하고 갑니다'], ['아니에요 이 성격땜 더 힘든거같아요 ㅋㅋ'], ['요리요~ 혼자 살면 굶어 죽을거 같아요ㅠ'], ['저도 요리도 넘 싫어요. 보람이 안느껴지고 시간낭비같아요..'], ['저도 요리요ㅋㅋㅋ\n다른것도 그닥이긴 한데 요리하고 상차리는거 싫어요ㅜㅜ 요리를 못해서ㅋㅋㅋ'], ['저는 딱 요리만 싫어요.\n설거지 및 상차리는거 환영입니다 ㅎㅎ'], ['화장실청소랑 설겆이요'], ['알고보면 결국은 집안일은 다 싫은거였어요....ㅎㅎ'], ['ㅎㅎ맞아요'], ['🤭환풍기청소 주방청소요 ㅜㅜ 매번 반찬하고 닦는데도 기름때는 어쩔수없나봅니다'], ['이것도 힘들죠...안닦여요...땀나요 ㅜㅜ'], ['걸레빨기요:: 걸레는 결혼시작부터 남편담당이에요 ㅋㅋㅋㅋㅋㅋ 다림질은 제가 안해여.. 신혼때 남방 하나 태워먹은 후로 안시키네요 ㅋㅋ 아웃핏터? 이거 광고보고 혹하긴 햇엇어요.. ㅎㅎㅎ'], ['오~~~ 고수이시군요. 태워먹기!! 좋은방법인데요?^^ 저는 이미 늦은것 같네요..'], ['빨래 개기요.... 빨래한거 널기까진 괜찮은데 개는게 너~ 무 싫어요... 개서 넣어놓기는 잘 하는데 ㅠㅠ'], ['그나마 빨래 개는시간은 유일하게 티비보는 시간이네요^^ 근데 그것도 하기 싫긴 해요. 벗고 다닐수도 없고 ㅜㅜ'], ['물걸레질요ㅡㅡㅋㅋ'], ['창틀청소요 ㅠㅠ'], ['정리정돈 ㅠ 재능이 없어요 ㅠ'], ['걸레 빠는거 제일 싫어요ㅠㅠ'], ['설거지 빨래 청소등등 넘많네요 ㅠㅠ'], ['음식쓰레기 재활용이요 ㅠ 음식쓰레기 버릴때 벌레들 너무 많이 튀어나와서 힘들어요'], ['요즘은 애들 삼시세끼 챙기는게 젤 힘드네요ㅠㅠ어여코로나가 없어졌음 해요'], ['빨래 개기는거용ㅋㅋㅋ\n음식하는거요ㅋㅋ요태기와서 요리시러요ㅠ'], ['댓글 보는데 진짜 다 싫으네요 ㅋㅋ'], ['저는 빨래개는거싫어요..ㅋ'], ['쓰레기버리기요...분류하고 정리하고 누르고 음식물은냄새에물에... 으....'], ['전 설겆이 하기싫은데 꾸준히 합니다ㅠ 유재석이 한 이야기가 생각나네요ㅎ'], ['음식쓰레기-하수구,변기닦기..근데 다 제가해요ㅡㅡ'], ['화장실청소'], ['청소기요 ㅋㅋㅋㅋㅋ'], ['밥 먹고 난 다음부터 해야하는 모든일이 다 ...싫어요 ㅋㅋㅋ'], ['저두 화장실청소요 ㅜㅜㅜ흑'], ['저는 설겆이요ㅜㅜ 설거지옥... 돈 벌면 식세기부터 살꺼에요'], ['빨래개는건 그래도 나은데 방마다 갖다넣기요~~ㅋㅋ'], ['저도 다림질 세상귀찮았는데\n스타일러사서 돌린이후로 다림질안해요^^ 넘편해요ㅋㅋ']]
    
    7634
    일과 살림 중 뭐가 더 힘든거 같으세요? 이런 질문이 참 의미없는거 알고케바케라서 답도 다 다르것지요..​그냥 궁금해서요.. 남들은 어떤지도 알고싶고요..​저는 살림이 1000% 더 힘들어요..평생 일만하고 살았으면 좋겠어요..​그런데 제 팔자는 일과 살림을 같이 할 팔자네요..ㅠㅠ설거지, 요리, 빨래, 정리.. 이런게 너무 너무 싫어요..오늘도 세시간 집을 치우고 나서정말 기진맥진해요..​참. 이상하죠.. 10시간 노는건 안힘든데이런 일들이 너무 힘들고 지긋지긋해요..​육아는 사실 힘들지 않아요.아이들과 노는게 즐겁거든요..근데 집안일 때문에 아이들과 잘 못 놀아주고 신경질을 내고늘 기다려 기다려를 입에 달고살고요..​무엇이 문제라서 이렇게 매일 촉박하게 사는걸까요??저는 왜 여유롭게 살지 못하는걸까요?이게 다 집안일 때문이란 생각이..​그냥 내 일만하고 퇴근해서는남이 차려준 저녁밥을 먹고 남이 설거지할 동안 나는 아이들과 신나게 놀아주고내가 잘 동안 남이 정리해주고다음날 출근할 옷도 다려주고 준비해놓고아침에 일어나면 남이 해준 아침을 먹고나는 일나가고 남은 아이들 학교에 데려다주고..​그런일상을 살면 얼마나 행복할까 생각을 해봅니다.입주이모님을 쓰면 되것지만그럼 먹고 살돈이 없네요..ㅋㅋㅋ​세 시간 집안일 하고 푸념해봅니다..끝이 없어요.이제 분리수거하고 또 빨래하려고요.​세탁기가 6킬로자리라..독일사람들은 이불을 안빠는지..보통가정집에서 6키로쓴다하는데 정말 불편하네요..​지금 3번 돌리고 있어요..ㅠㅠ
    
    [['동감입니다ㅜ 육아휴직하다가 복직하니 살겠더라는ㅠ 밥도 남이 해주는게 맛나네요ㅎㅎ'], ['제 동료가 복직해서 점심먹다가 울었다고..ㅠㅠ'], ['저 이글에 완전 공감해요ㅠ 얼마나 소모적인가요? 공들여 밥하고 금새 먹고 또 치우고 닦고, 치워놓으면 또 금방 먼지 쌓이고... 누가 알아주는 것도 아니고 사람쓰고싶어요 돈만 많으면ㅠ 전 살림보다 일이 훠얼 좋아요.'], ['22222222'], ['진짜 티가 안나요. 티만 나도 의욕이 있을텐데여.. 정말 일이 훨 좋죠.'], ['저는 육아요ㅠ일은 정말 출산하러 가기전 몇시간전까지 일했는데.'], ['맞아요. 육아도 힘들어요. 구래도 전 살림보단 덜 힘들어요. 애가 줄거워하는거 보면 기분이라도 좋지. 집안일은. 흑흑.'], ['결혼하기 전 하루 16시간씩 일하기가 부지기수에... 3일밤 새기도 하고.... 출장 가서 찜질방 자고 진짜 엄청 고생하며 젊은 날을 보냈는데.... 왠걸.... 살림 육아는 정말 비할바가 아니여요 ㅠㅠㅠ\n게다가 아무런 결과물이 없다능요....'], ['맞아요. 진짜.. 강도높은 일과 살림육아와 비교해도 살림육아가 갑일듯 싶어요.'], ['아이들과 놀아주는게 힘들지 않다니. 멋지네요'], ['아이들과 놀아주면 저도 신이나서..ㅋㅋㅋ 살림하다 애들한테 짜증을 내는게 문제네요. ㅠㅠ'], ['전 집안일두 너무 귀찮구 하기싫지만 직장다니는것보단 나은거같아요 결혼후 일하는건 집안일,육아와 다 병행해야하니까요ㅠㅠ'], ['그니까요. 저도 워킹맘팔자인데 복직하면 어찌살까 싶어요. 이모님 주 2회불러도 그나마 잠시것지요.ㅡㅠㅠ'], ['육아요'], ['육아도 힘들죠..'], ['저도요 살림이 훨 힘들어요 뭐가 이렇게 해도해도 끝이없고 티가 안나죠??????ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['끝이 없죠.. 어쩔땐 먹고 치우고 먹고 치우고 동물같이 느껴질때도 있어요..'], ['살림.'], ['살림이 진짜 갑!'], ['저도 육아휴직 끝나고 복직하니 숨통 트이는 느낌이었어요 ㅠㅠ 몇시간을 살림해도 티도 안나고... 잘 하지도 못해서 흑흑'], ['동료들이 다 그러더라고요.  에봐줄 사람있는 동료들은 3개월 휴직후에 다들 복직하더라고요..'], ['저도 일하는게 훨씬 즇아요~~~ 남자로 태어났으면 정말 한자리 하고 싶네요 일중독되서'], ['그러게요.. 남자들은 얼마나 팔자좋은지 모를거 같아요.. 한때는 일도 힘들었지만 최강의 힘듬을 맛보니 일은 아주 거뜬한거였어요.'], ['저도 살림 너무 힘드네요\n병행하는게 진짜 너무너무 제일 힘들고...\n휴직했을땐 그렇게 힘든 살림이지만 하나만 하니까 할만했어요 \n물론 살림안하고 돈만 버는게 제일 쉬워요!!!\n'], ['맞아요. 살림안하고 돈버는게 제일 쉽고.. 병행하는건 어렵고.. 몸까지 상하고요..'], ['살림...육아...일....요즘은 몽땅 다 힘들어요ㅠㅠ'], ['다 해서 그래요.. 다..ㅠㅠ'], ['저도집안일이힘들어요 이번 휴가 1주일 정말 힘들었어요 ㅠㅠㅠ 독일에서 다른가족도움없이 집에서 아이들 잘 돌보시는거 정말 최고에요!!회사가면 몸은편하해여 근데 집은 몸이 너무힘들어요 ㅠㅋ'], ['워킹맘은 쉴데가 없죠..ㅠㅠ'], ['전 일이 훨씬 힘드네요. 살림이나 육아는 육체적으론 힘들지만 오히려 남는 것이 많다는 생각이 들어요. 내가 가족과 사는 공간을 이쁘게 꾸미고 맛있는 것 해먹고 싶네요. 아이들 사랑도 더 많이 해줄 수 있고요. 직장일이 자기 몫으로 남는 일이면 정말 좋은데 제가 하는 일은 그렇지 않아서... 돈 노예생활을 빨리 청산하고 아이들과 즐겁게 지내고 싶네요. ㅜㅜ'], ['사람마다 재능이 다른거 같아요. 전 센스는 1도 없어 치워도 집은 항상 거지같아서..ㅋㅋㅋ'], ['살림은 쌓이는 게 없는 느낌이라 허무하고 육아는 하나도 내 맘대로 되지 않아서 힘들도 일도 힘들지만 월급 보면 또 힘나더라구요'], ['맞아요. 육아나 집안일도 누가 돈주면 할맛이 날런지요..휴.. 집안일 끝이 없어요.'], ['엄마가 해주는 밥먹으며 회사다닐때가 좋았죠 ㅠ'], ['진짜 그때가 제일 행복했는데 그때도 힘들다고 투장을 부렸네요. 사람은 고생을 해봐야 그때가 얼마나 안힘들었는지 깨닫게 되는거 같아요.'], ['장,단점이 명확한 영역인데 모든 같이 병행하면 힘든 것 같아요.사실 같이 한다는게 불가능하죠ㅜ'], ['그말이 정답이네요.모든걸 다 같이 하믄게 불가능인데 사회는 그걸 요구하고 말이죠!'], ['일이 더 어려워요... 살림은 내 몸 힘들면 스킵도 하고 대충 하면서 핑계도 댈 수 있고 귀찮으면 배달음식 시킬 수도 있는데 일은 그게 안되서요...'], ['그렇군요.. 저는 365일 집안일이 힘든게 문제네요. 왜이리 힘든지.. 대충해도 다음날 또 힘들다는게.함정이에요. 흑.'], ['전 25살때부터 얼마전까지 빡센데서 일하다 주부됐는데 집안일이 훨 쉬워요.. 청소에 밥차리고 설거지까지 삼시세끼 다 합쳐도 하루에 네시간도 안하는듯한데 집도 깨끗하고.. ㅎㅎ 제가 맞벌이였을때 워낙 개판으로 해놓고 살아그런가 남편도 저도 요새집상태에 대한 만족도가 높아요...'], ['워킹맘이라면 일도 힘들듯요. 다 한다는건 사실 불가능한거 같아요.. 님 몸도 설피면서 즐기세요..!'], ['전 일이 더 힘들어요.. 스트레스가 심해서 마음의 병까지 들었어요. 컴퓨터 앞에만 앉으면 심장이 터질 것 같아서 ㅜㅜ 살림이 좋아요 ㅜㅜ'], ['저도 살림이 싫어요\n누가 매일 쓰레기 버리고\n빨래 개서 넣어줬으면 좋겠어요\n\n진짜 돈벌고 싶어요'], ['일 살림 다 힘들다 치면.. 일은 하면 월급도 들어오고 살림은 해도 티도 안나고 안하면 티 팍팍 나서요. 살림이 훨어씬 힘들어요.'], ['살림이요~쉴틈이 없어요ㅠㅠ 계속 반복  반복 \n답답합니다--'], ['저도 일하는게 훨씬 좋아요 아이만 잘 커준다면 ^^'], ['전 일할땐일이힘들고 살림할땐 살림이힘들어요놀고시포요'], ['저도 살림이요..! 물론 회사다니는게 쉽지만은 않지만 상대적으로는 복직하고 예쁜 옷도 입고 어른들과(?) 맛있는 밥도 먹고 커피도 마시며 인간답게 살수 있어서 넘 좋더라구요. 물론 아기는 너무 이쁘니 육아까지야 그렇다 쳐도, 살림은 진짜 제 적성에 안맞나봐요... 진짜 재미도 없고 보람도 없고 잘 못하니까 안하고싶고 안하니까 계속 못하고 영원한 악순환같기도합니다..ㅎㅎ'], ['전 육아,살림요. 일하는게 편하더라고요.'], ['저도 살림이 제일 너무 힘들어요.. 진심 일과 육아는 아이가 좀 크니 할 만한데... 살림은 정말 ㅠㅠ 전 음식하는게 제일 힘들어유 ㅠㅠ'], ['전 음식하는게  세상 스트레스에요ㅜㅠ 요령도 없고 잘 못해서 그런가봐요 게다가 아기가 제가 움식하는거 잘 먹지도 않아서요ㅠㅠㅠ'], ['저도 저도 요리요.. 다향히 울 애들은 잘 먹어서요.. 애가 안먹으면 진짜 하기싫을거 같아요.. 먹고 치우는 일만 없어도 삶의 여유가 있더라고요.'], ['저도 살림이 제일 힘들어요! 저는 요리만 좋아하고요 나머진 그냥 너무 힘들어서 신랑한테 일임을.... 대신 저는 육아를 많이 하니까요!화장실도 자주 청소해야지, 쓰레기 버리기, 여기저기 물때 생활때 닦고.. 보면 끝이 없는데 티도 안나고 안하면 티나요 ㅎㅎ'], ['와. 요리를 좋아하시다니 부럽사와요. 저는 요리도 싫어하는데 꾸역꾸역하니 스트레스가 쌓이는 듯해요..'], ['근데 문제는 살림과 일 둘중 뭐가 어렵냐가 아니라 어차피 저희같은 엄마들은 일을 해도 살림과 아이케어를 아예 안할 수 없는 구조잖아요??ㅠㅠ 저는 그게 제일 싫어요!!!!! 둘다 해야하는거!!!!! 왜 일을 하면서도 집안일 아이들일에 동동거리는건 나만인가.....'], ['맞아요 공감 천만퍼센트네요. 왜 동동거려야하는 건가.. 저는 잘때 저만 동동거리기 싫어서 반반을 요구했는데 거절해서 이혼했어요. 어차피 혼자 다할껀데 뭣하러 같이사나요.. 밥도 지가 안처먹고 제가 해주갈 바라고요.'], ['저는 굳이 따지면 일이요7세까지 사람 한번 안쓰고 살림 육아 일 어찌어찌 오롯이 하는 중인데 일은 내가 돈 받는 만큼 밥값 해야하니 부담 및 의무감과 성과에 대한 스트레스가 크네요대신 살림과 육아는 적어도 완벽히 해내지 못할 망정 내기준 내만족이고 내새끼니 좀 부족한들 좋으니깐요^^'], ['사람마다 다르군요.저는 성과를 내는게 너므 좋더라고요. 물론 힘들지만 성과에 대한 보상이 있는데 살림은 뭐.. ㅠㅠ 제가 만족을 못해서 더 괴롭나봐요.'], ['저도  일이요.... \n일이 힘든직업이아니여서그렁가...\n집안일 너무 귀찮아요 ㅜㅜ'], ['저는 살림이 일보다는 적성에 맞는거같고 ㅋㅋ 육아가 젤 힘들어요...ㅜㅜ 육아휴직 해보니 집 꾸미고 관리하는건 재밌더라구요 ㅋㅋㅋ 한명이 집에있으니 확실히 집도 더 깨끗해지고!'], ['그냥.. 사는게 힘드네요 ㅋㅋㅋㅋㅋㅋ 애 유아식 시작하고 부터는 신랑 저녁도 거의 안차려주고.. 내 밥 챙겨먹기도 넘나 귀찮고... 겨우겨우 빨래랑 설거지 정도 하나봐요.. 휴..'], ['ㅋㅋㅋ 죄송해요! 저도 사는게 힘들다는 말에 격하게 공감해서^^;\n전 일이 심적 스트레스가 있어서 더 힘든 것 같아요!\n살림은 뭐 대~충, 겨우 먹고 살아요. 내가 안하고, 못해도 누가 뭐라지는 않으니...\n좀 드럽게, 편하게 살아요, 우리!(실은 저자신에게 하는 소리^^; 저도 요리, 청소 드럽게 싫어하고 못해요><)'], ['살림요 정말 살림이 힘들어요 정말 끝이 없네요. 매일 하는 청소빨래설거지다림질...도 힘든데 청소는 정말 끝이없는거같아요 가스레인지후드청소 숟가락통청소 식기건조대도 드러내면 장난아닌 물때에...공기청소기 속에 닦아야지 제습기에도 먼지끼지...신발장은 마구인데 볼때마다 한숨만 나오고 하아....너무 우울해요 진짜 ㅠ'], ['육아휴직하다가 복직했을때 노래부르고 다녔어요..일도 넘 재밌고 때 되면  월급 나오고 밥도 회사 식당밥 주변사람들은 불만도 하던데  전 너~~~무 맛나더라구요...마실 갔다 집에 가는 느낌..좋았됐더랬는데...지금은 본의 아니게 쉬네요ㅜㅜ 살림은 그래두 하겠는데 육아...점점 고집세지고 말안듣는 아이...통제 안되고 뭔가 문제 행동하면 내잭임인것만같고..그 부담이 넘 크네요...'], ['살림이요 다람쥐 쳇바퀴 하는 느낌!'], ['실체가 없고 끝이 없는 살림으로 인해 육아와 일이 모두 힝들어 지는것 같아요. 어느날 문득 내 스트레스의 많은 원인이 끝도 없는 살림을 당장 처리하고자 하는 조바심? 강박감?에서 오는게 아닌가 고민했어요. 그래서 살림으로 부터 벗어나기 위해 아무리 더러워도 일단 청소를 주1회만 하자 부터 정했는데 문제는 다음 순서로 줄일 수 있는게 없네요ㅜㅜ'], ['전 살림이요! 집안일은 해도 티안나고 안하면 티난다는 말이 있잖아요, 끝도 없는 일때문에 저는 기계의 힘을 빌려요. 식기세척기. 로봇청소기. 건조기, 한달에한번 청소도우미도 쓰구요. 자잘한 살림은 아이잘때(새벽에) 하고..저도 부지런하게 하진 않지만, 설거지 할시간에 기계의 힘을빌려 아이랑놀아주자! 생각으로 하고있어요^^항상 긍정적인 에너지 넘치시는 쉬바님! 살림은 조금더 내려놓으시고 쉬바님 시간을 더 가지면서 아껴주세요^^ '], ['저도 애랑 노는건 안 힘든데 집안일 너무 싫어요. 쳇바퀴도는 느낌..  집에 다 어지르는 사람만 있고 치우는건 저 하나니 정말 끝이 없어요. 지긋지긋..'], ['기본적으로 살림이 너무 싫은데 그래도 요즘같이 궂은 날씨에는 출근안해도 되는 살림이 나은가 하고 있어요'], ['육아>>>살림>>>>>>>>>일....... 애들 빨리 컸음 좋겠어요. 그럼 전 훅 늙어있겠죠?ㅠㅠ'], ['살람 힘들죠~~~ \n그래서 식섹기, 건조기 로봇청소기, 세트로 돌리니 좀 괜찮은거 같아요 \n하지만 이것외에도 ㅇ요리도 해야하고 이것저것 하다보니 힘든거겠죠??'], ['워킹맘인데요, 육아&살림이 만배어려워요.사회엔 일하며 눈치보고 나름노고가있지만 성취감도있고 혜택등도있지만 살림은 해도 티안나고 안하면 바로티나고 밖에서의 시선들은 밥먹고 노는데라고하고...ㅠㅠ\n일이 더 쉽죠~'], ['오늘 휴가 이틀째인데 집안일 하다 하루가 다 갔네요~정리를 해도 티도 않나공~자꾸 지치고  힘드네요~집안일  넘넘 싫어요ㅜㅜ~~']]
    
    7759
    음식물처리기 음식물처리기 이것저것 알아보다 아파트 전단지광고보고 미생물처리기라해서 사용하고있어요 여름이라그런지 너무만족하고있어요 그때그때 무게 재서버리는시스템이면 여름에 고생이 덜할꺼 같은데 세종시는 음식물쓰레기 봉지 채우기라 썩어서 냄새(우웩) 과일껍질있음 초파리생기고 ㅜㅠ여름엔 진짜 싫었는데 지금 몸과마음이편해서 강추해요~필터관리 이런거없고 한달쯤사용하니 전기세는3천원정도더나온것같아요~제일싫은 집안일 셋트에서 하나는벗어나서 만세부르는요즘 맘카페검색해보고 이거정보는없는것같아올려봅니다~~( 코로나좀진짜 여기넣어버리고싶네요 ㅜㅠ)
    
    [['마지막글 공감되요 저번~~~ 태풍 올때도...그태풍에 코로나19 가져갔으면 했네요 ㅎ'], ['흐엉 저도 현실같지않아서 엑시트영화에서처럼 비가와서 코로나가다씻겨갔으면좋겠다생각했었어요ㅜㅠ'], ['관심가요 가격등등 ㅎ 알고싶어서요'], ['쪽지보내드릴께요'], ['소음 심하지 않나요? 어디건지 궁금해요'], ['소음은 거의없어요~혹시필요하시면 전단지찍어 보내드릴까요'], ['네 그럼 부탁드릴께요'], ['쪽지드렸어요~'], ['여름되면 ㅠㅠ사고싶은게 음식물처리기에요'], ['냄새진짜 ㅜㅠ음쓰봉지다채우는거오래걸려너무힘들어요 반만채우자니 아깝고요'], ['언뜻보고 제꺼랑 똑같은줄 ㅎㅎ 다른회사네요 ㅎㅎ그래도 이렇게 분쇄형 안쓰시는거보면 반가워요 ㅎㅎㅎ미생물타입 자리 조금 차지하는거외엔 정말 다 좋은데 ㅎ 널리 알리고싶어도 오지랍일거같아 찾는경우에만 댓글달고 말았거든요 ㅎ 방갑방갑^^'], ['좋은건널리알려야하지말입니다 ㅎㅎ아줌마니깐같이좋아야되니깐요 ㅎ'], ['저도 알아보고 있는데 정보부탁드려요'], ['쪽지드렸어요'], ['저두  살짝  공유좀  부탁드려요'], ['쪽지드렸어요'], ['저도 공유 부탁드려요 ~!'], ['쪽지드렸어요'], ['저는 스마트카라쓰는데 여름엔 진짜 잘샀다 싶어요..ㅎ 이건 쓰봉을 아예 안쓰나요? 어떻게 처리하세요??'], ['양이좀찼다싶으면 쓰레기봉지에 덜어 버리면된다는데 한달동안 전 한번 3분의1쯤 버렸어요 코로나덕에 집밥만먹으니 전보다 음쓰양이 많이생기는것같아요ㅡㅜ'], ['정보좀주세요~~~'], ['쪽지보냈습니다'], ['초파리 싫어요 ㅜㅠ쪽지보냈습니다'], ['저도 정보 부탁드려요~~'], ['쪽지드렸어요'], ['저도 정보 부탁드려요~'], ['쪽지드렸어요'], ['오옷 좋네요. 저도 정보 부탁드려요~^^'], ['쪽지드렸어요'], ['저도 정보부탁드려요~~'], ['쪽지드렸어요'], ['저도 정보좀주세요~'], ['쪽지드렸어요~'], ['저도 음식물처리기 알아보고있는데 냄새가 심하다고해서 못사고있어요ㅜㅜ\n이건 냄새안나나요?저도 정보좀 부탁드려요~음식물쓰레기 스트레스에요ㅜㅜ'], ['쪽지보냈어요~뚜껑열때미생물냄새 적응은좀필요했구요 닫아있을땐 문제없었어요~~'], ['감사합니다^^'], ['저도 비슷하게 생긴 어썸#퀘어 쓰고있어요. 엄청 편하네요'], ['돈으로편리함을샀달까요 ㅎ'], ['저도 정보 부탁드려요 ~~!!'], ['쪽지드렸습니다~'], ['저도부탁드려요^^'], ['쪽지보냈습니다~'], ['저도 쪽지 부탁드려요~~^^'], ['쪽지보냈습니다'], ['저도 가격 정보 부탁드립니다'], ['쪽지보냈습니다~'], ['저두 부탁드려요'], ['쪽지보냈습니다'], ['저두 부탁드려요~~!'], ['쪽지보내드렸어요~~'], ['늦은시간 죄송해요저도 정보 주실 수 있으세요?'], ['쪽지보내드렸어요~'], ['저도 정보  좀 주세요 제품명과 가격이요'], ['쪽지보내드렸습니다'], ['저도 정보 좀 주세요~^^']]
    
    7836
    마지막 집안일 끝? 봉다리에서 진화한​반찬통 운동화 세탁법​쉐킷쉐킷 후​이따 헹굼만 해주면​오늘의 집안일 끝이네요​집안일 진작에  다 끝내셨지요?
    
    [['오 솔질이런거없이 그냥흔들면되는건가요 운동화사고 세탁한기억이 ㅡㅡ없어요'], ['솔질을 단한번도 안할순 없고요\n뜨거운물에 세제풀어 때를 불린뒤 헹굴때 살살 솔질해주네요'], ['아 솔질하긴해야되는군요 ㅜㅜ이상하게 운동화는 진짜 못빨겠어요 맨날 신고 ㅜ많이신고 더럽다싶음 버리고했거든요 애도 반계절마다 사이즈업하니 세탁할일없고 ㅎ'], ['세탁할 운동화라도 많으면 전문점에 맡기겠는데 몇개되지도 않는거 그냥 제가 세탁하네요\n하나사면 길게는 2년정도 신었는데  작년부턴 해바뀔때 한켤레 사주네요 둘째는 첫째오빠꺼 물려신고요'], ['저희는 오히려 반대라 ㅎ신랑꺼만 한번씩맡기니 세탁소가도 해주더라고요 제꺼랑애는 신다가 그냥 새신사신고 여러개돌려신으니 크게안더러워서 안빨고 물티슈로 쓱딱고 ㅎ가죽이런거라'], ['가죽운동화를  자주 사 신으시군요 그렇다면 굳이 물세탁 할필요 없으신게 맞네요'], ['이야~이런 방법도 있군요. 저도 매번 봉지에 했는데.. 하나 배웠어요ㅋ'], ['저도 초장기땐 봉다리에 했었는데 블로그보니 안쓰는 반찬통을 이용하길래 저도 따라해봤네요'], ['신기한데요 요건 어케하는 건가요?'], ['뜨거운물에 세제풀어 방치한후  솔로 살살 문질러서 헹구면 되네요'], ['아~저두 한번 해봐야겠네요^^'], ['운동화세탁법 검색해보면 더 자세하게  나올거에요..한번 해보세요'], ['네~좋은정보 감사합니다^^'], ['이 방법이 맘님께도 유용하게 적용되었으면 좋겠어요  굿밤되세요'], ['네~^^맘님두 굿밤되셔요~'], ['요고 어찌하는건가요?통에다 요래하니 신기한데요~ 방법 알려주세요'], ['이방법 초반에 봉다리로 했었는데 뜨거운물에 세제풀고 운동화 담궈서 불린뒤 헹굴때 솔질 좀 해주면 큰힘안들이고 운동화세탁할수있어요'], ['와저리도 운동화세척하는군요 저요증슬리퍼만계속신고잇어용ㅎ'], ['저는  발이 못난이라 가까운 슈퍼말곤 사시사철 운동화네요\n이 운동화는 드림할 운동화네요'], ['아글쿠나ㅎ 망님 드림마니하시는듯ㅎㅎ저는할게없어서못하고잇어용ㅎ'], ['오늘 애들옷 정리하다보니 작아진 운동화가 있길래 버리기도 그렇고 중고로 팔기는 양심상 상태가 많이 좋지않아서 드림하기로 했네요'], ['아네네ㅎㅎ 잘신었으며좋겠네용저도 옷같은거다시 드림할까싶어용ㅎ'], ['중고품 판매나 드림은 늘 조심스럽네요 상태양호라게 기준이 애매하니요'], ['이런방법이 있었군요 저는 맨날 힘들여서 팍팍문질렀는데 말이지요 세제는 아무세제 가능한건가요?'], ['저는 베이킹소다 쓰네요 근데 세제는 크게 상관없을것 같기도하고요'], ['아하 저도 요방법 써봐야 할까봐요 정말 무식하게 힘만들여서 씻었네요 ㅠ'], ['사람도 뜨신물에 몸을 불려야 때가 잘벗겨지듯이 운동화  세탁법도 마찬가지 원리네요ㅎㅎ'], ['아하 그렇군요 원리되로 생각하면 참 쉬운 방법이였는데 말이지요'], ['그러게말이에요\n그것도 모르고 예전엔 열심히 솔질만 했었었네요'], ['오 괜찮은 방법인데요 저렇게 불리면 때 잘지워질꺼같아요'], ['아무래도 좀 불렸다가 솔질하니 큰 힘은 안들고 좋은것 같아요'], ['운동화빠는거 은근 귀찮아요ㅠ 저희집에도 빨아야할 운동화가 한가득이네요 써먹어봐야겠습니다'], ['저는 그냥 다라이?에 넣었는데ㅋ반찬통 못쓰는거에 넣으시는군여ㅋ'], ['다라이에 담그면 위에까지 잠기지가 않으니 안쓰는 반찬통 적극 추천합니다 마구마구 흔들수도  있고요'], ['다라이에 담그니깐 자꾸 떠오르더라구여 ㅋㅋ반찬통좋으네영 뚜껑으로 닫을수있으니요 ㅋ'], ['자꾸 둥둥뜨지요 그렇다고 계속 누르고있다 뒤집어줄수도 없는 노릇이고요'], ['다른 다라이로 눌러놨었네여 근데 담궈놓으니 때가 더 잘빠지긴하더라구여 ㅎ'], ['덮어놓을 다라이가 또 있었군요 운동화도 사람도   물에 불려야 때가 잘 벗겨지네요ㅎㅎ'], ['낼 아침에되믄 완전 뽀얗게해사 하야이 되겠는데요 ㅎㅎ 꾸중물이빠이에 ㅎ'], ['제발 뽀얗게 되었으면 좋겠는데 워낙에 오래 묵혀둔 운동화라 큰 기대는 버렸네요'], ['아하 글쿠만요 그래도 하기전보다는 꺠끗해지겠찌요? 담에 저도 요런통 하나 사와서해봐야겠네요 지퍼팩에 했었는데ㅎ'], ['지퍼팩도 확실하게 가둬어두긴 하겠네요 그래도 두고두고 쓸수있는 안쓰는 반찬통도 괜찮겠지요'], ['넹 반찬통이 왠지 오래두고 쓸수있을꺼같애요 비닐은 자꾸 버리야하고 쓰레기나오니 ㅠ'], ['제 운동화는 색바랜 김치통  이용하고있네요\n자고로 사람은 남의 아이디어를 빌려쎠야하네요'], ['저도 뜨건물담가놨다 문지르는데...\n이리하면 더 잘 빠지나요?\n난 세탁 맡기고싶은데 아들래미가 엄마가 빨아돌래요 흑 ㅠㅠ'], ['아무래도 운동화 전체가 푹 담겨있으니 때가 잘 불려지는것 같아요'], ['세숫대야담그면 봉봉 뜨는데 \n저도 이거한번 해봐야겠어요 \n근데 들어갈 통이 음네여 ㅋㅋㅋ'], ['아이들 운동화는 반찬통이 딱 맞는데 제 운동화는 김치통 정도는 되어야하네요'], ['운동화 세탁하는거 \n은근히 귀찮기는하지만\n깔끔해진 운동화보면은\n기분이 엄청 개운하지요!^^\n저는 봉지에 넣고 하는데~ㅎㅎㅎ'], ['저도 초창기땐 큰 봉다리 한두개는 꼭 남겨두곤 그랬는데 이제는 더 간편한 반찬통을 이용하네요'], ['봉지는\n물가득 담고 묶어두니깐\n바닥에 펑퍼짐하게 퍼져서\n그닥이기는 하더라구요.\n큰 반찬통이 더 낫긴하겠어요.'], ['한번은 구멍난줄도 모르고 20분 방치했다보니 물 다 빠져있었구요ㅎㅎ\n제 운동화는 안쓰는 김치통 사용하네요'], ['김치통이\n큼직하니 사용하기 좋긴할듯해요.\n근데, 저희집에는 못 쓰는게 없어\n다이소에서 싼거라도\n하나 장만해야할까봐요.'], ['저희집은 김치가 귀해서 그런가 다먹은 빈 김치통만 수두룩하네요 친청에 다시 갔다줘야하는데  하나만 운동화 전용통으로 쓰고있네요']]
    
    7905
    남편들이 다들 알아서 집안일 도와주는거죠 잘 시키는 비법이 있나요어제 다들 음쓰 일반쓰레기 분리스거남편이 해준다해서요​울 남편은 돈번다고유세아닌 유세를하는데​가끔 무선 청소기는 자주 밀어주긴해요​그게 다거든요​안시켜도 잘하는게 신기한건 다시태어나야하나요
    
    [['다시 태어나도 남자는 안되지 싶네요ㅠㅠ'], ['음식쓰레기랑 재활용 일반쓰레기는 알아서버려줍니다 그외에는 해라해라해라 소리쳐야 가끔해줍니다'], ['거기에 저도 추가요 ㅋㅋㅋ'], ['33333'], ['아 공감되요'], ['55555'], ['66666'], ['저희집도  알아서 해주심'], ['저희 남편은 담배 피고 깊을때 핑계가 음식물+재활영 버리러 갔다올게 에요 ㅋㅋㅋ\n심지어 한번에 안가고 나눠 가요.... 그거 외엔 해달라 해야지 해주는것 같아요 ㅠㅠ'], ['울집도요..스스로한다기보다 담배피러갈때 애들이 어디가하면 쓰레기버리러..하면서 그때 치워주죠..애들 자면 안하죠...'], ['? 안해여 ㅋㅋ'], ['저희도 알아서 다해줘요\n칭찬~\n그리고 시켜놓으면 내맘에 안들어도 못본척, 간섭 안해요~\n처음엔 어설퍼도 하다보면 늘어요 ㅎㅎ'], ['저도저도~~\n시켜놓고 잘하든 못하든 일단 칭찬~~ 고맙다며~~\n자꾸하다보니 이젠 저보다 잘한답니다~^^'], ['문앞에 놔둬도 5번중 한두번해요 ㅡ.ㅡ'], ['아뇨 시켜야겨우해요 그나마쓰레기는본인일이라생각하고 버려주는거에감사해야합니다ㅜㅜ'], ['수많은 잔소리의 결과물이죠 ㅠㅠ ㅋㅋㅋ'], ['알아서는 안해도 시켜야해서 시켜요. 여보 이거좀해줘요. 애들목욕좀시켜놔요. 뭐이런식으로ㅋㅋ기분좋을때(?)시켜요.시킬때도 기분좋게ㅋㅋ시키고ㅋㅋ살림살이중 다른잔소리 대신 전 일절안해요.\n'], ['안 도와줍니다 그래서 포기 했습니다'], ['22222  포기포기'], ['청소기도 안밀어주는데여 ㅠㅠ나간데서 일반쓰레기봉투주며 버려달라니 알았다해놓고 나갈땟 걍 몸만 가네요 한두번이 아님요 깜빡했다한마디하고 😂😂'], ['쓰레기도 제가 버리는데요? 설겆이, 빨래 널고 개키는건 가끔 하네요'], ['꾸준히. 습관적으로 시키세요. ㅋ  저흰. 설겆이 빨래돌리고게는거  일반 요리도 해주구. 주말엔 청소기돌리기. 방닦기 다 도와줘요  음식물,쓰레기.버리기는 신랑이 다하구요. 애들도 씻겨주고 화장실 청소도 해줘요..'], ['오~~전생에 뭐하셨어요?'], ['집에 들어오면 진.짜 암것도 안해요.\n20킬로 쌀 주문한거 들어 옮겨주는 정도랄까요ㅠㅠ'], ['이미 포기했어요\n집에서는 손도 까딱 안해요'], ['쓰레기 분리해서 놔두면 그거 다 들고 내려가서 버리기만 해요 ㅋㅋ 음쓰는 자기가 알아서 들고 내려가고용. 그거 외에는 하나부터 열까지 다 알려주면서 간간히 확인 하고 감시 하면 해용ㅋㅋㅋㅋㅋ'], ['전업하고 있어서 안도와줘도 그러려니 해요 제가 직장맘인데 집안일  안도와주면 화나겠죠... 남편은 회사 일하고 왔으니 집안일은 제가 하려고 노력해요~'], ['저도요.. 같은생각으로 혼자다..하고있어요 .. 일시작하면 이제 분업하려구요ㅎㅎ'], ['2222 장사를해서 하루종일 밖에서 서서일하는 사람 집에와서 뭘 시키나 싶어 제가 다합니다.'], ['분리수거랑 일반쓰레기 버리는거 잘해줘요 ㅋㅋㅋㅋㅋㅋ\n일반쓰레기는 차타고 출근하는길에 가져다버리고? \n재활용도 주말에 어디 나갈일있으면 차에 실고 나가는길에 버리고가요 ㅋㅋㅋㅋㅋㅋ\n그외 저보다 일찍퇴근하면 밥 메인메뉴1개만 해놓고있구요 빨래 돌리는고 너는거는 저보다 많이해요 ㅋㅋㅋㅋㅋㅋㅋㅋ\n좀 마음에 안들게 해놓지만....해도 왜이렇게 마음에안들게했냐고 잔소리절대안하고 .도와줘서 고맙다고 이야기해주고......모른체합니다.....'], ['알아서 잘하는 유전자는 극히 드문것 같아요 ㅎ'], ['저희집도 알아서 다도와주네요.  감사하게 생각하고 있어요'], ['잘하는것만 시킵니다 ㅋㅋ 빨래널기,걷기,개기 음쓰버리기, 분리수거하기 그외는 안해요 ㅋㅋ'], ['정리해서 문 앞에두면 가져다 버려주고~ \n맏벌이하다보니 본인 휴무일땐 대청소와 저녁 준비 해 주고~ \n화장실 청소는 신랑 몫 입니다~'], ['전 아예 말안해요..ㅜㅜ은근 기대하거나 바라니 저만 지치거나  맘상하고..그런식으로 억지로할땐 하는 남편도 짜증스러운거같아 ..이게 뭐라고싶어 몇번 트러블후로는 말안해요..대신 제가 하든말든 잔소리도 하지못하게해요전 가끔 확 다운되면 다 내려놓고 아무것도..전혀 안하거든요.그리고  할거면 기분좋게하라고 얘기해줬어요..음.그랬더니 가뭄에 콩나듯이긴 하지만..알아서 재활용정도는 해주더라구요..대신..집에 아주 조그맣게라도 손볼게 생기면 그건 절대 제가 손대지않습니다 그랬더니 그건 자신의 일이라고 확실히 인지한듯해요..만일 그걸 남편에게 얘기했는데도 일주일이 넘어가도 이주일이 넘어가도 손볼생각안해도..절대 제가 먼저 급해서 손보지않고 할때까지 내버려둡니다.불편하면 불편한대로 생활합니다 잔소리도 않구요.처음엔정말 한달 넘기더니 이젠 알아서 손보는 단계까지왔어요..'], ['참..자기가 우짜다가 한번설거지든 세탁이든 건조든 우러나서 하면 맘에안들어도 오구오구 잘했다는 해줘요..그랬더니 우짜다 한번이라도 기분좋게하니 그게 더 낫더라구요ㅎ전업이라..마음을 비운것도 있는데 만일 전업아니라면ㅜㅜ 5대5까진 아니더라도 7대3에서 6대4까진 나누려고 했을것같아요'], ['저희 남편보니..다시 태어나도 안 될 것 같은데요..'], ['대부분 알아서 하는데 시야가 좁은 사람이라 미쳐 못본게 있으면 00시전에 음쓰 버려줘~/ 오늘은 씻고 욕실청소 해줘 라는 식으로 구체적 미션형식을 주고 갔다오면 고생했다 냄새가 너무 났는데 덕분에 숨이 쉬어진다 등 폭풍친창을 해줍니당 ㅋㅋㅋㅋ'], ['알아서안해요ㅋㅋㅋㅋ\n시켜야하는데 것도가끔 이상스한방향으로하죠..속터짐ㅜㅜ'], ['아침마다 음쓰랑 분리수거 들고나가고요...\n둘째 씻기는거랑 냉장고정리, 화장실청소,\n세탁기돌리고,건조기에서 빨래꺼내서 개기, 주말엔 청소기정도에요.\n\n한번도 시킨적은 없고~ 하나씩 하더니 점점 살림이 느는건지 갈수록 더 많이 하더라구요ㅎㅎㅎ\n\n친정에서 엄마가, 너거집에 깍두기있나??해줄까?물으면\n저는 모르고, 남편이 깍두기 1-2번 먹을거있습니다 하고대답해요ㅋㅋ\n\n다만...\n요리는 못해요..ㅠㅠ\n제가 감자썬어놓으면 자기가 볶는정도?\n생선굽기나 계란프라이가 다에요ㅎㅎㅎ'], ['전 제가 요리 & 아기케어 하고 그외에 다 남편이해요..ㅎㅎ.... 맞벌이 집안이라 몇번 다투고 협의했네요'], ['응????????????????정말 그런 남편들이 있다구요????????????전 애낳으러 가는날도 제가 음쓰버리고 화장실청소하고 갔는데ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이게 시켜야하지 알아서는 절대네버 안해요. 요즘은 제가 애둘본다고 짜증있는대로 내니까 가끔 분리수거 하고올까 쓰레기버릴꺼없나 물어보네요. 제발 자기 먹은거 쓰레기나 휴지통에 잘 넣어줬으면 좋겠어요'], ['알아서 해주면 안되냐니 일일이 말해줘야된다네요. 핑계거리죠'], ['도와주는편인데 잔소리 & 보조? 가 필요한\n이거갖고와라 저거갖고와라 ㅋ\n더 귀찮아요ㅜ'], ['알지를 못해요.. ㅡㅡ 쓰레기봉투 쌔매서 두고 버려주소 하면 버리고 음쓰도 버리게 해두고 버려주소 하면 버리고..제가 입을 열어야 하네요..남자가 알아서 할 수도 잇군요..부러워요ㅠㅜ'], ['음식쓰레기는알아서버리는데 다른건시켜야합니다즈ㅡ'], ['하나하나 이거 해줘~하고 잘한다잘한다 특급 칭찬으로 길들이고 성에 안차도 못본척하고 잘할때까지 기다리는중입니다~'], ['저흰 원래 시키면 참 잘했는데 요즘은 안시켜도 알아서 설거지며 청소기며 잘 하네요ㅋ\n원래 제가 한깔끔 했는데 손놔서 그런듯요ㅋㅋ\n쓰레기도 딱 채워놓음 알아서 들고 나가고ㅋㅋ\n손 놓을 맛 나요ㅋ'], ['배운바 있지만 본바 없는 남편들이 많아서 그럴꺼예요~~\n내아들,딸 결혼해서 잘 살기 바라면 보여줘야한다며\n시간을 정해서 시키세욤~~\n8시까지 버려줘~~으흐흐'], ['ㅋㅋㅋ안해요ㅠ 시키면 내일버릴게~ ㅡㅡ'], ['설거지,  밥, 청소 아기케어까지 다해줘요~~\n대신 돈은 제가 다벌어요ㅋㅋㅋ'], ['쓰레기통 한 번 비워본 적 없고, 유일하게 하는건 2주에 한번 모아놓은 박스 1층에 내려놓는 것만요. 회사 성실히 다니고, 도박,바람 안 피는 것만으로도 감사하며 나이 많은 아들 잘 키우는 중이에요.'], ['저조 음식물쓰레기는 신랑이~\n재활용은 신랑과 아들이 해요ㅎㅎ\n이제 아주 일상이되서 저는 손도 안댑니다ㅎㅎ'], ['저는 설거지 빨래만 합니다요.. 꼼꼼하지 못한 성격과 귀차니즘.. 신랑이 더 잘해서 본인이 하던데..'], ['저희 신랑은 퇴근후 밖에 나가는걸 싫어해서 음쓰 재활용 항상 제가 해요\n대신 신랑이 집에 있을땐 애들 둘다 아빠 껌딱지고 요즘 코로나 땜에 종일 애둘 보는거 힘들다고 아침밥 차리고 설겆이 정리정돈 신랑이 해요 어떤날은 삼시세끼 신랑이 하는 날도 있어요 일하다가 들어와서 점심 해주고 설겆이 하고 나가요ㅡㅡ\n제가 처음 시집 갔을때 아버님이 집안일을 너무 많이 하셔서 좀 놀랐는데 저희 어머님께시 처음 시집 가셨을때 시할아버지께서 마당쓸고 집안일 하셔서 놀라셨대요ㅋ\n보고 자라는게 중요한듯요'], ['집안환경 많이 좌우하는 것 같아요ㅡ저희는 시아버님이 애처가라 그런가 말 안해도 잘해주더라고요'], ['맞벌이 남매맘이에요. 신랑이 자발적으로 하는건 화장실 청소 하나인데 그냥 바닥이랑 변기 쓱싹하고 끝입니다;; 그리고 제가 복직하고 설겆이나 젖병(빨대컵) 씻었는데 5시에 퇴근하는날은 설겆이, 젖병, 아이들 목욕 다 합니다. 저희 아들은 절대 그렇게 안키우려고 노력중인데 주위를 둘러보면 남성의 DNA 자체가 역사적으로 그런건지 어휴....'], ['아기어릴땐 해주더니 어린이집가고 하니까 안해줘요ㅋㅋㅋㅋㅋㅋㅋ..🤣🤣🤣'], ['저희 같이해요\n분리수거 챙겨두면 같이 버리고, 주말 청소, 장보기...\n주말 장보고 음식은 알아서 해요\n그외 전업이라 제가 하구요'], ['알아서 안해요 시켜야 하죠,ㅋㅋ'], ['저희집에 돈 벌어온다는 유세로 십년넘게 손가락 하나 까딱!! 안하는 남자 한명있어요\n이번에 제가 아퍼서 수술날짜 받아오니 엄청 잘 움직이네요~~ㅋㅋ\n곰탱이가 저렇게 움직일 수도 있다는걸 이제야 알았어요ㅋㅋ\n설거지하면 바닥이 물난리이고 정리를 못하니 계속 물어봐서 귀가 아프지만.......'], ['음..알아서는 힘들지않나요~~??맘비우고 자꾸시켜야해요~~힘들면 도우미이모님비용받아서  가끔 맡기세요'], ['보통은 시켜서하죠 대신 빨래는 건조기에서 빼서 쇼파에 갖다놓으면 알아서 정리합니다ㅎ'], ['사람은 고쳐쓰면 안되는거라고 누가 그러던데 정말 맞는말인듯 다음생엠 그냥 혼자 사는걸로~'], ['타고나는거 같기는 해요. 근데 스스로 하는 남편은 잔소리 모드도 딸려옵니다ㅜ.ㅜ뭐 하나 먹고 잠시 안치워도 잔소리잔소리..ㅋㅋㅋㅋㅋ그리고 큼직한 청소빼고는 다 일일이 시켜야 해요ㅎㅎㅎ아 그리고 시댁분위기가 사아버님이 청소를 다 하세요.'], ['22222 \n진짜 집안일 지분이 비슷할정도로 잘도와주는데(저는전업) \n잔소이 진짜 ㅡㅡ ㅠㅠ 머리아플정도에요 잔소리'], ['설거지는 안해도 ㅎㅎ 씽크대 그릇 담기 전 매번 물로 한번 쓱 헹궈서 음식물 없이 넣어두고 청소기는 항상 밀고 현관 청소 본인이해요. 세탁기랑 건조기 돌리는거는 먼저 본사람이 하는데 거의 반반하는듯요 ㅎ 아이 목욕이랑 용변 뒷처리는 아빠가 더 많이하네요'], ['주말엔 식사준비,아이 밥먹는거 챙겨주고 설겆이에 마무리 커피까지 타다주는 신랑이라 평일 퇴근 늦는게 넘나 아쉬워요쓰레기랑 재활용 버리는것까지도 안시켜도 잘해주고 변기청소는 얘기하면 해주고요..애랑 저랑 아빠있는 주말만 목빠지게 기다립니다ㅋㅋㅋ'], ['아무것도 안해줘요'], ['즈희 신랑도 막담배 피러가니 음식물이랑 재활용정도는 해주네요 \n다른거는그닥'], ['담배덕에 합니다..\n담배피러 나갈때 분리수거ㅡㅡ'], ['재활용말고는 아무것도 안해요~~~~~~~~~'], ['맞벌이일때는 말안해도 도와주더니 전업맘되니 도로아미타불입니다. 너무싸워서 그냥 남의 편이가보다하고 포기했어요'], ['남자들은해라고해야합니다  시켜야되요 근데 일일이시키는것도짜증날때가있어요  ㅋㅋ 마치고오면 밥먹고 애들은씩이는데 그외엔 시켜요 누어서폰보는건알아서잘하네요 ㅋ'], ['아무것도 안하고 있음 알아서 합니다ㅋㅋ\n결혼전부터 집안일 싫어 결혼안한다 결혼하고싶음 딴여자 만나라 입에 달고 연애하고 리얼로 아무것도 안하니 자기가 다 합니다ㅋ'], ['시켜야 합니다'], ['암것도안하는데요?'], ['쓰레기봉투 밖에 내다주는것만해요\n굳이 뭘 시키고 싶은 생각도 없네요ㅋ\n맞벌이할땐 정말 많이 싸웠는데\n전업이니 그걸로 안싸워서 좋아요'], ['시아버지가 잘하셔서 보고 배운게 무섭네요~ ^^~도와준다는 개념이 아니라 70%이상은 신랑이 합니다. 맞벌이할 때는 90%신랑이 했구요~올해부터 전업들어서면서 제가 좀 더? 움직이려고 노력합니다. 근데~~근데~다 가질순 없겠죠?^^;성격이 예민하고 지랄?같아요 ㅋㅋㅋㅋ음~~더 이상 말 안하렵니다^^;'], ['본인 마음인듯해요. 음식쓰레기.재활용.일반 다 처리해줍니다. 음식쓰레기는 통도 한번씩 씼는데 집에서 퐁퐁으로 막 씨고 해서 바로 수세미는 제가 버려요,  머라하면 담에 안 하니 몰래 버려요. 그냥 이젠 습관된것 같아요. 내가 안 버리니 알아서 버려줘요. 흡연을 정당화하기 위해 나갈때마다 하나씩 들고가는것 같기도 해요.'], ['결혼 초반에 집안일.분담을 잘 해놔서 그런가 이젠 지 일인냥 착착해요 ㅋㅋㅋㅋ 개처럼 싸워서 승리했습니다...'], ['저희 남편 설거지 음쓰 분리수거 일반쓰레기 하는데....\n저는 남편이 할때꺼지 안해요^.^\n못본척하고 하면 칭찬을..... 이빠이 날려드립니다\n음식물분해기 사고싶다니 신랑이 하루에 한번 밥먹고 바로 설겆이하고 음쓰 버리내옄ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ']]
    
    7993
    맘님의 선택은? 극단적인 예입니다.1번,신랑이 완전 다정다감합니다.근데 말로만요.말은 너무 예뻐요.행동은 아예 안해요.집안일 하나안하고,먹은거,벗은거 그대로...​2번,신랑이 완전 경상도 사나이죠.말은 투명스럽게~~무뚝뚝근데 행동으로 넘 잘해요.설거지,청소등등 집안일을 적극적으로 도와주죠.​어떤 신랑이 좋나요?급 궁금하네요~
    
    [['2번요 ㅋㅋㅋㅋ입만 살아 있는거 싫어요..행동이 중요하죠'], ['이거 투표하기 그런거도 있던데~~할줄을 모릅니다ㅋㅋ젊은기.: 완전 기계치라ㅜ'], ['2번이죠ㅋㅋ'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['2번이요~'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['당연 2번이죠 ㅎㅎ'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['2번요\n행동이 중요하죠잉'], ['옳으신말씀~!'], ['저도 2번요~😆'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['222'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['무조건2번\n울집양반이 말만이 입니다\n말로는 지구도 구할양반인데\n액션은.....억장이무너지는수준이라는 ㅠ'], ['아,꺅...동지십니다ㅜ'], ['중학교 가정시간에 바느질좀 배워둘걸....이 인간 입에다 세발뜨기라도하구로..덴장..'], ['2번이요'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['2번ㅎㅎㅎ'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['둘다 싫어요  ㅋㅋㅋ'], ['앗,번외답변하기없기ㅋㅋㅋ어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['저는 2번에 가까운데 말이쁘게.하는것도 중요한거같아서요...아오'], ['무조건2번여  번지르르하게 입만나불대는거 짱나여ㅋㅋㅋㅋㅋ'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['전 그나마 2번이요 지금 1번남자와사는데 지인들은 다들 그래요 너밖에모르잖아 ㅠㅠ 정말 좋은거 맛있는거 고고 먹으면 저꼭 같이가자하는데 집에선 뱀 한마리 사는듯해요ㅡㅡ'], ['악~~동지십니다.미쵸요'], ['저만 뱀키우는게 아니였군요 ㅠㅠ'], ['둘중 정하라면 2번요~^^'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['저희 남편 1번요... 이혼도장 찍는 순간에도  말로 심쿵하게해서.... 빵터지고.. ㅠㅠ 도장 못 찍었어요.  돈은 잘 벌어와서 참아요.    \n2번 츤데레랑 살고 싶네요.'], ['우와~동지십니다.저도 그정도입니다'], ['ㅋㅋㅋㅋ 반갑지 않은 동지시네요 ㅠㅠ'], ['당근2번 입만살은사람 시러요 ㅋ'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['즤집남편이 2번이긴한데\n어차피할꺼 기분좋게좀해주지\n말로 사람 기분다잡쳐놓고하니\n별로고맙지도않아요ㅋㅋ'], ['와~~그래도 행동파ㅋ'], ['무조건 2번요~'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['악 ㅋㅋㅋ \n그 다정 ~몸소 보여주소서~🙏라고\n말씀 전합니다~~😆 아미타불~'], ['2번츤데레요ㅋ'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['2번이용~~'], ['어제 싸우면서 본인처럼 다정한사람어딧냐며...참고로 1번이랑삽니다.생각해보니 빡쳐서 물어봅니다ㅋㅋ'], ['저희 신랑 입은2번 행동은1번입니다 ㅋㅋㅋㅋㅋ 안좋은것만 섞였네요 ㅠ'], ['울 신랑이 딱 1번이네요. ㅋㅋㅋㅋ'], ['동지십니다ㅠ.ㅠ'], ['ㅎㅎ 대신 대화는 한 70퍼는 잘 통하는 편이라... 그나마... 가끔 울 신랑 백배는 잘하는거같은데 육아분담 버럭하는 글들보면 코웃음칩니다. ㅋㅋ 본인도 자기 못하는줄은 앎.. 능글넘어가거나 가~끔 나 이거해줬자나할 수 있게 가끔 설거지 물 뜨기. 정기적으로 쌀 소분 정도는 해주네요. ㅋㅋㅋ'], ['2번이요'], ['2222222222'], ['둘중에하나라면 2번여  ㅋㅋ'], ['저희남편은 2번 츤데레네요 근데일상생활에서는 심심해요쫌 ~ 무뚝뚝해서'], ['당연히2번요~'], ['저희신랑도 2번에 가까운데\nㅎ시키면ㅈ군소리없이 하긴해요.\n근데 말이별로없어 혼자있는거같단생각많이들어요ㅎㅎ']]
    
    8016
    제가 이상한건가요?생각 하면 화나요 맞벌이고 애없이 강아지한마리 키우는데...남편이 바지세탁소에 맡겨달라고 해서 맡겨주고 거스름돈으로커피사먹겠다고, 했는데  너도 돈버니까 니돈으로 사먹으라고하는 거있죠!! 이뿐만이 아니예요 주말에  배달음식시킬때도제가 돈내라고 하네요.생활비 30만원 줘요.  온전히 식비이지만요.  부족한건 너도 돈버니까 알아서 니돈 써!,,이러는데  얼탱이 없을 때가 많아요.전 저 생활비가 제 노동값의 가치같아요. 빨래 청소 집안일 일절하나도 안하는 남편이니 제가 다해요!불리수거까지도, 그리고 강아지 배변패드는 커녕 사료한번 사준적도 없어요. 뭐 제가 키우고 싶어 키우고있지만요.그리고 생활비도 월급날 재때주는 것도 아니예요!다들 이러고 살아요?  아니 가장이면 부양은 해야되지 않나요? 아님 집안일은 각자 해야되지 않나요?진짜 청소하기 힘들어서 생일 선물로 무선청소기 사달라고  한 제자신도 너무 어이없어 져요.​
    
    [['맞벌이인데도 그래요? 그냥 돈 받지 말고 집안일 반반하자 해보세요'], ['반반하자고 지정집안일을  줬는데 그것도 야근하고 늦게오면 안해요'], ['남편분이 이상한거같으네요 글에써진것만보면..ㅜ'], ['이상해요'], ['공용 생활비는 1/n 로 나누세요\n물론 강아지케어비용까지.....\n동의해서 키우는거니까요\n그리고 집안일도 나누시구요'], ['아이도 혼자 낳는거 아닌데 독박육아같이 독박이예요 집안일도 지정해줬는데'], ['그렇게 살면 뭐하러 살아요.\n목표가 없잖아요.집을 산다든가.노후준비를 한다든가\n님께서 남편 월급  관리하세요.용돈만주고요.\n너는 너 나는 나 늙어서 정없어지면 어떻게 살아가려구요 ㅠㅠ'], ['저라면  일을 그만두둔가  안살래요  \n일도 하고 내거말고 남의편 까지 내가 해야하고 이게 먼가요?  \n이런말해서죄송해요 \n저라면 그러게 할거같다는 애기에요'], ['지금도 생각중에있어요 잘못한 결혼인걸요'], ['그렇게 살거면 본인 엄마랑 살라고 하세요.\n남편분이 비정상적인 사고를 가지셨네요ㅡㅡ'], ['그쵸!!저도 돈버니까  이러는데 빈정상하더라고요'], ['30안받고 30주고 시킬래요ㅡㅡ'], ['뭔가 서운한게 있어서 저렇게 싹퉁이처럼 그런건 아닐까요? \n예를들면 집 명의가 님의 명의로만 되어있다던가..이런거요.\n\n각자 주머니면 생활비는 반반이어야하는데 남편 멘트는 뭔가 쥐어박듯이 이건 니가 내! 이러는거보면 남편 계산에 뭔가 억울한게 있어보여요. 손해본딘 생각하니 저런 행동이 나오죠.\n\n생활비 30만원이 님의 노동력값이라기엔 너무 헐값인데 그게 둘의 합의사항도 아닌듯하고요. \n\n둘이 대화가 많이 필요해보이는데 다 깔수없는 무언가가 있을수도 있고 부부의 일은 부부만 아는거니깐요.\n\n아무쪼록 잔돈으로 빈정 안상하게 님의 가사노동을 감사할줄 아는 그런 좋은 남편을 되찾길 바라요.'], ['대학생용돈이예요  시급도안나오는 돈이고요.  집안일 전혀안하고 집도 자기명의이고 뭐 싹퉁이 될께 없는데  진짜 화나요'], ['그렇다면 당신은 부처님! 그럼 잘생각해 보세요.\n나이들면 더하지 덜하지않아요.\n아침부터 굿모닝이여야하는데 이게 웬일!'], ['남편분이 생활비를 30요?? 0하나 빠진거 아니죠?? 무슨 대학생 용돈도 아니고'], ['대학생이죠ㅠ'], ['30은 생활비가 아니라 용돈 아닌가여..?'], ['30만원 생활비예요 ㅠ  주말 외식비랑 생필품사면 끝'], ['애생기고일그만두시면 남편 .... 더 심해질텐데ㅠㅠ 그전에 바로잡아야합니다'], ['전업주부하면 50만원가지고 다 생활하라고 할듯해요ㅜ'], ['너무하네요! 생활비 30이라니...;; 저라면 같이 30 내고 집안일도 똑같이 하자고 할래요! 생활비 낸거는 공용으로 쓸때만 쓰고...너무하네요!'], ['30만원이  고정이라서  턱없이 부족한거 제월급으로 써요 많이쓸때는 제가 60만원 매워서 써요'], ['그럼 나도돈버니 너도집안일같이하자하세요\n싫다하면답없어요 뭐하러같이살아요'], ['휴일에 토요일은 쇼파에서 티비랑 폰잡고 내려오지도 않고 일요일에는 회사출근해요'], ['에구..어찌 부부가 그렇게 지내나요오... 계산적인것같아 좀 그러네요..ㅠㅠ'], ['남같아서 같이 살고싶지 않고 뭔가 거리감이 크네요 30만원 주고 당당한 남자가 어딧어요ㅜ'], ['제말이요!!제가 아픈것도 병원비 자기카드긁고 돈 달래요'], ['많이 참고 사셨네요 .와이프 한테 부탁하면서 잔돈 받아달라고 한게 진짜 섭섭하죠.'], ['더 화나는건  퇴근길에 세탁물 찾아와달래요 이런 ㅂㅅ'], ['저는 제 월급에 대해선 남편이 아무런 관심 안가져요. 생활비, 공공요금, 휴대폰, 보험, 외식비, 애들학원비 모두 남편 월급통장에서 나가요. 가족부양의 책임은 남편에게 있고, 내 돈은 내가 모은다고 했어요.  허튼곳에 안쓰고 이사할때 몇천씩 제가 보태요. 맞벌이 하면 지출을 한군데로 몰고 한사람 월급은 모아야 목돈 만들어지더라구요.'], ['전 딱 생활비 30받고 제폰 제보험 저한테 들어가는 고정지출은 제가 내고 가끔 남편이 돈 얼마나 있냐고 물어보고 요 제가 인터넷 쇼핑하는거 엄청 신경써요'], ['맞벌인데 생활비는 왜 받아서 쓰시고 집안일은 왜 혼자하시는거에요?'], ['노예죠'], ['남편분이 저축하거나 대출금 갚고 계신가요? 그런거 아님 이해 안되네요'], ['저축은 모르겠어요. 결혼할때 생긴 집 대출금은 결혼하고 제가 모아둔돈 정리해서 반은 한번에 갚아서  지금결혼 5년차인데  다 갚앗을꺼예요'], ['둘다일하면 집안일도 같이해야죠 30이면 식비로도 부족할꺼 같은데요 혼자 돈모으나 라는 생각들꺼같아요'], ['식비로도 부족해오ㅠ 근데 둘다 집에서 거의 먹지않아요 . 먹어야 라면ㅠ'], ['왜 같이사는지 1도 이해가 안가는 상황이네요. 거지같은 남편이라도 아이가 있음 그래도 아이아빠니까 하는 이유같지도 않은 이유라도 있는데 ㅠㅠ 더늦기 전에 교통정리가 필요할듯한데요'], ['이번 추석이 교통정리할때인가봐요'], ['30... 두분식비만 덜렁주시는건가요?;;일은일대로 집안일은집안일대로 힘드시겠어요..아무리맞벌이라도 서로 어느정도벌고 앞으로 어떻게할거니까 얼마를 저축하고 생활비는 이렇게하고 뭔가 계획이라는걸 공유해야하는거 아닌가요?.저도맞벌이 했었지만 대부분 그러지 않나싶어요'], ['그런거없어요ㅜ'], ['같은 맞벌이인데도 왜그러나용 ㅜㅜㅜ집안일은 같이해야지요😭😭😭'], ['흠......어떻게 사실까요?  님 대단하시네요. 뭐 부부일이야 양쪽다 이야기를 들어봐야 하는거지만 님 말만들음 함께 살 이유가없네요. 30  ㅋㅋ 장난하나요..아 놔  가사도우미 주2회만 써도 32에요. 아놔 어이없네요.'], ['남편아니고 동거하는 룸메이트예요'], ['복받으실꺼에요^^'], ['ㅠㅠ복'], ['보통 남편 월급으로 생활비하고\n와이프 월급은 저금하죠~\n저는 그랬구요~\n근데 커피한잔도 니돈으로 사먹으라는건...\n농담 아닌이상~\n본인은 본인돈 벌어서 본인위해 모으며 하숙생도 아니고\n꼴랑30만원주면서~\n저같음 빨래도 밥도 집안일 전부 안해줍니다.\n각자 셀프로 해야죠~'], ['농담아니예요!!'], ['아이 생기기 전에...😭'], ['30만원은 왜 준대요? 아예 주지말지\n어이가 없네요 장난치나요 300 주라하세요'], ['아뇨!!아쉬운것도 남편이였어요!!전 아직도 32살 남편은 39살 5년전이면 전 27살이였는데 굳이 제가 왜 !!쓰면서도 이젠 그만 정리할때인가보다 하네요'], ['결혼전에 돈부분 상의 안하셨나요? 어처구니없네요 저같으면 같이 못살아요 30이 뭐예요 장난치나...왜결혼했대요?\n남편 30외에 남은돈들 어디에 쓰고 얼마나 저축하는지는 알려줘요? 이것도 안알려줄것같은데 ㅡㅡ 그리고 와이프한테 커피한잔 못사주나요? 연애때도 그랬어요?']]
    
    8076
    제일 하기 싫은 집안일이 뭘까요? 저는 화장실청소, 건조기 먼지털기, 빨래개서 서랍장 찾아넣기 등등 하기 싫은 일이 넘나 많아요누가 대신 좀 해주면 얼마나 좋을까요? ㅎㅎ.....이런 격한 공감일 줄이야요 🤦 이제 알겠어요 무보수로 하는 집안일이 너무 귀찮은거 같아요한 달에 2백만원 준다면 기꺼이 하나도 힘들지 않겠는데 말이죠 ㅎㅎㅎ
    
    [['저는 걸레질이요ㅜㅜ 환장하겠어요ㅜ'], ['아 저는 걸레질 안하는디요 ㅎㅎ\n생각나면 가끔 대충 밀어요'], ['전 집안일 다괜찮은디 빨래개는거젤싫어요ㅜ'], ['개는거까진 하겠는데 식구수대로 찾아넣는거 젤 귀찮네용ㅜㅜ'], ['저두.. 넣는거ㅋㅋㅋ 정말 구찮아요ㅜ'], ['맞아요ㅠㅠ 공감ㅋㅋ'], ['저도 그게 그렇게 구찮아요 ㅠ'], ['빨래개고 옷정리요'], ['빨래개는  기계좀 누가 개발했으면요옷가게도 아니고 각잡아 넣어야하니 진짜 죽겠네요 그걸 원하는 인간이 있어서요'], ['그 안간은 누구일까요? ㅎㅎㅎㅎㅎ'], ['군대갔다온 인간요ㅜㅜ군대에서 이상한걸 가르치니 문제입니다각잡는건 왜 가르치는걸까요배워서 결혼하면  써먹지도 않는걸요'], ['푸하하하  ㅋㅋㅋㅋ 역시나요'], ['에혀 본인옷 본인이 절대  정리안하고 각 잡은게 이게  뭐냐고  쑤셔넣기 대회 나가면 일등하겠다고 그러네요 옷도 군대처럼 스타일러같은곳에  던져 넣으면  각잡아 접어지는 기계좀 있었으면 좋겠네요'], ['스타일러 던져넣음 각잡아 지는 기계 개발하면 대박터질건데요 ㅎㅎㅎ'], ['그니깐요 왜 그런건 안말들고 뭣들하는건지요옷개는시간만 줄여도 청소시간 반은 줄더라구요'], ['요리..청소는 죽어라 할수있는데..아~  요리는ㅠㅠ'], ['저두 요리까지도 아니고 걍 밥하는 거 넘넘 귀찮아 죽겠어요 ㅜㅜ'], ['걸레질과 빨래개서 넣는거요 ㅋㅋ'], ['설겆이요. ㅠㅠ 너무 지겨워서 이모님들여놓을까 생각중요~'], ['저는 설거지요. 넘 시러요..오래 걸리고 힘들어요😭'], ['거지거지 설거지... ㅠㅠ'], ['저는 청소요'], ['빨래개는거요..ㅠ'], ['화장실청소 너무 싫어요..ㅜㅜ건조기 필터청소도 은근 구차닌요..ㅜㅜ'], ['전 청소여ㅠㅠㅠㅠㅠㅠㅠㅠ청소하는거 너무싫어요 ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ'], ['전 화장실청소요ㅠ\n너무싫은디..남편이  해주는것도 맘에 안드니  결국 제가 해유..'], ['계절별옷정리랑 설겆이요~음식하는것도요~'], ['개서 넣는거랑 걸레질이요.건조기 생겨도 꺼내는것도 구찮네요.'], ['밥하는거랑 걸레질이 세상싫어요 ㅜㅜ 그래서 요즘 일바쁘단핑계로 포장과 외식을 ^^; 걸레질은 롯봇님께 부탁드리네요'], ['저는 청소요~특히 물걸레질ㅎ'], ['욕실에 아이들때문에 미끄럼방지매트 깔았는데 그거 걷어내고 청소하는게 제일 싫어요 ㅜㅜ'], ['건조기를 이번에 구매했는데 \n이게 먼지청소 하는게 너무 귀찮네요.........ㅋㅋㅋㅋ 자꾸 한번만 더 돌리고 먼지청소하자 하게 되요..🤣'], ['빨래 널고 개고요.. 진짜 주방일이나 청소는 솔직히 재미(?)있는데 빨래는 왠지 모르게 너무 싫으네요 ㅡㅡ'], ['저도 밥하는거요 ㅠㅠ 오늘은 뭐먹지 생각하는거랑 재료 손질이 너무 귀찮아여 ㅠㅋㅋㅋㅋ'], ['빨래개서 옷정리하는거요'], ['화장실청소요  락스냄새도 넘 독하구요ㅜㅜ'], ['설거지요 ㅠㅠ 음쓰도싫어요!!!'], ['쓰레기버리기'], ['싲ㅌ.ㅣㄷ'], ['남편돌보기요...'], ['빨래널고 개는거요ㅜ'], ['설거지요 ..\n식기세척기가 하지만 그것도 정리해서 넣어야하니 진짜 그거 너무 하기 싫어요 ....'], ['화장실청소 걸래질 설거지!! 이것들 말고 다른건 좋아서해요ㅋㅋㅋ'], ['창틀 닦기요..'], ['전.. 빨래 너는거.. 딱 반듯이 널어야 했었어요.\n수건도 딱 반 애들 손수건도 딱 반씩 ㅋㅋㅋㅋㅋ\n그게 스트레스였는데 하기 싫지만 하면 저래 딱 맞아야 하구요.\n그래서 신랑이 건조기 사줬는데..\n이제 삘래 개서 넣는겤ㅋㅋㅋㅋㅋ싫어옄ㅋㅋㅋㅋㅋ\n그래서 개서 놓으면 애들껀 애들 시키네요 이제..ㅋㅋㅋㅋㅋ'], ['전 빨래 개고 세탁기에서 건조기 넣는거요 ㅋㅋ 싫어서 안하고 잇어요'], ['걸레질이 싫었는데... 물걸레청소기사고서는 걸레빠는게 싫어지네요ㅜㅜ 모아두고 빠는 저를 발견하고... 빨기싫어서 안돌리는ㅎㅎ'], ['요리하기랑 음쓰요ㅠㅠ'], ['화장실청소랑 요리요ㅠㅠ'], ['저도 빨래개서 장에 갖다놓는거여 ㅠㅠ 플러스 양말 찾으러 다녀서 득템한거마냥 좋아아하면서 장농에 놓는 저를 보며...참....... 고생이 많구나 생각이들어요 ㅋㅋㅋ'], ['전 요리요. ㅠㅠ'], ['설거지.물걸레요~^^'], ['빨래는 세탁기가 하지만 빨래개고 넣어놓기는건 제몫 이게 젤 싫으네요.청소는 청소기가 물걸레까지 해주니 만족하는데 이건 누가 해주지도 않고ㅜ 누굴 시키자니 마음에 안들고 \n차라리 마음에 안들바에 제가 고생하자 생각하고 해요 ㅋㅋ'], ['걸레 빨기 화장실 청소여'], ['삼시세끼 밥 차리는거요.. 정말 밥차리다 하루 끝나는거 같아요 일이 끝이 없는 느낌 청소는 마무리라도 있는데 ㅜㅜ'], ['전 화장실 청소여'], ['전 청소기 돌리는것만 좋고 나머진 다 싫어요ㅋㅋ'], ['옷정리요ㅜㅜ\n전 설거지가 차라리 나아요'], ['설거지요 밥 배불리 먹고 다들 각가 쉬거나 할 일 하는데 혼자 뒷정리하는게 좀 그래요...  \n화장실 청소는  해도해도 답 안나와요 \n식구가많으니 변기통은 늘 붙들고 닦는듯요ㅜㅜ'], ['걸레질이랑 화장실청소요ㅎ'], ['저는 변기청소요 ...ㅎ'], ['그냥 다 싫어요...'], ['다싫어요ㅠㅠ저도차라리 일만딱 하고싶네요ㅠㅠ'], ['저도 다 싫어요ㅠㅠㅠㅠㅠ'], ['빨래넣는거요 ㅜㅜㅜㅜ']]
    
    8118
    집안일 안하는 와이프 어떻게 하면 좋을까요? 집안일 안하는 와이프 어떻게 하면 좋을까요?​친구나 지인들한테는 말하기 창피해서 말도 못하고 답답해서 보내봅니다​결혼한지 5개월 됐고 와이프는 결혼준비하면서 직장을 그만둬서 현재는 외벌이 상태입니다.지금은 다른일 하고 싶다고 하루에 2시간 정도 공방을 다니고 있습니다.​저도 자취생활을 오래해서 집안일이 얼마나 힘들고 귀찮은지 알기 때문에결혼을 하게되면 저도 최대한 도와줄려고 했습니다.​​근데 문제는 와이프가 아예 집안을 일을 안해요설거지 같은 경우는 집에 있는 그릇, 수저 다 쓸때까지 안해요. (이틀 이상)이틀 이상 되면 설거지 통이 꽉 차는데 그래도 안하고 결혼할때 친구들한테 받은 식기세트를열어서 그걸로 사용해요집에 식기세척기가 있지만 덜 씻기는것같다고 찝찝해서 안쓴데요처음에는 목마른 사람이 우물을 판다고 제가 했죠그러다 보니 안하면 제가 하는줄 알고 당연히 안해요​빨래같은경우도 몰아서 한다고 일주일에 한번하는데 수건이 없어서정말 주방티슈에 얼굴 닦은적도 많구요​집안은 뭐 말할것도 없죠....​제일 화가 나는건 결혼 전 제가 자취를 할때 저희집에오면 항상 알아서 먼저 저희 집 청소도 하고 해줬거든요....그런데 결혼하고 완전 바꿨어요...​​설거지 같은 경우는 몇 달 동안 말해도 전혀 안들어서너무 화가나서 선물해준 친구들 한테 미안하지만 집에 있는 그릇 6개, 수저, 젓가락 4벌 빼고 다 버렸습니다.그릇, 수저가 적으면 설거지 거리도 적을거고 쓸게 없으면 설거지 하겠지 하는 마음에서요​근데 그날 밥상에 올라온건 나무 젓가락, 플라스틱 수저였어요....​같이 일을 하거나 임신을 한거라면 다 이해하고 제가 다하는데...집에서 놀고 있으면서 아무것도 안하는게 이해도 안되고 너무 화가 나요​​장모님이 와서 집안꼴 보고 저한테 미안하다고 할 정도인데와이프는 정신을 못차려요...​정말 5개월 동안 사정도 해보고 화도 내보고 다 해봤는데 1도 안바꿔요전혀 문제를 못 느끼는것같아요...​​그냥 내가 하고 살려니 너무 화가나고 힘들고뭐라해도 바뀌는거 하나 없고...​​이거 어떻게 하죠?​집안일 때문에 이혼을 생각할거라고는 정말 1도 안했는데.....​이와중에 와이프는 임신계획 잡고 있어요.....​​정말 답답해 미치겠어요... 이거 어떻게 해야하나요?
    
    [['잘안바뀌던데...사람성향이라서ㅜㅜ역할분담을해서 이틀에 한번꼴로 맡은일하자고 약속을 해놓는건어떠세요?아니면 방법이😂'], ['솔직하게 이러한 부분 때문에 스트레스 받는다 등을 담백하게 얘기하세요.. 대신 부드럽게 상대방이 기분 안 나쁘게..그게 어렵지만요. 그래도 안 비뀌어지면 솔선수범 계속 하시다가 그래도 안 바뀌면 안 바뀌는 겁니다... 그러면 본인이 떠안고 살거나 그게 싫으면 못사는거죠..'], ['여자분에게 이야기해보시고  그래도  안바뀌면 못살죠..차라리 혼자 사는게 속편하지  상전 모시고는 못살죠'], ['스트레스 안 받게 일찍 정리하시는것도 괜찮을듯'], ['그동안 노력했는데도 안되시는거라면마음을 가다듬으시고직접 다하셔야죠다만 아이 낳고 더 심한일 생길까 걱정은 되네요'], ['현재도 그정도라면  아이가 생기면 더 심해지실듯\n빨리 결단을 내리시는게 서로에게 좋을수 있습니다\n'], ['저라면못살듯.. 저도 정리정돈잘하는편은 아니지만 저정도는아니에요ㅋㅋㅋㅋ.. 부부생활의 왜 남편만집안일을다해야하나요? 다 나눠서서로돕고하는거지..거기다일까지쉬면 시간도많을텐데 ..'], ['노력을 하고 대화를 했음에도 안바뀐다면..\n그냥 헤어지시는게 맞는것 같아요.\n저런 성향들은 절대 바뀌지않아요.'], ['성향은 안바뀝니다. 님이 마음을 바꿔먹는게 빠를겁니다. 아내분은 님이 지적하는 그런 것들이 눈에 보이지 않을겁니다. 님 눈에만 보이죠. 그래서 님이 스트레스를 받는 거구요. 그러나 님이 아내분을 사랑한다면 님이 하시면 됩니다. 눈에 보일 때마다 싸우고 화내고 스트레스받고 지치면 같이 못살거든요. 님이 맞추던지 헤어지던지 입니다.'], ['대화해보고 변화가 없다면 답이 없다고 생각이 듭니다\n한 가지 해볼 수 있는 방법이 하나 생각나는데요, 넛지(nudge)를 활용해보는 겁니다\n직접적으로 말꺼내다가 사이가 더 멀어질 것 같다면 슬쩍 옆구리를 찔러주는거에요\n\n예를 들어서, 티비나 유튜브 등으로 본문에 나열하신 내용으로 부부가 전문가에게 상담받거나, 이혼 조정등으로 가는 과정을 다룬 방송을 같이 한번 보고나서 대화해보는 방향으로요\n아내분도 남편분의 의중을 어느정도 알게되고 행동이 변화하지 않을까 싶습니다🤔'], ['아이고 ㅠㅠㅠ  정말 바뀔수없는걸까요? 먼가 이유가 있지않을까 싶어요.  그냥  단순히 살림하기 싫다는건 아니실것 같은데...'], ['신혼이시니까 ..진지하게 고민이 있다고 얘기 좀 하자고 하시면서 말씀 나눠보세요 술 드시지 마시고 ..집안일 하는게 부담되냐고 좋게 시작하시고 내 생각에는 두 분이 돌아가면서 당번을 정해서 설거지,빨래, 청소, 쓰레기 버리는 날 이런걸 정하셔서 표를 작성하심이 어떨까요? 부인분 의견 들어보시고 그래도 안 하시면 정말 답이 없어요 애기 낳으면 아기도 무시할 사람 같네요ㅠㅠㅠㅠ'], ['사람이 바뀌기 싫다지만 저건 그런거랑 다른거 같습니다. 게으른거 아닌가요?  성격은 딱 보고 알자나요 게을러서 안하는건지 다른 이유때문인지\n게으른거면 애 나와도 육아도 님이 하실듯 \n잘 생각해보시길'], ['사람 안 바뀌죠.. 아이요? 절대..  아이를 위해서요.. \n\n남자분이 다 해줘서 그래요.. \n마음 정리 하세요.. 아내분도 이유가 있겠지만.. 남자분이 그 이유를 이해하실것 같지 않네요..'], ['더 늦기전에  ㅡ애 생기기전에ㅡ 정리하는게 쓰니님 정신건강에 좋을듯요 아님  쓰니님이 내가 선택한 여자니 다 끌어안고 가겠다로 집안일  해얄듯요 주위에  그래서 결혼하고  살이 칠키로 빠진분 있다고 들었어요'], ['집안일은 도와주는게 아니라 같이하는거에요~^^...  여자분이 결혼전에는 어땠나요?(님 자취방말고 자기 집에서도 그랬나요?)식기류가 없다고 1회용을 쓰시다니.. 하하;;게으른것보다 본인은 안하고 쌓아놓고 살아도 전혀 불편한게 없는데?    식기류 없으면 1회용품 써도 되고, 냄새 정도는 난 참을 수 있고..불편한 사람이 하겠지..이런거같네요;;이와중에 임신 생각이시라니;;진지하게 얘기하면서 도와준다는 말 하지마시고, 집안일 하는데 서로 분담하자고 해보세요.퇴근하고 와서 나도 집안일하는데~!이러면 더 안좋아 질 수 있으니 분담하는거에 대해서만 얘기하구요ㅠ왜 안하는지  .. 나까지  집안일을 안하면 우리집은 어떻게 될지...  그래도 안한다고하면 뭐.....같이 못 살거 같네요..여름에  먹고난 설거지 그릇들이 쌓이면;;;'], ['남편이 집 밖에서 고생하면 집에 있는 여자는 집안 내조를 잘하던가.. 어휴 외벌이 하는데 참..'], ['지금은 설거지만 쌓이죠...애있으면 집이 쓰레기장이 된다네요 저 아는사람 와입도 그런데 매일 배달시켜서 식탁에 쓰레기 그대로있고 거실에 과자봉지 굴러다닌데요 넘 힘들다고 이혼하고싶은데 애때문에 못한다네요.. 안바뀌나봐요'], ['맞벌이 부부라면 집안일을 서로 도와가며 하는건 맞다고 생각하지만 남편 혼자 외벌이라면 아내가 거의 전적으로 맡아서 하는게 맞다고 생각합니다..\n\n저도 최근에 퇴사 후 쉬고 있는데 남편은 저한테 계속 집안일도 하지말고 밥도 하지마라 그냥 쉬어라 하는데, 말은 고맙지만 그래도 할건 다 합니다\n남편이 설거지라도 하려하면 제가 못하게 해요 남편이 고집부려서 어쩔 수 없이 냅둘 때도 있지만요...\n솔직히 밖에서 스트레스 받아가며 힘들게 일하고 온 남편을 집안일 시키고 싶지 않아요\n저는 집에서 편하게 쉬고 있잖아요?\n그러니 시간도 많고.. 여자 혼자 집안일 충분히 할 수 있다고 생각해요\n물론 맞벌이면 이야기가 달라지겠지만요\n\n글쓴님 같은 경우엔 와이프분 성향이 바뀔 수 있도록 계속 대화 해보시고 그게 안된다면 어쩔 수 없이 이혼밖에 답이 없을것 같네요\n그리고 이런 상태에선 애기 절대 낳지 마세요 후회합니다'], ['화내거나 말을 계속 하고 극단적으로 버리기까지 했는데도 나무젓가락?플라스틱? 이 무슨 말도 안되는 소리죠. 집안일 안 맞아서 이혼하는 사람 많습니다. 신혼인데도 벌써 이러면 갈수록 심해지면 심해졌지 절대 안바뀝니다.. 이런 상황에서 아이까지 생기면 육아까지 다 맡기려고 할 것 같네요. 과감하게 결정해야합니다'], ['무슨 심리적으로 문제가 발생된건지요?\n분담에서 자기 할일만 하면 되는데 아예 안한다는건 정말 한쪽에선 지치는데요 이 상황에 임신은 반대입니다 제발..'], ['와이프는 눈치채고 이혼 못하게 하려고 임신계획 세우는거 아닐까요? ㅠㅜ 결혼이 무섭네요.'], ['와이프가 아니라 반려인이네요. 반려동물과 비슷한 수준의 케어가 필요한 듯... 맘 고생 많으시겠어요.'], ['아내분 정말ㅜㅜ . 이대로라면 애기 낳은 후도 더 큰 문제에요 ㅜㅜ  진지하게 대화를 해보셔야 할 듯..'], ['집청소 습관이 안든 젊은 사람이 많습니다.. 포기하시고 주 2회정도 청소용역 부르세요. 저 습관은 안고쳐져요. 자칫하면 쓰레기집 되구여'], ['말을 해도 지금 개선이 안되고 있는 상황이자나요?? 점잖게 말씀하지 마시고 확 엄청 쎄게 말씀해보세요. 지금 애 낳을때가 아니라고 생각되네요.'], ['아 사연보니까 끌어 안고 살 자신 없고 계속 스트레스 받으신다면 답은 이미 나와있는거 같네요 애기낳으면  더  힘드실텐데...그 전에 해결이 되야 될거 같아요 좋은해결되시길  바랍니다.'], ['배려없는 결혼생활은 악몽과 같을거예요\n\n노력하고 다짐받아도 실천안되고 개선이 없다면 답은 나온게  아닐까요..'], ['와 말을 해도해도 안통하면 포기해버리세요 아내가 일도 안하는데 아무것도 안하면, 돈 쓴이님이 관리하면서 용돈 주지마세요 아예 집 청소할때까지 밖에 나가계시는것도 좋을듯 여자분 스스로 느껴야해요 무엇이 잘못됬는지'], ['청소 안하는 이유가 도대체 멀까요?;;;;;;;; 성격 습관 바꾸기 정말 어려워요...잘 생각해보세요 ㅠㅠ'], ['같이 사는집인데 전 싫어요노력해도 안되는게 아니라 노력조차 안하고 있는거잖아요하루 공방 2시간...뭘 배우시는진 모르겠지만 몸이지칠만큼 노동적인건가요? 집안일 사실 힘들죠 저두 직장인이지만 퇴근하고 세탁기돌리고 그 사이 청소기 걸레질 하고 나면 진빠져요그치만 내가 사는 집이고 그 먼지 다 내입으로 들어가고.. 특히나 화장실은 이틀만 지나도 바닥에 머리카락이며환기안시키면 꿉꿉한 냄새납니다...집에있는 수저나 그릇을 다 쓸정도로 설거지 안하는건 게으른거 말곤 뭐라 할말이 없는것 같아요..갑작스런 단수거나 공사중이여도 미리 물받아놓고 씻을건 씻는 사람도 있어요결혼한지 5개월인데 공방다니는다는 핑계로 집안일 손땐거면애기생기면 더 할꺼에요 그 뒤치닥거리 고스란히 글쓴님 몫이구요 임신계획인 와이프 앉혀두고 말씀하세요..집안 환경이 이러면 임신해서도 서로가 힘들고 아이가 태어나도 병치래 한다구요..'], ['이런집도 있군요.. 신기하네요 ㅠㅠ 우째요 ㅠㅠ 애 생기기전에 갈라서는게 낫지 않을까요? 앞으로 몇십년을 살아야하는데 ㅠㅠ'], ['애까지 생기면 육아는 과연 할까요..? 몸도 더 힘들어질텐데...결혼하고 나서 아예 돌변했다는게 얼척이 없네요. 노력하고 뭐고 다 둘째치고 의도적인거같게 느껴져요;;;저는 남친이 가끔 놀러왔을때도 본인이 어지른거 안치우면 화가나는데, 글쓴이님은 어련하실까요..이혼 생각 하실수 있을것같아요....'], ['익명님.많이 힘드실 것 같아요.그리고 많이 착한 분이라는 생각도 드네요.결혼 전의 행동과 결혼 후 달라진 아내의 행동에 어떤 모습이 진짜 모습인지 많이 혼란스러울 것 같기도 하구요.오죽했으면... 익명게시판에 글을 올리셨겠어요.댓글의 절반이상... 전부가 이혼하라는 이야기가 먼저 나오는데.결혼도 쉬운 일이 아니고이혼도 쉬운 일이 아니라 생각됩니다.결혼이라는게 누구 덕 보려고 결혼하는 것은 아니라고 생각해요.지금 각자 살아온 인생만 20년이 넘을 텐데... 보통의 부부들은서로 맞춰가며 행복하게 살려고 결혼을 합니다.아내분은 이 결혼... 무엇 때문에 했을까요?반대로.. 남편분은... 어떤 이유로 아내분과 결혼까지 이어지게 되었을까요? 대화도 많이 해보신 것 같습니다.아내분이 원래부터 치우는 걸 싫어했던 사람인지...정신증이 있어서 치우는 것을 못 하는 사람인지. (우울증 및 강박증이 있어도 치우는 것을 못 하는 경우가 있습니다.이 경우 저장강박으로 넘어가면 이제는 버리는 것도 못 하고 집안에 계속 쌓이기만 하고결국 쓰레기장이 됩니다.)가장 잘 아는 사람은 장모님이라고 생각됩니다.혹은 정말 아내분의 베프... 원래 내 친구는 정말 드러운 애입니다. 라고 말해주는 친구혹은 아내분의 전 회사사람... 회사에서도 책상을 보면 알수 있다지요. 아이가 태어나면지금 보다 더 깨끗한 환경에서 생활해야하는데...결정은 남편분이 하시겠지만...부디 현명하게결정하시길 바랄게요. 몸도 마음도 건강하세요.'], ['아무리 성격, 성향 자라온 환경 등이 있다 하더라도... 결혼을 한 이상 노력해야 한다고 생각합니다. 다시 한번 진지하게 대화 나눠보시고 진전이 없다고 하면 결혼 생활을 다시 생각해보셔야하지 않을까요? 저라면 그렇게 할 것 같아요...'], ['와.......... 뭐지.......저런 분도 결혼을 하는구나.........ㅠㅠ남편분이 얘기를 해도 안들어먹으니.....아내분은 부모님 보기도 부끄럽지 않은가 보네요....'], ['결혼전 자취집 청소는 그렇게 열심히 해주구선 결혼후 같이 사는 집 청소는 안한다? 그것도 여자가??? 좀 이해가 안됨;;; 여자가 결혼후 바뀌었다는것도 첨 들어봄;;; 그것도 청소로 ㅋㅋㅋㅋ 그런 여자가 있구나;;; 하;;;; 뭐든 다 이유가 있을듯한데;;; 심각하게 서로 대화를........'], ['뭐하느라 집안일을 안하신대요? 정도가 심해서 아내분 잘못인듯 하긴 한데, 그래도 부부니까 사유라도 한번 들어보는게 중요할 것 같아요. 쓰니님 출근하신 시간에 내내 게임? 아니면 혹시 디스크라거나 이런 식으로 어디 아프신지?'], ['우선 얘기를 하고, 1. 설거지가 다 썩어서 문들어질때까지 놔둬보세요. 2. 같이 살아야 겠으면 가사도우미 불러야죠.'], ['아 스트레스 많으실듯...  근데 도대체 왜그러는지 원인부터 찾아봐야할것같긴해요.. 모쪼록 잘 해결되셨음좋겠어요..'], ['아내분이 많이 게으르세요. 혹시 어떤 계기가 있는 건가요? 아니면 원래부터 그런 분인가요? 대화도 해보고 여러 노력 해보셨는데도 상황의 심각성을 대소롭게 여기지 않는 아내분의 태도가 화가 나는 부분이네요. 지금의 행동들이 전혀 이해가 되지 않으면 결혼생활 이어나가기 힘들어요. 아이 낳는 것은 일단 미루셨으면 합니다. 마음의 준비가 전혀 되어 있지 않아요. 우선 같이 살아가는 부분에 대한 합의와 이행이 병행되어야 미래를 그려볼 수 있다고 생각합니다.'], ['아이가 태어나면 그 집안일의 더블로 더 집안일이 생기는데 그거 다 감안하셔야 해요.'], ['안녕히 계세요 하는게 좋을 듯...'], ['글쎄요 화가 나긴 하겠지만 이혼할만하진 않아요 우울증이 있을 수도 있고. 정말로 하기 싫다면 돈을 벌어서 청소해주시는 분을 불러서 같이 화목하게 사는 것도 방법이에요. 분명히 이상한 태도긴한데 방이 더러운것만 문제라면 해결법이 나름 명쾌한 부분이에요. 확대해서 이런 문제가 더 있겠다 하지마시고 빠르게 해결 보시고 더 평화롭게 사시는 걸 추천. 분명 좋은 부분이 있어서 하신 걸거에요. 사람만나기 얼마나 어려운데요. 결혼은 더어렵고요.  분명인연이 있었을거라고 믿어요'], ['안타깝네요. 위의 바닐라라님에 동의합니다. 이미 설득을 위한 충분한 대화를 나눠보신 듯 하네요. 원인을 찾아서 후회없는 결정을 내리시길 바랍니다.'], ['생활습관이란게 잘 고쳐지는게 아닙니다. 두 분이 대화를 통해서 일 분담을 하셔야 될 거 같네요.'], ['집안일 못 하면 적어도 맞벌이를 해야죠 애 갖는건 미루세여'], ['여자지만 최소한 이런아내는 돼지말아야지,,'], ['혹시 아내분 우울증이있으신건아닌지....정신건강은 괜찮은지 잘살펴보시고요 잘해결하길 바랍니다~'], ['이건 아닌듯. 저라면 못살듯요 ㅠㅠ'], ['사람 안바뀌는데 애없을때 빨리이혼하시는게 좋을텐데요'], ['집에만 있으면 사람이 더 게을러져요... 퇴근후 피곤하시더라도 운동같이 해보세요. 땀흘리고 그러면 몸에 아무리 기운없고 피곤하더라도 몸에 활력이 돌더라고요.'], ['형.. 그거 아냐.. 도망쳐..'], ['... 어떤 성격중에는 일을 미룰수 있을때까지 미루는 성격이 있다고해요ㅠㅠ그래도 진심으로 이야기 함이 좋을듯하네요 집안일이 많아도 하루 1시간만 바짝해도 금방 끝나긴 해요아이가 없으니 1시간 하면 끝나는데 이게 미루면 3시간 4시간 걸리는게 집안일이라서요'], ['같은여자이지만..아내분 너무 심해요..ㅠ 장모님께 혼을좀 내보라고 해보세요!! 그것도 안먹히려나..너무 힘드실거 같아요..적어도 설거지 빨래는 기본 아닌가요ㅠ 근데 아이 있으면 아이때문이라도 하지 않을까 싶은생각도 들지만 그 반대면 남편분이 너무 힘들거 같아요!! 왜그러세요 아내분.......................ㅠㅠ하루종일 모하는지 ..'], ['이런 와중에 임신 계획이라..... 육아까지 하시게 생기셨네요. .. 집안일도 저리 안하시고 위생 개념도 남다르신데... 육아는 잘하실지....'], ['하고 안하고를 떠나서요저건 상대를 생각하는 마음 배려가 있으면 자동적으로 하게되는거라 생각합니다.꼭 해야할것과 안해야할거 이런거 사실 얘들도 아니고..많이 답답하시겠네요..힘내세요;;'], ['이래서.. 결혼하기전 동거가 필수인것 같아요.. 사람은 고쳐쓰는거 아니래요.. 보살이 되시던가.. 임신하기전에 헤어져야죠'], ['ㅎㅎㅎ저는 그런것때문에 가끔씩 청소업체 불러요.~ 그럼 서로 편합니다~'], ['사랑과 전쟁에서 본듯한 내용...'], ['좀 찔리긴 합니다저도 집안일 하기 힘들어서 잘 안하는 여자1입니다..거의 남편이 하죠..그래도 남편이 치울때 눈치보며 치우긴 해요..설겆이나 빨래도 할 수 있을때 하구요..밥은 제가 차려주죠..전 일을 하고 있어서 그런지 남편이 항상 미안해하더라구요..'], ['일도 안해..살림도 안해..심각하네요..둘중에 하난해야지...'], ['결혼하고나서 본성이 드러나면 정말 어떻게 해야할까요.. 듣는 제가 난처하네요.'], ['하 답답하네요... 정말 힘드실거같아요ㅠㅠ']]
    
    8120
    집안일 중에 뭐 좋아하세요? ​질문부터가 좀 그런가요ㅋㅋㅋㅋ...​좋아하는 건 없지만굳이 꼽자면 저는 청소기 돌리기랑 설거지하기인데​빨래 개는 거 진짜 귀찮아요ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ​​퇴근해서 집 청소하고 나니까 이 시간이에요...😂😂​청소하고 앉아서 문득 생각난 게저희 집은 집안일을 나눠서 하거든요...이것도 힘들다고 찡찡거리는데... 매일 집안일 하시는 나무님들자취하시면서 혼자 다 하시는 나무님들일하시면서 식사도 준비하시고 집안일도 하시는 나무님들다 존경스럽다는 생각이 들었어요...​저는 제 방 닦고 제 몸 씻는 것만으로도 벅찬데...​
    
    [['집안일 다 싫어하는데요🤭ㅋㅋㅋㅋㅋㅋㅋㅋㅋ\n내외하는 냥이들 귀욥'], ['저두 싫어하지만 찡찡거리면서 해요 ㅋㅋㅋㅋㅋ😂😂'], ['전 설거지랑 청소기 돌리는거 재밌어요. 빨래 돌려서 너는거까지는 좋은데 개는거 진~짜 싫어요ㅜㅜ'], ['저랑 똑같으세요ㅋㅋㅋㅋㅋㅋ빨래 개는건 이상하게 진짜 싫어요ㅋ'], ['냥이들도 거리두기 중인가봐요ㅋㅋㅋㅋ\n저는 빨래개우는거 참 좋아하는데 \n갠거 정리하는게 싫어요^^;;;;;'], ['제가 정리해드릴게요...제 빨래도..개워주세요ㅠㅠㅠㅠㅜㅜㅜㅠㅜ'], ['덕존 정리하깈ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 좋아하는 일인데 1년에 한번 할까말까'], ['ㅋㅋㅋㅋㅋㅋㅋㅋ덕존 만들고 싶어서 진열장 샀는데 아직도 미루고 있는 사람🙋🏻\n신나는데 자꾸 미루게돼요 ㅋㅋㅋㅋㅋ'], ['그나마.. 상닦기..? 아 집안일 극혐 극혐'], ['상닦기 제일 금방 끝나는데 제일 귀찮아요..ㅋㅋㅋㅋㅋ 그래서 저는 쟁반에 두고 먹어요ㅋㅋㅋㅋ'], ['전 요리하는거 좋아해요  뚝딱뚝딱 해서 만들어 먹음 좋은데 설거지는 싫어요'], ['제가 설거지 할게요 요리해주떼요'], ['저 세탁기 돌리기요ㅋㅋ세탁기에 빨래쏟아넣고 세제넣고 뚜껑닿고 전원 누르는거만'], ['헐 제가 좋아하는거 \n제가 해드릴게요\n저 완전 잘해요'], ['빨래너는거요^^\n내스퇄대로 각잡아서 너는거 좋아요🙋🏻'], ['톽톽 털어서 널면 각잡혀 말려져서 좋은데 ..저는 보는거만 좋아요 보는거만'], ['고양이도 거리두기 중인가요 ㅋㅋㅋㅋㅋㅋ귀여워요!!\n전 집안일은 재미 없는거 같아요🤭 ㅋㅋㅋㅋ'], ['저는 재미도 재주도 없습니다ㅠㅠㅠㅠㅠㅠㅜㅠ안하면 먼지 쌓이니까 울면서 해요ㅋㅋㅋㅋ'], ['집안일 다 싫어여 ... 그래서 여행가면 그르케 좋아요ㅎㅎ'], ['집안일은 다 싫은데 전 여행다녀와서 캐리어 정리하는거 좋아해요ㅋㅋㅋㅋㅋㅋㅋㅋ이상한 논리'], ['저는 설거지요! 유튭 켜놓고 대장 영상 보면서 하면 시간 순삭이에요!!!\n입덕하고 설거지 시간 좋아진건 안비밀😆'], ['저는 보면서 설거지하면 넋놓고 보느라 아마 물 콸콸 낭비..할 거 같아서 후다닥 해치워버려요ㅎㅎㅎ'], ['저는 빨래개는거요ㅋㅋ딱 개는거까지만 좋아해요 넣는거 징쨔 너무 귀찮아요ㅠㅠㅋㅋ'], ['개는것도 넣는것도 귀찮아요ㅠㅠㅠㅠㅠ갑자기 엄마한테 감사한 마음이 드네요ㅋㅋㅋ'], ['제방은 절대 아무도 못건드려서 제방만 제가 청소하고 제옷만 빨래해요☺\n냥이들 v 자로 앉아있는거 귀여워요🐱🐱🐱'], ['ㅋㅋㅋㄱ빨래까지 하시는 나무님 👍🏻👍🏻 저는 다 못해요ㅠㅠㅠㅠㅠ'], ['전 집안일보단 .... 회사나가는게 적성에 맞네요 ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['저듀 일하는게 더 나은거 같아요...티가 안나요 집안일은ㅠㅠ'], ['오늘 설거지랑 청소중에 뭐할거냐고 해서 못골랐어요ㅋㅋㅋㅋㅋㅋ둘다 싫어요ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['둘 다 다음으로 미루고 싶습니다ㅋㅋㅋㅋㅋㅋ'], ['그나마 좋아하는건...세탁조 청소하는거요 클리너만 부어 넣으면 세탁기가 알아서 하니까요 ㅋㅋㅋ'], ['클리너 부어주기만 하면 되니까 얼마 전에 했어요 \n모든 집안일이 세탁기처럼 알아서 돌아가면 좋겠어요ㅋㅋㅋㅋㅋ'], ['하!!!성격이 삐~~~해서  아침저녁으로 물걸레청소하느라~퇴근후 여태 빨래, 청소 ,음식~ 댕이산책 냥이 화장실청소~와!!!!겁나 마니했어요~하고싶어서 한건 하나도없어요~'], ['나무님 진짜 대단하세요!! 혼자서 다하신거에요?ㅠㅠㅠㅠㅠ이제 푹 쉬세요'], ['제목부터 맘에안들어요ㅋㅋㅋㅋ좋아하는집안일이라뇨ㅜㅜㅜㅜㅜ'], ['ㅋㅋㅋㅋㅋㅋ그래서 질문부터가 이상하다고...ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ말이 안되죠ㅋㅋㅋㄱ좋아하는 집안일이라니😂😂'], ['음...화장실 청소???\n제 전용 화장실은 앉아서 밥도 먹을수 있어요 ㅋㅋㅋㅋㅋㅋㅋ'], ['ㅋㅋㄱㅋㅋㅋㄱㅋㅋㅋㅋㄱㅋㅋㅋㄱㅋㅋㄱㅋㅋㄱㅋㅋㅋㅋㄱㅋㅋㄱㅋㅋㅋ 얼마나 깨끗하기에 밥을 거기서👍🏻👍🏻ㅋㅋㅋㅋㅋ ㅋㅋㅋㅋㄱㅋㅋㅋㅋㄱㅋ'], ['건식으로 써서 ㅋㅋㅋㅋ 다들 모냐고 놀래요 😁 손 닦기 부담시렵다고 ㅋ'], ['전 그나마 설거지,,?ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 근데 애옹이들 너무 귀엽게 앉아있는거 아녀요? ㅠㅠㅠㅠ 아웅 귀여웡😁🥰'], ['냥이들이 낯도 안가려서 다가가니까 와서 빤히 보더라구요 귀여운 냥냥이들'], ['오모오모😍 미쳐따🤭'], ['없.음!!!!!! ㅋㅋㅋㅋㅋ 젤 시른건 빨래 널기요 ㅋㅋㅋ'], ['없는게 정답이죠 정답!! ㅋㅋㅋㅋㅋㅋㄱㅋ'], ['다른건 괜찮은데설거지 싫구요빨래개는거는 좋은데 칸칸이 넣기가 그르케시러여 ㅎ 알아서 기어 들어가주면 좋겠어요😆😆'], ['빨래 돌리면 알아서 개서 나오면 좋겠어요ㅠㅠㅜㅠㅜㅜㅠㅠ'], ['전부 싫어요ㅋㅋㅋㄱㅋ 좋은게없어요.ㅜ.ㅜ'], ['그나마 설거지...? 제방 안치워서 엄마요정이 저 몰래 청소기 돌려요ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['요리하기는 좋은데 다싫지만 어쩔수 없이 해야죠...ㅠㅠ집안 더러운거 못보는 사람 저요...'], ['저는 청소 설거지 좋아하는 편이에여😬'], ['진짜 다 싫어하는데 그나마 설거지는 좀 해요ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['설거지... 저요ㅋㅋㅋㅋㅋ 심지어 그걸로 머니도 법니다ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['집안 일 다 싫어해요ㅠㅠ 불량 주부ㅠㅠ']]
    
    8162
    끝도 없는 집안일.... 같이 골라주세요 * 댓글을 이렇게나 많이 주실줄은 정말 몰랐네요.일일히 답댓글 못드려 죄송하고요. 민트색을 제일 많은 분들이 골라주셨네요.님들 의견에 따라 민트색 주문할게요~댓글 주신 분들 모두 정말 감사합니다~^^​​​요즘 일교차가 심하죠? 밤에 잘때 춥길래 어제 겨울 이불로 싹 교체하고식구들 덮던 가을 차렵이불, 패드 등등 빨래하느라 세탁기를 4번 돌렸나봐요~​작년에 남편이랑 저랑 각자 이불로 바꾸고 나름 고심해서 골랐던 이불커버가 솜이랑 겉돌고 꼭 보자기 천 같고 이상해서 돈만 날리고..ㅠㅠ​이 진그레이색으로 주문해서 덮었는데요.​방이 너무 어두워 보이는거 같고 여유분으로 저렴이 커버 한세트 더 장만하려고 열심히 골랐는데요.저 몇 번으로 주문할까요?? 같이 골라주세요~​
    
    [['전 민트!'], ['저도 민트요'], ['민트요!침대 넘 이쁘네요 ㅎㅎ'], ['전 버터요.'], ['저는 버터요🙂색넘이뿌네여'], ['버터색요'], ['버터요'], ['1 버터요^^'], ['버터색'], ['민트요~~ 정보 부탁드려도 될까요?'], ['제가 사용해본게 아니어서 알려드려도 되나 싶지만...쁘리엘* 제품이에요.3개 다 오늘의🏠 쇼핑몰에서 봤어요~'], ['전 정보좀 주세요 ㅎㅎ커버 바꾸고싶어요~~'], ['3개 다 오늘의🏠 쇼핑몰에서 봤어요~이거 말고도 예쁜거 엄청 많으니 구경한번 가보세요~^^'], ['버터가 바닥색이나 프레임색에 어울릴거 같아요 근데 혹시 침대 싱글2개로 제작하신건가요?'], ['침대는 제작 아니고요.. 시몬* 제품이에요. *몬스 국민프레임이라고도 하더라고요~ㅎ'], ['민트요~~^^'], ['버터색이요'], ['민트!'], ['민트요^^ 안에는 거위털  솜 이런거따로넣는건가요? 폭신해보여서요^^\n침대도 어디꺼인지궁금하네요~'], ['이불 커버 안에 구스 솜 넣었고요..ㅅ몬스 침대에요~'], ['감사합니다 ^^ 알아봐야겠어용~~~ \n내년 이사예정인데 남편한테 보여주니 넘 편안해보인다며~ 찾아볼께용~'], ['전 민트나\n그린이요~'], ['저는 화이트요'], ['민트요'], ['민트예쁘네요. 버터색 관리 힘들더라구요\n 구스는 자주 빨수도 없으니깐요'], ['민트'], ['버터요'], ['민트 상콤하네요~'], ['민트 이쁘네요^^'], ['버터요'], ['버터요'], ['민트나 그린이요~'], ['저두 민트요!!^^'], ['버터요'], ['버터색이요. 잘어울릴거 같아요'], ['저는  버터 ~'], ['버터색이용!!'], ['버터색요^^'], ['민트 상큼해요ㅎ'], ['민트요'], ['민트한표요'], ['민트요~ 저도 침대 궁금해요^^'], ['시몬* 침대에요~ 저게 ㅅ몬스 국민프레임이라고 하더라고요~^^'], ['버터색이요\n근데 화이트도 좋을 것 같아요'], ['버터색이 더 잘 어울릴듯요'], ['민트요~'], ['민트요'], ['버터색이요'], ['버터요\n화이트도 이쁠꺼같아요'], ['버터요'], ['버터요. 집이 환해보일 것 같아요.'], ['222222'], ['민트'], ['따뜻한 버터요~'], ['커튼이랑 화분보니\n민트나 그린이 어울릴 듯 싶네요'], ['민트'], ['저도 민트요!! 저도 정보좀 주세용^^'], ['제가 사용해본게 아니어서 알려드려도 되나 싶지만...민트색은 쁘리엘* 제품이에요.3개 다 오늘의🏠 쇼핑몰에서 봤어요~^^'], ['2222민트요'], ['민트요!'], ['방에 그린이 잘 어울리 실 것 같은데요~'], ['버터'], ['민트넘이뻐요'], ['버터요~'], ['민트요'], ['버터는 칙칙 할듯싶어요 벽지도 그렇고 나머지가 나을듯오ㅡ'], ['버터요'], ['버터오'], ['어디껀가요?전 그린이뿌네요'], ['3개 다 오늘의🏠 쇼핑몰에서 봤어요~그린색은 "데👃뷰" 제품이에요~ㅋ'], ['이불정보가궁금해요 침대엔버터가어울릴듯한데 전민트사고싶어요'], ['3개 다 오늘의🏠 쇼핑몰에서 봤어요~이거 말고도 예쁜거 엄청 많으니 구경한번 가보세요~^^'], ['민트 이뿌네욛'], ['저도.침대 너무 예뻐서.궁금해요'], ['시몬* 침대에요. 저게 시몬* 국민프레임이라고 하더라고요~'], ['2번 민트요 \n침대 깔끔하니  이뻐요\n정보좀 부탁드려요'], ['시몬* 침대에요.저게 시몬* 국민프레임이라고 하더라고요~^^'], ['버터여. 화이트는 없나요?'], ['버터색 이뻐요~~'], ['화이트가 젤 예쁘고 깔끔할거같아요.기존 베게 그레이랑 믹스매치도 가능하고요'], ['다 이쁜데... 민트가 산뜻햐 보여서 눈에 들어와요. 저희도 곧 이사하고 침대 이불 바꾸려고 하는데... 침대랑 이불 정도 공유 부탁드려도 될까요?'], ['제가 사용해본게 아니어서 알려드려도 되나 싶지만...민트색은 쁘리엘* 제품이에요.3개 다 오늘의🏠 쇼핑몰에서 봤고요. 침대는 시몬* 에서 샀어요~^^'], ['저도 민트요! 깔끔해보여요'], ['버터색이요~'], ['버터색이요~'], ['민트!!'], ['민트'], ['저도 민트^^'], ['저도 민트가 예뻐요 맘님 혹시 안방에 죠기 보이는 식물들 이름이 뭘까요?   용도는 뭘까요 ? 안방이 건조해서 식물을 좀 키워볼까하는데 눈에 딱들어오네요'], ['아래카야자랑 스파티필름이에요.\n반그늘에도 잘 사는 종류여서 저희집 안방에서도 잘 살더라고요.\n특별한 용도는 없고요..\n식물 키우는걸 좋아해 집안 곳곳에 화분이 많아요..'], ['민트요~'], ['그린이요'], ['침대 헤드월이 있어서 넘 이뻐요, 침대정보부탁드려도될까요?'], ['민트요'], ['버터요'], ['버터용']]
    
    8515
    하기싫은 집안일은? 전 빨래개는것..너는것도싫지만ㅜ아~~건조기사고싶어지네요아~~진짜 빨래개주는 로봇하나있었음 싶다니깐요바꿔야할 가전제품만늘어가네요세탁기도다됐는지 돌아갈때 여간시끄러워요돌멩이넣고확돌려버릴까봐요🤣🤣
    
    [['빨래 널고 걷고 개는거여ㅠ건조기사면 해결될줄 알았는데ㅠ개는게 아직 해결안됐네여ㅠ'], ['걸래빨기요ㅎㅎ'], ['전 저부다 싫어요 끼니챙기는것도\n가사도우미분 쓰고싶네요'], ['빨래 개는거요^^;;;;;;;;;;'], ['전부다 귀찮지만  제일  싫은건  청소요'], ['반찬하기요.저는 자짜 김치나 멸치볶음 한개만 있어도 꿀맛인데.아들 남편 영양소 챙겨야하니 ....코로나땜시 더 도전이에요.외식도 마트도 맘대로 못가니까요'], ['설거지, 방닦기, 걸레빨기요. ㅠ'], ['설거지요'], ['빨래개는거요ㅠㅋㅋ'], ['반찬하는거책정리하는거ㅜㅜ  반찬  사먹는것도 이젠 안하다보니 한끼에 최선을다하고 나머지는  뭐그냥 대 충ㅜㅜ급 반성됩니다'], ['설거지요.'], ['청소요..특히 먼지 닦는거요 ㅠㅠ'], ['청소.음식만들기.설거지 등 모든 집안일이 적성에 안맞아요.\n도우미 이모님 부르고 싶어요.'], ['그중고르라면 설거지요\n제가 뒤만돌면 저지리를그렇게해요ㅡㅡ'], ['전 음식만드는거요 ㆍ설거지 청소가 더 적성에 맞아요'], ['저는 설거지요...ㅋㅋㅋ 식기세척기 사고싶어요~!!ㅋㅋㅋ'], ['청소랑 음식하기요~ 적성에 안맞나봐요 ㅋㅋㅋ'], ['돈만 벌고 싶어요..^^'], ['바닥 닦는거요..계단 있는 집이라 계단 닦는것도 일이네요.ㅡㅡ'], ['저도 빨래개고 넣는거요.^^;;;'], ['바닥 손걸레질이요.물걸레로 해도 구석구석안되는부분이있으니 승에 안차네요.ㅡㅡ'], ['물걸레질과 걸레 빠는거요진짜 너무 싫어요ㅜ'], ['저는 전부 다요~~~~~ 집안일 정말 재미도 없고, 힘들고 어렵고 하기 싫고.. 안 하면 금방 개판이고..ㅠㅠㅠㅠ'], ['설거지요ㅜㅜ 귀찮아욧'], ['갠 빨래 널어주는 거요. 넣는게 제일 귀찮아여.'], ['모두???모두다요 ㅠ'], ['집안일 모두 다요..\n전업맘인데..ㅜㅜ'], ['전 그나마 빨래 널기, 개기는 괜찮은데 청소, 음식하기가 젤로 귀찮네요. ㅎㅎ'], ['건조기는 정말 신세계라 강추드려요 ㅎㅎ\n전 설거지를 너무너무너무 싫어해서\n이번에 식기세척기 샀는데 진짜 이것도\n건조기만큼 좋고 편해요ㅠㅠ\n그 다음 싫은 집안일은 욕실청소..\n그래서 오늘 무선욕실청소기 구매했는데\n이것마저 힘들고 하기싫으면\n그냥 일이주에 한번씩 가사도우미 이모 부르려구요ㅠㅠ\n남편이 해줘도 완전 성에 안차서 ㅠㅠ'], ['빨래 갠 후에 넣는거요ㅋㅋ\n엉덩이가 무거워서 그런가봐요ㅠ'], ['설거지를 제일 늦게해요미루다미루다 억지루 하네요'], ['밥차리는거\n설거지하는거\n빨래 널고개고 제자리다시\n갖다놓는거요\n나중에되면 막 뒤죽박죽이네여'], ['설거지랑 음식하기요\n빨래야 몇일안개도 지장이없지만 설겆이나 음식은 매일  피할수가없네요ㅠ'], ['다 싫으네요.지져분하게 살고있어요.ㅋ'], ['저두 빨래개기요~건조기쓰고부터 더 싫어졌어요..'], ['반찬 만드는 거랑 빨래 개기요...\n그나마 첫째딸이 빨래 개서 수납장에 넣어주니 너무 고마워요 ㅠㅠ\n\n저희도 결혼한지 10년 다되가니\n고장나기 시작하네요\n\n밥솥. 티비. 냉장고는 1번씩 바꿨는데..\n에어컨. 세탁기 바꿀 때 되었어요 ㅠㅠ\n탈수돌릴때마다 너무 시끄러워서\n스트레스예요 ;;'], ['저도 빨래개는 가전은 안나오나 늘 생각해요 ^^; 그런게 나온다면 대박 히트 칠 것 같아요 ㅋㅋ'], ['요즘은 설겆이가 너무 싫어요뭐 해먹은것도 없는데 왜케 많은지그릇들은 그나마 난데집기류들과 냄비들  빨래 개는거는 개인적으로 좋아해서음악 틀어놓고 하고 있으면 개인적으로 힐링시간이에요 아무 생각 없이'], ['주방일이요.\n밥하기...설거지하기.'], ['설거지 ㅜㅜ'], ['방문세탁광고에 왜 빨래 개는 기계 안나오냐고 보고 격하게 공감했어요 ㅋㅋㅋ 지금 세탁기에선 다 된 빨래가 날 빼주시오 하고 기다리는데 외면중입니다 ㅋㅋㅋ'], ['설거지가젤  시러요 ㅠ'], ['저는 개는거 까진 하겠는데요 개어서 각각 서랍장에 넣는게 제일 싫어요ㅋ ㅋ수건.옷.속옷.사람마다 각각 넣는게 왜이리 귀찮을까요ㅋ'], ['빨래개는거, 반찬만들기... 등등 너무 많아서ㅠㅠ'], ['전 식탁닦기요..\n참 희안하게... 먹기전 먹은후 식탁닦눈게... 그렇게 시르네요 ㅠ\n식탁매트있음뭐해요 ㅠ'], ['바닥물걸레질이요ㅜ'], ['다 싫어요...ㅡ.ㅡ;;\n건조기 사고 싶어요.'], ['욕실청소요.... 바닥물때 변기오물 극혐 휴~ ㅠㅠ'], ['다 하기 싫지만 빨래개는게 젤로 힘들어요'], ['빨래 개기요 ㅎㅎ'], ['저도빨래개기 \n빨래가 산처럼 쌓여 있어요 ㅜㅡㅜ'], ['방닦는거랑, 화장실요 ㅎㅎ'], ['저는 물걸레질.걸레빨기요 ㅜ'], ['저는 빨래갠거 가져다 두기ㅎㅎ'], ['건조기.로봇청소기.식기세척기 있어서 다른건 다 부담없는데 음식하기가 제~~~~~일 힘들어요. 메뉴고민.장봐오기.요리하기.차리기.치우기 이것이요ㅜㅜ'], ['설거지가 젤 싫어요ㅠ'], ['걸레질이여'], ['건조기사보니 정말 신세계다~~ 싶은것도 잠시  이젠 빨래개어주는 기계안나오나?? 이런생각만ㅋㅋ'], ['청소랑 음식하기요. 청소해도 티도 안나고 저녁되면 먼지 쌓여있고ㅜㅜ왜 3끼를 먹어야하는지 1끼만 먹고 살았음 좋겠어요.'], ['설거지랑 물걸레청소 젤 싫어요~'], ['설거지요!!!']]
    
    8518
    집안일에 대한 강박, 저만 이런건 아니죠? 저는 정리는 그저그런 여자인데 집 청결에 대한 나름의 기준과 강박증이 좀 있어요.. 이게 충족이 안되면 스스로를 좀 괴롭히고 다른 사람들한테도 짜증을 자주 내는데 어떤 상황을 못참냐면1. 바닥에 보이는 머리카락 : 특히 머리 말리고 나서 머리카락을 줍지 않은 딸 방의 머리카락들 또는 화장실 바닥에 분산되어 있는 머리카락...2. 미끌거리거나 먼가 끈적거리는 주방 바닥,음식을 하고 기름 또는 음식물이 튀겨져있는 가스레인지 상부 3. 먼지가 쌓여있는 가구 위 4. 매끈해보이지 않고 물때가 껴있는 세면대, 물기가 흥건한 세면대 위, 물방울 얼룩이 져있는 욕실 거울5. 오줌 등 노폐물이 튀어있는 변기 이 외에도 많은데.. 쓰다보니 정신적으로 문제있는 사람처럼 느껴지네요 ㅜㅜ 저만 이런거 아니죠? 나이들수록 체력은 떨어지고 짜증이 많아져서 지저분한거 그냥 포기하고 살았음 하는데 잘 안되요. 문제는 나 말고 우리 집안 누구도 신경쓰지않는다는 거 ㅎㅎ
    
    [['저두요..격하게 공감이요\n저는 설거지랑 바닥 먼지,(머리카락포함) 빨래바구니.. 물건제자리 뭐이런거요\n저말고 아무도 안하니 저만스트레스에 짜증이랑 신경질나요'], ['아 물건 제자리... 도 포함이요... 정해진 자리에다 좀 놨음 해요... 물론 정해진 자리도 그닥 정리정돈돤건 아니라는게 함정이긴 한데 ㅎㅎ'], ['가스렌지주변 기름닦느라 요리하는 중간에 정신없어요.. 급 눈에보인 청소하느라 회사지각한적도 있고 청소하고싶어 회사가기싫을때도 있.. 쫌 성격이상하죠..'], ['아 저 비슷해요.. ㅋ 요리하면서 튀긴 기름 음식물 바로바로 안차우면 안됩니다. 외출 전에 보이는건 다 치워놓고 나가요. 돌어와서 눈에 보이면 화가 나니깐.. ㅠ'], ['격하게 공감요..가스레인지주변 튀기는 부침개..튀김요리 너무 싫다는..'], ['저두 그래요~~강박이라고 생각해본적 없이 당연한거라 생각했는데....'], ['근데 좀 강박인거 같아요 스스로를 억압하고다른 사람을 괴롭히잖아요 사실 그렇게 중요하지 않을수 있어요 ㅜㅜ'], ['저도 심했어요... 근데 고쳐지지 않는 두 남자들과 털 뿜뿜 댕냥이들 땜에 무뎌지나바요... 늙었나...🤦🏻\u200d♀️'], ['댕냥이 키우시면 포기하셔야죠 ㅎ 저는 그래서 애초에 털있는 애완은 안되요.. 지금 그나마 허락한게 도마뱀입니다 ㅎ'], ['도마뱀요? 오.....🤔'], ['쓰신거 모두랑 정리까지 강박 있습니다 ㅠㅠ하루종일 움직여요 ;;;'], ['티도 안나게 움직이는 거잖아요.. 이런들 저런들 별 차이 없는데 ㅜㅜ'], ['우리집오심 안되시겠어요 ㅜㅜ 변기만 빼곤 다 있는데 ㅜㅜ 셤니도 깔끔이라 쓰고 강박이 심하셨는데 노안오시곤 먼지가 안보이신대요. 즤엄니도 한깔끔 하셨는데 쓸고닦다 어깨십자나가서 수술하셨어요  수술 후 덧없다시며 깔끔떨며 몸축내지말고 그냥살래요마그네슘 함 드셔보세요.'], ['악 ㅋ 저는 친정부모남 주방, 화장실도 못보겠어요ㅠㅠ 자꾸 못볼걸 보게 되서'], ['아 마그네슘은 안그래도 많은 분들이 이야기하시더라구요... 저 갈수록 잡에서 짜증이 많아져서... 효과가 있나요?'], ['제 실험대상들은 ㅎㅎ  효과있어요. 마음의 평온... 까진 아니어도  덜 화내더라고요. 그리고 친정엄니는 잠이 잘오신대요. 칼슘 마그네슘 비타민디 로 되어있는게 좋아요'], ['저희집도 다들 신경 안써서 \n저도 이제 조금씩 내려놨어요 \n안그러면 나만 힘들거든요 \n청결도 중요하지만 저도 좀 편해지고 싶어서요\n화장실 변기는 애들 샤워전에 샤워기로 물 뿌리라고 시키니 한결 낫구요 \n주방은 주말 빼고 다들 식사시간 틀려서 어떤날은 혼자 밥먹을때도 있어서 청소거리가 줄었네요 \n'], ['털복숭이 포메땜에 바닥은 포기한지 오래예요 ㅎㅎㅎ\n'], ['친정엄마가 항상 이야기하세요.. 왜너는 사서 고생스럽게 사냐고..'], ['이제 다 포기하고 그냥 청소하네요.\n말해도 안되고, 화내도 안되고 방법이 없어요.ㅠ'], ['저도 혼자 다 하는데 씩씩대면서 해요 ㅎㅎ'], ['저두 모르면몰랏지 어느순간 보이면 앜!!!이러고 다 치워요. 글쓰신거 전부 해당ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ\n남편은 저한테 결벽이라는데 또 다른 엄마들은 저희집 드럽다든대옄ㅋㅋㅋㅋㅋㅋㅋ'], ['남편들 입장에서는 결벽, 쓸데없는 것들로 보이는거 같아요? 저희 남편도 생전 안하다가 나이드니 조금씩 저한테 맞춰주긴 해요'], ['저두요...\n설거지후 그릇 바로 물기 닦아서 안으로 넣기\n\n신발 바르게 정리하기\n\n냉장고에 들어가있는 우유 유제품등 상품라벨이 나를보게 정리하기\n\n살짝 삐뚤어진 물건 라인에 맞게 정리하기\n\n책... 눕혀놓지않고 책장에 꼽아두기..\n\n이거저거 엄청 많네요.. ㅜ'], ['아 정리 쪽으로도 엄격하신 분이네요? ㅋㅋ 저는 나름 정리한다고 하는데 잘하진 못해요'], ['기준이야 다들 다르실테고 친한 지인이 왔는데 쉴틈없이 움직이는 절 느끼고 미친년 같았어요^^;;'], ['아 저 예전에 가족모임때 다들 고스톱 치는데 혼자 청소하고 설겆이하고 ㅋㅋ 좀 이상한 사람 같죠 ㅠ'], ['맥주를 한잔했더니 표현이 좀 격했습니다^^;;  전 처음부터 그랬던건 아니구요. 휴직하고 집안살림을 챙기다보니 좀 변한 케이스예요.'], ['저두요. 하루종일 도우미마냥 치워야되서 그냥 안봐요. ㅠ'], ['차라리 안보고 눈감고 샆을때도 많아요 ㅜㅜ'], ['저도 머리카락 보면 하늘이 무너진 느낌이에요 진짜 하루 종일 움직여요'], ['ㅋㅋ 하늘이 무너진 느낌... 머리카락은 머리통에서 떨어진 순간 매우 불결한 존재가 됩니다'], ['ㅋㅋ..하늘이..ㅋ'], ['아 근데 이게 눈에 보이면 급 기분이 나빠지고 바로 해결해야하고.. 그리고 사실 이것보다 더 많은데, 이상한 사람처럼 보일까바 좀 덜 썼어요 ㅜㅜ 그리고 제 주변에 이 정도 안하는 사람들 너무 많이 봤어요 ㅎ'], ['저랑 사시면 저 쫒아다니면서 욕하실듯~~ㅎㅎ\n'], ['네 저 사람 좀 괴롭히는 편이라 ㅋㅋ'], ['청소기밀고 핸디청소기로 보이는 머리카락 수시로 밀고 분무기통에 소주 항상 넣어두고 주방바닥 싱크대 식탁은 항상 닦고 세면대 이틀에한번 욕조도 생각날때마다 변기는 거의 수시로체크해서 휴지로닦고..  강박이라 생각해본적은 없어요~~~'], ['저도 항상 주방 쪽은 알콜로 닦고 보송보송해져야 직성이 풀리는 사람이네요. 이 행동 자체가 강박은 아닌데 이게 안되어있음 초조하고 화가 나고 하는게 강박적인 증상인더 같아요ㅜ'], ['다 해당되고 설거지 한후 그릇 상부장에 넣어두기 신발 가지런히 정리 및 1인 1 신발만 내놓기,옷장 문 닫기,가방 각 맞춰 정리 ,거실 카페트랑 화장실 매트 각 맞추기등도 제대로 되어 있어야 마음이 편해요. 수전의 얼룩도 참기 힘드네요.회사 갔다와서 쉬지 않고 청소해요.저만 신경 쓰고 세 남자는 아무렇지 않아요.\n주말에는 창틀 먼지도 닦고 있어요. 거슬려서'], ['음 저도 대부분 해당돠는 것들이네요 ㅎ 발매트 거실 협탁 등 비뚤어져있는거 그냥 못넘어갑니다 ㅎ'], ['저 진심 격하게 공감했어요 나만 화나고 화내는데 아무도 안움직이고 나만하고 ㅋㅋㅋㅋㅋㅋㅋㅋ  남자만셋인집 매일매일 변기닦는 모습이 너무 초라해져서 눈물만나오는데 이상하게 점점 더 강박적이게 되요 ㅠㅠ'], ['저두요 청소하다가 애얼집맨날늦게 데려다준적도많고 요리하다가 가스렌지 기름때닦다가 시간다가요.요즘은 그럴시간도없지만 아직도 주말토일요일은 하루종일 움직이면서 쓸고닦고하고있어요'], ['제가.... 세탁기 3대 쓰고 방마다 청소기 있고 • _ • \n연년생 아기둘 4살때 까지 입주아즘마에 청소 이모 따로 쓰던 스탈인데.....  경제력 안좋아지고 \n연년생 아기들 어지는거로 인해\n저의 정리강박증 교육상도 안좋은거같아 포기하고 바뀐지 5년쯤 되는데 아직도 제 스스로 컨디션 안좋으면 제가 하다 폭팔해요 ㅠ ㅠ \n요즘 이사왔는데 전집보다 밝아서 \n눈에 머리카락 계속 보여서 \n힘드네요 • _ • !!!! \n이해해요 너무너무!!!!!!!!! \n내기준 더러우면 화나고 기분나쁜!!!!\nㅜ ㅜ'], ['저도 심한편이예요. 하루종일 몸이 부서져라 움직이죠. 돌돌이 구석구석 세워두고 틈날때마다 문지르고, 걸어가면서도 손 발 사용해서 책장 비뚤게 나와있는 책들 밀어주고, 화장실 앉아서 휴지걸이를 휴지로 닦고있고, 매일 무선, 로봇 청소기 돌리고. 가스레인지 기름은 설거지 할때마다 닦고...지금도 일 끝나고 방금 누웠어요. 쓰고보니 진짜 이상한 사람이네요.ㅋㅋㅋ 전업 10년하고 이렇게 됐어요ㅋ'], ['ㅋㅋㅋ 걸어가면서 책들 밀어주고... 격공하면서 웃었네요 왓다갔다하면서 그냥 지나치는 법이 없죠... 밀고 닦고 쓸고... 가스레인지는 당근 매일 닦고 바닥 물청소도 하루 한번 손걸레질로 해야 직성이 풀려요'], ['누구나 그런것들이 눈에 들어오면 청소해야지 싶어질듯한데요 몹시 피곤하면 나중에 해야지가 되거나 더 피곤하면 그런것들을 살필 겨를 없이 바로 잠을 청하게 되기도 하구요 저는 항상 몹시 피곤한게 문제네요ㅋㅋ'], ['저랑 신랑이랑 다 그런거 신경안써서 그냥 편하게 살아요. 성격인듯요.저는 너무 깔끔하지 못한게 어떨땐 좀 짜증나요. 중간쯤 되면 좋을텐데요. ㅎㅎ'], ['저두요! ㅠㅠ 덕분에 이제 손목도 너무 아프고 몸이 힘드니 짜증이 더 자주 나고 ㅜㅜ 누가 매일 싹~ 좀 해줬으면 좋겠어요 흑'], ['저도요... 근데 저는 청결(머리카락 먼지)은 별로 신경 안 쓰이는데 물건이 제자리에 안 있거나 너저분하게 나와 있음 답답해요. 무조건 안에 집어 넣어야 돼요 ㅋ'], ['저도 완전 공감해요~\n바닥에 머리카락, 먼지있음 바로 돌돌이로 치우고 부엌에 기름이나 물튄거, 이물질 떨어진거 못견뎌요~바로바로 닦아야하고 어질러진거 못봐요~어떤때는 제 자신이 피곤하기도 한데 더러운거 보는게 더 피곤하네요...'], ['정리정돈 드럽게 못하는데 오딜가나 물때가 그렇게 거슬려요 ㅋㅋㅋㅋㅋㅋ\n지난주에 친정 가서도 칫솔들고 세면기랑 싱크대 물때 청소하고 ㅠ 에잇.\n정리 청소 못해서 이럴바엔 미니멀로 살자 생각하는데 왜 못비우고 미니멀카페 눈팅만 하는지 ㅠ'], ['친정이라도 저는 남의 집 살림은 손안댑니다... 좀 지저분해보여도 그냥 냅두고 와요'], ['저도 위 다 해당되고요 저는 머하나라도 제자리에 없음 난리나요 날새서라도  찾고제자리에 놓아줍니다 전 예민한편이에요 신경이'], ['격하게 공감하고갑니다 아무도 치운지모르는데 혼자만 꼭꼭꼭 해야 되는집안일이 저도 있어요 삶이 참 피곤하네요'], ["제가 쓴 글인줄요. 요즘은 딸램한테 '머리 좀 묶어라' 가 제일 자주 하는 말인 듯요."], ['ㅋㅋㅋ 악 저 맨날 달고 사는 말 ㅜㅜ'], ['격공감합니다  어떤날 너무 지쳐서 나의 리스트들 중 안하고 내버려두면, 화장안지우고 자는느낌이랄까 ㅠ 너무 불편해요.. 주방일하고 물 튄 벽면이랑 싱크대, 암튼 물자국도 넘 시러용  식구들은 원래 보송보송한줄 알겠죠ㅡㅡ'], ['저.요!!!!!!! 아~~~~ 집안일에 대해 내 나름의 이상한 강박만 없으면 내 인생과 내 가족의 인생이 달라질 것만 같은 ㅠㅠㅠ다쿠와즈님 저와 비슷한 게 넘 많네요!!! 운동, 옷 등등 ㅎㅎㅎㅎ 저는 그 미모 살짝 못 따라가지만... 넘 비슷한 상황이신 것 같아요'], ['1인 추가요. 근데 아무렇지도 않은게 더 정신이 이상한 것 같아요'], ['우와~ 저는 정신과 상담한번 받아봐야 하는건가를 늘 고민했는데 비슷한 사람들이 이리 많네요. 전 이 모든게 해당되서 집에 누가 오는 걸 싫어해요.'], ['저도 솔직히 그래요... 나중에 가고 나서 치우는거 힘들어서 ㅜ'], ['어머!\n저도 그래요\n근데 오래된 아파트 이사오고 많이 내려놓아졌어요 \n제 한계를 느꼈거든요\n\n몇년 전엔 이걸 적성이라고 느껴서\n가사도이미가 되고싶기도 했어요 \n이모님 쓰는 지인집에 차마시러 갔는데\n그 날 오전에 다녀가셨다는데 제 눈에 청소가 전혀 되어있질 않아서 \n이렇게 하고 돈 받아가냐고 놀라고 \n난 더 잘할 수 있다고 느끼고 \n막 청소해주고 싶고;;\n그렇게 해 놓았을 때 뿌듯 보람도 느껴서 \n내 적성이 이거구나 싶더라구요 \n\n근데 ㅋㅋ \n너무 몸을 썼나봐요\n여기저기 몸이 뿌득뿌득(?)해요 \n내 집 정리도 버거운;;; ㅋㅋ'], ['쓰신건 보통 깔끔한 성격이면 눈에 거슬리는것들 같아요 전 정리가 더 거슬려서....집이 늘 깔끔하단 소리 들어요.근데 전 하면서 화가 나거나 짜증이 나진 않아요 오히려 청소하고 기분이 좋죠 강박보다는 내 안에 화를 다스려 보세요~~~'], ['저도 똑같아요 ㅎㅎㅎ 애낳기전에 진짜 지저분했는데 이렇게 바꼈어요. 머리카락 떨어진 꼴을 못보겠어요 ㅠㅠ 돌돌이와 한몸이라는...그리고 저는 장난감 부품 하나 없어지면 찾을때까지 다 뒤지는 이상한 성격이 추가됐어요....ㅋㅋㅋㅋㅋㅋ'], ['저는 뎃글들에 더 놀랍니다ㅜㅜ\n저 처럼 지저분하게 해놓고 편히 사는 분들이 잘 없네요\n몸은 편하나 정리 및 청소 잘 못하는게 나이들수록 한심하게 느껴져서요ㅜㅜ 이것도 적성이다 싶어요'], ['딱..저도 저만큼 강박증이,.ㅋ\n근데 또 정리정돈(특히 옷)같은건 눈에 안띄니그런가 뭐 그러려니 하거든요. 그런건 남편이 또 대박잘함.ㅋㅋ 게다 전 시력이 좋아서 진짜 잔잔한 먼지가 다 보여서..ㅜㅜ 무신경한 애들에 시력나쁘고 그런거 그닥 막 예민하지 않은 남편땜시....요샌 주방에서 뭐 해주는것도 별루에요 하고나면 바닥 물떨어뜨리고 슬리퍼로 밟고다녀서 얼룩대박에....싱크대 개수대얼룩..잔찌끄러기 부스러기 다 내눈에만 보임.ㅠㅠ 나한테 잔소리 안하니 나도 안해야는디 자꼬 궁시렁대서 남편도 싫을듯요.'], ['1.2.4.5 격공이요!!!\n저 숏컷.나머지 남자들인데도 머리카락 느므 싫어요\n침구청소기로 베개까지 쫙 밀어줘요\n강아지고양이 키우고 싶다가도 머리카락 집착하니..쇠약증 걸릴 듯 싶어 포기요 ㅠㅠ'], ['정말 웃긴게 어지러진것을 못보다보니 베란다에 화장지 가지러 가다가 세탁바구니에 세탁물보고 거기서 세탁기 돌리고 돌아오다 다시 화분잎이 말라 화분물주고 한참후 보면 어랏 화장지가 없었지 하고 다시 베란다로 가다가.. 이처럼 뭐라도 바로바로 해야해서 동선상에 있는 것들을 지나치지못해 일순서가 뒤죽박죽될때가 있어요. 요리하다 요리는 뒷전이고 기름닦는것처럼요. 그러다보니 중간에 누가 말거는거 예민해져요 아까하려던일 안잊어야해서.. 쓰고보니 집중력결여같네요 ㅎㅎ 지금도 분주하게 움직이다 30분전 화장지 가지러 가던거 잊었던거 생각나서 화장지는 뒷전 급 글적으러 왔어요. 또 가는길에 머릿카락및 이것저것 정리하며 가겠죠 ㅋㅋㅋ 저 이제 길떠나요'], ['와~~다쿠와즈님 제 남편하고 완전 비슷하세요. ㅎㅎ 저희남편도 각잡고 정리정돈같은건 크게 신경안쓰는데 먼지, 머리카락,기름때 물때..  이런걸 못참더라구요.주방에서 조리하고나면 그즉시 싱크대, 가스레인지 닦구요 주방바닥 닦고.. 샤워하면 머리카락 정리는 물론이고 수전에 물기까지 수건으로 다 닦고나오고...하루에 청소기 3번씩돌리고... 수시로 바닥 밀대들고 왔다갔다거리면서 닦고..제가 같이 사는동안 여간 눈치가 보이는게 아니었어요.내 영혼이 바짝바짝 마른다 싶은 느낌^^;;;그러다가 저의 전근으로인해 지금은 3시간거리 떨어져살고 한달에 두세번 만나는데... 좀 살거 같아요. 저는 지저분해도 별로 신경이 안쓰이는 성격이거든요.ㅎㅎㅎ다만, 남편 오기전날은 제가 집 뒤집어엎으며 대청소하는날이고 외부 감사받는것처럼 심장뛴다는 사실. ㅎㅎㅎ'], ['ㅋㅋㅋㅋ 아 근데 읽는 저는 너무 웃겨요.. 바닥 뽀송은 제 삶의 기본이예요 ㅎㅎ 한살림 살균소독제로 수시로 닦아요... 애들 과자뷰스러기 음식 흘리는거 너무 싫어하고 잔소리해서.. 가끔은 내가 왜이러나 싶어요.. 싱크대 가스레인지 머 묻어있는거 보면 혈압오르고 ㅜㅜ \n외부감사 d-1 에 콩닥콩닥하시는 거 생각하니 빵 터져요 ㅋ'], ['저희집에 무선청소기 2대, 침구청소기1대, 로봇청소기1대, 물걸레청소기1대, 밀대 2개, 먼지떼는 돌돌이 3개, 분무기 구연산, 가루구연산, 베이킹소다, 각종 소독제 등등...^^;;;;남편은 청소용품사면 마음이 편안해진대요 ㅋㅋㅋㅋㅋ그리고 자기는 서장훈심정 이해할거 같다고...'], ['ㅋㅋㅋ 그래도 너와함께한날들님 안괴롭히고 혼자 쓱쓱하시는 거면 이쁘게 봐주세요 ㅎ 덕분에 내 몸이라도 편하자나요 ㅎ'], ['저도.. 못참아서 매일혼자..쓸닦해요.3명 어지르고 혼자하려니 승질나요.ㅠ ㅠ 근데 또 못맡겨요.어차피제가다시해야해서']]
    
    8680
    집안일+요리잘하는 남편 많나용?ㅎㅎ 제 남편은 설겆이하라고 하면 딱 안에 있는거만 하고 뒷정리를 못해요원래 설겆이하면 남은 음쓰 음쓰봉투에 넣고 싱크대 주변도 닦고..그런건데 모르는걸까요?배수구에 잇는 음쓰 정리하라고 말하면 나중에~~하고 안합니다. 할줄 모르는건지ㅠㅠㅜ청소도 하면 청소기 돌리면 먼지를 비워야하는데 그것도 안하고 뭔가 뒷정리를 못하는 느낌? 모르는게겠죠?결혼전 거의 집안일 안하고 산 아들이어서ㅠㅜ집에 입주 아주머니 계셔서..​요리도 할줄아는건 라면이랑 고기굽는거요요리해달라하면 할줄 모른다해요 .ㅠㅠ 남자들은 인터넷에서 요리방법 찾아서 요리하는거 어려워하나요? ㅋㅋ 칼질도 한번 안해보셨네요ㅜ​좋은 점 많지만 똥손에 집안일 못하는거 답답하네요ㅜ​​
    
    [['제 예랑이는 저보다 섬세해서 시키면 잘하던데요 ㅋㅋ 제가 상대적으로 시간적 여유가 잇어서 더 하긴하지만요'], ['솔직히 모른다기보다 그냥 안한다는게 맞는것 같아요 만지기 더럽고 본인이 안하면 해 줄 사람이 있으니 안한다고 하는게 맞는것 같아요'], ['모르는게아니라 하기싫은거고 귀찮은것같아요.. 위생관념? 생활습관 그런건 알려주고 바꿔나갈수있게 도와주세여 ㅠ'], ['청소랑 설거지는 잘못해서 제가해요 ㅜ 신랑은 요리만 하고 저는 뒷정리,,, ㅋㅋㅋㅋㅋㅋ 요리는 잘하는데 진짜 주변을 엉망을 해놔서 저는 옆에서 ㄱㅖ속 치워줘요 ㅋㅋㅋㅋ'], ['요리는 배우고오신걸까요?'], ['아뇨아뇨 블로그나 유튜브 보고 하더라구요~ 요즘엔 밀키트도 잘나오니 밀키트로 하라고해보세요 ㅎㅎ'], ['저는 완전 집안일 꽝이고 저희남편은 예전부터 자취도 하고해서 거의 집안일 만렙이에요... 거의 다 해요'], ['ㅋㅋ 저는 제가 예랑인데.. 자취를 오래해서 기본적인거는 다 해요.. ㅋㅋ 요리는 잘 못하지만'], ['저희는 둘다 요리못하고 ㅋㅋㅋㅋㅋ둘다 라면이랑 고기만구울줄알아요 ㅋㅋㅋ배수구음쓰, 청소기 둘다 절반은 차야 비우구요! 이 글보니까 이런면은 둘이 잘 맞는거같네용ㅋㅋㅋㅋ 다행이 둘다 설거지하고 주변은 잘 닦습니당! 그냥 서로 맞고안맞고 차이 아닐까용?,,'], ['집안일을 정말 안해보신 신랑님 같으시네요..^^ 계속 하나씩 반복해서 가르쳐주심 됩니다 ㅎㅎ 다행이 울 남편은 요리하는걸 좋아해서 저는 아직까지 제대로 요리해본 적이 없네요 ㅎㅎ'], ['집안일은 잘하는데 음식은~ 애매해요ㅋㅋㅋㅋ 집안일은 예민한 사람이 더하게되는거 같아요ㅎㅎㅎ 신랑은 먼지같은거에 예민해서 더 청소하고, 주방은 제가 더 예민해서 더 하구요 ㅋㅋ'], ['전 저보다 신랑이 더 주방에 많이 있어요 설거지 요리 다 해요 ㅋㅋ 처음에 잘 알려주세요!!!ㅜㅜ'], ['잘하면 잘하는대로 잔소리가 있을 수 있어요. 그래도 말안해도 서로 도와가면서 이거저것 해주면 좋긴해요. 배우 김희선님도 예능 나와서 말하더라구요. 처음부터 아내가 이것저것 잘 하면 남편이 절대 안움직인다구요. 잘하더라도 못하고 적당히 숨기는 척 해야한다구요.'], ['제 남편은 집안일두 잘하구 음식도 잘해줘요 자기집에선 손하나 까딱 안햇다는뎈ㅋㅋㅋㅋ 해버릇 하니까 잘 하는 거 같아요'], ['부지런한 정도의 차이 같아요. 저희 남편은 자취해본 적도 없어서 걱정했는데 결혼하니 집안일 엄청 잘해요 ㅋㅋ 그런데 요리는 확실히 경험치가 쌓여야 실력이 느는듯해요.'], ['제 예랑일ㅏㅇ 같네요ㅜ ㅋㅋㅋㅋㅋ 시키면 하는데 뒷정리랑 섬세하게 잘 못하더라구요,, 시키면 잘 할런지ㅜㅜ'], ['저희는 예랑이가 주로 요리하고 저는 설거지해요 ㅎㅎ 청소는 주로 제가 하고 예랑이한테 뭐 해달라고 시키면 잘해주는데 제가 눈에 보이는 것만큼은 안 보이나봐요... 남자들은 다 그런가봐요ㅠㅠ'], ['예랑이가 설거지 하고 헹주로 뒷마무리까지 잘해요\n오히려 전 설거지만 하는 타입이에요 ㅎㅎ'], ['저희는 신랑이 요리해요^^; 뒷정리까지 다 할줄 알아서 잘해요.. 그러나 요리는 하되 뒷정리는 하기 귀찮으니까 저보고 해달라는 편.. 저도 뭐 신랑이 요리하니까 내가 뒷정리해야지 하고 그냥 제가 합니다~'], ['세상에 못하는게 어딨겠어여ㅎㅎ 안하는거지저도 결혼 전에는 한번도 안하고 살았는데 막상 결혼하니 인터넷으로 찾아서 하는걸요여자들은 다 알아봐서 하는데 남자라서 못하는건 아닌 것 같아요'], ['해본적이 없어서 모르는거지 해보면 잘하는거같아요 !저는 저보다 신랑이 뒷정리나 정리정돈 잘하고 설거지, 배수구비우는것도 꼬박꼬박 알아서 해요 ㅋㅋ'], ['모르는 사람 많죠;; 뭐 모르는것도있고..센스도 부족한거죠; 그런사람한테는 일 시킬때 자세하게 얘기해줘야해요~ 음식하는거야 뭐..남자라 못하는건가요 ~ 여자중에도 진짜 똥손들 많아요...ㅎㅎ'], ['저희는 반대입니다 남편이 모든 집안일을 잘하고 플러스 잔소리도 함께 옵니다 ㅋㅋㅋ 저는 발전이없어요 이해해주세요'], ['저희는 신랑이청소정리정돈잘해서주로하구 제가요리조아해서요리담당이에요 ㅋㅋㅋ'], ['남자라서가 아니고 사바사인거 같아용 원체 안해보고 자라셔서 그런듯요..ㅎ 제 예랑이는 항상 요리해줘서 제가 아직 할줄아는게 없어요... 청소같은 경우는 군대에서 많이 배우고 온다던데요~??'], ['음쓰는 비위가 약한 여린 남편덕에(?) 비위강한 제가 하고, 그냥 아들키운다 생각하고 해요 ㅎㅎ 시켜서 또 내가 할거라면 그냥 애초에 제가 깔끔하게 하는게 속편해요 ㅋㅋ'], ['저희 신랑은 요리 집안일 다 잘 하더라고요ㅋㅋㅋ'], ['나중에가 어딨어요 무조건 그자리에서 시킵니다 옆에서 하는거 지켜보고 있어요 저는 ㅋㅋ 할줄 몰라 나중에할게 ~ 는 그냥 안한단 소리죠 옆에 앉혀놓고 시키세요'], ['일단 전...집에선 수건접기밖에 안해요...\n예랑이 95 제가 5 합니다... 예랑이 잘하고 잘해서 저한테 안맡길라그래요'], ['남편이 저보다 잘해요ㅠㅠㅋ 요리도 남편이 다 하구 설거지도 뒷정리까지 완벽... 청소기도 항상 먼지까지 싹 비워요ㅋㅋㅋ 맨몸으로 시집와서 호강하구 살아요ㅠㅠㅋ'], ['저도 남편이 다 해주네요. 가정적이에요. 요리, 설거지,청소, 기타 집안일 다 많이 해줘요. (옷정리 빼고 다 잘하는거 같아욬ㅋㅋㅋㅋ)시키는것도 잘해주고요. 반면에 저는 똥손이라 빨래나 재활용품 분리 가끔 화장실청소 등등 하구요. 어머님도 요리를 자주 안하시던데 이런 아들이 어디서 태어났나 싶어요. 각자 알아서 잘하니까 싸울일이 없네용ㅎㅎㅎ'], ['저희 예랑인 청소는 좀 힘들어하는데 요리는 기똥차게 잘해요^^*요리대결입니다 ㅎㅎㅎ서로 ㅎㅎㅎㅎㅎ'], ['저희 예랑이도 요리랑 집안일 다 잘하는거 같아요 성격이 꼼꼼해서 그런지 따로 지적할게 거의 없어용'], ['저희 신랑도 저보다 집안일 잘해요~ 요리는 제가 낫긴하지만요~'], ['저희 남편은 저보다 요리랑 청소를 더 잘해여..밥은 늘 남편이 차려주고 설거지도 남편이 해요..전 구냥 빨래,세탁기만 돌려욬ㅋㅋㅋ'], ['저희 남편은 요리는 잘 못하지만 집안일은 저보다 훨씬 잘해요 :) ㅋㅋㅋ'], ['제 남편은 저보다 잘해서 오히려 제가 혼나요ㅎㅎ'], ['저희 신랑도 음식은 안해봐서그런지 잘 못하더라구용, 가끔씩 유튭보고 해보라고 시키고 있어요ㅎㅎ'], ['할줄 모르는게 아니고 안하는거에요 귀찮아서 그리고 자기가 안해도 치워지니까 당연하게 생각하는거에요ㅎㅎ 음식,집안일 다 잘해요 원래 타고난사람은 없는거같고 노력하기 나름같아요ㅎㅎ'], ['도와주려고 하긴 하는데 왜이렇게 못하는지 모르겠어요...ㅜㅜ'], ['저희남편은 진짜 못해요...ㅋㅋ 그래서 제가 잘 안 시켜용ㅎㅎ'], ['예랑이는 음식을 하고 저는 음식을 잘못해서 설겆이담당이예요'], ['저희신랑은 집안일,요리 잘해요~~;;ㅎㅎ제가 민망할 정도에요;;ㅎㅎ그런쪽으로 관심도 많고 좋아하더라구요~~ㅎㅎ'], ['저희는 집안일 청소만 잘 합니다. 오랜 유학생활에 자취를 오래해서 음식을 안 해먹더라구요. 그래서 청소만 시키고 음식은 배달 최대한 안 먹습니다.  회사 사옥에서 주는 밥을 먹고해서 그런 음식 말고 한 사람을 위해서  전 제가 정성스레 만든 음식 먹이고 싶어서, 요리는 제 몫이예요.\n 큰 청소만 시키시는거 어때요? 화장실 청소 분리수거 등등 .'], ['저희남편은 둘다잘해요!ㅋㅋㅋㅋ 요리랑청소랑 저보다 잘해요... 임신중이라 남편이 집안일 전담하고 있어요 ㅋㅋㅋ'], ['저희 예랑이는 둘다 잘해요! 뒷정리는 살짝 못하긴하는데 그건 제가 하면되요!'], ['저희 남편은 둘다 저보다 더 잘합니다 ㅎㅎ'], ['남자친구가 자치 기간이 길어서인지 요리도 제법하고 살림은 더 잘알더라구려 ㅠㅠ 답답해도 참으면서 하나씩 같이 해보서야 할 곳 같아요 ㅠㅠ'], ['저희 신랑은 알아서 잘 하더라구요ㅎㅎ 사람 나름인것 같아요!!'], ['집안일 하면 잘 하더라구요 어쩔땐 저보다 더 꼼꼼하게 잘 하는거 같아요 ㅎㅎ'], ['에혀 귀찮아서 안하는거죠뭐 처음부터 잘아는 사람이 어딨나요'], ['저희 예랑이는 집안일은 잘하는데 요리는 잘 못해요ㅎㅎ'], ['답답하고 화나실때도 있겠지만 차근차근 하나하나 다 가르쳐야 할 것 같아요 ㅠㅜ 안그럼 글쓴님이 나중에 넘 힘드실듯해용 ㅠ'], ['저희 남편은 결혼전 몇년간 자취해서 그런지 요리하고 청소 잘하더라고요~'], ['제 남편은 요리 수준은 저보다 쪼끔 더 나은 편이고 집안일은 저보다 많이 잘해요 ㅋㅋㅋㅋ 저는 오히려 제가 설거지하면 딱 설거지만 하는 정도고, 주변 물기 닦고 배수구망 세척하고 이런건 남편이 하는거 보고 제가 배웠어요...ㅋㅋㅋㅋ'], ['저는 제가 똥손이라 남편이 거의 다 해요. 요리해도 시키는게 마늘까기 파썰기 정도고 양파만 썰려고 해도 자기 쉬라고 해서 그냥 쉽니다.. 주말엔 남편이 삼시세끼 다 해줘요 ㅎㅎ 그래서 저는 어슬렁거리다가 음쓰나 버리고 와요(..) 관심없고 못하는데는 성별이 딱히 관련이 없는거 같아요.'], ['저희 남편도 그래요 ㅋㅋ 대부분 남자가 그런듯요... 말해야 알고ㅠㅠ 요리는.. 똥손인 제가 합니다'], ['제신랑도...해주는것만 해주지...거의 안해줘요ㅠㅋㅋㅋㅋ설거지도 어쩌다한번...해줘도 제가 다시 만지구요ㅠㅠ'], ['집안일은 잘하는데 요리는 정~~말 못해서 제가 다해요~!'], ['저희 남편도 그래요... 외동아들이라 그런가 세탁기도 못돌렸었어여..ㅎㅎ 설거지할땐 행주도 안빨고 정리도 안하구요! 아직까지 대청소는 제가 하고있긴한데 그래도 조금씩 가르쳐놔서 지금은 제법해여..! 제맘에 쏙은 안들지만요ㅋㅋㅋㅋ'], ['시키면 잘해요 ㅋㅋㅋㅋ차근차근 가르쳐주니 이젠 설거지 뒷정리까지 척척 잘해요'], ['설거지하는거 엄청 싫어서해서 \n빨래담당시켰어요ㅜㅜ \n주방일은 제가 그 외는 다 신랑이하네유'], ['저희 예랑이는 청소를 정말 잘해욬ㅋㅋㅋ제게 청소부분은 잘 못하구 대신 요리 담당해서 저흰 역할분담하구있어욬ㅋㅋㅋㅋ'], ['정말로 할 줄 모르시는거 같네요~ 안해보셨으면 앞으로 배우셔야죠~ 하나씩 가르쳐 주던지 아니면 둘 다 안해보세요ㅎ 그럼 본인이 해야할 필요성을 느낄지도 몰라요~'], ['저보다 남편이 요리도 잘하고 더 꼼꼼해요ㅎㅎ'], ['남자여도 유튜브에서 잘 찾아서 먹을거 매번 잘 해주던데용.. 뒷정리도 저보다 더 잘하고..제가 요리나 뒷정리 잘 못..ㅎ..넵..ㅋㅋㅋㅋㅜㅜ']]
    
    8834
    생일날  미역국은 누가~ 오늘 저 생일이예용 그런데 문득 생각나는게 다들 미역국은 먹었냐며 묻는데 다들 내 생일에 본인이 끓여 드시나요갑자기 궁금해져서요 ^^;;엄마는 참 다 챙겨주고 받지는 못하나봐요~~^^​일찍부터 케익들 받고 뭐 먼저 먹을까 행복한 고민하네요 매년 챙겨주는 지인들 덕분에 기분좋은 하루 시작이예요~^^정작 가족은 안 챙겨주고 신랑은 받은 쿠폰 보내주래요 받아온다고 ㅡㅡ​
    
    [['생일축하드려여 ❤️ 행복한 하루되세용'], ['감사합니다 ~^^'], ['생일축하드려요^^오늘은 집안일 내려두시고~커피한잔하며 푹쉬시고 저녁에 맛있는거 꼭드세요😊 남편님께 미역국 끓여달라하셔요^^\n행복한하루보내세요!'], ['네네 감사합니다 오늘은 아무것도 안하려고요 \n삼겹살좋아님도 행복한 하루 보내세요~^^'], ['전 항상 남편이 끓여줘요~\n10년 넘었네요.\n생일 축하드려요^^'], ['남편님 멋져요 ~~^^b'], ['우왕 생일축하드려요~~^^'], ['감사합니다 ~^^'], ['전 결혼 8년차인데 아직은 신랑이 끓여요 ~~~~생일 축하드려요 행복하게 보내세요 ^^'], ['부럽네요 전 9년차 한번도 안해주네요 ㅎ 오늘 생일 핑계로 이것저것 시켜야겠어요 ㅎ'], ['엎드려 미역국받기 해야해요 마구 시켜요~~~~남자들은 그래야 한다니깐요 ㅋ추카해요 한번 더 ~~~'], ['남자들은 말해줘야 알더라고요 \n감사합니다 ^^'], ['늘 제가 끓이다가 이제 내생일이든 남편생일이든 안끓입니다ㅋㅋ 의미없어요!\n생일축하드려요^^  케이꾸 부자시네요🎉  맛있겠어요 ㅋㅋ'], ['그쵸 ㅋ \n그냥 하루 편하게 행복하게만 보내면 되는거죵 ^^'], ['생일  축하합니다  행복한날  보내세요'], ['감사합니다 ~^^'], ['생일축하드려요'], ['감사합니다~^^'], ['신랑에게 미역국 끓이는 기술을 전수해주셔야해요^^ 생일축하드려요~ 맛있는거 많이 많이 드세요~ 가족분들과 예쁜케익으로 신나게 생파도 하시구요~~ 행복한 생일보내세요~'], ['현실은 남편은 돈으로 주면 좋겠어요 ㅋ \n감사합니다'], ['생일 축하드려요 행복하게 보내세요~~'], ['감사합니다 ~^^♡'], ['생일축하드려요..ㅎ전 생일날 미역국 안먹어요..ㅋ'], ['질리긴 해요\n아이가 미역국 좋아해서 수시로 끓이거든요 ㅜㅜ'], ['생일 축하해요~~^^\n미역국 안 먹으면 어떤가요??\n하루종일 집안일하지마시고\n먹고싶은거 먹고 하고 싶은거 하면서 쉬세요^^\n\n그리고 친정부모님께 전화 한통 하세요~~^^'], ['오늘은 내가 하고싶은거 \n오랜만에 카페놀이 좋아요~ㅎ'], ['가족 누구든 생일에 미역국 안 먹어요. 제 생일에 못 얻어 먹으니 불만 없이 공평하게~^^'], ['공평하게 넘 좋네요 ㅎ ~~^^'], ['생일 축하드려요~\n지인들 선물도 좋은데요\n오늘 휴업 하세요\n결혼 첫해는 남의편이 끓여주고 \n이후는 제가 끓여요'], ['저도 평소 아이 주려고 끓여서 얼려놓은거 많이 먹었던 기억이 ㅋ'], ['생일축하드려요~^^ 행복한 하루 되셔요~~~'], ['감사합니다 ~~^^'], ['생일 축하드려요~행복한 하루 보내세요🥰'], ['감사합니다 \n홍시님도 행복한 하루 보내세요~~^^♡'], ['생일 축하드려용~~💐🥳좋은 날 보내세요~~'], ['감사합니다\n소망사랑이맘님도 좋은 날 보내세요^^'], ['생일 축하드려요~~~^^전 처음부터 남편한테 선언했어요. 미역국을 안 먹었음 안 먹었지 절대 내 생일에 내 손으로 끓여먹지않을거라고.. 10년째 남편이 끓여주고 있네요~'], ['처음에 습관 잘하셨네요 \n남편분 멋져요~'], ['생일축하드려요^^ 저는결혼하고 친정엄마도 끓여주시고 남편도 끓였는데 신랑꺼는 맛이없어서 두해먹고는 하지말랬어요ㅋㅋ염치없지만 엄마가 해주시는거 먹어요'], ['맛없으면 먹는것도 곤욕이겠네요\n안먹는걸로요 ㅎ'], ['전 아들래미랑 생일이 같아서요^^;;; 강제 미역국을 제 스스로 끓여요 ㅋㅋㅋ'], ['좀더 커서는 아들이 끓여줄날 오겠죠 ^^'], ['저두  미역국 얘기하믄 짜증나요\n신랑,애들생일엔 미역국끓여줬냐 확인전화하시면서 정작 제생일엔 끓여먹었냐전화와요\n성질나서 안끓여요'], ['미역국 크게 생각 안했는데 그런전화 오면 싫겠어요 ㅜㅜ'], ['우와 생일 축하드려요 ^^'], ['감사합니다 ~^^'], ['생일 축하드려요~~ ❤️전 친하게 지내는 이웃지인들이돌아가면서 미역국 끓여줬었는데 지금은 이사를 와서 ㅠㅠ 애들이 크니까 비비땡미역국 사와서 뎁혀주더라구요~ 손에 물 뭍히지 마시고 맛난 음식 포장 배달해서 드세요'], ['지인들 돌아가며 좋네요 \n미역국 안먹어도 되고 \n손에 물 안묻히고 자유시간이 젤 좋죠~^^'], ['생일 축하드려요💕남편분께 미역국 끓여달라고 하세요~저희 남편은 미역국은 끓여줘요 ㅋㅋ'], ['남편분이 끓여주는분 많네요 \n보기 좋아요\n댓글들 신랑 보여줄까봐요~~^^'], ['생일축하해요~^^즐거운 시간 보내세요~ 저희는 남편이 끓여줘요ㅎㅎ'], ['우와 자상하신 남편님 최고예요~~\n축하 감사합니다 ~^^'], ['저는 아예 안 먹어요~귀찮아요\n생일 축하드립니다! 좋은 하루 되세요'], ['맞아요 지겨운 미역국 평소에도 많이 먹으니\n배달이 좋아요 ^^\n시인스머프님도 좋은하루 되세요~~'], ['남편이 1년 편하게 살려면 생일날 미역국 끓여야 한다고..늘 끓여주네요\n생일 🎉  축하해요~^^'], ['ㅋㅋ 옳은말인걸요 눈치 백점 훌륭한 남편님 이네요 ^^\n감사합니다'], ['생일 축하드려요~ 오늘은 배달찬스 이용하셔요~'], ['네네 오늘 주방 휴업이요\n감사합니다 ~^^♡'], ['전 반찬가게요ㅠ\n남표니는 끓일줄 몰라서요ㅠ\n생일 축하드려요~^^ 좋은하루보내세용!!'], ['뭐든 맛있게 먹고 행복한 시간 보내면 되는거죠\n감사합니다 ~~^^'], ['저희 신랑도 오늘 생일인데!!저도 제 생일에 몆년전까지 제가 미역국 끓여서 먹었다가 2년전엔가 한번 뒤집었어요. 신랑한테 끓이라고 시켰습니다. 그 후론 제 생일엔 무조건 미역국은 신랑 담당이요~~ 오늘부터 시키세요!!!!'], ['ㅋㅋ 잘하셨어요 \n저도 그럴까요 ^^']]
    
    8883
    빨래산더미 빨래가 이만큼 쌓였네여ㅠㅠ 너무 빨리 쌓이네요ㅠㅠ 빨리새고 설거지도 하고 집안일도 해야겠어용ㅎㅎ 집안일 시작합니당ㅠㅠ 맘님들은 뭐하시나영?ㅎㅎ
    
    [['전 오전 집안일 이제 끝냈습니다ㅠㅜ'], ['고생하셨네유 ㅠ'], ['저도 시작해야합니다.'], ['이모티콘 너무 귀여워요 ㅋㅋ'], ['귀엽죠.. 설거지안하고 생라면먹어요.'], ['생라면 구워먹으면 맛나쥬 ㅎㅎ'], ['귀찮아서.,  그냥먹는걸루..'], ['천천히하셔용 ㅎㅎ 저는이제 과외갑니다.총총'], ['과외 하러다니셔요?ㅎㅎ'], ['네넹ㅋㅋ 이제 곧 그만두겠지만요 ㅠㅠ'], ['아하ㅠㅠ 아쉬우시겠어요 ㅠㅠ'], ['그래도 이제출산하려면 어쩔수없죵'], ['그렇져ㅠ 아가가 우선이져 ㅎㅎ'], ['넹맞아용🥰'], ['저는 빨래하고 건조기 돌려요. ㅎㅎㅎㅎ'], ['건조기 너무좋죠?ㅎ'], ['최고예요^^ 여름 겨울엔 필수템입니다'], ['오옹ㅎㅎ 줄어든다고 하던데 괜찮나여?'], ['반반이요^^ 멀쩡한것도 있고요  저는 그냥 돌려요 ㅎㅎ'], ['오마야ㅠㅠ \n전 정리하고 앨범도 정리하고~~ 따꽁이렁 놀고잇어여ㅠ'], ['아구ㅠㅠ 앨범정리 너무 귀찮아여 ㅠ'], ['빨래 젤 귀찮아요 ㅠ'], ['잴 귀찮은데 할수밖에 없네유 ㅠ'], ['저희집빨래도 한동안 저만큼씩있었네용 ㅎㅎ'], ['ㅋㅋㅋ너무 빨리 쌓여요 ㅠㅠㅋ'], ['애둘이면 더많이쌓이겟죠ㅠㅠ'], ['엄청 많이 쌓여요 ㅠㅠ'], ['ㅠㅠㅠㅠㅠㅠㅠㅠ저희집은 빨래엄청많은데\n둘째나오면.....감당될지의문이네요...ㅋㅋㅋㅋㅋ'], ['빨래 개주는 기계는 언제 나올까요...'], ['어여나왔으면 좋겠어요 ㅠㅠ'], ['오늘 집안일 많으시다더니\n빨래가 진짜 한더미네융 ㅠㅠ\n집안일은 잘 하셨나용?'], ['넹 내일도 해야쥬 ㅠㅠ 끝이없네여 ㅠㅠ'], ['집안일은 해도 티도 안나고 끝도없쥬🤣'], ['해주는사람이 있었으면 좋겠어요~'], ['그러니깐유 ㅋㅋㅋ복직하면 청소업체는 한번씩.부르고 싶네요🤣'], ['오옹ㅋ 주위에보면 살림하는데도 5시간씩 해서 사람부르더라구요 ㅋㅋ'], ['아직은 돈이.그정도는 아니라 못하겠어융 ㅠㅠ 복직하면 저도 그렇게.해보려구용ㅋㅋ일하면 집안일하기 힘들테니 ㅠ'], ['맞아유 ㅠㅠ 저는 돈아까워서 사람 못부르겠더라구요 ㅠㅠ'], ['저도 아직은 그런데 일하면 힘들어서 부르고싶어질거 같습니다융 ㅠ'], ['잦아여 ㅠㅠ 일하면 말이 달라지져 ㅠㅠ'], ['넹 ㅠㅠ 지금도 집안일 힘든데.일하면 어쪄련지 벌써 걱정되네요 ㅠ'], ['그렇져 ㅠㅠ 그래도 다 하시더라구여 ㅠ'], ['어후 ㅜㅜ 일단은 복직하기전에.미리.잘 놀아놔야지유🤣'], ['맞아유 ㅎ 무슨일 하셔영?'], ['그냥직장댕겨융...ㅋㅋㅋ직장인이쥬 ㅋㅋㅋ'], ['아항ㅎㅎ 저는 미용했었눈데 습진이 너무심해서 다른일 해야할꺼같아여 ㅠㅠ'], ['미용하면 손이 진짜 남아나질 않더라구요 ㅠㅠ'], ['엉망진창이에여 ㅠ'], ['진짜 주변에 미용하는분들 보면 그렇더라구요 ㅠㅠ약이 엄청 독해서 그런거지유?'], ['넹ㅠㅠ 일을그만둬도 그때 일해서 피부가 약해져서 겨울만되면 심해지네여 ㅠㅠ'], ['겨울에 엄청 힘드시겠어요 ㅠㅠ저는 찬바람알러지있어서 손이 겨울되면 난리인데 고보습해주면 그나마 나아지더라구요 ㅠ'], ['너무 힘들어요 ㅠㅠ 피날정도니까유 ㅠㅠ'], ['저도 겨울되면 피부가 그래융 ㅠㅠ 진물나고 엄청 아프고 장난아니에유 ㅠㅠ'], ['헐ㅠㅠ 맘님도 고생많으시네요 ㅠㅠ'], ['넹 벌써 진물나오고 난리여유 ㅠㅠ저도 그래서 겨울이 너무나 싫쥬 ㅠㅠ'], ['ㅠㅠㅠ어휴.. 피부과는 가보셨어용?'], ['피부과 가도 안낫더라구요 ㅠㅠ고보습 해주는게 그나마 낫더라구요 ㅠ'], ['아규 ㅠㅠ보습이 답인가봐여 ㅠㅠ'], ['더운나라로 이민이 답입니다융🤣🤣'], ['이민까지 가야하나여ㅠㅋㅋ'], ['넵! 진심 겨울만 되면 이민가고싶습니다유 ㅠㅠ'], ['어휴 ㅠㅠ 가렵진 않으셔여?ㅠㅠ'], ['가려워서 긁으니 피나고 진물나고 ㅠㅠ 찬바람만 불어도 그렇게 되고 ㅠㅠ 찬바람때문에 그런거라 어찌 안고쳐져유 ㅠ'], ['아이구 ㅠㅠ 날이 따뜻해지는 그날만 손꼽아 기다립니다 ㅠㅠ'], ['저는 미뤄두고잇네용'], ['저두 미룰때많아여 ㅋㅋ 낼은 대청소 할려구용'], ['전 오늘 친정에서ㅜ거의 하루를...집안일은 내일할까해요 ㅠㅠㅠ 하루안하면 쌓이는..ㅠㅠㅎㅎ'], ['내일하셔용ㅎㅎㅎ 하면 티안나고 안하면 티 확나져 ㅠㅠ']]
    
    8904
    요리 하는게 집안일 중 힘든일 1순위? 저는 남편이랑 집안일을 공평하게 한다 생각하는데남편은 항상 아니다라도 해서요그 이유인 즉 요리는 다른 집안일과는 대체불가라고 하네요. ?전 요리. 먹는거에 별 관심이 없는데남편은 요리도 좋아하고 먹는거에 관심이 많아요. 그리고 직접 해먹는걸 좋아해요예를들면 칼국수를 먹더라도면은 사와서 할수 있자나요?그런데 면까지 집에서 반죽해서 뽑아요돈까스도 시중에 파는거 튀겨먹을수 있자나요?고기만 사와서 다지고 묻히고 튀기고 다 합니다. ?맞벌이라 전 주말에 좀 쉬고 싶으니적당히 먹었음 좋겠는데매번 이번주는 뭐먹을까 왜 맨날 나만 생각하고 고민하냐뭐 시켜먹자 말하면 뭘 시켜먹냐 집에서 만들어먹는게 더 맛있다 이런식이에요. (그렇다고 요리솜씨가 훌륭하진 않고 걍 보통이에요.)?물론 본인이 다 만들긴 하죠..그런데 너도 옆에서 보고 배워라음식만드는건 왜 맨날 나만하냐 너도 좀 해라잔소리가 넘 심해요. ?일단 주말에는 모든 설거지 제가 다하고청소. 빨래 제가 하고아침에 아기랑 일찍 일어나서 저녁까지 육아도 제가 해요?남편은 주말엔 오전 11시~12시까지 자고 일어나서주말 점심 저녁 4끼중 중 3끼 차리고음식물 쓰레기 버리고 이렇게 하는데?요리하는거에 다른 집안일 아무것도 갖다댈수 없다고제가 집안일을 항상 안한다고 더해야한다고 합니다. 다른건 그냥 몸만 힘들고 다라고. 요리는 신경쓸게 많다고요. ?근데 저는 소요시간이나 체력적으로 따지나 집안일 충분히 하고 있다 생각하거든요. 거기에 육아도 거진 제가 다 하고요?저 요알못이라 진심으로 물어보는데요리가 집안일중 제일 힘든거 맞아요?맞다면 제가 생각을 좀 고쳐먹어보려구요.. ?
    
    [['그렇게 죄~~~ 다 만들면 힘들긴 해요ㅠ 육아 제외하고 하기 싫은 집안일 고르라면 밥하기 싫어요.. 저는 요리 못하는게 아닌데도요ㅠ'], ['전 요리요~'], ['ㅋㅋㅋ헉 ㅋ 그렇게 다 만들면 당연히 힘들죠...반조리제품이라도 사시지 ㅜ'], ['육아제외하고 집안일중 제일 힘든건 요리가 맞는 것 같아요 ㅜㅜ'], ['종일 요리하는게 힘들긴해요ㅜ(특히 메뉴 고르는게 제일스트레스..)저는 전업이라 제가 다하긴하는데(남편이 요리는 1도못하기도하고요)..그래서 한끼정도 사먹었음 하는날도 많아요~근데  글쓴님 남편분이 사먹기시러하는거라..힘들다하심안될듯싶고요..아내분도 청소며 육아며 다하시니 니가잘했다 내가 잘했다 할건없는거같아요ㅜ.ㅜ집안일은 다 힘들어요ㅎㅎ'], ['근데 ..그렇게하면 당연히 힘들거같아여 남편분'], ['저는 요리가 제일 스트레스긴해요'], ['음식을 좋아하고 만드는것도 좋아하는 분은 힘들단 말은 잘 안하는거 같아요 해서 먹으면 행복하니까요... 그래서 남편분은 음식은 집안일에서 분리하는것이 아닐까하는데요...\n당근 여자분들은 집안일하고 먹고 치우고 반복적인데 남자들은 해봐야 한달 몇번해서 그럴수도 있고... 남편한테 그러세요 난 내가 만들어 먹는것보단 당신이 해주거 먹는게 더 좋다고 ㅎㅎㅎ'], ['저는 요리하는건 쉽던데.. 대신 설거지가너무 힘들어서 그건 신랑시키고 전 요리해요 요리가 더 편하다에 1표 (사람마다 다를듯!!)'], ['저도 요리스트레스가 제일 심해요진짜 요리만 해결되면 육아가 한결 편할 듯해요집안일중에 제일 힘드네요 저는'], ['집에서 다 만들어 먹지만요리가 제일쉬워여 ㅋㅋ재미도있구요 ㅋ뒷처리가힘듬 ㅠ 집안일은 더 싫구요 ㅋㅋ'], ['아니 본인이 좋아서 하는거면 스트레스를 받지 말아야 하는거 아닌가요 하질 말든가'], ['요리힘들어요근데 설거지는 정말하기시러요남편분이쫌만 양보하면되는데 아주 지극정성이네요그냥군말없이하던가.'], ['집안일 중 고르라면 요린데.. 제 기준.. 세상 가장 힘든 육아를 님이 하시니 투정부리면 안돼죠..'], ['요리가 그렇게 힘들면 시켜먹고 돈까스만들 때 고기 두드리기부터 할까요? 제가 볼땐 요리가 좋아서.그러는거.아니에요? 요리자체보다 뒤처리가 힘들잔아요. 그거 님이 다 하시고. 남편 요리하는 동안 육아도 님이 다 하시는데. . 단지 요리하나 한다고 집안일 운운하는건 아닌거같아요. 그냥 맞벌이고 힘드니.대충 먹자하세요. 요리하는것만금 뒤처리도 힘들다고'], ['저에게는 제일힘든게 요리인건 맞아요. 뭐먹을지 고르는거부터 재료사고 준비하고 만들고 전부다 스트레스고 힘들어요.ㅠ'], ['삼시세끼 만들어 먹으려면 머리통 지진 납니다주부들 오늘 저녁은 또 뭘 차려야 할지 고민하는게스트레스중 하나일 걸요.저도 예전엔 이틀에 한번꼴 반찬 대여섯가지에 국메인 메뉴까지  만들어 먹곤 했는데 넘 스트레스 받아서이젠 절반이상은 배달로 연명하는 중이에요 ㅠ'], ['전 차라리 요리가 나은데 남편분처럼 해먹으면 누구라도 요리가 젤 피곤할듯요.'], ['요리가 젤 힘들어요'], ['요리가 젤 스트레스에요..오죽하면신랑 저녁에 밥 안먹고 들어오나 매일매일기대합니다ㅎㅎ밥 약속 생기면준비해 놓은 저녁 내일 메뉴로미뤄진거 자체도 행복하고(내일 여유가 생기니)차리고 치우고 하는 행위 자체가 생략되니넘나 좋아요ㅠㅠ친구들 단톡에도오늘 뭐해먹냐  내일 뭐해먹냐 밥만 안해도살겠다 등등 빠지지 않는 고충이에요전 요리도 곧 잘 하고 맛있다고 하는 편이에요근데 진짜 넘나 힘들고 시러요ㅠㅠㅠㅠㅠ지금 이 순간에도 내일 저녁은 뭘 해먹어야 되나 고민중입니더ㅎㅎ'], ['그렇게 하면 힘들긴 한데 상대가 바라지도 않는데 그렇게 하고 생색내는 건 좀 아닌 거 같아요 ㅎㅎ'], ['동의합니다'], ['아..물론 음식만들고 먹는게 가장 고되고 루틴도 매번 비슷하니 고민은 맞는데요...남편분은 무튼 본인 좋아 하시는거쟎아요...뭘 사먹냐 집에서 만들어 먹지?!...본인 기준인거고...님도 노는게 아니네요...음식에 좀 관심 없다뿐인거지...제 보기엔 남편분이 알아.달라는거 같아요'], ['전 요리가 제일 힘들어요'], ['요리는 껌이고설거지가 가장 힘들어요'], ['그냥 제목만 봤을때\n집안일 중 가장 힘든거 요리 맞는거 같아요.\n다른 집안일보다 신경도 많이 쓰이고\n남편분처럼 하나하나 손수 하려면 재료 준비부터 식재료 관리까지 손이 많이 가니까요.\n\n그리고 솔직히 너무 부러워요ㅠㅠ..\n전 다시 결혼상대 고를 수 있다면 요리를 좋아하고 잘하는 사람 고르고 싶은 사람이라서요...'], ['저도 요리가 제일 힘든거 같아요. 저희 집도 주말에 신랑이 거의 요리 하는편이고 저는 청소 및 집안일 하는데.. 세끼 밥만 차려 준다면야 저는 청소가 몸은 힘들긴 해도 제기준에선 훨 편한거 같아요. 요리는 정말 너무 귀찮ㅠ 대신 아들이라 아빠랑 노는거 좋아해서 육아는 신랑이 다하지만요.  요리가 힘들다 해도 그것말고 1도 안하는건 좀 아닌거 같아요.'], ['걍 받아만 먹는 사람은 모르겠지만 있는거 대충 먹지않는 이상 한끼 차릴려면 기본 2시간은 서 있어야 되고요.. 메뉴 걱정해야 되고.. 설거지야 식세기가 한다쳐도 다 정리해서 넣어야 되서 있어도 귀찮아요'], ['저희도 애 셋 맞벌이인데남편이 요리 전담이거든요힘든 건 아는데너무 생색내고 본인만 고생하는 것처럼눈치보이게 해요밥 안 먹고 싶어요ㅠ같은 직업인데애 셋 낳으라 8년육아휴직하고 저는 나름 많이 희생했거든요. 그러다공황장애까지 생기고ㅜ제가 보기엔 돈도 벌고집안일도 완벽하게 하고애도 제가 전담해서 잘 키우고그랬음 좋겠나봐요ㅠ 저는 인간이지신이 아닌데요ㅠ'], ['제 마음이 님 마음이네요 ㅠ \n생색 엄청내고 본인만 고생하는것 처럼 눈치보이게 해서 저도 밥 안먹고 싶어요 ㅠㅠ \n뭘 그렇게 다 완벽하게 했음 하는건지. \n바쁘게 계속 움직이고 집안일 뭐 하나 더 할거 없는지 생각하고 더 완벽하게 하길 원하는데 주말엔 저도 좀 쉬고 싶거든요.. \n\n'], ['아기도있고 맞벌이인데 그렇게 요리하시면 당연히 힘들수밖에없을듯요약간 자처하는느낌도있는데요..'], ['요리가 젤힘든거 맞아요 딴거 내가 다할테니 누가 밥좀 해줬으면 좋겠어요 남편분은 더 힘들게 하는스타일이기도 한거같구요'], ['전 집안일 중 뿐만 아니라 제가 해본 일 중 요리가 제일 힘들어요.  육아, 공장, 노가다 다 할 수 있겠는데 결혼하고 요리 땜에 우울증도 걸렸어요.'], ['저도요리..'], ['죄다 어렵죠.. 요리도 맛내기가 어렵고..\n전 요리보다 설거지가 어렵습니다..'], ['요리가 젤 하기도싫고\n어렵기도해요..'], ['요리가 젤 스트레스에요ㅠㅠ'], ['요리가 제일 힘들다기 보다는 남편이 적당히 해도될걸 뒤지게 힘든길을 걸어가는거예요 ㅋㅋㅋ..'], ['요리를 못하지는 않는데,\n식구들 맛있게 잘 먹는 거 봐도 하고 싶은 생각이 1도 안 들어요.??\n요리가 제일 중노동으로 느껴져서.\n코로나로 쉴 땐 좀 하다가 다시 일 시작해서\n이젠 요리 안 합니다.ㅠ'], ['요리는 괜찮아요\n뒷정리가문제지\n뒷정리만 커버되면 요리는 면뽑아서라도 해요\n밀가루치워대고 기름 치우고 그런게힘들죠'], ['요리는 비교 불가 입니다....'], ['요리 제일 힘든 거 맞는 거 같아요.ㅜㅜ'], ['사실 요즘 세탁기있지..건조기있지.. 로봇청소기.음식물처리기.식기세척기.. 넘 힘들면 기계라도 있지 요리는 배달도 한계가있고 애도먹여야하고 간식챙기고 메뉴선정부터 어쩌다하면.재미있어도 매번하기엔 젤 힘들어요ㅜㅜ'], ['솔직히 집안일 중에 요리가 젤 힘들죠\n요리빼면 1도 안힘듦ㅠ 귀찮을 뿐\n집안일 중 요리는 매일 메뉴 고민부터 재료 준비 및 요리 과정,, 그리고 뒷정리까지 피곤쓰8,8~ 삼시세끼 차려먹우려면 힘들겠어여,,ㅠ'], ['요리도 같이하시고 집안일도 좀 분담하시면 안되나요?ㅜ\n\n저희도 주말은 신랑이 요리하는 편인데\n요리할때 제가 옆에서 다 보조하거든요ㅜ\n\n따지자면 요리 보조가 젤 힘들어요!ㅜㅜ'], ['요리가 제일 힘들어요\n맬같은 음식 올리기도 좀그렇고 끼니때마다 해야되고 영양도 생각해야되고 나물이잇음 생선또는 고기반찬\n간장양념잇음 고추장으로 된 양념잇어야되고\n시켜먹는게 편하긴한데 위생따지고 여튼 힘들어요ㅜㅜ'], ['요리제일싫어요 다른건 다하겠는데 요리는 극혐. . 전 보조만해도 행복해요. .'], ['집안일 중에 요리(부엌 뒷정리포함) 젤 힘든거 맞아요집안일 다합친거라 요리 하나 비슷하다 여겨져요'], ['요리하는것도힘들지만 어떻게 매번 해서먹나요?주부인저도주말엔시켜먹거나남편이해줘요ㅜㅜ자기가 하는건데 생색은왜내는지ㅜㅜ자기가좋아서 하는건데 본인이 설겆이도하셔야죠ㅜ맞벌이시면 더힘들텐데요ㅜ저희신랑도 힘들어서 한번씩 시켜먹어요~~'], ['저도 설거지가 더 힘들어서 요리가 편하다에 한표요~'], ['저도 요리 넘 스트레스 받아요ㅜㅜ\n남편 퇴근할 시간만 되면 저녁 뭘 어떻게 해먹을지 스트레스 넘 받아서 차라리 시댁에 있으면 어머님이 저녁 해주셔서 시댁에 있는 날이 훨씬 맘 편할 정도예요!'], ['평일에 7인분 주말에 8인분하는데, 요리가 젤 힘든단 생각은 안드는데요..'], ['요리...정말 짜증나고 힘들죠 식구들 드르렁하고 잘때도 낼은 뭘하나 매일 메뉴고민에 요리하는 과정에 설거지는 얼마나나오는지 맛내랴 재료손질하랴 요리진짜 귀찮아요']]
    
    9003
    집안일 남편이 다하시나요? 임신전에는 반반하다가 임신하고는 제가 잘못하고 있는데 남편은 저보고 너무 안한다고 하네요;;;그렇다고 본인이 다하지도 않아요 덩달아 안하는거같아요...똑같이 맞벌이하는데 일끝나고 쉬고있음 방에들어가 게임하고있어요 ..전 청소라도 좀  해줬음 좋겠는데 ㅠㅠ 집에 머리카락이랑 음식물 이런거 보면 스트레스 ........원래 깔끔한 성격음 아니지만 정말 너무 더러워요?뭐 먹고 싶은건 없는지 물어봐주고 밥 해주진 못하더라도..생각해주는게 어려울까요 남들은 영양가있는거 먹게 라면도 못먹게하고 챙겨주려하는거같은데...제가 늦게 들어오는 날은 밥도 안먹어요 괜히 신경쓰이게 ...저도 챙김받고싶은데 남편까지 신경쓰려니 가끔 울컥하고 억울해요 ?얘기하다보니 남편 욕하는거같은데^^ 평소엔 친구같고  좋은데 요즘은 왜이리 어린애같다는 생각이드는지.....
    
    [['요리, 설거지, 빨래, 청소 - 본인\n화장실청소, 분리수거, 바닥물걸레, 아이놀아주기+재우기+목욕시키기 -남편'], ['임신했어도 집안일 임신전이랑 비슷하게 한거 같아요ㅋ 남편한테 바라면 나만 실망하는 일이 생기니 그냥 냅뒀어요~ 대신 밥은 잘 안챙겨줬어요. 배고프면 시켜먹던 라면먹던 하라고'], ['마자요 자꾸 실망하게 되서....마음을 바꿔봐야겠어요 ㅠ감사합니다'], ['임신 기간에도 평소처럼 집안일 했어요. 제가 삼시세끼 꼬박꼬박 집밥 먹는 사람이라 다 해 먹었고요. 임신했다고 달라지는건 없었네요.'], ['222 저두요 ㅎㅎㅎ'], ['3333 댓글들 남편부럽네용.. 임신과 집안일은 상관없는줄알고 살아오고잇어요ㅋㅋㅋㅋ 20키로까진 무겁다생각안해서 쌀도 다옮겻는데..'], ['초기 하혈 이벤트로 식세기 구입후 설거지 벗어났고 맞벌이 아닌데도 남편이 퇴근 후 식사준비 도와줘요\n쓰레기들 다 버려주고 주말에 청소기, 물걸레질해주고 가끔 빨래널고 개기, 욕실청소 해줘요\n조금이라도 무리된다싶으면 쉬라고 해서 남편이 하는 일이 많네요\n임신전에도 많이 도와주는 편이긴 했고 너무 너무 어렵게 가진 아기라 더 도와주는 듯 해요'], ['저 25주까진 남편이 다했는데\n이제 안정되고\n입덧도 괜찮아지니 같이 안해요;;;;; 배달음식만먹어\n분리수거가 장난아니네요ㅠ'], ['임신 16주에 제가 일을 쉬고있는 상태인데.. 남편이 집안일 95% 정도 하고 저는 거의 아무것도 안해요.\n남편이 다음날 저 먹을 밥, 반찬, 국 다 해놓고 출근하고.. 집에오면 마사지도 해줍니다. 임신했을때 누리라면서 아무것도 하지말라네요'], ['헐 실화인가요?'], ['전 임신준비부터 일 안했는데 입덧시작하고 2달정도 집안일 아무것도 못하고 남편이 다해줬어요. 또 남편도 다행히 재택 해서 제가 먹을수 있겠다싶은 음식 해주고 사다나르고...입덧끝나고 해줬던게 고마워서 제가 집안일 했는데..저도 어리고 이기적인 사람이라 님 남편분처럼 했을지도 모른다는 생각은 들어요 ㅜㅜ임신이라는것만 빼면 손해보는 느낌이니까요ㅜ겉으로 티도 안나고 실감이 확 안나겠죵 에흉. 얼른 배속에 아이를 품고있는게 얼마나 소중하고 대단하고 고생스런 일인지 느끼셔야 할텐데 그런걸 억지로 주입하기가 어렵긴 하죵 ㅜ임신 다큐라도 한번 같이 보세요 ㅜ'], ['배부르기 전까진 힘들다고 얘기해도 남편이 실감안나하고 잘모르는거 같더라구요, 임신이란게 막연히 대단하고 힘들다고  생각하는거 같았어요~ 저도 다큐보면서 남편한테 주입?시키고 어리광부리고 정말 힘든순간들도 있었지만 더 엄살?부렸어요ㅎㅎ 전 역류성식도염을 동반한 먹덧이었는데 남편이랑 있을때 토하면 더 힘든티내고 남편 회사갔을때 토하면 꼭 보고했어요ㅎㅎ  이렇게 안하면 남편은 자기몸이 아니라 그런지? 실감이 안나서 그런지? 잘 모르더라구요~아, 이말도 했네요~ 나는 내 뼈와 살과 피를 다 바치고있다고...  ㅎㅎ'], ['ㅋㅋㅋ돌이켜보니 저도 힘들면 더 티냈네요 ㅋㅋ 먹은게 없어서 물만나와 ㅜ 나도 엎드리고 싶은데 못엎드리니까 허리아파 ㅜ 이러면서 찡찡찡ㅋㅋ 착한남편이지만 센스는 그저그래서 ㅋㅋ'], ['겪어보지 않은일을 이해하길 바라는 제가 너무 욕심이 과했나봐요 ....ㅠㅠㅠ남들은 다 그렇다니까 제 남편까지 똑같이하길 바랬던게 컸던거같아요 비교하지말아야하는데 .....감사합니다!'], ['ㅋㅋㅋ제 남편두요~ 착하지만 센스는...ㅎ남편은 자취경력 15년으로 집안일도 저한테 의존적인 남편은 아녔어서 알아서 잘하는 부분도 있었는데 (애기가 너무 순해서 지금은 의존적ㅜㅜ) 그래도 얘기하고 제 몸상태에 대해서 계속 표현해야 하더라구요ㅋ 6~7개월부턴 조금만 힘들다싶으면 엄청 불러댔어요ㅎㅎ'], ['비교는 하지 말아야하지만 배려는 받으셔야해요~이제 배 불러오고 힘들어지실테니까요~  부인이니까 배려받고싶으니까 라고 하지마시고 당신 애기 임신해서 힘드니까 라고 해보세요ㅎㅎ 여자라고 배려받고 남자라고 배려해줘야되냐 하는 분들도 계시니까 ㅜㅜ'], ['아....정말 그럴수도 있겠네요 ㅠ감사합니다..!!'], ['전 백수이고.. 임신 중 70%를 눕눕 중입니다. \n초기 ( 16주 )\n- 본인 ; 아침, 점심 본인 먹은거 요리 및 설거지,  저녁 가끔 요리 , 청소 (청소기 밀기)\n- 남편: 설거지, 빨래, 화장실 청소, 분리수거\nㅡㅡㅡㅡㅡ초기에 둘다 제대로 청소는 못했어요 ㅋㅋ\n현재 24주.... 초기랑 비슷한데 제가 청소는 화장실 빼고  다하고.. 분리수거나  빨래정리 , 저녁 설거지, 화장실 청소등은 아직도 남편이..하는 편이예요. 부끄럽지만.\n..저희 집은 청소 잘 안해요 ㅜㅡㅜ'], ['전 신랑시켜도 성에안차서 더 짜증나더라구요 ㅠㅠ 그냥 제가해야 맘이편해요 아 신랑이 쓰레기는 버려주네요ㅋㅋㅋㅋ'], ['전 처음 임신때 맞벌이에 평소처럼 집안일 대부분 나눠하거나 제가 좀 더 하고, 밥도 제가 해주며 지냈어요. 근데 계류유산 했고, 이번에 다시 임신했는데 입덧이 심해 무급휴직 중이에요. 남편이 지금 집안일 9할은 하구요. 이제 입덧 좀 잠잠해져서 밥은 해주고 있어요. 컨디션 안좋으면 못해주고 배달이지만요. 겪어보면 바뀌는게 있는 것 같아요. 그 전엔 모르는 것 같구요. 씁쓸하지만요..'], ['다는 아니지만 어느정도는 같이하죵..무거운건 아예 못드니까요'], ['저도 맞벌이였는데 요리만 제가하고 빨래 설거지 청소 남편이 다 했었어요 집안일 하라고 얘기한 적 없는데 남편 성격 같아요 대신 출산휴가 후에 설거지는 제가 하고 남편 출근할 때 도시락 챙겨주고 있어요!'], ['출산 2달 전까지 근무. 출산 전날까지 새벽 6시 20분까지 남편 먹을 아침식사 차리기, 화장실 청소, 빨래, 저녁식사 준비했구요. 남편은 주말에 집안 청소, 설거지 가끔, 재활용 쓰레기 버리기 했어요.지금은 아가 낳고는 아침식사 준비 안하고, 설거지는 식세기 사서 식세기 도움 받아요. 대신에 아기 케어의 90프로는 제가 다해요. 아기 목욕과 똥 기저귀 갈기는 전적으로 제가 하고 있네요... 아가 똥 싸면 남편이 저를 찾네요...ㅠ임신 중에 남편이 먹을거 사다 나르고, 집안일 많이 해주셨다는 분들 부럽네요. ㅠㅠ'], ['저희 남편도 똥기저귀 못 갈아요ㅋㅋ 임신 기간 동안 저는 그닥 먹고 싶은게 없었어서 사다달라고 한 적도 없네요ㅋ 집안일도 누가 대신 해줄 필요도 없었구요. 임신이 병은 아니니까요.'], ['저는 초기 유산 한번 하고 그뒤 또 임신해서 유산기로 눕눕 했고, 맞벌이 하는지라 70퍼 정도는 남편이 해줘요 ~~ 원래 결혼 하고 나서부터 빨래,  청소, 분리수거, 화장실 청소 , 밥하기 등 쪼금씩 시켰더니 이제는 알아서 다 하네요 ㅋㅋㅋㅋ \n역시 뭐든 조기교육(?)이 중요해요'], ['ㅋㅋㅋ 조기교육ㅋㅋ 맞아요\n음식도 가르쳐놨더니 쉬는 날 밥해줘서 좋아요'], ['임신전 맞벌이, 첫째 거의 독박육아에 집안일 거의 독박 \n남편이 하는건 어쩌다 한번씩 화장실청소, 쓰레기 버리고 오는건데 그것마저 저랑 같이 갔다와야함\n\n임신후 전업 \n입덧때문에 남편이 하는건 설거지와 음쓰버리기가 추가됫는데\n조기진통으로 입원하고나선 저보고 집안일 하지말라길래 남편이 할줄 알았는데 안함\n집이 난장판이여서 집안일,첫째케어 하다가 하혈하고 재입원\n남편은 그러게 하지말라니까 왜 하냐고 그냥 어지러진채로 두지 라고함\n\n둘째 출산휴가중인 지금의 남편은 시키면 다 하긴하는데 하라고 하면 몇일뒤에 해서 문제\n담주되면 남편 출근하는데 어케 바뀔지 모르겠네요'], ['ㅠㅠ몇일뒤에 해서 문제... 격한공감입니다ㅋㅋㅋ'], ['입덧 많이 심할땐 요리,설거지,빨래,청소 다 남편이 해줬는데 좀 가라앉고나니 같이해요~'], ['저는 맞벌이여서 임신했을때 집안일 아예 못했어요 회사를 무사히 출근하는 것만으로도 죽을맛이여서 집에서는 거의 기절상태였는데 남편이 군말없이 다했네요~몸 컨디션이 좋아서 같이 하면 좋겠지만 임산부랑 똑같이 할려고 하는건 너무한거 같네요ㅠㅠ'], ['임신땐 누워있기 티비보기 요가\n출산후 아기보기, 제 화장실청소, 아가방청소\n\n신랑이 청소,빨래,설거지,요리,분리수거,아기목욕,젖병씻기,아기놀아주기(3시간) 맡아서 봐주기하고 있어요\n나머지 시간은 게임을하던 티비,핸드폰보던 술을먹던 한마디도 안해요~더 놀으라고ㅋㅋㅋ'], ['나머지 시간이 있나요???남편분 대단하세요 ㅠㅠ'], ['하루에 다하는게아니고 하루는 청소 하루는 빨래자기가 정해놓고 하던데요??자꾸 해달라고하세요ㅜㅜ'], ['저희는 똑같이 지내요. 대신 요리는 항상 제가햇엇는데 이젠 요리할때 남편이 도와줘요. 요리하느라 오래 서잇기 힘들어서요. 그리고 남편 출근할때 항상 아침 챙겨줫엇는데(저는 재택근무) 이젠 안챙겨줘요. 12시까지 푹자려구요. 남편도 제가 마니자고 마니먹길원하구여'], ['임신전엔 제가 집에서 노니 다 했구요\n임신하고 난 뒤에는 입덧+두통으로 남편이 다 하다가 요즘은 약먹고 좀 살만해서 조금씩 슬금슬금 하고있어요'], ['원래도 남편이 밥자주 하고 옷 필요해지면 빨래 돌리고 그럼 제가 건조기 돌리고 서로서로 눈에 보이는거 청소기 돌리고 분리수거날은 같이 나갔다 오고 그랬는데 아기 생기고는 남편이 다 해요. 과일이나 간식도 챙겨주고.. 저는 입덧이 너무 심해서 그냥 눈뜨고 살아 있는것만으로도 힘들어해서 제가 토하고 입덧하면 손목이나 등 눌러주고.. 임신알자마자 초반에 맘똑티비 같이 보면서 임신 후 잠이 많아지고 체력 떨어지고 한다고 하는 영상 입덧영상 같이 봤어요. 지금부터라도 남편분이 많이 도와주셔야 아기 육아할때도 분담이 어느정도 될건데.. 남편분께서 임신후에 체력 떨어지는걸 모르시나봐요. 힘들면 힘든티 내야해요. 말안해주면 모를거예요.\n참, 초반에만 좀 도와달라고 안정기 오고 좀 움직일수 있어지면 내가 많이 할께~ 고마워 여보 ~ 하고 얘기했었네요. 남편 고생하는게 미안해서..'], ['전 입덧때문에 시체처럼있을때 남편이 요리, 첫째꺼 요리, 설거지, 빨래널기 해줬어요.. 그리고 청소해주는 아주머니 한달에 한번 부르구여. 그래도 제가 컨디션이괜찮을땐 최대한 설거지, 옷개우는것 애기 보는거 하려고했어요.. 커뮤니케이션많이해보세용 솔직한 이야기'], ['저 임신했을땐 집안일 평소보다 쫌더 안하는정도?집안일이 많이 힘들진 않았어요 운동삼아 하는정도라서요 근데 신랑이 왜 안하냐고하면 그건 또 기분 안좋을꺼같아요 ㅠ 말이라도 내가할게 쉬어~ 해줘야죠 ㅎ'], ['저도 원래하던것보단 조금줄여서 하고있어요ㅠ 음식냄새못맡으니 주방관련일은 아예손못대구요~ 그래도 15주 아직까진 남편이 나서서 많이 도와주긴하네요'], ['입덧땜에 한달넘게는 요리빼고는 다해줬고 지금은입덧 없어져서 그전처럼 집안일 제가 거의 다해요 \n청소기랑 화장실청소 쓰레기 버리는거는 신랑이 하구요'], ['넘 힘들어서 일하고 집에 드가면 아무것도 하기싫어서 누워있고 어느새 혼자 조용히 설거지하고있어요 자기도 일하고 이제 왔으면서ㅡ막 밥을 잘 차려주는건 아니지만 밥먹고 치우는거도 자기가 한다하고 아~ 음쓰 내일 버려야겠다 이러면서 빨래 개는거도 자기가 할수있을때 시간 조정해서 하고그렇게 해줘도 입덧땜에 힘든데요 ㅠ 나 혼자 다 겪는데'], ['저도 초기에 엄청 불안하다고 유산 가능성 있다고 하셔서 눕눕만 해서 집안일은 남편이 하고 있어요 재택하는 날이 있어서 아직까지는 수월하게 하고 있네요'], ['저희는 남편이 전담해요. 전 빨래 접는거 정도? 이것마저도 제자리 정리는 남편이... 초기에 하혈을 한번 하기도했지만 그 전부터 자기가 하면 덜 힘드니까 자기가 하는게 맞다고 하더라고요. 전 컨디션 좋으면 국 끓이거나 설거지할 때 옆에 서서 그릇정리하고.. 나름 무리되지 않는 선에서 같이 하려고 노력은 해요. 니가 많이하네 내가 많이하네 이런 생각을 떠나서 서로 생각해주면서 한발짝씩 양보해야 하는거 같아요ㅜ'], ['평소랑 똑같이해요. 일주일에 두번 출근하는 일 하구요. 평소에도 분리수거, 음쓰버리기, 쓰레기봉투 버리기는 남편이 했던거고.. 그냥 날 잡아서 하루는 청소 하루는 설거지 하루는 빨래 뭐 이렇게 돌아가면서 하는 중이에요.'], ['남편이 다 해요. 본인이 한다고 누워서 쉬라 해요. 가끔 귀찮아서 설거지 같은거 미룰때 있는데(전 자기전에 쌓여있는거 못보는 스타일) 그걸 왜 안하냐고 닥달하진 않아요. 남편이 하는거 제 성에 안차지만 저사람도 내 기준 맞추는게 힘들 수 있단 생각으로 항상 고맙다는 마음만 전해요. 저도 컨디션 좋고 기분 좋으면 좀 더 하려고 하고요. 그냥 서로 저 사람도 힘들겠구나 라는 마음으로 고맙다 표현하며 하면 덜 서운할 것 같아요! 가끔 제 기준에 너무 거슬리는게 있으면 부탁해요. 여보 미안한데 오늘은 이거(설거지, 분리수거 등등) 좀 나 대신 해주면 안될까? 라고요. 부탁하는데 싫어!! 하는 사람 드물지 않을까요ㅠ'], ['강아지 산책, 설거지, 밥, 반찬, 청소기, 분리수거, 음쓰, 빨래 등 남편이 전부 다 해주고 있어요. 저는 전업인데 임신 전에도 남편이 주로 하는 편이었고, 임신 후에는 아예 전적으로 맡아서 해주네요. 알아서 해주니 너무 고맙게 생각하고 있어요.. ㅠㅠ'], ['임신소식과함께 식기세척기랑 로봇청소기부터 샀어요결혼초부터 어짜피 집안일도 거의 신랑이 다하는데 설겆이는 제가 낮에 잔뜩먹고 담궈놓으면 퇴근후 저녁 잠들기전여 신랑이 돌려놓고자요청소기도 오후2시 타임을 맞춰놔서 알아서 청소해주니 할게없어요..'], ['저는 초기부터 피고임/ 피비침 있고, 입덧때문에 눕눕하느라 2달째 남편이 거의다해요..입덧 좀 괜찮아지면서 어제 첨으로 요리했네요. 근데 저희남편도 툴툴대면서 하긴해요..ㅋ'], ['임신전 : 쓰레기비우기, 빨래, 청소기돌리기 신랑\n임신후 : 전부신랑 .. 이 하고있어요.\n입덧도 있었고, 비염으로 힘들어하니 밥은 밖에서 샌드위치, 냉면 등등 포장해서 사다주네요.. (첫째때는 아무것도 안했어요)'], ['우선 저희는 맞벌이구요\n제 생각에 남편의 집안일 분담은 사랑의 크기보다는 남편의 타고난 성격이 더 좌우하는 것 같아요 \n저희 남편은 성격 자체가 워낙에 깔끔하고 더러운 걸 못참아서 본인이 다 하는 스타일이예요 ㅎㅎ \n저는 오히려 반대로 좀 어지럽혀져도 더러워도 그러려니 하구요\n원래 임신 전에도 요리 빼고는 남편이 다 했었고 \n임신 한 후로는 가끔 요리도 해주네요 \n근데 그러면서 잔소리 하면 그건 좀 최악인데 다행히 잔소리를 안하는 스타일 ㅋㅋ\n대신 전 맘껏 생색낼 수 있게 항상 입에 침이 마르게 칭찬해줘요 깔끔한 남편 덕분에 내가 호강하면서 산다 \n난 참 남편 복 하나는 타고났다 \n어쩜 이렇게 깔끔하게 잘 하냐 등등등 \n암튼 타고난 성격이 정말 중요한 것 같아요'], ['저도 지금12주 맞벌이요! 아무것도 안하는 중이에요 ㅎㅅㅎ 딩굴딩굴'], ['전 남편이 요리만 해주고 나머진 다 제가 해요 요리해주면서도 생색내는뎁??ㅋㅋㅋㅋ'], ['네 100퍼 다 해줘요~'], ['저는 원래 청소잘안해욬ㅋㄲㅋㅋㅋ그대신 요리 설거지는 좋아해서제가하는편이었는데 임신하구 손에물묻히는게싫어서 설거지도 안하는중이에요....ㅜ만사다귀찮....청소하는건 진짜 성격인거같아요ㅎㅎㅎㅎ'], ['저도 맞벌인데 세제에 입덧을 해서 초반엔 집안일 아예 안했어요ㅎㅎ 물론 입덧 없어진 지금은 설거지 빨래는 제가 해요ㅎㅎ엄살도 부리시고 아픈척도 하고 그러세요ㅠㅠ 지금 못누리면 평생 못누린다더라구요ㅠㅠ 남자들은 옆에서 계속 아프다 힘들다 해야지 알아먹어요ㅠ 안그럼 애기가 그냥 저절로 막 크는줄 알더라구요ㅠㅠ'], ['컨디션이 안좋아서 요리 설거지 청소 빨래 다 신랑이해주고있어요 ㅜㅜ 전업인데도요~ 얼른 컨디션이 돌아오길 기다리고있어요'], ['전업입니다\n 초기에 절박유산 진단받고ㅜ 요리외에는 남편이 전담하고 있어요 첫애라 남편이 걱정이 많아서;; 인제 안정기들어가니 저도 분담해서 해야죠ㅜㅜ'], ['전 12주인데 맞벌이라 ㅠㅠ 넘 힘들더라구요 아직까진 90프로는 남편이 다 해요 첫째 케어부터 ㅜㅜㅜ ㅎㅎ 그래도 엄마만 찾으니 ㅠㅠ'], ['초반에 하혈 한달동안 했었고 입덧때문에 너무 힘들어하니 100프로 남편이 다 해줘요 \n임신전에도 잘 해주는 편이었는데 이제는 밥차리고 치우고 설거지에 청소에...자기전에 마사지까지해줘요\n미안해서 매일 미안하고 고맙다고 얘기하네요\n어서 입덧끝나고 제가 할 수 있는날이 왔으면 좋겠어요\n'], ['집안일(요리 청소 분리수거 다포함)담당 제가하고 육아담당 남편 해용'], ['14주인데 입덧이 있는지라 남편이 백프로 다 해줘요저희는 외벌이인데 제가 응급실도 가고 이벤트가 있던지라 걱정되는지 아무것도 하지 말라고 본인이 더 불안하다며 다 전담하고 있어요임신하면 얼마나 힘든지 아직 몰라서 그러시는거 같아요 얼마나 아프고 불편한지 얘기 안하면 잘 모르는거 같아요 남자들은ㅜㅜ'], ['저는 극초기지만 안해요. 너무 어렵게 가진 아가기도 하고, 맞벌이만으로 벅차서요. 임신전엔 집안일 하나도 안하는 사람 막 시키고 있어요. 성에 차진 않지만...ㅋㅋㅋ빨래널기랑 가끔 청소기밀기 정도하는거 같아요.'], ['남편분 임산부체험 해보라하셔요 임산부에게 운동겸해서 집안일하라 하지만 임산부 운동과 집안일은 달라요 둘째때 무리하다가 몇년지나 몸아프다고 신호와요 \n충분히 대화하시고 나눌수 있는 부분은 나누셔야죠'], ['임신전에도 남편이 많이 했는데 임신하고나서 특히 입덧 생긴뒤로는 남편이 다 하는거같아요 \n입덧 없어지면 조금씩 도와서 하려구요'], ['전...임덧이 미친듯이 심해서... 남편이 독박 집안일중이에요..가끔 남편의 잔심부름 정도만^^냉장고에서 맥주갖다주기 이런거만..입덧안하고 집안일 반반하고싶어요..']]
    
    9017
    집안일 중 가장 하기 싫은게 뭘까요? 명전전 이사예정이라 집정리를 해야 하는데직장맘이기도하고나이도 반백이다보니 다 귀찮네요.정리정돈 좋아하는데지난주 신발장정리했고 이번주말 주방할까 예정중인데옷장정리는 왜이리 하기 싫은걸까요.?회원님들은 집안일중 뭐가제일 싫으세요?
    
    [['저는 정리하는것도 음식하는것도 좋아하는데\n요즘은 음식하는게 지겹네요 지긋지긋 ㅋ'], ['음식은 맘먹음 바로하기도하고\n시켜먹거나 외식이 가능해서요.ㅎ'], ['주방 후드 필터청소요~'], ['이건\n더럽기전에 미리미리하는게\n힘이 덜 들어요'], ['네~밥해먹을 때마다 미리 하기가 너무~구찮아요~'], ['저는 산더미 같은 빨래 게고서 각 각 자리에 갖다 놓는게 제일 귀찮아요...ㅜㅠㅠㅠㅠㅠ'], ['저두요'], ['그건 그래요.ㅎ\n그래도\n옷장정리보단\n빨리끝난다는거~~'], ['전 그래서 개기만 하고 애들 시킵니다^^;;;;; 아직 초등저학년이라 시키면 하더라구요~ㅎㅎ'], ['전 5,6세 애들 옛날부터 시켜요ㅠㅠㅋㅋ 자기의 일은 스스로 하자~~~ 알아서 척척척 스스로 어린이~~!! 이 노래불러주면서요ㅠㅠㅋㅋㅋㅋ 둘째는 뺀질거리고 잘 안하지만요 ㅋㅋ'], ['바닥 닦기요ㅜㅜ'], ['스팀청소기로 쓱싹~~\n집안일 할거 참 많죠^^'], ['빨래개는거 제일 안좋아해서 신랑한테 넘겼어요.'], ['잘하셨어요~~'], ['설거지요..혼자 벽보고 있는 시간이 지루하고 힘들어요..^^;;'], ['전 핸폰으로 라디오켜놓고\n주방일보니\n좀 나은거같아요'], ['빨래랑 설거지요 그나마 건조기 식세기가 있으니 낫네요'], ['식세기가 있으니\n좀 나으실거같아요'], ['전 빨래게는거..요..  집안일은 모두 재미없고 지루하네요..ㅠㅠ'], ['그러게요.해도 표시도안나고^^;;'], ['맞아요.안하면 안한 티 팍팍나고... 하면 한 티는 전혀 안나고..ㅠ 그래도 생활서 중요한 일들이니ㅠ 오늘도 화이팅해보아요.'], ['베란다 청소요 ㅠㅠ'], ['전 볘란다에 화초들이 많아 돌보느라 베란다좋아요'], ['부러워요~~ 정말 부지런하세요^^'], ['저는 장난감...이요 ㅠ 정리하고 뒤돌면 다시 꺼내놔요...ㅠㅠㅠㅠㅠ'], ['아이키우면서 정리안된거 너무 싫어 바로바로정리했는데\n아이한테\n별로 좋지않다네요.\n맘껏 어지르면서 놀라하고\n잠들면 한번만 밤에 정리하세요.ㅎ'], ['아 그래요? ㅎㅎ 그래야겠네요. 맨날 정리하라고 혼내내요; ㅎ'], ['설거지요.. ㅠㅜ 넘 찝찝..'], ['한번씩 간혹 배달음식으로^^;;'], ['빨래개서 각자자리에 넣는거 전 그게 글케싫드라구요 ㅎ 알아서들갖고감 좋겠어요. 이방저방 갖다놔야하니 ㅜ'], ['남편 길들이기하보세요~~'], ['다 싫지만 남편돌보기요ㅋㅋ'], ['죄송한데... 지나가다 댓 보고 빵 터졌어요ㅎㅎㅎ'], ['이건\n평생이라잖아요~'], ['전 냉장고 정리... ㅎㅎㅎㅎㅎ'], ['맞아요.ㅎ\n잘먹는집이면 좋은데\n저희도 그리 잘 먹는집이 아니다보니\n냉동실에 음식이 많아요'], ['빨래개기.....며칠째 건조기에 잇어요  오늘가서 꼭하려구요ㅋㅋ'], ['ㅋㅋ 전 쇼파위에 산처럼 쌓여있네요. 오늘은 꼭 정리해서 넣어야겠어요'], ['에궁.구겨짐이 있어서\n언능꺼내세요'], ['ㅎㅎ 333'], ['다요'], ['네\n맞아요.ㅎ'], ['빨래개서 제자리놓는 거요.ㅋㅋ'], ['의외로 제자리갖다놓기 많이들 싫어하시네요'], ['화장실청소요...^^'], ['샤워할때 하면 좀 시간절약?되는듯해요'], ['샤워할땐 샤워만 즐기고싶어요 ㅜㅋㅋ'], ['화장실청소랑 싱크대 배수구 청소요. T.T'], ['네\n다 귀찮은곳이죠'], ['정리정돈이요^^ 제일 싫어요~~~~~'], ['누가좀 해줬음 좋겠어요'], ['다요 다 하기싫어요!!!!ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['맞습니다.\n다 하기 귀찮죠.ㅎ'], ['무조건 밥이죠\n세끼다사먹고싶어요'], ['주부는\n누가\n반찬한가지에\n밥상만차려줘도\n감사하죠~'], ['빨래 개기랑 넣는거요 ㅎㅎㅎ 이틀삼일에 한번 개놔도 서랍에 또 안넣을때도 있어요 세상귀찮 ㅡ.ㅡ'], ['그렇죠\n의외로\n귀찮아요'], ['요리 빼고 다..요~먹는거에 온 정성과 진심을 다하는 스타일이라서요..ㅎㅎ'], ['그럼\n건강하실거예요.\n잘먹음 좋다잖아요'], ['저는 한깔끔 떨어서 쓸고 닦고 청소 정리는 좋아해요 집꾸미는것도 좋아하고요 근데 와이셔츠 다리기는 귀찮아요 ㅎㅎㅎ'], ['간혹\n세탁소에 맡기시고\n그시간에 좀 여유로운 개인시간 가지세요'], ['청소랑옷정리,.넘귀찮아요!\n그나마,.요리,빨래..설거지는..괜찮아요!^^'], ['옷정리가 정말 많이 귀찮아하는 일이네요~'], ['빨래개는거요ㅜㅜ'], ['ㅎ\n옷을 입지 않고 지내기?ㅎ\n맛점하세요'], ['창틀청소요ㅠㅠ!!!']]
    
    9066
    20) 집안일 고고 ?세탁기 돌아가는 동안 훌라후프 돌리면서댓글 좀 달렷습니다 ㅎㅎ이제 100개 좀 안되게 남앗는데 세탁기 거의 끝나가니 건조기 넣어주고미먼 좋으니 환기하면서 청소도 좀 할게요~~이따 또 오겟습니당 ㅎㅎ
    
    [['와우 집안일 하시고 오셔요 히히'], ['후딱 하고 마트 다녀옵니당 ㅎㅎ'], ['저도 오후에 집앞 마트 다녀왔지요'], ['맛난거 사와서 저녁 잘해주셧나요 ㅎㅎ'], ['3 그냥 집에 야채같은거 사왔어요 ㅎㅎ'], ['야채는 항상 잇어줘야지오'], ['4 맞아요 식재료 마트이런곳 갔다와야겠어요'], [': 저희는 이사하고 식자재마트가 좀 멀어져서 속상해요 ㅠㅠ'], ['5 거기 괜찮다고 해서 가볼려구요'], ['5) 가셔서 잔뜩 득템해ㅇㅎ시길요 ㅎㅎ'], ['아히히 오늘도 엄청 빨리 끝나시는군요 저도 곧 따라 가겠습니다'], ['애들 오기전에 끝내야겟어요 ㅎㅎ'], ['넵 손가락에 모터를 달고 또 달려 보아요'], ['넵 지금 바짝 집중하는중입니다 ㅎㅎ'], ['아히히 또 열심히 다려 봐아요'], ['네 손에 모터달고 다다다 달려여죵'], ['넵 저도 최선을 다해서 달리는중이요'], [': 오늘도 열심히 해야지요 ㅎㅎ'], ['5 아히히 그래야죠 저는 아직 미션도 못끝냈어여'], ['5) 제.계획은.20일까지.달리고 끝인데.그럴수가 없어서 힘드네요 ㅠㅠㅋㅋ'], ['오호 !대박 오늘진짜빠르게 ㅋㅋ\n저는 댓제는못하고 나만이정해둔 갯수채우기 하그있어요'], ['오늘은 오전에도 바짝 집중한덕에 금방이네요 ㅎㅎ'], ['ㅎㅎ오전바짝집중해야되는데 크.. 그게안되네유'], ['아무래도 아이가 어리면 손이 많이 가니~'], ['잠잘때 해야되는데 어머님이 있다보니..덜덜'], ['그러다보면 좀 신경쓰이긴 하지요~~~'], ['계속폰붙잡고있기가 그러니ㅠ.ㅠ'], [': 맞아요 ㅠㅠ 그래도 많이 하셧지요??'], ['5  ㅎㅎㅎ기본정도만 하다가.. 한 300~400개.. 오늘은 좀 더하기는 하고있는것같아요'], ['5) 저는 곧 애들 올시간이라 육아하면서 천천히 달려볼게요 ㅋㅋ'], ['으흐흐 저두이제끝이보입니다'], ['애들 오기전에 끝냅시다용 ㅋㅋㅋ'], ['오늘은쪼매늦게따라갈게유'], ['전 오늘도 분발해봐야지용'], ['으흐흐ㅋㅋ 분발같이해보자요'], ['얼마안남앗는데 시간이 촉박합니다!! ㅋㅋ'], ['저는한시간 단디달려볼게유'], [': 오늘도 거의 다 하셧으려나용'], ['오늘도조기퇴근하시겟군요'], ['오늘은 애들 오기전에 가야지유'], ['조기퇴근성공하신겁니까'], ['네 어제는 성공 ㅎㅎ 오늘은 어떠려너요'], ['ㅎ오늘도조기퇴근하신거같아용'], ['어제 하원전 성공은 햇지요 ㅎㅎ'], ['와우~조기퇴근하실때너무좋죠'], ['댓글 다들 어떻게 그렇게 잘 다시는지 대단하세요~'], ['손에 모터달고 열심히 달리는게지요 ㅎㅎ'], ['ㅎㅎ모터~ 그 모터 저도 갖고 싶네요'], ['빌려드릴수 잇다면 드리고 싶구만요 ㅎㅎ'], ['ㅎㅎ그 모터 진짜 부럽네요~ 오늘도 화이팅이요^^'], ['넵 오늘은 퇴근전에 댓제 할수잇게 달려야지용'], ['오늘 댓제하시길 바래요~'], [': 얼마안남아서 애들오기전에 끝내야지요'], ['5댓제 잘 끝내셨었나요?'], ['5) 네 잘 끝내고 쉬다가 지금 왓어용'], ['오우!! 한숨자고 오시더니 빨리 하셨네요~'], ['자기전에 좀 달려서 더 빨리 끝낸듯요 ㅎㅎ'], ['와 쑤남매맘님 매번 진짜 빠르셔요^^'], ['이제 천천히 달리려구요 ㅎㅎ']]
    
    9135
    애들 살림 시키시는 분 계신가요? 즤 큰애가 초딩 고학년이에요그래서 집에 있음 심심도 하고 엄빠도 도와주고 집안일도 배우고 이런저런 이유로 전기밥솥에 밥하기, 수건 개기와 정리, 자기옷 개서 정리하기, 쓰레기 버리기, 가끔은 세탁기 빨래 건조기 넣고 돌리기, 자기 방청소, 식사시간에 숟가락 놓고 반찬 옮기기 이런걸 시켜요,, 심지어 후다닥 잘 해요,, 아빠도 집안일 잘 합니다,,근데,, 하루는 이런거 하는 친구는 없다며,, 나만 한다고,, 왜 해야하냐고,, 그러더라구요,,?정말 이런거하는 친구들 없나요?제가 너무 많이 시키나요? ㅜ ㅜ?
    
    [['6살 초1도시켜요~걸레질하기,수건개기, 먹은그릇갖다놓기, 수저놓기 다잘해요ㅋㅋ'], ['6살 초1???? 그쵸!! 자연스레 다 하는거죠?'], ['네~~저 아는지인은 애가셋인데 밥먹고 안치우는건식당에서 밥값내고먹을때만그러는거라고... 설거지통에안갖다놓으면 돈받는대요;;;ㅋ   같이 캠핑다니는가족인데 아이들이 엄청센스있고싹싹해요ㅋ 다른어른들도 다들예뻐하시더라구용ㅋㅋ그런게가정교육이라고생각해요ㅎㅎ'], ['식당 얘기 확 와닿습니다!!'], ['이제 3,5학년 올라가요 밥먹은 그릇은 무조건 자기가 치우고 작은애는 빨래 개는걸 잘 도와줘요 음식할때도 자기가 해보겠단말 자주하구요'], ['그쵸!! 한집에 다같이 사는거니 집안일도 다같이 하는거쥬~~'], ['저희아들들 중1초5인데 설거지 쓰레기분리수거 청소가 돌리기 다시켜요~~^^'], ['크~ 댓글 보여줘야겠어요!!'], ['5살도 시켜요. 먹은그릇 가져다놓기, 장난감 정리하기, 빨 래같이개기, 등등 할 수 있는 선에서 같이 하는데, 집안일은 같이사는 사람들끼리 다 같이 하는게 맞다는 생각에 꾸준히 시킬 생각입니다.'], ['맞습니다! 맞습니다!!'], ['저희 아들초6인데 분리수거.쓰레기 버리기.수건.자기속옷.옷정리.청소기 돌리기.수저 놓는것등 집안일 많이 돕고 가끔 라면.짜장라면도 잘 끓여줘요~ㅋ커피도 아주 맛나게 잘 타주고요~'], ['크~ 최고최고!!'], ['저도 종종 시켜요. 특히 5인가족이라 빨래가 많은데 그중 수건개기는 자주시키고 바쁠땐 설거지도시키고 그래요. 할수있을만한건 시켜야지 안시키면 커도 안하더라구요.'], ['맞습니다 맞습니다!! 집안일은 구성원 모두 함께!!'], ['우리 5살 , 초1도 시켜요. 청소기돌리기, 빨래개기, 수저놓기, 반찬놓기, 물떠놓기, 다 먹은건 물담궈서 놓기, 현관신발정리, 쓰레기 분리수거 등.. 아이들 방은 당연한거구요.. 제가 너무 시키나싶지만 가족의 구성원으로 전 당연하다고 생각해서 자연스럽게 함께했어요. 물론 신랑도요.'], ['그쵸그쵸!! 자연스러운거 맞습니다~!!'], ['공짜밥을 줄 생각이 없습니다ㅎㅎ 요즘같이 24시간 붙어있는 이때에.... ㅎㅎ'], ['저도 사람이면 밥값은 하고 살자고ㅎㅎ'], ['밥은 압력솥에하니 못시키고 나머지는 다 시켜요..어릴때부터 해서 애들이 당연히 해야하는걸로 알아요..'], ['그니까요~ 어릴때부터 시켰는데 요게 좀 컸다고 내가 왜 해야되냐고해서 당황했어요;;'], ['3살짜리한테도 제가 빨래갠거 중에 양말같은거 날라오라고 일부러 시켜여ㅋㅋㅋ함께 정리하고 집안일하는 습관되게요ㅎ'], ['크~ 예쁜 3살이네요~~♡'], ['저희도 초6 여자아이요 자기가 먹은 그릇은 자가가 닦고 음쓰버리기 자기방청소하기 빨래개기 시켜요 가끔 라면도 끓이고 계란후라이도 시켜요 곧잘해요 자주는 말고 가끔시키면 잘해요'], ['다들 이리 잘 하고 계신데 제가 아들말만 듣고 어머 내가 너무 시키나 괜히 걱정했어요ㅎㅎ'], ['4살딸에게도 수건개기,바닥닦기,자기먹은 그릇 싱크대에놓기,자기물건정리하기 시켜요.집안 일은 같이하는 개념이라는거 알려주고싶어서요~^^'], ['암요암요~ 같이하는거죠!!'], ['집안 살림 위험하지 않는 범위내에선 자꾸 시켜봐야 안다구 생각해서 저도 시키는편이에요~ㅎㅎ'], ['역시 행복한 형제님~!!'], ['6살 딸램 빨래개기, 수저밥그릇 갖다놓기, 자기전이나 청소할때 방정리, 가끔 설거지랑 쌀씻기 하는데     9살딸램은 잘안하려해요ㅎㅎ'], ['즤 아들한테 한줄기 빛같은 댓글이네요ㅋㅋ'], ['올해 인제10살이 된 저희 첫째도 8살때부터 세탁기 돌려주고 끝나면 꺼내서 건조기 넣어서 돌려달라고하면 돌려줘요 빨래 분류는 제가 다 해놓구요ㅋ \n7살 동생은 먹은거 설거지통에 식사 젤 마지막에 한 사람이 식탁 닦기  수건개주기등등 해요~ㅋ'], ['식탁 닦기도 추가해야겠어요!!'], ['초2 큰아들 라면도 끓이고 가끔 엄마 쉬라면서 혼자 후라이나 스팸구워서  7살동생 밥차려줘요ㅋ\n쓰레기버리는것도 동생이랑 같이 하구요\n명절때도 꼬지끼우기 같은거 애들 꼭 시켜요 어릴때부터 습관이 되야 어른되서도 하니까요'], ['대박!! 아이들 보기만해도 배부르시겠어요~~'], ['초6학년 5학년때부터 5개월된 아기 아기띠 분유타기 똥기저귀갈기 엄마없을때 아기 쭉 봐줬고 설거지 걸레질하기 청소기돌리기 다 해요 초6 남자애예요^^'], ['와 초대박!!! 엄청 든든하시겠어요!!!'], ['저도 밥솥에 밥하기 빼고 다시키는거 같아요저희집은 아이들이 수저 안갖다 놓으면밥이 안나와요~밥그릇 안담궈 두면 그그릇에 점심 준다고 해요 ㅋ'], ['ㅋㅋㅋㅋㅋ 저두 밥상 차릴때 안하면 알아서 떠먹으라고 하고 그냥 앉아버려요ㅋㅋ'], ['저두 초딩아들딸 빨래개기, 밥상치우기, 본인방 걸레질 등 시키는걸요~~^^'], ['한국의 미래가 밝습니다!!ㅎㅎ'], ['저희 중1딸 저없으면 김치볶음밥,간장계란밥해서 동생도챙기구 설거지도 가끔해요~  둘째 초5남자아이는 빨래 정리,분리수거 잘하구요~~'], ['듬직한 남매를 두셨네요^^'], ['초5남자 아이 중1여 고1남 \n밥먹고 설거지통에 가져다 놓기 청소기 분리수거 합니다 돌아가면서요 빨래 널고 개고 하고요 5학년 밥 넘맛나게 해요'], ['중, 고딩은 사춘기라 안할것 같은데 잘하네요!! 즤 아들도 초5때 밥 시켰는데 엄마보다 잘한다고 칭찬해줬더니 밥은 신나게해요ㅋㅋ'], ['댓글보니 더 시켜야겠네요ㅋㅋ 전 초1아이 수건개기 밥그릇갖다놓기 과일씻기 이정도만시켜요~'], ['저도 초등저학년땐 몇가지만 시켰던것 같아요ㅎㅎ 시키다보니 잘하는것 같아서 자꾸 추가ㅋㅋ'], ['다른집 아이가 잘하니 칭찬은 하기싫고 그리 말하는겁니다 ㅋㅋㅋ 함께 집안일하는거요 너무좋은거에요!'], ['아니 이럴수가!! 그런 속내가 있었군요ㅋㅋㅋ'], ['저희 아이도 같이해요~ \n일방적으로 시키는게 아니라 같이 하는거죠!!! 뭐라고물어봐서 그럼 엄만 왜해야하냐고 나도 안한다~~ 그랬어요ㅋ 못할땐 어쩔수없이 참고 해주고 도와줬지만 이젠같이 하는거고 니가 독립하면 혼자해야하는 일이라 알려줬어요 ^^ \n지금은 가끔 뺀돌거리긴하지만 잘해줘요~~ ㅎ'], ['아이들의 생각도 비슷하고 엄마들의 답변도 비슷비슷하네요ㅎㅎ'], ['8살 11살인데 아무리 시켜도 죽어라 안하려해요.외출하고 들어오면 옷정리. 다 먹은 밥그릇 치우기가지고 논 장난감 책 정리 갈아입은 옷 빨래통에 넣기그 정도 시키는데 항상 처음처럼 잔소리 해야되요ㅎㅎ ^^;; 너무 늦게 시켰나 싶기도 하구요댓글 보니 부럽네요.'], ['저도 오죽하면 이런 글을,,^^;; 아주 속 터져유~ 저게 결혼은 할수 있으려나 결혼하믄 와이프 속만 썪이겠구나 내가 평생 데리고 살아야되나 등등 저도 아직 철이 안들었는데  애들 사람 만들려니 저희집은 동물의 왕국이 따로 없어유~ㅋㅋ'], ['6학년 큰아들도 집안일잘해요(밥차리기 쓰레기버리기 청소기돌리기등등) 시켜서가아니라 습관처럼 자기일이라고 생각하고 해요ㅎ 밑에 여동생둘인데 오빠가 하니 자연스럽게 도와요 ^^'], ['우와!! 자기일이라고 생각한다니 대단해요!!??'], ['저희애들7살5살인데 자기가먹은밥그릇 설겆이통에넣기,수저세팅하기 흘린곳물티슈로 닦기.외투옷걸이에걸기 장난감정리하기 옷 빨래통에넣기 해요~~\n저흰아들들이여서 나중에 며느리편하라고 제가미리 교육중 이예요 ㅋㅋㅋ'], ['저도 며느리 생각하면 더 시켜야할것 같아요ㅋㅋ'], ['제가 워킹맘이라\n아들셋 집안일 도움받고 있습니다~\n음식은 엄마가 하고 \n아이들은 챙겨서 밥먹고 뒷정리,설거지,청소기돌리기,빨래널기,빨래접기,쓰레기버리기\n요즘 코로나로 집안에만 있으니\n운동겸 시키고 있어요ㅋㅋ'], ['맞아요! 격한 공감합니다!!^^'], ['26개월딸도 먹은그릇가져다두고 쓰레기통에 버리는거지만 버리고 재활용이나 쓰레기 버리로나가면 하나라도 들고나와요ㅋ대신 빨래는 개놓으면 다 풀어헤쳐요ㅠㅠ'], ['우와!! 완전 귀요미 기특한 딸래미네요^^ 젤 귀여울 나이에요~~'], ['젤 이쁜데 젤힘드네용ㅋ']]
    
    9142
    반찬만들었어요^^ 오늘도 친구와 함께 반찬만들어 냉장고 채웠네요~집에만 있으니 반찬이 훅훅 주네요...ㅜㅜ오늘 메뉴는 ????????????,????????????,????????????????,???????????????,???????????????????,????????????????,?????????????????,???????????????????????????????????????????????????,???????????????????,?????????????????????,???????? 만들었어요~~*장보고 후다닥하느라 쉬지도 못했네요ㅜㅜ오늘하루도 육아하시고 집안일 하시느라모든 맘님들 수고하셨어용~~~^^
    
    [['반찬가게 내세요!'], ['코로나끝나면 추친해보겠습니당~~~!!!!ㅎ'], ['늘 칭구야랑 만든 3가지씩 그릇보며 부러워 보고 있어요. 오늘도 다양하고 맛난 찬들 수고 많으셨어요. 가족과 맛나게 자시고 행복하고 따뜻한 겨울 보내세요.♡♡♡'], ['감사합니당~~~^^오늘도수고하셨어요~~~*'], ['맘님 글 볼때마다 \n반찬 같이하는 마음맞는 친구가 있어서 참 좋으시겠어요~~'], ['얼매나다행인지몰라용~공동육아도하고 반찬품앗이도하구요~~~^^*'], ['정말 친구가 최고의 선물이네요??????'], ['우와!!!! 엄지척!!'], ['엄지엄지척척척~~~!!^^'], ['우아...진짜 넘 대단하셔요!!!'], ['감사합니당~~~^^요똥도먹고살려니하게되네요~~~^^'], ['이건 반찬가게아닌가요?! 와 진짜 같이만드시는 친구분도 대단하시구 반찬들이 다 맛있게 보여요♡'], ['친구랑저랑 손이 좀 크다보니ㅋㅋㅋㅋ이렇네여ㅎㅎㅎ 칭찬감사합니당~~~^^'], ['이런 손은 크셔도 됩니다용 코로나때문에 반찬이 순식간에 사라지더라구요ㅜㅎ 고생하신만큼 맛나게드세요^^'], ['그니까요ㅜㅜ집밥을많이먹으니 반찬도 금방 먹더라구요ㅜㅜㅎㅎ오늘도수고하셨어요~~~*'], ['네~ 현수맘님도 고생많으셨어요\n편안한 밤 되시길 바래요~'], ['반찬들이 맛나보여요ㅡㅡ부럽습니다 저는  똥손이라ㅡㅡ맛도 없고  그래서 안만들어요ㅡㅡ\n'], ['저도요똥이엇는데 하다보니 쪼매씩 느네요~~~~^^'], ['나중에 한수 가르쳐 주세요 가족에게 이쁨 받고 싶어용'], ['가르칠...ㅜㅜㅋㅋㅋㅋ실력까지는 안되용...ㅜㅜㅎㅎ 네이@찾아보고 합니당ㅋㅋㅋㅋ저두용'], ['우와 진짜 대박이네요~ 맘님 진짜 월반찬 배달 창업하셔도 되겠어요^^'], ['감사합니당ㅎㅎㅎ 반찬가게를 내야할까봐요ㅋ'], ['반찬가게하셔도될듯 맘맞는친구와 같이만들면 더 맛나겠어요 대단하세용 ㅋ'], ['감사합니당ㅎㅎㅎ 함께할수있는 친구가 있어 너무 좋습니당ㅎ'], ['우와 대단하세요!'], ['감사합니당ㅎㅎㅎ엄지척척!!'], ['멸치랑 마늘쫑 같이 볶은건가요? 맛나겠어요~감자조림도 먹음직스러워 보이고 다 맛있게 보여요~'], ['네~~ 마늘쫑멸치볶음이에요ㅎㅎㅎ고구마간장조림이랍니당ㅎ'], ['와!!!!!?? ?? ??  저는하루한가지도 못해서요 부럽습니다'], ['첨엔 저희도 몇개 못만들었는데 계속하다보니 양도늘고 가짓수도 많아졌어요ㅎ'], ['컬리플라워 두부무침 맛나쥬~ 저도 얼마전 해먹었어욯ㅎ 맘님 반찬에 정보 얻어서 주말에 저도 반찬 해야겠네용~ㅋㅋ'], ['앗진짜요ㅎㅎㅎ색이 이뻐서ㅎ하나 사봤는뎅ㅎ 맛도 좋더라구요ㅎ'], ['우와...무슨 일이래요..!후다닥 금손 맘님 부러워용ㅈ!'], ['감사합니당ㅎㅎ'], ['헐 대박 친구분이랑 큰일하셨네요\n만약 이런 소모임이 있다면 들고싶네요ㅜ'], ['네ㅜㅜㅎㅎ 힘들었어요...반나절을 주방에 서 있었더니...ㅠㅠ'], ['매번 보는데 친구분이랑 이렇게 같이 반찬 만드는거 너무 보기 좋아요\n솜씨도 너무 좋으시네요^^'], ['감사합니당~~~^^*'], ['그 친구 제가 할게요????\u200d♀?????\u200d♀?????\u200d♀? 맘님 반찬사진 올릴때마다 똥손엄마인게 미안해집니다ㅠㅠㅋㅋ'], ['ㅎㅎㅎ 할수있어요!!도오오오전~~~**'], ['이번에도 역시~~!! ??????\n맘님글 구독하고 보는 1인입니다~~~^^'], ['이런영광이....>_<!!ㅎㅎ부끄럽네요ㅎ'], ['캬~ 주문하고 싶은 충동이... 너무 맛있어보여요~~~'], ['맛은보장못합ㄴ ㅣ 다...ㅋㅋㅋㅋㅋ감사합니당ㅎ'], ['맘님 이리 반찬하면 배달음식은 잘 안드세요??'], ['네~ㅎㅎ주말엔 한번씩 시켜먹긴하지만 평일엔 집밥으로 먹어용ㅋ'], ['우와 사먹고싶어요!'], ['반찬가게 하면 사러오세요ㅋㅋㅋㅋㅋ'], ['반찬거리 고민이였는데 덕분에 참고합니다^^'], ['저도 메뉴좀 공유해주세용..3년정도하니 메뉴가 고갈되어 가고 있어용...ㅠㅠ'], ['진짜 반찬집하세요 볼때마다 감탄하고 가요 솜씨가 대단하세요 두분'], ['감사합니당~~~~^^'], ['반찬가게 하셔도 될듯해요~~^^'], ['엄지척척척척~~~^^추진해볼께용~~~~~']]
    
    9199
    잔소리터지는데 자기가 집안일 다하는 남편vs 입다물고 있는데 집안일은 열번말해야 하는 남편 집에 수저숫자도 다알고 있고 식기 구성이며 애키우는 방식이며 청소하는 방법전부 잔소리 터지는데 자기가 집안일 다하는 남편vs도배장판을 새로해도 잘 못알아보고 조용히 팥으로 메주를 쒀도 입다물고 있는데 집안일은 열번말해야 하는 남편?둘다 쉣이지만 그나마 누가 나을까요?ㅋㅋㅋㅋㅋ
    
    [['1번요ㅋㅋㅋ \n참고로 전 ...열번 말 안할것같아서요 ㅋㅋ\n'], ['1번 ㅎㅎㅎ'], ['저는 전문직에 한달3천버는 남편분이\n부인 집안일하는게 맘에 안들어서\n본인이 설거지.청소하는건 봤네요ㅋㅋ\n부인은 정작 그러려니ㅋ\n본인이 잔소리하면서 다한대요ㅋ\n그럼 1번인가요ㅋㅋ'], ['제 친구네요ㅋㅋ\n남편이 하다못해 한식조리사자격증도 따서 밥,청소 다해준데요ㅋㅋ\n그런데 잔소리는 하나 친구는 마냥 편함ㅋ'], ['총체적 난국이네요 그래도 2?'], ['전 1이요지금 2인 남자랑 살고 있어서 ㅎㅎ'], ['1번에  한달에한번이나 대청소 같이하는정도 하면서 잔소리는 또 얼마나해대는지 애들머리부터 학원까지 잔소리'], ['1번하곤 못살아요.귀에서 피나고 울화통 터질듯요.\n내가 다하고 2번이랑 살래요.\n어차피 설거지는 식세가 청소는 청소기가...등등\n하니께요.'], ['111¹11'], ['2번이요\n제가 잔소리나 싫은소리를 못견디는 편이라..'], ['집안일 안하다가 잔소리를 해서 잔소리하지말고 니가 하라했더니 그때부터 집안 청소,설겆이를 해대며 자꾸 잔소리를ㅡ..ㅡ;;\n아니.. 식세기 돌린다니까 왜자꾸 설겆이를 해대는지...\n그냥.. 잔소리 안하고 적당히 했음하네요..ㅠ'], ['전 2번이요 제 페이스 대로 하고 사는게 좋아요 현실은 1번 ㅠㅠ'], ['ㅋㅋ 전2하고 사는데 요새 노력하는듯합니다.근데 주위에 잔소리겁나하는 2  남편봤어요.ㅋㅋ'], ['1번과 살고있습니다.한귀로 듣고 한귀로 흘리면 정신건강 몸건강합니다. 입으로 떠들지언정 어쨋든 다해주니 편해요.'], ['푸하하 너무웃겨요'], ['저흰 2번인데.. 1번이 더 나아 보여요 ㅠ'], ['둘다싫어요ㅎㅎㅎ'], ['우리집은 1번인데 좀 약간 과라.. 그냥 제 몸 편해서 좋아요 ㅋㅋ'], ['저도요 ㅋㅋ'], ['2번이요. 그래도 10번 말하면 하기는 하니까요.\n딸둘 키우며 2번과 살고있어요.'], ['1번이요. \n1번이면서 행동은 2번인 남편과 살아요.\n차라리 1번이 나아요'], ['ㅋㅋㅋ 저희도 딱어쩌다하면서 잔소리. 설거지 하면서 한숨이때다 싶어 쓰레기 버리고오라하면 다음에 한다고 미루기. 나도 한번 여유좀 부리고 싶어 설겆이 할때 커피마셨더니 또 꿍시렁~~~ㅎ'], ['전1번 개인적으로 잔소리만하는 부류 싫어요. 잔소리하면서 자기가 하면  인정ㅋㅋㅋㅋㅋ \n열번말하면 하나 해주는  사람은  제가  지칠 것같아요.'], ['1이요\n전 늘 한귀로 흘릴수있다요ㅋㅋ'], ['무조건1111 이요~ 듣기싫은 소린 흘려버리면 그만이죠'], ['저희남편1이요..잔소리에 마빡터집니다..어차피 지가할꺼 잔소리는왜하는지 .. 잔소리도 듣다보면 자존감 떨어져요 ㅜ 제발 말안하고 , 안했음 좋겠어요'], ['1번남자와 살고있어요ㅎ 잔소리하는게 넘 싫어서 한번씩 굳이 일부러 남편 흠 잡아 저도 잔소리하면(너도 이런 사소한 잔소리 듣기 싫으면 나한테도 하지말라고) 이런 잔소리는 언제든 환영이라고 하네요 -.-;; 약오르고 밉지만 그래도 잔소리하는 만큼 집안일도 많이 해서 같이 살아준다지요. ㅎㅎ 굳이 선택하자면 2번보단 1번이요!'], ['ㅠㅠ 1번과 살고싶어요 2번이랑 사는데 몸이 늙고 안아픈곳이 없어요'], ['2번이요~ 남편이 1번이라 냉장고 다 뒤지고 열어가며 잔소리ㅠ 근데 치우는건 제가 해야해요ㅋㅋㅋ'], ['1이죠'], ['저희 신랑이 2번에 주는대로 먹고 주는대로 입는 스타일에 겨울에도 반팔티 주면 반팔티 입는사람이었는데 얼마전에 무슨 심경의 변화로 집청소를 시작하더니 진짜..\n잔소리때문에 짜증나서 살수가없어요..\n1도 안도와줘도되니까 간섭없는 남편이 나은듯요..'], ['ㅋㅋㅋ집안일은 1인데 신경은 2인 분과 사는데 세상 편합니다'], ['잔소리는 해야 맛이지.. 듣는건 쫌 아닌듯'], ['와 잔소리하면서 저 불러서 시키는 남편과 살아요 스트레스받아죽어요'], ['1번이요 제가 1번인데 신혼때는 짜증났으나 애둘키우는 지금은 도움되네요ㅋㅋ잔소리는 흘려들으니 살만해요ㅋㅋㅋ'], ['1번이요 ㅋㅋ \n애없다면 2번도 살아볼만 할거같은데 애들있음 무조겅 1번이죠 ㅋ'], ['저희 신랑이 1번ㅎㅎ\n결혼하고 1~2년은 듣기싫어 힘들었지만 시간지나니 적응되고 남편다루는 (?) 요령이 생겨서 편해요.성격이 꼼꼼해서 애도 저보다 더 잘봐요'], ['1번이요\n내몸이 편하다면  잔소리쯤이야 그리고 남편이니 받아치죠 뭐 ㅋ'], ['전 1번요~^^'], ['저도 1요\n내몸도 편해지고 싶네요ㅋ'], ['2번이요\nㅇ\n시아버지 1번 스탈이신데\n다녀오면 귀가 멍해요'], ['친구남편이1번에 가까운데 친구가 이혼 생각 여러번했다더라구요 요즘 그집남편이 회사에 일도 없어서 출근도 안해서 더 미치겠다고...저도봤는데 집에딱들어오면 모든물건에 각 잡고 본인이 뭔가스트레스받고 짜증나면 냉장고부터 다 엎고 그담 수납장...저는 차라리2번이 낫네요 저희남편이 약간2번같은 스타일인데 저는 제가 해야 속이편해서 2번이 낫네요'], ['전1번 행동파가 좋아요 \n2번이 더 열받고 천불날듯'], ['2번이랑 살아요.. 남자잔소리 짜증날것같아요.. 그냥 쭉~~ 제가하면서 제맘대로 하고 살래요^^'], ['잔소리 부스터 장착하고 가끔하는..??????'], ['1번은 정신적으로, 2번은 정신적+육체적으로 다 힘드니까요. 최악은 입으로 떠들면서 손까딱 안하는 사람이네요.'], ['1 번과삽니다 ... 2 번이 나아요 스트레스 장난 아니에오'], ['저 혼날까요 1번이랑 살아요.\n\n쌉무시하고 당연한척 ㅎㅎ'], ['2번과 사는데 요즘은 설겆이 정도는 늘 스스로해요 좀더살면 스스로 하는게 늘어날것 같긴해요\n2번이랑 살아봤으니 1번이랑 살아보고 싶긴한데 잔소리를 견딜수있을지 모르겠네요ㅋㅋ'], ['1번 잔소리 제 몸 편하다면참을 수 있어요~~'], ['겪어보면 1이 더 피곤할것같아요;;;; \n고르라면 2번 내가 하고말지..'], ['저기.. 잔소리는 터지는데, 정작 본인은 안하는 사람과 살고 있는 사람도 있습니다;; ㅜㅜ'], ['1번이요!!!\n잔소리들을래요'], ['2번요. 제 마음대로 할 수 있어서 좋아요. 첨엔 화나기도 하고 그랬는데 자꾸 시키니 본인이 하는 일도 점점 늘더라구요~'], ['잔소리 한귀로 듣고 흘릴수 있습니다ㅎㅎ'], ['2번요. 10번말하면 하자나요. 1번이랑 살려면 여자가 무조건 일해야지 싶어요.'], ['1번 남편과살고있습니다 신혼초 잔소리하지말라고 모라했더니 혼자 삭히는건지 잔소리 이제 거의안하고 본인이 스스로 열심히 집안일해요ㅡㅡㅋ첨엔 제가할일 남편이하니 스트레스였는데 이젠 제가 더시키게되는..ㅎㅎ 행동파라 시킴 바로바로 합니다ㅎ'], ['1번요..2번이랑 살고 있는데 짜증나서 화내기 일쑤인데 본인은 몰라요'], ['저도 1번요..\n2번인 남자랑 살고잇어요'], ['1번이 나을것같아요. 한귀로듣고 한귀로 흘리고 몸편하게 살고프네요 ㅎ'], ['1번 남자랑 살고있네요 \n전 덜렁이라ㅋㅋ\n잔소리 심하지만 집안일 안해줘도 \n잔소리는 할것같은 성격이라\n네네~ 하면서 맞춰드리고 살고있습니다\n 전 좋아요ㅋㅋ'], ['1번이랑 살고 있는데 청소뿐 아니라 빨래,건조,밥까지 다해줘서 편하네요,,ㅎㅎ\n집도 엄청 깨끗해서 오는 사람마다 호텔이냐고 할 정도네요\n쉬지않고 열일하는 남편덕에 전 책이랑 영화보며 지내요'], ['잔소리없는 1번과 살고싶네요ㅋㅋ현실은 노력하시는 2번?'], ['2번요 1번 이랑 사는데..잔소리 못견디고 살림 노력해도 그 만족을 못시켜요 ㅋㅋ'], ['1번요 한귀로 듣고 한귀로 흘릴래요\n2번이랑 사는데 항상 혈압올라요'], ['하 전 1번이랑 사는데 스트레스 ....가.......... 전 2번 택할랍니다']]
    
    9291
    집안일 빨래개고 손수건 빤거 널고수건돌리고 젖병닦고 열탕소독 끝!오늘은 쏘야두해야되고설거지.빨래널기도..지율이가 아침부터 책을 다 꺼내놔서다시 정리해야되는데??오늘은 할게 좀 많네요??
    
    [['젖병소독열탕으루 꼭 다하세요??'], ['지금 소독기 고장낫어여ㅠㅠ엊그제갑자기안되서저나햇더니 as해야지만된다고하더라구여ㅠㅠ'], ['언제 쉬시는거에요?ㅎ 계속 바쁘신것 같아요'], ['ㅋㅋㅋㅋㅋㅋㅋㅋ남편퇴근하면서 저도좀쉽니다!!!!!'], ['읭읭~~ 할꺼많은날은....다 내일로!!!!!! 누워버려요!!!!!! 귀찬하!!!!!!!!!!!!\n(맘님홧팅요ㅜㅠ마음의 소리가 좀 껏쥬ㅠ)'], ['앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ읽으면서 느낌표에이입되섴ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ빵터졋어옄ㅋㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋ앜ㅋㅋ나 맘님 반응 넘 좋아욬ㅋㅋㅋㅋㅋㅋㅋㅋ앜ㅋㅋㅋ맘님하고 소통하는거 재미져요~~??'], ['그런가용?!!\n절좋아하시는군여...\n부꾸롭게....'], ['고생이시네여ㅠㅠ집안일은끝이없어여ㅠㅠ저두장난감닦아야하는데ㅠㅠ'], ['장난감은남편이닦는걸로..'], ['으악  오늘도 여전히 집안일 한가득이구나ㅜ'], ['왜끝이없지..'], ['아 나 스트레스 ㅜㅡㅜ'], ['오늘 최악이얌...?ㅠㅠㅠ'], ['울엄마..내가방 택시에 두고내림????????????'], ['헐........................대박\n어뜨케...........................\n택시찾앗어?'], ['아니 못찾앗지ㅋㅋ다행히 지갑은없긴한데\n가방이고 뭐고 다 새거라ㅜㅡㅜ\n현금내서 찾지못한다네\n넘 열받아 죽것다ㅋㅋㅋㅋ'], ['헐....\n이래서 택시에서는 무조건 카드를 ㅠㅠㅠㅠ\n어머님 저번에 스벅도그렇고 왜그러신댜...ㅠㅠㅠ\n언니속상하긋다'], ['스벅ㅋㅋ갑자기 떠오르노????????동서한테 선물받은 기저귀파우치도잇는데ㅋㅋ와  미칠꺼같다ㅋㅋㅋㅋ컨트롤이안되'], ['ㅜㅜㅜ안그래도오늘힘들아서 더화날거같다ㅜㅜ'], ['응응ㅜㅡㅜ진짜 미쳐버릴꺼같애'], ['오늘 왜케 힘든날이거여 ㅠㅠ'], ['넘 힘든데\n주부놀이중이야??????'], ['아 난 반찬만들어야되는데귀찮아죽겟다'], ['아침부터바쁘네 ㅜㅜㅜㅜ 화이팅'], ['갑자기급졸려온다ㅜㅜ'], ['아궁.. 바쁘바쁘네요..\n저거  어디꺼예요? 젖병 걸오놓는거?'], ['저두잘몰라용..당근으로 지율이아기때산거라..??'], ['아.. 그러시군유,'], ['많이건조할수잇어서좋은거같아여!!'], ['쏘야 하시나유?? 소독기 고장ㅠㅠ 불편하시겠어유 ㅠ'], ['쏘야해야쥬..넘기차나여ㅜㅜ'], ['전 낼 장보러가서 반찬 생성 해야해용'], ['오오오~~~\n저희는 주말에 장볼거같기도해요 ㅎㅎㅎ\n아직먹을게많아서!?'], ['ㅎㅎ저흰 읎어용 ㅎㅎ'], ['전 이제 완전다비우고사보려고용ㅜㅜ'], ['조아유 조아유 ㅎㅎ 저희집 냉장고'], ['뭐가 없쥬 ㅋㅋ 냉동실고 비우고 낼 장보러 가요 ㅎㅎ 오늘 저녁 국수 입니당 ㅎㅎ'], ['우와 진짜깔끔하네영!!!\n저희집은....뭐없는데도그득그득 ㅠㅠ'], ['전 다버려용 ㅠㅠ'], ['버리기엔 산지얼마안된것들이라 ㅠㅠ'], ['아아 얼마안된건 드셔야쥬 ㅠㅠ 전 진짜 냉털 탈탈다했네유 ㅠㅠ'], ['부러워여ㅠㅠㅠ저도 냉털다하고나면  인증한번해보고싶네옄ㅋㅋㅋㅋ'], ['소독기고장나셨다니ㅜㅜ 언능고쳐야는디ㅜ'], ['5만원이라고해서 지금고민중이여...ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['우유병 열탕소독 하는 것도 번거로울 것 같아요 ㅠㅠ'], ['괜찮은듯 안괜찮은듯하네여ㅠㅠ'], ['안괜찮을 것 같아요 ㅠㅠ'], ['그래도 밤에만 젖병쓰는거라서 아침에 한번만하면되용!!ㅎㅎ'], ['오? 낮에는이유식 먹나용ㅋ'], ['지태는 60일도안되서..ㅋㅋㅋㅋㅋ\n직수합니댜~'], ['아아 둘째는 신생아였네용! 한참 귀엽겠어용ㅋ'], ['으...신생아아닌거같아여..너무무거워여...ㅠㅠㅠ'], ['ㅠㅠ 백일도 안됐는데 ㅠㅠ 아가 무게가 생각보다 무겁다더니 정말 그런가봐요'], ['지금6키로넘는거같아여ㅠㅠㅠㅠㅠㅠㅠㅠ손목이너덜너덜 ㅠㅠㅠ'], ['헉 정말 금방금방 자라네요....'], ['진자 넘 많이먹고 너무빨리커여ㅠㅠㅠ'], ['핳 잘먹고 잘크는 건 그만큼 건강하다는 거긴 하겠지만.... 엄만 힘들죵 ㅠㅠ'], ['ㅋㅋㅋ손목이 안그래도 유리손목인데 지금은 갈대가된거같아여ㅠㅠ'], ['흑 남의 얘기가 아니네요 ㅠㅠ 저도 손목 한 가늚 하거든요 ㅠㅠ'], ['전 터널증후군에 건초염...\n시술도많이받아서 또 마비처럼 못움직이면그때는수술해야된다고하더라구여ㅠㅠ'], ['헐 ㅠㅠ 그럼 어서 스마트폰을 내려놓으세요'], ['ㅋㅋㅋㅋㅋㅋㅋㅋ그래서 무리안가게하고잇슴다!!!!!!!!!!!!1'], ['손목보호대도 잘 쓰시구요!!'], ['손목보호대는 자주오래하면안좋대서 진짜너무아플때만합니다ㅠㅠ'], ['에고ㅠㅠ 또 자주하면 안좋군요'], ['네넨!!맘님도 다복이낳고너무많이하지마세요!!'], ['전 지금 허리가 아파서 복대할까 했는데 그것도 오래하면 안 좋다고 그래서 고민이에용'], ['마자여ㅠㅠ복대가인위적으로잡아주는거라서 허리힘더빠지게하는거라고하더라구여ㅠㅠ'], ['뭐하나 쉬운 게 없네용 ㅠㅠ'], ['그쳐....임신.츨산 진짜 대단한일하는거예요 우리!!'], ['남편은 뭐 하나 하는 게 없으니 살짝 억울하긴 해요ㅜㅜ'], ['바뀌는것도없고ㅠㅠㅠㅠ너무한..'], ['소독기고장났어??그김에 뜨거운열탕소독 제대로 되는구만 ㅎ 나더 책이랑 장난감방 정리해야하는데..해야지하면서...손이안간다잉 ㅜㅜ'], ['울집은 장난감은진짜별로없는듯...ㅎㅎㅎㅎ'], ['이사오면서 장난감방을 만들어주니...감당이..애초에 만들지 말았어야해 ㅠㅠㅋㅋ'], ['지율이도여기이사오면서만들어줬엇는데 다팔앗닼ㅋㄱㅋ'], ['둘째가딸이얐음 팔았겠지만...그대로..물려줘야할듯...ㅋㅋㅋㅋㅋㅋ반이상이 고장났지만ㅋㅋㅋ'], ['앜ㅋㅋㅋㅋㅋㅋㅋ난이번에정리병올라서고장난거다버렷더니 반도안남음..ㅋㅋㅋ'], ['그 정리병..나한테 넘겨줘봐 ㅜㅜ'], ['앜ㅋㅋㅋㅋㅋㅋㅋ지금정리병떠낫는데 너한테안갓뉘!!이자쉭어디루간거지 ㅋㅋㅋㅋ'], ['핫싀...어디간거야..ㅠㅠ-ㅋ'], ['정리병다시찾아오면그때너한테가라고꼭말해줄게 ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['제발..나 진짜 해야지하면서..장난감방문열기가 싫어..ㅠㅠㅋㅋ'], ['장난감정리할때박스들고가..\n그박스꽉차서나올거얔ㅋㅋㅋㄱㅋ내가그랫거든ㅋㅋㄱㅋ'], ['다재활용텅으로 ㄱㄱㄱ????'], ['우리동네 저런거주워가는할미잇어서 할무니다드림ㅋㅋㅋ'], ['어제 몇개버렸더니 울고불고 ...'], ['그래서없을때해야하는..????'], ['오늘 고생했넹!'], ['아침부터 집안일이쫌많앗엇지..ㅋㅋㅋ'], ['언닌 쉴틈이 없겠다 ㅠ'], ['오늘은 진짜낮잠한번을안잣네...'], ['대단하다 진짜 ㅜ'], ['근데 낼은 낮잠도자고해야지ㅠㅠ너무조금잣어..분명졸릴거야 ㅋㅋㅋ'], ['틈틈히 자ㅠㅠ 언니 몸 생각해야지 ㅜㅜ 몸 상한다 ㅠㅠ'], ['그럼그럼!!졸리면바로바로자야지ㅠㅠ'], ['마쟈마쟈ㅠㅠ졸릴땐 잠이 최고야'], ['세상에 잠만큼중요한것은없지!'], ['마쟈~ 잠이보약이라잖아 ㅠㅠ']]
    
    9422
    다들 좋아하는 집안일, 싫어하는 집안일 뭐예요?? 저는 설거지 제일 좋아하고 ㅋㅋㅋㅋㅋㅋㅋㅋ빨래 정리하는거 제일 싫어해요 ㅋㅋㅋㅋㅋㅋ?설거지는 귀찮아도 즉시 하는데 빨래개는건 하염없이 냅두게돼요... ㅜㅜㅜㅜㅜㅜㅜ 신랑몫 ㅌㅋㅋㅋㅋㅋ
    
    [['저는 청소, 요리, 정리 좋아하고... 빨래, 설겆이 완전 싫어요... 물닿는걸 싫어하나봐요 ㅎㅎㅎ'], ['전 설거지 싫어하고 청소기돌리는거 좋아요!'], ['전 빨래는 좋고 설거지는 싫어요ㅠㅠ 음쓰 치우는게 제일 싫고!'], ['저는 빨래만 좋아해요 ㅋㅋㅋ 나머지는 다 신랑이....ㅋㅋㅋㅋ'], ['다 좋지도않고 싫지도않은 것 같아요 ㅋㅋㅋㅋㅋㅋ 어쩔 수 없이 하는.. 빨래가 제일 귀찮네요 ㅠㅠ'], ['저는 화장실 청소 제일 좋아해요ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ물뿌리면서 뽀독뽀독 닦는 기분 너무좋아요. 밥먹은거 뒷처리(상 치우기, 설거지, 음쓰ㅜ)빨래돌리는거랑 개는거 싫어해요.........'], ['저는 집안일중에 설거지 싫어하고, 하루 한번 청소기 돌리거나 창문열고 집 환기 시키는건 좋아해요'], ['저도 엄마랑 살때 빨래 널어달라는게 그렇게 싫었어요..세탁기 돌리는것도..ㅋㅋㅋ 근데 지금은 건조기가 있으니 아직까진 재미 붙이는 중인데 그래도 굳이 꼽자면 설거지가 더 좋더라구요'], ['전 설거지가 젤시러요ㅜㅜ좋은건 없구용ㅎ그냥 몸이움직여서 합니다ㅎ'], ['헉 저는 설거지가 제일 싫어요!!!!ㅠㅠ 부엌데기가 된 기분이더라구요... 빨래개는건 티비보면서 얘기하면서 할수있어서 괜찮은데, 설거지는 혼자 고립되서 일하는 기분이라 싫더라구요ㅠㅠ'], ['전 다좋아하는데 바닥 물걸레질하는게 젤싫더라구요ㅠㅠ\n그래섴ㅋㅋ 신랑보고하라히욤'], ['걸레질이 잴 싫어요 ㅠㅠ'], ['저는요리! 남편은요리담당.저는 설거지 빨래 청소요 ㅎㅎ'], ['요리 좋고 설거지 싫어요.바닥 쓰는 거(청소기 포함)은 좋은데 물건 정리나 물건 위 먼지 닦기는 싫어요.분리수거나 쓰레기 버리기는 좋은데 화장실 청소는 싫어요..ㅋㅋㅋㅋ'], ['저두 빨래개기가 제일 손안가는거 같아요 ㅋㅋㅋ 다른건 습관적으로 하는거라 별생각 없는데 빨래는 한번도 안갤어요 ㅋㅋ'], ['청소기 돌리는거, 빨래하는거 요리하는 거 좋은데 그 이후 뒷정리가 힘들어요ㅋㅋ'], ['걸레질이요ㅜ'], ['저도 빨래너는거랑 빨래 개는거 너무 귀찮구 싫어요 설거지는 그럭저럭요 ㅠㅠ'], ['저랑비슷해뇨 ㅋㅋㅋ 설거지는 자리에서 뚝딱하면대는데 빨래는 개서 넣으러가는게 귀찬..'], ['헉 저랑 똑같아요 ㅋ 좋아하는건 설거지하는거싫어하는건 빨래 널고 정리하는거요 ㅋ설거지는 먹고 바로 해야 하는 성격인데빨래건조대에서 다마르면 하나씩 착용해요 ㅋㅋ신랑이 빨래갤때마다 빨래 돌려서 널어놓은게 분명 많았는데갤때되면 별로없대요.. '], ['헉ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ완전똑같아요'], ['어머! 저랑 같으시네요~  신랑은 빨래, 분리수거,전 설거지,요리 하는걸로 분담했어요ㅎㅎ'], ['저는 요리가 좋고 빨래널기가제일싫은거같아용ㅋㅋㅋ'], ['다 하겠는데 화장실청소는 싫어요ㅜㅜ'], ['전 그나마 건조기 있어서 빨래 너는건 없는데 개는건 귀찮아요.. ㅋㅋㅋㅋ 요리는 좋은데 설거지도 귀찮아요..ㅋㅋㅋㅋㅋ'], ['저는 화장실 청소, 분리수거, 음쓰버리는거요 ㅋㅋ 이건 다 남편이 해요 ㅋㅋ 나머지 청소기 돌리기, 빨래, 설거지는 재밌어요 전 ㅋㅋㅋ'], ['전빨래개키는거좋아해요ㅋㅋ화장실청소시러하구요ㅋㅋ'], ['와~~ 저는 요리는 좋은데 설거지는 시러여~~ 그리고 밥푸는 거 조아해여(아 이것도 요리에 해당되네여) 변기청소 조아해여!'], ['저는 빨래 개는건 괜찮은데 너는게 너무 귀차나여..ㅠㅠ설거지는 세상 싫은데 냄새나고할까봐 억지로 매번 하네용..ㅠㅠ저는 그래도 정리하는건 좋아하는 편 같아요 ㅎㅎ 청소나!ㅎㅎ'], ['요리하는거도괜찮은데장만보는거도괜찮은데장보고와서 정리해서 넣어두고재료다듬고 요리해서남은재료관리하고. 음식물쓰레기버리고이런거다극혐요ㅜㅜ엄마랑살때는요리해먹었는데결혼하고는 오히려다사먹는다는'], ['저는 요리 너무 싫어요ㅠ 만드는건 힘든데 먹는건 너무 쉽네요^^'], ['전 설거지가 너무너무 싫어요ㅠ음식 하는거 좋아해요:)'], ['저는 좋아하는거 없음, 싫어하는거 화장실&베란다 바닥청소'], ['설거지, 빨래, 스팀청소기 좋아하공\n음쓰비우기, 화장실청소 넘싫어요ㅋㅋㅋ'], ['저는 빨래개기 좋고 화장실청소가 싫어요 ㅋㅋㅋ'], ['저는 다른건 다 좋은데 걸레질이랑 옷 정리하는거 제일 싫어해요 ㅋㅋㅋㅋㅋ'], ['화장실청소 젤 귀찮고 ㅋㅋ 음식하고 청소기 미는건 좋아요 ㅎㅎ'], ['허!! 저는 빨래랑 청소를 좋아하고 설겆이를 싫어해요'], ['저는 설거지 배수구통 청소가 제일 싫어요ㅠㅠ거름망쓰고 있다만 고추기름끼는 음식하거나 먹고나서 버릴때면......ㄷㄷ'], ['음..저는 좋아하는 집안일은 빨래고(세탁기가 해주니까) 싫어하는 집안일은 그 외 다 인거 같아요^^'], ['설겆이는 좋은데 나머지 것들은 다 싫어요 특히 청소요!ㅎㅎ'], ['좋아하는건 없고,, ㅋㅋㅋ 싫어하는건 청소?ㅎㅎ'], ['전 설거지가 젤 귀찮아요 ㅠ 기름 안 닦이면 스트레스 ㅠㅠ 빨래 개는 건 좋으해요!'], ['전 먼지제거 이불정리 화장실청소싫어해요ㅠㅠ\n설거지 청소기 이런건 좋아해요ㅋㅋㅋ'], ['저는 걸레질하는게 넘 좋아요 ㅋㅋ 물걸레 청소기 있는데도 걍 손걸레질하게 되더라구여 젤 싫어하는건 빨래요ㅜ'], ['요리 청소기 빨래돌리는거 좋고\n쓰레기버리기 설거지 화장실청소 싫어요ㅛ'], ['전 분리수거 빼고 집안일 다 좋아해요!! 성격상 과하게 깔끔떠는 스탈이라서'], ['빨래, 요리는 좋고 설거지 진짜 제일 하기싫어요 미쳐버리겠어여 차라리 음식물 쓰레기 버리는게 나아요'], ['좋아하는 집안일은 없어요...\n근데 죽어도못하는건 음쓰 치우는 거랑 요리요..차라리 설거지가 나아요'], ['좋아하는 집안일은 없고 싫어하는 집안일이 설거지입니당..ㅋㅋㅋㅋ 먹고 쌓아두게되요ㅠ'], ['저는 요리 싫어하고요. 사실 다 귀찮답니다. 남편은 정리정돈 좋아해요 ㅎ'], ['저는 빨래하기 제일좋아하고 설겆이 시러하는데 식세기 생겨서 그나마 나아졌어욥 ㅎ'], ['저는 설거지후 그릇정리 싱크대 정리할때가 제일 좋구 싫은건 물걸레 청소요....바닥 ㅠㅠ'], ['설거지는 신랑몫이에요 빨래랑 청소기는 잘돌립니다 ㅎㅎ'], ['전 ㅋㅋㅋ다시른거같아요 ㅋㅋㅋㅋ청소는넘귀찮고 차라리설거지가낫네요'], ['전 좋아하는 건 없어요... ㅠㅠ 다 싫어하지만 그냥 해여되니까 해요ㅠㅠ'], ['청소가 제일좋고 제일 싫은건 빨래 널기요 ㅋㅋㅋㅋ'], ['전 설거지 제일 시로어요 ㅠㅠ 청소기가 젤 좋아요 ㅋㅋ'], ['저두요ㅜㅜ 빨래 개는거 너는게 왜케 귀찮죠? 설거지는 바로바로 하게 되던데..'], ['빨래 개고 정리하는 건 싫고 요리는 괜찮아요 근데 또 재료준비 귀찮아요ㅋㅋㅋ'], ['저는 청소기 돌리는게 제일 좋고 화장실 청소가 제일 싫어요 ㅎ'], ['전 차라리 설거지가 좋고 요리가 힘드네요ㅜㅜ'], ['전 요리, 정리 다 좋아하는데 가끔 청소기 돌리는 게 귀찮아욯ㅎㅎ ㅠㅠ'], ['뽀송한 빨래 개는건 좋은데, 세탁기 돌리자마자 빨래 꺼내는건 싫어요 축축']]
    
    9441
    긋머닝???♀? 젖병소독기 오늘 왔습니댜??근데 젖병이 더블하트라 열탕..????일단 등원.출근 시키고소독기도 제자리에 놓고 집안일 좀 하구 올게욥ㅎㅎ오늘도 호기심딱지로 시작
    
    [['오늘두 홧팅!!'], ['밍디도 오늘 홧팅하쟈!!!!!!!!아쟈!!!'], ['엥? \n더블하트는 소독이안되나?\n\n열일하구 만나~??'], ['ㅋㅋㅋㅋㅋㅋㅋㅋ소독기돌리면환경호르몬나올수도잇다는뎅..흠...'], ['헉..., 뭐냐.... 이런..'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋ그래서 열탕하려고 ㅠㅠ'], ['좋은아침이에요^^'], ['굿모닝이요~'], ['오늘도 홧팅하쟝?'], ['언니 새벽에많이깨서어째ㅠㅠ'], ['부지런하시네요  ~~^^굿모닝입니다'], ['굿모닝입니댱 ㅎㅎㅎ'], ['굿모닝~ 오늘도 활기찬 하루~'], ['굿모닝!!오늘도홧팅 ㅎㅎㅎ'], ['굿모닝요 화이팅하자용 ㅎㅎ'], ['굿모닝입니당!!!ㅎㅎㅎ\n오늘도 힘내봅시댜!!ㅎㅎ'], ['좋은아침입니다ㅎㅎ'], ['굿모닝입니댱ㅎㅎ'], ['젖소젖소?? 드디어 왓네????'], ['젖소~~?ㅋㅋㅋㅋㅋ\n드디어왓는데 열탕하러..ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['앜ㅋㅋㅋㅋㅋㅋ 그렇게도 줄이는군욬ㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋ난순간뭔말인가햇자낰ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['신세대 말인가봐....ㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋ너무 웃기게 줄여서 말햇나?ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['그건 아니고 그냥 줄였어요 ㅋㅋㅋㅋㅋㅋ아.........ㅋㅋㅋㅋㅋㅋ'], ['참신하니 좋았어요 ㅋㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ나보고젖소라하는줄알앗엇엌ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ개웃곀ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['앜ㄱㄱㄱㄱㅋ내가 언니테 그런말을왜햌ㅋㄱㅋㄱㄱㅋㅋㅋㅋㄱㄲㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ나넘이상하게생각햇짘ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋ젖소라니!!!!!!ㄲㄱㅋㅋㅋㅋㅋ근데 오해할만해ㅋㅋㅋ내가 갑분젖소젖소그랬으니까'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ아니나학교다닐때별명ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ염병'], ['앜ㅋㅋㄱㄱㄱㄱㅋㄱㄱㅋㅋㅋ그럼...쫌.....찔렸을수도 있겠닼ㄲㄱㅋ아ㅋㅋㅋㄱㅋㄱㅋㅋㄱㅋㄱ'], ['뭔가햇자나ㅋㄱㅋㅋㅋ'], ['좋은아침이야아아아아'], ['굿모닝이야언니~~~~~~~~~~'], ['젖병에 따라 소독 되는 게 있고 아닌 게 있구나ㅠㅠ'], ['응 ㅠㅠㅠ 에쒸..ㅠㅠㅠ'], ['드디어 왔는데...못쓰는군 ㅠㅠ'], ['글치ㅠㅠ'], ['그래두 하루한번먹징?'], ['2번!ㅎㅎㅎㅎ\n자기전이랑밤수한번 ㅋㅋㅋ'], ['아항ㅎㅎ 혼합좋은거같앙ㅎ'], ['편하다 내가잘수잇으니ㄱㄲ'], ['마쟈 ㅋㅋㅋㅋ 나도 혼합하고 싶다 ㅠ'], ['해외껀먹여바써?'], ['아닝 안먹여봤엉 요즘은 젖병자체를 거부해 ㅠ'], ['흠...지금당장은괜찮아도\n땔때문제겟다ㅠ'], ['마쟈 ㅠㅠ걱정이야ㅜㅠ'], ['서현인 다혀니땜에자연단유햇을꺼아녀ㅋㅋ'], ['마쟈 ㅋㅋㅋㅋ서현이는 분유도 잘먹었어 ㅋㅋ'], ['그럼다행이긴하지ㅎㅎ'], ['웅웅 주면 주는대로 다 먹어었엉'], ['그게좋지ㅎ'], ['그때가 그리워 ㅠ'], ['']]
    
    9476
    집안일 중 제일 하기 싫은 것 갑자기 신랑이 뜬금없이 집안일 중 뭐가 젤 싫냐고 물어보는거예요... 전 설겆이라고 말했는데~ 다른분들은 집안일 중 뭐가 젤 싫으세요??? ㅋㅋㅋㅋㅋㅋㅋㅋㅋ
    
    [['저도 설거지요~~~!'], ['설겆이는 하루에 몇번이고 셀수없이 많이 해야되서..... 힘들어요'], ['저도 설겆이요 ㅠㅜ 모아서 하게 되네요'], ['해도해도 끝이 없는고 티가 안나는게 집안일이지만 그중에서도 젤 자주해야되는게 설겆이라....'], ['설거지랑 빨래개기요~ㅎㅎ'], ['저도 그두개가 젤 싫지만 그중 또 고른다면 설겆이요 ㅋㅋㅋㅋㅋ'], ['설거지 2222222222222'], ['ㅠㅠㅠ 무한반복이야... 끝이없어'], ['진짜 너~~~무 싫지..하'], ['화장실청소 ㅜㅜ싫어요ㅋ빡빡문질러도 티가 안나성..??'], ['서방시키 ~~'], ['ㅠㅠㅠㅠ 맞어요맞어 그것도 티가 왜케 안나는지.... 했다는건 냄새로만 알아요ㅋㅋㅋㅋㅋ'], ['로또 밥상겸 술상차리는거요 ㅋㅋ'], ['전 음식하는 이유는 신랑밥차려쥬기위해서 하는뎁...;ㅋㅋ 밥에 아주 진심가득이시거든요...'], ['진심이라 ㅋㅋㅋ\n저도 진심으로 차려줘요\n제 진심 ㅋㅋ'], ['전 진심이 아니에요 ㅋㅋㅋㅋㅋ 강압에 의한ㅋㅋㅋㅋㅋㅋ 안차려쥬면 일안간데요 ㅠㅠㅠㅠㅠ'], ['일가지마라해요\n밥 안차려준다하고요 ㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋ 우리세식구 손가락 빨수없어서 ㅠㅠㅠㅠㅠ'], ['아무튼 이집저집 다 애기 한명은 더 키운다 봅니다 ㅋㅋ'], ['맞어요ㅋㅋㅋㅋㅋ신랑이 아니라 큰아들입니다..... 아주 아주 큰아들ㅋ\n등치도크고 머리도크고 키도크고 ㅋ'], ['저는 요리하는거요ㅋ\n의외로 설거지는 좋아합니다만..ㅋㅋㅋㅋ'], ['요리는 즐기지만 설겆이가 너무 많아요.... 초보라 그릇을 많이 사용하는 이유기도 ㅎㅏ지만 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['설거지한 그릇 정리하는거요......ㅋㅋㅋ'], ['맞아요맞어!! 정리도 ㅠㅠㅠㅠ 예전에 했지만 이젠 걍 마를때까지 ㅋㅋ 놔둬요 ㅋㅋㅋㅋㅋ포기'], ['식세기 사주시려나봅니다~^^'], ['진짤까요???? .... 내심 기대중 ㅋㅋㅋㅋ'], ['전 빨래개서 제자리 갖다놓는거요ㅜㅜㅜㅜ'], ['아이랑 같이 하면 시간이 아주 오래걸리지만 그것도 일종의 아이와의놀이로 생각해서 그건 좀 덜한거같아요~'], ['2222격한 공감요\n뽀나동생 오랜만이야~잠깐 인사 ^^'], ['222222222222222ㅎ'], ['저두 빨래개서 정리하는게 젤루 시르네유^^;;;'], ['누가누가 빨리 건조기에서 꺼내나 ㅋ 뭐...내기도하고 색깔별로 모으기도하고 아이꺼 엄마꺼 아빠꺼도 따로따로 모으기 이런 놀이를 하면서 그냥 저냥 하고있어요~'], ['설거지요.'], ['ㅠㅠㅠ 식세기가 필요합니다'], ['우리집 설거지는70프로 신랑이 하기에ㅋㅋㅋ전 빨래널기 개키기..늠 싫어욧ㅋ'], ['부럽습니다!!!! 저희 신랑은 집안일은 아애 안해요~ 다만 애는 좀 잘 봐주니.... 그것만해도 다행이다 싶기도하고 ㅠㅠㅠ'], ['설거지요 ㅎ\n빨래는 하루 한 번이나 미루면 이틀에 한 번하면되는데\n설거지는 미루면 다음 식사타임에 문제가..ㅜㅜ\n식기세척기 사 달라고 시위할려구요 ㅎㅎ'], ['그죠.... 저도저도ㅋㅋㅋㅋㅋ 식세기 사달라케야겠어요ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['식세기 완전 필요합니다ㅠㅠ'], ['이사가면 사준다켓는뎁.... 그게 언제가 될지ㅠㅠ'], ['다 싫어 ㅜ 특히 빨래개기 ㅋㅋ'], ['빨래 개주는 기계도 있다능디.... 해외에 있는뎁 100만원돈이나 한데욥 ㅍ'], ['헐 진짜 니가 사주믄 써보마 ㅍㅎㅎ'], ['언니가 내보다 부자믄서~ 그돈 다 언제다쓴데~ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['빚이 많은거지 ㅋㅋ'], ['언뉘!! 빚도 능력이 있어야 가질수있어요~'], ['알았다 사달라 안카께 ㅋ 담주에 밥사주께 됐제 ㅋㅋ'], ['ㅋㅋㅋ뭐 먹지?ㅋㅋ 담주 비워두겠슴돠 ㅋ'], ['오이야~~~'], ['저는 음쓰버리는거요.. 너무 귀찮아요ㅜ'], ['ㅠㅠㅠ안귀찮은 집안일이 읍어요ㅠㅠㅠㅠㅠㅠㅠㅠ 왜케 할일이 많은지'], ['요리요\n차라리 정리만 할래요'], ['전 요리는 하고나면 맛있다고하거나 아이가 잘먹음 뿌듯해서 할만한뎁... 예전 직업이 음식만드는 일이라 긍가?그건 몸에 베어있지만.... 오히려 배달이나 그런거 시켜먹는게 잘 안되요ㅠㅠㅠㅠ 한달에 한 2~3번 될까말까?'], ['저는 그냥ㅇ 청소자체가 넘 귀찮아효....ㅋㅋㅋㅋㅋ'], ['청소가 젤 티가 안나긴해요ㅠㅠㅠㅠ 청소해도 머리카락은 꼭 한두개씩 보이고ㅠㅠㅠㅠ 마음같아선 거실에 아무것도 안놔두고싶어요~'], ['걸레빨기가 제일 싫어요 설겆이하는거 제일 좋아해요 설겆이하다보면 기분이 좋아지더라구요 ㅎ'], ['오리날다님과 결혼을 했어야 하는데.........'], ['ㅎㅎㅎㅎㅎ'], ['저는 빨래개기요ㅋㅋ제일 귀찮아서~~'], ['맨날 몰아서 하기 ㅋㅋㅋㅋㅋ  흰거 빨았지만 정리는 색깔있는거 하기 ㅋㅋㅋㅋㅋㅋ'], ['음쓰버리기요 음쓰처리기좀 사도라켔는데 ㅡㅡ맨날 이사가서 사주께...이사는 언제가노 ㅎㅎㅎ'], ['맞아요!!!!뭐 해줘 뭐사죠 하면 ㅠㅠㅠ 이사가면 해준데요.... 도데체 이사는 언제가는건지.... 말뿐'], ['이사가쟈 이사가쟈'], ['ㅋㅋㅋㅋㅋㅋㅋ이사온지 1년도 안된집이라ㅠ 까마득합니다'], ['핫한집 이사했으니 넣읍시다'], ['자리가 안나와요.....'], ['저도요 ㅡㅡ싱크대 하는거말고 뒷베란다 따로빼고싶은데...헐..'], ['뒷베란다에도 설치가 가능해요?'], ['긴거요 제품이 여럿있던데 가루로 화분에 흙으로쓰고. 쓰레기통에 넣는제품요...갈아내리는거 말고요 다..장단점이있을텐데...안사주네요'], ['부지런히 알아보고 ㅋㅋㅋ 결제영수증만 청구합시다 ㅋㅋㅋㅋㅋㅋ'], ['오~♡굿 결제만 늦추면 배송이 늦어질뿐'], ['설거지랑 빨래요'], ['두개가 젤 박빙이네요~ 주부들은 다들 공감하는 부분인거같아요~'], ['그죠~평일에는 밥하는게제일싫더라고요~^^ㅎㅎ'], ['밥은 2번~3번이니 개안은데..... 설겆이는 왜 횟수제한이 없을까요?'], ['ㅎㅎ그죠~^^'], ['화장실청소요~진짜 하기싫어서 한번도 안해봤어요'], ['전 담엔 화장실 하나있는집 가자고 했어요~ 두개있으니 두배로 해야되서 싫다공 ㅠㅠㅠㅠㅠㅠ'], ['저는 화장실청소 젤 좋아하고 빨래개는거 넘무 시러용ㅠ'], ['화장실이 왜 좋으세요? 혹시 스트레스가 많으신가요??? 팍팍 문떼기 ㅋㅋㅋㅋ'], ['화장실 물기 한톨이라도 있고 더러운 꼴은 죽어도 몬봐요\n성격이라 눈만뜨만 화장실ㅎㅎㅎ\n하루 두번씩 두군데 총 4번  장실청소해요'], ['헉;;;;!!!!!!충격입니다 \n아이 목욕하는 날만 옆에서 청소해요 ㅋㅋㅋㅋㅋ 세제안쓰고 싶어서 개고생하지만 ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['빨래 정리....널고 개는건 할만한뎅 넣는거 너무 싫어요...ㅠ귀찮...'], ['빨래 다 해서 정리했는데 꺼낼때 흐트러트리는거 보면 딥빡쳐요!!!'], ['저는 빨래요 ㅌㅌ'], ['건조기도 나왔으니 좀 더 기다리다보면 뭔 기계가 나오겠죠 ㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ개비는게 너무시러요ㅠ'], ['전 빨래 개는거요ㅠㅠㅠㅠ젤 시러요 ㅠㅠ'], ['빨래도 살겆이만큼 많네요~'], ['설거지하는거 싫어서 이번에 식세기 샀어요~'], ['대박!!!!!! 부럽습니다 저희집은 놔둘자리가 안나와서ㅠㅠㅠㅠㅠㅠ 이사가면 해준데욥'], ['저는 요리좀 누가해주면 좋겠네요~~'], ['배달 고고 해요 ㅋㅋㅋㅋㅋ'], ['다른건 그래도 괜찮은데\n빨래 개어서 각자 옷장에 가져다 놓는게 제일 싫네요--;'], ['저희는 옷방안 건조기 넣어놔서 방안에서 모두 다 이뤄져서 동선이 짧아 할만해요~'], ['저 빨래개기요??\n설거지는 그래도 식세기이모님 모시니 낫네요??'], ['부러워요~ 언제쯤 전 이모님을 모실수 있을런지........ 생일 선물로 사달라칼지 ㅋㅋㅋㅋ 고민을 살짝꿍해봅니다'], ['전 화장실청소요 ㅠ.ㅠ'], ['해도 티안나죠 ㅠㅠ 전 세제안쓰니 허리랑 팔이 떨어져나갈꺼같아요~'], ['정리. ㅎㅎㅎ']]
    
    9490
    남편이 집안일 어느 정도까지 도와주시나요? 말그대로 남편이 집안일 어느 정도까지 도와주시나요? 저흰 설거지,세탁기 돌리기, 화장실청소,음식물쓰레기,종량제버리기,가끔 장보기,이정도 하는데 다들 남편께서 많이 도와주시나요? 궁금해서 여쭤봅니다! 남편도 궁금해 하더라구요! 알려주세요~~~
    
    [['많이 해주시네유~~~'], ['요리는 제가하고 도와줄려고 노력하는데 제가 됐다고해요 가끔 보조하는정도에요 많이 도와주더라구요~~~'], ['본문에다가 플러스 집청소도요~'], ['집청소할때 거실은 같이하고 방은 각자 따로 하네요~'], ['아하 전 전부다 남편이 해줘용 침대 먼지 닦는것도 해주드라구요 너무 좋아요ㅋㅋㅋ'], ['좋으시겠어요 ㅋㅋㅋ'], ['저희는 평일에는 제가 주말에는 남편이 해요~ :-)'], ['그렇군요'], ['세탁기도 돌려주고 청소기도 돌려주고 걸레질도 해주고 때 맞춰서 세탁조청소도 해주구요 분리수거도 해주고 음식이며 설거지 화장실 청소는 제가 하구요~ 방청소 해주니 저는 화장실 청소 담당인거죠 ㅎㅎ'], ['많이 도와주시네요 ㅎㅎ'], ['시켜야하죠~ 일단 모든 쓰레기는 정리하고 버려줘요 가끔 화장실청소 \n장은 뭐 목록 써주면 이것도 가끔요 설겆이는 1년에 한두번 할까말까 \n 잘못 길들였나봐여ㅠㅠ'], ['그러시군요ㅜㅜ'], ['많이 하시는 듯...'], ['많이 도와주기는해요'], ['전 외벌이라 제가 다해욤 ㅎㅎ'], ['그렇군요 ㅎㅎ'], ['전혀 안해요 자기가 사용한 물건 정리만이라도 하면 좋겠는데 그것마저도 흘리고 굴려서 결국 다 제몫이에요 잘 도와주시는 남편분들 너무 부럽답니다'], ['222222222'], ['힘드시겠어요'], ['손꾸락하나까딱안하는디요?\n개똥도내가치우고 ㅋㅋ'], ['2222222'], ['그런가요 ㅋㅋ'], ['2222 넘 웃프네유'], ['분노의 2 찍기ㅋㅋㅋ 이제그러려니합니다ㅋ'], ['저기에 음식도 만들어요~'], ['많이 도와주시네요~'], ['평일엔^^ 음식물버려주고, 분리수거해주고,\n주말엔^^ 애들씻겨주고, 설거지, 빨래,청소기, 걸레질도 해줘요^^ \n가끔ㅋㅋ 요리도 해줘요'], ['좋은데요 ㅋㅋ'], ['와~~남편분 마니 도와주시네요'], ['많이 도와주네요'], ['에휴....댓글보니 눈물나네요'], ['도와달라고 얘기해보세요'], ['손하나까딱안해요ㅋㅋㅋ대신 주말없이 일하니 이해합니다'], ['그렇군요'], ['다 같이해요~'], ['둘이하는게 좋은거죠~'], ['음쓰및 쓰레기 버리기,빨래돌리기,설거지,아이 목욕시키기,놀아주기 청소는 같이 정도요~'], ['많이 도와주시네요~'], ['시켜야 해요..ㅜ ㅜ 가끔은 붙박이장을 새로 들여놨구나 싶어요 ㅋㅋㅋㅋ후'], ['시켜서라도 하면 다행이죠'], ['비슷해요..  저흰. 맞벌인데 저정도 안하면 나쁜거죠 아침등원은 아빠가준비해서 등원시키고하원은 제가해요그외 아이는 백프로 제담당이구요분리수거 본인방화장실청소 식세기가 설거지를거의다하지만 그외 설거지랑 청소기밀기 빨래개기  장보기 음식도 주말엔 남편이해요..전 아이와 하루종일 놀아주고요 ㅜㅡ 외동이라서요'], ['많이 도와주시네요'], ['아뇨...  도와주는게 아니고 같이해야는거죠 우리나라는 인식자체가 잘못되었어요 자기밥은 자기가 해먹어야죠..육아까지 포함해서 집안일 비율이 저는 적어도 70 남편은 많아야 30인걸요  이것도 전 불공평하다고 생각해요 ...'], ['부럽습니다 손가락 1도 안움직이는 사람이랑 사는 사람입니다ㅠ 슬퍼지네요ㅠ 똑같이 밖에서 일 하는데~'], ['힘드시겠어요 힘내세요'], ['신랑분이 믾이 도와주고 계신대요 ㅋㅋ \n저흰 저녁이랑 주말 식사는 거의 신랑이 해요. 장보고 반찬만들고 ... 음쓰,쓰레기 버리고... 화장실 청소 하고 정리정돈 저보다 잘해서 그거 해주고 계셔요 ㅋ 전 설거지나 청소 세탁 정도 하나요?'], ['그런거요 ㅋㅋ'], ['설겆이 분리수거 등..모든 보이면 도와주려고 해요.. \n\n음식물쓰레기는 내가 전담해요.. 그냥 그건 내가 한다 합니다.. \n\n그래도 나 아프고나선 자신이 스스로 한다는게 기특하고 고맙습니다.. \n많이 도와주네요.. 신랑님 최고~~!!'], ['많이 도와줘서 고맙다고 가끔 한마디 해주네요~~~'], ['밥하는거 빼고 다해줘요... 그런데 제가 다시해야해요ㅡㅡ'], ['그렇군요'], ['설거지는 잘해줘요음쓰및종량제버리는건 본인이 담배땡기믄 솔선수범하네요....^^'], ['설거지 잘하는거도 대단한거에요'], ['안도와줘요ㅋㅋ\n가끔설거지나 밥한번씩해주고 저녁에 가게하다보니 낮에는자요 ????\n그래서 로봇청소기 두대 매일돌리고\n건조기도 사달라했어요??'], ['건조기 꼭 사달라하세요'], ['하루만빨래안돌려도 다음날 바구니 2개3개는기본이예요\n수건부터 아이옷 남의편옷 미챠요 ????'], ['꼭 필요하신거 같아요 건조기 화이팅요!'], ['네~^^\n이번에 꼭사고말께요 으쌰'], ['네~^^'], ['헉..저희 화장실만 전담이예요..쓰레기는 제가 시키면 하고요..시키는건 해요..설거지는 물을 넘 아껴서 안시켜요ㅜㅜ'], ['설거지하는데 물아끼면 안되는데 깨끗하게 할려면 충분히 헹궈야하는데 ㅜㅜ 시키는거 잘하셔서 좀 수월하시겠네요'], ['저희남편은 제가 모텔청소부인줄 알아요..ㅋㅋ 가끔 집안일합니다..싸우기싫어서 포기하고살아요. 식세기산지 1년반인데 어제시댁식구왔다고 돌려주더라구요. 세제어디에넣냐는말에 동서가 뜨악했죠..ㅋ'], ['식세기 사놓으시고 왜 안쓰시는거에요?'], ['남편이 처음 썼다구요.ㅋ 저는 계속써요.'], ['음식, 쓰레기버리기, 아이 목욕, 장보기 이런거 해요~  전 빨래 청소 하구요~ 전 시킨건 아니고 저런걸 좋아해서 그냥 자기가 하더라구요ㅎ'], ['많이 도와주시네요ㅎ'], ['외벌이인데 대부분 제가 다하지만 해달라는건 다해줘요.종류 안가리구요.'], ['그렇군요'], ['저희도 분리수거 청소걸레질 화장실청소 빨래개기 애들 등하원 씻기기 장보는건 거의 같이보거나 아님 사올것만 캡쳐해서 시키고요 설거지는 키가 커서 높이땜시 물바다가되서 어쩌다한번정도 시키는정도예요ㅎ'], ['잘도와주시네요ㅎ'], ['같이살땐 거의 같이했는데 이젠ㅠㅠㅋㅋㅋ 쉬는날만 와서해요ㅋㅋ'], ['주말부부세요?'], ['주말 아닌  5일제부부요ㅋㅋ 쉬는날이 주말도됐다 평일도됐다 그렇거든요ㅎ'], ['그렇군요 ㅋㅋ'], ['제가 음식 해놓으면 \n아이들 저녁차려주기, 설겆이\n아이들 머리말려주기, 운동화세탁\n음쓰,일반쓰레기,재활용ㅡ담배때문인듯\n빨래정리, 빨리널기ㅡ종종 같이 하고요\n\n지인들은 저한테 늘 이야기하죠~\n남편한테 잘해주라고!! \n잔소리 많이 듣습니다^^;;'], ['저랑 똑같은데요 저도 지인들이 남편한테 잘하라고 잔소리듣는데'], ['남편말이 본인아버지는 정말 손하나 안움직였다고.. 자기는 커서 그러지 말아야지라고 다짐했다고 지나가는 말로 했어요~ 많이 도와줘서 늘 고맙죠~~~'], ['좋은 남편분 만나셨네요~~~'], ['아이들한테 늘 최선을 다해줘서 감사해요^^ 전 그렇지 못해서;;'], ['저흰 제가 욕실청소나 쓰레기(음쓰포함)버리기, 분리수거 이런건 엉성해보인다고 신랑이 전적으로 하고 나머지는 같이해요.\n빨랫감갖다놨을때 많다싶다 느낀 사람이 돌리고 건조기돌리고 마르면 같이 개고. 청소도 같이하고.\n밥도 먼저 퇴근한 사람이 자연스럽게 하는 스타일이예요.'], ['좋은데요'], ['근데 장은 한번 시켰다가 안시켜요.\n남자들 대부분 그런건지 손이 왜케 큰걸까요.????'], ['맞아요 가면 필요한거외에 자꾸 뭘 잔뜩사와요????'], ['저희도 청소, 빨래, 설거지, 분리수거, 화장실청소, 냉장고청소...왠만한건 다 해요ㅋㅋㅋ\n코로나때메 목욕탕 못가서 딸들이랑 저 때도 밀어줘요????'], ['때까지 밀어주시군요 가끔저도 ㅋㅋ'], ['그래도 전 세신이모님이 좋아요 ㅋㅋㅋ'], ['이모님만 못하지요 ㅋㅋㅋ'], ['주말마다 분리수거,한달에 1번 화장실청소,1주일에 1번  청소기돌리기,  매일 애 목욕시키고 재우기 ㅡㅡㅡ 딱 요만큼요. 맞벌이고 아이 등하원 요리식사준비 설거지 빨래 주중청소 집정리정돈  일반쓰레기버리기등등 나머지는 다 제가해요  장보기만 같이 하고요~ 주중에는 아예 손가락 하나 까딱 안해서 애목욕이랑 재우기만큼은 무조건 남편이 하게 해요'], ['그러시군요'], ['모든쓰레기 버려주고 7세전까지 아이들 씻기고 재워주는 정도 였는데 애들이 크면서 이젠 아빠가 씻겨주는거 챙피하다고 하네요 ㅋ\n해달라는건 다해주는데..집안일할때 눈치껏 같이 해줬으면 좋겠어요ㅋ'], ['말안해도 알아서 해주면 얼마나 좋을까요 ㅋㅋ'], ['가끔 재활용쓰레기버려줘요. 그리고 국이나 찌개 끓여줘요. 그리고 없네요ㅜㅜ'], ['요리해주는 남편이 부럽네요ㅜㅜ'], ['맞벌이이고 각자 잘하는거 해요.\n제가 요리하면 남편이 설거지, 빨래너는건 제가하고  빨래개서 정리하는건 남편이, 주말청소때는 저는 청소기돌리고 남편은 스팀청소기 이런식으로요.\n'], ['저희랑 비슷한데요'], ['저희도 빨래,설거지,정리정돈,아이씻기기,쓰레기버리기,물끓여놓기, 주말엔 아침준비 및 아이랑놀기 전 늦잠  평일에도 아침 안먹고 출근하구요 \n전 청소기 돌리고 간혹 빨래개기 화장실청소 침대 청소 해요  먼가 저도 하는데 신랑이 많이 하네요 ㅎㅎ 뽀뽀해줘야 겠어요 ㅎㅎ'], ['좋은데요 뽀뽀 필히 해주세요 ㅎㅎ'], ['가족들생일에 미역국 끓이기,퇴근후 시키는것 사오기,어쩌다 한번 빨래개기, 어쩌다한번 설거지하기,어쩌다 한번 목욕하면서 욕실청소하기\n이렇답니다.ㅎㅎㅎ\n\n그래도 평소에 다른걸로 점수얻기에 집안일 같이 안해도  밉지는 않네요^^\n\n그리고 중요한것!\n남자가 집안일을 돕는게 아니라\n부부는 집안일,육아를 같이 하는거랍니다. 늘 그렇게 얘기하셔야 습관이 됩니다^^'], ['같이 도와가면서 잘하구 있어요^^'], ['전혀요']]
    
    9593
    집안일 중 어떤게가장 싫으세요? 전 설거지,욕실청소 좋아해요물과 거품과 함께 할 수 있는거요? 빨래개는게 너무싫어요그냥 빨아두고 널어둔채로 마르면 입거나 무더기로 쌓아놓고 꺼내입고 싶어요ㅋㅋㅋㅋㅋ?
    
    [['요리요 밥하는게 싫어요ㅜㅜ'], ['저두요리요~요똥이에요ㅋㅋ'], ['저랑똑같으시세요ㅡ설겆이.욕실청소만해요ㅋㅋ나머지는신랑찬스입니다ㅋ'], ['전 설겆이요'], ['저도빨래개기요ㅋㅋㅋ한그시 모아두고 개켜요ㅋㅋ지금도쌓여있다는ㅋㅋㅋ'], ['저도요. 몇시간째 방치 중..하아 ㅜ'], ['저도 빨래개기요ㅋㅋㅋㅋ  설거지는 좋아해요ㅋㅋㅋ'], ['설거지는 좀 시원한 기분이 들어서 좋은데 빨래개기는 진짜....짜증나요 ㅎ'], ['빨래개어 자리찾아넣는거욤...\n느무느무싫어요.ㅋㅋㅋ\n'], ['진짜 넘싫죠 ㅋㅋ 그냥 널어둔채로 놔두고싶어요'], ['빨래 갖다넣기요ㅠ'], ['그것도 별로에요 ㅜ'], ['빨래개기 넘 싫어요ㅜ'], ['그쵸 넘싫어요'], ['와 저도 빨래개기요..'], ['지금 수건하나깨고 딴짓하고 속옷하나개고 딴짓하고... 아 ㅜㅜ'], ['그나마 요리는 진짜 하기싫은날엔 사올수도 있고 밀키트도 있으니 나름의 대안이 있지만 빨래개기는 진짜... 누가해주지않는이상 탈출불가에요 그래서 넘싫어요'], ['전  빨래개기요. . 진짜  싫어요'], ['역시 빨래개는거 싫어하시는분들이 많네요.진짜 짱싫음'], ['바닥닦는거 너무 싫어용..ㅋㅋ'], ['닦아놓으면 깨끗하나 귀찮죠 ㅜ'], ['빨래개는거요~'], ['너무 싫어요 ㅜ'], ['저랑 반대네요\n저는 물일은 싫고 빨래는 의류매장처럼 반듯하게 잘 개요^^'], ['오.. 전 개는게 너무 싫어요 오늘도 딴짓하다 겨우겨우 갰어요'], ['방닦는거요ㅎㅎ'], ['앉아서 박박닦는게 깨끗하긴한데 힘들죠 ㅜ'], ['쌀씻기 ㅠㅠ'], ['저도 설렁설렁 씻어요 ㅋㅋ'], ['빨래개기고넣는거요ㅡ'], ['저도요 너무 싫어요 ㅜ'], ['빨래갠거 제자리찾아넣는거요...'], ['으...맞아요 개는것도 갖다놓는것도 싫어요 ㅜ'], ['저 밥하는거요 반찬하기 넘 싫어요ㅜㅜ'], ['빨래개서정리하는거요@.@'], ['설거지....'], ['그냥 다싫으네요ㅠ'], ['청소요ㅜ'], ['다 싫은데.. 고를수가 없어요 ㅎ'], ['설거지로ㅠ시작해서 다..싫어여..'], ['다 싫어요ㅎㅎ'], ['빨래랑 요리요ㅜㅜ'], ['빨래 개고 정리하는거요ㅠ'], ['창문..창문틀청소요'], ['다림질이요 12년간 늘질 않네요ㅋ 손빨래도 극혐요'], ['전부다요..ㅋ'], ['다요ㅋㅋ 그 중에 설거지랑..\n빨래 널기요'], ['빨래개고 갖다 정리하는거요. 세상 싫어요. 그래서 신랑 시킵니다.ㅋㅋ'], ['방닦기요.. 진짜 기어다니며 닦기 싫어요..ㅜㅜ'], ['빨래개기.밥하기요 번찬걱정 넘싫어요'], ['저는청소요'], ['저도 빨래개는거 제일 싫어요 ㅋㅋ 맨날 건조기에서 하나씩 꺼내입고 남편한테 혼나고 있어요..ㅋㅋㅋ'], ['요리요 매일 세끼 제일 싫어요'], ['찌찌뽕이요ㅋㅋ'], ['설거지 ㅎㅎ'], ['전 샤시.베란다청소 젤싫어요~~^^'], ['다 싫지만 젤 귀찮은 건 청소ㅜㅜ'], ['설거지요ㅜㅜ'], ['청소가 제일싫어요~  특히 닦는거는 더더욱~'], ['전 설거지 너무 싫어요 식세기 이모님 드릴려다 사이즈안나와서 대실패 ㅡㅡ'], ['빨래 개기, 청소기 밀기는 좋아요\n설거지가 제일 싫어요'], ['거지거지설거지요ㅎㅎ빨래개는것두요ㅎㅎㅎ'], ['빨래개기 설거지 정리정돈 좋아하는데 바닥닦기랑 손빨래가 너무 싫어요ㅎ'], ['집안일 다 싫지만 그중 빨래개고 넣기 너무 싫네요']]
    
    


```python
xx= 90
for i in range(len(df1[['review','comment']][df1['com_num']>xx])):
    print(df1['Unnamed: 0'][df1['com_num']>xx].iloc[i])
    print(df1['review'][df1['com_num']>xx].iloc[i])
    print()
    print(df1['comment'][df1['com_num']>xx].iloc[i])
    print()
```

    2052
    이제 집안일.. 무슨 빨래가..해도해도ㅋ끝이없쥬?ㅠㅠ따님셋-하루에한벌씩(학교입고간거)밖에나갔다오면 하루에 두벌씩도가능ㅠ 속옷도하루에하나씩남편님-작업복 속옷수건은 5명이 하루에 쓰는게 7장ㅋㅋㅋ거의 그정도하나요?애들옷빨래, 우리옷빨래,수건빨래전부나눠서 하는데ㅠ귀차나요ㅠ그냥같이하고픔ㅠㅠㅠ
    
    [['으아.... 지겨워라....\n저도 하루에 세번 할 때도 있어요... 겨울되니까 더 많아보여요 ㅎㅎㅎ\n맘님 진짜 힘들겠어요'], ['ㅋㅋㅋ징해요 징짜 ... 건조기가 있어서 다행이지 ..\n없었으면 맨날 징징거렸을꺼같아요ㅠㅠ'], ['ㅎㅎ 저도 수건은 하루에 세네개정도씩 나오다보니 건조기돌려요 ㅎ'], ['제가 요새 씻는것도 힘드니까 , 덜씻었더니,7장씩나와요 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ\n예전에는 하루에 두번씩씻었는데 ..... 배가 너무 나오니,.씻는게 왜케힘드까요ㅠㅠ'], ['아하하 다행이네요 맘님이라도 덜써서 ㅎㅎㅎㅎㅎ'], ['ㅋㅋㅋㅋㅋ예전엔 하루라도 안씻으면 난리낫엇는데ㅡ.ㅡ'], ['나도나도 ㅎㅎㅎㅎ\n지금은 추워서 피하고.... 괜찮은거같아서 피하고.... 하하하'], ['ㅋㅋㅋㅋㅋㅋ아근데, 씻으면 다리는 못씻으니까 그것도 짜증나고 ㅠㅠ\n앉아서 씻자니 .. 그냥 돌이라 찝찝하고 ㅋㅋㅋㅋㅋ\n언니는 어케 씻어요 ? 별아빠님이 씻겨주시나요 ?+_+'], ['난 욕실에 목욕탕 의자 사다놨지 ㅎㅎㅎㅎ 그러게 별아빠는 왜 발한번 씻겨준적이 없지.....하하하 아 전에 있었구나.... 우씨...ㅎ'], ['아하 , 욕실의자도 사놓으면 됐을텐데 ..멍청했군요 ㅋㅋ안씻어진다고 ㅋㅋㅋㅋ혼자그러다 \n닿는곳만 씻고 ..근데 요새 씻고나옴 너무 추와요 ㅠ'], ['어차피 발도 좀 담그거 하려면.. 그게 필요할거 같아서..'], ['글긴할꺼같아요 ㅠ.ㅠ\n이제 밥시간됐어요~!! ㅜㅜ'], ['저녁준비 ㅠㅠ 난 요즘 저녁준비 안해서 좋아 ㅎㅎㅎㅎ.'], ['흑흑 부러운말중에 젤 부럽네요ㅜ'], ['아침준비만 열심히 ㅎㅎ'], ['아침이 더싫을꺼같아요ㅜㅜ애들 밥줄때 어우ㅜ'], ['그지.. 지금 11시에 나가니 다행인데... ㅎㅎ\n쏙 인나서 씻고 아침먹고 나가는 하숙생같아 ㅎ'], ['저희집도 하숙생한분계시죠~ㅋㅋㅋ\n저녁밥만 먹습니다 ..간혹 본인이 해서 먹기도하죠~ㅋ'], ['아하하 너무 웃기넹\n그래도 잘해줘야지 그만큼 힘들텐데 ㅎ'], ['그쵸 ㅋㅋ서로 잘하긴해야하는데 .. 그게 참어렵네요~ㅠㅠㅠ'], ['누구든 어려운거 같아.. 나도 그래..'], ['그러니까요 .. 초반에는 안구랬는데 ㅠㅠ히융~'], ['초반에만...ㅎㅎ\n초반엔 다 글치 머 ㅎㅎ'], ['그러니까요 ㅋㅋㅋ아 ,. 초기때로 돌아가고싶다요~ㅋㅋ'], ['ㅎㅎㅎ 늘 초기때같은면 좋은디 뭐든 ㅎㅎ'], ['그러게말입니다 ㅠ'], ['그게 안돼 사람이라... 다들 그렇게 살거야.. 너무 힘들어하지느..'], ['그러게요 ... 안싸우는 부부들은 어떤삶을 살까요?ㅋㅋ부럽네요 ㅋㅋㅋ'], ['안싸우는 부부가 어딨어\n그냥 내색을 안할 뿐이지...\n우리도 맨날 싸워 ㅎㅎㅎ'], ['그러까요?아 노력을한다고 하는데도 ..잘안되니 ㅜㅜ'], ['다 본인들만의 사정이 있는거니까.....\n세상 좋은 사람들도 쌓아놓고 사는 사람도 있고.. 그걸 꺼내서 싸우고 사는 사람도 있고......'], ['그러니까요 .. 사람은 본인만이 아는거라지유~'], ['더 힘든 사람들도 많을거야.. 우린 양호하다고 생각하고 사는거지'], ['그러니까요..아근데 나는양호하다고 생각하지않은데ㅠㅠ\n남편월급이 작아서 ...하루하루가 비틀비틀 ㅠㅠ'], ['에효... 어뜨케... 맞춰서 살아야지.... \n산교갔는데 보험하는 사람이 자기 연봉 6억이라고.... 내가 속으로 저런건 왜 나한테는 안왔지...ㅎㅎㅎㅎ'], ['ㅋㅋㅋㅋㅋ연봉6억인데..어쩌라는거지? 라고 저는 생각했을텐데 .ㅎㅎ\n저 보험회사 2년했지만 ... 6억이라고 뜨는사람은없었는데 ..1억5천인가는 있엇궄ㅋ'], ['아하하 수령액이 6억이라던데.. 뻥쟁이가 ㅎㅎ'], ['저희 한번씩 타지역가서 교육듣고올때면\n그런얘기해주거든요ㅋ어찌어찌해야 돈을 버는지등등ㅋ'], ['아~ 근데 보험해서 6억벌면.. 나도 하고싶다 ㅎㅎㅎㅎㅎㅎㅎㅎㅎ'], ['글믄 진짜 나도 계속햇쥬ㅋㅋ\n내친구는 빚진경우도잇어요ㅜ'], ['으아... 영업은 하는 사람이나 하지... 못해'], ['ㅋㅋㅋ나도 돈은 벌긴했는데 .. 그만큼 힘들기도햇고ㅠㅠ'], ['아 대단해.. 난 죽어도 뫃할거 ㅎㅎ'], ['글지글지 ..그거 영업적인거라... 말도 잘해야하고.. 구걸아닌 구걸도해야함 ㅠㅠ'], ['으아.. 더럽고 치사해 ㅎㅎㅎ'], ['진짜 .. 어마무시하지유.. 진짜 더럽고 치사한직업이긴한듯;~!!!'], ['그래 영업이 얼마나 힘든건데...'], ['맞아맞아 ..진짜 힘두렁ㅋㅋ'], ['영업하는 사람들 진짜 대단해'], ['나도 2년동안하면서 진짜 드럽고 치사하고 ㅡㅡ'], ['그만하길.잘했어..\n진짜 우리나라에서 서비스직은 점점... 특히 영업직은... 아니 누구 아랫사람도 아닌데..'], ['그래도 그일을 하면서 애들을 잘키웠쥬~!!'], ['참 진짜 대단하다... 진짜 세아이 키운거 존경받을만 한거..'], ['별거시기라서 돈도안주더라고ㅡㅡ\n지새끼테 나가는돈마져..'], ['애초에 그런사람이었네....'], ['그런거같음 ~ 진짜 드릅고치사해서 영업하면서 돈벌어서 애들 키우고ㅠㅠ\n막내원비라도 내달라니까 ㅋㅋ내가 정해서 보낸곳이니 내가 해결하랰ㅋㅋㅋㅋㅋㅋㅋ\n심지어 애들 폰요금이라도 내달라니까 .. 그것도 내가 한거니 내가내랰ㅋㅋㅋ\n원비야 20만원이지만 .. 폰요금은 3만원인데 ㅡㅡ'], ['으아.. 아빠자격이 없는 사람이었네\n진짜 원에 있음 그런 전화 가끔 와.. 아빠한테 보내지 말라..엄마한테 보내지말라.. 이런거.. 경찰도오고 막.... 가정불화로 막 그런거 .. 진짜 몰래 데려가고 나쁜 사람들도 마나... 그냥 그런 사람이라 관심 없는게 다행일수도있럴'], ['그거 생각하면야 .. 진짜 다행중다행인데 ..\n14살에 데리고간다는말 들으니까 .. 아오 빡쳐서 ㅡㅡ'], ['참나.. 접근금지 신청해... 무슨 자격으로 데려가'], ['내년에나할까해~ \n양육비도 더 높게신청할꺼구!'], ['그래 할 스 있음 해야지\n그러다 진짜 나쁜맘 먹으면 무서우니까..'], ['아,애들 성이랑 친권바꾸고싶은데ㅜ\n그건 절대 동의안해줄듯하고ㅜㅜ아오짜증ㅠ'], ['그게 이혼했는데도 동의가 필요해? 그럼 새 가정을 꾸리면.. 전 남편 동의를 얻어야해?'], ['응응~!! 그래도 동의가 필요하는부분이야 .. 우리나라 법은 완젼 거지가틈 ;;'], ['와 이미 정리를 했는데.. 양육권도 준거자나.. 친자라 그런가... 이상하다 그건\n그럼 새가정이 불편하자나'], ['14살에 데리고 간다고 ㅈㄹ중임... 그래서 내년에는 친권이랑 포기하게끔 소송하던할려구~!!'], ['웃겨 머해줬다고..\n진짜 접근금지해야겠다\n아니 그동안 잘하고 친권에 대해 목맸음 말을 안해\n머 다키워놓으면 데려간대?'], ['ㅋㅋㅋ그래서 몇달전에 다 포기하랬더니 .. 데리고갈꺼래ㅡㅡ\n이번에 막내 치과비용좀 같이하자했더니 읽씹..ㅋㅋㅋㅋㅋㅋㅋ하하하\n안해줄꺼알고 보낸거긴했지만 .. 참 ..'], ['ㅎㅎㅎ 그런거 다 증거로 모아놔.. 개뿔 신경도 안쓰고 지돈 아끼면서 애들을 왜 데려거'], ['전 아직 콩콩이 태어나기전이라 아직은 많지않네욤\n\n아기 옷들 손수건 수건 양말등등 빨게ㅠ ㅠ\n\n오늘 하루도 끝까지 마무리 잘하세욤 ㅎ ㅎ ㅎ'], ['저희는 초딩님들 빨래인데도 많네요 ;\n이제 넷째태어나면 더많겠지요 ?ㅠㅠㅠ'], ['저도 옷빨래 애기빨래 수건빨래나눠서해요ㅋㅋ요즘은 하나더붙어서 애기이불빨래요ㅋㅋ'], ['저도 꼬봉이태어나면 ,. 더 많아지겠지요 ? ㅠㅠ 으허ㅜㅜ'], ['ㅋㅋㅋㅋ하루에다섯번하실거같네요'], ['최고많이햇던게 세탁기6번돌렷던적이잇엇죠ㅋㅋㅋ'], ['곧최고10번이되실수도ㅋㅋㄱㅋ 빨래만돌리다하루가시겠어요'], ['아앜ㅋㅋㅋㅋㅋ 생각만해도 너무 끔찍한대요 ?ㅋㅋ'], ['으아~빨래하다 하루다가겠어요ㅠ'], ['이제 집안일 다하고 쉬네요ㅠㅠ'], ['빨래하다가 하루 끝나겠네요ㅠ'], ['그러니까요 ;;빨래개는것만 2시간하는듯해요 ;ㅋㅋㅋ'], ['고생이 많으세요ㅠ'], ['ㅋㅋ괜찮아요 ㅋㅋ건조기 덕을 많이봐서~ㅋㅋ\n'], ['전 건조기가 없어요'], ['나중에 기회되시면 꼭 사셔요 ㅠㅠ'], ['빨래하다가 하루다가겠군요'], ['그러니까요ㅠㅠ 건조기가있어서 얼마나 다행인지 몰라요ㅠㅠ'], ['빨래를 하루에 세번이상 돌리는집이 대부분일꺼예요'], ['그쵸? 아이고 ,.뻐치네요 진짜 ㅜ'], ['저희은 하루에 두번에서 세번돌려용'], ['ㅋㅋ저희는 최대 6번까지 돌려봣어요 ㅠ'], ['식구가많은집은많이 돌리더라고요'], ['네넹ㅋㅋ그런거같아요ㅜ'], ['사람이 많으니 배네요^^'], ['그러니까요 ;ㅋㅋ 힘드네요 ㅠㅠ'], ['저흰 남편이 수건을 두번써서 별로 안나와요\n전 하루 한개 ㅎㅎㅎ \n빨래는 두 그룹으로 나눠서 빨구여~\n아이 태어나면 횟수가 어마어마해지겠쥬?ㅜㅠ'], ['저도 지금 걱정이랍니다 ㅜㅜ 우리 초딩님들 그리 깔끔하지도않은데 ;\n옷은 왜케 잘갈아입는건디 ... ㅜㅜㅜ'], ['ㅋㅋㅋ나 초딩때는 그냥 입던거 입었던 것 같은데\n은근히 깔끔쟁이들인가봐요~\n저도 미리 겁먹어서 건조기라도 사놓으려구요ㅎㅎ'], ['건조기 진짜 사세요~ 전 건조기 득을참많이바요~ㅋㅋㅋㅋㅋ'], ['네, 안그래도 가격 좀 안떨어지나하고 체크하고 있어요.\n이달안에 사려구요~\n특가로 좀 가격 빠졌음 좋겠어요ㅎㅎ']]
    
    2651
    몰스님들은 어떤 집안일이 제일 하기 싫으세요??? 전 설거지요ㅋㅋㅋㅋㅋ 저희 집 오면 다들 애 키우는 집 맞냐고 할정도로 깨끗하고 저 스스로도 청소하는걸 엄청 좋아해요ㅋㅋ 스트레스받으면 청소하는 스타일이거든요. 그런데 저는 설거지는 도저히 못하겠어요ㅜㅜ너무너무 더러워요. 기름이랑 고춧가루 묻은 그릇들을 손에 만진다고 생각하면 너무 싫어요. 밥은 제일 잘먹는다는게 함정ㅜㅜㅋㅋㅋ 저 진짜 이상하죠??그래서 설거지는 하루종일 안하고 남편이 퇴근하고 오면 해줘요ㅋㅋㅋ 근데 또 웃긴게 저 음식물쓰레기 버리는건 아무렇지도 않아요ㅋㅋㅋ 화장실청소도 한번도 남편 시킨적 없고 저 혼자서도 잘해요. 설거지할래 화장실청소할래 하면 전 화장실 청소합니다 ㅋㅋㅋ여튼... 전 설거지가 이상하리만치 너무 싫은데 몰스님들은 뭘 제일 싫어하시나욤?
    
    [['저는 빨래 개고 그 갠 빨래를 가져다 놓는거요 정말 싫어요ㅜㅜ'], ['전 빨래 개면서 희열을 느끼는 사람이에요ㅋㅋㅋㅋㅋ'], ['하루종일 거실에서 있어요'], ['정말요? 진정 빨래 개는게 희열이 느껴지시나요?\n전 애가 셋이라서 한번 개는데 40분정도 걸려요ㅜㅜ'], ['저도요 저도요\n개는 것도 힘든데 ...'], ['제가 손이 빠르기도 하고 각잡아서 넣을때의 희열이 있어요....ㅋㅋㅋㅋㅋㅋㅋ'], ['저두 빨래개는건 그렇다치고 가져다놓기 진짜시러요. 집이 31평인데 한 50평되서 수납공간많아지면 좀 나아지려나요ㅜㅜ'], ['오오 저랑 똑같네요 ㅎㅎ\n전 청소 설거지 요리 다 잘하는데 빨래널고 개고 그게 세상에서 젤 시러여ㅜㅜ\n그래서 빨래는 신랑이 해요\n애들이 옷찾을려면 아빠찾아여ㅋㅋ'], ['저두요 건조기 사서 좋긴좋은데 빨래무덤이 자주생겨요ㅠㅠ 빨래 개주는 기계 누가 안만드나요'], ['저도 건조기 써서 빨래 무덤이 생긴답니다 ㅋㅋ 제가 매일 신랑에게 하는 말이에요\n빨래 개주는 기계 좀 생기면 좋겠다고ㅎㅎ'], ['저두요ㅜㅡㅜ'], ['ㅋㅋㅋㅋ 건조기 생기니 이제 개고정리하는게 싫어지죠 ㅋㅋㅋㅋㅋ 아'], ['50평되면 둥선이 길어져서 더 귀찮아져요. ㅎ'], ['저는 아이들 밥먹고오는 식판이요ㅠㅠㅠㅠ'], ['그것도 냄새 진짜 역겹죠....ㅜㅜ 으'], ['저두요ㅋ너무 귀찮아요'], ['빨래 개서 서랍장에 넣는거요 ㅋ'], ['저 빨래 개는거는 진짜 잘해용ㅋㅋㅋ'], ['빨래 개기요 ㅠ'], ['빨래 개는걸 많은 분들이 안좋아하시나봐용. 전 개면서 희열을 느끼는 사람이에요ㅜㅜㅋㅋ'], ['설거지는 맞는데 구찮아서요 \n전 음쓰도 싫어요 ㅋ\n화장실은 뭐 건식 써야하니 하는거구요'], ['음쓰도 좋진 않지만 차라리 음쓰가 설거지보다 나아요 저는ㅜㅜㅋㅋㅋ'], ['전 진심 정리요..아 정리하기시러요.'], ['정리하려고 맘먹기가 쉽지 않아요ㅋㅋㅋ'], ['요리....맨날 뭐먹을까 고민.. 장보는것도 그렇구요  차라리 설거지가 속편해요'], ['전 설거지가 싫어서 요리를 못하는 여자에요ㅋㅋㅋ 집에서 고기 한번도 안꿔먹어봤을 정도에요ㅋㅋ 기름 닦기 싫어서'], ['저도 설겆이,화장실청소가 싫어요ㅜ\n사실 집안일중 좋은건 없어요;;;'], ['요리요 못해서요'], ['애들 재우고 힘빠진 상태에서 집안정리요 ㅜㅜ'], ['애들 재우면 그냥 퇴근하셔야죠 ㅋㅋ 전 안해요 그냥 ㅋㅋㅋ'], ['설거지요~다른 청소는 부지런히 하고 빨래는 좋아하는데 설거지는 너무 귀찮아요..'], ['싱크대 하수구 청소요;;'], ['밥이요. 그래서 남편시켜요ㅋㅋ'], ['저도 정리요ㅜㅜ 너무 못하고 하기 싫어요ㅜ'], ['걸레질이요'], ['전 설거지, 빨래개기 다 좋은데 청소요~~화장실도 거실도 다 하기싫네요'], ['빨래개고 정리하기~~'], ['전 화장실 하수구 머리카락 빼는거요 아 생각만해도 넘 싫어요'], ['정리요... 진짜 ...,하'], ['전 청소여 ㅜ 빨래 설거지는 그래두 괜찮은데 왤케 청소는 해도 한 거 같지두 않고 좀 있다 또 해야 할 거 같고 채력 소모도 많고 청소 한 번 하자면 할 곳은 왤케 또 많은 지...ㅠ'], ['전 청소기 돌리고 정리하는거요 ㅠㅠㅠ\n번외로 여행가방 짐풀기도 ㅜㅜㅜ'], ['전 정리랑 청소요 ㅎㅎ 좋진 않지만 그래도 요리,설거지,빨래는 훨씬 낫네요 ㅋ'], ['밥 차리기요ㅠㅠ'], ['밥 하는 일요.남이 차려주는 밥만 먹고 싶어요'], ['청소요ㅠ쓸고닦고'], ['걸래빠는거요ㅜㅠ'], ['전 화장실 청소요ㅜㅜㅜ'], ['전 화장실 청소요...진짜 너무 싫어요.'], ['설거지가 전 젤편해요 아이장난감정리하는거랑 걸레질이젤힘든거같아요. 그리고음식물쓰레기랑 분리수거는 평생 신랑담당..전 쓰레기버리는게 그렇게싫으네요..쥐가나올것만같고..'], ['저는 바닥에 널부러진 애들 장난감 남편이 대충벗어놓은 옷 정리하는게 젤싫고 짜증나요'], ['빨래개는것까진 괜찮은데 갖다놓기 분리해서 여기저기 넣는거 넘귀찮..ㅜ'], ['빨래 개서 서랍에 넣는거 \n이것땜에 집안일이 너무 오래걸려요--'], ['빨래너는게싫어서 건조기.물걸레질귀찮아서 로봇청소기. 설거지는 식세기가..\n화장실청소가 전 싫은데 대신 해줄수있는 기계가 없네요ㅜㅜ'], ['빨래 서랍에 넣는게 제일 귀찮아요ㅜ'], ['다림질요...'], ['설거지 걸레빨기 다 싫어요 어깨아픔ㅠㅠㅠ'], ['옷 개서 넣는거요. 진짜 젤 싫어요 ㅠㅠ 구석에 맨날 쌓아놔요 ㅋㅋㅋㅋㅋㅋ'], ['화장실청소요.ㅠ'], ['빨래여'], ['전 걸레빨기요'], ['어떻게 하나만 싫을 수 있나요???? \n전 ㄷ ㅏ !!!'], ['전 정리정돈이요 ㅜㅜ 설거지는 씻어서 건조해서 넣으면 되서 금방해요~~~근데 수납장도 부족하고 옷개기도 힘들고 ㅜㅜ'], ['하수구청소요...ㅜㅜ'], ['빨래정리하기 ,,  베개 이불 커버 씌우는거요 ㅎㅎ'], ['전부 다요. ㅠㅠ'], ['요리요'], ['빨래개고 서랍장에 정리하기요~~~. 근데 저랑 같은 분들 많으시네요~^^'], ['한개만 골라야 하나요? 너무 어려운 질문 ㅜㅜㅜㅜㅜ'], ['음식물쓰레기 버리는 거요~ 3년차 주부인데 여태 한번도 안 버려봤어요 ㅠㅠ 비위 상해서 도저히...ㅠㅠ'], ['빨래갠거 정리하는거요~\n빨래개는것도 하겠는데..\n정리하는거 진짜 귀찮귀찮ㅜ'], ['뭐가 싫다기 보다는 집안일 전체가 자잘한 잡일이 계속 있고~안하면 나중에 하기 힘들다는 총체적 난국이 된다는게 싫어요~ 내가 계속 움직여야 유지가 되니까요~ㅜㅜ'], ['빨래정리요. 세상 귀찮아요.'], ['싱크대 배수망 빼서 그 밑에 관 닦는거랑 화장실 하수구 밑에 거름망 빼서 청소하는거 최악이요'], ['빨래 각자 가져가는거 5살후반부터 교육시켰어요.ㅋㅋㅋㅋ\n지금 셋째 젖병이랑 빨대컵이요ㅠㅠ\n시즌상품으로..세탁실에 빨래 종류별로 분류해놓기..집안 구조상 세탁실이 너무 추워서 다들 세탁실 문앞에 던져 놓으면 제가 색깔별로 종류별로 나누는게 일이에요ㅠㅠ 아 생각만 해도 욕나와요 진짜 겨울엔 세탁실 저만 들어가요'], ['다 싫지만 그중에서 설거지가 젤 싫어요\n음식물쓰레기 뒷처리까지 포함이요 ㅠㅠ'], ['저도 빨래정리요 ㅋ 건조기쓰는데 빨래쌓아두면 거기서 찾아 입어요 ㅠ\n저 시엄니께서 너무 깔끔하게 사는거 아니냐 지적받은 며눌예요 ㅋ'], ['빨래 개기요... 건조기 사기전엔 빨래널기...ㅎㅎㅎ'], ['빨래의 전과정 제일 피곤해요.\n시간도 오래걸리는데 자주 해야하니까요.\n\n그중 다 갠 빨래 각 서랍에 넣고 걸고 이게 제일 귀찮아요.'], ['설거지요'], ['전 걸레빠는거가 너무 싫어요.\n걸레 전용 미니세탁기를 뒀어요. \n미니세탁기 버린후엔\n물티슈나 1회용포 물걸레포만 써요.\n꼭 걸레를 써야하는곳이면 메리야스나 수건 잘라서 쓰고 바로 버려요.'], ['저도요 ㅋㅋㅋ 걸레빠는거 베란다에서 쪼그려하는거 게을러서 못해요 ㅋㅋ 샤오미 걸레는 작아서 세면대에서 씻어요 ㅋㅋ'], ['갠빨래  각자 서랍장에 넣기..이게 젤루싫어요.\n아이가 셋이니 방방다니면서 넣는거 힘듬'], ['씽크대 하수구 청소랑 걸레질이요~~너무싫어요~'], ['밥하는거는 진짜 애들 있어서 하네요'], ['전 그냥 다요ㅠㅠ'], ['저는 철마다 옷장정리 하는거요\n우리나라는 왜 사계절인지'], ['전 설거지가 젤좋아요 청소가젤시러요 극혐..'], ['저도 설거지요..그담에 밥하는거ㅎㅎ 한끼 먹고나면 그릇이 뭐이리 많은지 ㅠㅠ 큰애가 5살인데 엄마는 뭘 제일 잘하냐고 했더니 설거지래요. 완전 충격ㅠㅠ 제가 매일 설거지만 하나봐요..'], ['빨래개는거요\n진짜 빨래개서 제자리 넣어놓는게 젤 귀찮아요;;;;'], ['저도 빨래 널고 개고 넣고 하는게 젤 싫던데 다림질도 싫고... 그러고 보니 옷이랑 관계 되는 거네요ㅎㅎ 옷은 좋나하는데 참.......ㅋㅋㅋ'], ['악 저랑 똑같아요 ㅋㅋㅋㅋ 설거지 진짜.. 손계속씻고싶어져요ㅜㅜ'], ['설겆이-식세기\n청소-다이슨, 브라바\n빨래-건조기\n\n빨래개키기... 제일 힘들어요. \n\n식세기 사세요.'], ['전 반대로 설겆이가 젤 좋아요\n씻다보면 홀가분한기분들어요\n대신 걸레질 음식물쓰레기가 젤 싫어요'], ['설거지가제일싫었는데 식기세척기산이후로는 괜찮아요 저는 정리가제일힘들어요ㅠ'], ['밥하기가 젤루 시러용 ㅋㅋㅋ'], ['음식물 쓰레기 버리는 거 진심 싫어요. ㅜ'], ['고무장갑 끼고 해서 전 설거지 조아해요~ 설거지다하고 부엌 물기까지 싹 말라있은거보면 세상뿌듯해요ㅋ 대신 음쓰랑 화장실청소는 못하겠어요 특히 변기.. 귀찮은건 빨래갠거 정리하기네요ㅋ'], ['반찬, 음식이요'], ['저도 설거지ㅠ그래서 남편시켜요ㅡㅡ'], ['저두요 그래서 식기세척기 없으면 안돼요']]
    
    2884
    (2월18일 월요일 오늘의출석은요?)우리아이들 심부름 집안일 도와주나요? 저희 둥이들은본인이 알아서 스스로하는게거의 없어요  ㅠㅠ자기책상치우는것도 무슨 큰일한것 처럼  ㅋㅋ대신 제가 시켜요..빨래 널자   정리하자설거지 오늘당번은할머니랑 샤워할사람 ㅋㅋ제가 지독하게 많이시키는건가요?어떤분들은 어차피 난중에 많이해야할 일인대우리애는 안시키고 싶다고...하시는 분들도 계시더라구요?어떠신가요?.울 어뭉님들은?
    
    [['아직 어리지만 자기 물건은 정리하기 시작했어요'], ['안시키니 안하드라구요. 그냥 청소해줘 책상정리해줘 시키는편이에요 ~'], ['어지르지 않는 편이지만 가지고 놀았던건 정리하도록 습관 들였더니 정리는 알아서 잘 해요.\n아직 집안일은 안시키네요.'], ['스스로 먹는것 챙겨먹기\n라면도 잘 끓여먹어요\n심부름도 잘하구요\n정리정돈은 지맘 내키면 하네요\n뭔 바람인가하고 기특해할때도 있는데 잘은 안하고 아주 가끔요^^'], ['저희집은 어릴때부터 엄마 도와주는게 잘 훈련되어 있었지요..저 혼자서 애들 다섯을 케어하는 자체가 힘들기에 고사리 손들을 빌릴 수 밖에 없었거든요ㅠㅠ지금은 학교 수업시간에 가사분담 교육도 받아서 종종 나서서 도와주는데..요즘 첫째가 그동안 쭈~욱 해왔던게 많이 지쳤는지??모르쇠..할때가 있더라구요...실과시간에 과일깎기 수행평가도 있고,바느질 하기,등등 수행평가가 그렇다보니 전혀 못해도 해볼 수 밖에 없더라구요...그러다보니 자연스레 혼자서도 밥 정도는 차려먹고...설거지,실내화빨기,빨래널고 개기 정도는 당연히 하더라구요..역시 공교육의 힘이 대단해요!!!'], ['정리는 가끔 하는데 다른건 얘기해야 해요^^;;'], ['이제 초3되는 딸램.. 자기방 정리외 책상정리밖에 못 시키네요.. 4학년쯤 되면 간단한 설겆이랑 청소기 돌리기정도는 시켜도 될꺼 같긴 한데..'], ['스스로 할 수 있는 것을 조금씩 늘려가며 시키는 편이에요..손까딱 안하는 신랑이랑 살다보니..넘 힘들어서..ㅜㅜ'], ['첫째는 취미가 코딩이라 폰만 가지고 있고 학습은 알아서 하고 모든 하고 나서 제자리에 두는데 작은 아이는 뭘 만들고 꾸미고 색칠하고 종이가 한가득 ㅜ\n그리고 그대로 ㅜ  \n집안 일은 거의 혼자해요 ㅜ\n흑...\n'], ['본인방 침대랑 책상정리는 학교 입학하면서 했고요..그 외 집안일은 부탁해요 ~해  말고 ~ 도와줄수 있어? 요로코롬요 그럼 냅따 넘어와요 ㅋㅋ'], ['큰딸보다는 아들이.. 잘도와줘여..수건정리하는거...재활용 버릴때..'], ['우리 딸들은 책상정리,가방정리,벗은옷 빨래통에 넣기,식사한 자기그릇 설거지통에 넣기..이정도요^^;;\n가끔 부탁하면 신발정리,빨래개기 정도 해줘요^^'], ['기본적인 습관은 만들어 놔야지가 맞아요. 나중에 지 마누라한테 구박 안 맞으려면 시켜야된다가 맞지요ㅋ'], ['분리수거. 수저정리, 자기옷 넣기등은 해요'], ['작은아이는 잘 도와주는데, 큰 아이는..... 그래도 가끔 힘쓰는일은 잘 도와줘요~ㅋ'], ['어릴때부터 자기물건은 자기가 정리하게 습관들였더니 잘해요~'], ['어릴때부터 습관이 중요한것 같아요 저희아이도 늘 미루지만 조금씩 해보게하고 있어요'], ['시켜야지 스스로 하는 버릇이 생기더라구요\n자주 시킬려구요~'], ['7세지만 자기 장난감은 자기가 스스로 치울 수 있도록 유도하는 편입니다.'], ['자기가 먹은 그릇정리,벗은 옷  빨래통넣기,공부후 책상정리 같은 자기주변정리만 시키고있는데 그 외는 “내가 왜?””왜나만?”이런 마인드네요'], ['저희집은  시켜도  잘 안해서  부글부글하지만   계속   정리하자  같이하자  하며 참여시켜요.훈련계속해야 할듯요.'], ['정리보다 쌓아놓는 습관이 길러진듯 새여'], ['집안일은 시킬때만 하네요.\n하지만 자기가 어지르는건 스스로 치우기 하네요'], ['스스로 정리하기 간단한 마트심부름 \n바닥청소하기'], ['다른건 기대안하고 최소한 가지고 놀았던 장난감은 꼭 스스로 치우도록 독려중입니다 ~ ^^'], ['자기 물건은 스스로 정리하게 하구요~ 심부름가는건 좋아해서 자주 시켜요 ㅋㅋㅋ'], ['속이 터져서 시키다가 제가 다 정리하는 것 같아요 ㅠㅠ'], ['하나씩  어떻게  정리하고 오라고  시킵니다\n당근을 주고요~'], ['자기가가지고논거나어지른거는정리하라고합니다나머진아직못하네요ㅜ'], ['자기 할일은 스스로~\n설겆이는 아이들이 하고나면 뒷처리가 더 많은거같아서 제가 하는데 신랑은 아이들과 같이하거나 시키네요 ㅎㅎ;;'], ['저희 애들도 시켜야 하죠...스스로 한적은 본인 게임하고싶을때 잘보일라고 하네요..ㅎㅎ'], ['전 많이 시키는 편이예요. 독서하고 있을 때 이럴 땐 안 시켜야 하는데 .. 혼자는 너무 힘들어요'], ['자기스스로는 절대 안해요 ㅎㅎ그래도 이것저것 자잘하게 시키는 심부름은 아직까지는 잘해줘요.두살어린 동생과 경쟁하면서 서로하려다 싸움납니다..ㅎㅎ'], ['시킵니다\n우선 본인 책상. 침대 위 정리부터 기본이구요. 바닥은 청소기 돌리지만 밀대로 닦는 거 시킵니다 큰 아이는 가끔 설겆이도 시킵니다 빨래 개는 거는 무조건 아이들 몫입니다\n단. 시켜야지 합니다 ^^;;'], ['제법 혼자하는것도 많이생기고\n빨래개는거 부터 집안일을 잘 도와줍니다.\n기특해요'], ['심부름은 물떠오는 정도.. 대신 주말마다 장난감 정리함 엎어서 다시 분류, 정리 시켜요~'], ['전 조금씩 치우라고해요 \n책상하나치우는것도 엄청대단한일한거만냥~~\n크면더않할것같아서 지금부터 조금씩 시키고있어요  심부름도시키구요 ^^'], ['자기방 정리정돈만 하죠..자기방만 치워요..^^::'], ['적당히 시켜야할꺼같아요. 울집은 애들이 옷골라입으면 옷장이 엉망이되서 제가 맨날 골라주는데 ..요즘 숙제입니다'], ['천천히 시키고 있은데 외동에 딸 이다 보니 ㅜㅜ 애교에 자꾸 넘어가네요'], ['성격에 따라 다른것 같아요..꽂히면 하는 스탈이네요..ㅎ\n제가 청소하고있으면.. 옆에서 자기도 한다고 책상정리하곤 합니다. 잘한다고 칭찬해주고 밑밥(?)깔아주면..ㅎㅎ 좀 합니다. 아직어려서인지 유도를 하면.. 하는편인데... 보니 습관 들여 놓는게 좋을것 같네요^^'], ['아들녀석은 자신이 먹는 과자봉지도 안치워요... (@@!!) 근데, 딸은, 작은손으로, 어쩔때는 설겆이도 도와주려고 하고, 빨래 널을때, 속옷이랑 양말은,,, 뭐, 척척~ 갖다 쌓기는 하지만 ㅋㅋ 그래도 많이 도와줍니다'], ['저는큰애7살부터 쉬운것부터 하게했어요\n빨래개기.세탁기에 다된빨래꺼내기,\n냉장고에 반찬꺼내 밥차려먹고씽크대 그릇정리해놓기\n남자들이라 지금말잘들어줄때 해놓지않으면 크면서 안할까봐\n제가 힘든건 다도와주라고하는편입니다^^'], ['어려서부터 하던거는 잘하구요(밥먹고 그릇치우기, 책상정리등)\n방정리며 거실에서 간식먹고는 그대로 널려 있네요 ㅠ 말해야 치우는 ㅋ'], ['시켜놓고 치울때까지 기다리는거 넘 힘들어서 자꾸 치워주게 된다능....ㅠㅠ\n이제 조금씩 습관을 잡아 가야 겠어요'], ['자기가먹은밥그릇은 정리하게해요.가방정리,놀잇감정리도요.계속이건 너희가해야하는것이라고, 말해주고..계속얘기해요.😊 \n(물론스스로하는날은가뭄의콩이납니다요🤣)'], ['시키는데 잘 안하네요~~~'], ['스스로 정리하기랑 벗은 옷 빨래통 넣기를 시키긴 했는데 생각날땐 하고 안할때는 쌓여 있네여^^;;'], ['출석]\n 심부름이 뭔가요??? \n하라고 하면 대답만 하고 안 움직이는걸요...\n답답해서 제가 다 하고 화내교..........'], ['저도 계속 시켜보곤 있는데....영 안되네요 ㅠㅠ 습관 들이기가 가장 힘든거 같아요 ㅠㅠ'], ['자기 물건 정리 하는것도 잘 안 된데요..'], ['초등 올라감 집안일 조금씩 시키는게 교육상 좋다는군요.근데 엄마 성엔 안차죠~'], ['전 직장맘이라 주말에 같이 하려고 하는편입니다. 물론 시켜야 하지요. \n신랑, 아이와 빨래개기.설거지등 시켜요.'], ['정리정돈은 잘해용~'], ['전 4살때부터 자기밥그릇 싱크대 갖다놓기,장난감정리 시켰고 저희딸은 초6이라 집안일 도와주며 용돈벌고 있어요 주로 설거지나 동생장난감정리 해줘요 그냥 용돈 받을땐 돈의 소중함을 몰랐는데 스스로 용돈벌기 하고 난후부터는 돈의 소중함을 알고 아껴쓰고 있어요'], ['댓글보고 도움 받아가요.\n습관잡히게 조금씩이라도\n시켜야겠어요.'], ['수저놓기해요'], ['청소기 돌리게 바닥 치워달라하면 싹 치우긴 하더라구요. 막 어딘가에 쑤셔넣어놔서 문제긴 하지만요ㅋㅋ'], ['심부름시키면되려저한테하라고하는두따님들ㅋㅋ대신놀았던자기물건들정리만해주는것도감사하게생각합니당'], ['요즘들어 큰애 초딩5 아이가 많이 도와줍니다 어릴때야 칭찬해주니 도와줬는데 좀 컸다고 자꾸 도와주네요^^'], ['ㅎㅎㅎ자꾸 도와준다는 부러운 댓글이 있네요 ㅎㅎㅎㅎ 부럽네요 진심 ㅎㅎㅎ'], ['정리는 가끔 마음내키면..\n집안일은 도와달라고 하지 않으면 안해요..'], ['정리를 한다고하는데.. 아직은 많이 서툴러욤~~'], ['저도 가끔 시키려고해요.'], ['댓글을 보니, 오늘 부터 전 뭐라도 시켜야 겠네요.\n그냥 아기라고 생각들어 오냐오냐 했나 싶네요.\n빨래 개기 부터 시켜 볼까 봐요.'], ['저도 이제 슬슬 시켜봐야 겠어요.'], ['여전히 잔소리해야 정리하는 아이들이네요ㅠ언제 스스로 좀 할까요?ㅠ'], ['아직은 엄마 껌딱지라 고사리같은 손으로 빨래도 함께 개주고 신발정리도 해줘요'], ['보상이 있어야하네요 ㅋ'], ['시키면 잘해요.. 안시키면 안하지요...'], ['공부도 중요하지만 나중에 어른이 되어 누구의 도움없이 스스로 잘 챙기면서 사는게 진정한 어른이고, 진정한 독립이라 생각해요. 그래서 조금씩 집안일은 늘려서 시켜볼 생각이에요.'], ['자기 물건은 스스로 정리 하라고 이야기하고 있습니다.'], ['엄청 어지르네요. .\n시키면 심부름은 잘하는데 치우기를 잘 안하려고해요..'], ['큰아이는 자기방 정리는 조금씩 하고 있어요..\n근데 다른일들을 자꾸 도와 주려고 하는데... 솔직히... 가만 있는게 도와주는거라... --;;;;'], ['제가 하는 일을 몇번 같이했어요..\n설겆이 쌀씻기 청소기..그랬더니 언젠가부터\n아이가 먼져 같이하자 말하네요ㅎㅎ\n요즘엔 같이 아이방 정리중이에요..\n언젠가 혼자할때 도움되길 바라네요..ㅎㅎ'], ['정리만 잘해요..'], ['네 잘하는편이에요~'], ['아직은 어려서 제가하는게 더편해요.ㅋㅋ'], ['집안일은 같이하도록 시키고있어요'], ['저희도 다둥이네라큰애한테미안하죠 오남매님대단하세요'], ['시키면 하는수준? ㅎ'], ['첫째는 아들이라 어려서부터 시켰어요..ㅋㅋ\n 분리수거한거 내다버리기..수건개서 욕실에 정리하기...빨래정리하면 각방에 갖다놓기..수저놓기..반찬통정리..밥먹고 설겆이통에 넣기..물컵정도는 마시고 씻어놓기..방정리등등.. 가끔 청소기도 밀어주고 제법 많이 도와줘요..\n둘째는  저정도는 아니지만 오빠보구선 자기 방정리는 아주 깨끗이  잘해요..\n'], ['저도안시키는편인데.그러니엉망이네요'], ['어리지만 둘도 없는 효녀라 잘 도와줘요\n엄마를 도와주는 일을 즐거워하는거 같아요'], ['빨래를 개줘서 너무 좋아요ㅎ'], ['7세,8세, 방정리와 책상정리 가끔이요 ㅎㅎ'], ['11세. 집안청소 밥차리기 다되요. 9세, 청소와 정리가 되구요, 8세, 물티슈로 방닦기를 해요;;;'], ['10세.작년부터스스로샤워정도는하네요.아직놀이감정리는동생보다떨어집니다ㅎㅎ'], ['정리 잘 하는 것만으로도 만족하네요 ㅎㅎ'], ['아니요~ 암것도 안해요 ㅠ'], ['자기방 치우기, 빨래돌려서 개기, 저녁시간에 숟가락 놓기.. 집안일 시켜야 해요..']]
    
    4083
    집안일이란... 뭘까요... 빡침   어제 화장실을 들어갔는데 갑자기 물때가 너무 거슬리는거예요 ㅋㅋㅋㅋㅋ 락스를 뿌리고 나왔는데 게임하고 있는 남편놈을 발견했습니다가만히 생각해보니 화장실 락스 청소 왜 맨날 나만 하지 싶은거예요 ㅋㅋㅋㅋㅋ 속으로 화가 났지만 티는 안내고 베란다에서 분리수거 쓰레기를 내놓으려고 사부작 댔어여  그랫더니 왜 자꾸 뭘 하냐면서 제가 집안일하니까 자기가 편히 쉴수 없다고 궁시렁 대며 설거지를 하더라구옄ㅋㅋㅋㅋㅋㅋ  지금 상황 마음에 안듦 얼굴에 써붙여놓곸ㅋㅋㅋ  어젯밤 쓰레기 버리러 오면서부터 서로 한마디도 안함.. ㅋㅋㅋㅋㅋㅋㅋㅋ 저희는 언쟁하는 스타일이 아니라서 이게 서로 기분 상한거고 어쩌면 싸운거랑 동급이거든요 ㅜㅜ  둘다 맞벌이에 주말에는 주말이라고 안하고 평일엔 평일이라고 안하고 ㅋㅋㅋㅋㅋ 도대체 언제 해요 ?ㅋㅋㅋㅋ  그렇다고 남편이 집안일을 안하는 건 아닌데 미뤄뒀다 하는 스타일이라 제가 그 새를 못참는 것 같아요 ㅠㅠ  아아...정말 집안일이란 뭘까요... ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ
    
    [['그 마음 저도 알죠ㅠㅠ자꾸 거슬리고..ㅠㅠ'], ['집안일이 그런거 같아요 미루고 미루다 하는 타입은 나중에 내가 할랬는데- 이러고 눈에 보이면 바로 치워하는 성미는 그거 다 내가 치우잖아! 하고 짜증나고.... 서로 조율을 잘 해보심이 어떨지요???'], ['저도 그런게 너무 걱정되는데.. 하 저는 메리지블루가 아니라 지금도 안하는데 결혼한다고 할까 답답하네요^^;'], ['열받을것같아요 한사람만 하는느낌이면'], ['집안일은..... 더 부지런하고 더 깔끔한 사람이 지는 게임인것 같더라고요 ㅠㅠㅠㅠ 둔한자가 승자'], ['저도 한 게으름 한 둔감함 하거든요? ㅋㅋㅋㅋㅋㅋㅋ 남편이 더 심한거같아요.... ㅜㅜㅜ 진듯'], ['말을 해야해요!! 저도 열받아서 한바탕 말했는데 일찍퇴근하는날은 하더라고요'], ['저는 신랑이 알아서 하는 편이긴 한데..뭔가 100% 딱 마음에 안들게 해요~ 하지만 그렇다고 뭐라 하면 한 성의가 있는데 기분 나쁘니까~ 말없이 제가 뒷처리를 마무리하죠~ 진짜 집안일은 해도 끝이 없어요.......ㅠㅠ'], ['저희 남편도 나름 자기는 한다고 설거지도 해놓구 박스도 치우구 하는데 왜 화장실 청소는 안할까욤... ^^...'], ['같이살게되면 정말 사소한 부분들도 많이 부딪히는것 같아요ㅜㅜㅜㅜ'], ['집안일 누구라고 할거없이 해야하는건데, 갑자기 훅 치밀때가 있어요 ㅠㅠ\n'], ['진짜 결혼하면 그런것 하나하나가 걱정이에요.. 안맞을까봐'], ['그래서 사실상 집안일은 서로 정해놓고 하는게 좋은 것 같아요'], ['헐 신부님 어쩜 저랑 똑같으세요ㅠㅠ\n집안일은 정말 더러운거못보는 사람이 하는건갸봐요ㅡㅡ 저도 넘 빡쳐서 막 하고잇음, 저희 남편도 쉬고싶다.. 편하게잇을수없다 하면서 같이해요... 집안 더러운거보고 어떻게 쉬고 어떻게 편하게있죠;;;;;ㅜㅜㅜㅜㅜㅜ'], ['저 결혼전에 엄마랑 살 때 개판쳐놓고 잇으면 엄마가 화내셨는데 이제 그 마음 이해가 돼욬ㅋㅋ \n왜냐면 저걸 치우는건 나일테니까 ...ㅠㅠㅠ'], ['보통 주변에 나이 좀 있는 맞벌이 부부 분들도 일주일에 하루 시간을 정해두고 하시더라구요ㅎㅎ 집 안에 룰을 만들어놓는 느낌? 제가 들은 거로는 다음 일주일을 위해서 일요일은 무.조.건 대청소를 하신대요. 요일을 딱 정해놓고 싹 다 냉장고, 화장실, 빨래, 바닥청소, 물건정리 다 해결해놓고 일주일 편하게 지내는 거죠 ㅋㅋ'], ['계속 그러시면 속에 쌓일거예요ㅠㅠ 평생햐야하는 일인데...정해두시고 서로 하시는 게 좋아요ㅠㅠ'], ['정해두고 하지 않는 이유가... 이거 당신 담당이잖아 라면서 책임전가하고싶지 않아서거든요 ㅜㅜ'], ['맞벌이부부들의 흔한 공통사이기도 하죠 ㅠ_ㅠ 저희는 그래서 청소 날짜 정해놓고 각자 맡은거 해결해요ㅎㅎ'], ['ㅠㅠ결혼해서는 서로 잘 말하며 조율해야지싶어요'], ['저도 비슷해요 ㅜㅜ 에휴 청소담당이 남편이라 언제 하나 두고보자 하면서 저도 안하고 지켜봤거든요 그랬더니 물때에 먼지에 하아..... 결국 제가 했네요 ㅜㅜ  화나서 막 뭐라고 했더니 결국 말싸움으로 번지고 서로 감정상하고 어떡해야할까요?'], ['그래서 저는 같이 해요. 얘기해요 남편한테'], ['후한 빡침 흔한 일상  맘을비우고 청소는 내일이라생각하고 하는게 맘편할듯'], ['아 집안일 생각만해도..같이 배여하면서 해야할듯요..'], ['제 말이 그말이요. 그래서 그노매 게임 하고 있는 꼴이 아주 보기 싫어 죽겠어요.\n어제는 하다하다. 난 내가 가정부 같다고. 스스로 일을 찾아서 하면 안되냐고 했는데 .................... \n그냥 신랑 눈에 안보이는거겠죠.. 시키기는 싫은데 알아서 하면 얼마나 좋을까요 ㅠㅠ'], ['저도 비슷한 성향인데 첨엔 집안일땜에 많이 빡쳤어요 저도ㅋㅋㅋ 요즘은 명확히 나는 이거 할테니까 자기는 저것 좀 해줘~ 좋게 말하니까 바로바로 일어나서 하더라규요'], ['집안일이란 해도해도 끝도없고 혼자하자니 왜 나혼자하지? 같이하자니 그래 힘들텐데 조금 배려하자 내적 갈등이 어마어마한 것이예요 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ\n그래서 저희는 미리 이야기해요, 아까 집보니까 이런거이런거는 조금 치워야 할 것 같은데 오늘할까? 내일할까?? 이렇게요\n피곤하다고 하면 그날은 같이 안하고 그냥 둬요'], ['청소할때 같이 하자고 말을 해봐요!ㅎㅎ'], ['못참는 사람이 결국하게되어있대요ㅠ\n저는 좀 놔두는 타입인거같은데..\n오빠가 안그런것같아요ㅠ'], ['같은 스타일이 아니라면 하는 쪽을 더 따라야하는게 아닐까요 ㅠ\n해서 나쁠껀 없으니까요 ㅠㅠ'], ['하.. 이런 생각하면 벌써부터 스트레스 받으려 해요ㅜㅜ'], ['헛 저랑 너무 비슷하신데요... ㅜㅡㅜ 제가 못 참아서 해요 ㅜㅡㅜ'], ['더 부지런떠는사람이 하게되는거죠뭐ㅠㅠ결국아쉬운쪽이하는거같아요'], ['아 ... 잠깐 남동생이랑 둘이 살때도 집안일때문에 박터지게 쌰웟는데 그때 동생이 아쉬운 사람이 하면 되는거 아니냐고 하던게 떠오르네여 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['헐 진짜 ㅜㅜ후 집안일'], ['규칙을 만드시는건 어떨까요? 맞벌이인데 어느 정도 나눠서 하셔야 될거같아요!!'], ['맞아요 공감가네요~ 해도해도 끝이없는집안일~ㅠㅠ'], ['남자분들 집안일 시키려면 몇시까지 이것좀 해줘 이렇게 부탁해 보세요 데드라인을 정하고 시켜야 움직여요! 이거 다큐멘터리에서도 나왔던 건데요 항상 신부님이 다하지 마세요.. 저녁 9시반까지 화장실 바닥이랑 변기청소 쓰레기 버리는거 해줘! 이렇게 말해요!!'], ['맞벌이 하시는데 혼자 하면 좀 화가 나긴 하죠.... 저희는 쓰니님댁하고 반대 성향이라 제가 몰아서 하는 편인데 지금 현재는 외벌이라... 불평없이 하고 있다죵ㅠㅠ'], ['저는 남편이 사부작댄다고 못 쉰다고 투덜대는 스타일은 아닌데 정말 느려요.... 그리고 매일 안해요... 그리고 해도 제 맘에 안들어요... ㅋㅋㅋㅋㅋㅋ 어찌해야 할까용ㅋㅋㅋㅋㅋ'], ['제가 먼가 하고 있으면 잘 도와주는 편이긴한데 그래도 본인 하기싫을때는 저도 청소하길 바라지 않더라구요..본인도 해야하니까..'], ['저희 남편도 그거예요 ㅋㅋㅋㅋㅋ 안하는 사람은 아닌데 쉬고싶을때 제가 청소하나까 그게 싫은거 ㅠㅠㅠ'], ['집안일 진짜 힘들죠ㅠㅠ 저희부부도 집안일 하느라 시간 다 쓰더라구요ㅠㅠ 저희는 둘이 같이하는편인데 \n제가 일 다니고부터는 퇴근이 저보다 빠른 신랑이 좀더 마니하는것같긴해요ㅋㅋㅋㅋ'], ['작은 섭섭함이 쌓이는 것 같아요 집안일은 엄청 많고ㅠㅠ'], ['그리고 맡겨서 깨끗하게하면 몰라 성에안차면 제가 마저해요'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 오 맞아요 \n설거지 해놓은거 쓰려다가 안쓰고 다시 설거지통에 넣음.....'], ['이런것 때문에 초반에 많이 싸우게 되는거 같더라구요 ㅠㅠ 서로 대화 많이 하면서 맞춰나가야 하지 않을까요'], ['같이하자고 분담해보셔요ㅎㅎ'], ['규칙을 만드시는게 좋을 것 같아요 저도 저만한다고 생각하면 화가나서 신랑한테 짜증내게 되더라구요ㅠㅠ'], ['남편이 화장실 청소 해주는데.. 물때 그대로 맞은거 보여요... ㅠㅠ 고마운데... 너무 고마운데.. 하아.. ㅋㅋ'], ['ㅋㅋㅋㅋ...... 남편들 눈엔 물때 안보이나봐요... \n저 제가 손잡고 들어가서 보여줬는데도 남편이 깨끗한데? 이랫어여 ㅋㅋㅋㅋ\n아 혈압 오를듯ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['저희는 제가 사부작대면 짝꿍이 뭐야? 뭐해?라고 오는편이에요..저러면 진짜 짜잉날거같아욬ㅋㅋㅋ'], ['거의 남자들은 비슷한거 같아욤 자기가 한다고 해놓고 세월아 네월아라 저도 제가 해버려욤ㅋㅋㅋ'], ['저도 눈에보이면 제가 먼저 하는편이라서 나중에는 그냥 시켰어요~ 내가 이거할테니까 여보는 이거 해줘 하고.. 무조건 직접하지 마시고 시키는 버릇?을 들여보는것도 좋을거 같아요!'], ['자꾸만 같이 하셔야 돼요 그러다가 온전히 본인 몫이 돼요ㅠㅠㅠㅠㅠ'], ['아... 맞벌이시면 나눠서 하셔야 하는데... 신랑분이 정 하기 싫다 하시면 일주일에 한두번정도 가사도우미 부르세요... 저는 백수인데도 신랑이 힘들면 가사도우미 부르라고 하는데.. 신랑분 너무 집안일에대한 중요성을 모르시는 것 같아요..'], ['ㅜㅜ 너무 스트레스 받지마세여~ 날짜를 정해서 하는게 어때요'], ['아 ㅠㅠ 분담을 하셔야 될 거같아요. 전 신랑이 오히려 더 깔끔해서 ㅠㅠ 아예 나눴어요. 나눴지만 각자 야근 또는 회식이 있으면 대신할때도 있구요 !!!! 조용히 혼자 하시면 나중에 뭐라하면 왜 이제와서 뭐라하냐고 할거에요. 첨에 잘 잡고 가야함...'], ['남자는 교육시키기 나름이라던데 넘 어려워요 ㅜㅜㅜ'], ['그러면 서로 정해놓고하세요~'], ['끝이없는 집안일 증말 휴 한숨나와요'], ['집안일 빡치죠 ㅜㅜ 저희뉸 화장실은 신랑이 하기로 햇는데 물때가 껴도 안하고 잇으니 참 거슬리는데 참습니닷 ㅋㅋㅋㅋ 제가 화내면 자기딴엔 그래도 설거지도 허고 빨래개달라그러면 개주고 한다면서 받아치네요 ;; 눈에 안보이는것도 알아서 해주면 좋겟는데 그건 무리겠죠 ㅜㅜ'], ['업무 처럼 서로 나눠서 해요'], ['ㅠㅠ 너무 힘들면 하지 마세요 ㅠ 스트레스받아여ㅠ 그리고 남편분에게도 역할을 정확히 분담해주는게 필요할거 같아여~ 혼자 끙끙앓지 마세여 ㅠ'], ['저희도 초반에 집안일로 엄청 싸우다가.. 물걸레로봇청소기 사고 식기세척기 사고 하면서 템빨 받아서 그나마 좀 나아졌어요..'], ['ㅋㅋㅋ 서로 조율하고 하기전에 같이하자고 해요 ㅋㅋ'], ['집안일 되게힘들죠ㅜ 저 결혼14일차고 신행다녀온뒤로 쉬는날없이 집안일이안끝나요ㅜㅜ'], ['진짜 진심 남자들은 한번에 하면되지 아님 안해도 되는거 굳이 한다고 생각해여ㅜㅜ 너무 힘든거 같아영'], ['게임하고 있는거 보면 화날거 같아요 시키는건 어떠세요?!'], ['에효.. 저도 그렇습니다. 업무분담하시고 남편이 할일은 절대 손대지 말고 냅두세요. 그러면 눈치보다가 결국에는 하더라구요. 심리전이 필요합니다.ㅠ'], ['ㅋㅋㅋㅋ 저도 그냥 방치하는 스타일인데 여름되니까 이걸 못참겠어요 ㅋㅋㅋㅋㅋㅋㅋ \n초파리도 생기고 ㅠㅜ 설거지통에 초파리 익사 현장 보고 얘기했더니 해맑게 닦아쓰면 되지 라고 하는뎈ㅋㅋㅋㅋㅋ 휴....'], ['ㅠㅠ 흐잉 저희 신랑은 화장실청소랑, 쓰레기통비우기랑, 음쓰 버리기 직접해요 ㅠㅠㅠ 요새는 다 남자들이 많이 하던데 ㅠㅠ'], ['분담 확실히하세요 더러운거 참기 힘들어서 내가 자꾸하면 다 내일됩니다'], ['맞아요! 맞벌이는 같이하면서 집안일은 제가 더 많이 하게되서 진심 빡칠때 있어요!'], ['에거 저도 분담 확실히 정하는거 추천해요 ㅠㅠ'], ['저희모습 보는거 같아요 ㅋㅋ'], ['청소는 같이 한번에 하는게 나은거같아요! 집안일 진짜 힘들어여ㅠㅠ'], ['어우 저랑똑같네요! 저도 눈에보이면해야되고 신랑은 주말에몰아서하려고해서 결국제가 계속움직이게되요ㅜㅜ'], ['집안일은 해도 티도 안나고........너무 지치는 거 같아요ㅠㅠ'], ['집안일 해도해도 끝이없는거같아요ㅠㅠ'], ['저두 속터져서 제가 하고말아요ㅠㅠ'], ['저도 그래요 ㅠㅠ특히 글쓴이님처럼 화장실이요..결국 제가 답답하고 지저분해서 청소하게되요 ㅠㅠ지치게되네요.'], ['저도 화장실 예민한데 남편은 화장실에 제일 둔감한거같아요 ㅋㅋㅋㅋㅋ 으어 ㅠㅠ'], ['ㅎㅎㅎ 욕실청소랑 쓰레기 버리는건 남편담당인데.... 욕실 바닥닦는건 잘모르는거같아여;;;; 물때껴도 잘못느끼는듯ㅋㅋ 그냥 제가하고.. 나중에 얘기해요 어떤상태가 됏을때는.. 청소해야한다고'], ['ㅋㅋㅋ 저도 제가 막 부산시럽게 움직이면~  그래도 신랑이 뭐하냐고 물어봐주고 다른 집안일 하고 해서~\n저는 그래도 괜찮은 것 같아요 ㅋㅋㅋ  그래도 집안일은 정말 힘들죠 ㅠㅠ'], ['먼저 움직이는 사람이 지는거죠 ㅠ'], ['할때 같이하면 조은데 그맘 충분히 이해되요'], ['저희는 반대;; 전 쇼파에 누우면 안움직이는데 신랑은 청소기 돌리고 물걸레돌리고 세탁기돌리고 출근할때 분리수거하고.. 가끔 미안하네요 ㅠㅠㅠ 그래서 화장실물때청소는 한달에 한번정도 제가해요;;'], ['같이 청소하자고 해요 ㅎㅎ 화장실이 근대 제일 문제'], ['네 저도 들었습니다만...  집안일(특히 정리정돈 및 청소)에 한해서는...\n비교적 더 둔하고, 개의치 않는 타입... 그리고 구태여 비교하면 좀더 지저분한 것을 잘 견디는 이가 승리한다고 들었습니다...\nㅠㅠㅠㅠㅠ 티안나게, 어떤건 티 엄청나게도 힘든 집안일 ㅜ_ㅜ 하'], ['그냥 못참겠더라도 참으세요~ㅠㅠ.. 신랑분이랑 신부님이랑 스타일이 다른거니까.. 그냥 이따 하겠지.. 하고 넘겨요!!'], ['ㅠㅠㅠ 저 원래 그런 타입인데... 어제는 정말 제안의 진짜 제가 참을수 없었나봐요 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 난 이 더러운 집에서 살고싶지 않다고 외쳤어요 ㅠㅠㅠㅠㅠㅠ'], ['저희도 둘다 맞벌이라 걱정이에요 ㅠㅠ 락스청소....진짜 남자들은 몰라요 ㅠㅠ'], ['요즘같을때... 서로 같이 도우면서 하셔야죠'], ['부지런한 분들이 먼저, 더 많이 하게된다하더라구요;;;'], ['맞벌이면 같이해야하는데 왠지 여자가 더 하느 느낌적인 느낌ㅠ'], ['할일을 정확히 정해서 하세요'], ['제가 그래요!! 약간 사서 고생하는 스타일? 남편 시켜도 되는데 깨끗하게 맘에 쏙 들게 안하니 다 제가 하게되는... 언제쯤 이런 성격을 내려놓을지ㅠㅠ 흑'], ['ㅠㅠ다그런거같아요 예비 신랑집 가끔 가는데 치우질 않으니.... 눈에 거슬리고..'], ['참는 사람이 승리라고 하던데..뭔가 구역을 좀 정해서 하시는게 좋을 것 같아요! 괜히 감정 상하면 안되니까요~']]
    
    6196
    끝없는 집안일 페넬로페의 베짜기 : 쉴 새 없이 하는 데도 끝나지 않는 일을 가리킬 때 쓰인다.아이가 올해 초등 입학이라 3월부터 육아휴직했어요.. 아이랑 둘이 집콕중인데 거실이나 주방에 뭐 나와있는걸 싫어해서 매일 정리하느라 바쁘네요.. 슬슬 지치기도 하지만 아직은 포기가 안되는데..이제 아이를 위해 거실에 책장을 놓고 이것저것 좀 붙여놓을까봐요..
    
    [['넘나 깨끗'], ['저희집보다가...안구정화되었어요'], ['이야~모델하우스같아요ㅎ대박쓰'], ['저도 눈이 시원... 지금 책쌓기 놀이해서 집구석이 아주그냥 ㅋㅋ 둘째좀크면 저도 이리살고픕니다ㅜㅜ'], ['이런집에서 죽기전에는 살아볼수 있을까요?엄지척! 입니다.'], ['와~~~~이런 집이 있었네요~~^^\n넘 깨끗해요~^^'], ['우리집 거실로 나가기 싫어지네요..넘 깨끗해요~'], ['부러워요ㅠㅠ'], ['우와 청소할게없네요.깔끔합니다'], ['헉 집보니까 이사가고 싶네요 꼭대기층인가요? 앞이 탁트여 너무 좋네요'], ['탑층은 아닌데 20층 이상이고 앞이 트여 있어서.. 요즘 집에만 있음에도 살만해요..'], ['진짜 거실에만 있어도 기분 좋으시겠어요\n내년 19층짜리 아파트 19층으로 가는데 맨 앞동이라 지금 너무 기대되네요\n요즘 인테리어 사진만 보고 사네요...'], ['오~ 좋으시겠어요~ 이쁘게 인테리어하세요^^'], ['어우 깨끗!!!\n얼마나 쓸고 닦고 치우셨을지 알아서 존경스럽기 까지 합니당...'], ['제가 원하는건데요.넘 멋지세요.\n그릇들은 다들 어디있는건가요?\n그릇건조기가 서랍안에 있는건가요?~^^'], ['그릇은 식세기 돌린후에 씽크대에  정리해요'], ['아~^^ 댓글 감사드립니다'], ['아..반성하게되는 사진입니다ㅋㅋ\n저..퇴근후..집청소 할께요^^;'], ['이런게 가능한가요? 에이 잡지사진이죠?🤣'], ['청소하기도 편하겠어요  이미 너무 깨끗해서^^'], ['매트 없는게 젤부럽네요'], ['우와 대단하십니다. 바로바로안치우면 저리안될텐데.. 저도 나와있는거싫어서 다 넣는데 밑에열면 와르르 ㅋㅋㅋ ㅠㅠ비법좀전수해주세요'], ['주말에 씽크대 정리도 다하느라 몸 부서질뻔 ㅠㅠ 매일매일 치워요 ㅠㅠ'], ['넘나 깨끗하네요. 저희집이랑 비교되네요.ㅠㅠ'], ['모델하우스같아요~ 넘 깔끔하고 이뻐요'], ['모델하우스인가요,?\n저희집은 바닥에 빈공간을 찾아 다녀야하는 수준ㅜ'], ['이렇게 깔끔히 사시니 집안일이 끝이 없죠.. 좀 내려놓으면 편해요ㅎㅎ 부러워서 하는 말이예요~~'], ['곧 내려놓을듯요 ㅋㅋㅋㅋㅋ'], ['부럽네요 ㅋㅋ 깔끔한집 ㅋㅋㅋ 저희집은 치우면 5분이내 원상복귀 ㅠㅠ'], ['씽크대 아래 발매트 정보궁금해요~~'], ['한샘에서 샀어요~'], ['와우! 즤집은 거실이 책장이 많아서 아무리 저리해도 ㅠㅠ'], ['거실에 책장 있음 그렇죠... 저도 이제 그 세계로 가려합니다~^^'], ['급 청소해야겠다는 의지가 ㅋ하지만 현실은 쇼파에 붙어있네요'], ['전망이 굿 넘나 좋네요'], ['와..  모델 하우스같아요^^'], ['제 눈이 깨끗해지는 기분이에요~~👍🏻'], ['ㅜㅜ 모델하우스 아니네요 부엌이 제일부럽네요'], ['진짜 주방이 제일 부럽네요~~~~\n모델하우스 저리가라네요ㅋㅋ'], ['전 청소고자라..이렇게 깔끔하게 사시는분들 보면 넘. 부럽고 제 자신이 부끄러워져요..언제나 저렇게 깔끔하게 살아볼까 싶어서요(이번생은 포기에요ㅠㅠ)'], ['와우~~~진정 부럽습니다~^^'], ['나름 저 깔끔쟁이인데 애둘 집에있으니 말끔히 치워도 10분만에 도루묵이라ㅜ 안구정화하고 갑니다~ ㅎㅎ'], ['저도 뭐 나와있는거 싫어서 맨날 치우는데 치우고 나면 애들 남편 여기저기 꺼내고 쌓아놓고ㅠ'], ['와 부럽네요'], ['집안일은 해도해도 끝이 없고 누가 알아주지도 않고..그와중에 정말 깨끗한 집이네요'], ['아이를 위해서???? 그러지 마세요~ !!! \n아이 정서에 이렇게 단조로운것 아주 많이 좋을꺼에요..ㅎㅎㅎ \n너무 너무 부럽습니다.'], ['사람 사는 집 맞죠? 존경합니다!!!'], ['집이 어찌 이렇게 깨끗하죠'], ['와~ 누가 우리집 좀 이리 치워줬음 좋겠네요. 부러버요.'], ['조리도구들도 다 안에 넣으시는거예요? 쓸때마다 빼시는건지.. 진심 부럽네요 ㅋㅋ'], ['조리도구랑 도마는 나와있어요~'], ['와~~~~!!!!!!!!!!!!!\n세간살이 어떤식으로 정리했는지 넘궁금해서 찬장문들 열어보고 싶은건 저뿐인가요^^???\n진짜 한수배우고싶어요'], ['ㅋㅋㅋ 몇개 열어볼까요?'], [''], [''], [''], [''], ['역쉬 대단하시네요 정리잘하시는것도 재능인거같아요'], ['혹시 실례가 안된다면 냄비랑 후라이팬 뚜껑같은건 서랍에 어찌 정리하시나요~? 정리의 신이네요ㅠ'], ['저희 집이 수납이 많은 편이라 정리가 쉬운것도 있어요~ 후라이팬은 이렇게 후라이팬 정리대요~'], ['자주 쓰는 냄비는 인덕션 옆에 수납하고 큰 냄비들은 따로 수납해요~'], ['ㅎ 사진 감사합니다 저도 수납장은 많은데 시작이 잘못된거같아요 낼은 저길 싹 엎어야겠어요 ㅎ'], ['아..넘 예뻐요ㅜ 저희집 하루에 최소 3번 청소기 돌리는데요....아무도 안 믿을거에요ㅜ 이를 어째ㅜ'], ['심플하면서도이쁘네요그와중에  토스트기계가눈에  들어오네요'], ['진짜부럽네요'], ['헉 이게 가능한 현실집인가요 저희집이랑 너무 비교되요 ㅎㅎ'], ['네???? 우리집도 이번에 초딩 입학하는 백수있는데.....왜이리 다른가요?\n이건....엄마 잘못이었군요ㅠㅠㅠㅠ'], ['우와.........................부엌에 저렇게 뭐 안올려둘수 있나요.....부러워요'], ['집이 너무예뻐요 ~~'], ['남편분 행복하실듯요.'], ['와우~~! 너무 깔끔하고 좋네요!!'], ['옛날에 딘딘님? 그분집처럼 깨끗하네요 전모델하우스인줄요\n대단하세요 엄지척'], ['반성합니다 ㅠㅠ'], ['제가 꿈꾸는 주방이네요ㅜㅜ'], ['ㅋ 아이키우시는집 맞으시죠? 대박짱이세요'], ['딸 하나라 그나마 좀 괜찮고 아이한테도 매일 쓰고 제 자리에 놓으라고 해요.. 잘 안되긴 하지만요 ㅎ'], ['우린 언제 저럴 수 있을까..보기만해도 좋네여 ㅎㅎ'], ['집만 봐도 너무 깨끗해서 기분이 좋네요...'], ['부럽쓰요'], ['몇평이에여?'], ['40 요~'], ['헐 모델하우인줄요 대단대단'], ['모델하우스인가요? 뷰도 좋네요♡'], ['휴롬옆에 있는건 뭔가요?집이 넘 깔끔하네요. 댓글에 수납사진도 감탄하고 갑니다^^'], ['토스터기요~ 감사합니다^^'], ['힐링 하우스 같아요. 평화롭네요'], ['저런집이면 집콕해도 힘들지않겠어요~넘 부럽네요~♡♡'], ['와~~\n실화인가요?\n너무 깨끗해서 어지르기 미안할듯요...\n정리 비법을 좀 알려주세요~\n진심 부럽다요~~'], ['깨끗하니 참 좋네요'], ['구조좋고. 깔끔하고. 최고에요~~'], ['우와  저도 저렇게 살고싶네요'], ['우와집좋다아'], ['안구정화 하고갑니다'], ['난 이렇게 할수있을까?ㅋㅋ\n리스펙입니다 ~~'], ['놀라울 따름입니다~!!'], ['최고네요.님 사진보고 제주방 보니 한숨만 나옵니다...'], ['진짜진짜궁금한데 이렇게 깔끔하려면 아이가 아끼는 코딱지만한 장난감이나 원에서 가져온 교재교구는 싹다버려야 하는거겠죠?'], ['그래서 제일 정리 어려운게 아이방이예요~ 유치원에서 종이접은거, 재활용으로 만든거 이런것도 못버리게 해서요.. 아이방 수납장에 넣어놨다가 버려도 될만한건 조금씩 버려요 ㅋㅋ'], ['아래층 올라올까봐 초1,초3 딸둘맘은 아직도 매트를 깔고살아요..\n이런 거실 언제쯤...'], ['어맛!!모델하우스라고 말해주세요~~~']]
    
    6658
    남편분들 집안일 어느정도 하시나요? ㅎ 밥먹고 반찬통 냉장고에 넣어주고 식탁행주로 닦아줘요 몇달에 한번? 현관신발 놓는곳 물티슈로 청소하고 분리수거, 음쓰 나올때마다 버리는거, 가끔 빨래정리, 아주 가끔 빨래 널기, 이불정리, 청소기 대충 가끔해주네요그외 걸레나 화장실은 절대 안하구요 ㅠ 이 정도면 집안일 많이 하는 편인가요? ㅎ다른 분들은 어떠신가요??
    
    [['제가 힘들어서 밀림 해요.\n가사도우미 모드 아주 가끔요.\n주말마다 ㅋㅋ ㅜㅜ'], ['저도 밀림 하긴 한답니다 ㅎ 가사도우미모드 좋네요^^'], ['전혀 안 합니다워킹맘에 아들 둘주말엔 본인 쉬어야 한대서아들둘 데리고일박이일 대한민국 안 다닌곳이 없이벌써 고등이네요일요일 밤 9시즈음기진맥진어린 아들  둘  데리고운전하랴 애들 챙기랴평일 내내 늦은 퇴근에피곤한데 주말도 반납..그 몸으로 집에 들어와씻기고 재워 놓음꽃게탕 끓여내라는신랑입니다~본인손으로 라면도 안 끓여 먹고식탁 셋팅도 안 해요~이 점 빼고는다 백점이라 데리고 살고 있습니다의리네요 ㅎㅎ'], ['저희신랑 전혀 안해요 이번생은 망했어요ㅠ'], ['그죠 ㅜ 시대가 아직 ㅠ'], ['평소에 많이 해줘요. 정리도 설거지도. 할줄 아는거에 한해서 요리도요.\n음쓰 ,분리수거는 온전히 남편 몫이네요.. 대신 장실은 제가 해요ㅇ'], ['장실 청소 너무 힘들어요 ㅋ'], ['분리수거,일주일 한번 주말에 청소(화장실도) 고정은 이것만 해요 나머지 도와달라고 말하면 도와주지만 거의 주중은 퇴근이 늦어 제 몫이죠^^'], ['하긴 일하고 오면 피곤하죠 ㅎ'], ['일주일에 한번 주말청소는 꼭 남편이 해요. 화장실은 제가 하고 제가 식사 만들어서 차리면 설거지는 해주죠 ㅋㅋ 그치만 결국 육아와 살림은 거의 제몫....ㅠㅠ 맞벌이에 항상 힘드네요 ㅠㅜ 방금까지 내일 먹을 반찬 만들었어요 ㅋㅋㅋ'], ['반찬 ㅜ 해도해도 끝이 없네요'], ['때론 마음 놓는게 편한 것 같아요'], ['빨래 너는건 건조기가. 설거지는 식세기가 하다보니 개는것과 정리 정도합니다'], ['요즘 정말 잘나오죠 ㅋ 저도 갖고 싶네요'], ['이정도면 훌륭한거 아닌가요?저희도 이 정도인데 전 엄청 훌륭한지 알고 살았네요 ㅋㅋㅋ다른 집 남자들보고 시댁가서 얘기했더니 거짓말이라고 절 비웃던데요 ;;;'], ['그죠 ㅋㅋ 저도 엄청 도와주는줄 ㅜㅜ'], ['정해진건 없어요.식사준비랑 욕실청소는 온전히 제몫이지만 나머진 그냥 아무나해요.'], ['욕실청소는 해줬으면 좋겠어요 ㅋ'], ['쓰레기는 버리라면 버려주고\n말안하면 안버림..ㅠㅠ\n주말에 집대청소,싱크대,화장실,창틀 기타등등\n그것외엔 집안일 일체 안하는데 그래도 전 뭐라안해요ㅋ\n애들 태어나서 \n젖병소독이며 밤중수유(분유먹일때)며 목욕까지 항상 시켜줘서요~\n작은애 육퇴도 신랑이해요ㅋㅋ;\n육아를 담당해줘서 집안일은 안해도..ㅋㅋ'], ['육아 담당도 크죠~^^'], ['담배피러 나가며 음식물쓰레기 버리고\n주말 아침 간단히 차려주고(계란볶음밥,토스트)\n가끔 요리해고(튀김류 잘해서 튀김은 항상 신랑이하네요.)\n라면 끓일일 있음 신랑이 끓여요. \n신랑 씻고있을때 걸레 나오면 빨아줘요. \n운동화 세탁할일 있음 직접 해줘요.\n뭐든 부탁하면 잘해줘요.\n\n딱히 하는거 없다 생각했는데\n적다보니 몇개 있긴 하네요ㅋㅋ\n'], ['멋져요~'], ['저희 남편은 오자마자 빨래,청소,주방부터 봐요.ㅋ 그것도 많이 좋진않아요.잔소리좀 있거든요ㅠ'], ['헉 잔소리 너무해요 ㅜ 차라리 혼자 할래요'], ['전혀 안해요. 가끔 쓰레기 버리라고 시키면 할까 ㅡㅡ 자기 밥 먹은 그릇 조차 정리해본일이 없네요.'], ['힝 ㅜ 밥그릇은 물에 담궈 주시지 ㅜ'], ['아무것도 안하지만, 시키지도 않아요.\n그냥 그 부분은 기대도 안했고,\n안해준다해서 속상하거나, 서운하거나, 화나지도 않아요.'], ['전 화장실은 해줬으면 좋겠어요 ㅋ'], ['전혀 안해요~~ㅠㅠ'], ['퇴근하고 오면 피곤하긴 하죠 ㅠ'], ['남편이 거의 재택이라 아침하기,(저는저녁담당) 쓰레기 버리기, 빨래 개어놓기는 남편 담당이요.  화장실은 각자 써서 각자 청소하구요.'], ['각자청소 좋네요 ㅎ'], ['우와 너무 너무 부럽네요~'], ['분리수거, 설거지, 음쓰버리기, 애들이 어질러놓은 장난감 정리, 청소기 돌리기, 빨래 개기가 매일 하는 코스요. 교대근무라 낮에 집에 있을 때가 많아서 많이 해요. 저보다 깔끔한 죄로 ㅜㅜ 쓰고보니 미안하네요. 근데 요리를 못 해서 식사준비는 항상 저요. 애들 공부시키는 것도 저요.'], ['딱 좋은데요? ^^'], ['똑같은 교대근무였는데....1도 안하는데.....심지어 저는 연년생 워킹맘....이예요......\n이래도 세상에서 자기가 제일 좋은 남편 아빠인줄 알고 사네요......휴........'], ['시키는 것만 아주 잘해요(설거지 요리 쓰레기버리기 청소 정도) 그치만 알아서 해주면 좋으련만....'], ['그죠 ㅎ 부지런한 남편들 부럽네요'], ['음... 나갈 때 쓰레기 버리고, 제가 정리를 잘 못하는지라 아주 가끔 집안 전체 정리정돈해주는데... 결정적으로 요리를 잘해줘요. 본인이 쉴 때 삼시세끼 중 한끼는 꼭 해줍니다. 기 ㅁ 수미씨나 배 ㄱ 종원씨 요리 벤치마킹해서요... 그런데, 이것도 사실 제가 요리를 잘못해서 입니다. 웃프네요..ㅠㅠㅜ'], ['요리.. 전 남편이 담배를 펴서 그냥 제가 해요 ㅋ'], ['스레기봉투,.분리수거..버리고 강아지 소변패드 갈아주는...딱..그것만..ㅠ 즤 남의편은 머리카락이 뭉치로 굴러다녀도..절대.,안치워요..대신 잔소리도 안해요;;;;;'], ['ㅋ 저도 청소 잔소리는 안해서 그냥 넘어가요'], ['음식빼고 웬만한건 거의 다해요~~ \n본인이 더러운거 못참아서 하는 스타일이에요 덕분에 몸은 편해요 맘이 불편해서 그렇지'], ['그래도 부럽네요 깔끔한 성격 ~'], ['거실화장실청소, 주말설거지 전부다, 평일저녁설거지 가끔, 커다란 택배박스 나올때마다 출근할때 가지고나가서 버림 요정도네요~남편이 결혼내내 주말에 청소기랑 스팀걸레 밀었는데 작년에 애브리봇이랑 로봇청소기 구매이후 안하더라구요ㅋㅋㅋ  저는 전업주부에요'], ['저도 화장실청소만큼은 좀 해줬으면 ㅜㅜ'], ['일반쓰레기만 버려요. 담배피러 갈 이유가 있어야 하니까요. 그렇게 쓰레기를 찾네요'], ['맞아요 백번공감 ㅋ'], ['다 똑같은데 신발장 물티슈로 닦아주는 건 안하네요. 해달라고 해야겠어요. ^^;;'], ['맞아요 ㅎ 먼지투성이인데 ㅜ 그정도야 ㅋ'], ['저희도 1도 안해요 자기는 돈벌지 않냐고 하는데 저 돈벌때도 안했어요'], ['조금이라도 도와주시면 좋을텐데 ㅜ'], ['주말 같이 대청소하고 평소엔 설겆이해요 밥 물양 잘맞춰 밥도 가끔 해줘요~저 가끔 주말출근하면 아이들 밥해주고 분리수거는 일주일에 한번 꼭 해주구요 무엇보다 아이들과 잘놀아주니 그게 제일 좋더라구여ㅋ'], ['멋져요~'], ['반반하는 것 같아요 육아시간 쓰고 있어서 저는 아침 남편은 저녁에 아이 보면서 집안일 해요\n한명은 아이 보고 한명은 밥 하고 먹고 나서는 교대요\n규칙은 아닌데 같이 살려니 자연스럽게 그렇게 되더라구요'], ['해달라는대로 다 하는데 꼭 말해야만 하네요. 왜 스스로 할수없는지 의문이에요.'], ['22222 \n딱 공감해요. 시키는건 군말안하고 다 하는데, 좀 스스로 하면 좋겠어요.\n매번 말하고 부탁하는것도 귀찮고 지침요ㅠㅠ'], ['남편이 6시에 퇴근해서 집에오는 순간부터 저는 집안일 퇴근, 남편은 출근이요. 밥준비부터 치우고 설거지 ,후식, 아이들씻기고 방에 들여보내고 쓰레기버리고 뒷 정리까지 전부다 해줘요...'], ['저희는 제가 전업인데 저녁 설겆이와 주 2회 걸레질 하고 평상시 정리 도와주고 화장실은 이야기 하면 해줘요.'], ['전업인데 집안일은 남편이 스스로 알아서 더 해요 제가 안하니 하는것도 있겠죠ㅋㅋㅋ\n밥도 잘못해서 고민하다 지난번에 요리못해서 어떡해하니 밥차릴라고 결혼했냐구 해서 감동한적 있네요ㅋㅋ'], ['시간날때는 꽤했는데\n\n요즘은 분리수거,음쓰,애 씻기기\n가~끔 빨래돌리고 개고\n그외 뒤치닥거리(고치기,보수하기)정도네요\n(최근 책 1~2권 읽어주기 시작)\n\n\n제가 생색 엄청 냅니다~~\n전 일하지만 재택근무라 \n비교적 시간이 많다보니 하거든요\n\n신랑도 시간나면 아이 등하원 ,유치원가방챙기기,아침이나 저녁 간단,간식 ,가끔 요리,빨래,세차 다하는타입이에요\n(설거지빼고)\n\n부부의 시기마다\n포지션 변경중이에요ㅎㅎ\n\n'], ['외벌이지만요 집안일 완전 잘해요 부탁안해도 손도 많이 빠르고 자취를 오래해서 알아서 눈에 보이는건 다해요(살림고수)\n퇴근후 샤워후 욕실청소하고, 개수대에 음식물쓰레기 보이면 갖다버리고요 평일 아침에 기분좋게 일찍일어나면(전날 본인이 좋아하는 요리를 제가 해줬을때) 밥해놓거나 아님 샌드위치 만들어두고 출근해요\n주말에는 대청소해주고, 애들하고만 나가서 2시간정도 외출했다가 들어오고요 마트에서 장봐오고요 주말에 두끼정도 요리해주고, 제가 손 느리다고 설거지도 주말에는 해주고요(비싼 그릇 다 이 나갔어요 ㅜㅜ)\n평일에 애들 자기전에 양치질도 시켜주고... 잠들기전 책한권씩 읽어주고요 여자아이들이라서 이젠 목욕 못시킨다고 아쉬워해요\n작년에는 큰아이 초1때 아침등교시키고 출근했어요\n잔소리는 살림의 참여도가 높아서 조금있어요'], ['담배 피러 가는김에종량제.음식물 딱 두가지요이것도 제가 정리해서 내놓으면요외벌이요'], ['외벌이... 것도 10년차 주말부부인데 아무것도 안해요\n본인은 타지에서 고생하고 집엔 쉬러온다 하는 인간이라\n진심이지 애들 케어만도 힘든데 이인간 집에 오는 금요일은 확마 인상부터 써져요\n어쩌다가 이런걸 만나 결혼을 했나 도끼질합니다'], ['저희 신랑은 주로 더러운거 담당이요 ㅎㅎ 음쓰 버리기, 분리수거, 화장실이랑 배란다 청소, 어항 관리요. 밥차릴 때 상 놔주고 치우는건 항상 같이 하구요'], ['빨래개기\n쓰레기와 분리수거 전부\n요리할때 볶고 다듬고 쌀씻고\n무거운거 들고 나르기 집안 잡일?\n이게 답니다.\n저희집도 식세기와 건조기와 로봇청소기와 로봇걸레가 있어요.그거외에 안합니다ㅜ\n대신 말하지 않아도 자기일인줄 알고 찾아서 딱 저만큼은 알아서해요.'], ['밥은 주로 제가하고 나머지 몽땅해요ㅜㅋ 저질체력이라 언제가부터 그리되어버렸어요. 화장실청소 빨래 집정리 계절옷정리 음쓰 분리수거...쓰다보니 뜨금하네요.맞벌이긴한데 제가 체력이 늘 딸리는 쪽이라 ㅜ'], ['음쓰, 분리수거, 화장실 청소는 무조건 신랑, 일주일 3번 정도 집 청소해 주고 설거지, 빨래널기, 개기 신랑이 자주 해주네요. \n주말 아침, 점심도 신랑이 애들 챙겨줘요.\n그래서인지 아이들이 탕 종류는 저보다 신랑이 한게 더 맛있다 하네요^^;;\n저 전업주부인데도 신랑이 집안일 함께 해줘서 너무 고마워요~'], ['1도안해요 애들이랑 열심히 놀아주기는해요ㅋㅋㅋ'], ['분리수거 음쓰 쓰레기버리기는 100퍼 신랑담당이요 ㅋㅋ아침스스로차려먹기 퇴근함 빨래정리 도와주구 주말엔 돌아가며 식사준비설거지 청소해줘요 전업인데 제가애보느라저질체력이라 신랑이 같이많이해요 맞벌이하면서 분담했던게 얼추남은것같아요^^; 집안일은같이하는게맞죠 대신 육아관련한것은거의제가전담이예요'], ['청소전담, 분리수거, 둘째 씻기기 치카, 본인 옷정리, 스타일러 전담...저 일찍 쉬거나 피곤해하면 설거지하고 배달시키면 뒷정리 다 하구요 사실 이정도 해도 제일이 산더미라는거ㅠㅠ 나 일할때 너 빈둥대는거 제일 싫다고 우리집은 너 없어도 문제없다했더니 그뒤로는 하네요'], ['분리수거 ,쓰레기버리기, 빨래통에 옷넣기'], ['전 9:1 남편:저'], ['전업인데 퇴근이 늦어서 평일엔 거의 못 도와주고 주말에 화장실 2개 청소는 꼭 해주고 간단식(라면, 토스트, 인스턴트 등)해주거나 가끔 청소해줘요 건조기, 식세기 있어서 나머지는 제가 하고 대신 애들 잘 케어해요(목욕담당)'], ['식사하고나면 둘 중 한 사람이 테이블정리 한 사람은 그릇 애벌해서 식세기에 넣구요~ 분리수거나 음식물 쓰레기는 주2회정도 버려줘요~ 그래서 그 때 분리수거 모으는 편이고 음쓰는 제가 먼저 버릴 때도 있구요~ \n매일 로봇청소기 비우고 걸레 씻어 끼우는 거 제가하고 남편이 큰 청소기로 돌리고 밀대 밀어주는 거 주1회 주말에 해요~\n대신 7시에 식사하고 이후 10시에 아이 잠들기 전까지 씻기고 놀아주고 재우는 등 아이케어 전적으로 다 맡아줘서 고맙게 생각해요'], ['서로 상부상조'], ['적어주신거 남편이 매일 다해요; \n남편이 요리만 못해서 요리는 제가해요\n최근에 식세기를 사서 남편이 더 좋아합니다 ㅋㅋ'], ['분리수거\n화장실청소\n운동화빨기\n쓰레기정리\n음식쓰레기버리기\n이불털어주기\n빨래널기\n이 정도요~\n설거지 해달라고 하면 해주고요'], ['저흰 맞벌이구요.\n집안일 육아의 비중을 따지자면\n남편90  저 10이에요.\n저는 저질체력이라 남편이 밥하고 빨래하고 청소하고 애보고 거의 다해요.'], ['빨래 분리세탁만  제가 하고 다른건 남편이  거의 다 해요 \n가구 옮기고 싶다. 책장 옮기고 싶다.. 그러면 나가서 놀다오라하고 혼자 다해놔요\n주말오전은 아이들 밥 챙겨주고 오후엔 저 쉬라하고 놀이터든 키카든 데리고 나가구요 \n전업인데 참 고맙네요'], ['저는 저녁식사담당\n신랑은 그시간에 빨래개우고 서랍에넣어주 는거해요\n화장실청소도 100번중90번신랑이해요. 분리수거는 신랑이버려주고 냄새예민한남자라 음쓰는 100번중 90번 제가버려요. 아이셋이고 결혼9년차.. 전업8년에 워킹맘1년하고 또다시 육아휴직중인데 워킹맘때처럼 여전히 빨래개우기랑 넣어주기 도와주는 남편이 고맙네요.. 대신 제가 육아는거의오롯이혼자..ㅋㅋㅋ 뚜벅이지만 애들셋데리고 여기저기 잘 다녀요. 평소에는 자전거뒤에 유아안장에막내태우고 아들딸 첫째둘째는 킥보드타고 같이 여기저기 다니기도하고 ㅎㅎ 목욕도 100번중 95번은 제가시켜요 아이들.. 아이들이 엄마껌딱지.. 운동화세탁도 항상 제가해요..'], ['아예 아무것도 안해요..\n쓰레기 현관앞에 내놓으면 출근할때 그것만 들고나가서 버리는것만..ㅠㅠ\n\n로봇청소기.건조기 사주고는 아예 안도와줘요ㅠㅠ\n일이 힘드니 그러려니 해요..\n돈잘벌어오니 그냥 그래..힘들지 하고 아무말안해요ㅎㅎ'], ['분리수거박스랑 안방화장실요'], ['맞벌이땐 같이 했어요.. 지금은 외벌이인데 거의 와이프가 해요. 같이 도와주고 싶어도 이젠 체력이 안되네요 ㅠ 대신 식세기 건조기 스타일러 등등... 살림에 도움될만한거 사달라고 하면 사줘요... 글 적다 보니 미안해지네요 ㅠ'], ['분리수거 중 박스만 버려줘요.\n로봇청소기, 식세기 등등 기계들였다고 내가 노는줄아나봐요.\n하지만 집안을 내맘대로 난리를 쳐도 가만히 있어요.'], ['음식쓰레기 매일 버리기.\n청소기 매일 돌리기\n물걸레 매일 밀기\n분리수거 담당\n\n\n딱 여기까지예요.\n정리 할 줄 모르고 요리할줄 몰라요;\n'], ['침대정리 식사후 자기자리정리 쓰레기 분리수거 쉬는날 청소기정도 하네요ㅎㅎ'], ['제 기준으론 많이 하시네요\n\n전 그냥 도와주면 고맙고 집안일은 내 몫이다 생각해요'], ['일할때는 안하고 백수되니 잘하네요'], ['그냥 딱 정해놓지는 않고 신랑퇴근전까지는 제 몫 퇴근후나 주말엔 신랑이해요 그러니 보통 요리는 제가하고 그 외는 신랑이요.. 주말엔 반반..'], ['맞벌이라 화장실청소 각종가전청소는 남편 전담이고 나머지는 거진 반씩 나눠서 해요 저는  빨래는 거의 제가  하고요'], ['요리빼고는 거의 다 해요.물론 저도 같이 하구요. 할수있는사람이 하고 있어요. 거의 남편이 전담으로 하는건 분리수거,일쓰, 음쓰버리기,화장실청소, 아이들데리고 놀이터가기,아이들목욕.\n 같이하는건 청소기밀기,설거지,빨래 정도 되네요ㅎㅎ'], ['저흰 맞벌이인데 저질체력에 둘째가지고있고 몸이 안좋아서 남편 90% 저 10% 대신 깨끗하게 치우고 살진않고 몰아서 청소하고 합니다. 요새 건조기랑 식세기샀는데 신세계예요. 남편 완전 만족중 ㅎㅎ'], ['음쓰버리는거 화장실청소 걸레질 결혼 10년인데 한번도 해본적 없어용~ 설겆이담당,막내 목욕도 신랑 담당 ㅋㅋ케'], ['매일저녁설거지 아침에 장난감 정리,분리수거버리기. 음식물버리기. 애기 유치원등원. 고기굽기.간단요리 해요~'], ['저도 망'], ['저는 그냥 제가 시켜요 남자들은 시켜야되나봐요 ㅠ 쓰님도 그냥 부리세요 어쩔수없어요']]
    
    7530
    제일 싫은 집안일은? 저는 설.거.지에요........도돌이표같은 느낌..?전 워킹맘인데 직업 특성상 주말 중에 하루는 출근하는 편이에요.근데 이번주 금토일을 쉬었어요.금요일에 쓱배송으로 이것저것 시켜서 금토일 중한끼빼고 집밥을 먹었어요. (왕감동.....)오징어 세마리 사서 오징어무국, 오징어볶음, 오징어간장볶음 (손질할때 온갖 인상 다 찌푸리고..ㅠㅠ)계란말이, 오뎅볶음, 앞다리 사서 제육볶음, 만둣국, 짜장면, 소세지볶음, 유부초밥, 황태국, 참치미역국.....3일동안 한거에요.냉장고에 반찬 만들어놓고 먹는 타입이 아니라 그때 그때해요 ㅠㅠ (냉장고에 들어가면 손 안대는 신랑^^)저는 정말 대단하다고 생각해욬ㅋㅋㅋㅋㅋㅋㅋ배달을 주로 시켜먹는 제가... 대견해요 ㅠㅠ근데 설거지가.. 해도해도 계속 나오고 ㅠㅠ저는 설거지하는 시간이 너무 아까워요 헝 ㅠㅠㅠㅠㅠ아이랑 조금이라도 더 놀아줄 수 있는데..식기세척기는 손안대도 알아서 해주나요?맘님들은 제일 싫은 집안일이 뭔가여..저 설거지 극혐이에요마무리하고 대자로뻗었어요
    
    [['저도 설거지가 제일 거지같고 하기싫어요... 그리고 빨래 다된거 개는것도 너무싫어요ㅠ 귀찮아요...'], ['하...저도 거지같아여...... 빨래든 뭐든 다 괜찮은데 설거지 ㅠㅠㅠㅠㅠ 악몽 ㅠㅠ'], ['식세기 넘 사고싶어요ㅜㅜ'], ['저도요ㅠㅠ 진짜 신세계인가여 ㅠ'], ['화장실청소요 방충망창틀도 싫고 힘들어요ㅜ'], ['ㅠㅠㅠㅠㅠㅠㅠ 아 창틀도 있었네요'], ['전 빨래 개는거 까진 괜찮은데 제자리 찾아 넣는게 제일!!!!싫어요..ㅜ'], ['2222222제발 누가좀 넣어줬으면ㅋㅋㄱㅋ'], ['33333 저두 이거요!! 이건 기계도 없고 넘 싫어요 ㅋㅋㅋ'], ['444444 저도 이거요ㅋㅋㅋ 개는거야  티비보면서 사부작 사부작 하겠는데 이리저리 자리찾아 넣는게 넘 귀찮아서 어쩔땐 거실에 개어논 옷들이 묵혀있을때도 있다는ㅋㅋ'], ['5555 남편있을때만 개요.. 개켜놓으면 넣어줘요ㅋㅋㅋㅋㅋ'], ['666666 오오오오오!! 이렇게 동지들이 많다니 ㅋㅋㅋㅋ 저두 제자리 넣는거 싫어해서 남편보고 넣으라고 거실에 늘어놔여 ㅋㅋㅋ'], ['와우 격하게 공감요! 개놓고 걍 자면 남편이 넣어놓을때도 있고 아닐때도 있고ㅋ'], ['악.. 저만 그런줄알았어요 !!!! ㅋㅋㅋㅋㅋㅋ'], ['2222'], ['앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 댓글 너무웃겨욬ㅋㅋ'], ['격공이요ㅠㅠ옷 빨고 건조기 돌리는 것 까지는 괜찮은데 개놓기 귀찮아서 맨날 건조기 안에 두고 꺼내입어요ㅋㅋㅋㅋㅋ'], ['저도요! 귀찮아도 잔소리듣기싫어서 개놓고 안볼때 그냥 농속에 쳐박아놔요! 나만 찾을수 있어서 이것도 언젠가 정리를 해야할듯. . .ㅜㅜ'], ['이거 적으러 들어왔어요ㅠㅠㅠ'], ['ㅋㅋ 저같은 분들이 많으시네요\n저도 제자리 찾아 넣는게 넘나 싫어서 그대로 둔적도 있어요 ㅋ'], ['와 대박 저도요 ㅠㅠㅠ ㅋㅋㅋㅋㅋㅋㅋ'], ['저 음식하는거요!!!\n너무 시러요 ㅠ\n화장실청소도 설겆이도 다다 제가 할 수 있어요.\n음식하는거보다 뒷정리가 전 더 낫다는 ㅋㅋㅋㅋㅋ\n식사요정이랑 같이 살았으면 좋겠어요 ㅎㅎ\n아이 낳기 전에는 그래도 요오리 제법 했던 것 같은데 이젠 정말 시르네요 ㅠㅠ'], ['저두요 ㅠㅠ 완전 음식하는거 지짜 너무싫어요 ㅠㅠ'], ['맞아요.... 음식하려면 맘먹고 하는 스타일..ㅋㅋㅋ'], ['전 다시름요ㅜㅜ'], ['그러네요.. 좋은게 없네요ㅠ'], ['집안일 3글자가 극혐...'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ \n끝도 없고 답도 없는 ㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 앜ㅋㅋㅋㅋ'], ['화장실청소,설거지요'], ['언니 댓글 보고 설거지라고 고쳤어요 ㅋㅋㅋㅋㅋㅋㅋ 받침 틀리는거 극혐인데 내가.... ㅠ_ㅠ'], ['응? 왜? 나 틀려써??'], ['아뇨 제가 설겆이라고 썼어욬ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['어릴땐 나도 설겆이라고 배움 ㅋ ㅋㅋ 아 나이 티나써..\n모르는 개 산책 이런거 아님 아무도 몰라.. 티안나 ㅋㅋㅋ'], ['앜ㅋㅋㅋㅋㅋ 갑분 나이 현타 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], [''], [''], ['모든 집안일이요..'], ['그쵸.. 전 그중에 설거지가 젤 시러여ㅠㅠ'], ['전 안싫은데 하루에 1번만 해요.\n화장실청소가 싫음'], ['하루에 한번 몰아하기.. 제가 잘해요 ㅋㅋㅋㅋㅋㅋㅋ\n근데 이번엔 한끼먹고 하고 한끼먹고 했더니 정신 나가겠어요 ㅋㅋ'], ['요리요;;;'], ['요리도 싫죠.. 그쵸ㅠㅠ 맞아요'], ['전 바닥청소요.... 거실 전체에 폴더매트라 다 들고 틈까지 닦아야해서...ㅜㅜ'], ['아.... 폴더매트 사이 끼는거 ㅠㅠㅠㅠ  저 그래서 아이 어릴때 바꿨어요ㅠㅠ'], ['설겆이ㅋㅋㅋ저는 설겆이가 싫어영'], ['역시 ㅋㅋㅋㅋ 저랑 같으세여>_<'], ['ㅋㅋㅋㅋ악 ㅋㅋㅋㅋ진짜 너무 싫어여ㅋㅋㅋㅋ'], ['빨래 개서 넣기요ㅠㅠㅠ 아 진짜 소오오오름돋게싫어요ㅠ'], ['앜ㅋㅋㅋㅋㅋㅋㅋ 저는 설거지가 소름돋던데.... 댓글보다보니 빨래 개서 넣기 싫어하는분들이 많네요 ㅋㅋㅋㅋㅋ'], ['저도 설겆이가 제일 시러요 ㅠㅠ  참다참다 요즘 식기세척기 검색중이예요 ㅠ'], ['진짜 식세기 사야할판이에요 ..'], ['저도 집안일은 다~~~전부다~~~싫어요ㅜㅜ십년을 넘게 해와도 늘지도 않아서 더 하기 싫은가봐요ㅋ'], ['맞아요 ㅠㅠㅠㅠㅠ 저도 다 싫은데 그중에 고르라면 설거지가 제일 싫더라고요'], ['저는 설겆이가 제일 좋...핸드폰 거치해서 드라마 보면서 하거든요... 저는 화장실이 젤 싫어요'], ['옴마 배우신분ㅋㅋ 배워감당ㅋㅋ'], ['자동차에 붙이는건... 자꾸 떨어져서ㅋㅋㅋ'], ['오 ㅋㅋㅋㅋㅋㅋㅋ 저는 노래틀고해요 ㅋㅋㅋ'], ['저는 닦는거요 먼지도 방바닥도 설거지도 닦는건 다 귀차네요0ㅜㅜ'], ['아... 전 그래서 놔버려요. 청소기만 돌리기도 하고.. 바닥 닦는건 귀차나서 몇일에 한번 ㅠㅠ'], ['저도 집안일 진짜 못하고 정리도 못하는데ㅠ 그중에 설거지가 진짜 제일 싫어요ㅠㅠ 지금도 싱크대에 설거지..,  아 하기싫어요ㅠ'], ['그쵸ㅠㅠ 제일 극혐 ㅠ 너무싫음 ㅠㅠㅠ'], ['저도 설겆이요..진짜진짜 싫어요. 청소랑 빨래는 어떻게라도(로봇청소기)하겠는데 설겆이는  식세기있는데도 진짜 하기싫어서 미뤄둡니다.ㅎㅎ'], ['진ㅁ자싫어욬ㅋㅋㅋㄴㅋ 식세기 있으면 편한가요?'], ['정작 식세기는 저보다 신랑이 더 자주써요. ㅎ 암튼 전 설겆이 NO!NO! ㅎㅎ'], ['집안일 너무 좋아하는데 ..빨래 넣는 건 정말 싫어요'], ['빨래넣기 ㅋㅋㅋㅋㅋㅋㅋㅋ 싫어하시는분 진짜 많으시네요'], ['그 묵은 물때낀거 청소하는거요\n그릇놓는 철제나 가스렌지 닦기등ㅜㅜ\n전 아직도 가스라 청소하기 귀찮네요..\n오히려 설겆이 화장실청소는 즐거운마음으로해요;'], ['아..... 저도 즐거운 마음을 갖도록 노력해봐야겠어요 ㅠㅠ'], ['음식하는거요..ㅋㅋ 다른것들은 그남아 할만한데..정말 음식하는거 싫어요.'], ['음식하는겈ㅋㅋㅋㅋㅋㄴ 맞아요 세상귀찮..'], ['집안일 자체가 너무싫어요 ㅋㅋㅋㅋ 차라리 야근하고 주말에 두탕뛰라면 그걸하겠어요 ㅋㅋㅋ 직장일은 해도 집안일은 너무싫어요.. 전업주부는 저랑 정말 안맞아요.'], ['맞아요 제가 딱 그거에요!!!! 선택할 수 있다면 일을 하겠어욬ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['설거지랑 손빨래요\n행주빨기 걸래빨기 어려워요'], ['손빨래요? ㅠㅠ 전 손빨래 해본적이.....\n행주도 일회용쓰고 걸레는 안쓰고 물티슈만 써요.. 그러고보니 저 진짜 게으르네여;;'], ['밥하는거 집안일이면 밥하는거요~'], ['아...... 그 큰 일이 있었죠.. ㅠㅠ'], ['설거지랑 음식하는거만 괜찮고 모든게 싫어요~설거지도 양많으면 싫어요 ㅠ청소가 제일 싫기는 해요 열심히 해도 티가 안나니ㅠ'], ['ㅋㅋㅋㅋㅋㅋ앜ㅋㅋㅋㅋ 설거지랑 음식만 괜찮으시다니 대단해요!!'], ['음쓰 처리요... 유일하게 못하는 집안일이에요..'], ['헉 진짜요? 전 그게 젤 쉽던데ㅠㅠ'], ['식세기는 설거지 짜증지수를 아느정도 카바가 되요.\n개수대가 그릇으로 넘치게 쌓여도 아 난 식세기가 있지? 하면서 원래 미루던겅 20프로정도 미루고\n갓 세척 하고 나온 따끈하고 코팅된 그릇들도참 예쁘고 이뻐요 ㅎ'], ['우와....... 식세기 질러야하나봐요 ㅠㅠ'], ['전 머리카락 치우기요...\n저나 딸 아이나 길어서 ㅜㅜ'], ['아... 전 청소기 슉슉 돌리면 끝나니까 차라리 그게 낫더라고요ㅠㅠ'], ['저는 빨래 개는게 제일 하기 싫어요.'], ['아.... 저는 그건 좀 낫더라고요ㅠㅠ 설겆이는 온몸이 다 힘들어요\n다리도 어깨도 손가락도 팔도 다힘듬 ㅠㅠ'], ['저는 화장실청소랑 바닥 물걸레청소요ㅠㅠ 청소기는 슥슥 쉬운데 물걸레랑 스팀하려면 진짜 맘먹고해야하는것같아요ㅠㅠ 설거지도 싫었는데 요새는 넷플릭스로 워킹데드 보면서 설거지해서 시즌1부터 지금 7까지 왔네요 어느새 설거지 즐기고있어욬ㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋ우왘ㅋㅋ 저도 노래부르면서 하렵니다ㅠㅠ'], ['남편ㅅㄲ 밥 차려주는거'], ['흐억 ㅠㅠ'], ['저는 빨래 개키고 정리하는 거요 ㅠㅠㅠㅠ 하아 정말 싫어요 ㅠㅠ'], ['ㅇ ㅏㅠㅠㅠㅠ 저는 설거지랑 빨래 선택하라하면 빨래요ㅠㅠ'], ['저는 설거지랑 화장실 청소요 ㅜㅜㅜ'], ['저랑 같으세여 ㅠㅠ'], ['저도 설거지.. 제일 효자는 세탁기... 근데 식세기 사도 냄비며, 설거지 품목들이 너무 다양해서 넣는것도 일이고 또 꺼내는 것도 일일것 같아. 고민중이네요.'], ['아.. 하긴 넣고 꺼내기가 ㅠㅠ 저도 고민해봐야겠어요'], ['전 빨래개는것까진좋은데 이걸 각각 가져다정리해놓는게 너무싫어요ㅠ'], ['ㅠㅠ 전 후다닥 갖다놓으니 괜찮던데ㅠ 설겆이는 너무 오래걸려요 ㅠㅠ'], ['남편시키 밥차려주는거요,.  흥칫뿡..ㅠㅠ']]
    
    9291
    집안일 빨래개고 손수건 빤거 널고수건돌리고 젖병닦고 열탕소독 끝!오늘은 쏘야두해야되고설거지.빨래널기도..지율이가 아침부터 책을 다 꺼내놔서다시 정리해야되는데??오늘은 할게 좀 많네요??
    
    [['젖병소독열탕으루 꼭 다하세요??'], ['지금 소독기 고장낫어여ㅠㅠ엊그제갑자기안되서저나햇더니 as해야지만된다고하더라구여ㅠㅠ'], ['언제 쉬시는거에요?ㅎ 계속 바쁘신것 같아요'], ['ㅋㅋㅋㅋㅋㅋㅋㅋ남편퇴근하면서 저도좀쉽니다!!!!!'], ['읭읭~~ 할꺼많은날은....다 내일로!!!!!! 누워버려요!!!!!! 귀찬하!!!!!!!!!!!!\n(맘님홧팅요ㅜㅠ마음의 소리가 좀 껏쥬ㅠ)'], ['앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ읽으면서 느낌표에이입되섴ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ빵터졋어옄ㅋㅋㅋㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋ앜ㅋㅋ나 맘님 반응 넘 좋아욬ㅋㅋㅋㅋㅋㅋㅋㅋ앜ㅋㅋㅋ맘님하고 소통하는거 재미져요~~??'], ['그런가용?!!\n절좋아하시는군여...\n부꾸롭게....'], ['고생이시네여ㅠㅠ집안일은끝이없어여ㅠㅠ저두장난감닦아야하는데ㅠㅠ'], ['장난감은남편이닦는걸로..'], ['으악  오늘도 여전히 집안일 한가득이구나ㅜ'], ['왜끝이없지..'], ['아 나 스트레스 ㅜㅡㅜ'], ['오늘 최악이얌...?ㅠㅠㅠ'], ['울엄마..내가방 택시에 두고내림????????????'], ['헐........................대박\n어뜨케...........................\n택시찾앗어?'], ['아니 못찾앗지ㅋㅋ다행히 지갑은없긴한데\n가방이고 뭐고 다 새거라ㅜㅡㅜ\n현금내서 찾지못한다네\n넘 열받아 죽것다ㅋㅋㅋㅋ'], ['헐....\n이래서 택시에서는 무조건 카드를 ㅠㅠㅠㅠ\n어머님 저번에 스벅도그렇고 왜그러신댜...ㅠㅠㅠ\n언니속상하긋다'], ['스벅ㅋㅋ갑자기 떠오르노????????동서한테 선물받은 기저귀파우치도잇는데ㅋㅋ와  미칠꺼같다ㅋㅋㅋㅋ컨트롤이안되'], ['ㅜㅜㅜ안그래도오늘힘들아서 더화날거같다ㅜㅜ'], ['응응ㅜㅡㅜ진짜 미쳐버릴꺼같애'], ['오늘 왜케 힘든날이거여 ㅠㅠ'], ['넘 힘든데\n주부놀이중이야??????'], ['아 난 반찬만들어야되는데귀찮아죽겟다'], ['아침부터바쁘네 ㅜㅜㅜㅜ 화이팅'], ['갑자기급졸려온다ㅜㅜ'], ['아궁.. 바쁘바쁘네요..\n저거  어디꺼예요? 젖병 걸오놓는거?'], ['저두잘몰라용..당근으로 지율이아기때산거라..??'], ['아.. 그러시군유,'], ['많이건조할수잇어서좋은거같아여!!'], ['쏘야 하시나유?? 소독기 고장ㅠㅠ 불편하시겠어유 ㅠ'], ['쏘야해야쥬..넘기차나여ㅜㅜ'], ['전 낼 장보러가서 반찬 생성 해야해용'], ['오오오~~~\n저희는 주말에 장볼거같기도해요 ㅎㅎㅎ\n아직먹을게많아서!?'], ['ㅎㅎ저흰 읎어용 ㅎㅎ'], ['전 이제 완전다비우고사보려고용ㅜㅜ'], ['조아유 조아유 ㅎㅎ 저희집 냉장고'], ['뭐가 없쥬 ㅋㅋ 냉동실고 비우고 낼 장보러 가요 ㅎㅎ 오늘 저녁 국수 입니당 ㅎㅎ'], ['우와 진짜깔끔하네영!!!\n저희집은....뭐없는데도그득그득 ㅠㅠ'], ['전 다버려용 ㅠㅠ'], ['버리기엔 산지얼마안된것들이라 ㅠㅠ'], ['아아 얼마안된건 드셔야쥬 ㅠㅠ 전 진짜 냉털 탈탈다했네유 ㅠㅠ'], ['부러워여ㅠㅠㅠ저도 냉털다하고나면  인증한번해보고싶네옄ㅋㅋㅋㅋ'], ['소독기고장나셨다니ㅜㅜ 언능고쳐야는디ㅜ'], ['5만원이라고해서 지금고민중이여...ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['우유병 열탕소독 하는 것도 번거로울 것 같아요 ㅠㅠ'], ['괜찮은듯 안괜찮은듯하네여ㅠㅠ'], ['안괜찮을 것 같아요 ㅠㅠ'], ['그래도 밤에만 젖병쓰는거라서 아침에 한번만하면되용!!ㅎㅎ'], ['오? 낮에는이유식 먹나용ㅋ'], ['지태는 60일도안되서..ㅋㅋㅋㅋㅋ\n직수합니댜~'], ['아아 둘째는 신생아였네용! 한참 귀엽겠어용ㅋ'], ['으...신생아아닌거같아여..너무무거워여...ㅠㅠㅠ'], ['ㅠㅠ 백일도 안됐는데 ㅠㅠ 아가 무게가 생각보다 무겁다더니 정말 그런가봐요'], ['지금6키로넘는거같아여ㅠㅠㅠㅠㅠㅠㅠㅠ손목이너덜너덜 ㅠㅠㅠ'], ['헉 정말 금방금방 자라네요....'], ['진자 넘 많이먹고 너무빨리커여ㅠㅠㅠ'], ['핳 잘먹고 잘크는 건 그만큼 건강하다는 거긴 하겠지만.... 엄만 힘들죵 ㅠㅠ'], ['ㅋㅋㅋ손목이 안그래도 유리손목인데 지금은 갈대가된거같아여ㅠㅠ'], ['흑 남의 얘기가 아니네요 ㅠㅠ 저도 손목 한 가늚 하거든요 ㅠㅠ'], ['전 터널증후군에 건초염...\n시술도많이받아서 또 마비처럼 못움직이면그때는수술해야된다고하더라구여ㅠㅠ'], ['헐 ㅠㅠ 그럼 어서 스마트폰을 내려놓으세요'], ['ㅋㅋㅋㅋㅋㅋㅋㅋ그래서 무리안가게하고잇슴다!!!!!!!!!!!!1'], ['손목보호대도 잘 쓰시구요!!'], ['손목보호대는 자주오래하면안좋대서 진짜너무아플때만합니다ㅠㅠ'], ['에고ㅠㅠ 또 자주하면 안좋군요'], ['네넨!!맘님도 다복이낳고너무많이하지마세요!!'], ['전 지금 허리가 아파서 복대할까 했는데 그것도 오래하면 안 좋다고 그래서 고민이에용'], ['마자여ㅠㅠ복대가인위적으로잡아주는거라서 허리힘더빠지게하는거라고하더라구여ㅠㅠ'], ['뭐하나 쉬운 게 없네용 ㅠㅠ'], ['그쳐....임신.츨산 진짜 대단한일하는거예요 우리!!'], ['남편은 뭐 하나 하는 게 없으니 살짝 억울하긴 해요ㅜㅜ'], ['바뀌는것도없고ㅠㅠㅠㅠ너무한..'], ['소독기고장났어??그김에 뜨거운열탕소독 제대로 되는구만 ㅎ 나더 책이랑 장난감방 정리해야하는데..해야지하면서...손이안간다잉 ㅜㅜ'], ['울집은 장난감은진짜별로없는듯...ㅎㅎㅎㅎ'], ['이사오면서 장난감방을 만들어주니...감당이..애초에 만들지 말았어야해 ㅠㅠㅋㅋ'], ['지율이도여기이사오면서만들어줬엇는데 다팔앗닼ㅋㄱㅋ'], ['둘째가딸이얐음 팔았겠지만...그대로..물려줘야할듯...ㅋㅋㅋㅋㅋㅋ반이상이 고장났지만ㅋㅋㅋ'], ['앜ㅋㅋㅋㅋㅋㅋㅋ난이번에정리병올라서고장난거다버렷더니 반도안남음..ㅋㅋㅋ'], ['그 정리병..나한테 넘겨줘봐 ㅜㅜ'], ['앜ㅋㅋㅋㅋㅋㅋㅋ지금정리병떠낫는데 너한테안갓뉘!!이자쉭어디루간거지 ㅋㅋㅋㅋ'], ['핫싀...어디간거야..ㅠㅠ-ㅋ'], ['정리병다시찾아오면그때너한테가라고꼭말해줄게 ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['제발..나 진짜 해야지하면서..장난감방문열기가 싫어..ㅠㅠㅋㅋ'], ['장난감정리할때박스들고가..\n그박스꽉차서나올거얔ㅋㅋㅋㄱㅋ내가그랫거든ㅋㅋㄱㅋ'], ['다재활용텅으로 ㄱㄱㄱ????'], ['우리동네 저런거주워가는할미잇어서 할무니다드림ㅋㅋㅋ'], ['어제 몇개버렸더니 울고불고 ...'], ['그래서없을때해야하는..????'], ['오늘 고생했넹!'], ['아침부터 집안일이쫌많앗엇지..ㅋㅋㅋ'], ['언닌 쉴틈이 없겠다 ㅠ'], ['오늘은 진짜낮잠한번을안잣네...'], ['대단하다 진짜 ㅜ'], ['근데 낼은 낮잠도자고해야지ㅠㅠ너무조금잣어..분명졸릴거야 ㅋㅋㅋ'], ['틈틈히 자ㅠㅠ 언니 몸 생각해야지 ㅜㅜ 몸 상한다 ㅠㅠ'], ['그럼그럼!!졸리면바로바로자야지ㅠㅠ'], ['마쟈마쟈ㅠㅠ졸릴땐 잠이 최고야'], ['세상에 잠만큼중요한것은없지!'], ['마쟈~ 잠이보약이라잖아 ㅠㅠ']]
    
    9476
    집안일 중 제일 하기 싫은 것 갑자기 신랑이 뜬금없이 집안일 중 뭐가 젤 싫냐고 물어보는거예요... 전 설겆이라고 말했는데~ 다른분들은 집안일 중 뭐가 젤 싫으세요??? ㅋㅋㅋㅋㅋㅋㅋㅋㅋ
    
    [['저도 설거지요~~~!'], ['설겆이는 하루에 몇번이고 셀수없이 많이 해야되서..... 힘들어요'], ['저도 설겆이요 ㅠㅜ 모아서 하게 되네요'], ['해도해도 끝이 없는고 티가 안나는게 집안일이지만 그중에서도 젤 자주해야되는게 설겆이라....'], ['설거지랑 빨래개기요~ㅎㅎ'], ['저도 그두개가 젤 싫지만 그중 또 고른다면 설겆이요 ㅋㅋㅋㅋㅋ'], ['설거지 2222222222222'], ['ㅠㅠㅠ 무한반복이야... 끝이없어'], ['진짜 너~~~무 싫지..하'], ['화장실청소 ㅜㅜ싫어요ㅋ빡빡문질러도 티가 안나성..??'], ['서방시키 ~~'], ['ㅠㅠㅠㅠ 맞어요맞어 그것도 티가 왜케 안나는지.... 했다는건 냄새로만 알아요ㅋㅋㅋㅋㅋ'], ['로또 밥상겸 술상차리는거요 ㅋㅋ'], ['전 음식하는 이유는 신랑밥차려쥬기위해서 하는뎁...;ㅋㅋ 밥에 아주 진심가득이시거든요...'], ['진심이라 ㅋㅋㅋ\n저도 진심으로 차려줘요\n제 진심 ㅋㅋ'], ['전 진심이 아니에요 ㅋㅋㅋㅋㅋ 강압에 의한ㅋㅋㅋㅋㅋㅋ 안차려쥬면 일안간데요 ㅠㅠㅠㅠㅠ'], ['일가지마라해요\n밥 안차려준다하고요 ㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋ 우리세식구 손가락 빨수없어서 ㅠㅠㅠㅠㅠ'], ['아무튼 이집저집 다 애기 한명은 더 키운다 봅니다 ㅋㅋ'], ['맞어요ㅋㅋㅋㅋㅋ신랑이 아니라 큰아들입니다..... 아주 아주 큰아들ㅋ\n등치도크고 머리도크고 키도크고 ㅋ'], ['저는 요리하는거요ㅋ\n의외로 설거지는 좋아합니다만..ㅋㅋㅋㅋ'], ['요리는 즐기지만 설겆이가 너무 많아요.... 초보라 그릇을 많이 사용하는 이유기도 ㅎㅏ지만 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['설거지한 그릇 정리하는거요......ㅋㅋㅋ'], ['맞아요맞어!! 정리도 ㅠㅠㅠㅠ 예전에 했지만 이젠 걍 마를때까지 ㅋㅋ 놔둬요 ㅋㅋㅋㅋㅋ포기'], ['식세기 사주시려나봅니다~^^'], ['진짤까요???? .... 내심 기대중 ㅋㅋㅋㅋ'], ['전 빨래개서 제자리 갖다놓는거요ㅜㅜㅜㅜ'], ['아이랑 같이 하면 시간이 아주 오래걸리지만 그것도 일종의 아이와의놀이로 생각해서 그건 좀 덜한거같아요~'], ['2222격한 공감요\n뽀나동생 오랜만이야~잠깐 인사 ^^'], ['222222222222222ㅎ'], ['저두 빨래개서 정리하는게 젤루 시르네유^^;;;'], ['누가누가 빨리 건조기에서 꺼내나 ㅋ 뭐...내기도하고 색깔별로 모으기도하고 아이꺼 엄마꺼 아빠꺼도 따로따로 모으기 이런 놀이를 하면서 그냥 저냥 하고있어요~'], ['설거지요.'], ['ㅠㅠㅠ 식세기가 필요합니다'], ['우리집 설거지는70프로 신랑이 하기에ㅋㅋㅋ전 빨래널기 개키기..늠 싫어욧ㅋ'], ['부럽습니다!!!! 저희 신랑은 집안일은 아애 안해요~ 다만 애는 좀 잘 봐주니.... 그것만해도 다행이다 싶기도하고 ㅠㅠㅠ'], ['설거지요 ㅎ\n빨래는 하루 한 번이나 미루면 이틀에 한 번하면되는데\n설거지는 미루면 다음 식사타임에 문제가..ㅜㅜ\n식기세척기 사 달라고 시위할려구요 ㅎㅎ'], ['그죠.... 저도저도ㅋㅋㅋㅋㅋ 식세기 사달라케야겠어요ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['식세기 완전 필요합니다ㅠㅠ'], ['이사가면 사준다켓는뎁.... 그게 언제가 될지ㅠㅠ'], ['다 싫어 ㅜ 특히 빨래개기 ㅋㅋ'], ['빨래 개주는 기계도 있다능디.... 해외에 있는뎁 100만원돈이나 한데욥 ㅍ'], ['헐 진짜 니가 사주믄 써보마 ㅍㅎㅎ'], ['언니가 내보다 부자믄서~ 그돈 다 언제다쓴데~ㅋㅋㅋㅋㅋㅋㅋㅋ'], ['빚이 많은거지 ㅋㅋ'], ['언뉘!! 빚도 능력이 있어야 가질수있어요~'], ['알았다 사달라 안카께 ㅋ 담주에 밥사주께 됐제 ㅋㅋ'], ['ㅋㅋㅋ뭐 먹지?ㅋㅋ 담주 비워두겠슴돠 ㅋ'], ['오이야~~~'], ['저는 음쓰버리는거요.. 너무 귀찮아요ㅜ'], ['ㅠㅠㅠ안귀찮은 집안일이 읍어요ㅠㅠㅠㅠㅠㅠㅠㅠ 왜케 할일이 많은지'], ['요리요\n차라리 정리만 할래요'], ['전 요리는 하고나면 맛있다고하거나 아이가 잘먹음 뿌듯해서 할만한뎁... 예전 직업이 음식만드는 일이라 긍가?그건 몸에 베어있지만.... 오히려 배달이나 그런거 시켜먹는게 잘 안되요ㅠㅠㅠㅠ 한달에 한 2~3번 될까말까?'], ['저는 그냥ㅇ 청소자체가 넘 귀찮아효....ㅋㅋㅋㅋㅋ'], ['청소가 젤 티가 안나긴해요ㅠㅠㅠㅠ 청소해도 머리카락은 꼭 한두개씩 보이고ㅠㅠㅠㅠ 마음같아선 거실에 아무것도 안놔두고싶어요~'], ['걸레빨기가 제일 싫어요 설겆이하는거 제일 좋아해요 설겆이하다보면 기분이 좋아지더라구요 ㅎ'], ['오리날다님과 결혼을 했어야 하는데.........'], ['ㅎㅎㅎㅎㅎ'], ['저는 빨래개기요ㅋㅋ제일 귀찮아서~~'], ['맨날 몰아서 하기 ㅋㅋㅋㅋㅋ  흰거 빨았지만 정리는 색깔있는거 하기 ㅋㅋㅋㅋㅋㅋ'], ['음쓰버리기요 음쓰처리기좀 사도라켔는데 ㅡㅡ맨날 이사가서 사주께...이사는 언제가노 ㅎㅎㅎ'], ['맞아요!!!!뭐 해줘 뭐사죠 하면 ㅠㅠㅠ 이사가면 해준데요.... 도데체 이사는 언제가는건지.... 말뿐'], ['이사가쟈 이사가쟈'], ['ㅋㅋㅋㅋㅋㅋㅋ이사온지 1년도 안된집이라ㅠ 까마득합니다'], ['핫한집 이사했으니 넣읍시다'], ['자리가 안나와요.....'], ['저도요 ㅡㅡ싱크대 하는거말고 뒷베란다 따로빼고싶은데...헐..'], ['뒷베란다에도 설치가 가능해요?'], ['긴거요 제품이 여럿있던데 가루로 화분에 흙으로쓰고. 쓰레기통에 넣는제품요...갈아내리는거 말고요 다..장단점이있을텐데...안사주네요'], ['부지런히 알아보고 ㅋㅋㅋ 결제영수증만 청구합시다 ㅋㅋㅋㅋㅋㅋ'], ['오~♡굿 결제만 늦추면 배송이 늦어질뿐'], ['설거지랑 빨래요'], ['두개가 젤 박빙이네요~ 주부들은 다들 공감하는 부분인거같아요~'], ['그죠~평일에는 밥하는게제일싫더라고요~^^ㅎㅎ'], ['밥은 2번~3번이니 개안은데..... 설겆이는 왜 횟수제한이 없을까요?'], ['ㅎㅎ그죠~^^'], ['화장실청소요~진짜 하기싫어서 한번도 안해봤어요'], ['전 담엔 화장실 하나있는집 가자고 했어요~ 두개있으니 두배로 해야되서 싫다공 ㅠㅠㅠㅠㅠㅠ'], ['저는 화장실청소 젤 좋아하고 빨래개는거 넘무 시러용ㅠ'], ['화장실이 왜 좋으세요? 혹시 스트레스가 많으신가요??? 팍팍 문떼기 ㅋㅋㅋㅋ'], ['화장실 물기 한톨이라도 있고 더러운 꼴은 죽어도 몬봐요\n성격이라 눈만뜨만 화장실ㅎㅎㅎ\n하루 두번씩 두군데 총 4번  장실청소해요'], ['헉;;;;!!!!!!충격입니다 \n아이 목욕하는 날만 옆에서 청소해요 ㅋㅋㅋㅋㅋ 세제안쓰고 싶어서 개고생하지만 ㅋㅋㅋㅋㅋㅋㅋㅋㅋ'], ['빨래 정리....널고 개는건 할만한뎅 넣는거 너무 싫어요...ㅠ귀찮...'], ['빨래 다 해서 정리했는데 꺼낼때 흐트러트리는거 보면 딥빡쳐요!!!'], ['저는 빨래요 ㅌㅌ'], ['건조기도 나왔으니 좀 더 기다리다보면 뭔 기계가 나오겠죠 ㅋㅋㅋ'], ['ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ개비는게 너무시러요ㅠ'], ['전 빨래 개는거요ㅠㅠㅠㅠ젤 시러요 ㅠㅠ'], ['빨래도 살겆이만큼 많네요~'], ['설거지하는거 싫어서 이번에 식세기 샀어요~'], ['대박!!!!!! 부럽습니다 저희집은 놔둘자리가 안나와서ㅠㅠㅠㅠㅠㅠ 이사가면 해준데욥'], ['저는 요리좀 누가해주면 좋겠네요~~'], ['배달 고고 해요 ㅋㅋㅋㅋㅋ'], ['다른건 그래도 괜찮은데\n빨래 개어서 각자 옷장에 가져다 놓는게 제일 싫네요--;'], ['저희는 옷방안 건조기 넣어놔서 방안에서 모두 다 이뤄져서 동선이 짧아 할만해요~'], ['저 빨래개기요??\n설거지는 그래도 식세기이모님 모시니 낫네요??'], ['부러워요~ 언제쯤 전 이모님을 모실수 있을런지........ 생일 선물로 사달라칼지 ㅋㅋㅋㅋ 고민을 살짝꿍해봅니다'], ['전 화장실청소요 ㅠ.ㅠ'], ['해도 티안나죠 ㅠㅠ 전 세제안쓰니 허리랑 팔이 떨어져나갈꺼같아요~'], ['정리. ㅎㅎㅎ']]
    
    9490
    남편이 집안일 어느 정도까지 도와주시나요? 말그대로 남편이 집안일 어느 정도까지 도와주시나요? 저흰 설거지,세탁기 돌리기, 화장실청소,음식물쓰레기,종량제버리기,가끔 장보기,이정도 하는데 다들 남편께서 많이 도와주시나요? 궁금해서 여쭤봅니다! 남편도 궁금해 하더라구요! 알려주세요~~~
    
    [['많이 해주시네유~~~'], ['요리는 제가하고 도와줄려고 노력하는데 제가 됐다고해요 가끔 보조하는정도에요 많이 도와주더라구요~~~'], ['본문에다가 플러스 집청소도요~'], ['집청소할때 거실은 같이하고 방은 각자 따로 하네요~'], ['아하 전 전부다 남편이 해줘용 침대 먼지 닦는것도 해주드라구요 너무 좋아요ㅋㅋㅋ'], ['좋으시겠어요 ㅋㅋㅋ'], ['저희는 평일에는 제가 주말에는 남편이 해요~ :-)'], ['그렇군요'], ['세탁기도 돌려주고 청소기도 돌려주고 걸레질도 해주고 때 맞춰서 세탁조청소도 해주구요 분리수거도 해주고 음식이며 설거지 화장실 청소는 제가 하구요~ 방청소 해주니 저는 화장실 청소 담당인거죠 ㅎㅎ'], ['많이 도와주시네요 ㅎㅎ'], ['시켜야하죠~ 일단 모든 쓰레기는 정리하고 버려줘요 가끔 화장실청소 \n장은 뭐 목록 써주면 이것도 가끔요 설겆이는 1년에 한두번 할까말까 \n 잘못 길들였나봐여ㅠㅠ'], ['그러시군요ㅜㅜ'], ['많이 하시는 듯...'], ['많이 도와주기는해요'], ['전 외벌이라 제가 다해욤 ㅎㅎ'], ['그렇군요 ㅎㅎ'], ['전혀 안해요 자기가 사용한 물건 정리만이라도 하면 좋겠는데 그것마저도 흘리고 굴려서 결국 다 제몫이에요 잘 도와주시는 남편분들 너무 부럽답니다'], ['222222222'], ['힘드시겠어요'], ['손꾸락하나까딱안하는디요?\n개똥도내가치우고 ㅋㅋ'], ['2222222'], ['그런가요 ㅋㅋ'], ['2222 넘 웃프네유'], ['분노의 2 찍기ㅋㅋㅋ 이제그러려니합니다ㅋ'], ['저기에 음식도 만들어요~'], ['많이 도와주시네요~'], ['평일엔^^ 음식물버려주고, 분리수거해주고,\n주말엔^^ 애들씻겨주고, 설거지, 빨래,청소기, 걸레질도 해줘요^^ \n가끔ㅋㅋ 요리도 해줘요'], ['좋은데요 ㅋㅋ'], ['와~~남편분 마니 도와주시네요'], ['많이 도와주네요'], ['에휴....댓글보니 눈물나네요'], ['도와달라고 얘기해보세요'], ['손하나까딱안해요ㅋㅋㅋ대신 주말없이 일하니 이해합니다'], ['그렇군요'], ['다 같이해요~'], ['둘이하는게 좋은거죠~'], ['음쓰및 쓰레기 버리기,빨래돌리기,설거지,아이 목욕시키기,놀아주기 청소는 같이 정도요~'], ['많이 도와주시네요~'], ['시켜야 해요..ㅜ ㅜ 가끔은 붙박이장을 새로 들여놨구나 싶어요 ㅋㅋㅋㅋ후'], ['시켜서라도 하면 다행이죠'], ['비슷해요..  저흰. 맞벌인데 저정도 안하면 나쁜거죠 아침등원은 아빠가준비해서 등원시키고하원은 제가해요그외 아이는 백프로 제담당이구요분리수거 본인방화장실청소 식세기가 설거지를거의다하지만 그외 설거지랑 청소기밀기 빨래개기  장보기 음식도 주말엔 남편이해요..전 아이와 하루종일 놀아주고요 ㅜㅡ 외동이라서요'], ['많이 도와주시네요'], ['아뇨...  도와주는게 아니고 같이해야는거죠 우리나라는 인식자체가 잘못되었어요 자기밥은 자기가 해먹어야죠..육아까지 포함해서 집안일 비율이 저는 적어도 70 남편은 많아야 30인걸요  이것도 전 불공평하다고 생각해요 ...'], ['부럽습니다 손가락 1도 안움직이는 사람이랑 사는 사람입니다ㅠ 슬퍼지네요ㅠ 똑같이 밖에서 일 하는데~'], ['힘드시겠어요 힘내세요'], ['신랑분이 믾이 도와주고 계신대요 ㅋㅋ \n저흰 저녁이랑 주말 식사는 거의 신랑이 해요. 장보고 반찬만들고 ... 음쓰,쓰레기 버리고... 화장실 청소 하고 정리정돈 저보다 잘해서 그거 해주고 계셔요 ㅋ 전 설거지나 청소 세탁 정도 하나요?'], ['그런거요 ㅋㅋ'], ['설겆이 분리수거 등..모든 보이면 도와주려고 해요.. \n\n음식물쓰레기는 내가 전담해요.. 그냥 그건 내가 한다 합니다.. \n\n그래도 나 아프고나선 자신이 스스로 한다는게 기특하고 고맙습니다.. \n많이 도와주네요.. 신랑님 최고~~!!'], ['많이 도와줘서 고맙다고 가끔 한마디 해주네요~~~'], ['밥하는거 빼고 다해줘요... 그런데 제가 다시해야해요ㅡㅡ'], ['그렇군요'], ['설거지는 잘해줘요음쓰및종량제버리는건 본인이 담배땡기믄 솔선수범하네요....^^'], ['설거지 잘하는거도 대단한거에요'], ['안도와줘요ㅋㅋ\n가끔설거지나 밥한번씩해주고 저녁에 가게하다보니 낮에는자요 ????\n그래서 로봇청소기 두대 매일돌리고\n건조기도 사달라했어요??'], ['건조기 꼭 사달라하세요'], ['하루만빨래안돌려도 다음날 바구니 2개3개는기본이예요\n수건부터 아이옷 남의편옷 미챠요 ????'], ['꼭 필요하신거 같아요 건조기 화이팅요!'], ['네~^^\n이번에 꼭사고말께요 으쌰'], ['네~^^'], ['헉..저희 화장실만 전담이예요..쓰레기는 제가 시키면 하고요..시키는건 해요..설거지는 물을 넘 아껴서 안시켜요ㅜㅜ'], ['설거지하는데 물아끼면 안되는데 깨끗하게 할려면 충분히 헹궈야하는데 ㅜㅜ 시키는거 잘하셔서 좀 수월하시겠네요'], ['저희남편은 제가 모텔청소부인줄 알아요..ㅋㅋ 가끔 집안일합니다..싸우기싫어서 포기하고살아요. 식세기산지 1년반인데 어제시댁식구왔다고 돌려주더라구요. 세제어디에넣냐는말에 동서가 뜨악했죠..ㅋ'], ['식세기 사놓으시고 왜 안쓰시는거에요?'], ['남편이 처음 썼다구요.ㅋ 저는 계속써요.'], ['음식, 쓰레기버리기, 아이 목욕, 장보기 이런거 해요~  전 빨래 청소 하구요~ 전 시킨건 아니고 저런걸 좋아해서 그냥 자기가 하더라구요ㅎ'], ['많이 도와주시네요ㅎ'], ['외벌이인데 대부분 제가 다하지만 해달라는건 다해줘요.종류 안가리구요.'], ['그렇군요'], ['저희도 분리수거 청소걸레질 화장실청소 빨래개기 애들 등하원 씻기기 장보는건 거의 같이보거나 아님 사올것만 캡쳐해서 시키고요 설거지는 키가 커서 높이땜시 물바다가되서 어쩌다한번정도 시키는정도예요ㅎ'], ['잘도와주시네요ㅎ'], ['같이살땐 거의 같이했는데 이젠ㅠㅠㅋㅋㅋ 쉬는날만 와서해요ㅋㅋ'], ['주말부부세요?'], ['주말 아닌  5일제부부요ㅋㅋ 쉬는날이 주말도됐다 평일도됐다 그렇거든요ㅎ'], ['그렇군요 ㅋㅋ'], ['제가 음식 해놓으면 \n아이들 저녁차려주기, 설겆이\n아이들 머리말려주기, 운동화세탁\n음쓰,일반쓰레기,재활용ㅡ담배때문인듯\n빨래정리, 빨리널기ㅡ종종 같이 하고요\n\n지인들은 저한테 늘 이야기하죠~\n남편한테 잘해주라고!! \n잔소리 많이 듣습니다^^;;'], ['저랑 똑같은데요 저도 지인들이 남편한테 잘하라고 잔소리듣는데'], ['남편말이 본인아버지는 정말 손하나 안움직였다고.. 자기는 커서 그러지 말아야지라고 다짐했다고 지나가는 말로 했어요~ 많이 도와줘서 늘 고맙죠~~~'], ['좋은 남편분 만나셨네요~~~'], ['아이들한테 늘 최선을 다해줘서 감사해요^^ 전 그렇지 못해서;;'], ['저흰 제가 욕실청소나 쓰레기(음쓰포함)버리기, 분리수거 이런건 엉성해보인다고 신랑이 전적으로 하고 나머지는 같이해요.\n빨랫감갖다놨을때 많다싶다 느낀 사람이 돌리고 건조기돌리고 마르면 같이 개고. 청소도 같이하고.\n밥도 먼저 퇴근한 사람이 자연스럽게 하는 스타일이예요.'], ['좋은데요'], ['근데 장은 한번 시켰다가 안시켜요.\n남자들 대부분 그런건지 손이 왜케 큰걸까요.????'], ['맞아요 가면 필요한거외에 자꾸 뭘 잔뜩사와요????'], ['저희도 청소, 빨래, 설거지, 분리수거, 화장실청소, 냉장고청소...왠만한건 다 해요ㅋㅋㅋ\n코로나때메 목욕탕 못가서 딸들이랑 저 때도 밀어줘요????'], ['때까지 밀어주시군요 가끔저도 ㅋㅋ'], ['그래도 전 세신이모님이 좋아요 ㅋㅋㅋ'], ['이모님만 못하지요 ㅋㅋㅋ'], ['주말마다 분리수거,한달에 1번 화장실청소,1주일에 1번  청소기돌리기,  매일 애 목욕시키고 재우기 ㅡㅡㅡ 딱 요만큼요. 맞벌이고 아이 등하원 요리식사준비 설거지 빨래 주중청소 집정리정돈  일반쓰레기버리기등등 나머지는 다 제가해요  장보기만 같이 하고요~ 주중에는 아예 손가락 하나 까딱 안해서 애목욕이랑 재우기만큼은 무조건 남편이 하게 해요'], ['그러시군요'], ['모든쓰레기 버려주고 7세전까지 아이들 씻기고 재워주는 정도 였는데 애들이 크면서 이젠 아빠가 씻겨주는거 챙피하다고 하네요 ㅋ\n해달라는건 다해주는데..집안일할때 눈치껏 같이 해줬으면 좋겠어요ㅋ'], ['말안해도 알아서 해주면 얼마나 좋을까요 ㅋㅋ'], ['가끔 재활용쓰레기버려줘요. 그리고 국이나 찌개 끓여줘요. 그리고 없네요ㅜㅜ'], ['요리해주는 남편이 부럽네요ㅜㅜ'], ['맞벌이이고 각자 잘하는거 해요.\n제가 요리하면 남편이 설거지, 빨래너는건 제가하고  빨래개서 정리하는건 남편이, 주말청소때는 저는 청소기돌리고 남편은 스팀청소기 이런식으로요.\n'], ['저희랑 비슷한데요'], ['저희도 빨래,설거지,정리정돈,아이씻기기,쓰레기버리기,물끓여놓기, 주말엔 아침준비 및 아이랑놀기 전 늦잠  평일에도 아침 안먹고 출근하구요 \n전 청소기 돌리고 간혹 빨래개기 화장실청소 침대 청소 해요  먼가 저도 하는데 신랑이 많이 하네요 ㅎㅎ 뽀뽀해줘야 겠어요 ㅎㅎ'], ['좋은데요 뽀뽀 필히 해주세요 ㅎㅎ'], ['가족들생일에 미역국 끓이기,퇴근후 시키는것 사오기,어쩌다 한번 빨래개기, 어쩌다한번 설거지하기,어쩌다 한번 목욕하면서 욕실청소하기\n이렇답니다.ㅎㅎㅎ\n\n그래도 평소에 다른걸로 점수얻기에 집안일 같이 안해도  밉지는 않네요^^\n\n그리고 중요한것!\n남자가 집안일을 돕는게 아니라\n부부는 집안일,육아를 같이 하는거랍니다. 늘 그렇게 얘기하셔야 습관이 됩니다^^'], ['같이 도와가면서 잘하구 있어요^^'], ['전혀요']]
    
    


```python
drop_data = [270,1350,1999,1781,2390,2426,2715,2966,2973, 
             3547,3881,4174,6732,7112,7572,8162,863,885,1006,
            1047,1517,1796,2362,2388,3036,3415,3901,4044,
            4667,4829,5913]
df1 = df0.drop(drop_data)

```


```python
len(df1)
```




    4287




```python
testing = [np.log(i+1) for i in df1.com_num]
df1['com_num_norm'] = testing
```


```python
imp = []

for i in range(len(df1)):
    imp.append(10*((df1['com_num_norm'].iloc[i]-df1['com_num_norm'].min()) / (df1['com_num_norm'].max() - df1['com_num_norm'].min())))
df1['imp'] = imp

sat = []
for i in range(len(df1)):
    sat.append(10*((df1['com_senti_mean'].iloc[i]-df1['com_senti_mean'].min()) / (df1['com_senti_mean'].max() - df1['com_senti_mean'].min())))
df1['sat'] = sat


opt = []
for i in range(len(df1)):
    
    차 = df1['imp'].iloc[i] - df1['sat'].iloc[i]
    if 차 > 0:
        opt.append(df1['imp'].iloc[i]+차)
    else:
        opt.append(df1['imp'].iloc[i])
        
df1['opt'] = opt
```


```python
5**(1/3)
```




    1.7099759466766968




```python
sns.distplot([round(i**(1/3),3) for i in df1['com_num']]) #hist=False kde=False
```

    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    




    <AxesSubplot:ylabel='Density'>




    
![png](output_78_2.png)
    



```python
sns.distplot([np.log(i+1)/np.log(10) for i in df0['com_num']]) #hist=False kde=False
```

    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    




    <AxesSubplot:ylabel='Density'>




    
![png](output_79_2.png)
    



```python
plt.hist([np.log(i+1) for i in df0['com_num']],bins=30,alpha=0.8,color='gray')
```




    (array([360.,   0.,   0., 408.,   0., 270.,   0., 410., 249., 359., 234.,
            282., 171., 364., 305., 212., 182., 206., 120., 177., 132.,  78.,
             95.,  68.,  51.,  48.,  37.,  14.,  12.,  20.]),
     array([0.69314718, 0.82387963, 0.95461207, 1.08534451, 1.21607696,
            1.3468094 , 1.47754185, 1.60827429, 1.73900674, 1.86973918,
            2.00047163, 2.13120407, 2.26193652, 2.39266896, 2.5234014 ,
            2.65413385, 2.78486629, 2.91559874, 3.04633118, 3.17706363,
            3.30779607, 3.43852852, 3.56926096, 3.69999341, 3.83072585,
            3.96145829, 4.09219074, 4.22292318, 4.35365563, 4.48438807,
            4.61512052]),
     <BarContainer object of 30 artists>)




    
![png](output_80_1.png)
    



```python
plt.hist([np.sqrt(i) for i in df0['com_num']],bins=30,alpha=0.8,color='gray')
```




    (array([360., 408., 270., 410., 608., 234., 453., 364., 305., 212., 262.,
            186., 162., 105., 102.,  55.,  86.,  49.,  44.,  38.,  30.,  36.,
             18.,  21.,   9.,   5.,   6.,   7.,   3.,  16.]),
     array([ 1. ,  1.3,  1.6,  1.9,  2.2,  2.5,  2.8,  3.1,  3.4,  3.7,  4. ,
             4.3,  4.6,  4.9,  5.2,  5.5,  5.8,  6.1,  6.4,  6.7,  7. ,  7.3,
             7.6,  7.9,  8.2,  8.5,  8.8,  9.1,  9.4,  9.7, 10. ]),
     <BarContainer object of 30 artists>)




    
![png](output_81_1.png)
    



```python
plt.hist([i**(1/3) for i in imp],bins=30,alpha=0.8,color='gray')
```




    (array([360.,   0.,   0.,   0.,   0.,   0., 408.,   0., 270., 410., 249.,
            593., 453., 364., 305., 322., 212., 186., 177., 132.,  78., 112.,
             51.,  51.,  48.,  30.,  19.,   8.,   7.,  19.]),
     array([0.        , 0.07181449, 0.14362898, 0.21544347, 0.28725796,
            0.35907245, 0.43088694, 0.50270143, 0.57451592, 0.64633041,
            0.7181449 , 0.78995939, 0.86177388, 0.93358837, 1.00540286,
            1.07721735, 1.14903183, 1.22084632, 1.29266081, 1.3644753 ,
            1.43628979, 1.50810428, 1.57991877, 1.65173326, 1.72354775,
            1.79536224, 1.86717673, 1.93899122, 2.01080571, 2.0826202 ,
            2.15443469]),
     <BarContainer object of 30 artists>)




    
![png](output_82_1.png)
    



```python
plt.hist(imp,bins=100)
```




    (array([500., 537., 335., 497., 291., 418., 265., 332., 218., 243., 197.,
            224., 138., 137., 130., 136.,  92., 111.,  79.,  77.,  65.,  71.,
             56.,  75.,  43.,  54.,  37.,  41.,  41.,  46.,  29.,  26.,  24.,
             31.,  24.,  28.,  26.,  25.,  20.,  21.,  17.,  14.,   6.,  12.,
              9.,  17.,  10.,  12.,  10.,  10.,   4.,   7.,  11.,  17.,   6.,
              6.,   9.,   4.,   4.,   8.,   8.,   5.,   4.,   4.,   9.,   4.,
              6.,   3.,   0.,   5.,   1.,   1.,   3.,   1.,   1.,   2.,   0.,
              3.,   1.,   1.,   3.,   0.,   2.,   1.,   1.,   1.,   2.,   1.,
              0.,   0.,   2.,   1.,   1.,   0.,   0.,   2.,   2.,   1.,   2.,
             15.]),
     array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,
             1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,
             2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3. ,  3.1,  3.2,
             3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,  4.3,
             4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,
             5.5,  5.6,  5.7,  5.8,  5.9,  6. ,  6.1,  6.2,  6.3,  6.4,  6.5,
             6.6,  6.7,  6.8,  6.9,  7. ,  7.1,  7.2,  7.3,  7.4,  7.5,  7.6,
             7.7,  7.8,  7.9,  8. ,  8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,
             8.8,  8.9,  9. ,  9.1,  9.2,  9.3,  9.4,  9.5,  9.6,  9.7,  9.8,
             9.9, 10. ]),
     <BarContainer object of 100 artists>)




    
![png](output_83_1.png)
    



```python
plt.hist(df0['com_senti_mean'],bins=100)
```




    (array([ 29.,   1.,   0.,   1.,   0.,   0.,   4.,   1.,   1.,   1.,   2.,
              0.,  23.,   0.,   2.,   5.,  10.,   5.,   9.,   8.,  13.,   5.,
              8.,   9.,  10., 155.,  16.,  22.,  29.,  32.,  46.,  70.,  55.,
             98.,  76.,  70.,  63., 198.,  64., 139.,  87., 144., 118., 143.,
            126., 135., 117., 109., 108.,  86., 773., 116., 114., 145., 160.,
             99., 122.,  87., 117.,  77., 106.,  83., 190.,  65.,  83.,  62.,
            126.,  46.,  75.,  57.,  49.,  60.,  43.,  34.,  23., 178.,  23.,
             31.,  32.,  26.,  12.,  24.,  12.,  42.,  14.,  18.,   7.,  53.,
              7.,   9.,  10.,  12.,   1.,  21.,   7.,   4.,   4.,   3.,   0.,
            117.]),
     array([-2.  , -1.96, -1.92, -1.88, -1.84, -1.8 , -1.76, -1.72, -1.68,
            -1.64, -1.6 , -1.56, -1.52, -1.48, -1.44, -1.4 , -1.36, -1.32,
            -1.28, -1.24, -1.2 , -1.16, -1.12, -1.08, -1.04, -1.  , -0.96,
            -0.92, -0.88, -0.84, -0.8 , -0.76, -0.72, -0.68, -0.64, -0.6 ,
            -0.56, -0.52, -0.48, -0.44, -0.4 , -0.36, -0.32, -0.28, -0.24,
            -0.2 , -0.16, -0.12, -0.08, -0.04,  0.  ,  0.04,  0.08,  0.12,
             0.16,  0.2 ,  0.24,  0.28,  0.32,  0.36,  0.4 ,  0.44,  0.48,
             0.52,  0.56,  0.6 ,  0.64,  0.68,  0.72,  0.76,  0.8 ,  0.84,
             0.88,  0.92,  0.96,  1.  ,  1.04,  1.08,  1.12,  1.16,  1.2 ,
             1.24,  1.28,  1.32,  1.36,  1.4 ,  1.44,  1.48,  1.52,  1.56,
             1.6 ,  1.64,  1.68,  1.72,  1.76,  1.8 ,  1.84,  1.88,  1.92,
             1.96,  2.  ]),
     <BarContainer object of 100 artists>)




    
![png](output_84_1.png)
    



```python
plt.hist(imp,bins=100)
```




    (array([500., 537., 335., 497., 291., 418., 265., 332., 218., 243., 197.,
            224., 138., 137., 130., 136.,  92., 111.,  79.,  77.,  65.,  71.,
             56.,  75.,  43.,  54.,  37.,  41.,  41.,  46.,  29.,  26.,  24.,
             31.,  24.,  28.,  26.,  25.,  20.,  21.,  17.,  14.,   6.,  12.,
              9.,  17.,  10.,  12.,  10.,  10.,   4.,   7.,  11.,  17.,   6.,
              6.,   9.,   4.,   4.,   8.,   8.,   5.,   4.,   4.,   9.,   4.,
              6.,   3.,   0.,   5.,   1.,   1.,   3.,   1.,   1.,   2.,   0.,
              3.,   1.,   1.,   3.,   0.,   2.,   1.,   1.,   1.,   2.,   1.,
              0.,   0.,   2.,   1.,   1.,   0.,   0.,   2.,   2.,   1.,   2.,
             15.]),
     array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,
             1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,
             2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3. ,  3.1,  3.2,
             3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,  4.3,
             4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,
             5.5,  5.6,  5.7,  5.8,  5.9,  6. ,  6.1,  6.2,  6.3,  6.4,  6.5,
             6.6,  6.7,  6.8,  6.9,  7. ,  7.1,  7.2,  7.3,  7.4,  7.5,  7.6,
             7.7,  7.8,  7.9,  8. ,  8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,
             8.8,  8.9,  9. ,  9.1,  9.2,  9.3,  9.4,  9.5,  9.6,  9.7,  9.8,
             9.9, 10. ]),
     <BarContainer object of 100 artists>)




    
![png](output_85_1.png)
    



```python
plt.hist(sat,bins=100)
```




    (array([ 29.,   1.,   0.,   1.,   0.,   0.,   4.,   1.,   1.,   2.,   1.,
              0.,  23.,   0.,   2.,   5.,  10.,   5.,   9.,   8.,  13.,   5.,
              8.,   9.,  10., 155.,  16.,  22.,  29.,  32.,  46.,  70.,  55.,
             98.,  54.,  92.,  63., 198.,  64., 108., 117., 145., 118., 143.,
             85., 176., 117., 111., 106.,  84., 775., 114., 114., 147., 128.,
            131., 122.,  87., 117.,  77., 106.,  83., 191.,  64.,  58.,  87.,
            126.,  46.,  75.,  40.,  67.,  59.,  43.,  34.,  23., 178.,  23.,
             31.,  32.,  16.,  22.,  24.,  11.,  43.,  14.,  18.,   7.,  53.,
              7.,   9.,  10.,  12.,   1.,  21.,   3.,   8.,   4.,   3.,   0.,
            117.]),
     array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,
             1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,
             2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3. ,  3.1,  3.2,
             3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,  4.3,
             4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,
             5.5,  5.6,  5.7,  5.8,  5.9,  6. ,  6.1,  6.2,  6.3,  6.4,  6.5,
             6.6,  6.7,  6.8,  6.9,  7. ,  7.1,  7.2,  7.3,  7.4,  7.5,  7.6,
             7.7,  7.8,  7.9,  8. ,  8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,
             8.8,  8.9,  9. ,  9.1,  9.2,  9.3,  9.4,  9.5,  9.6,  9.7,  9.8,
             9.9, 10. ]),
     <BarContainer object of 100 artists>)




    
![png](output_86_1.png)
    



```python
df1[['imp','sat','opt']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>imp</th>
      <th>sat</th>
      <th>opt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>6.988421</td>
      <td>4.583333</td>
      <td>9.393508</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.346659</td>
      <td>2.875000</td>
      <td>5.818318</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.767343</td>
      <td>7.500000</td>
      <td>1.767343</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.585718</td>
      <td>3.690789</td>
      <td>13.480647</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5.995388</td>
      <td>5.875000</td>
      <td>6.115776</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df0)
```




    4864




```python
df0['testing']=testing
```

    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    


```python
sum(df0['imp']>df0['sat'])
```




    168




```python
sum(df0['testing']>df0['sat'])
```




    714




```python
df0['imp'].mean(), df0['sat'].mean()
```




    (1.5267674559323006, 5.638648519029073)




```python
sum(df0['imp']>df0['imp'].mean())
```




    1577




```python
(10/8.9), 1.1*(10/8.9)
```




    (1.1235955056179774, 1.2359550561797752)




```python
testing = []
df123 = [np.log(i+1) for i in df0.imp]

for i in range(len(df0)):
    testing.append(10*((df123[i]-min(df123)) / (max(df123) - min(df123))))
```


```python
df0.sat.min()
```




    0.0




```python
df123
```




    [1.7115177268415682,
     0.9534625892455924,
     0.8408749651825217,
     0.44946657497549475,
     2.3783535600422527,
     1.3853490243227224,
     0.778498944161523,
     0.5504818825631803,
     0.6356417261637283,
     0.8989331499509895,
     0.31782086308186414,
     0.5504818825631803,
     0.5504818825631803,
     0.6356417261637283,
     0.9534625892455924,
     1.145919418253881,
     0.8408749651825217,
     1.145919418253881,
     1.4564381625088383,
     1.7115177268415682,
     0.8408749651825217,
     0.44946657497549475,
     0.44946657497549475,
     0.6356417261637283,
     0.6356417261637283,
     1.2712834523274565,
     0.5504818825631803,
     0.7106690545187014,
     0.9534625892455924,
     0.778498944161523,
     0.6356417261637283,
     0.44946657497549475,
     1.1009637651263606,
     1.005037815259212,
     0.31782086308186414,
     0.5504818825631803,
     1.2309149097933274,
     1.0540925533894598,
     0.0,
     0.31782086308186414,
     0.6356417261637283,
     0.8989331499509895,
     1.005037815259212,
     0.778498944161523,
     0.7106690545187014,
     0.44946657497549475,
     0.0,
     0.31782086308186414,
     0.0,
     0.0,
     0.5504818825631803,
     0.31782086308186414,
     1.7695516752310827,
     0.5504818825631803,
     1.005037815259212,
     0.44946657497549475,
     0.0,
     0.0,
     0.7106690545187014,
     1.005037815259212,
     2.420451581541316,
     0.9534625892455924,
     0.5504818825631803,
     1.2309149097933274,
     0.31782086308186414,
     1.0540925533894598,
     0.0,
     0.31782086308186414,
     0.7106690545187014,
     0.8408749651825217,
     0.5504818825631803,
     1.310408988511494,
     1.1009637651263606,
     0.6356417261637283,
     2.6014758825860307,
     0.6356417261637283,
     0.31782086308186414,
     0.5504818825631803,
     0.7106690545187014,
     0.7106690545187014,
     0.6356417261637283,
     0.44946657497549475,
     1.0540925533894598,
     1.2712834523274565,
     0.9534625892455924,
     0.0,
     0.31782086308186414,
     0.6356417261637283,
     0.31782086308186414,
     0.5504818825631803,
     0.31782086308186414,
     0.5504818825631803,
     0.7106690545187014,
     0.31782086308186414,
     0.9534625892455924,
     0.8989331499509895,
     0.5504818825631803,
     2.335496832484569,
     0.0,
     1.005037815259212,
     0.7106690545187014,
     0.7106690545187014,
     1.005037815259212,
     0.8408749651825217,
     0.44946657497549475,
     0.7106690545187014,
     0.31782086308186414,
     0.44946657497549475,
     0.44946657497549475,
     0.0,
     0.8408749651825217,
     0.9534625892455924,
     1.2309149097933274,
     0.44946657497549475,
     1.0540925533894598,
     0.8408749651825217,
     0.7106690545187014,
     0.0,
     1.959179378817529,
     0.0,
     0.6356417261637283,
     0.0,
     0.0,
     1.005037815259212,
     0.778498944161523,
     0.778498944161523,
     1.310408988511494,
     0.5504818825631803,
     0.8408749651825217,
     2.010075630518424,
     0.6356417261637283,
     0.7106690545187014,
     0.7106690545187014,
     0.6356417261637283,
     1.005037815259212,
     1.145919418253881,
     0.7106690545187014,
     1.556997888323046,
     0.31782086308186414,
     1.2309149097933274,
     0.44946657497549475,
     3.015113445777636,
     0.8989331499509895,
     0.8408749651825217,
     1.145919418253881,
     1.8257418583505536,
     1.2712834523274565,
     0.5504818825631803,
     1.1891767800211264,
     1.4907119849998598,
     2.542566904654913,
     0.8989331499509895,
     1.9332288373014037,
     0.9534625892455924,
     1.0540925533894598,
     0.31782086308186414,
     1.7695516752310827,
     1.9069251784911847,
     2.806917861068948,
     1.005037815259212,
     2.4822602928815023,
     0.5504818825631803,
     0.9534625892455924,
     2.3137708083419484,
     0.8408749651825217,
     0.9534625892455924,
     1.0540925533894598,
     0.5504818825631803,
     1.145919418253881,
     0.31782086308186414,
     0.7106690545187014,
     1.3483997249264843,
     0.7106690545187014,
     0.44946657497549475,
     1.6205747826813257,
     0.5504818825631803,
     0.6356417261637283,
     2.581988897471611,
     0.778498944161523,
     0.7106690545187014,
     0.7106690545187014,
     0.5504818825631803,
     0.6356417261637283,
     0.0,
     0.31782086308186414,
     1.3853490243227224,
     0.0,
     1.4907119849998598,
     0.8989331499509895,
     0.0,
     1.4213381090374029,
     1.959179378817529,
     0.31782086308186414,
     0.7106690545187014,
     1.3483997249264843,
     0.44946657497549475,
     1.2309149097933274,
     2.2696949467968492,
     0.44946657497549475,
     0.778498944161523,
     1.1009637651263606,
     0.7106690545187014,
     0.8989331499509895,
     0.9534625892455924,
     0.778498944161523,
     0.7106690545187014,
     0.7106690545187014,
     0.9534625892455924,
     0.5504818825631803,
     1.1891767800211264,
     0.6356417261637283,
     0.8989331499509895,
     0.0,
     2.4412283710451916,
     0.778498944161523,
     0.5504818825631803,
     0.44946657497549475,
     0.7106690545187014,
     0.31782086308186414,
     1.2309149097933274,
     0.31782086308186414,
     1.1891767800211264,
     1.1009637651263606,
     0.31782086308186414,
     0.778498944161523,
     0.9534625892455924,
     0.9534625892455924,
     0.0,
     0.9534625892455924,
     0.5504818825631803,
     0.31782086308186414,
     0.5504818825631803,
     0.9534625892455924,
     0.5504818825631803,
     0.7106690545187014,
     0.0,
     0.0,
     1.005037815259212,
     0.5504818825631803,
     0.8408749651825217,
     0.0,
     0.0,
     0.44946657497549475,
     0.8989331499509895,
     0.0,
     1.651445647689541,
     0.8408749651825217,
     0.7106690545187014,
     0.6356417261637283,
     0.44946657497549475,
     1.7115177268415682,
     0.7106690545187014,
     0.31782086308186414,
     0.0,
     0.0,
     0.7106690545187014,
     0.0,
     1.2712834523274565,
     0.5504818825631803,
     0.5504818825631803,
     1.1891767800211264,
     0.8408749651825217,
     0.7106690545187014,
     1.9847906537954927,
     0.778498944161523,
     1.3483997249264843,
     1.2309149097933274,
     0.5504818825631803,
     1.3853490243227224,
     0.5504818825631803,
     1.5891043154093205,
     1.9332288373014037,
     0.8408749651825217,
     1.145919418253881,
     0.9534625892455924,
     2.1555659689428777,
     0.8989331499509895,
     2.2473328748774737,
     0.778498944161523,
     1.0540925533894598,
     1.0540925533894598,
     0.5504818825631803,
     0.8989331499509895,
     1.2712834523274565,
     0.7106690545187014,
     0.7106690545187014,
     0.0,
     1.7115177268415682,
     1.6205747826813257,
     0.5504818825631803,
     0.9534625892455924,
     0.7106690545187014,
     0.5504818825631803,
     0.6356417261637283,
     1.1009637651263606,
     0.8989331499509895,
     1.145919418253881,
     0.5504818825631803,
     1.310408988511494,
     0.5504818825631803,
     0.778498944161523,
     0.6356417261637283,
     0.31782086308186414,
     0.8408749651825217,
     0.0,
     0.31782086308186414,
     0.44946657497549475,
     1.2712834523274565,
     1.0540925533894598,
     0.44946657497549475,
     0.5504818825631803,
     1.4907119849998598,
     0.8989331499509895,
     0.8989331499509895,
     0.7106690545187014,
     1.1891767800211264,
     0.8408749651825217,
     0.0,
     1.6817499303650434,
     1.4213381090374029,
     0.5504818825631803,
     0.7106690545187014,
     0.0,
     1.8802535827258875,
     0.7106690545187014,
     0.8989331499509895,
     1.3853490243227224,
     0.8408749651825217,
     0.5504818825631803,
     0.8408749651825217,
     0.31782086308186414,
     0.0,
     2.788866755113585,
     0.31782086308186414,
     0.31782086308186414,
     0.8408749651825217,
     0.31782086308186414,
     0.31782086308186414,
     1.1891767800211264,
     0.7106690545187014,
     0.6356417261637283,
     0.778498944161523,
     0.31782086308186414,
     0.5504818825631803,
     0.0,
     0.5504818825631803,
     1.2712834523274565,
     1.0540925533894598,
     0.9534625892455924,
     2.877990320141519,
     0.8408749651825217,
     1.2309149097933274,
     0.44946657497549475,
     1.2309149097933274,
     0.7106690545187014,
     2.2247460415730487,
     2.010075630518424,
     0.0,
     0.0,
     0.31782086308186414,
     0.778498944161523,
     2.178870062090612,
     0.8408749651825217,
     0.8989331499509895,
     0.31782086308186414,
     0.8408749651825217,
     0.7106690545187014,
     0.7106690545187014,
     0.9534625892455924,
     0.9534625892455924,
     0.44946657497549475,
     0.6356417261637283,
     0.5504818825631803,
     1.3853490243227224,
     1.4907119849998598,
     1.1009637651263606,
     1.005037815259212,
     1.310408988511494,
     0.7106690545187014,
     0.0,
     0.44946657497549475,
     0.31782086308186414,
     1.005037815259212,
     0.5504818825631803,
     0.0,
     1.145919418253881,
     0.0,
     1.1891767800211264,
     1.3483997249264843,
     0.778498944161523,
     0.8408749651825217,
     0.6356417261637283,
     1.3483997249264843,
     1.2712834523274565,
     0.0,
     0.5504818825631803,
     0.44946657497549475,
     1.5242153139344596,
     0.9534625892455924,
     0.7106690545187014,
     1.651445647689541,
     0.31782086308186414,
     1.005037815259212,
     0.7106690545187014,
     1.3483997249264843,
     0.44946657497549475,
     0.31782086308186414,
     1.2309149097933274,
     1.2712834523274565,
     1.1009637651263606,
     0.31782086308186414,
     1.1009637651263606,
     1.005037815259212,
     1.556997888323046,
     1.1009637651263606,
     0.9534625892455924,
     0.8408749651825217,
     1.5891043154093205,
     1.0540925533894598,
     1.1891767800211264,
     0.9534625892455924,
     0.778498944161523,
     0.7106690545187014,
     1.005037815259212,
     0.0,
     0.778498944161523,
     0.0,
     0.44946657497549475,
     0.5504818825631803,
     2.3137708083419484,
     1.1891767800211264,
     1.310408988511494,
     1.2712834523274565,
     0.31782086308186414,
     0.5504818825631803,
     1.3483997249264843,
     0.8408749651825217,
     0.7106690545187014,
     0.31782086308186414,
     1.1009637651263606,
     1.145919418253881,
     1.3483997249264843,
     2.461829819586655,
     0.778498944161523,
     0.0,
     0.8408749651825217,
     1.145919418253881,
     0.7106690545187014,
     0.31782086308186414,
     1.7407765595569784,
     1.1009637651263606,
     2.2696949467968492,
     0.5504818825631803,
     0.8408749651825217,
     1.5891043154093205,
     0.31782086308186414,
     1.4907119849998598,
     1.0540925533894598,
     1.005037815259212,
     0.778498944161523,
     0.44946657497549475,
     1.4564381625088383,
     0.5504818825631803,
     0.7106690545187014,
     0.44946657497549475,
     0.9534625892455924,
     1.4907119849998598,
     0.8408749651825217,
     0.44946657497549475,
     2.1555659689428777,
     0.6356417261637283,
     1.1009637651263606,
     0.7106690545187014,
     0.778498944161523,
     1.3853490243227224,
     0.8408749651825217,
     0.8408749651825217,
     0.7106690545187014,
     2.581988897471611,
     1.1891767800211264,
     1.0540925533894598,
     0.5504818825631803,
     0.6356417261637283,
     1.8257418583505536,
     1.1009637651263606,
     1.7115177268415682,
     0.31782086308186414,
     1.005037815259212,
     1.9069251784911847,
     1.1009637651263606,
     0.31782086308186414,
     1.5891043154093205,
     0.8408749651825217,
     0.6356417261637283,
     2.2019275302527213,
     0.778498944161523,
     0.8989331499509895,
     1.2309149097933274,
     0.8408749651825217,
     0.6356417261637283,
     0.8408749651825217,
     1.1891767800211264,
     0.0,
     2.178870062090612,
     0.7106690545187014,
     0.778498944161523,
     0.778498944161523,
     1.5891043154093205,
     1.1891767800211264,
     0.0,
     0.5504818825631803,
     0.778498944161523,
     0.31782086308186414,
     0.9534625892455924,
     0.9534625892455924,
     0.778498944161523,
     0.8989331499509895,
     0.778498944161523,
     2.3570226039551585,
     2.3994948963429277,
     0.31782086308186414,
     1.145919418253881,
     0.31782086308186414,
     0.5504818825631803,
     0.0,
     1.5242153139344596,
     1.0540925533894598,
     0.8408749651825217,
     0.8408749651825217,
     2.5025239784318276,
     1.4564381625088383,
     0.44946657497549475,
     0.8408749651825217,
     0.31782086308186414,
     0.8989331499509895,
     1.9069251784911847,
     0.9534625892455924,
     0.7106690545187014,
     0.31782086308186414,
     0.0,
     1.7407765595569784,
     1.2712834523274565,
     0.8989331499509895,
     0.9534625892455924,
     0.8408749651825217,
     0.0,
     0.7106690545187014,
     0.31782086308186414,
     1.005037815259212,
     0.6356417261637283,
     1.4907119849998598,
     0.44946657497549475,
     0.5504818825631803,
     0.778498944161523,
     2.0350464715613112,
     1.1009637651263606,
     1.6817499303650434,
     1.4907119849998598,
     0.778498944161523,
     1.4564381625088383,
     1.8257418583505536,
     0.5504818825631803,
     1.1891767800211264,
     0.31782086308186414,
     0.31782086308186414,
     0.5504818825631803,
     0.7106690545187014,
     0.8408749651825217,
     1.0540925533894598,
     1.005037815259212,
     2.1320071635561044,
     0.9534625892455924,
     1.3483997249264843,
     0.6356417261637283,
     0.0,
     1.9847906537954927,
     0.31782086308186414,
     0.9534625892455924,
     0.31782086308186414,
     0.5504818825631803,
     0.8989331499509895,
     0.0,
     0.31782086308186414,
     0.7106690545187014,
     0.0,
     1.4907119849998598,
     0.7106690545187014,
     0.31782086308186414,
     0.31782086308186414,
     0.5504818825631803,
     1.1009637651263606,
     0.0,
     0.31782086308186414,
     0.0,
     0.5504818825631803,
     1.6817499303650434,
     1.0540925533894598,
     0.7106690545187014,
     1.005037815259212,
     1.4213381090374029,
     0.778498944161523,
     1.3483997249264843,
     0.5504818825631803,
     0.8408749651825217,
     0.7106690545187014,
     1.005037815259212,
     0.5504818825631803,
     0.8408749651825217,
     0.6356417261637283,
     0.31782086308186414,
     0.8989331499509895,
     0.8408749651825217,
     2.2247460415730487,
     0.0,
     0.778498944161523,
     0.6356417261637283,
     0.8408749651825217,
     0.778498944161523,
     0.31782086308186414,
     1.1009637651263606,
     0.44946657497549475,
     1.4213381090374029,
     0.7106690545187014,
     1.7115177268415682,
     1.005037815259212,
     0.5504818825631803,
     1.1009637651263606,
     0.9534625892455924,
     0.8408749651825217,
     0.9534625892455924,
     1.6205747826813257,
     0.0,
     1.3483997249264843,
     1.5891043154093205,
     2.010075630518424,
     0.8989331499509895,
     1.0540925533894598,
     1.4907119849998598,
     0.6356417261637283,
     1.310408988511494,
     0.31782086308186414,
     0.31782086308186414,
     0.8408749651825217,
     0.6356417261637283,
     1.6817499303650434,
     1.6205747826813257,
     0.8408749651825217,
     0.44946657497549475,
     0.31782086308186414,
     2.3570226039551585,
     1.1009637651263606,
     1.1891767800211264,
     0.5504818825631803,
     0.7106690545187014,
     0.5504818825631803,
     0.31782086308186414,
     0.7106690545187014,
     0.7106690545187014,
     1.1009637651263606,
     1.2309149097933274,
     0.6356417261637283,
     0.44946657497549475,
     0.5504818825631803,
     0.7106690545187014,
     0.0,
     0.5504818825631803,
     1.6817499303650434,
     0.44946657497549475,
     0.31782086308186414,
     1.0540925533894598,
     0.31782086308186414,
     1.1009637651263606,
     0.778498944161523,
     1.4564381625088383,
     1.005037815259212,
     0.778498944161523,
     1.1009637651263606,
     0.7106690545187014,
     0.31782086308186414,
     1.0540925533894598,
     0.8989331499509895,
     0.9534625892455924,
     3.1622776601683795,
     1.1891767800211264,
     1.145919418253881,
     0.6356417261637283,
     0.31782086308186414,
     0.0,
     0.778498944161523,
     0.31782086308186414,
     0.6356417261637283,
     0.0,
     1.1009637651263606,
     0.8408749651825217,
     1.651445647689541,
     1.1891767800211264,
     0.8989331499509895,
     0.31782086308186414,
     0.778498944161523,
     1.1009637651263606,
     0.31782086308186414,
     1.5891043154093205,
     0.7106690545187014,
     0.8989331499509895,
     2.1320071635561044,
     0.31782086308186414,
     0.5504818825631803,
     1.4564381625088383,
     0.0,
     1.7695516752310827,
     0.8408749651825217,
     0.31782086308186414,
     0.31782086308186414,
     0.31782086308186414,
     1.0540925533894598,
     0.9534625892455924,
     1.6205747826813257,
     2.1320071635561044,
     0.6356417261637283,
     1.3853490243227224,
     0.778498944161523,
     0.31782086308186414,
     0.7106690545187014,
     0.44946657497549475,
     2.010075630518424,
     0.8408749651825217,
     0.44946657497549475,
     1.3853490243227224,
     0.0,
     0.31782086308186414,
     1.0540925533894598,
     0.7106690545187014,
     0.31782086308186414,
     1.0540925533894598,
     0.7106690545187014,
     1.145919418253881,
     0.6356417261637283,
     0.31782086308186414,
     2.0840907713999273,
     0.7106690545187014,
     0.44946657497549475,
     1.2309149097933274,
     1.4907119849998598,
     1.9332288373014037,
     0.5504818825631803,
     0.8408749651825217,
     1.4907119849998598,
     2.2696949467968492,
     0.6356417261637283,
     0.31782086308186414,
     1.4564381625088383,
     0.0,
     1.1009637651263606,
     1.4213381090374029,
     1.1891767800211264,
     1.0540925533894598,
     0.5504818825631803,
     1.310408988511494,
     0.44946657497549475,
     0.5504818825631803,
     0.5504818825631803,
     1.005037815259212,
     1.005037815259212,
     2.1320071635561044,
     0.5504818825631803,
     1.8257418583505536,
     1.005037815259212,
     2.4822602928815023,
     0.7106690545187014,
     0.44946657497549475,
     1.8802535827258875,
     0.7106690545187014,
     0.0,
     0.31782086308186414,
     1.0540925533894598,
     0.6356417261637283,
     1.3483997249264843,
     0.778498944161523,
     0.9534625892455924,
     1.2309149097933274,
     0.5504818825631803,
     0.0,
     1.1891767800211264,
     2.3137708083419484,
     0.31782086308186414,
     0.8408749651825217,
     0.5504818825631803,
     0.6356417261637283,
     0.778498944161523,
     2.2019275302527213,
     0.6356417261637283,
     0.5504818825631803,
     1.145919418253881,
     0.0,
     0.778498944161523,
     0.6356417261637283,
     1.005037815259212,
     0.0,
     0.8408749651825217,
     0.31782086308186414,
     0.44946657497549475,
     0.7106690545187014,
     0.778498944161523,
     0.8408749651825217,
     2.4412283710451916,
     0.8989331499509895,
     0.0,
     1.2712834523274565,
     1.3853490243227224,
     0.44946657497549475,
     0.6356417261637283,
     0.44946657497549475,
     0.0,
     0.8408749651825217,
     1.310408988511494,
     0.5504818825631803,
     1.3853490243227224,
     0.8408749651825217,
     0.8989331499509895,
     1.3853490243227224,
     1.145919418253881,
     1.145919418253881,
     0.7106690545187014,
     1.2309149097933274,
     1.797866299901979,
     0.8989331499509895,
     1.1891767800211264,
     0.5504818825631803,
     0.0,
     0.9534625892455924,
     2.1081851067789197,
     0.8408749651825217,
     0.31782086308186414,
     1.556997888323046,
     0.9534625892455924,
     0.5504818825631803,
     0.31782086308186414,
     0.0,
     1.145919418253881,
     1.6205747826813257,
     0.7106690545187014,
     0.5504818825631803,
     0.8989331499509895,
     1.145919418253881,
     1.310408988511494,
     0.31782086308186414,
     0.31782086308186414,
     0.8408749651825217,
     0.778498944161523,
     0.8989331499509895,
     1.4907119849998598,
     1.7115177268415682,
     1.1009637651263606,
     1.005037815259212,
     0.7106690545187014,
     1.5891043154093205,
     0.8989331499509895,
     1.0540925533894598,
     0.31782086308186414,
     0.7106690545187014,
     0.778498944161523,
     0.7106690545187014,
     0.7106690545187014,
     0.6356417261637283,
     0.44946657497549475,
     0.0,
     2.788866755113585,
     1.310408988511494,
     0.778498944161523,
     1.3853490243227224,
     0.5504818825631803,
     2.0350464715613112,
     0.0,
     0.6356417261637283,
     0.778498944161523,
     0.7106690545187014,
     0.6356417261637283,
     0.8408749651825217,
     0.8989331499509895,
     0.7106690545187014,
     1.5891043154093205,
     0.44946657497549475,
     1.1891767800211264,
     1.310408988511494,
     0.31782086308186414,
     0.5504818825631803,
     1.1891767800211264,
     0.5504818825631803,
     0.31782086308186414,
     0.5504818825631803,
     0.7106690545187014,
     0.778498944161523,
     0.8989331499509895,
     1.2309149097933274,
     0.0,
     2.542566904654913,
     1.5242153139344596,
     0.5504818825631803,
     1.3483997249264843,
     1.8531981638085644,
     0.5504818825631803,
     0.31782086308186414,
     0.6356417261637283,
     1.310408988511494,
     1.1009637651263606,
     0.8408749651825217,
     0.7106690545187014,
     1.145919418253881,
     1.145919418253881,
     0.44946657497549475,
     0.5504818825631803,
     1.005037815259212,
     1.959179378817529,
     1.9069251784911847,
     0.6356417261637283,
     3.1622776601683795,
     0.5504818825631803,
     1.7407765595569784,
     0.5504818825631803,
     1.797866299901979,
     1.7115177268415682,
     1.1891767800211264,
     0.44946657497549475,
     2.4412283710451916,
     0.8408749651825217,
     0.44946657497549475,
     1.1891767800211264,
     0.31782086308186414,
     1.2309149097933274,
     1.2712834523274565,
     0.5504818825631803,
     0.6356417261637283,
     1.2309149097933274,
     0.6356417261637283,
     0.5504818825631803,
     0.7106690545187014,
     0.7106690545187014,
     1.145919418253881,
     0.31782086308186414,
     0.0,
     0.9534625892455924,
     1.0540925533894598,
     0.5504818825631803,
     1.5891043154093205,
     0.0,
     0.31782086308186414,
     0.5504818825631803,
     0.44946657497549475,
     0.5504818825631803,
     0.0,
     0.778498944161523,
     1.3853490243227224,
     1.2309149097933274,
     0.8408749651825217,
     0.6356417261637283,
     0.9534625892455924,
     1.2712834523274565,
     1.9332288373014037,
     1.7407765595569784,
     1.145919418253881,
     1.7407765595569784,
     1.1891767800211264,
     1.6817499303650434,
     0.7106690545187014,
     0.0,
     1.005037815259212,
     1.0540925533894598,
     0.6356417261637283,
     1.0540925533894598,
     0.7106690545187014,
     1.0540925533894598,
     0.31782086308186414,
     0.6356417261637283,
     1.2309149097933274,
     0.44946657497549475,
     1.145919418253881,
     0.8408749651825217,
     0.6356417261637283,
     3.1622776601683795,
     0.7106690545187014,
     1.4564381625088383,
     0.7106690545187014,
     0.0,
     1.8257418583505536,
     0.8989331499509895,
     0.5504818825631803,
     2.0840907713999273,
     0.9534625892455924,
     1.2309149097933274,
     0.778498944161523,
     2.3570226039551585,
     0.5504818825631803,
     0.44946657497549475,
     1.7695516752310827,
     0.0,
     0.8408749651825217,
     1.145919418253881,
     0.44946657497549475,
     0.7106690545187014,
     0.778498944161523,
     ...]




```python
plt.scatter(testing,sat)
```




    <matplotlib.collections.PathCollection at 0x193ee063630>




    
![png](output_98_1.png)
    



```python
df1['imp'].mean(), df1['sat'].mean()
```




    (3.9437285377284796, 5.122305688617331)




```python
df1.imp.value_counts()
```




    1.033829    367
    2.336300    349
    0.000000    309
    3.194216    306
    3.835002    246
               ... 
    9.923118      1
    9.648688      1
    9.762028      1
    9.437346      1
    9.530074      1
    Name: imp, Length: 85, dtype: int64




```python
fig=plt.figure(figsize=(7,7))
plt.scatter(df1.imp, df1.sat, alpha=0.35, s=df1.opt*25) #c=df0.opt, cmap='viridis')

# satisfaction
# xdata = list(range(11))
# ydata = [(_*0.51)+4.9 for _ in xdata]
# plt.plot(xdata, ydata, 'k')

# importance
# x_data = list(range(int(df1['imp'].mean()), 11))
# y_data = [(10/8.9)*_-(1.1*(10/8.9)) for _ in x_data]
# plt.plot(x_data, y_data, 'k')

# plt.title('기회 점수', fontsize=14)
# plt.xlabel('중요도', fontsize=12)
# plt.ylabel('Satisfaction', fontsize=12)
plt.show()
```


    
![png](output_101_0.png)
    



```python
fig=plt.figure(figsize=(7,7))
plt.scatter(df1.imp, df1.sat, alpha=0.6, s=df1.opt*10) #c=df0.opt, cmap='viridis')

# satisfaction
xdata = list(range(11))
ydata = [(_*0.51)+4.9 for _ in xdata]
plt.plot(xdata, ydata, 'k')

# importance
x_data = list(range(int(df1['imp'].mean()), 11))
y_data = [(10/8.9)*_-(1.1*(10/8.9)) for _ in x_data]
plt.plot(x_data, y_data, 'k')

# plt.title('기회 점수', fontsize=14)
# plt.xlabel('중요도', fontsize=12)
# plt.ylabel('Satisfaction', fontsize=12)
plt.show()
```


    
![png](output_102_0.png)
    



```python
fig = px.scatter(data_frame = df0, x='imp',y='sat',hover_name='review',
                size='imp', size_max=20)
fig.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-121-d2a5159f61ad> in <module>
    ----> 1 fig = px.scatter(data_frame = df0, x='imp',y='sat',hover_name='review',
          2                 size='imp', size_max=20)
          3 fig.show()
    

    NameError: name 'px' is not defined



```python
df0.opt.mean()
```




    1.2086035564860094




```python
# 더 집중해서 봐야할 글들을 필터링할 수 있는 지표를 만든거
# 어디서 cut 할건지?
plt.hist(df0.opt.values,bins=100)
```




    (array([648., 553., 199., 452., 220., 308., 257.,  90., 158.,  84., 120.,
             46.,  85.,  76.,  39.,  56.,  27.,  52.,  26.,  35.,  33.,  12.,
             29.,  20.,  31.,  13.,  18.,  12.,  10.,  11.,   8.,  10.,   3.,
              4.,   8.,   3.,   5.,   3.,   3.,   6.,   5.,   5.,   3.,   3.,
              4.,   2.,   3.,   4.,   2.,   1.,   4.,   7.,   0.,   2.,   2.,
              1.,   2.,   0.,   2.,   1.,   0.,   4.,   0.,   2.,   0.,   0.,
              1.,   1.,   4.,   1.,   0.,   1.,   0.,   4.,   2.,   0.,   1.,
              0.,   1.,   0.,   0.,   1.,   1.,   0.,   0.,   1.,   1.,   0.,
              0.,   1.,   1.,   1.,   3.,   1.,   1.,   2.,   1.,   0.,   2.,
              1.]),
     array([ 0.        ,  0.15886667,  0.31773333,  0.4766    ,  0.63546667,
             0.79433333,  0.9532    ,  1.11206667,  1.27093333,  1.4298    ,
             1.58866667,  1.74753333,  1.9064    ,  2.06526667,  2.22413333,
             2.383     ,  2.54186667,  2.70073333,  2.8596    ,  3.01846667,
             3.17733333,  3.3362    ,  3.49506667,  3.65393333,  3.8128    ,
             3.97166667,  4.13053333,  4.2894    ,  4.44826667,  4.60713333,
             4.766     ,  4.92486667,  5.08373333,  5.2426    ,  5.40146667,
             5.56033333,  5.7192    ,  5.87806667,  6.03693333,  6.1958    ,
             6.35466667,  6.51353333,  6.6724    ,  6.83126667,  6.99013333,
             7.149     ,  7.30786667,  7.46673333,  7.6256    ,  7.78446667,
             7.94333333,  8.1022    ,  8.26106667,  8.41993333,  8.5788    ,
             8.73766667,  8.89653333,  9.0554    ,  9.21426667,  9.37313333,
             9.532     ,  9.69086667,  9.84973333, 10.0086    , 10.16746667,
            10.32633333, 10.4852    , 10.64406667, 10.80293333, 10.9618    ,
            11.12066667, 11.27953333, 11.4384    , 11.59726667, 11.75613333,
            11.915     , 12.07386667, 12.23273333, 12.3916    , 12.55046667,
            12.70933333, 12.8682    , 13.02706667, 13.18593333, 13.3448    ,
            13.50366667, 13.66253333, 13.8214    , 13.98026667, 14.13913333,
            14.298     , 14.45686667, 14.61573333, 14.7746    , 14.93346667,
            15.09233333, 15.2512    , 15.41006667, 15.56893333, 15.7278    ,
            15.88666667]),
     <BarContainer object of 100 artists>)




    
![png](output_105_1.png)
    



```python
df00 = df1[df1.opt > df1.opt.mean()]
```


```python
df00.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1852 entries, 3 to 9742
    Data columns (total 32 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Unnamed: 0         1852 non-null   int64  
     1   Unnamed: 0.1       1852 non-null   int64  
     2   Unnamed: 0.1.1     1852 non-null   int64  
     3   review             1852 non-null   object 
     4   after              1852 non-null   object 
     5   pos                1852 non-null   object 
     6   cluster            1852 non-null   int64  
     7   comment            1852 non-null   object 
     8   date               1852 non-null   int64  
     9   review_len         1852 non-null   int64  
     10  review_senti       1852 non-null   int64  
     11  review_senti_mean  1852 non-null   float64
     12  okt_pos            1852 non-null   object 
     13  com_okt_pos        1852 non-null   object 
     14  com_num            1852 non-null   int64  
     15  com_len_mean       1852 non-null   float64
     16  com_len_std        1852 non-null   float64
     17  com_senti_dist     1852 non-null   object 
     18  com_senti          1852 non-null   float64
     19  com_senti_mean     1852 non-null   float64
     20  com_senti_std      1852 non-null   float64
     21  year               1852 non-null   int64  
     22  month              1852 non-null   int64  
     23  Unnamed: 23        0 non-null      float64
     24  Unnamed: 24        0 non-null      float64
     25  Unnamed: 25        0 non-null      float64
     26  okt2               1852 non-null   object 
     27  cluster_review3    1852 non-null   int64  
     28  com_num_norm       1852 non-null   float64
     29  imp                1852 non-null   float64
     30  sat                1852 non-null   float64
     31  opt                1852 non-null   float64
    dtypes: float64(13), int64(11), object(8)
    memory usage: 477.5+ KB
    


```python
df00
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>review</th>
      <th>after</th>
      <th>pos</th>
      <th>cluster</th>
      <th>comment</th>
      <th>date</th>
      <th>review_len</th>
      <th>review_senti</th>
      <th>review_senti_mean</th>
      <th>...</th>
      <th>com_senti</th>
      <th>com_senti_mean</th>
      <th>com_senti_std</th>
      <th>year</th>
      <th>month</th>
      <th>okt2</th>
      <th>cluster_review3</th>
      <th>imp</th>
      <th>sat</th>
      <th>opt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8165</th>
      <td>8165</td>
      <td>집안일 하실 때요 17개월 아기를 키우고 있는 초보맘이에요심각한 엄마껌딱지라서 하루...</td>
      <td>집안일 하실 때 요 17개월 아기를 키우고 있는 초보맘이에 요심각한 엄마 껌 딱지라...</td>
      <td>['집안일', '아기', '키우', '초보', '엄마', '딱지', '하루하루', ...</td>
      <td>1</td>
      <td>[['전그래서 반찬 다 사서먹고있어요 설거지만후다닥하고  뭐든하나는 포기해야되여ㅜㅜ...</td>
      <td>1804</td>
      <td>471</td>
      <td>-2</td>
      <td>-1.000000</td>
      <td>...</td>
      <td>-44</td>
      <td>-2.000000</td>
      <td>4.067610</td>
      <td>18</td>
      <td>4</td>
      <td>[집안일, 때, 요, 개월, 아기, 키우다, 있다, 초보, 맘, 심각, 엄마, 껌,...</td>
      <td>1</td>
      <td>2.121212</td>
      <td>5.180723</td>
      <td>2.121212</td>
    </tr>
    <tr>
      <th>8176</th>
      <td>8176</td>
      <td>전업맘 분들 집안일 혼자 다 하시나요~? 작년 1월부터 임신한 몸으로 일하고 애 낳...</td>
      <td>전업맘 분들 집안일 혼자 다 하시나요 작년 1월부터 임신한 몸으로 일하고 애 낳고 ...</td>
      <td>['전업', '혼자', '작년', '임신', '출산', '휴가', '복직', '매일...</td>
      <td>1</td>
      <td>[['헐... 생각부터가 너무.... 노답인데요.. 저렇게 얘기할 정도면.. 고쳐지...</td>
      <td>1804</td>
      <td>1031</td>
      <td>-7</td>
      <td>-0.700000</td>
      <td>...</td>
      <td>-78</td>
      <td>-2.052632</td>
      <td>3.973248</td>
      <td>18</td>
      <td>4</td>
      <td>[전업, 맘, 집안일, 혼자, 작년, 임신, 몸, 일, 애, 낳다, 출산휴가, 개월...</td>
      <td>1</td>
      <td>3.737374</td>
      <td>5.149017</td>
      <td>3.737374</td>
    </tr>
    <tr>
      <th>8182</th>
      <td>8182</td>
      <td>둘째 임신 중일때 집안일 첫째케어 어떻게 하시나요 첫째 임신할 때는 한번의 아픔이 ...</td>
      <td>둘째 임신 중일 때 집안일 첫째 케어 어떻게 하시나요 첫째 임신할 때는 한 번의 아...</td>
      <td>['둘째', '임신', '첫째', '케어', '어떻게', '첫째', '임신', '아...</td>
      <td>1</td>
      <td>[['저도 둘째 임신 때 한 시간도 못 쉬고 집안일에 첫째가 어린이집도 안다녀서 혼...</td>
      <td>1804</td>
      <td>446</td>
      <td>0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-5</td>
      <td>-0.200000</td>
      <td>2.561250</td>
      <td>18</td>
      <td>4</td>
      <td>[둘째, 임신, 중일, 때, 집안일, 첫째, 케어, 어떻다, 첫째, 임신, 때, 번...</td>
      <td>1</td>
      <td>2.424242</td>
      <td>6.265060</td>
      <td>2.424242</td>
    </tr>
    <tr>
      <th>8195</th>
      <td>8195</td>
      <td>아기 돌보면서 집안일은 어케들 하시나여 57일  아가 맘이에요 밤잠은 잘자요 11시...</td>
      <td>아기 돌보면서 집안일은 어케들 하시나 여 57일 아가 맘이에 요 밤 잠은 잘 자요 ...</td>
      <td>['아기', '돌보', '집안일', '아가', '자요', '재우', '중간', '칭...</td>
      <td>1</td>
      <td>[['애기 모빌보고. 놀때하세요~~'], ['님 아기는 모빌 오래보나봐요\n글에 썼...</td>
      <td>1804</td>
      <td>656</td>
      <td>-9</td>
      <td>-1.500000</td>
      <td>...</td>
      <td>-52</td>
      <td>-2.363636</td>
      <td>3.599816</td>
      <td>18</td>
      <td>4</td>
      <td>[아기, 돌보다, 집안일, 어케들, 여, 아가, 맘, 요, 밤, 잠, 자다, 자서,...</td>
      <td>1</td>
      <td>2.121212</td>
      <td>4.961665</td>
      <td>2.121212</td>
    </tr>
    <tr>
      <th>8203</th>
      <td>8203</td>
      <td>집안일.다들 혼자다하시나요? 집안일.다들 혼자다하시나요?청소 빨래청소 설거지 반찬 ...</td>
      <td>집안일 다들 혼자 다 하시나요 집안일 다들 혼자 다 하시나 요청소 빨래 청소 설거지...</td>
      <td>['혼자', '혼자', '청소', '빨래', '청소', '설거지', '반찬', '국...</td>
      <td>1</td>
      <td>[['맘님 상황이시면 일주일에 두 번정도라도 도우미 쓸 거 같아여\n신랑이 퇴근이 ...</td>
      <td>1804</td>
      <td>267</td>
      <td>-9</td>
      <td>-2.250000</td>
      <td>...</td>
      <td>-5</td>
      <td>-0.238095</td>
      <td>3.610893</td>
      <td>18</td>
      <td>4</td>
      <td>[집안일, 들다, 혼자, 집안일, 들다, 혼자, 청소, 빨래, 청소, 설거지, 반찬...</td>
      <td>1</td>
      <td>2.020202</td>
      <td>6.242111</td>
      <td>2.020202</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7355</th>
      <td>7355</td>
      <td>애기키우는 맘인데 저만 이런가요..?ㅠㅠ 7개월 아기 키우는 맘이예여남편은 혼자 돈...</td>
      <td>애기 키우는 맘인데 저만 이런 가요 7개월 아기 키우는 맘이 예여 남편은 혼자 돈 ...</td>
      <td>['애기', '키우', '가요', '아기', '키우', '남편', '혼자', '독박...</td>
      <td>1</td>
      <td>[['남편이 일하고 와서 피곤한건 이해하겠지만 어느정도 육아의 참여를 해야 맘님이 ...</td>
      <td>2103</td>
      <td>261</td>
      <td>-9</td>
      <td>-1.285714</td>
      <td>...</td>
      <td>-40</td>
      <td>-1.176471</td>
      <td>3.148021</td>
      <td>21</td>
      <td>3</td>
      <td>[애기, 키우다, 맘, 이렇다, 가요, 개월, 아기, 키우다, 맘, 남편, 혼자, ...</td>
      <td>1</td>
      <td>3.333333</td>
      <td>5.676825</td>
      <td>3.333333</td>
    </tr>
    <tr>
      <th>7356</th>
      <td>7356</td>
      <td>직장산모님들 운동하시나요..? 운동을 하긴 해야하는데...?임신전에도 회사갔다오면 ...</td>
      <td>직장 산모님들 운동 하시나요 운동을 하긴 해야 하는데 임신 전에도 회사 갔다 오면 ...</td>
      <td>['직장', '산모', '운동', '운동', '임신', '회사', '집안일', '간...</td>
      <td>1</td>
      <td>[['저는 16주 이후부터 직장 다니면서 일주일에 3번 산전필라테스 다녔어요~~ 확...</td>
      <td>2103</td>
      <td>194</td>
      <td>6</td>
      <td>1.500000</td>
      <td>...</td>
      <td>17</td>
      <td>0.548387</td>
      <td>2.721938</td>
      <td>21</td>
      <td>3</td>
      <td>[직장, 산모, 들다, 운동, 운동, 임신, 전, 회사, 가다, 오다, 집안일, 간...</td>
      <td>1</td>
      <td>3.030303</td>
      <td>6.715896</td>
      <td>3.030303</td>
    </tr>
    <tr>
      <th>7366</th>
      <td>7366</td>
      <td>식세기 어디꺼 쓰시나요~ ?전업맘이예요.집안일 정말 정말 ㅋㅋㅋ 어렵네요손이 느린건...</td>
      <td>식세기 어디 꺼 쓰시나요 전업맘이예요집안일 정말 정말 어렵네요 손이 느린 건지 집안...</td>
      <td>['세기', '어디', '전업', '집안일', '정말', '정말', '어렵', '느...</td>
      <td>1</td>
      <td>[['lg 쓰는데 다른데건 안써봐서 비교는못하지만 워킹맘인 저한테는 좋아요ㅎ 신세계...</td>
      <td>2103</td>
      <td>130</td>
      <td>3</td>
      <td>0.750000</td>
      <td>...</td>
      <td>-21</td>
      <td>-0.840000</td>
      <td>2.722940</td>
      <td>21</td>
      <td>3</td>
      <td>[식, 세기, 어디, 끄다, 쓸다, 전업, 맘, 집안일, 정말, 정말, 어렵다, 손...</td>
      <td>1</td>
      <td>2.424242</td>
      <td>5.879518</td>
      <td>2.424242</td>
    </tr>
    <tr>
      <th>7367</th>
      <td>7367</td>
      <td>하. . . . 답 좀 주세요 . . . 첫번째 산후도우미님은일은 빠릿빠릿 잘하시고...</td>
      <td>하 답 좀 주세요 첫 번째 산 후도우미님은 일은 빠릿빠릿 잘하시고 하는데 모든 행동...</td>
      <td>['도우미', '빠릿빠릿', '행동', '거칠', '화장실', '다녀오', '나가'...</td>
      <td>1</td>
      <td>[['허... 세상에.... 그냥 취소는 안되나요? 저도 지금 도우미이모님 계시는데...</td>
      <td>2103</td>
      <td>860</td>
      <td>3</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-37</td>
      <td>-2.176471</td>
      <td>3.501853</td>
      <td>21</td>
      <td>3</td>
      <td>[답, 줄다, 첫, 산, 후, 도우미, 일, 모든, 행동, 크다, 거치다, 손, 씻...</td>
      <td>1</td>
      <td>1.616162</td>
      <td>5.074415</td>
      <td>1.616162</td>
    </tr>
    <tr>
      <th>7376</th>
      <td>7376</td>
      <td>싸우고 정떨어져요.. 남편보다 상대적으로 제가 더 깔끔한편이라 제가 못참고 치우다보...</td>
      <td>싸우고 정 떨어져요 남편보다 상대적으로 제가 더 깔끔한 편이라 제가 못 참고 치우다...</td>
      <td>['싸우', '떨어지', '남편', '상대', '치우', '집안일', '프로', '...</td>
      <td>1</td>
      <td>[['안마 같은 거 해주지 마세요...ㅜㅜ 답답하시겠어요ㅠㅠ'], ['ㅜㅜ시간을 두...</td>
      <td>2103</td>
      <td>819</td>
      <td>-10</td>
      <td>-0.476190</td>
      <td>...</td>
      <td>-66</td>
      <td>-2.062500</td>
      <td>3.131867</td>
      <td>21</td>
      <td>3</td>
      <td>[싸우다, 정, 떨어지다, 남편, 상대, 제, 더, 깔끔하다, 편이, 제, 못, 참...</td>
      <td>1</td>
      <td>3.131313</td>
      <td>5.143072</td>
      <td>3.131313</td>
    </tr>
  </tbody>
</table>
<p>1577 rows × 26 columns</p>
</div>




```python
df00.to_csv('페이퍼용df00감성다시.csv',encoding='utf-8-sig')
```


```python
df00 = pd.read_csv('페이퍼용df00기회2.csv',encoding='utf-8-sig').drop('Unnamed: 0',axis=1).drop('Unnamed: 0.1',axis=1)
```


```python
df00.columns
```




    Index(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'review', 'after',
           'pos', 'cluster', 'comment', 'date', 'review_len', 'review_senti',
           'review_senti_mean', 'okt_pos', 'com_okt_pos', 'com_num',
           'com_len_mean', 'com_len_std', 'com_senti_dist', 'com_senti',
           'com_senti_mean', 'com_senti_std', 'year', 'month', 'Unnamed: 23',
           'Unnamed: 24', 'Unnamed: 25', 'okt2', 'cluster_review3', 'com_num_norm',
           'imp', 'sat', 'opt'],
          dtype='object')




```python
len(df00)
```




    1852




```python
def split_comments(text):
    x = text[3:-3]   
    x = x.replace('[','>')
    x = x.replace(']','<')
    x = x.replace('> ','>')
    x = x.replace(' <','<')
    y = re.split("'<, >'",x)
    return y
```


```python
comment_s = []
for i in df00.comment:
    comment_s.append(split_comments(i))
```


```python
comment_s[0][4]
```




    '개발은 되었단거 저도 봤어요. 한 10년이면 빌트인으로 세탁기 건조기 그리고 붙박이장이 세트로 넣어주는 것까지 되지 않을까 기도해봐요. ㅋㅋ'




```python
df00['comment_s'] = comment_s
```

    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    


```python
com_okt2=[]
for j in df00.com_okt_pos:
    j = j.replace("[], ","")
    j = j.replace(", []","")
    j = j.replace("[] ,","")
    xx = split_comments(j)
    xxx=[]
    for i in xx:
        xxx.extend(i.split("', '"))
    com_okt2.append(xxx)
print(len(com_okt2))
df00['com_okt2'] = com_okt2
```

    1852
    

    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if sys.path[0] == '':
    


```python
df00.com_senti_dist
```




    3       [2.0, -1.6666666666666667, 2.0, 0.333333333333...
    4       [-1.5, 2.0, -2.0, -1.6666666666666667, 0, -1.3...
    8       [0.0, -0.3333333333333333, -1.0, 1.0, -2.0, 1....
    10      [0, 2.0, 0, -2.0, -1.0, -1.0, 2.0, 1.0, 0, 0, ...
    24      [0.0, -2.0, -2.0, 0, -1.0, -1.0, 0.0, 0.5, 2.0...
                                  ...                        
    9731    [-1.0, 0, 0.0, 0, 0.0, 0, -1.0, 0, 2.0, -1.666...
    9732    [2.0, 0.0, 2.0, 0.0, 1.8, 0, 1.75, 2.0, 2.0, 2...
    9739    [0.75, 1.0, 0, 0, -1.0, 0, -1.3333333333333333...
    9740    [2.0, 0, 1.0, 2.0, 0, 0, 0.6666666666666666, 0...
    9742    [2.0, 0, 2.0, 1.5, 2.0, 0, 0, 0.0, 2.0, 2.0, 0...
    Name: com_senti_dist, Length: 1852, dtype: object




```python
com_okt22=[]
for j in df00.com_okt_pos:
    j = j.replace("[], ","")
    j = j.replace(", []","")
    j = j.replace("[] ,","")
    xx = split_comments(j)
    xxx=[]
    for i in xx:
        xxx.append(i.split("', '"))
    com_okt22.append(xxx)
print(len(com_okt22))
df00['com_okt22'] = com_okt22
```

    1852
    

    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if sys.path[0] == '':
    


```python
int(2.0)
```




    2




```python
com_senti2=[]
for j in df00.com_senti_dist:
    xx = j[1:-1].split(',')
    xxx=[]
    for i in xx:
#         i = i
        xxx.append(i)
    com_senti2.append(xxx)
print(len(com_senti2))
df00['com_senti2'] = com_senti2
```

    1852
    

    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    


```python
com_okt2[18]
```


```python
b=6
print(df00.comment.iloc[b])
print(com_okt2[b])
print(com_okt22[b])
print(com_senti2[b])
```

    [['ㅋㅋ\n저도 제일하기 싫은 집안일이 다리미질입니다\n일주일 와이셔츠 5개\n그나마 겨울바지는 세탁소이용했는데 여름바지는 물빠래되는 재질이라\nㅜㅜ'], ['전 거의 3개월에 한번인데 정말 싫어요. ㅠㅠ'], ['걸레빠는거요.전..그래서 항상 신랑몫이네요'], ['ㅎㅎ 걸레빠는 것도 싫고요.'], ['전 하나도 안다려주는데.. 왠지 미안해지네요 ㅎㅎㅎㅎ'], ['그냥 계속 안다려주는걸로 하심이...^^'], ['저는 청소요~\n청소가 젤루 시러요..ㅠㅠㅠ'], ['청소는 그나마 괜찮아요.\n그!나!마!요. ㅎ'], ['세탁소에 맡기고있어요ㅜ 젤싫은건 설거지요'], ['전 설거지는 괜찮은데요. ㅠㅠ\n다림질은'], ['맞아요ㅜㅜ 다림질....ㅜㅜ'], ['다렸는데  쭈글거리고..ㅠㅠ'], ['전 걸래 빠는게 젤 싫던데~ㅋ'], ['걸레빠는것도 싫고... 이것저것중 하나죠.^^'], ['전 빨래개는게 제일 싫어요.. 집에 동물키우는 집이라ㅜㅡㅜ'], ['아..  동물을 안 키워봐서 그 느낌을 모르겠네요. ㅠㅠ'], ['털이 옷에 다 붙어서.. 돌돌이로 털 다떼매고 빨래개어 넣어야하네요 귀찮고.. 시간도 두배는 걸리는듯요ㅠ 건조기 필수인데 아직 구매를 못해서ㅎㅎ'], ['전ㅋㅋㅋ 다려달라했는데 오늘도 그냥 누워있었네요..ㅋㅋㅋㅋ\n전 청소가젤싫어요ㅜㅜ'], ['ㅎㅎㅎ 저도 미루다 미루다 아이 오기 전에 꾸역꾸역했어요.'], ['저두 청소요\n신랑이 쉬는날 열심히 해줘요^^'], ['신랑 만세~~~~'], ['저두요 다림질 너무 귀찮아요 ㅋㅋ'], ['다려도 다려도 펴지지 않아요. ㅎㅎ'], ['맞아요 다림질 넘 힘들어요'], ['그런 정교한 작업 싫으네요. ㅠㅠ'], ['개는건 하겠는데 각각 서랍에 넣는게 힘들어요.ㅎㅎ'], ['저는 계절마다 옷정리ㅜㅜ 옷정리 필요없는 넓은집 옷방 갖고싶습니다..'], ['아~~저도 패딩이 아직 그대로 있네요.ㅠㅠ'], ['저도 그래요.다림질 바느질이 안맞아요'], ['본문에도 썼지만 집안일은 다 싫네요. ㅎㅎ']]
    ['저', '제일', '싫다', '집안일', '다리미질', '이다', '일주일', '와이셔츠', '개', '겨울', '바지', '세탁소', '용하다', '여름', '바지', '물빠래되', '재질', '전', '거의', '개월', '한번', '정말', '싫다', '걸레', '빨다', '전', '항상', '신랑', '몫', '걸레', '빨다', '것', '싫다', '전', '하나', '다리다', '미안하다', '그냥', '계속', '다리다', '는걸', '심', '저', '청소', '청소', '젤루', '시르다', '청소', '괜찮다', '그', '나', '마', '세탁소', '맡다', '젤', '싫다', '설거지', '전', '설거지', '괜찮다', '다림질', '맞다', '다림질', '다리다', '쭈글거리', '전', '걸', '빨다', '젤', '싫다', '걸레', '빨다', '싫다', '것', '것', '하나', '전', '빨래', '개다', '제일', '싫다', '집', '동물', '키우다', '집', '동물', '안', '키우다', '보다', '그', '느낌', '모르다', '털', '옷', '붙다', '이로', '털', '떼', '매다', '빨래', '개다', '넣다', '귀찮다', '시간', '배', '걸리다', '듯', '건조기', '필수', '구매', '전', '다리다', '달라', '오늘', '그냥', '눕다', '전', '청소', '가젤', '싫다', '저', '미루다', '미루다', '아이', '오기', '전', '저', '청소', '신랑', '쉬다', '날', '해주다', '신랑', '만세', '두', '다림질', '귀찮다', '다리다', '다리다', '펴다', '않다', '맞다', '다림질', '넘다', '힘들다', '그렇다', '정교하다', '작업', '싫다', '개다', '각각', '서랍', '넣다', '힘들다', '저', '계절', '옷', '정리', '옷', '정리', '필요없다', '넓다', '집', '옷방', '갖다', '저', '패딩', '그대로', '있다', '저', '그렇다', '다림질', '바느질', '맞다', '본문', '써다', '집안일', '싫다']
    [['저', '제일', '싫다', '집안일', '다리미질', '이다', '일주일', '와이셔츠', '개', '겨울', '바지', '세탁소', '용하다', '여름', '바지', '물빠래되', '재질'], ['전', '거의', '개월', '한번', '정말', '싫다'], ['걸레', '빨다', '전', '항상', '신랑', '몫'], ['걸레', '빨다', '것', '싫다'], ['전', '하나', '다리다', '미안하다'], ['그냥', '계속', '다리다', '는걸', '심'], ['저', '청소', '청소', '젤루', '시르다'], ['청소', '괜찮다', '그', '나', '마'], ['세탁소', '맡다', '젤', '싫다', '설거지'], ['전', '설거지', '괜찮다', '다림질'], ['맞다', '다림질'], ['다리다', '쭈글거리'], ['전', '걸', '빨다', '젤', '싫다'], ['걸레', '빨다', '싫다', '것', '것', '하나'], ['전', '빨래', '개다', '제일', '싫다', '집', '동물', '키우다', '집'], ['동물', '안', '키우다', '보다', '그', '느낌', '모르다'], ['털', '옷', '붙다', '이로', '털', '떼', '매다', '빨래', '개다', '넣다', '귀찮다', '시간', '배', '걸리다', '듯', '건조기', '필수', '구매'], ['전', '다리다', '달라', '오늘', '그냥', '눕다', '전', '청소', '가젤', '싫다'], ['저', '미루다', '미루다', '아이', '오기', '전'], ['저', '청소', '신랑', '쉬다', '날', '해주다'], ['신랑', '만세'], ['두', '다림질', '귀찮다'], ['다리다', '다리다', '펴다', '않다'], ['맞다', '다림질', '넘다', '힘들다'], ['그렇다', '정교하다', '작업', '싫다'], ['개다', '각각', '서랍', '넣다', '힘들다'], ['저', '계절', '옷', '정리', '옷', '정리', '필요없다', '넓다', '집', '옷방', '갖다'], ['저', '패딩', '그대로', '있다'], ['저', '그렇다', '다림질', '바느질', '맞다'], ['본문', '써다', '집안일', '싫다']]
    ['-0.5', ' -2.0', ' 0', ' -2.0', ' -1.0', ' 0', ' 0', ' 1.0', ' -2.0', ' 1.0', ' 0', ' 0', ' -2.0', ' -2.0', ' -2.0', ' 0', ' -0.3333333333333333', ' -2.0', ' -1.0', ' 0', ' 0', ' -1.0', ' 0', ' -2.0', ' -2.0', ' -2.0', ' 0', ' 0', ' 0', ' -2.0']
    


```python
from gensim.models import Word2Vec

EMBEDDING_DIM = 20 # 임베딩 크기는 논문을 따름
model = Word2Vec(sentences=df00.com_okt2, sg=1, size=EMBEDDING_DIM, window=5, min_count=1) #sg 0은 CBOW, 1은 SKIP-GRAM
w2v_vocab = list(model.wv.vocab) # 임베딩 된 단어 리스트
print('Vocabulary size : ',len(w2v_vocab)) 
print('Vecotr shape :',model.wv.vectors.shape)
```

    Vocabulary size :  15420
    Vecotr shape : (15420, 20)
    


```python
w2v_vocab
```




    ['앗',
     '두',
     '필요하다',
     '건조대',
     '고르다',
     '입다',
     '이모',
     '오다',
     '개다',
     '때',
     '기다리다',
     '기도',
     '능',
     '역시',
     '거',
     '아니다',
     '화장실',
     '수건',
     '쓰다',
     '더미',
     '가다',
     '걸다',
     '해',
     '계시다',
     '부럽다',
     '유',
     '전',
     '미국',
     '광고',
     '건조',
     '후',
     '옷',
     '접다',
     '걸',
     '보고',
     '저',
     '진짜',
     '있다',
     '싶다',
     '울',
     '집',
     '좋다',
     '그렇다',
     '양말',
     '속옷',
     '정리',
     '제일',
     '흠',
     '오',
     '그',
     '기계',
     '국내',
     '도입',
     '시급하다',
     '상용',
     '되어다',
     '가격',
     '적당하다',
     '지다',
     '빨래',
     '빠르다',
     '핫',
     '개발',
     '보다',
     '빌트',
     '세탁기',
     '건조기',
     '붙박다',
     '이장',
     '세트',
     '넣다',
     '것',
     '되다',
     '않다',
     '해보다',
     '없다',
     '너',
     '해도',
     '담',
     '진도',
     '빼다',
     '맘',
     '먹다',
     '정말',
     '정도',
     '다가',
     '불쌍하다',
     '먼지',
     '쌓이다',
     '제자리',
     '줍다',
     '제',
     '얘기',
     '줄',
     '알다',
     '넘다',
     '위로',
     '돼다',
     '마음',
     '따뜻하다',
     '널다',
     '음식물',
     '쓰레기',
     '버리다',
     '최대한',
     '미루다',
     '주',
     '뒤',
     '맞다',
     '요즘',
     '기저귀',
     '음청',
     '나오다',
     '가기',
     '귀찮다',
     '짜다',
     '종량제',
     '봉투',
     '사서',
     '오래오래',
     '숙성',
     '시키다',
     '신랑',
     '부탁',
     '곧',
     '여름',
     '날',
     '파리',
     '들이다',
     '계셧',
     '그것',
     '예정',
     '게',
     '싫다',
     '사고',
     '나서다',
     '아주',
     '털다',
     '왜',
     '늦다',
     '물',
     '냄새',
     '돌아가다',
     '바로',
     '안되다',
     '끝나다',
     '멜로디',
     '나',
     '한참',
     '옮기다',
     '덕',
     '쉰내',
     '안나',
     '젤',
     '음',
     '입',
     '건',
     '또',
     '많다',
     '아이러니',
     '관련',
     '빨',
     '키우다',
     '딸',
     '학년',
     '도와주다',
     '얼마나',
     '모르다',
     '초딩',
     '따님',
     '쯤',
     '거의',
     '강산',
     '번',
     '변하다',
     '일해',
     '주다',
     '고맙다',
     '뿐이다',
     '가벼워지다',
     '괜찮다',
     '누가',
     '해주다',
     '세제',
     '향기',
     '탁탁',
     '널',
     '땐',
     '상쾌하다',
     '마르다',
     '문제',
     '항',
     '마지못하다',
     '스타일',
     '동지',
     '티',
     '잡',
     '안일',
     '하나',
     '잖다',
     '그치다',
     '확',
     '나다',
     '뭔가',
     '뭘',
     '끄다',
     '이렇다',
     '설거지',
     '더',
     '대박',
     '고무장갑',
     '끼다',
     '별로',
     '사가',
     '식기세척기',
     '로봇청소기',
     '사다',
     '제보',
     '청소',
     '옷장',
     '왜케시른',
     '막',
     '편하다',
     '일이',
     '공간',
     '바구니',
     '쌓다',
     '두다',
     '기전',
     '개',
     '사',
     '놓다',
     '가보다',
     '확실하다',
     '필요',
     '같다',
     '어제',
     '건조하다',
     '오늘',
     '아침',
     '남편',
     '집다',
     '안해',
     '눈치',
     '종종',
     '찬스',
     '써다',
     '말',
     '척척',
     '이쁘다',
     '받다',
     '문득',
     '생각',
     '드네',
     '아이',
     '셋',
     '양도',
     '어마어마하다',
     '지난달',
     '퇴근',
     '돌리다',
     '이불',
     '다르다',
     '갖다',
     '넣기',
     '시르다',
     '처리',
     '담당',
     '만들기',
     '험난하다',
     '여정',
     '내',
     '켜다',
     '짓',
     '아예',
     '신혼',
     '살림',
     '시작',
     '한번',
     '도저히',
     '흐',
     '느끼다',
     '연기',
     '살짝',
     '더하다',
     '듯',
     '애',
     '많아지다',
     '목욕',
     '몫',
     '대신',
     '맛있다',
     '음식',
     '집안일',
     '귀',
     '찮은일',
     '사람',
     '움직이다',
     '야하다',
     '힘들다',
     '뭐',
     '노래',
     '함',
     '꽃',
     '다운',
     '하루',
     '파이팅',
     '반사',
     '보내다',
     '힘',
     '청소기',
     '젖병',
     '닦다',
     '왤케',
     '죵',
     '맛',
     '롤케이크',
     '혼자',
     '임',
     '나누다',
     '크림',
     '치즈',
     '프레즐',
     '흥칫뿡',
     '어떡하다',
     '난',
     '좋아하다',
     '서우',
     '놀러와',
     '언니',
     '대다',
     '여',
     '딸리다',
     '애가',
     '둘',
     '데리',
     '이동',
     '몸',
     '명',
     '이기다',
     '붙다',
     '미니미',
     '갈다',
     '씐',
     '나용',
     '시어머니',
     '이용',
     '서진',
     '날짜',
     '잡다',
     '끝없다',
     '도돌이표',
     '깨끗하다',
     '유지',
     '일',
     '차다',
     '힘드다',
     '이다',
     '주부',
     '들다',
     '다',
     '힘내다',
     '내려놓다',
     '보이다',
     '곳',
     '우리',
     '엄마',
     '해내다',
     '네',
     '요리',
     '잊다',
     '그냥',
     '머',
     '시름',
     '만해',
     '맡기다',
     '들어서다',
     '안방',
     '애증',
     '관계',
     '위임',
     '중',
     '싸다',
     '설겆',
     '이하',
     '자다',
     '느낌',
     '걸레질',
     '징',
     '각',
     '맞추다',
     '댓글',
     '읽다',
     '각양각색',
     '워낙',
     '광범위하다',
     '안',
     '기분',
     '끝',
     '분수',
     '계속',
     '거리',
     '예쁘다',
     '종이접기',
     '잘못',
     '방치',
     '순위',
     '진심',
     '마다',
     '수',
     '타다',
     '제품',
     '자세하다',
     '셔츠',
     '바지',
     '신기하다',
     '팔',
     '아프다',
     '쌩뚱맞',
     '먹기',
     '아이스',
     '카라멜',
     '마끼아또',
     '일주일',
     '해달',
     '돌',
     '저리',
     '창틀',
     '센치',
     '중이',
     '쓰기',
     '능력',
     '부족하다',
     '튼튼하다',
     '체력',
     '커버',
     '분리',
     '부피',
     '커서',
     '다시',
     '기두',
     '매일',
     '저희',
     '아들',
     '떼다',
     '최근',
     '몇번',
     '방수',
     '구입',
     '자마자',
     '수도',
     '밥',
     '동생',
     '싫어하다',
     '가면',
     '노답',
     '싱크대',
     '방일',
     '싫어지다',
     '만사',
     '설다',
     '늘어지다',
     '놀다',
     '댕댕',
     '워',
     '얌전하다',
     '귀엽다',
     '랍',
     '니당다',
     '사진',
     '맥주',
     '쳐다보다',
     '술',
     '맡다',
     '기겁',
     '반대',
     '멍뭉이',
     '한잔',
     '아쉽다',
     '거참',
     '낮술',
     '운동',
     '마시다',
     '간',
     '신분',
     '조신',
     '느므',
     '긔엽',
     '남다',
     '라서',
     '무조건',
     '푹신하다',
     '감성',
     '돋다',
     '강',
     '지도',
     '귀요미',
     '용',
     '쫄보',
     '수컷',
     '낮',
     '오우',
     '스피커',
     '스타',
     '필드',
     '업다',
     '인테리어',
     '효과',
     '수고',
     '하셧네',
     '맛저',
     '편안하다',
     '월요일',
     '휴일',
     '한주',
     '시간',
     '즐겁다',
     '밀리다',
     '개운하다',
     '하니',
     '시원하다',
     '헐다',
     '포뇽님',
     '깔끔하다',
     '집안',
     '햄',
     '님',
     '감사하다',
     '낼',
     '출근',
     '위해',
     '굿',
     '욤',
     '머니',
     '굿밤',
     '어찌',
     '반짝거리다',
     '물티슈',
     '노력',
     '빛',
     '이나',
     '폭탄',
     '뇽님',
     '맞벌이',
     '데',
     '대단하다',
     '기',
     '대충',
     '늙다',
     '관절',
     '소리',
     '살라',
     '사실',
     '살',
     '어떻다',
     '깨끗',
     '하나요',
     '하얗다',
     '칫솔',
     '힘주다',
     '고역',
     '찌들다',
     '가스렌지',
     '완전',
     '늘',
     '반짝반짝하다',
     '반성',
     '내일',
     '혹시',
     '묻다',
     '짠님들',
     '집도',
     '깔다',
     '자체',
     '다리미질',
     '와이셔츠',
     '겨울',
     '세탁소',
     '용하다',
     '물빠래되',
     '재질',
     '개월',
     '걸레',
     '빨다',
     '항상',
     '다리다',
     '미안하다',
     '는걸',
     '심',
     '젤루',
     '마',
     '다림질',
     '쭈글거리',
     '동물',
     '털',
     '이로',
     '떼',
     '매다',
     '배',
     '걸리다',
     '필수',
     '구매',
     '달라',
     '눕다',
     '가젤',
     '오기',
     '쉬다',
     '만세',
     '펴다',
     '정교하다',
     '작업',
     '각각',
     '서랍',
     '계절',
     '필요없다',
     '넓다',
     '옷방',
     '패딩',
     '그대로',
     '바느질',
     '본문',
     '돌아서다',
     '너무하다',
     '하원',
     '모루',
     '겟',
     '공감',
     '지금',
     '지옥',
     '저녁',
     '준비',
     '잠깐',
     '들어오다',
     '큰일',
     '개요',
     '와이프',
     '절실',
     '애쓰다',
     '방전',
     '화이팅',
     '넵',
     '챙기다',
     '악',
     '치웟는데',
     '치우다',
     '가져다주다',
     '이르다',
     '싶쥬',
     '삶다',
     '냉장고',
     '연중행사',
     '평소',
     '아기',
     '휴지통',
     '비우기',
     '입덧',
     '절정',
     '평상시',
     '임신',
     '시',
     '적',
     '왜케귀찮쵸',
     '베란다',
     '오히려',
     '애기',
     '지겹다',
     '쓸다',
     '개귀',
     '바닥',
     '물걸레',
     '다해',
     '그리다',
     '생색',
     '내다',
     '극혐',
     '그다음',
     '차리다',
     '먹고살다',
     '메뉴',
     '정',
     '스트레스',
     '덥다',
     '하라',
     '방',
     '단지',
     '식구',
     '번하다',
     '은근',
     '세상',
     '스팀',
     '기요',
     '로사',
     '적도',
     '있어욬',
     '첫',
     '두번째',
     '욕실',
     '세번',
     '빨기',
     '솜',
     '시트',
     '씌우다',
     '통',
     '산',
     '만들다',
     '이보',
     '갠',
     '전이',
     '이어지다',
     '꺼내다',
     '채',
     '쇼파',
     '위',
     '쏟다',
     '므',
     '시로',
     '똑같다',
     '아줌마',
     '살다',
     '부르다',
     '뿐',
     '편이',
     '달',
     '옆',
     '궁',
     '시렁',
     '가끔',
     '매트',
     '솔로',
     '더럽다',
     '전다',
     '회사',
     '일만',
     '하므다',
     '정답',
     '세척',
     '해방',
     '엉엉',
     '콩알',
     '성',
     '서방',
     '짧다',
     '전생',
     '군자금',
     '손대다',
     '무선',
     '여전하다',
     '놨더',
     '마찬가지',
     '하나꼽으',
     '산이',
     '잇다',
     '거두다',
     '던지다',
     '겨우',
     '집어넣다',
     '글케',
     '총동원',
     '총',
     '동원',
     '파전',
     '이군',
     '식탁',
     '쭉',
     '각자',
     '가져가다',
     '싷어',
     '비슷하다',
     '주방',
     '아수라',
     '쌀',
     '씻다',
     '년',
     '지나다',
     '컵',
     '뜨다',
     '물재',
     '재밌다',
     '바깥',
     '고',
     '거기',
     '정리정돈',
     '번은',
     '비우다',
     '최고',
     '반찬',
     '부다',
     '흙',
     '적성',
     '백일',
     '잔치',
     '넹',
     '서해',
     '셀프',
     '일상',
     '밖',
     '홧팅',
     '축하',
     '상대',
     '일찍',
     '떡',
     '찾다',
     '점심',
     '와우',
     '추카',
     '드리다',
     '고생',
     '합',
     '다용',
     '가족',
     '행복하다',
     '상',
     '건안',
     '손님',
     '이집',
     '남자',
     '그게',
     '결혼',
     '분담',
     '원래',
     '진지하다',
     '도움',
     '요청',
     '성공하다',
     '예정일',
     '여태',
     '빵터지다',
     '울다',
     '서다',
     '척',
     '봣',
     '물때',
     '가해',
     '쥐다',
     '건지다',
     '손',
     '가게',
     '지치다',
     '간난',
     '분기',
     '별',
     '처럼',
     '여자',
     '일부러',
     '일몰',
     '바쁘다',
     '척해',
     '가하다',
     '놔두다',
     '여유',
     '유도',
     '낳다',
     '쓰레기봉투',
     '구더기',
     '가버리다',
     '고치다',
     '병',
     '글',
     '화나다',
     '근무시간',
     '놀래다',
     '대리',
     '부부',
     '영업',
     '겠다',
     '잡고',
     '잔소리',
     '땡큐',
     '출산',
     '방법',
     '배우다',
     '야근',
     '어쩌',
     '안대',
     '어이없다',
     '아빠',
     '나가다',
     '인사',
     '길',
     '들어가다',
     '욕조',
     '구',
     '단',
     '주말',
     '밀고',
     '사주다',
     '세면대',
     '거울',
     '변기',
     '솔질',
     '첨부',
     '터',
     '확률',
     '크다',
     '심하다',
     '가지',
     '꼭',
     '해달라다',
     '바라지다',
     '말다',
     '하나로',
     '늘리다',
     '안보',
     '모든',
     '바라지',
     '한가지',
     '당번',
     '가령',
     '거나',
     '지저분하다',
     '행동',
     '어렵다',
     '열',
     '남기다',
     '온',
     '뒤지다',
     '엎',
     '싹',
     '담날',
     '진통',
     '가다해',
     '시세',
     '끼',
     '일제',
     '돌반',
     '기관',
     '안버',
     '먹이다',
     '불만',
     '학생',
     '돈',
     '도안',
     '벌다',
     '이상하다',
     '아름답다',
     '보이',
     '쉬',
     '서로',
     '조율',
     '속상하다',
     '보통',
     '범주',
     '기네',
     '평화',
     '이주',
     '추천',
     '주중',
     '토요일',
     '이상',
     '자취',
     '드럽다',
     '불다',
     '이안',
     '응',
     '분말',
     '가사',
     '도우미',
     '늘다',
     '어필',
     '칭찬',
     '부리다',
     '응거',
     '넘기다',
     '화학',
     '성분',
     '안좋다',
     '라면',
     '크게',
     '달라지다',
     '평일',
     '나쁘다',
     '인지',
     '예전',
     '티비',
     '식',
     '기억',
     '실행',
     '공동',
     '어딧나',
     '따지다',
     '고리타',
     '따다',
     '오래되다',
     '가정',
     '중요하다',
     '현재',
     '자유시간',
     '즐기다',
     '분',
     '최소',
     '보살',
     '어디',
     '감당',
     '본인',
     '잎',
     '세탁',
     '원하다',
     '향',
     '라나',
     '내무부',
     ...]




```python
print(model.wv.most_similar('집안일'))
print()
print(model.wv.most_similar('남편'))
print()
print(model.wv.similarity('귀엽다', '아기'))
```

    [('그치다', 0.9304101467132568), ('그르게', 0.9273521304130554), ('지치다', 0.9248520135879517), ('우울하다', 0.9238431453704834), ('맞다', 0.9197430610656738), ('쳇바퀴', 0.9193933010101318), ('태산', 0.9175771474838257), ('일도', 0.9160487651824951), ('서럽다', 0.9158444404602051), ('더하다', 0.9151780605316162)]
    
    [('신랑', 0.9668986201286316), ('예랑', 0.9620644450187683), ('편이', 0.9606578946113586), ('입덧', 0.9600449800491333), ('와이프', 0.9578466415405273), ('제', 0.9570134878158569), ('미안하다', 0.9566431045532227), ('이해', 0.9545568227767944), ('나누다', 0.9512630701065063), ('프로', 0.9495726823806763)]
    
    0.7044062
    


```python
# save model in ASCII (word2vec) format
# 텍스트 파일로 단어들의 임베딩 벡터 저장
filename = 'imdb_embedding_word2vec_comment.txt'
model.wv.save_word2vec_format(filename, binary=False)
```


```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df00.com_okt2) # train 데이터
word_index = tokenizer.word_index

len(word_index), word_index
```




    (15420,
     {'저': 1,
      '있다': 2,
      '빨래': 3,
      '같다': 4,
      '집안일': 5,
      '청소': 6,
      '보다': 7,
      '좋다': 8,
      '없다': 9,
      '전': 10,
      '그렇다': 11,
      '되다': 12,
      '먹다': 13,
      '싫다': 14,
      '설거지': 15,
      '남편': 16,
      '오늘': 17,
      '제': 18,
      '때': 19,
      '진짜': 20,
      '집': 21,
      '자다': 22,
      '이다': 23,
      '더': 24,
      '정리': 25,
      '오다': 26,
      '들다': 27,
      '해주다': 28,
      '신랑': 29,
      '맞다': 30,
      '해': 31,
      '개다': 32,
      '아니다': 33,
      '애': 34,
      '넘다': 35,
      '가다': 36,
      '돌리다': 37,
      '힘들다': 38,
      '거': 39,
      '것': 40,
      '일': 41,
      '시키다': 42,
      '쉬다': 43,
      '시간': 44,
      '많다': 45,
      '않다': 46,
      '버리다': 47,
      '그냥': 48,
      '밥': 49,
      '끝': 50,
      '맘': 51,
      '나다': 52,
      '싶다': 53,
      '하루': 54,
      '네': 55,
      '정말': 56,
      '해도': 57,
      '알다': 58,
      '아이': 59,
      '맛': 60,
      '치우다': 61,
      '화장실': 62,
      '또': 63,
      '귀찮다': 64,
      '넣다': 65,
      '요리': 66,
      '주말': 67,
      '그': 68,
      '지금': 69,
      '생각': 70,
      '한번': 71,
      '감사하다': 72,
      '이제': 73,
      '티': 74,
      '안': 75,
      '쓰다': 76,
      '나오다': 77,
      '아침': 78,
      '청소기': 79,
      '세탁기': 80,
      '하나': 81,
      '말': 82,
      '용': 83,
      '날': 84,
      '건조기': 85,
      '맛있다': 86,
      '옷': 87,
      '저희': 88,
      '닦다': 89,
      '저녁': 90,
      '안나': 91,
      '엄마': 92,
      '두다': 93,
      '보이다': 94,
      '왜': 95,
      '뭐': 96,
      '사람': 97,
      '말다': 98,
      '해보다': 99,
      '보내다': 100,
      '크다': 101,
      '안되다': 102,
      '모르다': 103,
      '손': 104,
      '고생': 105,
      '사다': 106,
      '쓰레기': 107,
      '정도': 108,
      '매일': 109,
      '제일': 110,
      '놓다': 111,
      '널다': 112,
      '요즘': 113,
      '주다': 114,
      '좋아하다': 115,
      '바쁘다': 116,
      '부럽다': 117,
      '편하다': 118,
      '번': 119,
      '아기': 120,
      '차다': 121,
      '화이팅': 122,
      '받다': 123,
      '젤': 124,
      '커피': 125,
      '살다': 126,
      '씻다': 127,
      '힘드다': 128,
      '힘내다': 129,
      '도와주다': 130,
      '놀다': 131,
      '우리': 132,
      '나가다': 133,
      '시작': 134,
      '괜찮다': 135,
      '어제': 136,
      '후': 137,
      '혼자': 138,
      '계속': 139,
      '다시': 140,
      '보고': 141,
      '퇴근': 142,
      '끄다': 143,
      '야하다': 144,
      '음식': 145,
      '애기': 146,
      '이쁘다': 147,
      '이불': 148,
      '몸': 149,
      '물': 150,
      '공감': 151,
      '내': 152,
      '나': 153,
      '잇다': 154,
      '반찬': 155,
      '만들다': 156,
      '지다': 157,
      '이렇다': 158,
      '가끔': 159,
      '분리수거': 160,
      '눈': 161,
      '부지런하다': 162,
      '듯': 163,
      '구': 164,
      '못': 165,
      '내일': 166,
      '일이': 167,
      '걸레질': 168,
      '거의': 169,
      '늘다': 170,
      '남자': 171,
      '기분': 172,
      '안해': 173,
      '대단하다': 174,
      '완전': 175,
      '개': 176,
      '글': 177,
      '걸': 178,
      '아들': 179,
      '육아': 180,
      '써다': 181,
      '내다': 182,
      '끝내다': 183,
      '음': 184,
      '땐': 185,
      '잔소리': 186,
      '피곤하다': 187,
      '늦다': 188,
      '덥다': 189,
      '꼭': 190,
      '수건': 191,
      '유': 192,
      '귀엽다': 193,
      '조금': 194,
      '늘': 195,
      '수': 196,
      '게': 197,
      '깔끔하다': 198,
      '미루다': 199,
      '아프다': 200,
      '챙기다': 201,
      '자기': 202,
      '먼지': 203,
      '결혼': 204,
      '마음': 205,
      '가요': 206,
      '분': 207,
      '두': 208,
      '살': 209,
      '눕다': 210,
      '일어나다': 211,
      '여': 212,
      '빨다': 213,
      '쓸다': 214,
      '통': 215,
      '예쁘다': 216,
      '바닥': 217,
      '끝나다': 218,
      '가보다': 219,
      '재우다': 220,
      '오': 221,
      '준비': 222,
      '일찍': 223,
      '라면': 224,
      '기다': 225,
      '쌓이다': 226,
      '항상': 227,
      '설겆': 228,
      '깨끗하다': 229,
      '집안': 230,
      '출근': 231,
      '움직이다': 232,
      '중': 233,
      '줄': 234,
      '돼다': 235,
      '드리다': 236,
      '고민': 237,
      '점심': 238,
      '그때': 239,
      '남다': 240,
      '어떻다': 241,
      '행복하다': 242,
      '방': 243,
      '서다': 244,
      '너': 245,
      '식': 246,
      '한잔': 247,
      '쉬': 248,
      '음식물': 249,
      '건': 250,
      '그게': 251,
      '시르다': 252,
      '키우다': 253,
      '밤': 254,
      '앉다': 255,
      '댓글': 256,
      '그릇': 257,
      '둘째': 258,
      '다르다': 259,
      '달': 260,
      '아주': 261,
      '바로': 262,
      '대충': 263,
      '갖다': 264,
      '살림': 265,
      '꺼내다': 266,
      '들어가다': 267,
      '되어다': 268,
      '자주': 269,
      '주': 270,
      '역시': 271,
      '스트레스': 272,
      '똑같다': 273,
      '일단': 274,
      '종일': 275,
      '얘기': 276,
      '고맙다': 277,
      '자꾸': 278,
      '곳': 279,
      '수고': 280,
      '즐겁다': 281,
      '돈': 282,
      '본인': 283,
      '포기': 284,
      '대신': 285,
      '시원하다': 286,
      '다니다': 287,
      '햇': 288,
      '언제': 289,
      '갈다': 290,
      '임신': 291,
      '어디': 292,
      '마시다': 293,
      '세탁': 294,
      '아가': 295,
      '생기다': 296,
      '다해': 297,
      '앗': 298,
      '멋지다': 299,
      '맞벌이': 300,
      '냉장고': 301,
      '끼': 302,
      '벌써': 303,
      '낮잠': 304,
      '누가': 305,
      '낼': 306,
      '막': 307,
      '이르다': 308,
      '입다': 309,
      '빼다': 310,
      '주부': 311,
      '느낌': 312,
      '들어오다': 313,
      '싫어하다': 314,
      '오후': 315,
      '주방': 316,
      '걸레': 317,
      '걸리다': 318,
      '걱정': 319,
      '삶다': 320,
      '잠': 321,
      '둘': 322,
      '들이다': 323,
      '그것': 324,
      '깔다': 325,
      '비슷하다': 326,
      '최고': 327,
      '일주일': 328,
      '겠다': 329,
      '그거': 330,
      '이번': 331,
      '세상': 332,
      '어지르다': 333,
      '데': 334,
      '여유': 335,
      '중이': 336,
      '싹': 337,
      '칭찬': 338,
      '기도': 339,
      '뭔가': 340,
      '식기세척기': 341,
      '찾다': 342,
      '멀다': 343,
      '금방': 344,
      '달다': 345,
      '거실': 346,
      '위': 347,
      '기': 348,
      '날씨': 349,
      '어리다': 350,
      '친구': 351,
      '뒤': 352,
      '서로': 353,
      '평일': 354,
      '희다': 355,
      '오전': 356,
      '다행': 357,
      '힘': 358,
      '소리': 359,
      '담당': 360,
      '그리다': 361,
      '차리다': 362,
      '아빠': 363,
      '성격': 364,
      '카페': 365,
      '속': 366,
      '빠지다': 367,
      '퇴': 368,
      '뭘': 369,
      '원래': 370,
      '비': 371,
      '대박': 372,
      '때문': 373,
      '세기': 374,
      '장난감': 375,
      '몰다': 376,
      '하니': 377,
      '다림질': 378,
      '서랍': 379,
      '자고': 380,
      '빵': 381,
      '티비': 382,
      '달라': 383,
      '영': 384,
      '다녀오다': 385,
      '언니': 386,
      '잠시': 387,
      '싸우다': 388,
      '목욕': 389,
      '끓이다': 390,
      '욕실': 391,
      '낳다': 392,
      '반복': 393,
      '딸': 394,
      '지치다': 395,
      '어렵다': 396,
      '낫다': 397,
      '사용': 398,
      '가지': 399,
      '이해': 400,
      '먼저': 401,
      '필요하다': 402,
      '사랑': 403,
      '돌아가다': 404,
      '잡다': 405,
      '싸다': 406,
      '알': 407,
      '기다리다': 408,
      '집도': 409,
      '네네': 410,
      '남': 411,
      '빨': 412,
      '사실': 413,
      '나중': 414,
      '여름': 415,
      '표': 416,
      '배': 417,
      '더럽다': 418,
      '설겆이': 419,
      '친정': 420,
      '다음': 421,
      '푹': 422,
      '기계': 423,
      '고': 424,
      '바': 425,
      '젠': 426,
      '난': 427,
      '사진': 428,
      '여자': 429,
      '전업': 430,
      '냄새': 431,
      '뜨다': 432,
      '가하다': 433,
      '도우미': 434,
      '즐기다': 435,
      '안다': 436,
      '담다': 437,
      '마무리': 438,
      '넹': 439,
      '적다': 440,
      '치다': 441,
      '쥬': 442,
      '정신': 443,
      '다': 444,
      '님': 445,
      '니': 446,
      '깨다': 447,
      '새벽': 448,
      '부르다': 449,
      '편이': 450,
      '나누다': 451,
      '대다': 452,
      '맞추다': 453,
      '운동': 454,
      '와우': 455,
      '바꾸다': 456,
      '짜증': 457,
      '제자리': 458,
      '입': 459,
      '느끼다': 460,
      '여기': 461,
      '거리': 462,
      '밀리다': 463,
      '계시다': 464,
      '따다': 465,
      '빠르다': 466,
      '얼른': 467,
      '등': 468,
      '집다': 469,
      '말씀': 470,
      '날다': 471,
      '얼마나': 472,
      '개월': 473,
      '모두': 474,
      '쉬엄쉬엄': 475,
      '반': 476,
      '머리': 477,
      '다른': 478,
      '접다': 479,
      '타다': 480,
      '는걸': 481,
      '찌다': 482,
      '커피한잔': 483,
      '보이': 484,
      '불다': 485,
      '죽다': 486,
      '침대': 487,
      '이모': 488,
      '마르다': 489,
      '동안': 490,
      '얼마': 491,
      '첫째': 492,
      '등원': 493,
      '짜다': 494,
      '로봇청소기': 495,
      '만해': 496,
      '옆': 497,
      '미리': 498,
      '앞': 499,
      '문제': 500,
      '니당다': 501,
      '시': 502,
      '참다': 503,
      '열다': 504,
      '나서다': 505,
      '만': 506,
      '넣기': 507,
      '물걸레': 508,
      '차': 509,
      '따르다': 510,
      '싱크대': 511,
      '무조건': 512,
      '궁금하다': 513,
      '기전': 514,
      '셋': 515,
      '욤': 516,
      '어찌': 517,
      '정': 518,
      '세척': 519,
      '돌아서다': 520,
      '축하': 521,
      '건지다': 522,
      '습': 523,
      '김치': 524,
      '울': 525,
      '곧': 526,
      '애가': 527,
      '쇼파': 528,
      '먹이다': 529,
      '셧': 530,
      '별로': 531,
      '이틀': 532,
      '편': 533,
      '개운하다': 534,
      '비우다': 535,
      '식사': 536,
      '외': 537,
      '세제': 538,
      '몫': 539,
      '그대로': 540,
      '베란다': 541,
      '덜': 542,
      '잠들다': 543,
      '쌓다': 544,
      '읽다': 545,
      '겨울': 546,
      '심': 547,
      '샤워': 548,
      '신경': 549,
      '작다': 550,
      '등등': 551,
      '간식': 552,
      '옷장': 553,
      '진심': 554,
      '배우다': 555,
      '안좋다': 556,
      '가정': 557,
      '기본': 558,
      '아깝다': 559,
      '제대로': 560,
      '배달': 561,
      '처음': 562,
      '세': 563,
      '가능하다': 564,
      '터지다': 565,
      '잠깐': 566,
      '가사': 567,
      '워킹맘': 568,
      '방법': 569,
      '매번': 570,
      '뿌듯하다': 571,
      '모습': 572,
      '미치다': 573,
      '걸다': 574,
      '마자': 575,
      '스스로': 576,
      '올리다': 577,
      '회사': 578,
      '이상하다': 579,
      '산더미': 580,
      '보기': 581,
      '돌다': 582,
      '디': 583,
      '사먹다': 584,
      '털다': 585,
      '함': 586,
      '은근': 587,
      '지나다': 588,
      '밖': 589,
      '척': 590,
      '예전': 591,
      '걷다': 592,
      '보': 593,
      '당연하다': 594,
      '아': 595,
      '비다': 596,
      '이건': 597,
      '짜증나다': 598,
      '도움': 599,
      '머리카락': 600,
      '건가': 601,
      '일어나서': 602,
      '차라리': 603,
      '대요': 604,
      '다가': 605,
      '가면': 606,
      '정리정돈': 607,
      '답답하다': 608,
      '절대': 609,
      '부분': 610,
      '밉다': 611,
      '짐': 612,
      '시댁': 613,
      '습관': 614,
      '티나': 615,
      '양말': 616,
      '끼다': 617,
      '내려놓다': 618,
      '추다': 619,
      '떨어지다': 620,
      '부탁': 621,
      '헐다': 622,
      '홧팅': 623,
      '뭐라다': 624,
      '모으다': 625,
      '급': 626,
      '고프다': 627,
      '주무시다': 628,
      '혹시': 629,
      '거기': 630,
      '벌다': 631,
      '만나다': 632,
      '외출': 633,
      '한숨': 634,
      '자리': 635,
      '젖병': 636,
      '몇번': 637,
      '아쉽다': 638,
      '분담': 639,
      '열': 640,
      '직장': 641,
      '어머': 642,
      '재활용': 643,
      '정보': 644,
      '장': 645,
      '죵': 646,
      '머': 647,
      '모든': 648,
      '달달': 649,
      '간단하다': 650,
      '먹이': 651,
      '듣다': 652,
      '찌': 653,
      '하라': 654,
      '배고프다': 655,
      '물건': 656,
      '부지런': 657,
      '예랑': 658,
      '엉망': 659,
      '중간': 660,
      '환기': 661,
      '아예': 662,
      '보통': 663,
      '전쟁': 664,
      '에고': 665,
      '직접': 666,
      '맡다': 667,
      '산': 668,
      '가족': 669,
      '나머지': 670,
      '느껴지다': 671,
      '이유식': 672,
      '꼼꼼하다': 673,
      '처리': 674,
      '하원': 675,
      '메뉴': 676,
      '병': 677,
      '지저분하다': 678,
      '인': 679,
      '이야기': 680,
      '생각나다': 681,
      '왜케': 682,
      '순간': 683,
      '아하': 684,
      '이지': 685,
      '매트': 686,
      '로봇': 687,
      '마지막': 688,
      '이사': 689,
      '공부': 690,
      '넵': 691,
      '식구': 692,
      '각자': 693,
      '부부': 694,
      '무': 695,
      '생활': 696,
      '슬프다': 697,
      '스타일': 698,
      '신혼': 699,
      '위해': 700,
      '노력': 701,
      '겨우': 702,
      '안보': 703,
      '좋아지다': 704,
      '점': 705,
      '비싸다': 706,
      '때리다': 707,
      '코로나': 708,
      '따뜻하다': 709,
      '월요일': 710,
      '하나요': 711,
      '컵': 712,
      '해달라다': 713,
      '땡기다': 714,
      '사고': 715,
      '기억': 716,
      '점점': 717,
      '갑자기': 718,
      '웃다': 719,
      '아아': 720,
      '담': 721,
      '기저귀': 722,
      '켜다': 723,
      '체력': 724,
      '게으르다': 725,
      '내리다': 726,
      '지내다': 727,
      '조심하다': 728,
      '사서': 729,
      '그치다': 730,
      '제품': 731,
      '기요': 732,
      '울다': 733,
      '심하다': 734,
      '해먹': 735,
      '찍다': 736,
      '솜씨': 737,
      '말리다': 738,
      '담그다': 739,
      '이기다': 740,
      '길': 741,
      '추천': 742,
      '방학': 743,
      '속이다': 744,
      '병원': 745,
      '먹기': 746,
      '미안하다': 747,
      '출산': 748,
      '든든하다': 749,
      '래야': 750,
      '가득': 751,
      '가르치다': 752,
      '전부': 753,
      '재료': 754,
      '나이': 755,
      '닫다': 756,
      '하므다': 757,
      '마트': 758,
      '나름': 759,
      '허리': 760,
      '누구': 761,
      '차려': 762,
      '사주다': 763,
      '밀다': 764,
      '가장': 765,
      '덕분': 766,
      '키': 767,
      '존경': 768,
      '딱하다': 769,
      '어린이집': 770,
      '운동화': 771,
      '하하': 772,
      '바람': 773,
      '초': 774,
      '신세계': 775,
      '악': 776,
      '재밌다': 777,
      '거나': 778,
      '나쁘다': 779,
      '무리하다': 780,
      '지나가다': 781,
      '쪽': 782,
      '잔': 783,
      '자체': 784,
      '오히려': 785,
      '변기': 786,
      '어차피': 787,
      '기준': 788,
      '욬': 789,
      '미니': 790,
      '불': 791,
      '후다닥': 792,
      '거들다': 793,
      '문': 794,
      '땀': 795,
      '가야': 796,
      '쉽다': 797,
      '채우다': 798,
      '욧': 799,
      '답': 800,
      '힝': 801,
      '케어': 802,
      '학교': 803,
      '다리': 804,
      '틈': 805,
      '일도': 806,
      '최대한': 807,
      '펴다': 808,
      '적': 809,
      '믿다': 810,
      '버티다': 811,
      '요새': 812,
      '난리': 813,
      '뒷정리': 814,
      '감사': 815,
      '꾸다': 816,
      '무한': 817,
      '알아보다': 818,
      '독박': 819,
      '무섭다': 820,
      '책': 821,
      '그니': 822,
      '긋다': 823,
      '데리': 824,
      '구매': 825,
      '큰일': 826,
      '입덧': 827,
      '감': 828,
      '드': 829,
      '틀다': 830,
      '상태': 831,
      '락스': 832,
      '사오다': 833,
      '장난': 834,
      '김밥': 835,
      '주변': 836,
      '충전': 837,
      '낮': 838,
      '평소': 839,
      '도안': 840,
      '이상': 841,
      '중요하다': 842,
      '해주시': 843,
      '얼': 844,
      '하자': 845,
      '어른': 846,
      '현실': 847,
      '상황': 848,
      '완벽하다': 849,
      '푸다': 850,
      '놈': 851,
      '새다': 852,
      '확': 853,
      '돌': 854,
      '동생': 855,
      '맥주': 856,
      '물티슈': 857,
      '정답': 858,
      '놔두다': 859,
      '속상하다': 860,
      '금손': 861,
      '관리': 862,
      '쫌': 863,
      '전혀': 864,
      '고해': 865,
      '바라다': 866,
      '먹음': 867,
      '바뀌다': 868,
      '선물': 869,
      '노래': 870,
      '반성': 871,
      '자유시간': 872,
      '씻기다': 873,
      '반갑다': 874,
      '기름': 875,
      '확인': 876,
      '땜': 877,
      '감기': 878,
      '고르다': 879,
      '확실하다': 880,
      '살짝': 881,
      '꽃': 882,
      '창틀': 883,
      '가해': 884,
      '창문': 885,
      '태어나다': 886,
      '노동': 887,
      '넘어가다': 888,
      '주문': 889,
      '국': 890,
      '건조': 891,
      '사': 892,
      '명': 893,
      '각': 894,
      '계절': 895,
      '년': 896,
      '가게': 897,
      '이따': 898,
      '스럽다': 899,
      '타임': 900,
      '실': 901,
      '어유': 902,
      '뿌리다': 903,
      '구만': 904,
      '색': 905,
      '고요': 906,
      '돌아오다': 907,
      '건강': 908,
      '건조대': 909,
      '가스렌지': 910,
      '다리다': 911,
      '고치다': 912,
      '생': 913,
      '착하다': 914,
      '며칠': 915,
      '와중': 916,
      '힐링': 917,
      '해결': 918,
      '신나다': 919,
      '얼집': 920,
      '검색': 921,
      '불편하다': 922,
      '믹스': 923,
      '뚜껑': 924,
      '친정엄마': 925,
      '신기하다': 926,
      '부족하다': 927,
      '휴일': 928,
      '굿밤': 929,
      '파': 930,
      '앜': 931,
      '째다': 932,
      '발': 933,
      '꼴': 934,
      '여기저기': 935,
      '엄청나다': 936,
      '눈뜨다': 937,
      '추가': 938,
      '손목': 939,
      '과일': 940,
      '아무': 941,
      '널': 942,
      '안방': 943,
      '뎅': 944,
      '물기': 945,
      '도전': 946,
      '나니': 947,
      '런가': 948,
      '멍': 949,
      '차이': 950,
      '방금': 951,
      '뜨겁다': 952,
      '속옷': 953,
      '술': 954,
      '겁니다': 955,
      '짱': 956,
      '망': 957,
      '엄두': 958,
      '교육': 959,
      '야채': 960,
      '우유': 961,
      '줄다': 962,
      '대부분': 963,
      '아내': 964,
      '가격': 965,
      '적당하다': 966,
      '와이프': 967,
      '상': 968,
      '미세먼지': 969,
      '부엌': 970,
      '밀대': 971,
      '고양이': 972,
      '허다': 973,
      '더더': 974,
      '드라마': 975,
      '화': 976,
      '프로': 977,
      '무슨': 978,
      '외식': 979,
      '이면': 980,
      '과자': 981,
      '묵다': 982,
      '돌이': 983,
      '가시다': 984,
      '겁나다': 985,
      '해봤다': 986,
      '민트': 987,
      '구입': 988,
      '무선': 989,
      '던지다': 990,
      '온': 991,
      '크게': 992,
      '향': 993,
      '생기': 994,
      '코': 995,
      '재미': 996,
      '이후': 997,
      '밑': 998,
      '첨': 999,
      '토닥토닥': 1000,
      ...})




```python
df00_pad = tokenizer.texts_to_sequences(df00.com_okt2) 

print(df00_pad[0])
print(df00.com_okt2.iloc[0])
```

    [298, 208, 402, 909, 879, 309, 488, 26, 32, 19, 408, 339, 1232, 271, 39, 33, 62, 191, 76, 3703, 191, 36, 574, 31, 488, 464, 117, 192, 10, 4654, 3116, 891, 137, 87, 479, 178, 141, 1, 20, 2, 53, 20, 525, 21, 2, 8, 11, 10, 616, 953, 25, 110, 1407, 221, 68, 423, 4655, 8633, 3704, 5366, 268, 965, 966, 157, 10, 3, 32, 466, 1115, 2404, 268, 39, 1, 7, 2908, 80, 85, 3705, 5367, 2158, 65, 40, 12, 46, 339, 99, 10, 85, 9, 3, 245, 39, 57, 721, 3380, 310, 51, 13, 31, 56, 108, 605, 3, 2159, 203, 226, 32, 458, 65, 1091, 18, 276, 234, 58, 271, 39, 33, 35, 1009, 235, 205, 709, 208, 3, 112, 32, 249, 107, 47, 339, 3, 32, 807, 199, 270, 352, 32, 30, 113, 722, 3117, 77, 107, 47, 1740, 64, 494, 1741, 2405, 729, 3118, 3119, 42, 29, 621, 47, 526, 415, 26, 84, 3381, 323, 8634, 324, 1376, 1, 3, 245, 197, 110, 14, 85, 715, 505, 261, 8, 585, 245, 197, 95, 14, 85, 188, 310, 12, 80, 150, 431, 505, 404, 262, 310, 188, 310, 102, 1, 64, 3, 218, 5368, 153, 1162, 2, 85, 1688, 85, 1116, 8635, 91, 35, 8, 1, 3, 32, 124, 64, 184, 56, 95, 64, 459, 250, 9, 87, 63, 45, 3382, 3, 1451, 412, 112, 32, 65, 14, 113, 253, 394, 4081, 130, 472, 8, 103, 26, 1343, 1640, 130, 117, 394, 2283, 12, 169, 8636, 119, 1010, 68, 10, 3, 32, 80, 85, 1117, 114, 35, 277, 68, 19, 1545, 87, 5369, 415, 26, 8, 208, 3, 112, 135, 137, 305, 28, 30, 538, 2737, 1116, 3120, 585, 942, 185, 2284, 489, 137, 500, 1971, 1, 3, 110, 64, 245, 40, 32, 65, 199, 199, 4082, 698, 2564, 45, 192, 56, 74, 91, 1814, 4656, 81, 1506, 730, 74, 853, 52, 2564, 45, 340, 1009, 12, 369, 143, 153, 158, 33, 3, 14, 10, 15, 24, 14, 372, 30, 1742, 617, 531, 1815, 341, 495, 106, 184, 6510, 6, 14, 245, 39, 8, 32, 553, 65, 8637, 68, 65, 9, 307, 1, 85, 2, 118, 8, 3, 32, 167, 65, 1141, 9, 1069, 544, 93, 56, 65, 9, 85, 514, 489, 309, 953, 176, 892, 111, 24, 11, 219, 880, 87, 1452, 9, 4, 11, 33, 197, 95, 1009, 12, 136, 1893, 3, 17, 78, 32, 16, 141, 2, 469, 153, 173, 1011, 36, 32, 1453, 16, 1288, 181, 16, 95, 82, 46, 103, 42, 46, 1816, 147, 123, 4083, 87, 47, 70, 2160, 32, 70]
    ['앗', '두', '필요하다', '건조대', '고르다', '입다', '이모', '오다', '개다', '때', '기다리다', '기도', '능', '역시', '거', '아니다', '화장실', '수건', '쓰다', '더미', '수건', '가다', '걸다', '해', '이모', '계시다', '부럽다', '유', '전', '미국', '광고', '건조', '후', '옷', '접다', '걸', '보고', '저', '진짜', '있다', '싶다', '진짜', '울', '집', '있다', '좋다', '그렇다', '전', '양말', '속옷', '정리', '제일', '흠', '오', '그', '기계', '국내', '도입', '시급하다', '상용', '되어다', '가격', '적당하다', '지다', '전', '빨래', '개다', '빠르다', '핫', '개발', '되어다', '거', '저', '보다', '빌트', '세탁기', '건조기', '붙박다', '이장', '세트', '넣다', '것', '되다', '않다', '기도', '해보다', '전', '건조기', '없다', '빨래', '너', '거', '해도', '담', '진도', '빼다', '맘', '먹다', '해', '정말', '정도', '다가', '빨래', '불쌍하다', '먼지', '쌓이다', '개다', '제자리', '넣다', '줍다', '제', '얘기', '줄', '알다', '역시', '거', '아니다', '넘다', '위로', '돼다', '마음', '따뜻하다', '두', '빨래', '널다', '개다', '음식물', '쓰레기', '버리다', '기도', '빨래', '개다', '최대한', '미루다', '주', '뒤', '개다', '맞다', '요즘', '기저귀', '음청', '나오다', '쓰레기', '버리다', '가기', '귀찮다', '짜다', '종량제', '봉투', '사서', '오래오래', '숙성', '시키다', '신랑', '부탁', '버리다', '곧', '여름', '오다', '날', '파리', '들이다', '계셧', '그것', '예정', '저', '빨래', '너', '게', '제일', '싫다', '건조기', '사고', '나서다', '아주', '좋다', '털다', '너', '게', '왜', '싫다', '건조기', '늦다', '빼다', '되다', '세탁기', '물', '냄새', '나서다', '돌아가다', '바로', '빼다', '늦다', '빼다', '안되다', '저', '귀찮다', '빨래', '끝나다', '멜로디', '나', '한참', '있다', '건조기', '옮기다', '건조기', '덕', '쉰내', '안나', '넘다', '좋다', '저', '빨래', '개다', '젤', '귀찮다', '음', '정말', '왜', '귀찮다', '입', '건', '없다', '옷', '또', '많다', '아이러니', '빨래', '관련', '빨', '널다', '개다', '넣다', '싫다', '요즘', '키우다', '딸', '학년', '도와주다', '얼마나', '좋다', '모르다', '오다', '초딩', '따님', '도와주다', '부럽다', '딸', '쯤', '되다', '거의', '강산', '번', '변하다', '그', '전', '빨래', '개다', '세탁기', '건조기', '일해', '주다', '넘다', '고맙다', '그', '때', '뿐이다', '옷', '가벼워지다', '여름', '오다', '좋다', '두', '빨래', '널다', '괜찮다', '후', '누가', '해주다', '맞다', '세제', '향기', '덕', '탁탁', '털다', '널', '땐', '상쾌하다', '마르다', '후', '문제', '항', '저', '빨래', '제일', '귀찮다', '너', '것', '개다', '넣다', '미루다', '미루다', '마지못하다', '스타일', '동지', '많다', '유', '정말', '티', '안나', '잡', '안일', '하나', '잖다', '그치다', '티', '확', '나다', '동지', '많다', '뭔가', '위로', '되다', '뭘', '끄다', '나', '이렇다', '아니다', '빨래', '싫다', '전', '설거지', '더', '싫다', '대박', '맞다', '고무장갑', '끼다', '별로', '사가', '식기세척기', '로봇청소기', '사다', '음', '제보', '청소', '싫다', '너', '거', '좋다', '개다', '옷장', '넣다', '왜케시른', '그', '넣다', '없다', '막', '저', '건조기', '있다', '편하다', '좋다', '빨래', '개다', '일이', '넣다', '공간', '없다', '바구니', '쌓다', '두다', '정말', '넣다', '없다', '건조기', '기전', '마르다', '입다', '속옷', '개', '사', '놓다', '더', '그렇다', '가보다', '확실하다', '옷', '필요', '없다', '같다', '그렇다', '아니다', '게', '왜', '위로', '되다', '어제', '건조하다', '빨래', '오늘', '아침', '개다', '남편', '보고', '있다', '집다', '나', '안해', '눈치', '가다', '개다', '종종', '남편', '찬스', '써다', '남편', '왜', '말', '않다', '모르다', '시키다', '않다', '척척', '이쁘다', '받다', '문득', '옷', '버리다', '생각', '드네', '개다', '생각']
    


```python
# 딥러닝 모델에 넣을 용도로 사전에 훈련시킨 워드임베딩 데이터를 불러옵니다

import os
embedding_dict = {}
f = open(os.path.join('', 'imdb_embedding_word2vec_comment.txt'),  encoding = "utf-8")
for line in f: # 각 line은 단어, 임베딩백터값으로 구성된 하나의 문자열
    values = line.split() # [단어, 벡터값] 리스트 형성
    word = values[0] # 단어
    coefs = np.asarray(values[1:]) # 벡터
    embedding_dict[word] = coefs # key:단어 value:벡터
f.close()

embedding_dict
# 신경망에 사용할 embedding matrix 생성
embedding_matrix = np.zeros((len(embedding_dict), EMBEDDING_DIM))

print(len(embedding_dict))

# 여기서 word_index에선 OOV가 1입니다 
# word는 단어  i는 단어와 대응되는 정수토큰입니다 (숫자가 작을수록 빈도가 높습니다)
for word, i in word_index.items(): 
    if i >= len(embedding_dict): 
        continue      
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None: # get했는데 없으면 None 돌려줌
        embedding_matrix[i] = embedding_vector
print(np.shape(embedding_matrix))
print(embedding_matrix)
```

    15421
    (15421, 20)
    [[ 0.          0.          0.         ...  0.          0.
       0.        ]
     [-0.11181639 -0.15433578  0.942294   ...  0.08553856  0.18033539
       0.4665472 ]
     [-0.49699506  0.1189518   0.350456   ...  0.17615788  0.3980091
       0.7140281 ]
     ...
     [-0.05898003  0.09981006  0.17681794 ...  0.02728574  0.12766966
       0.07624301]
     [-0.07085981  0.08854144  0.15260004 ...  0.07493042  0.15008639
       0.11541756]
     [-0.07360359  0.08778126  0.18871379 ...  0.02020964  0.12866162
       0.07565418]]
    


```python
def doc_vectors(padding):
    document_embedding_list = []

    # 각 문장은 리스트의 리스트 형태로 숫자인코딩 된 벡터
    for line in padding:
        doc2vec = np.zeros(EMBEDDING_DIM) # 0값의 벡터를 만들어줍니다
        count = 0 # 평균을 내기위해 count해줍니다
        for token in line: # 정수토큰 하나하나를 가져옵니다                    
            if token in np.arange(1,len(embedding_matrix)):  # 제로패딩은 count안하게, Vocab_size까지만
                count += 1 
                # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
                if doc2vec is np.zeros(EMBEDDING_DIM): # 첫번째 단어때 필요한 문법
                    doc2vec = embedding_matrix[token]
                else:
                    doc2vec = doc2vec + embedding_matrix[token] # 단어의 w2v를 누적해서 더해줍니다
        
        # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠줍니다 = 문장 벡터
        doc2vec_average = doc2vec / (count+1) # 혹시나 있을 zero-divdend방지위해 +1
        document_embedding_list.append(doc2vec_average)

    # 각 문서에 대한 문서 벡터 리스트를 리턴
    return document_embedding_list
```


```python
com_vectors = doc_vectors(df00_pad)

# 각 문장을 단어평균 임베딩 벡터로
len(com_vectors),com_vectors[:10]
```




    (1139,
     [array([-0.4658676 , -0.01614783,  0.72571184, -0.384934  , -0.6727707 ,
              0.50545242,  0.27329041,  0.06997426, -0.65679376,  0.64757519,
              0.41641222, -0.15811663,  0.23359296, -0.37472742,  0.18272615,
              0.70092192,  0.29772215,  0.02979328,  0.30822611,  0.72659843]),
      array([-0.42481756, -0.09782244,  0.71398757, -0.3937993 , -0.57032554,
              0.58068676,  0.26196838,  0.10018336, -0.66077377,  0.68322961,
              0.4375357 , -0.12272573,  0.27908012, -0.44214521,  0.24694307,
              0.7182811 ,  0.273209  ,  0.08900394,  0.36349303,  0.67769026]),
      array([-0.25924049,  0.03374762,  0.67544996, -0.39963145, -0.46071247,
              0.53120876,  0.31188744,  0.13577138, -0.35553684,  0.56976045,
              0.00870236,  0.07198517,  0.1126707 , -0.25150391,  0.42613138,
              0.48171461,  0.30434027,  0.11380419,  0.20865462,  0.39723797]),
      array([-0.52152291, -0.25560217,  0.9159699 , -0.30795197, -0.69503847,
              0.52290614,  0.03899106, -0.00551495, -0.4670266 ,  0.89330986,
              0.16725166,  0.28252507,  0.32792206, -0.77704723,  0.56099208,
              0.79724499,  0.45014921,  0.08510941, -0.28680364,  0.54049346]),
      array([-0.43951297, -0.11459536,  0.7062616 , -0.35782013, -0.49192128,
              0.67930634,  0.10815773,  0.10336976, -0.51183098,  0.71853566,
              0.27524168,  0.02081598,  0.11573805, -0.36473243,  0.27085162,
              0.6265545 ,  0.23997892,  0.17073442,  0.15492775,  0.57895618]),
      array([-0.47224492, -0.07740531,  0.76313233, -0.37852413, -0.59363091,
              0.44742115,  0.21766907,  0.10799124, -0.72091855,  0.64740806,
              0.48425362, -0.28964596,  0.18953588, -0.46872477,  0.22314245,
              0.70003952,  0.27625058,  0.12993573,  0.39818564,  0.68013569]),
      array([-0.42946929, -0.21980021,  0.84006729, -0.41960804, -0.61244386,
              0.59485741,  0.09219326,  0.06196837, -0.46206732,  0.77090095,
              0.33694249,  0.00437393,  0.29761547, -0.56611593,  0.35743179,
              0.7271816 ,  0.32270438, -0.0797658 ,  0.03459976,  0.58074344]),
      array([-0.40063743, -0.1009926 ,  0.66583879, -0.36154214, -0.55448794,
              0.58604937,  0.22013563,  0.07606964, -0.79600795,  0.71762657,
              0.58107275, -0.32169861,  0.21553745, -0.59062183,  0.13049731,
              0.79720137,  0.26788436,  0.21135566,  0.4538367 ,  0.73730143]),
      array([-0.39407408, -0.14518003,  0.7219837 , -0.35058785, -0.56370124,
              0.54686773,  0.26634024,  0.08222737, -0.77021877,  0.72581265,
              0.51319163, -0.27282458,  0.25381833, -0.52859066,  0.21220135,
              0.79521521,  0.28085085,  0.14089817,  0.4064154 ,  0.69278634]),
      array([-0.42430786, -0.10042099,  0.82310339, -0.53983922, -0.45617692,
              0.62370138,  0.29588979,  0.12922875, -0.36109992,  0.62611917,
              0.06452565,  0.17294822,  0.30369525, -0.39306329,  0.39738955,
              0.50221108,  0.28001256,  0.1503284 ,  0.09978781,  0.49026628])])




```python
# 시각화가 필요없을 때 여기서 바로 시작합니다
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import silhouette_score

def visualize_silhouette_layer(data, num_cluster):
    clusters_range = range(2,int(num_cluster))
    results = []

    for i in clusters_range:
        clusterer = AgglomerativeClustering(n_clusters=i,linkage='ward')
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        results.append([i, silhouette_avg])

    result = pd.DataFrame(results, columns=["n_clusters", "silhouette_score"])
    pivot_ac = pd.pivot_table(result, index="n_clusters", values="silhouette_score")

    return result, pivot_ac
```


```python
from gensim.models import Word2Vec
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 벡터크기 = [10,30,100,300,500]
# 윈도우 = [3,5,10,15,20]

벡터크기 = [500]
윈도우 = [3]
for EMBEDDING_DIM in 벡터크기:
    for WINDOW_SIZE in 윈도우:

        # 임베딩 크기는 논문을 따름
        model = Word2Vec(sentences=df00.com_okt2, sg=1, size=EMBEDDING_DIM, window=WINDOW_SIZE, min_count=1) #sg 0은 CBOW, 1은 SKIP-GRAM
        w2v_vocab = list(model.wv.vocab) # 임베딩 된 단어 리스트

        # save model in ASCII (word2vec) format
        # 텍스트 파일로 단어들의 임베딩 벡터 저장
        filename = 'imdb_embedding_word2vec_comment.txt'
        model.wv.save_word2vec_format(filename, binary=False)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df00.com_okt2) # train 데이터
        word_index = tokenizer.word_index

        df00_pad = tokenizer.texts_to_sequences(df00.com_okt2) 

        # 딥러닝 모델에 넣을 용도로 사전에 훈련시킨 워드임베딩 데이터를 불러옵니다

        embedding_dict = {}
        f = open(os.path.join('', 'imdb_embedding_word2vec_comment.txt'),  encoding = "utf-8")
        for line in f: # 각 line은 단어, 임베딩백터값으로 구성된 하나의 문자열
            values = line.split() # [단어, 벡터값] 리스트 형성
            word = values[0] # 단어
            coefs = np.asarray(values[1:]) # 벡터
            embedding_dict[word] = coefs # key:단어 value:벡터
        f.close()

        embedding_dict
        # 신경망에 사용할 embedding matrix 생성
        embedding_matrix = np.zeros((len(embedding_dict), EMBEDDING_DIM))


        # 여기서 word_index에선 OOV가 1입니다 
        # word는 단어  i는 단어와 대응되는 정수토큰입니다 (숫자가 작을수록 빈도가 높습니다)
        for word, i in word_index.items(): 
            if i >= len(embedding_dict): 
                continue      
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None: # get했는데 없으면 None 돌려줌
                embedding_matrix[i] = embedding_vector

        def doc_vectors(padding):
            document_embedding_list = []

            # 각 문장은 리스트의 리스트 형태로 숫자인코딩 된 벡터
            for line in tqdm(padding):
                doc2vec = np.zeros(EMBEDDING_DIM) # 0값의 벡터를 만들어줍니다
                count = 0 # 평균을 내기위해 count해줍니다
                for token in line: # 정수토큰 하나하나를 가져옵니다                    
                    if token in np.arange(1,len(embedding_matrix)):  # 제로패딩은 count안하게, Vocab_size까지만
                        count += 1 
                        # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
                        if doc2vec is np.zeros(EMBEDDING_DIM): # 첫번째 단어때 필요한 문법
                            doc2vec = embedding_matrix[token]
                        else:
                            doc2vec = doc2vec + embedding_matrix[token] # 단어의 w2v를 누적해서 더해줍니다

                # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠줍니다 = 문장 벡터
                doc2vec_average = doc2vec / (count+1) # 혹시나 있을 zero-divdend방지위해 +1
                document_embedding_list.append(doc2vec_average)

            # 각 문서에 대한 문서 벡터 리스트를 리턴
            return document_embedding_list

        com_vectors = doc_vectors(df00_pad)

#         result, pivot_ac = visualize_silhouette_layer(com_vectors,31)


#         pd.DataFrame(['벡터크기:{} , 윈도우:{}'.format(EMBEDDING_DIM,WINDOW_SIZE)]).to_csv('댓글_하이퍼파라미터2.csv', encoding='utf-8-sig',mode='a')
#         result.T.to_csv('댓글_하이퍼파라미터2.csv', encoding='utf-8-sig',mode='a',header=False)


```

    100%|█████████████████████████████████████████████████████████████████████████████| 1852/1852 [00:11<00:00, 160.86it/s]
    


```python
visualize_silhouette([2, 3, 4, 5, 6], document_vectors)
```


    
![png](output_135_0.png)
    



```python
visualize_silhouette([2, 3, 4], com_vectors)
```


    
![png](output_136_0.png)
    



```python
# 시각화가 필요없을 때 여기서 바로 시작합니다
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import silhouette_score

def visualize_silhouette_layer(data, num_cluster):
    clusters_range = range(2,int(num_cluster))
    results = []

    for i in clusters_range:
        clusterer = AgglomerativeClustering(n_clusters=i,linkage='ward')
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        results.append([i, silhouette_avg])

    result = pd.DataFrame(results, columns=["n_clusters", "silhouette_score"])
    pivot_ac = pd.pivot_table(result, index="n_clusters", values="silhouette_score")

    return result, pivot_ac
```


```python
result, pivot_ac = visualize_silhouette_layer(com_vectors,30)
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_clusters</th>
      <th>silhouette_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.210688</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.219702</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0.185250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.184967</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.165525</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>0.167276</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0.165009</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>0.161769</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>0.148737</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11</td>
      <td>0.145443</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12</td>
      <td>0.141691</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13</td>
      <td>0.138039</td>
    </tr>
    <tr>
      <th>12</th>
      <td>14</td>
      <td>0.131269</td>
    </tr>
    <tr>
      <th>13</th>
      <td>15</td>
      <td>0.124331</td>
    </tr>
    <tr>
      <th>14</th>
      <td>16</td>
      <td>0.125816</td>
    </tr>
    <tr>
      <th>15</th>
      <td>17</td>
      <td>0.128147</td>
    </tr>
    <tr>
      <th>16</th>
      <td>18</td>
      <td>0.125560</td>
    </tr>
    <tr>
      <th>17</th>
      <td>19</td>
      <td>0.119174</td>
    </tr>
    <tr>
      <th>18</th>
      <td>20</td>
      <td>0.111819</td>
    </tr>
    <tr>
      <th>19</th>
      <td>21</td>
      <td>0.115326</td>
    </tr>
    <tr>
      <th>20</th>
      <td>22</td>
      <td>0.117249</td>
    </tr>
    <tr>
      <th>21</th>
      <td>23</td>
      <td>0.114253</td>
    </tr>
    <tr>
      <th>22</th>
      <td>24</td>
      <td>0.105171</td>
    </tr>
    <tr>
      <th>23</th>
      <td>25</td>
      <td>0.105462</td>
    </tr>
    <tr>
      <th>24</th>
      <td>26</td>
      <td>0.102880</td>
    </tr>
    <tr>
      <th>25</th>
      <td>27</td>
      <td>0.103061</td>
    </tr>
    <tr>
      <th>26</th>
      <td>28</td>
      <td>0.104507</td>
    </tr>
    <tr>
      <th>27</th>
      <td>29</td>
      <td>0.106011</td>
    </tr>
  </tbody>
</table>
</div>




```python
result.to_csv('df00실루엣결과review.csv')
```


```python
plt.plot(result.n_clusters, result.silhouette_score)
```




    [<matplotlib.lines.Line2D at 0x1d2d1d7b518>]




    
![png](output_140_1.png)
    



```python
# 위의 실루엣에 따른 적정 ccluster개수를 선정해 아래 n_clusters를 조정합니다
model = AgglomerativeClustering(n_clusters=2,linkage='ward')
com_cluster_info = model.fit_predict(com_vectors)

vis = pd.DataFrame(pd.Series(com_cluster_info).value_counts().values).reset_index()
vis.columns = ['cluster','num_review']
vis
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster</th>
      <th>num_review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1265</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>587</td>
    </tr>
  </tbody>
</table>
</div>




```python
824+712
```




    1536




```python
df00['com_cluster_info'] = com_cluster_info
```

    C:\Users\User\anaconda3\envs\tf2.0\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    


```python
df00.columns
```




    Index(['review', 'after', 'pos', 'cluster', 'comment', 'date', 'review_len',
           'review_senti', 'review_senti_mean', 'okt_pos', 'com_okt_pos',
           'com_num', 'com_len_mean', 'com_len_std', 'com_senti_dist', 'com_senti',
           'com_senti_mean', 'com_senti_std', 'year', 'month', 'okt2',
           'cluster_review3', 'imp', 'sat', 'opt', 'comment_s', 'com_okt2',
           'com_okt22', 'com_senti2', 'com_cluster_info'],
          dtype='object')




```python
df00['com_senti2'].iloc[1].index(max(df00['com_senti2'].iloc[1]))
df00['com_senti2'].iloc[1].index(min(df00['com_senti2'].iloc[1]))
df00['com_senti2'].iloc[1][0]
```




    -6




```python
df00.columns
```




    Index(['review', 'after', 'pos', 'cluster', 'comment', 'date', 'review_len',
           'review_senti', 'review_senti_mean', 'okt_pos', 'com_okt_pos',
           'com_num', 'com_len_mean', 'com_len_std', 'com_senti_dist', 'com_senti',
           'com_senti_mean', 'com_senti_std', 'year', 'month', 'okt2',
           'cluster_review3', 'imp', 'sat', 'opt', 'comment_s', 'com_okt2',
           'com_okt22', 'com_senti2', 'com_cluster_info'],
          dtype='object')




```python
df00[['review','comment','com_cluster_info','imp']][df00['com_cluster_info']==0].sort_values('imp', ascending=False).iloc[:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>comment</th>
      <th>com_cluster_info</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1494</th>
      <td>집안일 중 제일 하기 싫은 것 갑자기 신랑이 뜬금없이 집안일 중 뭐가 젤 싫냐고 물...</td>
      <td>[['저도 설거지요~~~!'], ['설겆이는 하루에 몇번이고 셀수없이 많이 해야되서...</td>
      <td>0</td>
      <td>10.00000</td>
    </tr>
    <tr>
      <th>304</th>
      <td>집안일 시작~ 미세먼지 최악인데 집안꼴도 최악이에요아직 아침안먹어서 배는고프지만 집...</td>
      <td>[['청소하려면 힘이있어야 하실텐데..!힘내세요!!'], ['넹 힘내서하고잇어요 ㅎ...</td>
      <td>0</td>
      <td>10.00000</td>
    </tr>
    <tr>
      <th>999</th>
      <td>남편분들 집안일 어느정도 하시나요? ㅎ 밥먹고 반찬통 냉장고에 넣어주고 식탁행주로 ...</td>
      <td>[['제가 힘들어서 밀림 해요.\n가사도우미 모드 아주 가끔요.\n주말마다 ㅋㅋ ㅜ...</td>
      <td>0</td>
      <td>9.69697</td>
    </tr>
  </tbody>
</table>
</div>




```python
from collections import Counter
Counter(['이건 아닐꺼야','그치'])
```




    Counter({'이건 아닐꺼야': 1, '그치': 1})




```python
df00[['review','imp']][df00['review'].str.contains('알면 좀 더 쉬운')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df00[df00['com_cluster_info']==0][['imp','sat','opt']].mean()
```




    imp    6.012142
    sat    4.546704
    opt    7.646481
    dtype: float64




```python
# 각 댓글군집별로 대표리뷰 3개씩 뽑고 감성 점수 상위,하위 각각 3개씩 추출
for i in df00.com_cluster_info.value_counts().index: # 0 1 2번 군집
    j = df00[df00['com_cluster_info']==i][['review','comment','com_cluster_info','imp','com_senti2']].sample(2)  #.iloc[:3]
#     .sort_values('imp', ascending=False).iloc[:30]
    print('### '+ str(i) + '번 댓글군집')
    for n in range(len(j)):
        print('기회점수')
        print(j.imp.iloc[n])
        print()
        print(j.review.iloc[n])
        print()
        print(j.comment.iloc[n])
        print()
        print(j.com_senti2.iloc[n])
        print()
        
```

    ### 0번 댓글군집
    기회점수
    5.602344506042856
    
    집안일1도 해본적 없다가 결혼하신 분들??? 어찌 집안일 익숙하신가요?????전 미스때 설거지도 안해본 1인이예요요리 설거지 청소 다안했어요그냥 엄빠밑에서 곱게큰거 같네요 ㅠㅠㅠ제가 거의10년 외동딸이었거든요자라면서도 1도 안시키셨는데안해본거하려니 아직도 너무 힘들어요주부7년차....하루종일 너~무 바빠요청소 빨래개기 세탁기돌리고 설거지하고 밥하고 중간중간 중고물품팔고 정리하고 화장실베란다 정리하구.... 애랑도 놀아줘야하구(유치원안다님)진짜 살림1도 안해봐서 손도 느리고요 너무바빠서 밤되면 지쳐 쓰러져용 ㅠㅠㅠㅠㅠ가끔은 살림 안가르쳐준 엄마가 원망스럽네요ㅋㅋ곱게곱게-_- 자라신분들~~어떠신가요?
    
    [['저는 결혼 3년차인데 집안꼴이 개판이에요ㅠ 그나마 신랑이 많이 하고 애기가 아직없어서~~ 이제 애기 태어나면 어쩌나 걱정이에요ㅠ'], ['저요ㅋㅋㅋㅋㅋ 신랑이 처음에는 잔소리하다가 포기했어요~ 차라리 나가서 일하는 게 좋네요ㅠㅠ 손도 느리고..'], ['곱게?는 아니지만 그냥 시키지도 않으셨고 가끔 시키시면 제가 동생을 시켜서ㅋㅋㅋ 저도 집안일 하느라 하루가 너무 바쁜데 게다가 요똥이라 요리가 제일 싫어요..ㅋㅋ 치우는건 성격이 깔끔한 편이라 그냥저냥 치우고 잘 살아요ㅋㅋ'], ['ㅋㅋㅋㅋ\n저요. 맏이인데도 온실의 화초? 지금은 잡초..ㅋㅋㅋ 애들 있으니 하게 됩니다. ^^;;;;;'], ['저도 어머니가 집안일 절대 안 시키셔서 그렇게 살다가 결혼했는데 집안일 못하겠어요...ㅋㅋㅋㅋ 음식 하지도 못하는데 메뉴 고민하는 것도 머리 아프고 청소도 전문 업체 어머니 모셔서 해요ㅠㅠㅋ 정년까지 일하기로 마음 먹었습니당ㅋㅋㅋㅋㅋ'], ['살림을 배우는 거라고 생각 해 본적이 없어요. 자기 스탈대로 터득하고 인터넷도 보고 하면서 스스로 쌓아가는거 아닐까요? 살림 하나도 않하고 직장생활 하면서 결혼사하고 계속 맞벌이. 남편과 집안일 나눠하는데 요리는 꽤 하고 정리는 잘 못하지만 모. . . 안가르쳐준 엄마를 원망해본적은 없네요. ^^;;;;'], ['어느정도 하다가 결혼해도 능숙하지 않아요 ㅎㅎ 엄마가 주"로 하시고 부"수적으로 제가 하는거랑 제가 주"가 되서 이끌어가는게... 영 능숙해지지는 않네요.'], ['저랑 똑같네요 저 그래서 집안일 신랑이 대부분해요..다행스럽게 전공이 식영과라 대략적으로 음식은 할줄알아서 음식은 제가 하네요ㅠㅠ'], ['결혼한지 일년이 넘었는데 아직도 살림하는 솜씨가 엉망진창이예요 ... ㅠㅠ'], ['하다보면 늘어용~ 음식은 인터넷검색. 빨래 청소는 하다보믄 늘구여.. 전 아직 수납정리가 어렵더라구요'], ['설거지 청소 등 세세한 부분은 남편 하는거 보고 배웠어요. 결혼한지 5년 만에 반찬 조금씩 만들기 시작했구요.'], ['오빠가요리하기를좋아해서 어렸을때부터 남매끼리뭘해도 전 늘 설거지담당이였어요\n엄마는 아프셔서 레시피물어봐도알려주시지못하고, 네이버검색하며 음식하는데 시어머니는 제가손도빠르고손맛도좋대요😅\n제가봤을때 전똥손막손인데말이죠ㅋㅋ'], ['저도 결혼전에 집안일 1도 안하고 시집왔거든요ㅎㅎ설거지 결혼전에 해본적이 거의 없고 진짜 못된 딸래미였는데ㅎㅎ회사갈때 치장만하고 심지어 방도 엄마가 치워주시고ㅎㅎ결혼하고 너무 힘들더라구요 이제 4년차되니 요리는 아직도 못하고 아이 반찬만 겨우 만드는정도고 나름 청소나 주변정리는 깨끗하게 하고있어요ㅎㅎ친정엄마가 엄청 깔끔 정리정돈 잘하셔서 그냥 배우지는 않았지만 깨끗한거 보면서 커서그런지 어지럽히보 지저분한걸 잘 못보니 그냥 치우게되네용'], ['저는 엄마가 이것저것 시켰는데도 그닥 별 도움은 안되더라고요.  직접 혼자서 실전에 부딪히고 해봐야 느는게 살림인것같아요..'], ['저도 그래서 결혼후에 나는 애한테 다 시켜야지!! 결심했었네욬ㅋㅋ진짜 첨엔 청소도 너무 오래걸렸고요. 애둘 낳고 키우다보니 좀 빨라지긴 하드라고요.  그래도 여전히 요리는 못해서 애들 이유식때부터 다 사서 먹였어요. 요리못하는 사람 특징ㅡ 재료다듬는척 다 버리고, 사방팔방 그릇 다 꺼내고, 결과물은 맛도없고, 설거지는 넘쳐나는ㅡ 다 가지고 있어서 지금도 다 사다먹네욤ㅜㅜ'], ['저예요 ㅠㅠ 온실화초는 아니었지만 밖에서 끼니 해결하며 지내고 청소는 딱 필요한만큼만 하며 살았는데..아기 출산후가 너무 걱정되어서 지금부터 요리며 청소며 연습하는데 쉽지 않네요. 항상 제 눈에 더럽고 만족스럽지 않아서 자존감이 떨어져요. 돈이라도 많이 버는 직장에 갔어야 했나봐요ㅠㅠ'], ['저요 ㅋㅋㅋ 근데 지저분하고 사먹는 음식만 먹는거 삶의 질이 넘 낮아지는거 같아 내가 사는 곳 쾌적하게 청소하고 집 꾸미고 건강한 요리 해먹어요~ 수준 높아지는거 같아 뿌듯 뿌듯']]
    
    ['-1.5', ' -0.25', ' -0.6666666666666666', ' 0', ' -2.0', ' -0.6666666666666666', ' 0', ' 0', ' 0', ' -2.0', ' 1.0', ' 0.75', ' -0.625', ' 1.0', ' -1.3333333333333333', ' -1.8', ' 0.6666666666666666']
    
    기회점수
    5.137472511357584
    
    끝이 없는 집안일 밥먹기-설거지-청소기돌리기-물걸레청소기돌리기-재활용품분리수거-빨래-화장실청소​주말에 집안일을 끝이 없네요
    
    [['ㅋㅋㅋㅋㅋ 저도 다들 집에 잇음 청소기 5번은 족히 돌려요'], ['집안일은 끝이없고 티도잘안나여ㅠ'], ['맞아요ㅜㅜ 내일 또 청소 해야하네요'], ['그쵸ㅜㅠ 무한반복ㅜㅠ 저는 오늘 삘 받아서 베란다 청소까지 ㅋㅋ 너무 힘들어서  급후회했다는ㅜㅜㅎㅎ'], ['맞아요 ㅠ 평일에도 다 치워놨는데 애들 하원하면 도루묵.. 무한반복이죠 ㅠ'], ['안하면 티는 금방나요 ㅠㅠ'], ['먼지하고 같이 사네요...ㅠㅠ'], ['하 진짜 공감이요ㅜㅜ치우고나면 티가 안나는데 안치우면 확티나고..'], ['먼지 휴...진짜 도루묵'], ['진짜 공감요...... ㅜㅜ 가끔은 내가 청소부인가 싶어요'], ['진짜 바쁘게 하는데도 티가 안나는 ㅜㅜ'], ['주말은 쉬는날이 아닙니다 ㅠ'], ['주말은 쉬는 날이 아니라 집안일 몰아서 하는날인거 같아요ㅠ'], ['ㅠㅠㅠ퇴근해도 육아출근이죠ㅠㅠㅠ']]
    
    ['0', ' -1.0', ' 0', ' -1.0', ' 0', ' 0', ' -1.0', ' 0', ' -1.0', ' 0', ' 0', ' -2.0', ' -2.0', ' 0']
    
    ### 1번 댓글군집
    기회점수
    7.223948510426579
    
    점심? 저녁?? 점저??? 토욜인데...아들은 독서실,남편님은 결혼식...딸램과 둘이서,중국집으로 전화를...ㅋ미니 탕슉!!!요즘은 탕슉이 없음 허전해서 자꾸...ㅋ딸램은 간짜장...소스 확 부었더니 비쥬얼이...ㅠㅠ전 짬뽕...먹다가 포기했지만...그래도 맛나네요~간만에 부지런히,집 청소, 화장실 청소, 빨래...집안일로 붙타올랐습니다~집안일 끝나면 환수,환수 끝나면 활착놀이할 계획인데...진짜 요즘은 자는 시간 빼고는 뭔가 할 일이 있는 듯한바쁜 날들 보내고 있구만요.
    
    [['저도.. 활착거리가 좀 있는데.. 어찌 할가 무쟈게 고민중이에요 ㅋ'], ['ㅋㅋㅋㅋㅋ...\n\n네 숙제를 빠르게 해치워야~ ㅋㅋㅋㅋㅋ\n\n기달려봐~ \n\n꼭 숙제완료해서 올릴 터이니~!!!!'], ['누나..\n이번거 부세들 아주...\n고오급 인거 아시지용~~'], ['아~~주 고급이니 잘 활착해 볼게~ ㅋㅋㅋ'], ['맛나보이네요...한~~~~입~~~만!!!^^(유세윤 버전입니다)'], ['ㅋㅋㅋㅋㅋ...\n\n지금 바로 배달앱으로 주문을...ㅋㅋㅋㅋ'], ['아 고 간짜장 맛나 보이네요 ㅎㅎ\n탕슉이 예전엔 어쩌다가 였는데\n이젠 필수가 ㅋㅋㄱㅋ'], ['그니까~~\n\n일년에 한번 먹을까 말까 하던 탕슉이 이제 꼭 시켜야 하는 걸로...ㅋㅋㅋㅋ'], ['앗! 짬뽕 주문해야쥐~~  ㅎㅎ\n\n수엄쉬엄 하세요~'], ['ㅋㅋㅋㅋㅋ...\n\n짬뽕 맛나게 드시와요^^'], ['저하고 점심 메뉴가 똑같아요...ㅎ..저도 딸아이가 먹고 싶다고해서...'], ['오~~~ 찌찌뿡!!! ㅋㅋㅋㅋ\n\n그 동네 중국집 맛나나요~? ㅋㅋㅋ'], ['저희가 오늘 다녀온집은 중국분이 하시는데 싸고 맛있어요^^...배달은 안해요(와서 먹어)...ㅋ'], ['저희는 꼬맹이가 짬뽕...ㅋ'], ['ㅋㅋㅋㅋㅋ... \n먹을 줄 아누만요...ㅋㅋ'], ['앗 언니 저는 매콤한 떡뽂이 먹다가\n포기 ㅎㅎ\n속이 살살 아파서\n죽먹어야겠어요~>_<\n맛점저~~!!!♡-♡하세유~!!!'], ['넌 매운 것 못먹음서 왜 도전했다니...ㅋㅋㅋㅋ\n\n죽 꼭 챙겨먹고 속을 달래~~'], ['맛나게 먹어\n부럽가'], ['오라버니는 정모에서 맛나게 먹어요~ ㅋ'], ['정모 끝나고  3차다 ㅜㅜ'], ['오늘은 좀 쉬어요~^^'], ['탕슉탕슉~~~ 저는 탕슉이 먹고싶었어요~~^^'], ['탕슉탕슉!!!\n\n오늘 저녁은 탕슉??? ㅋㅋㅋㅋ'], ['그래야될까봐요~~~~ 꼴깍 꼴깍~~~'], ['뭐든 맛난 거 먹어라~~^^'], ['나이들면 딸냄이 최고죠'], ['정말로 그렇게 될지~~ ㅋㅋㅋ'], ['오늘은 소박하시네요~'], ['음... 간만인데 기대 충족이 안됐죠??? ㅋㅋㅋㅋ'], ['쉬는날도 참 바쁘네요 ㅋ'], ['한동안 게으름을 많이 피우다보니...ㅋㅋ'], ['짬뽕 색깔이 아주 좋네요!!'], ['탕슉이랑 같이 먹다 보니...\n\n먹다 남겼...ㅋㅋㅋㅋ']]
    
    ['0', ' 1.0', ' 0', ' 2.0', ' 0', ' 0', ' 1.0', ' 0', ' 0', ' 0', ' 0', ' 0', ' 2.0', ' 0', ' 0', ' -1.6666666666666667', ' 0.0', ' -1.0', ' 0', ' 0', ' 0', ' 0', ' 0', ' 0', ' 0', ' 2.0', ' 0', ' 0', ' 1.0', ' 0', ' -1.0', ' 2.0', ' 0']
    
    기회점수
    5.302028757930042
    
    남편에게 집안일 시키면 안되는 이유 "여보~배추 절이게 반으로 좀 썰어놔~"""여보 거기 까만 봉지에 있는 감자 반만 깎아서 냄비에 좀 삶아줘"남편:만두 먹고 싶은데아내:아..고구마 찌고 있는데 그 위에 올려놔남푠님들 혹시 뭐가 잘못된 건지 모르신다면 허걱입니다~남자란  자세한 설명이 필요한 거겠죠?
    
    [['ㅋㅋㅋ 미챠네요~ 하나하나 자세히 설명해줘야는 남자들 애나 별반 다를게 없죠;;ㅠ'], ['귀엽네요 ㅎ'], ['어떤 남편은 전기밥솥에 밥좀 하라했더니 밥솥을 가스렌지 위에다 올리고 가스불을 켰데요~ㅋ'], ['ㅋㅋㅋ 혹시 이거 다 남편의 유머 아닐까요? 재밌으라고..  ㅎㅎ'], ['저희 신랑은 덜기 귀찮다고 냄비채 전자렌지 돌리더라구요 ㅡㅡ 그거보고 쓰러지는줄알았네요 ㅋㅋㅋㅋ'], ['아 진짜 너무 웃긴데...\n격하게 공감되네요^^'], ['아하 상세 설명'], ['또 게다가 상세하게 설명해주면  듣기 싫어해요;;;'], ['재밋네요'], ['정말 잼나네요.\n한참 웃었어요.\n생각이 기발하시네요.'], [''], ['아 진짜 그래요?'], ['다신 시키지 말라고 저러는 거라는데요? ㅋ'], ['다시는 못시키게 하려는 고도의 전략일수도 ㅋㅋ'], ['일부러 그런듯 ㅋㅋㅋ']]
    
    ['-1.0', ' 2.0', ' 0', ' -2.0', ' -1.5', ' 0', ' 0', ' -2.0', ' 0', ' 2.0', ' 0', ' 0', ' 0', ' 0', ' -1.0']
    
    


```python
print(com_okt2[3])
print(com_okt22[3])
print(com_senti2[3])
```


```python
df00['date'].value_counts().sort_index()
```




    1804    20
    1805    30
    1806    29
    1807    34
    1808    20
    1809    31
    1810    41
    1811    31
    1812    33
    1901    27
    1902    21
    1903    26
    1904    26
    1905    32
    1906    24
    1907    24
    1908    32
    1909    34
    1910    38
    1911    25
    1912    28
    2001    25
    2002    22
    2003    22
    2004    49
    2005    45
    2006    29
    2007    37
    2008    30
    2009    36
    2010    36
    2011    33
    2012    35
    2101    50
    2102    49
    2103    35
    Name: date, dtype: int64




```python

```


```python

```
