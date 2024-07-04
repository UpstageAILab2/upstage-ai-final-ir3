import os
import json
import torch
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
# SBERT 기반 모델보다는 RoBERTa 기반 모델이 성능이 더 좋았음
'''
    사용해본 모델
        SBERT 기반
            - snunlp/KR-SBERT-V40K-klueNLI-augSTS
            - sentence-transformers/sentence-t5-large
            - jhgan/ko-sbert-multitask
            - jhgan/ko-sbert-nli
        
        RoBERTa 기반
            - jhgan/ko-sroberta-multitask
            - jhgan/ko-sroberta-base-nli
            
'''
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# Reranker 모델 및 토크나이저 로드 (bge reranker model)
reranker_model_name = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)


# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    return model.encode(sentences)

# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings

# Elasticsearch 인덱스 생성 함수
def create_es_index(index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)

# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)

# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)


'''
# 전통적인 역색인 기법을 사용하여 단어의 빈도와 위치를 기반으로 검색
# 계산 자원이 적게 들고 빠른 검색이 가능하지만, 의미적 유사성을 잘 포착하지 못할 수 있음
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")


# 딥러닝 기반의 임베딩 벡터를 사용하여 의미적 유사성을 포착한 검색을 수행
# 계산 자원이 많이 들지만, 높은 의미적 관련성을 제공
def dense_retrieve(query_str, size):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]

    # kNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", knn=knn)
'''

# Hybrid retrieve (역색인 + 벡터 유사도 혼합)
def hybrid_retrieve(query_str, size):
    query_embedding = get_embedding([query_str])[0]
    
    # Using elasticsearch query DSL
    # 기존 match와 knn이 별도로 나열되어 있던 것을 should를 통해 결합 (테스트 X)
    body = {
        "query": {
            "bool": {
                "should": [{
                    "match": {
                        "content": {
                            "query": query_str,
                            "boost": 0.002
                            }
                        }
                    },
                    
                    {
                    "knn": {
                        "field": "embeddings",
                        "query_vector": query_embedding.tolist(),
                        "k": 5,
                        "num_candidates": 50,
                        "boost": 1
                        }
                    }]
                }
            },
        "size": size
    }

    return es.search(index="test", body=body)

# Reranker
def rerank(query, search_results):
    reranked_results = []

    for result in search_results['hits']['hits']:
        doc_text = result['_source']['content']
        inputs = tokenizer.encode_plus(query, doc_text, return_tensors='pt', truncation=True)
        with torch.no_grad():
            scores = reranker_model(**inputs).logits
        relevance_score = scores.item()
        reranked_results.append((relevance_score, result))

    # 점수에 따라 결과를 정렬
    reranked_results = sorted(reranked_results, key=lambda x: x[0], reverse=True)

    # 정렬된 결과를 반환
    return [result for score, result in reranked_results]


es_username = "elastic"
es_password = "pwd"

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="/usr/elasticsearch-8.8.0/config/certs/http_ca.crt")

# Elasticsearch client 정보 확인
print(es.info())


'''    
    BM25
        전통적인 역색인 기법과 빈도 기반 접근 방식을 결합한 모델
        문서의 길이와 단어 빈도를 고려

    LM Jelinek-Mercer
        쿼리와 문서 간의 확률적 유사성을 계산
        Jelinek-Mercer 스무딩을 사용하여, 쿼리 용어가 문서에 나타나지 않는 경우를 처리
'''

# 색인을 위한 setting 설정
settings = {            
    # BM25 (submission 22) -> LM Jelinek-Mercer (submission 30)
    # Public score는 LM Jelinek-Mercer가 더 낮았으나, Private score는 더 높았음
    "index": {
        "similarity": {
            "lm_jelinek_mercer": { 
                "type": "LMJelinekMercer", 
                "lambda": 0.7 
            } 
        }
    },
    
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            },
        }
    }
}

# 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
mappings = {
    "properties": {
        "content": {
            "type": "text", 
            "analyzer": "nori",
            # BM25 -> LM Jelinek-Mercer
            "similarity": "lm_jelinek_mercer"
        },
        
        "embeddings": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            # 문서 임베딩의 경우 벡터의 크기보다 방향이 더 중요한 경우가 많음
            # hybrid retrieve만 사용했을 경우 l2 norm이 cosine에 비해 점수가 소폭 높음
            # reranker 테스트에서는 cosine이 l2 norm보다 좀 더 좋은 결과를 뽑아냈음
            # 시간이 없어 오류를 고치지 못했고 최종 결과를 보지 못해서 아쉬움
            "similarity": "cosine"
        }
    }
}

# settings, mappings 설정된 내용으로 'test' 인덱스 생성
create_es_index("test", settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성
index_docs = []
with open("/root/Project/data/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)
                
# 생성한 임베딩을 색인할 필드로 추가
for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

# 'test' 인덱스에 대량 문서 추가
ret = bulk_add("test", index_docs)

# 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
print(ret)

test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

# 역색인 + Vector 유사도 혼합을 사용하는 검색 예제
search_result_retrieve = hybrid_retrieve(test_query, 15)

# Reranker를 통해 검색 결과 재정렬
reranked_results = rerank(test_query, search_result_retrieve)
# 최종적으로 선택할 문서 수
final_doc_count = 3  

# 결과 출력 테스트
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])



# 아래부터는 실제 RAG를 구현하는 코드입니다.
from openai import OpenAI
import traceback

# OpenAI API 키를 환경변수에 설정
os.environ["OPENAI_API_KEY"] = "your api keys"

client = OpenAI()
# 사용할 모델을 설정(여기서는 gpt-3.5-turbo-1106 모델 사용)
llm_model = "gpt-3.5-turbo-1106"

# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
'''
    질의 분석 변경점
        - 사용자가 지식에 관한 질문을 하는 경우에는 반드시 search 함수를 호출한다.
            1. 과학 지식으로 좁힐 경우, 과학 지식임에도 불구하고 처리가 안되는 경우 존재
               차라리 한정짓지 않고 다양한 주제의 지식 관련 질문을 처리할 수 있도록 변경
            2. 지식 관련 질문임에도 불구하고 함수가 호출되지 않는 경우 존재
               '반드시' 라는 단어를 포함시켜 호출되지 않는 경우를 처리
            
        - 나머지 메시지에는 함수 호출을 하지 않고 적절한 대답을 생성한다.
            1. 위의 경우를 제외한 나머지 메세지에 대한 처리
               함수를 호출하지 않겠다라는 내용을 포함시켜 함수가 호출되는 경우를 처리 
'''

persona_function_calling = """
## Role: 과학 상식 전문가

## Instruction
- 사용자가 지식에 관한 질문을 하는 경우에는 반드시 search 함수를 호출한다.
- 나머지 메시지에는 함수 호출을 하지 않고 적절한 대답을 생성한다.
"""

# Function calling에 사용할 함수 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "properties": {
                    # sparse vector 검색 시 영어 단어가 포함시 성능 저하
                    "standalone_query": {
                        "type": "string",            
                        # description
                        # ‘in Korean’ 추가해도 영어 단어가 한국어로 변환되지 않음.         
                        # ‘Full message if the message is single-turn.’ 추가 시 더 많은 키워드가 추출되지만 원하는 대로 질문 그대로 추출되지는 않음.
                        "description": "User's question in Korean. Full message if the message is single-turn."
                    }
                },
                "required": ["standalone_query"],
                "type": "object"
            }
        }
    },
]


# LLM과 검색엔진을 활용한 RAG 구현
def answer_question(messages):
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            functions=tools,
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception as e:
        traceback.print_exc()
        return response

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
    if result.choices[0].message.function_call:
        function_args = json.loads(result.choices[0].message.function_call["arguments"])
        standalone_query = function_args.get("standalone_query")

        # 검색 결과 추출 (초기 검색에서 많은 문서 선택)
        search_result = hybrid_retrieve(standalone_query, 15)

        # Reranking the search results
        reranked_results = rerank(standalone_query, search_result)

        response["standalone_query"] = standalone_query
        retrieved_context = []
        
        # 최종적으로 선택할 문서 수  
        final_doc_count = 5        
        
        # 상위 5개의 문서만 선택
        for i, rst in enumerate(reranked_results[:final_doc_count]):  
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

        content = json.dumps(retrieved_context)
        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                model=llm_model,
                messages=msg,
                temperature=0,
                seed=1,
                timeout=30
            )
        except Exception as e:
            traceback.print_exc()
            return response
        response["answer"] = qaresult.choices[0].message["content"]

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message["content"]

    return response

# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
eval_rag("/root/Project/data/eval.jsonl", "/root/Project/result/sample_submission.csv")

