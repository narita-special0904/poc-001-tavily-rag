"""LangGraph を使った Tavily RAG。

フロー:
  classify  : ユーザ質問に Web 検索が必要かを LLM に判定させる
  search    : 必要な場合のみ Tavily で検索し、コンテキストを作る
  generate  : (あれば) コンテキストを参考に最終回答を生成

Tavily 呼び出しは tavily_search.py / tavily_general_search.py を再利用する。
"""
from __future__ import annotations

import json
import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from tavily_search import aoai_client
from tavily_general_search import search_and_build_context

load_dotenv(override=True)

AOAI_MODEL = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT")


#===============================================
# State
#===============================================
class RagState(TypedDict, total=False):
    question: str
    need_search: bool
    search_query: str
    context: str
    answer: str


#===============================================
# Node: 検索要否の判定
#===============================================
def classify_node(state: RagState) -> RagState:
    question = state["question"]

    system_prompt = (
        "あなたはユーザの質問を分析するアシスタントです。"
        "質問に正確に答えるために最新情報や外部知識(Web検索)が必要かを判定してください。"
        "以下の JSON 形式のみで回答してください: "
        '{"need_search": true/false, "search_query": "検索に使うクエリ(必要な場合のみ)"}'
    )

    response = aoai_client.chat.completions.create(
        model=AOAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=300,
    )

    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        need_search = bool(parsed.get("need_search", False))
        search_query = parsed.get("search_query") or question
    except json.JSONDecodeError:
        need_search = True
        search_query = question

    print(f"🔎 検索要否: {need_search} / クエリ: {search_query}")
    return {"need_search": need_search, "search_query": search_query}


#===============================================
# Node: Tavily 検索
#===============================================
def search_node(state: RagState) -> RagState:
    query = state.get("search_query") or state["question"]
    context = search_and_build_context(query, max_results=5)
    return {"context": context}


#===============================================
# Node: 回答生成
#===============================================
def generate_node(state: RagState) -> RagState:
    question = state["question"]
    context = state.get("context", "")

    if context:
        system_prompt = (
            "あなたは信頼できるアシスタントです。"
            "提供された検索結果を根拠に、簡潔かつ正確に回答してください。"
            "検索結果に無い内容は推測せず『情報なし』と明記してください。"
            "可能であれば参照した情報源のURLを末尾に列挙してください。"
        )
        user_prompt = (
            f"### 質問\n{question}\n\n"
            f"### 検索結果\n{context}\n\n"
            "### 回答"
        )
    else:
        system_prompt = "あなたは有能なアシスタントです。簡潔かつ正確に回答してください。"
        user_prompt = question

    response = aoai_client.chat.completions.create(
        model=AOAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort="medium",
        max_completion_tokens=1500,
    )

    answer = response.choices[0].message.content.strip()
    usage = response.usage
    print(
        f"📈 トークン使用量 入力:{usage.prompt_tokens} / 出力:{usage.completion_tokens} "
        f"/ 合計:{usage.total_tokens}"
    )
    return {"answer": answer}


#===============================================
# 分岐
#===============================================
def route_after_classify(state: RagState) -> Literal["search", "generate"]:
    return "search" if state.get("need_search") else "generate"


#===============================================
# Graph 構築
#===============================================
def build_graph():
    graph = StateGraph(RagState)
    graph.add_node("classify", classify_node)
    graph.add_node("search", search_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"search": "search", "generate": "generate"},
    )
    graph.add_edge("search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


#===============================================
# エントリポイント
#===============================================
def ask(question: str) -> str:
    app = build_graph()
    final_state = app.invoke({"question": question})
    return final_state["answer"]


if __name__ == "__main__":
    question = input("質問をどうぞ：")

    print(f"\n{'=' * 60}\n質問: {question}\n{'=' * 60}")
    answer = ask(question)
    print(f"\n{'=' * 60}\n回答\n{'=' * 60}\n{answer}")
