"""汎用的なWeb検索ラッパー。

tavily_search.py は企業調査向けのクエリに特化しているため、
任意のクエリで検索する用途にはこちらを使用する。
Tavily クライアントと結果整形ロジック(build_context)は tavily_search.py から再利用する。
"""
from tavily_search import tavlily_client, build_context


def tavily_search_general(query: str, max_results: int = 5) -> dict:
    """任意のクエリで Tavily 検索を実行する。"""
    return tavlily_client.search(
        query=query,
        search_depth="advanced",
        topic="general",
        include_answer=True,
        include_raw_content=False,
        max_results=max_results,
    )


def search_and_build_context(query: str, max_results: int = 5) -> str:
    """検索を実行し、LLM に渡すコンテキスト文字列まで整形して返す。"""
    response = tavily_search_general(query, max_results=max_results)
    return build_context(response)
