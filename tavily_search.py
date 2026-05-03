import os
from tavily import TavilyClient
from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv(override=True)

#===============================================
# Tavily Client
#===============================================
tavlily_client = TavilyClient(
    api_base_url=os.getenv("TAVILY_BASE_URL"),
    api_key=os.getenv("TAVILY_API_KEY")
)

#===============================================
# AzureOpenAI Client
#===============================================
aoai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

#===============================================
# Tavily Seach Function
#===============================================
def tavily_search_company_info(company_name: str) -> dict:
    query = f"{company_name} 企業概要 事業内容"

    response = tavlily_client.search(
        query=query,
        search_depth="advanced",
        topic="general",
        include_answer=True,
        include_raw_content=False,
        max_results=3,
    )

    return response

#===============================================
# Tavily API検索結果をコンテキスト文字列に整形
#===============================================
def build_context(search_response: dict) -> str:
    context_parts = []

    if search_response.get("answer"):
        context_parts.append(f"【Tavily 簡易回答\n{search_response['answer']}\n")
    
    # 各検索結果のスニペットを追加
    results = search_response.get("results", [])
    print(f"✅️ {len(results)}件の検索結果を取得")
    
    for i, result in enumerate(results, start=1):
        part = (
            f"【情報源 {i}】{result.get('title', '不明')}\n"
            f"【URL】{result.get('url', '')}\n"
            f"【内容】{result.get('content', '')}\n"
            f"【関連度スコア】{result.get('score', 0)}"
        )
        context_parts.append(part)

    return "\n".join(context_parts)

#===============================================
# AOAIに要約させる
#===============================================
def summarize_with_aoai(company_name: str, context: str) -> str:
    system_prompt = """あなたは企業調査のエキスパートアナリストです。
"""

    user_prompt = f"""以下の情報を基に、「{company_name} について要約してください。
### 情報 ###
{context}

### 出力形式 ###
1. 企業名
2. 企業概要（設立・本社・代表など）
3. 業績や財務ハイライト（分かる範囲で）
4. 最近の動向
5. 総合評価コメント

### 制約条件 ###
 - 提供されていない情報にない内容は「情報なし」と記載し、推測で補完しないこと
"""
    
    response = aoai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        reasoning_effort="medium",
        max_completion_tokens=1500,
    )

    answer = response.choices[0].message.content.strip()
    usage = response.usage
    print(f"📈 【トークン使用量】入力：{usage.prompt_tokens} / 出力：{usage.completion_tokens} = 全体：{usage.total_tokens}")
    return answer

#===============================================
# メイン処理
#===============================================
def research_company(company_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"企業リサーチ開始：{company_name}")
    print(f"{'='*60}")

    # Step 1: Tavilyで検索
    search_result = tavily_search_company_info(company_name)

    # Step 2: コンテキストを整形
    context = build_context(search_result)

    # Step 3: GPT-5.4で要約
    summary = summarize_with_aoai(company_name, context)

    # Step 4: 結果出力
    print(f"\n{'='*60}")
    print(f"【企業情報要約】{company_name}")
    print(f"{'='*60}")
    print(summary)

#===============================================
# エントリポイント
#===============================================
if __name__ == "__main__":
    company_name = "トヨタ自動車"
    research_company(company_name)
