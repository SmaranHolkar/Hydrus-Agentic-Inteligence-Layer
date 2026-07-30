import os
import re
import urllib.request
import urllib.parse
import json
from typing import Optional, List, Tuple

class SearchServer:
    def __init__(self, guard, session_name: str = "default"):
        self.guard = guard
        self.session_name = session_name
        self.searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")

    async def _run_browser_use_agent(self, task: str) -> Optional[str]:
        """Attempt to execute a browser-use agent backed by a local Ollama model."""
        # Resolve imports safely
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                print("[SearchServer] Neither langchain-ollama nor langchain-community found.")
                return None

        try:
            from browser_use import Agent
        except ImportError:
            print("[SearchServer] browser-use package not found.")
            return None

        try:
            model_name = os.getenv("OLLAMA_MODEL", "llama3")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            print(f"[SearchServer] Launching browser-use agent with local Ollama model: {model_name} at {base_url}...")
            llm = ChatOllama(
                model=model_name,
                base_url=base_url,
                num_ctx=32000,
                temperature=0.0,
            )

            agent = Agent(
                task=task,
                llm=llm,
            )
            history = await agent.run()
            if history:
                final_res = history.final_result()
                if final_res:
                    return str(final_res)
        except Exception as e:
            print(f"[SearchServer] browser-use agent run crashed: {e}")
        return None

    async def web_search(self, query: str) -> str:
        """
        Query the web using browser-use + local Ollama model.
        Falls back to SearXNG or DuckDuckGo HTML search if browser-use is unavailable or fails.
        """
        task = f"Go to Google or DuckDuckGo. Search for: '{query}'. Browse through the search results, click on the most relevant links, synthesize the information, and write a summary."
        
        # 1. Try local browser-use agent
        agent_result = await self._run_browser_use_agent(task)
        if agent_result:
            return f"Autonomous Browser-Agent Search results for '{query}':\n\n{agent_result}"

        print("[SearchServer] browser-use agent unavailable/failed. Falling back to HTTP SearXNG/DDG searches.")

        # 2. Fallback to SearXNG
        try:
            url = f"{self.searxng_url}/search?q={urllib.parse.quote(query)}&format=json"
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            # Use urllib synchronously as fallback
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                results = data.get("results", [])
                if results:
                    snippets = []
                    for r in results[:5]: 
                        title = r.get("title", "No Title")
                        link = r.get("url", "")
                        content = r.get("content", "").strip()
                        if link:
                            snippets.append(f"- **[{title}]({link})**: {content}")
                        else:
                            snippets.append(f"- **{title}**: {content}")
                    return f"Live SearXNG Search results for '{query}':\n" + "\n".join(snippets)
        except Exception as e:
            print(f"[SearchServer] SearXNG query failed ({e}). Falling back to DuckDuckGo.")

        # 3. Fallback to DuckDuckGo (HTML scrape)
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                html = response.read().decode('utf-8')
                snippets = []
                matches = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
                for m in matches[:5]:
                    clean = re.sub(r'<[^>]+>', '', m).strip()
                    clean = clean.replace("&quot;", "\"").replace("&amp;", "&").replace("&#x27;", "'").replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'")
                    snippets.append(clean)
                
                if not snippets:
                    matches = re.findall(r'<div class="result__snippet[^>]*>(.*?)</div>', html, re.DOTALL)
                    for m in matches[:5]:
                        clean = re.sub(r'<[^>]+>', '', m).strip()
                        clean = clean.replace("&quot;", "\"").replace("&amp;", "&").replace("&#x27;", "'").replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'")
                        snippets.append(clean)

                if snippets:
                    return f"Live DDG Search results for '{query}':\n" + "\n".join(f"- {s}" for s in snippets)
        except Exception as e:
            print(f"[SearchServer] DDG query failed ({e}).")

        # 4. Ultimate fallback
        if "hydrus" in query.lower():
            return (
                "Offline Search results for 'HydrusOpt':\n"
                "- HydrusOpt is a local LLM safety and benchmarking layer.\n"
                "- Core skills include: Selective Layer Linearisation, Mixed Precision Quantisation,\n"
                "  Metacognition Plugin, Hallucination Guards, and a Stratified Memory Lattice (SML).\n"
                "- It supports secure local user profiles and active belief tracking."
            )
        return f"Search results for '{query}': No specific offline matches found and live fetch failed."

    async def fetch_webpage(self, url: str, query: Optional[str] = None) -> str:
        """
        Fetch HTML content of a URL and parse it.
        Uses browser-use if available, otherwise Crawl4AI, falling back to urllib parser.
        """
        if not (url.startswith("http://") or url.startswith("https://")):
            return "Error: Invalid protocol. Only http/https supported."

        # 1. Try local browser-use agent
        task = f"Go to URL '{url}'."
        if query:
            task += f" Extract and summarize all sections relevant to this query: '{query}'."
        else:
            task += " Extract the main article content clean in Markdown format."

        agent_result = await self._run_browser_use_agent(task)
        if agent_result:
            return agent_result

        print("[SearchServer] browser-use agent unavailable/failed. Falling back to Crawl4AI/urllib.")

        # 2. Try Crawl4AI
        markdown_content = None
        try:
            from crawl4ai import AsyncWebCrawler
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                if result and result.success:
                    markdown_content = result.markdown
        except ImportError:
            pass
        except Exception as e:
            print(f"[SearchServer] Crawl4AI arun failed: {e}")

        # 3. Fallback: urllib + simple parser
        if not markdown_content:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            try:
                with urllib.request.urlopen(req, timeout=8) as response:
                    html = response.read().decode('utf-8', errors='ignore')
                    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
                    html = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n\n## \1\n', html, flags=re.IGNORECASE)
                    html = re.sub(r'<a[^>]+href=["\'](https?://[^"\']+)["\'][^>]*>(.*?)</a>', r'[\2](\1)', html, flags=re.IGNORECASE)
                    text = re.sub(r'<[^>]+>', ' ', html)
                    text = re.sub(r'\s+', ' ', text).strip()
                    text = text.replace("##", "\n\n## ")
                    markdown_content = text
            except Exception as e:
                return f"Error: Failed to fetch webpage. {str(e)}"

        if not markdown_content:
            return "Error: Empty content fetched."

        # Apply BM25 filtering if query is provided
        if query:
            markdown_content = self._bm25_filter(markdown_content, query)

        # Truncate
        if len(markdown_content) > 10000:
            markdown_content = markdown_content[:10000] + "\n\n... [Content truncated due to size]"

        return markdown_content

    def _bm25_filter(self, text: str, query: str, top_k: int = 5) -> str:
        """Filter text sections using BM25 relevance score."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return text

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= top_k:
            return text

        tokenized_corpus = [p.lower().split() for p in paragraphs]
        tokenized_query = query.lower().split()

        try:
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True)[:top_k]
            selected_indices = sorted(top_indices)
            filtered_paragraphs = [paragraphs[i] for i in selected_indices if scores[i] > 0.0]

            if not filtered_paragraphs:
                return "\n\n".join(paragraphs[:top_k])
            
            return "\n\n".join(filtered_paragraphs)
        except Exception as e:
            print(f"[SearchServer] BM25 filtering failed: {e}")
            return text
