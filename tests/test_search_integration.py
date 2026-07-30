import os
import sys
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hydrus_agent.security import WorkspaceGuard
from hydrus_agent.servers.search import SearchServer

class TestSearchServer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.workspace = os.path.abspath("test_temp_workspace_search")
        self.guard = WorkspaceGuard(self.workspace)
        self.server = SearchServer(self.guard, "test_session")

    def tearDown(self):
        if os.path.exists(self.workspace):
            import shutil
            try:
                shutil.rmtree(self.workspace)
            except Exception:
                pass

    @patch('urllib.request.urlopen')
    async def test_web_search_searxng_fallback_success(self, mock_urlopen):
        # Mock SearXNG JSON response
        mock_response = MagicMock()
        mock_response.read.return_value = b'''{
            "results": [
                {"title": "Result 1", "url": "http://example.com/1", "content": "First relevant info"},
                {"title": "Result 2", "url": "http://example.com/2", "content": "Second relevant info"}
            ]
        }'''
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Call web_search (browser-use will fail to import in test, falling back to HTTP search)
        res = await self.server.web_search("test query")
        self.assertIn("Live SearXNG Search results", res)
        self.assertIn("[Result 1](http://example.com/1)", res)
        self.assertIn("First relevant info", res)

    @patch('urllib.request.urlopen')
    async def test_web_search_searxng_failure_ddg_success(self, mock_urlopen):
        # First call (SearXNG) raises exception, second call (DDG) returns HTML
        mock_response_ddg = MagicMock()
        mock_response_ddg.read.return_value = b'''
        <html>
            <body>
                <div class="result__snippet">DuckDuckGo fallback scrapable result 1</div>
            </body>
        </html>
        '''
        
        # Side effect: exception for SearXNG, return mock for DDG
        mock_urlopen.side_effect = [Exception("SearXNG down"), MagicMock(__enter__=MagicMock(return_value=mock_response_ddg))]

        res = await self.server.web_search("fallback query")
        self.assertIn("Live DDG Search results", res)
        self.assertIn("DuckDuckGo fallback scrapable result 1", res)

    @patch('urllib.request.urlopen')
    async def test_fetch_webpage_fallback_html_parsing(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'''
        <html>
            <head><style>body {color: red;}</style></head>
            <body>
                <h1>Main Heading</h1>
                <p>Paragraph text here with a <a href="http://link.com">hyperlink</a>.</p>
            </body>
        </html>
        '''
        mock_urlopen.return_value.__enter__.return_value = mock_response

        res = await self.server.fetch_webpage("http://somepage.com")
        self.assertNotIn("<style>", res)
        self.assertIn("Main Heading", res)
        self.assertIn("[hyperlink](http://link.com)", res)

    def test_bm25_filtering_logic(self):
        text = (
            "Paragraph one is about machine learning and neural networks.\n\n"
            "Paragraph two is totally unrelated, talking about recipes for cakes.\n\n"
            "Paragraph three covers deep learning, gradient descent, and training models."
        )
        
        res = self.server._bm25_filter(text, "deep learning", top_k=1)
        self.assertTrue(len(res) > 0)

        # Mock rank_bm25 to test filtering logic explicitly
        mock_bm25_module = MagicMock()
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.get_scores.return_value = [0.8, 0.0, 0.9]
        mock_bm25_module.BM25Okapi.return_value = mock_bm25_instance

        with patch.dict('sys.modules', {'rank_bm25': mock_bm25_module}):
            res_filtered = self.server._bm25_filter(text, "deep learning", top_k=2)
            self.assertIn("Paragraph one", res_filtered)
            self.assertIn("Paragraph three", res_filtered)
            self.assertNotIn("Paragraph two", res_filtered)

    @patch('hydrus_agent.servers.search.SearchServer._run_browser_use_agent')
    async def test_browser_use_agent_success_paths(self, mock_run_agent):
        # Verify that if the browser-use agent runs successfully, its result is returned directly
        mock_run_agent.return_value = "Mocked browser-use agent final answer summary."
        
        search_res = await self.server.web_search("test query")
        self.assertIn("Autonomous Browser-Agent Search results", search_res)
        self.assertIn("Mocked browser-use agent final answer summary.", search_res)

        fetch_res = await self.server.fetch_webpage("http://someurl.com", "my query")
        self.assertEqual(fetch_res, "Mocked browser-use agent final answer summary.")

    async def test_run_browser_use_agent_mocked_execution(self):
        # Mock langchain_ollama and browser_use modules
        mock_chat_ollama = MagicMock()
        mock_chat_ollama_class = MagicMock(return_value=mock_chat_ollama)
        
        mock_agent_instance = MagicMock()
        mock_history = MagicMock()
        mock_history.final_result.return_value = "Simulated agent browsing result"
        mock_agent_instance.run = AsyncMock(return_value=mock_history)
        mock_agent_class = MagicMock(return_value=mock_agent_instance)

        with patch.dict('sys.modules', {
            'langchain_ollama': MagicMock(ChatOllama=mock_chat_ollama_class),
            'browser_use': MagicMock(Agent=mock_agent_class)
        }):
            res = await self.server._run_browser_use_agent("Search something")
            self.assertEqual(res, "Simulated agent browsing result")
            mock_agent_class.assert_called_once()
            mock_chat_ollama_class.assert_called_once()

if __name__ == '__main__':
    unittest.main()
