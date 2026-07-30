import unittest
import os
import shutil
from pathlib import Path
from hydrus_agent.security import WorkspaceGuard
from hydrus_agent.llm_bridge import HydrusOptAdapter
from hydrus_agent.servers.file_system import FileSystemServer
from hydrus_agent.mcp_bus import MCPBus

class TestWorkspaceGuard(unittest.TestCase):
    def setUp(self):
        self.workspace = os.path.abspath("test_temp_workspace")
        self.guard = WorkspaceGuard(self.workspace)

    def tearDown(self):
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)

    def test_path_validation(self):
        # Inside workspace
        p = self.guard.validate_path("sub/file.txt")
        self.assertTrue(p.is_absolute())
        self.assertTrue(str(p).startswith(self.workspace))

        # Outside workspace (escape try)
        with self.assertRaises(PermissionError):
            self.guard.validate_path("../escaped.txt")

    def test_command_validation(self):
        # Safe commands
        allowed, danger, _ = self.guard.validate_command("git status")
        self.assertTrue(allowed)
        self.assertEqual(danger, "medium")

        allowed, danger, _ = self.guard.validate_command("pytest tests/")
        self.assertTrue(allowed)
        self.assertEqual(danger, "medium")

        # Blocked commands
        allowed, _, _ = self.guard.validate_command("rm -rf /")
        self.assertFalse(allowed)

        allowed, _, _ = self.guard.validate_command("sudo apt-get install python3")
        self.assertFalse(allowed)


class TestFileSystemServer(unittest.TestCase):
    def setUp(self):
        self.workspace = os.path.abspath("test_temp_workspace")
        self.guard = WorkspaceGuard(self.workspace)
        self.fs_server = FileSystemServer(self.guard, "test_session")
        
        # Create a test file
        self.test_file = os.path.join(self.workspace, "code.py")
        os.makedirs(self.workspace, exist_ok=True)
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("def hello():\n    print('hello world')\n\nhello()\n")

    def tearDown(self):
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)

    def test_read_and_write(self):
        # Read file
        res = self.fs_server.read_file("code.py")
        self.assertIn("def hello():", res)

        # Write file
        res = self.fs_server.write_file("new.txt", "content here")
        self.assertIn("Success", res)
        self.assertTrue(os.path.exists(os.path.join(self.workspace, "new.txt")))

    def test_patch_file(self):
        # Exact patch match
        res = self.fs_server.patch_file(
            "code.py",
            "print('hello world')",
            "print('hello from test')"
        )
        self.assertIn("Success", res)
        
        # Verify changes
        updated = self.fs_server.read_file("code.py")
        self.assertIn("hello from test", updated)
        self.assertNotIn("hello world", updated)

        # Confirm backup created
        backup_dir = os.path.join(self.workspace, "data", "backups", "test_session")
        self.assertTrue(os.path.exists(backup_dir))
        self.assertTrue(len(os.listdir(backup_dir)) > 0)

    def test_make_directory_and_delete_file(self):
        # Create directory
        res = self.fs_server.make_directory("test_dir")
        self.assertIn("Success", res)
        self.assertTrue(os.path.exists(os.path.join(self.workspace, "test_dir")))

        # Delete file
        res = self.fs_server.delete_file("code.py")
        self.assertIn("Success", res)
        self.assertFalse(os.path.exists(os.path.join(self.workspace, "code.py")))

        # Delete directory
        res = self.fs_server.delete_directory("test_dir")
        self.assertIn("Success", res)
        self.assertFalse(os.path.exists(os.path.join(self.workspace, "test_dir")))


class TestReActParser(unittest.TestCase):
    def setUp(self):
        # Dummy adapter
        self.adapter = HydrusOptAdapter(model_name="microsoft/Phi-3.5-mini-instruct", model=object(), tokenizer=object())

    def test_parsing_re_act(self):
        # Normal react thought and action
        text = "<thought>I should list files.</thought>\n<action>list_directory|{\"path\": \".\"}</action>"
        thought, name, args = self.adapter.parse_react_output(text)
        self.assertEqual(thought, "I should list files.")
        self.assertEqual(name, "list_directory")
        self.assertEqual(args, {"path": "."})

        # No closing tag fallback
        text_fallback = "<thought>Running a command.</thought>\n<action>run_command|pytest tests/"
        thought, name, args = self.adapter.parse_react_output(text_fallback)
        self.assertEqual(name, "run_command")
        self.assertEqual(args, {"command": "pytest tests/"})

    def test_robust_fallback_parsing(self):
        # 1. Action: Prefix with pipe
        text1 = "<thought>Searching...</thought>\nAction: web_search|{\"query\": \"Nothing Phone\"}"
        thought, name, args = self.adapter.parse_react_output(text1)
        self.assertEqual(name, "web_search")
        self.assertEqual(args, {"query": "Nothing Phone"})

        # 2. Action: Prefix with 'with args' / plain JSON (frequent LLM format)
        text2 = "<thought>Finishing</thought>\nAction: FinalAnswer' with args {\"answer\": \"Hello World\"}"
        thought, name, args = self.adapter.parse_react_output(text2)
        self.assertEqual(name, "FinalAnswer")
        self.assertEqual(args, {"answer": "Hello World"})

        # 3. Plain ToolName|JSON (no action tag or label prefix)
        text3 = "FinalAnswer|{\"answer\": \"Direct format test\"}"
        thought, name, args = self.adapter.parse_react_output(text3)
        self.assertEqual(name, "FinalAnswer")
        self.assertEqual(args, {"answer": "Direct format test"})


class TestMCPBus(unittest.TestCase):
    def setUp(self):
        self.workspace = os.path.abspath("test_temp_workspace")
        self.bus = MCPBus(self.workspace, "test_session")

    def tearDown(self):
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)

    def test_execute_argument_filtering(self):
        # We try to write a file, but pass an unexpected argument "extra_param"
        import asyncio
        # Create parent directory
        os.makedirs(self.workspace, exist_ok=True)
        
        loop = asyncio.new_event_loop()
        try:
            res = loop.run_until_complete(self.bus.execute("write_file", {
                "path": "test.txt",
                "content": "some content",
                "extra_param": "hallucinated value"
            }))
        finally:
            loop.close()
        self.assertIn("Success", res)
        self.assertTrue(os.path.exists(os.path.join(self.workspace, "test.txt")))


if __name__ == "__main__":
    unittest.main()
