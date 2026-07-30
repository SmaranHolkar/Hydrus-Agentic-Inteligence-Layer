"""
Structured unit test for ArtifactRenderer — no model required.
"""
import sys
import os
import unittest
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hydrusopt import ArtifactRenderer

class TestArtifact(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_artifacts"
        self.renderer = ArtifactRenderer(output_dir=self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except Exception:
                pass

    def test_render_code(self):
        py_code = 'def greet(name):\n    return f"Hello, {name}!"\n'
        p = self.renderer.render_code(py_code, lang="python", title="greet.py")
        self.assertTrue(os.path.exists(p))

    def test_render_html(self):
        html_body = '<h1>HydrusOPT Artifact Preview</h1>'
        p = self.renderer.render_html(html_body, title="preview.html")
        self.assertTrue(os.path.exists(p))

    def test_detect_and_render(self):
        model_output = '```html\n<p>Hello</p>\n```\n'
        paths = self.renderer.detect_and_render(model_output)
        self.assertEqual(len(paths), 1)
        self.assertTrue(os.path.exists(paths[0]))

if __name__ == "__main__":
    unittest.main()
