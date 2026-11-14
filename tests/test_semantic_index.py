import tempfile
import unittest
from pathlib import Path
from scripts.build_semantic_index import build_index


class TestSemanticIndex(unittest.TestCase):
    def test_symbols_reverse_dependencies_and_test_associations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'tests').mkdir()
            (root / 'src' / 'example.py').write_text('"""Example module."""\nclass RAGTrace: pass\n')
            (root / 'tests' / 'test_example.py').write_text('from src.example import RAGTrace\n')
            index = build_index(root)
            example = next(e for e in index['files'] if e['path'] == 'src/example.py')
            self.assertEqual(example['tests'], ['tests/test_example.py'])
            self.assertEqual(example['depended_on_by'], ['tests/test_example.py'])
            self.assertEqual(index['concepts']['RAGTrace'][0]['line'], 2)
            self.assertEqual(index, build_index(root))
