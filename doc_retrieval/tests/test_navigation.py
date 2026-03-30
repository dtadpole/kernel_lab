"""Tests for TOC browsing and section reading."""

import json

import pytest


@pytest.fixture
def nav_dir(tmp_path):
    """Create test TOC and sections JSONL files."""
    toc = [
        {"doc_id": "test-doc", "section_id": "intro", "title": "Introduction",
         "heading_level": 1, "parent_section_id": None,
         "children": ["basics", "advanced"],
         "deep_link": "https://example.com/test-doc/index.html#intro"},
        {"doc_id": "test-doc", "section_id": "basics", "title": "Basics",
         "heading_level": 2, "parent_section_id": "intro",
         "children": [],
         "deep_link": "https://example.com/test-doc/index.html#basics"},
        {"doc_id": "test-doc", "section_id": "advanced", "title": "Advanced",
         "heading_level": 2, "parent_section_id": "intro",
         "children": [],
         "deep_link": "https://example.com/test-doc/index.html#advanced"},
        {"doc_id": "other-doc", "section_id": "overview", "title": "Overview",
         "heading_level": 1, "parent_section_id": None,
         "children": [],
         "deep_link": "https://example.com/other-doc/index.html#overview"},
    ]
    sections = [
        {"doc_id": "test-doc", "section_id": "intro", "title": "Introduction",
         "heading_level": 1, "heading_path": ["Introduction"],
         "content": "<p>Welcome to the intro.</p>", "token_count": 10,
         "deep_link": "https://example.com/test-doc/index.html#intro"},
        {"doc_id": "test-doc", "section_id": "basics", "title": "Basics",
         "heading_level": 2, "heading_path": ["Introduction", "Basics"],
         "content": "<p>Basic content here.</p>", "token_count": 8,
         "deep_link": "https://example.com/test-doc/index.html#basics"},
        {"doc_id": "test-doc", "section_id": "advanced", "title": "Advanced",
         "heading_level": 2, "heading_path": ["Introduction", "Advanced"],
         "content": "<p>Advanced content here.</p>", "token_count": 8,
         "deep_link": "https://example.com/test-doc/index.html#advanced"},
        {"doc_id": "other-doc", "section_id": "overview", "title": "Overview",
         "heading_level": 1, "heading_path": ["Overview"],
         "content": "<p>Other doc overview.</p>", "token_count": 8,
         "deep_link": "https://example.com/other-doc/index.html#overview"},
    ]

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    with open(chunks_dir / "toc.jsonl", "w") as f:
        for entry in toc:
            f.write(json.dumps(entry) + "\n")
    with open(chunks_dir / "sections.jsonl", "w") as f:
        for sec in sections:
            f.write(json.dumps(sec) + "\n")
    (chunks_dir / "all_chunks.jsonl").touch()
    (tmp_path / "index").mkdir()

    return tmp_path


class TestBrowseToc:
    def test_top_level(self, nav_dir):
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=nav_dir / "index")
        s._runtime_root = nav_dir
        result = s.browse_toc("test-doc")
        assert len(result) == 1
        assert result[0]["section_id"] == "intro"
        assert result[0]["children_count"] == 2

    def test_expand_section(self, nav_dir):
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=nav_dir / "index")
        s._runtime_root = nav_dir
        result = s.browse_toc("test-doc", section_id="intro")
        assert result["section_id"] == "intro"
        assert len(result["children"]) == 2
        assert result["children"][0]["section_id"] == "basics"

    def test_unknown_doc(self, nav_dir):
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=nav_dir / "index")
        s._runtime_root = nav_dir
        result = s.browse_toc("nonexistent")
        assert result == []


class TestReadSection:
    def test_read_existing(self, nav_dir):
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=nav_dir / "index")
        s._runtime_root = nav_dir
        result = s.read_section("test-doc", "basics")
        assert result["content"] == "<p>Basic content here.</p>"
        assert result["title"] == "Basics"
        assert result["nav"]["parent"] == "intro"

    def test_sibling_navigation(self, nav_dir):
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=nav_dir / "index")
        s._runtime_root = nav_dir
        result = s.read_section("test-doc", "basics")
        assert result["nav"]["prev_sibling"] is None
        assert result["nav"]["next_sibling"] == "advanced"

    def test_not_found(self, nav_dir):
        from doc_retrieval.searcher import DocSearcher

        s = DocSearcher(index_dir=nav_dir / "index")
        s._runtime_root = nav_dir
        result = s.read_section("test-doc", "nonexistent")
        assert result is None
