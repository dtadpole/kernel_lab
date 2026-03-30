"""Tests for BeautifulSoup-based HTML parser."""

import tiktoken
import pytest

FIXTURE_HTML = """\
<!DOCTYPE html>
<html>
<body>
<section id="overview">
  <h1>Overview</h1>
  <p>This is the overview.</p>
  <section id="introduction">
    <h2>Introduction</h2>
    <p>Intro content with <code>code_example()</code> and <strong>bold text</strong>.</p>
    <pre><code>int x = 42;</code></pre>
  </section>
  <section id="getting-started">
    <h2>Getting Started</h2>
    <p>Getting started content.</p>
    <section id="installation">
      <h3>Installation</h3>
      <p>Install instructions.</p>
    </section>
    <section id="first-steps">
      <h3>First Steps</h3>
      <p>First steps content.</p>
    </section>
  </section>
</section>
<section id="programming-model">
  <h1>Programming Model</h1>
  <p>Programming model content.</p>
  <table><tr><th>Feature</th><th>Support</th></tr><tr><td>CUDA</td><td>Yes</td></tr></table>
</section>
</body>
</html>
"""


@pytest.fixture
def enc():
    return tiktoken.get_encoding("cl100k_base")


class TestExtractSections:
    def test_finds_all_sections(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        ids = [s["section_id"] for s in sections]
        assert "overview" in ids
        assert "introduction" in ids
        assert "getting-started" in ids
        assert "installation" in ids
        assert "programming-model" in ids
        assert len(sections) == 6

    def test_heading_levels(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        assert by_id["overview"]["heading_level"] == 1
        assert by_id["introduction"]["heading_level"] == 2
        assert by_id["installation"]["heading_level"] == 3

    def test_parent_child_relationships(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        assert by_id["introduction"]["parent_section_id"] == "overview"
        assert by_id["installation"]["parent_section_id"] == "getting-started"
        assert by_id["overview"]["parent_section_id"] is None
        assert "introduction" in by_id["overview"]["children"]
        assert "getting-started" in by_id["overview"]["children"]

    def test_deep_link(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        assert by_id["introduction"]["deep_link"] == "https://example.com/test-doc/index.html#introduction"

    def test_heading_path(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        assert by_id["installation"]["heading_path"] == ["Overview", "Getting Started", "Installation"]

    def test_content_excludes_nested_sections(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        # overview's direct content should NOT contain "Intro content"
        assert "Intro content" not in by_id["overview"]["content"]
        assert "This is the overview" in by_id["overview"]["content"]

    def test_content_preserves_semantic_html(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        content = by_id["introduction"]["content"]
        assert "<code>" in content
        assert "<strong>" in content
        assert "<pre>" in content

    def test_content_preserves_tables(self):
        from doc_retrieval.html_parser import extract_sections

        sections = extract_sections(FIXTURE_HTML, "test-doc", "https://example.com/test-doc/index.html")
        by_id = {s["section_id"]: s for s in sections}
        content = by_id["programming-model"]["content"]
        assert "<table>" in content
        assert "<th>" in content
