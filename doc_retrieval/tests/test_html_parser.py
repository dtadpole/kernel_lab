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


LARGE_SECTION_HTML = """\
<!DOCTYPE html>
<html><body>
<section id="big-section">
  <h1>Big Section</h1>
  {paragraphs}
</section>
</body></html>
""".format(paragraphs="".join(f"<p>Paragraph {i} with enough words to take up tokens. " * 8 + "</p>" for i in range(30)))

TINY_SECTIONS_HTML = """\
<!DOCTYPE html>
<html><body>
<section id="parent">
  <h1>Parent</h1>
  <p>Parent content.</p>
  <section id="tiny-a">
    <h2>Tiny A</h2>
    <p>Short.</p>
  </section>
  <section id="tiny-b">
    <h2>Tiny B</h2>
    <p>Also short.</p>
  </section>
  <section id="normal">
    <h2>Normal Section</h2>
    <p>This section has enough content to stand alone with many words repeated several times to ensure it exceeds the minimum token threshold for chunking purposes in our system.</p>
  </section>
</section>
</body></html>
"""


class TestChunkSections:
    def test_large_section_is_split(self, enc):
        from doc_retrieval.html_parser import parse_html_doc

        toc, sections, chunks = parse_html_doc(
            LARGE_SECTION_HTML, "test-doc",
            "https://example.com/test-doc/index.html", enc,
            max_tokens=512, min_tokens=128, overlap_tokens=64,
        )
        big_chunks = [c for c in chunks if c["section_id"] == "big-section"]
        assert len(big_chunks) > 1
        assert all(c["section_id"] == "big-section" for c in big_chunks)

    def test_tiny_sections_merged(self, enc):
        from doc_retrieval.html_parser import parse_html_doc

        toc, sections, chunks = parse_html_doc(
            TINY_SECTIONS_HTML, "test-doc",
            "https://example.com/test-doc/index.html", enc,
            max_tokens=512, min_tokens=128, overlap_tokens=64,
        )
        chunk_texts = " ".join(c["text"] for c in chunks)
        assert "Tiny A" in chunk_texts
        assert "Tiny B" in chunk_texts

    def test_toc_output(self, enc):
        from doc_retrieval.html_parser import parse_html_doc

        toc, sections, chunks = parse_html_doc(
            FIXTURE_HTML, "test-doc",
            "https://example.com/test-doc/index.html", enc,
        )
        assert len(toc) == 6
        by_id = {t["section_id"]: t for t in toc}
        assert "children" in by_id["overview"]
        assert by_id["overview"]["parent_section_id"] is None

    def test_sections_output(self, enc):
        from doc_retrieval.html_parser import parse_html_doc

        toc, sections, chunks = parse_html_doc(
            FIXTURE_HTML, "test-doc",
            "https://example.com/test-doc/index.html", enc,
        )
        assert len(sections) > 0
        sec = next(s for s in sections if s["section_id"] == "introduction")
        assert "content" in sec
        assert "token_count" in sec
        assert sec["token_count"] > 0

    def test_chunks_have_section_id(self, enc):
        from doc_retrieval.html_parser import parse_html_doc

        toc, sections, chunks = parse_html_doc(
            FIXTURE_HTML, "test-doc",
            "https://example.com/test-doc/index.html", enc,
        )
        for chunk in chunks:
            assert "section_id" in chunk
            assert "chunk_id" in chunk
            assert "doc_id" in chunk
            assert chunk["doc_id"] == "test-doc"
