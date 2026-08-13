"""Safety-case RAG: parser -> chunking -> embeddings -> store -> retriever.

The seed corpus is SYNTHETIC demonstration material. Every synthetic rule is
tagged SYNTHETIC_EXAMPLE / NOT_A_REAL_STANDARD in both content and metadata;
SOTIF (ISO 21448) entries are honest concept paraphrases labeled
PARAPHRASE_NOT_STANDARD_TEXT. The retriever ALWAYS returns
{source, document, version, section, retrieved_text, relevance_score}.
"""
