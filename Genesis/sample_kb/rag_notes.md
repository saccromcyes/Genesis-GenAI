# Retrieval-Augmented Generation (RAG)

RAG improves factual grounding by retrieving relevant passages from a corpus and placing them into the LLM context.

Key components:
1) Chunking: split documents into semantically meaningful chunks.
2) Embeddings: represent chunks as vectors.
3) Vector index: FAISS stores vectors for efficient similarity search.
4) Retrieval: fetch top-k chunks for a query.
5) Generation: prompt the LLM with retrieved chunks and the question.
6) Attribution: provide citations/snippets for transparency.

Failure modes:
- Over-retrieval: too many chunks reduces usable context.
- Under-retrieval: missing critical evidence.
- Wrong chunking: splits destroy meaning.
- Embedding drift: poor similarity.

Mitigations:
- MMR / diversity-aware retrieval
- hybrid re-ranking
- chunk overlap
- evaluation with known Q/A pairs
