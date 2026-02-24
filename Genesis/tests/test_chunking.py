from genesis.utils.text import chunk_text

def test_chunking_nonempty():
    t = "A\n\nB\n\nC"
    chunks = chunk_text(t, chunk_size=3, overlap=0)
    assert len(chunks) >= 1
