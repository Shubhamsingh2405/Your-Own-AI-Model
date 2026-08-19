from backend.retrieval.bm25 import BM25Index


def test_bm25_ranks_matching_doc():
    idx = BM25Index()
    idx.add_document(1, "binary search tree algorithm")
    idx.add_document(2, "pizza recipe tomato cheese")
    hits = idx.search("binary search tree", 2)
    assert hits[0][1] == 1
    assert hits[0][0] > hits[1][0]
