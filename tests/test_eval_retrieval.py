"""Offline checks for weighted RRF and the rerank wrapper (no database, no model download)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from eval_retrieval import Hybrid, Rerank, rrf


def c(id):
    return {'id': id, 'text': id, 'score': 0.0}


class FakeRetriever:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query, *, filters=None, top_k=3):
        return [c(i) for i in self.ids][:top_k]


class FakeCrossEncoder:
    def predict(self, pairs):
        return [len(text) for _, text in pairs]  # longer text = more relevant


class EvalRetrievalTests(unittest.TestCase):
    def test_weighted_rrf_lets_dense_win(self):
        dense, lexical = [c('d1'), c('x')], [c('l1'), c('x')]
        self.assertEqual(rrf(dense, lexical)[0]['id'], 'x')  # in both lists
        self.assertEqual(rrf(dense, lexical, weights=[1.0, 0.0])[0]['id'], 'd1')
        with self.assertRaises(ValueError):
            rrf(dense, lexical, weights=[1.0])

    def test_hybrid_passes_lexical_weight(self):
        h = Hybrid(FakeRetriever(['d1', 'x']), FakeRetriever(['l1', 'x']), lexical_weight=0.0)
        self.assertEqual([r['id'] for r in h.search('q', top_k=2)], ['d1', 'x'])

    def test_rerank_reorders_and_keeps_first_stage_rank(self):
        r = object.__new__(Rerank)  # skip loading a real model
        r.base, r.pool, r.model = FakeRetriever(['a', 'ccc', 'bb']), 20, FakeCrossEncoder()
        out = r.search('q', top_k=2)
        self.assertEqual([x['id'] for x in out], ['ccc', 'bb'])
        self.assertEqual(out[0]['retrieval_rank'], 2)


if __name__ == '__main__':
    unittest.main()
