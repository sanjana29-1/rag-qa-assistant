"""Evaluate retrieval quality with a small labelled question set.

Metrics:
  Hit@1 - is the correct section ranked first?
  Hit@3 - is the correct section in the top 3?
  MRR   - mean reciprocal rank of the correct section.
"""
from rag import Retriever, load_chunks

EVAL_SET = [  # (question, expected section) - phrased differently from the source text on purpose
    ("How long do I have to bring back a full price top?", "Returns policy"),
    ("Can I get my money back for something I bought on sale?", "Sale item returns"),
    ("My jacket's zip broke after a week, what are my rights?", "Faulty items"),
    ("How long will the shop keep my click and collect parcel?", "Online orders"),
    ("What deposit do I need to put an item on lay-by?", "Lay-by"),
    ("Do gift vouchers expire?", "Gift cards"),
    ("How many points do I need for a reward voucher?", "Loyalty program"),
    ("What time do you close on Thursday?", "Opening hours"),
    ("What discount do employees get?", "Staff discount"),
    ("When are the rosters released?", "Rostering"),
    ("What do I do if the till is short by $20?", "Cash handling"),
    ("Is the store open on Sunday morning?", "Opening hours"),
]


def evaluate(k: int = 3) -> dict:
    retriever = Retriever(load_chunks("knowledge_base.md"))
    hit1 = hitk = rr = 0.0
    print(f"{'Question':60} {'Expected':18} {'Top result':18} Rank")
    for q, expected in EVAL_SET:
        ranked = [c.title for c, _ in retriever.search(q, k=len(retriever.chunks))]
        rank = ranked.index(expected) + 1
        hit1 += rank == 1; hitk += rank <= k; rr += 1 / rank
        print(f"{q[:58]:60} {expected:18} {ranked[0]:18} {rank}")
    n = len(EVAL_SET)
    res = {"Hit@1": hit1 / n, f"Hit@{k}": hitk / n, "MRR": rr / n}
    print("\n" + "  ".join(f"{m}: {v:.2f}" for m, v in res.items()))
    return res


if __name__ == "__main__":
    evaluate()
