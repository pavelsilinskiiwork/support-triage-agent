import pytest

from triage.evaluate import load_tickets, retrieval_recall
from triage.retrieval import Index, load_runbooks


def test_all_runbooks_load_with_ids(runbooks):
    assert len(runbooks) == 13
    assert all(rid == rb.id and rb.text for rid, rb in runbooks.items())


@pytest.mark.parametrize("query,expected", [
    ("refund duplicate charge", "billing-refund-policy"),
    ("forgot password reset link", "account-password-reset"),
    ("lost phone two-factor recovery codes", "account-2fa-lost-device"),
    ("CNAME domain verification", "howto-custom-domain"),
    ("502 errors in every region", "outage-status-page"),
])
def test_top_hit(runbooks, query, expected):
    assert Index(runbooks).search(query, k=1)[0][0].id == expected


def test_no_overlap_returns_nothing(runbooks):
    assert Index(runbooks).search("zzzz qqqq", k=3) == []


def test_missing_front_matter_is_an_error(tmp_path):
    (tmp_path / "bad.md").write_text("no front matter")
    with pytest.raises(ValueError):
        load_runbooks(tmp_path)


def test_recall_at_3_on_the_labelled_tickets():
    hit, total = retrieval_recall(load_tickets())
    assert total >= 25
    assert hit / total >= 0.9
