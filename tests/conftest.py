import pytest

from triage.models import Customer, Ticket
from triage.retrieval import load_runbooks


@pytest.fixture(scope="session")
def runbooks():
    return load_runbooks()


@pytest.fixture
def make_ticket():
    def _make(id="T-1", subject="Charged twice", body="I was charged twice, please refund.", days=1, amount=29.0):
        return Ticket(id=id, subject=subject, body=body,
                      customer=Customer(customer_id="C-1", plan="pro", last_charge_days_ago=days, last_charge_amount=amount))
    return _make

