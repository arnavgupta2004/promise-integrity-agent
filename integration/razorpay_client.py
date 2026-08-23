"""
integration/razorpay_client.py — §13's exact Razorpay live-slice API calls,
wrapped. Zero business logic: every method here is a thin pass-through to
the official razorpay SDK (client.invoice.create/fetch, client.payment_link
.create/fetch, client.payment.fetch) -- no policy checks, no decisions, no
knowledge of invoices/customers/promises beyond the raw dicts the caller
gives it. Module purity (no policy/, models/, or agent/ imports) is
verified the same way Stage 2 verified features/feature_engine.py's purity
-- an ad hoc grep for import statements (also covered by
tests/test_razorpay_client.py::TestModulePurity), not a standing script.

Test-mode enforcement is the one thing this module insists on beyond pure
pass-through, because the cost of getting it wrong is a real financial API
call: RazorpayClient.__init__ refuses to construct (raises LiveModeKeyError)
unless RAZORPAY_KEY_ID starts with "rzp_test_". A live-mode key is rejected
loudly, not silently skipped or downgraded.
"""
from __future__ import annotations

import os
from typing import Optional

import razorpay

TEST_KEY_PREFIX = "rzp_test_"
LIVE_KEY_PREFIX = "rzp_live_"


class LiveModeKeyError(RuntimeError):
    """Raised instead of silently proceeding whenever a non-test-mode (or
    unrecognized/missing) Razorpay key is detected."""


def assert_test_mode_key(key_id: str) -> None:
    if not key_id:
        raise LiveModeKeyError("RAZORPAY_KEY_ID is empty -- cannot confirm test-mode. Refusing to proceed.")
    if key_id.startswith(LIVE_KEY_PREFIX):
        raise LiveModeKeyError(
            f"RAZORPAY_KEY_ID starts with {LIVE_KEY_PREFIX!r} -- this is a LIVE-MODE key. "
            "Refusing to proceed: this project must only ever use test-mode keys."
        )
    if not key_id.startswith(TEST_KEY_PREFIX):
        raise LiveModeKeyError(
            f"RAZORPAY_KEY_ID does not start with {TEST_KEY_PREFIX!r} -- cannot confirm this is a "
            f"test-mode key. Refusing to proceed. (got prefix: {key_id[:12]!r})"
        )


class RazorpayClient:
    """Thin wrapper, §13's exact 5 calls only. Constructed once; every
    method takes/returns plain dicts (whatever the SDK itself takes/returns)
    -- no Invoice/Customer ORM objects cross this boundary, keeping this
    module decoupled from backend.db as well as policy/models/agent.
    """

    def __init__(self, key_id: Optional[str] = None, key_secret: Optional[str] = None):
        key_id = key_id if key_id is not None else os.environ.get("RAZORPAY_KEY_ID", "")
        key_secret = key_secret if key_secret is not None else os.environ.get("RAZORPAY_KEY_SECRET", "")
        assert_test_mode_key(key_id)
        if not key_secret:
            raise LiveModeKeyError("RAZORPAY_KEY_SECRET is empty -- refusing to proceed.")
        self.key_id = key_id
        self._client = razorpay.Client(auth=(key_id, key_secret))
        print(f"[razorpay_client] confirmed test-mode key in use (prefix={key_id[:len(TEST_KEY_PREFIX) + 4]}...)")

    def create_invoice(self, invoice_data: dict) -> dict:
        return self._client.invoice.create(invoice_data)

    def fetch_invoice(self, invoice_id: str) -> dict:
        return self._client.invoice.fetch(invoice_id)

    def create_payment_link(self, link_data: dict) -> dict:
        return self._client.payment_link.create(link_data)

    def fetch_payment_link(self, payment_link_id: str) -> dict:
        return self._client.payment_link.fetch(payment_link_id)

    def fetch_payment(self, payment_id: str) -> dict:
        return self._client.payment.fetch(payment_id)
