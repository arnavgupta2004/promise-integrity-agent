"""
tests/test_razorpay_client.py — mock-based unit tests for
integration/razorpay_client.py. No network calls, no real credentials
needed: the razorpay.Client instance is monkeypatched after construction.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from integration.razorpay_client import (
    LIVE_KEY_PREFIX,
    TEST_KEY_PREFIX,
    LiveModeKeyError,
    RazorpayClient,
    assert_test_mode_key,
)


class TestTestModeEnforcement:
    def test_accepts_test_mode_key(self):
        client = RazorpayClient(key_id="rzp_test_abc123", key_secret="secret")
        assert client.key_id == "rzp_test_abc123"

    def test_rejects_live_mode_key(self):
        with pytest.raises(LiveModeKeyError, match="LIVE-MODE"):
            RazorpayClient(key_id="rzp_live_abc123", key_secret="secret")

    def test_rejects_unrecognized_key_prefix(self):
        with pytest.raises(LiveModeKeyError, match="cannot confirm this is a test-mode key"):
            RazorpayClient(key_id="some_other_prefix_abc", key_secret="secret")

    def test_rejects_empty_key_id(self):
        with pytest.raises(LiveModeKeyError):
            RazorpayClient(key_id="", key_secret="secret")

    def test_rejects_empty_key_secret(self):
        with pytest.raises(LiveModeKeyError):
            RazorpayClient(key_id="rzp_test_abc123", key_secret="")

    def test_assert_test_mode_key_helper_directly(self):
        assert_test_mode_key("rzp_test_x")  # does not raise
        with pytest.raises(LiveModeKeyError):
            assert_test_mode_key("rzp_live_x")
        with pytest.raises(LiveModeKeyError):
            assert_test_mode_key("")


class TestThinWrapperPassthrough:
    """Every method must be a pure pass-through: call the matching SDK
    sub-resource method with the given argument(s), return its result
    unmodified -- no transformation, no added logic."""

    @pytest.fixture
    def client_with_mock_sdk(self):
        client = RazorpayClient(key_id="rzp_test_abc123", key_secret="secret")
        client._client = MagicMock()
        return client

    def test_create_invoice(self, client_with_mock_sdk):
        client_with_mock_sdk._client.invoice.create.return_value = {"id": "inv_123"}
        result = client_with_mock_sdk.create_invoice({"amount": 1000})
        client_with_mock_sdk._client.invoice.create.assert_called_once_with({"amount": 1000})
        assert result == {"id": "inv_123"}

    def test_fetch_invoice(self, client_with_mock_sdk):
        client_with_mock_sdk._client.invoice.fetch.return_value = {"id": "inv_123", "status": "paid"}
        result = client_with_mock_sdk.fetch_invoice("inv_123")
        client_with_mock_sdk._client.invoice.fetch.assert_called_once_with("inv_123")
        assert result == {"id": "inv_123", "status": "paid"}

    def test_create_payment_link(self, client_with_mock_sdk):
        client_with_mock_sdk._client.payment_link.create.return_value = {"id": "plink_1", "short_url": "https://rzp.io/x"}
        result = client_with_mock_sdk.create_payment_link({"amount": 5000})
        client_with_mock_sdk._client.payment_link.create.assert_called_once_with({"amount": 5000})
        assert result["short_url"] == "https://rzp.io/x"

    def test_fetch_payment_link(self, client_with_mock_sdk):
        client_with_mock_sdk._client.payment_link.fetch.return_value = {"id": "plink_1", "status": "paid"}
        result = client_with_mock_sdk.fetch_payment_link("plink_1")
        client_with_mock_sdk._client.payment_link.fetch.assert_called_once_with("plink_1")
        assert result["status"] == "paid"

    def test_fetch_payment(self, client_with_mock_sdk):
        client_with_mock_sdk._client.payment.fetch.return_value = {"id": "pay_1", "amount": 5000, "status": "captured"}
        result = client_with_mock_sdk.fetch_payment("pay_1")
        client_with_mock_sdk._client.payment.fetch.assert_called_once_with("pay_1")
        assert result["status"] == "captured"


class TestModulePurity:
    def test_zero_policy_models_agent_imports(self):
        import ast
        source = Path("integration/razorpay_client.py").read_text()
        tree = ast.parse(source)
        forbidden = {"policy", "models", "agent"}
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in forbidden:
                        found.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top in forbidden:
                        found.add(top)
        assert not found, f"integration/razorpay_client.py imports forbidden module(s): {found}"
