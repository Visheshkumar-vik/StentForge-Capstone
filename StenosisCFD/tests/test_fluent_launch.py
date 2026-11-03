"""Tests for ``fluent_launch._resolve_product_version`` (16 Aug 2026, the Gadi
kit): the product version is derived from the environment, never pinned, so the
same code finds v261 on the laptop (AWP_ROOT261) and under
``module load ansys/2026r1`` on Gadi.  The laptop-unchanged property is the
load-bearing one and is pinned explicitly.
"""
from __future__ import annotations

import os
import re

import pytest

import fluent_launch as fl

_AWP_RE = re.compile(r"AWP_ROOT\d{3}")


def _clear_version_env(monkeypatch):
    """Remove every input the resolver reads, so each test sets up from zero."""
    monkeypatch.delenv(fl.ENV_PRODUCT_VERSION, raising=False)
    for key in [k for k in os.environ if _AWP_RE.fullmatch(k)]:
        monkeypatch.delenv(key, raising=False)


def test_explicit_override_wins(monkeypatch):
    """STENOSIS_FLUENT_PRODUCT_VERSION beats every AWP_ROOT*."""
    monkeypatch.setenv(fl.ENV_PRODUCT_VERSION, "23.2.0")
    monkeypatch.setenv("AWP_ROOT261", r"D:\whatever")
    assert fl._resolve_product_version() == "23.2.0"


def test_derives_from_the_newest_awp_root(monkeypatch):
    """The newest AWP_ROOT<vnn> is the loaded module; v232 must never win on the
    laptop, where both roots are set."""
    _clear_version_env(monkeypatch)
    monkeypatch.setenv("AWP_ROOT232", r"C:\old")
    monkeypatch.setenv("AWP_ROOT261", "/apps/ansys/2026r1/v261")
    assert fl._resolve_product_version() == "26.1.0"


def test_single_awp_root_derives(monkeypatch):
    _clear_version_env(monkeypatch)
    monkeypatch.setenv("AWP_ROOT232", r"C:\only-this-one")
    assert fl._resolve_product_version() == "23.2.0"


def test_fallback_is_the_historical_pin(monkeypatch):
    """No override and no AWP_ROOT* -> the historical laptop value, so a bare
    shell behaves exactly as the old hardcode did."""
    _clear_version_env(monkeypatch)
    assert fl._resolve_product_version() == "26.1.0"


def test_laptop_resolution_is_byte_identical_to_the_old_hardcode():
    """On the real laptop environment the derived value is exactly the old pin
    -- the Gadi-port change must not move laptop behaviour one bit."""
    if not any(_AWP_RE.fullmatch(k) and k == "AWP_ROOT261" for k in os.environ):
        pytest.skip("not the laptop environment (AWP_ROOT261 unset)")
    assert fl._resolve_product_version() == "26.1.0"
    assert fl.PRODUCT_VERSION == "26.1.0"
