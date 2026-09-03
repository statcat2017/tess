"""Shared pytest gates (network-dependent archive tests skip cleanly)."""

import socket

import pytest


def _network_ok() -> bool:
    try:
        socket.create_connection(("mast.stsci.edu", 443), timeout=5).close()
        return True
    except OSError:
        return False


needs_archive = pytest.mark.skipif(
    not _network_ok(), reason="MAST unreachable; replay needs the archive"
)
