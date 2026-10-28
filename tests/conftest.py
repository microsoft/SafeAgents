import os
import socket

import pytest


_network_guards = pytest.MonkeyPatch()


def _forbid_external_calls(*args, **kwargs):
    raise AssertionError("LLM, credential, and external network calls are forbidden in these tests.")


def pytest_configure(config):
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def local_connect(sock, address):
        if sock.family != socket.AF_UNIX:
            _forbid_external_calls()
        return original_connect(sock, address)

    def local_connect_ex(sock, address):
        if sock.family != socket.AF_UNIX:
            _forbid_external_calls()
        return original_connect_ex(sock, address)

    _network_guards.setattr(socket.socket, "connect", local_connect)
    _network_guards.setattr(socket.socket, "connect_ex", local_connect_ex)
    _network_guards.setattr(socket, "getaddrinfo", _forbid_external_calls)

    from azure.identity import AzureCliCredential
    from openai._base_client import AsyncAPIClient, SyncAPIClient

    _network_guards.setattr(AzureCliCredential, "get_token", _forbid_external_calls)
    _network_guards.setattr(AsyncAPIClient, "request", _forbid_external_calls)
    _network_guards.setattr(SyncAPIClient, "request", _forbid_external_calls)
    config.addinivalue_line("markers", "docker: opt-in direct Docker execution without LLM calls")


def pytest_unconfigure(config):
    _network_guards.undo()


def pytest_collection_modifyitems(config, items):
    if os.getenv("SAFEAGENTS_RUN_DOCKER_TESTS") != "1":
        skip = pytest.mark.skip(reason="Set SAFEAGENTS_RUN_DOCKER_TESTS=1 for direct Docker tests.")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def forbid_model_and_local_execution(monkeypatch):
    from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
    from safeagents import Team
    from safeagents.core.src.evaluation.assessment import Assessment
    from safeagents.core.src.frameworks.client_factory import ClientFactory

    monkeypatch.setattr(ClientFactory, "create_client", _forbid_external_calls)
    monkeypatch.setattr(Team, "create", _forbid_external_calls)
    monkeypatch.setattr(Assessment, "evaluate_logs", _forbid_external_calls)
    monkeypatch.setattr(LocalCommandLineCodeExecutor, "__init__", _forbid_external_calls)
    monkeypatch.setattr(LocalCommandLineCodeExecutor, "execute_code_blocks", _forbid_external_calls)
