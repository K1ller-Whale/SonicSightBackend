"""StreamProcess model-selection branch, tested without any model loaded."""

import asyncio
import os
import sys

import grpc
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from grpc_server import SonicSightServicer


class FakeAbort(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self.code = code
        self.details = details


class FakeContext:
    def __init__(self, metadata):
        self._metadata = metadata

    def invocation_metadata(self):
        return list(self._metadata.items())

    async def abort(self, code, details):
        raise FakeAbort(code, details)


async def _empty_iterator():
    return
    yield  # pragma: no cover


def _drive(metadata):
    async def run():
        servicer = SonicSightServicer()
        gen = servicer.StreamProcess(_empty_iterator(), FakeContext(metadata))
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            return None
    return asyncio.run(run())


def test_unknown_model_rejected_with_failed_precondition():
    with pytest.raises(FakeAbort) as exc:
        _drive({"sonicsight-model": "does-not-exist"})
    assert exc.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "Unknown model" in exc.value.details
    assert "sonicsight" in exc.value.details  # advertises what IS available


def test_known_model_without_loaded_checkpoint_rejected():
    # In this test environment no checkpoint is loaded, so even the default
    # model must be refused with FAILED_PRECONDITION rather than crash later.
    from inference import inference

    assert not inference.is_loaded  # precondition of this test environment
    with pytest.raises(FakeAbort) as exc:
        _drive({"sonicsight-model": "sonicsight"})
    assert exc.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "not loaded" in exc.value.details


def test_absent_metadata_defaults_to_sonicsight():
    # No metadata key -> default model. Same not-loaded rejection here, but
    # the message must name the DEFAULT model, proving the fallback happened.
    with pytest.raises(FakeAbort) as exc:
        _drive({})
    assert "'sonicsight'" in exc.value.details
