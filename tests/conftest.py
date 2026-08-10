"""Pytest collection config for the backend unit suite.

test_grpc_client.py is a MANUAL integration script (run via its own
__main__ against a live server); its test_* functions take a `stub`
argument, which pytest misreads as a missing fixture and reports as two
collection errors — observed identically on both measurement hosts
(p5-smoke and the 2026-08-10 E-M runs). Excluding it from collection
records that classification; it does not hide a failing unit test.
"""

collect_ignore = ["test_grpc_client.py"]
