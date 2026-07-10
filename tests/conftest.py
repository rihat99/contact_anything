"""Shared pytest configuration for the contact test suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: GPU / real-checkpoint integration test (deselect with -m 'not slow')",
    )
