"""Clearance labeler: the Python reference, its C++ port, and the benchmark harness."""

import os


def load_fastgjk():
    """Import the compiled fastgjk extension, with a build hint if it is missing."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        import fastgjk  # noqa: PLC0415

        return fastgjk
    except ImportError:
        pass
    import sys

    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import fastgjk

        return fastgjk
    except ImportError as e:
        raise ImportError(
            f"fastgjk extension not built. Run {os.path.join(here, 'build.sh')}"
        ) from e
