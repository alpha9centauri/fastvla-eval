"""Load fastvla/posttrain/config.env, so the shell scripts and the Python scripts in this
directory read one set of paths rather than two.

The copied scripts had the box's paths compiled into them; they now take them from here.
Override the file with POSTTRAIN_CONFIG, or any single value with a real environment
variable, which always wins.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "config.env")


def load(path=None):
    """Return the config as a dict. Real environment variables take precedence."""
    path = path or os.environ.get("POSTTRAIN_CONFIG") or DEFAULT
    out = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    out.update({k: v for k, v in os.environ.items() if k in out and v})
    return out


def get(key, default=None):
    return load().get(key, default)
