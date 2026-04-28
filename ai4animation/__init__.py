# Copyright (c) Meta Platforms, Inc. and affiliates.
__version__ = "1.0.0"

import ast as _ast
import importlib as _importlib
import pathlib as _pathlib
import sys as _sys
import types as _types

_pkg_dir = _pathlib.Path(__file__).resolve().parent


def _ensure_package(pkg_name, directory):
    """Register a directory as a package in sys.modules (no __init__.py needed)."""
    if pkg_name in _sys.modules:
        return _sys.modules[pkg_name]
    pkg = _types.ModuleType(pkg_name)
    pkg.__path__ = [str(directory)]
    pkg.__package__ = pkg_name
    _sys.modules[pkg_name] = pkg
    return pkg


def _register_packages(directory, pkg_name):
    """Recursively register subdirectories as packages with lazy module loading."""
    pkg = _ensure_package(pkg_name, directory)

    # Set __all__ based on .py files for 'from pkg import *' support
    pkg.__all__ = [
        f.stem for f in sorted(directory.glob("*.py")) if f.stem != "__init__"
    ]

    # Enable lazy loading: modules are imported on first access
    def _lazy_getattr(name, _pkg_name=pkg_name):
        try:
            return _importlib.import_module(f".{name}", _pkg_name)
        except ImportError:
            raise AttributeError(f"module {_pkg_name!r} has no attribute {name!r}")

    pkg.__getattr__ = _lazy_getattr

    # Recurse into subdirectories
    for d in sorted(directory.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "__")):
            continue
        sub_pkg_name = f"{pkg_name}.{d.name}"
        _register_packages(d, sub_pkg_name)
        setattr(pkg, d.name, _sys.modules[sub_pkg_name])


# Register all subdirectories as packages with lazy module loading
# (no __init__.py files needed in subdirectories)
for _d in sorted(_pkg_dir.iterdir()):
    if not _d.is_dir() or _d.name.startswith((".", "__")):
        continue
    _register_packages(_d, f"{__name__}.{_d.name}")

# Auto-import all top-level modules (after subdirectory registration so that
# top-level modules can import from subdirectories via the lazy __getattr__)
for _f in sorted(_pkg_dir.glob("*.py")):
    if _f.stem != "__init__":
        _importlib.import_module(f".{_f.stem}", __name__)


# Auto-discover and re-export all public classes for convenience
# (e.g., ai4animation.Motion instead of ai4animation.Animation.Motion.Motion)
def _find_classes(pyfile):
    return [
        n.name
        for n in _ast.iter_child_nodes(_ast.parse(pyfile.read_text(encoding="utf-8")))
        if isinstance(n, _ast.ClassDef) and not n.name.startswith("_")
    ]


def _auto_reexport():
    pkg = _sys.modules[__name__]
    exported = {}

    # Re-export Math modules as module objects (they contain functions, not classes)
    for f in sorted((_pkg_dir / "Math").glob("*.py")):
        if f.stem != "__init__":
            exported[f.stem] = _importlib.import_module(f".Math.{f.stem}", __name__)

    # Top-level modules first — set on pkg immediately so subdirectory imports
    # that do `from ai4animation import X` resolve correctly.
    # Modules with classes: export each class. Modules without: export the module object.
    for pyfile in sorted(_pkg_dir.glob("*.py")):
        if pyfile.name == "__init__.py":
            continue
        mod = _sys.modules.get(f"{__name__}.{pyfile.stem}")
        if mod is None:
            continue
        classes = _find_classes(pyfile)
        if classes:
            for name in classes:
                if name not in exported:
                    obj = getattr(mod, name, None)
                    if obj is not None:
                        exported[name] = obj
                        setattr(pkg, name, obj)
        else:
            exported[pyfile.stem] = mod
            setattr(pkg, pyfile.stem, mod)

    # Subdirectory classes and modules
    for pyfile in sorted(_pkg_dir.rglob("*.py")):
        if pyfile.name == "__init__.py" or pyfile.parent == _pkg_dir:
            continue
        names = _find_classes(pyfile)
        if not names:
            continue
        parts = pyfile.relative_to(_pkg_dir).with_suffix("").parts
        mod = _importlib.import_module(f".{'.'.join(parts)}", __name__)
        for name in names:
            if name not in exported:
                exported[name] = getattr(mod, name)
        # Also export the module by filename when it doesn't shadow a class
        if pyfile.stem not in exported:
            exported[pyfile.stem] = mod

    for name, obj in exported.items():
        setattr(pkg, name, obj)
    return list(exported.keys())


__all__ = _auto_reexport()
