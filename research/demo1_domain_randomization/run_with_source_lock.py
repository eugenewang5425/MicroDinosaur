"""Run actuator research with a private, hash-checked copy of Demo 1 Python imports."""

import argparse
import hashlib
import importlib.abc
from importlib.machinery import PathFinder
import json
from pathlib import Path
import runpy
import sys


class SnapshotLoader(importlib.abc.Loader):
    def __init__(self, origin, payload):
        self.origin = origin
        self.payload = payload

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(compile(self.payload, self.origin, "exec"), module.__dict__)


class SnapshotFinder(importlib.abc.MetaPathFinder):
    def __init__(self, study, bundle):
        self.study = study
        self.manifest = json.loads((bundle / "manifest.json").read_bytes())
        self.sources = {}
        for name, expected in self.manifest.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unexpected source lock path: {name}")
            payload = (bundle / "files" / relative).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Source lock hash mismatch: {name}")
            self.sources[relative.as_posix()] = payload

    def relative_native_path(self, origin):
        if origin:
            path = Path(origin).resolve()
            if path.suffix == ".py" and path.is_relative_to(self.study):
                return path.relative_to(self.study).as_posix()
        return None

    def find_spec(self, fullname, search=None, target=None):
        spec = PathFinder.find_spec(fullname, search, target)
        name = self.relative_native_path(spec.origin if spec else None)
        if name is None:
            return None
        if name not in self.sources:
            raise ImportError(f"Native source is outside the frozen manifest: {name}")
        spec.loader = SnapshotLoader(spec.origin, self.sources[name])
        return spec

    def verify_campaign(self, campaign, study):
        state = {}
        for module in tuple(sys.modules.values()):
            name = self.relative_native_path(getattr(module, "__file__", None))
            if name is not None:
                if (not isinstance(getattr(module, "__loader__", None), SnapshotLoader)
                        and Path(module.__file__).read_bytes() != self.sources.get(name)):
                    raise ValueError(f"Native source was loaded before its snapshot: {name}")
                state[name] = self.manifest[name]
        path = campaign.output / "native_sources.json"
        if path.exists():
            if json.loads(path.read_bytes()) != state:
                raise ValueError("The loaded Demo 1 snapshot differs from the previous run")
        else:
            import continue_training

            continue_training.replace_json(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    entry = args.entry.resolve()
    if entry.parent != Path(__file__).resolve().parent:
        raise ValueError("The training entry point must be in the same research directory")
    sys.path.insert(0, str(entry.parent))
    finder = SnapshotFinder(args.study.resolve(), args.source_lock)
    sys.meta_path.insert(0, finder)
    import continue_training

    def verify(campaign, study):
        finder.verify_campaign(campaign, study)

    continue_training.Campaign.verify_imports = verify
    sys.argv = [str(entry), "--study", str(args.study.resolve()), *remaining]
    if entry.name == "continue_training.py":
        continue_training.main()
    else:
        runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
