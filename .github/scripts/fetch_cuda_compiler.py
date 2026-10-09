"""Unpacks the parts of a CUDA toolkit that build Shorokoo's CUDA operators library (nvcc, the
CUDA headers and the static CUDA runtime) from NVIDIA's redistributable archives into one folder,
each archive checked against the SHA-256 NVIDIA's manifest for the release gives it. Nothing is
installed: point CMake at the folder with CUDAToolkit_ROOT.

    python fetch_cuda_compiler.py <CUDA release, e.g. 13.0.3> <folder>

Runs on Windows and on Linux, for x86-64.
"""

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = ["cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cudart", "cuda_cccl"]


def fetch(url: str, path: str) -> None:
    with urllib.request.urlopen(url) as response, open(path, "wb") as out:
        shutil.copyfileobj(response, out)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(root: str, relative: str) -> str:
    target = os.path.realpath(os.path.join(root, relative))
    if os.path.commonpath([target, os.path.realpath(root)]) != os.path.realpath(root):
        raise SystemExit(f"An archive entry points outside {root}: {relative}")
    return target


def unpack(archive: str, root: str) -> None:
    """Every regular file and link of the archive, less its top folder, under root."""
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                relative = entry.filename.split("/", 1)[1] if "/" in entry.filename else ""
                if not relative or entry.is_dir():
                    continue
                target = inside(root, relative)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(entry) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
        return
    with tarfile.open(archive) as t:
        for entry in t.getmembers():
            relative = entry.name.split("/", 1)[1] if "/" in entry.name else ""
            if not relative or entry.isdir():
                continue
            target = inside(root, relative)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if entry.issym():
                if os.path.isabs(entry.linkname):
                    raise SystemExit(f"An archive link points outside {root}: {entry.name}")
                inside(os.path.dirname(target), entry.linkname)
                if os.path.lexists(target):
                    os.remove(target)
                os.symlink(entry.linkname, target)
            elif entry.isfile():
                with t.extractfile(entry) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
                os.chmod(target, entry.mode & 0o755)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    release, root = sys.argv[1], os.path.abspath(sys.argv[2])
    platform = "windows-x86_64" if os.name == "nt" else "linux-x86_64"
    with urllib.request.urlopen(f"{REDIST}redistrib_{release}.json") as response:
        manifest = json.load(response)
    os.makedirs(root, exist_ok=True)
    with tempfile.TemporaryDirectory() as downloads:
        for component in COMPONENTS:
            entry = manifest.get(component, {}).get(platform)
            if entry is None:
                raise SystemExit(f"CUDA {release} has no {component} for {platform}.")
            archive = os.path.join(downloads, os.path.basename(entry["relative_path"]))
            fetch(REDIST + entry["relative_path"], archive)
            if sha256(archive) != entry["sha256"]:
                raise SystemExit(f"{archive} does not have the SHA-256 the CUDA {release} manifest gives it.")
            unpack(archive, root)
            print(f"{component} {manifest[component]['version']}")
    # The archives put Linux's libraries under lib/, where nvcc and CMake look under lib64/ too.
    if os.name != "nt" and not os.path.exists(os.path.join(root, "lib64")):
        os.symlink("lib", os.path.join(root, "lib64"))
    print(f"CUDA {release} compiler in {root}")


if __name__ == "__main__":
    main()
