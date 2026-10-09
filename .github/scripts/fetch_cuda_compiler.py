"""Unpacks the parts of a CUDA toolkit that build Shorokoo's CUDA operators library (nvcc, the
CUDA headers and the static CUDA runtime) from NVIDIA's redistributable archives into one folder,
each archive checked against the SHA-256 this script records for it. Nothing is installed: point
CMake at the folder with CUDAToolkit_ROOT.

    python fetch_cuda_compiler.py <CUDA release, e.g. 13.0.3> <folder>

Runs on Windows and on Linux, for x86-64. A release this script records no archives for is refused:
add its archives' versions and SHA-256s below (each archive's, as NVIDIA's redistrib_<release>.json
gives it, checked against the archive itself) to build with it.
"""

import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"

# Per CUDA release and platform, each archive the compiler takes: its component, its version and
# its SHA-256.
ARCHIVES = {
    "13.0.3": {
        "windows-x86_64": [
            ("cuda_nvcc", "13.0.88", "61760e24a44937c6c29a20ddea69067b4d3860ccd6a9a0564573f840d96e6c5f"),
            ("cuda_crt", "13.0.88", "ff413c89dbd96ed3b3dcfac923437de78acea32fb90309277197c8a127588112"),
            ("libnvvm", "13.0.88", "6dda4a82d22a2c173a65c66b4b4c933e3424a56ce845e2864bdca9cc66e6c524"),
            ("cuda_cudart", "13.0.96", "a2ed875f9997aa24904fb70cc9db3acd9308433cde99bc8e63ec1271c9da31b4"),
            ("cuda_cccl", "13.0.85", "f4f5187f7976d8057d76e1e04e72c8b7d0a2a22793107a671ad709b1d8d3408d"),
        ],
        "linux-x86_64": [
            ("cuda_nvcc", "13.0.88", "48e35be3cfbf4b4fbc16828eaec8a7048ee789403049dc409f7b643d6259cf7a"),
            ("cuda_crt", "13.0.88", "5a3279a049ffc1cdb951c44cb95206acfdde9e9ae5e87825fc18d7e4a6878bb0"),
            ("libnvvm", "13.0.88", "17ef1665b63670887eeba7d908da5669fa8c66bb73b5b4c1367f49929c086353"),
            ("cuda_cudart", "13.0.96", "25b8071951baba827be1580b841d363464f6ee6c39f48d33a81646f90cc95ed1"),
            ("cuda_cccl", "13.0.85", "ed845eae8c1767706b6ee91e40c608a03f6f633551a849b63f7346d32d73ee60"),
        ],
    },
}

# Seconds a download may wait on the server for any one read before it fails.
TIMEOUT = 120


def fetch(url: str, path: str) -> None:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response, open(path, "wb") as out:
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
            elif entry.islnk():
                linked = entry.linkname.split("/", 1)[1] if "/" in entry.linkname else ""
                shutil.copy2(inside(root, linked), target)
            elif entry.isfile():
                with t.extractfile(entry) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
                os.chmod(target, entry.mode & 0o755)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    release, root = sys.argv[1], os.path.abspath(sys.argv[2])
    platform = "windows-x86_64" if os.name == "nt" else "linux-x86_64"
    archives = ARCHIVES.get(release, {}).get(platform)
    if archives is None:
        raise SystemExit(f"This script records no CUDA {release} archives for {platform}; it records {', '.join(ARCHIVES)}.")
    os.makedirs(root, exist_ok=True)
    extension = "zip" if os.name == "nt" else "tar.xz"
    with tempfile.TemporaryDirectory() as downloads:
        for component, version, expected in archives:
            relative = f"{component}/{platform}/{component}-{platform}-{version}-archive.{extension}"
            archive = os.path.join(downloads, os.path.basename(relative))
            fetch(REDIST + relative, archive)
            if sha256(archive) != expected:
                raise SystemExit(f"{relative} does not have the SHA-256 recorded for it.")
            unpack(archive, root)
            print(f"{component} {version}")
    # The archives put Linux's libraries under lib/, where nvcc and CMake look under lib64/ too.
    if os.name != "nt" and not os.path.exists(os.path.join(root, "lib64")):
        os.symlink("lib", os.path.join(root, "lib64"))
    print(f"CUDA {release} compiler in {root}")


if __name__ == "__main__":
    main()
