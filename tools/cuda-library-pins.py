"""Writes cuda-libraries.txt beside each CUDA environment lock: the cuDNN and cuBLAS releases every
CUDA backend of a process shares, as the PyPI wheel each comes from and the files of it a backend
loads, with the SHA-256 the wheel's RECORD gives each file.

The releases are the ones the lock's PyTorch carries: on Linux the nvidia-cudnn-cu13 and
nvidia-cublas wheels the lock pins, which PyTorch depends on; on Windows the wheels whose files
PyTorch's own wheel bundles byte for byte in torch/lib, found by comparing RECORDs. Run it after
recompiling a lock:

    python tools/cuda-library-pins.py

Only the end of each wheel is read (its central directory and RECORD), with HTTP range requests.
"""

import io
import json
import os
import re
import sys
import urllib.request
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "src", "Backend", "Python", "Shorokoo.PythonHost", "Environments", "cu13")
AGENT = {"User-Agent": "shorokoo-cuda-library-pins"}
CUDA_MAJOR = "13"
LIBRARIES = [("cudnn", "nvidia-cudnn-cu13"), ("cublas", "nvidia-cublas")]


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0", **AGENT})) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size or len(buffer) == 0:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", **AGENT})
        with urllib.request.urlopen(request) as r:
            data = r.read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def record(url):
    """The wheel's RECORD, as (path, sha256, size) rows."""
    archive = zipfile.ZipFile(io.BufferedReader(RangeFile(url), 1 << 16))
    name = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
    rows = []
    for line in archive.read(name).decode().splitlines():
        path, digest, size = line.rsplit(",", 2)
        if digest.startswith("sha256="):
            rows.append((path, digest[len("sha256="):], int(size)))
    return rows


def pypi_wheel(package, version, platform_tag):
    with urllib.request.urlopen(urllib.request.Request(f"https://pypi.org/pypi/{package}/{version}/json", headers=AGENT)) as r:
        files = json.load(r)["urls"]
    wheel = next(f for f in files if f["filename"].endswith(platform_tag + ".whl"))
    return wheel["url"], wheel["digests"]["sha256"], wheel["size"]


def releases(package):
    with urllib.request.urlopen(urllib.request.Request(f"https://pypi.org/pypi/{package}/json", headers=AGENT)) as r:
        return list(json.load(r)["releases"])


def lock_version(lock, package):
    match = re.search(rf"^{re.escape(package)}==(\S+)", lock, re.MULTILINE)
    return match.group(1) if match else None


def loaded(library, path):
    """Whether a backend loads this file of the library's wheel: its shared libraries, and of
    cuBLAS's only cuBLAS and cuBLASLt."""
    name = path.rsplit("/", 1)[-1]
    if not re.search(r"\.dll$|\.so\.\d+$", name):
        return False
    return library != "cublas" or re.match(r"(lib)?cublas(Lt)?(64_\d+\.dll|\.so\.\d+)$", name) is not None


def load_order(library, path):
    """Dependencies first: what the others import by name loads before them."""
    name = path.rsplit("/", 1)[-1].lower()
    if library == "cublas":
        return (0 if "cublaslt" in name else 1, name)
    rank = 0 if "graph" in name else 1 if "_ops" in name else 3 if re.match(r"(lib)?cudnn(64_9\.dll|\.so\.9)$", name) else 2
    return (rank, name)


def torch_wheel(lock, index_url):
    version = lock_version(lock, "torch")
    page = urllib.request.urlopen(urllib.request.Request(index_url.rstrip("/") + "/torch/", headers=AGENT)).read().decode()
    wanted = f"torch-{version}-cp312-cp312-win_amd64.whl"
    for href, text in re.findall(r'href="([^"]+)"[^>]*>([^<]+)</a>', page):
        if text == wanted:
            url, _, digest = href.replace("&amp;", "&").partition("#sha256=")
            return version, url, digest
    raise SystemExit(f"{wanted} is not on {index_url}")


def write(platform, entries, bundled=None):
    lines = [
        f"# The NVIDIA libraries every CUDA backend of a process shares on {platform}, one release each, built",
        f"# for CUDA {CUDA_MAJOR}: the PyPI wheel each comes from, then the files of it a backend loads, in the order",
        "# they are loaded, each with the SHA-256 (unpadded base64url) and size the wheel's RECORD gives it.",
    ]
    if bundled:
        lines.append("# Last, the PyTorch wheel the lock beside this file pins, and its copies of those files.")
    lines += ["# Written by tools/cuda-library-pins.py from that lock: run it again whenever the lock changes.",
              "#",
              "# library <name> <version> <cuda major> <package> <wheel url> <wheel sha256> <wheel size>",
              "# file <library> <path in the wheel> <sha256> <size>"]
    if bundled:
        lines.append("# bundled <package>==<version> <wheel sha256>")
        lines.append("# bundles <path in the wheel> <sha256> <size>")
    for library, version, package, url, digest, size, files in entries:
        lines.append(f"library {library} {version} {CUDA_MAJOR} {package} {url} {digest} {size}")
        lines += [f"file {library} {path} {sha} {length}" for path, sha, length in files]
    if bundled:
        version, digest, files = bundled
        lines.append(f"bundled torch=={version} {digest}")
        lines += [f"bundles {path} {sha} {length}" for path, sha, length in files]
    with open(os.path.join(ROOT, platform, "cuda-libraries.txt"), "w", newline="\n") as out:
        out.write("\n".join(lines) + "\n")


def linux():
    lock = open(os.path.join(ROOT, "linux-x64", "requirements.txt")).read()
    entries = []
    for library, package in LIBRARIES:
        version = lock_version(lock, package)
        url, digest, size = pypi_wheel(package, version, "manylinux_2_27_x86_64")
        files = sorted((row for row in record(url) if loaded(library, row[0])), key=lambda row: load_order(library, row[0]))
        entries.append((library, version, package, url, digest, size, files))
    write("linux-x64", entries)


def windows():
    folder = os.path.join(ROOT, "win-x64")
    lock = open(os.path.join(folder, "requirements.txt")).read()
    index = open(os.path.join(folder, "uv-index.txt")).read().split()
    version, torch_url, torch_digest = torch_wheel(lock, index[index.index("--index-url") + 1])
    bundled = {path.rsplit("/", 1)[-1]: (path, sha, size) for path, sha, size in record(torch_url) if path.startswith("torch/lib/")}
    entries, bundles = [], []
    for library, package in LIBRARIES:
        for release in sorted(releases(package), key=lambda v: [int(p) if p.isdigit() else 0 for p in v.split(".")], reverse=True):
            try:
                url, digest, size = pypi_wheel(package, release, "win_amd64")
            except StopIteration:
                continue
            files = sorted((row for row in record(url) if loaded(library, row[0])), key=lambda row: load_order(library, row[0]))
            names = [path.rsplit("/", 1)[-1] for path, _, _ in files]
            if all(name in bundled and bundled[name][1] == sha for name, (_, sha, _) in zip(names, files)):
                entries.append((library, release, package, url, digest, size, files))
                bundles += [bundled[name] for name in names]
                break
        else:
            raise SystemExit(f"No {package} release on PyPI ships the {library} files torch {version} bundles")
    write("win-x64", entries, (version, torch_digest, bundles))


if __name__ == "__main__":
    for platform in sys.argv[1:] or ["linux-x64", "win-x64"]:
        {"linux-x64": linux, "win-x64": windows}[platform]()
