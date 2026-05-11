# /// script
# requires-python = ">=3.10"
# ///
"""NIXLBench runner for Kubernetes inference pods.

Runs nixlbench (NIXL data transfer benchmark) between pods using ETCD for
worker coordination.  Measures RDMA/GPU memory transfer throughput and latency
between pod pairs.

With --install-deps, builds nixlbench from source inside pods (requires CUDA
and UCX to be present in the container image).  Otherwise expects nixlbench
to be pre-installed.

Public API:
    run_nixlbench(pods, display_names) -> dict
    ensure_nixlbench(pod_name) -> None
    install_etcd(pod_name) -> None
    build_nixlbench_from_source(pod_name) -> bool
"""

import subprocess
import sys
import threading
import time

from importlib import import_module

# Import shared utilities from the common module.
_common = None


def _get_common():
    global _common
    if _common is None:
        _common = import_module("run-tests-common")
    return _common


# Delegated to run-tests-common
def run_cmd(*args, **kwargs):
    return _get_common().run_cmd(*args, **kwargs)


def exec_in_pod(*args, **kwargs):
    return _get_common().exec_in_pod(*args, **kwargs)


# ---------------------------------------------------------------------------
# NIXLBench constants
# ---------------------------------------------------------------------------
ETCD_VER = "v3.5.21"
ETCD_DOWNLOAD_URL = (
    f"https://github.com/etcd-io/etcd/releases/download/{ETCD_VER}"
    f"/etcd-{ETCD_VER}-linux-amd64.tar.gz"
)
ETCD_PORT = 2379
NIXLBENCH_BINARY = "nixlbench"
NIXLBENCH_TIMEOUT = 600  # 10 minutes per pair
NIXL_REPO = "https://github.com/ai-dynamo/nixl.git"
NIXL_BUILD_DIR = "/tmp/nixl-build"
NIXL_INSTALL_PREFIX = "/usr/local/nixl"
NIXLBENCH_INSTALL_PREFIX = "/usr/local/nixlbench"
ETCD_CPP_REPO = "https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git"
PROTOBUF_VER = "v21.12"
GRPC_VER = "v1.46.7"

# Defaults — overridden via configure() from run-tests.py
NIXLBENCH_BACKEND = "UCX"
NIXLBENCH_SEG_TYPE = "VRAM"
NIXLBENCH_BUFFER_SIZE = "8G"


# ---------------------------------------------------------------------------
# Binary checks
# ---------------------------------------------------------------------------
def _check_binary(pod_name, binary):
    """Check if a binary is available on a pod. Returns True if found."""
    # Include nixlbench install paths in case it was built from source
    # For nixlbench, also verify shared libraries can load (ldd check)
    env_prefix = (
        f'export PATH={NIXLBENCH_INSTALL_PREFIX}/bin:{NIXL_INSTALL_PREFIX}/bin:'
        f'/usr/local/bin:$PATH && '
        f'export LD_LIBRARY_PATH={NIXLBENCH_INSTALL_PREFIX}/lib:'
        f'{NIXLBENCH_INSTALL_PREFIX}/lib64:'
        f'{NIXL_INSTALL_PREFIX}/lib64:'
        f'{NIXL_INSTALL_PREFIX}/lib64/plugins:'
        f'{NIXL_INSTALL_PREFIX}/lib/$(uname -m)-linux-gnu:'
        f'{NIXL_INSTALL_PREFIX}/lib/$(uname -m)-linux-gnu/plugins:'
        f'/opt/ucx/lib:/usr/local/lib64:/usr/local/lib:${{LD_LIBRARY_PATH:-}}'
    )
    if binary == NIXLBENCH_BINARY:
        # For nixlbench: verify both found AND shared libs resolve
        check_cmd = (
            f'{env_prefix} && '
            f'if ! command -v {binary} >/dev/null 2>&1; then echo "MISSING"; '
            f'elif ldd $(command -v {binary}) 2>&1 | grep -q "not found"; then echo "MISSING_LIBS"; '
            f'else echo "FOUND"; fi'
        )
    else:
        check_cmd = (
            f'{env_prefix} && '
            f'command -v {binary} >/dev/null 2>&1 && echo "FOUND" || echo "MISSING"'
        )
    result = exec_in_pod(
        pod_name,
        ["bash", "-c", check_cmd],
        use_debug=False,
    )
    if binary == NIXLBENCH_BINARY:
        return result.returncode == 0 and "FOUND" in result.stdout and "MISSING_LIBS" not in result.stdout
    return result.returncode == 0 and "FOUND" in result.stdout


def _check_etcd_runtime(pod_name):
    """Check if nixlbench was built with ETCD runtime support."""
    env_prefix = (
        f'export PATH={NIXLBENCH_INSTALL_PREFIX}/bin:{NIXL_INSTALL_PREFIX}/bin:'
        f'/usr/local/bin:$PATH && '
        f'export LD_LIBRARY_PATH={NIXLBENCH_INSTALL_PREFIX}/lib:'
        f'{NIXLBENCH_INSTALL_PREFIX}/lib64:'
        f'{NIXL_INSTALL_PREFIX}/lib64:{NIXL_INSTALL_PREFIX}/lib64/plugins:'
        f'/opt/ucx/lib:/usr/local/lib64:/usr/local/lib:${{LD_LIBRARY_PATH:-}}'
    )
    # Run nixlbench with --etcd_endpoints dummy to check if ETCD runtime is valid
    result = exec_in_pod(
        pod_name,
        ["bash", "-c",
         f'{env_prefix} && nixlbench --etcd_endpoints http://127.0.0.1:1 '
         f'--benchmark_group test 2>&1 | head -5'],
        timeout=10, use_debug=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    # "Invalid runtime: ETCD" means ETCD runtime was not compiled in
    return "Invalid runtime" not in output


def ensure_nixlbench(pod_name, out=None):
    """Verify nixlbench binary is available; build from source if --install-deps."""
    _out = out or print
    if _check_binary(pod_name, NIXLBENCH_BINARY):
        # Also verify ETCD runtime is available
        if _check_etcd_runtime(pod_name):
            _out(f"  nixlbench binary available on {pod_name} (with ETCD runtime).")
            return
        elif _get_common().INSTALL_DEPS:
            _out(f"  nixlbench found but lacks ETCD runtime on {pod_name}, rebuilding ...")
        else:
            _out(f"  Warning: nixlbench on {pod_name} lacks ETCD runtime.")
            _out(f"  Re-run with --install-deps (-i) to rebuild with ETCD support.")
            sys.exit(1)
    if not _get_common().INSTALL_DEPS:
        _out(f"  Error: nixlbench binary not found on {pod_name}.")
        _out(f"  Re-run with --install-deps (-i) to build from source,")
        _out(f"  or use a pod image with nixlbench pre-installed.")
        _out(f"  See https://github.com/ai-dynamo/nixl/tree/main/benchmark/nixlbench")
        sys.exit(1)
    _out(f"  nixlbench not found on {pod_name}, building from source ...")
    if not build_nixlbench_from_source(pod_name, out=_out):
        _out(f"  Error: failed to build nixlbench on {pod_name}.")
        sys.exit(1)
    if not _check_binary(pod_name, NIXLBENCH_BINARY):
        _out(f"  Error: nixlbench still not available after build on {pod_name}.")
        sys.exit(1)
    _out(f"  nixlbench built and installed successfully on {pod_name}.")


# ---------------------------------------------------------------------------
# Build from source
# ---------------------------------------------------------------------------
def _run_build_step(pod_name, cmd, label, out, timeout=600, ignore_error=False):
    """Run a build step inside the pod. Returns True on success."""
    _out = out or print
    _out(f"    [{label}] ...")
    # Ensure /usr/local/bin is on PATH (pip installs meson/ninja there)
    # Also set PKG_CONFIG_PATH for locally-built libraries
    cmd_with_path = (
        f'export PATH=/usr/local/bin:$PATH && '
        f'export PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig:${{PKG_CONFIG_PATH:-}} && '
        f'{cmd}'
    )
    result = exec_in_pod(pod_name, ["bash", "-c", cmd_with_path], timeout=timeout, use_debug=False)
    if _get_common().VERBOSE and result.stdout.strip():
        _out(result.stdout[-500:])
    if result.returncode != 0:
        if ignore_error:
            _out(f"    [{label}] warning (exit {result.returncode}, ignored)")
            return True
        _out(f"    [{label}] FAILED (exit {result.returncode})")
        stderr = result.stderr.strip()
        if stderr:
            _out(f"    stderr: {stderr[-500:]}")
        return False
    return True


def install_nixlbench_deps(pod_name, out=None):
    """Install system packages and Python build tools needed for nixlbench."""
    _out = out or print
    _out(f"  Installing build dependencies on {pod_name} ...")

    # Package groups by distro family — installed one group at a time so a
    # failure in one doesn't block the rest.
    # Install each package individually to avoid one missing package
    # blocking the rest.  RHEL 9 UBI repos have limited -devel packages.
    rpm_groups = [
        ("gcc gcc-c++ ninja-build cmake pkgconfig git python3-pip",
         "core build tools"),
        ("rdma-core-devel",
         "RDMA core development headers"),
        ("gflags-devel",
         "gflags (required by nixlbench)"),
        ("openssl-devel",
         "OpenSSL development headers (for etcd-cpp-apiv3)"),
        ("re2-devel zlib-devel",
         "re2 and zlib development headers (for gRPC)"),
    ]
    apt_groups = [
        ("build-essential ninja-build cmake pkg-config git python3-pip",
         "core build tools"),
        ("libibverbs-dev librdmacm-dev rdma-core ibverbs-utils "
         "libibumad-dev ibverbs-providers libnuma-dev",
         "RDMA/IB development headers"),
        ("libgflags-dev libaio-dev liburing-dev libz-dev pybind11-dev",
         "I/O and misc libraries"),
        ("libssl-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc "
         "libgrpc++-dev libcpprest-dev",
         "protobuf/gRPC/OpenSSL (for etcd support)"),
    ]

    # Detect package manager
    for pkg_mgr, install_cmd in [
        ("dnf", "dnf install -y"),
        ("yum", "yum install -y"),
    ]:
        check = exec_in_pod(pod_name, ["which", pkg_mgr], use_debug=False)
        if check.returncode == 0:
            _out(f"    Detected {pkg_mgr} package manager")
            for pkgs, label in rpm_groups:
                result = exec_in_pod(
                    pod_name, ["bash", "-c", f"{install_cmd} {pkgs}"],
                    timeout=300, use_debug=False,
                )
                if result.returncode != 0:
                    _out(f"    Warning: failed to install {label}: {result.stderr.strip()[:200]}")
                else:
                    stdout = (result.stdout or "").lower()
                    if "already installed" in stdout or "nothing to do" in stdout:
                        _out(f"    Skipped {label} (already installed).")
                    else:
                        _out(f"    Installed {label}.")
            break
    else:
        # Debian/Ubuntu — apt-get
        exec_in_pod(pod_name, ["apt-get", "update", "-y"], timeout=120, use_debug=False)
        for pkgs, label in apt_groups:
            result = exec_in_pod(
                pod_name,
                ["bash", "-c", f"DEBIAN_FRONTEND=noninteractive apt-get install -y {pkgs}"],
                timeout=300, use_debug=False,
            )
            if result.returncode != 0:
                _out(f"    Warning: failed to install {label}: {result.stderr.strip()[:200]}")
            else:
                stdout = (result.stdout or "").lower()
                if "already the newest" in stdout:
                    _out(f"    Skipped {label} (already installed).")
                else:
                    _out(f"    Installed {label}.")

    # Python build tools — try pip3 first, then python3 -m pip
    _out(f"    Installing Python build tools (meson, pybind11, tomlkit) ...")
    result = exec_in_pod(
        pod_name,
        ["bash", "-c",
         "pip3 install meson pybind11 tomlkit 2>/dev/null || "
         "python3 -m pip install meson pybind11 tomlkit"],
        timeout=120, use_debug=False,
    )
    if result.returncode != 0:
        _out(f"    Warning: pip install failed: {result.stderr.strip()[:200]}")
    else:
        _out(f"    Installed Python build tools.")

    _out(f"  Build dependencies installed on {pod_name}.")
    return True


def _check_lib_installed(pod_name, pkg_config_name, lib_paths):
    """Check if a library is installed via pkg-config or file existence."""
    checks = [f"pkg-config --exists {pkg_config_name} 2>/dev/null"]
    for p in lib_paths:
        checks.append(f"test -f {p}")
    result = exec_in_pod(
        pod_name,
        ["bash", "-c", " || ".join(checks)],
        use_debug=False,
    )
    return result.returncode == 0


def _clean_build_dir(pod_name, build_dir, out):
    """Clean a build directory, handling stubborn permissions."""
    _run_build_step(
        pod_name,
        f"chmod -R u+rwx {build_dir} 2>/dev/null; rm -rf {build_dir}",
        "clean", out, ignore_error=True,
    )


def build_protobuf(pod_name, out=None):
    """Build protobuf from source (required by gRPC and etcd-cpp-apiv3)."""
    _out = out or print
    if _check_lib_installed(pod_name, "protobuf",
                            ["/usr/local/lib64/libprotobuf.so",
                             "/usr/local/lib/libprotobuf.so"]):
        _out(f"    protobuf already installed.")
        return True
    _out(f"  Building protobuf on {pod_name} ...")
    build_dir = f"{NIXL_BUILD_DIR}/protobuf"
    _clean_build_dir(pod_name, build_dir, _out)
    steps = [
        (f"git clone --depth 1 --branch {PROTOBUF_VER} "
         f"https://github.com/protocolbuffers/protobuf.git {build_dir}",
         "clone protobuf", 120),
        (f"cd {build_dir} && git submodule update --init --recursive",
         "init submodules", 120),
        (f"mkdir -p {build_dir}/build && cd {build_dir}/build && "
         f"cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local "
         f"-DCMAKE_POSITION_INDEPENDENT_CODE=ON "
         f"-Dprotobuf_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON",
         "cmake configure", 120),
        (f"cd {build_dir}/build && make -j$(nproc)", "build", 600),
        (f"cd {build_dir}/build && make install && ldconfig", "install", 120),
    ]
    for cmd, label, t in steps:
        if not _run_build_step(pod_name, cmd, label, _out, timeout=t):
            _out(f"  Warning: protobuf build failed at '{label}'.")
            return False
    _out(f"  protobuf built and installed on {pod_name}.")
    return True


def build_grpc(pod_name, out=None):
    """Build gRPC from source with bundled abseil (required by etcd-cpp-apiv3)."""
    _out = out or print
    if _check_lib_installed(pod_name, "grpc++",
                            ["/usr/local/lib/libgrpc++.so",
                             "/usr/local/lib64/libgrpc++.so"]):
        _out(f"    gRPC already installed.")
        return True
    _out(f"  Building gRPC on {pod_name} (this may take several minutes) ...")
    build_dir = f"{NIXL_BUILD_DIR}/grpc"
    _clean_build_dir(pod_name, build_dir, _out)
    steps = [
        (f"git clone --depth 1 --branch {GRPC_VER} "
         f"https://github.com/grpc/grpc.git {build_dir}",
         "clone gRPC", 120),
        # Init only needed submodules: abseil and c-ares
        (f"cd {build_dir} && git submodule update --init "
         f"third_party/abseil-cpp third_party/cares/cares",
         "init submodules", 120),
        (f"mkdir -p {build_dir}/build && cd {build_dir}/build && "
         f"cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local "
         f"-DBUILD_SHARED_LIBS=ON -DgRPC_INSTALL=ON "
         f"-DgRPC_BUILD_TESTS=OFF "
         f"-DgRPC_PROTOBUF_PROVIDER=package "
         f"-DgRPC_ABSL_PROVIDER=module "
         f"-DgRPC_CARES_PROVIDER=module "
         f"-DgRPC_RE2_PROVIDER=package "
         f"-DgRPC_SSL_PROVIDER=package "
         f"-DgRPC_ZLIB_PROVIDER=package "
         f"-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
         "cmake configure", 120),
        (f"cd {build_dir}/build && make -j$(nproc)", "build", 900),
        (f"cd {build_dir}/build && cmake --install . && ldconfig", "install", 120),
        # Remove gRPC's bundled abseil pkg-config files so they don't conflict
        # with NIXL's meson subproject abseil (which is much newer).
        # gRPC itself links abseil statically via its module provider.
        (f"rm -f /usr/local/lib/pkgconfig/absl_*.pc "
         f"/usr/local/lib64/pkgconfig/absl_*.pc 2>/dev/null; "
         f"dnf remove -y abseil-cpp abseil-cpp-devel 2>/dev/null; true",
         "clean abseil pkg-config", 60),
    ]
    for cmd, label, t in steps:
        if not _run_build_step(pod_name, cmd, label, _out, timeout=t):
            _out(f"  Warning: gRPC build failed at '{label}'.")
            return False
    _out(f"  gRPC built and installed on {pod_name}.")
    return True


def build_etcd_cpp_api(pod_name, out=None):
    """Build etcd-cpp-apiv3 from source (required for nixlbench etcd runtime).

    Requires protobuf and gRPC to be installed first.  Builds them from source
    if not already present (RHEL 9 UBI repos lack these C++ packages).
    """
    _out = out or print

    # Check if already installed
    if _check_lib_installed(pod_name, "etcd-cpp-api",
                            ["/usr/local/lib64/libetcd-cpp-api-core.so",
                             "/usr/local/lib/libetcd-cpp-api-core.so"]):
        _out(f"    etcd-cpp-apiv3 already available.")
        return True

    _out(f"  Building etcd stack (protobuf -> gRPC -> etcd-cpp-apiv3) ...")

    # Step 1: Build protobuf from source
    if not build_protobuf(pod_name, out=_out):
        _out(f"  Cannot build etcd-cpp-apiv3 without protobuf.")
        return False

    # Step 2: Build gRPC from source
    if not build_grpc(pod_name, out=_out):
        _out(f"  Cannot build etcd-cpp-apiv3 without gRPC.")
        return False

    # Step 3: Build etcd-cpp-apiv3
    _out(f"  Building etcd-cpp-apiv3 on {pod_name} ...")
    build_dir = f"{NIXL_BUILD_DIR}/etcd-cpp-apiv3"
    _clean_build_dir(pod_name, build_dir, _out)
    steps = [
        (f"git clone --depth 1 {ETCD_CPP_REPO} {build_dir}",
         "clone etcd-cpp-apiv3", 120),
        (f"mkdir -p {build_dir}/build && cd {build_dir}/build && "
         f"cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON "
         f"-DBUILD_ETCD_TESTS=OFF -DBUILD_ETCD_CORE_ONLY=ON "
         f"-DCMAKE_PREFIX_PATH=/usr/local",
         "cmake configure", 120),
        (f"cd {build_dir}/build && make -j$(nproc)", "build", 300),
        (f"cd {build_dir}/build && make install && ldconfig", "install", 120),
    ]
    for cmd, label, t in steps:
        if not _run_build_step(pod_name, cmd, label, _out, timeout=t):
            _out(f"  Warning: etcd-cpp-apiv3 build failed at '{label}'.")
            _out(f"  nixlbench will be built without etcd runtime support.")
            return False

    # Create pkg-config file (meson needs it to find etcd-cpp-api)
    pkgconfig_cmd = (
        'mkdir -p /usr/local/lib64/pkgconfig && '
        'cat > /usr/local/lib64/pkgconfig/etcd-cpp-api.pc << "PKGEOF"\n'
        'prefix=/usr/local\n'
        'exec_prefix=${prefix}\n'
        'libdir=${prefix}/lib64\n'
        'includedir=${prefix}/include\n'
        '\n'
        'Name: etcd-cpp-api\n'
        'Description: etcd C++ client API (core only)\n'
        'Version: 0.15.4\n'
        'Libs: -L${libdir} -letcd-cpp-api-core\n'
        'Cflags: -I${includedir}\n'
        'PKGEOF'
    )
    _run_build_step(pod_name, pkgconfig_cmd, "pkg-config file", _out, ignore_error=True)

    _out(f"  etcd-cpp-apiv3 built and installed on {pod_name}.")
    return True


def _detect_ucx_path(pod_name):
    """Detect UCX installation path on the pod."""
    # Check common locations: /opt/ucx, /usr/local, /usr
    for path in ["/opt/ucx", "/usr/local", "/usr"]:
        result = exec_in_pod(
            pod_name,
            ["bash", "-c", f"test -f {path}/lib/libucp.so && echo FOUND"],
            use_debug=False,
        )
        if result.returncode == 0 and "FOUND" in result.stdout:
            return path
    return None


def build_nixl(pod_name, out=None):
    """Clone and build NIXL library from source (UCX plugin only)."""
    _out = out or print
    _out(f"  Building NIXL library on {pod_name} ...")

    # Detect UCX path for meson
    ucx_path = _detect_ucx_path(pod_name)
    if ucx_path:
        _out(f"    Found UCX at {ucx_path}")
    else:
        _out(f"    Warning: UCX not found, NIXL build may fail.")

    nixl_src = f"{NIXL_BUILD_DIR}/nixl"
    meson_args = (
        f"--prefix={NIXL_INSTALL_PREFIX} --buildtype=release "
        f"-Denable_plugins=UCX -Dinstall_headers=true"
    )
    if ucx_path and ucx_path not in ("/usr", "/usr/local"):
        meson_args += f" -Ducx_path={ucx_path}"

    # Clean stale directory — chmod first to fix meson subproject perms, then rm
    _run_build_step(
        pod_name,
        f"chmod -R u+rwx {nixl_src} 2>/dev/null; rm -rf {nixl_src}",
        "clean", _out, timeout=60, ignore_error=True,
    )
    steps = [
        (f"git clone --depth 1 {NIXL_REPO} {nixl_src}", "clone NIXL", 300),
        (f"cd {nixl_src} && meson setup build {meson_args}",
         "meson configure", 900),  # downloads abseil subproject
        (f"cd {nixl_src}/build && ninja", "build", 900),
        (f"cd {nixl_src}/build && ninja install", "install", 120),
    ]
    for cmd, label, step_timeout in steps:
        if not _run_build_step(pod_name, cmd, label, _out, timeout=step_timeout):
            return False

    # Configure linker to find NIXL libraries.
    # Detect actual lib dir: RHEL uses lib64, Debian uses lib/<arch>-linux-gnu
    ldconfig_cmd = (
        f'NIXL_LIBDIR=$(find {NIXL_INSTALL_PREFIX} -maxdepth 1 -name "lib*" -type d | head -1) && '
        f'echo "$NIXL_LIBDIR" > /etc/ld.so.conf.d/nixl.conf && '
        f'[ -d "$NIXL_LIBDIR/plugins" ] && echo "$NIXL_LIBDIR/plugins" >> /etc/ld.so.conf.d/nixl.conf; '
        f'ldconfig'
    )
    if not _run_build_step(pod_name, ldconfig_cmd, "ldconfig", _out):
        _out(f"    Warning: ldconfig failed, may need LD_LIBRARY_PATH at runtime.")

    # Also create symlinks in /usr/local/lib64 (RHEL) or /usr/local/lib as
    # fallback — ensures the runtime linker finds libs even without ldconfig
    symlink_cmd = (
        f'NIXL_LIBDIR=$(find {NIXL_INSTALL_PREFIX} -maxdepth 1 -name "lib*" -type d | head -1) && '
        f'SYSLIB=$([ -d /usr/local/lib64 ] && echo /usr/local/lib64 || echo /usr/local/lib) && '
        f'mkdir -p $SYSLIB && '
        f'for f in $NIXL_LIBDIR/lib*.so*; do '
        f'  [ -f "$f" ] && ln -sf "$f" $SYSLIB/$(basename "$f") || true; '
        f'done; '
        f'for f in $NIXL_LIBDIR/plugins/lib*.so*; do '
        f'  [ -f "$f" ] && ln -sf "$f" $SYSLIB/$(basename "$f") || true; '
        f'done; ldconfig'
    )
    if not _run_build_step(pod_name, symlink_cmd, "library symlinks", _out):
        _out(f"    Warning: symlink creation failed.")

    _out(f"  NIXL built and installed at {NIXL_INSTALL_PREFIX} on {pod_name}.")
    return True


def _build_nixlbench_binary(pod_name, out=None):
    """Build nixlbench binary against installed NIXL."""
    _out = out or print
    _out(f"  Building nixlbench on {pod_name} ...")

    nixlbench_src = f"{NIXL_BUILD_DIR}/nixl/benchmark/nixlbench"
    # Set PKG_CONFIG_PATH so meson finds etcd-cpp-api and other locally-built libs
    pkg_env = "export PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
    meson_args = (
        f"-Dnixl_path={NIXL_INSTALL_PREFIX}/ "
        f"-Dprefix={NIXLBENCH_INSTALL_PREFIX} "
        f"-Detcd_inc_path=/usr/local/include "
        f"-Detcd_lib_path=/usr/local/lib64 "
        f"--buildtype=release"
    )
    steps = [
        (f"{pkg_env} && cd {nixlbench_src} && meson setup build {meson_args}",
         "meson configure", 300),
        (f"cd {nixlbench_src}/build && ninja", "build", 600),
        (f"cd {nixlbench_src}/build && ninja install", "install", 120),
    ]
    for cmd, label, step_timeout in steps:
        if not _run_build_step(pod_name, cmd, label, _out, timeout=step_timeout):
            return False

    # Add to PATH and LD_LIBRARY_PATH via profile.d for this session
    env_setup = (
        f'echo "export PATH={NIXLBENCH_INSTALL_PREFIX}/bin:{NIXL_INSTALL_PREFIX}/bin:\\$PATH" '
        f'> /etc/profile.d/nixlbench.sh && '
        f'echo "export LD_LIBRARY_PATH={NIXLBENCH_INSTALL_PREFIX}/lib:'
        f'{NIXLBENCH_INSTALL_PREFIX}/lib64:'
        f'{NIXL_INSTALL_PREFIX}/lib64:{NIXL_INSTALL_PREFIX}/lib64/plugins:'
        f'{NIXL_INSTALL_PREFIX}/lib/\\$(uname -m)-linux-gnu:'
        f'{NIXL_INSTALL_PREFIX}/lib/\\$(uname -m)-linux-gnu/plugins:'
        f'/usr/local/lib64:/usr/local/lib:/opt/ucx/lib:'
        f'\\$LD_LIBRARY_PATH" >> /etc/profile.d/nixlbench.sh && '
        f'chmod +x /etc/profile.d/nixlbench.sh && '
        # Also create symlinks in /usr/local/bin for immediate availability
        f'ln -sf {NIXLBENCH_INSTALL_PREFIX}/bin/nixlbench /usr/local/bin/nixlbench'
    )
    if not _run_build_step(pod_name, env_setup, "environment setup", _out):
        _out(f"    Warning: PATH setup failed. nixlbench may not be found via command -v.")

    _out(f"  nixlbench built and installed at {NIXLBENCH_INSTALL_PREFIX} on {pod_name}.")
    return True


def build_nixlbench_from_source(pod_name, out=None):
    """Full build pipeline: deps -> etcd-cpp-api -> NIXL -> nixlbench."""
    _out = out or print
    _out(f"  === Building nixlbench from source on {pod_name} ===")

    # Ensure build directory exists
    exec_in_pod(pod_name, ["mkdir", "-p", NIXL_BUILD_DIR], use_debug=False)

    # Step 1: Install system and Python build dependencies
    if not install_nixlbench_deps(pod_name, out=_out):
        return False

    # Step 2: Build etcd-cpp-apiv3 (optional but needed for multi-node)
    etcd_ok = build_etcd_cpp_api(pod_name, out=_out)
    if not etcd_ok:
        _out(f"  Continuing without etcd-cpp-apiv3 (nixlbench etcd runtime will be disabled).")

    # Step 3: Build NIXL library
    if not build_nixl(pod_name, out=_out):
        _out(f"  NIXL build failed. Cannot continue.")
        return False

    # Step 4: Build nixlbench binary
    if not _build_nixlbench_binary(pod_name, out=_out):
        _out(f"  nixlbench build failed.")
        return False

    _out(f"  === nixlbench build complete on {pod_name} ===")
    return True


def _check_etcd(pod_name):
    """Check if etcd and etcdctl are available on a pod."""
    return _check_binary(pod_name, "etcd") and _check_binary(pod_name, "etcdctl")


def install_etcd(pod_name, out=None):
    """Download and install etcd binary on a pod."""
    _out = out or print
    if _check_etcd(pod_name):
        _out(f"  etcd already available on {pod_name}.")
        return True

    if not _get_common().INSTALL_DEPS:
        _out(f"  Error: etcd not found on {pod_name}.")
        _out(f"  Re-run with --install-deps to automatically install etcd.")
        sys.exit(1)

    _out(f"  Installing etcd {ETCD_VER} on {pod_name} ...")
    install_script = (
        f'curl -L {ETCD_DOWNLOAD_URL} -o /tmp/etcd.tar.gz'
        f' && tar xzf /tmp/etcd.tar.gz -C /tmp'
        f' && cp /tmp/etcd-{ETCD_VER}-linux-amd64/etcd /usr/local/bin/'
        f' && cp /tmp/etcd-{ETCD_VER}-linux-amd64/etcdctl /usr/local/bin/'
        f' && rm -rf /tmp/etcd*'
        f' && echo "ETCD_INSTALLED"'
    )
    result = exec_in_pod(pod_name, ["bash", "-c", install_script], timeout=120, use_debug=False)
    if result.returncode != 0 or "ETCD_INSTALLED" not in result.stdout:
        _out(f"  Error: failed to install etcd: {result.stderr.strip()[:200]}")
        return False
    _out(f"  etcd {ETCD_VER} installed on {pod_name}.")
    return True


# ---------------------------------------------------------------------------
# ETCD lifecycle
# ---------------------------------------------------------------------------
def start_etcd_server(pod_name, pod_ip, out=None):
    """Start etcd server in background on a pod. Returns Popen process."""
    _out = out or print
    _c = _get_common()

    etcd_args = [
        "etcd",
        "--data-dir=/tmp/etcd-nixlbench-data",
        f"--listen-client-urls=http://0.0.0.0:{ETCD_PORT}",
        f"--advertise-client-urls=http://{pod_ip}:{ETCD_PORT}",
        f"--listen-peer-urls=http://0.0.0.0:2380",
        f"--initial-advertise-peer-urls=http://{pod_ip}:2380",
        f"--initial-cluster=default=http://{pod_ip}:2380",
    ]
    cmd = _c._build_remote_cmd(pod_name, etcd_args)

    _out(f"  Starting etcd server on {pod_name} ({pod_ip}:{ETCD_PORT}) ...")
    if _c.VERBOSE:
        _out(f"  $ {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _c._server_procs_append(proc)

    # Wait for etcd to be ready
    time.sleep(2)
    health_result = exec_in_pod(
        pod_name,
        ["etcdctl", "endpoint", "health", f"--endpoints=http://localhost:{ETCD_PORT}"],
        use_debug=False,
    )
    if health_result.returncode != 0:
        _out(f"  Warning: etcd health check failed, waiting 3 more seconds ...")
        time.sleep(3)

    _out(f"  etcd server started on {pod_name}.")
    return proc


def stop_etcd_server(proc, out=None):
    """Stop etcd server process."""
    _out = out or print
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except OSError:
            pass
    _get_common()._server_procs_remove(proc)
    _out("  etcd server stopped.")


def cleanup_etcd_state(pod_name, etcd_endpoint, group=None, out=None):
    """Clean up etcd state after a benchmark run."""
    _out = out or print
    prefix = f"xferbench/{group}" if group else "xferbench"
    exec_in_pod(
        pod_name,
        ["etcdctl", "del", prefix, "--prefix=true", f"--endpoints={etcd_endpoint}"],
        use_debug=False,
    )
    if _get_common().VERBOSE:
        _out(f"  Cleaned etcd state: prefix={prefix}")


# ---------------------------------------------------------------------------
# NIXLBench execution
# ---------------------------------------------------------------------------
def _build_nixlbench_cmd(etcd_endpoint, benchmark_group):
    """Build full shell command for nixlbench with proper PATH/LD_LIBRARY_PATH."""
    # Convert buffer size to raw bytes (nixlbench expects uint64, not "8G")
    buffer_bytes = _get_common()._parse_size(NIXLBENCH_BUFFER_SIZE)
    nixl_args = (
        f"--etcd_endpoints {etcd_endpoint} "
        f"--benchmark_group {benchmark_group} "
        f"--backend {NIXLBENCH_BACKEND} "
        f"--initiator_seg_type {NIXLBENCH_SEG_TYPE} "
        f"--target_seg_type {NIXLBENCH_SEG_TYPE} "
        f"--op_type WRITE "
        f"--total_buffer_size {buffer_bytes}"
    )
    # Wrap in bash to set PATH/LD_LIBRARY_PATH for built-from-source installs
    # Include both lib64 (RHEL) and lib/<arch>-linux-gnu (Debian) paths
    env_setup = (
        f"export PATH={NIXLBENCH_INSTALL_PREFIX}/bin:{NIXL_INSTALL_PREFIX}/bin:"
        f"/usr/local/bin:$PATH && "
        f"export LD_LIBRARY_PATH={NIXLBENCH_INSTALL_PREFIX}/lib:"
        f"{NIXLBENCH_INSTALL_PREFIX}/lib64:"
        f"{NIXL_INSTALL_PREFIX}/lib64:"
        f"{NIXL_INSTALL_PREFIX}/lib64/plugins:"
        f"{NIXL_INSTALL_PREFIX}/lib/$(uname -m)-linux-gnu:"
        f"{NIXL_INSTALL_PREFIX}/lib/$(uname -m)-linux-gnu/plugins:"
        f"/opt/ucx/lib:/usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH"
    )
    return ["bash", "-c", f"{env_setup} && {NIXLBENCH_BINARY} {nixl_args}"]


def run_nixlbench_pair(src_pod, src_ip, dst_pod, dst_ip, etcd_endpoint, group, out=None):
    """Run nixlbench between two pods. Returns parsed metrics dict or None.

    Launches target first (Popen), then initiator (subprocess.run blocking).
    Both self-register via etcd to get ranks.
    """
    _out = out or print
    _c = _get_common()

    nixl_cmd_args = _build_nixlbench_cmd(etcd_endpoint, group)
    short_desc = f"nixlbench --backend {NIXLBENCH_BACKEND} --benchmark_group {group}"

    # Start target (background)
    target_cmd = _c._build_remote_cmd(dst_pod, nixl_cmd_args)
    _out(f"  Target: {short_desc}")
    if _c.VERBOSE:
        _out(f"  $ {' '.join(target_cmd)}")
    target_proc = subprocess.Popen(target_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _c._server_procs_append(target_proc)

    # Brief delay for target to register with etcd
    time.sleep(2)

    # Start initiator (blocking)
    initiator_cmd = _c._build_remote_cmd(src_pod, nixl_cmd_args)
    _out(f"  Initiator: {short_desc}")
    if _c.VERBOSE:
        _out(f"  $ {' '.join(initiator_cmd)}")

    try:
        result = subprocess.run(
            initiator_cmd, capture_output=True, text=True, timeout=NIXLBENCH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        _out(f"    TIMEOUT after {NIXLBENCH_TIMEOUT}s")
        result = None

    # Clean up target and capture its output
    try:
        target_proc.terminate()
    except OSError:
        pass
    try:
        target_stdout, target_stderr = target_proc.communicate(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        target_stdout, target_stderr = b"", b""
        try:
            target_proc.kill()
        except OSError:
            pass
    _c._server_procs_remove(target_proc)

    if _c.VERBOSE and target_stdout:
        target_out = target_stdout.decode("utf-8", errors="replace") if isinstance(target_stdout, bytes) else target_stdout
        _out(f"  nixlbench target output ({len(target_out)} chars):\n{target_out[-2000:]}")

    # Clean up etcd state for this group
    cleanup_etcd_state(_etcd_pod_name, etcd_endpoint, group=group, out=_out)

    if result is None or result.returncode != 0:
        stderr_msg = result.stderr.strip()[:200] if result else "timeout"
        _out(f"    FAILED: {stderr_msg}")
        # Even if initiator fails, try parsing target output (it may have results)
        if target_stdout:
            target_text = target_stdout.decode("utf-8", errors="replace") if isinstance(target_stdout, bytes) else target_stdout
            parsed = parse_nixlbench_output(target_text, out=_out)
            if parsed:
                return parsed
        return None

    # Try initiator output first, then target output
    parsed = parse_nixlbench_output(result.stdout, out=_out)
    if not parsed and target_stdout:
        target_text = target_stdout.decode("utf-8", errors="replace") if isinstance(target_stdout, bytes) else target_stdout
        parsed = parse_nixlbench_output(target_text, out=_out)
    return parsed


# Module-level state for etcd pod (set during run_nixlbench)
_etcd_pod_name = None


def parse_nixlbench_output(stdout, out=None):
    """Parse nixlbench columnar output.

    nixlbench outputs fixed-column tables.  Known column orders:
      Short (8 cols): Block Size (B), Batch Size, B/W (GB/Sec), Avg Lat. (us),
                      Avg Prep (us), P99 Prep (us), Avg Post (us), P99 Post (us)
      Long (12 cols): adds Aggregate B/W (GB/Sec), Network Util (%),
                      Avg Tx (us), P99 Tx (us)  after B/W

    Data rows are whitespace-separated numbers.  The header row (multi-word
    column names) is detected but NOT split — we use positional mapping
    directly since the column order is fixed.

    Returns dict with bw_gbs, lat_avg_us, lat_p99_us (prep) for the largest
    block-size row, or None.
    """
    _out = out or print

    if not stdout or not stdout.strip():
        _out("    No output from nixlbench")
        return None

    if _get_common().VERBOSE:
        _out(f"  nixlbench raw output ({len(stdout)} chars):\n{stdout[-3000:]}")

    lines = stdout.strip().split("\n")

    # Collect candidate data rows — lines where ALL tokens are numeric.
    # Also note whether we see a header line (to confirm it's nixlbench output).
    saw_header = False
    data_rows = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if "block size" in lower or "b/w" in lower or "gb/sec" in lower:
            saw_header = True
            continue

        # A data row: first char is a digit and all whitespace-separated
        # tokens parse as floats.
        if stripped[0].isdigit():
            parts = stripped.split()
            try:
                vals = [float(p) for p in parts]
                if len(vals) >= 3:  # need at least block_size, batch, bw
                    data_rows.append(vals)
            except ValueError:
                pass

    if not data_rows:
        _out("    Warning: could not parse nixlbench output (no data rows)")
        return None

    # Use last data row (largest block size)
    row = data_rows[-1]
    ncols = len(row)
    metrics = {}

    # 10-col format (observed):
    #   0=Block Size  1=Batch  2=B/W  3=Avg Lat  4=Avg Prep  5=P99 Prep
    #   6=Avg Post  7=P99 Post  8=Avg Tx  9=P99 Tx
    # 8-col format (short — no Avg Tx / P99 Tx):
    #   0=Block Size  1=Batch  2=B/W  3=Avg Lat  4=Avg Prep  5=P99 Prep
    #   6=Avg Post  7=P99 Post
    # 12-col format (long — adds Aggregate B/W + Network Util after B/W):
    #   0=Block Size 1=Batch 2=B/W 3=Agg B/W 4=Net Util 5=Avg Lat
    #   6=Avg Prep 7=P99 Prep 8=Avg Post 9=P99 Post 10=Avg Tx 11=P99 Tx
    if ncols >= 8 and ncols <= 10:
        metrics["bw_gbs"] = row[2]
        metrics["lat_avg_us"] = row[3]     # Avg Lat
        metrics["lat_p99_us"] = row[7]     # P99 Post (completion/ack)
    elif ncols >= 11:
        metrics["bw_gbs"] = row[2]
        metrics["lat_avg_us"] = row[5]     # Avg Lat
        metrics["lat_p99_us"] = row[9]     # P99 Post
    else:
        metrics["bw_gbs"] = row[2] if ncols > 2 else row[-1]

    if not saw_header and _get_common().VERBOSE:
        _out(f"    Note: parsed {ncols}-column data row without header confirmation")

    block_size = int(row[0]) if row[0] == int(row[0]) else row[0]
    _out(f"    Parsed: block={block_size}  B/W={metrics.get('bw_gbs', '?')} GB/s"
         f"  avg_lat={metrics.get('lat_avg_us', '?')} us"
         f"  p99_lat={metrics.get('lat_p99_us', '?')} us")

    return metrics


def _run_nixlbench_matrix(pods, display_names, etcd_endpoint):
    """Run nixlbench for all pod pairs. Returns dict of metric_name -> NxN matrix."""
    _c = _get_common()
    n = len(pods)
    test_pairs = _c._cross_role_pairs(pods)
    total_pairs = len(test_pairs)

    print(f"\n{'─' * 50}")
    print(f"  Running nixlbench — backend={NIXLBENCH_BACKEND} "
          f"seg_type={NIXLBENCH_SEG_TYPE} buffer={NIXLBENCH_BUFFER_SIZE} "
          f"— {total_pairs} pair(s)")
    print(f"{'─' * 50}")

    waves = _c._schedule_parallel_pairs(n, pairs=test_pairs)

    # Metric matrices — populated as results come in
    bw_matrix = [[None] * n for _ in range(n)]
    lat_avg_matrix = [[None] * n for _ in range(n)]
    lat_p99_matrix = [[None] * n for _ in range(n)]

    done = 0

    for wave_idx, wave in enumerate(waves):
        pair_labels = ", ".join(
            f"{display_names[pods[i][0]]}->{display_names[pods[j][0]]}"
            for i, j in wave
        )
        print(f"\n  [wave {wave_idx + 1}/{len(waves)}] "
              f"{len(wave)} pair(s) in parallel: {pair_labels}")

        buffers = []
        results_slot = [None] * len(wave)
        done_events = [threading.Event() for _ in wave]

        def _worker(slot, i, j, buf, done_evt):
            try:
                src_name, src_ip = pods[i]
                dst_name, dst_ip = pods[j]
                src_short = display_names[src_name]
                dst_short = display_names[dst_name]

                def _out(msg):
                    buf.write(msg + "\n")

                _out(f"\n    {src_short} -> {dst_short}")
                group = f"pair_{i}_{j}"
                pair_metrics = run_nixlbench_pair(
                    src_name, src_ip, dst_name, dst_ip,
                    etcd_endpoint, group, out=_out,
                )
                if pair_metrics:
                    for k, v in pair_metrics.items():
                        _out(f"    {k}: {v}")
                else:
                    _out(f"    FAIL")
                results_slot[slot] = (i, j, pair_metrics)
            except Exception as exc:
                buf.write(f"\n    ERROR: {exc}\n")
                results_slot[slot] = (i, j, None)
            finally:
                done_evt.set()

        # Launch threads for this wave
        threads = []
        for slot, (i, j) in enumerate(wave):
            buf = _c._StreamingBuffer()
            buffers.append(buf)
            t = threading.Thread(
                target=_worker,
                args=(slot, i, j, buf, done_events[slot]),
                daemon=True,
            )
            threads.append(t)
            t.start()

        # Wait and flush output in pair order
        for slot in range(len(wave)):
            done_events[slot].wait()
            buffers[slot].flush_all()

        for t in threads:
            t.join(timeout=5)

        # Store results in matrices
        for slot in range(len(wave)):
            if results_slot[slot] is not None:
                i, j, pair_metrics = results_slot[slot]
                if pair_metrics:
                    if "bw_gbs" in pair_metrics:
                        bw_matrix[i][j] = pair_metrics["bw_gbs"]
                    if "lat_avg_us" in pair_metrics:
                        lat_avg_matrix[i][j] = pair_metrics["lat_avg_us"]
                    if "lat_p99_us" in pair_metrics:
                        lat_p99_matrix[i][j] = pair_metrics["lat_p99_us"]
                done += 1

    print(f"\n  Completed {done}/{total_pairs} pair(s)")

    # Build results dict — only include metrics that have at least one value
    results = {}
    if any(bw_matrix[i][j] is not None for i in range(n) for j in range(n)):
        results["NIXLBench Write BW (GB/s)"] = bw_matrix
    if any(lat_avg_matrix[i][j] is not None for i in range(n) for j in range(n)):
        results["NIXLBench Write Latency avg (usec)"] = lat_avg_matrix
    if any(lat_p99_matrix[i][j] is not None for i in range(n) for j in range(n)):
        results["NIXLBench Write Latency P99 (usec)"] = lat_p99_matrix

    return results


def run_nixlbench(pods, display_names):
    """Run nixlbench between all pod pairs.

    1. Verify nixlbench on all pods (parallel)
    2. Install/verify etcd on pod 0
    3. Start etcd on pod 0
    4. Run nixlbench for all pairs
    5. Stop etcd
    """
    global _etcd_pod_name
    _c = _get_common()
    n = len(pods)

    print(f"\n{'=' * 60}")
    print("  NIXLBENCH (NIXL Data Transfer Benchmark)")
    print(f"{'=' * 60}")
    print(f"Settings: backend={NIXLBENCH_BACKEND}, seg_type={NIXLBENCH_SEG_TYPE}, "
          f"buffer_size={NIXLBENCH_BUFFER_SIZE}")

    # --- Step 1: Verify nixlbench on all pods in parallel ---
    print("\nEnsuring nixlbench is available on all pods (parallel) ...")
    ensure_bufs = [_c._StreamingBuffer() for _ in range(n)]
    ensure_done = [threading.Event() for _ in range(n)]

    ensure_failed = [False] * n

    def _ensure_worker(idx, pod_name, buf, done_evt):
        try:
            ensure_nixlbench(pod_name, out=lambda msg: buf.write(msg + "\n"))
        except SystemExit:
            buf.write(f"  FATAL: nixlbench not found on {pod_name}\n")
            ensure_failed[idx] = True
        except Exception as exc:
            buf.write(f"  ERROR on {pod_name}: {exc}\n")
            ensure_failed[idx] = True
        finally:
            done_evt.set()

    ensure_threads = []
    for idx, (name, _ip) in enumerate(pods):
        t = threading.Thread(
            target=_ensure_worker,
            args=(idx, name, ensure_bufs[idx], ensure_done[idx]),
            daemon=True,
        )
        ensure_threads.append(t)
        t.start()

    for idx in range(n):
        while not ensure_done[idx].is_set():
            ensure_bufs[idx].flush_new()
            ensure_done[idx].wait(timeout=0.1)
        ensure_bufs[idx].flush_all()

    for t in ensure_threads:
        t.join(timeout=5)

    if any(ensure_failed):
        failed_pods = [display_names[pods[i][0]] for i in range(n) if ensure_failed[i]]
        print(f"\n  Aborting: nixlbench not available on {', '.join(failed_pods)}")
        return {}

    # --- Step 2: Install/verify etcd on pod 0 ---
    etcd_pod_name, etcd_pod_ip = pods[0]
    _etcd_pod_name = etcd_pod_name
    print(f"\nSetting up etcd on {display_names[etcd_pod_name]} ...")
    install_etcd(etcd_pod_name)

    # --- Step 3: Start etcd server ---
    etcd_proc = start_etcd_server(etcd_pod_name, etcd_pod_ip)
    etcd_endpoint = f"http://{etcd_pod_ip}:{ETCD_PORT}"

    try:
        # --- Step 4: Run nixlbench matrix ---
        results = _run_nixlbench_matrix(pods, display_names, etcd_endpoint)
    finally:
        # --- Step 5: Stop etcd ---
        stop_etcd_server(etcd_proc)

    return results


# ---------------------------------------------------------------------------
# Standalone entry point — allows:  uv run run-tests-nixlbench.py [options]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _USAGE = """\
Usage: uv run run-tests-nixlbench.py [options]

Run NIXLBench (NIXL data transfer benchmark) between Kubernetes inference pods.

Equivalent to: run-tests.sh -t nixlbench

Options:
  --nixlbench-backend BACKEND
                        NIXLBench backend (default: UCX).
  --nixlbench-seg-type TYPE
                        Segment type for initiator/target (default: VRAM).
  --nixlbench-buffer-size SIZE
                        Total buffer size (default: 8G).
  -D, --debug-image IMAGE
                        Use ephemeral debug containers with the given image.
  -e, --explain         Show the kubectl/shell commands behind each finding.
  -h, --help            Show this help message.
  -i, --install-deps    Build nixlbench from source and install etcd
                        if missing.  Requires CUDA and UCX in the pod image.
  -l, --label SELECTOR  Label selector to discover pods
                        (default: "llm-d.ai/inferenceServing=true").
  -n, --namespace NS    Kubernetes namespace for all kubectl commands.
  -v, --verbose         Print kubectl commands as they run.
  -x, --explain-verify  Run each explain command and verify output.
                        Implies --explain.
"""
    if "-h" in sys.argv or "--help" in sys.argv:
        print(_USAGE)
        sys.exit(0)

    _c = _get_common()
    _cfg = _c._parse_common_args(extra_flags={
        ("--nixlbench-backend",):       ("_NIXLBENCH_BACKEND", True),
        ("--nixlbench-seg-type",):      ("_NIXLBENCH_SEG_TYPE", True),
        ("--nixlbench-buffer-size",):   ("_NIXLBENCH_BUFFER_SIZE", True),
    })

    # Apply nixlbench-specific config
    _raw_backend = _cfg.pop("_NIXLBENCH_BACKEND", None)
    if _raw_backend:
        NIXLBENCH_BACKEND = _raw_backend
    _raw_seg = _cfg.pop("_NIXLBENCH_SEG_TYPE", None)
    if _raw_seg:
        NIXLBENCH_SEG_TYPE = _raw_seg
    _raw_buf = _cfg.pop("_NIXLBENCH_BUFFER_SIZE", None)
    if _raw_buf:
        NIXLBENCH_BUFFER_SIZE = _raw_buf

    _c.configure(**_cfg)

    _pods, _display_names = _c._discover_and_display()
    if _c.USE_DEBUG_CONTAINER:
        print("\nCreating debug containers ...")
        _c.create_debug_containers(_pods)

    _results = run_nixlbench(_pods, _display_names)
    # Print a simple summary
    for _title, _matrix in _results.items():
        print(f"\n{_title}:")
        _header = [""] + [_display_names[p[0]] for p in _pods]
        print("  " + "\t".join(_header))
        for _i, (_name, _ip) in enumerate(_pods):
            _row = [_display_names[_name]]
            for _j in range(len(_pods)):
                _val = _matrix[_i][_j]
                _row.append(f"{_val:.2f}" if _val is not None else "-")
            print("  " + "\t".join(_row))
    _c.print_results_summary(_pods, _results, _display_names)
