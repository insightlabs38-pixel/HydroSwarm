# SUB-3: Docker execution is not possible in this sandbox (root-cause confirmed)

Date: 2026-08-10

## Finding

`docker` (29.7.2) and `docker buildx` (0.36.1) are installed in this sandbox, per the
session setup, but **no `docker build` or `docker run` can succeed here** -- not for
arm64 (native to this aarch64 host), and not for amd64 (which would additionally need
QEMU emulation). This is a hard capability restriction of the sandbox container itself,
not a Dockerfile, network, or configuration problem.

## Root cause

This session is itself running inside a container (`/.dockerenv` present) with
`CAP_SYS_ADMIN` explicitly removed from the capability bounding set
(`capsh --print` → `!cap_sys_admin` in the IAB/dropped list). Confirmed directly,
independent of Docker's own error messages:

```
$ unshare --mount --uts echo test
unshare: unshare failed: Operation not permitted

$ unshare --user --map-root-user --mount echo test
unshare: unshare failed: Operation not permitted
```

The second test used `CLONE_NEWUSER` (a *user* namespace, not a privileged one) and
still failed even though `/proc/sys/kernel/unprivileged_userns_clone` reports `1`
(unprivileged user namespaces are kernel-enabled). That means the block is enforced by
this sandbox's own seccomp/LSM policy on the `unshare`/`clone` syscalls, not by a
capability or kernel-config gap that could be worked around from inside the container.

Consequences observed while diagnosing:
- `dockerd` itself starts, but only with `--iptables=false --bridge=none` (its default
  bridge-network setup needs `iptables`/`nftables` NAT chain creation, which needs real
  root network privileges this sandbox also does not grant -- `Permission denied (you
  must be root)` even though `id` reports uid 0).
- `docker build` (BuildKit): fails at the very first step (reading the Dockerfile)
  with a bind-mount error (`operation not permitted`) -- BuildKit's snapshotter needs
  to bind-mount snapshot layers, which needs `CAP_SYS_ADMIN`.
- `docker build` (legacy `DOCKER_BUILDKIT=0` builder): fails with
  `Error response from daemon: unshare: operation not permitted` -- confirms the
  daemon-side container/build step itself cannot create the namespaces it needs.
- `docker run --privileged tonistiigi/binfmt --install all` (the standard way to
  register QEMU interpreters for cross-arch emulation): fails to even extract the
  image layers (`failed to mount ... operation not permitted`) for the same reason.
- `mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc`: `permission denied`.

None of these are fixable from inside the sandbox without a capability this container
was not granted. `sudo` is not available/authorized for this session either way.

## Disposition

Per this session's operating instructions ("If docker is having key issues, skip those
sections and proceed until the gate... record the issue and continue with the next
independent task"):

- **SUB-3's Docker/release deliverables are implemented** (release compose,
  `.github/workflows/release.yml` with `docker/setup-qemu-action` +
  `docker/setup-buildx-action` for real multiarch builds, runtime zip script, release
  manifest generator) -- these steps work correctly on a real GitHub Actions runner
  (which grants the privileges this sandbox withholds) or on a developer machine with
  normal Docker Desktop/Engine privileges.
- **Docker/release packaging is explicitly NOT marked complete or "Tested" per the
  submission-readiness gate** ("do not mark Docker/release packaging complete until a
  real amd64 build/run and an arm64 build/smoke have executed successfully"). That
  execution proof could not be produced in this environment. It must be produced by:
  1. pushing `feature/submission-readiness-v1` (or its eventual PR) and letting
     `.github/workflows/release.yml`'s build job run on a real GitHub Actions runner
     (has full container privileges), or
  2. running `./setup_hydroswarm_linux.sh`-equivalent Docker verification manually on
     any machine with normal (non-sandboxed) Docker privileges:
     `docker build -t hydroswarm:smoke . && docker run --rm hydroswarm:smoke hydroswarm self-test --human`
- Continuing with the remaining independent SUB-3 sub-tasks (compose files, workflow,
  manifest/checksum generation, runtime zip) and the rest of the phase list; Docker
  execution verification remains an explicitly open item for a human or a real CI run.
