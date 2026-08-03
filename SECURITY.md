# Security Policy

Report vulnerabilities privately to the repository owner rather than opening a public
issue. Include the affected version, reproduction steps, impact, and any suggested fix.

HydroSwarm accepts only bounded, content-validated EPANET `.inp` uploads into controlled
local storage. The service binds to loopback, restricts browser origins, rejects external
file references and traversal, performs no URL fetching, and never executes input as shell
commands. Exact simulator verification and operator approval are mandatory safety gates.

Supported security updates target the current `main` branch. This hackathon/research build
does not provide production authentication, process sandboxing, actuator integration, or a
utility deployment safety guarantee.

