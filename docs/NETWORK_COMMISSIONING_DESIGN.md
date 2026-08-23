# Network commissioning design

Commissioning is the deliberate bridge between a parseable EPANET network and
one where calibrated learned evidence may support governed deterministic
planning. It is not implied by a successful upload.

1. Store immutable uploaded `.inp` bytes and validate parse/simulation.
2. Derive canonical structural identity and a separate mutable hydraulic-state
   provenance record.
3. Characterize nominal hydraulics and permitted operating-condition classes.
4. Generate governed signature libraries and synthetic commissioning scenarios.
5. Fit conformal calibration only on the designated commissioning calibration
   split; record coverage, candidate size, network/condition groups, and
   artifact checksums.
6. Generate the OOD topology/state references and verify unknown topology
   remains inapplicable.
7. Publish a readiness record with separate compatibility, applicability, and
   operational-readiness fields.

The prototype is intentionally a design only. It must preserve the same
canonical `.inp` identity semantics used by serving and never make an
uncommissioned uploaded network borrow calibration by display name.
