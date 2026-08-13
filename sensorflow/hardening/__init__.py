"""Production hardening layer for Sensorflow Studio.

Built from the audit in docs/hardening/audit.md. This package LAYERS
production-grade contracts, sampling, quality routing, HITL prioritization,
cache manifests, power analysis and interface seams on top of the existing
packages without modifying their public APIs. Adoption path documented in
each module.
"""
