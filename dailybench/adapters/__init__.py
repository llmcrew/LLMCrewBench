"""Official upstream adapters.

An adapter runs a pinned upstream dataset with the upstream scorer and keeps
the upstream metric separate from the native ``/100`` suite. It may only add
provenance, a frozen task selection and a machine-readable verdict; it never
rewrites the upstream score into a different scale.

The first adapter is :mod:`dailybench.adapters.bigcodebench`.
"""

from __future__ import annotations
