# Maintaining host-configs with `rad-bat`

Ordinary RAJA CI sets `RAD_BAT_MODE=host-config`. Every build-and-test job
uses the stable host-config filename stored beside its `SPEC`; a missing file
is an error and never causes an implicit Spack invocation.

To regenerate the complete maintained set, start a GitLab pipeline for the
mirrored GitHub branch with:

- `RAD_BAT_MODE=spack`
- optionally `RAD_BAT_PUSH_TO_BUILDCACHE=true` to publish dependencies

Every refresh job concretizes its `SPEC`, creates an immutable project-specific
dependency installation, and publishes its generated host-config as a job
artifact. Download all artifacts into `host-configs/`, review them, and commit
them through the normal GitHub branch and pull-request workflow. The GitLab
mirror is never modified by the refresh pipeline.

RAJA can temporarily pin CAMP during refresh through two explicit declarations:

- A pipeline or temporary `.gitlab-ci.yml` change defines `CAMP_REFERENCE`.
- `.rad-bat.json` maps a non-empty value to
  `^camp@git.{CAMP_REFERENCE}=main` in `spack.spec_fragments`.

The variable is intentionally unset by default: it is a temporary testing
override, not RAJA's normal dependency policy. `rad-bat` has no built-in
knowledge of CAMP or `CAMP_REFERENCE`, and defining the variable without its
configuration entry does not change the spec. If `CAMP_REFERENCE` is unset or
empty, the fragment is omitted and Spack chooses the CAMP version. Additional
optional dependency references are represented by additional entries in the
array.

This mechanism is only for dependencies expressed in the Spack spec. A source
override such as RAJAPerf's `RAJA_REFERENCE`, which checks out `tpl/RAJA`, is
handled by that project's build adapter instead.
