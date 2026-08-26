.. -*- mode: rst -*-

2.1.0.dev
=========

HIGHLIGHTS
----------

NEW
---

- :bdg-success:`Enhancement` Add the dmriprep workflow.
- :bdg-success:`Enhancement` Add the morphologist workflow.

Fixes
-----

- :bdg-danger:`Deprecation` Fix the containers that are using `mri_synthstrip`.
- :bdg-danger:`Deprecation` The run mapping file has been moved to avoid
  conflicts with FreeSurfer.

Enhancements
------------

- :bdg-success:`Enhancement` Add signature hook.
- :bdg-success:`Enhancement` Add live comand line monitoring support.
- :bdg-success:`Enhancement` Support multi-modality in Quasi-Raw workflow.
- :bdg-success:`Enhancement` Add `quick` mode in Quasi-Raw workflow.
- :bdg-success:`Enhancement` Support multi-modality in Deface workflow.

Changes
-------

- :bdg-danger:`Deprecation` Optimize the Quasi-Raw workflow steps.


2.0.0
=====

HIGHLIGHTS
----------

his is a major release featuring significant changes to both the API and
CLI. Please review the modifications below.

NEW
---

- :bdg-success:`Doc` Create doc with `furo <https://github.com/pradyunsg/furo>`_.
- :bdg-success:`Enhancement` Workflows generate a report file.
- :bdg-success:`Enhancement` Anonymize workflow outputs.
- :bdg-success:`Enhancement` workflows generate BIDS-compliant organization.
- :bdg-success:`Enhancement` New workflows can generate HTML reporting.
- :bdg-success:`Datasets` Toy datasets have been added to test the module.
- :bdg-success:`Enhancement` Quasi-RAW preprocessing compute the brain mask
  using `mri_synthstrip <https://surfer.nmr.mgh.harvard.edu/docs/synthstrip>`_.
- :bdg-success:`Enhancement` A build‑and‑test container strategy has been
  released.
- :bdg-success:`Enhancement` The software version has been updated, and the
  corresponding workflow has been revised accordingly.
- :bdg-success:`Enhancement` A consolidated quality‑check HTML report can
  now be generated directly from the CLI.

Enhancements
------------

- :bdg-success:`API` A `keep_intermediate` argument has been added to all
  workflows to retain intermediate results; useful for debugging.
- :bdg-success:`Doc` A user guide is now available.
- :bdg-success:`Enhancement` Each workflow now has its own dedicated container,
  enabling easier upgrades without impacting other workflows.”
- :bdg-success:`Enhancement` The brainprep command scope has been limited to
  the contents of considered container.
- :bdg-success:`Enhancement` Each workflow now includes its own dedicated
  automatic quality‑check procedure.

Changes
-------

- :bdg-danger:`Deprecation` Remove the old dmriprep workflow and tbss workflow
  (will be updated in v2.1.0).
- :bdg-success:`API` Building blocks are now grouped in an interfaces
  submodule.
- :bdg-danger:`Deprecation` Path custom white matter mask to `recon-all` has
  been deprecated.
- :bdg-success:`Enhancement` Use '--no-annot' in surfreg to avoid using the
  annotation (aparc) to help with the registration. This was described to
  create some artifacts on the edge of the medial wall
  (https://surfer.nmr.mgh.harvard.edu/fswiki/Xhemi).
- :bdg-danger:`Deprecation` Workflows have been renamed.


0.0.2
=====

**Released September 2022**

HIGHLIGHTS
----------

- :bdg-success:`API` This release includes different workflows to process
  antomical, functional and diffusion MR images.
- :bdg-success:`API` All workflows are integrated in a dedicated container
  to enforce reproducible research.

NEW
---

- :bdg-success:`API` The following workflows are released:

* fsreconall
* fsreconall-summary
* fsreconall-qc
* fsreconall-longitudinal
* cat12vbm
* cat12vbm-qc
* cat12vbm-roi
* quasiraw
* quasiraw-qc
* fmriprep
* fmriprep-conn
* mriqc
* mriqc-summary
* deface
* tbss-preproc
* tbss
* dmriprep

Fixes
-----

Enhancements
------------

Changes
-------

