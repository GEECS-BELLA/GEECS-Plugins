"""GeecsConfigurator: one editor over every GEECS-Schemas document in the configs repository.

A FastAPI service (``geecs-configurator``, port 8800) on the kit: the
catalogue of kinds with their storage, tier and consumers, every document of
every file kind, its schema for the theme's ``form.js``, validate, diff (with
the strictest changed tier) and YAML, all read-only. Saving (commits, pull
requests) arrives with the commit path.
"""
