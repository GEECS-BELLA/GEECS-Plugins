"""GeecsConfigurator: one editor over every GEECS-Schemas document in the configs repository.

A FastAPI service (``geecs-configurator``, port 8800) on the kit: the
catalogue of kinds with their storage, tier and consumers, every document of
every file kind, its schema for the theme's ``form.js``, validate, diff (with
the strictest changed tier) and YAML; with the commit path on, saving
(operator-tier commits to main, maintainer-tier pull requests as the GitHub
App), the inbox, the gate on main, the share's fast-forward and the log.
"""
