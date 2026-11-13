# GeecsConfigurator

One editor over every GEECS-Schemas document in the configs repository: the
catalogue of every registered kind (where it is kept, its tier, who reads it),
every document of every file kind drawn from its JSON Schema by the theme's
`form.js`, and validate, diff (with the strictest changed tier) and YAML on an
unsaved edit. **Read-only for now**: saving (a commit, or a pull request for
the maintainer tier) arrives with the commit path.

## Run

```bash
poetry install
GEECS_CONFIGS_ROOT=/path/to/GEECS-Plugins-Configs GEECS_EXPERIMENT=<experiment> poetry run geecs-configurator
```

Settings (`geecs_configurator/settings.py`): `GEECS_CONFIGURATOR_CONFIGS_ROOT`
(else `GEECS_CONFIGS_ROOT`, else `config.ini`'s `scanner_config_root_path`),
`GEECS_CONFIGURATOR_EXPERIMENT` (else `GEECS_EXPERIMENT`; any experiment in the
root can be browsed), `_BIND` (127.0.0.1), `_PORT` (8800), `_CORS_ORIGINS`
(the surfaces that embed the editor), `_TITLE`.

## API (read-only)

| Route | |
|---|---|
| `GET /api/kinds` | every registered kind: storage (`file`, `code`, `not-a-file` with the reason), tier, path pattern, readers, dry run |
| `GET /api/experiments` | the experiments in the configs root |
| `GET /api/schema/{kind}` | the model's JSON Schema, tiers included |
| `GET /api/documents/{kind}?experiment=` | the documents, valid or not |
| `GET /api/documents/{kind}/{id}?experiment=` | one document with its etag, errors and tier |
| `POST /api/validate/{kind}` | `{document, id?}` → errors at pydantic locations |
| `POST /api/diff/{kind}/{id}?experiment=` | `{document}` → the changes against the stored one, each with its tier, and the strictest |
| `POST /api/yaml` | `{document}` → its YAML |

## Deploy

A unit template (`deploy/geecs-configurator.service`) rendered from site.env
by `deploy/render_units.sh`; `deploy/bootstrap_host.sh --only configurator`
clones `configurator-checkout`, installs and prints the enable step. Site keys:
`GEECS_CONFIGURATOR_PORT` (unset: not deployed), `_BIND`, `_CORS_ORIGINS`,
`_EXTRA_ARGS`. The Control Room's launcher links to it through
`GEECS_CONTROL_ROOM_CONFIG_EDITOR_URL`.

The commit path (the GitHub App once per org; per services box its key, the
experiments it owns, its state directory, the share's clone, the trial on a
throwaway branch) is [DEPLOYMENT.md](DEPLOYMENT.md).
