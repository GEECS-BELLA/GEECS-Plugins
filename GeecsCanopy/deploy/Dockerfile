# The derived image: Canopy at pin.py's digest (passed as CANOPY_IMAGE; never
# build without it) plus the three GEECS packages and the DB driver. Context =
# the repository root, filtered by Dockerfile.dockerignore.
ARG CANOPY_IMAGE
FROM ${CANOPY_IMAGE}

# The image's venv (/app/.venv) has no pip; the base interpreter's pip
# installs into it with --python. The three GEECS packages go in without
# their dependency resolution (path deps); their third-party deps are the
# locked versions, named here. pydantic and PyYAML are Canopy's own.
USER root
COPY GEECS-Core /src/GEECS-Core
COPY GEECS-Schemas /src/GEECS-Schemas
COPY GeecsCanopy /src/GeecsCanopy
RUN pip --python /app/.venv/bin/python install --no-cache-dir --no-deps \
        /src/GEECS-Core /src/GEECS-Schemas /src/GeecsCanopy \
    && pip --python /app/.venv/bin/python install --no-cache-dir \
        mysql-connector-python==9.7.0 gest-api==0.1 \
    && rm -rf /src
USER appuser
