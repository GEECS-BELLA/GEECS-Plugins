# The derived GEECS Canopy image: SLAC's image, pinned by digest, plus this
# package, GEECS-Core and GEECS-Schemas, and the DB driver the transitional
# gateway source needs. Build context = the repository root (only the three
# package folders are sent: Dockerfile.dockerignore). render_conf.sh passes
# CANOPY_IMAGE from geecs_canopy/pin.py; never build without it.
#
#   docker build --platform linux/amd64 -f GeecsCanopy/deploy/Dockerfile \
#     --build-arg CANOPY_IMAGE=<pin.py CANOPY_IMAGE> -t geecs-canopy:<commit> .
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
