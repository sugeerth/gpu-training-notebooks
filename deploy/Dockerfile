# The API image. Python and this repository — nothing else.
#
# servingkit has no dependencies, which is what keeps this honest: there is no wheel-building
# stage, no compiler in the runtime layer, and no lockfile to drift. The image is the base plus
# a few hundred kilobytes of Python.
#
#   docker build -f deploy/Dockerfile -t servingkit/api:0.1.0 .
#   docker run --rm -p 8000:8000 servingkit/api:0.1.0
FROM python:3.12-slim AS runtime

# Fail fast and log straight through. A buffered stdout in a container means the last thing you
# see before a crash is not the thing that crashed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    SERVINGKIT_SCORECARD=/var/lib/servingkit/scorecard.json \
    SERVINGKIT_ENABLE_KERNELS=0

# The API image deliberately cannot compile kernels: no g++, and the route is off by default.
# Kernel verification is the other image's job, because it needs a toolchain and this does not,
# and merging them would put a C++ compiler in the container that faces the internet.
WORKDIR /app

# Only what the service needs at runtime. The notebooks are excluded by .dockerignore, so this
# image does not carry fifty .ipynb files — but `kernels/` and `kernelbench/` are here because
# /v1/kernels lists them and the drift check reads the notebooks it is pointed at.
COPY pyproject.toml README.md ./
COPY servingkit/ ./servingkit/
COPY kernels/ ./kernels/
COPY kernelbench/ ./kernelbench/

RUN python -m pip install --no-cache-dir --no-compile -e . \
    && python -c "import servingkit as sk; assert len(sk.LEVERS) >= 17, sk.LEVERS" \
    && mkdir -p /var/lib/servingkit \
    && useradd --create-home --uid 10001 servingkit \
    && chown -R servingkit:servingkit /var/lib/servingkit /app

USER 10001
EXPOSE 8000

# The check hits /readyz rather than /healthz on purpose: readiness runs the models and verifies
# the levers' declarations against measurement, so a container whose arithmetic broke fails its
# health check instead of serving wrong plans.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request as u, sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/readyz', timeout=4).status == 200 else 1)"

ENTRYPOINT ["python", "-m", "servingkit"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
