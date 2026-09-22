# The full app, drawing included, without installing Python.
#
#   docker run --rm -p 8731:8731 ghcr.io/samehaisaa/sonoform
#
# Then open http://127.0.0.1:8731/. The published demo can only show the
# presets, because it has no solver behind it. This has the solver.

FROM python:3.12-slim AS build
WORKDIR /src

# numpy, scipy and matplotlib all ship manylinux wheels, so there is no
# compiler here and the image stays small.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
LABEL org.opencontainers.image.source=https://github.com/samehaisaa/sonoform
LABEL org.opencontainers.image.description="Chladni figures of a free brass plate, solved and bowed"
LABEL org.opencontainers.image.licenses=MIT

COPY --from=build /install /usr/local

# Nothing here needs root, and the server writes only to a temp directory.
RUN useradd --create-home --uid 10001 plate
USER plate

EXPOSE 8731
# 0.0.0.0 so the port is reachable from outside the container's namespace.
# The default everywhere else stays on loopback.
CMD ["sonoform", "play", "--host", "0.0.0.0", "--no-browser"]
