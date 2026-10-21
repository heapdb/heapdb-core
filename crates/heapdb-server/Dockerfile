# heapdb-server container image (multi-stage, slim, non-root).
#
# Build from the repository root:
#   docker build -f crates/heapdb-server/Dockerfile -t heapdb-server .
# Run a single node:
#   docker run -p 8080:8080 -v heapdb-data:/data -e HEAPDB_ADMIN_TOKEN=... heapdb-server

FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked -p heapdb-server --features cluster-aws \
 && cp target/release/heapdb-server /heapdb-server

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --home-dir /data --shell /usr/sbin/nologin heapdb \
 && mkdir -p /data && chown heapdb:heapdb /data
COPY --from=build /heapdb-server /usr/local/bin/heapdb-server
USER heapdb
ENV HEAPDB_LISTEN=0.0.0.0:8080 \
    HEAPDB_INTERNAL_LISTEN=0.0.0.0:9090 \
    HEAPDB_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8080 9090
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD ["heapdb-server", "healthcheck", "--url", "http://127.0.0.1:8080"]
ENTRYPOINT ["heapdb-server"]
