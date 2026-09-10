ARG BUILD_IMAGE="registry.access.redhat.com/ubi9/nodejs-22:latest"

# Build stage
FROM ${BUILD_IMAGE} AS builder

COPY --chown=default:root package*.json ./
RUN npm ci

COPY --chown=default:root . .
RUN npm run build

# Production stage
FROM registry.access.redhat.com/ubi9/nginx-124:latest

# Copy built files
COPY --from=builder --chown=1001:0 /opt/app-root/src/dist .

# Add CORS header for Module Federation remote entry
RUN echo $'location /remoteEntry.js {\n    add_header Access-Control-Allow-Origin *;\n}' \
    > "${NGINX_DEFAULT_CONF_PATH}/cors.conf"

# Route nginx logs to stdout/stderr (the S2I run script does this
# automatically, but CMD below bypasses it)
RUN ln -sf /dev/stdout /var/log/nginx/access.log && \
    ln -sf /dev/stderr /var/log/nginx/error.log

EXPOSE 8080

USER 1001:0

CMD ["nginx", "-g", "daemon off;"]
