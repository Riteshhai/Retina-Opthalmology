FROM node:24-alpine@sha256:116b800c3754b440ee0aa3ed2f20f5bf3d99a554bfd9adeea09ea8017f6e85f5 AS sources
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .

FROM nginxinc/nginx-unprivileged:stable-alpine@sha256:f5116ed4606f3eb015dd4a8e28cb9b351476dc0ad4bc6e9f93b19cec130cdf03 AS nginx

FROM sources AS workspace-build
RUN npm run build:workspace

FROM nginx AS local-workspace
COPY workspace.nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=workspace-build /app/dist-workspace /usr/share/nginx/html

FROM sources AS public-build
RUN npm run build

FROM nginx AS public-site
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=public-build /app/dist /usr/share/nginx/html
