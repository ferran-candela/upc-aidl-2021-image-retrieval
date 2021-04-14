### STAGE 1: Build ###
FROM node:14.16-alpine AS build
WORKDIR /usr/src/app
COPY retrieval-app/package.json retrieval-app/package-lock.json ./
RUN npm install
COPY retrieval-app .
RUN npm run build

### STAGE 2: Run ###
FROM nginx:1.19.10-alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=build /usr/src/app/dist/retrieval-app /usr/share/nginx/html