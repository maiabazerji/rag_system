FROM node:22-alpine

WORKDIR /app

# Lockfile first, so the dependency layer is cached until it changes and the
# installed tree is exactly the one CI tested.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Vite binds to localhost unless VITE_HOST is set (see vite.config.ts). Inside
# the container it must listen on every interface for the published port to
# reach it; Compose publishes that port on 127.0.0.1 only.
ENV VITE_HOST=0.0.0.0

EXPOSE 5173

CMD ["npm", "run", "dev"]
