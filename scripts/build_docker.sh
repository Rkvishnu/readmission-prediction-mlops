#!/usr/bin/env bash
set -euo pipefail
docker build -f deployment/docker/Dockerfile.api -t readmission-api:latest .

