#!/bin/bash

# Docker Compose Cleanup Script
# Usage: ./cleanup.sh [--containers|--images|--volumes|--networks|--all]
# Default: cleans everything

set -e

# Determine compose command
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

# Get project name from directory
PROJECT_NAME=$(basename "$(pwd)" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')

# Parse arguments
CLEAN_CONTAINERS=false
CLEAN_IMAGES=false
CLEAN_VOLUMES=false
CLEAN_NETWORKS=false

case "${1:-all}" in
    --containers) CLEAN_CONTAINERS=true ;;
    --images) CLEAN_IMAGES=true ;;
    --volumes) CLEAN_VOLUMES=true ;;
    --networks) CLEAN_NETWORKS=true ;;
    --all|*) CLEAN_CONTAINERS=true; CLEAN_IMAGES=true; CLEAN_VOLUMES=true; CLEAN_NETWORKS=true ;;
esac

# Get images before stopping containers
if [ "$CLEAN_IMAGES" = true ]; then
    echo "Getting image list..."
    IMAGES=$($COMPOSE_CMD ps -q | xargs -r docker inspect --format='{{.Image}}' 2>/dev/null | sort -u || echo "")
fi

# Stop and remove containers
if [ "$CLEAN_CONTAINERS" = true ]; then
    echo "Stopping containers..."
    $COMPOSE_CMD down --remove-orphans 2>/dev/null || true
fi

# Remove images
if [ "$CLEAN_IMAGES" = true ]; then
    echo "Removing images..."
    # Remove images from compose file
    $COMPOSE_CMD config --images 2>/dev/null | xargs -r docker rmi -f 2>/dev/null || true
    # Remove images from stopped containers
    echo "$IMAGES" | while IFS= read -r image; do
        [ -n "$image" ] && docker rmi -f "$image" 2>/dev/null || true
    done
fi

# Remove volumes
if [ "$CLEAN_VOLUMES" = true ]; then
    echo "Removing volumes..."
    # Remove compose volumes
    $COMPOSE_CMD down --volumes --remove-orphans 2>/dev/null || true
    # Remove all project volumes
    docker volume ls -q --filter "name=${PROJECT_NAME}" | xargs -r docker volume rm -f 2>/dev/null || true
    # Remove dangling volumes
    docker volume prune -f 2>/dev/null || true
fi

# Remove networks
if [ "$CLEAN_NETWORKS" = true ]; then
    echo "Removing networks..."
    docker network ls -q --filter "name=${PROJECT_NAME}" | xargs -r docker network rm 2>/dev/null || true
    # Clean up dangling networks
    docker network prune -f 2>/dev/null || true
fi

echo "Cleanup complete!"

# Show what's left
echo "Remaining: $(docker ps -aq | wc -l) containers, $(docker images -q | wc -l) images, $(docker volume ls -q | wc -l) volumes"