#!/bin/bash
# VoIP Avatar Service Entrypoint
#
# This script starts both the health server and the main avatar service.
# It handles graceful shutdown when receiving SIGTERM/SIGINT.

set -e

# Default values (can be overridden by environment variables)
HEALTH_PORT="${HEALTH_PORT:-8080}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
PERSONA_NAME="${PERSONA_NAME:-default}"

echo "Starting VoIP Avatar Service"
echo "  Persona: ${PERSONA_NAME}"
echo "  Health port: ${HEALTH_PORT}"
echo "  Log level: ${LOG_LEVEL}"

# Check required environment variables
if [ -z "$LIVEKIT_URL" ]; then
    echo "ERROR: LIVEKIT_URL environment variable is required"
    exit 1
fi

if [ -z "$LIVEKIT_ROOM" ]; then
    echo "ERROR: LIVEKIT_ROOM environment variable is required"
    exit 1
fi

if [ -z "$LIVEKIT_API_KEY" ]; then
    echo "ERROR: LIVEKIT_API_KEY environment variable is required"
    exit 1
fi

if [ -z "$LIVEKIT_API_SECRET" ]; then
    echo "ERROR: LIVEKIT_API_SECRET environment variable is required"
    exit 1
fi

# Start the health server in the background
echo "Starting health server on port ${HEALTH_PORT}..."
python -m service.health_server --port "${HEALTH_PORT}" &
HEALTH_PID=$!

# Give the health server a moment to start
sleep 1

# Trap signals for graceful shutdown
cleanup() {
    echo "Shutting down..."
    if [ -n "$HEALTH_PID" ]; then
        kill "$HEALTH_PID" 2>/dev/null || true
    fi
    if [ -n "$SERVICE_PID" ]; then
        kill "$SERVICE_PID" 2>/dev/null || true
    fi
    wait
    echo "Shutdown complete"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start the main avatar service
echo "Starting avatar service..."
python -m service.livekit_avatar_service \
    --persona "${PERSONA_NAME}" \
    --log-level "${LOG_LEVEL}" \
    &
SERVICE_PID=$!

# Wait for the service to exit
wait $SERVICE_PID
EXIT_CODE=$?

# Clean up health server
kill "$HEALTH_PID" 2>/dev/null || true

exit $EXIT_CODE
