"""
Health Server - Simple HTTP server for health checks and debugging.

This module provides a lightweight HTTP server that exposes:
    GET /health - Returns {"status": "ok"} for container health checks
    GET /personas - Returns list of available persona names

The server runs in the background alongside the main avatar service.
"""

import asyncio
import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from .personas import load_all_personas


logger = logging.getLogger(__name__)

DEFAULT_PORT = 8080


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health and status endpoints."""

    def log_message(self, format: str, *args) -> None:
        """Override to use Python logging instead of stderr."""
        logger.debug(f"Health server: {format % args}")

    def _send_json_response(self, data: dict, status: int = HTTPStatus.OK) -> None:
        """Send a JSON response with the given data and status code."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/personas":
            self._handle_personas()
        elif self.path == "/":
            self._handle_root()
        else:
            self._handle_not_found()

    def _handle_health(self) -> None:
        """Handle /health endpoint."""
        self._send_json_response({"status": "ok"})

    def _handle_personas(self) -> None:
        """Handle /personas endpoint."""
        try:
            personas = load_all_personas()
            persona_list = [
                {
                    "name": p.name,
                    "display_name": p.display_name,
                    "video_path": p.video_path,
                }
                for p in personas.values()
            ]
            self._send_json_response({"personas": persona_list})
        except Exception as e:
            logger.error(f"Error loading personas: {e}")
            self._send_json_response(
                {"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def _handle_root(self) -> None:
        """Handle / endpoint - basic service info."""
        self._send_json_response(
            {
                "service": "voipavatar",
                "endpoints": ["/health", "/personas"],
            }
        )

    def _handle_not_found(self) -> None:
        """Handle unknown endpoints."""
        self._send_json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)


class HealthServer:
    """
    Background HTTP server for health checks.

    This server runs in a separate thread and provides health check
    endpoints for container orchestration (Docker, Kubernetes, etc.).
    """

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        """
        Initialize the health server.

        Args:
            port: Port to listen on (default: 8080)
        """
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        """Start the health server in a background thread."""
        self._server = HTTPServer(("0.0.0.0", self.port), HealthHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"Health server started on port {self.port}")

    def stop(self) -> None:
        """Stop the health server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Health server stopped")


async def run_health_server(port: int = DEFAULT_PORT) -> HealthServer:
    """
    Start the health server and return the server instance.

    This is a convenience function for starting the health server
    from async code.

    Args:
        port: Port to listen on (default: 8080)

    Returns:
        HealthServer instance (already started)
    """
    server = HealthServer(port)
    server.start()
    return server


def main() -> None:
    """Run the health server standalone (for testing)."""
    import argparse

    parser = argparse.ArgumentParser(description="Health check server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("HEALTH_PORT", DEFAULT_PORT)),
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = HealthServer(args.port)
    server.start()

    logger.info(f"Health server running on http://0.0.0.0:{args.port}")
    logger.info("Press Ctrl+C to stop")

    try:
        # Keep the main thread alive
        while True:
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop()


if __name__ == "__main__":
    main()
