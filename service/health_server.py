"""
Health Server - Simple HTTP server for health checks and debugging.

This module provides a lightweight HTTP server that exposes:
    GET /health - Returns {"status": "ok"} for container health checks
    GET /personas - Returns list of available persona names

The server runs in the background alongside the main avatar service.
"""
