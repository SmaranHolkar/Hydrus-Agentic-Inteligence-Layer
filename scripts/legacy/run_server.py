"""
run_server.py — Root launcher for the HydrusOPT API server.
Usage: python run_server.py [--model ...] [--neuro] [--port 8000]
"""
import sys
import os

# Ensure the project root is on the path so hcl_core and server_api resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server_api.server import run_server

if __name__ == "__main__":
    run_server()
