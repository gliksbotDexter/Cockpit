"""
Dexter-Gliksbot Backend Startup Script

This script initializes and starts the Dexter autonomy system backend.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dexter_autonomy.brain.init_db import init_databases
from dexter_autonomy.configs import get_global_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the Dexter backend."""
    logger.info("=" * 60)
    logger.info("Starting Dexter-Gliksbot Backend")
    logger.info("=" * 60)
    
    # Step 1: Load configuration
    logger.info("Loading configuration...")
    try:
        config = get_global_config()
        logger.info(f"Configuration loaded from: {config.config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Step 2: Initialize databases
    logger.info("Initializing databases...")
    try:
        data_dir = config.get("system.data_dir", "./data")
        init_databases(data_dir)
        logger.info("Databases initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize databases: {e}")
        return 1
    
    # Step 3: Start the UI Bridge (FastAPI server)
    logger.info("Starting UI Bridge API server...")
    try:
        from dexter_autonomy.ui_bridge.api import app
        import uvicorn
        
        host = config.get("ui_bridge.host", "0.0.0.0")
        port = config.get("ui_bridge.port", 8765)
        
        logger.info(f"UI Bridge will listen on {host}:{port}")
        logger.info("=" * 60)
        logger.info("Dexter Backend is ready!")
        logger.info("=" * 60)
        
        # Use Config and Server to run uvicorn in the existing event loop
        config_uvicorn = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config_uvicorn)
        await server.serve()
        
    except Exception as e:
        logger.error(f"Failed to start UI Bridge: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
