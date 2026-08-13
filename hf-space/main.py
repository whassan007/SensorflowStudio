"""
Accident Analysis API Gateway - Simplified Version
For demonstration purposes, providing core functionality only.
"""

import io
import logging
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import asyncio
from contextlib import asynccontextmanager

# Simplified implementations to avoid dependency issues
class MockAccidentValidator:
    def __init__(self, df):
        self.df = df
    
    def validate_schema(self):
        class Report:
            def __init__(self):
                self.valid = True
                self.errors = []
                self.quality_score = 0.95
        return Report()

class MockAccidentDataCleaner:
    def __init__(self, df):
        self.df = df
    
    def clean_dataset(self):
        # Remove any rows with missing values
        return self.df.dropna()

class MockAccidentAnalysisEngine:
    def __init__(self, df):
        self.df = df
        
    def run_full_pipeline(self):
        return {
            "metrics": {
                "total_records": len(self.df),
                "geospatial": {
                    "hotspots": [
                        {"lat": 37.7749, "lng": -122.4194, "count": 25},
                        {"lat": 34.0522, "lng": -118.2437, "count": 18}
                    ]
                }
            },
            "summary": "Analysis complete for dataset"
        }

class MockAccidentInsights:
    def __init__(self, metrics):
        self.metrics = metrics
    
    def get_risk_profile(self):
        return {
            "risk_index": 0.75,
            "high_risk_factors": ["urban", "night_hours"],
            "recommendations": [
                "Increase street lighting in high-risk areas",
                "Implement additional safety measures near hotspots"
            ],
            "confidence_score": 0.89
        }

class MockAccidentReport:
    def __init__(self, metrics, insights):
        self.metrics = metrics
        self.insights = insights
    
    def export_to_html(self):
        return f"""
        <html>
        <head><title>Executive Report</title></head>
        <body>
            <h1>Accident Analysis Executive Report</h1>
            <p>Total Records Processed: {self.metrics['metrics']['total_records']}</p>
            <p>Risk Index: {self.insights.get('risk_index', 'N/A')}</p>
            <h2>Key Findings</h2>
            <ul>
                <li>High-risk factors identified</li>
                <li>Hotspot locations provided</li>
            </ul>
        </body>
        </html>
        """

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lifecycle management for the application
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up")
    yield
    logger.info("Application shutting down")

app = FastAPI(
    title="Accident Analysis Platform",
    description="Simplified API Engine for demonstration purposes",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread-safe cache with expiration for processed datasets
class EngineCache:
    """Thread-safe cache implementation with TTL for processed datasets."""
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
        
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """Set a cached value with TTL."""
        async with self._lock:
            self._cache[key] = {
                "data": value,
                "timestamp": asyncio.get_event_loop().time(),
                "ttl": ttl
            }
            
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a cached value, checking TTL."""
        async with self._lock:
            if key not in self._cache:
                return None
                
            item = self._cache[key]
            current_time = asyncio.get_event_loop().time()
            
            # Check if expired
            if current_time - item["timestamp"] > item["ttl"]:
                del self._cache[key]
                return None
                
            return item["data"]
    
    async def cleanup_expired(self):
        """Remove expired cache entries."""
        async with self._lock:
            current_time = asyncio.get_event_loop().time()
            expired_keys = [
                key for key, item in self._cache.items()
                if current_time - item["timestamp"] > item["ttl"]
            ]
            for key in expired_keys:
                del self._cache[key]

# Global cache instance
engine_cache = EngineCache()

class ConnectionManager:
    """Manages WebSocket clients subscribing to real-time telemetry updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Add a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected WebSocket connection."""
        try:
            with self._lock:
                self.active_connections.remove(websocket)
        except ValueError:
            pass  # Already disconnected
    
    async def broadcast(self, message: str):
        """Broadcast a message to all active connections."""
        async with self._lock:
            # Create list of tasks to avoid potential modification during iteration
            tasks = []
            for connection in self.active_connections.copy():
                try:
                    tasks.append(connection.send_text(message))
                except Exception as e:
                    logger.warning(f"Failed to send to WebSocket: {e}")
                    # Connection might be closed, remove it
                    try:
                        self.active_connections.remove(connection)
                    except ValueError:
                        pass  # Already removed
            
            # Execute all sends concurrently
            await asyncio.gather(*tasks, return_exceptions=True)

manager = ConnectionManager()

@app.post("/api/analyze")
async def analyze_accident_data(file: UploadFile = File(...)):
    """
    Uploads a SWITRS formatted CSV dataset, runs validation, cleaning, and initiates the engine.
    
    Returns:
        JSON response with analysis results
    """
    logger.info(f"Received file upload for analysis: {file.filename}")
    
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV datasets are accepted.")
    
    t0 = time.perf_counter()
    try:
        # Validate file size (limit to 50MB)
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50 MB limit
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")
        
        # Parse CSV with error handling
        df = pd.read_csv(io.BytesIO(content))
        logger.info(f"CSV loaded successfully, shape: {df.shape}")
        
        # Mock Validation
        validator = MockAccidentValidator(df)
        report = validator.validate_schema()
        if not report.valid:
            logger.warning(f"Dataset validation failed: {report.errors}")

        # Mock Cleaning
        cleaner = MockAccidentDataCleaner(df) 
        clean_df = cleaner.clean_dataset()
        
        if clean_df.empty:
            raise HTTPException(status_code=422, detail="Cleaning produced an empty dataset.")

        # Mock Analysis
        engine = MockAccidentAnalysisEngine(clean_df)
        metrics = engine.run_full_pipeline()
        
        # Mock Insights
        insights = MockAccidentInsights(metrics)
        profile = insights.get_risk_profile()
        
        elapsed = round(time.perf_counter() - t0, 4)

        # Cache for sub-resource queries with TTL (1 hour)
        await engine_cache.set("latest", {
            "metrics": metrics,
            "insights": profile,
            "processing_time": elapsed,
        }, ttl=3600)
        
        # Broadcast trigger over WebSockets
        broadcast_message = f"New dataset processed. Total records: {len(clean_df)}. Risk index: {profile.get('risk_index', 'N/A')}"
        await manager.broadcast(broadcast_message)
        
        # `simulated`/`analysis_provenance` mark that validation, analysis and
        # insights come from mock engines with constant outputs — the payload
        # must never present them as real analysis results.
        return {
            "status": "success",
            "simulated": True,
            "analysis_provenance": "MOCK_ENGINE",
            "records_processed": len(clean_df),
            "quality_score": report.quality_score,
            "insights": profile,
            "processing_time_seconds": elapsed,
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV file provided.")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except Exception as e:
        logger.error(f"Error in dataset ingestion pipeline: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Pipeline error occurred during processing")

@app.get("/api/hotspots")
async def get_hotspots():
    """
    Retrieves geographic hotspots from the latest processed run.
    
    Returns:
        JSON response with geospatial hotspots
    """
    try:
        cached_data = await engine_cache.get("latest")
        if not cached_data:
            raise HTTPException(status_code=404, detail="No processed metrics found. Please POST to /api/analyze first.")
        
        geo = cached_data["metrics"].get("geospatial", {})
        hotspots = geo.get("hotspots", [])
        
        return {
            "status": "success",
            "simulated": True,
            "analysis_provenance": "MOCK_ENGINE",
            "hotspots": hotspots,
            "count": len(hotspots)
        }
    except Exception as e:
        logger.error(f"Error retrieving hotspots: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve hotspots")

@app.get("/api/insights")
async def get_insights():
    """
    Retrieves high-level risk factors and recommendations.
    
    Returns:
        JSON response with risk insights
    """
    try:
        cached_data = await engine_cache.get("latest")
        if not cached_data:
            raise HTTPException(status_code=404, detail="No processed metrics found. Please POST to /api/analyze first.")
        
        insights = cached_data["insights"]
        return {
            "status": "success",
            "simulated": True,
            "analysis_provenance": "MOCK_ENGINE",
            "insights": insights
        }
    except Exception as e:
        logger.error(f"Error retrieving insights: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve insights")

@app.get("/api/report", response_class=HTMLResponse)
async def get_executive_report():
    """
    Generates and downloads the executive HTML summary report.
    
    Returns:
        HTML content of the generated report
    """
    try:
        cached_data = await engine_cache.get("latest")
        if not cached_data:
            return HTMLResponse(content="<h1>No report generated. Please analyze a dataset first.</h1>", status_code=404)
        
        report = MockAccidentReport(
            metrics=cached_data["metrics"],
            insights=cached_data["insights"]
        )
        html_content = report.export_to_html()
        return HTMLResponse(content=html_content, status_code=200)
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate report")

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    Subscribes to live broadcasts of analytical status changes.
    
    Returns:
        WebSocket connection for real-time updates
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await websocket.receive_text()
            await websocket.send_text(f"Telemetry heartbeat: received {len(data)} bytes")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        JSON response indicating service status
    """
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "AccidentAnalysisAPI",
            "version": "1.0.0"
        },
        status_code=200
    )
