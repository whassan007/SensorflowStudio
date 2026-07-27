"""
Accident Analysis API Gateway
Exposes FastAPI REST endpoints and real-time streaming WebSocket connections for processed SWITRS datasets.
"""

import io
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import pandas as pd

from accident_importer import AccidentDataImporter
from accident_validator import AccidentDataValidator
from accident_cleaner import AccidentDataCleaner
from accident_analysis import AccidentAnalysisEngine
from accident_insights import AccidentInsights
from accident_report import AccidentReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Accident Analysis Platform",
    description="6-Layer Production Accident Data Analysis API Engine",
    version="1.0.0"
)

# Global memory caches to hold processed state for querying
engine_cache: Dict[str, Any] = {}

class ConnectionManager:
    """Manages WebSocket clients subscribing to real-time telemetry updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.post("/api/analyze")
async def analyze_accident_data(file: UploadFile = File(...)):
    """
    Uploads a SWITRS formatted CSV dataset, runs validation, cleaning, and initiates the engine.
    """
    logger.info(f"Received file upload for analysis: {file.filename}")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV datasets are accepted.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # 1. Validation
        validator = AccidentDataValidator(df)
        report = validator.validate_schema()
        if not report.valid:
            logger.warning(f"Dataset validation failed: {report.errors}")
            # Still proceed but log the errors

        # 2. Cleaning
        cleaner = AccidentDataCleaner(df)
        clean_df = cleaner.clean_dataset()
        if clean_df.empty:
            raise HTTPException(status_code=422, detail="Cleaning produced an empty dataset.")

        # 3. Compute Analysis
        engine = AccidentAnalysisEngine(clean_df)
        metrics = engine.run_full_pipeline()
        
        # 4. Insights
        insights = AccidentInsights(metrics)
        profile = insights.get_risk_profile()
        
        # Cache for sub-resource queries
        engine_cache["latest"] = {
            "metrics": metrics,
            "insights": profile
        }
        
        # Broadcast trigger over WebSockets
        await manager.broadcast(f"New dataset processed. Total records: {len(clean_df)}. Risk index: {profile['risk_index']}")
        
        return {
            "status": "success",
            "records_processed": len(clean_df),
            "quality_score": report.quality_score,
            "insights": profile
        }

    except Exception as e:
        logger.error(f"Error in dataset ingestion pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.get("/api/hotspots")
def get_hotspots():
    """
    Retrieves geographic hotspots from the latest processed run.
    """
    if "latest" not in engine_cache:
        raise HTTPException(status_code=404, detail="No processed metrics found. Please POST to /api/analyze first.")
    
    geo = engine_cache["latest"]["metrics"].get("geospatial", {})
    return {
        "status": "success",
        "hotspots": geo.get("hotspots", [])
    }

@app.get("/api/insights")
def get_insights():
    """
    Retrieves high-level risk factors and recommendations.
    """
    if "latest" not in engine_cache:
        raise HTTPException(status_code=404, detail="No processed metrics found. Please POST to /api/analyze first.")
    
    return {
        "status": "success",
        "insights": engine_cache["latest"]["insights"]
    }

@app.get("/api/report", response_class=HTMLResponse)
def get_executive_report():
    """
    Generates and downloads the executive HTML summary report.
    """
    if "latest" not in engine_cache:
        raise HTMLResponse(content="<h1>No report generated. Please analyze a dataset first.</h1>", status_code=404)
    
    report = AccidentReport(
        metrics=engine_cache["latest"]["metrics"],
        insights=engine_cache["latest"]["insights"]
    )
    return report.export_to_html()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    Subscribes to live broadcasts of analytical status changes.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await websocket.receive_text()
            await websocket.send_text(f"Telemetry heartbeat: received {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
