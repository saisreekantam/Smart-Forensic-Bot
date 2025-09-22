#!/usr/bin/env python3
"""
Simple test server for testing the Evidence Viewer frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import sqlite3
import json
from datetime import datetime

app = FastAPI(title="Test Evidence API", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    """Get SQLite database connection"""
    return sqlite3.connect('data/forensic_cases.db')

@app.get("/")
async def root():
    return {"message": "Evidence API Test Server", "status": "running"}

@app.get("/cases")
async def get_cases():
    """Get all cases"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, case_number, title, status, investigator_name, 
               created_at, updated_at
        FROM cases
    """)
    
    cases = []
    for row in cursor.fetchall():
        cases.append({
            "id": row[0],
            "case_number": row[1], 
            "title": row[2],
            "status": row[3],
            "investigator_name": row[4],
            "created_at": row[5],
            "updated_at": row[6],
            "total_evidence_count": 0,
            "processed_evidence_count": 0,
            "processing_progress": 0.0
        })
    
    conn.close()
    return cases

@app.get("/cases/{case_id}/evidence")
async def get_case_evidence(case_id: str):
    """Get evidence for a case"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get evidence for the case
    cursor.execute("""
        SELECT id, original_filename, evidence_type, processing_status,
               file_size, created_at, has_embeddings
        FROM evidence 
        WHERE case_id = ?
    """, (case_id,))
    
    evidence_list = []
    for row in cursor.fetchall():
        evidence_list.append({
            "id": row[0],
            "original_filename": row[1],
            "evidence_type": row[2] or "document",
            "processing_status": row[3] or "pending", 
            "file_size": row[4] or 0,
            "created_at": row[5],
            "has_embeddings": bool(row[6]) if row[6] is not None else False
        })
    
    conn.close()
    
    if not evidence_list:
        raise HTTPException(status_code=404, detail="No evidence found for this case")
    
    return evidence_list

@app.get("/cases/{case_id}/evidence/timeline")
async def get_evidence_timeline(case_id: str):
    """Get timeline data for evidence"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if case exists
    cursor.execute("SELECT id FROM cases WHERE id = ?", (case_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Get evidence with timestamps
    cursor.execute("""
        SELECT original_filename, evidence_type, created_at, processing_status
        FROM evidence 
        WHERE case_id = ?
        ORDER BY created_at
    """, (case_id,))
    
    timeline_events = []
    for row in cursor.fetchall():
        timeline_events.append({
            "timestamp": row[2],
            "event": f"Evidence uploaded: {row[0]}",
            "evidence": [row[0]],
            "importance": "high" if row[3] == "completed" else "medium"
        })
    
    conn.close()
    
    return {
        "timeline_events": timeline_events,
        "insights": ["Timeline generated from evidence upload dates"],
        "confidence": 0.8,
        "generated_by": "simple_timeline",
        "case_id": case_id
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple Evidence API Server...")
    print("📍 Server: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")