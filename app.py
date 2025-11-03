"""
PDF Scraper API
===============
Universal PDF data extraction service for Railway deployment.
Automatically extracts contacts, projects, and tenders from any structured PDF.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import logging
from pathlib import Path

from scraper import parse_pdf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="PDF Scraper API",
    description="Universal PDF data extraction - works with any structured PDF",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API information"""
    return {
        "service": "PDF Scraper API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "POST /extract": "Extract data from a single PDF",
            "POST /extract-batch": "Extract data from multiple PDFs",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "features": [
            "Universal PDF support (any company, any language)",
            "Automatic table detection and classification",
            "Intelligent data validation",
            "Confidence scoring",
            "Multi-method extraction (Camelot + pdfplumber)"
        ]
    }

@app.get("/health")
async def health():
    """Health check for Railway"""
    return {"status": "healthy"}

@app.post("/extract")
async def extract_pdf(file: UploadFile = File(...)):
    """
    Extract structured data from a PDF.
    
    Returns:
    - company_info: Company details (name, CVR, contact info)
    - contacts: List of contacts with names, phones, emails, roles
    - projects: List of projects with budgets, dates, stages
    - tenders: List of tenders/bids
    - quality: Extraction quality metrics
    """
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only PDF files are supported."
        )
    
    logger.info(f"Processing PDF: {file.filename}")
    
    tmp_path = None
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Parse PDF
        result = parse_pdf(tmp_path)
        
        # Add metadata
        result['metadata'] = {
            'filename': file.filename,
            'success': True
        }
        
        logger.info(
            f"Extracted: {result['summary']['contacts']} contacts, "
            f"{result['summary']['projects']} projects, "
            f"{result['summary']['tenders']} tenders"
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract data: {str(e)}"
        )
    
    finally:
        # Cleanup
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")

@app.post("/extract-batch")
async def extract_batch(files: list[UploadFile] = File(...)):
    """
    Extract data from multiple PDFs.
    
    Returns results for each file with success/error status.
    """
    
    results = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            results.append({
                'filename': file.filename,
                'success': False,
                'error': 'Invalid file type'
            })
            continue
        
        tmp_path = None
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            # Parse PDF
            result = parse_pdf(tmp_path)
            result['metadata'] = {
                'filename': file.filename,
                'success': True
            }
            
            results.append(result)
            
            # Cleanup
            os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'success': False,
                'error': str(e)
            })
    
    # Calculate summary
    successful = [r for r in results if r.get('metadata', {}).get('success')]
    failed = [r for r in results if not r.get('metadata', {}).get('success')]
    
    return JSONResponse(content={
        'results': results,
        'summary': {
            'total': len(files),
            'successful': len(successful),
            'failed': len(failed),
            'total_contacts': sum(r.get('summary', {}).get('contacts', 0) for r in successful),
            'total_projects': sum(r.get('summary', {}).get('projects', 0) for r in successful),
            'total_tenders': sum(r.get('summary', {}).get('tenders', 0) for r in successful)
        }
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
