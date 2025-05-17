from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from textblob import TextBlob
from typing import List, Optional
import uvicorn
import pandas as pd
import random
import os

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing text sentiments using a CSV dataset",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# Configuration
CSV_PATH = r"C:\Users\KIIT\Desktop\AD-LAB\Expt8\tweets.csv"
DEFAULT_SAMPLE_SIZE = 10
MAX_SAMPLE_SIZE = 100

# Response models
class TextResponse(BaseModel):
    text: str
    sentiment: str
    polarity: float
    confidence: Optional[float] = None

class HealthCheck(BaseModel):
    status: str
    sample_size: int
    loaded: bool

# Load dataset
def load_dataset():
    try:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"File not found at {CSV_PATH}")
        
        df = pd.read_csv(CSV_PATH)
        
        if 'text' not in df.columns:
            raise ValueError("CSV must contain a 'text' column")
            
        df['text'] = df['text'].fillna('').astype(str)
        return df['text'].tolist()
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load dataset: {str(e)}"
        )

SAMPLE_TWEETS = load_dataset()

# Sentiment analysis
def analyze_sentiment(text: str) -> dict:
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {
            "sentiment": sentiment,
            "polarity": round(polarity, 2),
            "confidence": round(subjectivity, 2)
        }
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return {
            "sentiment": "Unknown",
            "polarity": 0.0,
            "confidence": 0.0
        }

# API Endpoints
@app.get("/", response_model=HealthCheck)
async def health_check():
    return {
        "status": "operational",
        "sample_size": len(SAMPLE_TWEETS),
        "loaded": len(SAMPLE_TWEETS) > 0
    }

@app.get("/analyze/", response_model=List[TextResponse])
async def analyze_texts(
    keyword: str,
    count: int = DEFAULT_SAMPLE_SIZE,
    randomize: bool = True
):
    if not keyword.strip():
        raise HTTPException(
            status_code=400,
            detail="Keyword cannot be empty"
        )
        
    if not 0 < count <= MAX_SAMPLE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Count must be between 1 and {MAX_SAMPLE_SIZE}"
        )
    
    # Filter matching texts
    matches = [t for t in SAMPLE_TWEETS if keyword.lower() in t.lower()]
    
    # Handle insufficient matches
    if len(matches) < count:
        if randomize:
            remaining = [t for t in SAMPLE_TWEETS if t not in matches]
            matches += random.sample(
                remaining,
                min(len(remaining), count - len(matches))
            )
        else:
            count = min(count, len(matches))
    
    # Analyze selected texts
    results = []
    for text in (random.sample(matches, count) if randomize else matches[:count]):
        analysis = analyze_sentiment(text)
        results.append({
            "text": text,
            **analysis
        })
    
    return JSONResponse(content=results)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )