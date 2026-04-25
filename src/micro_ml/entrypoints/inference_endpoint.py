from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl
import skops.io as sio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parents[2] / "micro_ml" / "data" / "model"
print(MODEL_DIR)
MODEL_PATH = MODEL_DIR / "model.skops"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.skops"

# ── Target decoding (inverse of feature.yaml target_encodings) ────────────────
TARGET_DECODE = {2: "High", 1: "Medium", 0: "Low"}


# ── App state ─────────────────────────────────────────────────────────────────
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessor once at startup."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise RuntimeError(f"Preprocessor not found at {PREPROCESSOR_PATH}")

    state["model"] = sio.load(
        MODEL_PATH, trusted=sio.get_untrusted_types(file=MODEL_PATH)
    )
    state["preprocessor"] = sio.load(
        PREPROCESSOR_PATH, trusted=sio.get_untrusted_types(file=PREPROCESSOR_PATH)
    )
    yield
    state.clear()


app = FastAPI(title="micro_ml inference", lifespan=lifespan)


# ── Schema ────────────────────────────────────────────────────────────────────
class InferenceRequest(BaseModel):
    # Exact columns from feature.yaml -> columns -> features
    age: float
    experience_years: float
    daily_work_hours: float
    sleep_hours: float
    caffeine_intake: float
    bugs_per_day: float
    commits_per_day: float
    meetings_per_day: float
    screen_time: float
    exercise_hours: float
    stress_level: float


class InferenceResponse(BaseModel):
    burnout_level: str  # "Low" | "Medium" | "High"
    burnout_level_encoded: int  # 0 | 1 | 2


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    model_loaded = "model" in state and "preprocessor" in state
    return {
        "status": "ok" if model_loaded else "unavailable",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    try:
        # Build polars DataFrame — matches what the preprocessor was fit on
        df = pl.DataFrame([request.model_dump()])

        # Transform using saved sklearn Pipeline (SimpleImputer, polars output)
        transformed = state["preprocessor"].transform(df)

        # Predict — returns encoded int (0, 1, 2)
        encoded = int(state["model"].predict(transformed)[0])

        return InferenceResponse(
            burnout_level=TARGET_DECODE[encoded],
            burnout_level_encoded=encoded,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
