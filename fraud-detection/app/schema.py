from pydantic import BaseModel
from typing import Dict, Any

class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: str
    card1: float
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    dist2: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    C1: float | None = None
    C2: float | None = None
    C3: float | None = None
    # ... add more fields as you like — the model will ignore missing ones
    M1: str | None = None
    M2: str | None = None
    M3: str | None = None
    M4: str | None = None
    M5: str | None = None
    M6: str | None = None
    # Up to M9 if you want

    class Config:
        extra = "allow"  # accept extra fields without crashing