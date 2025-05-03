from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    fullname: str
    phone_number: str

class UserCreate(UserBase):
    password: str
    role: str = "user"

class UserResponse(UserBase):
    id: int
    role: str
    created_at: datetime

    class Config:
        orm_mode = True

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# Transaction schemas
class TransactionBase(BaseModel):
    sender_phone: str
    receiver_phone: str
    amount: float
    transaction_type: str
    location: Optional[str] = None
    device_id: Optional[str] = None

class TransactionCreate(TransactionBase):
    pass

class TransactionResponse(TransactionBase):
    id: int
    fraud_score: float
    created_at: datetime
    user_id: int

    class Config:
        orm_mode = True

# Fraud Alert schemas
class FraudAlertBase(BaseModel):
    transaction_id: int
    alert_type: str
    risk_score: float
    description: str
    is_resolved: bool = False

class FraudAlertCreate(FraudAlertBase):
    pass

class FraudAlertResponse(FraudAlertBase):
    id: int
    created_at: datetime
    resolved_by: Optional[int] = None
    resolved_at: Optional[datetime] = None

    class Config:
        orm_mode = True

# Dashboard statistics schema
class DashboardStats(BaseModel):
    total_transactions: int
    total_volume: float
    average_transaction: float
    fraud_stats: Optional[dict] = None