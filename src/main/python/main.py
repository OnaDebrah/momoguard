import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from config.db import get_db, engine, Base
from models.db.db import Transaction, FraudAlert, User
from models.pydantic.schemas import (
    TransactionCreate,
    TransactionResponse,
    FraudAlertResponse,
    UserCreate,
    UserResponse,
    LoginRequest
)
from utils.auth import create_access_token, get_current_user

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize fraud detection model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/fraud_detector.pkl")
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Fraud detection model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

app = FastAPI(
    title="MoMoGuard-GH API",
    description="Mobile Money Fraud Detection System for Ghana",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication endpoints
@app.post("/api/auth/register", response_model=UserResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    new_user = User(
        email=user.email,
        fullname=user.fullname,
        phone_number=user.phone_number,
        hashed_password=User.get_password_hash(user.password),
        role=user.role
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user

@app.post("/api/auth/login")
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user or not user.verify_password(login_data.password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "email": user.email,
        "role": user.role
    }

# Transaction endpoints
@app.post("/api/transactions/", response_model=TransactionResponse)
async def create_transaction(
        transaction: TransactionCreate,
        background_tasks: BackgroundTasks,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Create a new transaction and check for fraud"""

    # Create new transaction in DB
    new_transaction = Transaction(
        sender_phone=transaction.sender_phone,
        receiver_phone=transaction.receiver_phone,
        amount=transaction.amount,
        transaction_type=transaction.transaction_type,
        location=transaction.location,
        device_id=transaction.device_id,
        user_id=current_user.id
    )

    db.add(new_transaction)
    db.commit()
    db.refresh(new_transaction)

    # Check for fraud in the background
    background_tasks.add_task(
        check_transaction_for_fraud,
        transaction_id=new_transaction.id,
        db=db
    )

    return new_transaction

def check_transaction_for_fraud(transaction_id: int, db: Session):
    """Analyze a transaction for potential fraud"""
    transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()

    if not transaction:
        logger.error(f"Transaction {transaction_id} not found")
        return

    try:
        # Get user's transaction history
        user_id = transaction.user_id
        sender_phone = transaction.sender_phone

        # Get recent transactions from this sender
        recent_txns = db.query(Transaction).filter(
            Transaction.sender_phone == sender_phone,
            Transaction.created_at >= datetime.now() - timedelta(days=30)
        ).all()

        # Extract features for model
        features = extract_fraud_features(transaction, recent_txns)

        # Predict fraud probability if model is loaded
        if model is not None:
            # Convert features to array for prediction
            feature_array = np.array([
                features['amount'],
                features['is_foreign_receiver'],
                features['num_recent_transactions'],
                features['avg_transaction_amount'],
                features['transaction_frequency_change'],
                features['is_new_receiver'],
                features['time_of_day_risk']
            ]).reshape(1, -1)

            fraud_probability = model.predict_proba(feature_array)[0, 1]
            fraud_threshold = 0.7  # Configurable threshold

            # Update transaction with fraud score
            transaction.fraud_score = float(fraud_probability)
            db.commit()

            # Create fraud alert if score exceeds threshold
            if fraud_probability >= fraud_threshold:
                alert = FraudAlert(
                    transaction_id=transaction.id,
                    alert_type="HIGH_RISK_TRANSACTION",
                    risk_score=float(fraud_probability),
                    description=f"Suspicious transaction detected with {fraud_probability:.2f} risk score",
                    is_resolved=False
                )
                db.add(alert)
                db.commit()

                logger.warning(f"Fraud alert created for transaction {transaction_id}")

                # Here you would typically send notifications via SMS, email, etc.
        else:
            logger.warning("Model not loaded, skipping fraud detection")

    except Exception as e:
        logger.error(f"Error in fraud detection: {str(e)}")

def extract_fraud_features(transaction, recent_transactions):
    """Extract features from transaction for fraud detection model"""

    # Calculate various fraud detection features
    is_foreign_receiver = not transaction.receiver_phone.startswith('+233')  # Ghana code

    num_recent_transactions = len(recent_transactions)

    # Calculate average amount of recent transactions
    if num_recent_transactions > 0:
        avg_transaction_amount = sum(txn.amount for txn in recent_transactions) / num_recent_transactions
    else:
        avg_transaction_amount = 0

    # Check if receiver is new (not in recent transactions)
    receiver_phones = [txn.receiver_phone for txn in recent_transactions]
    is_new_receiver = transaction.receiver_phone not in receiver_phones

    # Calculate transaction frequency change (simplified)
    recent_week_count = len([
        txn for txn in recent_transactions
        if txn.created_at >= datetime.now() - timedelta(days=7)
    ])
    prev_week_count = len([
        txn for txn in recent_transactions
        if txn.created_at >= datetime.now() - timedelta(days=14)
           and txn.created_at < datetime.now() - timedelta(days=7)
    ])

    if prev_week_count > 0:
        transaction_frequency_change = (recent_week_count - prev_week_count) / prev_week_count
    else:
        transaction_frequency_change = recent_week_count  # Just use current week's count if no previous

    # Time of day risk (higher risk during night hours)
    hour_of_day = transaction.created_at.hour
    time_of_day_risk = 1 if (hour_of_day < 6 or hour_of_day >= 22) else 0

    return {
        'amount': transaction.amount,
        'is_foreign_receiver': int(is_foreign_receiver),
        'num_recent_transactions': num_recent_transactions,
        'avg_transaction_amount': avg_transaction_amount,
        'transaction_frequency_change': transaction_frequency_change,
        'is_new_receiver': int(is_new_receiver),
        'time_of_day_risk': time_of_day_risk
    }

@app.get("/api/transactions/", response_model=List[TransactionResponse])
def get_transactions(
        skip: int = 0,
        limit: int = 100,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Get a list of transactions"""
    # Admin can see all transactions, regular users only see their own
    if current_user.role == "admin":
        transactions = db.query(Transaction).offset(skip).limit(limit).all()
    else:
        transactions = db.query(Transaction).filter(
            Transaction.user_id == current_user.id
        ).offset(skip).limit(limit).all()

    return transactions

@app.get("/api/alerts/", response_model=List[FraudAlertResponse])
def get_fraud_alerts(
        skip: int = 0,
        limit: int = 100,
        resolved: Optional[bool] = None,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Get a list of fraud alerts"""
    # Ensure user is admin
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to view fraud alerts")

    query = db.query(FraudAlert)

    if resolved is not None:
        query = query.filter(FraudAlert.is_resolved == resolved)

    alerts = query.offset(skip).limit(limit).all()
    return alerts

@app.post("/api/alerts/{alert_id}/resolve")
def resolve_fraud_alert(
        alert_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Mark a fraud alert as resolved"""
    # Ensure user is admin
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to resolve alerts")

    alert = db.query(FraudAlert).filter(FraudAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.is_resolved = True
    alert.resolved_by = current_user.id
    alert.resolved_at = datetime.now()

    db.commit()

    return {"message": "Alert resolved successfully"}

@app.get("/api/stats/dashboard")
def get_dashboard_stats(
        days: int = Query(30, ge=1, le=365),
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Get dashboard statistics"""
    # Ensure user is admin for full stats
    is_admin = current_user.role == "admin"

    start_date = datetime.now() - timedelta(days=days)

    # Transactions stats
    if is_admin:
        transactions = db.query(Transaction).filter(
            Transaction.created_at >= start_date
        ).all()
    else:
        transactions = db.query(Transaction).filter(
            Transaction.user_id == current_user.id,
            Transaction.created_at >= start_date
        ).all()

    total_transactions = len(transactions)
    total_volume = sum(t.amount for t in transactions) if transactions else 0

    # Fraud stats (admin only)
    fraud_stats = {}
    if is_admin:
        # Get fraud alerts
        alerts = db.query(FraudAlert).filter(
            FraudAlert.created_at >= start_date
        ).all()

        total_alerts = len(alerts)
        resolved_alerts = len([a for a in alerts if a.is_resolved])

        # Get high-risk transactions (fraud score > 0.5)
        high_risk_txns = db.query(Transaction).filter(
            Transaction.created_at >= start_date,
            Transaction.fraud_score >= 0.5
        ).all()

        fraud_stats = {
            "total_alerts": total_alerts,
            "resolved_alerts": resolved_alerts,
            "unresolved_alerts": total_alerts - resolved_alerts,
            "high_risk_transactions": len(high_risk_txns),
            "fraud_rate": (len(high_risk_txns) / total_transactions) if total_transactions > 0 else 0
        }

    return {
        "total_transactions": total_transactions,
        "total_volume": total_volume,
        "average_transaction": total_volume / total_transactions if total_transactions > 0 else 0,
        "fraud_stats": fraud_stats if is_admin else None
    }

@app.get("/")
def root():
    return {"message": "Welcome to MoMoGuard-GH API. Access /docs for API documentation."}