# MoMoGuard: Mobile Money Fraud Detection

MoMoGuard-GH is an AI-powered fraud detection system designed specifically for mobile money transactions in Ghana. The system uses machine learning to identify potentially fraudulent transactions in real-time, helping to protect users from scams, unauthorized transactions, and SIM swap fraud.

## Features

- **Real-time Transaction Monitoring**: Analyzes mobile money transactions as they occur
- **Machine Learning-based Risk Assessment**: Uses advanced ML models to score transactions for fraud risk
- **User Behavior Analysis**: Learns normal patterns for each user and flags anomalies
- **SIM Swap Fraud Detection**: Specialized features to identify potential SIM swap attacks
- **Admin Dashboard**: Interface for fraud analysts to review flagged transactions
- **RESTful API**: Easy integration with existing mobile money platforms
- **Containerized Deployment**: Docker-based setup for easy deployment

## Technology Stack

- **Backend**: Python with FastAPI
- **Machine Learning**: Scikit-Learn
- **Database**: PostgreSQL
- **Containerization**: Docker & Docker Compose
- **Development Tools**: Jupyter Notebooks for model development

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- Git for repository management

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/momoguard-gh.git
   cd momoguard-gh
   ```

2. Configure environment variables (optional):
   ```
   # Create a .env file with your settings
   touch .env
   # Add variables like:
   # SECRET_KEY=your_secure_secret_key
   ```

3. Start the application:
   ```
   docker-compose up -d
   ```

4. Access the API documentation:
   ```
   http://localhost:8000/docs
   ```

5. Access the PgAdmin interface (for database management):
   ```
   http://localhost:5050
   Login with:
   Email: admin@momoguard.com
   Password: admin
   ```

## System Architecture

### API Endpoints

- `/api/auth/register` - Register new users
- `/api/auth/login` - Authenticate users
- `/api/transactions/` - Submit and retrieve transactions
- `/api/alerts/` - Manage fraud alerts
- `/api/stats/dashboard` - View system statistics

### Machine Learning Model

The system uses a Random Forest classifier trained on mobile money transaction patterns. Key features include:

- Transaction amount
- Whether the receiver is foreign
- Number of recent transactions by sender
- Average transaction amount
- Transaction frequency changes
- New receiver detection
- Time of day risk assessment

## Development

### Model Training

The machine learning model is developed in Jupyter notebooks located in the `/notebooks` directory. To retrain the model:

1. Navigate to the notebooks directory
2. Run the `fraud_model_training.ipynb` notebook
3. The new model will be saved to the `/models` directory

### Running Tests

```
pytest
```

## Deployment

For production deployment, consider:

1. Using a proper secrets management solution
2. Setting up database backups
3. Implementing proper logging and monitoring
4. Configuring HTTPS with a valid SSL certificate
5. Setting up CI/CD pipelines

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Developed for improving financial security in the growing mobile money ecosystem in Ghana
- Inspired by the need to protect vulnerable populations from financial fraud
