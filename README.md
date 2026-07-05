# Software Engineering for Machine Learning (SE4ML)

Welcome to the **Software Engineering for Machine Learning** course repository! This repository contains course materials, assignments, and resources for learning best practices in building production-quality machine learning systems.

## 📚 Course Overview

Software Engineering for Machine Learning focuses on the principles, practices, and tools needed to develop, deploy, and maintain ML systems in production environments. This course bridges the gap between data science and software engineering, emphasizing code quality, testing, reproducibility, and scalability.

## 📋 Course Structure

### Assignments
This course includes **2 comprehensive assignments** designed to reinforce key SE4ML concepts:

- **Assignment 1** - Foundation concepts and basic ML pipeline practices
  - Loan Default Prediction System
  - Demonstrates Pipe-and-Filter and Microservice patterns
  - Covers data preprocessing, model training, and REST API deployment
  
- **Assignment 2** - Advanced practices including testing, deployment, and monitoring

### Webinars
Throughout the course, **4 interactive webinars** provide deep dives into specialized topics:

1. **Webinar 1** - ML Systems Architecture and Design Patterns
2. **Webinar 2** - Testing and Validation Strategies for ML
3. **Webinar 3** - CI/CD Pipelines for Machine Learning
4. **Webinar 4** - Monitoring, Maintenance, and MLOps Best Practices

## 🎯 Key Topics

- ML system design and architecture
- Code quality and best practices
- Automated testing and validation
- Continuous Integration/Continuous Deployment (CI/CD)
- Model versioning and reproducibility
- Monitoring and observability
- Ethical AI and responsible ML
- Microservices for ML deployment
- Data quality and leakage prevention

## 📁 Repository Structure

```
SE4ML/
├── assignment_1/                    # Loan Default Prediction System
│   ├── loan_default_pipeline.py     # ML training pipeline
│   ├── loan_default_service.py      # Flask microservice for predictions
│   ├── loan_default_model.joblib    # Trained model artifact
│   ├── sample_request.json          # Example API requests
│   ├── loan_default_architecture.png # System architecture diagram
│   ├── Group_40.docx                # Assignment submission document
│   └── README.md                    # Assignment 1 documentation
├── assignment_2/                    # (Coming soon)
├── webinars/                        # Webinar resources and notes
└── README.md                        # This file
```

## 🚀 Getting Started

### Quick Start with Assignment 1

```bash
# 1. Navigate to assignment folder
cd assignment_1

# 2. Install dependencies
pip install scikit-learn pandas numpy flask joblib

# 3. Train the model
python loan_default_pipeline.py loan_default_dataset.csv

# 4. Start the microservice
python loan_default_service.py

# 5. Make predictions (in another terminal)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2019,
    "loan_amount": 116500,
    "Credit_Score": 758,
    "income": 1740
  }'
```

For detailed assignment instructions, see [Assignment 1 README](./assignment_1/README.md)

## 💻 Requirements

- Python 3.8+
- Familiarity with machine learning fundamentals
- Basic software development knowledge
- Git for version control

## 📚 Core Concepts

### Design Patterns Covered

- **Pipe-and-Filter Pattern**: Data processing pipelines with sequential transformations
- **Microservice Pattern**: Model serving via REST APIs
- **Factory Pattern**: Model instantiation and selection
- **Strategy Pattern**: Pluggable ML algorithms

### Best Practices

- ✅ Data leakage prevention and validation
- ✅ Fairness and ethical AI considerations
- ✅ Handling imbalanced datasets
- ✅ Model evaluation and metric selection
- ✅ Clean code and documentation
- ✅ Reproducibility and versioning
- ✅ API design for ML services

## 📊 Assignment 1: Loan Default Prediction System

A complete end-to-end ML system demonstrating:

- **Data Processing**: Handling missing values, categorical encoding, and feature scaling
- **Model Training**: Multiple classifier comparison (Logistic Regression, Random Forest, Histogram Gradient Boosting)
- **Model Evaluation**: Comprehensive metrics including PR-AUC for imbalanced data
- **Model Serving**: Flask microservice with REST API endpoints
- **Architecture**: Clean separation between training pipeline and inference service

**Key Features**:
- Prevents data leakage through careful feature selection
- Addresses fairness by removing protected attributes
- Handles class imbalance with balanced class weights
- Provides both single and batch prediction capabilities
- Includes health checks and error handling

## 🔧 Technologies & Tools

- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Web Framework**: Flask
- **Model Serialization**: joblib
- **Version Control**: Git/GitHub
- **Development**: Python 3.8+

## 📝 Documentation Standards

All code includes:
- Clear docstrings following PEP 257
- Inline comments for complex logic
- Type hints for function parameters
- Exception handling and logging

## 📞 Support & Resources

- **Assignment Documentation**: See individual assignment READMEs
- **Webinar Materials**: Check the `webinars/` directory
- **Course Platform**: (Link to course portal)
- **Office Hours**: (Schedule and contact)

## 🤝 Contributing

This is a course repository. For improvements or suggestions:
1. Create an issue describing the improvement
2. Fork and create a feature branch
3. Submit a pull request with detailed description

## 📝 Notes for Students

- All code should follow PEP 8 style guidelines
- Include proper documentation and comments
- Ensure reproducibility with fixed random seeds
- Write unit tests for critical components
- Submit assignments according to deadlines
- Review feedback from previous assignments

## 📜 License

This course material is provided for educational purposes as part of **AIMLCZG546 - Software Engineering for Machine Learning**.

---

## 📅 Course Timeline

| Week | Focus | Deliverable |
|------|-------|------------|
| 1-2 | ML System Design & Architecture | Webinar 1 |
| 3-4 | Assignment 1: ML Pipeline & Microservice | Assignment 1 Due |
| 5-6 | Testing & Validation Strategies | Webinar 2 |
| 7-8 | CI/CD for ML Systems | Webinar 3 |
| 9-10 | Assignment 2: Advanced Topics | Assignment 2 Due |
| 11-12 | MLOps & Monitoring | Webinar 4 |

---

**Last Updated:** July 2026  
**Course Code:** AIMLCZG546  
**Language:** Python  
**Repository Owner:** bitswilp
