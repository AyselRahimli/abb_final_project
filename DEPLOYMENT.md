# Bank360 Deployment Guide

## 🚀 Quick Deployment

### Option 1: Local Development Setup

1. **Prerequisites**
   ```bash
   # Ensure Python 3.8+ is installed
   python3 --version
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv bank360_env
   source bank360_env/bin/activate  # Linux/Mac
   # OR
   bank360_env\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Application**
   ```bash
   streamlit run bank360_app.py
   ```

5. **Access Application**
   - Open browser to `http://localhost:8501`
   - The app starts with sample data for demonstration

### Option 2: Docker Deployment

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "bank360_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**
   ```bash
   docker build -t bank360 .
   docker run -p 8501:8501 bank360
   ```

### Option 3: Cloud Deployment

#### Streamlit Cloud
1. Upload files to GitHub repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from repository

#### Heroku
1. Create `Procfile`:
   ```
   web: streamlit run bank360_app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Deploy via Heroku CLI or GitHub integration

#### AWS/Azure/GCP
- Use container services (ECS, Container Instances, Cloud Run)
- Or VM-based deployment with reverse proxy

## 🔧 Configuration Options

### Environment Variables
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export STREAMLIT_SERVER_PORT="8501"
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
```

### Production Settings
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f4e79"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f8ff"
textColor = "#262730"
```

## 📊 Performance Optimization

### Memory Optimization
- Adjust sample data size in `generate_sample_data()`
- Implement data pagination for large datasets
- Use Streamlit caching for expensive computations

### Load Balancing
- Use nginx for multiple Streamlit instances
- Implement Redis for session state management
- Consider microservices architecture for scaling

## 🔐 Security Considerations

### Production Security
1. **API Key Management**
   - Use environment variables
   - Implement key rotation
   - Monitor API usage

2. **Data Protection**
   - Enable HTTPS
   - Implement input validation
   - Add rate limiting

3. **Access Control**
   - Add authentication layer
   - Implement role-based access
   - Audit logging

### Compliance
- Ensure GDPR compliance for EU users
- Implement data retention policies
- Add consent management

## 📈 Monitoring & Maintenance

### Health Checks
```python
# Add to app for health monitoring
@st.cache_data
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
```

### Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### Metrics
- Monitor response times
- Track user interactions
- Monitor API call costs
- Alert on errors

## 🔄 Updates & Maintenance

### Version Control
- Use semantic versioning
- Maintain changelog
- Test updates in staging

### Database Integration
- Connect to production databases
- Implement data pipelines
- Add data quality checks

### Feature Flags
- Implement gradual rollouts
- A/B testing capabilities
- Quick feature toggling

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Port Conflicts**
   ```bash
   streamlit run bank360_app.py --server.port=8502
   ```

3. **Memory Issues**
   - Reduce sample data size
   - Clear browser cache
   - Restart application

4. **API Errors**
   - Check API key validity
   - Monitor rate limits
   - Implement retry logic

### Debug Mode
```bash
streamlit run bank360_app.py --logger.level=debug
```

## 📞 Support

For deployment issues:
- Check application logs
- Review system requirements
- Contact support team

---

**Bank360 - Ready for Enterprise Deployment** 🏦✨