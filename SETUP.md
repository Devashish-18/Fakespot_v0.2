# FAKESPOT - Instagram Account Authenticity Checker

A modern web application for detecting fake Instagram accounts using advanced AI-powered behavioral analysis. Built with React, Tailwind CSS, and Python Flask.

## Features

✨ **Account Prediction** - Instant REAL/FAKE classification using machine learning
📊 **Profile Metrics** - Visual analysis of followers, engagement, and activity patterns
🔍 **Fake Signal Explanation** - Detailed reasoning for why an account was flagged
📈 **Performance Charts** - Bar charts, radar charts, line charts, and doughnut charts
💾 **Export Reports** - Download analysis as JSON for further investigation
📱 **Responsive Design** - Works seamlessly on desktop, tablet, and mobile

## Tech Stack

### Frontend
- **React 18** - Modern UI library
- **Tailwind CSS** - Utility-first CSS framework
- **React Router** - Client-side routing
- **Recharts** - Interactive charts and graphs
- **Lucide React** - Icon library
- **Axios** - HTTP client

### Backend
- **Python 3.10+**
- **Flask** - Lightweight web framework
- **Flask-CORS** - Cross-origin resource sharing
- **scikit-learn** - Machine learning models
- **NumPy & Pandas** - Data processing

## Project Structure

```
Fakespot_v0.2-main/
├── frontend/                  # React application
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/       # Reusable React components
│   │   │   ├── Navbar.jsx
│   │   │   ├── Footer.jsx
│   │   │   ├── MetricCard.jsx
│   │   │   ├── ChartCard.jsx
│   │   │   └── ReasonBox.jsx
│   │   ├── pages/            # Page components
│   │   │   ├── Home.jsx
│   │   │   ├── Analyzer.jsx
│   │   │   ├── ResultPage.jsx
│   │   │   └── AnalysisPage.jsx
│   │   ├── utils/
│   │   │   ├── api.js        # API calls
│   │   │   └── explanations.js  # Dynamic explanations
│   │   ├── App.jsx
│   │   └── index.js
│   ├── package.json
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── app.py                    # Flask backend
├── requirements.txt          # Python dependencies
├── random_fake.pkl          # Trained Random Forest model
├── decision_fake.pkl        # Trained Decision Tree model
└── README.md
```

## Installation & Setup

### Prerequisites
- Node.js 16+ and npm
- Python 3.10+
- pip (Python package manager)

### Backend Setup

1. **Navigate to project root**
```bash
cd Fakespot_v0.2-main
```

2. **Create virtual environment (optional but recommended)**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Start Flask server**
```bash
python app.py
```
The backend will run on `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install Node dependencies**
```bash
npm install
```

3. **Create .env file** (optional, for custom API URL)
```bash
echo "REACT_APP_API_URL=http://localhost:5000" > .env
```

4. **Start React development server**
```bash
npm start
```
The frontend will open at `http://localhost:3000`

## How It Works

### 1. **Home Page**
   - Clean, modern landing page
   - Hero section with CTA button
   - Feature cards showcasing capabilities
   - How-it-works section with 3-step process

### 2. **Analyzer Section**
   - Username input form
   - Loading state animation
   - Error handling for invalid inputs

### 3. **Result Page**
   - Prediction badge (REAL/FAKE) with confidence percentage
   - Detailed profile metrics in a grid layout
   - "Why this result?" section with reasons and impact levels
   - Export report button
   - Link to detailed analysis

### 4. **Performance Analysis Page**
   - 4 interactive charts:
     - **Bar Chart**: Followers vs Following vs Posts
     - **Radar Chart**: Account health indicators
     - **Line Chart**: Engagement growth trend
     - **Doughnut Chart**: Suspicious score breakdown
   - Auto-generated explanations for each chart
   - Detailed metrics table with status indicators
   - Summary analysis

## API Endpoints

### GET /analyze
Analyzes an Instagram account for fake detection.

**Query Parameters:**
- `username` (string, required): Instagram username

**Response:**
```json
{
  "username": "example_user",
  "prediction": "FAKE" or "REAL",
  "confidence": 0.93,
  "profile_data": {
    "followers": 250,
    "following": 2200,
    "posts": 2,
    "bio_length": 5,
    "has_profile_pic": false,
    "is_private": true,
    "account_age_days": 15,
    "avg_likes": 6,
    "avg_comments": 0,
    "engagement_rate": 0.02
  },
  "reasons": [
    {
      "signal": "High following-to-follower ratio",
      "impact": "high",
      "detail": "Following is extremely high compared to followers"
    }
  ],
  "charts": {
    "bar": [...],
    "radar": [...],
    "line": [...]
  }
}
```

## Key Components

### Navbar
- Sticky navigation with links to all pages
- Mobile responsive menu
- Brand logo and CTA button

### MetricCard
- Displays individual metric with icon
- Color-coded border (primary, secondary, success, etc.)
- Smooth fade-in animation

### ChartCard
- Wraps Recharts components
- Includes auto-generated explanation below chart
- Responsive container

### ReasonBox
- Shows why account was flagged as fake/real
- Color-coded by impact (high=red, medium=yellow, low=green)
- Displays signal, detail, and severity badge

## Fake Detection Signals

The system analyzes multiple signals:

**High Impact:**
- High following-to-follower ratio (>3:1)
- Very new account (<30 days)
- No profile picture

**Medium Impact:**
- Low engagement (<1%)
- Empty biography
- Few posts with many followers

**Low Impact:**
- Private account
- Missing external URL

## Dynamic Explanations

Explanations are generated based on thresholds:
- Following/Followers ratio > 3 → Suspicious
- Account age < 30 days → Suspicious
- Engagement rate < 1% → Suspicious
- Posts < 5 and followers > 500 → Suspicious

## Development

### Building for Production

**Frontend:**
```bash
cd frontend
npm run build
# Creates optimized build in frontend/build/
```

**Backend:**
- No build step required
- Configure Flask for production in app.py

### Environment Variables

**Frontend (.env):**
```
REACT_APP_API_URL=http://your-api-url.com
```

**Backend (app.py):**
- Modify `app.run()` parameters for production settings

## Deployment

### Deploy Backend (Flask)
Use services like:
- Heroku
- Railway
- AWS/Azure/GCP
- DigitalOcean

### Deploy Frontend (React)
Use services like:
- Netlify
- Vercel
- GitHub Pages
- Any static hosting

### Docker Setup (Optional)
Create `Dockerfile`:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

## Troubleshooting

**Port 5000 already in use:**
```bash
python app.py
# Or specify different port
python -m flask run --port 8000
```

**Port 3000 already in use:**
```bash
cd frontend
PORT=3001 npm start
```

**CORS errors:**
- Backend already has CORS enabled
- Check `REACT_APP_API_URL` matches backend URL

**Models not loading:**
- Ensure `random_fake.pkl` and `decision_fake.pkl` exist
- Backend will continue without models (uses fallback)

## Future Enhancements

- [ ] Real Instagram API integration
- [ ] User authentication and history
- [ ] Batch account analysis
- [ ] Advanced reporting (PDF export)
- [ ] Email notifications
- [ ] Community sharing of results
- [ ] Mobile apps (React Native)

## License

MIT License - Feel free to use for personal or commercial projects

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API response structure
3. Check browser console for errors
4. Verify backend is running on correct port

---

**Built with ❤️ | FAKESPOT - Detect Fake Instagram Accounts**
