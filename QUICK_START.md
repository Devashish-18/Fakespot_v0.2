# Quick Start Guide - Instagram Fake Account Detection App

## ✅ What I Fixed

1. **Backend API Issue** - Fixed incomplete `generate_chart_data()` function in `app.py` 
   - Added missing doughnut chart data
   - Function now properly returns all 4 chart types
   - Backend is now running on `http://localhost:5000`

2. **Backend Status**
   - ✅ Flask server is running
   - ✅ `/analyze` endpoint is working
   - ✅ API is returning proper JSON responses
   - ✅ CORS is configured

## 📋 What You Need to Do

### Step 1: Install Node.js (if not already installed)

**Windows:**
- Download and install from: https://nodejs.org/ (recommended: LTS version)
- OR use: `winget install OpenJS.NodeJS`
- Restart your terminal/computer after installation
- Verify: `node --version` and `npm --version`

### Step 2: Install Frontend Dependencies

```bash
cd d:\Workspace\Fakespot_v0.2\Fakespot_v0.2-main\frontend
npm install
```

### Step 3: Start the Applications

**Terminal 1 - Backend (Flask):**
```bash
cd d:\Workspace\Fakespot_v0.2\Fakespot_v0.2-main
python app.py
# Server runs on http://localhost:5000
```

**Terminal 2 - Frontend (React):**
```bash
cd d:\Workspace\Fakespot_v0.2\Fakespot_v0.2-main\frontend
npm start
# App runs on http://localhost:3000
```

### Step 4: Use the App

1. Open http://localhost:3000 in your browser
2. Click "Analyze Now" on the home page
3. Enter any Instagram username (e.g., "cristiano", "instagram", "testuser")
4. Click "Analyze"
5. View results with prediction and key signals
6. Click "View Performance Graphs & Analysis" to see detailed breakdown
7. Click "Export JSON Report" to download results

## 🎯 Architecture

```
┌─────────────────────────────────────────────┐
│   React Frontend (http://localhost:3000)    │
│  - Home Page                                │
│  - Analyzer Form                            │
│  - Result Page                              │
│  - Analysis Page (4 Charts)                 │
└────────────────┬────────────────────────────┘
                 │
                 │ API Calls (axios)
                 │
┌────────────────▼────────────────────────────┐
│  Flask Backend (http://localhost:5000)      │
│  - GET /analyze endpoint                    │
│  - ML Model predictions                     │
│  - Data generation & formatting             │
└─────────────────────────────────────────────┘
```

## 📊 API Endpoint

```
GET http://localhost:5000/analyze?username=USERNAME
```

**Query Parameters (all optional except username):**
- `username` (required) - Instagram username to analyze
- `followers` (optional) - Formatted count (e.g., "40.7M", "1.2K")
- `following` (optional) - Formatted count
- `posts` (optional) - Formatted count

**Response Example:**
```json
{
  "username": "cristiano",
  "prediction": "REAL",
  "confidence": 0.92,
  "profile_data": {
    "followers": 625345200,
    "following": 592,
    "posts": 912,
    ...
  },
  "reasons": [...],
  "charts": {
    "bar": [...],
    "radar": [...],
    "line": [...],
    "doughnut": [...]
  }
}
```

## 🔧 Troubleshooting

### "npm: command not found"
- **Solution**: Install Node.js and restart your terminal
- Check: `node --version` should return a version number

### "Port 5000 already in use"
- **Solution**: The Flask server might be running from a previous session
- Kill it: `lsof -ti:5000 | xargs kill -9` (Mac/Linux) or use Task Manager (Windows)

### "Port 3000 already in use"
- **Solution**: Another app is using port 3000
- Kill it or run React on different port: `PORT=3001 npm start`

### "CORS error in browser console"
- **Solution**: Make sure Flask backend is running on port 5000
- Frontend is set to connect to `http://localhost:5000`

### "Cannot find module" errors during npm install
- **Solution**: Clear npm cache and reinstall
  ```bash
  npm cache clean --force
  npm install
  ```

## 🎨 Files Structure

```
Fakespot_v0.2-main/
├── app.py                          # Flask backend
├── requirements.txt                # Python dependencies
├── frontend/
│   ├── package.json               # npm dependencies
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── index.js
│       ├── App.js
│       ├── pages/
│       │   ├── Home.jsx            # Landing page
│       │   ├── Analyzer.jsx        # Form page
│       │   ├── ResultPage.jsx      # Prediction results
│       │   └── AnalysisPage.jsx    # Detailed analysis + charts
│       ├── components/
│       │   ├── Navbar.jsx
│       │   ├── Footer.jsx
│       │   ├── MetricCard.jsx
│       │   ├── ChartCard.jsx
│       │   └── ReasonBox.jsx
│       └── utils/
│           ├── api.js              # API calls
│           ├── explanations.js     # Smart explanations
│           └── countFormatter.js   # Number formatting
└── templates/                      # Old HTML templates (not used)
```

## ✨ Features

- ✅ Beautiful modern UI with Tailwind CSS
- ✅ 4 interactive charts (Bar, Radar, Line, Doughnut)
- ✅ Smart data-driven explanations
- ✅ Human-readable number formatting (40.7M, 1.2K)
- ✅ JSON export functionality
- ✅ FAKE/REAL prediction with confidence
- ✅ Mobile-responsive design
- ✅ No model comparison visible
- ✅ Professional styling with emojis for clarity

## 📚 Documentation

- `IMPLEMENTATION_COMPLETE.md` - Complete feature overview
- `UI_UX_IMPROVEMENTS.md` - Design improvements detail
- `USER_JOURNEY.md` - User flow documentation
- `QUICK_REFERENCE.md` - Quick function reference

---

**Ready to go! 🚀**

Once Node.js is installed and you run both servers, visit http://localhost:3000
