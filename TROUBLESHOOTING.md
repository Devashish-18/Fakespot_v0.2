# Troubleshooting Guide - FAKESPOT

Complete troubleshooting guide for common issues and their solutions.

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [Backend Issues](#backend-issues)
3. [Frontend Issues](#frontend-issues)
4. [API & Integration](#api--integration)
5. [Performance Issues](#performance-issues)
6. [Deployment Issues](#deployment-issues)

---

## Installation Issues

### Issue: Python not found

**Symptoms:**
```
'python' is not recognized as an internal or external command
```

**Solutions:**
1. Install Python from [python.org](https://www.python.org)
2. Add Python to PATH:
   - Windows: System Properties → Environment Variables
   - macOS: Already in PATH
   - Linux: `sudo apt-get install python3`

3. Verify installation:
```bash
python --version  # Should show Python 3.10+
```

---

### Issue: Node.js not found

**Symptoms:**
```
'npm' is not recognized as an internal or external command
```

**Solutions:**
1. Install Node.js from [nodejs.org](https://nodejs.org)
2. Choose LTS version (16+)
3. Verify installation:
```bash
node --version   # Should show v16+
npm --version    # Should show 8+
```

---

### Issue: pip install fails

**Symptoms:**
```
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions:**
```bash
# Update pip
python -m pip install --upgrade pip

# Try installing with specific Python version
python3 -m pip install -r requirements.txt

# Install one by one to identify problem
pip install flask==2.3.2
pip install flask-cors==4.0.0
# etc...
```

---

### Issue: npm install fails

**Symptoms:**
```
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
```

**Solutions:**
```bash
# Clear npm cache
npm cache clean --force

# Try install with legacy flag
npm install --legacy-peer-deps

# Delete node_modules and try again
rm -rf node_modules package-lock.json
npm install

# Check Node version compatibility
node --version  # Should be 16+
```

---

## Backend Issues

### Issue: Port 5000 already in use

**Symptoms:**
```
OSError: [Errno 48] Address already in use
RuntimeError: Address already in use
```

**Solutions:**

**Windows:**
```bash
# Find what's using port 5000
netstat -ano | findstr :5000

# Kill the process
taskkill /PID <PID> /F

# Or use different port in app.py
# app.run(debug=True, host='0.0.0.0', port=5001)
```

**macOS/Linux:**
```bash
# Find process
lsof -i :5000

# Kill it
kill -9 <PID>

# Or use different port
python app.py  # Edit app.py first to change port
```

---

### Issue: Models not loading

**Symptoms:**
```
WARNING: Could not load random_fake.pkl: No such file or directory
WARNING: Could not load decision_fake.pkl: No such file or directory
```

**Solutions:**
1. The app will work without models (uses fallback heuristics)
2. If you have models:
   ```bash
   # Place .pkl files in project root directory
   # Verify they exist:
   ls random_fake.pkl decision_fake.pkl
   ```

---

### Issue: ImportError - Module not found

**Symptoms:**
```
ModuleNotFoundError: No module named 'flask'
ImportError: cannot import name 'CORS'
```

**Solutions:**
```bash
# Reinstall all requirements
pip install -r requirements.txt

# Verify Flask is installed
pip list | grep flask

# Try direct installation
pip install flask==2.3.2 flask-cors==4.0.0
```

---

### Issue: API returning 500 error

**Symptoms:**
```
Internal Server Error
Error: Failed to analyze account
```

**Solutions:**
1. Check backend logs for error details
2. Verify request format:
   ```
   GET /analyze?username=cristiano
   ```
3. Check if username parameter is provided
4. Restart backend:
   ```bash
   python app.py
   ```

---

## Frontend Issues

### Issue: Port 3000 already in use

**Symptoms:**
```
Something is already running on port 3000
```

**Solutions:**

**Windows:**
```bash
# Kill process on port 3000
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or use different port
set PORT=3001
npm start
```

**macOS/Linux:**
```bash
# Kill process
lsof -i :3000
kill -9 <PID>

# Or use different port
PORT=3001 npm start
```

---

### Issue: npm start fails

**Symptoms:**
```
npm ERR! missing script: "start"
react-scripts: command not found
```

**Solutions:**
```bash
# Verify you're in frontend directory
cd frontend

# Reinstall dependencies
npm install

# Clear cache
npm cache clean --force

# Start again
npm start
```

---

### Issue: Blank white page

**Symptoms:**
- App loads but shows nothing
- Browser console errors

**Solutions:**
1. Open browser console (F12)
2. Check for JavaScript errors
3. Common issues:
   ```javascript
   // Check if App.jsx has default export
   export default function App() { }
   
   // Check if index.js is correct
   const root = ReactDOM.createRoot(document.getElementById('root'));
   root.render(<App />);
   ```
4. Restart dev server:
   ```bash
   npm start
   ```

---

### Issue: Styling not applied

**Symptoms:**
- Page looks unstyled
- No colors or layout

**Solutions:**
```bash
# Rebuild Tailwind CSS
npm run build

# Check index.css is imported
// In src/index.js
import './index.css';

# Verify Tailwind config
# tailwind.config.js should exist

# Try clearing cache
rm -rf node_modules/.cache
npm start
```

---

## API & Integration

### Issue: CORS error

**Symptoms:**
```
Access to XMLHttpRequest blocked by CORS policy
No 'Access-Control-Allow-Origin' header
```

**Solutions:**
1. Backend already has CORS enabled in `app.py`
2. Verify API URL in `.env`:
   ```
   REACT_APP_API_URL=http://localhost:5000
   ```
3. Ensure both are running:
   ```bash
   # Terminal 1
   python app.py  # Should run on :5000
   
   # Terminal 2
   npm start      # Should run on :3000
   ```
4. Check app.py has CORS initialization:
   ```python
   from flask_cors import CORS
   app = Flask(__name__)
   CORS(app)
   ```

---

### Issue: API not responding

**Symptoms:**
```
Failed to fetch
Request timeout
Network error
```

**Solutions:**
1. Verify backend is running:
   ```bash
   # Should see "Running on http://localhost:5000"
   python app.py
   ```
2. Test API manually:
   ```bash
   # In terminal
   curl "http://localhost:5000/analyze?username=test"
   
   # Should return JSON
   ```
3. Check network tab in browser (F12)
4. Increase timeout in api.js if needed

---

### Issue: 404 Not Found

**Symptoms:**
```
404 Not Found - /analyze
```

**Solutions:**
1. Verify endpoint exists in app.py:
   ```python
   @app.route('/analyze', methods=['GET'])
   def analyze():
   ```
2. Check URL in api.js:
   ```javascript
   const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
   // Result: http://localhost:5000/analyze
   ```
3. Restart backend

---

### Issue: Username parameter not working

**Symptoms:**
```
Invalid username error
No results returned
```

**Solutions:**
1. Check parameter is passed correctly:
   ```javascript
   // api.js should use 'username' param
   params: { username: username.trim() }
   ```
2. Verify no special characters in username
3. Check backend handles parameter:
   ```python
   username = request.args.get('username', '')
   if not username:
       return {'error': 'Username required'}
   ```

---

## Performance Issues

### Issue: Slow page load

**Symptoms:**
- Takes >3 seconds to load
- Charts render slowly

**Solutions:**
```bash
# Check bundle size
npm run build

# Optimize images
# Reduce animation durations in index.css

# Lazy load components
const AnalysisPage = React.lazy(() => import('./pages/AnalysisPage'));

# Enable browser caching
```

---

### Issue: High memory usage

**Symptoms:**
- Browser becomes slow
- Application crashes

**Solutions:**
1. Close unused tabs
2. Clear browser cache (DevTools → Storage)
3. Restart browser
4. Check for memory leaks in React:
   ```javascript
   useEffect(() => {
     return () => {
       // Cleanup here
     };
   }, []);
   ```

---

### Issue: Charts take long to render

**Symptoms:**
- Analysis page loads slowly
- Charts render with delay

**Solutions:**
```javascript
// In AnalysisPage.jsx, add loading state
if (!data) return <LoadingSpinner />;

// Memoize chart components
const MemoizedChart = React.memo(ChartCard);

// Reduce chart data points
data = data.slice(0, 100);
```

---

## Deployment Issues

### Issue: Deployed app shows blank page

**Symptoms:**
- Works locally, blank on production
- No errors visible

**Solutions:**
1. Check environment variables:
   ```bash
   REACT_APP_API_URL=https://your-backend-url.com
   ```
2. Check backend URL is correct
3. Test API endpoint directly:
   ```bash
   curl "https://your-backend-url.com/analyze?username=test"
   ```
4. Check browser console for errors

---

### Issue: Database connection failed

**Symptoms:**
```
Database connection refused
Could not connect to database
```

**Solutions:**
1. Current implementation doesn't use database (no issue)
2. If adding database later:
   ```bash
   # Verify database is running
   # Check connection string in environment variables
   # Verify credentials are correct
   ```

---

### Issue: 503 Service Unavailable

**Symptoms:**
```
503 Service Unavailable
Backend server is down
```

**Solutions:**
1. Check if backend is running:
   ```bash
   python app.py
   ```
2. Check server logs for errors
3. Verify all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```
4. Restart the application

---

## Verification Steps

### After Installation
- [ ] Backend runs without errors
- [ ] Frontend starts without errors
- [ ] No console errors in browser (F12)
- [ ] API responds to test query:
  ```bash
  curl "http://localhost:5000/analyze?username=test"
  ```

### After Setup
- [ ] Can enter username in analyzer
- [ ] Can click "Analyze" button
- [ ] Can see results on result page
- [ ] Can view charts on analysis page
- [ ] Can export JSON report
- [ ] Can navigate between pages

### After Deployment
- [ ] Frontend loads from deployed URL
- [ ] API URL correctly set
- [ ] Can analyze accounts
- [ ] Charts render properly
- [ ] No CORS errors
- [ ] Performance is acceptable

---

## Getting Help

If you still have issues:

1. **Check Documentation**
   - [SETUP.md](SETUP.md) - Setup guide
   - [QUICKSTART.md](QUICKSTART.md) - Quick start
   - [API.md](API.md) - API reference

2. **Check Browser Console**
   - Press F12
   - Go to Console tab
   - Look for red error messages
   - Take screenshot of full error

3. **Check Terminal Output**
   - Look at Flask server output
   - Look at npm start output
   - Copy full error message

4. **Test Manually**
   ```bash
   # Test API
   curl "http://localhost:5000/analyze?username=test"
   
   # Test npm
   npm list
   
   # Test pip
   pip list
   ```

5. **Restart Everything**
   ```bash
   # Kill all processes
   # Clear caches
   # Reinstall dependencies
   # Start fresh
   ```

---

## Common Error Messages

| Message | Cause | Solution |
|---------|-------|----------|
| Port already in use | Another app on port | Kill process or use different port |
| Module not found | Missing dependency | Run `pip install -r requirements.txt` |
| CORS error | Backend not running | Run `python app.py` |
| 404 Not Found | Wrong endpoint | Check API URL in code |
| Timeout error | Backend too slow | Increase timeout or restart |
| Blank page | React not rendering | Check console for errors |
| No styling | Tailwind not compiled | Run `npm start` again |

---

## Performance Benchmarks

**Expected Performance:**
- Backend startup: 1-2 seconds
- Frontend startup: 3-5 seconds
- API response: <500ms
- Chart render: <1 second
- Page transition: <200ms

If slower, check:
- System resources (CPU, RAM)
- Network speed
- Browser extensions
- Cache issues

---

**Still having issues? Try the nuclear option:**

```bash
# Backend - Remove and reinstall everything
rm -rf venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py

# Frontend - Remove and reinstall everything
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

---

**Last Resort:**
If all else fails, create a fresh install from the original files and follow QUICKSTART.md step by step.

Good luck! 🚀
