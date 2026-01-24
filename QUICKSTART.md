# 🚀 Quick Start Guide - FAKESPOT

Get the Instagram Authenticity Checker up and running in minutes!

## Prerequisites
- **Python 3.10+** - Download from [python.org](https://www.python.org)
- **Node.js 16+** - Download from [nodejs.org](https://nodejs.org)
- **Git** (optional) - For cloning/version control

## One-Command Setup (Windows)

```bash
setup.bat
```

## One-Command Setup (macOS/Linux)

```bash
chmod +x setup.sh
./setup.sh
```

## Manual Setup

### Step 1: Install Backend Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies
```bash
cd frontend
npm install
```

### Step 3: Start Backend (Terminal 1)
```bash
cd .. (if in frontend folder)
python app.py
```
✅ Backend running at `http://localhost:5000`

### Step 4: Start Frontend (Terminal 2)
```bash
cd frontend
npm start
```
✅ Frontend opens at `http://localhost:3000`

## 🎯 Usage

1. **Open Home Page** - Browse features and how it works
2. **Click "Analyze Account"** - Go to analyzer
3. **Enter Instagram Username** - Type any public account username
4. **View Results** - See prediction, confidence, and metrics
5. **Explore Analysis** - Click "View Detailed Analysis" for charts
6. **Export Report** - Download results as JSON

## 📊 Example Usernames to Test
- `cristiano` (Popular account)
- `billgates` (Well-established account)
- `randomuser123` (Typical account)

## 🔧 Configuration

### API URL
Edit `frontend/.env`:
```env
REACT_APP_API_URL=http://localhost:5000
```

### Backend Port
Edit `app.py` last line:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📁 Project Structure

```
Fakespot_v0.2-main/
├── frontend/          # React UI
├── app.py            # Flask Backend
├── requirements.txt  # Python packages
├── SETUP.md         # Full documentation
└── README.md        # Original readme
```

## ✨ Features

| Feature | Location |
|---------|----------|
| Input Form | Analyzer page |
| Prediction | Result page |
| Metrics | Result cards |
| Charts | Analysis page |
| Export | Result page button |

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Change port in `app.py` or `npm start` |
| Module not found | Run `pip install -r requirements.txt` or `npm install` |
| CORS error | Ensure backend running on correct port |
| Models not loading | App works without models (uses fallback) |

## 🌐 Deployment

### Free Hosting Options:
- **Backend**: Railway, Render, Heroku
- **Frontend**: Vercel, Netlify, GitHub Pages

See [SETUP.md](SETUP.md) for detailed deployment guide.

## 📚 Documentation

- [Full Setup Guide](SETUP.md)
- [API Documentation](#api-endpoints)
- [Component Guide](#components)

## 💡 Tips

- **Add to favorites** - Save to bookmark for quick access
- **Test multiple accounts** - Try different account types
- **Check explanations** - Read why signals are suspicious
- **Compare results** - Export and compare different accounts

## 🆘 Need Help?

1. Check [SETUP.md](SETUP.md) troubleshooting section
2. Verify both servers are running
3. Check browser console for errors (F12)
4. Check backend terminal for Flask errors

## 🎓 Learn More

- **React**: [react.dev](https://react.dev)
- **Tailwind**: [tailwindcss.com](https://tailwindcss.com)
- **Flask**: [flask.palletsprojects.com](https://flask.palletsprojects.com)
- **Recharts**: [recharts.org](https://recharts.org)

---

**Enjoy analyzing Instagram accounts! 🎉**

Need to make changes? Edit components in `frontend/src/` and files in root for backend.
