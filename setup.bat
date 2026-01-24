@echo off
REM FAKESPOT Setup Script for Windows

echo 🚀 FAKESPOT Setup Started

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.10+
    exit /b 1
)

REM Check Node
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found. Please install Node.js 16+
    exit /b 1
)

echo ✓ Python and Node.js detected

REM Backend setup
echo.
echo 📦 Setting up backend...
pip install -r requirements.txt
echo ✓ Backend dependencies installed

REM Frontend setup
echo.
echo 📦 Setting up frontend...
cd frontend
call npm install
echo ✓ Frontend dependencies installed

REM Create .env file
if not exist .env (
    copy .env.example .env
    echo ✓ .env file created
) else (
    echo ℹ️  .env file already exists
)

cd ..

echo.
echo ✅ Setup complete!
echo.
echo 📝 Next steps:
echo 1. Terminal 1: python app.py (Backend)
echo 2. Terminal 2: cd frontend && npm start (Frontend)
echo.
echo 🌐 Open http://localhost:3000 in your browser
