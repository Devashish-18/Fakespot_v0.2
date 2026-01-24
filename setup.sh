#!/bin/bash

# FAKESPOT Setup Script

echo "🚀 FAKESPOT Setup Started"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check Node
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 16+"
    exit 1
fi

echo "✓ Python and Node.js detected"

# Backend setup
echo ""
echo "📦 Setting up backend..."
pip install -r requirements.txt
echo "✓ Backend dependencies installed"

# Frontend setup
echo ""
echo "📦 Setting up frontend..."
cd frontend
npm install
echo "✓ Frontend dependencies installed"

# Create .env file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ .env file created"
else
    echo "ℹ️  .env file already exists"
fi

cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Terminal 1: python app.py (Backend)"
echo "2. Terminal 2: cd frontend && npm start (Frontend)"
echo ""
echo "🌐 Open http://localhost:3000 in your browser"
