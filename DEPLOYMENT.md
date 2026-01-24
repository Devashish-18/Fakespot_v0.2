# Deployment Guide - FAKESPOT

Complete guide to deploy FAKESPOT to production environments.

## Table of Contents
1. [Local Production Build](#local-production-build)
2. [Deploy to Railway](#deploy-to-railway)
3. [Deploy to Render](#deploy-to-render)
4. [Deploy to Vercel](#deploy-to-vercel)
5. [Deploy to Netlify](#deploy-to-netlify)
6. [Docker Deployment](#docker-deployment)
7. [Environment Configuration](#environment-configuration)

---

## Local Production Build

### Build Frontend
```bash
cd frontend
npm run build
```
Creates optimized production build in `frontend/build/`

### Configure Flask for Production
Edit `app.py`:
```python
if __name__ == '__main__':
    # Development
    app.run(debug=True, host='0.0.0.0', port=5000)
    
    # Production (commented)
    # app.run(debug=False, host='0.0.0.0', port=5000)
```

Change to:
```python
if __name__ == '__main__':
    import os
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
```

---

## Deploy to Railway

Railway is the easiest option for full-stack deployment.

### Step 1: Install Railway CLI
```bash
npm i -g @railway/cli
```

### Step 2: Login to Railway
```bash
railway login
```

### Step 3: Create Project
```bash
railway init
```
Select "Flask" when prompted.

### Step 4: Link to Git
```bash
railway link
```

### Step 5: Configure Environment
Create `railway.json`:
```json
{
  "buildCommand": "pip install -r requirements.txt",
  "startCommand": "python app.py"
}
```

### Step 6: Deploy
```bash
git push origin main
```
or
```bash
railway up
```

### Step 7: Set Up Frontend
```bash
railway add
# Select Next.js or static site
cd frontend
npm run build
```

---

## Deploy to Render

Great for both backend and frontend.

### Backend (Flask)

1. **Push to GitHub**
   - Commit and push code to GitHub

2. **Create New Service**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repo

3. **Configure**
   - Name: `fakespot-api`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Plan: Free

4. **Set Environment**
   - Add `FLASK_ENV=production`

5. **Deploy**
   - Click "Deploy"
   - Get your API URL from dashboard

### Frontend (React)

1. **Build Locally**
   ```bash
   cd frontend
   npm run build
   ```

2. **Create Static Site**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Static Site"
   - Connect your GitHub repo

3. **Configure**
   - Name: `fakespot`
   - Publish Directory: `frontend/build`
   - Build Command: `npm install && npm run build`

4. **Environment**
   ```
   REACT_APP_API_URL=https://fakespot-api.onrender.com
   ```

5. **Deploy**
   - Click "Deploy"

---

## Deploy to Vercel

Perfect for React frontend.

### Frontend Deployment

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Deploy**
   ```bash
   cd frontend
   vercel
   ```

3. **Configure**
   - Project name: `fakespot`
   - Framework: `Create React App`
   - Root directory: `./`

4. **Environment Variables**
   ```
   REACT_APP_API_URL=https://your-backend-url.com
   ```

5. **Done!**
   - Vercel provides a URL for your frontend

---

## Deploy to Netlify

Alternative React hosting option.

### Frontend Deployment

1. **Build Locally**
   ```bash
   cd frontend
   npm run build
   ```

2. **Connect to Netlify**
   - Go to [netlify.com](https://netlify.com)
   - Click "Add new site" → "Deploy manually"
   - Drag `frontend/build` folder

3. **Or Connect Git**
   - Link your GitHub repo
   - Set build command: `npm run build`
   - Set publish directory: `frontend/build`

4. **Environment Variables**
   - Go to Site Settings → Build & Deploy
   - Add `REACT_APP_API_URL=your-backend-url`

5. **Deploy**
   - Push to GitHub and Netlify auto-deploys

---

## Docker Deployment

### Create Docker Images

#### Backend Dockerfile
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_ENV=production

CMD ["python", "app.py"]
```

#### Frontend Dockerfile
```dockerfile
# frontend/Dockerfile
FROM node:16-alpine as build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./random_fake.pkl:/app/random_fake.pkl
      - ./decision_fake.pkl:/app/decision_fake.pkl

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://backend:5000

  nginx:
    image: nginx:alpine
    ports:
      - "8080:80"
    depends_on:
      - backend
```

### Run with Docker Compose
```bash
docker-compose up -d
```

---

## Environment Configuration

### Backend Variables
```bash
# app.py reads these:
FLASK_ENV=production
PORT=5000
DEBUG=False
```

### Frontend Variables
```bash
# frontend/.env.production
REACT_APP_API_URL=https://api.yourdomian.com
REACT_APP_ENV=production
```

### GitHub Actions (Auto-Deploy)
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to Render
        run: |
          curl https://api.render.com/deploy/srv-xxxxx?key=${{ secrets.RENDER_DEPLOY_KEY }}
```

---

## Post-Deployment Checklist

- [ ] Frontend loads without errors
- [ ] API calls work (check network tab)
- [ ] Analyzer accepts username input
- [ ] Results display correctly
- [ ] Charts render properly
- [ ] Export JSON works
- [ ] Mobile responsive
- [ ] No CORS errors
- [ ] Loading states work
- [ ] Error handling functions

---

## Monitoring & Maintenance

### Monitor Performance
- Render: Dashboard > Metrics
- Railway: Dashboard > Metrics
- Vercel: Analytics tab

### Update Dependencies
```bash
# Check for updates
npm outdated
pip list --outdated

# Update
npm update
pip install --upgrade -r requirements.txt
```

### View Logs
```bash
# Render
render logs -s your-service-id

# Railway
railway logs

# Local
tail -f app.log
```

---

## Troubleshooting Deployment

| Issue | Solution |
|-------|----------|
| CORS Error | Add `REACT_APP_API_URL` env var with correct backend URL |
| 404 Not Found | Check `REACT_APP_API_URL` matches deployed backend |
| Build fails | Ensure Node 16+, Python 3.10+ in build environment |
| Slow response | Add caching, optimize database queries |
| Out of memory | Increase dyno size, optimize image uploads |

---

## Cost Estimates

| Service | Frontend | Backend | Total |
|---------|----------|---------|-------|
| Vercel + Render | Free | $7/mo | $7/mo |
| Railway | $5/mo | $5/mo | $10/mo |
| Heroku | Free | $7/mo | $7/mo |
| AWS | $1/mo | $5/mo | $6/mo |

---

## Support Links

- [Railway Docs](https://docs.railway.app)
- [Render Docs](https://render.com/docs)
- [Vercel Docs](https://vercel.com/docs)
- [Netlify Docs](https://docs.netlify.com)
- [Docker Docs](https://docs.docker.com)

---

**Successfully deployed? Share your experience!**
