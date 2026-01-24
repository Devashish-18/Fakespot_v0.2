# FAKESPOT - Component & UI Guide

Complete guide to all components and UI sections in FAKESPOT.

## Table of Contents
1. [Navbar Component](#navbar-component)
2. [Home Page](#home-page)
3. [Analyzer Page](#analyzer-page)
4. [Result Page](#result-page)
5. [Analysis Page](#analysis-page)
6. [Footer Component](#footer-component)
7. [UI Components](#ui-components)
8. [Styling Guidelines](#styling-guidelines)

---

## Navbar Component

### Location
`frontend/src/components/Navbar.jsx`

### Features
- Sticky navigation (stays at top while scrolling)
- Logo with gradient background
- Desktop menu items: Home, Features, How It Works, Analyze
- Mobile hamburger menu (responsive)
- "Analyze Now" CTA button
- Smooth transitions on hover

### Usage
```jsx
import Navbar from './components/Navbar';

<Navbar />
```

### Props
None - uses React Router links internally

### Responsive Behavior
- Desktop: Full menu + CTA button
- Mobile: Logo + hamburger icon
- Menu toggles on mobile with smooth animation

---

## Home Page

### Location
`frontend/src/pages/Home.jsx`

### Sections

#### 1. Hero Section
```
[Logo FAKESPOT]

Instagram Account Authenticity Checker
[Subtitle text]

[Analyze Account Button] [Learn More Button]
```

**Features:**
- Large gradient title
- Descriptive subtitle
- Two CTA buttons (primary + secondary)
- Fade-in animation

#### 2. Features Section
```
[4 Feature Cards in Grid]
├── Card 1: Shield icon + Account Prediction
├── Card 2: Chart icon + Profile Metrics
├── Card 3: Zap icon + Fake Signal Explanation
└── Card 4: Download icon + Export Report
```

**Features:**
- Icon (20+ from Lucide)
- Title
- Description
- Hover effects (shadow + border color change)
- Responsive grid (1 col mobile, 4 col desktop)

#### 3. How It Works Section
```
Step 1           Step 2           Step 3
[Circle 1]       [Circle 2]       [Circle 3]
Enter Username → Extract & Predict → View Results
```

**Features:**
- 3-step process
- Numbered circles
- Horizontal connectors
- Responsive (stacked on mobile)

#### 4. CTA Section
```
[Gradient background]
Ready to Start?
[CTA Button]
```

---

## Analyzer Page

### Location
`frontend/src/pages/Analyzer.jsx`

### Layout
```
[Header]
Title: Analyze Instagram Account
Subtitle: Enter any public Instagram username...

[Form Card]
├── Error Alert (conditional)
├── Username Input
│   ├── @ symbol prefix
│   └── Placeholder: "john_doe"
├── Helper text
└── [Analyze Button]

[Info Box]
💡 Tip: Make sure the account is public...

[3 Feature Highlights]
├── 🚀 Instant Analysis
├── 📊 Detailed Insights
└── 🎯 High Accuracy
```

### Form Validation
- Username required
- Error messages displayed
- Loading state on submit
- Disabled input during loading

### States
- **Normal**: Ready for input
- **Loading**: Shows spinner + "Analyzing..."
- **Error**: Shows error message
- **Success**: Redirects to result page

---

## Result Page

### Location
`frontend/src/pages/ResultPage.jsx`

### Layout
```
[Header]
Analysis Result
@username

[Prediction Card]
REAL or FAKE badge (gradient)
93% Confidence
Summary text

[Metrics Grid] (8 columns on desktop)
├── Followers: 250
├── Following: 2200
├── Posts: 2
├── Engagement Rate: 2%
├── Account Age: 15 days
├── Bio Length: 5 chars
├── Profile Picture: Yes/No
└── Private: Yes/No

[Why This Result Section]
├── Reason Box 1 (high impact - red)
├── Reason Box 2 (medium impact - yellow)
└── Reason Box 3 (low impact - green)

[Action Buttons]
├── Export Report (border style)
└── View Detailed Analysis (gradient)

[Analyze Another]
Want to check another account?
[Link to analyzer]
```

### Color Coding
- **REAL**: Green gradient (#10b981)
- **FAKE**: Red/Orange gradient (#ef4444)
- **Confidence**: Large percentage display

### Metric Cards
```
[Border + Background]
📊 Metric Label
12500 unit
```

---

## Analysis Page

### Location
`frontend/src/pages/AnalysisPage.jsx`

### Charts Section

#### 1. Bar Chart
```
Account Metrics Overview

[Y-axis: Value]
|
|     [Bar] [Bar] [Bar]
|__________[================]__________
   Followers Following Posts

💡 Insight: This account follows significantly...
```

#### 2. Radar Chart
```
Account Health Indicators

        [Engagement]
       /      |      \
   [Age]      |      [Completeness]
    /         |         \
[Activity]    |    [Network]
    \         |         /
     \________|________/

💡 Insight: The radar shows multiple concerning...
```

#### 3. Line Chart
```
Engagement Growth Trend

[Value] |
        |     /‾‾\
        |    /    ‾‾‾\
        |___/__________|___
        Day1 Day2 Day3...

💡 Insight: Engagement trend shows minimal...
```

#### 4. Doughnut Chart
```
Suspicious Score Breakdown

    [Profile Signals]
   [Network Signals]
[Engagement Signals]
[Authenticity Score]

💡 Insight: The fake account score is distributed...
```

### Detailed Metrics Table
```
┌──────────────────┬─────────┬────────────┐
│ Metric           │ Value   │ Status     │
├──────────────────┼─────────┼────────────┤
│ Username         │ @user   │ ✓ Provided │
│ Followers        │ 12,500  │ 📈 Good    │
│ Following        │ 2,200   │ ⚠️ Suspicious
│ Posts            │ 2       │ 😴 Inactive│
│ Engagement Rate  │ 2.5%    │ ⚠️ Low     │
│ Account Age      │ 15 days │ ⚠️ New     │
│ Profile Picture  │ No      │ ✗ Missing  │
│ Bio Length       │ 0 chars │ ⚠️ Minimal │
└──────────────────┴─────────┴────────────┘
```

### Summary Box
```
[Blue background box]
📊 Analysis Summary

This account has been classified as FAKE with 93% confidence.
The analysis detected multiple suspicious patterns...
```

---

## Footer Component

### Location
`frontend/src/components/Footer.jsx`

### Layout
```
[Dark background]

[4 Column Grid]
├── FAKESPOT branding + description
├── Product links (Features, How It Works, Pricing)
├── Resource links (Blog, Documentation, Support)
└── Social links (Facebook, Twitter, LinkedIn icons)

[Divider line]
© 2026 FAKESPOT. All rights reserved.
```

### Features
- Company info
- Quick links
- Social media icons (clickable circles)
- Copyright notice
- Responsive (1 col mobile, 4 col desktop)

---

## UI Components

### MetricCard Component

**Location:** `frontend/src/components/MetricCard.jsx`

**Props:**
```jsx
{
  label: string,           // "Followers"
  value: number | string,  // 12500
  unit: string,            // "followers" (optional)
  icon: React.Component,   // Icon from lucide-react
  color: string            // "primary" | "secondary" | "success" | etc
}
```

**Example Usage:**
```jsx
<MetricCard 
  label="Followers" 
  value={12500}
  unit="accounts"
  color="primary"
/>
```

**Colors:**
- `primary` - Blue (#6366f1)
- `secondary` - Pink (#ec4899)
- `success` - Green (#10b981)
- `warning` - Orange (#f59e0b)
- `danger` - Red (#ef4444)

### ChartCard Component

**Location:** `frontend/src/components/ChartCard.jsx`

**Props:**
```jsx
{
  title: string,           // "Account Metrics Overview"
  children: React.Node,    // <ResponsiveContainer>...</ResponsiveContainer>
  explanation: string      // "This account follows significantly..."
}
```

**Features:**
- Title header
- Chart container
- Auto-generated explanation box
- Responsive design

### ReasonBox Component

**Location:** `frontend/src/components/ReasonBox.jsx`

**Props:**
```jsx
{
  reason: {
    signal: string,       // "High following-to-follower ratio"
    impact: string,       // "high" | "medium" | "low"
    detail: string        // Detailed explanation
  }
}
```

**Features:**
- Color-coded by impact
- Icon indicates severity
- Impact badge
- Smooth fade-in animation

---

## Styling Guidelines

### Color Palette
```javascript
// Tailwind colors configured in tailwind.config.js
primary:      #6366f1 (Indigo)
primary-light: #818cf8
primary-dark:  #4f46e5
secondary:    #ec4899 (Pink)
success:      #10b981 (Green)
warning:      #f59e0b (Orange)
danger:       #ef4444 (Red)
dark:         #1f2937 (Dark gray)
dark-light:   #374151 (Medium gray)
```

### Typography
```css
/* Titles */
h1: text-4xl sm:text-5xl font-bold

/* Subtitles */
h2: text-2xl sm:text-3xl font-bold

/* Card titles */
h3: text-lg font-bold

/* Body text */
p: text-gray-600 text-base

/* Small text */
span: text-sm text-gray-500
```

### Spacing
```css
/* Sections */
py-20 /* vertical padding on sections */

/* Cards */
p-6 or p-8 /* internal padding */

/* Gaps between items */
gap-4 or gap-8 /* flex/grid gaps */

/* Margins */
mb-4 or mb-6 /* margin bottom */
```

### Animations
```css
/* Fade in */
.fade-in {
  animation: fadeIn 0.6s ease-in;
}

/* Slide up */
.slide-up {
  animation: slideUp 0.5s ease-out;
}

/* Spinner */
.spinner {
  animation: spin 1s linear infinite;
}

/* Hover effects */
hover:shadow-lg
hover:border-primary
hover:-translate-y-1
transition
```

### Responsive Classes
```jsx
// Mobile first approach
<div className="
  w-full              // mobile width
  md:w-1/2            // medium (768px+) width 50%
  lg:w-1/3            // large (1024px+) width 33%
  
  text-base           // mobile text
  sm:text-lg          // small (640px+)
  md:text-xl          // medium
  lg:text-2xl         // large
  
  flex-col            // mobile stacked
  md:flex-row         // medium side-by-side
  
  grid-cols-1         // mobile single column
  md:grid-cols-2      // medium two columns
  lg:grid-cols-4      // large four columns
">
</div>
```

### Gradient Classes
```jsx
// Predefined gradients in index.css
className="gradient-primary"    // Blue to Pink
className="gradient-success"    // Green
className="gradient-danger"     // Red
```

---

## Page Routing

```javascript
// Routes in App.jsx
"/"                    → Home page
"/analyzer"            → Analyzer form
"/result/:username"    → Result page
"/analysis/:username"  → Analysis page
```

---

## Component Hierarchy

```
App
├── Navbar
├── Routes
│   ├── Home
│   │   ├── Hero Section
│   │   ├── Feature Cards
│   │   ├── How It Works
│   │   └── CTA Section
│   │
│   ├── Analyzer
│   │   ├── Header
│   │   ├── Form Card
│   │   └── Feature Highlights
│   │
│   ├── ResultPage
│   │   ├── Header
│   │   ├── Prediction Card
│   │   ├── MetricCard (x8)
│   │   ├── ReasonBox (x3)
│   │   ├── Action Buttons
│   │   └── Analyze Another
│   │
│   └── AnalysisPage
│       ├── Header
│       ├── ChartCard
│       │   └── Recharts
│       ├── ChartCard
│       │   └── Recharts
│       ├── ChartCard
│       │   └── Recharts
│       ├── ChartCard
│       │   └── Recharts
│       ├── Metrics Table
│       └── Summary Box
│
└── Footer
```

---

## CSS Custom Classes

```css
/* In index.css */

.gradient-primary {
  background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
}

.gradient-success {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

.gradient-danger {
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}

.fade-in {
  animation: fadeIn 0.6s ease-in;
}

.slide-up {
  animation: slideUp 0.5s ease-out;
}

.spinner {
  border: 4px solid rgba(99, 102, 241, 0.1);
  border-top: 4px solid #6366f1;
  animation: spin 1s linear infinite;
}

.chart-container {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
}

.glass-effect {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
}
```

---

## Responsive Breakpoints

```javascript
// Tailwind breakpoints
sm: 640px   // Small devices
md: 768px   // Medium (tablets)
lg: 1024px  // Large (laptops)
xl: 1280px  // Extra large
2xl: 1536px // 4K displays
```

---

## Accessibility Guidelines

- [ ] Use semantic HTML (nav, header, main, footer)
- [ ] Add alt text to images
- [ ] Use ARIA labels where needed
- [ ] Ensure color contrast meets WCAG standards
- [ ] Test with keyboard navigation
- [ ] Use focus states for interactive elements
- [ ] Provide error messages for form validation

---

## Performance Tips

1. **Code Splitting** - Use React.lazy for page components
2. **Image Optimization** - Use WebP with fallbacks
3. **Caching** - Implement localStorage for results
4. **Lazy Loading** - Load charts only when visible
5. **Bundle Size** - Monitor with `npm run build`
6. **Minification** - Automatic in production build

---

**End of Component Guide**
