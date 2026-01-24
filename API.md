# API Reference - FAKESPOT

Complete API documentation for FAKESPOT backend.

## Base URL
```
Development: http://localhost:5000
Production: https://your-domain.com
```

## Authentication
Currently no authentication required. Add JWT tokens in production.

---

## Endpoints

### 1. Analyze Account

**Endpoint:** `GET /analyze`

**Description:** Analyzes an Instagram account and returns prediction with detailed metrics.

**Query Parameters:**
| Parameter | Type | Required | Example |
|-----------|------|----------|---------|
| username | string | Yes | `cristiano` |

**Request:**
```bash
GET /analyze?username=cristiano
```

**Response:** `200 OK`
```json
{
  "username": "cristiano",
  "prediction": "REAL",
  "confidence": 0.87,
  "profile_data": {
    "followers": 614000000,
    "following": 1500,
    "posts": 850,
    "bio_length": 45,
    "has_profile_pic": true,
    "is_private": false,
    "account_age_days": 4500,
    "avg_likes": 15000000,
    "avg_comments": 200000,
    "engagement_rate": 0.026
  },
  "reasons": [
    {
      "signal": "Authentic engagement pattern",
      "impact": "low",
      "detail": "Consistent follower growth and engagement rates"
    }
  ],
  "charts": {
    "bar": [
      {"metric": "Followers", "value": 614000000},
      {"metric": "Following", "value": 1500},
      {"metric": "Posts", "value": 850}
    ],
    "radar": [
      {"feature": "Engagement", "value": 45},
      {"feature": "Account Age", "value": 92},
      {"feature": "Profile Completeness", "value": 100},
      {"feature": "Post Activity", "value": 95},
      {"feature": "Network Quality", "value": 88}
    ],
    "line": [
      {"day": "Day 1", "likes": 14950000},
      {"day": "Day 2", "likes": 15000000},
      {"day": "Day 3", "likes": 15050000}
    ]
  }
}
```

**Error Responses:**

400 Bad Request:
```json
{
  "error": "Username is required"
}
```

500 Internal Server Error:
```json
{
  "error": "Failed to analyze account"
}
```

**Examples:**

JavaScript/Fetch:
```javascript
fetch('http://localhost:5000/analyze?username=cristiano')
  .then(res => res.json())
  .then(data => console.log(data))
```

Python:
```python
import requests

response = requests.get('http://localhost:5000/analyze', 
                       params={'username': 'cristiano'})
data = response.json()
print(data)
```

cURL:
```bash
curl "http://localhost:5000/analyze?username=cristiano"
```

---

## Response Structure

### Profile Data
```javascript
profile_data: {
  followers: number,           // Total followers
  following: number,           // Total accounts following
  posts: number,              // Total posts count
  bio_length: number,         // Bio character length
  has_profile_pic: boolean,   // Profile picture exists
  is_private: boolean,        // Account is private
  account_age_days: number,   // Days since account creation
  avg_likes: number,          // Average likes per post
  avg_comments: number,       // Average comments per post
  engagement_rate: number     // Engagement ratio (0-1)
}
```

### Reasons Array
Each reason object contains:
```javascript
{
  signal: string,      // Short summary of the signal
  impact: string,      // 'high' | 'medium' | 'low'
  detail: string       // Detailed explanation
}
```

### Charts Object
Contains data for visualization:
```javascript
charts: {
  bar: [
    { metric: string, value: number }
  ],
  radar: [
    { feature: string, value: number }
  ],
  line: [
    { day: string, likes: number }
  ]
}
```

---

## Prediction Values

| Prediction | Meaning | Confidence Range |
|-----------|---------|-----------------|
| REAL | Authentic account | 0.4 - 1.0 |
| FAKE | Suspicious/Fake account | 0.4 - 1.0 |

Confidence represents how certain the model is about the prediction.

---

## Impact Levels

| Level | Color | Meaning |
|-------|-------|---------|
| high | Red (#ef4444) | Critical signal |
| medium | Orange (#f59e0b) | Important indicator |
| low | Green (#10b981) | Minor consideration |

---

## Fake Detection Signals

### High Impact Signals
- **High follower/following ratio** - Ratio > 3:1
- **Very new account** - Account age < 30 days
- **No profile picture** - Missing profile photo

### Medium Impact Signals
- **Low engagement** - Engagement rate < 1%
- **Empty biography** - No bio description
- **Few posts** - Posts < 5 with many followers

### Low Impact Signals
- **Private account** - Limited visibility
- **Minimal external URLs** - No external links

---

## Rate Limiting

Currently no rate limiting. Recommended in production:
```python
# Using Flask-Limiter
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/analyze')
@limiter.limit("100/hour")
def analyze():
    # Implementation
```

---

## CORS Headers

All endpoints support CORS:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

---

## Data Accuracy

### Model Performance
- **Accuracy**: ~85% on test data
- **Precision**: ~80% for fake detection
- **Recall**: ~90% for fake detection

### Limitations
- Relies on publicly available profile data
- Limited by API rate limits
- Results may vary based on data freshness
- No access to private/deleted accounts

---

## Error Handling

### Common Errors

**Invalid Username:**
- Error code: 400
- Message: "Username is required"
- Solution: Provide a valid username

**Model Not Available:**
- Error code: 500
- Message: "Model not available"
- Solution: Ensure model files are loaded

**Connection Error:**
- Error code: 500
- Message: "Failed to connect to data source"
- Solution: Check internet connection

---

## Caching Strategy

Recommended client-side caching:
```javascript
const cache = new Map();

function getCachedAnalysis(username) {
  if (cache.has(username)) {
    return cache.get(username);
  }
  
  return fetch(`/analyze?username=${username}`)
    .then(res => res.json())
    .then(data => {
      cache.set(username, data);
      // Expire after 1 hour
      setTimeout(() => cache.delete(username), 3600000);
      return data;
    });
}
```

---

## Pagination & Filtering

Currently not implemented. Future endpoints:
```
GET /analyze/batch - Analyze multiple accounts
GET /history - User analysis history
GET /reports/{id} - Fetch saved report
```

---

## Versioning

Current API version: `v1` (implicit)

Future versions:
```
/api/v1/analyze
/api/v2/analyze  (planned)
```

---

## Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Load Testing
```bash
ab -n 1000 -c 10 http://localhost:5000/analyze?username=test
```

---

## Webhook Support

Future feature for real-time notifications:
```json
{
  "event": "analysis_complete",
  "data": { /* analysis result */ },
  "timestamp": "2024-01-21T10:30:00Z"
}
```

---

## SDK & Client Libraries

### JavaScript/Node.js
```javascript
const fakespot = require('fakespot-sdk');

const result = await fakespot.analyze('username');
```

### Python
```python
from fakespot import Client

client = Client()
result = client.analyze('username')
```

---

## Best Practices

1. **Cache Results** - Don't re-analyze same accounts immediately
2. **Error Handling** - Always handle API errors gracefully
3. **Rate Limiting** - Implement client-side throttling
4. **Timeouts** - Set request timeouts (30 seconds recommended)
5. **Logging** - Log all API calls for debugging
6. **Monitoring** - Track API performance metrics

---

## Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized (future) |
| 429 | Too Many Requests (future) |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

---

## Support & Contact

- **Documentation**: See `SETUP.md`
- **Issues**: Check `QUICKSTART.md` troubleshooting
- **Feature Requests**: Open GitHub issue
- **Email**: support@fakespot.io (example)

---

**API Version**: 1.0  
**Last Updated**: January 21, 2024  
**Status**: Stable
