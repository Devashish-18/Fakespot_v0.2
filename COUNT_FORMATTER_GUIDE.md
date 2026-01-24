# Count Formatter Utility - Usage Guide

This utility provides functions to parse and format large numbers in human-readable format (K, M, B).

## Functions

### `parseCount(value)` - Frontend (JavaScript)
Converts human-readable count format to numeric value.

**Examples:**
```javascript
import { parseCount } from './utils/countFormatter';

parseCount("218K")        // → 218000
parseCount("40.7M")       // → 40700000
parseCount("1.2B")        // → 1200000000
parseCount("950")         // → 950
parseCount(950)           // → 950
parseCount("10.5k")       // → 10500 (case-insensitive)
```

**Error Handling:**
```javascript
try {
  parseCount("invalid");  // Throws: "Invalid number format"
  parseCount("12.34.56K"); // Throws: "Invalid number format"
  parseCount("");         // Throws: "Invalid number format"
} catch (error) {
  console.log(error);
}
```

**Safe Version with Fallback:**
```javascript
import { parseCountSafe } from './utils/countFormatter';

parseCountSafe("218K", 0)       // → 218000
parseCountSafe("invalid", 100)  // → 100 (fallback value)
parseCountSafe("", 500)         // → 500
```

**Validation:**
```javascript
import { isValidCountFormat } from './utils/countFormatter';

isValidCountFormat("218K")    // → true
isValidCountFormat("40.7M")   // → true
isValidCountFormat("invalid")  // → false
```

---

### `formatCount(number)` - Frontend (JavaScript)
Converts numeric value to human-readable format.

**Examples:**
```javascript
import { formatCount } from './utils/countFormatter';

formatCount(218000)       // → "218K"
formatCount(40700000)     // → "40.7M"
formatCount(1200000000)   // → "1.2B"
formatCount(950)          // → "950"
formatCount(1000)         // → "1K"
formatCount(1500)         // → "1.5K"
```

**Error Handling:**
```javascript
try {
  formatCount(-100);      // Throws: "Invalid number format"
  formatCount("invalid");  // Throws: "Invalid number format"
} catch (error) {
  console.log(error);
}
```

**Safe Version with Fallback:**
```javascript
import { formatCountSafe } from './utils/countFormatter';

formatCountSafe(218000, "N/A")    // → "218K"
formatCountSafe("invalid", "N/A") // → "N/A"
```

---

### `parse_count(value)` - Backend (Python)
Converts human-readable count format to numeric value in Flask API.

**Examples:**
```python
from app import parse_count

parse_count("218K")       # → 218000
parse_count("40.7M")      # → 40700000
parse_count("1.2B")       # → 1200000000
parse_count("950")        # → 950
parse_count(950)          # → 950
parse_count("10.5k")      # → 10500 (case-insensitive)
```

**API Usage:**
```bash
# API accepts optional formatted count parameters
GET /analyze?username=cristiano&followers=218K&following=40.7M&posts=1.2k

# Still works with plain numbers
GET /analyze?username=cristiano&followers=218000&following=40700000&posts=1200
```

**Error Handling:**
```python
try:
    parse_count("invalid")     # Raises ValueError: "Invalid number format"
    parse_count("12.34.56K")   # Raises ValueError: "Invalid number format"
except ValueError as e:
    print(f"Error: {e}")
```

---

## Features

### Supported Formats
- **K** - Thousands (1K = 1,000)
- **M** - Millions (1M = 1,000,000)
- **B** - Billions (1B = 1,000,000,000)
- Decimal support: 40.7M, 1.2K, etc.
- Case-insensitive: "218k" = "218K"
- Plain numbers: "950" → 950

### Display in UI Components

**MetricCard with Formatting:**
```jsx
<MetricCard 
  label="Followers" 
  value={formatCount(218000)}  // Shows "218K"
  formatAsCount={true}
/>
```

**Charts with Custom Tooltips:**
```jsx
// Charts automatically format large numbers in tooltips
const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const value = payload[0].value;
    const displayValue = formatCountSafe(value);
    return <div>{displayValue}</div>;
  }
};
```

---

## Implementation Details

### Rounding Behavior
- Numbers < 1,000 display as-is: "950"
- Whole number results show no decimals: "218K" (not "218.0K")
- Results with decimals show 1 decimal place: "40.7M"
- Internal calculations are rounded to nearest integer to avoid floating-point errors

### Validation Rules
1. Input must be a valid number or string
2. Suffix must be K, M, or B (or lowercase)
3. Numbers must be non-negative
4. Decimals allowed before suffix: "40.7M" ✓, "40M.7" ✗
5. No multiple decimal points: "40.5.7M" ✗
6. No spaces within number: "40 .7M" ✗ (leading/trailing spaces are trimmed)

---

## Browser & Node Compatibility

- **Frontend (JavaScript):** All modern browsers, Node.js 12+
- **Backend (Python):** Python 3.6+
- **Dependencies:** None (pure JavaScript/Python)

---

## Error Messages

| Error | Cause | Example |
|-------|-------|---------|
| "Invalid number format" | Non-numeric input | `parseCount("abc")` |
| "Invalid number format" | Multiple decimal points | `parseCount("40.5.7M")` |
| "Invalid number format" | Invalid suffix | `parseCount("40T")` |
| "Invalid number format" | Empty string | `parseCount("")` |
| "Invalid number format" | Negative number (formatCount only) | `formatCount(-100)` |

---

## Examples in Application

### ResultPage - Display Metric Cards
```jsx
import { formatCountSafe } from '../utils/countFormatter';

<MetricCard 
  label="Followers" 
  value={formatCountSafe(data.profile_data.followers)}
  formatAsCount={true}
/>
```

### AnalysisPage - Charts with Formatted Tooltips
```jsx
import { formatCountSafe } from '../utils/countFormatter';

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const value = payload[0].value;
    const displayValue = value > 1000 ? formatCountSafe(value) : value;
    return <div><strong>{payload[0].name}:</strong> {displayValue}</div>;
  }
};
```

### Backend API - Accept Formatted Input
```bash
# User provides counts in readable format
curl "http://localhost:5000/analyze?username=cristiano&followers=40.7M&following=10.5M&posts=1.2k"

# Backend parses and uses numeric values for analysis
# API response still contains numeric values internally
```

---

## Testing

### Frontend Test Examples
```javascript
// Valid formats
["218K", "40.7M", "1.2B", "950", "1k", "10.5m", "2B"].forEach(val => {
  console.log(`${val} → ${formatCount(parseCount(val))}`);
});

// Invalid formats
["invalid", "12.34.56K", "", "40T"].forEach(val => {
  try {
    parseCount(val);
  } catch (e) {
    console.log(`${val}: ${e.message}`);
  }
});
```

---

## Performance

- **parseCount:** O(1) - regex match + arithmetic
- **formatCount:** O(1) - conditional logic + division
- **No dependencies:** Both functions are lightweight and dependency-free
- **Memory:** Minimal - only temporary numeric values

---

## Future Enhancements

Possible additions (not currently implemented):
- Support for "T" (Trillion) suffix
- Locale-specific formatting (commas, periods)
- Custom decimal precision configuration
- Negative number support
- Scientific notation support
