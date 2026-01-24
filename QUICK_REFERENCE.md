# Quick Reference - Human-Readable Count Functions

## JavaScript Functions

### `parseCount(value)` - String/Number → Number
```javascript
import { parseCount } from './utils/countFormatter';

parseCount("218K")          // 218000
parseCount("40.7M")         // 40700000
parseCount("1.2B")          // 1200000000
parseCount("950")           // 950
parseCount(950)             // 950
```

### `formatCount(number)` - Number → String
```javascript
import { formatCount } from './utils/countFormatter';

formatCount(218000)         // "218K"
formatCount(40700000)       // "40.7M"
formatCount(1200000000)     // "1.2B"
formatCount(950)            // "950"
```

### Safe Variants with Fallback
```javascript
import { parseCountSafe, formatCountSafe } from './utils/countFormatter';

parseCountSafe("invalid", 0)     // 0 (not error)
formatCountSafe(218000)          // "218K" (handles errors)
```

### Validation
```javascript
import { isValidCountFormat } from './utils/countFormatter';

isValidCountFormat("218K")       // true
isValidCountFormat("invalid")    // false
```

---

## Python Functions

### `parse_count(value)` - String/Number → Integer
```python
from app import parse_count

parse_count("218K")          # 218000
parse_count("40.7M")         # 40700000
parse_count("1.2B")          # 1200000000
parse_count("950")           # 950
parse_count(950)             # 950
```

**Error on invalid input:**
```python
try:
    parse_count("invalid")
except ValueError:
    # "Invalid number format"
    pass
```

---

## API Usage

### Accept Formatted Counts
```bash
GET /analyze?username=cristiano&followers=40.7M&following=10.5M&posts=1.2k
```

### Still Works with Plain Numbers
```bash
GET /analyze?username=cristiano&followers=40700000&following=10500000&posts=1200
```

---

## Supported Formats

| Input | Output (formatCount) | Parse Value |
|-------|----------------------|-------------|
| "218K" | - | 218,000 |
| "40.7M" | - | 40,700,000 |
| "1.2B" | - | 1,200,000,000 |
| "950" | - | 950 |
| 218000 | "218K" | 218,000 |
| 40700000 | "40.7M" | 40,700,000 |
| 1200000000 | "1.2B" | 1,200,000,000 |
| 950 | "950" | 950 |

---

## In Components

### MetricCard
```jsx
<MetricCard 
  label="Followers"
  value={40700000}
  formatAsCount={true}  // Shows "40.7M"
/>
```

### Charts
```jsx
const CustomTooltip = ({ payload }) => {
  const value = payload[0].value;
  return <div>{formatCountSafe(value)}</div>;
};
```

---

## Error Messages

| Input | Error |
|-------|-------|
| "invalid" | Invalid number format |
| "12.34.56K" | Invalid number format |
| "" | Invalid number format |
| "40T" | Invalid number format |
| -100 (formatCount) | Invalid number format |

---

## Key Rules

✅ Case-insensitive: K/k, M/m, B/b
✅ Decimals allowed: 40.7M, 1.5K
✅ Plain numbers work: 950 → 950
✅ Smart formatting: No ".0" suffix
✅ Rounds internally to integer

---

## Common Patterns

### Accept user input, format for display
```javascript
const userInput = "40.7M";
const numericValue = parseCount(userInput);        // 40700000
// ... use numericValue in calculations ...
const displayText = formatCount(numericValue);     // "40.7M"
```

### Safe parsing with error handling
```javascript
try {
  const value = parseCount(userInput);
} catch (e) {
  alert(`Invalid format: ${e.message}`);
}

// Or use safe variant
const value = parseCountSafe(userInput, 0);
```

### Format large numbers in UI
```jsx
// Metric card
<MetricCard value={40700000} formatAsCount={true} />

// Chart tooltip
const displayValue = formatCountSafe(40700000);  // "40.7M"
```

---

For detailed documentation, see `COUNT_FORMATTER_GUIDE.md` and `HUMAN_READABLE_COUNTS_IMPLEMENTATION.md`
