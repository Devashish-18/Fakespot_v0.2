/**
 * Converts human-readable count format to numeric value
 * Examples: "218K" → 218000, "40.7M" → 40700000, "1.2B" → 1200000000, "950" → 950
 * 
 * @param {string|number} value - The value to parse (e.g., "218K", "40.7M", "950")
 * @returns {number} The parsed numeric value
 * @throws {Error} If the format is invalid
 */
export const parseCount = (value) => {
  // If already a number, return it
  if (typeof value === 'number') {
    if (isNaN(value)) {
      throw new Error('Invalid number format');
    }
    return Math.abs(value); // Ensure positive
  }

  // Convert to string and trim whitespace
  const trimmed = String(value).trim();

  if (!trimmed) {
    throw new Error('Invalid number format');
  }

  // Match pattern: optional digits, optional decimal, digits, optional suffix (K, M, B)
  const regex = /^(\d+\.?\d*|\.\d+)([kmb]?)$/i;
  const match = trimmed.match(regex);

  if (!match) {
    throw new Error('Invalid number format');
  }

  const numValue = parseFloat(match[1]);
  const suffix = (match[2] || '').toLowerCase();

  if (isNaN(numValue)) {
    throw new Error('Invalid number format');
  }

  // Apply multiplier based on suffix
  const multipliers = {
    k: 1_000,
    m: 1_000_000,
    b: 1_000_000_000,
  };

  const result = numValue * (multipliers[suffix] || 1);
  return Math.round(result); // Round to nearest integer to avoid floating point issues
};

/**
 * Converts numeric value to human-readable format
 * Examples: 218000 → "218K", 40700000 → "40.7M", 1200000000 → "1.2B", 950 → "950"
 * 
 * @param {number} value - The numeric value to format
 * @returns {string} The formatted string
 * @throws {Error} If the value is not a valid number
 */
export const formatCount = (value) => {
  const num = typeof value === 'string' ? parseFloat(value) : value;

  if (isNaN(num) || num < 0) {
    throw new Error('Invalid number format');
  }

  // If less than 1000, return as is
  if (num < 1_000) {
    return String(Math.round(num));
  }

  // Define thresholds
  const thresholds = [
    { value: 1_000_000_000, suffix: 'B' },
    { value: 1_000_000, suffix: 'M' },
    { value: 1_000, suffix: 'K' },
  ];

  for (const threshold of thresholds) {
    if (num >= threshold.value) {
      const divided = num / threshold.value;
      // If the result is a whole number or close to it, show without decimals
      if (Math.abs(divided - Math.round(divided)) < 0.05) {
        return `${Math.round(divided)}${threshold.suffix}`;
      }
      // Otherwise show one decimal place
      return `${(Math.round(divided * 10) / 10).toFixed(1)}${threshold.suffix}`;
    }
  }

  return String(Math.round(num));
};

/**
 * Validates if a value is in valid count format
 * @param {string|number} value - Value to validate
 * @returns {boolean} True if valid, false otherwise
 */
export const isValidCountFormat = (value) => {
  try {
    parseCount(value);
    return true;
  } catch {
    return false;
  }
};

/**
 * Safely parses count with default fallback
 * @param {string|number} value - Value to parse
 * @param {number} defaultValue - Fallback value if parsing fails
 * @returns {number} Parsed value or default
 */
export const parseCountSafe = (value, defaultValue = 0) => {
  try {
    return parseCount(value);
  } catch {
    return defaultValue;
  }
};
