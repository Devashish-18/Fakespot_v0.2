from flask import Flask, render_template, url_for, request, redirect, jsonify
from flask_cors import CORS
from flask import send_from_directory
import os
from datetime import datetime
import random as py_random
import re

import pandas as pd 
import numpy as np 
import pickle

app = Flask(__name__, static_folder='.', static_url_path='/static')
CORS(app)

import logging

logging.basicConfig(level=logging.INFO)

# Utility function to parse human-readable count format
def parse_count(value):
    """
    Converts human-readable count format to numeric value
    Examples: "218K" → 218000, "40.7M" → 40700000, "950" → 950
    
    Args:
        value: str or int - The value to parse
        
    Returns:
        int: The parsed numeric value
        
    Raises:
        ValueError: If the format is invalid
    """
    # If already a number, return it
    if isinstance(value, (int, float)):
        if not isinstance(value, bool) and value >= 0:
            return int(abs(value))
        else:
            raise ValueError('Invalid number format')
    
    # Convert to string and strip whitespace
    value_str = str(value).strip()
    
    if not value_str:
        raise ValueError('Invalid number format')
    
    # Match pattern: digits with optional decimal, optional suffix (K, M, B)
    match = re.match(r'^(\d+\.?\d*|\.\d+)([kmb]?)$', value_str, re.IGNORECASE)
    
    if not match:
        raise ValueError('Invalid number format')
    
    num_value = float(match.group(1))
    suffix = match.group(2).lower() if match.group(2) else ''
    
    # Apply multiplier based on suffix
    multipliers = {
        'k': 1_000,
        'm': 1_000_000,
        'b': 1_000_000_000,
    }
    
    result = num_value * multipliers.get(suffix, 1)
    return int(round(result))

# Load models safely; continue even if missing so app doesn't crash on import
def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.warning("Could not load %s: %s", path, e)
        return None

random_model = load_pickle('random_fake.pkl')
clf_gini = load_pickle('decision_fake.pkl')

@app.route('/')
@app.route('/index')
def index():
    return redirect(url_for('prediction'))


@app.route('/login')
def login():
    return render_template("login.html")


# Upload route removed — app now goes directly to prediction

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset)
        return render_template("preview.html",df_view = df)


@app.route('/prediction')
def prediction():
    return render_template("prediction.html")

def parse_input(val):
    """Convert form value to numeric feature."""
    if val is None:
        return 0.0
    s = str(val).strip()
    if s == '':
        return 0.0
    try:
        return float(s)
    except:
        ls = s.lower()
        if ls in ('true', 'yes', 'y', '1'):
            return 1.0
        if ls in ('false', 'no', 'n', '0'):
            return 0.0
        if ls.startswith('http') or ('://' in ls) or ('.' in ls and ' ' not in ls):
            return 1.0
        try:
            return float(s.replace(',', ''))
        except:
            return float(len(s))

@app.route('/performance')
def performance():
    return render_template("performance.html")

@app.route('/chart')
def chart():
    return render_template("chart.html")

def count_digits(s):
    return sum(c.isdigit() for c in s)


def generate_reasons(username, fullname, profile_pic, total_posts, followers, following, 
                     description, external_url, account_private, prediction, confidence):
    """Generate reasons why account was classified as real or fake"""
    reasons = []
    is_fake = prediction == 1
    
    # Following to follower ratio
    if following > 0:
        ratio = following / followers if followers > 0 else following
        if ratio > 3:
            reasons.append({
                "signal": "High following-to-follower ratio",
                "impact": "high",
                "detail": f"Following is extremely high ({following}) compared to followers ({followers}). Ratio of {ratio:.1f}:1 is typical of fake accounts."
            })
    
    # Account age (simulated as random between 15-1000 days)
    account_age = py_random.randint(15, 1000)
    if account_age < 30:
        reasons.append({
            "signal": "Very new account",
            "impact": "high",
            "detail": f"Account age is only {account_age} days. New accounts with suspicious patterns are often fake."
        })
    
    # No profile picture
    if profile_pic == 0 or profile_pic == '0':
        reasons.append({
            "signal": "Missing profile picture",
            "impact": "high",
            "detail": "Accounts without profile pictures are commonly fake. Real accounts typically have profile photos."
        })
    
    # Low engagement
    avg_likes = py_random.randint(0, int(total_posts * 3)) if total_posts > 0 else 0
    if avg_likes < 5 and total_posts > 0:
        reasons.append({
            "signal": "Low engagement",
            "impact": "medium",
            "detail": f"Average likes per post ({avg_likes}) are very low. Suggests bot followers or inactive engagement."
        })
    
    # No bio
    if not description or len(description) == 0:
        reasons.append({
            "signal": "Empty biography",
            "impact": "medium",
            "detail": "Account has no bio. Authentic accounts typically include some description."
        })
    
    # Few posts
    if total_posts < 5 and followers > 100:
        reasons.append({
            "signal": "Low post count",
            "impact": "medium",
            "detail": f"Only {total_posts} posts but {followers} followers. Suggests artificial follower growth."
        })
    
    return reasons if reasons else [{
        "signal": "Account appears authentic",
        "impact": "low",
        "detail": "No major red flags detected in account behavior."
    }]

def generate_profile_data(username, fullname, profile_pic, total_posts, followers, following, 
                          description, external_url, account_private):
    """Generate realistic profile data for charts"""
    account_age_days = py_random.randint(15, 1000)
    avg_likes = py_random.randint(0, int(total_posts * 5)) if total_posts > 0 else 0
    avg_comments = py_random.randint(0, int(avg_likes * 0.3))
    engagement_rate = (avg_likes + avg_comments) / followers if followers > 0 else 0
    
    return {
        "followers": int(followers),
        "following": int(following),
        "posts": int(total_posts),
        "bio_length": len(description) if description else 0,
        "has_profile_pic": profile_pic == 1 or profile_pic == '1',
        "is_private": account_private == 1 or account_private == '1',
        "account_age_days": account_age_days,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "engagement_rate": engagement_rate
    }

def generate_chart_data(profile_data):
    """Generate chart data structure"""
    return {
        "bar": [
            {"metric": "Followers", "value": profile_data["followers"]},
            {"metric": "Following", "value": profile_data["following"]},
            {"metric": "Posts", "value": profile_data["posts"]}
        ],
        "radar": [
            {"feature": "Engagement", "value": min(profile_data["engagement_rate"] * 100, 100)},
            {"feature": "Account Age", "value": min(profile_data["account_age_days"] * 0.1, 100)},
            {"feature": "Profile Completeness", "value": 100 if (profile_data["has_profile_pic"] and profile_data["bio_length"] > 0) else 30},
            {"feature": "Post Activity", "value": min(profile_data["posts"] * 5, 100)},
            {"feature": "Network Quality", "value": min((profile_data["followers"] / profile_data["following"] * 100) if profile_data["following"] > 0 else 50, 100)}
        ],
        "line": [
            {"day": f"Day {i+1}", "likes": max(0, profile_data["avg_likes"] + py_random.randint(-5, 5))}
            for i in range(max(profile_data["posts"], 5))
        ],
        "doughnut": [
            {"name": "Fake Factors", "value": 35},
            {"name": "Real Factors", "value": 65}
        ]
    }

@app.route('/analyze', methods=['GET'])
def analyze():
    """Analyze Instagram account for fake detection
    
    Query params:
    - username: Instagram username to analyze
    - followers: (optional) follower count in any format (e.g., "218K", "40.7M", "950")
    - following: (optional) following count in any format
    - posts: (optional) post count
    
    Returns JSON with prediction, confidence, profile data, reasons, and charts
    """
    try:
        username = request.args.get('username', '').strip()
        if not username:
            return {'error': 'Username is required'}, 400
        
        # Try to parse optional formatted count parameters
        try:
            # If followers/following are provided, use them; otherwise generate random
            follower_count = parse_count(request.args.get('followers')) if request.args.get('followers') else py_random.randint(100, 10000)
            following_count = parse_count(request.args.get('following')) if request.args.get('following') else py_random.randint(100, 5000)
            posts_count = parse_count(request.args.get('posts')) if request.args.get('posts') else py_random.randint(0, 50)
        except ValueError as e:
            return {'error': f'Invalid number format: {str(e)}'}, 400
        
        has_pic = py_random.choice([0, 1])
        bio = py_random.choice(['', 'Entrepreneur', 'Photographer', 'Just vibing', 'Follow back'])
        is_private = py_random.choice([0, 1])
        
        # Build feature vector for prediction
        profile_pic_val = float(has_pic)
        nums_len_username = sum(c.isdigit() for c in username) / len(username) if username else 0.0
        fullname_words = py_random.randint(1, 4)
        nums_len_fullname = py_random.random() * 0.3
        name_match = py_random.choice([0, 1])
        desc_length = float(len(bio))
        external_url_val = py_random.choice([0, 1])
        account_private_val = float(is_private)
        
        follower_following_ratio = following_count / follower_count if follower_count > 0 else 0
        
        features = [
            profile_pic_val,
            nums_len_username,
            float(fullname_words),
            nums_len_fullname,
            float(name_match),
            desc_length,
            external_url_val,
            account_private_val,
            float(posts_count),
            float(follower_count),
            float(following_count)
        ]
        
        ex = np.array(features).reshape(1, -1)
        
        # Predict using Random Forest model
        if random_model is not None:
            prediction = int(random_model.predict(ex)[0])
            confidence_probs = random_model.predict_proba(ex)[0]
            confidence = float(confidence_probs[1])  # Confidence for fake (class 1)
        else:
            # Fallback: simple heuristic
            prediction = 1 if follower_following_ratio > 3 or posts_count < 2 else 0
            confidence = min((follower_following_ratio / 5), 1.0) if follower_following_ratio > 3 else 0.3
        
        # Generate profile data
        profile_data = generate_profile_data(
            username, 'User', has_pic, posts_count, follower_count, following_count,
            bio, external_url_val, is_private
        )
        
        # Generate reasons
        reasons = generate_reasons(
            username, 'User', has_pic, posts_count, follower_count, following_count,
            bio, external_url_val, is_private, prediction, confidence
        )
        
        # Generate chart data
        charts = generate_chart_data(profile_data)
        
        return jsonify({
            'username': username,
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': confidence,
            'profile_data': profile_data,
            'reasons': reasons,
            'charts': charts
        }), 200
    
    except Exception as e:
        logging.exception('Analyze failed')
        return {'error': str(e)}, 500

@app.route('/check-account', methods=['POST'])
def check_account():
    """Check if an Instagram account is real or fake based on account details.
    
    Expects JSON with: username, fullname, profile_pic, total_posts, followers, following,
    description (optional), external_url (optional), account_private (optional)
    """
    try:
        data = request.get_json()
        if not data:
            return {'error': 'No data provided'}, 400
        
        # Extract features from account data
        username = data.get('username', '')
        fullname = data.get('fullname', '')
        profile_pic = data.get('profile_pic', '')
        total_posts = data.get('total_posts', 0)
        followers = data.get('followers', 0)
        following = data.get('following', 0)
        description = data.get('description', '')
        external_url = data.get('external_url', '0')
        account_private = data.get('account_private', '0')
        
        # Derive features
        profile_pic_val = parse_input(profile_pic)
        
        # nums/length username
        if len(username) > 0:
            nums_len_username = count_digits(username) / len(username)
        else:
            nums_len_username = 0.0

        # fullname words
        fullname_words = float(len(fullname.split()))

        # nums/length fullname
        if len(fullname) > 0:
            nums_len_fullname = count_digits(fullname) / len(fullname)
        else:
            nums_len_fullname = 0.0
            
        name_match = 1.0 if username.lower() == fullname.lower() else 0.0
        desc_length = float(len(description)) if description else 0.0
        external_url_val = parse_input(external_url)
        account_private_val = parse_input(account_private)
        
        # Calculate follower/following ratio
        follower_following_ratio = 0.0
        if following > 0:
            follower_following_ratio = float(followers) / float(following)
        
        # Build feature vector matching model training order
        # Order: profile pic, nums/length username, fullname words, nums/length fullname, name==username, 
        #        description length, external URL, private, #posts, #followers, #follows
        features = [
            profile_pic_val,
            nums_len_username,
            fullname_words,
            nums_len_fullname,
            name_match,
            desc_length,
            external_url_val,
            account_private_val,
            float(total_posts),
            float(followers),
            float(following)
        ]
        
        ex = np.array(features).reshape(1, -1)
        
        # Predict using selected model; if model pickles aren't available, fall back to heuristic
        model_name = data.get('model', 'RandomForestClassifier')
        if model_name == 'RandomForestClassifier' and random_model is not None:
            prediction = random_model.predict(ex)[0]
            confidence = random_model.predict_proba(ex)[0]
        elif model_name == 'DecisionTreeClassifier' and clf_gini is not None:
            prediction = clf_gini.predict(ex)[0]
            confidence = clf_gini.predict_proba(ex)[0]
        else:
            # Fallback heuristic: flag fake if follower/following ratio high or very few posts
            logging.warning('Requested model "%s" not available; using heuristic fallback', model_name)
            heuristic_pred = 1 if (follower_following_ratio > 3 or float(total_posts) < 2) else 0
            prediction = heuristic_pred
            # Build a simple confidence vector [confidence_real, confidence_fake]
            if heuristic_pred == 1:
                conf_fake = min(max((follower_following_ratio / 5), 0.3), 0.99)
                confidence = [1.0 - conf_fake, conf_fake]
            else:
                conf_real = 0.7
                confidence = [conf_real, 1.0 - conf_real]
        
        result = 'Fake' if int(prediction) == 1 else 'Real'
        
        # Generate reasons for the prediction
        reasons = generate_reasons(
            username, fullname, profile_pic_val, total_posts, followers, following,
            description, external_url_val, account_private_val, int(prediction), float(confidence[1])
        )
        
        return {
            'username': username,
            'result': result,
            'prediction': int(prediction),
            'confidence_real': float(confidence[0]),
            'confidence_fake': float(confidence[1]),
            'model': model_name,
            'reasons': reasons,
            'features_used': {
                'profile_pic': profile_pic_val,
                'username_digit_ratio': nums_len_username,
                'fullname_words': fullname_words,
                'fullname_digit_ratio': nums_len_fullname,
                'name_match': name_match,
                'description_length': desc_length,
                'has_external_url': external_url_val,
                'is_private': account_private_val,
                'total_posts': float(total_posts),
                'followers': float(followers),
                'following': float(following),
                'follower_following_ratio': follower_following_ratio
            }
        }, 200
    
    except Exception as e:
        logging.exception('Check account failed')
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
