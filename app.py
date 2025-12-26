from flask import Flask,render_template,url_for,request,redirect
from flask import send_from_directory
import os

import pandas as pd 
import numpy as np 
import pickle

app = Flask(__name__, static_folder='.', static_url_path='/static')

import logging

logging.basicConfig(level=logging.INFO)

# Load models safely; continue even if missing so app doesn't crash on import
def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.warning("Could not load %s: %s", path, e)
        return None

random = load_pickle('random_fake.pkl')
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
        
        # Predict using selected model
        model_name = data.get('model', 'RandomForestClassifier')
        if model_name == 'RandomForestClassifier' and random is not None:
            prediction = random.predict(ex)[0]
            confidence = random.predict_proba(ex)[0]
        elif model_name == 'DecisionTreeClassifier' and clf_gini is not None:
            prediction = clf_gini.predict(ex)[0]
            confidence = clf_gini.predict_proba(ex)[0]
        else:
            return {'error': 'Model not available'}, 500
        
        result = 'Fake' if int(prediction) == 1 else 'Real'
        
        return {
            'username': username,
            'result': result,
            'prediction': int(prediction),
            'confidence_real': float(confidence[0]),
            'confidence_fake': float(confidence[1]),
            'model': model_name,
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
 app.run(debug=True)
