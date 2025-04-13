from flask import Flask, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Configure the MySQL connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://admin123:abc123@localhost/university_expert_system'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy and LoginManager
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)

    def __init__(self, username, hashed_password, email=None):
        self.username = username
        self.password = hashed_password
        self.email = email

# NEW: a table for storing chat records: user message, assistant response, references
class ChatRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    assistant_response = db.Column(db.Text, nullable=False)
    reference_docs = db.Column(db.Text, nullable=True)  # can store JSON or a plain string

    # relationship back to the user
    user = db.relationship('User', backref=db.backref('chat_records', lazy=True))

    def __init__(self, user_id, user_message, assistant_response, reference_docs=None):
        self.user_id = user_id
        self.user_message = user_message
        self.assistant_response = assistant_response
        self.reference_docs = reference_docs

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Endpoint for user login
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    
    # Fetch the user from the database
    user = User.query.filter_by(username=username).first()
    if user:
        # Compare hashed password with the provided password
        if check_password_hash(user.password, password):
            login_user(user)
            return jsonify({"message": "Login successful"}), 200

    return jsonify({"message": "Invalid credentials"}), 401

# Endpoint for user signup
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    email = data.get("email")
    
    # Validate input
    if not username or not password:
        return jsonify({"message": "Username and password are required."}), 400
    
    # Check if the user already exists
    if User.query.filter_by(username=username).first():
        return jsonify({"message": "User already exists."}), 409
    
    # Hash the password before storing
    hashed_pw = generate_password_hash(password)
    
    # Create and save the new user (now storing hashed password!)
    new_user = User(username, hashed_pw, email)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({"message": "Signup successful."}), 201

# Endpoint for user profile (requires login)
@app.route("/profile", methods=["GET"])
@login_required
def profile():
    username = request.args.get("username")
    if not username:
        return jsonify({"message": "Username is required."}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"message": "User not found."}), 404

    # Return user profile details as JSON
    return jsonify({
        "username": user.username,
        "email": user.email
    }), 200

# Endpoint for logout (requires login)
@app.route("/logout", methods=["GET"])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out"}), 200

# Endpoint to update password
@app.route("/update_password", methods=["PUT"])
@login_required
def update_password():
    """
    Expects JSON:
    {
      "username": "theUser",
      "old_password": "oldPlainTextPassword",
      "new_password": "newPlainTextPassword"
    }
    """
    data = request.json
    username = data.get("username")
    old_password = data.get("old_password")
    new_password = data.get("new_password")

    if not username or not old_password or not new_password:
        return jsonify({"message": "All fields (username, old_password, new_password) are required."}), 400

    # Query the user by username
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"message": "User not found."}), 404

    # Verify the old password
    if not check_password_hash(user.password, old_password):
        return jsonify({"message": "Old password is incorrect."}), 401

    # Hash the new password
    hashed_pw = generate_password_hash(new_password)
    user.password = hashed_pw

    db.session.commit()
    return jsonify({"message": "Password updated successfully."}), 200

@app.route("/store_chat_record", methods=["POST"])
@login_required
def store_chat_record():
    """
    Expects JSON:
    {
      "username": "someUser",
      "user_message": "Hello, can you help me?",
      "assistant_response": "Sure, how can I help?",
      "reference_docs": "[{'source':'doc1.pdf','snippet':'...'}]"
    }
    """
    data = request.json
    username = data.get("username")
    user_message = data.get("user_message")
    assistant_response = data.get("assistant_response")
    reference_docs = data.get("reference_docs")

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"message": "User not found"}), 404

    # Insert a new record
    record = ChatRecord(
        user_id=user.id,
        user_message=user_message,
        assistant_response=assistant_response,
        reference_docs=reference_docs
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({"message": "Chat record stored successfully"}), 201

@app.route("/get_chat_records", methods=["GET"])
@login_required
def get_chat_records():
    username = request.args.get("username")
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"message": "User not found"}), 404

    records = ChatRecord.query.filter_by(user_id=user.id).all()
    results = []
    for r in records:
        results.append({
            "id": r.id,
            "user_message": r.user_message,
            "assistant_response": r.assistant_response,
            "reference_docs": r.reference_docs
        })
    return jsonify(results), 200


if __name__ == "__main__":
    # Manually push the app context and create tables
    with app.app_context():
        db.create_all()
    app.run(debug=True)
