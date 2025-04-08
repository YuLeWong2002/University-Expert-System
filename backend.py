from flask import Flask, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_sqlalchemy import SQLAlchemy
import ollama

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Configure the MySQL connection
# Replace 'admin123', 'abc123', and 'university_expert_system' with your MySQL credentials and database name.
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://admin123:abc123@localhost/university_expert_system'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy and LoginManager
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

# Define a User model to represent users in the database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)  # Auto-incrementing primary key
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)

    def __init__(self, username, password, email=None):
        self.username = username
        self.password = password  # In production, store hashed passwords!
        self.email = email

# User loader callback for Flask-Login
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
    # In production, compare hashed passwords
    if user and user.password == password:
        login_user(user)
        return jsonify({"message": "Login successful"}), 200
    return jsonify({"message": "Invalid credentials"}), 401

# Endpoint for RAG query (requires login)
@app.route("/rag", methods=["POST"])
@login_required
def rag_query():
    data = request.json
    query = data.get("query")
    
    # Generate response using Ollama (modify for LangChain if needed)
    response = ollama.generate(model="mistral", prompt=query)
    return jsonify({"answer": response["response"]})

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
    
    # Create and save the new user (remember to hash the password in production!)
    new_user = User(username, password, email)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({"message": "Signup successful."}), 201

# Endpoint for user profile (requires login)
@app.route("/profile", methods=["GET"])
@login_required
def profile():
    # Get the username from the query parameters
    username = request.args.get("username")
    if not username:
        return jsonify({"message": "Username is required."}), 400

    # Query the user from the database
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

if __name__ == "__main__":
    # Manually push the app context and create tables (replacing the deprecated before_first_request decorator)
    with app.app_context():
        db.create_all()
    app.run(debug=True)

@app.route("/update_profile", methods=["PUT"])
@login_required
def update_profile():
    data = request.json
    username = data.get("username")
    new_email = data.get("email")
    new_password = data.get("password")  # optional if user wants to change password

    if not username:
        return jsonify({"message": "Username is required."}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"message": "User not found."}), 404

    # Update fields as present
    if new_email is not None:
        user.email = new_email
    if new_password is not None and new_password.strip():
        user.password = new_password  # In production, store a hash!

    db.session.commit()

    return jsonify({"message": "Profile updated successfully"}), 200

