# networks
## Aim:
To develop a fraud detection system for bank transactions using ensemble machine learning techniques, incorporating data preprocessing, feature selection, model evaluation, and deployment in a Flask-based web application, with integrated Exploratory Data Analysis (EDA) for insights. 

## Data Creation:
```
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of records
num_records = 10000

# Helper function to generate random dates
def random_date(start_date, end_date):
    return start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))

# Helper function to simulate churn based on features
def churn_logic(row):
    if row['login_frequency'] < 5 and row['purchases'] < 1 and row['ratings_count'] < 2:
        return 1  # Churned
    return 0  # Not churned

# Generate synthetic data
data = []
for i in range(num_records):
    user_id = i + 1
    login_frequency = np.random.randint(1, 31)  # Between 1 and 30 logins per month
    last_login_time = random_date(datetime(2024, 1, 1), datetime(2025, 1, 1))
    purchase_history = np.random.randint(0, 50)  # Between 0 and 50 total purchases
    ratings_count = np.random.randint(0, 100)  # Between 0 and 100 ratings
    session_duration = np.random.randint(50, 500)  # Total session time in minutes
    avg_session_time = np.random.randint(5, 30)  # Average session time per session in minutes
    purchases = np.random.randint(0, 10)  # Purchases in the last month
    last_purchase_days_ago = np.random.randint(1, 60)  # Days since the last purchase
    
    churn_status = churn_logic({
        'login_frequency': login_frequency,
        'purchases': purchases,
        'ratings_count': ratings_count
    })

    data.append([
        user_id, login_frequency, last_login_time, purchase_history, ratings_count, 
        session_duration, avg_session_time, purchases, last_purchase_days_ago, churn_status
    ])

# Create a DataFrame
columns = [
    'user_id', 'login_frequency', 'last_login_time', 'purchase_history', 'ratings_count', 
    'session_duration', 'avg_session_time', 'purchases', 'last_purchase_days_ago', 'churn_status'
]
df = pd.DataFrame(data, columns=columns)

# Save the dataset to a CSV file
df.to_csv('Book_Review.csv', index=False)

# Show the first few rows of the dataset
df.head()
```

## App:
### Libraries:
```
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
```
### Flask App:
```
app = Flask(__name__)
app.secret_key = 'User@123'  # Replace with a more secure key for production

# Set up the database configuration (SQLite for simplicity)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Use SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model for the database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    user_id = db.Column(db.Integer, unique=True, nullable=False)

# Create the database
with app.app_context():
    db.create_all()

```
### Dataset:
```
books = pd.read_csv('Dataset/Books.csv')
users = pd.read_csv('Dataset/Users.csv')
ratings = pd.read_csv('Dataset/Ratings.csv')

# Clean books dataset
books.drop_duplicates(keep="first", inplace=True)
books.reset_index(inplace=True, drop=True)
books.drop(columns=["Image-URL-L", "Image-URL-M"], inplace=True)

# Clean users dataset
users.drop_duplicates(keep="first", inplace=True)
users.reset_index(inplace=True, drop=True)

# Clean ratings dataset
ratings.drop_duplicates(keep="first", inplace=True)
ratings.reset_index(inplace=True, drop=True)

# Filter users with more than 50 ratings
num_ratings = ratings.groupby("User-ID")["Book-Rating"].count()
num_ratings = pd.DataFrame(num_ratings)
num_ratings.rename(columns={"Book-Rating": "num_rating"}, inplace=True)
ratings = pd.merge(ratings, num_ratings, on="User-ID")
ratings = ratings[ratings["num_rating"] > 50]

```
### Merge ratings with books
```
df = pd.merge(ratings, books, on="ISBN")

# Filter books with more than 100 ratings
book_rating_counts = df.groupby("Book-Title")["Book-Rating"].count()
df = df[df["Book-Title"].isin(book_rating_counts[book_rating_counts > 100].index)]

# Aggregate ratings to ensure each user has one rating per book
aggregated_df = df.groupby(['User-ID', 'Book-Title']).agg({'Book-Rating': 'mean'}).reset_index()

# Merge the aggregated data to drop duplicate ratings
df = pd.merge(aggregated_df, df.drop(columns=['Book-Rating']), on=['User-ID', 'Book-Title'])
df.drop_duplicates(subset=['User-ID', 'Book-Title'], keep='first', inplace=True)

# Create a sparse matrix of users and their ratings
pivot = df.pivot(index='Book-Title', columns='User-ID', values='Book-Rating')
pivot.fillna(value=0, inplace=True)
matrix = csr_matrix(pivot)

# Build k-NN model for personalized recommendations
model1 = NearestNeighbors(algorithm="brute", metric="cosine", n_neighbors=11)
model1.fit(csr_matrix(pivot.T))

# Function for personalized book recommendations
def personalized_book_recommendations(user_id, model, transpose_pivot, df, books):
    if user_id not in transpose_pivot.index:
        return "User ID not found in the dataset."  # Return error message if user is not found in dataset.

    # Get nearest neighbors (similar users)
    distances, indices = model.kneighbors(transpose_pivot.loc[user_id, :].values.reshape(1, -1), n_neighbors=11)
    similar_users = transpose_pivot.index[indices.flatten()][1:]  # Skip the first one as it's the current user.

    # Collect books rated by similar users
    recommended_books = set()
    for user in similar_users:
        top_books = df[df['User-ID'] == user][['Book-Title', 'ISBN']].drop_duplicates()
        # Update the set with each book's (Book-Title, ISBN) pair as a tuple
        recommended_books.update([tuple(book) for book in top_books.values.tolist()])

    # Exclude books already rated by the input user
    user_rated_books = df[df['User-ID'] == user_id]['Book-Title'].unique()
    final_recommendations = [book for book in recommended_books if book[0] not in user_rated_books]

    # Now collect detailed information of the recommended books
    book_details = []
    for book_title, isbn in final_recommendations:
        # Fetch book details from the books dataframe
        book_info = books[books['ISBN'] == isbn][['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S']]
        if not book_info.empty:
            book_details.append(book_info.iloc[0].to_dict())  # Convert the first row to a dictionary

    return book_details

# Route for the index page (landing page)
@app.route('/')
def index():
    return render_template('index.html')

# Route for registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        email = request.form['email']
        password = request.form['password']
        phone = request.form['phone']

        # Check if the email already exists in the database
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return "Email already registered. Please use a different email."  # Display error if email exists
        
        # Load the existing recommendations CSV file
        recommendations_df = pd.read_csv('Dataset/user_book_recommendations.csv')
        
        # Extract User-IDs from the recommendations dataset
        existing_user_ids = recommendations_df['User-ID'].tolist()
        
        # Find the first available User-ID not in the database
        available_user_id = None
        for user_id in existing_user_ids:
            if not User.query.filter_by(user_id=user_id).first():  # Check if the User ID is already in the database
                available_user_id = user_id
                break
        
        # If no available User-ID is found, return an error
        if available_user_id is None:
            return "No available User ID found. Please try again later."
        
        # Create a new user object with the selected User-ID
        new_user = User(name=name, age=age, email=email, password=password, phone=phone, user_id=available_user_id)
        
        # Add the new user to the database
        db.session.add(new_user)
        db.session.commit()

        # Store user details in session
        session['user_id'] = available_user_id
        session['user_name'] = name

        # Redirect to the home page after successful registration
        return redirect(url_for('home'))  # Redirect to home after registration
    
    # If the request is GET, show the registration form
    return render_template('register.html')

# Route for login page (only need the User ID for login)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']  # Only need the User ID for login
        
        # Check if the User ID is within the valid range
        if not (183 <= int(user_id) <= 278843):
            return "User ID out of valid range. Please try again."

        # Check if the User ID exists in the database
        user = User.query.filter_by(user_id=int(user_id)).first()
        if not user:
            return "User ID not found. Please try again."  # Error message if User ID is not found.
        
        # Store user details in session
        session['user_id'] = user.user_id
        session['user_name'] = user.name  # Store name in the session if needed (for personalization)

        # Redirect to the home page with recommendations
        return redirect(url_for('home'))
    
    # Login page, will show input for User ID
    return render_template('login.html')

# Route for home page where recommendations are shown
@app.route('/home')
def home():
    user_id = session.get('user_id')
    if user_id is None:
        return redirect(url_for('login'))
    
    # Fetch the book recommendations with detailed info
    recommendations = personalized_book_recommendations(user_id, model1, pivot.T, df, books)
```

### Return:
```
return render_template('home.html', recommendations=recommendations, user_name=session.get('user_name'))


@app.route('/book_now/<book_id>')
def book_now(book_id):
    # Handle the book booking logic here
    return f"Book Now for book {book_id}"


# Route for logout
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    return redirect(url_for('index'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
```

## Result:
The fraud detection system accurately classifies bank transactions using ensemble machine learning models, achieving high performance with precision, recall, F1-score, and ROC-AUC metrics. The Flask-based web app enables real-time predictions and EDA visualizations, enhancing fraud prevention in banking
