import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load datasets
books = pd.read_csv('Dataset/Books.csv')
users = pd.read_csv('Dataset/Users.csv')
ratings = pd.read_csv('Dataset/Ratings.csv')

# Clean books dataset
books.drop_duplicates(keep="first", inplace=True)
books.reset_index(inplace=True, drop=True)
books.drop(columns=["Image-URL-S", "Image-URL-L"], inplace=True)

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

# Merge ratings with books
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

# Build k-NN model for generalized book recommendations
model = NearestNeighbors(n_neighbors=11, algorithm="brute", metric="cosine")
model.fit(matrix)

# Function for generalized book recommendations
def get_book_recommendations(book_title, model, pivot, books):
    if book_title not in pivot.index:
        return "Book title not found."

    book_index = pivot.index.get_loc(book_title)
    distances, indices = model.kneighbors(pivot.iloc[book_index, :].values.reshape(1, -1), n_neighbors=11)
    recommended_books = [pivot.index[i] for i in indices.flatten()][1:]  # Exclude the input book
    
    # Fetch book details from books DataFrame
    book_details = []
    for title in recommended_books:
        book_info = books[books['Book-Title'] == title][['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-M']].iloc[0]
        book_details.append(book_info)
    
    return book_details

input_book = df["Book-Title"].iloc[1]
print("Recommendations for book:", input_book)
recommendation = get_book_recommendations(input_book, model, pivot, books)

for book in recommendation:
    print(f"Title: {book['Book-Title']}, Author: {book['Book-Author']}, Year: {book['Year-Of-Publication']}, Publisher: {book['Publisher']}, Image: {book['Image-URL-M']}")

# Prepare transpose pivot for personalized recommendations
transpose_pivot = pivot.T
matrix_t = csr_matrix(transpose_pivot)

# Build k-NN model for personalized recommendations
model1 = NearestNeighbors(algorithm="brute", metric="cosine", n_neighbors=11)
model1.fit(matrix_t)

# Function for personalized book recommendations
def personalized_book_recommendations(user_id, model, transpose_pivot, df, books):
    if user_id not in transpose_pivot.index:
        return "User ID not found in the dataset."

    # Get nearest neighbors (similar users)
    distances, indices = model.kneighbors(transpose_pivot.loc[user_id, :].values.reshape(1, -1), n_neighbors=11)
    similar_users = transpose_pivot.index[indices.flatten()][1:]

    # Collect books rated by similar users
    recommended_books = set()
    for user in similar_users:
        top_books = df[df['User-ID'] == user]['Book-Title'].unique()
        recommended_books.update(top_books)

    # Exclude books already rated by the input user
    user_rated_books = df[df['User-ID'] == user_id]['Book-Title'].unique()
    final_recommendations = list(recommended_books.difference(user_rated_books))

    # Fetch book details from books DataFrame
    book_details = []
    for title in final_recommendations:
        book_info = books[books['Book-Title'] == title][['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-M']].iloc[0]
        book_details.append(book_info)
    
    return book_details

# Example: Personalized recommendation for a user
user_inp = df["User-ID"].iloc[0]
recommendations = personalized_book_recommendations(user_inp, model1, transpose_pivot, df, books)

print(f"Personalized recommendations for user {user_inp}:")
for book in recommendations:
    print(f"Title: {book['Book-Title']}, Author: {book['Book-Author']}, Year: {book['Year-Of-Publication']}, Publisher: {book['Publisher']}, Image: {book['Image-URL-M']}")
