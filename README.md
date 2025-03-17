# AI-Capstone-hw1
Dataset: netflix_trailers.csv
This dataset contains information about Netflix movie trailers, including:

Video ID - ID of the YouTube video
Title – Name of the movie
Duration – Length of the trailer (in seconds)
Views – Number of views the trailer received
Category - Series genre  
              1. Drama\
              2. Comedy\
              3. Action/Adventure\
              4. Sci-Fi/Fantasy\
              5. Horror/Thriller\
              6. Documentary\
              7. Animation\
| Video ID  | Title  | Duration (s) | Views   | Category        | Category ID | Like Rate | Comment Rate |
|-----------|----------------------------------------------|-------------|---------|----------------|-------------|------------|--------------|
| QIw6ITiwgBU | The Electric State | Final Trailer | Netflix | 131 | 1801529 | Sci-Fi/Fantasy | 4 | 0.8 | 0.09 |
| a_1-vyLcBpM | How to Sell Drugs Online (Fast): The Final Season | 117 | 30781 | Drama | 1 | 3.39 | 0.24 |
Category ID – Encoded genre of the movie  
Like Rate – Ratio of likes to total interactions  
Comment Rate – Ratio of comment to total interactions  
Scripts:  
Kmean.py – Performs unsupervised clustering using K-Means to group trailers based on Duration, Views, and Like Rate.  
supervised.py – Implements regression models to analyze relationships between trailer length and view count.  
dnn.py – Uses a Deep Neural Network (DNN) for genre classification based on trailer features.  
