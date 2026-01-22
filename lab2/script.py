#!/usr/bin/env python3
"""
Lab 2: TMDb API Data Collection
This script fetches Comedy movies from TMDb API and finds similar movies.
"""

import sys
import requests
import time
import csv

# TMDb API base URL
BASE_URL = "https://api.themoviedb.org/3"

# Rate limiting: 40 requests per 10 seconds
MAX_REQUESTS = 40
TIME_WINDOW = 10
request_times = []


def rate_limit():
    """Implement rate limiting: 40 requests per 10 seconds"""
    global request_times
    current_time = time.time()
    
    # Remove requests older than 10 seconds
    request_times = [t for t in request_times if current_time - t < TIME_WINDOW]
    
    # If we've reached the limit, wait
    if len(request_times) >= MAX_REQUESTS:
        sleep_time = TIME_WINDOW - (current_time - request_times[0]) + 0.1
        if sleep_time > 0:
            time.sleep(sleep_time)
            # Clean up again after sleeping
            request_times = [t for t in request_times if time.time() - t < TIME_WINDOW]
    
    request_times.append(time.time())


def make_request(url, params):
    """Make an API request with rate limiting"""
    rate_limit()
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_comedy_genre_id(api_key):
    """Get the genre ID for Comedy"""
    url = f"{BASE_URL}/genre/movie/list"
    params = {"api_key": api_key}
    data = make_request(url, params)
    
    for genre in data["genres"]:
        if genre["name"] == "Comedy":
            return genre["id"]
    
    raise ValueError("Comedy genre not found")


def get_300_comedy_movies(api_key):
    """
    Part B: Retrieve 300 most popular Comedy movies from 2000 or later.
    Returns a list of tuples: [(movie_id, movie_name), ...]
    """
    genre_id = get_comedy_genre_id(api_key)
    movies = []
    page = 1
    
    url = f"{BASE_URL}/discover/movie"
    
    while len(movies) < 300:
        params = {
            "api_key": api_key,
            "with_genres": genre_id,
            "primary_release_date.gte": "2000-01-01",
            "sort_by": "popularity.desc",
            "page": page
        }
        
        data = make_request(url, params)
        results = data.get("results", [])
        
        if not results:
            break
        
        for movie in results:
            if len(movies) >= 300:
                break
            movie_id = movie["id"]
            movie_name = movie["title"]
            movies.append((movie_id, movie_name))
        
        # Check if there are more pages
        if data.get("page", 0) >= data.get("total_pages", 0):
            break
        
        page += 1
    
    return movies[:300]


def get_similar_movies(api_key, movie_id):
    """
    Get similar movies for a given movie ID.
    Returns a list of movie IDs.
    """
    url = f"{BASE_URL}/movie/{movie_id}/similar"
    params = {
        "api_key": api_key,
        "page": 1
    }
    
    data = make_request(url, params)
    results = data.get("results", [])
    
    # Return up to 5 similar movie IDs
    similar_ids = [movie["id"] for movie in results[:5]]
    return similar_ids


def save_movie_names(movies, filename):
    """Save movie IDs and names to CSV file (Part B)"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for movie_id, movie_name in movies:
            writer.writerow([movie_id, movie_name])


def get_all_similar_pairs(api_key, movies):
    """
    Part C: Get similar movies for all movies and return pairs.
    Returns a set of tuples (id1, id2) where id1 < id2 to avoid duplicates.
    """
    pairs = set()
    
    print(f"Fetching similar movies for {len(movies)} movies...")
    for idx, (movie_id, _) in enumerate(movies, 1):
        try:
            similar_ids = get_similar_movies(api_key, movie_id)
            for similar_id in similar_ids:
                # Ensure we only store pairs where id1 < id2
                pair = tuple(sorted([movie_id, similar_id]))
                pairs.add(pair)
            
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(movies)} movies...")
        except Exception as e:
            print(f"Error processing movie {movie_id}: {e}")
            continue
    
    return pairs


def save_similar_pairs(pairs, filename):
    """Save similar movie pairs to CSV file (Part C)"""
    # Sort pairs for consistent output
    sorted_pairs = sorted(pairs)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for id1, id2 in sorted_pairs:
            writer.writerow([id1, id2])


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 script.py <API_KEY>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    print("Part B: Fetching 300 most popular Comedy movies from 2000 or later...")
    movies = get_300_comedy_movies(api_key)
    print(f"Found {len(movies)} movies")
    
    print("Saving movie_ID_name.csv...")
    save_movie_names(movies, "movie_ID_name.csv")
    
    print("\nPart C: Fetching similar movies for each movie...")
    pairs = get_all_similar_pairs(api_key, movies)
    print(f"Found {len(pairs)} unique similar movie pairs")
    
    print("Saving movie_ID_sim_movie_ID.csv...")
    save_similar_pairs(pairs, "movie_ID_sim_movie_ID.csv")
    
    print("\nDone! Generated movie_ID_name.csv and movie_ID_sim_movie_ID.csv")


if __name__ == "__main__":
    main()
