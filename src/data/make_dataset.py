from typing import Optional, List
from time import sleep, time
import spotipy
from lyricsgenius import Genius
import pandas as pd


def create_dataset(genres_types: List[str],
                   sp: spotipy.Spotify,
                   genius: Genius,
                   limit: Optional[int] = 50,
                   how_many_in_genre: Optional[int] = 2_000,
                   sleep_time: Optional[int] = 30,
                   path_to_save: Optional[str] = None,
                   verbose: Optional[bool] = True,
                   save_progess: Optional[bool] = True) -> pd.DataFrame:
    """
    Function scrapes fixed number of songs by genre from Spotify API
    and their lyrics from Genius API.

    Parameters
    ----------
    genres_types (list): list of selected genres to scrape.
    sp (spotipy.Spotify): connector to Spotify API. It should be defined with credentials specified.
    genius (Genius) : connector to Genius API. It should be defined with credentials specified.
    limit (int) : How many observations get with single request. The max is 50. Needs to be smaller than
                  how_many_in_genre and limit % how_many_in_genre == 0
    how_many_in_genre (int) : how many observations should be in single genre class. Then, the whole dataset has
                              len(genre_types) * how_many_in_genre observations.
    sleep_time (int) : seconds of function sleep after each spotipy request. Set this up to avoid too many request error
    path_to_save (str) : path to save resulted data frame
    verbose (bool) : if true then it will print progress of our function
    save_progess (bool) : if ture then after each spotipy request the current result will be saved

    Returns
    -------
        df (pd.DataFrame) : scraped dataframe with author, title, popularity, genre and lyrics columns
    """
    # iterator needed when verbose is true
    i_sample = 0
    max_sample = len(genres_types) * how_many_in_genre

    # Define empty lists to locate scraped data
    artist_names = []
    track_names = []
    popularities = []
    genres = []
    lyrics = []
    # If verbose then print progress
    if verbose:
        print(f"Number of scraped samples: {i_sample}/{max_sample}")

    for genre in genres_types:
        # offset is the start index of API results
        # maximum limit of request is 50 observations so we need to change offset in each iteration
        for offset in range(0, how_many_in_genre, limit):
            t1 = time()
            # Scraped data from spotify
            track_results = sp.search(q=f'genre:"{genre}"', type='track', limit=limit, offset=offset)
            # We will iterate over our results
            for i, t in enumerate(track_results['tracks']['items']):
                # Extract artist name
                artist_name = t['artists'][0]['name']
                # Extract title of song
                track_name = t['name']

                # Save results
                artist_names.append(artist_name)
                track_names.append(track_name)
                popularities.append(t['popularity'])
                genres.append(genre)
                # We use try/except because genius has limited timeout to single request
                # and raise error when there is timeout
                # If this situation occurs wie will need to rescrape certain lyrics
                try:
                    # Searching for lyrics to scraped song
                    text = genius.search_song(track_name, artist_name).lyrics
                except:
                    text = f"Error in {artist_name} - {track_name}"
                lyrics.append(text)
            # After each iteration the function will sleep to avoid too many request error
            sleep(sleep_time)
            # Monitoring progess
            i_sample += limit
            if verbose:
                t2 = time()
                print(f"Number of scraped samples: {i_sample}/{max_sample}. Time: {(t2 - t1) / 60:.2f} min")
            # We can save result after each request
            if save_progess:
                df = pd.DataFrame({'artist_name': artist_names,
                                   'track_name': track_names,
                                   'popularity': popularities,
                                   "genre": genres,
                                   "lyrics": lyrics})
                if path_to_save:
                    df.to_csv(path_to_save)

    # Creating data frame
    df = pd.DataFrame({'artist_name': artist_names,
                       'track_name': track_names,
                       'popularity': popularities,
                       "genre": genres,
                       "lyrics": lyrics})

    # Shuffle pandas df rows, reset index and drop duplicates
    df = df.sample(frac=1, random_state=7).reset_index(drop=True).drop_duplicates()

    # Save to specified path
    if path_to_save:
        df.to_csv(path_to_save)

    return df


def read_main_df(path: str,
                 drop_columns: Optional[List[str]]=None,
                 unstructured_columns: Optional[List[str]]=None) -> pd.DataFrame:
    """
    Function read main data frame and performs basic data type formatting

    Args:
        path: path to csv file with data set
        drop_columns: list of unnecessary columns to remove
        unstructured_columns: list of columns which may contains not valid data structures for pandas
                            like lists or dictionaries

    Returns:
        df: ready data set
    """
    # Importing csv file
    df = pd.read_csv(path)

    # Drop unused columns
    df.drop(drop_columns, axis=1, inplace=True)

    # Converting columns with lists/dicts to usable structure
    for column in unstructured_columns:
        df[column] = df[column].apply(eval)

    return df
