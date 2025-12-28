"""
Books dataset loading module.

This module provides functions to load the Books dataset from CSV files
and prepare it for recommendation models.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Default column names
DEFAULT_USER_COL = "userID"
DEFAULT_ITEM_COL = "itemID"
DEFAULT_RATING_COL = "rating"
DEFAULT_HEADER = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]

# Data directory path
DATA_DIR = Path(__file__).parent


def load_pandas_df(
    header: Optional[list[str]] = None,
    title_col: Optional[str] = None,
    author_col: Optional[str] = None,
    year_col: Optional[str] = None,
    publisher_col: Optional[str] = None,
    sample_frac: Optional[float] = None,
    min_rating: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load the Books dataset as a pandas DataFrame.

    Loads ratings data from Ratings.csv and optionally merges with
    book metadata from Books.csv.

    Args:
        header: Column names for the rating data [userID, itemID, rating].
            Defaults to DEFAULT_HEADER if None.
        title_col: Book title column name. If None, the column will not be loaded.
        author_col: Author column name. If None, the column will not be loaded.
        year_col: Publication year column name. If None, the column will not be loaded.
        publisher_col: Publisher column name. If None, the column will not be loaded.
        sample_frac: Fraction of data to sample (0.0 to 1.0). If None, load all data.
        min_rating: Minimum rating threshold. Only ratings >= min_rating are kept.
        seed: Random seed for sampling.

    Returns:
        pd.DataFrame: Books rating dataset with columns based on header.

    Raises:
        FileNotFoundError: If data files are not found in the data directory.

    Examples:
        >>> # Load basic rating data
        >>> df = load_pandas_df()
        >>> df.head()
           userID  itemID  rating

        >>> # Load with book metadata
        >>> df = load_pandas_df(
        ...     title_col='Title',
        ...     author_col='Author',
        ...     year_col='Year'
        ... )
    """
    if header is None:
        header = DEFAULT_HEADER.copy()
    elif len(header) < 2:
        raise ValueError("Header must contain at least user and item column names")
    elif len(header) > 3:
        header = header[:3]

    # Load ratings data
    ratings_path = DATA_DIR / "Ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")

    df = pd.read_csv(
        ratings_path,
        encoding="ISO-8859-1",
        on_bad_lines="skip",
    )

    # Rename columns to match header
    df = df.rename(columns={
        "User-ID": header[0],
        "ISBN": header[1],
        "Book-Rating": header[2] if len(header) > 2 else "rating",
    })

    # Keep only required columns
    keep_cols = header[:min(len(header), 3)]
    df = df[keep_cols]

    # Convert rating to float
    if len(header) > 2:
        df[header[2]] = df[header[2]].astype(float)

    # Filter by minimum rating if specified
    if min_rating is not None and len(header) > 2:
        df = df[df[header[2]] >= min_rating]

    # Sample data if specified
    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)

    # Load book metadata if any metadata columns are requested
    if any([title_col, author_col, year_col, publisher_col]):
        item_df = load_item_df(
            item_col=header[1],
            title_col=title_col,
            author_col=author_col,
            year_col=year_col,
            publisher_col=publisher_col,
        )
        if item_df is not None:
            df = df.merge(item_df, on=header[1], how="left")

    return df.reset_index(drop=True)


def load_item_df(
    item_col: str = DEFAULT_ITEM_COL,
    title_col: Optional[str] = None,
    author_col: Optional[str] = None,
    year_col: Optional[str] = None,
    publisher_col: Optional[str] = None,
) -> pd.DataFrame | None:
    """
    Load book metadata from Books.csv.

    Args:
        item_col: Column name for book identifier (ISBN).
        title_col: Book title column name. If None, the column will not be loaded.
        author_col: Author column name. If None, the column will not be loaded.
        year_col: Publication year column name. If None, the column will not be loaded.
        publisher_col: Publisher column name. If None, the column will not be loaded.

    Returns:
        pd.DataFrame or None: Book metadata DataFrame, or None if no columns requested.

    Raises:
        FileNotFoundError: If Books.csv is not found.
    """
    if all(col is None for col in [title_col, author_col, year_col, publisher_col]):
        return None

    books_path = DATA_DIR / "Books.csv"
    if not books_path.exists():
        raise FileNotFoundError(f"Books file not found: {books_path}")

    # Define column mapping
    col_mapping = {"ISBN": item_col}
    usecols = ["ISBN"]

    if title_col:
        col_mapping["Book-Title"] = title_col
        usecols.append("Book-Title")
    if author_col:
        col_mapping["Book-Author"] = author_col
        usecols.append("Book-Author")
    if year_col:
        col_mapping["Year-Of-Publication"] = year_col
        usecols.append("Year-Of-Publication")
    if publisher_col:
        col_mapping["Publisher"] = publisher_col
        usecols.append("Publisher")

    item_df = pd.read_csv(
        books_path,
        usecols=usecols,
        encoding="ISO-8859-1",
        on_bad_lines="skip",
    )

    item_df = item_df.rename(columns=col_mapping)

    # Clean year column if present
    if year_col and year_col in item_df.columns:
        item_df[year_col] = pd.to_numeric(item_df[year_col], errors="coerce")

    return item_df


def load_user_df(
    user_col: str = DEFAULT_USER_COL,
    location_col: Optional[str] = None,
    age_col: Optional[str] = None,
) -> pd.DataFrame | None:
    """
    Load user metadata from Users.csv.

    Args:
        user_col: Column name for user identifier.
        location_col: Location column name. If None, the column will not be loaded.
        age_col: Age column name. If None, the column will not be loaded.

    Returns:
        pd.DataFrame or None: User metadata DataFrame, or None if no columns requested.

    Raises:
        FileNotFoundError: If Users.csv is not found.
    """
    if all(col is None for col in [location_col, age_col]):
        return None

    users_path = DATA_DIR / "Users.csv"
    if not users_path.exists():
        raise FileNotFoundError(f"Users file not found: {users_path}")

    col_mapping = {"User-ID": user_col}
    usecols = ["User-ID"]

    if location_col:
        col_mapping["Location"] = location_col
        usecols.append("Location")
    if age_col:
        col_mapping["Age"] = age_col
        usecols.append("Age")

    user_df = pd.read_csv(
        users_path,
        usecols=usecols,
        encoding="ISO-8859-1",
        on_bad_lines="skip",
    )

    user_df = user_df.rename(columns=col_mapping)

    # Clean age column if present
    if age_col and age_col in user_df.columns:
        user_df[age_col] = pd.to_numeric(user_df[age_col], errors="coerce")

    return user_df


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    user_col: str = DEFAULT_USER_COL,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets using random split.

    Args:
        df: Input DataFrame with user-item-rating data.
        test_size: Fraction of data to use for testing (0.0 to 1.0).
        user_col: Name of the user column.
        seed: Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    np.random.seed(seed)

    # Shuffle the data
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Calculate split point
    split_idx = int(len(df_shuffled) * (1 - test_size))

    train_df = df_shuffled.iloc[:split_idx].reset_index(drop=True)
    test_df = df_shuffled.iloc[split_idx:].reset_index(drop=True)

    return train_df, test_df


def create_user_item_mappings(
    df: pd.DataFrame,
    user_col: str = DEFAULT_USER_COL,
    item_col: str = DEFAULT_ITEM_COL,
) -> tuple[dict, dict, dict, dict]:
    """
    Create mappings between original IDs and contiguous indices.

    Args:
        df: Input DataFrame with user and item columns.
        user_col: Name of the user column.
        item_col: Name of the item column.

    Returns:
        tuple containing:
            - user2idx: Dict mapping user IDs to indices
            - idx2user: Dict mapping indices to user IDs
            - item2idx: Dict mapping item IDs to indices
            - idx2item: Dict mapping indices to item IDs
    """
    unique_users = df[user_col].unique()
    unique_items = df[item_col].unique()

    user2idx = {user: idx for idx, user in enumerate(unique_users)}
    idx2user = {idx: user for user, idx in user2idx.items()}

    item2idx = {item: idx for idx, item in enumerate(unique_items)}
    idx2item = {idx: item for item, idx in item2idx.items()}

    return user2idx, idx2user, item2idx, idx2item


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Get basic statistics about the dataset.

    Args:
        df: Input DataFrame with user-item-rating data.

    Returns:
        dict: Dictionary containing dataset statistics.
    """
    stats = {
        "n_ratings": len(df),
        "n_users": df[DEFAULT_USER_COL].nunique() if DEFAULT_USER_COL in df.columns else None,
        "n_items": df[DEFAULT_ITEM_COL].nunique() if DEFAULT_ITEM_COL in df.columns else None,
    }

    if DEFAULT_RATING_COL in df.columns:
        stats["rating_mean"] = df[DEFAULT_RATING_COL].mean()
        stats["rating_std"] = df[DEFAULT_RATING_COL].std()
        stats["rating_min"] = df[DEFAULT_RATING_COL].min()
        stats["rating_max"] = df[DEFAULT_RATING_COL].max()

    if stats["n_users"] and stats["n_items"]:
        stats["sparsity"] = 1 - (stats["n_ratings"] / (stats["n_users"] * stats["n_items"]))

    return stats
