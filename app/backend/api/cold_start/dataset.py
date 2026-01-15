import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import re

# Constants
DEFAULT_USER_COL = "User-ID"
DEFAULT_ITEM_COL = "ISBN"
DEFAULT_RATING_COL = "Book-Rating"

class BooksColdStartDataLoader:
    def __init__(self, data_dir, device='cpu', bsz=32, shuffle=True, maml_episode=False):
        self.data_dir = Path(data_dir)
        self.device = device
        self.bsz = bsz
        self.shuffle = shuffle
        self.maml_episode = maml_episode

        self.books_path = self.data_dir / "Books.csv"
        self.users_path = self.data_dir / "Users.csv"
        self.ratings_path = self.data_dir / "Ratings.csv"

        # Load and Preprocess Data
        self._load_data()
        self._preprocess_features()
        self._create_splits()

    def _load_data(self):
        """Loads CSV files."""
        print(f"Loading data from {self.data_dir}...")
        self.books = pd.read_csv(self.books_path, dtype={DEFAULT_ITEM_COL: str})
        self.users = pd.read_csv(self.users_path, dtype={DEFAULT_USER_COL: int})
        self.ratings = pd.read_csv(self.ratings_path, dtype={DEFAULT_USER_COL: int, DEFAULT_ITEM_COL: str})

        # Clean IDs
        self.books[DEFAULT_ITEM_COL] = self.books[DEFAULT_ITEM_COL].str.strip()
        self.ratings[DEFAULT_ITEM_COL] = self.ratings[DEFAULT_ITEM_COL].str.strip()

        # Filter ratings using merge (Items ONLY)
        # Users.csv has disjoint IDs from Ratings.csv in this dataset version, so we ignore Users.csv for filtering.
        print("DEBUG: Filtering ratings by Books only (ignoring Users.csv due to mismatch)...")
        valid_items = self.books[[DEFAULT_ITEM_COL]]

        self.ratings = self.ratings.merge(valid_items, on=DEFAULT_ITEM_COL, how='inner')
        print(f"DEBUG: Filtered Ratings: {len(self.ratings)}")

    def _preprocess_features(self):
        """Encodes features and prepares description dictionary."""
        print("Preprocessing features...")

        # --- User Features ---
        self.users['Location'] = self.users['Location'].fillna('').astype(str).str.lower()

        # Build Location Vocabulary
        all_loc_tokens = set()
        for loc in self.users['Location']:
            parts = [p.strip() for p in loc.split(',')]
            all_loc_tokens.update(parts)

        self.loc_token2idx = {t: i + 1 for i, t in enumerate(all_loc_tokens)} # 0 for padding
        self.num_loc_tokens = len(self.loc_token2idx) + 1

        self.num_age_buckets = 7

        # --- Item Features ---
        self.author_encoder = LabelEncoder()
        self.books['Book-Author'] = self.books['Book-Author'].fillna('Unknown')
        self.books['Author_Idx'] = self.author_encoder.fit_transform(self.books['Book-Author'])
        self.num_authors = len(self.author_encoder.classes_)

        self.books['Year-Of-Publication'] = pd.to_numeric(self.books['Year-Of-Publication'], errors='coerce').fillna(2000)
        self.books['Year_Idx'] = pd.cut(self.books['Year-Of-Publication'], bins=[0, 1950, 1980, 1990, 2000, 2010, 2025], labels=False)
        self.books['Year_Idx'] = self.books['Year_Idx'].fillna(3).astype(int)
        self.num_years = 7

        # 5. Publisher
        self.publisher_encoder = LabelEncoder()
        self.books['Publisher'] = self.books['Publisher'].fillna('Unknown')
        self.books['Publisher_Idx'] = self.publisher_encoder.fit_transform(self.books['Publisher'])
        self.num_publishers = len(self.publisher_encoder.classes_)

        # 6. Title
        self.books['Title_Len'] = self.books['Book-Title'].str.len().fillna(0).astype(int)

        # Map IDs to Indices
        unique_rating_users = self.ratings[DEFAULT_USER_COL].unique()
        self.user_encoder = LabelEncoder()
        self.user_encoder.fit(unique_rating_users)
        self.ratings['User_Idx'] = self.user_encoder.transform(self.ratings[DEFAULT_USER_COL])

        self.item_encoder = LabelEncoder()
        self.books['Item_Idx'] = self.item_encoder.fit_transform(self.books[DEFAULT_ITEM_COL])
        # Transform items in ratings
        self.ratings['Item_Idx'] = self.item_encoder.transform(self.ratings[DEFAULT_ITEM_COL])

        self.num_users = len(self.user_encoder.classes_)
        self.num_items = len(self.item_encoder.classes_)

        # Store Feature Description for Model
        self.description = {
            'user_id': (self.num_users, 'spr'),
            'item_id': (self.num_items, 'spr'),
            'location': (self.num_loc_tokens, 'seq'),
            'age': (self.num_age_buckets, 'spr'),
            'author': (self.num_authors, 'spr'),
            'year': (self.num_years, 'spr'),
            'publisher': (self.num_publishers, 'spr'),
        }

    def _create_splits(self):
        """Creates Train (Warm) / Test (Cold) splits."""
        if len(self.ratings) == 0:
            print("WARNING: No ratings found after filtering!")
            self.train_data = pd.DataFrame()
            self.test_data = pd.DataFrame()
            self.item_features_map = {}
            self.user_features_map = {}
            return

        # Items are already transformed in preprocess
        rated_items = self.ratings['Item_Idx'].unique()
        n_warm = int(0.8 * len(rated_items))

        np.random.seed(42)
        warm_items = np.random.choice(rated_items, n_warm, replace=False)
        self.warm_items_set = set(warm_items)

        # Create lookups (Drop duplicates to ensure unique index)
        unique_books = self.books.drop_duplicates(subset=['Item_Idx'])
        self.item_features_map = unique_books.set_index('Item_Idx')[['Author_Idx', 'Year_Idx', 'Publisher_Idx']].to_dict('index')

        # User Features Map: DEFAULT
        self.user_features_map = {}

        self.ratings['Label'] = (self.ratings[DEFAULT_RATING_COL] > 5).astype(float)

        self.train_data = self.ratings[self.ratings['Item_Idx'].isin(self.warm_items_set)]
        self.test_data = self.ratings[~self.ratings['Item_Idx'].isin(self.warm_items_set)]

        print(f"Data Loaded: {len(self.train_data)} train interactions, {len(self.test_data)} test interactions.")

    def get_loader(self, mode='train'):
        if mode == 'train':
            dataset = BooksDataset(self.train_data, self.item_features_map, self.user_features_map, self.loc_token2idx)
        else:
            dataset = BooksDataset(self.test_data, self.item_features_map, self.user_features_map, self.loc_token2idx)

        return DataLoader(dataset, batch_size=self.bsz, shuffle=self.shuffle)


class BooksDataset(Dataset):
    def __init__(self, ratings_df, item_features, user_features, loc_token2idx):
        self.data = ratings_df.reset_index(drop=True)
        self.item_features = item_features
        self.user_features = user_features
        self.loc_token2idx = loc_token2idx
        self.max_loc_len = 5 # Truncate location tokens to 5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        uid, iid, label = int(row['User_Idx']), int(row['Item_Idx']), float(row['Label'])

        # User Features (Handling missing map)
        if uid in self.user_features:
             u_feat = self.user_features[uid]
             age_idx = u_feat['Age_Bucket']
             loc_str = u_feat['Location']
             if pd.isna(age_idx): age_idx = 0
        else:
             # Default
             age_idx = 0
             loc_str = ""

        # Location Processing (Set -> Indices)
        loc_tokens = [p.strip() for p in loc_str.split(',')]
        loc_indices = [self.loc_token2idx.get(t, 0) for t in loc_tokens][:self.max_loc_len]
        # Pad
        if len(loc_indices) < self.max_loc_len:
            loc_indices += [0] * (self.max_loc_len - len(loc_indices))

        # Item Features
        i_feat = self.item_features[iid]

        features = {
            'user_id': torch.tensor(uid, dtype=torch.long),
            'item_id': torch.tensor(iid, dtype=torch.long),
            'age': torch.tensor(int(age_idx), dtype=torch.long),
            'location': torch.tensor(loc_indices, dtype=torch.long),
            'author': torch.tensor(i_feat['Author_Idx'], dtype=torch.long),
            'year': torch.tensor(i_feat['Year_Idx'], dtype=torch.long),
            'publisher': torch.tensor(i_feat['Publisher_Idx'], dtype=torch.long),
        }

        return features, torch.tensor(label, dtype=torch.float)

if __name__ == "__main__":
    # Test Block
    API_DIR = Path(__file__).parent.parent
    DATA_DIR = API_DIR / "data"
    print(f"Testing loader with data from {DATA_DIR}")

    loader = BooksColdStartDataLoader(DATA_DIR, bsz=4)
    train_dl = loader.get_loader('train')

    for batch_features, batch_labels in train_dl:
        print("Batch Features keys:", batch_features.keys())
        print("User ID shape:", batch_features['user_id'].shape)
        print("Location shape:", batch_features['location'].shape)
        print("Label shape:", batch_labels.shape)
        break
