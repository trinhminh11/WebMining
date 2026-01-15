import unittest
import torch
from pathlib import Path
from app.backend.api.cold_start.dataset import BooksColdStartDataLoader, BooksDataset

class TestBooksDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Point to the actual data directory for this test since we want to verifying integration with real file formats
        # Ideally we would create a small synthetic CSV here, but using real data confirms parsing works.
        cls.data_dir = Path("/Users/thefool/Local/Project/WebMining/Books/app/backend/api/data")

    def test_loader_initialization(self):
        """Test that loader initializes and processes features without error."""
        try:
            loader = BooksColdStartDataLoader(self.data_dir, bsz=4, shuffle=False)
            self.loader = loader
        except Exception as e:
            self.fail(f"Loader initialization failed: {e}")

        self.assertGreater(loader.num_users, 0)
        self.assertGreater(loader.num_items, 0)
        self.assertGreater(loader.num_loc_tokens, 0)

        # Check description
        self.assertIn('location', loader.description)
        self.assertEqual(loader.description['location'][1], 'seq')

    def test_batch_shapes(self):
        """Test that the DataLoader yields batches with correct shapes."""
        loader = BooksColdStartDataLoader(self.data_dir, bsz=4, shuffle=False)
        dl = loader.get_loader('train')

        batch = next(iter(dl))
        features, labels = batch

        # Check Batch Size
        self.assertEqual(labels.shape[0], 4)

        # Check Feature Tensos
        self.assertTrue(torch.is_tensor(features['user_id']))
        self.assertTrue(torch.is_tensor(features['location']))

        # Check Location Sequence Length (should be 5 based on implementation)
        self.assertEqual(features['location'].shape[1], 5)

        # Check Types
        self.assertEqual(features['user_id'].dtype, torch.long)
        self.assertEqual(labels.dtype, torch.float)

if __name__ == '__main__':
    unittest.main()
