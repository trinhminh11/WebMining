import unittest
import torch
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.backend.api.cold_start.model import GNN, EmerG
from app.backend.api.cold_start.infer import infer

class TestInfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a dummy checkpoint
        cls.ckpt_dir = Path("app/backend/api/cold_start/tests/ckpt")
        cls.ckpt_dir.mkdir(parents=True, exist_ok=True)
        cls.ckpt_path = cls.ckpt_dir / "dummy_model.pt"

        cls.description = {
            'user_id': (100, 'spr'),
            'item_id': (50, 'spr'),
            'location': (20, 'seq'),
            'age': (7, 'spr'),
            'author': (30, 'spr'),
            'year': (10, 'spr'),
            'publisher': (15, 'spr')
        }
        embed_dim = 32
        gnn = GNN(cls.description, embed_dim, gnn_layers=2)
        model = EmerG(gnn, embed_dim * 4, device='cpu')

        torch.save(model.state_dict(), cls.ckpt_path)

    @classmethod
    def tearDownClass(cls):
        if cls.ckpt_dir.exists():
            shutil.rmtree(cls.ckpt_dir)

    @patch('app.backend.api.cold_start.infer.BooksColdStartDataLoader')
    def test_inference_run(self, MockLoader):
        # Setup Mock
        mock_instance = MockLoader.return_value
        mock_instance.description = self.description

        # Mock DataLoader to return one batch
        bsz = 4
        # Create dummy batch
        batch_features = {
            'user_id': torch.randint(0, 100, (bsz,)),
            'item_id': torch.tensor([0, 0, 1, 1], dtype=torch.long), # Ensure valid items
            'location': torch.randint(0, 20, (bsz, 5)),
            'age': torch.randint(0, 7, (bsz,)),
            'author': torch.randint(0, 30, (bsz,)),
            'year': torch.randint(0, 10, (bsz,)),
            'publisher': torch.randint(0, 15, (bsz,))
        }
        batch_labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        mock_instance.get_loader.return_value = [(batch_features, batch_labels)]

        # Mock args
        class Args:
            data_dir = "dummy_dir"
            ckpt_path = str(self.ckpt_path)
            output_path = "app/backend/api/cold_start/tests/results.json"
            batch_size = 4
            embed_dim = 32
            gnn_layers = 2
            device = 'cpu'

        # Run inference
        try:
            infer(Args())
        except Exception as e:
            self.fail(f"Inference failed: {e}")

        # Check output
        out_path = Path(Args.output_path)
        self.assertTrue(out_path.exists())

        with open(out_path, 'r') as f:
            data = json.load(f)
            self.assertIn('avg_loss', data)
            self.assertIn('details', data)

if __name__ == '__main__':
    unittest.main()
