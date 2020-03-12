import unittest
import torch
from model import Squad
import pickle

class ModelCheck(unittest.TestCase):
    def forward_check(self):
        input_features = torch.load('./save/input_feature.pt')
        attention_mask = torch.load('./save/attention_mask.pt')
        token_type_ids = torch.load('./save/token_type_ids.pt')
        start_positions = torch.load('./save/start_positions.pt')
        end_positions = torch.load('./save/end_positions.pt')
        is_impossible = torch.load('./save/is_impossible.pt')
        with open('./save/ids.pt', 'rb') as f:
            ids = pickle.load(f)
        with open('./save/answer_text.pt', 'rb') as f:
            answer_text = pickle.load(f)

        assert input_features.shape == attention_mask.shape == token_type_ids.shape
        network = Squad()
        start_logits, end_logits, loss = network.forward(input_features[:10, :], attention_mask[:10, :], token_type_ids[:10, :], start_positions[:10], end_positions[:10])
        print(loss.item())
        assert start_logits.shape == input_features[:10, :].shape
        assert end_logits.shape == input_features[:10, :].shape

if __name__ == "__main__":
    unittest.TextTestRunner().run(ModelCheck("forward_check"))
        
