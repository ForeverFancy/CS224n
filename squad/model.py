import torch
import transformers
import torch.nn as nn

class Squad(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(Squad, self).__init__()
        model_class = transformers.BertModel
        UNCASED = '../bert_base'
        self.bert_model = model_class.from_pretrained(UNCASED)
        # labels:2 (start token, end token)
        self.linear = nn.Linear(in_features=768, out_features=2)

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, token_type_ids: torch.tensor, start_positions: torch.tensor = None, end_positions: torch.tensor = None):
        '''
        @param input_ids(torch.tensor) of shape(batch_size, sentence_length)

        @param attention_mask(torch.tensor) of shape(batch_size, sentence_length)
        
        @param token_type_ids(torch.tensor) of shape(batch_size, sentence_length)
        
        @param start_positions(torch.tensor) of shape(batch_size, sentence_length)
        
        @param end_positions(torch.tensor) of shape(batch_size, sentence_length)

        @return start_logits of shape (batch_size, sentence_length): score of each token being start token

        @return end_logits of shape (batch_size, sentence_length): score of each token being end token

        @return mean_loss: (start position loss + end position loss )/2 CrossEntropyLoss, could be None
        '''
        outputs = self.bert_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]  # shape (batch_size, sentence_length, hidden_size)
        
        logits = self.linear(sequence_output)  # output shape (batch_size, sentence_length, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        # shape (batch_size, sentence_length): use this to calculate the probability being start token
        start_logits = start_logits.squeeze(-1)
        # shape (batch_size, sentence_length): use this to calculate the probability being end token
        end_logits = end_logits.squeeze(-1)

        mean_loss = None
        if start_positions is not None and end_positions is not None:
            ignore_index = start_logits.size(1)
            # Prevent start position and end position out of our sentence
            start_positions.clamp_(0, ignore_index)
            end_positions.clamp_(0, ignore_index)

            celoss = nn.CrossEntropyLoss(ignore_index=ignore_index)
            start_loss = celoss(start_logits, start_positions)
            end_loss = celoss(start_logits, end_positions)
            mean_loss = (start_loss + end_loss)/2

        return (start_logits, end_logits, mean_loss)

