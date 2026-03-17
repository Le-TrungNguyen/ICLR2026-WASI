from torch.utils.data import Dataset  # Import từ đây
# Thêm class mới cho C4 dataset
class TextDataset(Dataset):
    """Dataset wrapper cho C4 text data"""
    def __init__(self, data, tokenizer, max_length=512, task='lm'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task  # 'lm' for language modeling, 'cls' for classification
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        if self.task == 'lm':
            # Language modeling: labels = input_ids
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
            # Classification task - cần thêm label (ví dụ: sentiment, topic...)
            # Tạm thời dùng dummy label
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': 0  # Dummy label cho classification
            }