import transformers
import torch
from torch.utils.data import RandomSampler, DataLoader
from transformers import AdamW
from model import Squad
from tqdm import tqdm


def train(epoch: int, batch_size: int, dataset: torch.utils.data.TensorDataset, device, lr: float = 2e-5, adam_epsilon: float = 1e-8, verbose: int = 1):
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(
        dataset, sampler=train_sampler, batch_size=batch_size)
    model = Squad()
    optimzer = AdamW(model.parameters(), lr=lr, eps=adam_epsilon)
    model.to(device)
    model.train()
    model.zero_grad()
    total_loss = 0.0

    for i in range(epoch):
        print('-' * 80)
        print('Begin epoch {}.'.format(i))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[4],
                "end_positions": batch[5]
            }

            outputs = model(**inputs)

            loss = outputs[2]
            loss.backward()

            total_loss += loss.item()
            if (step + 1) % verbose == 0:
                optimzer.step()
                print("Step: {} | loss: {}".format(step + 1, total_loss / verbose))
                total_loss = 0
                model.zero_grad()


def run():
    pass


def get_dataset():
    dataset = torch.load("./save/dataset.pt")
    return dataset


if __name__ == "__main__":
    dataset = get_dataset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train(2, 4, dataset=dataset, device=torch.device("cpu"))
