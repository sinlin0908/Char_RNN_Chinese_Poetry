import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim


import matplotlib.pyplot as plt
import numpy as np

import pickle

from model import Model
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.manual_seed(1)
torch.manual_seed(1)

config = config.Config


def create_data_set(data: list) -> np.array:
    x = np.array(data)

    print("data shape:{}\n ".format(x.shape))
    return torch.from_numpy(x)


def create_data_loader(data: list) -> Data.DataLoader:

    x = create_data_set(data)
    return Data.DataLoader(
        dataset=x,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )


def draw_loss(loss_history):

    plt.plot(loss_history, label="loss", linewidth=1)
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, int(max(loss_history))))
    plt.show()


def train(model, data_loader):

    print("\n\n------ Start training -------\n\n")
    model.train()
    l_h = []
    epoch_loss = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epoch):
        epoch_loss = 0
        count = 0

        for step, d in enumerate(data_loader):
            count += 1
            optimizer.zero_grad()
            d = d.long().transpose(1, 0).contiguous()  # torch.Size([24, 32])

            # input  第0個 到 倒數第二個 torch.Size([23, 32])
            input_, target = d[:-1, :].to(device), d[1:, :].to(device)
            # output 第一個 到 最後 torch.Size([23, 32])
            hidden = model.init_hidden(input_.size(1)).to(device)

            out, hidden = model(input_, hidden)

            loss = criterion(out, target.view(-1))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            epoch_loss += loss.item()

            if (step+1) % 50 == 0:

                print('Epoch: ', epoch, '| step ', step+1,
                      '| train loss: %.4f' % loss.item())

        print("======== Epoch {} Toal loss {} =========".format(
            epoch, epoch_loss/count))

        l_h.append(epoch_loss/count)

    draw_loss(l_h)
    torch.save(model, 'poetry_model.pickle')
    with open("result_history.pickle", 'wb')as f:
        pickle.dump(l_h)


if __name__ == '__main__':

    with open("./data/id2char_dict.pickle", "rb") as f:
        id2char_dict = pickle.load(f)

    with open("./data/poetry2id_seqs.pickle", "rb") as f:
        poetry2id_seq = pickle.load(f)

    data_loader = create_data_loader(poetry2id_seq)

    model = Model(
        vocab_size=len(id2char_dict),
        embedding_dim=300,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
    ).to(device)

    print("Model Structure\n\n", model)

    train(model, data_loader)
