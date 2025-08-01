import torch

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

inputs = torch.LongTensor(x_data).unsqueeze(0) # 形状变为[1,5]
print(inputs)
print(inputs.shape)
labels = torch.LongTensor(y_data)

num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)
    
    def forward(self, x):
        hidden = torch.zeros(num_layers, batch_size, hidden_size)
        print('hidden.shape:', hidden.shape)
        x = self.emb(x)
        print('x.shape:', x.shape)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class)
    
net = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[i] for i in idx]), end='')
    print(', Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1,15, loss.item()))                  