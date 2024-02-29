import torch


class Generator(torch.nn.Module):
    def __init__(self, input_length, output_image_size, p_dropout):
        super(Generator, self).__init__()
        self.input = torch.nn.Linear(input_length, 256)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.h1 = torch.nn.Linear(self.input.out_features, self.input.out_features*2)
        self.h2 = torch.nn.Linear(self.h1.out_features, self.h1.out_features*2)
        self.output = torch.nn.Linear(self.h2.out_features, output_image_size[1] * output_image_size[2])
        self.sigmoid = torch.nn.Tanh()

    def forward(self, x):
        x = self.input(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.h1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.h2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, input_image_size, p_dropout):
        super(Discriminator, self).__init__()

        self.input = torch.nn.Linear(input_image_size[1] * input_image_size[2], 1024)
        self.activation = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.h1 = torch.nn.Linear(self.input.out_features, self.input.out_features // 2)
        self.h2 = torch.nn.Linear(self.h1.out_features, self.h1.out_features // 2)
        self.output = torch.nn.Linear(self.h2.out_features, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.h1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.h2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
