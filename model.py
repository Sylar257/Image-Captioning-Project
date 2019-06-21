import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    ### For Encoder we used the ResNet50 model pre-trained from image-nets. More powerful model can be applied here. However, ResNet50 performs reasonably well with a decent amount of training. We would freeze all the weights in the pre-trained model and remove its output layer. The removed output layer is replaced with our own Linear layer that takes in the number of output features and produce a feature of size: embedding size. Therefore, it is important that when creating an instance of our EncoderCNN model we make sure to take in parameter embed_size to specify our network sufficiently.       
    
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch_norm = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batch_norm(features)
        return features
    

class DecoderRNN(nn.Module):
    ### The decoder is a simple LSTM architecture with dropout for regularization. When specifying out decoder, make sure we have the correct vocab_size which is determined when we decide our vocab_threshold and construct our Vocabulary. It is also important to imbed our target captions using nn.Embedding() first before feeding into our network.
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.embed_size = embed_size
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,dropout=0.4,batch_first=True)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(hidden_size, vocab_size)
#         self.softmax = nn.Softmax(dim=1)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens,_ = self.lstm(embeddings)
        print("the hiddens.shape:",hiddens.shape) # keep track of the output during training, you will see that caption lengths are differnt from each batch depending on the sampling
        outputs = self.fc(hiddens[:,:-1,:])
        return outputs        
        

    def sample(self, inputs, states=None, max_len=25):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids=[]
        # Terminate captioning producing of max_len of words are reached
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states) # hiddens:  (batch_size,1,hidden_size)
            outputs = self.fc(hiddens.squeeze(1))       # outputs:  (batch_size,vocab_size)
            _, predicted = outputs.max(1)               # predicted:(batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)              # inputs:   (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                # inputs:   (batch_size,1,embed_size)
            
        sampled_ids = torch.stack(sampled_ids, 1)       # sampled_ids:(batch_size, max_seq_length)
        sampled_ids = sampled_ids[0].tolist() # in the case of batch_size = 1(only testing on one image)
        return sampled_ids