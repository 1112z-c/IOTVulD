import torch.nn as nn
from functions import ReverseLayerF
import torch
import copy


def l2norm(t):
    return torch.nn.functional.normalize(t, dim=-1)


class BertModel(nn.Module):
    def __init__(self, encoder,config,tokenizer,args):
        super(BertModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.max_pool = nn.MaxPool1d(kernel_size=1, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=768, kernel_size=3, padding=1)
        self.fc = nn.Linear(768, 2)
        self.to_text_latent1 = nn.Linear(768, 768, bias=False)
        self.batch_norm = nn.BatchNorm1d(768)
        self.to_text_latent2 = nn.Linear(768,768, bias=False)
        self.classifier_domain = nn.Sequential(
            nn.Linear(in_features=768, out_features=512, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=2),
            nn.LogSoftmax(dim=-1)
        )
    def extract_feature_conv(self, x):
        x = x.unsqueeze(1)  # Add a dimension for Conv1D
        x = x.permute(0, 2, 1)  # Change dimensions for Conv1D
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.dropout(x)
        return x


    def forward(self, input_ids=None, alpha=0):

        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True, return_dict=True)

        cls_first = outputs.hidden_states[1][:, 0, :]
        cls_last = outputs.hidden_states[-1][:, 0, :]

        cls = cls_first + cls_last

        cls = self.extract_feature_conv(cls)
        cls = l2norm(self.to_text_latent1(cls))
        cls = self.batch_norm(self.to_text_latent2(cls))

        reverse_feature = ReverseLayerF.apply(cls, alpha)
        domain_output = self.classifier_domain(reverse_feature)


        logits = self.fc(cls)
        return logits, cls, domain_output

