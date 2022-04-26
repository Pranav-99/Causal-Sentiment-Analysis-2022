import torch

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import wordpunct_tokenize


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def kl_loss_function(z_mu, z_logvar):
    '''
    z_mu.shape: (B, z_dim)
    z_logvar.shape: (B, z_dim)
    '''
    z_var = torch.exp(z_logvar)
    return - 0.5 * (1 + z_logvar - (z_mu ** 2) - z_var).sum(dim=1)


def recon_loss_function(x, x_hat):
    return ((x - x_hat) ** 2).sum(dim=1)



def prepare_data(all_train_df, test_df, ind=1):
    
    train_df = all_train_df[ind]
    train_df = train_df.sample(frac=1)

    train_df['text'] = train_df['text'].apply(lambda x: x.lower())
    test_df['text'] = test_df['text'].apply(lambda x: x.lower())

    train_df['text'] = train_df['text'].apply(lambda x: " ".join(wordpunct_tokenize(x)))
    test_df['text'] = test_df['text'].apply(lambda x: " ".join(wordpunct_tokenize(x)))

    tokenizer = Tokenizer(num_words=2000, lower=True, split=' ', filters='#%&()*+-/:;<=>@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(train_df['text'].values)

    X_train = tokenizer.texts_to_sequences(train_df['text'].values)
    X_train = pad_sequences(X_train, maxlen=350)

    X_test = tokenizer.texts_to_sequences(test_df['text'].values)
    X_test = pad_sequences(X_test, maxlen=350)

    return X_train, X_test, train_df, test_df
