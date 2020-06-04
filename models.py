import torch.nn as nn
import torch
import torch.nn.functional as F
    
    
class LSTM_Statement(nn.Module):

    def __init__(self, embedding_info, embed_size, output_size, hidden_size, fnn_layer):
        super(LSTM_Statement, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_info = embedding_info
        self.embed_size = embed_size

        self.embedding_statement = self.embed_init('statement', grad=True)
        self.lstm = self.lstm_statement = self.lstm_init('statement', hidden_size)
        
        self.bn1 = nn.BatchNorm1d(2*2*hidden_size)
        self.dropout1 = nn.Dropout(0.2)

        self.fnn = nn.Linear(2*2*hidden_size, fnn_layer)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(fnn_layer)
        self.dropout2 = nn.Dropout(0.2)

        self.label = nn.Linear(fnn_layer, output_size)
        
    def lstm_hidden_transform(self, hidden):
        return torch.cat([hidden[i,:,:] for i in range(int(hidden.shape[0]))], dim=1)
    
    def embed_init(self, feature, grad):
        embed = nn.Embedding(self.embedding_info[feature]['vocab_size'], self.embed_size[feature], padding_idx=1)
        embed.weight = nn.Parameter(self.embedding_info[feature]['field'].vocab.vectors, requires_grad=grad)
        return embed

    def lstm_init(self, feature, size, num_layers=2):
        return nn.LSTM(self.embed_size[feature], size, bidirectional=True, num_layers=num_layers)

    def forward(self, statement, subject, speaker, speaker_job, state, party, context, justification):
        statement_embed = self.embedding_statement(statement)
        
        output, (final_hidden_state, final_cell_state) = self.lstm(statement_embed)
        final_hidden_state = self.lstm_hidden_transform(final_hidden_state)
        final_hidden_state = self.dropout1(self.bn1(final_hidden_state))
        
        fnn = self.dropout2(self.bn2(self.relu2(self.fnn(final_hidden_state))))
        final_output = self.label(fnn)

        return final_output
    
    
class CNN_Statement(nn.Module):

    def __init__(self, embedding_info, embed_size, output_size, fnn_layer, out_channels=128, kernels=(3,4,5)):
        super(CNN_Statement, self).__init__()
        self.output_size = output_size
        self.embed_size = embed_size
        self.embedding_info = embedding_info
        self.embedding_statement = self.embed_init('statement', grad=True)
        
        self.conv1 = nn.Conv2d(1, out_channels, (kernels[0], self.embed_size['statement']), padding=(kernels[0] - 1, 0))
        self.conv2 = nn.Conv2d(1, out_channels, (kernels[1], self.embed_size['statement']), padding=(kernels[1] - 1, 0))
        self.conv3 = nn.Conv2d(1, out_channels, (kernels[2], self.embed_size['statement']), padding=(kernels[2] - 1, 0))
      
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        self.bn1 = nn.BatchNorm1d(len(kernels)*out_channels)
        self.bn2 = nn.BatchNorm1d(fnn_layer)

        self.relu = nn.ReLU()
        self.fnn = nn.Linear(len(kernels)*out_channels, fnn_layer)
       
        self.label = nn.Linear(fnn_layer, output_size)
        
    def embed_init(self, feature, grad):
        embed = nn.Embedding(self.embedding_info[feature]['vocab_size'], self.embed_size[feature], padding_idx=1)
        embed.weight = nn.Parameter(self.embedding_info[feature]['field'].vocab.vectors, requires_grad=grad)
        return embed

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return max_out


    def forward(self, statement, subject, speaker, speaker_job, state, party, context, justification):
        statement_embed = self.embedding_statement(statement.T).unsqueeze(1)
        max_out1 = self.conv_block(statement_embed, self.conv1)
        max_out2 = self.conv_block(statement_embed, self.conv2)
        max_out3 = self.conv_block(statement_embed, self.conv3)
      
        all_out = self.dropout1(self.bn1(torch.cat((max_out1, max_out2, max_out3), 1)))
        fnn = self.dropout2(self.bn2(self.relu(self.fnn(all_out))))
        final_output = self.label(fnn)

        return final_output

    
class LSTM_statement_metadata(nn.Module):


    def __init__(self, embedding_info, embed_size, output_size, fnn_meta_layer, fnn_final_layer):
        super(LSTM_statement_metadata, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embedding_info = embedding_info
        self.one_hot_length = embedding_info['state']['vocab_size'] + \
        embedding_info['party']['vocab_size'] + embedding_info['speaker']['vocab_size']

        # Embeddings
        self.embedding_statement = self.embed_init('statement', grad=True)
        self.embedding_justification = self.embed_init('justification', grad=True)
        self.embedding_subject = self.embed_init('subject', grad=True)
        self.embedding_speaker_job = self.embed_init('speaker_job', grad=True)
        self.embedding_context = self.embed_init('context', grad=True)
        self.embedding_state = self.embed_init('state', grad=False)
        self.embedding_party = self.embed_init('party', grad=False)
        self.embedding_speaker = self.embed_init('speaker', grad=False)

        # Lstm 
        self.lstm_statement = self.lstm_init('statement', hidden_size)
        self.lstm_subject = self.lstm_init('subject', hidden_size)
        self.lstm_speaker_job = self.lstm_init('speaker_job', hidden_size)
        self.lstm_context = self.lstm_init('context', hidden_size)
        
        # Meta data layer
        self.bn_concat = nn.BatchNorm1d(4*3*hidden_size+self.one_hot_length)
        self.drop_concat = nn.Dropout()
        self.fnn_meta = nn.Linear(4*3*hidden_size+self.one_hot_length, fnn_meta_layer)
        self.bn_meta = nn.BatchNorm1d(fnn_meta_layer)
        self.relu_meta = nn.ReLU()
        self.drop_meta = nn.Dropout()
        
        self.drop_statement = nn.Dropout()
        self.bn_statement =  nn.BatchNorm1d(4*hidden_size)

        #Fnn layer
        self.fnn_final = nn.Linear(fnn_meta_layer+4*hidden_size, fnn_final_layer)
        self.bn_final = nn.BatchNorm1d(fnn_final_layer)
        self.relu_final = nn.ReLU()
        self.drop_final = nn.Dropout()
        
        #Output
        self.label = nn.Linear(fnn_final_layer, output_size)

    def lstm_hidden_transform(self, hidden):
        return torch.cat([hidden[i,:,:] for i in range(int(hidden.shape[0]))], dim=1)
    
    def embed_init(self, feature, grad):
        embed = nn.Embedding(self.embedding_info[feature]['vocab_size'], self.embed_size[feature], padding_idx=1)
        embed.weight = nn.Parameter(self.embedding_info[feature]['field'].vocab.vectors, requires_grad=grad)
        return embed

    def lstm_init(self, feature, size, num_layers=2):
        return nn.LSTM(self.embed_size[feature], size, bidirectional=True, num_layers=num_layers)


    def forward(self, statement, subject, speaker, speaker_job, state, party, context, justification):
      
        statement_embed = self.embedding_statement(statement)
        subject_embed = self.embedding_subject(subject)
        speaker_embed = self.embedding_speaker(speaker)
        speaker_job_embed = self.embedding_speaker_job(speaker_job)
        state_embed = self.embedding_state(state)
        party_embed = self.embedding_party(party)
        context_embed = self.embedding_context(context)

        _, (statement_lstm_hidden, _) = self.lstm_statement(statement_embed)
        _, (subject_lstm_hidden, _) = self.lstm_subject(subject_embed)
        _, (speaker_job_lstm_hidden, _) = self.lstm_speaker_job(speaker_job_embed)
        _, (context_lstm_hidden, _) = self.lstm_context(context_embed)


        statement_lstm_hidden = self.lstm_hidden_transform(statement_lstm_hidden)
        subject_lstm_hidden = self.lstm_hidden_transform(subject_lstm_hidden)
        speaker_job_lstm_hidden = self.lstm_hidden_transform(speaker_job_lstm_hidden)
        context_lstm_hidden = self.lstm_hidden_transform(context_lstm_hidden)

        meta_data = torch.cat([
        speaker_embed, state_embed, party_embed, subject_lstm_hidden, speaker_job_lstm_hidden, context_lstm_hidden
        ], dim=1)

        meta_data = self.drop_concat(self.bn_concat(meta_data))
        meta_data_embed = self.drop_meta(self.bn_meta(self.relu_meta(self.fnn_meta(meta_data))))

        statement_lstm_hidden = self.drop_statement(self.bn_statement(statement_lstm_hidden))
        final = torch.cat([meta_data_embed, statement_lstm_hidden], dim=1)
        fnn = self.drop_final(self.bn_final(self.relu_final(self.fnn_final(final))))

        final_output = self.label(fnn)

        return final_output

    
class LSTMFullData(nn.Module):
    
    def __init__(self, embedding_info, output_size, fnn_meta_layer, fnn_final_layer, embed_size,
                hidden_size):
        super(LSTMFullData, self).__init__()

        self.output_size = output_size
        self.one_hot_length = embed_size['state'] + embed_size['party'] + embed_size['speaker']
        self.embedding_info = embedding_info
        self.embed_size = embed_size
        self.meta_embedding_len = 4*3*hidden_size+self.one_hot_length

        # Embeddings
        self.embedding_statement = self.embed_init('statement', grad=True)
        self.embedding_justification = self.embed_init('justification', grad=True)
        self.embedding_subject = self.embed_init('subject', grad=True)
        self.embedding_speaker_job = self.embed_init('speaker_job', grad=True)
        self.embedding_context = self.embed_init('context', grad=True)
        self.embedding_state = self.embed_init('state', grad=False)
        self.embedding_party = self.embed_init('party', grad=False)
        self.embedding_speaker = self.embed_init('speaker', grad=False)

        # Lstm 
        self.lstm_statement = self.lstm_init('statement', hidden_size)
        self.lstm_justification = self.lstm_init('justification', hidden_size)
        self.lstm_subject = self.lstm_init('subject', hidden_size)
        self.lstm_speaker_job = self.lstm_init('speaker_job', hidden_size)
        self.lstm_context = self.lstm_init('context', hidden_size)

        # Meta data layer
        self.bn_concat_meta = nn.BatchNorm1d(self.meta_embedding_len)
        self.drop_concat_meta = nn.Dropout()
        self.fnn_meta = nn.Linear(self.meta_embedding_len, fnn_meta_layer)
        self.bn_meta = nn.BatchNorm1d(fnn_meta_layer)
        self.relu_meta = nn.ReLU()
        self.drop_meta = nn.Dropout()

        self.drop_statement = nn.Dropout()
        self.bn_statement =  nn.BatchNorm1d(4*hidden_size)

        self.drop_justification = nn.Dropout()
        self.bn_justification =  nn.BatchNorm1d(4*hidden_size)

        #Fnn layer
        self.fnn_final = nn.Linear(fnn_meta_layer+8*hidden_size, fnn_final_layer)
        self.bn_final = nn.BatchNorm1d(fnn_final_layer)
        self.relu_final = nn.ReLU()
        self.drop_final = nn.Dropout()

        #Output
        self.label = nn.Linear(fnn_final_layer, output_size)

    def lstm_hidden_transform(self, hidden):
        return torch.cat([hidden[i,:,:] for i in range(int(hidden.shape[0]))], dim=1)
    
    def embed_init(self, feature, grad):
        embed = nn.Embedding(self.embedding_info[feature]['vocab_size'], self.embed_size[feature], padding_idx=1)
        embed.weight = nn.Parameter(self.embedding_info[feature]['field'].vocab.vectors, requires_grad=grad)
        return embed

    def lstm_init(self, feature, size, num_layers=2):
        return nn.LSTM(self.embed_size[feature], size, bidirectional=True, num_layers=num_layers)


    def forward(self, statement, subject, speaker, speaker_job, state, party, context, justification):
        
        # Embeddings
        statement_embed = self.embedding_statement(statement)
        justification_embed = self.embedding_justification(justification)
        subject_embed = self.embedding_subject(subject)
        speaker_embed = self.embedding_speaker(speaker)
        speaker_job_embed = self.embedding_speaker_job(speaker_job)
        state_embed = self.embedding_state(state)
        party_embed = self.embedding_party(party)
        context_embed = self.embedding_context(context)
        
        # Lstm 
        _, (statement_lstm_hidden, _) = self.lstm_statement(statement_embed)
        _, (justification_lstm_hidden, _) = self.lstm_justification(justification_embed)
        _, (subject_lstm_hidden, _) = self.lstm_subject(subject_embed)
        _, (speaker_job_lstm_hidden, _) = self.lstm_speaker_job(speaker_job_embed)
        _, (context_lstm_hidden, _) = self.lstm_context(context_embed)
        
        statement_lstm_hidden = self.lstm_hidden_transform(statement_lstm_hidden)
        justification_lstm_hidden = self.lstm_hidden_transform(justification_lstm_hidden)
        subject_lstm_hidden = self.lstm_hidden_transform(subject_lstm_hidden)
        speaker_job_lstm_hidden = self.lstm_hidden_transform(speaker_job_lstm_hidden)
        context_lstm_hidden = self.lstm_hidden_transform(context_lstm_hidden)
        
        # Meta data layer
        meta_data = torch.cat([
            speaker_embed, state_embed, party_embed, subject_lstm_hidden, speaker_job_lstm_hidden, context_lstm_hidden
        ], dim=1)
        meta_data = self.drop_concat_meta(self.bn_concat_meta(meta_data))
        meta_data_embed = self.drop_meta(self.bn_meta(self.relu_meta(self.fnn_meta(meta_data))))

        statement_lstm_hidden = self.drop_statement(self.bn_statement(statement_lstm_hidden))
        justification_lstm_hidden = self.drop_justification(self.bn_justification(justification_lstm_hidden))
        
        #Fnn layer
        final = torch.cat([meta_data_embed, statement_lstm_hidden, justification_lstm_hidden], dim=1)
        fnn = self.drop_final(self.bn_final(self.relu_final(self.fnn_final(final))))
        
        #Output
        final_output = self.label(fnn)

        return final_output

class LSTMattentionFullData(nn.Module):

    def __init__(self, embedding_info, output_size, fnn_meta_layer, fnn_final_layer, embed_size, hidden_size):
        super(LSTMattentionFullData, self).__init__()

        self.output_size = output_size
        self.one_hot_length = embed_size['state'] + embed_size['party'] + embed_size['speaker']
        self.embedding_info = embedding_info
        self.embed_size = embed_size
        self.meta_embedding_len = 4*3*hidden_size+self.one_hot_length
        self.hidden_size = hidden_size
        # Embeddings
        self.embedding_statement = self.embed_init('statement', grad=True)
        self.embedding_justification = self.embed_init('justification', grad=True)
        self.embedding_subject = self.embed_init('subject', grad=True)
        self.embedding_speaker_job = self.embed_init('speaker_job', grad=True)
        self.embedding_context = self.embed_init('context', grad=True)
        self.embedding_state = self.embed_init('state', grad=False)
        self.embedding_party = self.embed_init('party', grad=False)
        self.embedding_speaker = self.embed_init('speaker', grad=False)

        # Lstm 
        self.lstm_statement = self.lstm_init('statement', hidden_size)
        self.lstm_justification = self.lstm_init('justification', hidden_size)
        self.lstm_subject = self.lstm_init('subject', hidden_size)
        self.lstm_speaker_job = self.lstm_init('speaker_job', hidden_size)
        self.lstm_context = self.lstm_init('context', hidden_size)

        self.drop_statement = nn.Dropout()
        self.bn_statement =  nn.BatchNorm1d(4*hidden_size)
        self.drop_justification = nn.Dropout()
        self.bn_justification =  nn.BatchNorm1d(4*hidden_size)

        # Meta data layer
        self.bn_concat_meta = nn.BatchNorm1d(self.meta_embedding_len)
        self.drop_concat_meta = nn.Dropout()
        self.fnn_meta = nn.Linear(self.meta_embedding_len, fnn_meta_layer)
        self.bn_meta = nn.BatchNorm1d(fnn_meta_layer)
        self.relu_meta = nn.ReLU()
        self.drop_meta = nn.Dropout()

        self.bn_attention_statement = nn.BatchNorm1d(2*hidden_size)
        self.bn_attention_justification = nn.BatchNorm1d(2*hidden_size)
        self.drop_attention_statement = nn.Dropout()
        self.drop_attention_justification = nn.Dropout()

        #Fnn layer
        self.fnn_final = nn.Linear(fnn_meta_layer+2*2*hidden_size, fnn_final_layer)
        self.bn_final = nn.BatchNorm1d(fnn_final_layer)
        self.relu_final = nn.ReLU()
        self.drop_final = nn.Dropout()

        #Output
        self.label = nn.Linear(fnn_final_layer, output_size)

    def lstm_hidden_transform(self, hidden):
        return torch.cat([hidden[i,:,:] for i in range(int(hidden.shape[0]))], dim=1)
    
    def embed_init(self, feature, grad):
        embed = nn.Embedding(self.embedding_info[feature]['vocab_size'], self.embed_size[feature], padding_idx=1)
        embed.weight = nn.Parameter(self.embedding_info[feature]['field'].vocab.vectors, requires_grad=grad)
        return embed

    def lstm_init(self, feature, size, num_layers=2):
        return nn.LSTM(self.embed_size[feature], size, bidirectional=True, num_layers=num_layers)
    
    def apply_attention(self, rnn_output, final_hidden_state):
        final_hidden_state = final_hidden_state[:, :self.hidden_size*2]
        hidden_state = final_hidden_state.unsqueeze(2)
        
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2)
        attention_output = torch.bmm(rnn_output.permute(0,2,1), soft_attention_weights).squeeze(2)
        return attention_output

    def forward(self, statement, subject, speaker, speaker_job, state, party, context, justification):
        
        # Embeddings
        statement_embed = self.embedding_statement(statement)
        justification_embed = self.embedding_justification(justification)
        subject_embed = self.embedding_subject(subject)
        speaker_embed = self.embedding_speaker(speaker)
        speaker_job_embed = self.embedding_speaker_job(speaker_job)
        state_embed = self.embedding_state(state)
        party_embed = self.embedding_party(party)
        context_embed = self.embedding_context(context)
        
        # Lstm 
        statement_lstm_output, (statement_lstm_hidden, _) = self.lstm_statement(statement_embed)
        justification_lstm_output, (justification_lstm_hidden, _) = self.lstm_justification(justification_embed)
        _, (subject_lstm_hidden, _) = self.lstm_subject(subject_embed)
        _, (speaker_job_lstm_hidden, _) = self.lstm_speaker_job(speaker_job_embed)
        _, (context_lstm_hidden, _) = self.lstm_context(context_embed)

        statement_lstm_hidden = self.lstm_hidden_transform(statement_lstm_hidden)
        justification_lstm_hidden = self.lstm_hidden_transform(justification_lstm_hidden)
        subject_lstm_hidden = self.lstm_hidden_transform(subject_lstm_hidden)
        speaker_job_lstm_hidden = self.lstm_hidden_transform(speaker_job_lstm_hidden)
        context_lstm_hidden = self.lstm_hidden_transform(context_lstm_hidden)

        statement_lstm_hidden = self.drop_statement(self.bn_statement(statement_lstm_hidden))
        justification_lstm_hidden = self.drop_justification(self.bn_justification(justification_lstm_hidden))
        
        #Attention
        statement_attention = self.apply_attention(statement_lstm_output.permute(1,0,2), statement_lstm_hidden)
        justification_attention = self.apply_attention(justification_lstm_output.permute(1,0,2), justification_lstm_hidden)
        statement_attention = self.bn_attention_statement(statement_attention)
        justification_attention = self.bn_attention_justification(justification_attention)
        
        # Meta data layer
        meta_data = torch.cat([ 
            speaker_embed, state_embed, party_embed, subject_lstm_hidden, speaker_job_lstm_hidden, context_lstm_hidden
        ], dim=1)
        meta_data = self.drop_concat_meta(self.bn_concat_meta(meta_data))
        meta_data_embed = self.drop_meta(self.bn_meta(self.relu_meta(self.fnn_meta(meta_data))))
        
        #Fnn layer
        final = torch.cat([meta_data_embed, statement_attention, justification_attention], dim=1)
        fnn = self.drop_final(self.bn_final(self.relu_final(self.fnn_final(final))))
        #Output
        final_output = self.label(fnn)

        return final_output