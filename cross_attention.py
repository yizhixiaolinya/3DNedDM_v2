import torch
import torch.nn as nn

class CrossAttentionModule(nn.Module):
    def __init__(self, num_heads, d_model):
        super(CrossAttentionModule, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads)

    def forward(self, sequences):
        # sequences: List of tensors with shape (b, c, l)

        # Stack the sequences along the time dimension to create a tensor of shape (b, c, l_sum)
        sequences_stacked = torch.cat(sequences, dim=2)

        # Calculate self-attention
        output, _ = self.cross_attention(sequences_stacked, sequences_stacked, sequences_stacked)

        return output

def main():
    num_heads = 8
    d_model = 256
    sequence_lengths = 100
    batch_size = 16

    cross_attention_module = CrossAttentionModule(num_heads, d_model)

    num_sequences = torch.randint(1, 5, (1,)).item()  # Randomly choose 1 to 4 sequences
    sequences = []
    for _ in range(num_sequences):
        sequence = torch.rand(batch_size, d_model, sequence_lengths)
        sequences.append(sequence)

    output_sequence = cross_attention_module(sequences)
    # print("Output sequence shape:", output_sequence.shape)

if __name__ == "__main__":
    main()
