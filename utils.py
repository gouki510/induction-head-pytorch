import matplotlib.pyplot as plt


def visalize_attention(model, layer_i):
    attn_matrix = model.get_attention_matrix(layer_i)
    num_heads = attn_matrix.size(0)
    fig, ax = plt.subplots(num_heads, 1, figsize=(5*num_heads, 5))
    if num_heads == 1:
        ax = [ax]
    for j in range(num_heads):
        ax[j].imshow(attn_matrix[j].detach().cpu().numpy(), cmap="Blues")
    plt.tight_layout()
    return fig
