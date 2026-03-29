import torch
import lightning as L
import math
import torch.nn.functional

class MyTransformer(L.LightningModule):
    def __init__(self, vocab_size, d_model, max_length, seq_length, layer, heads):
        super(MyTransformer, self).__init__()
        # self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.seq_length = seq_length
        self.layer = layer
        self.heads = heads

        self._build_model()
        # initialize the linear layer and embedding layer
        self._init_weights()
        # (my own word) make the final layer use same tuned parameter as embedding
        # in embedding, vocable index is converted to a vector, the final_layer does it reversely
        # GPT: We use the same embedding matrix to measure similarity between the hidden state
        # and each token embedding, instead of learning a second separate matrix.
        self.final_layer.weight = self.token_embedding.weight

        # define loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _build_model(self):
        # embedding layer
        self.token_embedding = torch.nn.Embedding(self.vocab_size, self.d_model)
        # position encoder
        self.position_encoder = PositionalEncoding(self.max_length, self.d_model)
        # create mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.max_length, self.max_length, dtype=torch.bool))
        )
        # decoder model
        self.layers = torch.nn.ModuleList([MyTransformerLayer(self.d_model, self.heads) for _ in range(self.layer)])
        # linear layer
        self.final_layer = torch.nn.Linear(self.d_model, self.vocab_size, bias = False)

    # proposed by GPT, because modern transformers usually initialize linear layers with xavier_uniform
    # embedding with normal(0, 0.02
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # no_grad disables gradient calculation.
    # This is useful for prediction because it will not call Tensor.backward().
    # No parameter tuning. It is std used for trainer.prediction/test
    @torch.no_grad()
    def generate(self, prompt, char_to_idx, idx_to_char, max_new_tokens = 1000, temperature = 1.0, top_k = None):
        assert temperature > 0

        # eval is a method in nn.module
        # LightningModule inherits from nn.Module
        # it switch module to evaluation mode. no dropout, no batch normalization.
        # it doesn't run evaluation or call training logic, it just change internal layer behavior
        self.eval()
        # okay it just to find out on which device we current are.
        # it is unnecessary to use all parameters because they are on same device
        device = next(self.parameters()).device

        # encode prompt (string → tensor of token ids)
        idx = torch.tensor(
            # convert char to token
            [char_to_idx[ch] for ch in prompt],
            dtype = torch.long,
            device = device
        ).unsqueeze(0)

        # interation till the maximum new char is generated
        for _ in range(max_new_tokens):

            # keep only last seq_length tokens
            # prompt can have any length > seq_length, indeed, only the last seq_length chars are taken
            last_chars = idx[:, -self.seq_length:]

            # forward pass
            # logits has size[1 seq_length vocab_size]
            logits = self(last_chars)

            # take last char/token
            # scale the probability with temperature
            logits = logits[:, -1, :] / temperature

            # if top_k is not none, the probability is lower then the last top_k is masked
            # it will not taken, useful for the multinominal.
            # similar mechanism is top_p
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                min_values = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, torch.full_like(logits, -float('inf')), logits)

            probs = torch.nn.functional.softmax(logits, dim=-1)

            # ref Log "multinomial"
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

        # decode tokens → string
        return ''.join([idx_to_char[i] for i in idx[0].tolist()])



    def forward(self, x):
        batch_size, seq_length = x.shape
        # good to know *  math.sqrt(self.d_model) was done in original transformer
        # the modern GPT oft removes this scaling
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.position_encoder(x)
        # get mask
        mask = self.causal_mask[:seq_length, :seq_length]
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        # x_hat [batch_size, seq_length, vocab_size]
        # y [batch_size, seq_length]
        # CrossEntropy expects input [N C], target [N]
        B, T, V = x_hat.shape
        loss = self.loss_fn(
            x_hat.reshape(B * T, V),
            y.reshape(B * T)
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        B, T, V = x_hat.shape
        loss = self.loss_fn(
            x_hat.reshape(B * T, V),
            y.reshape(B * T)
        )

        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        B, T, V = x_hat.shape
        loss = self.loss_fn(
            x_hat.reshape(B * T, V),
            y.reshape(B * T)
        )

        self.log("test_loss", loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=3e-4, weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=300
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # interval has default value epoch. as name said scheduler is adapted scheduler wise
        # it can also be set step, in this way T_max means number of steps, not h epoch anymore
        # ref...
        # "lr_scheduler" : {"scheduler": scheduler}, "interval": "step"}

class MyTransformerLayer(torch.nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads

        assert self.d_model % self.heads == 0

        self._transformer()
        self._feed_forward()
        self.norm_attn = torch.nn.LayerNorm(d_model)
        self.norm_ffn = torch.nn.LayerNorm(d_model)

    def _transformer(self):

        self.W_q = torch.nn.Linear(self.d_model, self.d_model, bias=False, )
        self.W_k = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = torch.nn.Linear(self.d_model, self.d_model, bias=False)

        self.attn_dropout = torch.nn.Dropout(0.1)
        self.attention_out_dropout = torch.nn.Dropout(0.1)
        self.ffn_dropout = torch.nn.Dropout(0.1)

        self.out_proj = torch.nn.Linear(self.d_model, self.d_model)

    def _feed_forward(self):
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, 4 * self.d_model),
            torch.nn.GELU(),
            torch.nn.Linear(4 * self.d_model, self.d_model),
        )

    def attention(self, x, mask):
        # following code calculate the Q, K and V at first
        # 2nd split the Q, K and V to multiple head attention by
        #  .view(batch_size, seq_length, heads, d_model/heads]
        #  .transpose(2,1) exchange the data between the heads and seq_length
        t = self.d_model // self.heads
        sq_len = x.size(1)
        Q = self.W_q(x).view(x.size(0), sq_len, self.heads, t).transpose(1, 2) # [batch_size, heads, seq_length, d_model//heads]
        K = self.W_k(x).view(x.size(0), sq_len, self.heads, t).transpose(1, 2) # [batch_size, heads, seq_length, d_model//heads]
        V = self.W_v(x).view(x.size(0), sq_len, self.heads, t).transpose(1, 2) # [batch_size, heads, seq_length, d_model//heads]

        # Batch multiplication, the batch dimension is kept
        # transpose(-2, -1), exchange 2nd last dimension with last dimension
        # [batch_size, seq_length, d_model] * [batch_size, d_model, seq_length]
        # next two code line \frac{Q\cdot K^T}{\sqrt{d_k}}
        # d_k = d_model // heads
        scores = Q @ K.transpose(-2, -1) # [batch_size, heads, seq_length, seq_length]
        scores /= math.sqrt(t)

        # mask is so called causal (look-ahead) mask.
        # It enforces: token i can only attend to tokens ≤ i
        # To prevent peeking into the future during language modeling.
        # Without it, token 3 could attend to token 10 — which breaks next-token prediction.
        # It was also defined in decoder of the original Vaswani et al paper, but not in encoder.
        # for more refer chat gpt conversation : Causal Mask Explanation
        #
        # the code unsqueeze(0).unsqueeze(0)
        # scores is in [batch, heads, seq_len, seq_len]
        # mask is in [seq_len, seq_len}
        # the maks works only because broadcasts automatically
        # using unsqueeze(0).unsqueeze(0) shapes maks to [1, 1, seq_len, seq_len]
        if mask is not None:
            scores = scores.masked_fill(
                ~mask.unsqueeze(0).unsqueeze(0),
                float('-inf'))

        attn = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = attn @ V #[batch_size, heads, seq_length, d_model]

        # reshape to #[batch_size, seq_length, d_model]
        out = out.transpose(1, 2).contiguous()  #[batch_size, seq_length, heads, d_model]
        out = out.view(x.size(0), sq_len, self.d_model) #[batch_size, seq_length, d_model]
        out = self.out_proj(out)

        return out

    def forward(self, x, mask = None):
        # following left of **post ln** is the original transformer procedure
        # it calls post normalization, which means the normalization is done after attention and residual connection
        # This process is less stable for deep stacks, multiple transformer blocks
        # Most modern models uses pre LN, it means the the attention process is applied on the normalized data
        # x  => self_attention ** <= post LN ** pre LN => **. x => norm_1st => self_attention
        # => + x
        # => norm_1st. ** <= post LN **   pre LN => **. => norm_2nd
        # => feed_forward
        # => + norm_1st
        # => norm_2nd

        # USING PRE LN
        attention_out = self.attention(self.norm_attn(x), mask)
        x = x + self.attention_out_dropout(attention_out)

        ffn = self.feed_forward(self.norm_ffn(x))
        x = x + self.ffn_dropout(ffn)

        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        nominator = torch.arange(0, max_len).unsqueeze(1).float()
        denominator = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(nominator * denominator)
        pe[:, 1::2] = torch.cos(nominator * denominator)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)

# Log
# the following class implementation was original ideal create the position encoding table
# the ideal overwrite getitem method comes from the orch.utils.dataset
# to return the position encoding look up table
# unfortunately this ideal has some lacks for the machine learning
# 1: the instance will stay in the cpu, if the model moves to gpu, model.cuda(),
#     the code ist needed:   self.pe.pe = self.pe.pe.to(x.device)
# 2: Dtype mismatches, if model runs float16 or bfloat16, pe is defined float32
#      correction needed:  self.pe.pe = self.pe.pe.to(dtype=x.dtype)
# 3: Not saved in state_dict
#   checkpoints won’t contain it; reproducibility relies on code, not state;
# Therefor the following code is dropout. The new implementation, ref above, is very similar,
# instead of plain class, it is a subclass of nn.module

# class PositionalEncoding():
#     def __init__(self, max_len, d_model):
#         self.max_len = max_len
#         self.d_model = d_model
#         self.pe = torch.zeros(max_len, d_model)
#         nominator = torch.arange(0, max_len).unsqueeze(1).float()
#         denominator = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         self.pe[:, 0::2] = torch.sin(nominator * denominator)
#         self.pe[:, 1::2] = torch.cos(nominator * denominator)
#
#
#     # pls be aware the argument slice is not single index it shall like start_position:end_position
#     def __getitem__(self, slice):
#         return self.pe[slice]

# Log
# key word : register_buffer
# is ideal for storing non-trainable data like masks, statistics or further information.
# register_buffer is part of model states and moves with model together to cpu/gpu. with register_buffer defined
# object is stored in the state_dict()
# e.g. following code define a mask named causal_mask
# self.register_buffer('causal_mask', torch.tril(torch.ones(self.max_length, self.max_length, dtype=torch.bool)))
#
# In first implementation, I used register_buffer define the causal_mask, It means each transformer layer has it own mask.
# But:
# 1. The mask is a property of the task, not the layer.
# 2. All layers use the same mask.
# 3. The mask depends on input length.
# 4. Some tasks use different masks (padding mask, causal mask, cross mask).
#
# Redesign using of the mask, following code was implemented within _transformer to define the mask
#        if self.use_causal_mask:
#             self.register_buffer('causal_mask', torch.tril(torch.ones(self.max_length, self.max_length, dtype=torch.bool)))
# It is removed because the mask shall be defined on the taks level

# Log "multinomial"
# Normally we take the highest probability for the next token by using argmax, like CNN classification
#
# Assume the outcoming of the softmax has 70% for apple 30% banana, we will always predict apple
# the prediction might stuck in a loop, "i like apple, i like apple..."
# With multinomial, instead of argmax, if we run the prediction we will 70% chance of predicting apple
# 30% chance banana. This mechanism is best for chat, stories, and brainstorming where you want variety.
#
# To avoid that very very low probability token can be taken we use top_k/top_p to maks token with low probability
# Greedy (argmax): Best for facts, math, or translation where there is one "right" answer.
#
# Sampling (multinomial): Best for chat, stories, and brainstorming where you want variety.
