import torch

from .BaseTrimmer import BaseTrimmer


class M2M100Trimmer(BaseTrimmer):
    def __init__(self, model, config, tokenizer):
        super().__init__(model, config, tokenizer)

    def trim_weights(self):
        if 'lm_head.weight' in self.model.state_dict():
            # LM head matrix
            lmh = self.model.lm_head.weight.detach().numpy()
            self.trimmed_weights['lm_head'] = lmh[self.trimmed_vocab_ids, :]

            # embedding matrix
            em = self.model.model.shared.weight.detach().numpy()
        else:
            em = self.model.shared.weight.detach().numpy()
        self.trimmed_weights['shared'] = em[self.trimmed_vocab_ids, :]

    def initialize_new_model(self):
        # arch = self.config.architectures[0]
        arch = self.model.__class__.__name__
        if arch=='M2M100Model':
            from transformers import M2M100Model
            model = M2M100Model(self.config)
            changed_params = [
                'shared.weight',
                'encoder.embed_tokens.weight',
                'decoder.embed_tokens.weight',
            ]
        elif arch=='M2M100ForConditionalGeneration':
            from transformers import M2M100ForConditionalGeneration
            model = M2M100ForConditionalGeneration(self.config)
            changed_params = [
                'model.shared.weight',
                'model.encoder.embed_tokens.weight',
                'model.decoder.embed_tokens.weight',
                'lm_head.weight',
            ]
        else:
            raise NotImplementedError('ERROR: M2M100Trimmer does not support this architecture!')

        self.trimmed_model = model
        self.changed_params = changed_params

    def trim_model(self):
        # copy unchanged params over from the old model
        for param in self.model.state_dict().keys():
            if param in self.changed_params:
                continue
            self.trimmed_model.state_dict()[param].copy_(self.model.state_dict()[param])

        # set trimmed params
        prunedEmbeddingMatrix = torch.nn.Embedding.from_pretrained(torch.Tensor(self.trimmed_weights['shared']),
                                                                    freeze=False,
                                                                    padding_idx=self.tokenizer.pad_token_id,
        )
        self.trimmed_model.set_input_embeddings(prunedEmbeddingMatrix)

        if 'lm_head' in self.trimmed_weights:
            prunedLMHeadMatrix = torch.Tensor(self.trimmed_weights['lm_head'])
            _ = self.trimmed_model.lm_head.weight.data.copy_(prunedLMHeadMatrix)

        # tie weights as set up in config
        self.trimmed_model.tie_weights()
