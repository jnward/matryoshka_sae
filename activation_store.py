import torch
import transformer_lens.utils as utils
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens.hook_points import HookedRootModule


class ActivationsStore:
    def __init__(
        self,
        model: HookedRootModule,
        cfg: dict,
    ):
        self.model = model
        self.dataset = iter(load_dataset(cfg["dataset_path"], split="train", streaming=True, trust_remote_code=True))

        # Use the TransformerLens utility to properly construct the hook point name
        layer = cfg.get("layer", 0)
        site = cfg.get("site", "resid_post")
        self.hook_point = utils.get_act_name(site, layer)

        self.context_size = min(cfg.get("seq_len", 1024), model.cfg.n_ctx)
        self.model_batch_size = cfg.get("model_batch_size", 8)
        self.device = cfg.get("device", "cpu")
        self.num_batches_in_buffer = cfg.get("num_batches_in_buffer", 10)
        self.tokens_column = self._get_tokens_column()
        self.config = cfg
        self.tokenizer = model.tokenizer

        # Initialize activation buffer and dataloader iterator
        self.activation_buffer = self._fill_buffer()
        self.dataloader = self._get_dataloader()
        self.dataloader_iter = iter(self.dataloader)

    def _get_tokens_column(self):
        sample = next(self.dataset)
        if "tokens" in sample:
            return "tokens"
        elif "input_ids" in sample:
            return "input_ids"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")

    def get_batch_tokens(self):
        all_tokens = []
        while len(all_tokens) < self.model_batch_size * self.context_size:
            batch = next(self.dataset)
            if self.tokens_column == "text":
                tokens = self.model.to_tokens(
                    batch["text"], truncate=True, move_to_device=True, prepend_bos=True
                ).squeeze(0)
            else:
                tokens = batch[self.tokens_column]
            all_tokens.extend(tokens)
        token_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.device)[
            : self.model_batch_size * self.context_size
        ]
        return token_tensor.view(self.model_batch_size, self.context_size)

    def get_activations(self, batch_tokens: torch.Tensor):
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.config["layer"] + 1,
            )
        return cache[self.hook_point]

    def _fill_buffer(self):
        all_activations = []
        for _ in range(self.num_batches_in_buffer):
            batch_tokens = self.get_batch_tokens()
            activations = self.get_activations(batch_tokens).reshape(-1, self.config["act_size"])
            all_activations.append(activations)
        return torch.cat(all_activations, dim=0)

    def _get_dataloader(self):
        return DataLoader(TensorDataset(self.activation_buffer), batch_size=self.config["batch_size"], shuffle=True)

    def next_batch(self):
        try:
            return next(self.dataloader_iter)[0]
        except (StopIteration, AttributeError):
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)[0]
