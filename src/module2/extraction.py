import logging
import torch

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """
    Wraps the CSP model with forward hooks on MLP hidden states.

    IMPORTANT: The CSP model is a CUSTOM architecture (circuitgpt),
    NOT standard GPT-2. Layer names differ from vanilla GPT-2.

    The model ships with a built-in hook_recorder (see hook_utils.py
    in the HuggingFace repo). If available, prefer using it:
        with hook_recorder() as rec:
            model(idx)
        # rec keys: "0.mlp.act_in", "0.attn.act_out", etc.

    If hook_recorder is not available from the HF download, use manual
    register_forward_hook after identifying layer names via
    model.named_modules().
    """

    def __init__(self, model, tokenizer, device, n_layers=8,
                 use_hook_recorder=False, hook_recorder_fn=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_layers = n_layers
        self.use_hook_recorder = use_hook_recorder and hook_recorder_fn is not None
        self.hook_recorder_fn = hook_recorder_fn
        self.activations = {}
        self.hooks = []
        self.hook_pattern = None  # e.g. "blocks.{layer_id}.mlp" — set after inspecting named_modules

    def set_hook_pattern(self, pattern: str):
        """Set the module name pattern, e.g. 'blocks.{layer_id}.mlp'."""
        self.hook_pattern = pattern

    def register_hooks(self):
        """Attach forward hooks to all N MLP layers. Return hook handles."""
        if self.use_hook_recorder:
            logger.info("Using built-in hook_recorder — no manual hooks needed")
            return
        if self.hook_pattern is None:
            raise ValueError(
                "hook_pattern not set. Inspect model.named_modules() and call "
                "set_hook_pattern() with the correct pattern before registering hooks."
            )
        module_dict = dict(self.model.named_modules())
        for i in range(self.n_layers):
            name = self.hook_pattern.format(layer_id=i)
            if name not in module_dict:
                logger.error(f"Could not find layer: {name}")
                continue
            handle = module_dict[name].register_forward_hook(self._make_hook(i))
            self.hooks.append(handle)
        logger.info(f"Registered {len(self.hooks)} manual hooks")

    def remove_hooks(self):
        """Remove all hooks to prevent memory leaks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def extract(self, prompt_text: str, token_pos: int = -1) -> dict:
        """
        Run a single forward pass, return activations at all layers.

        Args:
            prompt_text: the code string
            token_pos: which token to extract from (-1 = last)

        Returns:
            dict[int, torch.Tensor]: {layer_id: activation_vector}
            Each tensor is shape [n_neurons] (1D, the MLP expansion dim)
        """
        # CRITICAL: add_special_tokens=False per CSP requirements
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

        if self.use_hook_recorder:
            with self.hook_recorder_fn() as rec:
                with torch.no_grad():
                    self.model(inputs)
            result = {}
            for key, tensor in rec.items():
                if "mlp" in key and "act" in key:
                    layer_id = int(key.split(".")[0])
                    result[layer_id] = tensor[0, token_pos, :].cpu()
            return result
        else:
            self.activations = {}
            with torch.no_grad():
                self.model(inputs)
            result = {}
            for lid, act in self.activations.items():
                result[lid] = act[0, token_pos, :].cpu()
            self.activations = {}
            return result

    def _make_hook(self, layer_id: int):
        """Factory that returns a hook function capturing activations."""
        def fn(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            # out shape: [batch, seq_len, mlp_dim]
            self.activations[layer_id] = out.detach()
        return fn
