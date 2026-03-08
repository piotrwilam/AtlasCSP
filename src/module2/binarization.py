import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PairRepresentationBuilder:
    """
    Processes N variations for a single (ast, builtin) pair.
    Computes the Consistency Score and applies hard masking.

    Memory protocol: maintain a running integer sum per layer.
    Never hold more than one prompt's activations in memory at a time.
    """

    def __init__(self, epsilon: float, consistency_threshold: float, n_layers: int,
                 batch_size: int = 64):
        self.epsilon = epsilon
        self.consistency_threshold = consistency_threshold
        self.n_layers = n_layers
        self.batch_size = batch_size

    def build(self, extractor, prompts: list) -> dict:
        """
        Process all prompts for one pair, return Pair Representation.

        Args:
            extractor: ActivationExtractor instance
            prompts: list of prompt strings (nominally 100)

        Returns:
            dict[int, np.ndarray]: {layer_id: bool_mask}
            Each bool_mask is shape [n_neurons].
        """
        N = len(prompts)
        sums = {}  # layer_id -> int array (running sum of binary activations)

        for i in range(0, N, self.batch_size):
            batch = prompts[i:i + self.batch_size]
            batch_acts = extractor.extract_batch(batch)
            for acts in batch_acts:
                for lid, vec in acts.items():
                    binary = (vec.abs() > self.epsilon).numpy().astype(np.int32)
                    if lid not in sums:
                        sums[lid] = np.zeros_like(binary)
                    sums[lid] += binary
            del batch_acts

        # Step B + C: Consistency score -> hard mask
        pair_masks = {}
        for lid, total in sums.items():
            consistency = total / N  # float array, 0.0 to 1.0
            pair_masks[lid] = consistency >= self.consistency_threshold  # bool array

        return pair_masks
