from typing import Any

import torch
from torch import nn

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.decoder import get_decoder
from training.architecture.transformer.encoder import get_encoder


class TrOMR(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.config = config

    def eval_mode(self) -> None:
        self.decoder.eval()
        self.encoder.eval()

    def forward(
        self,
        inputs: torch.Tensor,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        sampling_prob: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        context = self.encoder(inputs)
        loss = self.decoder(
            rhythms=rhythms,
            pitchs=pitchs,
            lifts=lifts,
            articulations=articulations,
            positions=positions,
            context=context,
            mask=mask,
            sampling_prob=sampling_prob,
            **kwargs,
        )
        return loss

    @torch.no_grad()
    def generate(self, x: torch.Tensor) -> list[EncodedSymbol]:
        start_token = torch.tensor([[1]], dtype=torch.long, device=x.device)
        nonote_token = torch.tensor([[0]], dtype=torch.long, device=x.device)

        context = self.encoder(x)
        out = self.decoder.generate(start_token, nonote_token, context=context)

        return out

    def freeze_decoder(self) -> None:
        """Freeze all decoder parameters to prevent updates during training."""
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters to prevent updates during training."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_backbone(self) -> None:
        """Freeze only the encoder backbone."""
        if hasattr(self.encoder, "freeze_backbone"):
            self.encoder.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        """Unfreeze the encoder backbone."""
        if hasattr(self.encoder, "unfreeze_backbone"):
            self.encoder.unfreeze_backbone()

    def unfreeze_lift_decoder(self) -> None:
        for param in self.decoder.net.lift_emb.parameters():
            param.requires_grad = True
        for param in self.decoder.net.to_logits_lift.parameters():
            param.requires_grad = True

    def unfreeze_rhythm_decoder(self) -> None:
        """Also train the rhythm branch.

        Needed for the makam tokens that are not accidentals: the key signature
        arrives as keyAccidental symbols and the usul as timeSignature_N/D, both
        of which are rhythm tokens. With the branch frozen the model can never
        emit either, however well it learns the accidental glyphs themselves.
        """
        for param in self.decoder.net.rhythm_emb.parameters():
            param.requires_grad = True
        for param in self.decoder.net.to_logits_rhythm.parameters():
            param.requires_grad = True


def _grow_to_fit_vocabulary(
    model: TrOMR, state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Let a checkpoint from a smaller vocabulary load into a larger one.

    Adding tokens changes the first dimension of the embedding and output layers,
    which makes load_state_dict fail on a shape mismatch even with strict=False.
    Tokens are only ever appended (see vocabulary.build_lift / build_rhythm), so
    the old rows still mean the same thing: copy them across and leave the rows
    for the new tokens freshly initialised.
    """
    current = model.state_dict()
    adjusted = {}
    for key, saved in state.items():
        target = current.get(key)
        if target is None or saved.shape == target.shape:
            adjusted[key] = saved
            continue
        if saved.dim() != target.dim() or any(
            s > t for s, t in zip(saved.shape, target.shape)
        ):
            eprint(f"Skipping {key}: checkpoint shape {tuple(saved.shape)} does not fit")
            continue
        grown = target.clone()
        grown[tuple(slice(0, s) for s in saved.shape)] = saved.to(grown.dtype)
        adjusted[key] = grown
        eprint(f"Grew {key} from {tuple(saved.shape)} to {tuple(target.shape)}")
    return adjusted


def load_model(config: Config) -> TrOMR:
    """Load model from checkpoint."""
    model = TrOMR(config)
    checkpoint_path = config.filepaths.checkpoint
    if checkpoint_path.endswith(".safetensors"):
        import safetensors  # noqa: PLC0415

        tensors = {}
        with safetensors.safe_open(checkpoint_path, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(_grow_to_fit_vocabulary(model, tensors), strict=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensors = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(_grow_to_fit_vocabulary(model, tensors), strict=False)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model
