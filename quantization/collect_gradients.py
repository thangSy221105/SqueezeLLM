import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from squeezellm.datautils import get_loaders
from squeezellm.model_parse import (
    get_layers,
    get_module_names,
    get_modules,
    parse_model,
)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="model to load")
parser.add_argument(
    "--output_path",
    type=str,
    default=None,
    help="path to save per-layer gradient-square chunks",
)
parser.add_argument(
    "--output", dest="output_path", type=str, help="alias of --output_path"
)
parser.add_argument(
    "--model_type",
    type=str,
    default=None,
    help="model type",
    choices=["llama", "opt", "mistral"],
)
parser.add_argument(
    "--dataset",
    type=str,
    default="c4",
    choices=["wikitext2", "ptb", "c4"],
    help="dataset used to estimate gradient-square statistics",
)
parser.add_argument("--nsamples", type=int, default=128, help="number of samples")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument(
    "--seqlen",
    type=int,
    default=2048,
    help="sequence length used for gradient collection",
)
parser.add_argument("--cache_dir", type=str, default=None, help="cache directory")


def main():
    args = parser.parse_args()

    if args.output_path is None:
        raise ValueError("Please provide --output_path (or --output).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    model_type = args.model_type or parse_model(model)
    model.seqlen = min(getattr(model.config, "max_position_embeddings", 2048), args.seqlen)
    model.to(device)
    model.eval()
    model.config.use_cache = False

    trainloader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    layers = get_layers(model, model_type)
    module_names = get_module_names(model_type)
    grad_squares = []
    for layer in layers:
        layer_stats = {}
        for module, name in zip(get_modules(layer, model_type), module_names):
            layer_stats[name] = torch.zeros(
                module.weight.shape, dtype=torch.float32, device="cpu"
            )
        grad_squares.append(layer_stats)

    print(f"Collecting gradient-square statistics from {len(trainloader)} samples")
    for input_ids, labels in tqdm(trainloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        for layer_idx, layer in enumerate(layers):
            for module, name in zip(get_modules(layer, model_type), module_names):
                if module.weight.grad is None:
                    continue
                grad_squares[layer_idx][name] += (
                    module.weight.grad.detach().float().cpu().pow(2)
                )

    os.makedirs(args.output_path, exist_ok=True)
    for layer_idx, layer_stats in enumerate(grad_squares):
        torch.save(layer_stats, os.path.join(args.output_path, f"layer_{layer_idx}.pt"))

    print(f"Saved gradient-square chunks to {args.output_path}")


if __name__ == "__main__":
    main()
